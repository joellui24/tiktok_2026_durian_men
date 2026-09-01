"""Optional free-form semantic retrieval over the frozen product catalogue.

The production :mod:`starter.agent` imports this module only after a message has
already been routed to the free-form path.  The official formatted-query path
therefore does not import FastEmbed, load the model, or touch the embedding
artifact.

Hard constraints are deliberately outside this module.  ``dense_rank`` only
orders identifiers supplied by its caller and can never restore a filtered
product.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Keeping documents concise materially reduces the one-time CPU build while
# retaining the high-signal title/category/brand and leading feature text.
DEFAULT_DOCUMENT_CHAR_LIMIT = 300
DEFAULT_RRF_K = 60
ARTIFACT_VERSION = "semantic-catalog-v1"


def _flatten(value: object) -> list[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, Mapping):
        return [
            f"{key}: {item}"
            for key, item in value.items()
            if item not in (None, "", [])
        ]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item not in (None, "", [])]
    return [str(value)]


def product_document(
    product: Mapping[str, object], *, character_limit: int = DEFAULT_DOCUMENT_CHAR_LIMIT
) -> str:
    """Return stable, high-signal text for one catalogue product.

    Title/category/brand precede feature clauses so they survive model
    truncation.  Only two feature clauses and any remaining room for one short
    description are used; dates, dimensions and other noisy details are not
    embedded.
    """

    title = " ".join(_flatten(product.get("title")))
    category = " > ".join(_flatten(product.get("categories")))
    store = " ".join(_flatten(product.get("store")))
    features = " ".join(_flatten(product.get("features"))[:2])
    description = " ".join(_flatten(product.get("description"))[:1])
    parts = [
        f"Title: {title}",
        f"Category: {category}",
        f"Brand: {store}",
        f"Features: {features}",
        f"Description: {description}",
    ]
    return " | ".join(part for part in parts if not part.endswith(": "))[
        : max(1, int(character_limit))
    ]


def _catalog_rows(catalog_path: str | Path) -> tuple[list[str], list[str], str]:
    identifiers: list[str] = []
    documents: list[str] = []
    digest = hashlib.sha256()
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            product = json.loads(line)
            parent_asin = str(product.get("parent_asin", "")).strip()
            if not parent_asin:
                raise ValueError(f"catalogue line {line_number} has no parent_asin")
            document = product_document(product)
            identifiers.append(parent_asin)
            documents.append(document)
            digest.update(parent_asin.encode("utf-8"))
            digest.update(b"\0")
            digest.update(document.encode("utf-8"))
            digest.update(b"\n")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("catalogue contains duplicate parent_asin values")
    return identifiers, documents, digest.hexdigest()


def _embedding_model(
    *,
    model_name: str,
    cache_dir: str | Path | None,
    model_path: str | Path | None,
    local_files_only: bool,
    threads: int | None,
):
    # FastEmbed is intentionally imported only when semantic work is requested.
    from fastembed import TextEmbedding

    kwargs: dict[str, object] = {
        "model_name": model_name,
        "threads": threads,
        "local_files_only": local_files_only,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if model_path is not None:
        kwargs["specific_model_path"] = str(model_path)
    return TextEmbedding(**kwargs)


def build_embedding_artifact(
    catalog_path: str | Path,
    output_path: str | Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    local_files_only: bool = False,
    threads: int | None = None,
    batch_size: int = 128,
    parallel: int | None = None,
) -> dict[str, object]:
    """Build a compressed float16 catalogue embedding artifact."""

    identifiers, documents, catalog_sha256 = _catalog_rows(catalog_path)
    model = _embedding_model(
        model_name=model_name,
        cache_dir=cache_dir,
        model_path=model_path,
        local_files_only=local_files_only,
        threads=threads,
    )
    started = time.perf_counter()
    vectors = np.asarray(
        list(
            model.passage_embed(
                documents,
                batch_size=max(1, int(batch_size)),
                parallel=parallel,
            )
        ),
        dtype=np.float32,
    )
    build_seconds = time.perf_counter() - started
    if vectors.ndim != 2 or vectors.shape[0] != len(identifiers):
        raise RuntimeError(
            f"unexpected embedding shape {vectors.shape}; expected {len(identifiers)} rows"
        )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "catalog_sha256": catalog_sha256,
        "document_char_limit": DEFAULT_DOCUMENT_CHAR_LIMIT,
        "embedding_dimension": int(vectors.shape[1]),
        "model_name": model_name,
        "product_count": len(identifiers),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        identifiers=np.asarray(identifiers, dtype="U16"),
        embeddings=vectors.astype(np.float16),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {
        **metadata,
        "build_seconds": build_seconds,
        "artifact_bytes": output_path.stat().st_size,
        "output_path": str(output_path),
    }


@dataclass(frozen=True)
class RetrievalTiming:
    model_load_seconds: float
    query_embedding_seconds: float
    search_seconds: float


class SemanticCatalogIndex:
    """Lazy query encoder plus exact cosine search over 50,000 products."""

    def __init__(
        self,
        artifact_path: str | Path,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        local_files_only: bool = True,
        threads: int | None = None,
    ) -> None:
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"semantic artifact not found: {artifact_path}")
        started = time.perf_counter()
        with np.load(artifact_path, allow_pickle=False) as payload:
            self.identifiers = tuple(str(value) for value in payload["identifiers"])
            # Convert once at load; float32 BLAS is materially faster than
            # repeated float16 cosine calculations on CPU.
            self.embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
            metadata_value = payload["metadata"]
            self.metadata = json.loads(str(metadata_value.item()))
        if self.metadata.get("artifact_version") != ARTIFACT_VERSION:
            raise ValueError("unsupported semantic artifact version")
        if self.embeddings.ndim != 2 or len(self.identifiers) != len(self.embeddings):
            raise ValueError("semantic artifact identifiers and vectors do not align")
        if len(set(self.identifiers)) != len(self.identifiers):
            raise ValueError("semantic artifact contains duplicate identifiers")
        if self.metadata.get("product_count") != len(self.identifiers):
            raise ValueError("semantic artifact product count is inconsistent")
        if self.metadata.get("embedding_dimension") != self.embeddings.shape[1]:
            raise ValueError("semantic artifact embedding dimension is inconsistent")
        if self.metadata.get("model_name") != model_name:
            raise ValueError("semantic artifact was built with a different model")
        if not np.isfinite(self.embeddings).all():
            raise ValueError("semantic artifact contains non-finite vectors")
        vector_norms = np.linalg.norm(self.embeddings, axis=1)
        if not np.allclose(vector_norms, 1.0, atol=0.02):
            raise ValueError("semantic artifact vectors are not normalized")
        self.row_for_id = {
            parent_asin: position
            for position, parent_asin in enumerate(self.identifiers)
        }
        self.artifact_load_seconds = time.perf_counter() - started
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model_path = model_path
        self.local_files_only = local_files_only
        self.threads = threads
        self._model = None
        self.model_load_seconds = 0.0
        self.last_timing = RetrievalTiming(0.0, 0.0, 0.0)

    def _load_model(self):
        if self._model is None:
            started = time.perf_counter()
            self._model = _embedding_model(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                model_path=self.model_path,
                local_files_only=self.local_files_only,
                threads=self.threads,
            )
            self.model_load_seconds = time.perf_counter() - started
        return self._model

    def dense_rank(
        self,
        query: str,
        candidate_ids: Iterable[str],
        *,
        limit: int | None = None,
    ) -> list[str]:
        """Rank only supplied hard-filtered identifiers by cosine similarity."""

        query = " ".join(str(query).split())
        candidates = list(dict.fromkeys(str(value) for value in candidate_ids))
        if not query or not candidates:
            return candidates[: max(0, limit or len(candidates))]
        rows_and_ids = [
            (self.row_for_id[parent_asin], parent_asin)
            for parent_asin in candidates
            if parent_asin in self.row_for_id
        ]
        if not rows_and_ids:
            return candidates[: max(0, limit or len(candidates))]

        model = self._load_model()
        started = time.perf_counter()
        query_vector = np.asarray(next(iter(model.query_embed([query]))), dtype=np.float32)
        norm = float(np.linalg.norm(query_vector))
        if not math.isfinite(norm) or norm <= 0.0:
            return candidates[: max(0, limit or len(candidates))]
        query_vector /= norm
        query_seconds = time.perf_counter() - started

        started = time.perf_counter()
        row_numbers = np.fromiter((row for row, _ in rows_and_ids), dtype=np.int64)
        scores = self.embeddings[row_numbers] @ query_vector
        ranked_positions = sorted(
            range(len(rows_and_ids)),
            key=lambda position: (-float(scores[position]), rows_and_ids[position][1]),
        )
        if limit is not None:
            ranked_positions = ranked_positions[: max(0, int(limit))]
        ranked = [rows_and_ids[position][1] for position in ranked_positions]
        search_seconds = time.perf_counter() - started
        self.last_timing = RetrievalTiming(
            self.model_load_seconds, query_seconds, search_seconds
        )
        return ranked


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    weights: Sequence[float] | None = None,
    k: int = DEFAULT_RRF_K,
    limit: int = 10,
) -> list[str]:
    """Fuse heterogeneous rankings without assuming comparable score scales."""

    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must align with rankings")
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    seen_order = 0
    for ranking, weight in zip(rankings, weights, strict=True):
        if weight <= 0:
            continue
        for rank, parent_asin in enumerate(dict.fromkeys(ranking), start=1):
            if parent_asin not in first_seen:
                first_seen[parent_asin] = seen_order
                seen_order += 1
            scores[parent_asin] = scores.get(parent_asin, 0.0) + float(weight) / (
                k + rank
            )
    return sorted(
        scores,
        key=lambda parent_asin: (
            -scores[parent_asin],
            first_seen[parent_asin],
            parent_asin,
        ),
    )[: max(0, int(limit))]


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the free-form semantic index")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output", default="data/semantic_embeddings.npz")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--cache-dir")
    parser.add_argument("--model-path")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--parallel", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    result = build_embedding_artifact(
        args.catalog,
        args.output,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        model_path=args.model_path,
        local_files_only=args.local_files_only,
        threads=args.threads,
        batch_size=args.batch_size,
        parallel=args.parallel,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
