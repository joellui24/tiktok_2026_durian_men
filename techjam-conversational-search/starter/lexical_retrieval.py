"""Optional SQLite FTS5 retrieval for free-form shopping queries.

This module is intentionally independent of :mod:`starter.agent`.  Building an
artifact is an explicit offline operation, while :class:`LexicalCatalogIndex`
opens that artifact lazily and read-only on the first search.  Callers must
supply the identifiers that survived deterministic hard filtering; search
results are always a subset of those identifiers.

The catalogue document layout mirrors
``starter.semantic_retrieval.product_document`` without importing optional ML
dependencies.  Lexical and dense indexes can therefore be built from the same
stable product text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


ARTIFACT_VERSION = "lexical-catalog-v1"
DEFAULT_DOCUMENT_CHAR_LIMIT = 300
DEFAULT_LIMIT = 500
DEFAULT_MAX_QUERY_TOKENS = 16
FTS_TABLE = "product_fts"

# Shopping boilerplate contributes little relevance and can cause an OR query
# to match most of a 50,000-product catalogue.  Hard operators are also omitted
# from lexical syntax; their semantics belong to the deterministic parser and
# candidate set supplied by the caller.
QUERY_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "all",
        "an",
        "and",
        "any",
        "anything",
        "are",
        "avoid",
        "be",
        "below",
        "but",
        "can",
        "could",
        "do",
        "does",
        "except",
        "exclude",
        "excluding",
        "find",
        "for",
        "from",
        "get",
        "give",
        "i",
        "in",
        "is",
        "it",
        "looking",
        "me",
        "need",
        "no",
        "not",
        "of",
        "on",
        "or",
        "please",
        "prefer",
        "preferably",
        "really",
        "show",
        "some",
        "something",
        "than",
        "that",
        "the",
        "these",
        "things",
        "this",
        "those",
        "to",
        "under",
        "want",
        "with",
        "without",
        "would",
    }
)
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


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
    product: Mapping[str, object],
    *,
    character_limit: int = DEFAULT_DOCUMENT_CHAR_LIMIT,
) -> str:
    """Return the same bounded, high-signal text used by dense retrieval."""

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


def query_tokens(
    query: str,
    *,
    max_tokens: int = DEFAULT_MAX_QUERY_TOKENS,
) -> tuple[str, ...]:
    """Return bounded, de-duplicated tokens without exposing FTS syntax.

    The raw query is never passed to ``MATCH``.  Unicode normalization and the
    alphanumeric-only token pattern remove quotes, parentheses, wildcards,
    column selectors, and boolean operators that could alter the FTS query.
    """

    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    normalized = unicodedata.normalize("NFKC", str(query)).casefold()
    result: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall(normalized):
        if token in QUERY_STOPWORDS or token in seen:
            continue
        if len(token) < 2 and not token.isdigit():
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= max_tokens:
            break
    return tuple(result)


def fts5_query(
    query: str,
    *,
    max_tokens: int = DEFAULT_MAX_QUERY_TOKENS,
) -> str:
    """Compile user text into a safe token-level FTS5 OR expression."""

    # TOKEN_RE cannot emit a quote, but explicit replacement keeps this helper
    # safe if its tokenizer is broadened in the future.
    return " OR ".join(
        f'"{token.replace(chr(34), chr(34) * 2)}"'
        for token in query_tokens(query, max_tokens=max_tokens)
    )


def _catalog_rows(
    catalog_path: str | Path,
    *,
    document_builder: Callable[[Mapping[str, object]], str],
) -> tuple[list[tuple[str, str]], str]:
    rows: list[tuple[str, str]] = []
    identifiers: set[str] = set()
    digest = hashlib.sha256()
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            product = json.loads(line)
            parent_asin = str(product.get("parent_asin", "")).strip()
            if not parent_asin:
                raise ValueError(
                    f"catalogue line {line_number} has no parent_asin"
                )
            if parent_asin in identifiers:
                raise ValueError(
                    f"catalogue contains duplicate parent_asin: {parent_asin}"
                )
            identifiers.add(parent_asin)
            document = str(document_builder(product))
            rows.append((parent_asin, document))
            digest.update(parent_asin.encode("utf-8"))
            digest.update(b"\0")
            digest.update(document.encode("utf-8"))
            digest.update(b"\n")
    return rows, digest.hexdigest()


def build_lexical_artifact(
    catalog_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    document_builder: Callable[[Mapping[str, object]], str] = product_document,
) -> dict[str, object]:
    """Build an atomic, portable SQLite FTS5 catalogue artifact."""

    catalog_path = Path(catalog_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass overwrite=True to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".building")
    temporary_path.unlink(missing_ok=True)

    rows, catalog_sha256 = _catalog_rows(
        catalog_path, document_builder=document_builder
    )
    started = time.perf_counter()
    connection = sqlite3.connect(temporary_path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode = OFF;
            PRAGMA synchronous = OFF;

            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE product_fts USING fts5(
                parent_asin UNINDEXED,
                document,
                tokenize = 'porter unicode61 remove_diacritics 2'
            );
            """
        )
        connection.executemany(
            "INSERT INTO product_fts(parent_asin, document) VALUES (?, ?)", rows
        )
        metadata = {
            "artifact_version": ARTIFACT_VERSION,
            "catalog_sha256": catalog_sha256,
            "document_char_limit": str(DEFAULT_DOCUMENT_CHAR_LIMIT),
            "product_count": str(len(rows)),
            "tokenizer": "porter unicode61 remove_diacritics 2",
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        # Merge FTS segments once offline so runtime queries remain read-only.
        connection.execute("INSERT INTO product_fts(product_fts) VALUES ('optimize')")
        connection.commit()
    except Exception:
        connection.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        connection.close()

    temporary_path.replace(output_path)
    return {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_bytes": output_path.stat().st_size,
        "build_seconds": time.perf_counter() - started,
        "catalog_sha256": catalog_sha256,
        "output_path": str(output_path),
        "product_count": len(rows),
    }


@dataclass(frozen=True)
class LexicalHit:
    parent_asin: str
    score: float


class LexicalCatalogIndex:
    """Lazy read-only BM25 search over a prebuilt FTS5 artifact."""

    def __init__(
        self,
        artifact_path: str | Path,
        *,
        max_query_tokens: int = DEFAULT_MAX_QUERY_TOKENS,
    ) -> None:
        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"lexical artifact not found: {artifact_path}")
        if max_query_tokens < 0:
            raise ValueError("max_query_tokens must be non-negative")
        self.max_query_tokens = int(max_query_tokens)
        self._connection: sqlite3.Connection | None = None
        self.metadata: dict[str, str] | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        uri = self.artifact_path.resolve().as_uri() + "?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        try:
            connection.execute("PRAGMA query_only = ON")
            metadata = {
                str(key): str(value)
                for key, value in connection.execute(
                    "SELECT key, value FROM metadata"
                )
            }
            if metadata.get("artifact_version") != ARTIFACT_VERSION:
                raise ValueError("unsupported lexical artifact version")
            row = connection.execute(
                "SELECT COUNT(*) FROM product_fts"
            ).fetchone()
            product_count = 0 if row is None else int(row[0])
            if product_count != int(metadata.get("product_count", "-1")):
                raise ValueError("lexical artifact product count does not match metadata")
        except Exception:
            connection.close()
            raise
        self._connection = connection
        self.metadata = metadata
        return connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> "LexicalCatalogIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def search(
        self,
        query: str,
        candidate_ids: Iterable[str],
        *,
        limit: int | None = DEFAULT_LIMIT,
    ) -> list[LexicalHit]:
        """Return BM25 hits, restricted to caller-supplied candidate IDs.

        FTS retrieves matching documents in global BM25 order, then the explicit
        candidate set is applied before a result is appended.  No fallback IDs
        are manufactured when there is no lexical match.
        """

        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or None")
        if limit == 0:
            return []
        allowed = {str(value) for value in candidate_ids}
        expression = fts5_query(query, max_tokens=self.max_query_tokens)
        if not allowed or not expression:
            return []

        connection = self._connect()
        cursor = connection.execute(
            """
            SELECT parent_asin, bm25(product_fts, 0.0, 1.0) AS score
            FROM product_fts
            WHERE product_fts MATCH ?
            ORDER BY score, parent_asin
            """,
            (expression,),
        )
        hits: list[LexicalHit] = []
        for parent_asin, score in cursor:
            identifier = str(parent_asin)
            if identifier not in allowed:
                continue
            hits.append(LexicalHit(identifier, float(score)))
            if limit is not None and len(hits) >= limit:
                break
        return hits

    def lexical_rank(
        self,
        query: str,
        candidate_ids: Iterable[str],
        *,
        limit: int | None = DEFAULT_LIMIT,
    ) -> list[str]:
        """Return only identifiers from :meth:`search`, best match first."""

        return [
            hit.parent_asin
            for hit in self.search(query, candidate_ids, limit=limit)
        ]


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the free-form SQLite FTS5 catalogue index"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output", default="data/lexical_index.sqlite3")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    result = build_lexical_artifact(
        args.catalog,
        args.output,
        overwrite=args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
