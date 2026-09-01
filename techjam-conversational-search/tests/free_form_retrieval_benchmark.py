"""Frozen end-to-end retrieval benchmark for natural shopping requests.

This diagnostic is deliberately separate from ``evaluator`` and ``starter``.
It imports the already-frozen GLiNER unseen cases, derives relevance labels only
from their expected fields and immutable catalog evidence, and can evaluate any
adapter implementing :class:`RecommendationProvider`.

The benchmark does *not* use parser output, model scores, or retrieved products
to construct labels.  That separation makes it suitable for comparing the
current Agent with later lexical or semantic retrieval variants.

Run the current baseline from the project directory with::

    py -3.13 -B tests/free_form_retrieval_benchmark.py \
        --split development --output retrieval-baseline-development.json

The official evaluator and its public dataset are never imported or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starter.agent import Agent  # noqa: E402
from starter.attribute_index import AttributeIndex, normalize_value  # noqa: E402
from starter.category_index import CategoryIndex  # noqa: E402
from tests.gliner_unseen_cases import (  # noqa: E402
    CASES,
    FROZEN_CORPUS_SHA256,
    QueryCase,
    _stable_json_value,
    manifest_is_valid,
)


BENCHMARK_VERSION = "free-form-retrieval-v1"
CUTOFF = 10
RELEVANT_GRADE = 2

# Whole-group selection was declared before any semantic retrieval output was
# inspected.  Do not add or remove individual cases based on model results.
SELECTED_GROUPS = frozenset(
    {
        "paraphrase",
        "category_synonym",
        "hard_buying",
        "browsing",
        "vague_use_case",
        "feature",
    }
)
EXPECTED_SELECTION_HASHES = {
    "all": "48475e314eb496c7a622823e4d067efcd6b3fd4b92e83c6586d1d6a164262062",
    "development": "1c1889a20f32891a2bb7b6c0357f4a6c7e3a07493917fcf4eeb2920c85d2f1d1",
    "confirmation": "ab264661fbb5b37c1ecb40b4b9f026bde4509b89b6a5a78ebea81740a226449b",
}
EXPECTED_RELEVANCE_HASHES = {
    "all": "09fcce829fbb821c7c21f9050cd6336223baef18b3903eb57117a51ca2617784",
    "development": "307805afd4705b0627d4e0bc60b6846a63231e243832b1d8e7402586383e089d",
    "confirmation": "8c57a47e8dcecbfc6795b7d3ef8741faae0c4fee453f49cf04b92663abeefd99",
}

# These targets are copied into the benchmark so future production category
# rewrites cannot silently change frozen relevance labels.
FROZEN_CATEGORY_TARGETS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "running shoes": (("Running",), ("Shoes",)),
    "walking shoes": (("Walking",), ("Shoes",)),
    "sneakers": (("Fashion Sneakers", "Sneakers"), ("Shoes",)),
    "sandals": (("Sandals",), ("Shoes",)),
    "shoes": (("Shoes",), ()),
    "shirts": (("Shirts", "T-Shirts", "Blouses & Button-Down Shirts"), ()),
    "tunics": (("Tunics",), ()),
    "boots": (("Boots",), ("Shoes",)),
    "dresses": (("Dresses",), ()),
    "jackets": (("Jackets",), ()),
    "jeans": (("Jeans",), ()),
    "pants": (("Pants",), ()),
    "skirts": (("Skirts",), ()),
    "socks": (("Socks", "Athletic Socks"), ()),
    "belts": (("Belts",), ()),
    "watches": (("Wrist Watches",), ()),
    "earrings": (("Earrings",), ()),
    "necklaces": (("Necklaces",), ()),
    "rings": (("Rings",), ()),
    "slippers": (("Slippers",), ("Shoes",)),
    "loafers": (("Loafers & Slip-Ons",), ("Shoes",)),
    "pumps": (("Pumps",), ("Shoes",)),
    "flats": (("Flats",), ("Shoes",)),
    "hoodies": (("Fashion Hoodies & Sweatshirts",), ()),
    "hats": (("Hats & Caps",), ()),
    "sunglasses": (("Sunglasses",), ()),
}

# Exact, human-declared surface evidence for each canonical frozen label.  The
# list contains ordinary inflections or established lexical forms only; it is
# intentionally not expanded from retrieval/model output.
CANONICAL_EVIDENCE_TERMS: dict[str, dict[str, tuple[str, ...]]] = {
    "feature": {
        "breathable": ("breathable", "breathability"),
        "comfort": ("comfort", "comfortable", "comfy"),
        "cushioned": ("cushion", "cushioned", "cushioning"),
        "durable": ("durable", "durability"),
        "lightweight": ("lightweight", "light weight"),
        "polarized": ("polarized", "polarised"),
        "soft": ("soft",),
        "style": ("style", "stylish", "fashionable"),
        "supportive": ("support", "supportive", "arch support"),
        "warm": ("warm", "warmth"),
        "waterproof": ("waterproof", "water proof"),
    },
    "use_case": {
        "beach": ("beach",),
        "formal": ("formal", "black tie", "gala"),
        "gym": ("gym", "workout", "fitness"),
        "hiking": ("hike", "hiking", "trail"),
        "lounge": ("lounge", "lounging", "loungewear"),
        "outdoor": ("outdoor", "outdoors"),
        "running": ("run", "running", "jogging"),
        "summer": ("summer",),
        "travel": ("travel", "traveling", "travelling"),
        "walking": ("walk", "walking"),
        "winter": ("winter",),
        "work": ("work", "office"),
    },
}

# The existing catalog-backed Agent treats these exact fields as product-
# removing constraints.  Size is retained in case metadata but is not judged as
# a hard violation: parent-ASIN metadata does not expose a canonical variant
# size, and the attribute index stores free-text size statements rather than
# normalized values such as ``11`` or ``xl``.
VERIFIABLE_HARD_ATTRIBUTES = ("brand", "color", "material")
UNVERIFIABLE_EXPECTED_ATTRIBUTES = ("size",)

SEARCHABLE_FIELDS = (
    "title",
    "features",
    "details",
    "description",
    "categories",
    "store",
)


def _selection_hash(cases: Sequence[QueryCase]) -> str:
    encoded = json.dumps(
        _stable_json_value(tuple(cases)),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def selected_cases(split: str = "all") -> tuple[QueryCase, ...]:
    """Return the immutable whole-group retrieval selection.

    Odd numeric case IDs form development and even IDs form confirmation,
    preserving five examples from every selected group in each half.
    """

    if split not in EXPECTED_SELECTION_HASHES:
        raise ValueError(
            "split must be one of: " + ", ".join(EXPECTED_SELECTION_HASHES)
        )
    cases = tuple(case for case in CASES if case.group in SELECTED_GROUPS)
    if split != "all":
        parity = 1 if split == "development" else 0
        cases = tuple(
            case for case in cases if int(case.case_id.removeprefix("GLQ")) % 2 == parity
        )
    expected = EXPECTED_SELECTION_HASHES[split]
    actual = _selection_hash(cases)
    if not manifest_is_valid() or actual != expected:
        raise RuntimeError(
            "frozen retrieval benchmark manifest mismatch: "
            f"source={FROZEN_CORPUS_SHA256}, split={split}, "
            f"expected={expected}, actual={actual}"
        )
    return cases


def benchmark_manifest() -> dict[str, Any]:
    return {
        "version": BENCHMARK_VERSION,
        "source_corpus_sha256": FROZEN_CORPUS_SHA256,
        "selected_groups": sorted(SELECTED_GROUPS),
        "selection_hashes": dict(EXPECTED_SELECTION_HASHES),
        "relevance_hashes": dict(EXPECTED_RELEVANCE_HASHES),
        "selection_counts": {
            split: len(selected_cases(split)) for split in EXPECTED_SELECTION_HASHES
        },
        "cutoff": CUTOFF,
        "relevant_grade": RELEVANT_GRADE,
        "verifiable_hard_attributes": list(VERIFIABLE_HARD_ATTRIBUTES),
        "unverifiable_expected_attributes": list(
            UNVERIFIABLE_EXPECTED_ATTRIBUTES
        ),
    }


def _flatten(value: object) -> list[str]:
    if isinstance(value, dict):
        return [
            f"{key} {item}"
            for key, item in value.items()
            if item not in (None, "", [])
        ]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _evidence_text(product: Mapping[str, Any]) -> str:
    raw = " ".join(
        part
        for field_name in SEARCHABLE_FIELDS
        for part in _flatten(product.get(field_name))
    )
    normalized = unicodedata.normalize("NFKC", raw).casefold()
    normalized = "".join(character if character.isalnum() else " " for character in normalized)
    return " ".join(normalized.split())


@dataclass(frozen=True)
class CaseRelevance:
    """Frozen catalog-derived relevance information for one query."""

    case_id: str
    group: str
    message: str
    grades: Mapping[str, int]
    hard_eligible: frozenset[str]
    hard_constraint_labels: tuple[str, ...]
    soft_evidence_labels: tuple[str, ...]
    unverifiable_constraints: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )

    @property
    def scorable(self) -> bool:
        return any(grade >= RELEVANT_GRADE for grade in self.grades.values())


class CatalogRelevanceBuilder:
    """Create qrels from frozen expectations and exact catalog evidence."""

    def __init__(
        self,
        catalog_path: str | Path,
        *,
        category_index_path: str | Path | None = None,
        attribute_index_path: str | Path | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        data_directory = self.catalog_path.parent
        self.category_index = CategoryIndex(
            category_index_path or data_directory / "category_index.sqlite3"
        )
        self.attribute_index = AttributeIndex(
            attribute_index_path or data_directory / "attribute_index.sqlite3"
        )
        self.catalog_ids: frozenset[str]
        self.leaf_categories: dict[str, str] = {}
        texts: dict[str, str] = {}
        identifiers: set[str] = set()
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                identifiers.add(parent_asin)
                categories = product.get("categories") or []
                self.leaf_categories[parent_asin] = (
                    str(categories[-1]) if categories else "<uncategorized>"
                )
                texts[parent_asin] = _evidence_text(product)
        self.catalog_ids = frozenset(identifiers)

        required_evidence = {
            (field_name, value)
            for case in selected_cases("all")
            for field_name in CANONICAL_EVIDENCE_TERMS
            for value in case.attributes.get(field_name, ())
        }
        self.evidence_postings: dict[tuple[str, str], frozenset[str]] = {}
        for field_name, value in sorted(required_evidence):
            try:
                terms = CANONICAL_EVIDENCE_TERMS[field_name][value]
            except KeyError as error:
                raise RuntimeError(
                    f"no frozen canonical evidence terms for {field_name}={value}"
                ) from error
            normalized_terms = tuple(
                _evidence_text({"title": term}) for term in terms
            )
            self.evidence_postings[(field_name, value)] = frozenset(
                parent_asin
                for parent_asin, text in texts.items()
                if any(
                    f" {term} " in f" {text} " for term in normalized_terms
                )
            )
        del texts

        self._hard_maps = {
            attribute: self.attribute_index.load_hashmap(attribute)
            for attribute in VERIFIABLE_HARD_ATTRIBUTES
        }
        self._category_cache: dict[str, frozenset[str]] = {}
        self._budget_cache: dict[float, frozenset[str]] = {}

    def close(self) -> None:
        self.category_index.close()
        self.attribute_index.close()

    def __enter__(self) -> "CatalogRelevanceBuilder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _category_products(self, category: str) -> frozenset[str]:
        cached = self._category_cache.get(category)
        if cached is not None:
            return cached
        target = FROZEN_CATEGORY_TARGETS.get(category)
        if target is None:
            result = frozenset()
        else:
            names, path_terms = target
            result = frozenset(
                self.category_index.products_for_category_names(
                    names, required_path_terms=path_terms
                )
            )
        self._category_cache[category] = result
        return result

    def _budget_products(self, maximum: float) -> frozenset[str]:
        maximum = float(maximum)
        cached = self._budget_cache.get(maximum)
        if cached is None:
            cached = frozenset(
                self.attribute_index.filter_products(maximum_price=maximum)
            )
            self._budget_cache[maximum] = cached
        return cached

    def for_case(self, case: QueryCase) -> CaseRelevance:
        eligible = set(self.catalog_ids)
        hard_labels: list[str] = []
        if case.category:
            eligible.intersection_update(self._category_products(case.category))
            hard_labels.append(f"category:{case.category}")

        for attribute in VERIFIABLE_HARD_ATTRIBUTES:
            for value in case.attributes.get(attribute, ()):
                eligible.intersection_update(
                    self._hard_maps[attribute].get(normalize_value(value), ())
                )
                hard_labels.append(f"{attribute}:{value}")

        if case.maximum_price is not None:
            eligible.intersection_update(self._budget_products(case.maximum_price))
            hard_labels.append(f"budget:<{case.maximum_price:g}")

        soft_pairs = tuple(
            (field_name, value)
            for field_name in ("feature", "use_case")
            for value in case.attributes.get(field_name, ())
        )
        soft_labels = tuple(
            f"{field_name}:{value}" for field_name, value in soft_pairs
        )
        postings = tuple(self.evidence_postings[pair] for pair in soft_pairs)
        scoped_request = bool(hard_labels)
        grades: dict[str, int] = {}
        for parent_asin in eligible:
            if not postings:
                grades[parent_asin] = 2
                continue
            matched = sum(parent_asin in values for values in postings)
            if matched == len(postings):
                grades[parent_asin] = 3
            elif matched:
                grades[parent_asin] = 2
            elif scoped_request:
                # Correct product type/hard constraints, but no catalog evidence
                # for the requested soft meaning.
                grades[parent_asin] = 1

        unverifiable = {
            attribute: tuple(case.attributes.get(attribute, ()))
            for attribute in UNVERIFIABLE_EXPECTED_ATTRIBUTES
            if case.attributes.get(attribute)
        }
        return CaseRelevance(
            case_id=case.case_id,
            group=case.group,
            message=case.message,
            grades=grades,
            hard_eligible=frozenset(eligible),
            hard_constraint_labels=tuple(hard_labels),
            soft_evidence_labels=soft_labels,
            unverifiable_constraints=unverifiable,
        )

    def build(
        self, cases: Sequence[QueryCase]
    ) -> dict[str, CaseRelevance]:
        return {case.case_id: self.for_case(case) for case in cases}


def relevance_hash(labels: Mapping[str, CaseRelevance]) -> str:
    """Hash every deterministic qrel and hard-eligibility decision."""

    payload = []
    for case_id in sorted(labels):
        label = labels[case_id]
        payload.append(
            {
                "case_id": case_id,
                "grades": sorted(label.grades.items()),
                "hard_eligible": sorted(label.hard_eligible),
                "hard_constraint_labels": list(label.hard_constraint_labels),
                "soft_evidence_labels": list(label.soft_evidence_labels),
                "unverifiable_constraints": {
                    key: list(values)
                    for key, values in sorted(
                        label.unverifiable_constraints.items()
                    )
                },
            }
        )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_relevance_hash(
    labels: Mapping[str, CaseRelevance], split: str
) -> str:
    if split not in EXPECTED_RELEVANCE_HASHES:
        raise ValueError(
            "split must be one of: " + ", ".join(EXPECTED_RELEVANCE_HASHES)
        )
    actual = relevance_hash(labels)
    expected = EXPECTED_RELEVANCE_HASHES[split]
    if actual != expected:
        raise RuntimeError(
            "frozen catalog relevance manifest mismatch: "
            f"split={split}, expected={expected}, actual={actual}"
        )
    return actual


@runtime_checkable
class RecommendationProvider(Protocol):
    """Minimal adapter seam shared by Agent and experimental retrievers."""

    name: str

    def recommend(self, case: QueryCase, top_k: int) -> Sequence[object]:
        """Return ordered ASIN strings or recommendation dictionaries."""


class AgentAdapter:
    """Adapt any evaluator-compatible Agent without changing that Agent."""

    def __init__(self, agent: Any, *, name: str = "agent") -> None:
        self.agent = agent
        self.name = name

    def recommend(self, case: QueryCase, top_k: int) -> Sequence[object]:
        session_id = f"retrieval-benchmark-{self.name}-{case.case_id}"
        self.agent.reset(session_id, {})
        response = self.agent.respond(session_id, case.message, 1, top_k)
        if not isinstance(response, Mapping):
            return ()
        recommendations = response.get("recommendations")
        return (
            recommendations
            if isinstance(recommendations, Sequence)
            and not isinstance(recommendations, (str, bytes))
            else ()
        )


@dataclass
class CallableAdapter:
    """Convenience adapter for a retrieval function used by experiments."""

    name: str
    callback: Callable[[QueryCase, int], Sequence[object]]

    def recommend(self, case: QueryCase, top_k: int) -> Sequence[object]:
        return self.callback(case, top_k)


def normalize_recommendations(
    payload: Sequence[object], catalog_ids: frozenset[str], *, cutoff: int = CUTOFF
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in payload:
        value = item.get("parent_asin", "") if isinstance(item, Mapping) else item
        parent_asin = str(value).strip()
        if not parent_asin or parent_asin in seen or parent_asin not in catalog_ids:
            continue
        seen.add(parent_asin)
        result.append(parent_asin)
        if len(result) >= cutoff:
            break
    return result


def _dcg(grades: Sequence[int]) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )


def _pairwise_category_diversity(categories: Sequence[str]) -> float:
    pair_count = len(categories) * (len(categories) - 1) // 2
    if not pair_count:
        return 0.0
    different = sum(
        left != right
        for position, left in enumerate(categories)
        for right in categories[position + 1 :]
    )
    return different / pair_count


def score_ranked_list(
    relevance: CaseRelevance,
    ranked: Sequence[str],
    leaf_categories: Mapping[str, str],
    *,
    cutoff: int = CUTOFF,
) -> dict[str, Any]:
    ranked = list(ranked[:cutoff])
    grades = [int(relevance.grades.get(parent_asin, 0)) for parent_asin in ranked]
    ideal = sorted(relevance.grades.values(), reverse=True)[:cutoff]
    ideal_dcg = _dcg(ideal)
    ndcg = _dcg(grades) / ideal_dcg if ideal_dcg else None
    first_relevant_rank = next(
        (rank for rank, grade in enumerate(grades, start=1) if grade >= RELEVANT_GRADE),
        None,
    )
    relevant_ids = [
        parent_asin
        for parent_asin, grade in zip(ranked, grades, strict=True)
        if grade >= RELEVANT_GRADE
    ]
    hard_violations = [
        parent_asin for parent_asin in ranked if parent_asin not in relevance.hard_eligible
    ]
    categories = [leaf_categories.get(parent_asin, "<unknown>") for parent_asin in ranked]
    relevant_categories = {
        leaf_categories.get(parent_asin, "<unknown>") for parent_asin in relevant_ids
    }
    return {
        "case_id": relevance.case_id,
        "group": relevance.group,
        "message": relevance.message,
        "scorable": relevance.scorable,
        "ranked": ranked,
        "grades": grades,
        "ndcg_at_10": ndcg,
        "success_at_10": None
        if not relevance.scorable
        else int(first_relevant_rank is not None),
        "first_relevant_rank": first_relevant_rank,
        "first_relevant_mrr": None
        if not relevance.scorable
        else (0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank),
        "precision_at_10": None
        if not relevance.scorable
        else len(relevant_ids) / cutoff,
        "hard_violation_count": len(hard_violations),
        "hard_violations": hard_violations,
        "hard_safe": not hard_violations,
        "diversity_at_10": _pairwise_category_diversity(categories),
        "relevant_leaf_category_count_at_10": len(relevant_categories),
        "hard_constraint_labels": list(relevance.hard_constraint_labels),
        "soft_evidence_labels": list(relevance.soft_evidence_labels),
        "unverifiable_constraints": {
            key: list(values)
            for key, values in relevance.unverifiable_constraints.items()
        },
        "hard_eligible_count": len(relevance.hard_eligible),
        "relevant_product_count": sum(
            grade >= RELEVANT_GRADE for grade in relevance.grades.values()
        ),
    }


def _mean_present(rows: Sequence[Mapping[str, Any]], field_name: str) -> float | None:
    values = [float(row[field_name]) for row in rows if row.get(field_name) is not None]
    return statistics.fmean(values) if values else None


def aggregate_rows(rows: Sequence[Mapping[str, Any]], *, provider_name: str) -> dict[str, Any]:
    scorable = [row for row in rows if row["scorable"]]
    browsing = [row for row in rows if row["group"] == "browsing"]
    returned = sum(len(row["ranked"]) for row in rows)
    violations = sum(int(row["hard_violation_count"]) for row in rows)
    by_group: dict[str, dict[str, Any]] = {}
    for group in sorted({str(row["group"]) for row in rows}):
        group_rows = [row for row in rows if row["group"] == group]
        by_group[group] = {
            "query_count": len(group_rows),
            "scorable_query_count": sum(bool(row["scorable"]) for row in group_rows),
            "ndcg_at_10": _mean_present(group_rows, "ndcg_at_10"),
            "success_at_10": _mean_present(group_rows, "success_at_10"),
            "first_relevant_mrr": _mean_present(group_rows, "first_relevant_mrr"),
            "precision_at_10": _mean_present(group_rows, "precision_at_10"),
            "hard_safe_query_rate": statistics.fmean(
                float(bool(row["hard_safe"])) for row in group_rows
            ),
        }
    return {
        "provider": provider_name,
        "query_count": len(rows),
        "scorable_query_count": len(scorable),
        "unscorable_query_count": len(rows) - len(scorable),
        "ndcg_at_10": _mean_present(rows, "ndcg_at_10"),
        "success_at_10": _mean_present(rows, "success_at_10"),
        "first_relevant_mrr": _mean_present(rows, "first_relevant_mrr"),
        "precision_at_10": _mean_present(rows, "precision_at_10"),
        "hard_violation_rate": violations / returned if returned else 0.0,
        "hard_safe_query_rate": statistics.fmean(
            float(bool(row["hard_safe"])) for row in rows
        )
        if rows
        else 0.0,
        "diversity_at_10": _mean_present(browsing, "diversity_at_10"),
        "browsing_relevant_leaf_category_count_at_10": _mean_present(
            browsing, "relevant_leaf_category_count_at_10"
        ),
        "by_group": by_group,
    }


def evaluate_provider(
    provider: RecommendationProvider,
    cases: Sequence[QueryCase],
    relevance: Mapping[str, CaseRelevance],
    *,
    catalog_ids: frozenset[str],
    leaf_categories: Mapping[str, str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for case in cases:
        try:
            payload = provider.recommend(case, CUTOFF)
            ranked = normalize_recommendations(payload, catalog_ids)
        except Exception as error:  # diagnostic parity with evaluator safety
            ranked = []
            errors.append(
                {
                    "case_id": case.case_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        rows.append(
            score_ranked_list(relevance[case.case_id], ranked, leaf_categories)
        )
    return {
        "manifest": benchmark_manifest(),
        "aggregate": aggregate_rows(rows, provider_name=provider.name),
        "errors": errors,
        "cases": rows,
    }


def label_summary(
    labels: Mapping[str, CaseRelevance], *, relevance_sha256: str | None = None
) -> dict[str, Any]:
    relevant_counts = [
        sum(grade >= RELEVANT_GRADE for grade in label.grades.values())
        for label in labels.values()
    ]
    hard_eligible_counts = [len(label.hard_eligible) for label in labels.values()]
    return {
        "case_count": len(labels),
        "relevance_sha256": relevance_sha256 or relevance_hash(labels),
        "scorable_case_count": sum(label.scorable for label in labels.values()),
        "zero_relevant_case_ids": sorted(
            case_id for case_id, label in labels.items() if not label.scorable
        ),
        "cases_with_unverifiable_constraints": sorted(
            case_id
            for case_id, label in labels.items()
            if label.unverifiable_constraints
        ),
        "relevant_products": {
            "minimum": min(relevant_counts, default=0),
            "median": statistics.median(relevant_counts) if relevant_counts else 0,
            "maximum": max(relevant_counts, default=0),
        },
        "hard_eligible_products": {
            "minimum": min(hard_eligible_counts, default=0),
            "median": statistics.median(hard_eligible_counts)
            if hard_eligible_counts
            else 0,
            "maximum": max(hard_eligible_counts, default=0),
        },
        "groups": dict(sorted(Counter(label.group for label in labels.values()).items())),
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen catalog-derived free-form retrieval benchmark"
    )
    parser.add_argument(
        "--split",
        choices=tuple(EXPECTED_SELECTION_HASHES),
        default="development",
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--category-index")
    parser.add_argument("--attribute-index")
    parser.add_argument("--output")
    parser.add_argument(
        "--retrieval-mode",
        choices=("off", "lexical", "dense", "hybrid"),
        default="off",
        help="free-form candidate-ranking architecture to benchmark",
    )
    parser.add_argument("--lexical-index")
    parser.add_argument("--semantic-artifact")
    parser.add_argument("--semantic-model-path")
    parser.add_argument("--semantic-cache-dir")
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="build and summarize frozen qrels without running an Agent",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    cases = selected_cases(args.split)
    with CatalogRelevanceBuilder(
        args.catalog,
        category_index_path=args.category_index,
        attribute_index_path=args.attribute_index,
    ) as builder:
        labels = builder.build(cases)
        relevance_sha256 = validate_relevance_hash(labels, args.split)
        payload: dict[str, Any] = {
            "manifest": benchmark_manifest(),
            "split": args.split,
            "labels": label_summary(
                labels, relevance_sha256=relevance_sha256
            ),
        }
        if not args.labels_only:
            with Agent(
                args.catalog,
                category_index_path=args.category_index,
                attribute_index_path=args.attribute_index,
                free_form_retrieval_mode=args.retrieval_mode,
                lexical_index_path=args.lexical_index,
                semantic_artifact_path=args.semantic_artifact,
                semantic_model_path=args.semantic_model_path,
                semantic_cache_dir=args.semantic_cache_dir,
            ) as agent:
                result = evaluate_provider(
                    AgentAdapter(agent, name=f"agent-{args.retrieval_mode}"),
                    cases,
                    labels,
                    catalog_ids=builder.catalog_ids,
                    leaf_categories=builder.leaf_categories,
                )
            payload["result"] = result

    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "split": args.split,
                    "labels": payload["labels"],
                    "aggregate": payload.get("result", {}).get("aggregate"),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
