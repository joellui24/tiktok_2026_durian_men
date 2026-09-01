from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from starter.attribute_index import build_attribute_database
from starter.category_index import build_category_database
from tests.free_form_retrieval_benchmark import (
    AgentAdapter,
    BENCHMARK_VERSION,
    CaseRelevance,
    CatalogRelevanceBuilder,
    EXPECTED_RELEVANCE_HASHES,
    EXPECTED_SELECTION_HASHES,
    RecommendationProvider,
    SELECTED_GROUPS,
    _dcg,
    aggregate_rows,
    benchmark_manifest,
    normalize_recommendations,
    score_ranked_list,
    selected_cases,
)
from tests.gliner_unseen_cases import QueryCase


class FrozenSelectionTests(unittest.TestCase):
    def test_selection_is_balanced_and_hash_validated(self) -> None:
        all_cases = selected_cases("all")
        development = selected_cases("development")
        confirmation = selected_cases("confirmation")

        self.assertEqual(len(all_cases), 60)
        self.assertEqual(len(development), 30)
        self.assertEqual(len(confirmation), 30)
        self.assertEqual({case.group for case in all_cases}, set(SELECTED_GROUPS))
        for group in SELECTED_GROUPS:
            self.assertEqual(sum(case.group == group for case in development), 5)
            self.assertEqual(sum(case.group == group for case in confirmation), 5)
        self.assertTrue(
            {case.case_id for case in development}.isdisjoint(
                case.case_id for case in confirmation
            )
        )

        manifest = benchmark_manifest()
        self.assertEqual(manifest["version"], BENCHMARK_VERSION)
        self.assertEqual(manifest["selection_hashes"], EXPECTED_SELECTION_HASHES)
        self.assertEqual(manifest["relevance_hashes"], EXPECTED_RELEVANCE_HASHES)

    def test_unknown_split_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            selected_cases("future")


class CatalogRelevanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.catalog_path = root / "catalog.jsonl"
        self.category_path = root / "category.sqlite3"
        self.attribute_path = root / "attribute.sqlite3"
        products = (
            {
                "parent_asin": "P1",
                "title": "Nike Black Leather Breathable Running Shoes",
                "features": ["Breathable upper for comfortable running"],
                "description": [],
                "price": 80.0,
                "categories": [
                    "Clothing, Shoes & Jewelry",
                    "Men",
                    "Shoes",
                    "Athletic",
                    "Running",
                ],
                "details": {},
                "average_rating": 4.5,
                "rating_number": 10,
                "store": "Nike",
            },
            {
                "parent_asin": "P2",
                "title": "Adidas Black Leather Breathable Running Shoes",
                "features": ["Breathable running design"],
                "description": [],
                "price": 70.0,
                "categories": [
                    "Clothing, Shoes & Jewelry",
                    "Men",
                    "Shoes",
                    "Athletic",
                    "Running",
                ],
                "details": {},
                "average_rating": 4.4,
                "rating_number": 8,
                "store": "Adidas",
            },
            {
                "parent_asin": "P3",
                "title": "Nike Black Leather Breathable Hiking Boots",
                "features": ["Breathable trail boot"],
                "description": [],
                "price": 75.0,
                "categories": [
                    "Clothing, Shoes & Jewelry",
                    "Men",
                    "Shoes",
                    "Outdoor",
                    "Hiking Boots",
                ],
                "details": {},
                "average_rating": 4.3,
                "rating_number": 6,
                "store": "Nike",
            },
            {
                "parent_asin": "P4",
                "title": "Nike Black Leather Running Shoes",
                "features": ["Firm racing shoe"],
                "description": [],
                "price": 90.0,
                "categories": [
                    "Clothing, Shoes & Jewelry",
                    "Men",
                    "Shoes",
                    "Athletic",
                    "Running",
                ],
                "details": {},
                "average_rating": 4.2,
                "rating_number": 5,
                "store": "Nike",
            },
            {
                "parent_asin": "P5",
                "title": "Nike Black Leather Breathable Running Shoes Premium",
                "features": ["Breathable running design"],
                "description": [],
                "price": 150.0,
                "categories": [
                    "Clothing, Shoes & Jewelry",
                    "Men",
                    "Shoes",
                    "Athletic",
                    "Running",
                ],
                "details": {},
                "average_rating": 4.8,
                "rating_number": 12,
                "store": "Nike",
            },
        )
        self.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        build_category_database(
            self.catalog_path, self.category_path, overwrite=False
        )
        build_attribute_database(
            self.catalog_path, self.attribute_path, overwrite=False
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_qrels_use_category_hard_constraints_and_canonical_evidence(self) -> None:
        case = QueryCase(
            case_id="SYN001",
            split="synthetic",
            group="feature",
            message="Black Nike leather runners under 100, breathable, size 11",
            intent="buying",
            category="running shoes",
            attributes={
                "brand": ("Nike",),
                "color": ("black",),
                "material": ("leather",),
                "size": ("11",),
                "feature": ("breathable",),
                "use_case": ("running",),
            },
            maximum_price=100.0,
        )
        with CatalogRelevanceBuilder(
            self.catalog_path,
            category_index_path=self.category_path,
            attribute_index_path=self.attribute_path,
        ) as builder:
            label = builder.for_case(case)
            self.assertEqual(label.hard_eligible, frozenset({"P1", "P4"}))
            self.assertEqual(label.grades["P1"], 3)
            # P4 has exact running evidence through its title/category but no
            # breathable evidence, so it is a partial semantic match.
            self.assertEqual(label.grades["P4"], 2)
            self.assertNotIn("P2", label.grades)  # wrong brand
            self.assertNotIn("P3", label.grades)  # wrong category
            self.assertNotIn("P5", label.grades)  # over budget
            self.assertEqual(label.unverifiable_constraints, {"size": ("11",)})
            self.assertEqual(
                builder.evidence_postings[("feature", "breathable")],
                frozenset({"P1", "P2", "P3", "P5"}),
            )


class MetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.relevance = CaseRelevance(
            case_id="Q1",
            group="browsing",
            message="A useful browsing request",
            grades={"A": 3, "B": 2, "C": 1},
            hard_eligible=frozenset({"A", "B", "C"}),
            hard_constraint_labels=("category:shoes",),
            soft_evidence_labels=("feature:comfort",),
        )
        self.leaves = {
            "A": "Shoes",
            "B": "Shoes",
            "C": "Dresses",
            "X": "Jewelry",
        }

    def test_rank_metrics_and_hard_violations(self) -> None:
        row = score_ranked_list(
            self.relevance, ["B", "X", "A", "C"], self.leaves
        )
        expected_ndcg = _dcg([2, 0, 3, 1]) / _dcg([3, 2, 1])
        self.assertAlmostEqual(row["ndcg_at_10"], expected_ndcg)
        self.assertEqual(row["success_at_10"], 1)
        self.assertEqual(row["first_relevant_rank"], 1)
        self.assertEqual(row["first_relevant_mrr"], 1.0)
        self.assertEqual(row["precision_at_10"], 0.2)
        self.assertEqual(row["hard_violations"], ["X"])
        self.assertFalse(row["hard_safe"])
        self.assertAlmostEqual(row["diversity_at_10"], 5 / 6)
        self.assertEqual(row["relevant_leaf_category_count_at_10"], 1)

    def test_aggregate_uses_scorable_queries_and_item_weighted_violations(self) -> None:
        first = score_ranked_list(self.relevance, ["B", "X", "A"], self.leaves)
        unscorable = CaseRelevance(
            case_id="Q2",
            group="feature",
            message="Impossible constraints",
            grades={},
            hard_eligible=frozenset(),
            hard_constraint_labels=("budget:<1",),
            soft_evidence_labels=(),
        )
        second = score_ranked_list(unscorable, [], self.leaves)
        aggregate = aggregate_rows([first, second], provider_name="fake")
        self.assertEqual(aggregate["query_count"], 2)
        self.assertEqual(aggregate["scorable_query_count"], 1)
        self.assertEqual(aggregate["unscorable_query_count"], 1)
        self.assertAlmostEqual(aggregate["hard_violation_rate"], 1 / 3)
        self.assertEqual(aggregate["success_at_10"], 1.0)

    def test_normalization_deduplicates_and_rejects_unknown_ids(self) -> None:
        ranked = normalize_recommendations(
            [
                {"parent_asin": "A"},
                "A",
                {"parent_asin": "missing"},
                "B",
            ],
            frozenset({"A", "B"}),
        )
        self.assertEqual(ranked, ["A", "B"])


class AdapterTests(unittest.TestCase):
    def test_agent_adapter_uses_public_agent_contract(self) -> None:
        class FakeAgent:
            def __init__(self) -> None:
                self.reset_calls: list[tuple[str, dict]] = []

            def reset(self, session_id: str, profile: dict) -> None:
                self.reset_calls.append((session_id, profile))

            def respond(
                self, session_id: str, message: str, turn: int, top_k: int
            ) -> dict:
                self.last_response_call = (session_id, message, turn, top_k)
                return {"recommendations": [{"parent_asin": "P1"}]}

        case = QueryCase(
            case_id="SYN002",
            split="synthetic",
            group="feature",
            message="A lightweight jacket",
        )
        fake = FakeAgent()
        adapter = AgentAdapter(fake, name="fake")
        self.assertIsInstance(adapter, RecommendationProvider)
        self.assertEqual(adapter.recommend(case, 10), [{"parent_asin": "P1"}])
        self.assertEqual(len(fake.reset_calls), 1)
        self.assertEqual(fake.last_response_call[1:], (case.message, 1, 10))


if __name__ == "__main__":
    unittest.main()
