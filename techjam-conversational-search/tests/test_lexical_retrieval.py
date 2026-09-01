from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from starter.lexical_retrieval import (
    ARTIFACT_VERSION,
    LexicalCatalogIndex,
    build_lexical_artifact,
    fts5_query,
    product_document,
    query_tokens,
)
from starter.semantic_retrieval import product_document as semantic_product_document


PRODUCTS = (
    {
        "parent_asin": "BREATHABLE",
        "title": "Airy Summer Walking Shirt",
        "features": [
            "Breathable mesh keeps you cool",
            "Lightweight fabric for hot weather",
        ],
        "description": ["Comfortable for long days outside."],
        "categories": ["Clothing", "Shirts"],
        "store": "Breeze",
    },
    {
        "parent_asin": "WINTER",
        "title": "Insulated Winter Coat",
        "features": ["Warm wool lining", "Wind resistant outer shell"],
        "description": ["For cold and snowy days."],
        "categories": ["Clothing", "Coats"],
        "store": "North",
    },
    {
        "parent_asin": "COMFORT",
        "title": "Cushioned Walking Shoes",
        "features": ["All-day comfort", "Supportive cushioned sole"],
        "description": ["Suitable for long walks."],
        "categories": ["Shoes", "Walking"],
        "store": "Stride",
    },
)


class LexicalRetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        root = Path(self.temporary_directory.name)
        self.catalog_path = root / "catalog.jsonl"
        self.artifact_path = root / "lexical.sqlite3"
        with self.catalog_path.open("w", encoding="utf-8") as handle:
            for product in PRODUCTS:
                handle.write(json.dumps(product) + "\n")
        self.build_result = build_lexical_artifact(
            self.catalog_path, self.artifact_path
        )

    def test_document_is_exactly_compatible_with_semantic_retrieval(self) -> None:
        self.assertEqual(
            product_document(PRODUCTS[0]), semantic_product_document(PRODUCTS[0])
        )
        self.assertEqual(
            product_document(PRODUCTS[0], character_limit=91),
            semantic_product_document(PRODUCTS[0], character_limit=91),
        )

    def test_builder_records_metadata_and_refuses_implicit_overwrite(self) -> None:
        self.assertEqual(self.build_result["artifact_version"], ARTIFACT_VERSION)
        self.assertEqual(self.build_result["product_count"], len(PRODUCTS))
        self.assertGreater(self.build_result["artifact_bytes"], 0)
        with closing(sqlite3.connect(self.artifact_path)) as connection:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        self.assertEqual(metadata["artifact_version"], ARTIFACT_VERSION)
        self.assertEqual(int(metadata["product_count"]), len(PRODUCTS))
        with self.assertRaises(FileExistsError):
            build_lexical_artifact(self.catalog_path, self.artifact_path)

    def test_query_compiler_removes_fts_syntax_and_bounds_tokens(self) -> None:
        malicious = 'black" OR parent_asin:* (summer) NOT winter'
        self.assertEqual(
            query_tokens(malicious, max_tokens=3),
            ("black", "parent", "asin"),
        )
        expression = fts5_query(malicious, max_tokens=3)
        self.assertEqual(expression, '"black" OR "parent" OR "asin"')
        self.assertNotIn("*", expression)
        self.assertNotIn("parent_asin:", expression)

    def test_natural_wording_retrieves_morphological_matches(self) -> None:
        with LexicalCatalogIndex(self.artifact_path) as index:
            breathable = index.lexical_rank(
                "I want something breathable for summer",
                {"BREATHABLE", "WINTER", "COMFORT"},
            )
            walking = index.lexical_rank(
                "comfortable for walking all day",
                {"BREATHABLE", "WINTER", "COMFORT"},
            )
        self.assertEqual(breathable[0], "BREATHABLE")
        self.assertEqual(walking[0], "COMFORT")

    def test_search_never_returns_an_identifier_outside_candidates(self) -> None:
        index = LexicalCatalogIndex(self.artifact_path)
        self.addCleanup(index.close)
        ranked = index.lexical_rank(
            "breathable summer walking",
            {"WINTER", "COMFORT", "NOT_IN_ARTIFACT"},
        )
        self.assertTrue(set(ranked).issubset({"WINTER", "COMFORT"}))
        self.assertNotIn("BREATHABLE", ranked)
        self.assertNotIn("NOT_IN_ARTIFACT", ranked)

    def test_no_match_returns_empty_instead_of_restoring_candidates(self) -> None:
        with LexicalCatalogIndex(self.artifact_path) as index:
            self.assertEqual(
                index.lexical_rank("diamond tiara", {"WINTER", "COMFORT"}), []
            )

    def test_runtime_connection_is_lazy_and_read_only(self) -> None:
        index = LexicalCatalogIndex(self.artifact_path)
        self.addCleanup(index.close)
        self.assertIsNone(index._connection)
        self.assertIsNone(index.metadata)
        index.lexical_rank("winter coat", {"WINTER"})
        self.assertIsNotNone(index._connection)
        self.assertEqual(index.metadata["artifact_version"], ARTIFACT_VERSION)
        with self.assertRaises(sqlite3.OperationalError):
            index._connection.execute("DELETE FROM metadata")

    def test_empty_inputs_and_zero_limit_do_not_open_artifact(self) -> None:
        index = LexicalCatalogIndex(self.artifact_path)
        self.addCleanup(index.close)
        self.assertEqual(index.lexical_rank("", {"WINTER"}), [])
        self.assertEqual(index.lexical_rank("winter", set()), [])
        self.assertEqual(index.lexical_rank("winter", {"WINTER"}, limit=0), [])
        self.assertIsNone(index._connection)

    def test_invalid_limit_is_rejected(self) -> None:
        with LexicalCatalogIndex(self.artifact_path) as index:
            with self.assertRaises(ValueError):
                index.lexical_rank("winter", {"WINTER"}, limit=-1)


if __name__ == "__main__":
    unittest.main()
