from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from starter.semantic_retrieval import (
    ARTIFACT_VERSION,
    SemanticCatalogIndex,
    product_document,
    reciprocal_rank_fusion,
)


class FakeEmbeddingModel:
    def __init__(self, vector: list[float]) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)
        self.calls = 0

    def query_embed(self, queries: list[str]):
        self.calls += 1
        self.last_queries = list(queries)
        yield self.vector


class SemanticRetrievalTest(unittest.TestCase):
    def test_document_prioritizes_high_signal_fields_and_is_bounded(self) -> None:
        document = product_document(
            {
                "title": "Walking Shoe",
                "categories": ["Shoes", "Walking"],
                "store": "Example",
                "features": ["Cushioned all-day comfort", "Breathable mesh"],
                "description": ["A long description"],
                "details": {"Date First Available": "today"},
            },
            character_limit=120,
        )
        self.assertLessEqual(len(document), 120)
        self.assertIn("Walking Shoe", document)
        self.assertIn("Shoes > Walking", document)
        self.assertNotIn("Date First Available", document)

    def test_rrf_uses_rank_not_incomparable_raw_scores(self) -> None:
        fused = reciprocal_rank_fusion(
            [["A", "B", "C"], ["B", "C", "D"]],
            weights=[1.0, 1.0],
            k=10,
            limit=4,
        )
        self.assertEqual(fused[0], "B")
        self.assertEqual(set(fused), {"A", "B", "C", "D"})

    def test_dense_rank_never_introduces_non_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "semantic.npz"
            np.savez_compressed(
                path,
                identifiers=np.asarray(["A", "B", "C"], dtype="U16"),
                embeddings=np.asarray(
                    [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float16
                ),
                metadata=np.asarray(
                    json.dumps(
                        {
                            "artifact_version": ARTIFACT_VERSION,
                            "embedding_dimension": 2,
                            "model_name": "fake",
                            "product_count": 3,
                        }
                    )
                ),
            )
            index = SemanticCatalogIndex(path, model_name="fake")
            fake = FakeEmbeddingModel([1.0, 0.0])
            index._model = fake
            ranked = index.dense_rank("comfort", {"B", "C"})
            self.assertEqual(ranked, ["C", "B"])
            self.assertNotIn("A", ranked)
            self.assertEqual(fake.calls, 1)

    def test_missing_artifact_fails_closed(self) -> None:
        with self.assertRaises(FileNotFoundError):
            SemanticCatalogIndex("does-not-exist.npz")

    def test_artifact_metadata_and_identifiers_are_validated(self) -> None:
        invalid_payloads = (
            (
                "duplicates",
                ["A", "A"],
                [[1.0, 0.0], [0.0, 1.0]],
                {"product_count": 2, "embedding_dimension": 2, "model_name": "fake"},
            ),
            (
                "count",
                ["A", "B"],
                [[1.0, 0.0], [0.0, 1.0]],
                {"product_count": 3, "embedding_dimension": 2, "model_name": "fake"},
            ),
            (
                "dimension",
                ["A", "B"],
                [[1.0, 0.0], [0.0, 1.0]],
                {"product_count": 2, "embedding_dimension": 3, "model_name": "fake"},
            ),
            (
                "model",
                ["A", "B"],
                [[1.0, 0.0], [0.0, 1.0]],
                {"product_count": 2, "embedding_dimension": 2, "model_name": "other"},
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for label, identifiers, embeddings, metadata in invalid_payloads:
                with self.subTest(label=label):
                    path = Path(directory) / f"{label}.npz"
                    np.savez_compressed(
                        path,
                        identifiers=np.asarray(identifiers, dtype="U16"),
                        embeddings=np.asarray(embeddings, dtype=np.float16),
                        metadata=np.asarray(
                            json.dumps(
                                {"artifact_version": ARTIFACT_VERSION, **metadata}
                            )
                        ),
                    )
                    with self.assertRaises(ValueError):
                        SemanticCatalogIndex(path, model_name="fake")


if __name__ == "__main__":
    unittest.main()
