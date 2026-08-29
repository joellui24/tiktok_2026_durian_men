from __future__ import annotations

import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path

from starter.hybrid_model import NO_ANSWER, PortableHybridModel


class PortableHybridModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "model.sqlite3"
        connection = sqlite3.connect(self.path)
        connection.executescript(
            """
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE context_features(
                feature_id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                field TEXT NOT NULL,
                vector BLOB NOT NULL
            );
            CREATE TABLE item_features(
                feature_id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                field TEXT NOT NULL,
                linear_weight REAL NOT NULL,
                vector BLOB NOT NULL
            );
            CREATE TABLE products(
                parent_asin TEXT PRIMARY KEY,
                base_score REAL NOT NULL,
                vector BLOB NOT NULL,
                item_feature_ids BLOB NOT NULL
            );
            CREATE TABLE cross_weights(
                context_feature_id INTEGER NOT NULL,
                item_feature_id INTEGER NOT NULL,
                positive_support INTEGER NOT NULL,
                negative_support INTEGER NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY(context_feature_id, item_feature_id)
            );
            CREATE TABLE reply_values(
                parent_asin TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                attribute TEXT NOT NULL,
                normalized_value TEXT NOT NULL,
                PRIMARY KEY(parent_asin, ordinal)
            );
            """
        )
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES(?,?)",
            (("schema_version", "1"), ("dimension", "2"), ("temperature", "1")),
        )
        connection.execute(
            "INSERT INTO context_features VALUES(0,?,?,?)",
            ("ctx:use_case=running", "use_case", struct.pack("<2f", 1.0, 0.0)),
        )
        connection.executemany(
            "INSERT INTO item_features VALUES(?,?,?,?,?)",
            (
                (0, "item:feature=cushioned", "feature", 0.0, struct.pack("<2f", 1.0, 0.0)),
                (1, "item:feature=plain", "feature", 0.0, struct.pack("<2f", 0.0, 1.0)),
            ),
        )
        connection.executemany(
            "INSERT INTO products VALUES(?,?,?,?)",
            (
                ("CUSHIONED", 0.0, struct.pack("<2f", 1.0, 0.0), struct.pack("<I", 0)),
                ("PLAIN", 0.0, struct.pack("<2f", 0.0, 1.0), struct.pack("<I", 1)),
            ),
        )
        connection.execute(
            "INSERT INTO cross_weights VALUES(0,0,25,40,1.5)"
        )
        connection.executemany(
            "INSERT INTO reply_values VALUES(?,?,?,?)",
            (
                ("CUSHIONED", 0, "use_case", "running"),
                ("CUSHIONED", 1, "feature", "cushioned"),
                ("PLAIN", 0, "feature", "plain"),
            ),
        )
        connection.commit()
        connection.close()
        self.model = PortableHybridModel(self.path)

    def test_latent_and_explicit_interactions_rank_supported_relationship(self) -> None:
        context = ["ctx:use_case=running"]
        fm_scores = self.model.score_many(
            ("CUSHIONED", "PLAIN"), context, mode="fm"
        )
        hybrid_scores = self.model.score_many(
            ("CUSHIONED", "PLAIN"), context, mode="hybrid"
        )
        self.assertGreater(fm_scores["CUSHIONED"], fm_scores["PLAIN"])
        self.assertAlmostEqual(
            hybrid_scores["CUSHIONED"] - fm_scores["CUSHIONED"], 1.5
        )
        self.assertEqual(
            self.model.rank(("PLAIN", "CUSHIONED"), context, 2),
            ["CUSHIONED", "PLAIN"],
        )

    def test_field_pair_ablation_removes_only_explicit_contribution(self) -> None:
        context = ["ctx:use_case=running"]
        fm = self.model.score_many(("CUSHIONED",), context, mode="fm")
        ablated = self.model.score_many(
            ("CUSHIONED",),
            context,
            mode="hybrid",
            disabled_field_pair=("use_case", "feature"),
        )
        self.assertEqual(fm, ablated)

    def test_reply_prediction_respects_disclosure_history(self) -> None:
        self.assertEqual(
            self.model.predicted_reply("CUSHIONED", "feature", set()),
            ("cushioned",),
        )
        self.assertEqual(
            self.model.predicted_reply("CUSHIONED", "feature", {"cushioned"}),
            (NO_ANSWER,),
        )
        self.assertEqual(
            self.model.predicted_reply("CUSHIONED", "other", {"running"}),
            ("cushioned",),
        )

    def test_scores_are_deterministic(self) -> None:
        context = ["ctx:use_case=running"]
        self.assertEqual(
            self.model.score_many(("CUSHIONED", "PLAIN"), context),
            self.model.score_many(("CUSHIONED", "PLAIN"), context),
        )


if __name__ == "__main__":
    unittest.main()
