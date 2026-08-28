from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path

import numpy as np


TRAINER_PATH = Path(__file__).resolve().parents[2] / "approach 1" / "train_fm.py"
SPEC = importlib.util.spec_from_file_location("approach1_train_fm_tests", TRAINER_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAINER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAINER
SPEC.loader.exec_module(TRAINER)


class FactorizationTrainingTest(unittest.TestCase):
    def test_product_split_is_stable(self) -> None:
        first = [TRAINER.split_for(f"A{index}") for index in range(100)]
        second = [TRAINER.split_for(f"A{index}") for index in range(100)]
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"train", "validation", "test"})

    def test_supported_cross_is_learned_from_hard_negatives(self) -> None:
        products = []
        for index in range(25):
            products.append(
                TRAINER.Product(
                    f"RUN{index:02d}",
                    "shoes",
                    (
                        ("use_case", "running", "running"),
                        ("feature", "cushioned", "cushioned"),
                    ),
                    "brand",
                    "50_75",
                    "4.5",
                    "5",
                )
            )
        for index in range(25):
            products.append(
                TRAINER.Product(
                    f"PLAIN{index:02d}",
                    "shoes",
                    (
                        ("use_case", "outdoor", "outdoor"),
                        ("feature", "plain", "plain"),
                    ),
                    "brand",
                    "50_75",
                    "4.5",
                    "5",
                )
            )

        TRAINER._SPLIT_CACHE = ["train"] * len(products)
        features = TRAINER.build_feature_data(products)
        negatives = np.empty(
            (len(features.states), TRAINER.NEGATIVES_PER_STATE),
            dtype=np.int32,
        )
        for state_index, state in enumerate(features.states):
            negatives[state_index] = (
                np.arange(25, 33) % 25 + 25
                if state.product_index < 25
                else np.arange(8)
            )
        rows = np.arange(len(features.states), dtype=np.int32)
        pairs, positive_support, negative_support = TRAINER.eligible_crosses(
            features, rows, negatives
        )
        relationship = (
            features.context_name_to_id["ctx:use_case=running"],
            features.item_name_to_id["item:feature=cushioned"],
        )
        self.assertIn(relationship, pairs)
        relationship_index = pairs.index(relationship)
        self.assertGreaterEqual(
            positive_support[relationship_index], TRAINER.MIN_CROSS_SUPPORT
        )
        self.assertGreaterEqual(
            negative_support[relationship_index], TRAINER.MIN_CROSS_SUPPORT
        )

        with contextlib.redirect_stdout(io.StringIO()):
            parameters, history, _ = TRAINER.train_model(
                features,
                rows,
                negatives,
                pairs,
                8,
                variant="hybrid",
            )
        self.assertLess(history[-1]["loss"], history[0]["loss"])
        self.assertGreater(parameters.cross_weights[relationship_index], 0.0)


if __name__ == "__main__":
    unittest.main()
