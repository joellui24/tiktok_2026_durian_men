from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from starter.agent import Agent, SessionState
from starter.hybrid_model import PortableHybridModel


TRAINER_PATH = Path(__file__).resolve().parents[2] / "approach 1" / "train_fm.py"
SPEC = importlib.util.spec_from_file_location("approach1_train_fm_tests", TRAINER_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAINER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAINER
SPEC.loader.exec_module(TRAINER)


@dataclass(frozen=True)
class TrainingState:
    trajectory_id: int
    state_index: int
    product_index: int
    known_constraints: tuple[tuple[str, tuple[str, ...]], ...]
    split: str = "train"
    scenario: str = "browsing"
    scenario_state: str = "browsing"
    turn: int = 2
    intent_epoch: int = 0
    state_weight: float = 1.0


class TrainingDataset:
    def __init__(self, states: list[TrainingState], survivors: list[list[int]]) -> None:
        self.states = tuple(states)
        self._survivors = tuple(
            np.asarray(values, dtype=np.uint32) for values in survivors
        )

    def state_survivors(self, state_index: int) -> np.ndarray:
        return self._survivors[state_index]


class FactorizationTrainingTest(unittest.TestCase):
    def test_v2_artifact_round_trip_preserves_dual_other_scores(self) -> None:
        products = [
            TRAINER.Product(
                "HIKING",
                "outdoor gear",
                (("use_case", "for hiking", "For HIKING"),),
                None,
                "50_75",
                "4.5",
                "5",
                1,
            ),
            TRAINER.Product(
                "PLAIN",
                "outdoor gear",
                (("feature", "plain", "Plain"),),
                None,
                "50_75",
                "4.5",
                "5",
                1,
            ),
        ]
        state = TrainingState(
            trajectory_id=0,
            state_index=0,
            product_index=0,
            known_constraints=(("other", ("For HIKING",)),),
        )
        feature_data = TRAINER.build_feature_data(
            products,
            TrainingDataset([state], [[0, 1]]),
            product_splits=("train", "train"),
            minimum_value_support=1,
            other_encoding="dual",
        )

        dual_other_names = (
            "ctx:answer_source=other",
            "ctx:other=for hiking",
            "ctx:use_case=for hiking",
        )
        context_vectors = np.zeros(
            (len(feature_data.context_names), 2), dtype=np.float32
        )
        for name, weight in zip(
            dual_other_names, (0.25, 0.50, 0.75), strict=True
        ):
            context_vectors[feature_data.context_name_to_id[name], 0] = weight
        item_vectors = np.zeros(
            (len(feature_data.item_names), 2), dtype=np.float32
        )
        hiking_item_id = feature_data.item_name_to_id[
            "item:use_case=for hiking"
        ]
        item_vectors[hiking_item_id, 0] = 2.0
        cross_pair = (
            feature_data.context_name_to_id["ctx:answer_source=other"],
            hiking_item_id,
        )
        parameters = TRAINER.Parameters(
            context_vectors=context_vectors,
            item_vectors=item_vectors,
            item_linear=np.zeros(len(feature_data.item_names), dtype=np.float32),
            cross_weights=np.asarray([1.25], dtype=np.float32),
        )
        item_sum, item_base = TRAINER.item_components(feature_data, parameters)
        expected = np.asarray(
            [
                TRAINER.score_pair(
                    feature_data,
                    parameters,
                    0,
                    product_index,
                    item_sum,
                    item_base,
                    {cross_pair: 0},
                )
                for product_index in (0, 1)
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.jsonl"
            catalog_path.write_text("{}\n", encoding="utf-8")
            artifact_path = root / "model.sqlite3"
            config = TRAINER.TrainingConfig(
                dimension=2,
                minimum_value_support=1,
                minimum_cross_support=1,
                other_encoding="dual",
            )
            TRAINER.write_artifact(
                artifact_path,
                catalog_path,
                feature_data,
                parameters,
                (cross_pair,),
                np.asarray([1], dtype=np.int32),
                np.asarray([1], dtype=np.int32),
                1.0,
                1,
                "hybrid",
                config,
                {
                    "dataset_version": "fm-trajectories-v2",
                    "trajectory_count": 1,
                    "state_count": 1,
                },
            )

            model = PortableHybridModel(artifact_path)
            agent = object.__new__(Agent)
            agent.model = model
            runtime_context = agent._context_features(
                SessionState(
                    scenario_state="browsing",
                    coarse_category="outdoor gear",
                    known_constraints={"other": ["For HIKING"]},
                ),
                turn=2,
            )
            actual = model.score_many(
                ("HIKING", "PLAIN"), runtime_context, mode="hybrid"
            )

        self.assertEqual(
            model.metadata["feature_schema_version"],
            "conversation-features-v2",
        )
        self.assertEqual(model.metadata["category_only_weight"], "0.05")
        self.assertEqual(model.metadata["evidence_saturation"], "3")
        self.assertTrue(set(dual_other_names).issubset(runtime_context))
        self.assertEqual(
            set(model.context_ids_for(runtime_context)),
            set(feature_data.state_context_ids[0]),
        )
        np.testing.assert_allclose(
            [actual["HIKING"], actual["PLAIN"]], expected, rtol=0, atol=1e-7
        )
        self.assertGreater(actual["HIKING"], actual["PLAIN"])

    def test_supervision_band_boundaries_match_evaluator_contract(self) -> None:
        self.assertEqual(TRAINER.supervision_weight_band(0.0), "zero")
        self.assertEqual(TRAINER.supervision_weight_band(0.25), "low")
        self.assertEqual(TRAINER.supervision_weight_band(0.75), "medium")
        self.assertEqual(TRAINER.supervision_weight_band(1.0), "high")

    def test_product_split_is_stable(self) -> None:
        first = [TRAINER.split_for(f"A{index}") for index in range(100)]
        second = [TRAINER.split_for(f"A{index}") for index in range(100)]
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"train", "validation", "test"})

    def test_heldout_category_uses_train_fitted_rare_fallback(self) -> None:
        products = [
            TRAINER.Product(
                "TRAIN",
                "common category",
                (("feature", "shared", "shared"),),
                None,
                "20_35",
                "4.5",
                "3",
                1,
            ),
            TRAINER.Product(
                "HELDOUT",
                "heldout-only category",
                (("feature", "unseen", "unseen"),),
                None,
                "20_35",
                "4.5",
                "3",
                1,
            ),
        ]
        states = [
            TrainingState(0, 0, 0, (), split="train"),
            TrainingState(1, 0, 1, (), split="validation"),
        ]
        features = TRAINER.build_feature_data(
            products,
            TrainingDataset(states, [[0, 1], [0, 1]]),
            product_splits=("train", "validation"),
            minimum_value_support=1,
        )
        self.assertIn("ctx:category=common category", features.context_name_to_id)
        self.assertIn("ctx:category=<rare>", features.context_name_to_id)
        self.assertNotIn(
            "ctx:category=heldout-only category", features.context_name_to_id
        )
        heldout_item_names = {
            features.item_names[feature_id]
            for feature_id in features.product_item_ids[1]
        }
        self.assertIn("item:category=<rare>", heldout_item_names)

    def test_supported_cross_is_learned_from_survivor_pairs(self) -> None:
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

        states: list[TrainingState] = []
        survivors: list[list[int]] = []
        for product_index in range(len(products)):
            running = product_index < 25
            states.append(
                TrainingState(
                    trajectory_id=product_index,
                    state_index=0,
                    product_index=product_index,
                    known_constraints=(
                        ("use_case", (("running" if running else "outdoor"),)),
                    ),
                )
            )
            competitor = 25 + product_index % 25 if running else product_index % 25
            survivors.append([product_index, competitor])

        dataset = TrainingDataset(states, survivors)
        features = TRAINER.build_feature_data(
            products,
            dataset,
            product_splits=["train"] * len(products),
            minimum_value_support=1,
        )
        rows = np.arange(len(states), dtype=np.int32)
        config = TRAINER.TrainingConfig(
            seed=2026,
            dimension=8,
            negatives_per_state=1,
            negative_pre_pool_size=1,
            max_epochs=8,
            supervision_policy="downweight_ties",
        )
        initial_parameters = TRAINER.initialize_parameters(
            features, 0, "hybrid", config
        )
        sampled_pairs = TRAINER.sample_dynamic_negatives(
            features,
            rows,
            initial_parameters,
            (),
            epoch=0,
            config=config,
        )
        pairs, positive_support, comparable_support = TRAINER.eligible_crosses(
            features, sampled_pairs
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
            comparable_support[relationship_index], TRAINER.MIN_CROSS_SUPPORT
        )

        with contextlib.redirect_stdout(io.StringIO()):
            parameters, history, _, _ = TRAINER.train_model(
                features,
                rows,
                pairs,
                validation_rows=None,
                variant="hybrid",
                config=config,
            )
        self.assertLess(
            history[-1]["weighted_bpr_loss"],
            history[0]["weighted_bpr_loss"],
        )
        self.assertGreater(parameters.cross_weights[relationship_index], 0.0)


if __name__ == "__main__":
    unittest.main()
