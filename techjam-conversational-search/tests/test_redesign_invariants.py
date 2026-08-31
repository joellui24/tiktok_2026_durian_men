from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


APPROACH_ROOT = Path(__file__).resolve().parents[2] / "approach 1"
sys.path.insert(0, str(APPROACH_ROOT))

import evaluate_fm as EVALUATION  # noqa: E402
import fm_training as TRAINING  # noqa: E402
import trajectory_data as TRAJECTORY  # noqa: E402
from starter.conversation_features import context_feature_names  # noqa: E402


def _product(
    index: int,
    *,
    category: str = "shared category",
    constraints: tuple[TRAJECTORY.Constraint, ...] | None = None,
    hard_constraint_count: int = 1,
) -> TRAJECTORY.Product:
    if constraints is None:
        hard = f"hard group {index % 4}"
        old = f"old preference {(index // 4) % 2}"
        constraints = (
            ("feature", hard, hard),
            ("feature", old, old),
        )
    return TRAJECTORY.Product(
        parent_asin=f"P{index:03d}",
        category=category,
        constraints=constraints,
        brand=None,
        price_bucket="20_35",
        rating_bucket="4.5",
        popularity_bucket="4",
        hard_constraint_count=hard_constraint_count,
    )


def _raw_product(product: TRAJECTORY.Product) -> dict[str, object]:
    return {
        "parent_asin": product.parent_asin,
        "title": product.parent_asin,
        "categories": ["Root", product.category],
        "features": [display for _, _, display in product.constraints],
    }


@dataclass(frozen=True)
class _TrainingState:
    trajectory_id: int
    state_index: int
    product_index: int
    known_constraints: tuple[tuple[str, tuple[str, ...]], ...]
    state_weight: float = 1.0
    split: str = "train"
    scenario: str = "browsing"
    scenario_state: str = "browsing"
    turn: int = 2
    intent_epoch: int = 0
    has_other_answer: bool = False


class _ArrayDataset:
    def __init__(
        self, states: list[_TrainingState], survivors: list[np.ndarray]
    ) -> None:
        self.states = tuple(states)
        self._survivors = tuple(
            np.asarray(values, dtype=np.uint32) for values in survivors
        )

    def state_survivors(self, state_index: int) -> np.ndarray:
        return self._survivors[state_index]


def _feature_fixture(
    products: list[TRAJECTORY.Product],
    survivors: np.ndarray,
    *,
    known_constraints: tuple[tuple[str, tuple[str, ...]], ...],
    state_weight: float = 1.0,
) -> TRAINING.FeatureData:
    state = _TrainingState(
        trajectory_id=7,
        state_index=0,
        product_index=0,
        known_constraints=known_constraints,
        state_weight=state_weight,
    )
    dataset = _ArrayDataset([state], [survivors])
    return TRAINING.build_feature_data(
        products,
        dataset,
        product_splits=["train"] * len(products),
        minimum_value_support=1,
    )


def _zero_parameters(
    features: TRAINING.FeatureData, config: TRAINING.TrainingConfig
) -> TRAINING.Parameters:
    parameters = TRAINING.initialize_parameters(features, 0, "fm", config)
    parameters.context_vectors.fill(0.0)
    parameters.item_vectors.fill(0.0)
    parameters.item_linear.fill(0.0)
    return parameters


class ScenarioAndSplitInvariantTest(unittest.TestCase):
    def test_public_scenario_counts_are_exact_and_plan_sizes_are_nested(self) -> None:
        schedules = {
            count: TRAJECTORY.allocate_scenarios(count, "public", seed=81)
            for count in (25_000, 50_000, 100_000)
        }
        for count, schedule in schedules.items():
            self.assertEqual(
                Counter(schedule),
                {
                    "buying": count * 8 // 20,
                    "browsing": count * 8 // 20,
                    "boundary": count // 20,
                    "intent_override": count * 3 // 20,
                },
            )
        self.assertEqual(schedules[100_000][:50_000], schedules[50_000])
        self.assertEqual(schedules[50_000][:25_000], schedules[25_000])

    def test_balanced_scenario_counts_are_exact_and_nested(self) -> None:
        small = TRAJECTORY.allocate_scenarios(25_000, "balanced", seed=91)
        large = TRAJECTORY.allocate_scenarios(50_000, "balanced", seed=91)
        self.assertEqual(
            Counter(small),
            {scenario: 6_250 for scenario in TRAJECTORY.SCENARIOS},
        )
        self.assertEqual(
            Counter(large),
            {scenario: 12_500 for scenario in TRAJECTORY.SCENARIOS},
        )
        self.assertEqual(large[:25_000], small)

    def test_product_splits_are_deterministic_and_category_stratified(self) -> None:
        products = [
            _product(index, category=f"category-{index // 10}")
            for index in range(30)
        ]
        first = TRAJECTORY.build_product_splits(products, seed=123)
        second = TRAJECTORY.build_product_splits(products, seed=123)
        self.assertEqual(first, second)

        split_sets = {name: set(indices) for name, indices in first.items()}
        self.assertEqual(
            {name: len(indices) for name, indices in split_sets.items()},
            {"train": 24, "validation": 3, "test": 3},
        )
        self.assertEqual(set().union(*split_sets.values()), set(range(30)))
        for left_index, left in enumerate(TRAJECTORY.SPLITS):
            for right in TRAJECTORY.SPLITS[left_index + 1 :]:
                self.assertTrue(split_sets[left].isdisjoint(split_sets[right]))

        for category_number in range(3):
            category_indices = set(
                range(category_number * 10, (category_number + 1) * 10)
            )
            self.assertEqual(
                {
                    split: len(indices & category_indices)
                    for split, indices in split_sets.items()
                },
                {"train": 8, "validation": 1, "test": 1},
            )

        reversed_products = list(reversed(products))
        reversed_splits = TRAJECTORY.build_product_splits(
            reversed_products, seed=123
        )
        labels = {
            products[index].parent_asin: split
            for split, indices in first.items()
            for index in indices
        }
        reversed_labels = {
            reversed_products[index].parent_asin: split
            for split, indices in reversed_splits.items()
            for index in indices
        }
        self.assertEqual(reversed_labels, labels)

    def test_category_apportionment_avoids_global_schedule_slice_bias(self) -> None:
        products = []
        for category, size in (("tiny", 1), ("ten", 10), ("nine", 9)):
            for _ in range(size):
                products.append(_product(len(products), category=category))

        splits = TRAJECTORY.build_product_splits(products, seed=7)
        labels = {
            index: split
            for split, indices in splits.items()
            for index in indices
        }
        ten_counts = Counter(
            labels[index]
            for index, product in enumerate(products)
            if product.category == "ten"
        )

        # The previous implementation sliced a shuffled global schedule at
        # category boundaries and produced 7/2/1 for this feasible 10-product
        # category.  Per-category apportionment must preserve exact 8/1/1.
        self.assertEqual(
            ten_counts,
            {"train": 8, "validation": 1, "test": 1},
        )
        self.assertEqual(
            {split: len(indices) for split, indices in splits.items()},
            {"train": 16, "validation": 2, "test": 2},
        )
        self.assertTrue(
            all(
                any(
                    labels[index] == "train"
                    for index, value in enumerate(products)
                    if value.category == category
                )
                for category in {product.category for product in products}
            )
        )

    def test_real_catalog_gives_every_category_train_coverage_when_feasible(self) -> None:
        catalog_path = (
            APPROACH_ROOT.parent
            / "techjam-conversational-search"
            / "data"
            / "catalog.jsonl"
        )
        products: list[TRAJECTORY.Product] = []
        with catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                products.append(
                    TRAJECTORY.Product(
                        parent_asin=str(raw["parent_asin"]),
                        category=TRAJECTORY.coarse_category(
                            [str(value) for value in raw.get("categories") or []]
                        ),
                        constraints=(),
                        brand=None,
                        price_bucket="missing",
                        rating_bucket="missing",
                        popularity_bucket="missing",
                        hard_constraint_count=0,
                    )
                )

        splits = TRAJECTORY.build_product_splits(products, seed=2026)
        expected = TRAJECTORY._nearest_ratio_counts(
            len(products), TRAJECTORY.SPLIT_COUNTS_PER_TEN
        )
        self.assertEqual(
            {split: len(indices) for split, indices in splits.items()}, expected
        )
        categories = {product.category for product in products}
        self.assertGreaterEqual(len(splits["train"]), len(categories))
        train_categories = {
            products[index].category for index in splits["train"]
        }
        self.assertEqual(train_categories, categories)


class TrajectoryInvariantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.products = [_product(index) for index in range(20)]
        raw_products = [_raw_product(product) for product in cls.products]
        cls.dataset = TRAJECTORY.generate_trajectory_dataset(
            cls.products,
            raw_products,
            TRAJECTORY.TrajectoryConfig(
                trajectory_count=20,
                seed=2026,
                split_seed=303,
                max_turns=6,
                extended_fraction=1.0,
            ),
        )

    def _trajectory_states(self, trajectory_id: int) -> tuple[object, ...]:
        start = int(self.dataset.trajectory_state_offsets[trajectory_id])
        stop = int(self.dataset.trajectory_state_offsets[trajectory_id + 1])
        return self.dataset.states[start:stop]

    def _global_state_index(self, state: object) -> int:
        start = int(self.dataset.trajectory_state_offsets[state.trajectory_id])
        return start + state.state_index

    def test_browsing_and_boundary_turn_one_features_have_no_future_label(self) -> None:
        checked = Counter()
        for state in self.dataset.states:
            if state.turn != 1 or state.scenario not in {"browsing", "boundary"}:
                continue
            checked[state.scenario] += 1
            self.assertEqual(state.scenario_state, "exploring_unknown")
            names = context_feature_names(
                coarse_category=self.products[state.product_index].category,
                scenario_state=state.scenario_state,
                turn=state.turn,
                intent_epoch=state.intent_epoch,
                known_constraints=state.known_constraint_mapping(),
            )
            self.assertIn("ctx:scenario=exploring_unknown", names)
            self.assertNotIn("ctx:scenario=browsing", names)
            self.assertNotIn("ctx:scenario=boundary", names)
        self.assertEqual(checked, {"browsing": 8, "boundary": 1})

    def test_question_choice_does_not_inspect_the_target_reply(self) -> None:
        products = [
            _product(
                index,
                constraints=(
                    (
                        "feature" if index % 2 == 0 else "material",
                        "battery life" if index % 2 == 0 else "cotton",
                        "battery life" if index % 2 == 0 else "cotton",
                    ),
                ),
                hard_constraint_count=1,
            )
            for index in range(12)
        ]
        postings = TRAJECTORY.build_catalog_postings(
            products, [_raw_product(product) for product in products]
        )
        config = TRAJECTORY.TrajectoryConfig(
            trajectory_count=20,
            seed=41,
            max_turns=2,
            extended_fraction=0.0,
        )

        # Targets 0 and 1 have incompatible private replies, while their
        # turn-one visible state is identical: same category, candidates,
        # scenario, turn, seed, and empty conversation history.  The policy
        # must therefore ask the same question for either hidden label.
        even_target, _ = TRAJECTORY._simulate_trajectory(
            trajectory_id=7,
            product_index=0,
            split="train",
            scenario="browsing",
            products=products,
            postings=postings,
            config=config,
        )
        odd_target, _ = TRAJECTORY._simulate_trajectory(
            trajectory_id=7,
            product_index=1,
            split="train",
            scenario="browsing",
            products=products,
            postings=postings,
            config=config,
        )

        self.assertNotEqual(
            products[0].answer_values("feature", set()),
            products[1].answer_values("feature", set()),
        )
        self.assertIsNotNone(even_target[0].asked_attribute)
        self.assertEqual(
            even_target[0].asked_attribute,
            odd_target[0].asked_attribute,
        )

    def test_pending_reply_enters_only_the_following_state(self) -> None:
        checked = 0
        for trajectory_id, scenario in enumerate(self.dataset.trajectory_scenarios):
            if scenario != "browsing":
                continue
            states = self._trajectory_states(trajectory_id)
            for before, after in zip(states, states[1:]):
                attribute = before.asked_attribute
                if attribute is None or after.intent_epoch != before.intent_epoch:
                    continue
                before_constraints = before.known_constraint_mapping()
                after_constraints = after.known_constraint_mapping()
                pending_values = tuple(after_constraints.get(attribute, ()))
                if not pending_values:
                    continue
                checked += 1
                self.assertNotIn(attribute, before_constraints)
                for value in pending_values:
                    self.assertNotIn(
                        value,
                        {
                            existing
                            for values in before_constraints.values()
                            for existing in values
                        },
                    )
        self.assertGreater(checked, 0)

    def test_seeded_question_variation_stays_within_current_roadmap_stage(self) -> None:
        attributes = ("feature", "style", "material", "size", "budget")
        choices = {
            TRAJECTORY._choose_attribute(
                TRAJECTORY._roadmap_candidates(attributes),
                seed=53,
                trajectory_id=trajectory_id,
                turn=2,
                phase="informative",
            )
            for trajectory_id in range(40)
        }
        self.assertGreater(len(choices), 1)
        self.assertTrue(choices.issubset({"feature", "style", "material"}))

    def test_override_clears_old_evidence_and_rebuilds_from_category_pool(self) -> None:
        override_count = 0
        for trajectory_id, scenario in enumerate(self.dataset.trajectory_scenarios):
            if scenario != "intent_override":
                continue
            override_count += 1
            states = self._trajectory_states(trajectory_id)
            target = self.products[states[0].product_index]
            old = target.soft_preferences[-1]
            replacement = target.hard_constraints[0]

            turn_one_values = {
                value
                for _, values in states[0].known_constraints
                for value in values
            }
            self.assertIn(old[2], turn_one_values)
            post_override = next(state for state in states if state.intent_epoch == 1)
            post_values = {
                value
                for _, values in post_override.known_constraints
                for value in values
            }
            self.assertNotIn(old[2], post_values)
            self.assertIn(replacement[2], post_values)

            expected = {
                index
                for index, product in enumerate(self.products)
                if any(
                    attribute == replacement[0] and normalized == replacement[1]
                    for attribute, normalized, _ in product.constraints
                )
            }
            survivors = set(
                map(
                    int,
                    self.dataset.state_survivors(
                        self._global_state_index(post_override)
                    ),
                )
            )
            self.assertEqual(survivors, expected)
            self.assertTrue(
                any(
                    product.soft_preferences[-1][1] != old[1]
                    for index, product in enumerate(self.products)
                    if index in survivors
                )
            )
        self.assertEqual(override_count, 3)

    def test_target_survives_every_state_and_state_weights_sum_to_one(self) -> None:
        for trajectory_id in range(self.dataset.trajectory_count):
            states = self._trajectory_states(trajectory_id)
            self.assertTrue(states)
            self.assertAlmostEqual(
                sum(state.state_weight for state in states), 1.0, places=12
            )
            for state in states:
                survivors = self.dataset.state_survivors(
                    self._global_state_index(state)
                )
                self.assertIn(state.product_index, survivors)
                self.assertEqual(state.survivor_count, len(survivors))

    def test_survivors_use_compact_read_only_flat_storage(self) -> None:
        self.assertEqual(self.dataset.survivor_values.dtype, np.dtype(np.uint32))
        self.assertEqual(self.dataset.survivor_offsets.dtype, np.dtype(np.uint64))
        self.assertEqual(
            len(self.dataset.survivor_offsets), len(self.dataset.states) + 1
        )
        self.assertFalse(hasattr(self.dataset.states[0], "survivors"))
        for state_index, state in enumerate(self.dataset.states):
            view = self.dataset.state_survivors(state_index)
            self.assertFalse(view.flags.writeable)
            self.assertTrue(np.shares_memory(view, self.dataset.survivor_values))
            self.assertEqual(len(view), state.survivor_count)


class NegativeSamplingAndSupervisionInvariantTest(unittest.TestCase):
    @staticmethod
    def _unique_products(count: int) -> list[TRAJECTORY.Product]:
        return [
            _product(
                index,
                constraints=(("feature", f"value {index}", f"value {index}"),),
                hard_constraint_count=1,
            )
            for index in range(count)
        ]

    def test_small_pool_uses_every_survivor_except_target(self) -> None:
        products = self._unique_products(4)
        features = _feature_fixture(
            products,
            np.arange(4, dtype=np.uint32),
            known_constraints=(("feature", ("value 0",)),),
        )
        config = TRAINING.TrainingConfig(
            seed=19,
            dimension=2,
            negatives_per_state=8,
            negative_pre_pool_size=8,
            supervision_policy="downweight_ties",
        )
        batch = TRAINING.sample_dynamic_negatives(
            features,
            np.asarray([0], dtype=np.int32),
            _zero_parameters(features, config),
            (),
            epoch=0,
            config=config,
        )
        self.assertEqual(set(map(int, batch.negatives)), {1, 2, 3})
        self.assertTrue(np.all(batch.positives == 0))
        self.assertEqual(set(batch.sampler_types), {"all"})
        self.assertTrue(np.allclose(batch.sampling_weights, 1.0 / 3.0))

    def test_static_and_product_fixed_small_pools_use_every_valid_candidate(self) -> None:
        products = self._unique_products(4)
        features = _feature_fixture(
            products,
            np.asarray([0, 1], dtype=np.uint32),
            known_constraints=(("feature", ("value 0",)),),
        )
        rows = np.asarray([0], dtype=np.int32)
        for mode, expected in (
            ("survivor_static", {1}),
            ("product_fixed", {1, 2, 3}),
        ):
            with self.subTest(mode=mode):
                config = TRAINING.TrainingConfig(
                    seed=17,
                    dimension=2,
                    negatives_per_state=8,
                    negative_pre_pool_size=8,
                    negative_mode=mode,
                    supervision_policy="downweight_ties",
                )
                batch = TRAINING.sample_negatives(
                    features,
                    rows,
                    _zero_parameters(features, config),
                    (),
                    epoch=6,
                    config=config,
                )
                self.assertEqual(set(map(int, batch.negatives)), expected)
                self.assertEqual(len(batch.negatives), len(expected))
                self.assertTrue(
                    np.allclose(batch.sampling_weights, 1.0 / len(expected))
                )
                self.assertEqual(batch.diagnostics["negative_mode"], mode)
                self.assertTrue(
                    all(row["negative_mode"] == mode for row in batch.audit_rows)
                )

    def test_training_negatives_never_cross_product_splits(self) -> None:
        products = self._unique_products(6)
        state = _TrainingState(
            trajectory_id=9,
            state_index=0,
            product_index=0,
            known_constraints=(("feature", ("value 0",)),),
            split="train",
        )
        features = TRAINING.build_feature_data(
            products,
            _ArrayDataset([state], [np.arange(6, dtype=np.uint32)]),
            product_splits=(
                "train",
                "train",
                "validation",
                "validation",
                "test",
                "test",
            ),
            minimum_value_support=1,
        )
        config = TRAINING.TrainingConfig(
            seed=23,
            dimension=2,
            negatives_per_state=8,
            negative_pre_pool_size=8,
            supervision_policy="downweight_ties",
        )
        batch = TRAINING.sample_dynamic_negatives(
            features,
            np.asarray([0], dtype=np.int32),
            _zero_parameters(features, config),
            (),
            epoch=0,
            config=config,
        )
        self.assertEqual(tuple(map(int, batch.negatives)), (1,))
        self.assertEqual(
            batch.diagnostics["excluded_product_split_candidate_count"], 4
        )
        self.assertEqual(batch.audit_rows[0]["eligible_negative_pool_size"], 1)
        self.assertEqual(batch.audit_rows[0]["excluded_product_split_count"], 4)

    def test_dynamic_negatives_are_valid_reproducible_and_refreshed(self) -> None:
        products = self._unique_products(40)
        survivors = np.arange(40, dtype=np.uint32)
        features = _feature_fixture(
            products,
            survivors,
            known_constraints=(("feature", ("value 0",)),),
        )
        config = TRAINING.TrainingConfig(
            seed=29,
            dimension=2,
            negatives_per_state=8,
            negative_pre_pool_size=16,
            supervision_policy="downweight_ties",
        )
        parameters = _zero_parameters(features, config)

        def sample(epoch: int) -> TRAINING.PairBatch:
            return TRAINING.sample_dynamic_negatives(
                features,
                np.asarray([0], dtype=np.int32),
                parameters,
                (),
                epoch=epoch,
                config=config,
            )

        first = sample(3)
        repeated = sample(3)
        refreshed = sample(4)
        configured_default = TRAINING.sample_negatives(
            features,
            np.asarray([0], dtype=np.int32),
            parameters,
            (),
            epoch=3,
            config=config,
        )
        self.assertTrue(np.array_equal(first.negatives, repeated.negatives))
        self.assertTrue(np.array_equal(first.negatives, configured_default.negatives))
        self.assertTrue(np.array_equal(first.sampler_types, repeated.sampler_types))
        self.assertNotEqual(tuple(first.negatives), tuple(refreshed.negatives))
        self.assertEqual(len(first.negatives), 8)
        self.assertEqual(len(set(map(int, first.negatives))), 8)
        self.assertNotIn(0, first.negatives)
        self.assertTrue(
            set(map(int, first.negatives)).issubset(set(map(int, survivors)))
        )

    def test_survivor_static_is_deterministic_and_cached_across_training_epochs(self) -> None:
        products = self._unique_products(40)
        features = _feature_fixture(
            products,
            np.arange(40, dtype=np.uint32),
            known_constraints=(("feature", ("value 0",)),),
        )
        rows = np.asarray([0], dtype=np.int32)
        config = TRAINING.TrainingConfig(
            seed=31,
            dimension=2,
            negatives_per_state=8,
            negative_pre_pool_size=16,
            negative_mode="survivor_static",
            supervision_policy="downweight_ties",
            max_epochs=3,
        )
        parameters = _zero_parameters(features, config)
        first = TRAINING.sample_negatives(
            features, rows, parameters, (), epoch=1, config=config
        )
        repeated = TRAINING.sample_negatives(
            features, rows, parameters, (), epoch=9, config=config
        )
        self.assertTrue(np.array_equal(first.negatives, repeated.negatives))
        self.assertTrue(np.array_equal(first.positives, repeated.positives))
        self.assertTrue(np.array_equal(first.sampler_types, repeated.sampler_types))
        self.assertEqual(first.diagnostics["selection_epoch"], 0)
        self.assertEqual(repeated.diagnostics["selection_epoch"], 0)

        with contextlib.redirect_stdout(io.StringIO()):
            _, history, _, audit = TRAINING.train_model(
                features,
                rows,
                (),
                validation_rows=None,
                variant="fm",
                config=config,
            )
        self.assertEqual(len(history), 3)
        assignments = {
            epoch: tuple(
                row["negative_parent_asin"]
                for row in audit
                if row["epoch"] == epoch
            )
            for epoch in (1, 2, 3)
        }
        self.assertTrue(assignments[1])
        self.assertEqual(assignments[1], assignments[2])
        self.assertEqual(assignments[2], assignments[3])
        self.assertTrue(
            all(
                record["negative_sampling"]["negative_mode"]
                == "survivor_static"
                for record in history
            )
        )

    def test_product_fixed_pool_is_split_safe_but_not_survivor_bound(self) -> None:
        products = [
            _product(
                index,
                category=("shared" if index < 4 else "other"),
                constraints=(("feature", f"value {index}", f"value {index}"),),
                hard_constraint_count=1,
            )
            for index in range(5)
        ]
        state = _TrainingState(
            trajectory_id=12,
            state_index=0,
            product_index=0,
            known_constraints=(("feature", ("value 0",)),),
            split="train",
        )
        features = TRAINING.build_feature_data(
            products,
            _ArrayDataset(
                [state],
                [np.asarray([0, 1], dtype=np.uint32)],
            ),
            product_splits=("train", "train", "train", "validation", "train"),
            minimum_value_support=1,
        )
        config = TRAINING.TrainingConfig(
            seed=41,
            dimension=2,
            negatives_per_state=8,
            negative_pre_pool_size=8,
            negative_mode="product_fixed",
            supervision_policy="downweight_ties",
        )
        rows = np.asarray([0], dtype=np.int32)
        first = TRAINING.sample_negatives(
            features,
            rows,
            _zero_parameters(features, config),
            (),
            epoch=1,
            config=config,
        )
        repeated = TRAINING.sample_negatives(
            features,
            rows,
            _zero_parameters(features, config),
            (),
            epoch=99,
            config=config,
        )
        self.assertEqual(set(map(int, first.negatives)), {1, 2})
        self.assertTrue(np.array_equal(first.negatives, repeated.negatives))
        self.assertNotIn(3, first.negatives)
        self.assertNotIn(4, first.negatives)
        self.assertEqual(
            first.diagnostics["excluded_product_split_candidate_count"], 1
        )
        self.assertEqual(
            first.diagnostics["negative_outside_survivor_pair_count"], 1
        )
        by_negative = {
            row["negative_parent_asin"]: row for row in first.audit_rows
        }
        self.assertTrue(
            by_negative[products[1].parent_asin]["negative_in_survivor_set"]
        )
        self.assertFalse(
            by_negative[products[2].parent_asin]["negative_in_survivor_set"]
        )

    def test_config_and_cli_default_to_dynamic_survivors(self) -> None:
        self.assertEqual(
            TRAINING.TrainingConfig().negative_mode,
            "survivor_dynamic",
        )
        self.assertEqual(
            TRAINING._parser().parse_args([]).negative_mode,
            "survivor_dynamic",
        )
        with self.assertRaises(ValueError):
            TRAINING.TrainingConfig(negative_mode="not-a-mode")

    def test_artifact_metadata_records_negative_mode(self) -> None:
        products = self._unique_products(2)
        features = _feature_fixture(
            products,
            np.asarray([0, 1], dtype=np.uint32),
            known_constraints=(("feature", ("value 0",)),),
        )
        config = TRAINING.TrainingConfig(
            dimension=2,
            negative_mode="survivor_static",
            supervision_policy="downweight_ties",
        )
        parameters = _zero_parameters(features, config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            catalog.write_text("{}\n", encoding="utf-8")
            artifact = root / "candidate.sqlite3"
            TRAINING.write_artifact(
                artifact,
                catalog,
                features,
                parameters,
                (),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                1.0,
                1,
                "fm",
                config,
                {
                    "dataset_version": "test-v1",
                    "trajectory_count": 1,
                    "state_count": 1,
                },
            )
            with sqlite3.connect(artifact) as connection:
                metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        self.assertEqual(metadata["negative_mode"], "survivor_static")

    def test_ambiguity_policies_and_weight_components_are_auditable(self) -> None:
        products = [
            _product(
                0,
                constraints=(("feature", "shared", "shared"),),
                hard_constraint_count=1,
            ),
            _product(
                1,
                constraints=(("feature", "shared", "shared"),),
                hard_constraint_count=1,
            ),
            _product(
                2,
                constraints=(("feature", "different 2", "different 2"),),
                hard_constraint_count=1,
            ),
            _product(
                3,
                constraints=(("feature", "different 3", "different 3"),),
                hard_constraint_count=1,
            ),
        ]
        features = _feature_fixture(
            products,
            np.arange(4, dtype=np.uint32),
            known_constraints=(("feature", ("shared",)),),
            state_weight=0.5,
        )
        rows = np.asarray([0], dtype=np.int32)

        def sample(policy: str, epoch: int = 0) -> TRAINING.PairBatch:
            config = TRAINING.TrainingConfig(
                seed=37,
                dimension=2,
                negatives_per_state=8,
                negative_pre_pool_size=8,
                supervision_policy=policy,
                tie_weight=0.2,
                evidence_saturation=2,
            )
            return TRAINING.sample_dynamic_negatives(
                features,
                rows,
                _zero_parameters(features, config),
                (),
                epoch=epoch,
                config=config,
            )

        skipped = sample("skip_ties")
        self.assertEqual(set(map(int, skipped.negatives)), {2, 3})
        self.assertTrue(np.all(skipped.positives == 0))
        self.assertEqual(skipped.diagnostics["skipped_tie_count"], 1)

        downweighted = sample("downweight_ties")
        self.assertEqual(set(map(int, downweighted.negatives)), {1, 2, 3})
        tie_row = np.flatnonzero(downweighted.negatives == 1)
        clear_rows = np.flatnonzero(downweighted.negatives != 1)
        self.assertTrue(np.allclose(downweighted.ambiguity_weights[tie_row], 0.2))
        self.assertTrue(np.allclose(downweighted.ambiguity_weights[clear_rows], 1.0))
        self.assertTrue(np.allclose(downweighted.trajectory_state_weights, 0.5))
        self.assertTrue(np.allclose(downweighted.evidence_weights, 0.5))
        self.assertTrue(np.allclose(downweighted.sampling_weights, 1.0 / 3.0))
        self.assertTrue(
            np.allclose(
                downweighted.effective_weights,
                downweighted.trajectory_state_weights
                * downweighted.evidence_weights
                * downweighted.sampling_weights
                * downweighted.ambiguity_weights,
            )
        )
        self.assertTrue(downweighted.audit_rows)
        self.assertTrue(
            {
                "trajectory_state_weight",
                "evidence_weight",
                "sampling_weight",
                "ambiguity_weight",
            }.issubset(downweighted.audit_rows[0])
        )

        set_valued_batches = [
            sample("set_valued_positives", epoch) for epoch in range(8)
        ]
        for batch in set_valued_batches:
            self.assertEqual(set(map(int, batch.negatives)), {2, 3})
            self.assertTrue(set(map(int, batch.positives)).issubset({0, 1}))
        self.assertIn(
            1,
            {
                int(positive)
                for batch in set_valued_batches
                for positive in batch.positives
            },
        )


@dataclass(frozen=True)
class _EvaluationState:
    state_id: str = "state-0"
    trajectory_id: str = "trajectory-0"
    target_parent_asin: str = "B"
    split: str = "validation"
    scenario: str = "browsing"
    scenario_state: str = "browsing"
    turn: int = 2


class _EvaluationDataset:
    states = (_EvaluationState(),)

    @staticmethod
    def state_survivors(index: int) -> tuple[str, ...]:
        del index
        return ("D", "B", "A", "C")

    @staticmethod
    def state_context_features(state: _EvaluationState) -> tuple[str, ...]:
        del state
        return ()


class _EvaluationModel:
    metadata = {"seed": "1"}

    def __init__(self) -> None:
        self.scored_widths: list[int] = []

    def score_many(
        self,
        parent_asins: tuple[str, ...],
        context_names: tuple[str, ...],
        *,
        mode: str,
    ) -> dict[str, float]:
        del context_names, mode
        self.scored_widths.append(len(parent_asins))
        scores = {"A": 1.0, "B": 1.0, "C": 0.5, "D": 2.0}
        return {parent_asin: scores[parent_asin] for parent_asin in parent_asins}


class FullSurvivorInvariantTest(unittest.TestCase):
    def test_full_survivor_denominator_and_asin_tie_order_are_exact(self) -> None:
        model = _EvaluationModel()
        rows = EVALUATION.score_full_survivor_states(
            _EvaluationDataset(), model, mode="fm"
        )
        self.assertEqual(model.scored_widths, [4])
        self.assertEqual(rows[0]["candidate_width"], 4)
        self.assertEqual(rows[0]["target_rank"], 3)
        self.assertAlmostEqual(rows[0]["rank_percentile"], 2.0 / 3.0)
        self.assertEqual(
            EVALUATION.exact_rank({"C": 2.0, "B": 1.0, "A": 1.0}, "B"),
            3,
        )


if __name__ == "__main__":
    unittest.main()
