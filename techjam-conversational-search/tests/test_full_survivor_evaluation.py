from __future__ import annotations

import contextlib
import io
import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


EVALUATOR_PATH = (
    Path(__file__).resolve().parents[2] / "approach 1" / "evaluate_fm.py"
)
SPEC = importlib.util.spec_from_file_location(
    "approach1_full_survivor_evaluation_tests", EVALUATOR_PATH
)
assert SPEC is not None and SPEC.loader is not None
EVALUATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EVALUATION
SPEC.loader.exec_module(EVALUATION)


@dataclass(frozen=True)
class FakeState:
    state_id: str
    trajectory_id: str
    target_parent_asin: str
    split: str = "validation"
    scenario: str = "browsing"
    scenario_state: str = "browsing"
    turn: int = 1
    turn_bucket: str = "early"
    has_other_answer: bool = False
    evidence_weight: float = 1.0


class FakeDataset:
    def __init__(self, states: list[FakeState], survivors: list[list[str]]) -> None:
        self.states = tuple(states)
        self._survivors = tuple(tuple(values) for values in survivors)
        self.seed = 77

    def state_survivors(self, index: int) -> tuple[str, ...]:
        return self._survivors[index]

    @staticmethod
    def state_context_features(state: FakeState) -> tuple[str, ...]:
        return (f"ctx:state={state.state_id}",)


class FakeModel:
    def __init__(self, scores: dict[str, dict[str, float]], seed: int = 9) -> None:
        self._scores = scores
        self.metadata = {"seed": str(seed)}
        self.scored_candidate_counts: list[int] = []

    def score_many(
        self, parent_asins: tuple[str, ...], context_names: tuple[str, ...], *, mode: str
    ) -> dict[str, float]:
        del mode
        self.scored_candidate_counts.append(len(parent_asins))
        state_id = context_names[0].split("=", 1)[1]
        return {parent_asin: self._scores[state_id][parent_asin] for parent_asin in parent_asins}


class FullSurvivorEvaluationTest(unittest.TestCase):
    def test_skip_official_cli_never_reads_or_writes_public_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trajectory_module_path = root / "trajectory_data.py"
            trajectory_module_path.touch()
            official_output = root / "official"
            ranker_output = root / "ranker_only"
            fake_module = SimpleNamespace()
            fake_dataset = object()
            full_survivor_report = {
                "models": {
                    name: {"overall": {"primary": {"mrr": score}}}
                    for name, score in (
                        ("linear", 0.1),
                        ("fm", 0.2),
                        ("candidate", 0.3),
                    )
                }
            }

            with (
                mock.patch.object(
                    EVALUATION.evaluator,
                    "load_jsonl",
                    side_effect=AssertionError("public dataset must not be read"),
                ) as load_public,
                mock.patch.object(
                    EVALUATION.evaluator,
                    "catalog_index",
                    side_effect=AssertionError(
                        "official catalog indexing must not run"
                    ),
                ) as official_catalog,
                mock.patch.object(
                    EVALUATION,
                    "Agent",
                    side_effect=AssertionError("official agent must not be created"),
                ) as official_agent,
                mock.patch.object(
                    EVALUATION,
                    "_load_python_module",
                    return_value=fake_module,
                ),
                mock.patch.object(
                    EVALUATION,
                    "generate_fixed_heldout_dataset",
                    return_value=fake_dataset,
                ) as generate_dataset,
                mock.patch.object(
                    EVALUATION,
                    "write_full_survivor_outputs",
                    return_value=full_survivor_report,
                ) as write_ranker_only,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                EVALUATION.main(
                    [
                        "--skip-official",
                        "--dataset",
                        str(root / "must-not-be-read.jsonl"),
                        "--output-dir",
                        str(official_output),
                        "--redesign-output-dir",
                        str(ranker_output),
                        "--trajectory-module",
                        str(trajectory_module_path),
                        "--third-model-name",
                        "candidate",
                    ]
                )

            load_public.assert_not_called()
            official_catalog.assert_not_called()
            official_agent.assert_not_called()
            generate_dataset.assert_called_once()
            write_ranker_only.assert_called_once()
            self.assertFalse(official_output.exists())
            self.assertTrue(ranker_output.is_dir())
            self.assertEqual(
                json.loads(stdout.getvalue())["full_survivor"]["status"],
                "completed",
            )

    def test_cli_rejects_skipping_both_evaluation_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "must-not-be-created"
            stderr = io.StringIO()
            with (
                mock.patch.object(EVALUATION.evaluator, "load_jsonl") as load_public,
                contextlib.redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                EVALUATION.main(
                    [
                        "--skip-official",
                        "--skip-full-survivor",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("skip all evaluation work", stderr.getvalue())
            load_public.assert_not_called()
            self.assertFalse(output_dir.exists())

    def test_official_bootstrap_uses_dynamic_candidate_label(self) -> None:
        rows = []
        for model, score in (("linear", 0.0), ("fm", 0.5), ("candidate", 1.0)):
            rows.append(
                {
                    "model": model,
                    "sample_id": "s0",
                    "scenario_type": "buying",
                    "hit": score > 0,
                    "reciprocal_rank": score,
                    "efficiency": score,
                    "technical_score_contribution": score,
                }
            )
        comparisons = {
            row["comparison"]
            for row in EVALUATION.bootstrap_rows(rows, replicates=10)
        }
        self.assertIn("candidate_minus_fm", comparisons)
        self.assertIn("candidate_minus_linear", comparisons)
        self.assertNotIn("hybrid_minus_fm", comparisons)

    def test_context_schema_matches_frozen_and_v2_runtime_paths(self) -> None:
        dataset = SimpleNamespace(
            products=(SimpleNamespace(category="outdoor gear"),)
        )
        state = SimpleNamespace(
            product_index=0,
            scenario_state="browsing",
            turn=2,
            intent_epoch=0,
            known_constraints=(("other", ("waterproof hiking",)),),
        )
        legacy = EVALUATION._state_context_features(
            dataset, state, SimpleNamespace(metadata={})
        )
        version_two = EVALUATION._state_context_features(
            dataset,
            state,
            SimpleNamespace(
                metadata={"feature_schema_version": "conversation-features-v2"}
            ),
        )
        self.assertIn("ctx:other=waterproof hiking", legacy)
        self.assertNotIn("ctx:answer_source=other", legacy)
        self.assertNotIn("ctx:use_case=waterproof hiking", legacy)
        self.assertIn("ctx:answer_source=other", version_two)
        self.assertIn("ctx:other=waterproof hiking", version_two)
        self.assertIn("ctx:use_case=waterproof hiking", version_two)

    def test_exact_rank_scores_every_survivor_and_gives_ties_half_credit(self) -> None:
        state = FakeState("s0", "t0", "B")
        dataset = FakeDataset([state], [["D", "B", "A", "C"]])
        model = FakeModel({"s0": {"A": 1.0, "B": 1.0, "C": 0.5, "D": 2.0}})

        rows = EVALUATION.score_full_survivor_states(
            dataset, model, mode="fm", model_name="candidate"
        )

        self.assertEqual(model.scored_candidate_counts, [4])
        self.assertEqual(rows[0]["target_rank"], 3)
        self.assertAlmostEqual(rows[0]["reciprocal_rank"], 1.0 / 3.0)
        self.assertEqual(rows[0]["hit_at_1"], 0)
        self.assertEqual(rows[0]["hit_at_5"], 1)
        self.assertEqual(rows[0]["hit_at_10"], 1)
        self.assertAlmostEqual(rows[0]["rank_percentile"], 2.0 / 3.0)
        self.assertAlmostEqual(rows[0]["pairwise_accuracy"], 0.5)

    def test_trajectory_macro_is_primary_and_differs_from_state_micro(self) -> None:
        states = [
            FakeState("s0", "t0", "A"),
            FakeState("s1", "t0", "A", turn=4, turn_bucket="middle"),
            FakeState("s2", "t1", "A"),
        ]
        dataset = FakeDataset(states, [["A", "B"], ["A", "B"], ["A", "B"]])
        model = FakeModel(
            {
                "s0": {"A": 2.0, "B": 1.0},
                "s1": {"A": 1.0, "B": 2.0},
                "s2": {"A": 2.0, "B": 1.0},
            }
        )
        rows = EVALUATION.score_full_survivor_states(dataset, model)
        summary = EVALUATION.summarize_full_survivor(rows)

        self.assertAlmostEqual(summary["primary"]["mrr"], 0.875)
        self.assertAlmostEqual(summary["state_micro"]["mrr"], 5.0 / 6.0)
        self.assertEqual(summary["primary"]["aggregation"], "trajectory_macro")
        self.assertEqual(summary["trajectory_count"], 2)

    def test_split_filter_preserves_dataset_global_survivor_offsets(self) -> None:
        states = [
            FakeState("train-0", "t0", "TRAIN0", split="train"),
            FakeState("validation-0", "t1", "TARGET0", split="validation"),
            FakeState("train-1", "t2", "TRAIN1", split="train"),
            FakeState("validation-1", "t3", "TARGET1", split="validation"),
        ]
        dataset = FakeDataset(
            states,
            [
                ["TRAIN0"],
                ["TARGET0", "OTHER0"],
                ["TRAIN1"],
                ["TARGET1", "OTHER1", "OTHER2"],
            ],
        )
        model = FakeModel(
            {
                "validation-0": {"TARGET0": 2.0, "OTHER0": 1.0},
                "validation-1": {
                    "TARGET1": 3.0,
                    "OTHER1": 2.0,
                    "OTHER2": 1.0,
                },
            }
        )

        rows = EVALUATION.score_full_survivor_states(
            dataset, model, split="validation"
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            [(row["state_id"], row["candidate_width"]) for row in rows],
            [("validation-0", 2), ("validation-1", 3)],
        )

    def test_hit_at_ten_is_suppressed_for_narrow_groups(self) -> None:
        state = FakeState("s0", "t0", "A")
        dataset = FakeDataset([state], [["A", "B"]])
        model = FakeModel({"s0": {"A": 2.0, "B": 1.0}})
        rows = EVALUATION.score_full_survivor_states(dataset, model)

        summary = EVALUATION.summarize_full_survivor(rows)
        self.assertIsNone(summary["primary"]["hit_rate_at_10"])
        self.assertEqual(summary["primary"]["hit_rate_at_10_raw"], 1.0)
        self.assertFalse(summary["primary"]["hit_rate_at_10_informative"])

        breakdowns = EVALUATION.full_survivor_breakdowns(rows)
        width = next(
            row
            for row in breakdowns
            if row["dimension"] == "survivor_width" and row["value"] == "<=10"
        )
        self.assertIsNone(width["metrics"]["primary"]["hit_rate_at_10"])

    def test_hit_at_ten_excludes_narrow_states_from_overall_rate(self) -> None:
        states = [
            FakeState("narrow", "t0", "A"),
            FakeState("wide", "t1", "A"),
        ]
        wide_survivors = ["A", *[f"P{index:02d}" for index in range(11)]]
        dataset = FakeDataset(states, [["A", "B"], wide_survivors])
        model = FakeModel(
            {
                "narrow": {"A": 2.0, "B": 1.0},
                "wide": {
                    "A": 0.0,
                    **{f"P{index:02d}": float(11 - index) for index in range(11)},
                },
            }
        )
        summary = EVALUATION.summarize_full_survivor(
            EVALUATION.score_full_survivor_states(dataset, model)
        )["primary"]
        self.assertEqual(summary["hit_rate_at_10_raw"], 0.5)
        self.assertEqual(summary["hit_rate_at_10"], 0.0)
        self.assertEqual(summary["hit_rate_at_10_informative_state_count"], 1)

    def test_width_and_supervision_bands_use_exact_boundaries(self) -> None:
        self.assertEqual(
            [EVALUATION.width_bucket(value) for value in (10, 11, 50, 51, 200, 201)],
            ["<=10", "11-50", "11-50", "51-200", "51-200", ">200"],
        )
        self.assertEqual(EVALUATION.supervision_weight_band(0.0), "zero")
        self.assertEqual(EVALUATION.supervision_weight_band(0.25), "low")
        self.assertEqual(EVALUATION.supervision_weight_band(0.75), "medium")
        self.assertEqual(EVALUATION.supervision_weight_band(1.0), "high")

    def test_supervision_weights_follow_each_artifacts_training_config(self) -> None:
        state = SimpleNamespace(
            state_id="s0",
            trajectory_id="t0",
            target_parent_asin="A",
            split="validation",
            scenario="browsing",
            scenario_state="browsing",
            turn=1,
            turn_bucket="early",
            has_other_answer=False,
            known_constraints=(),
        )
        dataset = FakeDataset([state], [["A", "B"]])
        baseline = FakeModel({"s0": {"A": 1.0, "B": 0.0}})
        candidate = FakeModel({"s0": {"A": 1.0, "B": 0.0}})
        baseline.metadata.update(
            {"category_only_weight": "0.05", "evidence_saturation": "3"}
        )
        candidate.metadata.update(
            {"category_only_weight": "0.50", "evidence_saturation": "2"}
        )
        baseline_rows = EVALUATION.score_full_survivor_states(
            dataset, baseline, model_name="baseline"
        )
        candidate_rows = EVALUATION.score_full_survivor_states(
            dataset, candidate, model_name="candidate"
        )
        self.assertEqual(baseline_rows[0]["supervision_weight"], 0.05)
        self.assertEqual(candidate_rows[0]["supervision_weight"], 0.5)
        self.assertEqual(baseline_rows[0]["supervision_weight_band"], "low")
        self.assertEqual(candidate_rows[0]["supervision_weight_band"], "medium")
        paired = EVALUATION.pair_full_survivor_rows(
            candidate_rows, baseline_rows
        )
        self.assertEqual(paired[0]["supervision_weight_band"], "medium")

    def test_paired_join_is_strict_and_bootstraps_trajectories(self) -> None:
        states = [
            FakeState("s0", "t0", "A"),
            FakeState("s1", "t0", "A"),
            FakeState("s2", "t1", "A"),
        ]
        dataset = FakeDataset(states, [["A", "B"]] * 3)
        baseline = FakeModel(
            {state.state_id: {"A": 1.0, "B": 2.0} for state in states}, seed=1
        )
        candidate = FakeModel(
            {state.state_id: {"A": 2.0, "B": 1.0} for state in states}, seed=2
        )
        baseline_rows = EVALUATION.score_full_survivor_states(
            dataset, baseline, model_name="baseline"
        )
        candidate_rows = EVALUATION.score_full_survivor_states(
            dataset, candidate, model_name="candidate"
        )
        paired = EVALUATION.pair_full_survivor_rows(
            candidate_rows,
            baseline_rows,
            candidate_name="candidate",
            baseline_name="baseline",
        )
        self.assertTrue(all(row["rank_improvement"] == 1 for row in paired))

        first = EVALUATION.paired_trajectory_bootstrap(
            paired, replicates=100, seed=42
        )
        second = EVALUATION.paired_trajectory_bootstrap(
            paired, replicates=100, seed=42
        )
        self.assertEqual(first, second)
        mrr = next(row for row in first if row["metric"] == "mrr_delta")
        self.assertEqual(mrr["trajectory_count"], 2)
        self.assertGreater(mrr["observed_delta"], 0.0)
        hit_ten = next(row for row in first if row["metric"] == "hit_at_10_delta")
        self.assertFalse(hit_ten["informative"])
        self.assertIsNone(hit_ten["observed_delta"])

        with self.assertRaisesRegex(ValueError, "paired state keys differ"):
            EVALUATION.pair_full_survivor_rows(
                candidate_rows[:-1], baseline_rows
            )


if __name__ == "__main__":
    unittest.main()
