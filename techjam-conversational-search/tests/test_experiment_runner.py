from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


RUNNER_PATH = Path(__file__).resolve().parents[2] / "approach 1" / "run_fm_experiments.py"
SPEC = importlib.util.spec_from_file_location("approach1_experiment_runner_tests", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _learning_configuration() -> dict[str, object]:
    return {
        "variant": "fm",
        "dataset_version": RUNNER.EXPECTED_DATASET_VERSION,
        "catalog_sha256": "catalog-bytes-sha256",
        "catalog_records_sha256": "catalog-records-sha256",
        "scenario_mix": "public",
        "extended_fraction": 0.1,
        "supervision_policy": "set_valued_positives",
        "tie_weight": 0.1,
        "category_only_weight": 0.05,
        "evidence_saturation": 3,
        "negative_count": 16,
        "negative_pre_pool_size": 128,
        "negative_mode": "survivor_dynamic",
        "hard_fraction": 0.5,
        "near_fraction": 0.25,
        "random_fraction": 0.25,
        "other_encoding": "dual",
        "dimension": 16,
        "learning_rate": 0.01,
        "latent_l2": 1e-5,
        "linear_l2": 1e-5,
        "cross_l2": 1e-4,
        "minimum_value_support": 5,
        "minimum_cross_support": 20,
        "max_epochs": 60,
        "patience": 7,
        "validation_interval": 1,
        "pair_batch_size": 65536,
    }


def _learning_rows(
    means: dict[int, float],
    *,
    mutate: tuple[int, int, str, object] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for count in RUNNER.LEARNING_CURVE_SIZES:
        for seed in RUNNER.FINAL_SEEDS:
            row = {
                **_learning_configuration(),
                "group": "E7_learning_curve",
                "trajectory_count": count,
                "model_seed": seed,
                "trajectory_seed": 2026,
                "split_seed": 2026,
                "validation_mrr": means[count],
            }
            if mutate is not None and (count, seed) == mutate[:2]:
                row[mutate[2]] = mutate[3]
            rows.append(row)
    return rows


def _training_spec(root: Path, **overrides: object) -> RUNNER.RunSpec:
    values: dict[str, object] = {
        "python": Path(sys.executable),
        "catalog": root / "catalog.jsonl",
        "output_root": root,
        "run_id": "run",
        "group": "E7_learning_curve",
        "experiment": "E7",
        "description": "test",
        "trajectory_count": 25_000,
        "scenario_mix": "public",
        "trajectory_seed": 2026,
        "seed": 2027,
        "split_seed": 2026,
        "supervision_policy": "set_valued_positives",
        "negatives": 16,
        "dimension": 16,
        "learning_rate": 0.01,
        "latent_l2": 1e-5,
        "linear_l2": 1e-5,
        "max_epochs": 60,
        "patience": 7,
        "extended_fraction": 0.1,
        "hard_fraction": 0.5,
        "near_fraction": 0.25,
        "random_fraction": 0.25,
        "other_encoding": "dual",
    }
    values.update(overrides)
    return RUNNER._training_spec(  # type: ignore[arg-type]
        **values,
    )


def _matching_metrics(spec: RUNNER.RunSpec) -> dict[str, object]:
    command = RUNNER._command_configuration(spec.command)
    trajectory = {
        "dataset_version": RUNNER.EXPECTED_DATASET_VERSION,
        "trajectory_count": int(command["trajectory_count"]),
        "scenario_mix": command["scenario_mix"],
        "extended_fraction": float(command["extended_fraction"]),
        "seed": int(command["trajectory_seed"]),
        "split_seed": int(command["split_seed"]),
    }
    training_names = {
        "seed": "seed",
        "supervision_policy": "supervision_policy",
        "tie_weight": "tie_weight",
        "category_only_weight": "category_only_weight",
        "evidence_saturation": "evidence_saturation",
        "negatives": "negatives_per_state",
        "negative_pre_pool_size": "negative_pre_pool_size",
        "hard_fraction": "hard_fraction",
        "near_fraction": "near_fraction",
        "random_fraction": "random_fraction",
        "other_encoding": "other_encoding",
        "dimension": "dimension",
        "learning_rate": "learning_rate",
        "latent_l2": "latent_l2",
        "linear_l2": "linear_l2",
        "cross_l2": "cross_l2",
        "minimum_value_support": "minimum_value_support",
        "minimum_cross_support": "minimum_cross_support",
        "max_epochs": "max_epochs",
        "patience": "patience",
        "validation_interval": "validation_interval",
        "pair_batch_size": "pair_batch_size",
    }
    integer_fields = {
        "seed",
        "evidence_saturation",
        "negatives",
        "negative_pre_pool_size",
        "dimension",
        "minimum_value_support",
        "minimum_cross_support",
        "max_epochs",
        "patience",
        "validation_interval",
        "pair_batch_size",
    }
    string_fields = {"supervision_policy", "other_encoding"}
    training = {
        payload_name: (
            command[command_name]
            if command_name in string_fields
            else int(command[command_name])
            if command_name in integer_fields
            else float(command[command_name])
        )
        for command_name, payload_name in training_names.items()
    }
    if "negative_mode" in command:
        training["negative_mode"] = command["negative_mode"]
    catalog_hashes = RUNNER._catalog_input_hashes(Path(command["catalog"]))
    return {
        "schema_version": RUNNER.EXPECTED_TRAINING_SCHEMA_VERSION,
        "model": command["variant"],
        "artifact": str(spec.artifact_path),
        "trajectory_config": trajectory,
        "training_config": training,
        "dataset_manifest": {
            "dataset_version": RUNNER.EXPECTED_DATASET_VERSION,
            "config": dict(trajectory),
            "input_sha256": catalog_hashes,
            "product_count": 1,
        },
        "full_survivor_validation": {"mrr": 0.5},
    }


def _write_catalog(root: Path, *, title: str = "test product") -> Path:
    catalog = root / "catalog.jsonl"
    catalog.write_text(
        json.dumps(
            {
                "parent_asin": "TEST-ASIN",
                "title": title,
                "categories": ["Test"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    RUNNER._CATALOG_HASH_CACHE.clear()
    return catalog


def _write_valid_model_artifact(
    spec: RUNNER.RunSpec, payload: dict[str, object]
) -> None:
    assert spec.artifact_path is not None
    manifest = payload["dataset_manifest"]
    assert isinstance(manifest, dict)
    catalog_hashes = manifest["input_sha256"]
    assert isinstance(catalog_hashes, dict)
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    with sqlite3.connect(spec.artifact_path) as connection:
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
                PRIMARY KEY(context_feature_id,item_feature_id)
            ) WITHOUT ROWID;
            CREATE TABLE reply_values(
                parent_asin TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                attribute TEXT NOT NULL,
                normalized_value TEXT NOT NULL,
                PRIMARY KEY(parent_asin,ordinal)
            ) WITHOUT ROWID;
            """
        )
        metadata = {
            "schema_version": "1",
            "training_schema_version": RUNNER.EXPECTED_TRAINING_SCHEMA_VERSION,
            "dataset_version": RUNNER.EXPECTED_DATASET_VERSION,
            "dataset_manifest_sha256": manifest_sha256,
            "catalog_sha256": str(catalog_hashes["catalog"]),
            "model_type": "second_order_factorization_machine",
            "product_count": "1",
        }
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)", metadata.items()
        )
        connection.execute(
            "INSERT INTO context_features VALUES (1,'ctx:test','test',?)", (b"v",)
        )
        connection.execute(
            "INSERT INTO item_features VALUES (1,'item:test','test',0.0,?)", (b"v",)
        )
        connection.execute(
            "INSERT INTO products VALUES ('TEST-ASIN',0.0,?,?)", (b"v", b"i")
        )
    command = RUNNER._command_configuration(spec.command)
    RUNNER._write_json(Path(command["manifest"]), manifest)
    Path(command["negative_audit"]).write_text(
        ",".join(sorted(RUNNER._REQUIRED_NEGATIVE_AUDIT_COLUMNS)) + "\n",
        encoding="utf-8",
    )
    Path(command["cross_audit"]).write_text(
        ",".join(sorted(RUNNER._REQUIRED_CROSS_AUDIT_COLUMNS)) + "\n",
        encoding="utf-8",
    )


def _write_failed_validation_manifest(
    spec: RUNNER.RunSpec, *, validation_errors: list[str] | None = None
) -> dict[str, object]:
    manifest = RUNNER._manifest_payload(spec, "failed_validation")
    manifest.update(
        {
            "started_utc": "2026-01-01T00:00:00+00:00",
            "finished_utc": "2026-01-01T00:01:00+00:00",
            "return_code": 0,
            "metrics_complete": False,
        }
    )
    if validation_errors is not None:
        manifest["validation_errors"] = validation_errors
    RUNNER._write_json(spec.manifest_path, manifest)
    return manifest


def _write_completed_training(
    spec: RUNNER.RunSpec, payload: dict[str, object] | None = None
) -> dict[str, object]:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    payload = _matching_metrics(spec) if payload is None else payload
    _write_valid_model_artifact(spec, payload)
    RUNNER._write_json(spec.completion_path, payload)
    manifest = RUNNER._manifest_payload(spec, "completed")
    manifest.update(
        {
            "started_utc": "2026-01-01T00:00:00+00:00",
            "finished_utc": "2026-01-01T00:01:00+00:00",
            "return_code": 0,
            "metrics_complete": True,
            "validation_errors": [],
        }
    )
    RUNNER._write_json(spec.manifest_path, manifest)
    return payload


def _ranker_summary(candidate_artifact: Path, *, candidate_mrr: float = 0.60) -> dict:
    def report(artifact: Path, mrr: float, hit_at_10: float) -> dict:
        breakdowns = []
        for dimension, value in (("scenario", "buying"), ("turn_bucket", "early")):
            breakdowns.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "metrics": {
                        "state_count": 10,
                        "trajectory_count": 5,
                        "primary": {
                            "hit_rate_at_10": hit_at_10,
                            "hit_rate_at_10_informative": True,
                        },
                    },
                }
            )
        return {
            "artifact": str(artifact),
            "artifact_sha256": RUNNER._sha256_path(artifact),
            "overall": {
                "state_count": 10,
                "trajectory_count": 5,
                "primary": {"mrr": mrr},
            },
            "breakdowns": breakdowns,
        }

    dataset_manifest = {
        "dataset_version": RUNNER.EXPECTED_DATASET_VERSION,
        "config": {},
        "input_sha256": {},
    }
    cohort = {
        "split": "validation",
        "state_count": 10,
        "trajectory_count": 5,
        "state_cohort_sha256": "a" * 64,
        "dataset_manifest": dataset_manifest,
        "dataset_manifest_sha256": RUNNER._json_sha256(dataset_manifest),
    }
    cohort["cohort_sha256"] = RUNNER._json_sha256(cohort)
    return {
        "schema_version": RUNNER.FULL_SURVIVOR_SCHEMA_VERSION,
        "evaluation_protocol": RUNNER.FULL_SURVIVOR_PROTOCOL,
        "split": "validation",
        "cohort": cohort,
        "models": {
            "fm": report(RUNNER.APPROACH_ROOT / "fm_only_model.sqlite3", 0.50, 0.50),
            "candidate": report(candidate_artifact, candidate_mrr, 0.51),
        },
    }


def _correctness_report() -> dict[str, object]:
    output = (
        "----------------------------------------------------------------------\n"
        f"Ran {len(set(RUNNER.CORRECTNESS_TEST_IDS.values()))} tests in 0.001s\n\nOK\n"
    )
    return {
        "schema_version": RUNNER.CORRECTNESS_EVIDENCE_SCHEMA_VERSION,
        "status": "passed",
        "return_code": 0,
        "tests_run": len(set(RUNNER.CORRECTNESS_TEST_IDS.values())),
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "command": list(RUNNER._correctness_command(Path(sys.executable))),
        "working_directory": str(RUNNER.PROJECT_ROOT),
        "source_sha256": RUNNER._correctness_source_sha256(),
        "test_ids": list(dict.fromkeys(RUNNER.CORRECTNESS_TEST_IDS.values())),
        "required_checks": {
            name: {"test_id": test_id, "passed": True}
            for name, test_id in RUNNER.CORRECTNESS_TEST_IDS.items()
        },
        "output": output,
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }


class CommandAndCumulativeSpecTest(unittest.TestCase):
    def test_default_command_remains_legacy_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = _training_spec(Path(directory))
        command = list(spec.command)
        self.assertNotIn("--negative-mode", command)
        self.assertEqual(command[command.index("--tie-weight") + 1], "0.10")
        self.assertEqual(
            command[command.index("--category-only-weight") + 1], "0.05"
        )
        self.assertEqual(command[command.index("--evidence-saturation") + 1], "3")

    def test_nondefault_sampler_and_weights_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = RUNNER.build_parser().parse_args(
                [
                    "--mode",
                    "smoke",
                    "--output-root",
                    directory,
                    "--negative-mode",
                    "product_fixed",
                    "--tie-weight",
                    "0.25",
                    "--category-only-weight",
                    "0.5",
                    "--evidence-saturation",
                    "2",
                ]
            )
            spec = RUNNER.specs_for_mode(args)[0]
        configuration = RUNNER._command_configuration(spec.command)
        self.assertEqual(configuration["negative_mode"], "product_fixed")
        self.assertEqual(configuration["tie_weight"], "0.25")
        self.assertEqual(configuration["category_only_weight"], "0.5")
        self.assertEqual(configuration["evidence_saturation"], "2")

    def test_cumulative_mode_is_an_exact_exploratory_ladder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = RUNNER.build_parser().parse_args(
                ["--mode", "cumulative", "--output-root", directory]
            )
            specs = RUNNER.specs_for_mode(args)
        self.assertEqual(
            [spec.experiment for spec in specs],
            ["E1", "E2", "E3", "E4a", "E4", "E5", "E6"],
        )
        self.assertTrue(
            all("exploratory" in spec.group.lower() for spec in specs)
        )
        self.assertTrue(
            all("not promotion evidence" in spec.description for spec in specs)
        )
        configurations = [
            RUNNER._command_configuration(spec.command) for spec in specs
        ]
        for configuration in configurations:
            configuration["negative_mode"] = RUNNER._canonical_negative_mode(
                configuration.get("negative_mode")
            )
            self.assertEqual(configuration["trajectory_count"], "25000")
            self.assertEqual(configuration["seed"], "2026")

        e1, e2, e3, e4a, e4, e5, e6 = configurations
        self.assertEqual(
            (
                e1["scenario_mix"],
                e1["extended_fraction"],
                e1["negative_mode"],
                e1["negatives"],
                e1["supervision_policy"],
                e1["tie_weight"],
                e1["category_only_weight"],
                e1["evidence_saturation"],
                e1["other_encoding"],
            ),
            (
                "balanced",
                "0.0",
                "product_fixed",
                "8",
                "downweight_ties",
                "1.0",
                "1.0",
                "1",
                "legacy",
            ),
        )
        self.assertEqual(e2["scenario_mix"], "public")
        self.assertEqual(e2["extended_fraction"], "0.1")
        self.assertEqual(e2["negative_mode"], "product_fixed")
        self.assertEqual(
            (
                e3["negative_mode"],
                e3["hard_fraction"],
                e3["near_fraction"],
                e3["random_fraction"],
            ),
            ("survivor_static", "0.0", "0.0", "1.0"),
        )
        self.assertEqual(
            (
                e4a["negative_mode"],
                e4a["negatives"],
                e4a["hard_fraction"],
                e4a["near_fraction"],
                e4a["random_fraction"],
            ),
            ("survivor_dynamic", "8", "0.5", "0.25", "0.25"),
        )
        self.assertEqual(e4["negatives"], "16")
        self.assertEqual(e5["supervision_policy"], "set_valued_positives")
        self.assertEqual(e5["other_encoding"], "legacy")
        self.assertEqual(e6["other_encoding"], "dual")
        held_fixed = (
            "scenario_mix",
            "extended_fraction",
            "negative_mode",
            "negatives",
            "hard_fraction",
            "near_fraction",
            "random_fraction",
            "supervision_policy",
            "tie_weight",
            "category_only_weight",
            "evidence_saturation",
        )
        self.assertTrue(all(e5[field] == e6[field] for field in held_fixed))

    def test_cumulative_omits_duplicate_selected_e4_at_eight_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = RUNNER.build_parser().parse_args(
                [
                    "--mode",
                    "cumulative",
                    "--negatives",
                    "8",
                    "--output-root",
                    directory,
                ]
            )
            experiments = [
                spec.experiment for spec in RUNNER.specs_for_mode(args)
            ]
        self.assertEqual(experiments, ["E1", "E2", "E3", "E4a", "E5", "E6"])

class PlateauDecisionTest(unittest.TestCase):
    def test_later_rebound_prevents_early_plateau(self) -> None:
        decision = RUNNER.plateau_decision(
            _learning_rows({25_000: 0.500, 50_000: 0.501, 100_000: 0.510})
        )
        self.assertTrue(decision["all_required_sizes_complete"])
        self.assertEqual(decision["selected_trajectory_count"], 100_000)

    def test_smallest_size_is_selected_only_when_every_later_gain_is_small(self) -> None:
        decision = RUNNER.plateau_decision(
            _learning_rows({25_000: 0.5000, 50_000: 0.5010, 100_000: 0.5015})
        )
        self.assertEqual(decision["selected_trajectory_count"], 25_000)

    def test_mismatched_configuration_invalidates_learning_curve(self) -> None:
        rows = _learning_rows(
            {25_000: 0.500, 50_000: 0.501, 100_000: 0.502},
            mutate=(50_000, 2028, "negative_count", 32),
        )
        decision = RUNNER.plateau_decision(rows)
        self.assertFalse(decision["all_required_sizes_complete"])
        self.assertIsNone(decision["selected_trajectory_count"])

    def test_mismatched_seed_set_invalidates_learning_curve(self) -> None:
        rows = _learning_rows(
            {25_000: 0.500, 50_000: 0.501, 100_000: 0.502},
            mutate=(100_000, 2028, "model_seed", 2029),
        )
        decision = RUNNER.plateau_decision(rows)
        self.assertFalse(decision["consistent_model_seed_sets"])
        self.assertIsNone(decision["selected_trajectory_count"])

    def test_learning_curve_chart_covers_both_metrics_and_all_cost_axes(self) -> None:
        svg = RUNNER.learning_curve_svg(
            [
                {
                    "trajectory_count": 25_000,
                    "mean_state_count": 80_000,
                    "mean_effective_weighted_pairs": 900_000,
                    "mean_training_seconds": 12.5,
                    "validation_mrr": 0.61,
                    "validation_hit_at_10": 0.88,
                }
            ]
        )
        for label in (
            "Validation MRR",
            "Validation HR@10",
            "Complete trajectories",
            "Retained states",
            "Effective weighted pairs",
            "Training time (seconds)",
        ):
            self.assertIn(label, svg)


class ResumeValidationTest(unittest.TestCase):
    def test_manifestless_training_result_is_adopted_only_after_exact_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_catalog(root)
            spec = _training_spec(root)
            spec.output_dir.mkdir(parents=True)
            payload = _matching_metrics(spec)
            _write_valid_model_artifact(spec, payload)
            RUNNER._write_json(spec.completion_path, payload)

            self.assertTrue(RUNNER._is_complete(spec))
            self.assertEqual(RUNNER.aggregate_metric_rows(root), [])
            with mock.patch.object(RUNNER.subprocess, "run") as run:
                self.assertEqual(
                    RUNNER.execute_spec(spec, force=False), "skipped_completed"
                )
            run.assert_not_called()
            manifest = json.loads(spec.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "adopted_completed")
            rows = RUNNER.aggregate_metric_rows(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["negative_mode"], "survivor_dynamic")

    def test_nondefault_negative_mode_requires_an_exact_stored_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_catalog(root)
            spec = _training_spec(root, negative_mode="survivor_static")
            spec.output_dir.mkdir(parents=True)
            payload = _matching_metrics(spec)
            _write_valid_model_artifact(spec, payload)
            self.assertEqual(
                payload["training_config"]["negative_mode"],  # type: ignore[index]
                "survivor_static",
            )
            self.assertEqual(RUNNER._training_payload_errors(spec, payload), [])
            del payload["training_config"]["negative_mode"]  # type: ignore[index]
            self.assertIn(
                "training_config.negative_mode does not match",
                RUNNER._training_payload_errors(spec, payload),
            )

    def test_manifestless_result_rejects_config_version_and_artifact_mismatches(self) -> None:
        mutations = (
            lambda payload, root: payload["training_config"].__setitem__(
                "dimension", 32
            ),
            lambda payload, root: payload["trajectory_config"].__setitem__(
                "dataset_version", "fm-trajectories-v1"
            ),
            lambda payload, root: payload.__setitem__(
                "artifact", str(root / "different.sqlite3")
            ),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_catalog(root)
                spec = _training_spec(root)
                spec.output_dir.mkdir(parents=True)
                payload = _matching_metrics(spec)
                _write_valid_model_artifact(spec, payload)
                mutate(payload, root)
                RUNNER._write_json(spec.completion_path, payload)
                self.assertFalse(RUNNER._is_complete(spec))

    def test_manifestless_result_rejects_changed_catalog_and_corrupt_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_catalog(root)
            spec = _training_spec(root)
            spec.output_dir.mkdir(parents=True)
            payload = _matching_metrics(spec)
            _write_valid_model_artifact(spec, payload)
            RUNNER._write_json(spec.completion_path, payload)
            self.assertTrue(RUNNER._is_complete(spec))

            _write_catalog(root, title="catalog changed after training")
            self.assertFalse(RUNNER._is_complete(spec))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_catalog(root)
            spec = _training_spec(root)
            spec.output_dir.mkdir(parents=True)
            assert spec.artifact_path is not None
            spec.artifact_path.touch()
            RUNNER._write_json(spec.completion_path, _matching_metrics(spec))
            self.assertFalse(RUNNER._is_complete(spec))

    def test_corrupt_manifest_is_not_treated_as_manifestless_adoption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_catalog(root)
            spec = _training_spec(root)
            spec.output_dir.mkdir(parents=True)
            payload = _matching_metrics(spec)
            _write_valid_model_artifact(spec, payload)
            RUNNER._write_json(spec.completion_path, payload)
            spec.manifest_path.write_text("{", encoding="utf-8")

            self.assertFalse(RUNNER._is_complete(spec))
            self.assertIn(
                "failed-validation runner manifest is missing or invalid JSON",
                RUNNER._recoverable_training_output_errors(spec),
            )

    def test_training_completion_requires_all_diagnostic_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_catalog(root)
            spec = _training_spec(root)
            spec.output_dir.mkdir(parents=True)
            payload = _matching_metrics(spec)
            _write_valid_model_artifact(spec, payload)
            command = RUNNER._command_configuration(spec.command)
            Path(command["negative_audit"]).unlink()
            self.assertTrue(
                any(
                    "negative_audit" in error
                    for error in RUNNER._training_payload_errors(spec, payload)
                )
            )

    def test_zero_exit_failed_validation_output_can_be_adopted_without_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_catalog(root)
            spec = _training_spec(root)
            spec.output_dir.mkdir(parents=True)
            payload = _matching_metrics(spec)
            _write_valid_model_artifact(spec, payload)
            RUNNER._write_json(spec.completion_path, payload)
            original = _write_failed_validation_manifest(
                spec, validation_errors=["diagnostic was not yet visible"]
            )

            with mock.patch.object(RUNNER.subprocess, "run") as run:
                result = RUNNER.execute_spec(
                    spec, force=False, adopt_valid_output=True
                )

            run.assert_not_called()
            self.assertEqual(result, "adopted_completed")
            adopted = json.loads(spec.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(adopted["status"], "adopted_completed")
            self.assertTrue(adopted["metrics_complete"])
            self.assertEqual(adopted["validation_errors"], [])
            self.assertEqual(adopted["command"], original["command"])
            self.assertEqual(adopted["finished_utc"], original["finished_utc"])
            self.assertEqual(
                adopted["recovery"]["previous_validation_errors"],
                ["diagnostic was not yet visible"],
            )
            self.assertTrue(RUNNER._is_complete(spec))

    def test_failed_validation_adoption_rejects_manifest_and_artifact_mismatches(self) -> None:
        for mutation in ("command", "return_code", "artifact"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_catalog(root)
                spec = _training_spec(root)
                spec.output_dir.mkdir(parents=True)
                payload = _matching_metrics(spec)
                _write_valid_model_artifact(spec, payload)
                RUNNER._write_json(spec.completion_path, payload)
                manifest = _write_failed_validation_manifest(spec)
                if mutation == "command":
                    manifest["command"] = [*manifest["command"], "--unexpected"]
                    RUNNER._write_json(spec.manifest_path, manifest)
                elif mutation == "return_code":
                    manifest["return_code"] = 1
                    RUNNER._write_json(spec.manifest_path, manifest)
                else:
                    assert spec.artifact_path is not None
                    spec.artifact_path.write_bytes(b"not a sqlite model")

                with mock.patch.object(RUNNER.subprocess, "run") as run:
                    with self.assertRaisesRegex(RuntimeError, "cannot adopt"):
                        RUNNER.execute_spec(
                            spec, force=False, adopt_valid_output=True
                        )
                run.assert_not_called()
                recorded = json.loads(
                    spec.manifest_path.read_text(encoding="utf-8")
                )
                self.assertEqual(recorded["status"], "failed_validation")

    def test_failed_validation_adoption_is_training_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = RUNNER.RunSpec(
                run_id="evaluation",
                group="test",
                experiment="test",
                description="",
                output_dir=root / "evaluation",
                command=(sys.executable, "evaluate.py"),
                completion_path=root / "evaluation" / "summary.json",
                kind="evaluation",
            )
            self.assertEqual(
                RUNNER._recoverable_training_output_errors(spec),
                ["only training outputs can be recovered without re-execution"],
            )

    def test_adoption_cli_targets_only_the_named_run_and_skips_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = _write_catalog(root)
            spec = _training_spec(root)
            spec.output_dir.mkdir(parents=True)
            payload = _matching_metrics(spec)
            _write_valid_model_artifact(spec, payload)
            RUNNER._write_json(spec.completion_path, payload)
            _write_failed_validation_manifest(spec)

            with (
                mock.patch.object(RUNNER, "specs_for_mode", return_value=[spec]),
                mock.patch.object(
                    RUNNER, "stage_prerequisite_errors", return_value=[]
                ),
                mock.patch.object(RUNNER, "aggregate_metric_rows") as aggregate,
                mock.patch.object(RUNNER.subprocess, "run") as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                return_code = RUNNER.main(
                    [
                        "--mode",
                        "cumulative",
                        "--execute",
                        "--adopt-valid-run",
                        spec.run_id,
                        "--catalog",
                        str(catalog),
                        "--output-root",
                        str(root),
                    ]
                )

            self.assertEqual(return_code, 0)
            run.assert_not_called()
            aggregate.assert_not_called()
            manifest = json.loads(spec.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "adopted_completed")


class LearningCurveEvidenceReuseTest(unittest.TestCase):
    @staticmethod
    def _matched_specs(root: Path) -> tuple[RUNNER.RunSpec, RUNNER.RunSpec]:
        settings = {
            "seed": 2026,
            "supervision_policy": "downweight_ties",
            "negatives": 32,
            "hard_fraction": 0.0,
            "near_fraction": 0.5,
            "random_fraction": 0.5,
            "other_encoding": "dual",
        }
        source = _training_spec(
            root,
            **settings,
            run_id="mixture_no_model_hard_n32_25k_tseed2026_mseed2026",
            group="E4_negative_mixture",
            experiment="E4",
        )
        target = _training_spec(
            root,
            **settings,
            run_id="025k_public_tseed2026_mseed2026",
            group="E7_learning_curve",
            experiment="E7",
        )
        return source, target

    def test_exact_source_is_reused_without_copy_or_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_catalog(root)
            source, target = self._matched_specs(root)
            _write_completed_training(source)

            plan = RUNNER._plan_learning_curve_evidence(
                [target], [source.completion_path], root
            )
            self.assertEqual(set(plan), {target.run_id})
            self.assertEqual(
                RUNNER._write_learning_curve_evidence(
                    target, source.completion_path, root
                ),
                "reused_evidence",
            )
            self.assertTrue(RUNNER._is_complete(target))
            with mock.patch.object(RUNNER.subprocess, "run") as run:
                result = RUNNER.execute_spec(target, force=False)
            run.assert_not_called()
            self.assertEqual(result, "skipped_reused_evidence")

            self.assertEqual(
                {path.name for path in target.output_dir.iterdir()},
                {RUNNER.LEARNING_CURVE_EVIDENCE_FILENAME},
            )
            rows = RUNNER.aggregate_metric_rows(root)
            e7_rows = [row for row in rows if row.get("group") == target.group]
            self.assertEqual(len(e7_rows), 1)
            row = e7_rows[0]
            self.assertEqual(row["run_id"], target.run_id)
            self.assertTrue(row["evidence_reused"])
            self.assertEqual(row["evidence_source_run_id"], source.run_id)
            self.assertEqual(row["metrics_path"], str(source.completion_path.resolve()))
            self.assertEqual(row["artifact"], str(source.artifact_path))

    def test_reuse_rejects_any_generating_configuration_mismatch(self) -> None:
        mismatches = {
            "model seed": {"seed": 2027},
            "mixture": {
                "hard_fraction": 0.5,
                "near_fraction": 0.25,
                "random_fraction": 0.25,
            },
            "dimension": {"dimension": 32},
            "near-equal regularization": {"latent_l2": 1.00000001e-5},
        }
        for label, overrides in mismatches.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_catalog(root)
                source, target = self._matched_specs(root)
                _write_completed_training(source)
                target_settings = {
                    "seed": 2026,
                    "supervision_policy": "downweight_ties",
                    "negatives": 32,
                    "hard_fraction": 0.0,
                    "near_fraction": 0.5,
                    "random_fraction": 0.5,
                    "other_encoding": "dual",
                    **overrides,
                }
                target = _training_spec(
                    root,
                    run_id=target.run_id,
                    group=target.group,
                    experiment=target.experiment,
                    **target_settings,
                )
                with self.assertRaisesRegex(ValueError, "matches 0 exact E7 targets"):
                    RUNNER._plan_learning_curve_evidence(
                        [target], [source.completion_path], root
                    )
                self.assertFalse(target.output_dir.exists())

    def test_reference_fails_closed_when_source_evidence_changes(self) -> None:
        mutations = {
            "catalog": lambda root, source: _write_catalog(
                root, title="changed after evidence was recorded"
            ),
            "artifact": lambda root, source: source.artifact_path.write_bytes(
                b"not the attested sqlite artifact"
            ),
            "manifest": lambda root, source: RUNNER._write_json(
                source.manifest_path,
                {
                    **json.loads(source.manifest_path.read_text(encoding="utf-8")),
                    "metrics_complete": False,
                },
            ),
            "diagnostic": lambda root, source: Path(
                RUNNER._command_configuration(source.command)["negative_audit"]
            ).write_text("changed\n", encoding="utf-8"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_catalog(root)
                source, target = self._matched_specs(root)
                _write_completed_training(source)
                RUNNER._write_learning_curve_evidence(
                    target, source.completion_path, root
                )

                mutate(root, source)

                self.assertTrue(RUNNER._learning_curve_evidence_errors(target))
                self.assertFalse(RUNNER._is_complete(target))
                self.assertEqual(
                    [
                        row
                        for row in RUNNER.aggregate_metric_rows(root)
                        if row.get("group") == "E7_learning_curve"
                    ],
                    [],
                )

    def test_direct_target_output_wins_and_is_not_duplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_catalog(root)
            source, target = self._matched_specs(root)
            _write_completed_training(source)
            RUNNER._write_learning_curve_evidence(
                target, source.completion_path, root
            )
            _write_completed_training(target)

            rows = [
                row
                for row in RUNNER.aggregate_metric_rows(root)
                if row.get("group") == "E7_learning_curve"
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], target.run_id)
            self.assertNotIn("evidence_reused", rows[0])

    def test_partial_direct_diagnostic_invalidates_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_catalog(root)
            source, target = self._matched_specs(root)
            _write_completed_training(source)
            RUNNER._write_learning_curve_evidence(
                target, source.completion_path, root
            )
            target_dataset_manifest = Path(
                RUNNER._command_configuration(target.command)["manifest"]
            )
            target_dataset_manifest.write_text("{}\n", encoding="utf-8")

            self.assertIn(
                "direct target output exists alongside reused evidence",
                RUNNER._learning_curve_evidence_errors(target),
            )
            self.assertFalse(RUNNER._is_complete(target))
            self.assertEqual(
                [
                    row
                    for row in RUNNER.aggregate_metric_rows(root)
                    if row.get("group") == "E7_learning_curve"
                ],
                [],
            )
            with self.assertRaisesRegex(ValueError, "partial direct output"):
                RUNNER._plan_learning_curve_evidence(
                    [target], [source.completion_path], root
                )

    def test_reuse_evidence_binds_local_training_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_catalog(root)
            source, target = self._matched_specs(root)
            mutable_trainer = root / "same-path-trainer.py"
            mutable_trainer.write_text("original source\n", encoding="utf-8")

            def with_trainer(spec: RUNNER.RunSpec) -> RUNNER.RunSpec:
                command = list(spec.command)
                command[1] = str(mutable_trainer)
                return RUNNER.RunSpec(
                    run_id=spec.run_id,
                    group=spec.group,
                    experiment=spec.experiment,
                    description=spec.description,
                    output_dir=spec.output_dir,
                    command=tuple(command),
                    completion_path=spec.completion_path,
                    artifact_path=spec.artifact_path,
                    prerequisites=spec.prerequisites,
                    kind=spec.kind,
                )

            source = with_trainer(source)
            target = with_trainer(target)
            _write_completed_training(source)
            RUNNER._write_learning_curve_evidence(
                target, source.completion_path, root
            )
            payload = json.loads(
                RUNNER._learning_curve_evidence_path(target).read_text(
                    encoding="utf-8"
                )
            )
            files = payload["source"]["files"]
            for name in (
                "trainer_source",
                "fm_training_source",
                "trajectory_source",
                "conversation_features_source",
            ):
                self.assertIn(name, files)
                self.assertEqual(len(files[name]["sha256"]), 64)

            mutable_trainer.write_text("changed source\n", encoding="utf-8")
            errors = RUNNER._learning_curve_evidence_errors(target)
            self.assertIn(
                "learning-curve evidence file changed: trainer_source",
                errors,
            )
            self.assertFalse(RUNNER._is_complete(target))

    def test_invalid_direct_attestation_blocks_direct_and_reused_rows(self) -> None:
        mutations = {
            "metrics incomplete": {"metrics_complete": False},
            "validation errors": {"validation_errors": ["failed validation"]},
            "nonzero return": {"return_code": 1},
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_catalog(root)
                source, target = self._matched_specs(root)
                _write_completed_training(source)
                RUNNER._write_learning_curve_evidence(
                    target, source.completion_path, root
                )
                _write_completed_training(target)
                manifest = json.loads(
                    target.manifest_path.read_text(encoding="utf-8")
                )
                manifest.update(mutation)
                RUNNER._write_json(target.manifest_path, manifest)

                self.assertTrue(
                    RUNNER._training_manifest_attestation_errors(manifest)
                )
                self.assertFalse(RUNNER._is_complete(target))
                self.assertEqual(
                    [
                        row
                        for row in RUNNER.aggregate_metric_rows(root)
                        if row.get("group") == "E7_learning_curve"
                    ],
                    [],
                )

    def test_cli_reuse_flag_is_learning_curve_only_and_force_safe(self) -> None:
        parser = RUNNER.build_parser()
        with tempfile.TemporaryDirectory() as directory:
            evidence = str(Path(directory) / "metrics.json")
            for arguments, message in (
                (
                    ["--mode", "smoke", "--reuse-learning-curve-evidence", evidence],
                    "requires --mode learning-curve",
                ),
                (
                    [
                        "--mode",
                        "learning-curve",
                        "--force",
                        "--reuse-learning-curve-evidence",
                        evidence,
                    ],
                    "cannot be combined with --force",
                ),
            ):
                with self.subTest(arguments=arguments):
                    parser.parse_args(arguments)
                    with self.assertRaisesRegex(SystemExit, message):
                        RUNNER.main(arguments)

    def test_cli_records_reference_and_never_launches_reused_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = _write_catalog(root)
            source, target = self._matched_specs(root)
            unrelated = _training_spec(
                root,
                seed=2027,
                supervision_policy="downweight_ties",
                negatives=32,
                hard_fraction=0.0,
                near_fraction=0.5,
                random_fraction=0.5,
                other_encoding="dual",
                run_id="025k_public_tseed2026_mseed2027",
                group="E7_learning_curve",
                experiment="E7",
            )
            _write_completed_training(source)
            active_manifest = RUNNER._manifest_payload(unrelated, "launching")
            active_manifest["started_utc"] = RUNNER.utc_now()
            RUNNER._write_json(unrelated.manifest_path, active_manifest)
            arguments = [
                "--mode",
                "learning-curve",
                "--execute",
                "--catalog",
                str(catalog),
                "--output-root",
                str(root),
                "--supervision-policy",
                "downweight_ties",
                "--negatives",
                "32",
                "--hard-fraction",
                "0",
                "--near-fraction",
                "0.5",
                "--random-fraction",
                "0.5",
                "--reuse-learning-curve-evidence",
                str(source.completion_path),
            ]
            with (
                mock.patch.object(
                    RUNNER, "specs_for_mode", return_value=[target, unrelated]
                ),
                mock.patch.object(
                    RUNNER, "stage_prerequisite_errors", return_value=[]
                ),
                mock.patch.object(RUNNER.subprocess, "run") as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(RUNNER.main(arguments), 0)

            run.assert_not_called()
            self.assertTrue(
                (target.output_dir / RUNNER.LEARNING_CURVE_EVIDENCE_FILENAME).is_file()
            )
            self.assertFalse(target.completion_path.exists())
            assert target.artifact_path is not None
            self.assertFalse(target.artifact_path.exists())
            self.assertEqual(
                json.loads(unrelated.manifest_path.read_text(encoding="utf-8")),
                active_manifest,
            )


class StageSafetyTest(unittest.TestCase):
    def test_execute_spec_fails_closed_for_missing_prerequisite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = RUNNER.RunSpec(
                run_id="missing",
                group="test",
                experiment="test",
                description="",
                output_dir=root / "run",
                command=(sys.executable, "does-not-run.py"),
                completion_path=root / "run" / "summary.json",
                prerequisites=(root / "missing.sqlite3",),
                kind="evaluation",
            )
            with mock.patch.object(RUNNER.subprocess, "run") as run:
                with self.assertRaises(FileNotFoundError):
                    RUNNER.execute_spec(spec, force=False)
            run.assert_not_called()

    def test_execute_spec_refuses_launching_manifest_even_with_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = _training_spec(root)
            RUNNER._write_json(
                spec.manifest_path, RUNNER._manifest_payload(spec, "launching")
            )
            for force in (False, True):
                with self.subTest(force=force), mock.patch.object(
                    RUNNER.subprocess, "run"
                ) as run:
                    with self.assertRaisesRegex(
                        RuntimeError, "refusing a duplicate subprocess"
                    ):
                        RUNNER.execute_spec(spec, force=force)
                run.assert_not_called()

    def test_execute_spec_rejects_zero_return_without_valid_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = _training_spec(root)
            completed = mock.Mock(returncode=0)
            with mock.patch.object(RUNNER.subprocess, "run", return_value=completed):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "validated outputs"):
                        RUNNER.execute_spec(spec, force=False)
            manifest = json.loads(spec.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed_validation")
            self.assertFalse(manifest["metrics_complete"])
            self.assertIn(
                "completion payload is missing or invalid JSON",
                manifest["validation_errors"],
            )

    def test_execute_spec_records_successful_terminal_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_catalog(root)
            spec = _training_spec(root)
            spec.output_dir.mkdir(parents=True)
            payload = _matching_metrics(spec)
            _write_valid_model_artifact(spec, payload)
            RUNNER._write_json(spec.completion_path, payload)
            completed = mock.Mock(returncode=0)

            with mock.patch.object(
                RUNNER.subprocess, "run", return_value=completed
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = RUNNER.execute_spec(spec, force=True)

            self.assertEqual(result, "completed")
            manifest = json.loads(spec.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue(manifest["metrics_complete"])
            self.assertEqual(manifest["validation_errors"], [])

    def test_execute_spec_rejects_malformed_evaluator_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = RUNNER._evaluation_spec(
                python=Path(sys.executable),
                catalog=root / "catalog.jsonl",
                output_root=root,
                run_id="evaluation",
                group="E0_full_survivor_baseline",
                experiment="E0",
                description="",
                linear_model=RUNNER.APPROACH_ROOT / "linear_model.sqlite3",
                fm_model=RUNNER.APPROACH_ROOT / "fm_only_model.sqlite3",
                hybrid_model=RUNNER.APPROACH_ROOT / "fm_model.sqlite3",
                trajectory_count=800,
                trajectory_seed=2026,
                split_seed=2026,
                scenario_mix="public",
                prerequisites=(
                    RUNNER.APPROACH_ROOT / "linear_model.sqlite3",
                    RUNNER.APPROACH_ROOT / "fm_only_model.sqlite3",
                    RUNNER.APPROACH_ROOT / "fm_model.sqlite3",
                ),
            )
            spec.completion_path.parent.mkdir(parents=True)
            RUNNER._write_json(spec.completion_path, {})
            completed = mock.Mock(returncode=0)
            with mock.patch.object(RUNNER.subprocess, "run", return_value=completed):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "validated outputs"):
                        RUNNER.execute_spec(spec, force=False)

    def test_all_inventory_has_no_official_specs_and_execution_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = RUNNER.build_parser().parse_args(
                ["--mode", "all", "--output-root", directory]
            )
            specs = RUNNER.specs_for_mode(args)
            self.assertNotIn("E8_official_handoff", {spec.group for spec in specs})
            self.assertNotIn(
                "E1_E6_cumulative_exploratory", {spec.group for spec in specs}
            )
            with mock.patch.object(RUNNER.subprocess, "run") as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(SystemExit, "E1-E6"):
                        RUNNER.main(
                            [
                                "--mode",
                                "all",
                                "--execute",
                                "--output-root",
                                directory,
                            ]
                        )
            run.assert_not_called()

    def test_standalone_downstream_modes_refuse_missing_decisions(self) -> None:
        for mode in ("tuning", "final-seeds", "official"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                with mock.patch.object(RUNNER.subprocess, "run") as run:
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            RUNNER.main(
                                [
                                    "--mode",
                                    mode,
                                    "--execute",
                                    "--output-root",
                                    directory,
                                ]
                            )
                run.assert_not_called()

    def test_downstream_dry_run_reports_blocker_without_fabricated_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                return_code = RUNNER.main(
                    ["--mode", "tuning", "--output-root", directory]
                )
            self.assertEqual(return_code, 0)
            self.assertIn("BLOCKED: E7", output.getvalue())
            self.assertNotIn("Bounded full-survivor validation tuning", output.getvalue())


class PromotionEvidenceTest(unittest.TestCase):
    def test_promotion_evidence_must_match_one_candidate_and_four_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = [root / f"seed-{seed}.sqlite3" for seed in RUNNER.FINAL_SEEDS]
            for artifact in artifacts:
                artifact.touch()
            ranker_reports = []
            for seed, artifact in zip(RUNNER.FINAL_SEEDS, artifacts, strict=True):
                path = root / f"ranker-{seed}.json"
                path.write_text(
                    json.dumps(_ranker_summary(artifact)), encoding="utf-8"
                )
                ranker_reports.append(path)
            correctness = root / "correctness.json"
            correctness.write_text(
                json.dumps(_correctness_report()),
                encoding="utf-8",
            )
            report = root / "promotion.json"
            payload = {
                "schema_version": RUNNER.PROMOTION_EVIDENCE_SCHEMA_VERSION,
                "decision": "approved_for_official_evaluation",
                "selected_candidate_artifact": str(artifacts[0]),
                "final_seed_artifacts": [str(path) for path in artifacts],
                "gates": {
                    **{
                        name: {
                            "passed": True,
                            "evidence_paths": [str(path) for path in ranker_reports],
                        }
                        for name in RUNNER.RANKER_PROMOTION_GATES
                    },
                    "correctness_tests_passed": {
                        "passed": True,
                        "evidence_paths": [str(correctness)],
                    },
                },
            }
            report.write_text(json.dumps(payload), encoding="utf-8")
            specs = [
                RUNNER.RunSpec(
                    run_id=str(index),
                    group="E8_final_seeds",
                    experiment="E8",
                    description="",
                    output_dir=root / str(index),
                    command=(),
                    completion_path=root / str(index) / "metrics.json",
                    artifact_path=artifact,
                )
                for index, artifact in enumerate(artifacts)
            ]
            args = Namespace(
                candidate_artifact=[artifacts[0]], promotion_evidence=report
            )
            self.assertEqual(RUNNER.promotion_evidence_errors(args, specs), [])

            payload["gates"][RUNNER.PRE_OFFICIAL_PROMOTION_GATES[0]]["passed"] = False
            report.write_text(json.dumps(payload), encoding="utf-8")
            errors = RUNNER.promotion_evidence_errors(args, specs)
            self.assertTrue(any("did not pass" in error for error in errors))

    def test_arbitrary_text_and_failing_ranker_metrics_cannot_approve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = [root / f"seed-{seed}.sqlite3" for seed in RUNNER.FINAL_SEEDS]
            for artifact in artifacts:
                artifact.touch()
            ranker_reports = []
            for seed, artifact in zip(RUNNER.FINAL_SEEDS, artifacts, strict=True):
                path = root / f"ranker-{seed}.json"
                path.write_text(json.dumps(_ranker_summary(artifact)), encoding="utf-8")
                ranker_reports.append(path)
            arbitrary = root / "verified.txt"
            arbitrary.write_text("verified\n", encoding="utf-8")
            report = root / "promotion.json"
            payload = {
                "schema_version": RUNNER.PROMOTION_EVIDENCE_SCHEMA_VERSION,
                "decision": "approved_for_official_evaluation",
                "selected_candidate_artifact": str(artifacts[0]),
                "final_seed_artifacts": [str(path) for path in artifacts],
                "gates": {
                    **{
                        name: {
                            "passed": True,
                            "evidence_paths": [str(path) for path in ranker_reports],
                        }
                        for name in RUNNER.RANKER_PROMOTION_GATES
                    },
                    "correctness_tests_passed": {
                        "passed": True,
                        "evidence_paths": [str(arbitrary)],
                    },
                },
            }
            report.write_text(json.dumps(payload), encoding="utf-8")
            specs = [
                RUNNER.RunSpec(
                    run_id=str(index),
                    group="E8_final_seeds",
                    experiment="E8",
                    description="",
                    output_dir=root / str(index),
                    command=(),
                    completion_path=root / str(index) / "metrics.json",
                    artifact_path=artifact,
                )
                for index, artifact in enumerate(artifacts)
            ]
            args = Namespace(candidate_artifact=[artifacts[0]], promotion_evidence=report)
            errors = RUNNER.promotion_evidence_errors(args, specs)
            self.assertTrue(any("not valid JSON" in error for error in errors))

            ranker_reports[0].write_text(
                json.dumps(_ranker_summary(artifacts[0], candidate_mrr=0.40)),
                encoding="utf-8",
            )
            self.assertTrue(
                any(
                    "machine-validated promotion gate failed" in error
                    for error in RUNNER.promotion_evidence_errors(args, specs)
                )
            )

    def test_ranker_evidence_is_bound_to_artifact_and_identical_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = [root / f"seed-{seed}.sqlite3" for seed in RUNNER.FINAL_SEEDS]
            for artifact in artifacts:
                artifact.write_bytes(b"model")
            paths = []
            for seed, artifact in zip(RUNNER.FINAL_SEEDS, artifacts, strict=True):
                path = root / f"ranker-{seed}.json"
                path.write_text(json.dumps(_ranker_summary(artifact)), encoding="utf-8")
                paths.append(path)
            results, errors = RUNNER._machine_ranker_gate_results(
                paths, {path.resolve() for path in artifacts}
            )
            self.assertEqual(errors, [])
            self.assertTrue(all(results.values()))

            artifacts[0].write_bytes(b"changed model")
            _, errors = RUNNER._machine_ranker_gate_results(
                paths, {path.resolve() for path in artifacts}
            )
            self.assertTrue(any("artifact digest differs" in error for error in errors))

            artifacts[0].write_bytes(b"model")
            changed = json.loads(paths[0].read_text(encoding="utf-8"))
            changed["cohort"]["state_cohort_sha256"] = "b" * 64
            cohort_without_digest = {
                key: value
                for key, value in changed["cohort"].items()
                if key != "cohort_sha256"
            }
            changed["cohort"]["cohort_sha256"] = RUNNER._json_sha256(
                cohort_without_digest
            )
            paths[0].write_text(json.dumps(changed), encoding="utf-8")
            _, errors = RUNNER._machine_ranker_gate_results(
                paths, {path.resolve() for path in artifacts}
            )
            self.assertTrue(any("identical bound cohort" in error for error in errors))

    def test_correctness_evidence_is_source_bound_and_can_be_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "correctness.json"
            payload = _correctness_report()
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                RUNNER._machine_correctness_evidence_errors([path]), []
            )

            payload["source_sha256"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(
                any(
                    "stale" in error
                    for error in RUNNER._machine_correctness_evidence_errors([path])
                )
            )

            path.write_text(json.dumps(_correctness_report()), encoding="utf-8")
            with mock.patch.object(
                RUNNER, "_run_correctness_suite", return_value={"status": "failed"}
            ):
                errors = RUNNER._machine_correctness_evidence_errors(
                    [path], verify_execution=True
                )
            self.assertTrue(any("re-execution" in error for error in errors))

    def test_post_official_gate_enforces_thresholds_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.sqlite3"
            candidate.write_bytes(b"candidate")
            spec = RUNNER._evaluation_spec(
                python=Path(sys.executable),
                catalog=root / "catalog.jsonl",
                output_root=root,
                run_id="official",
                group="E8_official_handoff",
                experiment="E8/promotion",
                description="",
                linear_model=root / "linear.sqlite3",
                fm_model=root / "fm.sqlite3",
                hybrid_model=candidate,
                trajectory_count=800,
                trajectory_seed=2026,
                split_seed=2026,
                scenario_mix="public",
                third_model_name="candidate",
                third_model_mode="fm",
            )
            official_dir = spec.output_dir / "official"
            ablation_rows = [
                {
                    "model": "candidate",
                    "scenario": "overall",
                    "sample_count": 200,
                    "correct_answers": 199,
                    "mrr": 0.70,
                },
                *[
                    {
                        "model": "candidate",
                        "scenario": scenario,
                        "sample_count": 1,
                        "correct_answers": 1,
                        "mrr": 1.0,
                    }
                    for scenario in ("buying", "browsing", "boundary", "intent_override")
                ],
            ]
            RUNNER._write_csv(
                official_dir / "model_ablation.csv",
                ("model", "scenario", "sample_count", "correct_answers", "mrr"),
                ablation_rows,
            )
            RUNNER._write_csv(
                official_dir / "model_ablation_sessions.csv",
                ("model", "sample_id"),
                [
                    {"model": "candidate", "sample_id": f"s{index:03d}"}
                    for index in range(200)
                ],
            )
            RUNNER._write_csv(
                official_dir / "model_ablation_bootstrap.csv",
                (
                    "comparison",
                    "metric",
                    "sample_count",
                    "observed_delta",
                    "ci_95_lower",
                    "ci_95_upper",
                ),
                [
                    {
                        "comparison": f"candidate_minus_{baseline}",
                        "metric": metric,
                        "sample_count": 200,
                        "observed_delta": 0.01,
                        "ci_95_lower": -0.01,
                        "ci_95_upper": 0.03,
                    }
                    for baseline in ("fm", "linear")
                    for metric in ("accuracy", "mrr", "efficiency", "technical_score")
                ],
            )
            payload = RUNNER._post_official_gate_payload(spec)
            self.assertEqual(payload["decision"], "eligible_for_manual_promotion")
            RUNNER._write_json(
                spec.output_dir / "post_official_promotion_decision.json", payload
            )
            self.assertEqual(RUNNER._post_official_gate_errors(spec), [])

            ablation_rows[0]["mrr"] = RUNNER.OFFICIAL_MRR_GATE
            RUNNER._write_csv(
                official_dir / "model_ablation.csv",
                ("model", "scenario", "sample_count", "correct_answers", "mrr"),
                ablation_rows,
            )
            failed = RUNNER._post_official_gate_payload(spec)
            self.assertEqual(failed["decision"], "not_eligible_for_promotion")


if __name__ == "__main__":
    unittest.main()
