#!/usr/bin/env python3
"""Resumable orchestration for the trajectory-aligned FM E0--E8 program.

The default behavior is deliberately read-only: commands and completion state
are printed, but no directories are created and no subprocesses are launched.
Pass ``--execute`` to write manifests, run commands, and refresh aggregates.
All generated artifacts live below a versioned experiment root; the committed
Linear/FM/Hybrid reference artifacts are inputs only and are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
APPROACH_ROOT = SCRIPT_PATH.parent
REPOSITORY_ROOT = APPROACH_ROOT.parent
PROJECT_ROOT = REPOSITORY_ROOT / "techjam-conversational-search"
TRAINER_PATH = APPROACH_ROOT / "train_fm.py"
EVALUATOR_PATH = APPROACH_ROOT / "evaluate_fm.py"
DEFAULT_CATALOG = PROJECT_ROOT / "data/catalog.jsonl"
DEFAULT_OUTPUT_ROOT = APPROACH_ROOT / "results" / "redesign" / "experiments_v2"
EXPERIMENT_SCHEMA_VERSION = "fm-experiment-runner-v2"
EXPECTED_DATASET_VERSION = "fm-trajectories-v2"
EXPECTED_TRAINING_SCHEMA_VERSION = "fm-training-v2"
DEFAULT_NEGATIVE_MODE = "survivor_dynamic"
LEARNING_CURVE_EVIDENCE_SCHEMA_VERSION = "fm-learning-curve-evidence-v2"
LEARNING_CURVE_EVIDENCE_FILENAME = "reused_training_evidence.json"
PROMOTION_EVIDENCE_SCHEMA_VERSION = "fm-promotion-evidence-v1"
CORRECTNESS_EVIDENCE_SCHEMA_VERSION = "fm-correctness-evidence-v2"
FULL_SURVIVOR_SCHEMA_VERSION = "1"
FULL_SURVIVOR_PROTOCOL = "exact_full_survivor_ranker_only_v1"
PLATEAU_MRR_THRESHOLD = 0.002
FINAL_SEEDS = (2026, 2027, 2028)
LEARNING_CURVE_SIZES = (25_000, 50_000, 100_000)
TUNING_DIMENSIONS = (8, 16, 32)
TUNING_LEARNING_RATES = (0.005, 0.01, 0.02)
TUNING_REGULARIZATIONS = (1e-6, 1e-5, 1e-4)
PRE_OFFICIAL_PROMOTION_GATES = (
    "full_survivor_validation_mrr_improved",
    "no_material_hit_at_10_regression",
    "correctness_tests_passed",
    "consistent_direction_across_three_seeds",
)
RANKER_PROMOTION_GATES = (
    "full_survivor_validation_mrr_improved",
    "no_material_hit_at_10_regression",
    "consistent_direction_across_three_seeds",
)
REQUIRED_CORRECTNESS_CHECKS = (
    "filtering",
    "rollback",
    "intent_override",
    "ten_turn",
)
CORRECTNESS_TEST_IDS = {
    "filtering": (
        "tests.test_agent.ProgressiveAgentTest."
        "test_same_and_different_attribute_values_use_and_with_rollback"
    ),
    "rollback": (
        "tests.test_agent.ProgressiveAgentTest."
        "test_same_and_different_attribute_values_use_and_with_rollback"
    ),
    "intent_override": (
        "tests.test_agent.ProgressiveAgentTest."
        "test_intent_override_replaces_obsolete_filters_atomically"
    ),
    "ten_turn": (
        "tests.test_agent.ProgressiveAgentTest."
        "test_early_stop_returns_all_survivors_and_turn_ten_never_asks"
    ),
}
OFFICIAL_CORRECT_ANSWERS_GATE = 199
OFFICIAL_MRR_GATE = 0.645863
POST_OFFICIAL_GATE_SCHEMA_VERSION = "fm-post-official-gate-v1"
LEARNING_CURVE_MATCH_FIELDS = (
    "variant",
    "dataset_version",
    "catalog_sha256",
    "catalog_records_sha256",
    "scenario_mix",
    "extended_fraction",
    "supervision_policy",
    "tie_weight",
    "category_only_weight",
    "evidence_saturation",
    "negative_count",
    "negative_pre_pool_size",
    "negative_mode",
    "hard_fraction",
    "near_fraction",
    "random_fraction",
    "other_encoding",
    "dimension",
    "learning_rate",
    "latent_l2",
    "linear_l2",
    "cross_l2",
    "minimum_value_support",
    "minimum_cross_support",
    "max_epochs",
    "patience",
    "validation_interval",
    "pair_batch_size",
)

FROZEN_ARTIFACTS = {
    (APPROACH_ROOT / "fm_model.sqlite3").resolve(),
    (APPROACH_ROOT / "fm_only_model.sqlite3").resolve(),
    (APPROACH_ROOT / "linear_model.sqlite3").resolve(),
    (APPROACH_ROOT / "training_metrics.json").resolve(),
    (APPROACH_ROOT / "fm_only_training_metrics.json").resolve(),
    (APPROACH_ROOT / "linear_training_metrics.json").resolve(),
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    group: str
    experiment: str
    description: str
    output_dir: Path
    command: tuple[str, ...]
    completion_path: Path
    artifact_path: Path | None = None
    prerequisites: tuple[Path, ...] = ()
    kind: str = "training"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "run_manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or ())
            return [dict(row) for row in reader], fields
    except (OSError, csv.Error, UnicodeError):
        return [], []


def _resolve_recorded_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _same_config_value(expected: object, actual: object) -> bool:
    """Compare command-line strings with the typed values stored in metrics."""

    if isinstance(actual, bool):
        return expected is actual or str(expected).lower() == str(actual).lower()
    expected_number = _as_float(expected)
    actual_number = _as_float(actual)
    if expected_number is not None and actual_number is not None:
        return math.isclose(expected_number, actual_number, rel_tol=1e-12, abs_tol=1e-12)
    return str(expected) == str(actual)


def _same_generation_option(expected: object, actual: object) -> bool:
    """Compare two command options without accepting near-equal numerics."""

    try:
        expected_number = Decimal(str(expected))
        actual_number = Decimal(str(actual))
    except (InvalidOperation, ValueError):
        return str(expected) == str(actual)
    if not expected_number.is_finite() or not actual_number.is_finite():
        return str(expected) == str(actual)
    return expected_number == actual_number


def _canonical_negative_mode(value: object) -> str:
    """Map pre-mode v2 records to their historical dynamic behavior."""

    if value is None or value == "":
        return DEFAULT_NEGATIVE_MODE
    return str(value)


_CATALOG_HASH_CACHE: dict[
    tuple[str, int, int, int, int, int], dict[str, str]
] = {}


def _catalog_input_hashes(catalog_path: Path) -> dict[str, str]:
    """Hash the exact catalog bytes and the parsed JSONL record stream.

    The latter intentionally mirrors ``trajectory_data.hash_catalog_records``.
    The stat-keyed cache avoids reparsing the 50k-product catalog for every
    completed seed while still invalidating on ordinary in-place replacement.
    """

    resolved = catalog_path.resolve()
    stat = resolved.stat()
    cache_key = (
        str(resolved),
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )
    cached = _CATALOG_HASH_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    file_digest = hashlib.sha256()
    record_digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            file_digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid catalog JSON on line {line_number}: {resolved}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"catalog line {line_number} is not a JSON object: {resolved}"
                )
            record_digest.update(
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
            record_digest.update(b"\n")
    result = {
        "catalog": file_digest.hexdigest(),
        "catalog_records": record_digest.hexdigest(),
    }
    _CATALOG_HASH_CACHE.clear()
    _CATALOG_HASH_CACHE[cache_key] = result
    return dict(result)


_REQUIRED_MODEL_COLUMNS = {
    "metadata": {"key", "value"},
    "context_features": {"feature_id", "name", "field", "vector"},
    "item_features": {
        "feature_id",
        "name",
        "field",
        "linear_weight",
        "vector",
    },
    "products": {"parent_asin", "base_score", "vector", "item_feature_ids"},
    "cross_weights": {
        "context_feature_id",
        "item_feature_id",
        "positive_support",
        "negative_support",
        "weight",
    },
    "reply_values": {"parent_asin", "ordinal", "attribute", "normalized_value"},
}

_REQUIRED_NEGATIVE_AUDIT_COLUMNS = {
    "epoch",
    "trajectory_id",
    "state_index",
    "target_parent_asin",
    "negative_parent_asin",
    "survivor_pool_size",
    "sampler_type",
    "trajectory_state_weight",
    "evidence_weight",
    "sampling_weight",
}
_REQUIRED_CROSS_AUDIT_COLUMNS = {
    "context_feature",
    "item_feature",
    "field_pair",
    "positive_support",
    "negative_support",
    "learned_weight",
}


def _model_artifact_errors(
    spec: RunSpec,
    payload: Mapping[str, object],
    *,
    expected_catalog_sha256: str | None,
) -> list[str]:
    """Perform a read-only integrity and schema check on a trained artifact."""

    artifact = spec.artifact_path
    if artifact is None or not artifact.is_file():
        return ["model artifact is missing or is not a regular file"]
    if artifact.stat().st_size == 0:
        return ["model artifact is empty"]

    errors: list[str] = []
    dataset_manifest = payload.get("dataset_manifest")
    dataset_manifest = (
        dataset_manifest if isinstance(dataset_manifest, Mapping) else {}
    )
    expected_manifest_sha256 = hashlib.sha256(
        json.dumps(dataset_manifest, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    expected_model_type = {
        "linear": "linear_pairwise_ranker",
        "fm": "second_order_factorization_machine",
        "hybrid": "second_order_fm_plus_explicit_crosses",
    }.get(str(payload.get("model")))

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"{artifact.resolve().as_uri()}?mode=ro", uri=True
        )
        connection.execute("PRAGMA query_only = ON")
        quick_check = connection.execute("PRAGMA quick_check(1)").fetchone()
        if quick_check is None or quick_check[0] != "ok":
            errors.append("model artifact failed SQLite quick_check")

        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing_tables = sorted(set(_REQUIRED_MODEL_COLUMNS) - tables)
        if missing_tables:
            errors.append(
                "model artifact is missing required tables: "
                + ", ".join(missing_tables)
            )
        for table, required_columns in _REQUIRED_MODEL_COLUMNS.items():
            if table not in tables:
                continue
            columns = {
                str(row[1])
                for row in connection.execute(f'PRAGMA table_info("{table}")')
            }
            missing_columns = sorted(required_columns - columns)
            if missing_columns:
                errors.append(
                    f"model artifact table {table} is missing columns: "
                    + ", ".join(missing_columns)
                )

        metadata = (
            {
                str(key): str(value)
                for key, value in connection.execute("SELECT key, value FROM metadata")
            }
            if "metadata" in tables
            else {}
        )
        expected_metadata = {
            "schema_version": "1",
            "training_schema_version": EXPECTED_TRAINING_SCHEMA_VERSION,
            "dataset_version": EXPECTED_DATASET_VERSION,
            "dataset_manifest_sha256": expected_manifest_sha256,
        }
        if expected_catalog_sha256 is not None:
            expected_metadata["catalog_sha256"] = expected_catalog_sha256
        if expected_model_type is not None:
            expected_metadata["model_type"] = expected_model_type
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                errors.append(f"model artifact metadata {key} does not match")

        if "products" in tables:
            product_count = int(
                connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]
            )
            if product_count <= 0:
                errors.append("model artifact contains no products")
            manifest_product_count = _as_int(dataset_manifest.get("product_count"))
            metadata_product_count = _as_int(metadata.get("product_count"))
            if manifest_product_count != product_count:
                errors.append("model product count disagrees with dataset manifest")
            if metadata_product_count != product_count:
                errors.append("model product count disagrees with metadata")
        for table in ("context_features", "item_features"):
            if table in tables:
                count = int(
                    connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                )
                if count <= 0:
                    errors.append(f"model artifact contains no {table}")
    except (OSError, sqlite3.DatabaseError, ValueError) as error:
        errors.append(f"model artifact is not a valid readable model database: {error}")
    finally:
        if connection is not None:
            connection.close()
    return errors


def _training_payload_errors(
    spec: RunSpec, payload: Mapping[str, object]
) -> list[str]:
    """Prove that a training result was produced for this exact ``RunSpec``.

    This validation intentionally uses the trainer's embedded configuration,
    rather than trusting filenames.  It is also the only path by which a
    manifestless result can be adopted.
    """

    errors: list[str] = []
    command = _command_configuration(spec.command)
    training = payload.get("training_config")
    training = training if isinstance(training, Mapping) else {}
    trajectory = payload.get("trajectory_config")
    trajectory = trajectory if isinstance(trajectory, Mapping) else {}
    dataset_manifest = payload.get("dataset_manifest")
    dataset_manifest = (
        dataset_manifest if isinstance(dataset_manifest, Mapping) else {}
    )
    dataset_config = dataset_manifest.get("config")
    dataset_config = dataset_config if isinstance(dataset_config, Mapping) else {}

    if payload.get("schema_version") != EXPECTED_TRAINING_SCHEMA_VERSION:
        errors.append("training schema version does not match the runner")
    if payload.get("model") != command.get("variant"):
        errors.append("model variant does not match the expected command")

    recorded_artifact = _resolve_recorded_path(payload.get("artifact"))
    if spec.artifact_path is None or recorded_artifact != spec.artifact_path.resolve():
        errors.append("metrics artifact path does not match the expected artifact")

    trajectory_fields = {
        "trajectory_count": "trajectory_count",
        "scenario_mix": "scenario_mix",
        "extended_fraction": "extended_fraction",
        "trajectory_seed": "seed",
        "split_seed": "split_seed",
    }
    training_fields = {
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
    for command_key, payload_key in trajectory_fields.items():
        if command_key not in command or not _same_config_value(
            command[command_key], trajectory.get(payload_key)
        ):
            errors.append(f"trajectory_config.{payload_key} does not match")
    for command_key, payload_key in training_fields.items():
        if command_key not in command or not _same_config_value(
            command[command_key], training.get(payload_key)
        ):
            errors.append(f"training_config.{payload_key} does not match")
    expected_negative_mode = _canonical_negative_mode(command.get("negative_mode"))
    recorded_negative_mode = _canonical_negative_mode(training.get("negative_mode"))
    if expected_negative_mode != recorded_negative_mode:
        errors.append("training_config.negative_mode does not match")

    versions = (
        trajectory.get("dataset_version"),
        dataset_manifest.get("dataset_version"),
        dataset_config.get("dataset_version"),
    )
    if any(value != EXPECTED_DATASET_VERSION for value in versions):
        errors.append(
            f"dataset version must be {EXPECTED_DATASET_VERSION} in every embedded manifest"
        )
    for key in ("trajectory_count", "scenario_mix", "extended_fraction"):
        if not _same_config_value(trajectory.get(key), dataset_config.get(key)):
            errors.append(f"dataset manifest config disagrees on {key}")
    for trajectory_key, dataset_key in (("seed", "seed"), ("split_seed", "split_seed")):
        if not _same_config_value(
            trajectory.get(trajectory_key), dataset_config.get(dataset_key)
        ):
            errors.append(f"dataset manifest config disagrees on {dataset_key}")
    input_hashes = dataset_manifest.get("input_sha256")
    input_hashes = input_hashes if isinstance(input_hashes, Mapping) else {}
    catalog_path = _resolve_recorded_path(command.get("catalog"))
    expected_catalog_hashes: dict[str, str] = {}
    if catalog_path is None:
        errors.append("training command does not identify a catalog")
    else:
        try:
            expected_catalog_hashes = _catalog_input_hashes(catalog_path)
        except (OSError, ValueError) as error:
            errors.append(f"current catalog cannot be hashed: {error}")
    for key, expected in expected_catalog_hashes.items():
        if input_hashes.get(key) != expected:
            errors.append(f"dataset manifest input hash does not match current {key}")

    diagnostic_specs = (
        ("manifest", None),
        ("negative_audit", _REQUIRED_NEGATIVE_AUDIT_COLUMNS),
        ("cross_audit", _REQUIRED_CROSS_AUDIT_COLUMNS),
    )
    for command_key, required_columns in diagnostic_specs:
        path = _resolve_recorded_path(command.get(command_key))
        if path is None or not path.is_file() or path.stat().st_size == 0:
            errors.append(f"required training diagnostic is missing or empty: {command_key}")
            continue
        if command_key == "manifest":
            disk_manifest = _read_json(path)
            if disk_manifest != dataset_manifest:
                errors.append(
                    "on-disk dataset manifest differs from embedded dataset manifest"
                )
            continue
        _, columns = _read_csv_rows(path)
        missing_columns = sorted(set(required_columns or ()) - set(columns))
        if missing_columns:
            errors.append(
                f"training diagnostic {command_key} is missing columns: "
                + ", ".join(missing_columns)
            )
    errors.extend(
        _model_artifact_errors(
            spec,
            payload,
            expected_catalog_sha256=expected_catalog_hashes.get("catalog"),
        )
    )
    return errors


def _evaluation_payload_errors(
    spec: RunSpec, payload: Mapping[str, object]
) -> list[str]:
    """Validate both ranker-only and requested official evaluator outputs."""

    errors: list[str] = []
    command = _command_configuration(spec.command)
    if payload.get("schema_version") != FULL_SURVIVOR_SCHEMA_VERSION:
        errors.append("full-survivor summary schema is invalid")
    if payload.get("evaluation_protocol") != FULL_SURVIVOR_PROTOCOL:
        errors.append("full-survivor summary protocol is invalid")
    if payload.get("split") != command.get("full_survivor_split"):
        errors.append("full-survivor summary split differs from the command")
    models = payload.get("models")
    models = models if isinstance(models, Mapping) else {}
    model_commands = {
        "linear": command.get("linear_model"),
        "fm": command.get("fm_model"),
        str(command.get("third_model_name", "hybrid")): command.get("hybrid_model"),
    }
    for name, expected_artifact_value in model_commands.items():
        report = models.get(name)
        if not isinstance(report, Mapping):
            errors.append(f"full-survivor summary lacks model: {name}")
            continue
        overall = report.get("overall")
        overall = overall if isinstance(overall, Mapping) else {}
        primary = overall.get("primary")
        primary = primary if isinstance(primary, Mapping) else {}
        if _as_float(primary.get("mrr")) is None:
            errors.append(f"full-survivor summary lacks finite MRR for {name}")
        recorded_artifact = _evidence_recorded_path(
            report.get("artifact"), spec.completion_path
        )
        expected_artifact = _resolve_recorded_path(expected_artifact_value)
        if recorded_artifact != expected_artifact:
            errors.append(f"full-survivor artifact differs for {name}")

    if command.get("skip_official") is not True:
        official_dir = _resolve_recorded_path(command.get("output_dir"))
        if official_dir is None:
            errors.append("evaluation command does not identify its official output")
        else:
            official_summary = _read_json(official_dir / "summary.json")
            official_200 = _read_json(official_dir / "official_200.json")
            if official_summary is None or not isinstance(
                official_summary.get("official_200"), Mapping
            ):
                errors.append("official evaluator summary is missing or invalid")
            if official_200 is None or _as_int(official_200.get("sample_count")) != 200:
                errors.append("official 200-session result is missing or invalid")
            for name in (
                "model_ablation.csv",
                "model_ablation_sessions.csv",
                "model_ablation_bootstrap.csv",
            ):
                path = official_dir / name
                if not path.is_file() or path.stat().st_size == 0:
                    errors.append(f"official evaluator output is missing or empty: {name}")
            if official_200 is not None:
                sessions = official_200.get("sessions")
                scenarios = official_200.get("scenario_metrics")
                if not isinstance(sessions, list) or len(sessions) != 200:
                    errors.append("official result must contain 200 session records")
                if not isinstance(scenarios, Mapping) or not scenarios:
                    errors.append("official result lacks per-scenario failure breakdowns")

            third_name = str(command.get("third_model_name", "hybrid"))
            ablation_rows, ablation_columns = _read_csv_rows(
                official_dir / "model_ablation.csv"
            )
            required_ablation_columns = {
                "model",
                "scenario",
                "sample_count",
                "correct_answers",
                "mrr",
            }
            if not required_ablation_columns.issubset(ablation_columns):
                errors.append("official model ablation CSV has an invalid schema")
            expected_models = {"linear", "fm", third_name}
            overall_models = {
                row.get("model")
                for row in ablation_rows
                if row.get("scenario") == "overall"
                and _as_int(row.get("sample_count")) == 200
            }
            if overall_models != expected_models:
                errors.append("official model ablation lacks complete 200-session models")
            scenario_names = {
                row.get("scenario")
                for row in ablation_rows
                if row.get("model") == third_name and row.get("scenario") != "overall"
            }
            if not {"buying", "browsing", "boundary", "intent_override"}.issubset(
                scenario_names
            ):
                errors.append("official model ablation lacks candidate scenario rows")

            session_rows, session_columns = _read_csv_rows(
                official_dir / "model_ablation_sessions.csv"
            )
            required_session_columns = {
                "model",
                "sample_id",
                "scenario_type",
                "hit",
                "reciprocal_rank",
                "efficiency",
                "technical_score_contribution",
            }
            if not required_session_columns.issubset(session_columns):
                errors.append("official per-session ablation CSV has an invalid schema")
            session_ids: dict[str, set[str]] = {
                name: {
                    row.get("sample_id", "")
                    for row in session_rows
                    if row.get("model") == name and row.get("sample_id")
                }
                for name in expected_models
            }
            if (
                any(len(values) != 200 for values in session_ids.values())
                or len({frozenset(values) for values in session_ids.values()}) != 1
            ):
                errors.append("official per-session cohorts are incomplete or unpaired")

            bootstrap_rows, bootstrap_columns = _read_csv_rows(
                official_dir / "model_ablation_bootstrap.csv"
            )
            required_bootstrap_columns = {
                "comparison",
                "metric",
                "sample_count",
                "observed_delta",
                "ci_95_lower",
                "ci_95_upper",
            }
            if not required_bootstrap_columns.issubset(bootstrap_columns):
                errors.append("official bootstrap CSV has an invalid schema")
            required_pairs = {
                (f"{third_name}_minus_fm", metric)
                for metric in ("accuracy", "mrr", "efficiency", "technical_score")
            } | {
                (f"{third_name}_minus_linear", metric)
                for metric in ("accuracy", "mrr", "efficiency", "technical_score")
            }
            observed_pairs = {
                (row.get("comparison", ""), row.get("metric", ""))
                for row in bootstrap_rows
                if _as_int(row.get("sample_count")) == 200
                and _as_float(row.get("observed_delta")) is not None
                and _as_float(row.get("ci_95_lower")) is not None
                and _as_float(row.get("ci_95_upper")) is not None
            }
            if not required_pairs.issubset(observed_pairs):
                errors.append("official bootstrap lacks paired candidate uncertainty rows")
    return errors


def _manifest_errors(
    spec: RunSpec,
    manifest: Mapping[str, object],
    *,
    allowed_statuses: frozenset[str] = frozenset(
        {"completed", "adopted_completed"}
    ),
) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        errors.append("runner schema version differs")
    if manifest.get("status") not in allowed_statuses:
        errors.append("manifest is not completed")
    for key, expected in (
        ("run_id", spec.run_id),
        ("group", spec.group),
        ("experiment", spec.experiment),
        ("kind", spec.kind),
    ):
        if manifest.get(key) != expected:
            errors.append(f"manifest {key} differs")
    if manifest.get("command") != list(spec.command):
        errors.append("manifest command differs")
    if manifest.get("configuration") != _command_configuration(spec.command):
        errors.append("manifest configuration differs")
    if (
        _resolve_recorded_path(manifest.get("completion_path"))
        != spec.completion_path.resolve()
    ):
        errors.append("manifest completion path differs")
    expected_artifact = spec.artifact_path.resolve() if spec.artifact_path else None
    if _resolve_recorded_path(manifest.get("artifact_path")) != expected_artifact:
        errors.append("manifest artifact path differs")
    return errors


def _training_manifest_attestation_errors(
    manifest: Mapping[str, object],
) -> list[str]:
    """Reject explicitly unsuccessful terminal training attestations.

    Historical completed manifests did not always record ``validation_errors``
    or ``return_code``. Missing legacy fields remain compatible, while any
    present value must attest success.
    """

    errors: list[str] = []
    if manifest.get("metrics_complete") is not True:
        errors.append("training manifest does not attest metrics_complete=true")
    if "validation_errors" in manifest and manifest.get("validation_errors") != []:
        errors.append("training manifest retains validation errors")
    if "return_code" in manifest and manifest.get("return_code") != 0:
        errors.append("training manifest records a nonzero return code")
    return errors


def _completion_errors(
    spec: RunSpec,
    *,
    allowed_manifest_statuses: frozenset[str] = frozenset(
        {"completed", "adopted_completed"}
    ),
    require_manifest: bool | None = None,
) -> list[str]:
    """Return machine-readable reasons that a run is not complete.

    Training payloads may normally be adopted without a manifest because they
    embed the full generating configuration. Evaluations cannot. Callers that
    validate a terminal runner manifest can override the accepted status while
    retaining all immutable manifest/spec checks.
    """

    errors: list[str] = []
    payload = _read_json(spec.completion_path)
    if payload is None:
        errors.append("completion payload is missing or invalid JSON")
    manifest = _read_json(spec.manifest_path)
    manifest_required = (
        spec.kind != "training" if require_manifest is None else require_manifest
    )
    if manifest is None:
        if manifest_required or spec.manifest_path.exists():
            errors.append("runner manifest is missing or invalid JSON")
    else:
        errors.extend(
            _manifest_errors(
                spec, manifest, allowed_statuses=allowed_manifest_statuses
            )
        )
    if payload is None:
        return errors
    if spec.kind == "training":
        if not isinstance(payload.get("full_survivor_validation"), dict):
            errors.append("training metrics lack full-survivor validation")
        if spec.artifact_path is None:
            errors.append("training spec lacks an artifact path")
        elif not spec.artifact_path.is_file():
            errors.append("training artifact is missing")
        errors.extend(_training_payload_errors(spec, payload))
        return errors
    # Evaluation outputs cannot be adopted without their exact completed runner
    # manifest, and both the ranker-only and requested official products must be
    # structurally complete.
    errors.extend(_evaluation_payload_errors(spec, payload))
    if spec.group == "E8_official_handoff":
        errors.extend(_post_official_gate_errors(spec))
    return errors


def _terminal_completion_errors(spec: RunSpec) -> list[str]:
    """Validate a persisted result, including its terminal attestation."""

    errors = _completion_errors(spec)
    manifest = _read_json(spec.manifest_path)
    if manifest is not None and spec.kind == "training":
        errors.extend(_training_manifest_attestation_errors(manifest))
    return list(dict.fromkeys(errors))


def _is_complete(spec: RunSpec) -> bool:
    if not _terminal_completion_errors(spec):
        return True
    return not _learning_curve_evidence_errors(spec)


def _assert_safe_output(path: Path, output_root: Path) -> None:
    resolved = path.resolve()
    if resolved in FROZEN_ARTIFACTS:
        raise ValueError(f"refusing to overwrite frozen artifact: {resolved}")
    try:
        resolved.relative_to(output_root.resolve())
    except ValueError as error:
        raise ValueError(
            f"generated output must remain below {output_root.resolve()}: {resolved}"
        ) from error


def _default_preserving_number(
    value: int | float, default: int | float, historical_token: str
) -> str:
    """Render new overrides without changing historical default commands."""

    if math.isclose(float(value), float(default), rel_tol=0.0, abs_tol=1e-12):
        return historical_token
    return str(value)


def _training_spec(
    *,
    python: Path,
    catalog: Path,
    output_root: Path,
    run_id: str,
    group: str,
    experiment: str,
    description: str,
    trajectory_count: int,
    scenario_mix: str,
    trajectory_seed: int,
    seed: int,
    split_seed: int,
    supervision_policy: str,
    negatives: int,
    dimension: int,
    learning_rate: float,
    latent_l2: float,
    linear_l2: float,
    max_epochs: int,
    patience: int,
    extended_fraction: float,
    hard_fraction: float,
    near_fraction: float,
    random_fraction: float,
    other_encoding: str,
    negative_mode: str = DEFAULT_NEGATIVE_MODE,
    tie_weight: float = 0.10,
    category_only_weight: float = 0.05,
    evidence_saturation: int = 3,
) -> RunSpec:
    output_dir = output_root / group / run_id
    artifact = output_dir / "model.sqlite3"
    metrics = output_dir / "metrics.json"
    dataset_manifest = output_dir / "dataset_manifest.json"
    negative_audit = output_dir / "negative_audit.csv"
    cross_audit = output_dir / "cross_weights.csv"
    for path in (artifact, metrics, dataset_manifest, negative_audit, cross_audit):
        _assert_safe_output(path, output_root)
    command = (
        str(python),
        str(TRAINER_PATH),
        "--catalog",
        str(catalog),
        "--trajectory-count",
        str(trajectory_count),
        "--scenario-mix",
        scenario_mix,
        "--extended-fraction",
        str(extended_fraction),
        "--trajectory-seed",
        str(trajectory_seed),
        "--seed",
        str(seed),
        "--split-seed",
        str(split_seed),
        "--variant",
        "fm",
        "--supervision-policy",
        supervision_policy,
        "--tie-weight",
        _default_preserving_number(tie_weight, 0.10, "0.10"),
        "--category-only-weight",
        _default_preserving_number(category_only_weight, 0.05, "0.05"),
        "--evidence-saturation",
        _default_preserving_number(evidence_saturation, 3, "3"),
        "--negatives",
        str(negatives),
        "--negative-pre-pool-size",
        "128",
        *(
            ("--negative-mode", negative_mode)
            if negative_mode != DEFAULT_NEGATIVE_MODE
            else ()
        ),
        "--hard-fraction",
        str(hard_fraction),
        "--near-fraction",
        str(near_fraction),
        "--random-fraction",
        str(random_fraction),
        "--other-encoding",
        other_encoding,
        "--dimension",
        str(dimension),
        "--learning-rate",
        str(learning_rate),
        "--latent-l2",
        str(latent_l2),
        "--linear-l2",
        str(linear_l2),
        "--cross-l2",
        "1e-4",
        "--minimum-value-support",
        "5",
        "--minimum-cross-support",
        "20",
        "--max-epochs",
        str(max_epochs),
        "--patience",
        str(patience),
        "--validation-interval",
        "1",
        "--pair-batch-size",
        "65536",
        "--output",
        str(artifact),
        "--metrics",
        str(metrics),
        "--manifest",
        str(dataset_manifest),
        "--negative-audit",
        str(negative_audit),
        "--cross-audit",
        str(cross_audit),
    )
    return RunSpec(
        run_id=run_id,
        group=group,
        experiment=experiment,
        description=description,
        output_dir=output_dir,
        command=command,
        completion_path=metrics,
        artifact_path=artifact,
    )


def _evaluation_spec(
    *,
    python: Path,
    catalog: Path,
    output_root: Path,
    run_id: str,
    group: str,
    experiment: str,
    description: str,
    linear_model: Path,
    fm_model: Path,
    hybrid_model: Path,
    trajectory_count: int,
    trajectory_seed: int,
    split_seed: int,
    scenario_mix: str,
    third_model_name: str = "hybrid",
    third_model_mode: str = "hybrid",
    prerequisites: Sequence[Path] = (),
) -> RunSpec:
    output_dir = output_root / group / run_id
    official_dir = output_dir / "official"
    full_survivor_dir = output_dir / "full_survivor"
    for path in (official_dir, full_survivor_dir):
        _assert_safe_output(path, output_root)
    command = (
        str(python),
        str(EVALUATOR_PATH),
        "--catalog",
        str(catalog),
        "--output-dir",
        str(official_dir),
        "--linear-model",
        str(linear_model),
        "--fm-model",
        str(fm_model),
        "--hybrid-model",
        str(hybrid_model),
        "--third-model-name",
        third_model_name,
        "--third-model-mode",
        third_model_mode,
        "--trajectory-module",
        str(APPROACH_ROOT / "trajectory_data.py"),
        "--trajectory-count",
        str(trajectory_count),
        "--trajectory-seed",
        str(trajectory_seed),
        "--split-seed",
        str(split_seed),
        "--scenario-mix",
        scenario_mix,
        "--full-survivor-split",
        "validation",
        "--bootstrap-replicates",
        "10000",
        "--redesign-output-dir",
        str(full_survivor_dir),
    )
    return RunSpec(
        run_id=run_id,
        group=group,
        experiment=experiment,
        description=description,
        output_dir=output_dir,
        command=command,
        completion_path=full_survivor_dir / "summary.json",
        artifact_path=None,
        prerequisites=tuple(prerequisites) + (APPROACH_ROOT / "trajectory_data.py",),
        kind="evaluation",
    )


def _common_training_values(
    args: argparse.Namespace,
    *,
    max_epochs: int | None = None,
    patience: int | None = None,
    dimension: int | None = None,
    learning_rate: float | None = None,
    latent_l2: float | None = None,
    linear_l2: float | None = None,
    hard_fraction: float | None = None,
    near_fraction: float | None = None,
    random_fraction: float | None = None,
    other_encoding: str | None = None,
    extended_fraction: float | None = None,
    negative_mode: str | None = None,
    tie_weight: float | None = None,
    category_only_weight: float | None = None,
    evidence_saturation: int | None = None,
) -> dict[str, object]:
    return {
        "python": args.python,
        "catalog": args.catalog,
        "output_root": args.output_root,
        "trajectory_seed": args.trajectory_seed,
        "split_seed": args.split_seed,
        "dimension": args.dimension if dimension is None else dimension,
        "learning_rate": (
            args.learning_rate if learning_rate is None else learning_rate
        ),
        "latent_l2": args.latent_l2 if latent_l2 is None else latent_l2,
        "linear_l2": args.linear_l2 if linear_l2 is None else linear_l2,
        "max_epochs": args.max_epochs if max_epochs is None else max_epochs,
        "patience": args.patience if patience is None else patience,
        "extended_fraction": (
            args.extended_fraction
            if extended_fraction is None
            else extended_fraction
        ),
        "hard_fraction": (
            args.hard_fraction if hard_fraction is None else hard_fraction
        ),
        "near_fraction": (
            args.near_fraction if near_fraction is None else near_fraction
        ),
        "random_fraction": (
            args.random_fraction if random_fraction is None else random_fraction
        ),
        "other_encoding": (
            args.other_encoding if other_encoding is None else other_encoding
        ),
        "negative_mode": (
            getattr(args, "negative_mode", DEFAULT_NEGATIVE_MODE)
            if negative_mode is None
            else negative_mode
        ),
        "tie_weight": (
            getattr(args, "tie_weight", 0.10)
            if tie_weight is None
            else tie_weight
        ),
        "category_only_weight": (
            getattr(args, "category_only_weight", 0.05)
            if category_only_weight is None
            else category_only_weight
        ),
        "evidence_saturation": (
            getattr(args, "evidence_saturation", 3)
            if evidence_saturation is None
            else evidence_saturation
        ),
    }


def smoke_specs(args: argparse.Namespace) -> list[RunSpec]:
    return [
        _training_spec(
            **_common_training_values(
                args,
                max_epochs=min(2, args.max_epochs),
                patience=min(2, args.patience),
            ),
            run_id=(
                f"800_public_tseed{args.trajectory_seed}_mseed{args.seed}"
            ),
            group="E1_smoke",
            experiment="E1-E3",
            description=(
                "800-trajectory invariant smoke: complete transitions, survivor "
                "sets, weighting, and dynamic negatives"
            ),
            trajectory_count=800,
            scenario_mix="public",
            seed=args.seed,
            supervision_policy=args.supervision_policy,
            negatives=args.negatives,
        )
    ]


def cumulative_specs(args: argparse.Namespace) -> list[RunSpec]:
    """Build the honest, single-seed cumulative E1--E6 ablation ladder."""

    group = "E1_E6_cumulative_exploratory"
    identity = f"25k_tseed{args.trajectory_seed}_mseed{args.seed}"
    caveat = (
        "Exploratory single-seed cumulative ablation; this run is descriptive "
        "and is not promotion evidence."
    )

    def make_spec(
        *,
        stage: str,
        run_label: str,
        change: str,
        scenario_mix: str,
        extended_fraction: float,
        negative_mode: str,
        negatives: int,
        hard_fraction: float,
        near_fraction: float,
        random_fraction: float,
        supervision_policy: str,
        tie_weight: float,
        category_only_weight: float,
        evidence_saturation: int,
        other_encoding: str,
    ) -> RunSpec:
        return _training_spec(
            **_common_training_values(
                args,
                extended_fraction=extended_fraction,
                negative_mode=negative_mode,
                hard_fraction=hard_fraction,
                near_fraction=near_fraction,
                random_fraction=random_fraction,
                tie_weight=tie_weight,
                category_only_weight=category_only_weight,
                evidence_saturation=evidence_saturation,
                other_encoding=other_encoding,
            ),
            run_id=f"{run_label}_{identity}",
            group=group,
            experiment=stage,
            description=f"{caveat} {stage}: {change}",
            trajectory_count=25_000,
            scenario_mix=scenario_mix,
            seed=args.seed,
            supervision_policy=supervision_policy,
            negatives=negatives,
        )

    unweighted = {
        "supervision_policy": "downweight_ties",
        "tie_weight": 1.0,
        "category_only_weight": 1.0,
        "evidence_saturation": 1,
        "other_encoding": "legacy",
    }
    result = [
        make_spec(
            stage="E1",
            run_label="e1_balanced_product_fixed_n08_legacy",
            change=(
                "complete trajectories and corrected transitions over the balanced "
                "mix, with the product-fixed legacy comparator"
            ),
            scenario_mix="balanced",
            extended_fraction=0.0,
            negative_mode="product_fixed",
            negatives=8,
            hard_fraction=0.50,
            near_fraction=0.25,
            random_fraction=0.25,
            **unweighted,
        ),
        make_spec(
            stage="E2",
            run_label="e2_public_coverage_product_fixed_n08_legacy",
            change="public scenario mix plus extended early/middle/late coverage",
            scenario_mix="public",
            extended_fraction=0.10,
            negative_mode="product_fixed",
            negatives=8,
            hard_fraction=0.50,
            near_fraction=0.25,
            random_fraction=0.25,
            **unweighted,
        ),
        make_spec(
            stage="E3",
            run_label="e3_survivor_static_random_n08_legacy",
            change="state-specific survivor negatives sampled once per state",
            scenario_mix="public",
            extended_fraction=0.10,
            negative_mode="survivor_static",
            negatives=8,
            hard_fraction=0.0,
            near_fraction=0.0,
            random_fraction=1.0,
            **unweighted,
        ),
        make_spec(
            stage="E4a",
            run_label="e4a_survivor_dynamic_n08_legacy",
            change="epoch-refreshed survivor negatives with a 50/25/25 mixture",
            scenario_mix="public",
            extended_fraction=0.10,
            negative_mode=DEFAULT_NEGATIVE_MODE,
            negatives=8,
            hard_fraction=0.50,
            near_fraction=0.25,
            random_fraction=0.25,
            **unweighted,
        ),
    ]
    if args.negatives != 8:
        result.append(
            make_spec(
                stage="E4",
                run_label=f"e4_selected_dynamic_n{args.negatives:02d}_legacy",
                change=f"selected dynamic negative count of {args.negatives}",
                scenario_mix="public",
                extended_fraction=0.10,
                negative_mode=DEFAULT_NEGATIVE_MODE,
                negatives=args.negatives,
                hard_fraction=0.50,
                near_fraction=0.25,
                random_fraction=0.25,
                **unweighted,
            )
        )

    selected_information = {
        "supervision_policy": args.supervision_policy,
        "tie_weight": args.tie_weight,
        "category_only_weight": args.category_only_weight,
        "evidence_saturation": args.evidence_saturation,
    }
    result.extend(
        (
            make_spec(
                stage="E5",
                run_label=(
                    f"e5_information_{args.supervision_policy}_"
                    f"n{args.negatives:02d}_legacy"
                ),
                change=(
                    f"selected information-aware policy {args.supervision_policy} "
                    "and requested supervision weights"
                ),
                scenario_mix="public",
                extended_fraction=0.10,
                negative_mode=DEFAULT_NEGATIVE_MODE,
                negatives=args.negatives,
                hard_fraction=0.50,
                near_fraction=0.25,
                random_fraction=0.25,
                other_encoding="legacy",
                **selected_information,
            ),
            make_spec(
                stage="E6",
                run_label=(
                    f"e6_dual_{args.supervision_policy}_n{args.negatives:02d}"
                ),
                change="dual OTHER encoding, with every E5 setting held fixed",
                scenario_mix="public",
                extended_fraction=0.10,
                negative_mode=DEFAULT_NEGATIVE_MODE,
                negatives=args.negatives,
                hard_fraction=0.50,
                near_fraction=0.25,
                random_fraction=0.25,
                other_encoding="dual",
                **selected_information,
            ),
        )
    )
    return result


def supervision_specs(args: argparse.Namespace) -> list[RunSpec]:
    return [
        _training_spec(
            **_common_training_values(args),
            run_id=(
                f"{policy}_25k_tseed{args.trajectory_seed}_mseed{args.seed}"
            ),
            group="E5_supervision",
            experiment="E5",
            description=f"Information-aware supervision policy: {policy}",
            trajectory_count=25_000,
            scenario_mix="public",
            seed=args.seed,
            supervision_policy=policy,
            negatives=args.negatives,
        )
        for policy in ("skip_ties", "downweight_ties", "set_valued_positives")
    ]


def negative_specs(args: argparse.Namespace) -> list[RunSpec]:
    return [
        _training_spec(
            **_common_training_values(args),
            run_id=(
                f"count_n{count:02d}_25k_tseed{args.trajectory_seed}_mseed{args.seed}"
            ),
            group="E4_negatives",
            experiment="E4",
            description=f"Dynamic survivor-negative comparison with {count} negatives",
            trajectory_count=25_000,
            scenario_mix="public",
            seed=args.seed,
            supervision_policy=args.supervision_policy,
            negatives=count,
        )
        for count in (8, 16, 32)
    ]


def negative_mixture_specs(args: argparse.Namespace) -> list[RunSpec]:
    profiles = (
        ("hard_heavy", 0.75, 0.125, 0.125),
        ("balanced", 0.34, 0.33, 0.33),
        ("no_model_hard", 0.0, 0.50, 0.50),
    )
    return [
        _training_spec(
            **_common_training_values(
                args,
                hard_fraction=hard,
                near_fraction=near,
                random_fraction=random,
            ),
            run_id=(
                f"mixture_{name}_n{args.negatives:02d}_25k_"
                f"tseed{args.trajectory_seed}_mseed{args.seed}"
            ),
            group="E4_negative_mixture",
            experiment="E4",
            description=(
                f"Dynamic survivor-negative mixture {name}: "
                f"hard={hard:g}, near={near:g}, random={random:g}"
            ),
            trajectory_count=25_000,
            scenario_mix="public",
            seed=args.seed,
            supervision_policy=args.supervision_policy,
            negatives=args.negatives,
        )
        for name, hard, near, random in profiles
    ]


def sensitivity_specs(args: argparse.Namespace) -> list[RunSpec]:
    return [
        _training_spec(
            **_common_training_values(args),
            run_id=(
                f"{mix}_25k_tseed{args.trajectory_seed}_mseed{args.seed}"
            ),
            group="E2_sensitivity",
            experiment="E2",
            description=f"Scenario-distribution sensitivity: {mix}",
            trajectory_count=25_000,
            scenario_mix=mix,
            seed=args.seed,
            supervision_policy=args.supervision_policy,
            negatives=args.negatives,
        )
        for mix in ("public", "balanced")
    ]


def other_encoding_specs(args: argparse.Namespace) -> list[RunSpec]:
    return [
        _training_spec(
            **_common_training_values(args, other_encoding=encoding),
            run_id=(
                f"other_{encoding}_25k_tseed{args.trajectory_seed}_mseed{args.seed}"
            ),
            group="E6_other_encoding",
            experiment="E6",
            description=f"Meaningful OTHER answer encoding: {encoding}",
            trajectory_count=25_000,
            scenario_mix="public",
            seed=args.seed,
            supervision_policy=args.supervision_policy,
            negatives=args.negatives,
        )
        for encoding in ("legacy", "dual")
    ]


def learning_curve_specs(args: argparse.Namespace) -> list[RunSpec]:
    return [
        _training_spec(
            **_common_training_values(args),
            run_id=(
                f"{count // 1000:03d}k_public_tseed{args.trajectory_seed}_"
                f"mseed{seed}"
            ),
            group="E7_learning_curve",
            experiment="E7",
            description=(
                f"Nested learning curve at {count:,} complete trajectories; "
                f"fixed trajectory seed {args.trajectory_seed}, model seed {seed}"
            ),
            trajectory_count=count,
            scenario_mix="public",
            seed=seed,
            supervision_policy=args.supervision_policy,
            negatives=args.negatives,
        )
        for count in LEARNING_CURVE_SIZES
        for seed in args.learning_curve_seeds
    ]


def _learning_rows_from_disk(output_root: Path) -> list[dict[str, object]]:
    rows = aggregate_metric_rows(output_root)
    return [row for row in rows if row.get("group") == "E7_learning_curve"]


def plateau_decision(
    rows: Sequence[Mapping[str, object]],
    threshold: float = PLATEAU_MRR_THRESHOLD,
    expected_model_seeds: int = 3,
) -> dict[str, object]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        count = _as_int(row.get("trajectory_count"))
        mrr = _as_float(row.get("validation_mrr"))
        if count is None or mrr is None:
            continue
        seed = _as_int(row.get("model_seed", row.get("seed")))
        trajectory_seed = _as_int(row.get("trajectory_seed"))
        split_seed = _as_int(row.get("split_seed"))
        configuration = {
            field: (
                _canonical_negative_mode(row.get(field))
                if field == "negative_mode"
                else row.get(field)
            )
            for field in LEARNING_CURVE_MATCH_FIELDS
        }
        grouped.setdefault(count, []).append(
            {
                "model_seed": seed,
                "validation_mrr": mrr,
                "trajectory_seed": trajectory_seed,
                "split_seed": split_seed,
                "configuration": configuration,
            }
        )
    points: list[dict[str, object]] = []
    point_configuration_keys: dict[int, str | None] = {}
    for count, seeded_rows in sorted(grouped.items()):
        values = [float(row["validation_mrr"]) for row in seeded_rows]
        model_seeds = sorted(
            {
                int(row["model_seed"])
                for row in seeded_rows
                if row["model_seed"] is not None
            }
        )
        trajectory_seeds = sorted(
            {
                int(row["trajectory_seed"])
                for row in seeded_rows
                if row["trajectory_seed"] is not None
            }
        )
        split_seeds = sorted(
            {
                int(row["split_seed"])
                for row in seeded_rows
                if row["split_seed"] is not None
            }
        )
        configurations = [
            row["configuration"]
            for row in seeded_rows
            if isinstance(row["configuration"], Mapping)
        ]
        configuration_complete = all(
            all(configuration.get(field) is not None for field in LEARNING_CURVE_MATCH_FIELDS)
            for configuration in configurations
        ) and len(configurations) == len(seeded_rows)
        configuration_keys = {
            json.dumps(configuration, sort_keys=True, separators=(",", ":"))
            for configuration in configurations
        }
        configuration_matched = configuration_complete and len(configuration_keys) == 1
        point_configuration_keys[count] = (
            next(iter(configuration_keys)) if configuration_matched else None
        )
        seed_mrr = {
            str(row["model_seed"]): row["validation_mrr"]
            for row in seeded_rows
            if row["model_seed"] is not None
        }
        points.append(
            {
                "trajectory_count": count,
                "model_seed_count": len(model_seeds),
                "model_seeds": model_seeds,
                "trajectory_seeds": trajectory_seeds,
                "split_seeds": split_seeds,
                "validation_mrr": fmean(values),
                "validation_mrr_min": min(values),
                "validation_mrr_max": max(values),
                "validation_mrr_stddev": pstdev(values) if len(values) > 1 else 0.0,
                "validation_mrr_by_model_seed": seed_mrr,
                "configuration_fields": list(LEARNING_CURVE_MATCH_FIELDS),
                "configuration": configurations[0] if configuration_matched else None,
                "configuration_matched_within_size": configuration_matched,
                "complete": (
                    len(model_seeds) == expected_model_seeds
                    and len(seeded_rows) == expected_model_seeds
                    and len(trajectory_seeds) == 1
                    and len(split_seeds) == 1
                    and configuration_matched
                ),
            }
        )
    completed_points = [
        point
        for point in points
        if point["complete"] and point["trajectory_count"] in LEARNING_CURVE_SIZES
    ]
    completed_seed_sets = {
        tuple(point["model_seeds"]) for point in completed_points
    }
    consistent_seed_sets = len(completed_seed_sets) == 1
    completed_data_seeds = {
        (
            tuple(point["trajectory_seeds"]),
            tuple(point["split_seeds"]),
        )
        for point in completed_points
    }
    consistent_data_seeds = len(completed_data_seeds) == 1
    completed_configuration_keys = {
        point_configuration_keys[int(point["trajectory_count"])]
        for point in completed_points
    }
    consistent_configurations = (
        len(completed_configuration_keys) == 1
        and None not in completed_configuration_keys
    )
    required_points = [
        next(
            (
                point
                for point in completed_points
                if point["trajectory_count"] == size
            ),
            None,
        )
        for size in LEARNING_CURVE_SIZES
    ]
    required_complete = all(
        point is not None for point in required_points
    ) and consistent_seed_sets and consistent_data_seeds and consistent_configurations
    selected: int | None = None
    comparisons: list[dict[str, object]] = []
    comparable_points = [point for point in required_points if point is not None]
    for left, right in zip(comparable_points, comparable_points[1:]):
        improvement = float(right["validation_mrr"]) - float(left["validation_mrr"])
        justified = improvement >= threshold - 1e-12
        left_by_seed = left["validation_mrr_by_model_seed"]
        right_by_seed = right["validation_mrr_by_model_seed"]
        assert isinstance(left_by_seed, Mapping)
        assert isinstance(right_by_seed, Mapping)
        seed_improvements = {
            str(seed): float(right_by_seed[str(seed)]) - float(left_by_seed[str(seed)])
            for seed in left["model_seeds"]
            if str(seed) in left_by_seed and str(seed) in right_by_seed
        }
        all_seed_gains_below = (
            len(seed_improvements) == expected_model_seeds
            and all(value < threshold - 1e-12 for value in seed_improvements.values())
        )
        comparisons.append(
            {
                "from_trajectories": left["trajectory_count"],
                "to_trajectories": right["trajectory_count"],
                "mrr_improvement": improvement,
                "larger_dataset_justified": justified,
                "mrr_improvement_by_model_seed": seed_improvements,
                "all_model_seed_gains_below_threshold": all_seed_gains_below,
            }
        )
    if required_complete:
        # A local dip is not a plateau if a later size rebounds. Select the
        # first size only when every subsequent adjacent gain remains below the
        # threshold in both the mean and each matched model seed.
        for index, point in enumerate(comparable_points[:-1]):
            subsequent = comparisons[index:]
            if all(
                not comparison["larger_dataset_justified"]
                and comparison["all_model_seed_gains_below_threshold"]
                for comparison in subsequent
            ):
                selected = int(point["trajectory_count"])
                break
        if selected is None:
            selected = int(comparable_points[-1]["trajectory_count"])
    return {
        "rule": (
            "average full-survivor validation MRR across three model seeds; "
            "select a smaller dataset only when every subsequent matched-seed "
            "gain remains below threshold"
        ),
        "threshold": threshold,
        "expected_model_seeds_per_size": expected_model_seeds,
        "all_required_sizes_complete": required_complete,
        "consistent_model_seed_sets": consistent_seed_sets,
        "consistent_trajectory_and_split_seeds": consistent_data_seeds,
        "consistent_training_and_trajectory_configuration": consistent_configurations,
        "selected_trajectory_count": selected,
        "points": points,
        "comparisons": comparisons,
    }


def _requested_learning_configuration(args: argparse.Namespace) -> dict[str, object]:
    catalog_hashes = _catalog_input_hashes(args.catalog)
    return {
        "variant": "fm",
        "dataset_version": EXPECTED_DATASET_VERSION,
        "catalog_sha256": catalog_hashes["catalog"],
        "catalog_records_sha256": catalog_hashes["catalog_records"],
        "scenario_mix": "public",
        "extended_fraction": args.extended_fraction,
        "supervision_policy": args.supervision_policy,
        "tie_weight": args.tie_weight,
        "category_only_weight": args.category_only_weight,
        "evidence_saturation": args.evidence_saturation,
        "negative_count": args.negatives,
        "negative_pre_pool_size": 128,
        "negative_mode": args.negative_mode,
        "hard_fraction": args.hard_fraction,
        "near_fraction": args.near_fraction,
        "random_fraction": args.random_fraction,
        "other_encoding": args.other_encoding,
        "dimension": args.dimension,
        "learning_rate": args.learning_rate,
        "latent_l2": args.latent_l2,
        "linear_l2": args.linear_l2,
        "cross_l2": 1e-4,
        "minimum_value_support": 5,
        "minimum_cross_support": 20,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "validation_interval": 1,
        "pair_batch_size": 65536,
    }


def _requested_tuning_fixed_configuration(
    args: argparse.Namespace,
) -> dict[str, object]:
    configuration = _requested_learning_configuration(args)
    for field in ("dimension", "learning_rate", "latent_l2", "linear_l2"):
        configuration.pop(field)
    return configuration


def plateau_prerequisite_errors(args: argparse.Namespace) -> list[str]:
    decision = plateau_decision(
        _learning_rows_from_disk(args.output_root), args.plateau_threshold
    )
    selected = _as_int(decision.get("selected_trajectory_count"))
    errors: list[str] = []
    if not decision.get("all_required_sizes_complete"):
        errors.append(
            "E7 is incomplete or does not use identical configuration and shared "
            "model/trajectory/split seed sets at 25k, 50k, and 100k"
        )
    if selected not in set(LEARNING_CURVE_SIZES):
        errors.append("E7 has no valid plateau selection")
    required_points = [
        point
        for point in decision.get("points", [])
        if isinstance(point, Mapping)
        and _as_int(point.get("trajectory_count")) in set(LEARNING_CURVE_SIZES)
    ]
    expected_model_seeds = sorted(args.learning_curve_seeds)
    if decision.get("all_required_sizes_complete") and any(
        point.get("model_seeds") != expected_model_seeds
        or point.get("trajectory_seeds") != [args.trajectory_seed]
        or point.get("split_seeds") != [args.split_seed]
        for point in required_points
    ):
        errors.append("E7 seed sets do not match the requested runner settings")
    expected_configuration = _requested_learning_configuration(args)
    if selected in set(LEARNING_CURVE_SIZES):
        selected_point = next(
            (
                point
                for point in decision.get("points", [])
                if isinstance(point, Mapping)
                and _as_int(point.get("trajectory_count")) == selected
            ),
            None,
        )
        recorded_configuration = (
            selected_point.get("configuration")
            if isinstance(selected_point, Mapping)
            else None
        )
        if not isinstance(recorded_configuration, Mapping) or any(
            not _same_config_value(expected, recorded_configuration.get(field))
            for field, expected in expected_configuration.items()
        ):
            errors.append(
                "E7 configuration does not match the requested runner settings"
            )
    if (
        args.final_trajectories is not None
        and selected is not None
        and args.final_trajectories != selected
    ):
        errors.append(
            f"--final-trajectories={args.final_trajectories} conflicts with the "
            f"completed E7 selection of {selected}"
        )
    return errors


def selected_final_trajectory_count(args: argparse.Namespace) -> int:
    errors = plateau_prerequisite_errors(args)
    if errors:
        raise ValueError("; ".join(errors))
    decision = plateau_decision(
        _learning_rows_from_disk(args.output_root), args.plateau_threshold
    )
    selected = _as_int(decision.get("selected_trajectory_count"))
    assert selected is not None
    return selected


def _tuning_grid() -> tuple[tuple[int, float, float, float], ...]:
    """Nine bounded one-factor-at-a-time E8 configurations."""

    base_dimension = 16
    base_learning_rate = 0.01
    base_regularization = 1e-5
    configurations = {
        (dimension, base_learning_rate, base_regularization, base_regularization)
        for dimension in TUNING_DIMENSIONS
    }
    configurations.update(
        (
            base_dimension,
            learning_rate,
            base_regularization,
            base_regularization,
        )
        for learning_rate in TUNING_LEARNING_RATES
    )
    configurations.update(
        (
            base_dimension,
            base_learning_rate,
            regularization,
            base_regularization,
        )
        for regularization in TUNING_REGULARIZATIONS
    )
    configurations.update(
        (
            base_dimension,
            base_learning_rate,
            base_regularization,
            regularization,
        )
        for regularization in TUNING_REGULARIZATIONS
    )
    return tuple(sorted(configurations))


def tuning_specs(args: argparse.Namespace) -> list[RunSpec]:
    trajectory_count = selected_final_trajectory_count(args)
    result: list[RunSpec] = []
    for dimension, learning_rate, latent_l2, linear_l2 in _tuning_grid():
        config_slug = _slug(
            f"d{dimension}_lr{learning_rate:g}_zl{latent_l2:g}_wn{linear_l2:g}"
        )
        result.append(
            _training_spec(
                **_common_training_values(
                    args,
                    dimension=dimension,
                    learning_rate=learning_rate,
                    latent_l2=latent_l2,
                    linear_l2=linear_l2,
                ),
                run_id=(
                    f"{trajectory_count // 1000:03d}k_{config_slug}_"
                    f"tseed{args.trajectory_seed}_mseed{args.tuning_seed}"
                ),
                group="E8_tuning",
                experiment="E8/tuning",
                description=(
                    "Bounded full-survivor validation tuning: "
                    f"dimension={dimension}, learning_rate={learning_rate:g}, "
                    f"latent_l2={latent_l2:g}, linear_l2={linear_l2:g}"
                ),
                trajectory_count=trajectory_count,
                scenario_mix="public",
                seed=args.tuning_seed,
                supervision_policy=args.supervision_policy,
                negatives=args.negatives,
            )
        )
    return result


def tuning_decision(
    rows: Sequence[Mapping[str, object]],
    trajectory_count: int | None = None,
    *,
    trajectory_seed: int | None = None,
    split_seed: int | None = None,
    model_seed: int | None = None,
    required_configuration: Mapping[str, object] | None = None,
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    expected_configurations = set(_tuning_grid())
    for row in rows:
        if row.get("group") != "E8_tuning":
            continue
        row_trajectory_count = _as_int(row.get("trajectory_count"))
        if trajectory_count is not None and row_trajectory_count != trajectory_count:
            continue
        row_trajectory_seed = _as_int(row.get("trajectory_seed"))
        row_split_seed = _as_int(row.get("split_seed"))
        row_model_seed = _as_int(row.get("model_seed", row.get("seed")))
        if trajectory_seed is not None and row_trajectory_seed != trajectory_seed:
            continue
        if split_seed is not None and row_split_seed != split_seed:
            continue
        if model_seed is not None and row_model_seed != model_seed:
            continue
        if required_configuration is not None and any(
            not _same_config_value(expected, row.get(field))
            for field, expected in required_configuration.items()
        ):
            continue
        validation_mrr = _as_float(row.get("validation_mrr"))
        dimension = _as_int(row.get("dimension"))
        learning_rate = _as_float(row.get("learning_rate"))
        latent_l2 = _as_float(row.get("latent_l2"))
        linear_l2 = _as_float(row.get("linear_l2"))
        if None in (validation_mrr, dimension, learning_rate, latent_l2, linear_l2):
            continue
        configuration = (
            int(dimension),
            float(learning_rate),
            float(latent_l2),
            float(linear_l2),
        )
        if configuration not in expected_configurations:
            continue
        candidates.append(
            {
                "run_id": row.get("run_id"),
                "validation_mrr": validation_mrr,
                "training_seconds": _as_float(row.get("training_seconds")),
                "trajectory_count": row_trajectory_count,
                "model_seed": row_model_seed,
                "trajectory_seed": row_trajectory_seed,
                "split_seed": row_split_seed,
                "dimension": dimension,
                "learning_rate": learning_rate,
                "latent_l2": latent_l2,
                "linear_l2": linear_l2,
                "artifact": row.get("artifact"),
                "metrics_path": row.get("metrics_path"),
            }
        )
    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(row["validation_mrr"]),
            float(row["training_seconds"])
            if row["training_seconds"] is not None
            else math.inf,
            int(row["dimension"]),
            float(row["learning_rate"]),
            float(row["latent_l2"]),
            float(row["linear_l2"]),
            str(row["run_id"]),
        ),
    )
    configuration_counts: dict[tuple[int, float, float, float], int] = {}
    for row in ranked:
        configuration = (
            int(row["dimension"]),
            float(row["learning_rate"]),
            float(row["latent_l2"]),
            float(row["linear_l2"]),
        )
        configuration_counts[configuration] = configuration_counts.get(configuration, 0) + 1
    duplicates = [
        list(configuration)
        for configuration, count in sorted(configuration_counts.items())
        if count > 1
    ]
    grid_complete = (
        set(configuration_counts) == expected_configurations and not duplicates
    )
    provisional_selected = ranked[0] if ranked else None
    selected = provisional_selected if grid_complete else None
    return {
        "selection_metric": "full_survivor_validation_mrr",
        "selection_rule": "highest MRR; then lower runtime and stable config order",
        "bounded_grid_size": len(_tuning_grid()),
        "trajectory_count": trajectory_count,
        "trajectory_seed": trajectory_seed,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "required_configuration": dict(required_configuration or {}),
        "completed_metric_rows": len(ranked),
        "completed_configurations": len(configuration_counts),
        "duplicate_configurations": duplicates,
        "grid_complete": grid_complete,
        "selected": selected,
        "provisional_selected": provisional_selected,
        "ranked_candidates": ranked,
    }


def selected_tuning_config(args: argparse.Namespace) -> dict[str, int | float]:
    decision = tuning_decision(
        aggregate_metric_rows(args.output_root),
        selected_final_trajectory_count(args),
        trajectory_seed=args.trajectory_seed,
        split_seed=args.split_seed,
        model_seed=args.tuning_seed,
        required_configuration=_requested_tuning_fixed_configuration(args),
    )
    selected = decision.get("selected")
    if isinstance(selected, Mapping):
        values = {
            "dimension": _as_int(selected.get("dimension")),
            "learning_rate": _as_float(selected.get("learning_rate")),
            "latent_l2": _as_float(selected.get("latent_l2")),
            "linear_l2": _as_float(selected.get("linear_l2")),
        }
        if all(value is not None for value in values.values()):
            return {key: value for key, value in values.items() if value is not None}
    raise ValueError(
        "E8 tuning grid is incomplete for the selected E7 trajectory count and "
        "shared trajectory/split/model seed; refusing default hyperparameters"
    )


def final_seed_specs(args: argparse.Namespace) -> list[RunSpec]:
    trajectory_count = selected_final_trajectory_count(args)
    tuning = selected_tuning_config(args)
    tuning_slug = _slug(
        f"d{int(tuning['dimension'])}_lr{float(tuning['learning_rate']):g}_"
        f"zl{float(tuning['latent_l2']):g}_wn{float(tuning['linear_l2']):g}"
    )
    return [
        _training_spec(
            **_common_training_values(
                args,
                dimension=int(tuning["dimension"]),
                learning_rate=float(tuning["learning_rate"]),
                latent_l2=float(tuning["latent_l2"]),
                linear_l2=float(tuning["linear_l2"]),
            ),
            run_id=(
                f"{trajectory_count // 1000:03d}k_tuned_"
                f"{tuning_slug}_tseed{args.trajectory_seed}_mseed{seed}"
            ),
            group="E8_final_seeds",
            experiment="E8",
            description=(
                f"Final independent FM candidate seed {seed} at "
                f"{trajectory_count:,} trajectories; inherits selected E8 "
                f"configuration {tuning}"
            ),
            trajectory_count=trajectory_count,
            scenario_mix="public",
            seed=seed,
            supervision_policy=args.supervision_policy,
            negatives=args.negatives,
        )
        for seed in args.final_seeds
    ]


def baseline_specs(args: argparse.Namespace) -> list[RunSpec]:
    return [
        _evaluation_spec(
            python=args.python,
            catalog=args.catalog,
            output_root=args.output_root,
            run_id=(
                f"frozen_models_{args.baseline_trajectory_count}_"
                f"tseed{args.trajectory_seed}_sseed{args.split_seed}"
            ),
            group="E0_full_survivor_baseline",
            experiment="E0",
            description="Frozen Linear, FM, and Hybrid under exact full-survivor evaluation",
            linear_model=APPROACH_ROOT / "linear_model.sqlite3",
            fm_model=APPROACH_ROOT / "fm_only_model.sqlite3",
            hybrid_model=APPROACH_ROOT / "fm_model.sqlite3",
            trajectory_count=args.baseline_trajectory_count,
            trajectory_seed=args.trajectory_seed,
            split_seed=args.split_seed,
            scenario_mix="public",
            prerequisites=(
                APPROACH_ROOT / "linear_model.sqlite3",
                APPROACH_ROOT / "fm_only_model.sqlite3",
                APPROACH_ROOT / "fm_model.sqlite3",
            ),
        )
    ]


def _slug(value: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in value)
    return "_".join(part for part in cleaned.split("_") if part) or "candidate"


def _evidence_recorded_path(value: object, evidence_path: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = evidence_path.parent / path
    return path.resolve()


def _gate_evidence_paths(
    gate: Mapping[str, object], evidence_path: Path, gate_name: str
) -> tuple[list[Path], list[str]]:
    values = gate.get("evidence_paths")
    if not isinstance(values, list) or not values:
        return [], [f"promotion gate lacks evidence paths: {gate_name}"]
    paths = [
        path
        for value in values
        if (path := _evidence_recorded_path(value, evidence_path)) is not None
    ]
    errors: list[str] = []
    if len(paths) != len(values) or len(set(paths)) != len(paths):
        errors.append(f"promotion gate has invalid or duplicate evidence paths: {gate_name}")
    if any(not path.is_file() for path in paths):
        errors.append(f"promotion gate has missing evidence: {gate_name}")
    return paths, errors


def _ranker_breakdown_hit_at_10(
    model: Mapping[str, object], *, evidence_path: Path, model_name: str
) -> tuple[dict[str, tuple[float | None, int | None, int | None]], list[str]]:
    rows = model.get("breakdowns")
    if not isinstance(rows, list):
        return {}, [f"ranker evidence lacks breakdowns for {model_name}: {evidence_path}"]
    result: dict[str, tuple[float | None, int | None, int | None]] = {}
    errors: list[str] = []
    seen_dimensions: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        dimension = str(row.get("dimension", ""))
        if dimension not in {"scenario", "turn_bucket"}:
            continue
        value = str(row.get("value", ""))
        metrics = row.get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        primary = metrics.get("primary")
        primary = primary if isinstance(primary, Mapping) else {}
        informative = primary.get("hit_rate_at_10_informative")
        hit_at_10 = _as_float(primary.get("hit_rate_at_10"))
        if informative is True and hit_at_10 is None:
            errors.append(
                f"informative HR@10 is missing for {model_name} {dimension}/{value}"
            )
        if informative is not True and informative is not False:
            errors.append(
                f"HR@10 informativeness flag is missing for {model_name} "
                f"{dimension}/{value}"
            )
        key = f"{dimension}/{value}"
        if key in result:
            errors.append(f"duplicate ranker breakdown {key}: {evidence_path}")
            continue
        result[key] = (
            hit_at_10 if informative is True else None,
            _as_int(metrics.get("state_count")),
            _as_int(metrics.get("trajectory_count")),
        )
        seen_dimensions.add(dimension)
    for dimension in ("scenario", "turn_bucket"):
        if dimension not in seen_dimensions:
            errors.append(
                f"ranker evidence lacks {dimension} HR@10 breakdowns: {evidence_path}"
            )
    return result, errors


def _machine_ranker_gate_results(
    evidence_paths: Sequence[Path],
    expected_final_paths: set[Path],
    *,
    expected_catalog_hashes: Mapping[str, str] | None = None,
    expected_trajectory_count: int | None = None,
    expected_trajectory_seed: int | None = None,
    expected_split_seed: int | None = None,
    expected_scenario_mix: str | None = None,
) -> tuple[dict[str, bool], list[str]]:
    """Derive three pre-official ranker gates from evaluator summaries."""

    results = {gate_name: False for gate_name in RANKER_PROMOTION_GATES}
    errors: list[str] = []
    if len(evidence_paths) != 3 or len(set(evidence_paths)) != 3:
        return results, ["ranker promotion gates require three distinct summary JSON files"]

    covered_candidates: set[Path] = set()
    mrr_deltas: list[float] = []
    incumbent_signatures: list[tuple[float, object]] = []
    cohort_signatures: list[str] = []
    hit_regressions: list[str] = []
    incumbent_artifact = (APPROACH_ROOT / "fm_only_model.sqlite3").resolve()
    for evidence_path in evidence_paths:
        payload = _read_json(evidence_path)
        if payload is None:
            errors.append(f"ranker evidence is not valid JSON: {evidence_path}")
            continue
        if payload.get("schema_version") != FULL_SURVIVOR_SCHEMA_VERSION:
            errors.append(f"ranker evidence schema is invalid: {evidence_path}")
        if payload.get("evaluation_protocol") != FULL_SURVIVOR_PROTOCOL:
            errors.append(f"ranker evidence protocol is invalid: {evidence_path}")
        if payload.get("split") != "validation":
            errors.append(f"ranker evidence must use the validation split: {evidence_path}")
        cohort = payload.get("cohort")
        cohort = cohort if isinstance(cohort, Mapping) else {}
        cohort_sha256 = cohort.get("cohort_sha256")
        cohort_without_digest = {
            key: value for key, value in cohort.items() if key != "cohort_sha256"
        }
        if (
            not isinstance(cohort_sha256, str)
            or cohort_sha256 != _json_sha256(cohort_without_digest)
        ):
            errors.append(f"ranker evidence cohort digest is invalid: {evidence_path}")
        else:
            cohort_signatures.append(cohort_sha256)
        state_cohort_sha256 = cohort.get("state_cohort_sha256")
        if not isinstance(state_cohort_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", state_cohort_sha256
        ):
            errors.append(f"ranker evidence state cohort digest is invalid: {evidence_path}")
        dataset_manifest = cohort.get("dataset_manifest")
        dataset_manifest = (
            dataset_manifest if isinstance(dataset_manifest, Mapping) else {}
        )
        if cohort.get("dataset_manifest_sha256") != _json_sha256(dataset_manifest):
            errors.append(f"ranker evidence manifest digest is invalid: {evidence_path}")
        dataset_config = dataset_manifest.get("config")
        dataset_config = dataset_config if isinstance(dataset_config, Mapping) else {}
        expected_config = {
            "trajectory_count": expected_trajectory_count,
            "seed": expected_trajectory_seed,
            "split_seed": expected_split_seed,
            "scenario_mix": expected_scenario_mix,
        }
        for key, expected in expected_config.items():
            if expected is not None and not _same_config_value(
                expected, dataset_config.get(key)
            ):
                errors.append(
                    f"ranker evidence cohort {key} differs from the registered cohort: "
                    f"{evidence_path}"
                )
        if dataset_manifest.get("dataset_version") != EXPECTED_DATASET_VERSION:
            errors.append(f"ranker evidence dataset version is invalid: {evidence_path}")
        manifest_input_hashes = dataset_manifest.get("input_sha256")
        manifest_input_hashes = (
            manifest_input_hashes
            if isinstance(manifest_input_hashes, Mapping)
            else {}
        )
        for key, expected in (expected_catalog_hashes or {}).items():
            if manifest_input_hashes.get(key) != expected:
                errors.append(
                    f"ranker evidence catalog hash differs for {key}: {evidence_path}"
                )
        models = payload.get("models")
        models = models if isinstance(models, Mapping) else {}
        candidate = models.get("candidate")
        incumbent = models.get("fm")
        if not isinstance(candidate, Mapping) or not isinstance(incumbent, Mapping):
            errors.append(
                f"ranker evidence must contain candidate and incumbent fm: {evidence_path}"
            )
            continue
        candidate_path = _evidence_recorded_path(candidate.get("artifact"), evidence_path)
        incumbent_path = _evidence_recorded_path(incumbent.get("artifact"), evidence_path)
        if candidate_path is None or candidate_path not in expected_final_paths:
            errors.append(
                f"ranker evidence candidate is not an expected final seed: {evidence_path}"
            )
        else:
            covered_candidates.add(candidate_path)
        if incumbent_path != incumbent_artifact:
            errors.append(
                f"ranker evidence does not use the frozen incumbent FM: {evidence_path}"
            )
        for model_name, model, artifact_path in (
            ("candidate", candidate, candidate_path),
            ("fm", incumbent, incumbent_path),
        ):
            recorded_digest = model.get("artifact_sha256")
            if artifact_path is None or not artifact_path.is_file():
                errors.append(
                    f"ranker evidence {model_name} artifact is missing: {evidence_path}"
                )
            elif recorded_digest != _sha256_path(artifact_path):
                errors.append(
                    f"ranker evidence {model_name} artifact digest differs: {evidence_path}"
                )

        candidate_overall = candidate.get("overall")
        incumbent_overall = incumbent.get("overall")
        candidate_overall = (
            candidate_overall if isinstance(candidate_overall, Mapping) else {}
        )
        incumbent_overall = (
            incumbent_overall if isinstance(incumbent_overall, Mapping) else {}
        )
        candidate_primary = candidate_overall.get("primary")
        incumbent_primary = incumbent_overall.get("primary")
        candidate_primary = (
            candidate_primary if isinstance(candidate_primary, Mapping) else {}
        )
        incumbent_primary = (
            incumbent_primary if isinstance(incumbent_primary, Mapping) else {}
        )
        candidate_mrr = _as_float(candidate_primary.get("mrr"))
        incumbent_mrr = _as_float(incumbent_primary.get("mrr"))
        if candidate_mrr is None or incumbent_mrr is None:
            errors.append(f"ranker evidence lacks finite primary MRR: {evidence_path}")
        else:
            mrr_deltas.append(candidate_mrr - incumbent_mrr)
        if (
            _as_int(candidate_overall.get("state_count"))
            != _as_int(incumbent_overall.get("state_count"))
            or _as_int(candidate_overall.get("trajectory_count"))
            != _as_int(incumbent_overall.get("trajectory_count"))
        ):
            errors.append(
                f"candidate and incumbent ranker cohorts differ: {evidence_path}"
            )
        if (
            _as_int(cohort.get("state_count"))
            != _as_int(candidate_overall.get("state_count"))
            or _as_int(cohort.get("trajectory_count"))
            != _as_int(candidate_overall.get("trajectory_count"))
        ):
            errors.append(f"ranker cohort counts do not match reports: {evidence_path}")

        candidate_hits, candidate_errors = _ranker_breakdown_hit_at_10(
            candidate, evidence_path=evidence_path, model_name="candidate"
        )
        incumbent_hits, incumbent_errors = _ranker_breakdown_hit_at_10(
            incumbent, evidence_path=evidence_path, model_name="fm"
        )
        errors.extend(candidate_errors)
        errors.extend(incumbent_errors)
        if set(candidate_hits) != set(incumbent_hits):
            errors.append(
                f"candidate and incumbent ranker breakdown keys differ: {evidence_path}"
            )
        for key in sorted(set(candidate_hits) & set(incumbent_hits)):
            candidate_hit, candidate_states, candidate_trajectories = candidate_hits[key]
            incumbent_hit, incumbent_states, incumbent_trajectories = incumbent_hits[key]
            if (candidate_states, candidate_trajectories) != (
                incumbent_states,
                incumbent_trajectories,
            ):
                errors.append(f"ranker breakdown cohort differs for {key}: {evidence_path}")
            if (candidate_hit is None) != (incumbent_hit is None):
                errors.append(
                    f"ranker HR@10 informativeness differs for {key}: {evidence_path}"
                )
            elif (
                candidate_hit is not None
                and incumbent_hit is not None
                and candidate_hit < incumbent_hit - 1e-12
            ):
                hit_regressions.append(
                    f"{candidate_path}:{key}={candidate_hit - incumbent_hit:+.12g}"
                )
        if incumbent_mrr is not None:
            incumbent_signatures.append(
                (
                    incumbent_mrr,
                    json.dumps(incumbent_hits, sort_keys=True, separators=(",", ":")),
                )
            )

    if covered_candidates != expected_final_paths:
        errors.append("ranker evidence does not cover the exact three final seed artifacts")
    if len(set(incumbent_signatures)) != 1:
        errors.append("ranker evidence does not use one identical frozen validation cohort")
    if len(cohort_signatures) != 3 or len(set(cohort_signatures)) != 1:
        errors.append("ranker evidence does not contain one identical bound cohort")
    if hit_regressions:
        errors.append("material HR@10 regression detected: " + "; ".join(hit_regressions))

    structurally_valid = not errors and len(mrr_deltas) == 3
    results["full_survivor_validation_mrr_improved"] = (
        structurally_valid and fmean(mrr_deltas) > 0.0
    )
    results["consistent_direction_across_three_seeds"] = (
        structurally_valid and all(delta > 0.0 for delta in mrr_deltas)
    )
    results["no_material_hit_at_10_regression"] = (
        structurally_valid and not hit_regressions
    )
    return results, errors


def _correctness_command(python: Path) -> tuple[str, ...]:
    test_ids = tuple(dict.fromkeys(CORRECTNESS_TEST_IDS.values()))
    return (str(python.resolve()), "-m", "unittest", "-q", *test_ids)


def _correctness_source_sha256() -> str:
    paths = (
        PROJECT_ROOT / "starter" / "agent.py",
        PROJECT_ROOT / "starter" / "conversation_features.py",
        PROJECT_ROOT / "starter" / "hybrid_model.py",
        PROJECT_ROOT / "tests" / "test_agent.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPOSITORY_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _run_correctness_suite(python: Path) -> dict[str, object]:
    command = _correctness_command(python)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    match = re.search(r"Ran\s+(\d+)\s+tests?", output)
    tests_run = int(match.group(1)) if match else 0
    passed = completed.returncode == 0 and tests_run == len(set(CORRECTNESS_TEST_IDS.values()))
    return {
        "schema_version": CORRECTNESS_EVIDENCE_SCHEMA_VERSION,
        "generated_utc": utc_now(),
        "status": "passed" if passed else "failed",
        "return_code": completed.returncode,
        "tests_run": tests_run,
        "failures": 0 if passed else None,
        "errors": 0 if passed else None,
        "skipped": 0,
        "command": list(command),
        "working_directory": str(PROJECT_ROOT),
        "source_sha256": _correctness_source_sha256(),
        "test_ids": list(dict.fromkeys(CORRECTNESS_TEST_IDS.values())),
        "required_checks": {
            name: {"test_id": test_id, "passed": passed}
            for name, test_id in CORRECTNESS_TEST_IDS.items()
        },
        "output": output,
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }


def _machine_correctness_evidence_errors(
    evidence_paths: Sequence[Path],
    *,
    python: Path | None = None,
    verify_execution: bool = False,
) -> list[str]:
    if len(evidence_paths) != 1:
        return ["correctness gate requires exactly one structured JSON report"]
    evidence_path = evidence_paths[0]
    payload = _read_json(evidence_path)
    if payload is None:
        return [f"correctness evidence is not valid JSON: {evidence_path}"]
    errors: list[str] = []
    if payload.get("schema_version") != CORRECTNESS_EVIDENCE_SCHEMA_VERSION:
        errors.append("correctness evidence schema is invalid")
    if payload.get("status") != "passed" or _as_int(payload.get("return_code")) != 0:
        errors.append("correctness evidence does not record a passing command")
    if (_as_int(payload.get("tests_run")) or 0) <= 0:
        errors.append("correctness evidence must record at least one executed test")
    for field in ("failures", "errors", "skipped"):
        if _as_int(payload.get(field)) != 0:
            errors.append(f"correctness evidence {field} must be zero")
    expected_python = (python or Path(sys.executable)).resolve()
    command = payload.get("command")
    if command != list(_correctness_command(expected_python)):
        errors.append("correctness evidence command is not the fixed required suite")
    if payload.get("working_directory") != str(PROJECT_ROOT):
        errors.append("correctness evidence working directory is invalid")
    if payload.get("source_sha256") != _correctness_source_sha256():
        errors.append("correctness evidence is stale for the current source tree")
    expected_test_ids = list(dict.fromkeys(CORRECTNESS_TEST_IDS.values()))
    if payload.get("test_ids") != expected_test_ids:
        errors.append("correctness evidence test IDs differ from the required suite")
    if _as_int(payload.get("tests_run")) != len(expected_test_ids):
        errors.append("correctness evidence test count differs from the required suite")
    output = payload.get("output")
    if not isinstance(output, str) or payload.get("output_sha256") != hashlib.sha256(
        (output if isinstance(output, str) else "").encode("utf-8")
    ).hexdigest():
        errors.append("correctness evidence output digest is invalid")
    checks = payload.get("required_checks")
    checks = checks if isinstance(checks, Mapping) else {}
    for check in REQUIRED_CORRECTNESS_CHECKS:
        result = checks.get(check)
        result = result if isinstance(result, Mapping) else {}
        if (
            result.get("passed") is not True
            or result.get("test_id") != CORRECTNESS_TEST_IDS[check]
        ):
            errors.append(f"correctness evidence did not pass required check: {check}")
    if verify_execution:
        rerun = _run_correctness_suite(expected_python)
        if rerun.get("status") != "passed":
            errors.append("current correctness suite failed independent re-execution")
    return errors


def promotion_evidence_errors(
    args: argparse.Namespace,
    expected_final_specs: Sequence[RunSpec] = (),
    *,
    verify_correctness: bool = False,
) -> list[str]:
    """Validate the explicit pre-official promotion-gate evidence document."""

    errors: list[str] = []
    candidates = list(args.candidate_artifact or ())
    if len(candidates) != 1:
        errors.append("official evaluation requires exactly one --candidate-artifact")
    evidence_path = args.promotion_evidence
    if evidence_path is None:
        errors.append("official evaluation requires --promotion-evidence")
        return errors
    evidence_path = evidence_path.resolve()
    payload = _read_json(evidence_path)
    if payload is None:
        errors.append(f"promotion evidence is missing or invalid JSON: {evidence_path}")
        return errors
    if payload.get("schema_version") != PROMOTION_EVIDENCE_SCHEMA_VERSION:
        errors.append(
            f"promotion evidence schema must be {PROMOTION_EVIDENCE_SCHEMA_VERSION}"
        )
    if payload.get("decision") != "approved_for_official_evaluation":
        errors.append("promotion evidence decision is not approved_for_official_evaluation")

    selected_candidate = _evidence_recorded_path(
        payload.get("selected_candidate_artifact"), evidence_path
    )
    candidate = candidates[0].resolve() if len(candidates) == 1 else None
    if candidate is not None and selected_candidate != candidate:
        errors.append("promotion evidence selects a different candidate artifact")

    recorded_final = payload.get("final_seed_artifacts")
    if not isinstance(recorded_final, list):
        errors.append("promotion evidence must list three final_seed_artifacts")
        recorded_final_paths: list[Path] = []
    else:
        recorded_final_paths = [
            path
            for value in recorded_final
            if (path := _evidence_recorded_path(value, evidence_path)) is not None
        ]
        if len(recorded_final_paths) != 3 or len(set(recorded_final_paths)) != 3:
            errors.append("promotion evidence must list three distinct final seed artifacts")
        elif any(not path.is_file() for path in recorded_final_paths):
            errors.append("promotion evidence references missing final seed artifacts")
    expected_final_paths = {
        spec.artifact_path.resolve()
        for spec in expected_final_specs
        if spec.artifact_path is not None
    }
    if expected_final_paths and set(recorded_final_paths) != expected_final_paths:
        errors.append("promotion evidence final seed artifacts do not match E8 outputs")
    if candidate is not None and candidate not in recorded_final_paths:
        errors.append("selected official candidate is not one of the three final seeds")

    gates = payload.get("gates")
    gates = gates if isinstance(gates, Mapping) else {}
    gate_paths: dict[str, list[Path]] = {}
    for gate_name in PRE_OFFICIAL_PROMOTION_GATES:
        gate = gates.get(gate_name)
        if not isinstance(gate, Mapping):
            errors.append(f"promotion gate did not pass: {gate_name}")
            continue
        if gate.get("passed") is not True:
            errors.append(f"promotion gate did not pass: {gate_name}")
        paths, path_errors = _gate_evidence_paths(gate, evidence_path, gate_name)
        errors.extend(path_errors)
        gate_paths[gate_name] = paths

    ranker_path_sets = {
        tuple(gate_paths.get(gate_name, ())) for gate_name in RANKER_PROMOTION_GATES
    }
    if len(ranker_path_sets) != 1:
        errors.append("all three ranker promotion gates must reference the same summaries")
    elif all(gate_name in gate_paths for gate_name in RANKER_PROMOTION_GATES):
        ranker_paths = list(next(iter(ranker_path_sets)))
        catalog_hashes: Mapping[str, str] | None = None
        catalog_path = getattr(args, "catalog", None)
        if isinstance(catalog_path, Path) and catalog_path.is_file():
            try:
                catalog_hashes = _catalog_input_hashes(catalog_path)
            except (OSError, ValueError) as error:
                errors.append(f"promotion catalog cannot be hashed: {error}")
        machine_results, machine_errors = _machine_ranker_gate_results(
            ranker_paths,
            expected_final_paths or set(recorded_final_paths),
            expected_catalog_hashes=catalog_hashes,
            expected_trajectory_count=getattr(args, "baseline_trajectory_count", None),
            expected_trajectory_seed=getattr(args, "trajectory_seed", None),
            expected_split_seed=getattr(args, "split_seed", None),
            expected_scenario_mix=(
                "public" if hasattr(args, "baseline_trajectory_count") else None
            ),
        )
        errors.extend(machine_errors)
        for gate_name in RANKER_PROMOTION_GATES:
            if not machine_results.get(gate_name, False):
                errors.append(f"machine-validated promotion gate failed: {gate_name}")

    correctness_paths = gate_paths.get("correctness_tests_passed", [])
    errors.extend(
        _machine_correctness_evidence_errors(
            correctness_paths,
            python=getattr(args, "python", None),
            verify_execution=verify_correctness,
        )
    )
    return errors


def official_specs(args: argparse.Namespace) -> list[RunSpec]:
    candidates = list(args.candidate_artifact or ())
    result: list[RunSpec] = []
    for ordinal, candidate in enumerate(candidates):
        if candidate is None:
            continue
        label = _slug(f"{candidate.parent.name}_{candidate.stem}_{ordinal}")
        result.append(
            _evaluation_spec(
                python=args.python,
                catalog=args.catalog,
                output_root=args.output_root,
                run_id=(
                    f"{label}_{args.official_trajectory_count}_"
                    f"tseed{args.trajectory_seed}_sseed{args.split_seed}_official"
                ),
                group="E8_official_handoff",
                experiment="E8/promotion",
                description=(
                    "Explicitly gate-approved official evaluation: frozen linear, "
                    "frozen incumbent FM, "
                    f"and candidate {candidate} in the evaluator's third slot"
                ),
                linear_model=APPROACH_ROOT / "linear_model.sqlite3",
                fm_model=APPROACH_ROOT / "fm_only_model.sqlite3",
                hybrid_model=candidate,
                trajectory_count=args.official_trajectory_count,
                trajectory_seed=args.trajectory_seed,
                split_seed=args.split_seed,
                scenario_mix="public",
                third_model_name="candidate",
                third_model_mode="fm",
                prerequisites=(
                    APPROACH_ROOT / "linear_model.sqlite3",
                    APPROACH_ROOT / "fm_only_model.sqlite3",
                    candidate,
                ),
            )
        )
    return result


def specs_for_mode(args: argparse.Namespace) -> list[RunSpec]:
    builders = {
        "baseline": baseline_specs,
        "smoke": smoke_specs,
        "cumulative": cumulative_specs,
        "supervision": supervision_specs,
        "negatives": negative_specs,
        "negative-mixture": negative_mixture_specs,
        "sensitivity": sensitivity_specs,
        "other-encoding": other_encoding_specs,
        "learning-curve": learning_curve_specs,
        "tuning": tuning_specs,
        "final-seeds": final_seed_specs,
        "official": official_specs,
    }
    if args.mode in {"aggregate", "correctness"}:
        return []
    if args.mode != "all":
        return builders[args.mode](args)
    result: list[RunSpec] = []
    for name in (
        "baseline",
        "smoke",
        "supervision",
        "negatives",
        "negative-mixture",
        "sensitivity",
        "other-encoding",
        "learning-curve",
    ):
        result.extend(builders[name](args))
    return result


def tuning_prerequisite_errors(args: argparse.Namespace) -> list[str]:
    errors = plateau_prerequisite_errors(args)
    if errors:
        return errors
    trajectory_count = selected_final_trajectory_count(args)
    decision = tuning_decision(
        aggregate_metric_rows(args.output_root),
        trajectory_count,
        trajectory_seed=args.trajectory_seed,
        split_seed=args.split_seed,
        model_seed=args.tuning_seed,
        required_configuration=_requested_tuning_fixed_configuration(args),
    )
    if not decision.get("grid_complete"):
        errors.append(
            "E8 tuning grid is incomplete for the selected E7 size and exact "
            "trajectory/split/model seeds"
        )
    return errors


def stage_prerequisite_errors(args: argparse.Namespace) -> list[str]:
    if args.mode == "all":
        return [
            "--mode all execution remains disabled: the exploratory E1-E6 "
            "cumulative ladder must be reviewed before selected settings are "
            "propagated into E7/E8; run named stages explicitly"
        ]
    if args.mode == "tuning":
        return plateau_prerequisite_errors(args)
    if args.mode == "final-seeds":
        return tuning_prerequisite_errors(args)
    if args.mode != "official":
        return []

    errors = tuning_prerequisite_errors(args)
    if errors:
        return errors
    final_specs = final_seed_specs(args)
    incomplete = [spec.run_id for spec in final_specs if not _is_complete(spec)]
    if incomplete:
        errors.append(
            "all three exact E8 final-seed outputs must be complete before official "
            "evaluation: " + ", ".join(incomplete)
        )
    errors.extend(
        promotion_evidence_errors(
            args, final_specs, verify_correctness=bool(getattr(args, "execute", False))
        )
    )
    return errors


def _command_configuration(command: Sequence[str]) -> dict[str, object]:
    """Recover explicit long-option values for the pre-launch manifest."""

    result: dict[str, object] = {}
    position = 2
    while position < len(command):
        token = command[position]
        if not token.startswith("--"):
            position += 1
            continue
        key = token[2:].replace("-", "_")
        if position + 1 < len(command) and not command[position + 1].startswith("--"):
            result[key] = command[position + 1]
            position += 2
        else:
            result[key] = True
            position += 1
    return result


_TRAINING_DESTINATION_KEYS = frozenset(
    {"output", "metrics", "manifest", "negative_audit", "cross_audit"}
)
_LEARNING_EVIDENCE_FILE_KEYS = frozenset(
    {
        "metrics",
        "runner_manifest",
        "artifact",
        "catalog",
        "dataset_manifest",
        "negative_audit",
        "cross_audit",
        "trainer_source",
        "fm_training_source",
        "trajectory_source",
        "conversation_features_source",
        "attribute_index_source",
        "category_index_source",
        "hybrid_model_source",
    }
)


def _learning_curve_evidence_path(spec: RunSpec) -> Path:
    return spec.output_dir / LEARNING_CURVE_EVIDENCE_FILENAME


def _path_is_below(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _training_output_path_errors(spec: RunSpec) -> list[str]:
    """Bind every destination option to the paths represented by ``spec``."""

    errors: list[str] = []
    command = _command_configuration(spec.command)
    expected = {
        "output": spec.artifact_path.resolve() if spec.artifact_path else None,
        "metrics": spec.completion_path.resolve(),
    }
    for key, expected_path in expected.items():
        if _resolve_recorded_path(command.get(key)) != expected_path:
            errors.append(f"training command {key} path differs from the spec")
    for key in ("manifest", "negative_audit", "cross_audit"):
        path = _resolve_recorded_path(command.get(key))
        if path is None or path.parent != spec.output_dir.resolve():
            errors.append(f"training command {key} path differs from the run directory")
    return errors


def _training_direct_output_paths(spec: RunSpec) -> tuple[Path, ...]:
    """Return every path whose presence proves a direct target run started."""

    command = _command_configuration(spec.command)
    paths = [spec.manifest_path.resolve()]
    for key in sorted(_TRAINING_DESTINATION_KEYS):
        path = _resolve_recorded_path(command.get(key))
        if path is not None:
            paths.append(path)
    return tuple(dict.fromkeys(paths))


def _training_generation_errors(source: RunSpec, target: RunSpec) -> list[str]:
    """Compare training semantics while intentionally ignoring output locations."""

    errors: list[str] = []
    if source.kind != "training" or target.kind != "training":
        return ["evidence reuse requires two training specifications"]
    for position, label in ((0, "python interpreter"), (1, "trainer")):
        try:
            source_path = Path(source.command[position]).expanduser().resolve()
            target_path = Path(target.command[position]).expanduser().resolve()
        except IndexError:
            errors.append(f"training command lacks {label}")
            continue
        if source_path != target_path:
            errors.append(f"{label} differs")

    source_config = _command_configuration(source.command)
    target_config = _command_configuration(target.command)
    source_generation = {
        key: value
        for key, value in source_config.items()
        if key not in _TRAINING_DESTINATION_KEYS
    }
    target_generation = {
        key: value
        for key, value in target_config.items()
        if key not in _TRAINING_DESTINATION_KEYS
    }
    source_generation["negative_mode"] = _canonical_negative_mode(
        source_generation.get("negative_mode")
    )
    target_generation["negative_mode"] = _canonical_negative_mode(
        target_generation.get("negative_mode")
    )
    if set(source_generation) != set(target_generation):
        errors.append("generating command option sets differ")
        return errors
    for key in sorted(source_generation):
        source_value = source_generation[key]
        target_value = target_generation[key]
        if key == "catalog":
            source_value = _resolve_recorded_path(source_value)
            target_value = _resolve_recorded_path(target_value)
        if not _same_generation_option(source_value, target_value):
            errors.append(f"generating command option differs: {key}")
    return errors


def _learning_curve_target_record(
    spec: RunSpec, output_root: Path
) -> dict[str, object]:
    return {
        "run_id": spec.run_id,
        "group": spec.group,
        "experiment": spec.experiment,
        "kind": spec.kind,
        "description": spec.description,
        "output_root": str(output_root.resolve()),
        "output_dir": str(spec.output_dir.resolve()),
        "command": list(spec.command),
        "completion_path": str(spec.completion_path.resolve()),
        "artifact_path": (
            str(spec.artifact_path.resolve()) if spec.artifact_path else None
        ),
    }


def _target_spec_from_learning_evidence(
    reference_path: Path, target_record: Mapping[str, object]
) -> tuple[RunSpec | None, Path | None, list[str]]:
    errors: list[str] = []
    command = target_record.get("command")
    if not isinstance(command, list) or not all(
        isinstance(value, str) for value in command
    ):
        return None, None, ["evidence target command is invalid"]
    output_root = _resolve_recorded_path(target_record.get("output_root"))
    output_dir = _resolve_recorded_path(target_record.get("output_dir"))
    completion_path = _resolve_recorded_path(target_record.get("completion_path"))
    artifact_path = _resolve_recorded_path(target_record.get("artifact_path"))
    if output_root is None or output_dir is None or completion_path is None:
        return None, output_root, ["evidence target paths are invalid"]
    spec = RunSpec(
        run_id=str(target_record.get("run_id", "")),
        group=str(target_record.get("group", "")),
        experiment=str(target_record.get("experiment", "")),
        description=str(target_record.get("description", "")),
        output_dir=output_dir,
        command=tuple(command),
        completion_path=completion_path,
        artifact_path=artifact_path,
        kind=str(target_record.get("kind", "")),
    )
    if spec.group != "E7_learning_curve" or spec.experiment != "E7":
        errors.append("evidence target is not an E7 learning-curve run")
    if spec.kind != "training" or spec.artifact_path is None:
        errors.append("evidence target is not a training run")
    if spec.output_dir != output_root / spec.group / spec.run_id:
        errors.append("evidence target directory is not canonical")
    if reference_path.resolve() != _learning_curve_evidence_path(spec).resolve():
        errors.append("evidence reference path differs from the target")
    if not _path_is_below(spec.output_dir, output_root):
        errors.append("evidence target lies outside its output root")
    if spec.completion_path != spec.output_dir / "metrics.json":
        errors.append("evidence target metrics path is not canonical")
    if spec.artifact_path != spec.output_dir / "model.sqlite3":
        errors.append("evidence target artifact path is not canonical")
    errors.extend(_training_output_path_errors(spec))
    return spec, output_root, errors


def _validated_learning_evidence_source(
    metrics_path: Path, output_root: Path
) -> tuple[RunSpec, dict[str, object], dict[str, object]]:
    metrics_path = metrics_path.resolve()
    errors: list[str] = []
    if not _path_is_below(metrics_path, output_root):
        errors.append("source metrics lie outside the experiment output root")
    payload = _read_json(metrics_path)
    if payload is None:
        raise ValueError("source metrics are missing or invalid JSON")
    source = _validated_training_metric_spec(metrics_path, payload)
    if source is None:
        raise ValueError("source training payload, manifest, or artifact is invalid")
    manifest = _read_json(source.manifest_path)
    assert manifest is not None
    if manifest.get("metrics_complete") is not True:
        errors.append("source manifest does not attest metrics_complete=true")
    if manifest.get("validation_errors") != []:
        errors.append("source manifest retains validation errors")
    errors.extend(_training_output_path_errors(source))
    errors.extend(_completion_errors(source, require_manifest=True))
    for path in (source.completion_path, source.manifest_path, source.artifact_path):
        if path is not None and not _path_is_below(path, output_root):
            errors.append("source run artifacts lie outside the experiment output root")
            break
    if errors:
        raise ValueError("; ".join(dict.fromkeys(errors)))
    return source, payload, manifest


def _learning_evidence_source_paths(source: RunSpec) -> dict[str, Path]:
    command = _command_configuration(source.command)
    trainer = (
        Path(source.command[1]).expanduser().resolve()
        if len(source.command) > 1
        else None
    )
    catalog = _resolve_recorded_path(command.get("catalog"))
    dataset_manifest = _resolve_recorded_path(command.get("manifest"))
    negative_audit = _resolve_recorded_path(command.get("negative_audit"))
    cross_audit = _resolve_recorded_path(command.get("cross_audit"))
    if (
        source.artifact_path is None
        or trainer is None
        or catalog is None
        or dataset_manifest is None
        or negative_audit is None
        or cross_audit is None
    ):
        raise ValueError("source command does not identify every required evidence file")
    return {
        "metrics": source.completion_path.resolve(),
        "runner_manifest": source.manifest_path.resolve(),
        "artifact": source.artifact_path.resolve(),
        "catalog": catalog,
        "dataset_manifest": dataset_manifest,
        "negative_audit": negative_audit,
        "cross_audit": cross_audit,
        "trainer_source": trainer,
        "fm_training_source": (APPROACH_ROOT / "fm_training.py").resolve(),
        "trajectory_source": (APPROACH_ROOT / "trajectory_data.py").resolve(),
        "conversation_features_source": (
            PROJECT_ROOT / "starter" / "conversation_features.py"
        ).resolve(),
        "attribute_index_source": (
            PROJECT_ROOT / "starter" / "attribute_index.py"
        ).resolve(),
        "category_index_source": (
            PROJECT_ROOT / "starter" / "category_index.py"
        ).resolve(),
        "hybrid_model_source": (
            PROJECT_ROOT / "starter" / "hybrid_model.py"
        ).resolve(),
    }


def _learning_curve_evidence_payload(
    target: RunSpec, source_metrics: Path, output_root: Path
) -> dict[str, object]:
    source, _, _ = _validated_learning_evidence_source(source_metrics, output_root)
    generation_errors = _training_generation_errors(source, target)
    if generation_errors:
        raise ValueError("source is not exact-compatible: " + "; ".join(generation_errors))
    source_paths = _learning_evidence_source_paths(source)
    files = {
        name: {"path": str(path), "sha256": _sha256_path(path)}
        for name, path in source_paths.items()
    }
    return {
        "schema_version": LEARNING_CURVE_EVIDENCE_SCHEMA_VERSION,
        "created_utc": utc_now(),
        "reuse_policy": "exact_generating_command_except_destinations",
        "target": _learning_curve_target_record(target, output_root),
        "source": {
            "run_id": source.run_id,
            "group": source.group,
            "experiment": source.experiment,
            "kind": source.kind,
            "command": list(source.command),
            "files": files,
        },
    }


def _learning_curve_evidence_validation(
    reference_path: Path, expected_target: RunSpec | None = None
) -> tuple[
    list[str],
    RunSpec | None,
    RunSpec | None,
    Path | None,
    dict[str, object] | None,
]:
    reference_path = reference_path.resolve()
    reference = _read_json(reference_path)
    if reference is None:
        return ["learning-curve evidence reference is missing or invalid JSON"], None, None, None, None
    errors: list[str] = []
    if reference.get("schema_version") != LEARNING_CURVE_EVIDENCE_SCHEMA_VERSION:
        errors.append("learning-curve evidence schema differs")
    if reference.get("reuse_policy") != "exact_generating_command_except_destinations":
        errors.append("learning-curve evidence reuse policy differs")
    if not isinstance(reference.get("created_utc"), str):
        errors.append("learning-curve evidence lacks a creation timestamp")
    target_record = reference.get("target")
    if not isinstance(target_record, Mapping):
        return errors + ["learning-curve evidence target is invalid"], None, None, None, None
    target, output_root, target_errors = _target_spec_from_learning_evidence(
        reference_path, target_record
    )
    errors.extend(target_errors)
    if target is None or output_root is None:
        return errors, target, None, None, None
    if expected_target is not None and target_record != _learning_curve_target_record(
        expected_target, output_root
    ):
        errors.append("learning-curve evidence target differs from the requested spec")
    # A reference never masks or competes with a partial/direct target result.
    for path in _training_direct_output_paths(target):
        if path.exists():
            errors.append("direct target output exists alongside reused evidence")
            break

    source_record = reference.get("source")
    if not isinstance(source_record, Mapping):
        return errors + ["learning-curve evidence source is invalid"], target, None, None, None
    file_records = source_record.get("files")
    if not isinstance(file_records, Mapping) or set(file_records) != _LEARNING_EVIDENCE_FILE_KEYS:
        return errors + ["learning-curve evidence file inventory is invalid"], target, None, None, None
    source_paths: dict[str, Path] = {}
    for name in sorted(_LEARNING_EVIDENCE_FILE_KEYS):
        record = file_records.get(name)
        if not isinstance(record, Mapping):
            errors.append(f"learning-curve evidence file record is invalid: {name}")
            continue
        path = _resolve_recorded_path(record.get("path"))
        digest = record.get("sha256")
        if path is None or not isinstance(digest, str) or len(digest) != 64:
            errors.append(f"learning-curve evidence file record is invalid: {name}")
            continue
        source_paths[name] = path
        try:
            if not path.is_file() or _sha256_path(path) != digest:
                errors.append(f"learning-curve evidence file changed: {name}")
        except OSError as error:
            errors.append(f"learning-curve evidence file cannot be read: {name}: {error}")
    source_metrics = source_paths.get("metrics")
    if source_metrics is None:
        return errors, target, None, None, None
    if source_metrics == target.completion_path:
        errors.append("learning-curve evidence cannot reference its target output")
    try:
        source, payload, _ = _validated_learning_evidence_source(
            source_metrics, output_root
        )
    except ValueError as error:
        return errors + [f"learning-curve source validation failed: {error}"], target, None, source_metrics, None
    actual_source_record = {
        "run_id": source.run_id,
        "group": source.group,
        "experiment": source.experiment,
        "kind": source.kind,
        "command": list(source.command),
    }
    for key, value in actual_source_record.items():
        if source_record.get(key) != value:
            errors.append(f"learning-curve evidence source {key} differs")
    try:
        actual_paths = _learning_evidence_source_paths(source)
    except ValueError as error:
        errors.append(str(error))
        actual_paths = {}
    for name, path in actual_paths.items():
        if source_paths.get(name) != path:
            errors.append(f"learning-curve evidence source path differs: {name}")
    errors.extend(_training_generation_errors(source, target))
    return list(dict.fromkeys(errors)), target, source, source_metrics, payload


def _learning_curve_evidence_errors(spec: RunSpec) -> list[str]:
    if spec.group != "E7_learning_curve" or spec.kind != "training":
        return ["run is not eligible for learning-curve evidence reuse"]
    errors, _, _, _, _ = _learning_curve_evidence_validation(
        _learning_curve_evidence_path(spec), expected_target=spec
    )
    return errors


def _plan_learning_curve_evidence(
    specs: Sequence[RunSpec], source_metrics: Sequence[Path], output_root: Path
) -> dict[str, tuple[RunSpec, Path]]:
    targets = [
        spec
        for spec in specs
        if spec.group == "E7_learning_curve" and spec.kind == "training"
    ]
    plan: dict[str, tuple[RunSpec, Path]] = {}
    for source_path in source_metrics:
        source_path = source_path.resolve()
        source, _, _ = _validated_learning_evidence_source(source_path, output_root)
        matches = [
            target
            for target in targets
            if not _training_generation_errors(source, target)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{source_path} matches {len(matches)} exact E7 targets; expected one"
            )
        target = matches[0]
        if target.run_id in plan:
            raise ValueError(f"more than one evidence source maps to {target.run_id}")
        if not _terminal_completion_errors(target):
            raise ValueError(f"{target.run_id} already has a direct completed output")
        for path in _training_direct_output_paths(target):
            if path.exists():
                raise ValueError(
                    f"{target.run_id} has partial direct output; refusing evidence reuse"
                )
        plan[target.run_id] = (target, source_path)
    return plan


def _write_learning_curve_evidence(
    target: RunSpec, source_metrics: Path, output_root: Path
) -> str:
    reference_path = _learning_curve_evidence_path(target)
    _assert_safe_output(reference_path, output_root)
    existing_errors, _, _, existing_source, _ = _learning_curve_evidence_validation(
        reference_path, expected_target=target
    )
    if not existing_errors and existing_source == source_metrics.resolve():
        return "existing_reused_evidence"
    payload = _learning_curve_evidence_payload(target, source_metrics, output_root)
    _write_json(reference_path, payload)
    errors, _, _, _, _ = _learning_curve_evidence_validation(
        reference_path, expected_target=target
    )
    if errors:
        raise RuntimeError(
            f"written evidence for {target.run_id} failed validation: "
            + "; ".join(errors)
        )
    return "reused_evidence"


def _manifest_payload(spec: RunSpec, status: str) -> dict[str, object]:
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "run_id": spec.run_id,
        "group": spec.group,
        "experiment": spec.experiment,
        "kind": spec.kind,
        "description": spec.description,
        "status": status,
        "updated_utc": utc_now(),
        "working_directory": str(REPOSITORY_ROOT),
        "command": list(spec.command),
        "command_shell": shlex.join(spec.command),
        "configuration": _command_configuration(spec.command),
        "completion_path": str(spec.completion_path),
        "artifact_path": str(spec.artifact_path) if spec.artifact_path else None,
        "prerequisites": [str(path) for path in spec.prerequisites],
    }


def _existing_manifest_execution_errors(spec: RunSpec) -> list[str]:
    """Refuse to relaunch over a run that is still marked active."""

    manifest = _read_json(spec.manifest_path)
    if manifest is None:
        return []
    if manifest.get("status") == "launching":
        return [
            "run manifest is still launching; refusing a duplicate subprocess"
        ]
    return []


def print_plan(
    specs: Sequence[RunSpec],
    *,
    force: bool,
    adopt_only: bool = False,
    adoption_errors: Mapping[str, Sequence[str]] | None = None,
    reuse_evidence: Mapping[str, Path] | None = None,
) -> None:
    if not specs:
        print("No subprocess commands in aggregate mode.")
        return
    for ordinal, spec in enumerate(specs, start=1):
        if adopt_only:
            recovery_errors = (
                list(adoption_errors[spec.run_id])
                if adoption_errors is not None and spec.run_id in adoption_errors
                else _recoverable_training_output_errors(spec)
            )
            status = (
                "ADOPT validated zero-exit output"
                if not recovery_errors
                else "REFUSE adoption: " + "; ".join(recovery_errors)
            )
        elif reuse_evidence is not None and spec.run_id in reuse_evidence:
            status = f"REUSE exact evidence from {reuse_evidence[spec.run_id]}"
        else:
            reused_evidence = not _learning_curve_evidence_errors(spec)
            direct_complete = not _terminal_completion_errors(spec)
            complete = direct_complete or reused_evidence
            if reused_evidence and not force:
                status = "SKIP exact reused evidence"
            elif direct_complete and not force:
                status = "SKIP completed"
            elif (execution_errors := _existing_manifest_execution_errors(spec)):
                status = "WAIT " + "; ".join(execution_errors)
            elif any(not path.exists() for path in spec.prerequisites):
                status = "WAIT missing prerequisite"
            elif complete and force:
                status = "RUN forced"
            else:
                status = "RUN"
        print(f"[{ordinal:02d}] {status}: {spec.experiment} {spec.run_id}")
        print(f"     {spec.description}")
        print(f"     {shlex.join(spec.command)}")


def _post_official_gate_payload(spec: RunSpec) -> dict[str, object]:
    command = _command_configuration(spec.command)
    official_dir = _resolve_recorded_path(command.get("output_dir"))
    candidate_artifact = _resolve_recorded_path(command.get("hybrid_model"))
    candidate_name = str(command.get("third_model_name", "candidate"))
    if official_dir is None or candidate_artifact is None:
        raise ValueError("official specification lacks output or candidate artifact")
    if not candidate_artifact.is_file():
        raise ValueError("official candidate artifact is missing")

    ablation_rows, _ = _read_csv_rows(official_dir / "model_ablation.csv")
    candidate_overall = next(
        (
            row
            for row in ablation_rows
            if row.get("model") == candidate_name and row.get("scenario") == "overall"
        ),
        None,
    )
    if candidate_overall is None:
        raise ValueError("official ablation lacks the candidate overall row")
    correct_answers = _as_int(candidate_overall.get("correct_answers"))
    mrr = _as_float(candidate_overall.get("mrr"))

    scenario_rows = [
        row
        for row in ablation_rows
        if row.get("model") == candidate_name and row.get("scenario") != "overall"
    ]
    required_scenarios = {"buying", "browsing", "boundary", "intent_override"}
    scenarios_reported = required_scenarios.issubset(
        {row.get("scenario") for row in scenario_rows}
    )

    session_rows, _ = _read_csv_rows(official_dir / "model_ablation_sessions.csv")
    candidate_session_ids = {
        row.get("sample_id", "")
        for row in session_rows
        if row.get("model") == candidate_name and row.get("sample_id")
    }
    sessions_complete = len(candidate_session_ids) == 200

    bootstrap_rows, _ = _read_csv_rows(
        official_dir / "model_ablation_bootstrap.csv"
    )
    required_uncertainty = {
        (f"{candidate_name}_minus_{baseline}", metric)
        for baseline in ("fm", "linear")
        for metric in ("accuracy", "mrr", "efficiency", "technical_score")
    }
    observed_uncertainty = {
        (row.get("comparison", ""), row.get("metric", ""))
        for row in bootstrap_rows
        if _as_int(row.get("sample_count")) == 200
        and _as_float(row.get("observed_delta")) is not None
        and _as_float(row.get("ci_95_lower")) is not None
        and _as_float(row.get("ci_95_upper")) is not None
    }
    uncertainty_reported = required_uncertainty.issubset(observed_uncertainty)

    gates = {
        "official_correct_answers": {
            "operator": ">=",
            "threshold": OFFICIAL_CORRECT_ANSWERS_GATE,
            "observed": correct_answers,
            "passed": correct_answers is not None
            and correct_answers >= OFFICIAL_CORRECT_ANSWERS_GATE,
        },
        "official_mrr": {
            "operator": ">",
            "threshold": OFFICIAL_MRR_GATE,
            "observed": mrr,
            "passed": mrr is not None and mrr > OFFICIAL_MRR_GATE,
        },
        "per_scenario_failures_reported": {
            "required_scenarios": sorted(required_scenarios),
            "passed": scenarios_reported,
        },
        "paired_session_uncertainty_reported": {
            "required_comparisons": sorted(
                f"{comparison}/{metric}" for comparison, metric in required_uncertainty
            ),
            "passed": uncertainty_reported,
        },
        "candidate_sessions_complete": {
            "expected": 200,
            "observed": len(candidate_session_ids),
            "passed": sessions_complete,
        },
    }
    all_passed = all(
        isinstance(value, Mapping) and value.get("passed") is True
        for value in gates.values()
    )
    return {
        "schema_version": POST_OFFICIAL_GATE_SCHEMA_VERSION,
        "generated_utc": utc_now(),
        "run_id": spec.run_id,
        "decision": (
            "eligible_for_manual_promotion"
            if all_passed
            else "not_eligible_for_promotion"
        ),
        "automatic_promotion_performed": False,
        "candidate_artifact": str(candidate_artifact),
        "candidate_artifact_sha256": _sha256_path(candidate_artifact),
        "gates": gates,
        "evidence": {
            "model_ablation": str(official_dir / "model_ablation.csv"),
            "per_session_failures": str(
                official_dir / "model_ablation_sessions.csv"
            ),
            "paired_uncertainty": str(
                official_dir / "model_ablation_bootstrap.csv"
            ),
        },
    }


def _post_official_gate_errors(spec: RunSpec) -> list[str]:
    path = spec.output_dir / "post_official_promotion_decision.json"
    recorded = _read_json(path)
    if recorded is None:
        return ["post-official promotion decision is missing or invalid"]
    try:
        expected = _post_official_gate_payload(spec)
    except (OSError, ValueError) as error:
        return [f"post-official promotion evidence cannot be validated: {error}"]
    recorded_comparable = {
        key: value for key, value in recorded.items() if key != "generated_utc"
    }
    expected_comparable = {
        key: value for key, value in expected.items() if key != "generated_utc"
    }
    if recorded_comparable != expected_comparable:
        return ["post-official promotion decision disagrees with official outputs"]
    return []


def _write_post_official_gate(spec: RunSpec) -> None:
    _write_json(
        spec.output_dir / "post_official_promotion_decision.json",
        _post_official_gate_payload(spec),
    )


def _recoverable_training_output_errors(spec: RunSpec) -> list[str]:
    """Validate a runner-produced, zero-exit training output for adoption."""

    errors: list[str] = []
    if spec.kind != "training":
        return ["only training outputs can be recovered without re-execution"]
    manifest = _read_json(spec.manifest_path)
    if manifest is None:
        return ["failed-validation runner manifest is missing or invalid JSON"]
    if manifest.get("status") != "failed_validation":
        errors.append("manifest status is not failed_validation")
    return_code = manifest.get("return_code")
    if type(return_code) is not int or return_code != 0:
        errors.append("manifest does not record an exact zero subprocess return code")
    if manifest.get("metrics_complete") is not False:
        errors.append("manifest does not record the failed output validation")
    for key in ("started_utc", "finished_utc"):
        value = manifest.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"manifest lacks {key}")

    errors.extend(
        _manifest_errors(
            spec,
            manifest,
            allowed_statuses=frozenset({"failed_validation"}),
        )
    )
    expected = _manifest_payload(spec, "failed_validation")
    for key in (
        "description",
        "working_directory",
        "command_shell",
        "prerequisites",
    ):
        if manifest.get(key) != expected.get(key):
            errors.append(f"manifest {key} differs")
    errors.extend(
        _completion_errors(
            spec,
            allowed_manifest_statuses=frozenset({"failed_validation"}),
            require_manifest=True,
        )
    )
    # Avoid repeating a manifest mismatch that is discovered by both the
    # identity and completion passes while preserving deterministic ordering.
    return list(dict.fromkeys(errors))


def _adopt_valid_training_output(spec: RunSpec) -> str:
    errors = _recoverable_training_output_errors(spec)
    if errors:
        raise RuntimeError(f"cannot adopt {spec.run_id}: " + "; ".join(errors))
    manifest = _read_json(spec.manifest_path)
    assert manifest is not None  # Proved above; retained for type checkers.
    previous_validation_errors = manifest.get("validation_errors")
    previous_validation_errors = (
        previous_validation_errors
        if isinstance(previous_validation_errors, list)
        and all(isinstance(value, str) for value in previous_validation_errors)
        else []
    )
    adopted_utc = utc_now()
    manifest["status"] = "adopted_completed"
    manifest["metrics_complete"] = True
    manifest["validation_errors"] = []
    manifest["updated_utc"] = adopted_utc
    manifest["recovery"] = {
        "adopted_utc": adopted_utc,
        "previous_status": "failed_validation",
        "previous_validation_errors": previous_validation_errors,
    }
    _write_json(spec.manifest_path, manifest)
    if not _is_complete(spec):
        # A concurrent artifact change after the pre-write validation must not
        # leave a completed attestation behind.
        post_write_errors = _completion_errors(
            spec,
            allowed_manifest_statuses=frozenset({"adopted_completed"}),
            require_manifest=True,
        )
        manifest["status"] = "failed_validation"
        manifest["metrics_complete"] = False
        manifest["validation_errors"] = post_write_errors
        manifest["updated_utc"] = utc_now()
        _write_json(spec.manifest_path, manifest)
        raise RuntimeError(
            f"{spec.run_id} changed during adoption validation: "
            + "; ".join(post_write_errors)
        )
    return "adopted_completed"


def execute_spec(
    spec: RunSpec, *, force: bool, adopt_valid_output: bool = False
) -> str:
    if adopt_valid_output:
        if force:
            raise ValueError("--force cannot be combined with output adoption")
        return _adopt_valid_training_output(spec)
    if not force and not _learning_curve_evidence_errors(spec):
        return "skipped_reused_evidence"
    if not _terminal_completion_errors(spec) and not force:
        # Manifestless training outputs are adopted only after their embedded
        # trajectory/training configs, versions, and artifact path have been
        # proven equivalent by ``_is_complete``. Evaluation outputs cannot be
        # adopted because they do not embed the full generating configuration.
        if not spec.manifest_path.exists():
            manifest = _manifest_payload(spec, "adopted_completed")
            manifest["metrics_complete"] = True
            manifest["validation_errors"] = []
            _write_json(spec.manifest_path, manifest)
        return "skipped_completed"
    execution_errors = _existing_manifest_execution_errors(spec)
    if execution_errors:
        raise RuntimeError(
            f"cannot execute {spec.run_id}: " + "; ".join(execution_errors)
        )
    missing = [path for path in spec.prerequisites if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"cannot execute {spec.run_id}; missing prerequisite(s): "
            + ", ".join(str(path) for path in missing)
        )

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_payload(spec, "launching")
    manifest["started_utc"] = utc_now()
    _write_json(spec.manifest_path, manifest)
    print(f"Launching {spec.run_id}: {shlex.join(spec.command)}", flush=True)
    completed = subprocess.run(spec.command, cwd=REPOSITORY_ROOT, check=False)
    postprocess_error: str | None = None
    if completed.returncode == 0 and spec.group == "E8_official_handoff":
        try:
            _write_post_official_gate(spec)
        except (OSError, ValueError) as error:
            postprocess_error = str(error)
    manifest["finished_utc"] = utc_now()
    manifest["updated_utc"] = manifest["finished_utc"]
    manifest["return_code"] = completed.returncode
    manifest["status"] = "completed" if completed.returncode == 0 else "failed"
    # Persist a terminal status before validating so the validator observes the
    # exact manifest/spec identity that will be retained on success.
    _write_json(spec.manifest_path, manifest)
    validation_errors = _completion_errors(
        spec,
        allowed_manifest_statuses=frozenset({str(manifest["status"])}),
        require_manifest=True,
    )
    if postprocess_error is not None:
        manifest["postprocess_error"] = postprocess_error
        validation_errors.append(f"postprocessing failed: {postprocess_error}")
    if completed.returncode:
        validation_errors.insert(
            0, f"subprocess returned nonzero exit status {completed.returncode}"
        )
    validation_errors = list(dict.fromkeys(validation_errors))
    manifest["validation_errors"] = validation_errors
    manifest["metrics_complete"] = (
        completed.returncode == 0 and not validation_errors
    )
    if not manifest["metrics_complete"] and completed.returncode == 0:
        manifest["status"] = "failed_validation"
    _write_json(spec.manifest_path, manifest)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, spec.command)
    if not manifest["metrics_complete"]:
        raise RuntimeError(
            f"{spec.run_id} returned success without complete validated outputs: "
            + "; ".join(validation_errors)
        )
    return "completed"


def _as_float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _selected_history_record(payload: Mapping[str, object]) -> Mapping[str, object]:
    history = payload.get("training_history")
    if not isinstance(history, list):
        return {}
    selected_epoch = _as_int(payload.get("selected_epoch"))
    candidates = [row for row in history if isinstance(row, Mapping)]
    if selected_epoch is not None:
        for row in candidates:
            if _as_int(row.get("epoch")) == selected_epoch:
                return row
    return candidates[-1] if candidates else {}


def metric_row(metrics_path: Path, payload: Mapping[str, object]) -> dict[str, object]:
    run_manifest = _read_json(metrics_path.parent / "run_manifest.json") or {}
    training = payload.get("training_config")
    training = training if isinstance(training, Mapping) else {}
    trajectory = payload.get("trajectory_config")
    trajectory = trajectory if isinstance(trajectory, Mapping) else {}
    dataset_manifest = payload.get("dataset_manifest")
    dataset_manifest = dataset_manifest if isinstance(dataset_manifest, Mapping) else {}
    input_hashes = dataset_manifest.get("input_sha256")
    input_hashes = input_hashes if isinstance(input_hashes, Mapping) else {}
    validation = payload.get("full_survivor_validation")
    validation = validation if isinstance(validation, Mapping) else {}
    test = payload.get("full_survivor_test")
    test = test if isinstance(test, Mapping) else {}
    history = _selected_history_record(payload)
    return {
        "metric_scope": "product_heldout_training_artifact",
        "experiment": run_manifest.get("experiment", "unknown"),
        "group": run_manifest.get("group", metrics_path.parent.parent.name),
        "run_id": run_manifest.get("run_id", metrics_path.parent.name),
        "variant": payload.get("model"),
        "seed": training.get("seed"),
        "model_seed": training.get("seed"),
        "trajectory_seed": trajectory.get("seed"),
        "split_seed": trajectory.get("split_seed"),
        "dataset_version": trajectory.get("dataset_version"),
        "catalog_sha256": input_hashes.get("catalog"),
        "catalog_records_sha256": input_hashes.get("catalog_records"),
        "scenario_mix": trajectory.get("scenario_mix"),
        "trajectory_count": trajectory.get("trajectory_count"),
        "extended_fraction": trajectory.get("extended_fraction"),
        "state_count": dataset_manifest.get("state_count"),
        "effective_weighted_pairs": history.get("effective_weighted_pairs"),
        "training_seconds": payload.get("elapsed_seconds"),
        "selected_epoch": payload.get("selected_epoch"),
        "supervision_policy": training.get("supervision_policy"),
        "tie_weight": training.get("tie_weight"),
        "category_only_weight": training.get("category_only_weight"),
        "evidence_saturation": training.get("evidence_saturation"),
        "negative_count": training.get("negatives_per_state"),
        "negative_pre_pool_size": training.get("negative_pre_pool_size"),
        "negative_mode": _canonical_negative_mode(training.get("negative_mode")),
        "hard_fraction": training.get("hard_fraction"),
        "near_fraction": training.get("near_fraction"),
        "random_fraction": training.get("random_fraction"),
        "other_encoding": training.get("other_encoding"),
        "dimension": training.get("dimension"),
        "learning_rate": training.get("learning_rate"),
        "latent_l2": training.get("latent_l2"),
        "linear_l2": training.get("linear_l2"),
        "cross_l2": training.get("cross_l2"),
        "minimum_value_support": training.get("minimum_value_support"),
        "minimum_cross_support": training.get("minimum_cross_support"),
        "max_epochs": training.get("max_epochs"),
        "patience": training.get("patience"),
        "validation_interval": training.get("validation_interval"),
        "pair_batch_size": training.get("pair_batch_size"),
        "validation_mrr": validation.get("mrr"),
        "validation_hit_at_1": validation.get("hit_rate_at_1"),
        "validation_hit_at_5": validation.get("hit_rate_at_5"),
        "validation_hit_at_10": validation.get("hit_rate_at_10"),
        "validation_mean_rank": validation.get("mean_rank"),
        "validation_candidate_width": validation.get("mean_candidate_width"),
        "test_mrr": test.get("mrr"),
        "test_hit_at_10": test.get("hit_rate_at_10"),
        "artifact": payload.get("artifact"),
        "metrics_path": str(metrics_path),
    }


def _validated_training_metric_spec(
    metrics_path: Path, payload: Mapping[str, object]
) -> RunSpec | None:
    """Reconstruct and validate the exact completed spec used by aggregation."""

    manifest = _read_json(metrics_path.parent / "run_manifest.json")
    if manifest is None:
        return None
    command = manifest.get("command")
    artifact = _resolve_recorded_path(manifest.get("artifact_path"))
    if (
        not isinstance(command, list)
        or not all(isinstance(value, str) for value in command)
        or artifact is None
    ):
        return None
    spec = RunSpec(
        run_id=str(manifest.get("run_id", "")),
        group=str(manifest.get("group", "")),
        experiment=str(manifest.get("experiment", "")),
        description=str(manifest.get("description", "")),
        output_dir=metrics_path.parent,
        command=tuple(command),
        completion_path=metrics_path,
        artifact_path=artifact,
        kind="training",
    )
    if (
        _manifest_errors(spec, manifest)
        or _training_manifest_attestation_errors(manifest)
        or _training_payload_errors(spec, payload)
        or not artifact.exists()
    ):
        return None
    return spec


def _ancestor_run_manifest(path: Path, output_root: Path) -> dict[str, object]:
    directory = path.parent
    while True:
        candidate = directory / "run_manifest.json"
        payload = _read_json(candidate)
        if payload is not None:
            return payload
        if directory == output_root or directory.parent == directory:
            return {}
        directory = directory.parent


def evaluation_metric_row(
    metrics_path: Path,
    payload: Mapping[str, object],
    output_root: Path,
) -> dict[str, object] | None:
    overall = payload.get("overall")
    overall = overall if isinstance(overall, Mapping) else {}
    primary = overall.get("primary")
    primary = primary if isinstance(primary, Mapping) else {}
    if not primary or "mrr" not in primary:
        return None
    width = overall.get("candidate_width")
    width = width if isinstance(width, Mapping) else {}
    run_manifest = _ancestor_run_manifest(metrics_path, output_root)
    evaluator_slot = metrics_path.name.removesuffix("_metrics.json")
    group = run_manifest.get("group", metrics_path.parent.parent.name)
    model_name = (
        "candidate"
        if group == "E8_official_handoff" and evaluator_slot == "hybrid"
        else evaluator_slot
    )
    configuration = run_manifest.get("configuration")
    configuration = configuration if isinstance(configuration, Mapping) else {}
    return {
        "metric_scope": "ranker_only_full_survivor_evaluation",
        "experiment": run_manifest.get("experiment", "E0/E8 evaluation"),
        "group": group,
        "run_id": f"{run_manifest.get('run_id', metrics_path.parent.name)}:{model_name}",
        "variant": model_name,
        "evaluator_slot": evaluator_slot,
        "seed": None,
        "model_seed": None,
        "trajectory_seed": configuration.get("trajectory_seed"),
        "split_seed": configuration.get("split_seed"),
        "scenario_mix": configuration.get("scenario_mix"),
        "trajectory_count": overall.get("trajectory_count"),
        "state_count": overall.get("state_count"),
        "effective_weighted_pairs": None,
        "training_seconds": None,
        "selected_epoch": None,
        "supervision_policy": None,
        "tie_weight": None,
        "negative_count": None,
        "negative_pre_pool_size": None,
        "negative_mode": None,
        "hard_fraction": None,
        "near_fraction": None,
        "random_fraction": None,
        "other_encoding": None,
        "dimension": None,
        "learning_rate": None,
        "latent_l2": None,
        "linear_l2": None,
        "validation_mrr": primary.get("mrr"),
        "validation_hit_at_1": primary.get("hit_rate_at_1"),
        "validation_hit_at_5": primary.get("hit_rate_at_5"),
        "validation_hit_at_10": primary.get("hit_rate_at_10"),
        "validation_mean_rank": primary.get("mean_rank"),
        "validation_candidate_width": width.get("mean"),
        "test_mrr": None,
        "test_hit_at_10": None,
        "artifact": payload.get("artifact"),
        "metrics_path": str(metrics_path),
    }


def aggregate_metric_rows(output_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not output_root.exists():
        return rows
    for metrics_path in sorted(output_root.rglob("metrics.json")):
        payload = _read_json(metrics_path)
        if payload is None or not isinstance(payload.get("full_survivor_validation"), dict):
            continue
        if _validated_training_metric_spec(metrics_path, payload) is None:
            continue
        rows.append(metric_row(metrics_path, payload))
    direct_learning_run_ids = {
        str(row.get("run_id"))
        for row in rows
        if row.get("group") == "E7_learning_curve"
    }
    for reference_path in sorted(
        output_root.rglob(LEARNING_CURVE_EVIDENCE_FILENAME)
    ):
        errors, target, source, source_metrics, payload = (
            _learning_curve_evidence_validation(reference_path)
        )
        if (
            errors
            or target is None
            or source is None
            or source_metrics is None
            or payload is None
            or target.run_id in direct_learning_run_ids
        ):
            continue
        row = metric_row(source_metrics, payload)
        row.update(
            {
                "experiment": target.experiment,
                "group": target.group,
                "run_id": target.run_id,
                "evidence_reused": True,
                "evidence_reference": str(reference_path.resolve()),
                "evidence_source_run_id": source.run_id,
                "evidence_source_group": source.group,
            }
        )
        rows.append(row)
        direct_learning_run_ids.add(target.run_id)
    for metrics_path in sorted(output_root.rglob("*_metrics.json")):
        payload = _read_json(metrics_path)
        if payload is None:
            continue
        row = evaluation_metric_row(metrics_path, payload, output_root)
        if row is not None:
            rows.append(row)
    return rows


ABLATION_FIELDS = (
    "metric_scope",
    "experiment",
    "group",
    "run_id",
    "variant",
    "evaluator_slot",
    "seed",
    "model_seed",
    "trajectory_seed",
    "split_seed",
    "dataset_version",
    "catalog_sha256",
    "catalog_records_sha256",
    "scenario_mix",
    "trajectory_count",
    "extended_fraction",
    "state_count",
    "effective_weighted_pairs",
    "training_seconds",
    "selected_epoch",
    "supervision_policy",
    "tie_weight",
    "category_only_weight",
    "evidence_saturation",
    "negative_count",
    "negative_pre_pool_size",
    "negative_mode",
    "hard_fraction",
    "near_fraction",
    "random_fraction",
    "other_encoding",
    "dimension",
    "learning_rate",
    "latent_l2",
    "linear_l2",
    "cross_l2",
    "minimum_value_support",
    "minimum_cross_support",
    "max_epochs",
    "patience",
    "validation_interval",
    "pair_batch_size",
    "validation_mrr",
    "validation_hit_at_1",
    "validation_hit_at_5",
    "validation_hit_at_10",
    "validation_mean_rank",
    "validation_candidate_width",
    "test_mrr",
    "test_hit_at_10",
    "artifact",
    "metrics_path",
    "evidence_reused",
    "evidence_reference",
    "evidence_source_run_id",
    "evidence_source_group",
)


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def learning_curve_rows(
    rows: Sequence[Mapping[str, object]], threshold: float
) -> tuple[list[dict[str, object]], dict[str, object]]:
    retained = [row for row in rows if row.get("group") == "E7_learning_curve"]
    decision = plateau_decision(retained, threshold)
    selected = _as_int(decision.get("selected_trajectory_count"))
    comparisons = {
        _as_int(row.get("from_trajectories")): row
        for row in decision.get("comparisons", [])  # type: ignore[union-attr]
        if isinstance(row, Mapping)
    }
    output: list[dict[str, object]] = []
    points = decision.get("points")
    points = points if isinstance(points, list) else []
    for point in points:
        if not isinstance(point, Mapping):
            continue
        count = _as_int(point.get("trajectory_count"))
        matching = [
            row
            for row in retained
            if _as_int(row.get("trajectory_count")) == count
        ]
        comparison = comparisons.get(count, {})

        def mean_value(key: str) -> float | None:
            values = [
                value
                for value in (_as_float(row.get(key)) for row in matching)
                if value is not None
            ]
            return fmean(values) if values else None

        model_seeds = sorted(
            {
                seed
                for seed in (
                    _as_int(row.get("model_seed", row.get("seed")))
                    for row in matching
                )
                if seed is not None
            }
        )
        training_times = [
            value
            for value in (
                _as_float(row.get("training_seconds")) for row in matching
            )
            if value is not None
        ]
        output.append(
            {
                "trajectory_count": count,
                "model_seed_count": len(model_seeds),
                "model_seeds": ";".join(str(seed) for seed in model_seeds),
                "trajectory_seed": matching[0].get("trajectory_seed") if matching else None,
                "split_seed": matching[0].get("split_seed") if matching else None,
                "mean_state_count": mean_value("state_count"),
                "mean_effective_weighted_pairs": mean_value(
                    "effective_weighted_pairs"
                ),
                "mean_training_seconds": (
                    fmean(training_times) if training_times else None
                ),
                "total_training_seconds": sum(training_times),
                "validation_mrr": point.get("validation_mrr"),
                "validation_mrr_min": point.get("validation_mrr_min"),
                "validation_mrr_max": point.get("validation_mrr_max"),
                "validation_mrr_stddev": point.get("validation_mrr_stddev"),
                "validation_hit_at_10": mean_value("validation_hit_at_10"),
                "mrr_improvement_to_next": comparison.get("mrr_improvement"),
                "next_larger_dataset_justified": comparison.get(
                    "larger_dataset_justified"
                ),
                "plateau_threshold": threshold,
                "plateau_selected": count == selected,
                "three_seed_complete": point.get("complete"),
                "metrics_paths": ";".join(
                    str(row.get("metrics_path")) for row in matching
                ),
            }
        )
    return output, decision


LEARNING_FIELDS = (
    "trajectory_count",
    "model_seed_count",
    "model_seeds",
    "trajectory_seed",
    "split_seed",
    "mean_state_count",
    "mean_effective_weighted_pairs",
    "mean_training_seconds",
    "total_training_seconds",
    "validation_mrr",
    "validation_mrr_min",
    "validation_mrr_max",
    "validation_mrr_stddev",
    "validation_hit_at_10",
    "mrr_improvement_to_next",
    "next_larger_dataset_justified",
    "plateau_threshold",
    "plateau_selected",
    "three_seed_complete",
    "metrics_paths",
)


TUNING_FIELDS = (
    "rank",
    "selected",
    "run_id",
    "validation_mrr",
    "training_seconds",
    "trajectory_count",
    "model_seed",
    "trajectory_seed",
    "split_seed",
    "dimension",
    "learning_rate",
    "latent_l2",
    "linear_l2",
    "artifact",
    "metrics_path",
)


def learning_curve_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1240, 760
    title = "Full-survivor validation learning curves"
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" '
        f'font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
    ]
    x_axes = (
        ("trajectory_count", "Complete trajectories"),
        ("mean_state_count", "Retained states"),
        ("mean_effective_weighted_pairs", "Effective weighted pairs"),
        ("mean_training_seconds", "Training time (seconds)"),
    )
    y_axes = (
        ("validation_mrr", "Validation MRR", "#2563eb"),
        ("validation_hit_at_10", "Validation HR@10", "#059669"),
    )
    panel_width, panel_height = 280, 300
    column_gap, row_gap = 20, 42
    origin_x, origin_y = 40, 58
    plot_left, plot_right, plot_top, plot_bottom = 48, 12, 38, 48

    def tick_label(value: float) -> str:
        magnitude = abs(value)
        if magnitude >= 1_000_000:
            return f"{value / 1_000_000:.1f}m"
        if magnitude >= 1_000:
            return f"{value / 1_000:.1f}k"
        return f"{value:.1f}" if not value.is_integer() else str(int(value))

    for row_index, (y_key, y_label, color) in enumerate(y_axes):
        for column_index, (x_key, x_label) in enumerate(x_axes):
            panel_x = origin_x + column_index * (panel_width + column_gap)
            panel_y = origin_y + row_index * (panel_height + row_gap)
            left = panel_x + plot_left
            top = panel_y + plot_top
            plot_width = panel_width - plot_left - plot_right
            plot_height = panel_height - plot_top - plot_bottom
            points = [
                (_as_float(row.get(x_key)), _as_float(row.get(y_key)))
                for row in rows
            ]
            points = [
                (x, y) for x, y in points if x is not None and y is not None
            ]
            elements.append(
                f'<text x="{panel_x + panel_width / 2}" y="{panel_y + 18}" '
                'text-anchor="middle" font-family="sans-serif" font-size="13">'
                f'{html.escape(y_label)} vs {html.escape(x_label)}</text>'
            )
            elements.extend(
                (
                    f'<line x1="{left}" y1="{top}" x2="{left}" '
                    f'y2="{top + plot_height}" stroke="#444"/>',
                    f'<line x1="{left}" y1="{top + plot_height}" '
                    f'x2="{left + plot_width}" y2="{top + plot_height}" '
                    'stroke="#444"/>',
                )
            )
            if not points:
                elements.append(
                    f'<text x="{left + plot_width / 2}" '
                    f'y="{top + plot_height / 2}" text-anchor="middle" '
                    'font-family="sans-serif" font-size="12" fill="#666">'
                    "No completed E7 metrics</text>"
                )
                continue
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            x_min, x_max = min(x_values), max(x_values)
            raw_y_min, raw_y_max = min(y_values), max(y_values)
            y_padding = max(0.002, (raw_y_max - raw_y_min) * 0.15)
            y_min = max(0.0, raw_y_min - y_padding)
            y_max = min(1.0, raw_y_max + y_padding)
            if math.isclose(y_min, y_max):
                y_min = max(0.0, y_min - 0.01)
                y_max = min(1.0, y_max + 0.01)

            def x_coordinate(value: float) -> float:
                if math.isclose(x_min, x_max):
                    return left + plot_width / 2
                return left + (value - x_min) / (x_max - x_min) * plot_width

            def y_coordinate(value: float) -> float:
                return top + (y_max - value) / (y_max - y_min) * plot_height

            for tick in range(4):
                value = y_min + (y_max - y_min) * tick / 3
                y = y_coordinate(value)
                elements.append(
                    f'<line x1="{left}" y1="{y:.2f}" '
                    f'x2="{left + plot_width}" y2="{y:.2f}" stroke="#ececec"/>'
                )
                elements.append(
                    f'<text x="{left - 5}" y="{y + 4:.2f}" text-anchor="end" '
                    f'font-family="sans-serif" font-size="9">{value:.3f}</text>'
                )
            ordered = sorted(points)
            polyline = " ".join(
                f"{x_coordinate(x):.2f},{y_coordinate(y):.2f}" for x, y in ordered
            )
            elements.append(
                f'<polyline points="{polyline}" fill="none" stroke="{color}" '
                'stroke-width="2.5"/>'
            )
            for x, y in ordered:
                elements.append(
                    f'<circle cx="{x_coordinate(x):.2f}" cy="{y_coordinate(y):.2f}" '
                    f'r="4" fill="{color}"/>'
                )
            for value, anchor in ((x_min, "start"), (x_max, "end")):
                elements.append(
                    f'<text x="{x_coordinate(value):.2f}" '
                    f'y="{top + plot_height + 16}" text-anchor="{anchor}" '
                    f'font-family="sans-serif" font-size="9">{tick_label(value)}</text>'
                )
            elements.append(
                f'<text x="{left + plot_width / 2}" '
                f'y="{panel_y + panel_height - 8}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="11">{html.escape(x_label)}</text>'
            )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def write_aggregates(
    output_root: Path,
    threshold: float,
    tuning_trajectory_count: int | None = None,
    *,
    tuning_trajectory_seed: int | None = None,
    tuning_split_seed: int | None = None,
    tuning_model_seed: int | None = None,
) -> dict[str, object]:
    rows = aggregate_metric_rows(output_root)
    learning_rows, decision = learning_curve_rows(rows, threshold)
    if tuning_trajectory_count is None:
        tuning_trajectory_count = _as_int(decision.get("selected_trajectory_count"))
    if tuning_trajectory_count is None:
        tuning = tuning_decision(
            [],
            None,
            trajectory_seed=tuning_trajectory_seed,
            split_seed=tuning_split_seed,
            model_seed=tuning_model_seed,
        )
    else:
        tuning = tuning_decision(
            rows,
            tuning_trajectory_count,
            trajectory_seed=tuning_trajectory_seed,
            split_seed=tuning_split_seed,
            model_seed=tuning_model_seed,
        )
    ranked = tuning.get("ranked_candidates")
    ranked = ranked if isinstance(ranked, list) else []
    selected_tuning = tuning.get("selected")
    selected_tuning = (
        selected_tuning if isinstance(selected_tuning, Mapping) else None
    )
    tuning_rows = [
        {
            **row,
            "rank": rank,
            "selected": (
                selected_tuning is not None
                and row.get("metrics_path") == selected_tuning.get("metrics_path")
            ),
        }
        for rank, row in enumerate(ranked, start=1)
        if isinstance(row, Mapping)
    ]
    _write_csv(output_root / "ablation.csv", ABLATION_FIELDS, rows)
    _write_csv(output_root / "learning_curve.csv", LEARNING_FIELDS, learning_rows)
    _write_csv(output_root / "tuning.csv", TUNING_FIELDS, tuning_rows)
    _atomic_write_text(
        output_root / "learning_curve.svg", learning_curve_svg(learning_rows)
    )
    _write_json(output_root / "plateau_decision.json", decision)
    _write_json(output_root / "tuning_decision.json", tuning)
    summary = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "generated_utc": utc_now(),
        "completed_metric_rows": len(rows),
        "completed_training_runs": sum(
            row.get("metric_scope") == "product_heldout_training_artifact"
            for row in rows
        ),
        "completed_evaluation_rows": sum(
            row.get("metric_scope") == "ranker_only_full_survivor_evaluation"
            for row in rows
        ),
        "completed_learning_curve_runs": sum(
            row.get("group") == "E7_learning_curve" for row in rows
        ),
        "completed_learning_curve_sizes": len(learning_rows),
        "completed_tuning_runs": len(tuning_rows),
        "plateau": decision,
        "tuning": tuning,
        "ablation_csv": str(output_root / "ablation.csv"),
        "learning_curve_csv": str(output_root / "learning_curve.csv"),
        "learning_curve_svg": str(output_root / "learning_curve.svg"),
        "tuning_csv": str(output_root / "tuning.csv"),
    }
    _write_json(output_root / "aggregation_manifest.json", summary)
    return summary


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if len(result) != 3:
        raise argparse.ArgumentTypeError("experiment requires exactly three seeds")
    if len(set(result)) != 3:
        raise argparse.ArgumentTypeError("the three seeds must be distinct")
    return result


def _parse_block_count(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("trajectory count must be an integer") from error
    if result <= 0 or result % 20:
        raise argparse.ArgumentTypeError(
            "trajectory count must be positive and divisible by 20"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "List or execute the resumable E0-E8 FM experiment program. "
            "Without --execute this command is a read-only dry run."
        )
    )
    parser.add_argument(
        "--mode",
        choices=(
            "all",
            "baseline",
            "smoke",
            "cumulative",
            "supervision",
            "negatives",
            "negative-mixture",
            "sensitivity",
            "other-encoding",
            "learning-curve",
            "tuning",
            "final-seeds",
            "official",
            "correctness",
            "aggregate",
        ),
        default="all",
        help="experiment family to list or execute (default: all)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="launch subprocesses and write manifests/aggregates",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun completed versioned outputs; frozen references remain protected",
    )
    parser.add_argument(
        "--adopt-valid-run",
        action="append",
        default=None,
        metavar="RUN_ID",
        help=(
            "adoption-only recovery for a selected failed_validation training run; "
            "requires its exact zero-exit manifest/spec and fully validated outputs "
            "and never launches a subprocess (repeatable)"
        ),
    )
    parser.add_argument(
        "--reuse-learning-curve-evidence",
        type=Path,
        action="append",
        default=None,
        metavar="METRICS_JSON",
        help=(
            "reuse a fully validated, exact-compatible completed training result "
            "for its matching E7 target without copying metrics or models "
            "(learning-curve mode only; repeatable and reuse-only when executed)"
        ),
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="model initialization, minibatch, and negative-sampler seed",
    )
    parser.add_argument(
        "--trajectory-seed",
        type=int,
        default=2026,
        help="fixed trajectory-generation seed, independent from --seed",
    )
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument(
        "--supervision-policy",
        choices=("skip_ties", "downweight_ties", "set_valued_positives"),
        default="set_valued_positives",
    )
    parser.add_argument("--tie-weight", type=float, default=0.10)
    parser.add_argument("--category-only-weight", type=float, default=0.05)
    parser.add_argument("--evidence-saturation", type=int, default=3)
    parser.add_argument("--negatives", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument(
        "--negative-mode",
        choices=("product_fixed", "survivor_static", "survivor_dynamic"),
        default=DEFAULT_NEGATIVE_MODE,
    )
    parser.add_argument("--hard-fraction", type=float, default=0.50)
    parser.add_argument("--near-fraction", type=float, default=0.25)
    parser.add_argument("--random-fraction", type=float, default=0.25)
    parser.add_argument(
        "--other-encoding", choices=("legacy", "dual"), default="dual"
    )
    parser.add_argument("--dimension", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--latent-l2", type=float, default=1e-5)
    parser.add_argument("--linear-l2", type=float, default=1e-5)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--extended-fraction", type=float, default=0.10)
    parser.add_argument(
        "--plateau-threshold", type=float, default=PLATEAU_MRR_THRESHOLD
    )
    parser.add_argument(
        "--learning-curve-seeds",
        type=_parse_seeds,
        default=FINAL_SEEDS,
        metavar="2026,2027,2028",
        help="exactly three distinct model seeds for every E7 size",
    )
    parser.add_argument(
        "--final-trajectories",
        type=int,
        choices=(25_000, 50_000, 100_000),
        default=None,
        help="confirm the completed learning-curve decision (must match it)",
    )
    parser.add_argument(
        "--final-seeds",
        type=_parse_seeds,
        default=FINAL_SEEDS,
        metavar="2026,2027,2028",
    )
    parser.add_argument(
        "--tuning-seed",
        type=int,
        default=2026,
        help="model seed shared by the bounded E8 tuning configurations",
    )
    parser.add_argument(
        "--candidate-artifact",
        type=Path,
        action="append",
        default=None,
        help="single gate-approved candidate artifact for explicit official evaluation",
    )
    parser.add_argument(
        "--promotion-evidence",
        type=Path,
        default=None,
        help=(
            "fm-promotion-evidence-v1 JSON proving the four pre-official gates; "
            "required with --mode official --execute"
        ),
    )
    parser.add_argument(
        "--correctness-evidence-output",
        type=Path,
        default=None,
        help=(
            "structured output for --mode correctness (default: "
            "<output-root>/correctness_evidence.json)"
        ),
    )
    parser.add_argument(
        "--baseline-trajectory-count",
        type=_parse_block_count,
        default=25_000,
        help="E0 full-survivor trajectories (default: 25000)",
    )
    parser.add_argument(
        "--official-trajectory-count", type=_parse_block_count, default=800
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    adoption_run_ids = tuple(dict.fromkeys(args.adopt_valid_run or ()))
    reuse_source_metrics = tuple(
        dict.fromkeys(path.resolve() for path in args.reuse_learning_curve_evidence or ())
    )
    if adoption_run_ids and args.force:
        raise SystemExit("--adopt-valid-run cannot be combined with --force")
    if adoption_run_ids and args.mode in {"aggregate", "correctness"}:
        raise SystemExit(
            "--adopt-valid-run requires an experiment mode containing the run"
        )
    if reuse_source_metrics and args.mode != "learning-curve":
        raise SystemExit(
            "--reuse-learning-curve-evidence requires --mode learning-curve"
        )
    if reuse_source_metrics and args.force:
        raise SystemExit(
            "--reuse-learning-curve-evidence cannot be combined with --force"
        )
    if reuse_source_metrics and adoption_run_ids:
        raise SystemExit(
            "--reuse-learning-curve-evidence cannot be combined with "
            "--adopt-valid-run"
        )
    # Ordinary execution retains canonical interpreter paths and therefore its
    # existing command hashes. Adoption preserves the supplied token so an
    # older manifest that recorded a symlink (for example /usr/local/bin/python3)
    # can still be matched exactly when that same token is supplied explicitly.
    if not adoption_run_ids:
        args.python = args.python.resolve()
    args.catalog = args.catalog.resolve()
    args.output_root = args.output_root.resolve()
    if args.candidate_artifact:
        args.candidate_artifact = [path.resolve() for path in args.candidate_artifact]
    if args.promotion_evidence is not None:
        args.promotion_evidence = args.promotion_evidence.resolve()
    if args.correctness_evidence_output is not None:
        args.correctness_evidence_output = args.correctness_evidence_output.resolve()
    if args.output_root in {APPROACH_ROOT.resolve(), REPOSITORY_ROOT.resolve()}:
        raise SystemExit("--output-root must be a dedicated versioned subdirectory")
    if args.plateau_threshold < 0:
        raise SystemExit("--plateau-threshold must be non-negative")
    fractions = (args.hard_fraction, args.near_fraction, args.random_fraction)
    if min(fractions) < 0 or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise SystemExit(
            "--hard-fraction + --near-fraction + --random-fraction must equal 1"
        )
    if args.learning_rate <= 0 or min(args.latent_l2, args.linear_l2) < 0:
        raise SystemExit(
            "learning rate must be positive and regularization values non-negative"
        )
    if not 0.0 <= args.tie_weight <= 1.0:
        raise SystemExit("--tie-weight must be between 0 and 1")
    if not 0.0 <= args.category_only_weight <= 1.0:
        raise SystemExit("--category-only-weight must be between 0 and 1")
    if args.evidence_saturation <= 0:
        raise SystemExit("--evidence-saturation must be positive")
    if not 0.0 <= args.extended_fraction <= 1.0:
        raise SystemExit("--extended-fraction must be between 0 and 1")
    if args.max_epochs <= 0 or args.patience <= 0:
        raise SystemExit("--max-epochs and --patience must be positive")
    if args.mode == "correctness":
        output_path = (
            args.correctness_evidence_output
            if args.correctness_evidence_output is not None
            else args.output_root / "correctness_evidence.json"
        )
        _assert_safe_output(output_path, args.output_root)
        print(f"Correctness command: {shlex.join(_correctness_command(args.python))}")
        print(f"Correctness evidence: {output_path}")
        if not args.execute:
            print("Dry run only. Re-run with --execute to run the fixed suite.")
            return 0
        report = _run_correctness_suite(args.python)
        _write_json(output_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        if report.get("status") != "passed":
            raise SystemExit("required correctness suite failed")
        return 0
    blockers = stage_prerequisite_errors(args)
    # `all` remains a useful inventory of the safe, pre-selection stages. For
    # gated standalone modes, do not fabricate downstream commands while their
    # prerequisite decision is absent.
    specs = specs_for_mode(args) if not blockers or args.mode == "all" else []
    if adoption_run_ids:
        specs_by_id = {spec.run_id: spec for spec in specs}
        missing_run_ids = [
            run_id for run_id in adoption_run_ids if run_id not in specs_by_id
        ]
        if missing_run_ids:
            raise SystemExit(
                "--adopt-valid-run is not present in the selected mode: "
                + ", ".join(missing_run_ids)
            )
        specs = [specs_by_id[run_id] for run_id in adoption_run_ids]
    try:
        reuse_plan = (
            _plan_learning_curve_evidence(
                specs, reuse_source_metrics, args.output_root
            )
            if reuse_source_metrics
            else {}
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot reuse learning-curve evidence: {error}") from error
    adoption_validation = (
        {
            spec.run_id: _recoverable_training_output_errors(spec) for spec in specs
        }
        if adoption_run_ids
        else {}
    )
    print(
        f"Mode={args.mode} execute={args.execute} force={args.force} "
        f"adopt_only={bool(adoption_run_ids)} "
        f"reuse_evidence={len(reuse_plan)} output_root={args.output_root}"
    )
    print_plan(
        specs,
        force=args.force,
        adopt_only=bool(adoption_run_ids),
        adoption_errors=adoption_validation,
        reuse_evidence={
            run_id: source_path
            for run_id, (_, source_path) in reuse_plan.items()
        },
    )
    for blocker in blockers:
        print(f"BLOCKED: {blocker}")
    if adoption_run_ids:
        if not args.execute:
            print(
                "Dry run only. Re-run with --execute to adopt only the listed "
                "validated output(s); no subprocess will be launched."
            )
            return 0
        if blockers:
            raise SystemExit("; ".join(blockers))
        if not args.catalog.exists():
            raise SystemExit(f"catalog does not exist: {args.catalog}")
        failed_adoptions = {
            run_id: errors
            for run_id, errors in adoption_validation.items()
            if errors
        }
        if failed_adoptions:
            raise SystemExit(
                "; ".join(
                    f"cannot adopt {run_id}: " + "; ".join(errors)
                    for run_id, errors in failed_adoptions.items()
                )
            )
        for spec in specs:
            execute_spec(spec, force=False, adopt_valid_output=True)
            print(f"Adopted validated output: {spec.run_id}")
        print(
            "Adoption complete. Run --mode aggregate --execute separately to "
            "refresh aggregate reports."
        )
        return 0
    if reuse_plan and args.execute:
        if blockers:
            raise SystemExit("; ".join(blockers))
        if not args.catalog.exists():
            raise SystemExit(f"catalog does not exist: {args.catalog}")
        for target, source_path in reuse_plan.values():
            result = _write_learning_curve_evidence(
                target, source_path, args.output_root
            )
            print(f"Recorded {result}: {target.run_id} <- {source_path}")
        print(
            "Evidence reuse complete. No training subprocesses were launched; "
            "run the stage again without --reuse-learning-curve-evidence to continue."
        )
        return 0
    existing_rows = aggregate_metric_rows(args.output_root)
    existing_decision = plateau_decision(
        [row for row in existing_rows if row.get("group") == "E7_learning_curve"],
        args.plateau_threshold,
    )
    print(
        "Existing completed metric rows: "
        f"{len(existing_rows)}; plateau selection: "
        f"{existing_decision.get('selected_trajectory_count')}"
    )
    existing_selected_count = _as_int(
        existing_decision.get("selected_trajectory_count")
    )
    existing_tuning = tuning_decision(
        existing_rows if existing_selected_count is not None else [],
        existing_selected_count,
        trajectory_seed=args.trajectory_seed,
        split_seed=args.split_seed,
        model_seed=args.tuning_seed,
    )
    selected_tuning = existing_tuning.get("selected")
    print(
        "Existing E8 tuning completion: "
        f"{existing_tuning.get('completed_configurations')}/{len(_tuning_grid())}; "
        f"selected: {selected_tuning}"
    )
    if not args.execute:
        print("Dry run only. Re-run with --execute to launch and write outputs.")
        return 0

    if blockers:
        raise SystemExit("; ".join(blockers))

    if not args.catalog.exists() and specs:
        raise SystemExit(f"catalog does not exist: {args.catalog}")
    for spec in specs:
        execute_spec(spec, force=args.force)
    post_rows = aggregate_metric_rows(args.output_root)
    post_decision = plateau_decision(
        [row for row in post_rows if row.get("group") == "E7_learning_curve"],
        args.plateau_threshold,
    )
    selected_count = _as_int(post_decision.get("selected_trajectory_count"))
    summary = write_aggregates(
        args.output_root,
        args.plateau_threshold,
        selected_count,
        tuning_trajectory_seed=args.trajectory_seed,
        tuning_split_seed=args.split_seed,
        tuning_model_seed=args.tuning_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
