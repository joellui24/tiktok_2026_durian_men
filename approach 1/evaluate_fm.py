from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import numbers
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPOSITORY_ROOT / "techjam-conversational-search"
sys.path.insert(0, str(PROJECT_ROOT))

from evaluator import local_evaluator as evaluator  # noqa: E402
from starter.agent import Agent  # noqa: E402
from starter.conversation_features import (  # noqa: E402
    context_feature_names,
    legacy_context_feature_names,
)
from starter.hybrid_model import PortableHybridModel, turn_bucket  # noqa: E402


FULL_SURVIVOR_SCHEMA_VERSION = "1"
FULL_SURVIVOR_PROTOCOL = "exact_full_survivor_ranker_only_v1"
_MISSING = object()


def _field(value: object, *names: str, default: object = _MISSING) -> object:
    """Read the first matching mapping key or object attribute."""

    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    if default is not _MISSING:
        return default
    raise AttributeError(f"none of {names!r} is present on {type(value).__name__}")


def _stable_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _parent_asin(dataset: object, value: object) -> str:
    """Resolve an ASIN from either a product object, string, or product index."""

    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and "parent_asin" in value:
        return str(value["parent_asin"])
    if hasattr(value, "parent_asin"):
        return str(getattr(value, "parent_asin"))
    if isinstance(value, numbers.Integral):
        products = _field(dataset, "products")
        product = products[int(value)]  # type: ignore[index]
        return _parent_asin(dataset, product)
    raise TypeError(f"cannot resolve parent_asin from {value!r}")


def _state_id(state: object, ordinal: int) -> str:
    explicit = _field(state, "state_id", "id", default=None)
    if explicit not in (None, ""):
        return str(explicit)
    trajectory_id = str(_field(state, "trajectory_id", default="trajectory"))
    state_index = _field(
        state, "state_index", "state_ordinal", "turn", default=ordinal
    )
    return f"{trajectory_id}:{state_index}"


def _state_target(dataset: object, state: object) -> str:
    direct = _field(
        state,
        "target_parent_asin",
        "target_asin",
        "parent_asin",
        default=None,
    )
    if direct not in (None, ""):
        return _parent_asin(dataset, direct)
    product_index = _field(
        state, "target_product_index", "product_index", "target_index"
    )
    return _parent_asin(dataset, product_index)


def _state_survivors(
    dataset: object, state: object, state_ordinal: int
) -> tuple[str, ...]:
    provider = getattr(dataset, "state_survivors", None)
    if callable(provider):
        try:
            values = provider(state_ordinal)
        except (TypeError, AttributeError):
            values = provider(state)
    else:
        values = _field(
            state,
            "survivor_parent_asins",
            "surviving_candidates",
            "survivors",
        )
    result = tuple(sorted({_parent_asin(dataset, value) for value in values}))
    if not result:
        raise ValueError("a full-survivor state cannot have an empty survivor set")
    return result


def _state_context_features(
    dataset: object, state: object, model: object | None = None
) -> tuple[str, ...]:
    for method_name in (
        "state_context_features",
        "context_features_for_state",
        "state_feature_names",
    ):
        provider = getattr(dataset, method_name, None)
        if callable(provider):
            return tuple(str(value) for value in provider(state))
    values = _field(
        state,
        "context_features",
        "context_names",
        "feature_names",
        default=None,
    )
    if values is None:
        product_index = int(
            _field(state, "target_product_index", "product_index", "target_index")
        )
        product = _field(dataset, "products")[product_index]  # type: ignore[index]
        category = str(_field(product, "category", "coarse_category"))
        constraints = _field(state, "known_constraints", default=())
        known_constraints = (
            dict(constraints)
            if not isinstance(constraints, Mapping)
            else dict(constraints)
        )
        arguments = {
            "coarse_category": category,
            "scenario_state": str(
                _field(state, "scenario_state", "scenario", default="unknown")
            ),
            "turn": int(_field(state, "turn", default=1)),
            "intent_epoch": int(_field(state, "intent_epoch", default=0)),
            "known_constraints": known_constraints,
        }
        metadata = getattr(model, "metadata", {})
        feature_schema = (
            metadata.get("feature_schema_version")
            if isinstance(metadata, Mapping)
            else None
        )
        builder = (
            context_feature_names
            if feature_schema == "conversation-features-v2"
            else legacy_context_feature_names
        )
        return tuple(builder(**arguments))
    return tuple(str(value) for value in values)


def _state_evidence_sha256(dataset: object, state: object) -> str:
    """Hash model-independent visible state evidence for strict pairing."""

    product_index = _field(
        state, "target_product_index", "product_index", "target_index", default=None
    )
    category = None
    if product_index is not None:
        product = _field(dataset, "products")[int(product_index)]  # type: ignore[index]
        category = str(_field(product, "category", "coarse_category"))
    constraints = _field(state, "known_constraints", default=())
    if isinstance(constraints, Mapping):
        normalized_constraints = [
            (str(attribute), tuple(str(value) for value in values))
            for attribute, values in sorted(constraints.items())
        ]
    else:
        normalized_constraints = [
            (
                str(attribute),
                (str(values),)
                if isinstance(values, str)
                else tuple(str(value) for value in values),
            )
            for attribute, values in constraints
        ]
    payload = {
        "category": category,
        "scenario_state": str(
            _field(state, "scenario_state", "scenario", default="unknown")
        ),
        "turn": int(_field(state, "turn", default=1)),
        "intent_epoch": int(_field(state, "intent_epoch", default=0)),
        "known_constraints": normalized_constraints,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _dataset_states(dataset: object, split: str | None) -> list[object]:
    source = _field(dataset, "states")
    if callable(source):
        try:
            values = list(source(split=split))
        except TypeError:
            values = list(source())
    elif isinstance(source, Mapping):
        values = list(source[split]) if split is not None else [
            state for group in source.values() for state in group
        ]
    else:
        values = list(source)  # type: ignore[arg-type]
    if split is None:
        return values
    return [
        state
        for state in values
        if str(_field(state, "split", default=split)) == split
    ]


def width_bucket(width: int) -> str:
    if width <= 10:
        return "<=10"
    if width <= 50:
        return "11-50"
    if width <= 200:
        return "51-200"
    return ">200"


def supervision_weight_band(weight: float) -> str:
    if weight <= 0.0:
        return "zero"
    if weight <= 0.25:
        return "low"
    if weight <= 0.75:
        return "medium"
    return "high"


def exact_rank(scores: Mapping[str, float], target: str) -> int:
    """Return exact deterministic rank using runtime's score/ASIN tie break."""

    if target not in scores:
        raise ValueError(f"target {target!r} was not scored")
    ordered = sorted(scores, key=lambda parent_asin: (-scores[parent_asin], parent_asin))
    return ordered.index(target) + 1


def _pairwise_accuracy(scores: Mapping[str, float], target: str) -> float | None:
    target_score = scores[target]
    comparisons = [score for parent_asin, score in scores.items() if parent_asin != target]
    if not comparisons:
        return None
    credit = sum(
        1.0 if target_score > score else 0.5 if target_score == score else 0.0
        for score in comparisons
    )
    return credit / len(comparisons)


def score_full_survivor_states(
    dataset: object,
    model: PortableHybridModel | object,
    *,
    states: Sequence[object] | None = None,
    split: str | None = "validation",
    mode: str = "fm",
    model_name: str = "fm",
    seed: int | str | None = None,
) -> list[dict[str, object]]:
    """Score every survivor for immutable held-out trajectory states.

    ``dataset`` is intentionally duck typed. It must expose ``states`` and
    ``state_survivors(state)``; states expose their target, trajectory metadata,
    and context feature names either directly or through a dataset helper.
    """

    all_states = _dataset_states(dataset, None)
    if states is None:
        selected_pairs = [
            (index, state)
            for index, state in enumerate(all_states)
            if split is None or str(_field(state, "split", default=split)) == split
        ]
    else:
        index_by_identity = {id(state): index for index, state in enumerate(all_states)}
        selected_pairs = []
        for state in states:
            index = index_by_identity.get(id(state))
            if index is None:
                raise ValueError("an explicitly supplied state is not in dataset.states")
            selected_pairs.append((index, state))
    metadata = getattr(model, "metadata", {})
    model_seed = seed if seed is not None else (
        metadata.get("seed", "unknown") if isinstance(metadata, Mapping) else "unknown"
    )
    training_scope = (
        metadata.get("training_scope", "legacy_all_products")
        if isinstance(metadata, Mapping)
        else "unknown"
    )
    try:
        category_only_weight = float(
            metadata.get("category_only_weight", 0.05)
            if isinstance(metadata, Mapping)
            else 0.05
        )
    except (TypeError, ValueError):
        category_only_weight = 0.05
    try:
        evidence_saturation = float(
            metadata.get("evidence_saturation", 3)
            if isinstance(metadata, Mapping)
            else 3
        )
    except (TypeError, ValueError):
        evidence_saturation = 3.0
    if not math.isfinite(category_only_weight) or not 0.0 <= category_only_weight <= 1.0:
        category_only_weight = 0.05
    if not math.isfinite(evidence_saturation) or evidence_saturation <= 0.0:
        evidence_saturation = 3.0
    rows: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str]] = set()
    for ordinal, (dataset_state_index, state) in enumerate(selected_pairs):
        state_split = str(_field(state, "split", default=split or "unknown"))
        state_id = _state_id(state, ordinal)
        key = (state_split, state_id)
        if key in seen_keys:
            raise ValueError(f"duplicate full-survivor state key: {key!r}")
        seen_keys.add(key)

        target = _state_target(dataset, state)
        survivors = _state_survivors(dataset, state, dataset_state_index)
        if target not in survivors:
            raise ValueError(
                f"target {target!r} is absent from survivor set for state {state_id!r}"
            )
        context_features = _state_context_features(dataset, state, model)
        scores = model.score_many(survivors, context_features, mode=mode)
        missing = set(survivors) - set(scores)
        extra = set(scores) - set(survivors)
        if missing or extra:
            raise ValueError(
                f"model coverage mismatch for {state_id!r}: "
                f"missing={len(missing)} extra={len(extra)}"
            )
        numeric_scores = {str(key): float(value) for key, value in scores.items()}
        rank = exact_rank(numeric_scores, target)
        width = len(survivors)
        turn = int(_field(state, "turn", default=1))
        scenario_type = str(
            _field(state, "scenario_type", "scenario", default="unknown")
        )
        scenario_state = str(
            _field(state, "scenario_state", default=scenario_type)
        )
        other = bool(
            _field(
                state,
                "has_other_answer",
                "contains_other_answer",
                "other_answer_present",
                default=False,
            )
        )
        explicit_evidence_weight = _field(
            state,
            "supervision_weight",
            "evidence_weight",
            "weight_evidence",
            default=None,
        )
        if explicit_evidence_weight is None:
            constraints = _field(state, "known_constraints", default=())
            if isinstance(constraints, Mapping):
                evidence_count = sum(len(values) for values in constraints.values())
            else:
                evidence_count = sum(
                    len(values) if not isinstance(values, str) else 1
                    for _, values in constraints
                )
            evidence_weight = (
                category_only_weight
                if evidence_count == 0
                else min(1.0, evidence_count / evidence_saturation)
            )
        else:
            evidence_weight = float(explicit_evidence_weight)
        band = str(
            _field(
                state,
                "supervision_weight_band",
                default=supervision_weight_band(evidence_weight),
            )
        )
        generation_seed = _field(
            state,
            "generation_seed",
            default=_field(
                _field(dataset, "config", default=dataset),
                "seed",
                "generation_seed",
                default="unknown",
            ),
        )
        percentile = 0.0 if width == 1 else (rank - 1) / (width - 1)
        rows.append(
            {
                "schema_version": FULL_SURVIVOR_SCHEMA_VERSION,
                "evaluation_protocol": FULL_SURVIVOR_PROTOCOL,
                "model": model_name,
                "mode": mode,
                "seed": str(model_seed),
                "training_scope": str(training_scope),
                "generation_seed": str(generation_seed),
                "split": state_split,
                "state_id": state_id,
                "trajectory_id": str(
                    _field(state, "trajectory_id", default=state_id)
                ),
                "target_parent_asin": target,
                "scenario_type": scenario_type,
                "scenario_state": scenario_state,
                "turn": turn,
                "turn_bucket": str(
                    _field(state, "turn_bucket", default=turn_bucket(turn))
                ),
                "has_other_answer": other,
                "supervision_weight": evidence_weight,
                "supervision_weight_band": band,
                "candidate_width": width,
                "survivor_width_bucket": width_bucket(width),
                "survivor_sha256": _stable_digest(survivors),
                "state_evidence_sha256": _state_evidence_sha256(dataset, state),
                "context_sha256": _stable_digest(context_features),
                "target_rank": rank,
                "reciprocal_rank": 1.0 / rank,
                "hit_at_1": int(rank <= 1),
                "hit_at_5": int(rank <= 5),
                "hit_at_10": int(rank <= 10),
                "rank_percentile": percentile,
                "pairwise_accuracy": _pairwise_accuracy(numeric_scores, target),
            }
        )
    return rows


def _mean(values: Iterable[float | int | None]) -> float | None:
    retained = [float(value) for value in values if value is not None]
    return statistics.fmean(retained) if retained else None


def _median(values: Iterable[float | int | None]) -> float | None:
    retained = [float(value) for value in values if value is not None]
    return float(statistics.median(retained)) if retained else None


def _metric_block(rows: Sequence[Mapping[str, object]], *, macro: bool) -> dict[str, object]:
    if not rows:
        return {
            "aggregation": "trajectory_macro" if macro else "state_micro",
            "mrr": 0.0,
            "hit_rate_at_1": 0.0,
            "hit_rate_at_5": 0.0,
            "hit_rate_at_10": None,
            "hit_rate_at_10_raw": 0.0,
            "hit_rate_at_10_informative": False,
            "hit_rate_at_10_informative_state_count": 0,
            "hit_rate_at_10_informative_trajectory_count": 0,
            "mean_rank": None,
            "median_rank": None,
            "mean_rank_percentile": None,
            "pairwise_accuracy": None,
        }

    metric_keys = {
        "mrr": "reciprocal_rank",
        "hit_rate_at_1": "hit_at_1",
        "hit_rate_at_5": "hit_at_5",
        "mean_rank": "target_rank",
        "mean_rank_percentile": "rank_percentile",
        "pairwise_accuracy": "pairwise_accuracy",
    }
    if macro:
        grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["trajectory_id"])].append(row)
        values = {
            output: _mean(
                _mean(group[raw] for group in groups)  # type: ignore[index]
                for groups in grouped.values()
            )
            for output, raw in metric_keys.items()
        }
        trajectory_medians = [
            _median(group["target_rank"] for group in groups)
            for groups in grouped.values()
        ]
        median_rank = _median(trajectory_medians)
    else:
        values = {
            output: _mean(row[raw] for row in rows)  # type: ignore[index]
            for output, raw in metric_keys.items()
        }
        median_rank = _median(row["target_rank"] for row in rows)
    hit_10_raw = float(_mean(row["hit_at_10"] for row in rows) or 0.0)
    informative_rows = [
        row for row in rows if int(row["candidate_width"]) > 10
    ]
    if macro:
        informative_groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in informative_rows:
            informative_groups[str(row["trajectory_id"])].append(row)
        hit_10 = _mean(
            _mean(row["hit_at_10"] for row in group)
            for group in informative_groups.values()
        )
    else:
        hit_10 = _mean(row["hit_at_10"] for row in informative_rows)
    return {
        "aggregation": "trajectory_macro" if macro else "state_micro",
        **values,
        "median_rank": median_rank,
        "hit_rate_at_10": hit_10,
        "hit_rate_at_10_raw": hit_10_raw,
        "hit_rate_at_10_informative": bool(informative_rows),
        "hit_rate_at_10_informative_state_count": len(informative_rows),
        "hit_rate_at_10_informative_trajectory_count": len(
            {str(row["trajectory_id"]) for row in informative_rows}
        ),
    }


def summarize_full_survivor(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    widths = [int(row["candidate_width"]) for row in rows]
    return {
        "schema_version": FULL_SURVIVOR_SCHEMA_VERSION,
        "evaluation_protocol": FULL_SURVIVOR_PROTOCOL,
        "state_count": len(rows),
        "trajectory_count": len({str(row["trajectory_id"]) for row in rows}),
        "primary": _metric_block(rows, macro=True),
        "state_micro": _metric_block(rows, macro=False),
        "candidate_width": {
            "minimum": min(widths) if widths else None,
            "maximum": max(widths) if widths else None,
            "mean": _mean(widths),
            "median": _median(widths),
        },
    }


def full_survivor_breakdowns(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    dimensions: tuple[tuple[str, Callable[[Mapping[str, object]], str]], ...] = (
        ("scenario", lambda row: str(row["scenario_type"])),
        ("scenario_state", lambda row: str(row["scenario_state"])),
        ("turn_bucket", lambda row: str(row["turn_bucket"])),
        ("survivor_width", lambda row: str(row["survivor_width_bucket"])),
        (
            "other_answer",
            lambda row: "with_other" if bool(row["has_other_answer"]) else "without_other",
        ),
        (
            "supervision_weight",
            lambda row: str(row["supervision_weight_band"]),
        ),
        ("seed", lambda row: str(row["seed"])),
        ("generation_seed", lambda row: str(row["generation_seed"])),
    )
    output: list[dict[str, object]] = []
    for dimension, getter in dimensions:
        grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[getter(row)].append(row)
        for value, group in sorted(grouped.items()):
            output.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "metrics": summarize_full_survivor(group),
                }
            )
    return output


def evaluate_full_survivor(
    dataset: object,
    model: PortableHybridModel | object,
    **kwargs: object,
) -> dict[str, object]:
    rows = score_full_survivor_states(dataset, model, **kwargs)
    return {
        "schema_version": FULL_SURVIVOR_SCHEMA_VERSION,
        "evaluation_protocol": FULL_SURVIVOR_PROTOCOL,
        "overall": summarize_full_survivor(rows),
        "breakdowns": full_survivor_breakdowns(rows),
        "states": rows,
    }


def pair_full_survivor_rows(
    candidate_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
    *,
    candidate_name: str = "candidate",
    baseline_name: str = "baseline",
) -> list[dict[str, object]]:
    """Strictly join identical frozen states and calculate paired deltas."""

    def indexed(
        rows: Sequence[Mapping[str, object]], label: str
    ) -> dict[tuple[str, str], Mapping[str, object]]:
        result: dict[tuple[str, str], Mapping[str, object]] = {}
        for row in rows:
            key = (str(row["split"]), str(row["state_id"]))
            if key in result:
                raise ValueError(f"duplicate {label} state key: {key!r}")
            result[key] = row
        return result

    candidate = indexed(candidate_rows, candidate_name)
    baseline = indexed(baseline_rows, baseline_name)
    if set(candidate) != set(baseline):
        missing_candidate = sorted(set(baseline) - set(candidate))
        missing_baseline = sorted(set(candidate) - set(baseline))
        raise ValueError(
            "paired state keys differ: "
            f"missing_candidate={missing_candidate[:5]!r} "
            f"missing_baseline={missing_baseline[:5]!r}"
        )

    invariant_fields = (
        "trajectory_id",
        "target_parent_asin",
        "scenario_type",
        "scenario_state",
        "turn_bucket",
        "has_other_answer",
        "candidate_width",
        "survivor_width_bucket",
        "survivor_sha256",
        "state_evidence_sha256",
    )
    output: list[dict[str, object]] = []
    for key in sorted(candidate):
        candidate_row = candidate[key]
        baseline_row = baseline[key]
        for field in invariant_fields:
            if candidate_row[field] != baseline_row[field]:
                raise ValueError(
                    f"paired state {key!r} differs on {field}: "
                    f"{candidate_row[field]!r} != {baseline_row[field]!r}"
                )
        candidate_pairwise = candidate_row["pairwise_accuracy"]
        baseline_pairwise = baseline_row["pairwise_accuracy"]
        output.append(
            {
                "comparison": f"{candidate_name}_minus_{baseline_name}",
                "candidate": candidate_name,
                "baseline": baseline_name,
                "candidate_seed": candidate_row["seed"],
                "baseline_seed": baseline_row["seed"],
                "split": key[0],
                "state_id": key[1],
                "trajectory_id": candidate_row["trajectory_id"],
                "scenario_type": candidate_row["scenario_type"],
                "scenario_state": candidate_row["scenario_state"],
                "turn_bucket": candidate_row["turn_bucket"],
                "has_other_answer": candidate_row["has_other_answer"],
                "supervision_weight_band": candidate_row[
                    "supervision_weight_band"
                ],
                "candidate_width": candidate_row["candidate_width"],
                "survivor_width_bucket": candidate_row[
                    "survivor_width_bucket"
                ],
                "candidate_context_sha256": candidate_row["context_sha256"],
                "baseline_context_sha256": baseline_row["context_sha256"],
                "candidate_rank": candidate_row["target_rank"],
                "baseline_rank": baseline_row["target_rank"],
                "rank_improvement": int(baseline_row["target_rank"])
                - int(candidate_row["target_rank"]),
                "mrr_delta": float(candidate_row["reciprocal_rank"])
                - float(baseline_row["reciprocal_rank"]),
                "hit_at_1_delta": int(candidate_row["hit_at_1"])
                - int(baseline_row["hit_at_1"]),
                "hit_at_5_delta": int(candidate_row["hit_at_5"])
                - int(baseline_row["hit_at_5"]),
                "hit_at_10_delta": int(candidate_row["hit_at_10"])
                - int(baseline_row["hit_at_10"]),
                "rank_percentile_improvement": float(
                    baseline_row["rank_percentile"]
                )
                - float(candidate_row["rank_percentile"]),
                "pairwise_accuracy_delta": (
                    None
                    if candidate_pairwise is None or baseline_pairwise is None
                    else float(candidate_pairwise) - float(baseline_pairwise)
                ),
            }
        )
    return output


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_trajectory_bootstrap(
    paired_rows: Sequence[Mapping[str, object]],
    *,
    replicates: int = 10_000,
    seed: int = 2026,
) -> list[dict[str, object]]:
    """Paired cluster bootstrap using trajectories as independent units."""

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not paired_rows:
        return []
    metrics = (
        "mrr_delta",
        "hit_at_1_delta",
        "hit_at_5_delta",
        "hit_at_10_delta",
        "rank_improvement",
        "rank_percentile_improvement",
        "pairwise_accuracy_delta",
    )
    comparison = str(paired_rows[0]["comparison"])
    output: list[dict[str, object]] = []
    for metric in metrics:
        # Hit@10 is meaningful only where more than ten candidates remain.
        # Narrow states are retained in the raw diagnostic but never allowed to
        # inflate the reported estimate or confidence interval.
        metric_rows = (
            [row for row in paired_rows if int(row["candidate_width"]) > 10]
            if metric == "hit_at_10_delta"
            else list(paired_rows)
        )
        raw_rows = list(paired_rows)
        raw_by_trajectory: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in raw_rows:
            raw_by_trajectory[str(row["trajectory_id"])].append(row)
        raw_cluster_values = [
            _mean(row[metric] for row in rows)
            for _, rows in sorted(raw_by_trajectory.items())
        ]
        observed_raw = _mean(raw_cluster_values)
        by_trajectory: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in metric_rows:
            by_trajectory[str(row["trajectory_id"])].append(row)
        trajectory_ids = sorted(by_trajectory)
        cluster_values = {
            trajectory_id: _mean(row[metric] for row in rows)
            for trajectory_id, rows in by_trajectory.items()
        }
        retained_ids = [
            trajectory_id
            for trajectory_id in trajectory_ids
            if cluster_values[trajectory_id] is not None
        ]
        if not retained_ids:
            output.append(
                {
                    "comparison": comparison,
                    "comparison_unit": "trajectory",
                    "metric": metric,
                    "state_count": 0,
                    "trajectory_count": 0,
                    "observed_delta": None,
                    "observed_delta_raw": observed_raw,
                    "bootstrap_replicates": replicates,
                    "bootstrap_seed": seed,
                    "ci_95_lower": None,
                    "ci_95_upper": None,
                    "ci_excludes_zero": None,
                    "informative": False,
                    "wins": 0,
                    "ties": 0,
                    "losses": 0,
                }
            )
            continue
        observed = statistics.fmean(
            float(cluster_values[trajectory_id]) for trajectory_id in retained_ids
        )
        metric_seed = int.from_bytes(
            hashlib.sha256(f"{seed}\0{comparison}\0{metric}".encode()).digest()[:8],
            "big",
        )
        values = np.asarray(
            [float(cluster_values[trajectory_id]) for trajectory_id in retained_ids],
            dtype=np.float64,
        )
        rng = np.random.default_rng(metric_seed)
        draws: list[float] = []
        # Chunked vectorization keeps memory bounded for the 100k-trajectory
        # evaluation while avoiding billions of Python-level RNG iterations.
        chunk_size = max(1, min(256, 2_000_000 // len(values)))
        for start in range(0, replicates, chunk_size):
            count = min(chunk_size, replicates - start)
            indices = rng.integers(
                0, len(values), size=(count, len(values)), dtype=np.int32
            )
            draws.extend(np.mean(values[indices], axis=1).tolist())
        lower = _quantile(draws, 0.025)
        upper = _quantile(draws, 0.975)
        output.append(
            {
                "comparison": comparison,
                "comparison_unit": "trajectory",
                "metric": metric,
                "state_count": len(metric_rows),
                "trajectory_count": len(retained_ids),
                "observed_delta": observed,
                "observed_delta_raw": observed_raw,
                "bootstrap_replicates": replicates,
                "bootstrap_seed": seed,
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "ci_excludes_zero": lower > 0.0 or upper < 0.0,
                "informative": True,
                "wins": sum(float(row[metric]) > 0 for row in metric_rows if row[metric] is not None),
                "ties": sum(float(row[metric]) == 0 for row in metric_rows if row[metric] is not None),
                "losses": sum(float(row[metric]) < 0 for row in metric_rows if row[metric] is not None),
            }
        )
    return output


def paired_full_survivor_breakdowns(
    paired_rows: Sequence[Mapping[str, object]],
    *,
    replicates: int = 10_000,
    seed: int = 2026,
) -> list[dict[str, object]]:
    # Overall paired uncertainty keeps the caller's full replicate budget.
    # Breakdown CIs are diagnostic and numerous, so cap each cell to keep the
    # 25k/50k/100k protocol tractable while preserving cluster resampling.
    effective_replicates = min(replicates, 1_000)
    dimensions: tuple[tuple[str, Callable[[Mapping[str, object]], str]], ...] = (
        ("scenario", lambda row: str(row["scenario_type"])),
        ("scenario_state", lambda row: str(row["scenario_state"])),
        ("turn_bucket", lambda row: str(row["turn_bucket"])),
        ("survivor_width", lambda row: str(row["survivor_width_bucket"])),
        (
            "other_answer",
            lambda row: "with_other" if bool(row["has_other_answer"]) else "without_other",
        ),
        (
            "supervision_weight",
            lambda row: str(row["supervision_weight_band"]),
        ),
        ("candidate_seed", lambda row: str(row["candidate_seed"])),
    )
    output: list[dict[str, object]] = []
    for dimension, getter in dimensions:
        grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in paired_rows:
            grouped[getter(row)].append(row)
        for value, group in sorted(grouped.items()):
            for metric in paired_trajectory_bootstrap(
                group, replicates=effective_replicates, seed=seed
            ):
                output.append(
                    {
                        "dimension": dimension,
                        "value": value,
                        "requested_bootstrap_replicates": replicates,
                        **metric,
                    }
                )
    return output


def scored_summary(sessions: list[dict]) -> dict[str, object]:
    summary = evaluator.metric_summary(sessions)
    accuracy = float(summary["hit_rate_at_10"])
    mrr = float(summary["mrr"])
    mttc = summary["mttc"]
    efficiency = (
        0.0
        if mttc is None
        else max(0.0, min(1.0, (11.0 - float(mttc)) / 10.0))
    )
    return {
        **summary,
        "correct_answers": sum(int(row["hit"]) for row in sessions),
        "accuracy": round(accuracy, 6),
        "efficiency": round(efficiency, 6),
        "technical_score": round(
            0.50 * accuracy + 0.30 * mrr + 0.20 * efficiency, 6
        ),
    }


def legacy_official_result(
    frozen_result: Mapping[str, object],
    *,
    cohort: str,
    intent_override_implemented: bool | None = None,
) -> dict[str, object]:
    """Adapt frozen evaluator output without changing its scoring semantics."""

    sessions = list(frozen_result["sessions"])  # type: ignore[arg-type]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    result: dict[str, object] = {
        **scored_summary(sessions),
        "scenario_metrics": {
            scenario: scored_summary(rows)
            for scenario, rows in sorted(grouped.items())
        },
        "sessions": sessions,
        "cohort": cohort,
    }
    if intent_override_implemented is not None:
        result["intent_override_implemented"] = intent_override_implemented
    return result


def simulate(
    agent: Agent,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    *,
    full_horizon: bool,
) -> tuple[dict, list[dict]]:
    sessions: list[dict] = []
    traces: list[dict] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        session_id = f"fm_{'full' if full_horizon else 'official'}_{sample_id}"
        target = str(sample["ground_truth"]["parent_asin"])
        scenario = str(sample["scenario_type"])
        agent.reset(session_id, sample["user_profile"])
        card, behavior = evaluator.materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = scenario != "intent_override"
        user_message = evaluator.initial_message(
            effective,
            evaluator.coarse_category(categories.get(target, [])),
            disclosed,
        )
        hit_turn: int | None = None
        best_rank: int | None = None
        trace: dict[str, object] = {
            "sample_id": sample_id,
            "scenario_type": scenario,
            "target_parent_asin": target,
        }

        for turn in range(1, evaluator.MAX_TURNS + 1):
            response = agent.respond(session_id, user_message, turn, evaluator.TOP_K)
            ranked = evaluator.normalize_recommendations(
                response.get("recommendations"), catalog_ids
            )
            state = agent._sessions[session_id]
            trace[f"candidate_count_turn_{turn}"] = len(state.surviving_candidates)
            trace[f"recommendation_count_turn_{turn}"] = len(ranked)
            trace[f"ask_attribute_turn_{turn}"] = response.get("ask_attribute") or ""
            trace[f"recommendations_turn_{turn}"] = ";".join(ranked)
            trace[f"target_survives_turn_{turn}"] = target in state.surviving_candidates
            trace[f"intent_epoch_turn_{turn}"] = state.intent_epoch

            if hit_turn is None and override_applied and target in ranked:
                hit_turn = turn
                best_rank = ranked.index(target) + 1
                if not full_horizon:
                    break
            if turn == evaluator.MAX_TURNS:
                break

            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(
                    override.get(
                        "message", "Actually, please ignore my earlier preference."
                    )
                )
            else:
                user_message, boundary_used = evaluator.customer_reply(
                    effective,
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )

        session = {
            "sample_id": sample_id,
            "scenario_type": scenario,
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        }
        sessions.append(session)
        final_count = int(trace.get("candidate_count_turn_10", len(state.surviving_candidates)))
        trace.update(
            {
                "hit": hit_turn is not None,
                "first_hit_turn": hit_turn or "",
                "best_rank": best_rank or "",
                "actually_reached_turn_10": hit_turn is None or hit_turn == 10,
                "surviving_parent_asins_turn_10": final_count,
                "exactly_10_survivors_turn_10": final_count == 10,
                "at_most_10_survivors_turn_10": final_count <= 10,
            }
        )
        traces.append(trace)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in sessions:
        grouped[str(row["scenario_type"])].append(row)
    overall = scored_summary(sessions)
    return (
        {
            **overall,
            "scenario_metrics": {
                scenario: scored_summary(rows)
                for scenario, rows in sorted(grouped.items())
            },
            "sessions": sessions,
        },
        traces,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ablation_rows(
    samples: list[dict],
    catalog_path: Path,
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    model_specs: Sequence[tuple[str, Path, str]],
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    session_rows: list[dict] = []
    for model_name, model_path, mode in model_specs:
        with Agent(
            catalog_path,
            model_path=model_path,
            ranking_mode=mode,
        ) as agent:
            if agent.model is None:
                raise RuntimeError(agent.model_error or f"{mode} model is unavailable")
            frozen_result = evaluator.evaluate(
                agent, samples, catalog_ids, categories, products
            )
            result = legacy_official_result(
                frozen_result, cohort="official_public_200"
            )
        for session in result["sessions"]:
            hit_turn = session["first_hit_turn"]
            efficiency = (
                0.0
                if hit_turn is None
                else max(0.0, min(1.0, (11.0 - float(hit_turn)) / 10.0))
            )
            session_rows.append(
                {
                    "model": model_name,
                    **session,
                    "efficiency": efficiency,
                    "technical_score_contribution": (
                        0.50 * int(session["hit"])
                        + 0.30 * float(session["reciprocal_rank"])
                        + 0.20 * efficiency
                    ),
                }
            )
        for scenario, summary in [("overall", result), *result["scenario_metrics"].items()]:
            rows.append(
                {
                    "model": model_name,
                    "scenario": scenario,
                    "sample_count": summary["sample_count"],
                    "correct_answers": summary["correct_answers"],
                    "accuracy": summary["accuracy"],
                    "mrr": summary["mrr"],
                    "mttc": summary["mttc"],
                    "efficiency": summary["efficiency"],
                    "technical_score": summary["technical_score"],
                }
            )
    return rows, session_rows


def bootstrap_rows(session_rows: list[dict], replicates: int = 10_000) -> list[dict]:
    by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    model_names = list(dict.fromkeys(str(row["model"]) for row in session_rows))
    for row in session_rows:
        by_model[str(row["model"])][str(row["sample_id"])] = row
    comparisons = tuple(
        (model_names[right], model_names[left])
        for right in range(1, len(model_names))
        for left in range(right)
    )
    metrics = {
        "accuracy": lambda row: float(bool(row["hit"])),
        "mrr": lambda row: float(row["reciprocal_rank"]),
        "efficiency": lambda row: float(row["efficiency"]),
        "technical_score": lambda row: float(row["technical_score_contribution"]),
    }
    output: list[dict] = []
    for candidate, baseline in comparisons:
        candidate_ids = set(by_model[candidate])
        baseline_ids = set(by_model[baseline])
        if candidate_ids != baseline_ids:
            raise ValueError(
                f"official paired keys differ for {candidate} and {baseline}"
            )
        sample_ids = sorted(candidate_ids)
        for sample_id in sample_ids:
            if (
                by_model[candidate][sample_id]["scenario_type"]
                != by_model[baseline][sample_id]["scenario_type"]
            ):
                raise ValueError(f"scenario mismatch for sample {sample_id!r}")
        for metric, getter in metrics.items():
            deltas = [
                getter(by_model[candidate][sample_id])
                - getter(by_model[baseline][sample_id])
                for sample_id in sample_ids
            ]
            observed = sum(deltas) / len(deltas)
            metric_seed = int.from_bytes(
                hashlib.sha256(
                    f"2026\0{candidate}\0{baseline}\0{metric}".encode()
                ).digest()[:8],
                "big",
            )
            rng = random.Random(metric_seed)
            draws = sorted(
                sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
                for _ in range(replicates)
            )
            lower = _quantile(draws, 0.025)
            upper = _quantile(draws, 0.975)
            output.append(
                {
                    "comparison": f"{candidate}_minus_{baseline}",
                    "metric": metric,
                    "sample_count": len(deltas),
                    "observed_delta": observed,
                    "bootstrap_replicates": replicates,
                    "ci_95_lower": lower,
                    "ci_95_upper": upper,
                    "ci_excludes_zero": lower > 0 or upper < 0,
                }
            )
    return output


def _load_python_module(path: Path) -> object:
    spec = importlib.util.spec_from_file_location("approach1_trajectory_data", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load trajectory module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def generate_fixed_heldout_dataset(
    module: object,
    *,
    catalog_path: Path,
    trajectory_count: int,
    seed: int,
    split_seed: int | None = None,
    scenario_mix: str = "public",
) -> object:
    """Call a trajectory module factory without coupling evaluation to its internals."""

    load_products = getattr(module, "load_products", None)
    config_class = getattr(module, "TrajectoryConfig", None)
    canonical_factory = getattr(module, "generate_trajectory_dataset", None)
    if callable(load_products) and callable(config_class) and callable(canonical_factory):
        products, raw_products = load_products(catalog_path)
        effective_split_seed = seed if split_seed is None else split_seed
        config = config_class(
            trajectory_count=trajectory_count,
            seed=seed,
            split_seed=effective_split_seed,
            scenario_mix=scenario_mix,
            max_turns=10,
            extended_fraction=0.10,
        )
        build_splits = getattr(module, "build_product_splits", None)
        product_splits = (
            build_splits(products, effective_split_seed)
            if callable(build_splits)
            else None
        )
        return canonical_factory(
            products,
            raw_products,
            config,
            product_splits=product_splits,
        )

    factory_names = (
        "generate_evaluation_dataset",
        "generate_trajectory_dataset",
        "build_trajectory_dataset",
        "generate_dataset",
    )
    factory = next(
        (
            getattr(module, name)
            for name in factory_names
            if callable(getattr(module, name, None))
        ),
        None,
    )
    if factory is None:
        dataset_class = getattr(module, "TrajectoryDataset", None)
        class_factory = getattr(dataset_class, "generate", None)
        if callable(class_factory):
            factory = class_factory
    if factory is None:
        raise AttributeError(
            "trajectory_data.py must expose generate_evaluation_dataset, "
            "generate_trajectory_dataset, build_trajectory_dataset, generate_dataset, "
            "or TrajectoryDataset.generate"
        )

    available: dict[str, object] = {
        "catalog": catalog_path,
        "catalog_path": catalog_path,
        "dataset_size": trajectory_count,
        "trajectory_count": trajectory_count,
        "num_trajectories": trajectory_count,
        "count": trajectory_count,
        "seed": seed,
        "global_seed": seed,
        "split_seed": seed if split_seed is None else split_seed,
        "scenario_mix": scenario_mix,
    }
    signature = inspect.signature(factory)
    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs = available if accepts_kwargs else {
        name: available[name] for name in parameters if name in available
    }
    missing = [
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and name not in kwargs
    ]
    if missing:
        raise TypeError(
            f"unsupported required trajectory factory parameters: {', '.join(missing)}"
        )
    return factory(**kwargs)


def write_full_survivor_outputs(
    dataset: object,
    *,
    model_specs: Sequence[tuple[str, Path, str]],
    output_dir: Path,
    split: str,
    bootstrap_replicates: int,
    dataset_manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    state_rows: dict[str, list[dict[str, object]]] = {}
    reports: dict[str, dict[str, object]] = {}
    for model_name, model_path, mode in model_specs:
        model = PortableHybridModel(model_path)
        report = evaluate_full_survivor(
            dataset,
            model,
            split=split,
            mode=mode,
            model_name=model_name,
        )
        rows = list(report.pop("states"))  # type: ignore[arg-type]
        report["artifact"] = str(model_path)
        report["artifact_sha256"] = _file_sha256(model_path)
        report["training_scope"] = model.metadata.get(
            "training_scope", "legacy_all_products"
        )
        report["feature_schema_version"] = model.metadata.get(
            "feature_schema_version", "legacy"
        )
        state_rows[model_name] = rows
        reports[model_name] = report
        write_csv(output_dir / f"{model_name}_states.csv", rows)
        (output_dir / f"{model_name}_metrics.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    model_names = [name for name, _, _ in model_specs]
    comparison_pairs = tuple(
        (model_names[right], model_names[left])
        for right in range(1, len(model_names))
        for left in range(right)
    )
    paired_rows: list[dict[str, object]] = []
    paired_bootstrap: list[dict[str, object]] = []
    paired_breakdowns: list[dict[str, object]] = []
    for candidate, baseline in comparison_pairs:
        if candidate not in state_rows or baseline not in state_rows:
            continue
        paired = pair_full_survivor_rows(
            state_rows[candidate],
            state_rows[baseline],
            candidate_name=candidate,
            baseline_name=baseline,
        )
        paired_rows.extend(paired)
        paired_bootstrap.extend(
            paired_trajectory_bootstrap(
                paired, replicates=bootstrap_replicates
            )
        )
        paired_breakdowns.extend(
            paired_full_survivor_breakdowns(
                paired, replicates=bootstrap_replicates
            )
        )
    write_csv(output_dir / "paired_state_deltas.csv", paired_rows)
    write_csv(output_dir / "paired_trajectory_bootstrap.csv", paired_bootstrap)
    write_csv(output_dir / "paired_breakdowns.csv", paired_breakdowns)
    reference_rows = state_rows[model_names[0]] if model_names else []
    cohort_fields = (
        "split",
        "state_id",
        "trajectory_id",
        "target_parent_asin",
        "scenario_type",
        "scenario_state",
        "turn",
        "turn_bucket",
        "has_other_answer",
        "candidate_width",
        "survivor_sha256",
        "state_evidence_sha256",
    )
    cohort_states = [
        {field: row.get(field) for field in cohort_fields}
        for row in sorted(
            reference_rows,
            key=lambda row: (str(row.get("split")), str(row.get("state_id"))),
        )
    ]
    manifest_payload = dict(dataset_manifest or {})
    cohort = {
        "split": split,
        "state_count": len(cohort_states),
        "trajectory_count": len(
            {str(row.get("trajectory_id")) for row in cohort_states}
        ),
        "state_cohort_sha256": _json_sha256(cohort_states),
        "dataset_manifest": manifest_payload,
        "dataset_manifest_sha256": _json_sha256(manifest_payload),
    }
    cohort["cohort_sha256"] = _json_sha256(cohort)
    manifest = {
        "schema_version": FULL_SURVIVOR_SCHEMA_VERSION,
        "evaluation_protocol": FULL_SURVIVOR_PROTOCOL,
        "split": split,
        "cohort": cohort,
        "models": reports,
        "paired_comparisons": paired_bootstrap,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _evaluate_official_public(
    args: argparse.Namespace,
    evaluation_model_specs: Sequence[tuple[str, Path, str]],
) -> dict[str, object]:
    """Run and write the public-session evaluation suite.

    Keeping this path separate makes ``--skip-official`` a hard boundary: the
    caller can run ranker-only evaluation without even reading the public set.
    """

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = evaluator.load_jsonl(args.dataset)
    supported = [
        row for row in samples if row.get("scenario_type") != "intent_override"
    ]
    catalog_ids, categories, products = evaluator.catalog_index(args.catalog)

    with Agent(
        args.catalog,
        model_path=args.hybrid_model,
        ranking_mode=args.third_model_mode,
    ) as agent:
        if agent.model is None:
            raise RuntimeError(agent.model_error or "hybrid model is unavailable")
        frozen_supported = evaluator.evaluate(
            agent, supported, catalog_ids, categories, products
        )
        frozen_official = evaluator.evaluate(
            agent, samples, catalog_ids, categories, products
        )
        _, supported_traces = simulate(
            agent,
            supported,
            catalog_ids,
            categories,
            products,
            full_horizon=False,
        )
        _, official_traces = simulate(
            agent,
            samples,
            catalog_ids,
            categories,
            products,
            full_horizon=False,
        )
        full_result, full_traces = simulate(
            agent,
            samples,
            catalog_ids,
            categories,
            products,
            full_horizon=True,
        )

    supported_result = legacy_official_result(
        frozen_supported, cohort="buying_browsing_boundary_170"
    )
    official_result = legacy_official_result(
        frozen_official,
        cohort="official_public_200",
        intent_override_implemented=True,
    )
    full_result["cohort"] = "official_public_200_full_horizon_diagnostic"
    for name, payload in (
        ("non_override_170.json", supported_result),
        ("official_200.json", official_result),
        ("full_horizon_200.json", full_result),
    ):
        (args.output_dir / name).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    write_csv(args.output_dir / "non_override_170.csv", supported_traces)
    write_csv(args.output_dir / "official_200.csv", official_traces)
    write_csv(args.output_dir / "full_horizon_200.csv", full_traces)

    ablations, ablation_sessions = ablation_rows(
        samples,
        args.catalog,
        catalog_ids,
        categories,
        products,
        evaluation_model_specs,
    )
    write_csv(args.output_dir / "model_ablation.csv", ablations)
    write_csv(args.output_dir / "model_ablation_sessions.csv", ablation_sessions)
    write_csv(
        args.output_dir / "model_ablation_bootstrap.csv",
        bootstrap_rows(ablation_sessions),
    )

    return {
        "non_override_170": {
            key: value
            for key, value in supported_result.items()
            if key not in {"sessions", "scenario_metrics"}
        },
        "official_200": {
            key: value
            for key, value in official_result.items()
            if key not in {"sessions", "scenario_metrics"}
        },
        "full_horizon_turn_10": {
            "exactly_10": sum(
                row["exactly_10_survivors_turn_10"] is True for row in full_traces
            ),
            "at_most_10": sum(
                row["at_most_10_survivors_turn_10"] is True for row in full_traces
            ),
            "more_than_10": sum(
                int(row["surviving_parent_asins_turn_10"]) > 10
                for row in full_traces
            ),
            "naturally_reached_turn_10": sum(
                row["actually_reached_turn_10"] is True for row in full_traces
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Approach 1 FM")
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data/catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/public_set.jsonl")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results") / "redesign" / "evaluation_v2",
    )
    parser.add_argument(
        "--linear-model", type=Path, default=Path(__file__).with_name("linear_model.sqlite3")
    )
    parser.add_argument(
        "--fm-model", type=Path, default=Path(__file__).with_name("fm_only_model.sqlite3")
    )
    parser.add_argument(
        "--hybrid-model", type=Path, default=Path(__file__).with_name("fm_model.sqlite3")
    )
    parser.add_argument(
        "--third-model-name",
        default="hybrid",
        help="label for --hybrid-model in reports (for example, candidate)",
    )
    parser.add_argument(
        "--third-model-mode",
        choices=("fm", "hybrid"),
        default="hybrid",
        help="scoring mode for --hybrid-model",
    )
    parser.add_argument(
        "--trajectory-module",
        type=Path,
        default=Path(__file__).with_name("trajectory_data.py"),
    )
    parser.add_argument("--trajectory-count", type=int, default=800)
    parser.add_argument("--trajectory-seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument(
        "--scenario-mix", choices=("public", "balanced"), default="public"
    )
    parser.add_argument(
        "--full-survivor-split",
        choices=("train", "validation", "test"),
        default="validation",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument(
        "--redesign-output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--skip-full-survivor", action="store_true")
    parser.add_argument(
        "--skip-official",
        action="store_true",
        help=(
            "skip public-session evaluation and write only ranker-only "
            "full-survivor outputs"
        ),
    )
    args = parser.parse_args(argv)
    if args.skip_official and args.skip_full_survivor:
        parser.error(
            "--skip-official and --skip-full-survivor cannot be used together "
            "because they skip all evaluation work"
        )
    if args.third_model_name in {"linear", "fm"}:
        parser.error("--third-model-name must differ from linear and fm")
    if not args.third_model_name.replace("_", "").replace("-", "").isalnum():
        parser.error("--third-model-name may contain only letters, digits, _ and -")

    evaluation_model_specs = (
        ("linear", args.linear_model, "linear"),
        ("fm", args.fm_model, "fm"),
        (args.third_model_name, args.hybrid_model, args.third_model_mode),
    )
    summary = {
        "evaluated_third_model": {
            "name": args.third_model_name,
            "mode": args.third_model_mode,
            "artifact": str(args.hybrid_model),
        },
    }
    if not args.skip_official:
        summary.update(_evaluate_official_public(args, evaluation_model_specs))
    if not args.skip_full_survivor:
        if not args.trajectory_module.exists():
            summary["full_survivor"] = {
                "status": "skipped",
                "reason": f"trajectory module not found: {args.trajectory_module}",
            }
        else:
            trajectory_module = _load_python_module(args.trajectory_module)
            trajectory_dataset = generate_fixed_heldout_dataset(
                trajectory_module,
                catalog_path=args.catalog,
                trajectory_count=args.trajectory_count,
                seed=args.trajectory_seed,
                split_seed=args.split_seed,
                scenario_mix=args.scenario_mix,
            )
            redesign_output = (
                args.redesign_output_dir
                if args.redesign_output_dir is not None
                else args.output_dir / "ranker_only"
            )
            redesign_output.mkdir(parents=True, exist_ok=True)
            manifest_builder = getattr(trajectory_module, "manifest", None)
            trajectory_manifest = None
            if callable(manifest_builder):
                trajectory_manifest = manifest_builder(trajectory_dataset)
                (redesign_output / "trajectory_manifest.json").write_text(
                    json.dumps(trajectory_manifest, indent=2) + "\n",
                    encoding="utf-8",
                )
            redesign_summary = write_full_survivor_outputs(
                trajectory_dataset,
                model_specs=evaluation_model_specs,
                output_dir=redesign_output,
                split=args.full_survivor_split,
                bootstrap_replicates=args.bootstrap_replicates,
                dataset_manifest=trajectory_manifest,
            )
            summary["full_survivor"] = {
                "status": "completed",
                "evaluation_protocol": FULL_SURVIVOR_PROTOCOL,
                "output_directory": str(redesign_output),
                "split": args.full_survivor_split,
                "trajectory_count": args.trajectory_count,
                "models": {
                    name: report["overall"]
                    for name, report in redesign_summary["models"].items()
                },
            }
    if not args.skip_official:
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
