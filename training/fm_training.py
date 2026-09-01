from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sqlite3
import struct
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPOSITORY_ROOT
MODULE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MODULE_ROOT))

import trajectory_data as trajectory  # noqa: E402
from src.conversation_features import (  # noqa: E402
    constraint_context_features,
    context_feature_names,
    legacy_context_feature_names,
    normalize_constraint,
)
from src.hybrid_model import MODEL_SCHEMA_VERSION, RARE_VALUE, turn_bucket  # noqa: E402


Product = trajectory.Product
State = trajectory.TrajectoryState
stable_int = trajectory.stable_int
split_for = trajectory.split_for
price_bucket = trajectory.price_bucket
rating_bucket = trajectory.rating_bucket
popularity_bucket = trajectory.popularity_bucket
load_products = trajectory.load_products

ATTRIBUTES = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
)
SCENARIO_STATES = (
    "buying",
    "exploring_unknown",
    "browsing",
    "boundary",
    "provisional_override",
    "intent_override",
    "unknown",
)
TURN_BUCKETS = ("early", "middle", "late")

# Compatibility names remain importable, but all command-line values are now
# explicit configuration rather than hidden training constants.
SEED = 2026
DIMENSION = 16
MIN_VALUE_SUPPORT = 5
MIN_CROSS_SUPPORT = 20
NEGATIVES_PER_STATE = 16
FM_L2 = 1e-5
LINEAR_L2 = 1e-5
CROSS_L2 = 1e-4
LEARNING_RATE = 0.01
NEGATIVE_MODES = (
    "product_fixed",
    "survivor_static",
    "survivor_dynamic",
)


@dataclass(frozen=True)
class TrainingConfig:
    schema_version: str = "fm-training-v2"
    seed: int = SEED
    dimension: int = DIMENSION
    minimum_value_support: int = MIN_VALUE_SUPPORT
    minimum_cross_support: int = MIN_CROSS_SUPPORT
    negatives_per_state: int = NEGATIVES_PER_STATE
    negative_pre_pool_size: int = 128
    hard_fraction: float = 0.50
    near_fraction: float = 0.25
    random_fraction: float = 0.25
    other_encoding: str = "dual"
    supervision_policy: str = "set_valued_positives"
    tie_weight: float = 0.10
    category_only_weight: float = 0.05
    evidence_saturation: int = 3
    learning_rate: float = LEARNING_RATE
    latent_l2: float = FM_L2
    linear_l2: float = LINEAR_L2
    cross_l2: float = CROSS_L2
    max_epochs: int = 60
    patience: int = 7
    validation_interval: int = 1
    pair_batch_size: int = 65_536
    negative_mode: str = "survivor_dynamic"

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.negatives_per_state <= 0:
            raise ValueError("negatives_per_state must be positive")
        if self.negative_pre_pool_size < self.negatives_per_state:
            raise ValueError("negative_pre_pool_size must cover negatives_per_state")
        if self.negative_mode not in NEGATIVE_MODES:
            raise ValueError(
                "negative_mode must be product_fixed, survivor_static, "
                "or survivor_dynamic"
            )
        if self.supervision_policy not in {
            "skip_ties",
            "downweight_ties",
            "set_valued_positives",
        }:
            raise ValueError("unsupported supervision policy")
        if self.other_encoding not in {"legacy", "dual"}:
            raise ValueError("other_encoding must be legacy or dual")
        if not 0.0 <= self.tie_weight <= 1.0:
            raise ValueError("tie_weight must be between zero and one")
        if min(self.hard_fraction, self.near_fraction, self.random_fraction) < 0.0:
            raise ValueError("negative sampler fractions must be non-negative")
        mixture = self.hard_fraction + self.near_fraction + self.random_fraction
        if not math.isclose(mixture, 1.0, abs_tol=1e-9):
            raise ValueError("negative sampler fractions must sum to one")


@dataclass
class FeatureData:
    products: Sequence[Product]
    dataset: object
    context_names: list[str]
    item_names: list[str]
    context_fields: list[str]
    item_fields: list[str]
    context_name_to_id: dict[str, int]
    item_name_to_id: dict[str, int]
    product_item_ids: list[tuple[int, ...]]
    product_item_sets: list[frozenset[int]]
    product_evidence: list[frozenset[tuple[str, str]]]
    item_matrix: sparse.csr_matrix
    state_context_ids: list[tuple[int, ...]]
    state_matrix: sparse.csr_matrix
    positives: np.ndarray
    trajectory_state_weights: np.ndarray
    state_splits: np.ndarray
    product_splits: np.ndarray


@dataclass
class Parameters:
    context_vectors: np.ndarray
    item_vectors: np.ndarray
    item_linear: np.ndarray
    cross_weights: np.ndarray


@dataclass
class PairBatch:
    state_indices: np.ndarray
    positives: np.ndarray
    negatives: np.ndarray
    trajectory_state_weights: np.ndarray
    evidence_weights: np.ndarray
    sampling_weights: np.ndarray
    ambiguity_weights: np.ndarray
    effective_weights: np.ndarray
    sampler_types: np.ndarray
    diagnostics: dict[str, object]
    audit_rows: list[dict[str, object]]

    def __len__(self) -> int:
        return int(len(self.state_indices))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def field_from_name(name: str) -> str:
    return name.split(":", 1)[1].split("=", 1)[0]


def _state_value(state: object, *names: str, default: object = None) -> object:
    for name in names:
        if hasattr(state, name):
            return getattr(state, name)
    return default


def _state_target(state: object) -> int:
    value = _state_value(state, "product_index", "target_index")
    if value is None:
        raise AttributeError("trajectory state has no target product index")
    return int(value)


def _state_trajectory_id(state: object) -> str:
    return str(_state_value(state, "trajectory_id", default=""))


def _state_local_index(state: object) -> int:
    return int(_state_value(state, "state_index", "trajectory_state_index", default=0))


def _state_split(state: object) -> str:
    return str(_state_value(state, "split", default="train"))


def _state_scenario(state: object) -> str:
    return str(_state_value(state, "scenario_state", "scenario", default="unknown"))


def _state_scenario_type(state: object) -> str:
    return str(_state_value(state, "scenario_type", "scenario", default="unknown"))


def _state_turn(state: object) -> int:
    return int(_state_value(state, "turn", default=1))


def _state_intent_epoch(state: object) -> int:
    explicit = _state_value(state, "intent_epoch")
    if explicit is not None:
        return int(explicit)
    return int(str(_state_value(state, "override", default="pre")) == "post")


def _state_known_constraints(state: object) -> tuple[tuple[str, str], ...]:
    raw = _state_value(state, "known_constraints", "visible_constraints", default=())
    if isinstance(raw, Mapping):
        return tuple(
            (str(attribute), str(value))
            for attribute, values in sorted(raw.items())
            for value in values
        )
    result: list[tuple[str, str]] = []
    for attribute, values in raw:
        if isinstance(values, str):
            result.append((str(attribute), values))
        else:
            result.extend((str(attribute), str(value)) for value in values)
    return tuple(result)


def _known_constraint_mapping(state: object) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for attribute, value in _state_known_constraints(state):
        if value not in result[attribute]:
            result[attribute].append(value)
    return dict(result)


def _state_has_other(state: object) -> bool:
    explicit = _state_value(state, "has_other_answer", "has_other")
    if explicit is not None:
        return bool(explicit)
    return any(attribute == "other" for attribute, _ in _state_known_constraints(state))


def _dataset_survivors(dataset: object, state_index: int) -> np.ndarray:
    values = dataset.state_survivors(state_index)
    array = np.asarray(values, dtype=np.int32)
    if array.ndim != 1:
        raise ValueError("state survivors must be one-dimensional")
    return array


def _split_lookup(
    products: Sequence[Product], product_splits: Mapping[object, str] | Sequence[str] | None
) -> list[str]:
    if product_splits is None:
        return [split_for(product.parent_asin) for product in products]
    if isinstance(product_splits, Mapping):
        return [
            str(
                product_splits.get(index, product_splits.get(product.parent_asin, "train"))
            )
            for index, product in enumerate(products)
        ]
    if len(product_splits) != len(products):
        raise ValueError("product split vector length does not match products")
    return [str(value) for value in product_splits]


def _constraint_support(
    products: Sequence[Product], splits: Sequence[str]
) -> tuple[Counter[tuple[str, str]], Counter[str]]:
    support: Counter[tuple[str, str]] = Counter()
    brand_support: Counter[str] = Counter()
    for product, split in zip(products, splits, strict=True):
        if split != "train":
            continue
        values: set[tuple[str, str]] = set()
        values.add(("category", normalize_constraint(product.category)))
        for attribute, normalized, _ in product.constraints:
            values.add((attribute, normalized))
            values.add(("other", normalized))
        support.update(values)
        if product.brand:
            brand_support[product.brand] += 1
    return support, brand_support


def mapped_value(
    attribute: str,
    value: str,
    support: Counter[tuple[str, str]],
    minimum_support: int = MIN_VALUE_SUPPORT,
) -> str:
    return value if support[(attribute, value)] >= minimum_support else RARE_VALUE


def _map_context_name(
    name: str,
    support: Counter[tuple[str, str]],
    minimum_support: int,
) -> str:
    if "=" not in name:
        return name
    field = field_from_name(name)
    if field not in ATTRIBUTES:
        return name
    value = name.split("=", 1)[1]
    mapped = mapped_value(field, value, support, minimum_support)
    return f"ctx:{field}={mapped}"


def build_feature_data(
    products: Sequence[Product],
    dataset: object,
    product_splits: Mapping[object, str] | Sequence[str] | None = None,
    *,
    minimum_value_support: int = MIN_VALUE_SUPPORT,
    other_encoding: str = "dual",
) -> FeatureData:
    """Fit the sparse vocabulary on training products and encode all states.

    Validation/test product values are transformed through training support;
    values not supported by the training split use typed rare fallbacks.
    """

    if other_encoding not in {"legacy", "dual"}:
        raise ValueError("other_encoding must be legacy or dual")
    splits = _split_lookup(products, product_splits)
    support, brand_support = _constraint_support(products, splits)

    context_names_set: set[str] = {
        *(f"ctx:scenario={value}" for value in SCENARIO_STATES),
        *(f"ctx:turn={value}" for value in TURN_BUCKETS),
        "ctx:override=pre",
        "ctx:override=post",
    }
    if other_encoding == "dual":
        context_names_set.add("ctx:answer_source=other")
    item_names_set: set[str] = set()
    for product in products:
        normalized_category = mapped_value(
            "category",
            normalize_constraint(product.category),
            support,
            minimum_value_support,
        )
        context_names_set.add(f"ctx:category={normalized_category}")
        item_names_set.add(f"item:category={normalized_category}")
        for attribute, value, raw in product.constraints:
            other_context_names = (
                constraint_context_features("other", raw)
                if other_encoding == "dual"
                else (f"ctx:other={value}",)
            )
            for context_name in (f"ctx:{attribute}={value}", *other_context_names):
                context_names_set.add(
                    _map_context_name(context_name, support, minimum_value_support)
                )
            for field in {attribute, "other"}:
                mapped = mapped_value(field, value, support, minimum_value_support)
                item_names_set.add(f"item:{field}={mapped}")
        brand = (
            product.brand
            if product.brand
            and brand_support[product.brand] >= minimum_value_support
            else RARE_VALUE
        )
        item_names_set.update(
            {
                f"item:brand={brand}",
                f"item:price={product.price_bucket}",
                f"item:rating={product.rating_bucket}",
                f"item:popularity={product.popularity_bucket}",
            }
        )

    for attribute in ATTRIBUTES:
        context_names_set.add(f"ctx:{attribute}={RARE_VALUE}")
        item_names_set.add(f"item:{attribute}={RARE_VALUE}")

    context_names = sorted(context_names_set)
    item_names = sorted(item_names_set)
    context_name_to_id = {name: index for index, name in enumerate(context_names)}
    item_name_to_id = {name: index for index, name in enumerate(item_names)}

    product_item_ids: list[tuple[int, ...]] = []
    product_item_sets: list[frozenset[int]] = []
    product_evidence: list[frozenset[tuple[str, str]]] = []
    item_rows: list[int] = []
    item_columns: list[int] = []
    for product_index, product in enumerate(products):
        normalized_category = mapped_value(
            "category",
            normalize_constraint(product.category),
            support,
            minimum_value_support,
        )
        names = {
            f"item:category={normalized_category}",
            f"item:price={product.price_bucket}",
            f"item:rating={product.rating_bucket}",
            f"item:popularity={product.popularity_bucket}",
        }
        brand = (
            product.brand
            if product.brand
            and brand_support[product.brand] >= minimum_value_support
            else RARE_VALUE
        )
        names.add(f"item:brand={brand}")
        evidence: set[tuple[str, str]] = set()
        for attribute, value, _ in product.constraints:
            evidence.add((attribute, value))
            evidence.add(("other", value))
            for field in {attribute, "other"}:
                names.add(
                    f"item:{field}="
                    f"{mapped_value(field, value, support, minimum_value_support)}"
                )
        identifiers = tuple(sorted(item_name_to_id[name] for name in names))
        product_item_ids.append(identifiers)
        product_item_sets.append(frozenset(identifiers))
        product_evidence.append(frozenset(evidence))
        item_rows.extend([product_index] * len(identifiers))
        item_columns.extend(identifiers)

    item_matrix = sparse.csr_matrix(
        (
            np.ones(len(item_rows), dtype=np.float32),
            (np.asarray(item_rows), np.asarray(item_columns)),
        ),
        shape=(len(products), len(item_names)),
        dtype=np.float32,
    )

    states = list(dataset.states)
    state_context_ids: list[tuple[int, ...]] = []
    state_rows: list[int] = []
    state_columns: list[int] = []
    trajectory_counts = Counter(_state_trajectory_id(state) for state in states)
    weights: list[float] = []
    for state_index, state in enumerate(states):
        product = products[_state_target(state)]
        context_builder = (
            context_feature_names
            if other_encoding == "dual"
            else legacy_context_feature_names
        )
        names = context_builder(
            coarse_category=product.category,
            scenario_state=_state_scenario(state),
            turn=_state_turn(state),
            intent_epoch=_state_intent_epoch(state),
            known_constraints=_known_constraint_mapping(state),
        )
        mapped_names = {
            _map_context_name(name, support, minimum_value_support) for name in names
        }
        missing = mapped_names.difference(context_name_to_id)
        if missing:
            raise AssertionError(f"context vocabulary parity failure: {sorted(missing)!r}")
        identifiers = tuple(sorted(context_name_to_id[name] for name in mapped_names))
        state_context_ids.append(identifiers)
        state_rows.extend([state_index] * len(identifiers))
        state_columns.extend(identifiers)
        explicit_weight = _state_value(state, "trajectory_state_weight", "state_weight")
        weights.append(
            float(explicit_weight)
            if explicit_weight is not None
            else 1.0 / trajectory_counts[_state_trajectory_id(state)]
        )

    state_matrix = sparse.csr_matrix(
        (
            np.ones(len(state_rows), dtype=np.float32),
            (np.asarray(state_rows), np.asarray(state_columns)),
        ),
        shape=(len(states), len(context_names)),
        dtype=np.float32,
    )
    return FeatureData(
        products=products,
        dataset=dataset,
        context_names=context_names,
        item_names=item_names,
        context_fields=[field_from_name(name) for name in context_names],
        item_fields=[field_from_name(name) for name in item_names],
        context_name_to_id=context_name_to_id,
        item_name_to_id=item_name_to_id,
        product_item_ids=product_item_ids,
        product_item_sets=product_item_sets,
        product_evidence=product_evidence,
        item_matrix=item_matrix,
        state_context_ids=state_context_ids,
        state_matrix=state_matrix,
        positives=np.asarray([_state_target(state) for state in states], dtype=np.int32),
        trajectory_state_weights=np.asarray(weights, dtype=np.float32),
        state_splits=np.asarray([_state_split(state) for state in states], dtype=object),
        product_splits=np.asarray(splits, dtype=object),
    )


def subset_rows(feature_data: FeatureData, split: str | None) -> np.ndarray:
    if split is None:
        return np.arange(len(feature_data.dataset.states), dtype=np.int32)
    return np.flatnonzero(feature_data.state_splits == split).astype(np.int32)


def evidence_weight(state: object, config: TrainingConfig) -> float:
    count = len(set(_state_known_constraints(state)))
    if count == 0:
        return float(config.category_only_weight)
    return min(1.0, count / max(1, config.evidence_saturation))


def supervision_weight_band(value: float) -> str:
    if value <= 0.0:
        return "zero"
    if value <= 0.25:
        return "low"
    if value <= 0.75:
        return "medium"
    return "high"


def evidence_match_count(
    feature_data: FeatureData, state_index: int, product_index: int
) -> int:
    evidence = set(_state_known_constraints(feature_data.dataset.states[state_index]))
    return sum(pair in feature_data.product_evidence[product_index] for pair in evidence)


def indistinguishable(
    feature_data: FeatureData, state_index: int, candidate_index: int
) -> bool:
    target = int(feature_data.positives[state_index])
    return evidence_match_count(
        feature_data, state_index, candidate_index
    ) >= evidence_match_count(feature_data, state_index, target)


class Adam:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.step = 0

    def update(self, parameter: np.ndarray, gradient: np.ndarray, rate: float) -> None:
        self.step += 1
        self.m *= 0.9
        self.m += 0.1 * gradient
        self.v *= 0.999
        self.v += 0.001 * gradient * gradient
        corrected_m = self.m / (1.0 - 0.9**self.step)
        corrected_v = self.v / (1.0 - 0.999**self.step)
        parameter -= rate * corrected_m / (np.sqrt(corrected_v) + 1e-8)


def initialize_parameters(
    feature_data: FeatureData,
    cross_count: int,
    variant: str,
    config: TrainingConfig,
) -> Parameters:
    if variant not in {"linear", "fm", "hybrid"}:
        raise ValueError("variant must be linear, fm, or hybrid")
    rng = np.random.default_rng(config.seed)
    context_vectors = rng.normal(
        0.0, 0.02, (len(feature_data.context_names), config.dimension)
    ).astype(np.float32)
    item_vectors = rng.normal(
        0.0, 0.02, (len(feature_data.item_names), config.dimension)
    ).astype(np.float32)
    if variant == "linear":
        context_vectors.fill(0.0)
        item_vectors.fill(0.0)
    return Parameters(
        context_vectors=context_vectors,
        item_vectors=item_vectors,
        item_linear=np.zeros(len(feature_data.item_names), dtype=np.float32),
        cross_weights=np.zeros(cross_count, dtype=np.float32),
    )


def item_components(
    feature_data: FeatureData, parameters: Parameters
) -> tuple[np.ndarray, np.ndarray]:
    item_sum = np.asarray(feature_data.item_matrix @ parameters.item_vectors)
    squared_sum = np.asarray(feature_data.item_matrix @ (parameters.item_vectors**2))
    item_quadratic = 0.5 * np.sum(item_sum * item_sum - squared_sum, axis=1)
    item_base = np.asarray(feature_data.item_matrix @ parameters.item_linear).ravel()
    item_base += item_quadratic
    return item_sum, item_base


def _score_candidates(
    feature_data: FeatureData,
    parameters: Parameters,
    state_index: int,
    candidate_indices: np.ndarray,
    item_sum: np.ndarray,
    item_base: np.ndarray,
    pair_to_id: Mapping[tuple[int, int], int],
) -> np.ndarray:
    context_ids = feature_data.state_context_ids[state_index]
    context = np.sum(parameters.context_vectors[list(context_ids)], axis=0)
    scores = item_base[candidate_indices] + item_sum[candidate_indices] @ context
    if pair_to_id and len(parameters.cross_weights):
        scores = np.asarray(scores, dtype=np.float64)
        for local_index, product_index in enumerate(candidate_indices):
            for context_id in context_ids:
                for item_id in feature_data.product_item_ids[int(product_index)]:
                    cross_id = pair_to_id.get((context_id, item_id))
                    if cross_id is not None:
                        scores[local_index] += float(parameters.cross_weights[cross_id])
    return np.asarray(scores)


def _sample_negatives(
    feature_data: FeatureData,
    rows: np.ndarray,
    parameters: Parameters,
    cross_pairs: Sequence[tuple[int, int]],
    *,
    epoch: int,
    config: TrainingConfig,
    negative_mode: str,
    audit_limit: int = 250,
) -> PairBatch:
    """Build one deterministic negative batch for the requested pool policy."""

    if negative_mode not in NEGATIVE_MODES:
        raise ValueError(f"unsupported negative mode: {negative_mode}")

    if negative_mode == "product_fixed":
        item_sum = np.empty((0, 0), dtype=np.float32)
        item_base = np.empty(0, dtype=np.float32)
    else:
        item_sum, item_base = item_components(feature_data, parameters)
    pair_to_id = {pair: index for index, pair in enumerate(cross_pairs)}
    output_states: list[int] = []
    output_positives: list[int] = []
    output_negatives: list[int] = []
    trajectory_weights: list[float] = []
    evidence_weights: list[float] = []
    sampling_weights: list[float] = []
    ambiguity_weights: list[float] = []
    sampler_types: list[str] = []
    audit_rows: list[dict[str, object]] = []
    states_with_pool = 0
    states_with_pairs = 0
    total_survivors = 0
    total_candidates = 0
    total_excluded_product_split = 0
    total_outside_survivor = 0
    total_ties = 0
    skipped_ties = 0
    sampler_counts: Counter[str] = Counter()
    product_groups: dict[tuple[str, str], tuple[int, ...]] = {}
    product_category_counts: Counter[str] = Counter()
    product_fixed_pools: dict[int, tuple[int, ...]] = {}
    product_fixed_excluded_splits: dict[int, int] = {}
    if negative_mode == "product_fixed":
        grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
        for product_index, product in enumerate(feature_data.products):
            split = str(feature_data.product_splits[product_index])
            grouped[(product.category, split)].append(product_index)
            product_category_counts[product.category] += 1
        product_groups = {
            key: tuple(
                sorted(
                    indices,
                    key=lambda candidate: feature_data.products[
                        candidate
                    ].parent_asin,
                )
            )
            for key, indices in grouped.items()
        }
        targets = {int(feature_data.positives[int(row)]) for row in rows}
        for target in sorted(targets):
            product = feature_data.products[target]
            target_split = str(feature_data.product_splits[target])
            candidates = [
                candidate
                for candidate in product_groups.get(
                    (product.category, target_split), ()
                )
                if candidate != target
            ]
            product_fixed_excluded_splits[target] = (
                product_category_counts[product.category] - 1 - len(candidates)
            )
            if len(candidates) > config.negative_pre_pool_size:
                pool_rng = random.Random(
                    stable_int(
                        f"product-fixed-pool\0{config.seed}\0{product.parent_asin}"
                    )
                )
                candidates = pool_rng.sample(
                    candidates, config.negative_pre_pool_size
                )
            target_features = feature_data.product_item_sets[target]
            product_fixed_pools[target] = tuple(
                sorted(
                    candidates,
                    key=lambda candidate: (
                        -len(
                            target_features
                            & feature_data.product_item_sets[candidate]
                        ),
                        stable_int(
                            f"product-fixed\0{config.seed}\0"
                            f"{product.parent_asin}\0"
                            f"{feature_data.products[candidate].parent_asin}"
                        ),
                        feature_data.products[candidate].parent_asin,
                    ),
                )
            )

    for raw_state_index in rows:
        state_index = int(raw_state_index)
        state = feature_data.dataset.states[state_index]
        target = int(feature_data.positives[state_index])
        survivors = _dataset_survivors(feature_data.dataset, state_index)
        if target not in survivors:
            raise AssertionError(f"target missing from survivor state {state_index}")
        survivor_set = set(map(int, survivors))
        target_split = str(feature_data.product_splits[target])
        if target_split != _state_split(state):
            raise AssertionError(
                f"state {state_index} target belongs to {target_split}, "
                f"not {_state_split(state)}"
            )
        # Product-held-out evaluation requires held-out products to have no
        # influence on fitting, even as negative labels.  Every mode therefore
        # restricts candidates to the target product's split.  The legacy
        # product-fixed comparator intentionally ignores the current hard-
        # constraint survivor filter, but remains category- and split-local.
        if negative_mode == "product_fixed":
            pool = list(product_fixed_pools[target])
            excluded_product_split_count = product_fixed_excluded_splits[target]
        else:
            # Keep the frozen state snapshot all-catalog for exact evaluation,
            # while restricting training eligibility to the state product split.
            # Sorting makes sampling independent of set iteration details.
            pool = sorted(
                candidate
                for candidate in survivor_set
                if candidate != target
                and str(feature_data.product_splits[candidate]) == target_split
            )
            excluded_product_split_count = len(survivor_set) - 1 - len(pool)
        total_excluded_product_split += excluded_product_split_count
        total_survivors += len(survivors)
        total_candidates += len(pool)
        if not pool:
            continue
        states_with_pool += 1

        ties = [
            candidate
            for candidate in pool
            if indistinguishable(feature_data, state_index, candidate)
        ]
        tie_set = set(ties)
        total_ties += len(ties)
        if config.supervision_policy in {"skip_ties", "set_valued_positives"}:
            valid = [candidate for candidate in pool if candidate not in tie_set]
            skipped_ties += len(ties)
        else:
            valid = pool
        if not valid:
            continue

        if negative_mode == "survivor_dynamic":
            # Preserve the original v2 seed exactly: existing dynamic runs are
            # reproducible byte-for-byte after adding the comparator modes.
            seed_key = (
                f"negative\0{config.seed}\0{epoch}\0"
                f"{_state_trajectory_id(state)}\0{_state_local_index(state)}"
            )
            selection_epoch = epoch
        else:
            seed_key = (
                f"negative\0{config.seed}\0{negative_mode}\0"
                f"{_state_trajectory_id(state)}\0{_state_local_index(state)}"
            )
            selection_epoch = 0
        rng = random.Random(stable_int(seed_key))
        if negative_mode == "product_fixed":
            # Filtering a precomputed order preserves a fixed product-level
            # comparator while allowing each state's information policy to
            # remove or down-weight ambiguous pairs.
            selected = valid[: config.negatives_per_state]
            labels = ["product_fixed"] * len(selected)
        elif len(valid) <= config.negatives_per_state:
            selected = list(valid)
            labels = ["all"] * len(selected)
        else:
            pre_pool = (
                list(valid)
                if len(valid) <= config.negative_pre_pool_size
                else rng.sample(valid, config.negative_pre_pool_size)
            )
            desired = config.negatives_per_state
            hard_count = int(round(desired * config.hard_fraction))
            near_count = int(round(desired * config.near_fraction))
            random_count = desired - hard_count - near_count

            pre_array = np.asarray(pre_pool, dtype=np.int32)
            scores = _score_candidates(
                feature_data,
                parameters,
                state_index,
                pre_array,
                item_sum,
                item_base,
                pair_to_id,
            )
            hard_order = [
                int(pre_array[index])
                for index in sorted(
                    range(len(pre_array)),
                    key=lambda index: (
                        -float(scores[index]),
                        feature_data.products[int(pre_array[index])].parent_asin,
                    ),
                )
            ]
            target_features = feature_data.product_item_sets[target]
            near_order = sorted(
                pre_pool,
                key=lambda candidate: (
                    -len(target_features & feature_data.product_item_sets[candidate]),
                    stable_int(f"near\0{seed_key}\0{candidate}"),
                ),
            )
            random_order = list(pre_pool)
            rng.shuffle(random_order)
            selected = []
            labels = []
            selected_set: set[int] = set()

            def take(values: Sequence[int], count: int, label: str) -> None:
                for candidate in values:
                    if len([value for value in labels if value == label]) >= count:
                        break
                    if candidate in selected_set:
                        continue
                    selected.append(candidate)
                    labels.append(label)
                    selected_set.add(candidate)

            take(hard_order, hard_count, "model_hard")
            take(near_order, near_count, "near_match")
            take(random_order, random_count, "random")
            fill = list(pre_pool)
            rng.shuffle(fill)
            for candidate in fill:
                if len(selected) >= desired:
                    break
                if candidate not in selected_set:
                    selected.append(candidate)
                    labels.append("fill")
                    selected_set.add(candidate)

        if not selected:
            continue
        states_with_pairs += 1
        acceptable = [target, *ties]
        per_pair_sampling_weight = 1.0 / len(selected)
        state_evidence_weight = evidence_weight(state, config)
        for pair_index, (negative, sampler_type) in enumerate(
            zip(selected, labels, strict=True)
        ):
            if negative == target:
                raise AssertionError("negative sampler selected the target")
            negative_in_survivors = negative in survivor_set
            if negative_mode != "product_fixed" and not negative_in_survivors:
                raise AssertionError("negative sampler violated survivor membership")
            if str(feature_data.product_splits[negative]) != target_split:
                raise AssertionError("negative sampler crossed a product split")
            if negative_mode == "product_fixed" and not negative_in_survivors:
                total_outside_survivor += 1
            if config.supervision_policy == "set_valued_positives":
                positive = acceptable[
                    stable_int(f"positive\0{seed_key}\0{pair_index}") % len(acceptable)
                ]
            else:
                positive = target
            ambiguity = config.tie_weight if negative in tie_set else 1.0
            output_states.append(state_index)
            output_positives.append(positive)
            output_negatives.append(negative)
            trajectory_weights.append(
                float(feature_data.trajectory_state_weights[state_index])
            )
            evidence_weights.append(state_evidence_weight)
            sampling_weights.append(per_pair_sampling_weight)
            ambiguity_weights.append(ambiguity)
            sampler_types.append(sampler_type)
            sampler_counts[sampler_type] += 1
            if len(audit_rows) < audit_limit:
                audit_rows.append(
                    {
                        "epoch": epoch,
                        "selection_epoch": selection_epoch,
                        "negative_mode": negative_mode,
                        "trajectory_id": _state_trajectory_id(state),
                        "state_index": _state_local_index(state),
                        "global_state_index": state_index,
                        "target_parent_asin": feature_data.products[target].parent_asin,
                        "positive_parent_asin": feature_data.products[positive].parent_asin,
                        "negative_parent_asin": feature_data.products[negative].parent_asin,
                        "survivor_pool_size": len(survivors) - 1,
                        "eligible_negative_pool_size": len(pool),
                        "excluded_product_split_count": excluded_product_split_count,
                        "candidate_pre_pool_size": min(
                            len(valid), config.negative_pre_pool_size
                        ),
                        "sampler_type": sampler_type,
                        "negative_in_survivor_set": negative_in_survivors,
                        "indistinguishable": negative in tie_set,
                        "trajectory_state_weight": float(
                            feature_data.trajectory_state_weights[state_index]
                        ),
                        "evidence_weight": state_evidence_weight,
                        "sampling_weight": per_pair_sampling_weight,
                        "ambiguity_weight": ambiguity,
                    }
                )

    arrays = (
        np.asarray(output_states, dtype=np.int32),
        np.asarray(output_positives, dtype=np.int32),
        np.asarray(output_negatives, dtype=np.int32),
        np.asarray(trajectory_weights, dtype=np.float32),
        np.asarray(evidence_weights, dtype=np.float32),
        np.asarray(sampling_weights, dtype=np.float32),
        np.asarray(ambiguity_weights, dtype=np.float32),
    )
    effective = arrays[3] * arrays[4] * arrays[5] * arrays[6]
    diagnostics: dict[str, object] = {
        "epoch": epoch,
        "selection_epoch": epoch if negative_mode == "survivor_dynamic" else 0,
        "negative_mode": negative_mode,
        "state_count": int(len(rows)),
        "states_with_negative_pool": states_with_pool,
        "active_state_count": states_with_pairs,
        "active_state_rate": states_with_pairs / max(1, len(rows)),
        "pair_count": int(len(output_states)),
        "effective_weighted_pairs": float(np.sum(effective)),
        "mean_survivor_width": total_survivors / max(1, len(rows)),
        "excluded_product_split_candidate_count": total_excluded_product_split,
        "negative_outside_survivor_pair_count": total_outside_survivor,
        "tie_candidate_rate": total_ties / max(1, total_candidates),
        "skipped_tie_count": skipped_ties,
        "sampler_counts": dict(sorted(sampler_counts.items())),
        "supervision_policy": config.supervision_policy,
    }
    return PairBatch(
        state_indices=arrays[0],
        positives=arrays[1],
        negatives=arrays[2],
        trajectory_state_weights=arrays[3],
        evidence_weights=arrays[4],
        sampling_weights=arrays[5],
        ambiguity_weights=arrays[6],
        effective_weights=effective,
        sampler_types=np.asarray(sampler_types, dtype=object),
        diagnostics=diagnostics,
        audit_rows=audit_rows,
    )


def sample_dynamic_negatives(
    feature_data: FeatureData,
    rows: np.ndarray,
    parameters: Parameters,
    cross_pairs: Sequence[tuple[int, int]],
    *,
    epoch: int,
    config: TrainingConfig,
    audit_limit: int = 250,
) -> PairBatch:
    """Refresh state-specific survivor negatives reproducibly for one epoch.

    This compatibility entry point always performs dynamic survivor sampling,
    regardless of ``config.negative_mode``. New trainer code should call
    :func:`sample_negatives` so the configured comparator is honored.
    """

    return _sample_negatives(
        feature_data,
        rows,
        parameters,
        cross_pairs,
        epoch=epoch,
        config=config,
        negative_mode="survivor_dynamic",
        audit_limit=audit_limit,
    )


def sample_negatives(
    feature_data: FeatureData,
    rows: np.ndarray,
    parameters: Parameters,
    cross_pairs: Sequence[tuple[int, int]],
    *,
    epoch: int,
    config: TrainingConfig,
    audit_limit: int = 250,
) -> PairBatch:
    """Sample using the negative mode recorded in ``config``.

    ``survivor_static`` and ``product_fixed`` use an epoch-independent seed.
    The training loop additionally caches their first batch so model updates
    cannot change model-hard ordering between epochs.
    """

    return _sample_negatives(
        feature_data,
        rows,
        parameters,
        cross_pairs,
        epoch=epoch,
        config=config,
        negative_mode=config.negative_mode,
        audit_limit=audit_limit,
    )


def _pair_batch_for_epoch(pairs: PairBatch, epoch: int) -> PairBatch:
    """Reuse a static batch while keeping per-epoch diagnostics auditable."""

    diagnostics = dict(pairs.diagnostics)
    diagnostics["epoch"] = epoch
    audit_rows = [dict(row, epoch=epoch) for row in pairs.audit_rows]
    return PairBatch(
        state_indices=pairs.state_indices,
        positives=pairs.positives,
        negatives=pairs.negatives,
        trajectory_state_weights=pairs.trajectory_state_weights,
        evidence_weights=pairs.evidence_weights,
        sampling_weights=pairs.sampling_weights,
        ambiguity_weights=pairs.ambiguity_weights,
        effective_weights=pairs.effective_weights,
        sampler_types=pairs.sampler_types,
        diagnostics=diagnostics,
        audit_rows=audit_rows,
    )


def cross_matrix(
    feature_data: FeatureData,
    state_indices: np.ndarray,
    product_indices: np.ndarray,
    pair_to_id: Mapping[tuple[int, int], int],
) -> sparse.csr_matrix:
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    for local_row, (state_index, product_index) in enumerate(
        zip(state_indices, product_indices, strict=True)
    ):
        active: set[int] = set()
        for context_id in feature_data.state_context_ids[int(state_index)]:
            for item_id in feature_data.product_item_ids[int(product_index)]:
                cross_id = pair_to_id.get((context_id, item_id))
                if cross_id is not None:
                    active.add(cross_id)
        matrix_rows.extend([local_row] * len(active))
        matrix_columns.extend(sorted(active))
    return sparse.csr_matrix(
        (
            np.ones(len(matrix_rows), dtype=np.float32),
            (np.asarray(matrix_rows), np.asarray(matrix_columns)),
        ),
        shape=(len(state_indices), len(pair_to_id)),
        dtype=np.float32,
    )


def eligible_crosses(
    feature_data: FeatureData,
    pairs: PairBatch,
    *,
    minimum_support: int = MIN_CROSS_SUPPORT,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """Select auditable explicit crosses from actual survivor comparisons."""

    if not len(pairs):
        return [], np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    context_allowed = np.asarray(
        [
            field in {"category", *ATTRIBUTES} and RARE_VALUE not in name
            for name, field in zip(
                feature_data.context_names, feature_data.context_fields, strict=True
            )
        ]
    )
    item_allowed = np.asarray(
        [RARE_VALUE not in name for name in feature_data.item_names]
    )
    context_ids = np.flatnonzero(context_allowed)
    item_ids = np.flatnonzero(item_allowed)
    x = feature_data.state_matrix[pairs.state_indices][:, context_ids]
    positive_counts = (
        x.T @ feature_data.item_matrix[pairs.positives][:, item_ids]
    ).tocoo()
    context_exposure = np.asarray(x.sum(axis=0)).ravel().astype(np.int64)

    candidate_pairs: list[tuple[int, int]] = []
    positive_support: list[int] = []
    comparable_support: list[int] = []
    for row, column, count in zip(
        positive_counts.row,
        positive_counts.col,
        positive_counts.data,
        strict=True,
    ):
        if int(count) < minimum_support:
            continue
        candidate_pairs.append((int(context_ids[row]), int(item_ids[column])))
        positive_support.append(int(count))
        comparable_support.append(int(context_exposure[row]))

    mask = np.asarray(comparable_support, dtype=np.int64) >= minimum_support
    return (
        [pair for pair, keep in zip(candidate_pairs, mask, strict=True) if keep],
        np.asarray(positive_support, dtype=np.int32)[mask],
        np.asarray(comparable_support, dtype=np.int32)[mask],
    )


def score_pair(
    feature_data: FeatureData,
    parameters: Parameters,
    state_index: int,
    product_index: int,
    item_sum: np.ndarray,
    item_base: np.ndarray,
    pair_to_id: Mapping[tuple[int, int], int],
) -> float:
    return float(
        _score_candidates(
            feature_data,
            parameters,
            state_index,
            np.asarray([product_index], dtype=np.int32),
            item_sum,
            item_base,
            pair_to_id,
        )[0]
    )


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values:
        return 0.0
    weight_array = np.asarray(weights, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    total = float(weight_array.sum())
    return float(np.dot(value_array, weight_array) / total) if total > 0 else 0.0


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values:
        return 0.0
    order = np.argsort(np.asarray(values), kind="stable")
    ordered_values = np.asarray(values, dtype=np.float64)[order]
    ordered_weights = np.asarray(weights, dtype=np.float64)[order]
    cutoff = ordered_weights.sum() / 2.0
    position = int(np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left"))
    return float(ordered_values[min(position, len(ordered_values) - 1)])


def evaluate_full_survivor(
    feature_data: FeatureData,
    parameters: Parameters,
    rows: np.ndarray,
    cross_pairs: Sequence[tuple[int, int]],
    *,
    return_rows: bool = False,
) -> dict[str, object]:
    """Rank each exact target against every current survivor."""

    item_sum, item_base = item_components(feature_data, parameters)
    pair_to_id = {pair: index for index, pair in enumerate(cross_pairs)}
    ranks: list[float] = []
    reciprocal_ranks: list[float] = []
    hit_1: list[float] = []
    hit_5: list[float] = []
    hit_10: list[float] = []
    informative_hit_10: list[float] = []
    informative_hit_10_weights: list[float] = []
    percentiles: list[float] = []
    pairwise: list[float] = []
    widths: list[float] = []
    weights: list[float] = []
    state_rows: list[dict[str, object]] = []
    for raw_state_index in rows:
        state_index = int(raw_state_index)
        state = feature_data.dataset.states[state_index]
        target = int(feature_data.positives[state_index])
        survivors = _dataset_survivors(feature_data.dataset, state_index)
        if target not in survivors:
            raise AssertionError(f"target missing from full survivor state {state_index}")
        scores = _score_candidates(
            feature_data,
            parameters,
            state_index,
            survivors,
            item_sum,
            item_base,
            pair_to_id,
        )
        target_position = int(np.flatnonzero(survivors == target)[0])
        target_score = float(scores[target_position])
        target_asin = feature_data.products[target].parent_asin
        better = 0
        wins = 0.0
        for local_index, candidate in enumerate(survivors):
            candidate = int(candidate)
            if candidate == target:
                continue
            score = float(scores[local_index])
            candidate_asin = feature_data.products[candidate].parent_asin
            if score > target_score or (
                score == target_score and candidate_asin < target_asin
            ):
                better += 1
            if target_score > score:
                wins += 1.0
            elif target_score == score:
                wins += 0.5
        rank = better + 1
        width = len(survivors)
        percentile = (rank - 1) / max(1, width - 1)
        # A singleton survivor set contains no adverse pair and is therefore
        # vacuously correct.  This also avoids reporting a perfect rank as zero
        # pairwise accuracy for fully resolved trajectories.
        pairwise_value = 1.0 if width == 1 else wins / (width - 1)
        state_weight = float(feature_data.trajectory_state_weights[state_index])
        ranks.append(float(rank))
        reciprocal_ranks.append(1.0 / rank)
        hit_1.append(float(rank <= 1))
        hit_5.append(float(rank <= 5))
        hit_10.append(float(rank <= 10))
        if width > 10:
            informative_hit_10.append(float(rank <= 10))
            informative_hit_10_weights.append(state_weight)
        percentiles.append(percentile)
        pairwise.append(pairwise_value)
        widths.append(float(width))
        weights.append(state_weight)
        if return_rows:
            state_evidence_weight = evidence_weight(state, TrainingConfig())
            state_rows.append(
                {
                    "state_id": str(
                        _state_value(state, "state_id", default=state_index)
                    ),
                    "trajectory_id": _state_trajectory_id(state),
                    "split": _state_split(state),
                    "scenario_type": _state_scenario_type(state),
                    "scenario_state": _state_scenario(state),
                    "turn": _state_turn(state),
                    "turn_bucket": turn_bucket(_state_turn(state)),
                    "has_other_answer": _state_has_other(state),
                    "supervision_weight_band": supervision_weight_band(
                        state_evidence_weight
                    ),
                    "target_parent_asin": target_asin,
                    "rank": rank,
                    "reciprocal_rank": 1.0 / rank,
                    "hit_at_1": int(rank <= 1),
                    "hit_at_5": int(rank <= 5),
                    "hit_at_10": int(rank <= 10),
                    "rank_percentile": percentile,
                    "candidate_width": width,
                    "pairwise_accuracy": pairwise_value,
                    "trajectory_state_weight": state_weight,
                }
            )

    summary: dict[str, object] = {
        "evaluation_protocol": "exact_full_survivor_v1",
        "state_count": len(ranks),
        "trajectory_count": len(
            {
                _state_trajectory_id(feature_data.dataset.states[int(index)])
                for index in rows
            }
        ),
        "mrr": _weighted_mean(reciprocal_ranks, weights),
        "hit_rate_at_1": _weighted_mean(hit_1, weights),
        "hit_rate_at_5": _weighted_mean(hit_5, weights),
        "hit_rate_at_10": (
            _weighted_mean(informative_hit_10, informative_hit_10_weights)
            if informative_hit_10
            else None
        ),
        "hit_rate_at_10_raw": _weighted_mean(hit_10, weights),
        "hit_rate_at_10_informative": bool(informative_hit_10),
        "hit_rate_at_10_informative_state_count": len(informative_hit_10),
        "mean_rank": _weighted_mean(ranks, weights),
        "median_rank": _weighted_median(ranks, weights),
        "mean_rank_percentile": _weighted_mean(percentiles, weights),
        "mean_candidate_width": _weighted_mean(widths, weights),
        "median_candidate_width": _weighted_median(widths, weights),
        "pairwise_accuracy": _weighted_mean(pairwise, weights),
        "state_micro_mrr": float(np.mean(reciprocal_ranks)) if ranks else 0.0,
        "state_micro_hit_rate_at_10": (
            float(np.mean(informative_hit_10)) if informative_hit_10 else None
        ),
        "state_micro_hit_rate_at_10_raw": (
            float(np.mean(hit_10)) if ranks else 0.0
        ),
    }
    if return_rows:
        summary["states"] = state_rows
    return summary


def calibrate_temperature(
    feature_data: FeatureData,
    parameters: Parameters,
    rows: np.ndarray,
    cross_pairs: Sequence[tuple[int, int]],
) -> tuple[float, dict[str, float]]:
    item_sum, item_base = item_components(feature_data, parameters)
    pair_to_id = {pair: index for index, pair in enumerate(cross_pairs)}
    score_sets: list[tuple[np.ndarray, int, float]] = []
    for raw_state_index in rows:
        state_index = int(raw_state_index)
        target = int(feature_data.positives[state_index])
        survivors = _dataset_survivors(feature_data.dataset, state_index)
        target_position = int(np.flatnonzero(survivors == target)[0])
        scores = _score_candidates(
            feature_data,
            parameters,
            state_index,
            survivors,
            item_sum,
            item_base,
            pair_to_id,
        )
        score_sets.append(
            (
                np.asarray(scores, dtype=np.float64),
                target_position,
                float(feature_data.trajectory_state_weights[state_index]),
            )
        )
    losses: dict[str, float] = {}
    for temperature in (0.25, 0.5, 1.0, 2.0, 4.0):
        nll: list[float] = []
        weights: list[float] = []
        for scores, target_position, state_weight in score_sets:
            values = scores / temperature
            values -= np.max(values)
            nll.append(
                float(-values[target_position] + np.log(np.exp(values).sum()))
            )
            weights.append(state_weight)
        losses[str(temperature)] = _weighted_mean(nll, weights)
    best = min((float(key) for key in losses), key=lambda value: losses[str(value)])
    return best, losses


def _copy_parameters(parameters: Parameters) -> Parameters:
    return Parameters(
        context_vectors=np.array(parameters.context_vectors, copy=True),
        item_vectors=np.array(parameters.item_vectors, copy=True),
        item_linear=np.array(parameters.item_linear, copy=True),
        cross_weights=np.array(parameters.cross_weights, copy=True),
    )


def train_model(
    feature_data: FeatureData,
    rows: np.ndarray,
    cross_pairs: Sequence[tuple[int, int]],
    *,
    validation_rows: np.ndarray | None,
    variant: str,
    config: TrainingConfig,
) -> tuple[Parameters, list[dict[str, object]], int, list[dict[str, object]]]:
    """Train independent Linear/FM/Hybrid parameters with weighted BPR."""

    parameters = initialize_parameters(feature_data, len(cross_pairs), variant, config)
    optimizers = {
        "context": Adam(parameters.context_vectors.shape),
        "item": Adam(parameters.item_vectors.shape),
        "linear": Adam(parameters.item_linear.shape),
        "cross": Adam(parameters.cross_weights.shape),
    }
    pair_to_id = {pair: index for index, pair in enumerate(cross_pairs)}
    history: list[dict[str, object]] = []
    sampler_audit: list[dict[str, object]] = []
    best_mrr = -1.0
    best_epoch = 0
    best_parameters: Parameters | None = None
    stale_checks = 0
    static_pairs: PairBatch | None = None

    for epoch in range(1, config.max_epochs + 1):
        if config.negative_mode == "survivor_dynamic":
            pairs = sample_negatives(
                feature_data,
                rows,
                parameters,
                cross_pairs,
                epoch=epoch,
                config=config,
            )
        else:
            if static_pairs is None:
                static_pairs = sample_negatives(
                    feature_data,
                    rows,
                    parameters,
                    cross_pairs,
                    epoch=epoch,
                    config=config,
                )
            pairs = _pair_batch_for_epoch(static_pairs, epoch)
        if not len(pairs) or float(pairs.effective_weights.sum()) <= 0:
            raise RuntimeError(
                "no effective survivor pairs remain; inspect tie diagnostics before scaling"
            )
        sampler_audit.extend(pairs.audit_rows)
        order = np.arange(len(pairs), dtype=np.int64)
        np.random.default_rng(stable_int(f"pair-order\0{config.seed}\0{epoch}")).shuffle(
            order
        )
        epoch_loss_numerator = 0.0
        epoch_weight = 0.0
        batch_count = math.ceil(len(order) / config.pair_batch_size)
        for batch_start in range(0, len(order), config.pair_batch_size):
            selection = order[batch_start : batch_start + config.pair_batch_size]
            state_indices = pairs.state_indices[selection]
            positives = pairs.positives[selection]
            negatives = pairs.negatives[selection]
            weights = pairs.effective_weights[selection].astype(np.float32)
            normalizer = float(weights.sum())
            if normalizer <= 0:
                continue

            item_sum, item_base = item_components(feature_data, parameters)
            x = feature_data.state_matrix[state_indices]
            context_sum = np.asarray(x @ parameters.context_vectors)
            delta = item_base[positives] - item_base[negatives]
            delta += np.sum(
                context_sum * (item_sum[positives] - item_sum[negatives]), axis=1
            )
            if pair_to_id:
                positive_cross = cross_matrix(
                    feature_data, state_indices, positives, pair_to_id
                )
                negative_cross = cross_matrix(
                    feature_data, state_indices, negatives, pair_to_id
                )
                cross_difference = positive_cross - negative_cross
                delta += np.asarray(
                    cross_difference @ parameters.cross_weights
                ).ravel()
            else:
                cross_difference = sparse.csr_matrix((len(selection), 0))

            q = 1.0 / (1.0 + np.exp(np.clip(delta, -30.0, 30.0)))
            g = (-q * weights / normalizer).astype(np.float32)
            epoch_loss_numerator += float(
                np.dot(weights, np.logaddexp(0.0, -delta))
            )
            epoch_weight += normalizer

            difference = item_sum[positives] - item_sum[negatives]
            gradient_context = np.asarray(x.T @ (g[:, None] * difference))
            gradient_context += (
                config.latent_l2 / batch_count
            ) * parameters.context_vectors

            relation = sparse.coo_matrix(
                (
                    np.concatenate((g, -g)),
                    (
                        np.concatenate(
                            (np.arange(len(selection)), np.arange(len(selection)))
                        ),
                        np.concatenate((positives, negatives)),
                    ),
                ),
                shape=(len(selection), len(feature_data.products)),
            ).tocsr()
            product_scalar = np.asarray(relation.sum(axis=0)).ravel().astype(
                np.float32
            )
            product_context = np.asarray(relation.T @ context_sum)
            gradient_item = np.asarray(feature_data.item_matrix.T @ product_context)
            gradient_item += np.asarray(
                feature_data.item_matrix.T @ (product_scalar[:, None] * item_sum)
            )
            feature_scalar = np.asarray(
                feature_data.item_matrix.T @ product_scalar
            ).ravel()
            gradient_item -= parameters.item_vectors * feature_scalar[:, None]
            gradient_item += (
                config.latent_l2 / batch_count
            ) * parameters.item_vectors
            gradient_linear = np.asarray(
                feature_data.item_matrix.T @ product_scalar
            ).ravel()
            gradient_linear += (
                config.linear_l2 / batch_count
            ) * parameters.item_linear
            gradient_cross = (
                np.asarray(cross_difference.T @ g).ravel()
                + (config.cross_l2 / batch_count) * parameters.cross_weights
                if pair_to_id
                else parameters.cross_weights
            )

            if variant != "linear":
                optimizers["context"].update(
                    parameters.context_vectors,
                    gradient_context.astype(np.float32),
                    config.learning_rate,
                )
                optimizers["item"].update(
                    parameters.item_vectors,
                    gradient_item.astype(np.float32),
                    config.learning_rate,
                )
            optimizers["linear"].update(
                parameters.item_linear,
                gradient_linear.astype(np.float32),
                config.learning_rate,
            )
            if variant == "hybrid" and pair_to_id:
                optimizers["cross"].update(
                    parameters.cross_weights,
                    gradient_cross.astype(np.float32),
                    config.learning_rate,
                )

        checked = (
            validation_rows is not None
            and len(validation_rows)
            and epoch % config.validation_interval == 0
        )
        validation = (
            evaluate_full_survivor(
                feature_data,
                parameters,
                validation_rows,
                cross_pairs,
            )
            if checked
            else {}
        )
        record: dict[str, object] = {
            "epoch": epoch,
            "weighted_bpr_loss": epoch_loss_numerator / max(epoch_weight, 1e-12),
            "pair_count": len(pairs),
            "effective_weighted_pairs": float(pairs.effective_weights.sum()),
            "negative_sampling": pairs.diagnostics,
        }
        if validation:
            record["full_survivor_validation"] = validation
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

        if validation_rows is None:
            best_epoch = epoch
            continue
        if not checked:
            continue
        validation_mrr = float(validation["mrr"])
        if validation_mrr > best_mrr + 1e-7:
            best_mrr = validation_mrr
            best_epoch = epoch
            stale_checks = 0
            best_parameters = _copy_parameters(parameters)
        else:
            stale_checks += 1
            if stale_checks >= config.patience:
                break

    return best_parameters or parameters, history, best_epoch or config.max_epochs, sampler_audit


def write_artifact(
    output_path: Path,
    catalog_path: Path,
    feature_data: FeatureData,
    parameters: Parameters,
    cross_pairs: Sequence[tuple[int, int]],
    positive_support: np.ndarray,
    negative_support: np.ndarray,
    temperature: float,
    selected_epoch: int,
    variant: str,
    training_config: TrainingConfig,
    dataset_manifest: Mapping[str, object],
    *,
    training_scope: str = "train_only",
) -> None:
    """Write a schema-v1 portable artifact without overwriting frozen E0 models."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".building")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
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
        manifest_json = json.dumps(
            dataset_manifest, sort_keys=True, separators=(",", ":")
        )
        metadata = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "feature_schema_version": (
                "conversation-features-v2"
                if training_config.other_encoding == "dual"
                else "conversation-features-v1"
            ),
            "training_schema_version": training_config.schema_version,
            "model_type": {
                "linear": "linear_pairwise_ranker",
                "fm": "second_order_factorization_machine",
                "hybrid": "second_order_fm_plus_explicit_crosses",
            }[variant],
            "training_scope": training_scope,
            "catalog_sha256": sha256_path(catalog_path),
            "dataset_manifest_sha256": hashlib.sha256(
                manifest_json.encode("utf-8")
            ).hexdigest(),
            "dataset_version": str(dataset_manifest.get("dataset_version", "")),
            "trajectory_count": str(dataset_manifest.get("trajectory_count", "")),
            "state_count": str(dataset_manifest.get("state_count", "")),
            "product_count": str(len(feature_data.products)),
            "dimension": str(parameters.context_vectors.shape[1]),
            "temperature": str(temperature),
            "selected_epoch": str(selected_epoch),
            "seed": str(training_config.seed),
            "minimum_value_support": str(training_config.minimum_value_support),
            "minimum_cross_support": str(training_config.minimum_cross_support),
            "negatives_per_state": str(training_config.negatives_per_state),
            "negative_pre_pool_size": str(training_config.negative_pre_pool_size),
            "negative_mode": training_config.negative_mode,
            "negative_mixture": json.dumps(
                {
                    "model_hard": training_config.hard_fraction,
                    "near_match": training_config.near_fraction,
                    "random": training_config.random_fraction,
                },
                sort_keys=True,
            ),
            "supervision_policy": training_config.supervision_policy,
            "tie_weight": str(training_config.tie_weight),
            "category_only_weight": str(training_config.category_only_weight),
            "evidence_saturation": str(training_config.evidence_saturation),
            "latent_l2": str(training_config.latent_l2),
            "linear_l2": str(training_config.linear_l2),
            "cross_l2": str(training_config.cross_l2),
            "learning_rate": str(training_config.learning_rate),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES(?,?)", metadata.items()
        )
        connection.executemany(
            "INSERT INTO context_features(feature_id,name,field,vector) VALUES(?,?,?,?)",
            (
                (
                    feature_id,
                    name,
                    feature_data.context_fields[feature_id],
                    sqlite3.Binary(
                        parameters.context_vectors[feature_id]
                        .astype("<f4")
                        .tobytes()
                    ),
                )
                for feature_id, name in enumerate(feature_data.context_names)
            ),
        )
        connection.executemany(
            """
            INSERT INTO item_features(feature_id,name,field,linear_weight,vector)
            VALUES(?,?,?,?,?)
            """,
            (
                (
                    feature_id,
                    name,
                    feature_data.item_fields[feature_id],
                    float(parameters.item_linear[feature_id]),
                    sqlite3.Binary(
                        parameters.item_vectors[feature_id].astype("<f4").tobytes()
                    ),
                )
                for feature_id, name in enumerate(feature_data.item_names)
            ),
        )
        item_sum, item_base = item_components(feature_data, parameters)
        connection.executemany(
            """
            INSERT INTO products(parent_asin,base_score,vector,item_feature_ids)
            VALUES(?,?,?,?)
            """,
            (
                (
                    product.parent_asin,
                    float(item_base[index]),
                    sqlite3.Binary(item_sum[index].astype("<f4").tobytes()),
                    sqlite3.Binary(
                        struct.pack(
                            f"<{len(feature_data.product_item_ids[index])}I",
                            *feature_data.product_item_ids[index],
                        )
                    ),
                )
                for index, product in enumerate(feature_data.products)
            ),
        )
        connection.executemany(
            """
            INSERT INTO cross_weights(
                context_feature_id,item_feature_id,positive_support,
                negative_support,weight
            ) VALUES(?,?,?,?,?)
            """,
            (
                (
                    context_id,
                    item_id,
                    int(positive_support[index]),
                    int(negative_support[index]),
                    float(parameters.cross_weights[index]),
                )
                for index, (context_id, item_id) in enumerate(cross_pairs)
            ),
        )
        reply_rows: list[tuple[str, int, str, str]] = []
        for product in feature_data.products:
            for ordinal, (attribute, normalized, _) in enumerate(product.constraints):
                reply_rows.append(
                    (product.parent_asin, ordinal, attribute, normalized)
                )
        connection.executemany(
            """
            INSERT INTO reply_values(parent_asin,ordinal,attribute,normalized_value)
            VALUES(?,?,?,?)
            """,
            reply_rows,
        )
        connection.commit()
        connection.execute("PRAGMA optimize")
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        connection.close()
    temporary.replace(output_path)


def write_cross_audit(
    output_path: Path,
    feature_data: FeatureData,
    parameters: Parameters,
    cross_pairs: Sequence[tuple[int, int]],
    positive_support: np.ndarray,
    negative_support: np.ndarray,
) -> None:
    rows = [
        {
            "context_feature": feature_data.context_names[context_id],
            "item_feature": feature_data.item_names[item_id],
            "field_pair": (
                f"{feature_data.context_fields[context_id]}×"
                f"{feature_data.item_fields[item_id]}"
            ),
            "positive_support": int(positive_support[index]),
            "negative_support": int(negative_support[index]),
            "learned_weight": float(parameters.cross_weights[index]),
            "absolute_weight": abs(float(parameters.cross_weights[index])),
        }
        for index, (context_id, item_id) in enumerate(cross_pairs)
    ]
    rows.sort(
        key=lambda row: (
            -float(row["absolute_weight"]),
            str(row["context_feature"]),
            str(row["item_feature"]),
        )
    )
    write_csv(
        output_path,
        rows,
        (
            "context_feature",
            "item_feature",
            "field_pair",
            "positive_support",
            "negative_support",
            "learned_weight",
            "absolute_weight",
        ),
    )


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fallback_fields: Sequence[str] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else list(fallback_fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        seed=args.seed,
        dimension=args.dimension,
        minimum_value_support=args.minimum_value_support,
        minimum_cross_support=args.minimum_cross_support,
        negatives_per_state=args.negatives,
        negative_pre_pool_size=args.negative_pre_pool_size,
        negative_mode=args.negative_mode,
        hard_fraction=args.hard_fraction,
        near_fraction=args.near_fraction,
        random_fraction=args.random_fraction,
        other_encoding=args.other_encoding,
        supervision_policy=args.supervision_policy,
        tie_weight=args.tie_weight,
        category_only_weight=args.category_only_weight,
        evidence_saturation=args.evidence_saturation,
        learning_rate=args.learning_rate,
        latent_l2=args.latent_l2,
        linear_l2=args.linear_l2,
        cross_l2=args.cross_l2,
        max_epochs=args.max_epochs,
        patience=args.patience,
        validation_interval=args.validation_interval,
        pair_batch_size=args.pair_batch_size,
    )


def _parser() -> argparse.ArgumentParser:
    redesign_dir = Path(__file__).with_name("outputs")
    parser = argparse.ArgumentParser(
        description="Train an independent FM on realistic survivor trajectories"
    )
    parser.add_argument(
        "--catalog", type=Path, default=PROJECT_ROOT / "data/catalog.jsonl"
    )
    parser.add_argument("--trajectory-count", type=int, default=25_000)
    parser.add_argument("--scenario-mix", choices=("public", "balanced"), default="public")
    parser.add_argument("--extended-fraction", type=float, default=0.10)
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="model initialization, minibatch, and dynamic-sampler seed",
    )
    parser.add_argument(
        "--trajectory-seed",
        type=int,
        default=None,
        help="fixed synthetic-data seed (defaults to --seed for compatibility)",
    )
    parser.add_argument("--split-seed", type=int, default=SEED)
    parser.add_argument("--variant", choices=("linear", "fm", "hybrid"), default="fm")
    parser.add_argument(
        "--supervision-policy",
        choices=("skip_ties", "downweight_ties", "set_valued_positives"),
        default="set_valued_positives",
    )
    parser.add_argument("--tie-weight", type=float, default=0.10)
    parser.add_argument("--category-only-weight", type=float, default=0.05)
    parser.add_argument("--evidence-saturation", type=int, default=3)
    parser.add_argument("--negatives", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--negative-pre-pool-size", type=int, default=128)
    parser.add_argument(
        "--negative-mode",
        choices=NEGATIVE_MODES,
        default="survivor_dynamic",
        help=(
            "product-level fixed competitors, one static survivor sample, or "
            "epoch-refreshed survivor samples"
        ),
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
    parser.add_argument("--cross-l2", type=float, default=1e-4)
    parser.add_argument("--minimum-value-support", type=int, default=5)
    parser.add_argument("--minimum-cross-support", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--pair-batch-size", type=int, default=65_536)
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=redesign_dir / "fm_candidate_v2.sqlite3"
    )
    parser.add_argument(
        "--metrics", type=Path, default=redesign_dir / "training_metrics_v2.json"
    )
    parser.add_argument(
        "--manifest", type=Path, default=redesign_dir / "dataset_manifest_v2.json"
    )
    parser.add_argument(
        "--negative-audit", type=Path, default=redesign_dir / "negative_audit_v2.csv"
    )
    parser.add_argument(
        "--cross-audit", type=Path, default=redesign_dir / "cross_weights_v2.csv"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    started = time.time()
    training_config = _training_config_from_args(args)
    trajectory_seed = args.seed if args.trajectory_seed is None else args.trajectory_seed
    trajectory_config = trajectory.TrajectoryConfig(
        trajectory_count=args.trajectory_count,
        seed=trajectory_seed,
        split_seed=args.split_seed,
        scenario_mix=args.scenario_mix,
        max_turns=10,
        extended_fraction=args.extended_fraction,
    )
    products, raw_products = load_products(args.catalog)
    product_splits = trajectory.build_product_splits(products, args.split_seed)
    dataset = trajectory.generate_trajectory_dataset(
        products,
        raw_products,
        trajectory_config,
        product_splits=product_splits,
        input_hashes={"catalog": sha256_path(args.catalog)},
    )
    dataset_manifest = trajectory.write_manifest(
        args.manifest, dataset, input_paths={"catalog": args.catalog}
    )
    print(
        json.dumps(
            {
                "loaded_products": len(products),
                "trajectory_count": dataset.trajectory_count,
                "state_count": len(dataset.states),
                "scenario_counts": dataset_manifest["trajectory_counts_by_scenario"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dataset_only:
        print(json.dumps(dataset_manifest, indent=2, sort_keys=True))
        return

    feature_data = build_feature_data(
        products,
        dataset,
        dataset.product_split_labels,
        minimum_value_support=training_config.minimum_value_support,
        other_encoding=training_config.other_encoding,
    )
    train_rows = subset_rows(feature_data, "train")
    validation_rows = subset_rows(feature_data, "validation")
    test_rows = subset_rows(feature_data, "test")
    print(
        f"States train={len(train_rows)} validation={len(validation_rows)} "
        f"test={len(test_rows)}",
        flush=True,
    )

    if args.variant == "hybrid":
        initializer = initialize_parameters(feature_data, 0, "fm", training_config)
        initial_pairs = sample_negatives(
            feature_data,
            train_rows,
            initializer,
            (),
            epoch=0,
            config=training_config,
        )
        cross_pairs, positive_support, negative_support = eligible_crosses(
            feature_data,
            initial_pairs,
            minimum_support=training_config.minimum_cross_support,
        )
    else:
        cross_pairs = []
        positive_support = np.zeros(0, dtype=np.int32)
        negative_support = np.zeros(0, dtype=np.int32)

    parameters, history, selected_epoch, sampler_audit = train_model(
        feature_data,
        train_rows,
        cross_pairs,
        validation_rows=validation_rows,
        variant=args.variant,
        config=training_config,
    )
    full_validation = evaluate_full_survivor(
        feature_data, parameters, validation_rows, cross_pairs
    )
    full_test = evaluate_full_survivor(
        feature_data, parameters, test_rows, cross_pairs
    )
    temperature, temperature_losses = calibrate_temperature(
        feature_data, parameters, validation_rows, cross_pairs
    )
    write_artifact(
        args.output,
        args.catalog,
        feature_data,
        parameters,
        cross_pairs,
        positive_support,
        negative_support,
        temperature,
        selected_epoch,
        args.variant,
        training_config,
        dataset_manifest,
    )
    write_cross_audit(
        args.cross_audit,
        feature_data,
        parameters,
        cross_pairs,
        positive_support,
        negative_support,
    )
    write_csv(
        args.negative_audit,
        sampler_audit,
        (
            "epoch",
            "selection_epoch",
            "negative_mode",
            "trajectory_id",
            "state_index",
            "global_state_index",
            "target_parent_asin",
            "positive_parent_asin",
            "negative_parent_asin",
            "survivor_pool_size",
            "eligible_negative_pool_size",
            "excluded_product_split_count",
            "candidate_pre_pool_size",
            "sampler_type",
            "negative_in_survivor_set",
            "indistinguishable",
            "trajectory_state_weight",
            "evidence_weight",
            "sampling_weight",
            "ambiguity_weight",
        ),
    )
    report = {
        "schema_version": training_config.schema_version,
        "model": args.variant,
        "training_scope": "train_only",
        "negative_mode": training_config.negative_mode,
        "training_config": asdict(training_config),
        "trajectory_config": asdict(trajectory_config),
        "dataset_manifest": dataset_manifest,
        "context_feature_count": len(feature_data.context_names),
        "item_feature_count": len(feature_data.item_names),
        "explicit_cross_count": len(cross_pairs),
        "selected_epoch": selected_epoch,
        "temperature": temperature,
        "temperature_nll": temperature_losses,
        "full_survivor_validation": full_validation,
        "full_survivor_test": full_test,
        "training_history": history,
        "elapsed_seconds": round(time.time() - started, 3),
        "artifact": str(args.output),
        "manifest": str(args.manifest),
        "negative_audit": str(args.negative_audit),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


__all__ = [
    "ATTRIBUTES",
    "CROSS_L2",
    "DIMENSION",
    "FM_L2",
    "FeatureData",
    "LEARNING_RATE",
    "LINEAR_L2",
    "MIN_CROSS_SUPPORT",
    "MIN_VALUE_SUPPORT",
    "NEGATIVE_MODES",
    "NEGATIVES_PER_STATE",
    "PairBatch",
    "Parameters",
    "Product",
    "SEED",
    "State",
    "TrainingConfig",
    "build_feature_data",
    "calibrate_temperature",
    "eligible_crosses",
    "evaluate_full_survivor",
    "evidence_match_count",
    "evidence_weight",
    "indistinguishable",
    "initialize_parameters",
    "item_components",
    "load_products",
    "main",
    "mapped_value",
    "sample_dynamic_negatives",
    "sample_negatives",
    "score_pair",
    "sha256_path",
    "split_for",
    "stable_int",
    "subset_rows",
    "supervision_weight_band",
    "train_model",
    "write_artifact",
]
