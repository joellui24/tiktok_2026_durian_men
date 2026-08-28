from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sqlite3
import struct
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPOSITORY_ROOT / "techjam-conversational-search"
sys.path.insert(0, str(PROJECT_ROOT))

from evaluator import local_evaluator as evaluator  # noqa: E402
from starter.attribute_index import normalize_value  # noqa: E402
from starter.category_index import coarse_category  # noqa: E402
from starter.hybrid_model import MODEL_SCHEMA_VERSION, RARE_VALUE  # noqa: E402


ATTRIBUTES = (
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
SCENARIOS = (
    "buying",
    "browsing",
    "boundary",
    "provisional_override",
    "intent_override",
)
TURN_BUCKETS = ("early", "middle", "late")
SEED = 2026
DIMENSION = 16
MIN_VALUE_SUPPORT = 5
MIN_CROSS_SUPPORT = 20
NEGATIVES_PER_STATE = 8
FM_L2 = 1e-5
CROSS_L2 = 1e-4
LEARNING_RATE = 0.01


@dataclass(frozen=True)
class Product:
    parent_asin: str
    category: str
    constraints: tuple[tuple[str, str, str], ...]
    brand: str | None
    price_bucket: str
    rating_bucket: str
    popularity_bucket: str


@dataclass(frozen=True)
class State:
    product_index: int
    scenario: str
    turn_bucket: str
    override: str
    known_constraints: tuple[tuple[str, str], ...]


@dataclass
class FeatureData:
    context_names: list[str]
    item_names: list[str]
    context_fields: list[str]
    item_fields: list[str]
    context_name_to_id: dict[str, int]
    item_name_to_id: dict[str, int]
    product_item_ids: list[tuple[int, ...]]
    item_matrix: sparse.csr_matrix
    states: list[State]
    state_context_ids: list[tuple[int, ...]]
    state_matrix: sparse.csr_matrix
    positives: np.ndarray


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def split_for(parent_asin: str) -> str:
    bucket = stable_int(f"split\0{parent_asin}") % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "validation"
    return "test"


def price_bucket(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "missing"
    price = float(value)
    boundaries = (10, 20, 35, 50, 75, 100, 150, 250)
    lower = 0
    for upper in boundaries:
        if price < upper:
            return f"{lower}_{upper}"
        lower = upper
    return "250_plus"


def rating_bucket(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "missing"
    return f"{round(float(value) * 2) / 2:.1f}"


def popularity_bucket(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "missing"
    count = max(0, int(value))
    return str(min(16, int(math.log2(count + 1))))


def load_products(catalog_path: Path) -> tuple[list[Product], list[dict]]:
    products: list[Product] = []
    raw_products: list[dict] = []
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            card = evaluator.intent_card(raw)
            constraints = tuple(
                (
                    evaluator.classify_constraint(str(value)),
                    normalize_value(str(value)),
                    str(value),
                )
                for value in (
                    *card.get("hard_constraints", []),
                    *card.get("soft_preferences", []),
                )
            )
            brand = normalize_value(str(raw.get("store"))) if raw.get("store") else None
            products.append(
                Product(
                    parent_asin=str(raw["parent_asin"]),
                    category=normalize_value(
                        coarse_category([str(value) for value in raw.get("categories") or []])
                    ),
                    constraints=constraints,
                    brand=brand,
                    price_bucket=price_bucket(raw.get("price")),
                    rating_bucket=rating_bucket(raw.get("average_rating")),
                    popularity_bucket=popularity_bucket(raw.get("rating_number")),
                )
            )
            raw_products.append(raw)
    return products, raw_products


def mapped_value(attribute: str, value: str, support: Counter[tuple[str, str]]) -> str:
    return value if support[(attribute, value)] >= MIN_VALUE_SUPPORT else RARE_VALUE


def make_states(products: Sequence[Product]) -> list[State]:
    states: list[State] = []
    for product_index, product in enumerate(products):
        constraints = tuple((attribute, value) for attribute, value, _ in product.constraints)
        unique_constraints = tuple(dict.fromkeys(constraints))
        first = unique_constraints[:1]
        first_two = unique_constraints[:2]
        all_values = unique_constraints[:4]
        old = unique_constraints[-1:] if unique_constraints else ()
        states.extend(
            (
                State(product_index, "browsing", "early", "pre", ()),
                State(product_index, "buying", "early", "pre", first),
                State(product_index, "browsing", "early", "pre", first),
                State(product_index, "browsing", "early", "pre", first_two),
                State(product_index, "browsing", "middle", "pre", all_values),
                State(product_index, "boundary", "early", "pre", ()),
                State(product_index, "provisional_override", "early", "pre", old),
                State(product_index, "intent_override", "early", "post", first),
            )
        )
    return states


def field_from_name(name: str) -> str:
    return name.split(":", 1)[1].split("=", 1)[0]


def build_feature_data(products: Sequence[Product]) -> FeatureData:
    constraint_support: Counter[tuple[str, str]] = Counter()
    brand_support: Counter[str] = Counter()
    for product in products:
        constraint_support.update(
            set((attribute, value) for attribute, value, _ in product.constraints)
        )
        if product.brand:
            brand_support[product.brand] += 1

    context_names_set: set[str] = {
        *(f"ctx:scenario={value}" for value in SCENARIOS),
        *(f"ctx:turn={value}" for value in TURN_BUCKETS),
        "ctx:override=pre",
        "ctx:override=post",
    }
    item_names_set: set[str] = set()
    for product in products:
        context_names_set.add(f"ctx:category={product.category}")
        item_names_set.add(f"item:category={product.category}")
        for attribute, value, _ in product.constraints:
            mapped = mapped_value(attribute, value, constraint_support)
            context_names_set.add(f"ctx:{attribute}={mapped}")
            item_names_set.add(f"item:{attribute}={mapped}")
        brand = (
            product.brand
            if product.brand and brand_support[product.brand] >= MIN_VALUE_SUPPORT
            else RARE_VALUE
        )
        item_names_set.add(f"item:brand={brand}")
        item_names_set.add(f"item:price={product.price_bucket}")
        item_names_set.add(f"item:rating={product.rating_bucket}")
        item_names_set.add(f"item:popularity={product.popularity_bucket}")

    # Always provide typed rare fallbacks even if one type has no rare values.
    for attribute in ATTRIBUTES:
        context_names_set.add(f"ctx:{attribute}={RARE_VALUE}")
        item_names_set.add(f"item:{attribute}={RARE_VALUE}")

    context_names = sorted(context_names_set)
    item_names = sorted(item_names_set)
    context_name_to_id = {name: index for index, name in enumerate(context_names)}
    item_name_to_id = {name: index for index, name in enumerate(item_names)}

    product_item_ids: list[tuple[int, ...]] = []
    item_rows: list[int] = []
    item_columns: list[int] = []
    for product_index, product in enumerate(products):
        names = {
            f"item:category={product.category}",
            f"item:price={product.price_bucket}",
            f"item:rating={product.rating_bucket}",
            f"item:popularity={product.popularity_bucket}",
        }
        brand = (
            product.brand
            if product.brand and brand_support[product.brand] >= MIN_VALUE_SUPPORT
            else RARE_VALUE
        )
        names.add(f"item:brand={brand}")
        for attribute, value, _ in product.constraints:
            names.add(
                f"item:{attribute}={mapped_value(attribute, value, constraint_support)}"
            )
        identifiers = tuple(sorted(item_name_to_id[name] for name in names))
        product_item_ids.append(identifiers)
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

    states = make_states(products)
    state_context_ids: list[tuple[int, ...]] = []
    state_rows: list[int] = []
    state_columns: list[int] = []
    for state_index, state in enumerate(states):
        product = products[state.product_index]
        names = {
            f"ctx:category={product.category}",
            f"ctx:scenario={state.scenario}",
            f"ctx:turn={state.turn_bucket}",
            f"ctx:override={state.override}",
        }
        for attribute, value in state.known_constraints:
            names.add(
                f"ctx:{attribute}={mapped_value(attribute, value, constraint_support)}"
            )
        identifiers = tuple(sorted(context_name_to_id[name] for name in names))
        state_context_ids.append(identifiers)
        state_rows.extend([state_index] * len(identifiers))
        state_columns.extend(identifiers)

    state_matrix = sparse.csr_matrix(
        (
            np.ones(len(state_rows), dtype=np.float32),
            (np.asarray(state_rows), np.asarray(state_columns)),
        ),
        shape=(len(states), len(context_names)),
        dtype=np.float32,
    )
    return FeatureData(
        context_names=context_names,
        item_names=item_names,
        context_fields=[field_from_name(name) for name in context_names],
        item_fields=[field_from_name(name) for name in item_names],
        context_name_to_id=context_name_to_id,
        item_name_to_id=item_name_to_id,
        product_item_ids=product_item_ids,
        item_matrix=item_matrix,
        states=states,
        state_context_ids=state_context_ids,
        state_matrix=state_matrix,
        positives=np.asarray([state.product_index for state in states], dtype=np.int32),
    )


def build_hard_negatives(
    products: Sequence[Product],
    feature_data: FeatureData,
    *,
    restrict_split: bool,
) -> np.ndarray:
    by_group: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_category: dict[str, list[int]] = defaultdict(list)
    for index, product in enumerate(products):
        split = split_for(product.parent_asin) if restrict_split else "all"
        by_group[(product.category, split)].append(index)
        by_category[product.category].append(index)
    for values in by_group.values():
        values.sort(key=lambda index: products[index].parent_asin)

    product_negatives: list[tuple[int, ...]] = []
    item_sets = [set(values) for values in feature_data.product_item_ids]
    for product_index, product in enumerate(products):
        split = split_for(product.parent_asin) if restrict_split else "all"
        candidates = [
            value
            for value in by_group[(product.category, split)]
            if value != product_index
        ]
        if not candidates and restrict_split:
            candidates = [
                value
                for value in by_category[product.category]
                if value != product_index
            ]
        if not candidates:
            product_negatives.append(())
            continue
        if len(candidates) > 64:
            start = stable_int(f"negative-pool\0{product.parent_asin}") % len(candidates)
            step = max(1, len(candidates) // 61)
            pool = list(dict.fromkeys(candidates[(start + step * offset) % len(candidates)] for offset in range(64)))
        else:
            pool = candidates
        target_features = item_sets[product_index]
        pool.sort(
            key=lambda candidate: (
                -len(target_features.intersection(item_sets[candidate])),
                stable_int(
                    f"negative-rank\0{product.parent_asin}\0{products[candidate].parent_asin}"
                ),
            )
        )
        selected = [pool[offset % len(pool)] for offset in range(NEGATIVES_PER_STATE)]
        product_negatives.append(tuple(selected))

    rows: list[tuple[int, ...]] = []
    for state in feature_data.states:
        values = product_negatives[state.product_index]
        rows.append(values if values else (state.product_index,) * NEGATIVES_PER_STATE)
    return np.asarray(rows, dtype=np.int32)


def subset_rows(feature_data: FeatureData, split: str | None) -> np.ndarray:
    if split is None:
        return np.arange(len(feature_data.states), dtype=np.int32)
    return np.asarray(
        [
            index
            for index, state in enumerate(feature_data.states)
            if split_for_product_index(feature_data, state.product_index) == split
        ],
        dtype=np.int32,
    )


_SPLIT_CACHE: list[str] = []


def split_for_product_index(feature_data: FeatureData, product_index: int) -> str:
    # Set by main after product loading; keeping this helper local avoids storing
    # redundant strings in every synthetic State.
    del feature_data
    return _SPLIT_CACHE[product_index]


def eligible_crosses(
    feature_data: FeatureData,
    rows: np.ndarray,
    negatives: np.ndarray,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
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
    x = feature_data.state_matrix[rows][:, context_ids]
    positives = feature_data.positives[rows]
    positive_counts = (x.T @ feature_data.item_matrix[positives][:, item_ids]).tocoo()

    context_exposure = np.asarray(x.sum(axis=0)).ravel().astype(np.int64)
    candidate_pairs: list[tuple[int, int]] = []
    positive_support: list[int] = []
    comparable_negative_support: list[int] = []
    for row, column, count in zip(
        positive_counts.row,
        positive_counts.col,
        positive_counts.data,
        strict=True,
    ):
        if int(count) >= MIN_CROSS_SUPPORT:
            candidate_pairs.append((int(context_ids[row]), int(item_ids[column])))
            positive_support.append(int(count))
            # A negative comparison is informative even when the candidate
            # does not activate the item-side feature: that absence is what
            # lets BPR increase a supported positive cross. Every fixed hard
            # negative exposed under the active context is therefore counted.
            comparable_negative_support.append(
                int(context_exposure[row]) * NEGATIVES_PER_STATE
            )

    if not candidate_pairs:
        return [], np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    negative_support = np.asarray(comparable_negative_support, dtype=np.int64)
    mask = negative_support >= MIN_CROSS_SUPPORT
    filtered_pairs = [pair for pair, keep in zip(candidate_pairs, mask, strict=True) if keep]
    return (
        filtered_pairs,
        np.asarray(positive_support, dtype=np.int32)[mask],
        negative_support.astype(np.int32)[mask],
    )


def cross_matrix(
    feature_data: FeatureData,
    rows: np.ndarray,
    product_indices: np.ndarray,
    pair_to_id: dict[tuple[int, int], int],
) -> sparse.csr_matrix:
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    for local_row, (state_index, product_index) in enumerate(
        zip(rows, product_indices, strict=True)
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
        shape=(len(rows), len(pair_to_id)),
        dtype=np.float32,
    )


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


@dataclass
class Parameters:
    context_vectors: np.ndarray
    item_vectors: np.ndarray
    item_linear: np.ndarray
    cross_weights: np.ndarray


def initialize_parameters(
    feature_data: FeatureData, cross_count: int, variant: str
) -> Parameters:
    rng = np.random.default_rng(SEED)
    context_vectors = rng.normal(
        0.0, 0.02, (len(feature_data.context_names), DIMENSION)
    ).astype(np.float32)
    item_vectors = rng.normal(
        0.0, 0.02, (len(feature_data.item_names), DIMENSION)
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


def train_model(
    feature_data: FeatureData,
    rows: np.ndarray,
    negatives: np.ndarray,
    cross_pairs: list[tuple[int, int]],
    epochs: int,
    *,
    validation_rows: np.ndarray | None = None,
    variant: str = "hybrid",
) -> tuple[Parameters, list[dict[str, float]], int]:
    if variant not in {"linear", "fm", "hybrid"}:
        raise ValueError("variant must be linear, fm, or hybrid")
    parameters = initialize_parameters(feature_data, len(cross_pairs), variant)
    optimizers = {
        "context": Adam(parameters.context_vectors.shape),
        "item": Adam(parameters.item_vectors.shape),
        "linear": Adam(parameters.item_linear.shape),
        "cross": Adam(parameters.cross_weights.shape),
    }
    pair_to_id = {pair: index for index, pair in enumerate(cross_pairs)}
    x = feature_data.state_matrix[rows]
    positives = feature_data.positives[rows]
    z_positive = cross_matrix(feature_data, rows, positives, pair_to_id)
    z_negatives = [
        cross_matrix(feature_data, rows, negatives[rows, column], pair_to_id)
        for column in range(NEGATIVES_PER_STATE)
    ]

    history: list[dict[str, float]] = []
    best_mrr = -1.0
    best_epoch = 1
    best_parameters: Parameters | None = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        negative = negatives[rows, (epoch - 1) % NEGATIVES_PER_STATE]
        item_sum, item_base = item_components(feature_data, parameters)
        context_sum = np.asarray(x @ parameters.context_vectors)
        z_difference = z_positive - z_negatives[(epoch - 1) % NEGATIVES_PER_STATE]
        delta = item_base[positives] - item_base[negative]
        delta += np.sum(context_sum * (item_sum[positives] - item_sum[negative]), axis=1)
        if len(cross_pairs):
            delta += np.asarray(z_difference @ parameters.cross_weights).ravel()
        clipped = np.clip(delta, -30.0, 30.0)
        q = 1.0 / (1.0 + np.exp(clipped))
        g = (-q / len(rows)).astype(np.float32)
        loss = float(np.mean(np.logaddexp(0.0, -delta)))

        difference = item_sum[positives] - item_sum[negative]
        gradient_context = np.asarray(x.T @ (g[:, None] * difference))
        gradient_context += FM_L2 * parameters.context_vectors

        relation = sparse.coo_matrix(
            (
                np.concatenate((g, -g)),
                (
                    np.concatenate((np.arange(len(rows)), np.arange(len(rows)))),
                    np.concatenate((positives, negative)),
                ),
            ),
            shape=(len(rows), len(feature_data.product_item_ids)),
        ).tocsr()
        product_scalar = np.asarray(relation.sum(axis=0)).ravel().astype(np.float32)
        product_context = np.asarray(relation.T @ context_sum)
        gradient_item = np.asarray(feature_data.item_matrix.T @ product_context)
        gradient_item += np.asarray(
            feature_data.item_matrix.T @ (product_scalar[:, None] * item_sum)
        )
        feature_scalar = np.asarray(feature_data.item_matrix.T @ product_scalar).ravel()
        gradient_item -= parameters.item_vectors * feature_scalar[:, None]
        gradient_item += FM_L2 * parameters.item_vectors
        gradient_linear = np.asarray(feature_data.item_matrix.T @ product_scalar).ravel()
        gradient_linear += FM_L2 * parameters.item_linear
        gradient_cross = (
            np.asarray(z_difference.T @ g).ravel()
            + CROSS_L2 * parameters.cross_weights
            if len(cross_pairs)
            else parameters.cross_weights
        )

        if variant != "linear":
            optimizers["context"].update(
                parameters.context_vectors, gradient_context.astype(np.float32), LEARNING_RATE
            )
            optimizers["item"].update(
                parameters.item_vectors, gradient_item.astype(np.float32), LEARNING_RATE
            )
        optimizers["linear"].update(
            parameters.item_linear, gradient_linear.astype(np.float32), LEARNING_RATE
        )
        if variant == "hybrid" and len(cross_pairs):
            optimizers["cross"].update(
                parameters.cross_weights,
                gradient_cross.astype(np.float32),
                LEARNING_RATE,
            )

        validation = (
            evaluate_sampled(
                feature_data,
                parameters,
                validation_rows,
                negatives,
                cross_pairs,
                max_states=5000,
            )
            if validation_rows is not None and len(validation_rows)
            else {"mrr": 0.0, "hit_rate_at_10": 0.0, "pairwise_accuracy": 0.0}
        )
        record = {
            "epoch": float(epoch),
            "loss": loss,
            **validation,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

        if validation_rows is None:
            best_epoch = epoch
            continue
        if validation["mrr"] > best_mrr + 1e-7:
            best_mrr = validation["mrr"]
            best_epoch = epoch
            stale_epochs = 0
            best_parameters = Parameters(
                *(np.array(value, copy=True) for value in (
                    parameters.context_vectors,
                    parameters.item_vectors,
                    parameters.item_linear,
                    parameters.cross_weights,
                ))
            )
        else:
            stale_epochs += 1
            if stale_epochs >= 3:
                break

    return best_parameters or parameters, history, best_epoch


def score_pair(
    feature_data: FeatureData,
    parameters: Parameters,
    state_index: int,
    product_index: int,
    item_sum: np.ndarray,
    item_base: np.ndarray,
    pair_to_id: dict[tuple[int, int], int],
) -> float:
    context_ids = feature_data.state_context_ids[state_index]
    context = np.sum(parameters.context_vectors[list(context_ids)], axis=0)
    score = float(item_base[product_index] + context @ item_sum[product_index])
    for context_id in context_ids:
        for item_id in feature_data.product_item_ids[product_index]:
            cross_id = pair_to_id.get((context_id, item_id))
            if cross_id is not None:
                score += float(parameters.cross_weights[cross_id])
    return score


def evaluate_sampled(
    feature_data: FeatureData,
    parameters: Parameters,
    rows: np.ndarray,
    negatives: np.ndarray,
    cross_pairs: list[tuple[int, int]],
    *,
    max_states: int,
) -> dict[str, float]:
    if len(rows) > max_states:
        rows = rows[
            np.linspace(0, len(rows) - 1, max_states, dtype=np.int32)
        ]
    item_sum, item_base = item_components(feature_data, parameters)
    pair_to_id = {pair: index for index, pair in enumerate(cross_pairs)}
    reciprocal_ranks: list[float] = []
    hits: list[int] = []
    pairwise: list[float] = []
    for state_index in rows:
        target = int(feature_data.positives[state_index])
        candidate_list = list(dict.fromkeys([target, *map(int, negatives[state_index])]))
        scored = [
            (
                score_pair(
                    feature_data,
                    parameters,
                    int(state_index),
                    product_index,
                    item_sum,
                    item_base,
                    pair_to_id,
                ),
                product_index,
            )
            for product_index in candidate_list
        ]
        scored.sort(key=lambda row: (-row[0], row[1]))
        rank = next(index for index, (_, value) in enumerate(scored, 1) if value == target)
        reciprocal_ranks.append(1.0 / rank)
        hits.append(int(rank <= 10))
        target_score = next(score for score, value in scored if value == target)
        negative_scores = [score for score, value in scored if value != target]
        pairwise.append(
            sum(target_score > value for value in negative_scores) / max(1, len(negative_scores))
        )
    return {
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "hit_rate_at_10": float(np.mean(hits)) if hits else 0.0,
        "pairwise_accuracy": float(np.mean(pairwise)) if pairwise else 0.0,
    }


def calibrate_temperature(
    feature_data: FeatureData,
    parameters: Parameters,
    rows: np.ndarray,
    negatives: np.ndarray,
    cross_pairs: list[tuple[int, int]],
) -> tuple[float, dict[str, float]]:
    if len(rows) > 5000:
        rows = rows[np.linspace(0, len(rows) - 1, 5000, dtype=np.int32)]
    item_sum, item_base = item_components(feature_data, parameters)
    pair_to_id = {pair: index for index, pair in enumerate(cross_pairs)}
    score_sets: list[list[float]] = []
    for state_index in rows:
        target = int(feature_data.positives[state_index])
        candidates = list(dict.fromkeys([target, *map(int, negatives[state_index])]))
        # Put target first for the NLL calculation.
        candidates.remove(target)
        candidates.insert(0, target)
        score_sets.append(
            [
                score_pair(
                    feature_data,
                    parameters,
                    int(state_index),
                    product_index,
                    item_sum,
                    item_base,
                    pair_to_id,
                )
                for product_index in candidates
            ]
        )
    losses: dict[str, float] = {}
    for temperature in (0.25, 0.5, 1.0, 2.0, 4.0):
        nll: list[float] = []
        for scores in score_sets:
            values = np.asarray(scores) / temperature
            values -= np.max(values)
            nll.append(float(-values[0] + np.log(np.exp(values).sum())))
        losses[str(temperature)] = float(np.mean(nll)) if nll else 0.0
    best = min((float(key) for key in losses), key=lambda value: losses[str(value)])
    return best, losses


def write_artifact(
    output_path: Path,
    catalog_path: Path,
    products: Sequence[Product],
    feature_data: FeatureData,
    parameters: Parameters,
    cross_pairs: list[tuple[int, int]],
    positive_support: np.ndarray,
    negative_support: np.ndarray,
    temperature: float,
    selected_epoch: int,
    variant: str,
) -> None:
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
                PRIMARY KEY(context_feature_id, item_feature_id)
            ) WITHOUT ROWID;
            CREATE TABLE reply_values(
                parent_asin TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                attribute TEXT NOT NULL,
                normalized_value TEXT NOT NULL,
                PRIMARY KEY(parent_asin, ordinal)
            ) WITHOUT ROWID;
            """
        )
        metadata = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_type": {
                "linear": "linear_pairwise_ranker",
                "fm": "second_order_factorization_machine",
                "hybrid": "second_order_fm_plus_explicit_crosses",
            }[variant],
            "catalog_sha256": sha256_path(catalog_path),
            "product_count": str(len(products)),
            "dimension": str(DIMENSION),
            "temperature": str(temperature),
            "selected_epoch": str(selected_epoch),
            "seed": str(SEED),
            "minimum_value_support": str(MIN_VALUE_SUPPORT),
            "minimum_cross_support": str(MIN_CROSS_SUPPORT),
            "negatives_per_state": str(NEGATIVES_PER_STATE),
            "fm_l2": str(FM_L2),
            "cross_l2": str(CROSS_L2),
            "learning_rate": str(LEARNING_RATE),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        connection.executemany(
            "INSERT INTO context_features(feature_id,name,field,vector) VALUES(?,?,?,?)",
            (
                (
                    feature_id,
                    name,
                    feature_data.context_fields[feature_id],
                    sqlite3.Binary(parameters.context_vectors[feature_id].astype("<f4").tobytes()),
                )
                for feature_id, name in enumerate(feature_data.context_names)
            ),
        )
        connection.executemany(
            "INSERT INTO item_features(feature_id,name,field,linear_weight,vector) VALUES(?,?,?,?,?)",
            (
                (
                    feature_id,
                    name,
                    feature_data.item_fields[feature_id],
                    float(parameters.item_linear[feature_id]),
                    sqlite3.Binary(parameters.item_vectors[feature_id].astype("<f4").tobytes()),
                )
                for feature_id, name in enumerate(feature_data.item_names)
            ),
        )
        item_sum, item_base = item_components(feature_data, parameters)
        connection.executemany(
            "INSERT INTO products(parent_asin,base_score,vector,item_feature_ids) VALUES(?,?,?,?)",
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
                for index, product in enumerate(products)
            ),
        )
        connection.executemany(
            """
            INSERT INTO cross_weights(
                context_feature_id,item_feature_id,positive_support,negative_support,weight
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
        for product in products:
            for ordinal, (attribute, normalized, _) in enumerate(product.constraints):
                reply_rows.append((product.parent_asin, ordinal, attribute, normalized))
        connection.executemany(
            "INSERT INTO reply_values(parent_asin,ordinal,attribute,normalized_value) VALUES(?,?,?,?)",
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
    cross_pairs: list[tuple[int, int]],
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
    rows.sort(key=lambda row: (-float(row["absolute_weight"]), str(row["context_feature"]), str(row["item_feature"])))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "context_feature", "item_feature", "field_pair", "positive_support",
            "negative_support", "learned_weight", "absolute_weight",
        ])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global SEED
    parser = argparse.ArgumentParser(description="Train the Approach 1 hybrid FM")
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data/catalog.jsonl")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("fm_model.sqlite3"))
    parser.add_argument("--metrics", type=Path, default=Path(__file__).with_name("training_metrics.json"))
    parser.add_argument("--cross-audit", type=Path, default=Path(__file__).with_name("cross_weights.csv"))
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--skip-full-retrain", action="store_true")
    parser.add_argument(
        "--variant", choices=("linear", "fm", "hybrid"), default="hybrid"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    SEED = int(args.seed)
    started = time.time()
    products, _ = load_products(args.catalog)
    global _SPLIT_CACHE
    _SPLIT_CACHE = [split_for(product.parent_asin) for product in products]
    print(f"Loaded {len(products)} products", flush=True)
    feature_data = build_feature_data(products)
    negatives = build_hard_negatives(products, feature_data, restrict_split=True)
    train_rows = subset_rows(feature_data, "train")
    validation_rows = subset_rows(feature_data, "validation")
    test_rows = subset_rows(feature_data, "test")
    print(
        f"States train={len(train_rows)} validation={len(validation_rows)} test={len(test_rows)}",
        flush=True,
    )
    if args.variant == "hybrid":
        cross_pairs, positive_support, negative_support = eligible_crosses(
            feature_data, train_rows, negatives
        )
    else:
        cross_pairs = []
        positive_support = np.zeros(0, dtype=np.int32)
        negative_support = np.zeros(0, dtype=np.int32)
    print(
        f"Features context={len(feature_data.context_names)} item={len(feature_data.item_names)} "
        f"explicit_crosses={len(cross_pairs)}",
        flush=True,
    )

    validation_parameters, history, selected_epoch = train_model(
        feature_data,
        train_rows,
        negatives,
        cross_pairs,
        args.max_epochs,
        validation_rows=validation_rows,
        variant=args.variant,
    )
    internal_validation = evaluate_sampled(
        feature_data,
        validation_parameters,
        validation_rows,
        negatives,
        cross_pairs,
        max_states=10000,
    )
    internal_test = evaluate_sampled(
        feature_data,
        validation_parameters,
        test_rows,
        negatives,
        cross_pairs,
        max_states=10000,
    )
    temperature, temperature_losses = calibrate_temperature(
        feature_data,
        validation_parameters,
        validation_rows,
        negatives,
        cross_pairs,
    )

    if args.skip_full_retrain:
        final_parameters = validation_parameters
    else:
        all_rows = subset_rows(feature_data, None)
        final_parameters, _, _ = train_model(
            feature_data,
            all_rows,
            negatives,
            cross_pairs,
            selected_epoch,
            validation_rows=None,
            variant=args.variant,
        )

    write_artifact(
        args.output,
        args.catalog,
        products,
        feature_data,
        final_parameters,
        cross_pairs,
        positive_support,
        negative_support,
        temperature,
        selected_epoch,
        args.variant,
    )
    write_cross_audit(
        args.cross_audit,
        feature_data,
        final_parameters,
        cross_pairs,
        positive_support,
        negative_support,
    )
    report = {
        "model": args.variant,
        "seed": SEED,
        "catalog_sha256": sha256_path(args.catalog),
        "product_count": len(products),
        "state_count": len(feature_data.states),
        "train_state_count": len(train_rows),
        "validation_state_count": len(validation_rows),
        "test_state_count": len(test_rows),
        "context_feature_count": len(feature_data.context_names),
        "item_feature_count": len(feature_data.item_names),
        "explicit_cross_count": len(cross_pairs),
        "selected_epoch": selected_epoch,
        "temperature": temperature,
        "temperature_nll": temperature_losses,
        "internal_validation": internal_validation,
        "internal_test": internal_test,
        "training_history": history,
        "elapsed_seconds": round(time.time() - started, 3),
        "artifact": str(args.output),
    }
    args.metrics.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
