from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from array import array
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPOSITORY_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

from src.attribute_index import attribute_values, normalize_value  # noqa: E402
from src.category_index import coarse_category  # noqa: E402
from src.conversation_features import (  # noqa: E402
    classify_constraint,
    normalize_constraint,
)
from src.hybrid_model import turn_bucket  # noqa: E402


DATASET_VERSION = "fm-trajectories-v2"
SPLITS = ("train", "validation", "test")
SPLIT_COUNTS_PER_TEN = {"train": 8, "validation": 1, "test": 1}
SCENARIOS = ("buying", "browsing", "boundary", "intent_override")
PUBLIC_SCENARIO_COUNTS_PER_TWENTY = {
    "buying": 8,
    "browsing": 8,
    "boundary": 1,
    "intent_override": 3,
}
BALANCED_SCENARIO_COUNTS_PER_TWENTY = {
    "buying": 5,
    "browsing": 5,
    "boundary": 5,
    "intent_override": 5,
}
SCENARIO_MIXES = {
    "public": PUBLIC_SCENARIO_COUNTS_PER_TWENTY,
    "balanced": BALANCED_SCENARIO_COUNTS_PER_TWENTY,
}
TRAJECTORY_BLOCK_SIZE = 20
TURN_BUCKETS = ("early", "middle", "late")
SURVIVOR_WIDTH_BUCKETS = ("<=10", "11-50", "51-200", ">200")
QUESTION_ATTRIBUTES = (
    "use_case",
    "feature",
    "style",
    "material",
    "size",
    "budget",
    "brand",
    "color",
    "other",
)
ROADMAP_STAGES = (
    ("use_case",),
    ("feature", "style", "material"),
    ("size", "budget", "brand"),
    ("color",),
    ("other",),
)


Constraint = tuple[str, str, str]
KnownConstraints = tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class Product:
    parent_asin: str
    category: str
    constraints: tuple[Constraint, ...]
    brand: str | None
    price_bucket: str
    rating_bucket: str
    popularity_bucket: str
    hard_constraint_count: int = 2

    def __post_init__(self) -> None:
        if not 0 <= self.hard_constraint_count <= len(self.constraints):
            raise ValueError("hard_constraint_count is outside constraints")

    @property
    def hard_constraints(self) -> tuple[Constraint, ...]:
        return self.constraints[: self.hard_constraint_count]

    @property
    def soft_preferences(self) -> tuple[Constraint, ...]:
        return self.constraints[self.hard_constraint_count :]

    def answer_values(
        self, attribute: str, disclosed_values: set[str]
    ) -> tuple[str, ...]:
        """Return the evaluator-compatible answer to one attribute question."""

        matches: list[str] = []
        seen: set[str] = set()
        for candidate_attribute, normalized, display in self.constraints:
            if normalized in disclosed_values or normalized in seen:
                continue
            if attribute != "other" and candidate_attribute != attribute:
                continue
            seen.add(normalized)
            matches.append(display)
            if len(matches) == 2:
                break
        return tuple(matches)


@dataclass(frozen=True)
class TrajectoryConfig:
    trajectory_count: int = 25_000
    seed: int = 2026
    split_seed: int = 2026
    scenario_mix: str = "public"
    max_turns: int = 10
    extended_fraction: float = 0.10
    dataset_version: str = DATASET_VERSION

    def __post_init__(self) -> None:
        if self.trajectory_count <= 0:
            raise ValueError("trajectory_count must be positive")
        if self.trajectory_count % TRAJECTORY_BLOCK_SIZE:
            raise ValueError(
                f"trajectory_count must be divisible by {TRAJECTORY_BLOCK_SIZE} "
                "so every nested prefix has its exact requested mix"
            )
        if self.scenario_mix not in SCENARIO_MIXES:
            raise ValueError(
                f"scenario_mix must be one of: {', '.join(sorted(SCENARIO_MIXES))}"
            )
        if not 1 <= self.max_turns <= 10:
            raise ValueError("max_turns must be between 1 and 10")
        if not 0.0 <= self.extended_fraction <= 1.0:
            raise ValueError("extended_fraction must be between 0 and 1")


@dataclass(frozen=True)
class TrajectoryState:
    trajectory_id: int
    state_index: int
    product_index: int
    target_parent_asin: str
    split: str
    scenario: str
    scenario_state: str
    turn: int
    turn_bucket: str
    intent_epoch: int
    known_constraints: KnownConstraints
    asked_attribute: str | None
    has_other_answer: bool
    survivor_count: int
    state_weight: float
    extended_trajectory: bool
    after_normal_cutoff: bool

    def known_constraint_mapping(self) -> dict[str, tuple[str, ...]]:
        """Return the shape consumed by the shared context-feature builder."""

        return dict(self.known_constraints)


@dataclass(frozen=True)
class TrajectoryDataset:
    config: TrajectoryConfig
    products: tuple[Product, ...]
    product_split_labels: tuple[str, ...]
    states: tuple[TrajectoryState, ...]
    survivor_values: np.ndarray
    survivor_offsets: np.ndarray
    trajectory_state_offsets: np.ndarray
    trajectory_targets: np.ndarray
    trajectory_scenarios: tuple[str, ...]
    trajectory_splits: tuple[str, ...]
    input_hashes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.survivor_values.dtype != np.uint32:
            raise TypeError("survivor_values must use uint32")
        if len(self.survivor_offsets) != len(self.states) + 1:
            raise ValueError("survivor_offsets must contain one boundary per state")
        if len(self.trajectory_state_offsets) != self.config.trajectory_count + 1:
            raise ValueError(
                "trajectory_state_offsets must contain one boundary per trajectory"
            )
        if int(self.survivor_offsets[-1]) != len(self.survivor_values):
            raise ValueError("the final survivor offset does not match survivor_values")
        if int(self.trajectory_state_offsets[-1]) != len(self.states):
            raise ValueError("the final trajectory offset does not match states")
        if len(self.trajectory_targets) != self.config.trajectory_count:
            raise ValueError("trajectory_targets length does not match the config")
        if len(self.trajectory_scenarios) != self.config.trajectory_count:
            raise ValueError("trajectory_scenarios length does not match the config")
        if len(self.trajectory_splits) != self.config.trajectory_count:
            raise ValueError("trajectory_splits length does not match the config")

    @property
    def trajectory_count(self) -> int:
        return self.config.trajectory_count

    def state_survivors(self, index: int) -> np.ndarray:
        """Return a zero-copy, read-only view of one state's complete survivors."""

        if index < 0:
            index += len(self.states)
        if not 0 <= index < len(self.states):
            raise IndexError(index)
        start = int(self.survivor_offsets[index])
        stop = int(self.survivor_offsets[index + 1])
        result = self.survivor_values[start:stop].view()
        result.flags.writeable = False
        return result

    def prefix(self, trajectory_count: int) -> "TrajectoryDataset":
        """Return a nested prefix sharing the full dataset's compact arrays."""

        if trajectory_count == self.trajectory_count:
            return self
        if not 0 < trajectory_count < self.trajectory_count:
            raise ValueError("trajectory_count must select a non-empty strict prefix")
        if trajectory_count % TRAJECTORY_BLOCK_SIZE:
            raise ValueError(
                f"trajectory_count must be divisible by {TRAJECTORY_BLOCK_SIZE}"
            )
        state_stop = int(self.trajectory_state_offsets[trajectory_count])
        survivor_stop = int(self.survivor_offsets[state_stop])
        return TrajectoryDataset(
            config=replace(self.config, trajectory_count=trajectory_count),
            products=self.products,
            product_split_labels=self.product_split_labels,
            states=self.states[:state_stop],
            survivor_values=self.survivor_values[:survivor_stop],
            survivor_offsets=self.survivor_offsets[: state_stop + 1],
            trajectory_state_offsets=self.trajectory_state_offsets[
                : trajectory_count + 1
            ],
            trajectory_targets=self.trajectory_targets[:trajectory_count],
            trajectory_scenarios=self.trajectory_scenarios[:trajectory_count],
            trajectory_splits=self.trajectory_splits[:trajectory_count],
            input_hashes=self.input_hashes,
        )

    def manifest(
        self, input_paths: Mapping[str, str | Path] | None = None
    ) -> dict[str, object]:
        return manifest(self, input_paths=input_paths)


@dataclass(frozen=True)
class _CatalogPostings:
    category_pools: Mapping[str, np.ndarray]
    attribute_postings: Mapping[str, Mapping[str, np.ndarray]]
    reply_values: Mapping[str, tuple[tuple[str, ...], ...]]


@dataclass
class _MutableSession:
    scenario_state: str
    survivors: np.ndarray
    known_constraints: dict[str, list[str]]
    unindexed_values: set[tuple[str, str]]
    remaining_attributes: set[str]
    disclosed_values: set[str]
    intent_epoch: int = 0
    boundary_used: bool = False


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def split_for(parent_asin: str, seed: int = 2026) -> str:
    """Return a stable hash split when category context is unavailable."""

    bucket = stable_int(f"split\0{seed}\0{parent_asin}") % 10
    return "train" if bucket < 8 else ("validation" if bucket == 8 else "test")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_catalog_records(raw_products: Sequence[dict]) -> str:
    """Hash the parsed catalog content when an exact source path is unavailable."""

    digest = hashlib.sha256()
    for raw in raw_products:
        digest.update(
            json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def price_bucket(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "missing"
    price = float(value)
    lower = 0
    for upper in (10, 20, 35, 50, 75, 100, 150, 250):
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


def _constraints_from_values(values: Mapping[str, Sequence[str]]) -> tuple[Constraint, ...]:
    constraints: list[Constraint] = []
    for display in values.get("other", ()):
        normalized = normalize_constraint(display)
        if normalized:
            constraints.append((classify_constraint(display), normalized, str(display)))
    return tuple(constraints)


def load_products(catalog_path: str | Path) -> tuple[list[Product], list[dict]]:
    """Load catalog products without reading any public or private sessions."""

    products: list[Product] = []
    raw_products: list[dict] = []
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            parent_asin = str(raw.get("parent_asin", "")).strip()
            if not parent_asin:
                raise ValueError(f"missing parent_asin on catalog line {line_number}")
            values = attribute_values(raw)
            constraints = _constraints_from_values(values)
            hard = constraints[:2]
            soft = constraints[2:4] or constraints[:1]
            store = str(raw.get("store") or "").strip()
            products.append(
                Product(
                    parent_asin=parent_asin,
                    category=coarse_category(
                        [str(value) for value in raw.get("categories") or []]
                    ),
                    constraints=hard + soft,
                    brand=normalize_value(store) if store else None,
                    price_bucket=price_bucket(raw.get("price")),
                    rating_bucket=rating_bucket(raw.get("average_rating")),
                    popularity_bucket=popularity_bucket(raw.get("rating_number")),
                    hard_constraint_count=len(hard),
                )
            )
            raw_products.append(raw)
    if len({product.parent_asin for product in products}) != len(products):
        raise ValueError("catalog contains duplicate parent_asin values")
    return products, raw_products


def _counted_labels(
    counts: Mapping[str, int], *, seed_key: str
) -> list[str]:
    occurrences = [
        (label, ordinal)
        for label, count in counts.items()
        for ordinal in range(count)
    ]
    occurrences.sort(
        key=lambda item: (
            stable_int(f"{seed_key}\0{item[0]}\0{item[1]}"),
            item,
        )
    )
    return [label for label, _ in occurrences]


def _nearest_ratio_counts(total: int, counts: Mapping[str, int]) -> dict[str, int]:
    denominator = sum(counts.values())
    exact = {name: total * value / denominator for name, value in counts.items()}
    result = {name: int(math.floor(value)) for name, value in exact.items()}
    missing = total - sum(result.values())
    order = sorted(
        counts,
        key=lambda name: (-(exact[name] - result[name]), tuple(counts).index(name)),
    )
    for name in order[:missing]:
        result[name] += 1
    return result


def _ratio_schedule(
    total: int,
    counts: Mapping[str, int],
    *,
    seed: int,
    purpose: str,
) -> tuple[str, ...]:
    block_size = sum(counts.values())
    labels: list[str] = []
    full_blocks, remainder = divmod(total, block_size)
    for block in range(full_blocks):
        labels.extend(
            _counted_labels(
                counts,
                seed_key=f"{purpose}\0{seed}\0block\0{block}",
            )
        )
    if remainder:
        partial_counts = _nearest_ratio_counts(remainder, counts)
        labels.extend(
            _counted_labels(
                partial_counts,
                seed_key=f"{purpose}\0{seed}\0partial\0{full_blocks}",
            )
        )
    return tuple(labels)


def _category_split_allocations(
    category_sizes: Mapping[str, int], seed: int
) -> dict[str, dict[str, int]]:
    """Apportion exact global split totals into near-proportional category rows.

    Each category starts with Hamilton (largest-remainder) apportionment of its
    own 80/10/10 ideal.  If those independently rounded rows miss the required
    global totals, deterministic within-category transfers repeatedly choose
    the smallest available change in squared distance from the ideal.  All cost
    calculations use integer tenths so the result does not depend on
    floating-point behavior.
    """

    if any(size <= 0 for size in category_sizes.values()):
        raise ValueError("category sizes must be positive")
    total = sum(category_sizes.values())
    global_targets = _nearest_ratio_counts(total, SPLIT_COUNTS_PER_TEN)
    denominator = sum(SPLIT_COUNTS_PER_TEN.values())
    allocations: dict[str, dict[str, int]] = {}

    for category in sorted(category_sizes):
        size = int(category_sizes[category])
        counts = {
            split: size * SPLIT_COUNTS_PER_TEN[split] // denominator
            for split in SPLITS
        }
        missing = size - sum(counts.values())
        remainder_order = sorted(
            SPLITS,
            key=lambda split: (
                -(size * SPLIT_COUNTS_PER_TEN[split] % denominator),
                stable_int(
                    f"category-split-remainder\0{seed}\0{category}\0{split}"
                ),
                SPLITS.index(split),
            ),
        )
        for split in remainder_order[:missing]:
            counts[split] += 1
        allocations[category] = counts

    global_counts = {
        split: sum(counts[split] for counts in allocations.values())
        for split in SPLITS
    }
    protect_train_coverage = global_targets["train"] >= len(category_sizes)

    def cell_error(category: str, split: str, count: int) -> int:
        scaled_difference = (
            denominator * count
            - category_sizes[category] * SPLIT_COUNTS_PER_TEN[split]
        )
        return scaled_difference * scaled_difference

    while global_counts != global_targets:
        surplus_splits = [
            split
            for split in SPLITS
            if global_counts[split] > global_targets[split]
        ]
        deficit_splits = [
            split
            for split in SPLITS
            if global_counts[split] < global_targets[split]
        ]
        options: list[tuple[int, int, str, str, str]] = []
        for category in sorted(category_sizes):
            counts = allocations[category]
            for source in surplus_splits:
                lower_bound = (
                    1 if protect_train_coverage and source == "train" else 0
                )
                if counts[source] <= lower_bound:
                    continue
                for destination in deficit_splits:
                    delta = (
                        cell_error(category, source, counts[source] - 1)
                        - cell_error(category, source, counts[source])
                        + cell_error(
                            category, destination, counts[destination] + 1
                        )
                        - cell_error(category, destination, counts[destination])
                    )
                    options.append(
                        (
                            delta,
                            stable_int(
                                "category-split-transfer\0"
                                f"{seed}\0{category}\0{source}\0{destination}"
                            ),
                            category,
                            source,
                            destination,
                        )
                    )
        if not options:
            raise AssertionError("category split apportionment became infeasible")
        _, _, category, source, destination = min(options)
        allocations[category][source] -= 1
        allocations[category][destination] += 1
        global_counts[source] -= 1
        global_counts[destination] += 1

    if protect_train_coverage and any(
        counts["train"] < 1 for counts in allocations.values()
    ):
        raise AssertionError("a feasible category was left without train products")
    return allocations


def build_product_splits(
    products: Sequence[Product], seed: int = 2026
) -> dict[str, tuple[int, ...]]:
    """Build deterministic, category-stratified 80/10/10 product splits.

    The exact global totals are preserved while each category receives the
    closest practical integer allocation.  When the global train quota can
    cover every nonempty category, at least one product from each is protected
    in train.
    """

    by_category: dict[str, list[int]] = defaultdict(list)
    for index, product in enumerate(products):
        by_category[product.category].append(index)

    allocations = _category_split_allocations(
        {category: len(indices) for category, indices in by_category.items()},
        seed,
    )
    category_order = sorted(
        by_category,
        key=lambda category: (
            stable_int(f"category-split\0{seed}\0{category}"),
            category,
        ),
    )
    result: dict[str, list[int]] = {split: [] for split in SPLITS}
    for category in category_order:
        indices = sorted(
            by_category[category],
            key=lambda index: (
                stable_int(
                    f"product-split-order\0{seed}\0{category}\0"
                    f"{products[index].parent_asin}"
                ),
                products[index].parent_asin,
            ),
        )
        category_labels = _counted_labels(
            allocations[category],
            seed_key=f"product-split-labels\0{seed}\0{category}",
        )
        for index, split in zip(indices, category_labels, strict=True):
            result[split].append(index)

    frozen = {split: tuple(sorted(result[split])) for split in SPLITS}
    expected_totals = _nearest_ratio_counts(
        len(products), SPLIT_COUNTS_PER_TEN
    )
    if {split: len(frozen[split]) for split in SPLITS} != expected_totals:
        raise AssertionError("product splits do not have exact global totals")
    sets = {split: set(indices) for split, indices in frozen.items()}
    if any(
        sets[left].intersection(sets[right])
        for left in SPLITS
        for right in SPLITS
        if left < right
    ):
        raise AssertionError("product splits overlap")
    if set().union(*sets.values()) != set(range(len(products))):
        raise AssertionError("product splits do not cover the catalog")
    return frozen


def allocate_scenarios(
    trajectory_count: int, scenario_mix: str = "public", seed: int = 2026
) -> tuple[str, ...]:
    """Return exact per-20-block scenarios, making all valid prefixes nested."""

    if trajectory_count <= 0 or trajectory_count % TRAJECTORY_BLOCK_SIZE:
        raise ValueError(
            f"trajectory_count must be positive and divisible by {TRAJECTORY_BLOCK_SIZE}"
        )
    try:
        counts = SCENARIO_MIXES[scenario_mix]
    except KeyError as error:
        raise ValueError(f"unknown scenario mix: {scenario_mix}") from error
    return _ratio_schedule(
        trajectory_count,
        counts,
        seed=seed,
        purpose=f"scenario-{scenario_mix}",
    )


scenario_allocation = allocate_scenarios


def allocate_trajectory_splits(
    trajectory_count: int, seed: int = 2026
) -> tuple[str, ...]:
    """Return exact 80/10/10 labels for every valid 20-trajectory prefix."""

    if trajectory_count <= 0 or trajectory_count % TRAJECTORY_BLOCK_SIZE:
        raise ValueError(
            f"trajectory_count must be positive and divisible by {TRAJECTORY_BLOCK_SIZE}"
        )
    counts = {"train": 16, "validation": 2, "test": 2}
    return _ratio_schedule(
        trajectory_count,
        counts,
        seed=seed,
        purpose="trajectory-splits",
    )


def _validate_product_split_mapping(
    products: Sequence[Product], product_splits: Mapping[str, Sequence[int]]
) -> tuple[str, ...]:
    labels: list[str | None] = [None] * len(products)
    for split in SPLITS:
        for raw_index in product_splits.get(split, ()):
            index = int(raw_index)
            if not 0 <= index < len(products):
                raise ValueError(f"invalid product index {index} in {split}")
            if labels[index] is not None:
                raise ValueError(f"product index {index} occurs in multiple splits")
            labels[index] = split
    if any(value is None for value in labels):
        raise ValueError("product split mapping does not cover every product")
    return tuple(str(value) for value in labels)


def select_trajectory_targets(
    products: Sequence[Product],
    product_splits: Mapping[str, Sequence[int]],
    trajectory_splits: Sequence[str],
    seed: int = 2026,
) -> np.ndarray:
    """Select targets uniformly within each split using reproducible cycles."""

    for split in SPLITS:
        if not product_splits.get(split):
            raise ValueError(f"cannot select targets from empty {split} split")
    occurrences: Counter[str] = Counter()
    cycle_orders: dict[tuple[str, int], tuple[int, ...]] = {}
    selected = np.empty(len(trajectory_splits), dtype=np.uint32)
    for trajectory_id, split in enumerate(trajectory_splits):
        if split not in SPLITS:
            raise ValueError(f"unknown trajectory split: {split}")
        pool = product_splits[split]
        occurrence = occurrences[split]
        occurrences[split] += 1
        cycle, position = divmod(occurrence, len(pool))
        key = (split, cycle)
        order = cycle_orders.get(key)
        if order is None:
            order = tuple(
                sorted(
                    (int(index) for index in pool),
                    key=lambda index: (
                        stable_int(
                            f"trajectory-target\0{seed}\0{split}\0{cycle}\0"
                            f"{products[index].parent_asin}"
                        ),
                        products[index].parent_asin,
                    ),
                )
            )
            cycle_orders[key] = order
        selected[trajectory_id] = order[position]
    return selected


def build_catalog_postings(
    products: Sequence[Product], raw_products: Sequence[dict]
) -> _CatalogPostings:
    """Build exact in-memory postings from the runtime attribute extractor."""

    if len(products) != len(raw_products):
        raise ValueError("products and raw_products must be aligned")
    category_lists: dict[str, list[int]] = defaultdict(list)
    posting_lists: dict[str, dict[str, list[int]]] = {
        attribute: defaultdict(list) for attribute in QUESTION_ATTRIBUTES
    }
    for product_index, (product, raw) in enumerate(
        zip(products, raw_products, strict=True)
    ):
        if str(raw.get("parent_asin", "")).strip() != product.parent_asin:
            raise ValueError("products and raw_products have different order")
        category_lists[product.category].append(product_index)
        values = attribute_values(raw)
        for attribute in QUESTION_ATTRIBUTES:
            normalized_values = {
                normalized
                for value in values.get(attribute, ())
                if (normalized := normalize_constraint(value))
            }
            for normalized in normalized_values:
                posting_lists[attribute][normalized].append(product_index)

    category_pools = {
        category: np.asarray(indices, dtype=np.uint32)
        for category, indices in category_lists.items()
    }
    attribute_postings = {
        attribute: {
            value: np.asarray(indices, dtype=np.uint32)
            for value, indices in values.items()
        }
        for attribute, values in posting_lists.items()
    }
    # These are the exact normalized replies exposed by the trained runtime
    # model.  Keeping every value (rather than only the first two) lets a later
    # state remove already-disclosed values before applying the runtime's
    # two-value reply cap.
    reply_values = {
        attribute: tuple(
            tuple(
                dict.fromkeys(
                    normalized
                    for candidate_attribute, normalized, _ in product.constraints
                    if attribute == "other" or candidate_attribute == attribute
                )
            )
            for product in products
        )
        for attribute in QUESTION_ATTRIBUTES
    }
    return _CatalogPostings(category_pools, attribute_postings, reply_values)


def _clean_display(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip(" -;,.\t\n")


def _record_constraint(
    session: _MutableSession, attribute: str, display_value: str
) -> None:
    values = session.known_constraints.setdefault(attribute, [])
    if display_value not in values:
        values.append(display_value)


def _apply_values(
    session: _MutableSession,
    postings: _CatalogPostings,
    attribute: str,
    values: Sequence[str],
) -> bool:
    """Mirror runtime AND filtering and its empty-intersection rollback."""

    cleaned_values = list(
        dict.fromkeys(
            cleaned for value in values if (cleaned := _clean_display(value))
        )
    )
    if not cleaned_values:
        return False
    previous = session.survivors
    filtered = previous
    missing: list[str] = []
    mapping = postings.attribute_postings[attribute]
    for display in cleaned_values:
        _record_constraint(session, attribute, display)
        normalized = normalize_constraint(display)
        candidate_postings = mapping.get(normalized)
        if candidate_postings is None:
            missing.append(display)
            filtered = np.empty(0, dtype=np.uint32)
            continue
        if len(filtered):
            filtered = np.intersect1d(
                filtered, candidate_postings, assume_unique=True
            ).astype(np.uint32, copy=False)
    if len(filtered):
        session.survivors = filtered
        return True
    rejected = missing or cleaned_values
    session.unindexed_values.update(
        (attribute, normalize_constraint(display)) for display in rejected
    )
    session.survivors = previous
    return False


def _advance_past_stage(session: _MutableSession, attribute: str) -> None:
    for stage_index, stage in enumerate(ROADMAP_STAGES):
        if attribute not in stage:
            continue
        for completed_stage in ROADMAP_STAGES[: stage_index + 1]:
            session.remaining_attributes.difference_update(completed_stage)
        return


def _freeze_known_constraints(session: _MutableSession) -> KnownConstraints:
    return tuple(
        (attribute, tuple(values))
        for attribute, values in sorted(session.known_constraints.items())
    )


def _informative_attributes(
    session: _MutableSession,
    postings: _CatalogPostings,
) -> list[str]:
    """Return fields whose predicted replies vary among visible survivors.

    Question choice must be a policy decision based on conversation-visible
    state.  In particular, it cannot inspect the labeled target's undisclosed
    reply or preview the target-specific survivor intersection.  This mirrors
    the runtime model's ``predicted_reply`` bucketing, but uses an unweighted
    survivor distribution because the trajectory simulator has no ranker
    posterior of its own.
    """

    result: list[str] = []
    for attribute in QUESTION_ATTRIBUTES:
        if attribute not in session.remaining_attributes:
            continue
        first_reply: tuple[str, ...] | None = None
        for raw_product_index in session.survivors:
            product_index = int(raw_product_index)
            reply = tuple(
                value
                for value in postings.reply_values[attribute][product_index]
                if value not in session.disclosed_values
            )[:2]
            if first_reply is None:
                first_reply = reply
            elif reply != first_reply:
                result.append(attribute)
                break
    return result


def _roadmap_candidates(attributes: Sequence[str]) -> list[str]:
    """Return candidates from the earliest roadmap stage that can help."""

    available = set(attributes)
    for stage in ROADMAP_STAGES:
        candidates = [attribute for attribute in stage if attribute in available]
        if candidates:
            return candidates
    return []


def _choose_attribute(
    attributes: Sequence[str], *, seed: int, trajectory_id: int, turn: int, phase: str
) -> str | None:
    if not attributes:
        return None
    ordered = sorted(set(attributes))
    choice = stable_int(
        f"question\0{seed}\0{trajectory_id}\0{turn}\0{phase}"
    ) % len(ordered)
    return ordered[choice]


def _is_extended(config: TrajectoryConfig, trajectory_id: int, scenario: str) -> bool:
    if config.extended_fraction <= 0:
        return False
    if config.extended_fraction >= 1:
        return True
    threshold = int(config.extended_fraction * 1_000_000)
    draw = stable_int(
        f"extended\0{config.seed}\0{scenario}\0{trajectory_id}"
    ) % 1_000_000
    return draw < threshold


def _initialize_session(
    product: Product,
    product_index: int,
    scenario: str,
    postings: _CatalogPostings,
) -> tuple[_MutableSession, str | None]:
    category_pool = postings.category_pools.get(product.category)
    if category_pool is None or product_index not in category_pool:
        raise AssertionError("target is absent from its category pool")
    session = _MutableSession(
        scenario_state=(
            "buying"
            if scenario == "buying"
            else (
                "provisional_override"
                if scenario == "intent_override"
                else "exploring_unknown"
            )
        ),
        survivors=category_pool,
        known_constraints={},
        unindexed_values=set(),
        remaining_attributes=set(QUESTION_ATTRIBUTES),
        disclosed_values=set(),
    )
    override_value: str | None = None
    if scenario == "buying" and product.hard_constraints:
        attribute, normalized, display = product.hard_constraints[0]
        _apply_values(session, postings, attribute, (display,))
        session.disclosed_values.add(normalized)
        _advance_past_stage(session, attribute)
    elif scenario == "intent_override":
        old_constraint = (
            product.soft_preferences[-1]
            if product.soft_preferences
            else ("feature", normalize_constraint("different style"), "different style")
        )
        old_attribute, _, old_display = old_constraint
        _apply_values(session, postings, old_attribute, (old_display,))
        if product.hard_constraints:
            _, _, override_value = product.hard_constraints[0]
        else:
            override_value = product.constraints[0][2]
    return session, override_value


def _simulate_trajectory(
    *,
    trajectory_id: int,
    product_index: int,
    split: str,
    scenario: str,
    products: Sequence[Product],
    postings: _CatalogPostings,
    config: TrajectoryConfig,
) -> tuple[list[TrajectoryState], list[np.ndarray]]:
    product = products[product_index]
    session, override_value = _initialize_session(
        product, product_index, scenario, postings
    )
    override_turn = (
        3
        + stable_int(f"override-turn\0{config.seed}\0{trajectory_id}") % 2
        if scenario == "intent_override"
        else None
    )
    extended = _is_extended(config, trajectory_id, scenario)
    states: list[TrajectoryState] = []
    survivor_snapshots: list[np.ndarray] = []
    pending_question: str | None = None
    cutoff_seen = False

    for turn in range(1, config.max_turns + 1):
        if turn > 1:
            if override_turn is not None and turn == override_turn:
                session.survivors = postings.category_pools[product.category]
                session.known_constraints.clear()
                session.unindexed_values.clear()
                session.scenario_state = "intent_override"
                session.intent_epoch += 1
                pending_question = None
                if override_value:
                    override_attribute = classify_constraint(override_value)
                    _apply_values(
                        session,
                        postings,
                        override_attribute,
                        (override_value,),
                    )
                    session.disclosed_values.add(
                        normalize_constraint(override_value)
                    )
            else:
                if session.scenario_state == "exploring_unknown":
                    session.scenario_state = (
                        "boundary"
                        if scenario == "boundary" and pending_question is not None
                        else "browsing"
                    )
                if pending_question is not None:
                    if scenario == "boundary" and not session.boundary_used:
                        session.boundary_used = True
                    else:
                        answers = product.answer_values(
                            pending_question, session.disclosed_values
                        )
                        if answers:
                            _apply_values(
                                session, postings, pending_question, answers
                            )
                            session.disclosed_values.update(
                                normalize_constraint(value) for value in answers
                            )
                    session.remaining_attributes.discard(pending_question)
                pending_question = None

        if product_index not in session.survivors:
            raise AssertionError(
                f"trajectory {trajectory_id} removed its target at turn {turn}"
            )

        informative = _informative_attributes(session, postings)
        roadmap_informative = _roadmap_candidates(informative)
        normal_stop = (
            turn >= config.max_turns
            or len(session.survivors) <= 10
            or not informative
        )
        after_normal_cutoff = cutoff_seen
        if normal_stop:
            cutoff_seen = True

        if turn >= config.max_turns:
            next_question = None
        elif not normal_stop:
            next_question = _choose_attribute(
                roadmap_informative,
                seed=config.seed,
                trajectory_id=trajectory_id,
                turn=turn,
                phase="informative",
            )
        elif extended and session.remaining_attributes:
            next_question = _choose_attribute(
                roadmap_informative
                or _roadmap_candidates(tuple(session.remaining_attributes)),
                seed=config.seed,
                trajectory_id=trajectory_id,
                turn=turn,
                phase="extended",
            )
        else:
            next_question = None

        known_constraints = _freeze_known_constraints(session)
        states.append(
            TrajectoryState(
                trajectory_id=trajectory_id,
                state_index=len(states),
                product_index=product_index,
                target_parent_asin=product.parent_asin,
                split=split,
                scenario=scenario,
                scenario_state=session.scenario_state,
                turn=turn,
                turn_bucket=turn_bucket(turn),
                intent_epoch=session.intent_epoch,
                known_constraints=known_constraints,
                asked_attribute=next_question,
                has_other_answer=any(
                    attribute == "other" for attribute, _ in known_constraints
                ),
                survivor_count=len(session.survivors),
                state_weight=0.0,
                extended_trajectory=extended,
                after_normal_cutoff=after_normal_cutoff,
            )
        )
        survivor_snapshots.append(session.survivors)

        override_pending = override_turn is not None and turn < override_turn
        if next_question is None and not override_pending:
            break
        pending_question = next_question

    weight = 1.0 / len(states)
    states = [replace(state, state_weight=weight) for state in states]
    return states, survivor_snapshots


def generate_trajectory_dataset(
    products: Sequence[Product],
    raw_products: Sequence[dict],
    config: TrajectoryConfig,
    *,
    product_splits: Mapping[str, Sequence[int]] | None = None,
    input_hashes: Mapping[str, str] | None = None,
) -> TrajectoryDataset:
    """Generate the largest dataset once; use ``prefix`` for nested runs."""

    if not products:
        raise ValueError("products must not be empty")
    split_mapping = (
        build_product_splits(products, config.split_seed)
        if product_splits is None
        else {split: tuple(product_splits.get(split, ())) for split in SPLITS}
    )
    product_split_labels = _validate_product_split_mapping(products, split_mapping)
    scenarios = allocate_scenarios(
        config.trajectory_count, config.scenario_mix, config.seed
    )
    trajectory_splits = allocate_trajectory_splits(
        config.trajectory_count, config.split_seed
    )
    targets = select_trajectory_targets(
        products, split_mapping, trajectory_splits, config.seed
    )
    postings = build_catalog_postings(products, raw_products)

    all_states: list[TrajectoryState] = []
    survivor_buffer = array("I")
    if survivor_buffer.itemsize != np.dtype(np.uint32).itemsize:
        raise RuntimeError("native unsigned-int storage is not 32 bits")
    survivor_offsets: list[int] = [0]
    trajectory_state_offsets: list[int] = [0]

    for trajectory_id, (raw_product_index, split, scenario) in enumerate(
        zip(targets, trajectory_splits, scenarios, strict=True)
    ):
        product_index = int(raw_product_index)
        if product_split_labels[product_index] != split:
            raise AssertionError("selected target does not belong to trajectory split")
        states, snapshots = _simulate_trajectory(
            trajectory_id=trajectory_id,
            product_index=product_index,
            split=split,
            scenario=scenario,
            products=products,
            postings=postings,
            config=config,
        )
        all_states.extend(states)
        for survivors in snapshots:
            survivor_buffer.frombytes(
                np.asarray(survivors, dtype=np.uint32).tobytes(order="C")
            )
            survivor_offsets.append(len(survivor_buffer))
        trajectory_state_offsets.append(len(all_states))

    survivor_values = np.frombuffer(survivor_buffer, dtype=np.uint32)
    survivor_values.flags.writeable = False
    resolved_input_hashes = {"catalog_records": hash_catalog_records(raw_products)}
    resolved_input_hashes.update(input_hashes or {})
    return TrajectoryDataset(
        config=config,
        products=tuple(products),
        product_split_labels=product_split_labels,
        states=tuple(all_states),
        survivor_values=survivor_values,
        survivor_offsets=np.asarray(survivor_offsets, dtype=np.uint64),
        trajectory_state_offsets=np.asarray(
            trajectory_state_offsets, dtype=np.uint64
        ),
        trajectory_targets=targets,
        trajectory_scenarios=scenarios,
        trajectory_splits=trajectory_splits,
        input_hashes=tuple(sorted(resolved_input_hashes.items())),
    )


generate_trajectories = generate_trajectory_dataset


def survivor_width_bucket(width: int) -> str:
    if width <= 10:
        return "<=10"
    if width <= 50:
        return "11-50"
    if width <= 200:
        return "51-200"
    return ">200"


def _counter_dict(counter: Counter[str], order: Sequence[str]) -> dict[str, int]:
    return {name: int(counter.get(name, 0)) for name in order}


def manifest(
    dataset: TrajectoryDataset,
    *,
    input_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, object]:
    """Return an auditable manifest without serializing survivor IDs."""

    trajectory_split_counts = Counter(dataset.trajectory_splits)
    trajectory_scenario_counts = Counter(dataset.trajectory_scenarios)
    trajectory_joint_counts = Counter(
        f"{split}/{scenario}"
        for split, scenario in zip(
            dataset.trajectory_splits, dataset.trajectory_scenarios, strict=True
        )
    )
    state_split_counts = Counter(state.split for state in dataset.states)
    state_scenario_counts = Counter(state.scenario for state in dataset.states)
    observed_scenario_counts = Counter(
        state.scenario_state for state in dataset.states
    )
    turn_counts = Counter(state.turn_bucket for state in dataset.states)
    width_counts = Counter(
        survivor_width_bucket(state.survivor_count) for state in dataset.states
    )
    other_counts = Counter(
        "with_other" if state.has_other_answer else "without_other"
        for state in dataset.states
    )
    state_joint_counts = Counter(
        f"{state.split}/{state.scenario}" for state in dataset.states
    )
    product_split_counts = Counter(dataset.product_split_labels)
    product_split_hashes = {
        split: stable_hash(
            "\n".join(
                sorted(
                    product.parent_asin
                    for product, label in zip(
                        dataset.products,
                        dataset.product_split_labels,
                        strict=True,
                    )
                    if label == split
                )
            )
        )
        for split in SPLITS
    }
    config_payload = asdict(dataset.config)
    config_hash = stable_hash(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":"))
    )
    hashes = dict(dataset.input_hashes)
    for name, path in (input_paths or {}).items():
        hashes[str(name)] = sha256_path(path)
    widths = np.fromiter(
        (state.survivor_count for state in dataset.states), dtype=np.int64
    )
    trajectory_weight_error = 0.0
    for trajectory_id in range(dataset.trajectory_count):
        start = int(dataset.trajectory_state_offsets[trajectory_id])
        stop = int(dataset.trajectory_state_offsets[trajectory_id + 1])
        total = sum(state.state_weight for state in dataset.states[start:stop])
        trajectory_weight_error = max(trajectory_weight_error, abs(total - 1.0))

    return {
        "dataset_version": dataset.config.dataset_version,
        "config": config_payload,
        "config_sha256": config_hash,
        "input_sha256": dict(sorted(hashes.items())),
        "product_count": len(dataset.products),
        "trajectory_count": dataset.trajectory_count,
        "state_count": len(dataset.states),
        "survivor_value_count": len(dataset.survivor_values),
        "product_counts_by_split": _counter_dict(
            product_split_counts, SPLITS
        ),
        "product_split_sha256": product_split_hashes,
        "trajectory_counts_by_split": _counter_dict(
            trajectory_split_counts, SPLITS
        ),
        "trajectory_counts_by_scenario": _counter_dict(
            trajectory_scenario_counts, SCENARIOS
        ),
        "trajectory_counts_by_split_scenario": dict(
            sorted(trajectory_joint_counts.items())
        ),
        "state_counts_by_split": _counter_dict(state_split_counts, SPLITS),
        "state_counts_by_scenario": _counter_dict(
            state_scenario_counts, SCENARIOS
        ),
        "state_counts_by_observed_scenario": dict(
            sorted(observed_scenario_counts.items())
        ),
        "state_counts_by_split_scenario": dict(sorted(state_joint_counts.items())),
        "state_counts_by_turn_bucket": _counter_dict(turn_counts, TURN_BUCKETS),
        "state_counts_by_survivor_width": _counter_dict(
            width_counts, SURVIVOR_WIDTH_BUCKETS
        ),
        "state_counts_by_other": _counter_dict(
            other_counts, ("with_other", "without_other")
        ),
        "extended_trajectory_count": len(
            {
                state.trajectory_id
                for state in dataset.states
                if state.extended_trajectory
            }
        ),
        "states_after_normal_cutoff": sum(
            int(state.after_normal_cutoff) for state in dataset.states
        ),
        "survivor_width_summary": {
            "minimum": int(widths.min()) if len(widths) else 0,
            "mean": float(widths.mean()) if len(widths) else 0.0,
            "median": float(np.median(widths)) if len(widths) else 0.0,
            "maximum": int(widths.max()) if len(widths) else 0,
        },
        "maximum_trajectory_weight_sum_error": trajectory_weight_error,
    }


def write_manifest(
    path: str | Path,
    dataset: TrajectoryDataset,
    *,
    input_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, object]:
    report = manifest(dataset, input_paths=input_paths)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


__all__ = [
    "BALANCED_SCENARIO_COUNTS_PER_TWENTY",
    "DATASET_VERSION",
    "PUBLIC_SCENARIO_COUNTS_PER_TWENTY",
    "QUESTION_ATTRIBUTES",
    "SCENARIOS",
    "SCENARIO_MIXES",
    "SPLITS",
    "SURVIVOR_WIDTH_BUCKETS",
    "TRAJECTORY_BLOCK_SIZE",
    "TURN_BUCKETS",
    "Product",
    "TrajectoryConfig",
    "TrajectoryDataset",
    "TrajectoryState",
    "allocate_scenarios",
    "allocate_trajectory_splits",
    "build_catalog_postings",
    "build_product_splits",
    "generate_trajectories",
    "generate_trajectory_dataset",
    "hash_catalog_records",
    "load_products",
    "manifest",
    "popularity_bucket",
    "price_bucket",
    "rating_bucket",
    "scenario_allocation",
    "select_trajectory_targets",
    "sha256_path",
    "split_for",
    "stable_hash",
    "stable_int",
    "survivor_width_bucket",
    "write_manifest",
]
