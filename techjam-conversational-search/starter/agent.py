from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from starter.attribute_index import AttributeIndex, normalize_value
from starter.category_index import CategoryIndex
from starter.hybrid_model import (
    NO_ANSWER,
    PortableHybridModel,
    default_model_path,
    turn_bucket,
)


# ``category`` is supported by the evaluator, but the exact coarse category in
# the initial message is handled by CategoryIndex rather than asked again.
EVALUATOR_ATTRIBUTES = (
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
ROADMAP_STAGES = (
    ("use_case",),
    ("feature", "style", "material"),
    ("size", "budget", "brand"),
    ("color",),
    ("other",),
)
ROADMAP_ATTRIBUTES = tuple(
    attribute for stage in ROADMAP_STAGES for attribute in stage
)
MATERIALS = (
    "cotton",
    "polyester",
    "nylon",
    "leather",
    "wool",
    "spandex",
    "silk",
    "rayon",
    "fabric",
)
BUYING_MARKER = ". A key requirement is:"
EXPLORING_MARKER = ", but I'm still exploring."
BOUNDARY_MARKER = "please use your judgment"
ANSWER_PREFIX = "For that, what matters is:"
INITIAL_PREFIX_RE = re.compile(r"^\s*I(?:'m| am) looking for\s+", re.IGNORECASE)
OVERRIDE_MARKERS = (
    "ignore my earlier preference",
    "ignore my previous preference",
    "changed my mind",
    "change my mind",
    "instead, what i need",
)

QUESTION_TEXT = {
    "use_case": "What use case or occasion should this work best for?",
    "feature": "Which product feature matters most to you?",
    "style": "Do you have a preferred style or fit?",
    "material": "Do you have a material preference?",
    "size": "Are there any size, width, or fit constraints?",
    "budget": "What budget should I stay within?",
    "brand": "Do you have a preferred brand?",
    "color": "Do you have a color preference?",
    "other": "Is there another requirement I should prioritize?",
}


def classify_constraint(value: str) -> str:
    """Mirror evaluator.local_evaluator.classify_constraint exactly.

    Classification cannot be inferred from index membership because every
    simulator constraint is also stored under ``other``.
    """

    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(
        word in lowered
        for word in ("color", "black", "white", "blue", "red", "pink", "green")
    ):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def _clean_disclosed_value(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")


@dataclass
class SessionState:
    scenario_state: str = "unknown"
    coarse_category: str = "clothing item"
    surviving_candidates: set[str] = field(default_factory=set)
    known_constraints: dict[str, list[str]] = field(default_factory=dict)
    unindexed_values: set[tuple[str, str]] = field(default_factory=set)
    remaining_attributes: set[str] = field(
        default_factory=lambda: set(ROADMAP_ATTRIBUTES)
    )
    exhausted_attributes: set[str] = field(default_factory=set)
    roadmap_stage: int = 0
    last_asked_attribute: str | None = None
    historical_disclosures: set[str] = field(default_factory=set)
    intent_epoch: int = 0
    override_count: int = 0
    initialized: bool = False


class Agent:
    """Session-aware progressive attribute-filtering shopping agent."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        category_index_path: str | Path | None = None,
        attribute_index_path: str | Path | None = None,
        model_path: str | Path | None = None,
        ranking_mode: str = "hybrid",
        disabled_field_pair: tuple[str, str] | None = None,
    ) -> None:
        catalog_path = Path(catalog_path)
        data_directory = catalog_path.parent
        self.catalog_path = catalog_path
        self.category_index = CategoryIndex(
            category_index_path or data_directory / "category_index.sqlite3"
        )
        self.attribute_index = AttributeIndex(
            attribute_index_path or data_directory / "attribute_index.sqlite3"
        )
        self._sessions: dict[str, SessionState] = {}
        self._attribute_hashmaps: dict[str, dict[str, tuple[str, ...]]] = {}
        self._indexed_products: dict[str, frozenset[str]] = {}
        self._all_catalog_ids: set[str] | None = None
        if ranking_mode not in {"linear", "fm", "hybrid"}:
            raise ValueError("ranking_mode must be linear, fm, or hybrid")
        self.ranking_mode = ranking_mode
        self.disabled_field_pair = disabled_field_pair
        self.model: PortableHybridModel | None = None
        self.model_error: str | None = None
        requested_model_path = Path(model_path) if model_path else default_model_path()
        if requested_model_path.exists():
            try:
                candidate_model = PortableHybridModel(requested_model_path)
                if candidate_model.matches_catalog(self._catalog_ids()):
                    self.model = candidate_model
                else:
                    self.model_error = "FM artifact does not match this catalog"
            except Exception as error:  # safe evaluator fallback
                self.model_error = f"FM artifact could not be loaded: {error}"
        else:
            self.model_error = f"FM artifact not found: {requested_model_path}"

    def close(self) -> None:
        self.category_index.close()
        self.attribute_index.close()

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The current deterministic filtering policy does not personalize yet.
        del user_profile
        self._sessions[session_id] = SessionState()

    def _hashmap(self, attribute: str) -> dict[str, tuple[str, ...]]:
        mapping = self._attribute_hashmaps.get(attribute)
        if mapping is None:
            mapping = self.attribute_index.load_hashmap(attribute)
            self._attribute_hashmaps[attribute] = mapping
            self._indexed_products[attribute] = frozenset(
                parent_asin
                for postings in mapping.values()
                for parent_asin in postings
            )
        return mapping

    def _catalog_ids(self) -> set[str]:
        if self._all_catalog_ids is None:
            self._all_catalog_ids = {
                str(row[0])
                for row in self.category_index.connection.execute(
                    "SELECT parent_asin FROM products"
                )
            }
        return set(self._all_catalog_ids)

    @staticmethod
    def _parse_initial_message(user_message: str) -> tuple[str, str, str | None]:
        """Return scenario state, exact coarse category, and Buying constraint."""

        message = user_message.strip()
        without_prefix = INITIAL_PREFIX_RE.sub("", message, count=1)
        if BUYING_MARKER in without_prefix:
            category, constraint = without_prefix.split(BUYING_MARKER, 1)
            return "buying", category.strip(), _clean_disclosed_value(constraint)
        if without_prefix.endswith(EXPLORING_MARKER):
            category = without_prefix[: -len(EXPLORING_MARKER)].strip()
            return "exploring_unknown", category, None

        # The simulator's Intent Override scenario starts with a category and
        # a provisional preference. It is not scored until the later explicit
        # override, but using the provisional evidence still supports useful
        # questions before that transition.
        if "." in without_prefix:
            category, provisional = without_prefix.split(".", 1)
            provisional = _clean_disclosed_value(provisional)
            if category.strip() and provisional:
                return "provisional_override", category.strip(), provisional

        category = without_prefix.split(".", 1)[0].strip(" ,.")
        return "unknown", category or "clothing item", None

    @staticmethod
    def _parse_override_message(user_message: str) -> str | None:
        lowered = user_message.casefold()
        if not any(marker in lowered for marker in OVERRIDE_MARKERS):
            return None
        if ":" in user_message:
            replacement = user_message.rsplit(":", 1)[1]
        else:
            match = re.search(r"\b(?:need|want|instead)\b\s+(?:is\s+)?(.+)$", user_message, re.I)
            replacement = match.group(1) if match else ""
        return _clean_disclosed_value(replacement)

    def _initial_candidates(self, coarse_category: str) -> tuple[str, set[str]]:
        candidates = set(
            self.category_index.products_for_coarse_category(coarse_category)
        )
        if candidates:
            return coarse_category, candidates

        fallback = "clothing item"
        fallback_candidates = set(
            self.category_index.products_for_coarse_category(fallback)
        )
        return (
            (fallback, fallback_candidates)
            if fallback_candidates
            else (fallback, self._catalog_ids())
        )

    @staticmethod
    def _record_constraint(
        state: SessionState, attribute: str, value: str
    ) -> None:
        values = state.known_constraints.setdefault(attribute, [])
        if value not in values:
            values.append(value)

    def _apply_values(
        self, state: SessionState, attribute: str, values: list[str]
    ) -> bool:
        """Atomically AND exact postings, rolling back an empty result."""

        cleaned_values = list(
            dict.fromkeys(
                cleaned
                for value in values
                if (cleaned := _clean_disclosed_value(value))
            )
        )
        if not cleaned_values:
            return False

        mapping = self._hashmap(attribute)
        filtered = set(state.surviving_candidates)
        missing: list[str] = []
        for value in cleaned_values:
            self._record_constraint(state, attribute, value)
            postings = mapping.get(normalize_value(value))
            if not postings:
                missing.append(value)
                filtered.clear()
                continue
            filtered.intersection_update(postings)

        if filtered:
            state.surviving_candidates = filtered
            return True

        # A value may exist globally but still be incompatible with the current
        # category/filters. In either case it was not safely indexable in this
        # session, so preserve the previous survivor set.
        rejected = missing or cleaned_values
        state.unindexed_values.update(
            (attribute, normalize_value(value)) for value in rejected
        )
        return False

    @staticmethod
    def _exhaust_attribute(state: SessionState, attribute: str) -> None:
        state.remaining_attributes.discard(attribute)
        state.exhausted_attributes.add(attribute)

    @staticmethod
    def _advance_past_stage(state: SessionState, attribute: str) -> None:
        for stage_index, stage in enumerate(ROADMAP_STAGES):
            if attribute not in stage:
                continue
            for completed_stage in ROADMAP_STAGES[: stage_index + 1]:
                for completed_attribute in completed_stage:
                    state.remaining_attributes.discard(completed_attribute)
                    state.exhausted_attributes.add(completed_attribute)
            state.roadmap_stage = stage_index + 1
            return

    def _initialize(self, state: SessionState, user_message: str) -> None:
        scenario, parsed_category, buying_constraint = self._parse_initial_message(
            user_message
        )
        coarse_category, candidates = self._initial_candidates(parsed_category)
        state.scenario_state = scenario
        state.coarse_category = coarse_category
        state.surviving_candidates = candidates
        state.initialized = True

        if scenario in {"buying", "provisional_override"} and buying_constraint:
            attribute = classify_constraint(buying_constraint)
            self._apply_values(state, attribute, [buying_constraint])
            state.historical_disclosures.add(normalize_value(buying_constraint))
            if scenario == "buying":
                self._advance_past_stage(state, attribute)

    def _split_answer_values(self, attribute: str, payload: str) -> list[str]:
        payload = _clean_disclosed_value(payload)
        if not payload:
            return []
        mapping = self._hashmap(attribute)
        if normalize_value(payload) in mapping:
            return [payload]

        # The simulator joins at most two constraints with "; ". A catalog
        # constraint may itself contain semicolons, so prefer a split whose two
        # complete values both exist in the relevant attribute index.
        split_points = [match.start() for match in re.finditer(r";\s+", payload)]
        for split_point in split_points:
            left = _clean_disclosed_value(payload[:split_point])
            right = _clean_disclosed_value(payload[split_point + 1 :])
            if normalize_value(left) in mapping and normalize_value(right) in mapping:
                return [left, right]
        return [
            value
            for part in re.split(r";\s+", payload)
            if (value := _clean_disclosed_value(part))
        ]

    def _process_reply(self, state: SessionState, user_message: str) -> None:
        replacement = self._parse_override_message(user_message)
        if replacement is not None:
            if not replacement:
                # Do not destroy a valid state when an override is incomplete.
                state.last_asked_attribute = None
                return
            _, candidates = self._initial_candidates(state.coarse_category)
            state.surviving_candidates = candidates
            state.known_constraints.clear()
            state.unindexed_values.clear()
            state.scenario_state = "intent_override"
            state.intent_epoch += 1
            state.override_count += 1
            state.last_asked_attribute = None
            attribute = classify_constraint(replacement)
            self._apply_values(state, attribute, [replacement])
            state.historical_disclosures.add(normalize_value(replacement))
            return

        asked = state.last_asked_attribute
        if state.scenario_state == "exploring_unknown":
            state.scenario_state = (
                "boundary"
                if BOUNDARY_MARKER in user_message.casefold()
                else "browsing"
            )
        if asked is None:
            return

        if user_message.lstrip().startswith(ANSWER_PREFIX):
            payload = user_message.lstrip()[len(ANSWER_PREFIX) :]
            values = self._split_answer_values(asked, payload)
            self._apply_values(state, asked, values)
            state.historical_disclosures.update(normalize_value(value) for value in values)

        # Answers, no-preference replies, and unrecognized simulator replies all
        # consume the question. This prevents a session getting stuck.
        self._exhaust_attribute(state, asked)
        state.last_asked_attribute = None

    def _largest_bucket(self, attribute: str, candidates: set[str]) -> int:
        mapping = self._hashmap(attribute)
        indexed = self._indexed_products[attribute]
        largest = sum(parent_asin not in indexed for parent_asin in candidates)
        for postings in mapping.values():
            if len(postings) <= largest:
                continue
            bucket_size = sum(parent_asin in candidates for parent_asin in postings)
            if bucket_size > largest:
                largest = bucket_size
        return largest

    def _choose_next_attribute(self, state: SessionState, turn: int = 1) -> str | None:
        if self.model is not None and state.surviving_candidates:
            posterior = self.model.posterior(
                state.surviving_candidates,
                self._context_features(state, turn=turn),
                mode=self.ranking_mode,
                disabled_field_pair=self.disabled_field_pair,
            )
            if posterior:
                roadmap_order = {
                    attribute: position
                    for position, attribute in enumerate(ROADMAP_ATTRIBUTES)
                }
                scored: list[tuple[float, float, int, str]] = []
                for attribute in ROADMAP_ATTRIBUTES:
                    if attribute not in state.remaining_attributes:
                        continue
                    buckets: dict[tuple[str, ...], float] = {}
                    for parent_asin, probability in posterior.items():
                        reply = self.model.predicted_reply(
                            parent_asin,
                            attribute,
                            state.historical_disclosures,
                        )
                        buckets[reply] = buckets.get(reply, 0.0) + probability
                    entropy = -sum(
                        mass * math.log(mass)
                        for mass in buckets.values()
                        if mass > 0.0
                    )
                    no_answer_probability = buckets.get((NO_ANSWER,), 0.0)
                    scored.append(
                        (
                            entropy,
                            1.0 - no_answer_probability,
                            -roadmap_order[attribute],
                            attribute,
                        )
                    )
                if scored:
                    return max(scored)[3]

        while state.roadmap_stage < len(ROADMAP_STAGES):
            stage = ROADMAP_STAGES[state.roadmap_stage]
            available = [
                attribute
                for attribute in stage
                if attribute in state.remaining_attributes
            ]
            if available:
                order = {attribute: position for position, attribute in enumerate(stage)}
                return min(
                    available,
                    key=lambda attribute: (
                        self._largest_bucket(
                            attribute, state.surviving_candidates
                        ),
                        order[attribute],
                    ),
                )
            state.roadmap_stage += 1
        return None

    def _context_features(self, state: SessionState, turn: int) -> list[str]:
        features = [
            f"ctx:category={normalize_value(state.coarse_category)}",
            f"ctx:scenario={state.scenario_state}",
            f"ctx:turn={turn_bucket(turn)}",
            f"ctx:override={'post' if state.intent_epoch else 'pre'}",
        ]
        for attribute, values in sorted(state.known_constraints.items()):
            features.extend(
                f"ctx:{attribute}={normalize_value(value)}" for value in values
            )
        return features

    def _recommendations(
        self, state: SessionState, top_k: int, turn: int = 1
    ) -> list[dict[str, str]]:
        limit = max(0, min(10, int(top_k)))
        candidates = state.surviving_candidates
        if self.model is None:
            ranked = sorted(candidates)[:limit]
        else:
            ranked = self.model.rank(
                candidates,
                self._context_features(state, turn),
                limit,
                mode=self.ranking_mode,
                disabled_field_pair=self.disabled_field_pair,
            )
        return [{"parent_asin": parent_asin} for parent_asin in ranked]

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")

        if not state.initialized:
            self._initialize(state, user_message)
        else:
            self._process_reply(state, user_message)

        ask_attribute: str | None = None
        if turn < 10 and len(state.surviving_candidates) > 10:
            ask_attribute = self._choose_next_attribute(state, turn)
        state.last_asked_attribute = ask_attribute

        recommendations = self._recommendations(state, top_k, turn)
        if ask_attribute is None:
            message = "Here are the best matching products from the remaining options."
        else:
            message = (
                "Here are some current matches. " + QUESTION_TEXT[ask_attribute]
            )
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
