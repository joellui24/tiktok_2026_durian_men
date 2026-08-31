from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
except ImportError:  # exact templates still work without the optional wheel
    rapidfuzz_fuzz = None

from starter.attribute_index import AttributeIndex, normalize_value
from starter.category_index import CategoryIndex
from starter.conversation_features import (
    classify_constraint,
    context_feature_names,
    legacy_context_feature_names,
)
from starter.hybrid_model import (
    NO_ANSWER,
    PortableHybridModel,
    default_linear_model_path,
    default_model_path,
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
BUYING_MARKER = ". A key requirement is:"
EXPLORING_MARKER = ", but I'm still exploring."
BOUNDARY_MARKER = "please use your judgment"
ANSWER_PREFIX = "For that, what matters is:"
INITIAL_PREFIX_RE = re.compile(
    r"^\s*I(?:['\N{RIGHT SINGLE QUOTATION MARK}]m| am) looking for\s+",
    re.IGNORECASE,
)
OVERRIDE_MARKERS = (
    "ignore my earlier preference",
    "ignore my previous preference",
    "changed my mind",
    "change my mind",
    "instead, what i need",
)
INTENT_CUES = {
    "buying": "a key requirement is",
    "browsing": "but i'm still exploring",
    "boundary": "please use your judgment",
}
BUYING_FUZZY_THRESHOLD = 80.0
NON_BUYING_FUZZY_THRESHOLD = 85.0

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


def _clean_disclosed_value(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")


def _normalize_intent_text(value: str) -> str:
    """Normalize wording without retaining category or constraint punctuation."""

    value = value.casefold().replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _contains_intent_cue(user_message: str, intent: str) -> bool:
    return _normalize_intent_text(INTENT_CUES[intent]) in _normalize_intent_text(
        user_message
    )


def _fuzzy_intent_score(user_message: str, intent: str) -> float:
    if rapidfuzz_fuzz is None:
        return 0.0
    return float(
        rapidfuzz_fuzz.partial_ratio(
            _normalize_intent_text(user_message),
            _normalize_intent_text(INTENT_CUES[intent]),
        )
    )


def _classify_initial_intent(user_message: str) -> str:
    """Classify visible opening wording without evaluator-only labels."""

    normalized = _normalize_intent_text(user_message)
    if any(
        _normalize_intent_text(marker) in normalized for marker in OVERRIDE_MARKERS
    ):
        return "intent_override"

    # An exact non-Buying marker wins even if unrelated words happen to look
    # similar to the shorter Buying cue.
    if _contains_intent_cue(user_message, "browsing"):
        return "exploring_unknown"
    if _contains_intent_cue(user_message, "boundary"):
        return "boundary"
    if _contains_intent_cue(user_message, "buying"):
        return "buying"

    # The asymmetric thresholds deliberately favor Buying recall. Unknown and
    # low-confidence wording remains on the safer Hybrid route.
    if _fuzzy_intent_score(user_message, "buying") >= BUYING_FUZZY_THRESHOLD:
        return "buying"
    if (
        _fuzzy_intent_score(user_message, "browsing")
        >= NON_BUYING_FUZZY_THRESHOLD
    ):
        return "exploring_unknown"
    if (
        _fuzzy_intent_score(user_message, "boundary")
        >= NON_BUYING_FUZZY_THRESHOLD
    ):
        return "boundary"
    return "unknown"


def _is_boundary_reply(user_message: str) -> bool:
    if _contains_intent_cue(user_message, "boundary"):
        return True
    return (
        _fuzzy_intent_score(user_message, "boundary")
        >= NON_BUYING_FUZZY_THRESHOLD
    )


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
        linear_model_path: str | Path | None = None,
        ranking_mode: str | None = None,
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
        if ranking_mode is None:
            # A caller supplying one explicit artifact retains the historical
            # single-Hybrid behavior. The normal submission entrypoint, which
            # supplies no artifact override, enables intent routing.
            ranking_mode = "hybrid" if model_path is not None else "routed"
        if ranking_mode not in {"linear", "fm", "hybrid", "routed"}:
            raise ValueError("ranking_mode must be linear, fm, hybrid, or routed")
        self.ranking_mode = ranking_mode
        self.disabled_field_pair = disabled_field_pair
        self.model: PortableHybridModel | None = None
        self.linear_model: PortableHybridModel | None = None
        self.model_error: str | None = None
        requested_model_path = Path(model_path) if model_path else default_model_path()
        self.model, primary_error = self._load_model(
            requested_model_path, "primary"
        )

        errors = [primary_error] if primary_error else []
        if ranking_mode == "routed":
            requested_linear_path = (
                Path(linear_model_path)
                if linear_model_path
                else default_linear_model_path()
            )
            self.linear_model, linear_error = self._load_model(
                requested_linear_path, "linear"
            )
            if linear_error:
                errors.append(linear_error)
        self.model_error = "; ".join(errors) or None

    def _load_model(
        self, requested_path: Path, label: str
    ) -> tuple[PortableHybridModel | None, str | None]:
        if not requested_path.exists():
            return None, f"{label} model artifact not found: {requested_path}"
        try:
            candidate_model = PortableHybridModel(requested_path)
            if candidate_model.matches_catalog(self._catalog_ids()):
                return candidate_model, None
            return None, f"{label} model artifact does not match this catalog"
        except Exception as error:  # safe evaluator fallback
            return None, f"{label} model artifact could not be loaded: {error}"

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
        scenario = _classify_initial_intent(message)
        buying_match = re.search(
            r"\.\s*a\s+key\s+requirement\s+is\s*:\s*",
            without_prefix,
            re.IGNORECASE,
        )
        if scenario == "buying" and buying_match is not None:
            category = without_prefix[: buying_match.start()]
            constraint = without_prefix[buying_match.end() :]
            return "buying", category.strip(), _clean_disclosed_value(constraint)
        if scenario == "buying":
            category = re.split(r"[.,]", without_prefix, maxsplit=1)[0].strip()
            constraint = (
                _clean_disclosed_value(without_prefix.rsplit(":", 1)[1])
                if ":" in without_prefix
                else None
            )
            return "buying", category or "clothing item", constraint
        if scenario == "exploring_unknown":
            category = re.split(r",|\bbut\b", without_prefix, maxsplit=1, flags=re.I)[
                0
            ].strip()
            return "exploring_unknown", category, None
        if scenario == "boundary":
            category = re.split(r"[.,]", without_prefix, maxsplit=1)[0].strip()
            return "boundary", category or "clothing item", None
        if scenario == "intent_override":
            category = re.split(r"[.,]", without_prefix, maxsplit=1)[0].strip()
            return "intent_override", category or "clothing item", None

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

    def _active_model(
        self, state: SessionState
    ) -> tuple[PortableHybridModel | None, str]:
        """Return the available model and score mode for this visible state."""

        if self.ranking_mode != "routed":
            return self.model, self.ranking_mode
        if state.scenario_state == "buying":
            if self.linear_model is not None:
                return self.linear_model, "linear"
            if self.model is not None:
                return self.model, "hybrid"
        else:
            if self.model is not None:
                return self.model, "hybrid"
            if self.linear_model is not None:
                return self.linear_model, "linear"
        return None, "hybrid"

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
                if _is_boundary_reply(user_message)
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
        active_model, active_mode = self._active_model(state)
        if active_model is not None and state.surviving_candidates:
            posterior = active_model.posterior(
                state.surviving_candidates,
                self._context_features(state, turn=turn, model=active_model),
                mode=active_mode,
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
                        reply = active_model.predicted_reply(
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

    def _context_features(
        self,
        state: SessionState,
        turn: int,
        model: PortableHybridModel | None = None,
    ) -> list[str]:
        features = context_feature_names(
            coarse_category=state.coarse_category,
            scenario_state=state.scenario_state,
            turn=turn,
            intent_epoch=state.intent_epoch,
            known_constraints=state.known_constraints,
        )
        if model is None:
            model = getattr(self, "model", None)
        if (
            model is None
            or model.metadata.get("feature_schema_version")
            == "conversation-features-v2"
        ):
            return features

        # Frozen E0 artifacts predate dual OTHER encoding. Reconstruct their
        # exact legacy context so baseline evaluation is not changed merely by
        # loading the new runtime. Newly trained v2 artifacts take the shared
        # path above and receive the source, retained, and inferred features.
        return legacy_context_feature_names(
            coarse_category=state.coarse_category,
            scenario_state=state.scenario_state,
            turn=turn,
            intent_epoch=state.intent_epoch,
            known_constraints=state.known_constraints,
        )

    def _recommendations(
        self, state: SessionState, top_k: int, turn: int = 1
    ) -> list[dict[str, str]]:
        limit = max(0, min(10, int(top_k)))
        candidates = state.surviving_candidates
        active_model, active_mode = self._active_model(state)
        if active_model is None:
            ranked = sorted(candidates)[:limit]
        else:
            ranked = active_model.rank(
                candidates,
                self._context_features(state, turn, model=active_model),
                limit,
                mode=active_mode,
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
