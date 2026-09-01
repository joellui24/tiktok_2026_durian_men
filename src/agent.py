from __future__ import annotations

import gzip
import math
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
except ImportError:  # exact templates still work without the optional wheel
    rapidfuzz_fuzz = None

from src.attribute_index import AttributeIndex, normalize_value
from src.category_index import CategoryIndex
from src.conversation_features import (
    classify_constraint,
    context_feature_names,
    legacy_context_feature_names,
)
from src.free_form_parser import FreeFormParse, parse_free_form_message
from src.hybrid_model import (
    NO_ANSWER,
    PortableHybridModel,
    default_linear_model_path,
    default_model_path,
)


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]


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
FREE_FORM_BLANKET_OVERRIDE_MARKERS = (
    "ignore my earlier preference",
    "ignore my previous preference",
)
INTENT_CUES = {
    "buying": "a key requirement is",
    "browsing": "but i'm still exploring",
    "boundary": "please use your judgment",
}
BUYING_FUZZY_THRESHOLD = 80.0
NON_BUYING_FUZZY_THRESHOLD = 85.0

# Canonical free-form category labels resolve to exact nodes in the catalog's
# category tree. Required path terms disambiguate labels such as Running, which
# also exists under sport-specific clothing.
FREE_FORM_CATEGORY_TARGETS = {
    "running shoes": (("Running",), ("Shoes",)),
    "walking shoes": (("Walking",), ("Shoes",)),
    "sneakers": (("Fashion Sneakers", "Sneakers"), ("Shoes",)),
    "sandals": (("Sandals",), ("Shoes",)),
    "shoes": (("Shoes",), ()),
    "shirts": (("Shirts", "T-Shirts", "Blouses & Button-Down Shirts"), ()),
    "tunics": (("Tunics",), ()),
    "boots": (("Boots",), ("Shoes",)),
    "dresses": (("Dresses",), ()),
    "jackets": (("Jackets",), ()),
    "jeans": (("Jeans",), ()),
    "pants": (("Pants",), ()),
    "skirts": (("Skirts",), ()),
    "socks": (("Socks", "Athletic Socks"), ()),
    "belts": (("Belts",), ()),
    "watches": (("Wrist Watches",), ()),
    "earrings": (("Earrings",), ()),
    "necklaces": (("Necklaces",), ()),
    "rings": (("Rings",), ()),
    "slippers": (("Slippers",), ("Shoes",)),
    "loafers": (("Loafers & Slip-Ons",), ("Shoes",)),
    "pumps": (("Pumps",), ("Shoes",)),
    "flats": (("Flats",), ("Shoes",)),
    "hoodies": (("Fashion Hoodies & Sweatshirts",), ()),
    "hats": (("Hats & Caps",), ()),
    "sunglasses": (("Sunglasses",), ()),
}

FREE_FORM_RETRIEVAL_MODES = frozenset({"off", "lexical", "dense", "hybrid"})
FREE_FORM_HARD_ATTRIBUTES = frozenset({"color", "material", "brand"})
# Raw operator text is a poor embedding target (dense encoders do not reliably
# honour logical negation).  On these turns retrieval uses the deterministic,
# positive state representation assembled below instead.
SEMANTIC_OPERATOR_RE = re.compile(
    r"\b(?:not|no|except|exclude|excluding|avoid|without|or|either|instead|"
    r"rather\s+than|remove|clear|forget|don['\N{RIGHT SINGLE QUOTATION MARK}]?t|"
    r"doesn['\N{RIGHT SINGLE QUOTATION MARK}]?t|anything\s+but|other\s+than|"
    r"irrelevant|unimportant)\b",
    re.IGNORECASE,
)


def _reciprocal_rank_fusion(
    rankings: list[list[str]], weights: list[float], limit: int, *, k: int = 60
) -> list[str]:
    """Dependency-free fusion so the lexical fallback needs no ML packages."""

    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, parent_asin in enumerate(dict.fromkeys(ranking), start=1):
            first_seen.setdefault(parent_asin, len(first_seen))
            scores[parent_asin] = scores.get(parent_asin, 0.0) + weight / (k + rank)
    return sorted(
        scores,
        key=lambda parent_asin: (
            -scores[parent_asin],
            first_seen[parent_asin],
            parent_asin,
        ),
    )[:limit]

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
    free_form_active: bool = False
    hard_constraints: dict[str, list[str]] = field(default_factory=dict)
    alternative_constraints: dict[str, list[str]] = field(default_factory=dict)
    excluded_constraints: dict[str, list[str]] = field(default_factory=dict)
    maximum_price: float | None = None
    maximum_price_inclusive: bool = False
    semantic_fragments: list[str] = field(default_factory=list)
    retrieval_debug: dict[str, object] = field(default_factory=dict)


class Agent:
    """Session-aware progressive attribute-filtering shopping agent."""

    def __init__(
        self,
        catalog_path: str | Path | None = None,
        *,
        category_index_path: str | Path | None = None,
        attribute_index_path: str | Path | None = None,
        model_path: str | Path | None = None,
        linear_model_path: str | Path | None = None,
        ranking_mode: str | None = None,
        disabled_field_pair: tuple[str, str] | None = None,
        free_form_retrieval_mode: str = "dense",
        semantic_artifact_path: str | Path | None = None,
        semantic_model_path: str | Path | None = None,
        semantic_cache_dir: str | Path | None = None,
        lexical_index_path: str | Path | None = None,
    ) -> None:
        data_directory = SUBMISSION_ROOT / "data"
        catalog_path = (
            Path(catalog_path)
            if catalog_path
            else data_directory / "catalog.jsonl"
        )
        self.catalog_path = catalog_path
        self._attribute_temp_directory: tempfile.TemporaryDirectory | None = None
        self.category_index = CategoryIndex(
            category_index_path or data_directory / "category_index.sqlite3"
        )
        resolved_attribute_index = (
            Path(attribute_index_path)
            if attribute_index_path is not None
            else data_directory / "attribute_index.sqlite3"
        )
        if not resolved_attribute_index.exists() and attribute_index_path is None:
            archive_path = data_directory / "attribute_index.sqlite3.gz"
            if archive_path.exists():
                self._attribute_temp_directory = tempfile.TemporaryDirectory(
                    prefix="techjam-agent-"
                )
                resolved_attribute_index = (
                    Path(self._attribute_temp_directory.name)
                    / "attribute_index.sqlite3"
                )
                with gzip.open(
                    archive_path, "rb"
                ) as source, resolved_attribute_index.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
        self.attribute_index = AttributeIndex(resolved_attribute_index)
        self._sessions: dict[str, SessionState] = {}
        self._attribute_hashmaps: dict[str, dict[str, tuple[str, ...]]] = {}
        self._indexed_products: dict[str, frozenset[str]] = {}
        self._known_brands: tuple[str, ...] | None = None
        self._all_catalog_ids: set[str] | None = None
        if free_form_retrieval_mode not in FREE_FORM_RETRIEVAL_MODES:
            raise ValueError(
                "free_form_retrieval_mode must be off, lexical, dense, or hybrid"
            )
        self.free_form_retrieval_mode = free_form_retrieval_mode
        self.semantic_artifact_path = Path(
            semantic_artifact_path or data_directory / "semantic_embeddings.npz"
        )
        bundled_semantic_model = SUBMISSION_ROOT / "models" / "bge-small-en-v1.5"
        self.semantic_model_path = (
            Path(semantic_model_path)
            if semantic_model_path is not None
            else bundled_semantic_model if bundled_semantic_model.exists() else None
        )
        self.semantic_cache_dir = (
            Path(semantic_cache_dir) if semantic_cache_dir is not None else None
        )
        self.lexical_index_path = Path(
            lexical_index_path or data_directory / "lexical_index.sqlite3"
        )
        self._semantic_index: object | None = None
        self._lexical_index: object | None = None
        self._semantic_unavailable = False
        self._lexical_unavailable = False
        self.free_form_retrieval_errors: list[str] = []
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
        lexical_index = self._lexical_index
        if lexical_index is not None:
            close = getattr(lexical_index, "close", None)
            if close is not None:
                close()
        self.category_index.close()
        self.attribute_index.close()
        if self._attribute_temp_directory is not None:
            self._attribute_temp_directory.cleanup()
            self._attribute_temp_directory = None

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

    def _brands(self) -> tuple[str, ...]:
        if self._known_brands is None:
            rows = self.attribute_index.connection.execute(
                """
                SELECT display_value, COUNT(*) AS product_count
                FROM attribute_values
                WHERE attribute = 'brand'
                GROUP BY normalized_value
                HAVING product_count >= 2
                ORDER BY LENGTH(display_value) DESC, display_value
                """
            ).fetchall()
            self._known_brands = tuple(
                str(row[0])
                for row in rows
                if str(row[0]).strip().casefold() not in {"no", "generic", "unknown"}
            )
        return self._known_brands

    def _parse_free_form(self, user_message: str) -> FreeFormParse:
        return parse_free_form_message(user_message, known_brands=self._brands())

    @staticmethod
    def _semantic_fragment(
        user_message: str, parsed: FreeFormParse | None = None
    ) -> str | None:
        """Return safe raw semantic evidence, never unresolved operator text."""

        fragment = re.sub(r"\s+", " ", user_message).strip()
        if (
            len(fragment) < 3
            or SEMANTIC_OPERATOR_RE.search(fragment)
            or (
                parsed is not None
                and bool(
                    parsed.alternatives
                    or parsed.excluded
                    or parsed.remove_attributes
                )
            )
        ):
            return None
        return fragment

    @staticmethod
    def _append_semantic_fragment(state: SessionState, fragment: str | None) -> None:
        if fragment is None:
            return
        normalized = fragment.casefold()
        if any(existing.casefold() == normalized for existing in state.semantic_fragments):
            return
        # Recent language is most useful and this bound prevents an extended
        # conversation from growing beyond the encoder's useful context.
        state.semantic_fragments.append(fragment)
        del state.semantic_fragments[:-3]

    @staticmethod
    def _semantic_query(state: SessionState) -> str:
        """Combine safe conversational language with current positive state."""

        parts = list(state.semantic_fragments)
        category_alternatives = state.alternative_constraints.get("category", [])
        if category_alternatives:
            parts.extend(
                f"product category alternative {value}"
                for value in category_alternatives
            )
        elif state.coarse_category and state.coarse_category != "clothing item":
            parts.append(f"product category {state.coarse_category}")
        for attribute in (
            "use_case",
            "feature",
            "style",
            "material",
            "color",
            "brand",
            "size",
            "budget",
        ):
            values = state.known_constraints.get(attribute, [])
            for value in values:
                # Numeric ceilings are enforced structurally, not semantically.
                if attribute == "budget" and value.startswith("maximum $"):
                    continue
                parts.append(f"{attribute.replace('_', ' ')} {value}")
        return " | ".join(dict.fromkeys(part for part in parts if part))

    def _free_form_candidates(self, category: str | None) -> tuple[str, set[str]]:
        if category:
            target = FREE_FORM_CATEGORY_TARGETS.get(category)
            if target:
                names, path_terms = target
                candidates = set(
                    self.category_index.products_for_category_names(
                        names, required_path_terms=path_terms
                    )
                )
                if candidates:
                    return category, candidates
        fallback_category, fallback = self._initial_candidates("clothing item")
        return fallback_category, fallback

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

    def _apply_alternatives(
        self, state: SessionState, attribute: str, values: list[str]
    ) -> bool:
        """Intersect candidates with the union of same-field alternatives."""

        mapping = self._hashmap(attribute)
        matched: set[str] = set()
        for value in values:
            postings = mapping.get(normalize_value(value))
            if postings:
                matched.update(postings)
        filtered = state.surviving_candidates.intersection(matched)
        if not filtered:
            state.unindexed_values.update(
                (attribute, normalize_value(value)) for value in values
            )
            return False
        state.surviving_candidates = filtered
        return True

    def _rebuild_free_form_candidates(self, state: SessionState) -> None:
        """Reapply free-form hard semantics without restoring violations.

        The official formatted path keeps its historical rollback policy in
        :meth:`_apply_values`.  Free-form constraints instead fail closed: an
        incompatible exact value produces no candidates and a clarification,
        never products that contradict the user's price/negation/OR request.
        """

        category_alternatives = state.alternative_constraints.get("category", [])
        if category_alternatives:
            candidates: set[str] = set()
            for category_value in category_alternatives:
                _, category_candidates = self._free_form_candidates(category_value)
                candidates.update(category_candidates)
            category = state.coarse_category
        else:
            category, candidates = self._free_form_candidates(state.coarse_category)
        state.coarse_category = category
        state.surviving_candidates = candidates
        state.unindexed_values.clear()

        for attribute, values in state.hard_constraints.items():
            mapping = self._hashmap(attribute)
            filtered = set(state.surviving_candidates)
            for value in values:
                postings = mapping.get(normalize_value(value))
                if not postings:
                    state.unindexed_values.add((attribute, normalize_value(value)))
                    filtered.clear()
                    break
                filtered.intersection_update(postings)
            if not filtered:
                state.unindexed_values.update(
                    (attribute, normalize_value(value)) for value in values
                )
            state.surviving_candidates = filtered
        for attribute, values in state.alternative_constraints.items():
            if attribute == "category" or attribute not in FREE_FORM_HARD_ATTRIBUTES:
                continue
            mapping = self._hashmap(attribute)
            matched = {
                parent_asin
                for value in values
                for parent_asin in mapping.get(normalize_value(value), ())
            }
            state.surviving_candidates.intersection_update(matched)
            if not state.surviving_candidates:
                state.unindexed_values.update(
                    (attribute, normalize_value(value)) for value in values
                )
        for attribute, values in state.excluded_constraints.items():
            if attribute != "category" and attribute not in FREE_FORM_HARD_ATTRIBUTES:
                continue
            had_candidates = bool(state.surviving_candidates)
            if attribute == "category":
                excluded = set()
                for value in values:
                    _, category_candidates = self._free_form_candidates(value)
                    excluded.update(category_candidates)
            else:
                mapping = self._hashmap(attribute)
                excluded = {
                    parent_asin
                    for value in values
                    for parent_asin in mapping.get(normalize_value(value), ())
                }
            state.surviving_candidates.difference_update(excluded)
            if had_candidates and not state.surviving_candidates:
                state.unindexed_values.update(
                    (attribute, f"excluded:{normalize_value(value)}")
                    for value in values
                )
        if state.maximum_price is not None:
            maximum_price = (
                math.nextafter(state.maximum_price, math.inf)
                if state.maximum_price_inclusive
                else state.maximum_price
            )
            priced = set(
                self.attribute_index.filter_products(
                    maximum_price=maximum_price
                )
            )
            state.surviving_candidates.intersection_update(priced)
            if not state.surviving_candidates:
                state.unindexed_values.add(
                    ("budget", f"maximum:{state.maximum_price:g}")
                )

    @staticmethod
    def _clear_free_form_preferences(
        state: SessionState, *, clear_semantic: bool
    ) -> None:
        """Clear every mutable preference store as one atomic state operation."""

        state.known_constraints.clear()
        state.hard_constraints.clear()
        state.alternative_constraints.clear()
        state.excluded_constraints.clear()
        state.maximum_price = None
        state.maximum_price_inclusive = False
        state.unindexed_values.clear()
        state.historical_disclosures.clear()
        state.remaining_attributes = set(ROADMAP_ATTRIBUTES)
        state.exhausted_attributes.clear()
        state.roadmap_stage = 0
        state.last_asked_attribute = None
        if clear_semantic:
            state.semantic_fragments.clear()

    @staticmethod
    def _drop_values(
        constraints: dict[str, list[str]], attribute: str, values: list[str]
    ) -> None:
        """Remove normalized values from one field, deleting empty state."""

        removed = {normalize_value(value) for value in values}
        remaining = [
            value
            for value in constraints.get(attribute, [])
            if normalize_value(value) not in removed
        ]
        if remaining:
            constraints[attribute] = remaining
        else:
            constraints.pop(attribute, None)

    def _store_free_form_parse(
        self,
        state: SessionState,
        parsed: FreeFormParse,
        *,
        replace_category: bool = False,
    ) -> None:
        category_alternatives = parsed.alternatives.get("category", [])
        if replace_category and (parsed.category or category_alternatives):
            if parsed.category:
                state.coarse_category = parsed.category
            self._clear_free_form_preferences(state, clear_semantic=False)
        elif parsed.category:
            # A category supplied later can be a refinement (for example,
            # "something comfortable" -> "shoes"). Keep unrelated fields.
            state.coarse_category = parsed.category

        for attribute in parsed.remove_attributes:
            state.known_constraints.pop(attribute, None)
            state.hard_constraints.pop(attribute, None)
            state.alternative_constraints.pop(attribute, None)
            state.excluded_constraints.pop(attribute, None)
            if attribute == "budget":
                state.maximum_price = None
                state.maximum_price_inclusive = False

        for attribute, values in parsed.attributes.items():
            self._drop_values(state.excluded_constraints, attribute, values)
            state.known_constraints[attribute] = list(values)
            state.alternative_constraints.pop(attribute, None)
            if attribute in FREE_FORM_HARD_ATTRIBUTES:
                state.hard_constraints[attribute] = list(values)
            else:
                state.hard_constraints.pop(attribute, None)
            state.historical_disclosures.update(normalize_value(value) for value in values)
            state.remaining_attributes.discard(attribute)

        for attribute, values in parsed.alternatives.items():
            self._drop_values(state.excluded_constraints, attribute, values)
            state.known_constraints[attribute] = list(values)
            state.alternative_constraints[attribute] = list(values)
            state.hard_constraints.pop(attribute, None)
            state.historical_disclosures.update(normalize_value(value) for value in values)
            state.remaining_attributes.discard(attribute)

        for attribute, values in parsed.excluded.items():
            self._drop_values(state.known_constraints, attribute, values)
            self._drop_values(state.hard_constraints, attribute, values)
            self._drop_values(state.alternative_constraints, attribute, values)
            exclusions = state.excluded_constraints.setdefault(attribute, [])
            existing = {normalize_value(value) for value in exclusions}
            exclusions.extend(
                value for value in values if normalize_value(value) not in existing
            )
            state.remaining_attributes.discard(attribute)

        if parsed.maximum_price is not None:
            state.maximum_price = parsed.maximum_price
            state.maximum_price_inclusive = parsed.maximum_price_inclusive
            state.known_constraints["budget"] = [
                f"maximum ${parsed.maximum_price:g}"
            ]
            state.historical_disclosures.add(f"maximum ${parsed.maximum_price:g}")
            state.remaining_attributes.discard("budget")
        elif parsed.qualitative_budget:
            state.known_constraints["budget"] = [parsed.qualitative_budget]
            state.remaining_attributes.discard("budget")

        self._rebuild_free_form_candidates(state)

    def _initialize_free_form(
        self, state: SessionState, user_message: str
    ) -> bool:
        parsed = self._parse_free_form(user_message)
        semantic_fragment = self._semantic_fragment(user_message, parsed)
        parsed_signal = bool(
            parsed.category or parsed.has_constraints or parsed.intent == "browsing"
        )
        semantic_signal = bool(
            self.free_form_retrieval_mode != "off" and semantic_fragment
        )
        if not (parsed_signal or semantic_signal):
            return False
        state.free_form_active = True
        state.scenario_state = parsed.intent
        state.coarse_category, state.surviving_candidates = self._free_form_candidates(
            parsed.category
        )
        self._append_semantic_fragment(state, semantic_fragment)
        self._store_free_form_parse(state, parsed)
        return True

    @staticmethod
    def _replaces_existing_preference(
        state: SessionState, parsed: FreeFormParse
    ) -> bool:
        """Detect implicit same-field replacement so stale language is dropped."""

        for source in (parsed.attributes, parsed.alternatives):
            for attribute, values in source.items():
                existing = state.known_constraints.get(attribute)
                if existing and {
                    normalize_value(value) for value in existing
                } != {normalize_value(value) for value in values}:
                    return True
        for attribute, values in parsed.excluded.items():
            positive = state.known_constraints.get(attribute, [])
            if {normalize_value(value) for value in positive}.intersection(
                normalize_value(value) for value in values
            ):
                return True
        return bool(
            parsed.maximum_price is not None
            and state.maximum_price is not None
            and (
                parsed.maximum_price != state.maximum_price
                or parsed.maximum_price_inclusive
                != state.maximum_price_inclusive
            )
        )

    def _replace_all_free_form_preferences(
        self, state: SessionState, replacement: str
    ) -> None:
        """Apply a formatted-style blanket override to a free-form session."""

        parsed = self._parse_free_form(replacement)
        self._clear_free_form_preferences(state, clear_semantic=True)
        if parsed.category:
            state.coarse_category = parsed.category
        state.scenario_state = "intent_override"
        state.intent_epoch += 1
        state.override_count += 1
        self._append_semantic_fragment(
            state, self._semantic_fragment(replacement, parsed)
        )
        self._store_free_form_parse(state, parsed)

    def _augment_contextual_answer(
        self,
        state: SessionState,
        user_message: str,
        parsed: FreeFormParse,
    ) -> None:
        """Interpret a short answer using the field the agent just asked for."""

        asked = state.last_asked_attribute
        if (
            asked is None
            or parsed.category is not None
            or parsed.intent == "browsing"
            or parsed.has_constraints
            or parsed.remove_attributes
        ):
            return
        payload = _clean_disclosed_value(user_message)
        if not payload:
            return
        if re.fullmatch(
            r"(?:no preference|doesn['\N{RIGHT SINGLE QUOTATION MARK}]?t matter|"
            r"does not matter|anything|any|whatever|no)",
            payload,
            re.IGNORECASE,
        ):
            parsed.remove_attributes.add(asked)
            return
        if asked == "budget":
            money = re.fullmatch(
                r"(?:about\s+|around\s+)?(?:usd\s*)?\$?\s*"
                r"([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{1,2})?|"
                r"[0-9]+(?:\.\d{1,2})?)"
                r"\s*(?:dollars?)?",
                payload,
                re.IGNORECASE,
            )
            if money:
                parsed.maximum_price = float(money.group(1).replace(",", ""))
                parsed.maximum_price_inclusive = True
            return
        if asked == "size":
            size = re.fullmatch(
                r"(?:size\s+)?([0-9]{1,2}(?:\.[05])?|x{0,2}[sml]|medium|"
                r"small|large|extra\s+large)",
                payload,
                re.IGNORECASE,
            )
            if size:
                parsed.attributes["size"] = [size.group(1).casefold()]
            return
        if asked == "brand":
            mapping = self._hashmap("brand")
            if normalize_value(payload) in mapping:
                parsed.attributes["brand"] = [payload]
            return
        if asked in {"use_case", "feature", "style", "other"} and re.search(
            r"[A-Za-z]", payload
        ):
            parsed.attributes[asked] = [payload]

    def _process_free_form_reply(
        self, state: SessionState, user_message: str
    ) -> bool:
        parsed = self._parse_free_form(user_message)
        self._augment_contextual_answer(state, user_message, parsed)
        category_alternatives = parsed.alternatives.get("category", [])
        category_changed = bool(
            parsed.category and parsed.category != state.coarse_category
            or category_alternatives
            and set(category_alternatives)
            != set(state.alternative_constraints.get("category", []))
        )
        browsing_switch = bool(
            parsed.intent == "browsing" and state.scenario_state != "browsing"
        )
        semantic_fragment = self._semantic_fragment(user_message, parsed)
        semantic_edit = bool(
            self.free_form_retrieval_mode != "off" and semantic_fragment
        )
        has_edit = bool(
            category_changed
            or parsed.has_constraints
            or parsed.remove_attributes
            or browsing_switch
            or semantic_edit
        )
        if not has_edit:
            return False
        preference_replaced = self._replaces_existing_preference(state, parsed)
        override_language = bool(
            re.search(
                r"\b(?:actually|changed? my mind|make that|instead|"
                r"on second thought|would be better|switch to|prefer .+ now|"
                r"cap it|(?:raise|lower|set) (?:the )?limit)\b",
                user_message,
                re.IGNORECASE,
            )
        )
        category_replaced = bool(category_changed and override_language)
        explicit_override = bool(
            parsed.remove_attributes
            or preference_replaced
            or override_language
        )
        if browsing_switch:
            self._clear_free_form_preferences(state, clear_semantic=True)
            state.scenario_state = "browsing"
            state.intent_epoch += 1
            state.override_count += 1
        elif explicit_override:
            state.scenario_state = "intent_override"
            state.intent_epoch += 1
            state.override_count += 1
        elif parsed.intent != "unknown":
            state.scenario_state = parsed.intent
        if explicit_override or browsing_switch:
            state.semantic_fragments.clear()
        self._append_semantic_fragment(state, semantic_fragment)
        self._store_free_form_parse(
            state, parsed, replace_category=category_replaced
        )
        state.last_asked_attribute = None
        return True

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
        if scenario == "unknown" and self._initialize_free_form(
            state, user_message
        ):
            state.initialized = True
            return
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
        if state.free_form_active:
            blanket_override = bool(
                replacement
                and any(
                    marker in user_message.casefold()
                    for marker in FREE_FORM_BLANKET_OVERRIDE_MARKERS
                )
            )
            if blanket_override:
                self._replace_all_free_form_preferences(state, replacement)
                return
            if self._process_free_form_reply(state, user_message):
                return
            if replacement is not None:
                # An incomplete or unrecognized free-form edit must preserve
                # the valid state rather than falling into legacy slot logic.
                state.last_asked_attribute = None
                return
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

    def _record_free_form_retrieval_error(self, source: str, error: Exception) -> None:
        message = f"{source} retrieval unavailable: {error}"
        if message not in self.free_form_retrieval_errors:
            self.free_form_retrieval_errors.append(message)

    def _dense_ranking(
        self, query: str, candidates: set[str], limit: int
    ) -> list[str]:
        if self._semantic_unavailable:
            return []
        try:
            if self._semantic_index is None:
                from src.semantic_retrieval import SemanticCatalogIndex

                semantic_index = SemanticCatalogIndex(
                    self.semantic_artifact_path,
                    cache_dir=self.semantic_cache_dir,
                    model_path=self.semantic_model_path,
                    local_files_only=True,
                    threads=4,
                )
                if set(semantic_index.identifiers) != self._catalog_ids():
                    raise ValueError(
                        "semantic artifact identifiers do not match the catalog"
                    )
                self._semantic_index = semantic_index
            return self._semantic_index.dense_rank(  # type: ignore[union-attr]
                query, candidates, limit=limit
            )
        except Exception as error:  # optional layer must retain a safe fallback
            self._semantic_unavailable = True
            self._record_free_form_retrieval_error("dense", error)
            return []

    def _lexical_ranking(
        self, query: str, candidates: set[str], limit: int
    ) -> list[str]:
        if self._lexical_unavailable:
            return []
        try:
            if self._lexical_index is None:
                from src.lexical_retrieval import LexicalCatalogIndex

                self._lexical_index = LexicalCatalogIndex(self.lexical_index_path)
            return self._lexical_index.lexical_rank(  # type: ignore[union-attr]
                query, candidates, limit=limit
            )
        except Exception as error:  # optional layer must retain a safe fallback
            self._lexical_unavailable = True
            self._record_free_form_retrieval_error("lexical", error)
            return []

    def _recommendations(
        self, state: SessionState, top_k: int, turn: int = 1
    ) -> list[dict[str, str]]:
        limit = max(0, min(10, int(top_k)))
        candidates = state.surviving_candidates
        active_model, active_mode = self._active_model(state)
        # This is the locked official formatted-query branch.  It intentionally
        # precedes semantic query construction and all optional imports.
        if not state.free_form_active or self.free_form_retrieval_mode == "off":
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

        retrieval_depth = min(len(candidates), max(200, limit))
        if active_model is None:
            base_ranking = sorted(candidates)[:retrieval_depth]
        else:
            base_ranking = active_model.rank(
                candidates,
                self._context_features(state, turn, model=active_model),
                retrieval_depth,
                mode=active_mode,
                disabled_field_pair=self.disabled_field_pair,
            )
        query = self._semantic_query(state)
        rankings: list[list[str]] = [base_ranking]
        weights: list[float] = [0.75]
        source_counts: dict[str, int] = {"existing_ranker": len(base_ranking)}

        if query and self.free_form_retrieval_mode in {"lexical", "hybrid"}:
            lexical = self._lexical_ranking(query, candidates, retrieval_depth)
            if lexical:
                rankings.append(lexical)
                weights.append(1.0 if self.free_form_retrieval_mode == "lexical" else 0.75)
            source_counts["lexical"] = len(lexical)
        if query and self.free_form_retrieval_mode in {"dense", "hybrid"}:
            dense = self._dense_ranking(query, candidates, retrieval_depth)
            if dense:
                rankings.append(dense)
                weights.append(1.5)
            source_counts["dense"] = len(dense)
            # Dense is the selected production variant.  The compact FTS index
            # remains a dependency-free failover when its optional runtime or
            # prebuilt vectors are unavailable.
            if self.free_form_retrieval_mode == "dense" and not dense:
                lexical = self._lexical_ranking(query, candidates, retrieval_depth)
                if lexical:
                    rankings.append(lexical)
                    weights.append(1.0)
                source_counts["lexical_fallback"] = len(lexical)

        if len(rankings) == 1:
            ranked = base_ranking[:limit]
        else:
            ranked = _reciprocal_rank_fusion(rankings, weights, limit)
        state.retrieval_debug = {
            "candidate_count": len(candidates),
            "errors": list(self.free_form_retrieval_errors),
            "mode": self.free_form_retrieval_mode,
            "query": query,
            "source_counts": source_counts,
        }
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
        hard_conflict = bool(
            state.free_form_active
            and not state.surviving_candidates
            and (
                state.unindexed_values
                or state.hard_constraints
                or state.maximum_price is not None
                or any(
                    attribute == "category"
                    or attribute in FREE_FORM_HARD_ATTRIBUTES
                    for attribute in state.alternative_constraints
                )
                or any(
                    attribute == "category"
                    or attribute in FREE_FORM_HARD_ATTRIBUTES
                    for attribute in state.excluded_constraints
                )
            )
        )
        if turn < 10 and hard_conflict:
            conflict_attributes = {
                attribute for attribute, _ in state.unindexed_values
            }
            ask_attribute = next(
                (
                    attribute
                    for attribute in ROADMAP_ATTRIBUTES
                    if attribute in conflict_attributes
                ),
                "use_case",
            )
        elif turn < 10 and len(state.surviving_candidates) > 10:
            ask_attribute = self._choose_next_attribute(state, turn)
        state.last_asked_attribute = ask_attribute

        recommendations = self._recommendations(state, top_k, turn)
        if hard_conflict and ask_attribute is not None:
            message = (
                "I couldn't find a product satisfying every hard constraint. "
                + QUESTION_TEXT[ask_attribute]
            )
        elif ask_attribute is None:
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
