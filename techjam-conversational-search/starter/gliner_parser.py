"""Opt-in GLiNER augmentation for the deterministic free-form parser.

This module is deliberately not imported by :mod:`starter.agent`.  Experiments
may call :class:`GLiNERAugmenter` after ``parse_free_form_message``; production
and the official structured evaluator therefore remain rules-only.
"""

from __future__ import annotations

import copy
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from starter.free_form_parser import (
    CATEGORY_ALIASES,
    BRAND_STOPWORDS,
    COLORS,
    MATERIALS,
    FreeFormParse,
    normalize_text,
)


DEFAULT_MODEL_ID = "gliner-community/gliner_small-v2.5"
DEFAULT_MODEL_REVISION = "f227d3cd637bd4e6757ae143935316d062393341"
DEFAULT_THRESHOLD = 0.3
DEFAULT_ENABLED_FIELDS = frozenset(
    {"category", "brand", "color", "material", "feature", "use_case"}
)
DEFAULT_HARD_THRESHOLD = 0.7

# Seven labels keep the zero-shot prompt small enough for CPU inference.  The
# model identifies spans; validation below decides whether a span can become a
# typed value.  GLiNER never parses operators, intent, or budgets.
FIELD_LABELS = {
    "category": "product category",
    "brand": "brand",
    "color": "color",
    "material": "material",
    "size": "clothing or shoe size",
    "feature": "product feature",
    "use_case": "shopping use case or occasion",
}
LABEL_FIELDS = {normalize_text(label): field for field, label in FIELD_LABELS.items()}

HARD_FIELDS = frozenset({"category", "brand", "color", "material"})
SOFT_FIELDS = frozenset({"size", "feature", "use_case"})

# Any operator-bearing message stays entirely deterministic.  This is broader
# than ordinary negation on purpose: a false negative is safer than turning a
# negated or alternative value into a positive hard constraint.
AUTHORITATIVE_OPERATOR_RE = re.compile(
    r"\b(?:not|except|exclude|excluding|avoid|without|or|instead|"
    r"rather\s+than|other\s+than|anything\s+but)\b|"
    r"\bno\b(?!\s+(?:rush|more\s+than|fixed\s+item))|"
    r"\b(?:do\s+not|don't|dont)\s+want\b|"
    r"\b(?:doesn't|does\s+not|doesnt)\s+matter\b|"
    r"\b(?:don't|do\s+not|dont)\s+(?:really\s+)?care\b|"
    r"\bno\s+preference\b|\birrelevant\b|"
    r"\b(?:remove|clear|forget|drop)\s+(?:the\s+|my\s+|an?\s+)?"
    r"(?:earlier\s+|previous\s+)?(?:brand|colou?r|material|fabric|size|"
    r"budget|price|feature|use(?:\s+case)?|occasion)\b|"
    r"\bany\s+(?:brand|colou?r|material|fabric|size)\s+"
    r"(?:is\s+)?(?:fine|okay|ok|works|acceptable|will\s+do)\b|"
    r"\b(?:brand|colou?r|material|fabric|size|budget|price|feature|"
    r"use(?:\s+case)?|occasion)\s+(?:isn't|is\s+not)\s+"
    r"(?:important|a\s+priority)\b",
    re.IGNORECASE,
)


CATEGORY_SYNONYMS = {
    "runner": "running shoes",
    "runners": "running shoes",
    "road runner": "running shoes",
    "road runners": "running shoes",
    "jogging shoe": "running shoes",
    "jogging shoes": "running shoes",
    "trainer": "sneakers",
    "trainers": "sneakers",
    "kick": "sneakers",
    "kicks": "sneakers",
    "tennis shoe": "sneakers",
    "tennis shoes": "sneakers",
    "flip flop": "sandals",
    "flip flops": "sandals",
    "flip-flop": "sandals",
    "flip-flops": "sandals",
    "tee": "shirts",
    "tees": "shirts",
    "top": "shirts",
    "tops": "shirts",
    "blouse": "shirts",
    "blouses": "shirts",
    "trouser": "pants",
    "trousers": "pants",
    "hooded sweatshirt": "hoodies",
    "hooded sweatshirts": "hoodies",
    "pullover": "hoodies",
    "pullovers": "hoodies",
    "coat": "jackets",
    "coats": "jackets",
    "outerwear": "jackets",
    "ankle bootie": "boots",
    "ankle booties": "boots",
    "slip on": "loafers",
    "slip ons": "loafers",
    "slip-on": "loafers",
    "slip-ons": "loafers",
    "court heel": "pumps",
    "court heels": "pumps",
    "shade": "sunglasses",
    "shades": "sunglasses",
    "eyewear": "sunglasses",
}

FEATURE_ALIASES = {
    "comfortable": "comfort",
    "comfort": "comfort",
    "comfy": "comfort",
    "pain free": "comfort",
    "pain-free": "comfort",
    "pillowy": "cushioned",
    "padded": "cushioned",
    "cushion": "cushioned",
    "cushioned": "cushioned",
    "cushioning": "cushioned",
    "light": "lightweight",
    "lightweight": "lightweight",
    "barely weighs": "lightweight",
    "breathable": "breathable",
    "breathability": "breathable",
    "air circulate": "breathable",
    "ventilated": "breathable",
    "warm": "warm",
    "warmth": "warm",
    "insulated": "warm",
    "holds in heat": "warm",
    "durable": "durable",
    "sturdy": "durable",
    "rugged": "durable",
    "soft": "soft",
    "gentle": "soft",
    "supportive": "supportive",
    "support": "supportive",
    "arches": "supportive",
    "waterproof": "waterproof",
    "rainproof": "waterproof",
    "keep rain": "waterproof",
    "polarized": "polarized",
    "machine washable": "machine washable",
    "stylish": "style",
    "fashionable": "style",
    "current look": "style",
}

USE_CASE_ALIASES = {
    "run": "running",
    "running": "running",
    "jog": "running",
    "jogging": "running",
    "pounding the pavement": "running",
    "walk": "walking",
    "walking": "walking",
    "long walk": "walking",
    "long walks": "walking",
    "sightseeing on foot": "walking",
    "stand all day": "work",
    "hike": "hiking",
    "hiking": "hiking",
    "trek": "hiking",
    "gym": "gym",
    "workout": "gym",
    "lifting weights": "gym",
    "beach": "beach",
    "seaside": "beach",
    "sand sea": "beach",
    "winter": "winter",
    "snowy": "winter",
    "cold weather": "winter",
    "january": "winter",
    "outdoor": "outdoor",
    "outdoors": "outdoor",
    "outdoorsy": "outdoor",
    "travel": "travel",
    "travelling": "travel",
    "traveling": "travel",
    "trip": "travel",
    "holiday": "travel",
    "city break": "travel",
    "long flight": "travel",
    "long flights": "travel",
    "summer": "summer",
    "hot weather": "summer",
    "work": "work",
    "office": "work",
    "shift": "work",
    "shifts": "work",
    "formal": "formal",
    "gala": "formal",
    "black tie": "formal",
    "black-tie": "formal",
    "reception": "formal",
    "red carpet": "formal",
    "red-carpet": "formal",
    "lounge": "lounge",
    "lounging": "lounge",
    "relax": "lounge",
    "sofa": "lounge",
}

SIZE_ALIASES = {
    "extra small": "xs",
    "extra-small": "xs",
    "small": "s",
    "medium": "m",
    "large": "l",
    "extra large": "xl",
    "extra-large": "xl",
    "one size": "one size",
    "one-size-fits-all": "one size",
}
SIZE_RE = re.compile(
    r"^(?:(?:us|eu|uk)(?:\s+(?:women'?s|men'?s))?\s+)?"
    r"\d{1,2}(?:\.5)?(?:\s+(?:wide|narrow|petite))?$|"
    r"^(?:x{0,3}[sml]|one size)(?:\s+petite)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EntityPrediction:
    text: str
    label: str
    score: float
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True)
class EntityDecision:
    field: str
    raw_text: str
    label: str
    score: float
    value: str | None
    accepted: bool
    hard_constraint: bool
    reason: str


@dataclass
class GLiNERAugmentation:
    parse: FreeFormParse
    additions: dict[str, list[str]] = field(default_factory=dict)
    decisions: list[EntityDecision] = field(default_factory=list)
    predictions: list[EntityPrediction] = field(default_factory=list)
    inference_seconds: float = 0.0
    skipped_reason: str | None = None


def has_authoritative_operator(message: str) -> bool:
    return AUTHORITATIVE_OPERATOR_RE.search(normalize_text(message)) is not None


def _phrase_value(text: str, aliases: Mapping[str, str]) -> str | None:
    normalized = normalize_text(text).replace("'", "")
    for phrase in sorted(aliases, key=len, reverse=True):
        if re.search(
            rf"(?<![a-z0-9]){re.escape(normalize_text(phrase))}(?![a-z0-9])",
            normalized,
        ):
            return aliases[phrase]
    return None


def _strip_category_wrapper(text: str) -> str:
    normalized = normalize_text(text).strip(" .,-")
    return re.sub(
        r"^(?:(?:a|an|the|some)\s+|(?:a\s+)?pair\s+of\s+)", "", normalized
    ).strip()


class GLiNERAugmenter:
    """Lazy, schema-validating GLiNER fallback for experimental use only."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = DEFAULT_MODEL_REVISION,
        threshold: float = DEFAULT_THRESHOLD,
        hard_threshold: float = DEFAULT_HARD_THRESHOLD,
        enabled_fields: Iterable[str] = DEFAULT_ENABLED_FIELDS,
        known_brands: Iterable[str] = (),
        valid_categories: Iterable[str] = (),
        valid_colors: Iterable[str] = COLORS,
        valid_materials: Iterable[str] = MATERIALS,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not 0.0 <= hard_threshold <= 1.0:
            raise ValueError("hard_threshold must be between 0 and 1")
        unknown_fields = set(enabled_fields).difference(FIELD_LABELS)
        if unknown_fields:
            raise ValueError(f"unsupported enabled fields: {sorted(unknown_fields)}")
        self.model_id = model_id
        self.revision = revision
        self.threshold = threshold
        self.hard_threshold = hard_threshold
        self.enabled_fields = frozenset(enabled_fields)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.local_files_only = local_files_only
        self._model = model
        self.load_seconds = 0.0
        self.model_loaded_here = False

        blocked_brands = set(BRAND_STOPWORDS).union(
            {"no", "none", "yes", "generic", "unknown"}
        )
        self._brands = {
            normalized: str(value)
            for value in known_brands
            if (normalized := normalize_text(str(value)))
            and len(normalized) >= 3
            and normalized not in blocked_brands
        }
        self._colors = {
            normalize_text(value): normalize_text(value) for value in valid_colors
        }
        self._materials = {
            normalize_text(value): normalize_text(value) for value in valid_materials
        }
        canonical_categories = {
            normalize_text(value) for value in CATEGORY_ALIASES.values()
        }
        requested_categories = {
            normalize_text(value) for value in valid_categories if str(value).strip()
        }
        invalid_categories = requested_categories.difference(canonical_categories)
        if invalid_categories:
            raise ValueError(
                "valid_categories must use canonical free-form keys: "
                f"{sorted(invalid_categories)}"
            )
        self._categories = requested_categories or canonical_categories
        self._category_aliases = {
            normalize_text(alias): canonical
            for alias, canonical in {**CATEGORY_ALIASES, **CATEGORY_SYNONYMS}.items()
            if normalize_text(canonical) in self._categories
        }
        for category in self._categories:
            self._category_aliases.setdefault(category, category)
        if self._model is not None:
            evaluate = getattr(self._model, "eval", None)
            if callable(evaluate):
                evaluate()

    @property
    def model(self) -> Any:
        if self._model is None:
            self.load()
        return self._model

    def load(self) -> float:
        """Load the model lazily and return this call's wall-clock seconds."""

        if self._model is not None:
            return 0.0
        from gliner import GLiNER

        started = time.perf_counter()
        self._model = GLiNER.from_pretrained(
            self.model_id,
            revision=self.revision,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            map_location="cpu",
            low_cpu_mem_usage=True,
        )
        evaluate = getattr(self._model, "eval", None)
        if callable(evaluate):
            evaluate()
        elapsed = time.perf_counter() - started
        self.load_seconds = elapsed
        self.model_loaded_here = True
        return elapsed

    def _missing_fields(self, parsed: FreeFormParse) -> list[str]:
        missing = []
        for field_name in FIELD_LABELS:
            if field_name not in self.enabled_fields:
                continue
            if field_name == "category":
                present = bool(
                    parsed.category
                    or parsed.alternatives.get(field_name)
                    or parsed.excluded.get(field_name)
                    or field_name in parsed.remove_attributes
                )
            else:
                present = bool(
                    parsed.attributes.get(field_name)
                    or parsed.alternatives.get(field_name)
                    or parsed.excluded.get(field_name)
                    or field_name in parsed.remove_attributes
                )
            if not present:
                missing.append(field_name)
        return missing

    def predict(
        self,
        message: str,
        fields: Sequence[str],
        *,
        threshold: float | None = None,
    ) -> tuple[list[EntityPrediction], float]:
        labels = [FIELD_LABELS[field] for field in fields]
        if not labels:
            return [], 0.0
        predictor = getattr(self.model, "predict_entities", None)
        if not callable(predictor):
            raise RuntimeError("loaded GLiNER model has no predict_entities method")
        started = time.perf_counter()
        raw_predictions = predictor(
            message,
            labels,
            threshold=self.threshold if threshold is None else threshold,
        )
        elapsed = time.perf_counter() - started
        predictions = [
            EntityPrediction(
                text=str(item.get("text", "")),
                label=str(item.get("label", "")),
                score=float(item.get("score", 0.0)),
                start=(None if item.get("start") is None else int(item["start"])),
                end=(None if item.get("end") is None else int(item["end"])),
            )
            for item in raw_predictions
            if str(item.get("text", "")).strip()
        ]
        return predictions, elapsed

    def _field_for_label(self, label: str) -> str | None:
        return LABEL_FIELDS.get(normalize_text(label))

    def _validate(self, field_name: str, text: str) -> tuple[str | None, str]:
        normalized = normalize_text(text)
        if field_name == "brand":
            value = self._brands.get(normalized)
            return (value, "catalog brand") if value else (None, "unsupported brand")
        if field_name == "color":
            value = self._colors.get(normalized)
            return (value, "supported color") if value else (None, "unsupported color")
        if field_name == "material":
            value = self._materials.get(normalized)
            return (
                (value, "supported material")
                if value
                else (None, "unsupported material")
            )
        if field_name == "category":
            unwrapped = _strip_category_wrapper(text)
            value = self._category_aliases.get(unwrapped)
            return (
                (value, "validated category alias")
                if value
                else (None, "unsupported category")
            )
        if field_name == "feature":
            value = _phrase_value(text, FEATURE_ALIASES)
            return (
                (value, "controlled soft feature")
                if value
                else (None, "unsupported feature")
            )
        if field_name == "use_case":
            value = _phrase_value(text, USE_CASE_ALIASES)
            return (
                (value, "controlled soft use case")
                if value
                else (None, "unsupported use case")
            )
        if field_name == "size":
            value = _phrase_value(text, SIZE_ALIASES) or normalized
            return (
                (value, "validated soft size")
                if SIZE_RE.fullmatch(value)
                else (None, "unsupported size")
            )
        return None, "unsupported field"

    def augment(
        self,
        message: str,
        deterministic: FreeFormParse,
        *,
        predictions: Sequence[EntityPrediction] | None = None,
        threshold: float | None = None,
    ) -> GLiNERAugmentation:
        """Fill missing fields without changing deterministic decisions."""

        combined = copy.deepcopy(deterministic)
        if (
            deterministic.alternatives
            or deterministic.excluded
            or deterministic.remove_attributes
            or has_authoritative_operator(message)
        ):
            return GLiNERAugmentation(
                parse=combined, skipped_reason="deterministic operator present"
            )

        missing = self._missing_fields(deterministic)
        if not missing:
            return GLiNERAugmentation(parse=combined, skipped_reason="no missing fields")

        inference_seconds = 0.0
        if predictions is None:
            predictions, inference_seconds = self.predict(
                message, missing, threshold=threshold
            )
        active_threshold = self.threshold if threshold is None else threshold
        if not 0.0 <= active_threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        eligible = [
            prediction
            for prediction in predictions
            if self._field_for_label(prediction.label) in missing
        ]
        eligible.sort(
            key=lambda item: (
                -item.score,
                item.start if item.start is not None else -1,
                item.text,
            )
        )

        decisions: list[EntityDecision] = []
        additions: dict[str, list[str]] = {}
        filled: set[str] = set()
        for prediction in eligible:
            field_name = self._field_for_label(prediction.label)
            if field_name is None:
                continue
            required_threshold = (
                max(active_threshold, self.hard_threshold)
                if field_name in HARD_FIELDS
                else active_threshold
            )
            if prediction.score < required_threshold:
                continue
            span_is_valid = bool(
                prediction.start is not None
                and prediction.end is not None
                and 0 <= prediction.start < prediction.end <= len(message)
                and message[prediction.start : prediction.end] == prediction.text
            )
            if not span_is_valid:
                decisions.append(
                    EntityDecision(
                        field=field_name,
                        raw_text=prediction.text,
                        label=prediction.label,
                        score=prediction.score,
                        value=None,
                        accepted=False,
                        hard_constraint=field_name in HARD_FIELDS,
                        reason="invalid or non-verbatim source span",
                    )
                )
                continue
            value, reason = self._validate(field_name, prediction.text)
            accepted = value is not None and field_name not in filled
            if value is not None and field_name in filled:
                reason = "lower-scoring duplicate field"
            decisions.append(
                EntityDecision(
                    field=field_name,
                    raw_text=prediction.text,
                    label=prediction.label,
                    score=prediction.score,
                    value=value,
                    accepted=accepted,
                    hard_constraint=field_name in HARD_FIELDS,
                    reason=reason,
                )
            )
            if not accepted or value is None:
                continue
            filled.add(field_name)
            additions[field_name] = [value]
            if field_name == "category":
                combined.category = value
            else:
                combined.attributes[field_name] = [value]

        return GLiNERAugmentation(
            parse=combined,
            additions=additions,
            decisions=decisions,
            predictions=list(predictions),
            inference_seconds=inference_seconds,
        )


class GLiNERExperimentalAgent:
    """Factory namespace for an Agent subclass without a production import.

    ``create`` avoids making ``starter.agent`` import this optional module.  It
    returns a normal Agent subclass whose only override is the free-form parse
    seam.  Structured evaluator messages still take Agent's earlier template
    branch and never call this override.
    """

    @staticmethod
    def create(augmenter: GLiNERAugmenter, *args: Any, **kwargs: Any) -> Any:
        from starter.agent import Agent

        class _ExperimentalAgent(Agent):
            def __init__(self, *agent_args: Any, **agent_kwargs: Any) -> None:
                super().__init__(*agent_args, **agent_kwargs)
                self.gliner_augmentations: list[GLiNERAugmentation] = []

            def _parse_free_form(self, user_message: str) -> FreeFormParse:
                deterministic = super()._parse_free_form(user_message)
                result = augmenter.augment(user_message, deterministic)
                self.gliner_augmentations.append(result)
                return result.parse

        return _ExperimentalAgent(*args, **kwargs)


__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "DEFAULT_THRESHOLD",
    "DEFAULT_HARD_THRESHOLD",
    "DEFAULT_ENABLED_FIELDS",
    "FIELD_LABELS",
    "HARD_FIELDS",
    "SOFT_FIELDS",
    "EntityPrediction",
    "EntityDecision",
    "GLiNERAugmentation",
    "GLiNERAugmenter",
    "GLiNERExperimentalAgent",
    "has_authoritative_operator",
]
