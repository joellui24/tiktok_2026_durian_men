from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from starter.attribute_index import normalize_value
from starter.hybrid_model import turn_bucket


# Keep this deliberately aligned with evaluator.local_evaluator.  In
# particular, the evaluator's color vocabulary is narrower than the catalog
# attribute index's vocabulary.
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


def classify_constraint(value: str) -> str:
    """Return the evaluator's attribute classification for a constraint."""

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
    if any(
        word in lowered
        for word in ("department", "style", "fit", "sleeve", "neck")
    ):
        return "style"
    if any(
        word in lowered
        for word in ("hiking", "running", "gym", "winter", "outdoor", "work")
    ):
        return "use_case"
    return "feature"


def normalize_constraint(value: str) -> str:
    """Normalize disclosed constraint text identically for train and runtime."""

    cleaned = re.sub(r"\s+", " ", value).strip(" -;,.\t\n")
    return normalize_value(cleaned)


def constraint_context_features(attribute: str, value: str) -> tuple[str, ...]:
    """Encode one disclosed constraint as context feature names.

    An answer to the evaluator's ``other`` question retains both its provenance
    and free-form meaning.  The evaluator classifier also supplies the typed
    feature; its deliberate default to ``feature`` makes every non-empty
    evaluator constraint classifiable without a separate runtime heuristic.
    """

    normalized = normalize_constraint(value)
    if not normalized:
        return ()

    normalized_attribute = attribute.strip().casefold()
    if normalized_attribute != "other":
        return (f"ctx:{normalized_attribute}={normalized}",)

    inferred_attribute = classify_constraint(value)
    return (
        "ctx:answer_source=other",
        f"ctx:other={normalized}",
        f"ctx:{inferred_attribute}={normalized}",
    )


def context_feature_names(
    *,
    coarse_category: str,
    scenario_state: str,
    turn: int,
    intent_epoch: int,
    known_constraints: Mapping[str, Sequence[str]],
) -> list[str]:
    """Build the canonical ordered context names used by training and runtime."""

    features = [
        f"ctx:category={normalize_value(coarse_category)}",
        f"ctx:scenario={scenario_state}",
        f"ctx:turn={turn_bucket(turn)}",
        f"ctx:override={'post' if intent_epoch else 'pre'}",
    ]
    for attribute, values in sorted(known_constraints.items()):
        for value in values:
            features.extend(constraint_context_features(attribute, value))

    # Multiple OTHER values share one answer-source marker, and duplicated
    # disclosures should not make a binary sparse feature appear more than once.
    return list(dict.fromkeys(features))


def legacy_context_feature_names(
    *,
    coarse_category: str,
    scenario_state: str,
    turn: int,
    intent_epoch: int,
    known_constraints: Mapping[str, Sequence[str]],
) -> list[str]:
    """Reconstruct the feature schema used by frozen pre-v2 artifacts.

    This intentionally does not dual-encode ``OTHER``.  Keeping the legacy
    path shared by runtime and offline evaluation prevents an E0 baseline from
    changing merely because the current feature extractor was improved.
    """

    features = [
        f"ctx:category={normalize_value(coarse_category)}",
        f"ctx:scenario={scenario_state}",
        f"ctx:turn={turn_bucket(turn)}",
        f"ctx:override={'post' if intent_epoch else 'pre'}",
    ]
    for attribute, values in sorted(known_constraints.items()):
        features.extend(
            f"ctx:{attribute}={normalize_value(value)}" for value in values
        )
    return list(dict.fromkeys(features))
