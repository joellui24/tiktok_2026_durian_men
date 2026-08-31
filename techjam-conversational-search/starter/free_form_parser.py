from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable


COLORS = (
    "black",
    "white",
    "blue",
    "red",
    "pink",
    "green",
    "brown",
    "gray",
    "grey",
    "purple",
    "yellow",
    "orange",
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
)

# Values are canonical query labels. Candidate resolution remains catalog-backed
# in Agent, so these aliases do not invent product IDs or catalog attributes.
CATEGORY_ALIASES = {
    "road running shoes": "running shoes",
    "trail running shoes": "running shoes",
    "running sneakers": "running shoes",
    "running shoe": "running shoes",
    "running shoes": "running shoes",
    "walking shoe": "walking shoes",
    "walking shoes": "walking shoes",
    "fashion sneaker": "sneakers",
    "fashion sneakers": "sneakers",
    "sneaker": "sneakers",
    "sneakers": "sneakers",
    "sandal": "sandals",
    "sandals": "sandals",
    "shirt": "shirts",
    "shirts": "shirts",
    "t shirt": "shirts",
    "t shirts": "shirts",
    "tshirt": "shirts",
    "tshirts": "shirts",
    "tunic": "tunics",
    "tunics": "tunics",
    "shoe": "shoes",
    "shoes": "shoes",
    "boot": "boots",
    "boots": "boots",
    "dress": "dresses",
    "dresses": "dresses",
    "jacket": "jackets",
    "jackets": "jackets",
    "jean": "jeans",
    "jeans": "jeans",
    "pants": "pants",
    "trousers": "pants",
    "skirt": "skirts",
    "skirts": "skirts",
    "sock": "socks",
    "socks": "socks",
    "belt": "belts",
    "belts": "belts",
    "watch": "watches",
    "watches": "watches",
    "earring": "earrings",
    "earrings": "earrings",
    "necklace": "necklaces",
    "necklaces": "necklaces",
    "ring": "rings",
    "rings": "rings",
    "slipper": "slippers",
    "slippers": "slippers",
    "loafers": "loafers",
    "pumps": "pumps",
    "flats": "flats",
    "hoodie": "hoodies",
    "hoodies": "hoodies",
    "hat": "hats",
    "hats": "hats",
    "sunglasses": "sunglasses",
}

USE_CASE_ALIASES = {
    "jogging": "running",
    "jog": "running",
    "running": "running",
    "run": "running",
    "walking": "walking",
    "walk": "walking",
    "hiking": "hiking",
    "hike": "hiking",
    "gym": "gym",
    "workout": "gym",
    "beach": "beach",
    "winter": "winter",
    "outdoor": "outdoor",
}

FEATURE_ALIASES = {
    "comfortable": "comfort",
    "comfort": "comfort",
    "comfy": "comfort",
    "cushioned": "cushioned",
    "cushioning": "cushioned",
    "stylish": "style",
}

BRAND_STOPWORDS = {
    "find",
    "need",
    "want",
    "something",
    "style",
    "fashion",
    "men",
    "women",
    "unisex",
    "generic",
    "unknown",
}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
    return re.sub(r"[^a-z0-9$.'-]+", " ", value).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _first_alias(text: str, aliases: dict[str, str]) -> str | None:
    for phrase in sorted(aliases, key=len, reverse=True):
        if _contains_phrase(text, phrase):
            return aliases[phrase]
    return None


def _all_aliases(text: str, aliases: dict[str, str]) -> list[str]:
    return list(
        dict.fromkeys(
            aliases[phrase]
            for phrase in sorted(aliases, key=len, reverse=True)
            if _contains_phrase(text, phrase)
        )
    )


def _is_negated(text: str, value: str) -> bool:
    patterns = (
        rf"\bnot\s+(?:a\s+)?{re.escape(value)}\b",
        rf"\bno\s+{re.escape(value)}\b",
        rf"\banything\s+but\s+{re.escape(value)}\b",
        rf"\bexclude\s+{re.escape(value)}\b",
        rf"\bwithout\s+{re.escape(value)}\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _maximum_price(text: str) -> float | None:
    patterns = (
        r"\b(?:under|below|less\s+than|up\s+to|at\s+most|maximum(?:\s+of)?|max)\s*"
        r"(?:usd\s*)?\$?\s*(\d+(?:\.\d{1,2})?)\s*(?:dollars?)?\b",
        r"\bbudget\s+(?:is|of|around)?\s*(?:usd\s*)?\$?\s*"
        r"(\d+(?:\.\d{1,2})?)\s*(?:dollars?)?\b",
        r"\bkeep\s+it\s+(?:under|below)\s*(?:usd\s*)?\$?\s*"
        r"(\d+(?:\.\d{1,2})?)\s*(?:dollars?)?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


@dataclass
class FreeFormParse:
    intent: str = "unknown"
    category: str | None = None
    attributes: dict[str, list[str]] = field(default_factory=dict)
    alternatives: dict[str, list[str]] = field(default_factory=dict)
    excluded: dict[str, list[str]] = field(default_factory=dict)
    remove_attributes: set[str] = field(default_factory=set)
    maximum_price: float | None = None
    qualitative_budget: str | None = None

    @property
    def has_constraints(self) -> bool:
        return bool(
            self.attributes
            or self.alternatives
            or self.excluded
            or self.maximum_price is not None
            or self.qualitative_budget
        )


def parse_free_form_message(
    user_message: str,
    *,
    known_brands: Iterable[str] = (),
) -> FreeFormParse:
    """Conservatively extract typed shopping slots from non-template wording.

    Exact vocabulary and explicit syntax are preferred. This function does not
    fuzzy-match hard constraints; an uncertain value is safer as a soft signal
    than as a candidate-removing filter.
    """

    text = normalize_text(user_message)
    parsed = FreeFormParse(category=_first_alias(text, CATEGORY_ALIASES))

    if re.search(
        r"\b(?:colou?r|brand|material|size|budget)\s+(?:doesn't|does not|doesnt)\s+matter\b",
        text,
    ):
        attribute_match = re.search(
            r"\b(colou?r|brand|material|size|budget)\s+", text
        )
        if attribute_match:
            parsed.remove_attributes.add(
                "color" if attribute_match.group(1).startswith("colo") else attribute_match.group(1)
            )
    if re.search(r"\b(?:don't|do not|dont)\s+(?:really\s+)?care\s+about\s+brand\b", text):
        parsed.remove_attributes.add("brand")

    colors = sorted(
        (color for color in COLORS if _contains_phrase(text, color)),
        key=text.find,
    )
    positive_colors = [color for color in colors if not _is_negated(text, color)]
    excluded_colors = [color for color in colors if _is_negated(text, color)]
    if excluded_colors:
        parsed.excluded["color"] = excluded_colors
    if len(positive_colors) > 1 and re.search(
        rf"\b(?:{'|'.join(map(re.escape, positive_colors))})\b\s+or\s+"
        rf"\b(?:{'|'.join(map(re.escape, positive_colors))})\b",
        text,
    ):
        parsed.alternatives["color"] = positive_colors
    elif positive_colors:
        parsed.attributes["color"] = positive_colors

    materials = sorted(
        (material for material in MATERIALS if _contains_phrase(text, material)),
        key=text.find,
    )
    positive_materials = [
        material for material in materials if not _is_negated(text, material)
    ]
    excluded_materials = [
        material for material in materials if _is_negated(text, material)
    ]
    if positive_materials:
        parsed.attributes["material"] = positive_materials
    if excluded_materials:
        parsed.excluded["material"] = excluded_materials

    if "brand" not in parsed.remove_attributes:
        brands = []
        for brand in sorted(set(known_brands), key=len, reverse=True):
            normalized_brand = normalize_text(brand)
            if len(normalized_brand) < 3 or normalized_brand in BRAND_STOPWORDS:
                continue
            if _contains_phrase(text, normalized_brand) and not _is_negated(
                text, normalized_brand
            ):
                brands.append(brand)
        if brands:
            parsed.attributes["brand"] = [brands[0]]

    use_cases = _all_aliases(text, USE_CASE_ALIASES)
    if use_cases:
        parsed.attributes["use_case"] = use_cases
    features = _all_aliases(text, FEATURE_ALIASES)
    if features:
        parsed.attributes["feature"] = features
    if parsed.category is None and features:
        if "running" in use_cases:
            parsed.category = "running shoes"
        elif "walking" in use_cases:
            parsed.category = "walking shoes"

    size_match = re.search(
        r"\bsize\s+(?:is\s+)?([0-9]{1,2}(?:\.[05])?|[xsml]{1,3})\b", text
    )
    if size_match:
        parsed.attributes["size"] = [size_match.group(1)]

    parsed.maximum_price = _maximum_price(text)
    if parsed.maximum_price is None and re.search(
        r"\b(?:cheap|budget-friendly|inexpensive|affordable)\b", text
    ):
        parsed.qualitative_budget = "affordable"

    browsing_cue = bool(
        re.search(
            r"\b(?:show me|some cool things|something nice|ideas? for|still exploring)\b",
            text,
        )
        or ("beach" in text and parsed.category is None and not parsed.maximum_price)
    )
    if browsing_cue:
        parsed.intent = "browsing"
    elif parsed.has_constraints:
        parsed.intent = "buying"
    else:
        # A broad category-only request should trigger clarification rather than
        # forcing the linear Buying route.
        parsed.intent = "unknown"
    return parsed
