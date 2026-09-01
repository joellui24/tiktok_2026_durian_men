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
COLOR_ALIASES = {
    **{color: color for color in COLORS},
    "jet black": "black",
    "jet-black": "black",
    "crimson": "red",
    "navy blue": "blue",
    "navy": "blue",
}
COLOR_IDIOM_RE = re.compile(
    r"\b(?:black\s+friday|black[-\s]+tie|red[-\s]+carpet|"
    r"blue[-\s]+collar|white[-\s]+collar)\b"
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
    "road runners": "running shoes",
    "road runner": "running shoes",
    "running footwear": "running shoes",
    "walking footwear": "walking shoes",
    "footwear for walking": "walking shoes",
    "hooded sweatshirts": "hoodies",
    "hooded sweatshirt": "hoodies",
    "flip-flops": "sandals",
    "flip-flop": "sandals",
    "flip flops": "sandals",
    "flip flop": "sandals",
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
    "runners": "running shoes",
    "runner": "running shoes",
    "footwear": "shoes",
}

USE_CASE_ALIASES = {
    "jogging": "running",
    "jog": "running",
    "running": "running",
    "run": "running",
    "runs": "running",
    "walking": "walking",
    "walk": "walking",
    "hiking": "hiking",
    "hike": "hiking",
    "gym": "gym",
    "workout": "gym",
    "beach": "beach",
    "summer": "summer",
    "winter": "winter",
    "outdoor": "outdoor",
    "formal": "formal",
    "travel": "travel",
    "travelling": "travel",
    "traveling": "travel",
    "sightseeing": "travel",
    "on foot": "walking",
    "lounge": "lounge",
    "lounging": "lounge",
}

FEATURE_ALIASES = {
    "comfortable": "comfort",
    "comfort": "comfort",
    "comfy": "comfort",
    "cushioned": "cushioned",
    "cushioning": "cushioned",
    "stylish": "style",
    "breathable": "breathable",
    "lightweight": "lightweight",
    "supportive": "supportive",
    "durable": "durable",
    "waterproof": "waterproof",
    "soft": "soft",
    "warm": "warm",
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
    "refresh",
} | set(COLORS) | set(MATERIALS) | set(USE_CASE_ALIASES) | set(FEATURE_ALIASES)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
    # Preserve a numeric token before general punctuation cleanup. Without
    # this, "$1,200" becomes "$1 200" and the hard-price parser sees $1.
    value = re.sub(r"(?<=\d),(?=\d{3}\b)", "", value)
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


def _alias_mentions(
    text: str, aliases: dict[str, str]
) -> list[tuple[str, int, int]]:
    """Return ordered, non-overlapping canonical alias mentions."""

    mentions: list[tuple[str, int, int]] = []
    occupied: list[tuple[int, int]] = []
    for phrase in sorted(aliases, key=len, reverse=True):
        for match in re.finditer(
            rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text
        ):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            mentions.append((aliases[phrase], span[0], span[1]))
            occupied.append(span)
    mentions.sort(key=lambda item: item[1])
    return mentions


def _is_negated(text: str, value: str) -> bool:
    value = normalize_text(value)
    optional_noun = r"(?:a\s+|an\s+|any\s+)?"
    patterns = (
        rf"\bnot\s+(?:for\s+)?{optional_noun}{re.escape(value)}\b",
        rf"\bno\s+{optional_noun}{re.escape(value)}\b",
        rf"\banything\s+(?:but|except)\s+{re.escape(value)}\b",
        rf"\b(?:except|exclude|excluding|avoid|without)\s+{optional_noun}{re.escape(value)}\b",
        rf"\bother\s+than\s+{re.escape(value)}\b",
        rf"\b(?:do\s+not|don't|dont)\s+want\s+{optional_noun}{re.escape(value)}\b",
        rf"\b(?:instead\s+of|rather\s+than)\s+{re.escape(value)}\b",
        rf"\b{re.escape(value)}\s+(?:is\s+)?irrelevant\b",
        rf"\bnon[-\s]+{re.escape(value)}\b",
        rf"\b{re.escape(value)}[-\s]+free\b",
        rf"\bfree\s+of\s+{re.escape(value)}\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _uses_or(text: str, mentions: list[tuple[str, int, int]]) -> bool:
    if len(mentions) < 2:
        return False
    between = text[mentions[0][1] : mentions[-1][2]]
    return bool(
        re.search(r"\bor\b", between)
        or re.search(r"\beither\b", text[: mentions[0][1] + 1])
        or re.search(r"\bworks?\s+too\b", text[mentions[0][1] :])
    )


def _negation_flags(
    text: str, mentions: list[tuple[str, int, int]]
) -> list[bool]:
    """Carry a negation across a simple same-field ``and/or`` list."""

    flags = [
        _is_negated(text, text[start:end]) for _, start, end in mentions
    ]
    for index in range(1, len(mentions)):
        if flags[index] or not flags[index - 1]:
            continue
        connector = text[mentions[index - 1][2] : mentions[index][1]]
        if re.fullmatch(r"\s*(?:,\s*)?(?:and|or)\s*", connector):
            flags[index] = True
    return flags


def _store_mentions(
    parsed: "FreeFormParse",
    attribute: str,
    text: str,
    mentions: list[tuple[str, int, int]],
) -> None:
    negated = _negation_flags(text, mentions)
    positives = list(
        dict.fromkeys(
            value
            for (value, _, _), is_negated in zip(mentions, negated, strict=True)
            if not is_negated
        )
    )
    negatives = list(
        dict.fromkeys(
            value
            for (value, _, _), is_negated in zip(mentions, negated, strict=True)
            if is_negated
        )
    )
    if negatives:
        parsed.excluded[attribute] = negatives
    if _uses_or(
        text, [mention for mention in mentions if mention[0] in positives]
    ):
        parsed.alternatives[attribute] = positives
    elif positives:
        parsed.attributes[attribute] = positives


def _removed_attributes(text: str) -> set[str]:
    aliases = {
        "color": r"colou?r",
        "brand": r"brand",
        "material": r"material|fabric",
        "size": r"size",
        "budget": r"budget|price",
        "feature": r"feature",
        "use_case": r"use(?:\s+case)?|occasion",
    }
    removed: set[str] = set()
    for attribute, phrase in aliases.items():
        patterns = (
            rf"\b(?:{phrase})\s+(?:(?:doesn't|does\s+not|doesnt)\s+matter|is\s+(?:irrelevant|unimportant)|is\s+not\s+important)\b",
            rf"\b(?:don't|do\s+not|dont)\s+(?:really\s+)?care\s+about\s+(?:the\s+)?(?:{phrase})\b",
            rf"\bno\s+preference\s+(?:for|on)\s+(?:the\s+)?(?:{phrase})\b",
            rf"\bany\s+(?:{phrase})\s+(?:is\s+)?(?:fine|okay|ok|works)\b",
            rf"\b(?:remove|clear|forget)\s+(?:the\s+|my\s+)?(?:{phrase})(?:\s+preference)?\b",
        )
        if any(re.search(pattern, text) for pattern in patterns):
            removed.add(attribute)
    if re.search(
        r"\b(?:no|without)\s+(?:a\s+)?(?:budget|price)\s+(?:limit|cap|ceiling)\b|"
        r"\b(?:no|unlimited)\s+budget\b",
        text,
    ):
        removed.add("budget")
    return removed


def _maximum_price(text: str) -> tuple[float, bool] | None:
    """Return the numeric ceiling and whether the boundary is inclusive."""

    patterns = (
        (
            r"\b(?:under|below|less\s+than)\s*(?:usd\s*)?\$?\s*"
            r"(\d+(?:\.\d{1,2})?)\s*(?:dollars?)?\b",
            False,
        ),
        (
            r"\bkeep\s+it\s+(?:under|below)\s*(?:usd\s*)?\$?\s*"
            r"(\d+(?:\.\d{1,2})?)\s*(?:dollars?)?\b",
            False,
        ),
        (
            r"\b(?:capped|priced)\s+at\s*(?:usd\s*)?\$?\s*"
            r"(\d+(?:\.\d{1,2})?)\s*(?:dollars?)?\b",
            True,
        ),
        (
            r"\b(?:up\s+to|at\s+most|no\s+more\s+than|not\s+over|"
            r"maximum(?:\s+of)?|max)\s*(?:usd\s*)?\$?\s*"
            r"(\d+(?:\.\d{1,2})?)\s*(?:dollars?)?\b",
            True,
        ),
        (
            r"\bbudget\s+(?:is|of|around)?\s*(?:usd\s*)?\$?\s*"
            r"(\d+(?:\.\d{1,2})?)\s*(?:dollars?)?\b",
            True,
        ),
        (
            r"\b(?:cap\s+it|raise\s+(?:the\s+)?limit|lower\s+(?:the\s+)?limit|"
            r"set\s+(?:the\s+)?limit)\s+(?:at|to)\s*(?:usd\s*)?\$?\s*"
            r"(\d+(?:\.\d{1,2})?)\s*(?:dollars?)?\b",
            True,
        ),
        (
            r"\b(?:my\s+)?(?:limit|cap|ceiling)\s+(?:is|of|at)?\s*"
            r"(?:usd\s*)?\$?\s*(\d+(?:\.\d{1,2})?)\s*(?:dollars?)?\b",
            True,
        ),
    )
    for pattern, inclusive in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1)), inclusive
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
    maximum_price_inclusive: bool = False
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
    parsed = FreeFormParse()
    parsed.remove_attributes = _removed_attributes(text)

    # Resolve complete known-brand spans first. Attribute-looking words inside
    # a brand such as "Cotton On" must not become unrelated hard filters.
    brand_mentions: list[tuple[str, int, int]] = []
    occupied_brand_spans: list[tuple[int, int]] = []
    for brand in sorted(set(known_brands), key=len, reverse=True):
        normalized_brand = normalize_text(brand)
        if len(normalized_brand) < 3 or normalized_brand in BRAND_STOPWORDS:
            continue
        for match in re.finditer(
            rf"(?<![a-z0-9]){re.escape(normalized_brand)}(?![a-z0-9])", text
        ):
            if any(
                match.start() < end and start < match.end()
                for start, end in occupied_brand_spans
            ):
                continue
            brand_mentions.append((brand, match.start(), match.end()))
            occupied_brand_spans.append(match.span())
    brand_mentions.sort(key=lambda item: item[1])
    brand_spans = [(start, end) for _, start, end in brand_mentions]

    def outside_brand(start: int, end: int) -> bool:
        return not any(
            brand_start <= start and end <= brand_end
            for brand_start, brand_end in brand_spans
        )

    category_mentions = [
        mention
        for mention in _alias_mentions(text, CATEGORY_ALIASES)
        if outside_brand(mention[1], mention[2])
    ]
    category_negated = _negation_flags(text, category_mentions)
    positive_categories = [
        mention
        for mention, is_negated in zip(
            category_mentions, category_negated, strict=True
        )
        if not is_negated
    ]
    negative_categories = [
        mention
        for mention, is_negated in zip(
            category_mentions, category_negated, strict=True
        )
        if is_negated
    ]
    if negative_categories:
        parsed.excluded["category"] = list(
            dict.fromkeys(value for value, _, _ in negative_categories)
        )
    if _uses_or(text, positive_categories):
        parsed.alternatives["category"] = list(
            dict.fromkeys(value for value, _, _ in positive_categories)
        )
    elif positive_categories:
        # A broad wrapper such as "footwear" must not hide a more specific
        # exact phrase later in the request ("footwear for road runners").
        parsed.category = next(
            (value for value, _, _ in positive_categories if value != "shoes"),
            positive_categories[0][0],
        )

    if "color" not in parsed.remove_attributes:
        protected_color_spans = [match.span() for match in COLOR_IDIOM_RE.finditer(text)]
        color_mentions = [
            mention
            for mention in _alias_mentions(text, COLOR_ALIASES)
            if outside_brand(mention[1], mention[2])
            and not any(
                start <= mention[1] and mention[2] <= end
                for start, end in protected_color_spans
            )
        ]
        _store_mentions(parsed, "color", text, color_mentions)

    if "material" not in parsed.remove_attributes:
        material_mentions = [
            (material, match.start(), match.end())
            for material in MATERIALS
            for match in re.finditer(
                rf"(?<![a-z0-9]){re.escape(material)}(?![a-z0-9])", text
            )
            if outside_brand(match.start(), match.end())
            and not re.search(
                r"\b(?:vegan|faux|synthetic|artificial|pu)\s+$",
                text[max(0, match.start() - 16) : match.start()],
            )
            and not re.match(
                r"(?:-like|\s+like|\s+look(?:ing)?)\b",
                text[match.end() : match.end() + 12],
            )
        ]
        _store_mentions(
            parsed, "material", text, sorted(material_mentions, key=lambda item: item[1])
        )

    if "brand" not in parsed.remove_attributes:
        _store_mentions(
            parsed, "brand", text, brand_mentions
        )

    use_case_mentions = [
        mention
        for mention in _alias_mentions(text, USE_CASE_ALIASES)
        if outside_brand(mention[1], mention[2])
        if not any(
            category_start <= mention[1] and mention[2] <= category_end
            for _, category_start, category_end in negative_categories
        )
    ]
    if "use_case" not in parsed.remove_attributes:
        _store_mentions(parsed, "use_case", text, use_case_mentions)
    feature_mentions = [
        mention
        for mention in _alias_mentions(text, FEATURE_ALIASES)
        if outside_brand(mention[1], mention[2])
    ]
    if "feature" not in parsed.remove_attributes:
        _store_mentions(parsed, "feature", text, feature_mentions)
    use_cases = parsed.attributes.get("use_case", []) + parsed.alternatives.get(
        "use_case", []
    )
    if parsed.category in {None, "shoes"}:
        if "running" in use_cases:
            parsed.category = "running shoes"
        elif "walking" in use_cases:
            parsed.category = "walking shoes"

    if "size" not in parsed.remove_attributes:
        size_match = re.search(
            r"\bsize\s+(?:is\s+)?([0-9]{1,2}(?:\.[05])?|[xsml]{1,3})"
            r"(?:\s+or\s+([0-9]{1,2}(?:\.[05])?|[xsml]{1,3}))?\b",
            text,
        )
        if size_match:
            sizes = [value for value in size_match.groups() if value]
            if len(sizes) > 1:
                parsed.alternatives["size"] = sizes
            else:
                parsed.attributes["size"] = sizes

    if "budget" not in parsed.remove_attributes:
        maximum_price = _maximum_price(text)
        if maximum_price is not None:
            parsed.maximum_price, parsed.maximum_price_inclusive = maximum_price
    if parsed.maximum_price is None and re.search(
        r"\b(?:cheap|budget-friendly|inexpensive|affordable)\b", text
    ):
        parsed.qualitative_budget = "affordable"

    strong_browsing_cue = bool(
        re.search(
            r"\b(?:some cool things|something nice|ideas? for|still exploring|"
            r"just browsing|general ideas|no rush)\b",
            text,
        )
    )
    browsing_cue = bool(
        strong_browsing_cue
        or (re.search(r"\bshow me\b", text) and not parsed.has_constraints)
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
