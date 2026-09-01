"""Frozen unseen corpus for the rules-versus-GLiNER experiment.

IMPORTANT: This corpus was written and frozen before any GLiNER output was
inspected.  It must not be edited in response to model successes or failures.
Create a separately versioned corpus if additional examples are needed.

Only syntax and structural validation are permitted before the first scored
run.  ``corpus_sha256()`` serializes the labels deterministically, and
``FROZEN_CORPUS_SHA256`` records the version-one manifest digest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


CORPUS_VERSION = "gliner-unseen-v1"
FROZEN_BEFORE_GLINER_EVALUATION = True
FROZEN_ON = "2026-08-31"


@dataclass(frozen=True)
class QueryCase:
    case_id: str
    split: str
    group: str
    message: str
    intent: str = "unknown"
    category: str | None = None
    attributes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    alternatives: dict[str, tuple[str, ...]] = field(default_factory=dict)
    excluded: dict[str, tuple[str, ...]] = field(default_factory=dict)
    remove_attributes: frozenset[str] = frozenset()
    maximum_price: float | None = None
    qualitative_budget: str | None = None


def Q(case_id: str, group: str, message: str, **expected: object) -> QueryCase:
    return QueryCase(case_id, "unseen", group, message, **expected)


CASES = (
    # Natural paraphrases whose phrasing is absent from the historical corpus.
    Q("GLQ001", "paraphrase", "I'm after jet-black footwear for weekend runs", intent="buying", category="running shoes", attributes={"color": ("black",), "use_case": ("running",)}),
    Q("GLQ002", "paraphrase", "Could use a sky-blue cotton tee", intent="buying", category="shirts", attributes={"color": ("blue",), "material": ("cotton",)}),
    Q("GLQ003", "paraphrase", "Hoping to pick up brown boots made of leather", intent="buying", category="boots", attributes={"color": ("brown",), "material": ("leather",)}),
    Q("GLQ004", "paraphrase", "Find a pair of Adidas kicks for gym sessions", intent="buying", category="sneakers", attributes={"brand": ("Adidas",), "use_case": ("gym",)}),
    Q("GLQ005", "paraphrase", "I'd love a crimson silk dress for a formal dinner", intent="buying", category="dresses", attributes={"color": ("red",), "material": ("silk",), "use_case": ("formal",)}),
    Q("GLQ006", "paraphrase", "On the hunt for a warm wool coat", intent="buying", category="jackets", attributes={"feature": ("warm",), "material": ("wool",)}),
    Q("GLQ007", "paraphrase", "Can you source white Puma footwear for walking?", intent="buying", category="walking shoes", attributes={"brand": ("Puma",), "color": ("white",), "use_case": ("walking",)}),
    Q("GLQ008", "paraphrase", "I'd be keen on some green trousers in polyester", intent="buying", category="pants", attributes={"color": ("green",), "material": ("polyester",)}),
    Q("GLQ009", "paraphrase", "Please track down a stylish black skirt", intent="buying", category="skirts", attributes={"feature": ("style",), "color": ("black",)}),
    Q("GLQ010", "paraphrase", "I'm trying to find soft socks for home lounging", intent="buying", category="socks", attributes={"feature": ("soft",), "use_case": ("lounge",)}),

    # Category synonyms and colloquial product names.
    Q("GLQ011", "category_synonym", "Need navy road runners for training", intent="buying", category="running shoes", attributes={"color": ("blue",), "use_case": ("running",)}),
    Q("GLQ012", "category_synonym", "Show me casual trainers in white", intent="buying", category="sneakers", attributes={"color": ("white",)}),
    Q("GLQ013", "category_synonym", "Looking for beach flip-flops in yellow", intent="buying", category="sandals", attributes={"color": ("yellow",), "use_case": ("beach",)}),
    Q("GLQ014", "category_synonym", "A medium crew-neck tee in cotton, please", intent="buying", category="shirts", attributes={"size": ("m",), "material": ("cotton",)}),
    Q("GLQ015", "category_synonym", "Do you have office trousers in grey?", intent="buying", category="pants", attributes={"color": ("grey",), "use_case": ("work",)}),
    Q("GLQ016", "category_synonym", "Some polarized shades for travelling", intent="buying", category="sunglasses", attributes={"feature": ("polarized",), "use_case": ("travel",)}),
    Q("GLQ017", "category_synonym", "A warm hooded sweatshirt for winter", intent="buying", category="hoodies", attributes={"feature": ("warm",), "use_case": ("winter",)}),
    Q("GLQ018", "category_synonym", "Black ankle booties with cushioning", intent="buying", category="boots", attributes={"color": ("black",), "feature": ("cushioned",)}),
    Q("GLQ019", "category_synonym", "Brown slip-on loafers for the office", intent="buying", category="loafers", attributes={"color": ("brown",), "use_case": ("work",)}),
    Q("GLQ020", "category_synonym", "Red court heels for a formal occasion", intent="buying", category="pumps", attributes={"color": ("red",), "use_case": ("formal",)}),

    # Explicit buying requests with several simultaneous hard constraints.
    Q("GLQ021", "hard_buying", "Men's size 11 black Nike runners capped at $135", intent="buying", category="running shoes", attributes={"size": ("11",), "color": ("black",), "brand": ("Nike",), "use_case": ("running",)}, maximum_price=135.0),
    Q("GLQ022", "hard_buying", "Find size 7 white Adidas walking shoes for no more than 95 dollars", intent="buying", category="walking shoes", attributes={"size": ("7",), "color": ("white",), "brand": ("Adidas",), "use_case": ("walking",)}, maximum_price=95.0),
    Q("GLQ023", "hard_buying", "I need an XL blue polyester jacket below $80", intent="buying", category="jackets", attributes={"size": ("xl",), "color": ("blue",), "material": ("polyester",)}, maximum_price=80.0),
    Q("GLQ024", "hard_buying", "Get me a medium red cotton shirt under forty-five bucks", intent="buying", category="shirts", attributes={"size": ("m",), "color": ("red",), "material": ("cotton",)}, maximum_price=45.0),
    Q("GLQ025", "hard_buying", "Brown leather loafers, size 9, budget ceiling $160", intent="buying", category="loafers", attributes={"size": ("9",), "color": ("brown",), "material": ("leather",)}, maximum_price=160.0),
    Q("GLQ026", "hard_buying", "A warm grey wool hoodie under $110 in large", intent="buying", category="hoodies", attributes={"feature": ("warm",), "color": ("grey",), "material": ("wool",), "size": ("l",)}, maximum_price=110.0),
    Q("GLQ027", "hard_buying", "Waterproof black Columbia boots for hiking, EU 43, below $180", intent="buying", category="boots", attributes={"feature": ("waterproof",), "color": ("black",), "brand": ("Columbia",), "use_case": ("hiking",), "size": ("eu 43",)}, maximum_price=180.0),
    Q("GLQ028", "hard_buying", "Purple silk dress for a formal event, size small, maximum $220", intent="buying", category="dresses", attributes={"color": ("purple",), "material": ("silk",), "use_case": ("formal",), "size": ("s",)}, maximum_price=220.0),
    Q("GLQ029", "hard_buying", "Comfortable Skechers sneakers in black, US 10, under $105", intent="buying", category="sneakers", attributes={"feature": ("comfort",), "brand": ("Skechers",), "color": ("black",), "size": ("us 10",)}, maximum_price=105.0),
    Q("GLQ030", "hard_buying", "Orange nylon gym bag-style backpack under $65", intent="buying", attributes={"color": ("orange",), "material": ("nylon",), "use_case": ("gym",)}, maximum_price=65.0),

    # Exploratory requests should retain browsing intent and avoid invented fields.
    Q("GLQ031", "browsing", "I'm gathering outfit ideas for a seaside break", intent="browsing", attributes={"use_case": ("beach",)}),
    Q("GLQ032", "browsing", "What kinds of things work for a snowy holiday?", intent="browsing", attributes={"use_case": ("winter",)}),
    Q("GLQ033", "browsing", "Let me browse a few polished office looks", intent="browsing", attributes={"use_case": ("work",)}),
    Q("GLQ034", "browsing", "Could you inspire me for an upcoming gym refresh?", intent="browsing", attributes={"use_case": ("gym",)}),
    Q("GLQ035", "browsing", "I'm exploring what people wear on long flights", intent="browsing", attributes={"use_case": ("travel",)}),
    Q("GLQ036", "browsing", "Show me a mix of smart options for a gala", intent="browsing", attributes={"use_case": ("formal",)}),
    Q("GLQ037", "browsing", "Surprise me with breathable summer pieces", intent="browsing", attributes={"feature": ("breathable",), "use_case": ("summer",)}),
    Q("GLQ038", "browsing", "I'm only window-shopping for comfortable everyday gear", intent="browsing", attributes={"feature": ("comfort",)}),
    Q("GLQ039", "browsing", "Give me a few outdoorsy style directions", intent="browsing", attributes={"use_case": ("outdoor",)}),
    Q("GLQ040", "browsing", "No fixed item in mind; suggest something for a beach day", intent="browsing", attributes={"use_case": ("beach",)}),

    # Vague or indirectly expressed use cases.
    Q("GLQ041", "vague_use_case", "I need footwear that can handle hours of sightseeing on foot", intent="buying", category="walking shoes", attributes={"use_case": ("walking",)}),
    Q("GLQ042", "vague_use_case", "A pair for pounding the pavement every morning", intent="buying", category="running shoes", attributes={"use_case": ("running",)}),
    Q("GLQ043", "vague_use_case", "What could I take on a trek through muddy hills?", intent="browsing", attributes={"use_case": ("hiking",)}),
    Q("GLQ044", "vague_use_case", "Something I can wear while lifting weights", intent="browsing", attributes={"use_case": ("gym",)}),
    Q("GLQ045", "vague_use_case", "Clothes that won't leave me freezing in January", intent="browsing", attributes={"use_case": ("winter",)}),
    Q("GLQ046", "vague_use_case", "A dress appropriate for a black-tie reception", intent="buying", category="dresses", attributes={"use_case": ("formal",)}),
    Q("GLQ047", "vague_use_case", "Footwear for sand, sea and hot weather", intent="browsing", attributes={"use_case": ("beach",)}),
    Q("GLQ048", "vague_use_case", "A jacket to bring on a city break abroad", intent="buying", category="jackets", attributes={"use_case": ("travel",)}),
    Q("GLQ049", "vague_use_case", "Shoes suitable for shifts where I stand all day", intent="buying", category="shoes", attributes={"use_case": ("work",)}),
    Q("GLQ050", "vague_use_case", "Soft things to relax in on the sofa", intent="browsing", attributes={"feature": ("soft",), "use_case": ("lounge",)}),

    # Semantic feature language, including phrases rather than label words.
    Q("GLQ051", "feature", "Shoes that won't destroy my feet after twelve hours", intent="buying", category="shoes", attributes={"feature": ("comfort",)}),
    Q("GLQ052", "feature", "Running shoes with a pillowy feel underfoot", intent="buying", category="running shoes", attributes={"feature": ("cushioned",), "use_case": ("running",)}),
    Q("GLQ053", "feature", "A jacket that barely weighs anything", intent="buying", category="jackets", attributes={"feature": ("lightweight",)}),
    Q("GLQ054", "feature", "Gym shirts that let plenty of air circulate", intent="buying", category="shirts", attributes={"feature": ("breathable",), "use_case": ("gym",)}),
    Q("GLQ055", "feature", "Boots that keep rain from soaking my feet", intent="buying", category="boots", attributes={"feature": ("waterproof",)}),
    Q("GLQ056", "feature", "A coat that holds in heat on cold mornings", intent="buying", category="jackets", attributes={"feature": ("warm",), "use_case": ("winter",)}),
    Q("GLQ057", "feature", "Sandals sturdy enough for rough daily use", intent="buying", category="sandals", attributes={"feature": ("durable",)}),
    Q("GLQ058", "feature", "Walking shoes that properly hold up my arches", intent="buying", category="walking shoes", attributes={"feature": ("supportive",), "use_case": ("walking",)}),
    Q("GLQ059", "feature", "I want socks that feel gentle against the skin", intent="buying", category="socks", attributes={"feature": ("soft",)}),
    Q("GLQ060", "feature", "Find a dress with a fashionable, current look", intent="buying", category="dresses", attributes={"feature": ("style",)}),

    # Negation and deterministic operators: excluded values never become positive.
    Q("GLQ061", "negation_operator", "Find shoes in every colour except orange", intent="buying", category="shoes", excluded={"color": ("orange",)}),
    Q("GLQ062", "negation_operator", "I'd like sneakers, but definitely not Puma", intent="buying", category="sneakers", excluded={"brand": ("Puma",)}),
    Q("GLQ063", "negation_operator", "Avoid anything made from rayon", intent="buying", excluded={"material": ("rayon",)}),
    Q("GLQ064", "negation_operator", "Black boots without leather", intent="buying", category="boots", attributes={"color": ("black",)}, excluded={"material": ("leather",)}),
    Q("GLQ065", "negation_operator", "No white sandals, thanks", intent="buying", category="sandals", excluded={"color": ("white",)}),
    Q("GLQ066", "negation_operator", "A cotton shirt from any label other than Gap", intent="buying", category="shirts", attributes={"material": ("cotton",)}, excluded={"brand": ("Gap",)}),
    Q("GLQ067", "negation_operator", "I do not want polyester trousers", intent="buying", category="pants", excluded={"material": ("polyester",)}),
    Q("GLQ068", "negation_operator", "Choose walking footwear rather than running shoes", intent="buying", category="walking shoes", attributes={"use_case": ("walking",)}, excluded={"category": ("running shoes",)}),
    Q("GLQ069", "negation_operator", "Something stylish, not comfort-focused", intent="buying", attributes={"feature": ("style",)}, excluded={"feature": ("comfort",)}),
    Q("GLQ070", "negation_operator", "Exclude Adidas; Nike running shoes are fine", intent="buying", category="running shoes", attributes={"brand": ("Nike",), "use_case": ("running",)}, excluded={"brand": ("Adidas",)}),
    Q("GLQ071", "negation_operator", "Pink or purple dresses, just avoid silk", intent="buying", category="dresses", alternatives={"color": ("pink", "purple")}, excluded={"material": ("silk",)}),
    Q("GLQ072", "negation_operator", "Boots for hiking, but nothing above $145 and no brown", intent="buying", category="boots", attributes={"use_case": ("hiking",)}, excluded={"color": ("brown",)}, maximum_price=145.0),

    # OR semantics within fields; values are alternatives, not simultaneous ANDs.
    Q("GLQ073", "alternatives", "Red or yellow shirts would both work", intent="buying", category="shirts", alternatives={"color": ("red", "yellow")}),
    Q("GLQ074", "alternatives", "Either New Balance or ASICS for my runs", intent="buying", category="running shoes", attributes={"use_case": ("running",)}, alternatives={"brand": ("New Balance", "ASICS")}),
    Q("GLQ075", "alternatives", "A jacket made of wool or nylon", intent="buying", category="jackets", alternatives={"material": ("wool", "nylon")}),
    Q("GLQ076", "alternatives", "I'm choosing between loafers and flats for work", intent="buying", attributes={"use_case": ("work",)}, alternatives={"category": ("loafers", "flats")}),
    Q("GLQ077", "alternatives", "Size 8.5 or 9 walking shoes", intent="buying", category="walking shoes", attributes={"use_case": ("walking",)}, alternatives={"size": ("8.5", "9")}),
    Q("GLQ078", "alternatives", "Grey, green, or black hoodies", intent="buying", category="hoodies", alternatives={"color": ("grey", "green", "black")}),
    Q("GLQ079", "alternatives", "Cotton, silk or rayon dresses under $125", intent="buying", category="dresses", alternatives={"material": ("cotton", "silk", "rayon")}, maximum_price=125.0),
    Q("GLQ080", "alternatives", "Nike is okay and Reebok would also be acceptable", intent="buying", alternatives={"brand": ("Nike", "Reebok")}),
    Q("GLQ081", "alternatives", "Sandals or slippers for relaxing at the resort", intent="browsing", attributes={"use_case": ("lounge",)}, alternatives={"category": ("sandals", "slippers")}),
    Q("GLQ082", "alternatives", "Small, medium, or large cotton tee", intent="buying", category="shirts", attributes={"material": ("cotton",)}, alternatives={"size": ("s", "m", "l")}),
    Q("GLQ083", "alternatives", "Comfort or cushioning matters most in the shoes", intent="buying", category="shoes", alternatives={"feature": ("comfort", "cushioned")}),
    Q("GLQ084", "alternatives", "Something for hiking or outdoor everyday use", intent="browsing", alternatives={"use_case": ("hiking", "outdoor")}),

    # Numeric and qualitative budget expressions.
    Q("GLQ085", "budget", "My top spend for sneakers is $72", intent="buying", category="sneakers", maximum_price=72.0),
    Q("GLQ086", "budget", "Keep the boots at 119 dollars or less", intent="buying", category="boots", maximum_price=119.0),
    Q("GLQ087", "budget", "A shirt with a price ceiling of USD 38.50", intent="buying", category="shirts", maximum_price=38.5),
    Q("GLQ088", "budget", "I've set aside no more than $205 for a watch", intent="buying", category="watches", maximum_price=205.0),
    Q("GLQ089", "budget", "Can we stay south of 90 bucks for sandals?", intent="buying", category="sandals", maximum_price=90.0),
    Q("GLQ090", "budget", "The most I can pay is 140 dollars for running shoes", intent="buying", category="running shoes", attributes={"use_case": ("running",)}, maximum_price=140.0),
    Q("GLQ091", "budget", "I need wallet-friendly loafers", intent="buying", category="loafers", qualitative_budget="affordable"),
    Q("GLQ092", "budget", "Show me inexpensive jackets", intent="buying", category="jackets", qualitative_budget="affordable"),
    Q("GLQ093", "budget", "About $65 for a pair of walking shoes", intent="buying", category="walking shoes", attributes={"use_case": ("walking",)}, maximum_price=65.0),
    Q("GLQ094", "budget", "Premium watches; I haven't set a spending cap", intent="unknown", category="watches"),

    # Shoe, clothing, regional and fit-related size language.
    Q("GLQ095", "size", "Women's nine-and-a-half black sneakers", intent="buying", category="sneakers", attributes={"size": ("9.5",), "color": ("black",)}),
    Q("GLQ096", "size", "A size XXL cotton hoodie", intent="buying", category="hoodies", attributes={"size": ("xxl",), "material": ("cotton",)}),
    Q("GLQ097", "size", "European 41 leather boots", intent="buying", category="boots", attributes={"size": ("eu 41",), "material": ("leather",)}),
    Q("GLQ098", "size", "US women's 7 sandals in white", intent="buying", category="sandals", attributes={"size": ("us 7",), "color": ("white",)}),
    Q("GLQ099", "size", "Extra-small red dress", intent="buying", category="dresses", attributes={"size": ("xs",), "color": ("red",)}),
    Q("GLQ100", "size", "Large or extra-large wool jacket", intent="buying", category="jackets", attributes={"material": ("wool",)}, alternatives={"size": ("l", "xl")}),
    Q("GLQ101", "size", "One-size-fits-all blue hat", intent="buying", category="hats", attributes={"size": ("one size",), "color": ("blue",)}),
    Q("GLQ102", "size", "Men's size 12 in a wide fit", intent="buying", attributes={"size": ("12 wide",)}),
    Q("GLQ103", "size", "Narrow-fitting walking shoes in an 8", intent="buying", category="walking shoes", attributes={"size": ("8 narrow",), "use_case": ("walking",)}),
    Q("GLQ104", "size", "Medium petite black trousers", intent="buying", category="pants", attributes={"size": ("m petite",), "color": ("black",)}),

    # Adversarial words that resemble a supported entity in the wrong context.
    Q("GLQ105", "adversarial_collision", "White shoes for the Black Friday sale", intent="buying", category="shoes", attributes={"color": ("white",)}),
    Q("GLQ106", "adversarial_collision", "An orange-blossom print dress on a white background", intent="buying", category="dresses", attributes={"color": ("white",)}),
    Q("GLQ107", "adversarial_collision", "Cotton On polyester shirts", intent="buying", category="shirts", attributes={"brand": ("Cotton On",), "material": ("polyester",)}),
    Q("GLQ108", "adversarial_collision", "Coach me through finding a Nike watch", intent="buying", category="watches", attributes={"brand": ("Nike",)}),
    Q("GLQ109", "adversarial_collision", "I'm running late, but I need footwear for walking", intent="buying", category="walking shoes", attributes={"use_case": ("walking",)}),
    Q("GLQ110", "adversarial_collision", "Nike-inspired styling, but the brand must be Adidas", intent="buying", attributes={"brand": ("Adidas",)}),
    Q("GLQ111", "adversarial_collision", "A silk-feel polyester blouse", intent="buying", category="shirts", attributes={"material": ("polyester",)}),
    Q("GLQ112", "adversarial_collision", "Leather-look vegan boots", intent="buying", category="boots"),
    Q("GLQ113", "adversarial_collision", "Brown boots for blue-collar work", intent="buying", category="boots", attributes={"color": ("brown",), "use_case": ("work",)}),
    Q("GLQ114", "adversarial_collision", "I'm comfortable with a price below $60 for socks", intent="buying", category="socks", maximum_price=60.0),
    Q("GLQ115", "adversarial_collision", "While walking through ideas, show me shoes for running", intent="buying", category="running shoes", attributes={"use_case": ("running",)}),
    Q("GLQ116", "adversarial_collision", "A black dress for a red-carpet event", intent="buying", category="dresses", attributes={"color": ("black",), "use_case": ("formal",)}),

    # Unsupported concepts: retain supported fields but invent no unsupported slot.
    Q("GLQ117", "unsupported", "I need carbon-neutral sneakers", intent="buying", category="sneakers"),
    Q("GLQ118", "unsupported", "Black boots that can arrive tomorrow", intent="buying", category="boots", attributes={"color": ("black",)}),
    Q("GLQ119", "unsupported", "Show only shirts rated at least 4.8 stars", intent="buying", category="shirts"),
    Q("GLQ120", "unsupported", "A watch with a lifetime warranty under $300", intent="buying", category="watches", maximum_price=300.0),
    Q("GLQ121", "unsupported", "Cruelty-free red pumps", intent="buying", category="pumps", attributes={"color": ("red",)}),
    Q("GLQ122", "unsupported", "Locally manufactured blue jeans", intent="buying", category="jeans", attributes={"color": ("blue",)}),
    Q("GLQ123", "unsupported", "A shirt with custom name embroidery", intent="buying", category="shirts"),
    Q("GLQ124", "unsupported", "Sneakers currently in stock at the downtown branch", intent="buying", category="sneakers"),
    Q("GLQ125", "unsupported", "A vegan-certified belt in brown", intent="buying", category="belts", attributes={"color": ("brown",)}),
    Q("GLQ126", "unsupported", "Socks sold through a monthly subscription", intent="buying", category="socks"),

    # Intent boundaries: constraints imply buying; broad ideation implies browsing.
    Q("GLQ127", "intent_boundary", "Footwear, maybe?", intent="unknown"),
    Q("GLQ128", "intent_boundary", "I need a jacket", intent="unknown", category="jackets"),
    Q("GLQ129", "intent_boundary", "Let me look around without choosing yet", intent="browsing"),
    Q("GLQ130", "intent_boundary", "Recommend a few things and surprise me", intent="browsing"),
    Q("GLQ131", "intent_boundary", "I must buy black boots today", intent="buying", category="boots", attributes={"color": ("black",)}),
    Q("GLQ132", "intent_boundary", "Do you carry anything by Reebok?", intent="buying", attributes={"brand": ("Reebok",)}),
    Q("GLQ133", "intent_boundary", "Maybe browse some purple options", intent="browsing", attributes={"color": ("purple",)}),
    Q("GLQ134", "intent_boundary", "What would you suggest for winter?", intent="browsing", attributes={"use_case": ("winter",)}),
    Q("GLQ135", "intent_boundary", "A pair of sandals", intent="unknown", category="sandals"),
    Q("GLQ136", "intent_boundary", "Find the cheapest Nike shoes you have", intent="buying", category="shoes", attributes={"brand": ("Nike",)}, qualitative_budget="affordable"),

    # Preference removal utterances contain no replacement hard value.
    Q("GLQ137", "preference_removal", "Colour is no longer important to me", remove_attributes=frozenset({"color"})),
    Q("GLQ138", "preference_removal", "Any brand is acceptable now", remove_attributes=frozenset({"brand"})),
    Q("GLQ139", "preference_removal", "Forget the material requirement", remove_attributes=frozenset({"material"})),
    Q("GLQ140", "preference_removal", "I have no size preference anymore", remove_attributes=frozenset({"size"})),
    Q("GLQ141", "preference_removal", "Remove the spending limit", remove_attributes=frozenset({"budget"})),
    Q("GLQ142", "preference_removal", "Comfort doesn't matter after all", remove_attributes=frozenset({"feature"})),
    Q("GLQ143", "preference_removal", "The use case is irrelevant now", remove_attributes=frozenset({"use_case"})),
    Q("GLQ144", "preference_removal", "I don't care which colour it is", remove_attributes=frozenset({"color"})),
    Q("GLQ145", "preference_removal", "There is no maximum budget anymore", remove_attributes=frozenset({"budget"})),
    Q("GLQ146", "preference_removal", "Drop my earlier brand preference", remove_attributes=frozenset({"brand"})),
)


@dataclass(frozen=True)
class ConversationCase:
    case_id: str
    split: str
    group: str
    messages: tuple[str, ...]
    expected_category: str | None = None
    expected_attributes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    expected_removed: frozenset[str] = frozenset()
    expected_maximum_price: float | None = None
    expected_intent: str = "intent_override"


def C(case_id: str, group: str, messages: tuple[str, ...], **expected: object) -> ConversationCase:
    return ConversationCase(case_id, "unseen", group, messages, **expected)


CONVERSATIONS = (
    C("GLC001", "change_category", ("I need running shoes for mornings", "Actually, sandals instead"), expected_category="sandals"),
    C("GLC002", "replace_color", ("Show me black shirts", "On second thought, make them green"), expected_category="shirts", expected_attributes={"color": ("green",)}),
    C("GLC003", "remove_color", ("I'd like blue sneakers", "Colour isn't important anymore"), expected_category="sneakers", expected_removed=frozenset({"color"})),
    C("GLC004", "remove_brand", ("Find Nike walking shoes", "Any brand will do now"), expected_category="walking shoes", expected_removed=frozenset({"brand"})),
    C("GLC005", "replace_material", ("I want a cotton jacket", "I'd prefer wool instead"), expected_category="jackets", expected_attributes={"material": ("wool",)}),
    C("GLC006", "lower_budget", ("My ceiling is $175 for boots", "Reduce that maximum to $125"), expected_category="boots", expected_maximum_price=125.0),
    C("GLC007", "raise_budget", ("Keep sneakers below $70", "I can stretch the limit to $95"), expected_category="sneakers", expected_maximum_price=95.0),
    C("GLC008", "remove_budget", ("Find a watch under $180", "Forget the price cap"), expected_category="watches", expected_removed=frozenset({"budget"}), expected_maximum_price=None),
    C("GLC009", "replace_with_alternatives", ("White sandals, please", "Actually red or yellow would be better"), expected_category="sandals", expected_attributes={"color": ("red", "yellow")}),
    C("GLC010", "change_category", ("I'm looking for dresses", "Switch that to hoodies"), expected_category="hoodies"),
    C("GLC011", "switch_to_browsing", ("Black Adidas shoes under $110", "Never mind the specifics; I'm just browsing now"), expected_category="shoes", expected_intent="browsing"),
    C("GLC012", "switch_to_buying", ("Show me ideas for winter", "I actually need a black wool jacket below $100"), expected_category="jackets", expected_attributes={"color": ("black",), "material": ("wool",)}, expected_maximum_price=100.0, expected_intent="buying"),
    C("GLC013", "replace_size", ("Size 9 running shoes", "Make that a 10 instead"), expected_category="running shoes", expected_attributes={"size": ("10",)}),
    C("GLC014", "replace_brand", ("Puma sneakers would work", "Use Reebok instead"), expected_category="sneakers", expected_attributes={"brand": ("Reebok",)}),
    C("GLC015", "change_use_case", ("I need shoes for running", "Actually these are for long walks"), expected_category="walking shoes", expected_attributes={"use_case": ("walking",)}),
    C("GLC016", "remove_feature", ("Comfortable black shoes", "Comfort isn't a priority any longer"), expected_category="shoes", expected_attributes={"color": ("black",)}, expected_removed=frozenset({"feature"})),
)


def _stable_json_value(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation."""

    if is_dataclass(value):
        return _stable_json_value(asdict(value))
    if isinstance(value, dict):
        return {key: _stable_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, frozenset):
        return sorted(_stable_json_value(item) for item in value)
    if isinstance(value, (tuple, list)):
        return [_stable_json_value(item) for item in value]
    return value


def corpus_sha256() -> str:
    """Return the SHA-256 digest for all frozen inputs and expected labels."""

    payload = {
        "version": CORPUS_VERSION,
        "queries": CASES,
        "conversations": CONVERSATIONS,
    }
    encoded = json.dumps(
        _stable_json_value(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# Filled from ``corpus_sha256()`` using structure-only validation before any
# parser or GLiNER evaluation.  A mismatch means the frozen labels changed.
FROZEN_CORPUS_SHA256 = "730152c94d8246d43a451f2da216b495a17ac1dbcb1260aadfa0182377eabf03"


FROZEN_MANIFEST = {
    "version": CORPUS_VERSION,
    "frozen_on": FROZEN_ON,
    "query_count": len(CASES),
    "conversation_count": len(CONVERSATIONS),
    "sha256": FROZEN_CORPUS_SHA256,
}


def manifest_is_valid() -> bool:
    return (
        FROZEN_BEFORE_GLINER_EVALUATION
        and len(CASES) >= 120
        and len(CONVERSATIONS) >= 12
        and corpus_sha256() == FROZEN_CORPUS_SHA256
    )
