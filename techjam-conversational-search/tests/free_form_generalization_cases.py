"""Fixed development and held-out corpus for free-form generalization checks.

The split is part of the data. Do not move held-out examples into development
after observing results; add new development cases instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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


def Q(case_id: str, split: str, group: str, message: str, **expected: object) -> QueryCase:
    return QueryCase(case_id, split, group, message, **expected)


CASES = (
    # Direct but independently worded requests.
    Q("DIR01", "dev", "direct", "I want red leather shoes", intent="buying", category="shoes", attributes={"color": ("red",), "material": ("leather",)}),
    Q("DIR02", "heldout", "direct", "Find me a white cotton shirt", intent="buying", category="shirts", attributes={"color": ("white",), "material": ("cotton",)}),
    Q("DIR03", "dev", "direct", "Blue Nike sneakers please", intent="buying", category="sneakers", attributes={"color": ("blue",), "brand": ("Nike",)}),
    Q("DIR04", "heldout", "direct", "I need brown sandals", intent="buying", category="sandals", attributes={"color": ("brown",)}),
    Q("DIR05", "dev", "direct", "Show me Adidas running shoes", intent="buying", category="running shoes", attributes={"brand": ("Adidas",), "use_case": ("running",)}),
    Q("DIR06", "heldout", "direct", "A black wool jacket would be good", intent="buying", category="jackets", attributes={"color": ("black",), "material": ("wool",)}),
    Q("DIR07", "dev", "direct", "Looking for green polyester pants", intent="buying", category="pants", attributes={"color": ("green",), "material": ("polyester",)}),
    Q("DIR08", "heldout", "direct", "Need purple silk dresses", intent="buying", category="dresses", attributes={"color": ("purple",), "material": ("silk",)}),
    Q("DIR09", "dev", "direct", "Black Puma socks", intent="buying", category="socks", attributes={"color": ("black",), "brand": ("Puma",)}),
    Q("DIR10", "heldout", "direct", "Orange rayon skirt", intent="buying", category="skirts", attributes={"color": ("orange",), "material": ("rayon",)}),

    # Natural paraphrases and less regular syntax.
    Q("NAT01", "dev", "natural", "Need something comfy for long walks", intent="buying", category="walking shoes", attributes={"feature": ("comfort",), "use_case": ("walking",)}),
    Q("NAT02", "heldout", "natural", "Looking for some runners in black", intent="buying", category="running shoes", attributes={"color": ("black",), "use_case": ("running",)}),
    Q("NAT03", "dev", "natural", "Got anything from Nike below a hundred bucks?", intent="buying", attributes={"brand": ("Nike",)}, maximum_price=100.0),
    Q("NAT04", "heldout", "natural", "Could do with a pair of trainers for jogging", intent="buying", category="running shoes", attributes={"use_case": ("running",)}),
    Q("NAT05", "dev", "natural", "After a soft cotton top in blue", intent="buying", category="shirts", attributes={"color": ("blue",), "material": ("cotton",)}),
    Q("NAT06", "heldout", "natural", "Something sturdy for hiking would help", intent="buying", attributes={"feature": ("durable",), "use_case": ("hiking",)}),
    Q("NAT07", "dev", "natural", "Any comfy sandals for walking around town?", intent="buying", category="sandals", attributes={"feature": ("comfort",), "use_case": ("walking",)}),
    Q("NAT08", "heldout", "natural", "I could use warm boots for winter", intent="buying", category="boots", attributes={"feature": ("warm",), "use_case": ("winter",)}),
    Q("NAT09", "dev", "natural", "Help me pick an affordable pair of sneakers", intent="buying", category="sneakers", qualitative_budget="affordable"),
    Q("NAT10", "heldout", "natural", "I'm after a formal black dress", intent="buying", category="dresses", attributes={"color": ("black",), "use_case": ("formal",)}),

    # Constraint-first and unusual sentence structures.
    Q("STR01", "dev", "structure", "Under $80, black, preferably Adidas", intent="buying", attributes={"color": ("black",), "brand": ("Adidas",)}, maximum_price=80.0),
    Q("STR02", "heldout", "structure", "For jogging I need something cushioned", intent="buying", category="running shoes", attributes={"feature": ("cushioned",), "use_case": ("running",)}),
    Q("STR03", "dev", "structure", "Cotton is essential; blue would be ideal; make it a shirt", intent="buying", category="shirts", attributes={"color": ("blue",), "material": ("cotton",)}),
    Q("STR04", "heldout", "structure", "My limit is 60 dollars and I want sandals", intent="buying", category="sandals", maximum_price=60.0),
    Q("STR05", "dev", "structure", "Nike, black, running - those are the priorities", intent="buying", category="running shoes", attributes={"brand": ("Nike",), "color": ("black",), "use_case": ("running",)}),
    Q("STR06", "heldout", "structure", "Size 9 first, then comfort, for walking", intent="buying", category="walking shoes", attributes={"size": ("9",), "feature": ("comfort",), "use_case": ("walking",)}),
    Q("STR07", "dev", "structure", "If it is leather and under 150 dollars, show me boots", intent="buying", category="boots", attributes={"material": ("leather",)}, maximum_price=150.0),
    Q("STR08", "heldout", "structure", "For the beach: sandals, preferably white", intent="buying", category="sandals", attributes={"color": ("white",), "use_case": ("beach",)}),
    Q("STR09", "dev", "structure", "What I care about is wool; the item should be a jacket", intent="buying", category="jackets", attributes={"material": ("wool",)}),
    Q("STR10", "heldout", "structure", "Budget 45, colour red, category shirts", intent="buying", category="shirts", attributes={"color": ("red",)}, maximum_price=45.0),

    # Category language and synonyms.
    Q("CAT01", "dev", "category_synonym", "Show me black runners", intent="buying", category="running shoes", attributes={"color": ("black",), "use_case": ("running",)}),
    Q("CAT02", "heldout", "category_synonym", "I need trainers under $90", intent="buying", category="sneakers", maximum_price=90.0),
    Q("CAT03", "dev", "category_synonym", "Looking for tennis shoes in white", intent="buying", category="sneakers", attributes={"color": ("white",)}),
    Q("CAT04", "heldout", "category_synonym", "Any flip flops for the beach?", intent="buying", category="sandals", attributes={"use_case": ("beach",)}),
    Q("CAT05", "dev", "category_synonym", "Need a tee made from cotton", intent="buying", category="shirts", attributes={"material": ("cotton",)}),
    Q("CAT06", "heldout", "category_synonym", "Find me some trousers in black", intent="buying", category="pants", attributes={"color": ("black",)}),
    Q("CAT07", "dev", "category_synonym", "I want a pair of shades", intent="unknown", category="sunglasses"),
    Q("CAT08", "heldout", "category_synonym", "A pullover hoodie in grey", intent="buying", category="hoodies", attributes={"color": ("grey",)}),
    Q("CAT09", "dev", "category_synonym", "Show me ankle boots", intent="unknown", category="boots"),
    Q("CAT10", "heldout", "category_synonym", "Need walking sneakers", intent="buying", category="walking shoes", attributes={"use_case": ("walking",)}),

    # Feature and use-case language.
    Q("USE01", "dev", "feature_use_case", "Comfortable shoes for walking all day", intent="buying", category="walking shoes", attributes={"feature": ("comfort",), "use_case": ("walking",)}),
    Q("USE02", "heldout", "feature_use_case", "Cushioning matters for my daily runs", intent="buying", category="running shoes", attributes={"feature": ("cushioned",), "use_case": ("running",)}),
    Q("USE03", "dev", "feature_use_case", "Something waterproof for hiking", intent="buying", attributes={"feature": ("waterproof",), "use_case": ("hiking",)}),
    Q("USE04", "heldout", "feature_use_case", "Warm clothes for a winter trip", intent="browsing", attributes={"feature": ("warm",), "use_case": ("winter",)}),
    Q("USE05", "dev", "feature_use_case", "A stylish dress for a formal dinner", intent="buying", category="dresses", attributes={"feature": ("style",), "use_case": ("formal",)}),
    Q("USE06", "heldout", "feature_use_case", "Breathable gear for the gym", intent="browsing", attributes={"feature": ("breathable",), "use_case": ("gym",)}),
    Q("USE07", "dev", "feature_use_case", "Durable sandals for outdoor use", intent="buying", category="sandals", attributes={"feature": ("durable",), "use_case": ("outdoor",)}),
    Q("USE08", "heldout", "feature_use_case", "Soft socks for lounging at home", intent="buying", category="socks", attributes={"feature": ("soft",), "use_case": ("lounge",)}),
    Q("USE09", "dev", "feature_use_case", "Supportive shoes for standing at work", intent="buying", category="shoes", attributes={"feature": ("supportive",), "use_case": ("work",)}),
    Q("USE10", "heldout", "feature_use_case", "Lightweight trainers for travel", intent="buying", category="sneakers", attributes={"feature": ("lightweight",), "use_case": ("travel",)}),

    # Negation must not become a positive constraint.
    Q("NEG01", "dev", "negation", "Anything except black shoes", intent="buying", category="shoes", excluded={"color": ("black",)}),
    Q("NEG02", "heldout", "negation", "Running shoes, but not Nike", intent="buying", category="running shoes", attributes={"use_case": ("running",)}, excluded={"brand": ("Nike",)}),
    Q("NEG03", "dev", "negation", "Leather boots but not brown", intent="buying", category="boots", attributes={"material": ("leather",)}, excluded={"color": ("brown",)}),
    Q("NEG04", "heldout", "negation", "No polyester shirts please", intent="buying", category="shirts", excluded={"material": ("polyester",)}),
    Q("NEG05", "dev", "negation", "Blue sneakers without Adidas", intent="buying", category="sneakers", attributes={"color": ("blue",)}, excluded={"brand": ("Adidas",)}),
    Q("NEG06", "heldout", "negation", "Avoid red dresses", intent="buying", category="dresses", excluded={"color": ("red",)}),
    Q("NEG07", "dev", "negation", "Sandals in any colour other than white", intent="buying", category="sandals", excluded={"color": ("white",)}),
    Q("NEG08", "heldout", "negation", "I do not want wool", intent="buying", excluded={"material": ("wool",)}),
    Q("NEG09", "dev", "negation", "Exclude Puma and show running shoes", intent="buying", category="running shoes", attributes={"use_case": ("running",)}, excluded={"brand": ("Puma",)}),
    Q("NEG10", "heldout", "negation", "Black is fine, just no leather", intent="buying", attributes={"color": ("black",)}, excluded={"material": ("leather",)}),

    # Alternatives are OR within a field, never AND.
    Q("ALT01", "dev", "alternatives", "Black or blue sneakers", intent="buying", category="sneakers", alternatives={"color": ("black", "blue")}),
    Q("ALT02", "heldout", "alternatives", "Nike or Adidas running shoes", intent="buying", category="running shoes", attributes={"use_case": ("running",)}, alternatives={"brand": ("Nike", "Adidas")}),
    Q("ALT03", "dev", "alternatives", "Cotton or wool shirts", intent="buying", category="shirts", alternatives={"material": ("cotton", "wool")}),
    Q("ALT04", "heldout", "alternatives", "Sandals in red or white", intent="buying", category="sandals", alternatives={"color": ("red", "white")}),
    Q("ALT05", "dev", "alternatives", "Puma is okay, but Reebok works too", intent="buying", alternatives={"brand": ("Puma", "Reebok")}),
    Q("ALT06", "heldout", "alternatives", "Either leather boots or leather shoes", intent="buying", alternatives={"category": ("boots", "shoes")}, attributes={"material": ("leather",)}),
    Q("ALT07", "dev", "alternatives", "Grey, black, or blue trousers", intent="buying", category="pants", alternatives={"color": ("grey", "black", "blue")}),
    Q("ALT08", "heldout", "alternatives", "Nylon or polyester jacket under $70", intent="buying", category="jackets", alternatives={"material": ("nylon", "polyester")}, maximum_price=70.0),
    Q("ALT09", "dev", "alternatives", "Sneakers or sandals for the trip", intent="browsing", alternatives={"category": ("sneakers", "sandals")}, attributes={"use_case": ("travel",)}),
    Q("ALT10", "heldout", "alternatives", "Size 9 or 10 running shoes", intent="buying", category="running shoes", attributes={"use_case": ("running",)}, alternatives={"size": ("9", "10")}),

    # Vague exploratory intent should not invent hard constraints.
    Q("BRW01", "dev", "browsing", "Something nice for a beach holiday", intent="browsing", attributes={"use_case": ("beach",)}),
    Q("BRW02", "heldout", "browsing", "What should I wear for winter?", intent="browsing", attributes={"use_case": ("winter",)}),
    Q("BRW03", "dev", "browsing", "Show me something suitable for hiking", intent="browsing", attributes={"use_case": ("hiking",)}),
    Q("BRW04", "heldout", "browsing", "Give me ideas for a formal event", intent="browsing", attributes={"use_case": ("formal",)}),
    Q("BRW05", "dev", "browsing", "I'm just browsing summer styles", intent="browsing", attributes={"use_case": ("summer",)}),
    Q("BRW06", "heldout", "browsing", "What looks good for a weekend away?", intent="browsing", attributes={"use_case": ("travel",)}),
    Q("BRW07", "dev", "browsing", "Surprise me with gym clothing", intent="browsing", attributes={"use_case": ("gym",)}),
    Q("BRW08", "heldout", "browsing", "Any outfit inspiration for the office?", intent="browsing", attributes={"use_case": ("work",)}),
    Q("BRW09", "dev", "browsing", "I'm exploring comfortable everyday options", intent="browsing", attributes={"feature": ("comfort",)}),
    Q("BRW10", "heldout", "browsing", "Show me a few fun holiday ideas", intent="browsing", attributes={"use_case": ("holiday",)}),

    # Hard buying with several simultaneous requirements.
    Q("BUY01", "dev", "hard_buying", "Black Nike running shoes under $120 size 10", intent="buying", category="running shoes", attributes={"color": ("black",), "brand": ("Nike",), "use_case": ("running",), "size": ("10",)}, maximum_price=120.0),
    Q("BUY02", "heldout", "hard_buying", "White Adidas sneakers below $85 in size 8", intent="buying", category="sneakers", attributes={"color": ("white",), "brand": ("Adidas",), "size": ("8",)}, maximum_price=85.0),
    Q("BUY03", "dev", "hard_buying", "Blue cotton shirt size M no more than $40", intent="buying", category="shirts", attributes={"color": ("blue",), "material": ("cotton",), "size": ("m",)}, maximum_price=40.0),
    Q("BUY04", "heldout", "hard_buying", "Brown leather boots under 150 dollars size 9", intent="buying", category="boots", attributes={"color": ("brown",), "material": ("leather",), "size": ("9",)}, maximum_price=150.0),
    Q("BUY05", "dev", "hard_buying", "Comfortable black walking shoes from Skechers under $100", intent="buying", category="walking shoes", attributes={"feature": ("comfort",), "color": ("black",), "brand": ("Skechers",), "use_case": ("walking",)}, maximum_price=100.0),
    Q("BUY06", "heldout", "hard_buying", "Red silk formal dress below $200", intent="buying", category="dresses", attributes={"color": ("red",), "material": ("silk",), "use_case": ("formal",)}, maximum_price=200.0),
    Q("BUY07", "dev", "hard_buying", "Waterproof hiking boots in black, size 11, below $140", intent="buying", category="boots", attributes={"feature": ("waterproof",), "use_case": ("hiking",), "color": ("black",), "size": ("11",)}, maximum_price=140.0),
    Q("BUY08", "heldout", "hard_buying", "Warm wool jacket, grey, under $90", intent="buying", category="jackets", attributes={"feature": ("warm",), "material": ("wool",), "color": ("grey",)}, maximum_price=90.0),
    Q("BUY09", "dev", "hard_buying", "Puma gym shoes in white or black below $75", intent="buying", category="shoes", attributes={"brand": ("Puma",), "use_case": ("gym",)}, alternatives={"color": ("white", "black")}, maximum_price=75.0),
    Q("BUY10", "heldout", "hard_buying", "Leather sandals, not brown, maximum $60", intent="buying", category="sandals", attributes={"material": ("leather",)}, excluded={"color": ("brown",)}, maximum_price=60.0),

    # Budget syntax and qualitative budget safeguards.
    Q("BUD01", "dev", "budget", "Sneakers under $50", intent="buying", category="sneakers", maximum_price=50.0),
    Q("BUD02", "heldout", "budget", "My ceiling is 75 dollars for boots", intent="buying", category="boots", maximum_price=75.0),
    Q("BUD03", "dev", "budget", "Keep the shirt below forty bucks", intent="buying", category="shirts", maximum_price=40.0),
    Q("BUD04", "heldout", "budget", "I can spend at most USD 130 on a watch", intent="buying", category="watches", maximum_price=130.0),
    Q("BUD05", "dev", "budget", "Budget-friendly sandals", intent="buying", category="sandals", qualitative_budget="affordable"),
    Q("BUD06", "heldout", "budget", "Nothing expensive, just some shoes", intent="buying", category="shoes", qualitative_budget="affordable"),
    Q("BUD07", "dev", "budget", "Around $100 for running shoes", intent="buying", category="running shoes", attributes={"use_case": ("running",)}, maximum_price=100.0),
    Q("BUD08", "heldout", "budget", "No more than 89.99 for Adidas sneakers", intent="buying", category="sneakers", attributes={"brand": ("Adidas",)}, maximum_price=89.99),
    Q("BUD09", "dev", "budget", "Cheap shoes", intent="buying", category="shoes", qualitative_budget="affordable"),
    Q("BUD10", "heldout", "budget", "Premium leather boots", intent="buying", category="boots", attributes={"material": ("leather",)}),

    # Sizes and fit expressions.
    Q("SIZ01", "dev", "size", "Size 10 running shoes", intent="buying", category="running shoes", attributes={"size": ("10",), "use_case": ("running",)}),
    Q("SIZ02", "heldout", "size", "Women's 8.5 walking shoes", intent="buying", category="walking shoes", attributes={"size": ("8.5",), "use_case": ("walking",)}),
    Q("SIZ03", "dev", "size", "Medium cotton shirt", intent="buying", category="shirts", attributes={"size": ("m",), "material": ("cotton",)}),
    Q("SIZ04", "heldout", "size", "An XL wool jacket", intent="buying", category="jackets", attributes={"size": ("xl",), "material": ("wool",)}),
    Q("SIZ05", "dev", "size", "Wide fit black shoes", intent="buying", category="shoes", attributes={"size": ("wide",), "color": ("black",)}),
    Q("SIZ06", "heldout", "size", "Narrow size 7 sandals", intent="buying", category="sandals", attributes={"size": ("7 narrow",)}),
    Q("SIZ07", "dev", "size", "EU 42 Adidas trainers", intent="buying", category="sneakers", attributes={"size": ("eu 42",), "brand": ("Adidas",)}),
    Q("SIZ08", "heldout", "size", "US men's 11 Nike runners", intent="buying", category="running shoes", attributes={"size": ("us 11",), "brand": ("Nike",), "use_case": ("running",)}),
    Q("SIZ09", "dev", "size", "One size hat in blue", intent="buying", category="hats", attributes={"size": ("one size",), "color": ("blue",)}),
    Q("SIZ10", "heldout", "size", "Small or medium shirt", intent="buying", category="shirts", alternatives={"size": ("s", "m")}),

    # Intent and clarification boundaries.
    Q("INT01", "dev", "intent", "I need shoes", intent="unknown", category="shoes"),
    Q("INT02", "heldout", "intent", "Show me shoes", intent="unknown", category="shoes"),
    Q("INT03", "dev", "intent", "Just browsing", intent="browsing"),
    Q("INT04", "heldout", "intent", "I have no idea what I want", intent="unknown"),
    Q("INT05", "dev", "intent", "Find me Nike products", intent="buying", attributes={"brand": ("Nike",)}),
    Q("INT06", "heldout", "intent", "Maybe something blue", intent="browsing", attributes={"color": ("blue",)}),
    Q("INT07", "dev", "intent", "I definitely need black boots today", intent="buying", category="boots", attributes={"color": ("black",)}),
    Q("INT08", "heldout", "intent", "Could you suggest a few options?", intent="browsing"),
    Q("INT09", "dev", "intent", "Surprise me", intent="browsing"),
    Q("INT10", "heldout", "intent", "A shirt", intent="unknown", category="shirts"),

    # Adversarial related-but-different concepts.
    Q("ADV01", "dev", "adversarial", "Walking shoes, not running shoes", intent="buying", category="walking shoes", attributes={"use_case": ("walking",)}, excluded={"category": ("running shoes",)}),
    Q("ADV02", "heldout", "adversarial", "Navy sneakers", intent="buying", category="sneakers"),
    Q("ADV03", "dev", "adversarial", "Polyester shirt, definitely not cotton", intent="buying", category="shirts", attributes={"material": ("polyester",)}, excluded={"material": ("cotton",)}),
    Q("ADV04", "heldout", "adversarial", "Stylish shoes rather than comfortable ones", intent="buying", category="shoes", attributes={"feature": ("style",)}, excluded={"feature": ("comfort",)}),
    Q("ADV05", "dev", "adversarial", "Sandals, not sneakers", intent="buying", category="sandals", excluded={"category": ("sneakers",)}),
    Q("ADV06", "heldout", "adversarial", "Adidas only, no Nike", intent="buying", attributes={"brand": ("Adidas",)}, excluded={"brand": ("Nike",)}),
    Q("ADV07", "dev", "adversarial", "Dark blue shoes", intent="buying", category="shoes", attributes={"color": ("blue",)}),
    Q("ADV08", "heldout", "adversarial", "A cotton-like synthetic shirt", intent="buying", category="shirts"),
    Q("ADV09", "dev", "adversarial", "Running-style casual sneakers, not for running", intent="buying", category="sneakers", excluded={"use_case": ("running",)}),
    Q("ADV10", "heldout", "adversarial", "Comfort is irrelevant; I care about style", intent="buying", attributes={"feature": ("style",)}, excluded={"feature": ("comfort",)}),

    # Unsupported or ambiguous concepts must not create a supported hard value.
    Q("UNS01", "dev", "unsupported", "Burgundy shoes", intent="unknown", category="shoes"),
    Q("UNS02", "heldout", "unsupported", "Vegan leather boots", intent="buying", category="boots", attributes={"feature": ("vegan leather",)}),
    Q("UNS03", "dev", "unsupported", "Carbon-neutral sneakers", intent="buying", category="sneakers", attributes={"feature": ("carbon-neutral",)}),
    Q("UNS04", "heldout", "unsupported", "Shoes with excellent arch support", intent="buying", category="shoes", attributes={"feature": ("arch support",)}),
    Q("UNS05", "dev", "unsupported", "A dress that makes me look taller", intent="buying", category="dresses", attributes={"feature": ("look taller",)}),
    Q("UNS06", "heldout", "unsupported", "Ethically made cotton shirts", intent="buying", category="shirts", attributes={"material": ("cotton",), "feature": ("ethical",)}),
    Q("UNS07", "dev", "unsupported", "Machine-washable wool jacket", intent="buying", category="jackets", attributes={"material": ("wool",), "feature": ("machine washable",)}),
    Q("UNS08", "heldout", "unsupported", "Sneakers available for delivery tomorrow", intent="unknown", category="sneakers"),
    Q("UNS09", "dev", "unsupported", "Shoes rated at least four stars", intent="unknown", category="shoes"),
    Q("UNS10", "heldout", "unsupported", "A locally manufactured black shirt", intent="buying", category="shirts", attributes={"color": ("black",), "feature": ("locally manufactured",)}),

    # Broader category coverage.
    Q("VAR01", "dev", "category_variety", "Silver earrings under $40", intent="buying", category="earrings", maximum_price=40.0),
    Q("VAR02", "heldout", "category_variety", "A black belt from Calvin Klein", intent="buying", category="belts", attributes={"color": ("black",), "brand": ("Calvin Klein",)}),
    Q("VAR03", "dev", "category_variety", "Gold necklace for a formal event", intent="buying", category="necklaces", attributes={"use_case": ("formal",)}),
    Q("VAR04", "heldout", "category_variety", "Warm slippers below $50", intent="buying", category="slippers", attributes={"feature": ("warm",)}, maximum_price=50.0),
    Q("VAR05", "dev", "category_variety", "Brown loafers for work", intent="buying", category="loafers", attributes={"color": ("brown",), "use_case": ("work",)}),
    Q("VAR06", "heldout", "category_variety", "Red pumps for a formal dinner", intent="buying", category="pumps", attributes={"color": ("red",), "use_case": ("formal",)}),
    Q("VAR07", "dev", "category_variety", "Comfortable black flats", intent="buying", category="flats", attributes={"feature": ("comfort",), "color": ("black",)}),
    Q("VAR08", "heldout", "category_variety", "A blue hat for the beach", intent="buying", category="hats", attributes={"color": ("blue",), "use_case": ("beach",)}),
    Q("VAR09", "dev", "category_variety", "Polarized sunglasses for travel", intent="buying", category="sunglasses", attributes={"feature": ("polarized",), "use_case": ("travel",)}),
    Q("VAR10", "heldout", "category_variety", "A waterproof watch under $200", intent="buying", category="watches", attributes={"feature": ("waterproof",)}, maximum_price=200.0),

    # Exact field collision and brand/material/color checks.
    Q("FLD01", "dev", "field_collision", "Cotton On shirts", intent="buying", category="shirts", attributes={"brand": ("Cotton On",)}),
    Q("FLD02", "heldout", "field_collision", "Orange shoes from Nike", intent="buying", category="shoes", attributes={"color": ("orange",), "brand": ("Nike",)}),
    Q("FLD03", "dev", "field_collision", "Coach leather bag", intent="buying", attributes={"brand": ("Coach",), "material": ("leather",)}),
    Q("FLD04", "heldout", "field_collision", "Guess black watch", intent="buying", category="watches", attributes={"brand": ("Guess",), "color": ("black",)}),
    Q("FLD05", "dev", "field_collision", "Gap cotton shirt", intent="buying", category="shirts", attributes={"brand": ("Gap",), "material": ("cotton",)}),
    Q("FLD06", "heldout", "field_collision", "Under Armour gym shoes", intent="buying", category="shoes", attributes={"brand": ("Under Armour",), "use_case": ("gym",)}),
    Q("FLD07", "dev", "field_collision", "Columbia outdoor jacket", intent="buying", category="jackets", attributes={"brand": ("Columbia",), "use_case": ("outdoor",)}),
    Q("FLD08", "heldout", "field_collision", "Clarks brown walking shoes", intent="buying", category="walking shoes", attributes={"brand": ("Clarks",), "color": ("brown",), "use_case": ("walking",)}),
    Q("FLD09", "dev", "field_collision", "ASICS blue runners", intent="buying", category="running shoes", attributes={"brand": ("ASICS",), "color": ("blue",), "use_case": ("running",)}),
    Q("FLD10", "heldout", "field_collision", "Skechers comfortable trainers", intent="buying", category="sneakers", attributes={"brand": ("Skechers",), "feature": ("comfort",)}),
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


CONVERSATIONS = (
    ConversationCase("OVR01", "dev", "override_category", ("I want running shoes", "Actually make those sandals"), expected_category="sandals"),
    ConversationCase("OVR02", "heldout", "override_category", ("Show me black boots", "On second thought, sneakers instead"), expected_category="sneakers"),
    ConversationCase("OVR03", "dev", "override_budget", ("Keep it below $150", "Changed my mind, under $90"), expected_maximum_price=90.0),
    ConversationCase("OVR04", "heldout", "override_budget", ("My budget is $200", "Cap it at $120 instead"), expected_maximum_price=120.0),
    ConversationCase("OVR05", "dev", "override_remove", ("I want black shoes", "Actually colour doesn't matter anymore"), expected_category="shoes", expected_removed=frozenset({"color"})),
    ConversationCase("OVR06", "heldout", "override_remove", ("Nike sneakers please", "I don't care about brand now"), expected_category="sneakers", expected_removed=frozenset({"brand"})),
    ConversationCase("OVR07", "dev", "override_replace", ("Blue shirts", "Make that red"), expected_category="shirts", expected_attributes={"color": ("red",)}),
    ConversationCase("OVR08", "heldout", "override_replace", ("Cotton jacket", "Polyester would be better"), expected_category="jackets", expected_attributes={"material": ("polyester",)}),
    ConversationCase("OVR09", "dev", "override_intent", ("Black Nike shoes under $100", "Actually I'm just browsing now"), expected_category="shoes", expected_intent="browsing"),
    ConversationCase("OVR10", "heldout", "override_intent", ("I need Adidas runners", "No rush, show me general ideas instead"), expected_category="running shoes", expected_intent="browsing"),
    ConversationCase("OVR11", "dev", "override_category", ("Looking for dresses", "Switch to jackets"), expected_category="jackets"),
    ConversationCase("OVR12", "heldout", "override_category", ("Find sandals", "Actually I need walking shoes"), expected_category="walking shoes"),
    ConversationCase("OVR13", "dev", "override_budget", ("Shoes under $80", "Raise the limit to $110"), expected_category="shoes", expected_maximum_price=110.0),
    ConversationCase("OVR14", "heldout", "override_budget", ("Budget is $100", "There is no budget limit now"), expected_removed=frozenset({"budget"}), expected_maximum_price=None),
    ConversationCase("OVR15", "dev", "override_remove", ("Leather boots", "Material doesn't matter"), expected_category="boots", expected_removed=frozenset({"material"})),
    ConversationCase("OVR16", "heldout", "override_replace", ("Black sandals", "White or red would be better"), expected_category="sandals", expected_attributes={"color": ("white", "red")}),
)
