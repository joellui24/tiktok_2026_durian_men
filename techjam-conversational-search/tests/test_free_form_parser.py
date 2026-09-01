from __future__ import annotations

import unittest

from starter.free_form_parser import parse_free_form_message


BRANDS = ("Nike", "Adidas", "Puma")


class FreeFormParserTest(unittest.TestCase):
    def parse(self, message: str):
        return parse_free_form_message(message, known_brands=BRANDS)

    def test_direct_multi_slot_request(self) -> None:
        parsed = self.parse("I want black Nike running shoes under $120")
        self.assertEqual(parsed.intent, "buying")
        self.assertEqual(parsed.category, "running shoes")
        self.assertEqual(parsed.attributes["color"], ["black"])
        self.assertEqual(parsed.attributes["brand"], ["Nike"])
        self.assertEqual(parsed.attributes["use_case"], ["running"])
        self.assertEqual(parsed.maximum_price, 120.0)

    def test_direct_material_request(self) -> None:
        parsed = self.parse("I need a blue cotton shirt")
        self.assertEqual(parsed.category, "shirts")
        self.assertEqual(parsed.attributes["color"], ["blue"])
        self.assertEqual(parsed.attributes["material"], ["cotton"])

    def test_synonyms_are_explicit_and_field_scoped(self) -> None:
        comfy = self.parse("I want comfy shoes")
        self.assertEqual(comfy.attributes["feature"], ["comfort"])
        cushioned = self.parse("Something cushioned for running")
        self.assertEqual(cushioned.category, "running shoes")
        self.assertEqual(cushioned.attributes["feature"], ["cushioned"])
        self.assertEqual(cushioned.attributes["use_case"], ["running"])

    def test_qualitative_words_do_not_invent_hard_values(self) -> None:
        dark = self.parse("dark coloured sneakers")
        self.assertNotIn("color", dark.attributes)
        cheap = self.parse("cheap running shoes")
        self.assertIsNone(cheap.maximum_price)
        self.assertEqual(cheap.qualitative_budget, "affordable")

    def test_or_and_negation_have_distinct_representations(self) -> None:
        alternatives = self.parse("Blue or black sneakers below $100")
        self.assertEqual(
            alternatives.alternatives["color"], ["blue", "black"]
        )
        self.assertNotIn("color", alternatives.attributes)
        excluded = self.parse("Leather shoes but not brown")
        self.assertEqual(excluded.attributes["material"], ["leather"])
        self.assertEqual(excluded.excluded["color"], ["brown"])

    def test_no_preference_and_constraint_removal(self) -> None:
        no_brand = self.parse(
            "Need some sneakers for jogging, don't really care about brand"
        )
        self.assertIn("brand", no_brand.remove_attributes)
        self.assertNotIn("brand", no_brand.attributes)
        no_color = self.parse("Actually colour doesn't matter anymore")
        self.assertEqual(no_color.remove_attributes, {"color"})

    def test_operator_negation_is_recorded_not_promoted_to_positive(self) -> None:
        cases = (
            ("Anything except black shoes", "color", "black"),
            ("Avoid red dresses", "color", "red"),
            ("Blue sneakers without Adidas", "brand", "Adidas"),
            ("I do not want wool", "material", "wool"),
            ("Sandals in any colour other than white", "color", "white"),
        )
        for message, attribute, value in cases:
            with self.subTest(message=message):
                parsed = self.parse(message)
                self.assertIn(value, parsed.excluded[attribute])
                self.assertNotIn(value, parsed.attributes.get(attribute, ()))

    def test_or_creates_same_field_alternatives(self) -> None:
        cases = (
            ("Nike or Adidas running shoes", "brand", ["Nike", "Adidas"]),
            ("Cotton or wool shirts", "material", ["cotton", "wool"]),
            ("Sneakers or sandals", "category", ["sneakers", "sandals"]),
            ("Size 9 or 10 running shoes", "size", ["9", "10"]),
        )
        for message, attribute, values in cases:
            with self.subTest(message=message):
                parsed = self.parse(message)
                self.assertEqual(parsed.alternatives[attribute], values)
                self.assertNotIn(attribute, parsed.attributes)

    def test_instead_and_preference_removal_phrases(self) -> None:
        replacement = self.parse("White instead of black shoes")
        self.assertEqual(replacement.attributes["color"], ["white"])
        self.assertEqual(replacement.excluded["color"], ["black"])

        cases = (
            ("I have no preference for colour", "color"),
            ("Any brand is fine", "brand"),
            ("Material is irrelevant now", "material"),
            ("There is no budget limit now", "budget"),
        )
        for message, attribute in cases:
            with self.subTest(message=message):
                self.assertIn(attribute, self.parse(message).remove_attributes)

    def test_browsing_and_underspecified_intent(self) -> None:
        browsing = self.parse("Show me some cool things for a beach holiday")
        self.assertEqual(browsing.intent, "browsing")
        vague_show = self.parse("Show me shoes")
        self.assertEqual(vague_show.intent, "browsing")
        constrained_show = self.parse(
            "Show me black Nike running shoes under $120"
        )
        self.assertEqual(constrained_show.intent, "buying")
        leather_show = self.parse(
            "If it is leather and under 150 dollars, show me boots"
        )
        self.assertEqual(leather_show.intent, "buying")
        broad = self.parse("I need shoes")
        self.assertEqual(broad.intent, "unknown")
        self.assertEqual(broad.category, "shoes")

    def test_related_but_different_values_remain_distinct(self) -> None:
        walking = self.parse("comfortable shoes for walking")
        self.assertEqual(walking.attributes["use_case"], ["walking"])
        self.assertNotIn("running", walking.attributes["use_case"])
        dark_blue = self.parse("dark blue shoes")
        self.assertEqual(dark_blue.attributes["color"], ["blue"])
        polyester = self.parse("polyester shirt")
        self.assertEqual(polyester.attributes["material"], ["polyester"])
        self.assertNotIn("cotton", polyester.attributes["material"])
        adidas = self.parse("Adidas sandals")
        self.assertEqual(adidas.attributes["brand"], ["Adidas"])
        self.assertNotIn("Nike", adidas.attributes["brand"])

    def test_common_request_words_are_not_treated_as_brands(self) -> None:
        parsed = parse_free_form_message(
            "Can you find me something comfortable for walking all day?",
            known_brands=("find", "Nike"),
        )
        self.assertNotIn("brand", parsed.attributes)
        refresh = parse_free_form_message(
            "Could you inspire me for an upcoming gym refresh?",
            known_brands=("REFRESH", "Nike"),
        )
        self.assertNotIn("brand", refresh.attributes)

    def test_conservative_catalog_synonyms_become_exact_constraints(self) -> None:
        cases = (
            ("jet-black footwear for weekend runs", "running shoes", "black"),
            ("crimson silk dress", "dresses", "red"),
            ("navy road runners", "running shoes", "blue"),
            ("yellow beach flip-flops", "sandals", "yellow"),
        )
        for message, category, color in cases:
            with self.subTest(message=message):
                parsed = self.parse(message)
                self.assertEqual(parsed.category, category)
                self.assertEqual(parsed.attributes["color"], [color])

    def test_broad_dark_wording_stays_unconstrained(self) -> None:
        parsed = self.parse("dark coloured footwear for sightseeing")
        self.assertNotIn("color", parsed.attributes)
        self.assertNotIn("color", parsed.alternatives)

    def test_capped_budget_and_walking_footwear_are_structured(self) -> None:
        budget = self.parse("Nike runners capped at $135")
        self.assertEqual(budget.category, "running shoes")
        self.assertEqual(budget.maximum_price, 135.0)
        walking = self.parse("footwear that can handle sightseeing on foot")
        self.assertEqual(walking.category, "walking shoes")
        self.assertEqual(walking.attributes["use_case"], ["travel", "walking"])

    def test_brand_spans_and_nonliteral_materials_do_not_create_hard_filters(self) -> None:
        brand = parse_free_form_message(
            "Cotton On shirts", known_brands=("Cotton On",)
        )
        self.assertEqual(brand.attributes["brand"], ["Cotton On"])
        self.assertNotIn("material", brand.attributes)

        for message in ("cotton-like synthetic shirt", "vegan leather boots"):
            with self.subTest(message=message):
                parsed = self.parse(message)
                self.assertNotIn("material", parsed.attributes)

    def test_category_alias_negation_uses_the_users_actual_words(self) -> None:
        cases = (
            ("not runners", "running shoes"),
            ("avoid road runners", "running shoes"),
            ("anything except flip flops", "sandals"),
        )
        for message, category in cases:
            with self.subTest(message=message):
                parsed = self.parse(message)
                self.assertIn(category, parsed.excluded["category"])
                self.assertNotEqual(parsed.category, category)

    def test_colour_idioms_are_not_promoted_to_hard_colours(self) -> None:
        for message in (
            "Black Friday shoe deals",
            "a black-tie dress",
            "red-carpet style",
            "blue-collar work boots",
        ):
            with self.subTest(message=message):
                parsed = self.parse(message)
                self.assertNotIn("color", parsed.attributes)

    def test_longest_overlapping_brand_span_wins(self) -> None:
        parsed = parse_free_form_message(
            "Nike Golf shoes", known_brands=("Nike", "Nike Golf")
        )
        self.assertEqual(parsed.attributes["brand"], ["Nike Golf"])

    def test_numeric_ceiling_preserves_boundary_semantics(self) -> None:
        exclusive = self.parse("shoes under $120")
        self.assertEqual(exclusive.maximum_price, 120.0)
        self.assertFalse(exclusive.maximum_price_inclusive)

        for message in (
            "shoes up to $120",
            "shoes at most 120 dollars",
            "shoes no more than $120",
            "shoes not over 120 dollars",
        ):
            with self.subTest(message=message):
                parsed = self.parse(message)
                self.assertEqual(parsed.maximum_price, 120.0)
                self.assertTrue(parsed.maximum_price_inclusive)

        thousands = self.parse("shoes under $1,200")
        self.assertEqual(thousands.maximum_price, 1200.0)
        self.assertFalse(thousands.maximum_price_inclusive)

    def test_material_free_wording_is_an_exclusion(self) -> None:
        for message in (
            "leather-free shoes",
            "non-leather shoes",
            "shoes free of leather",
        ):
            with self.subTest(message=message):
                parsed = self.parse(message)
                self.assertEqual(parsed.excluded["material"], ["leather"])
                self.assertNotIn("leather", parsed.attributes.get("material", ()))


if __name__ == "__main__":
    unittest.main()
