from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import starter.agent as agent_module
from starter.agent import (
    Agent,
    EVALUATOR_ATTRIBUTES,
    ROADMAP_ATTRIBUTES,
    SessionState,
    classify_constraint,
)
from starter.attribute_index import build_attribute_database, normalize_value
from starter.category_index import build_category_database


LARGE_CATEGORY = "Tops Tunics"
SMALL_CATEGORY = "Accessories Belts"


class RecordingModel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.metadata = {"feature_schema_version": "conversation-features-v2"}
        self.calls: list[tuple[str, str]] = []

    def posterior(
        self,
        parent_asins: set[str],
        context_names: list[str],
        *,
        mode: str,
        disabled_field_pair: tuple[str, str] | None,
    ) -> dict[str, float]:
        del context_names, disabled_field_pair
        self.calls.append(("posterior", mode))
        probability = 1.0 / len(parent_asins)
        return {parent_asin: probability for parent_asin in parent_asins}

    def predicted_reply(
        self,
        parent_asin: str,
        attribute: str,
        disclosed_values: set[str],
    ) -> tuple[str, ...]:
        del disclosed_values
        return (f"{attribute}:{parent_asin}",)

    def rank(
        self,
        parent_asins: set[str],
        context_names: list[str],
        limit: int,
        *,
        mode: str,
        disabled_field_pair: tuple[str, str] | None,
    ) -> list[str]:
        del context_names, disabled_field_pair
        self.calls.append(("rank", mode))
        return sorted(parent_asins)[:limit]


def _catalog_products() -> list[dict]:
    products: list[dict] = []
    for index in range(15):
        products.append(
            {
                "parent_asin": f"A{index:02d}",
                "title": f"Test tunic {index}",
                "categories": [
                    "Clothing, Shoes & Jewelry",
                    "Women",
                    "Tops",
                    "Tunics",
                ],
                "features": [f"feature-{index}", "common option"],
                "details": {
                    "Fabric Type": "wool" if index in (2, 4) else "cotton",
                    "Color": "red" if index % 2 == 0 else "blue",
                },
                "store": f"Brand-{index}",
                "price": 20.0 + index,
            }
        )
    for index in range(3):
        products.append(
            {
                "parent_asin": f"B{index:02d}",
                "title": f"Test belt {index}",
                "categories": [
                    "Clothing, Shoes & Jewelry",
                    "Men",
                    "Accessories",
                    "Belts",
                ],
                "features": [f"belt-feature-{index}"],
                "details": {"Color": "black"},
                "store": f"BeltBrand-{index}",
                "price": 10.0 + index,
            }
        )
    return products


class ProgressiveAgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        cls.data_directory = Path(cls._temporary_directory.name)
        cls.catalog_path = cls.data_directory / "catalog.jsonl"
        cls.category_path = cls.data_directory / "category_index.sqlite3"
        cls.attribute_path = cls.data_directory / "attribute_index.sqlite3"
        cls.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in _catalog_products()),
            encoding="utf-8",
        )
        build_category_database(cls.catalog_path, cls.category_path)
        build_attribute_database(cls.catalog_path, cls.attribute_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def setUp(self) -> None:
        self.agent = Agent(
            self.catalog_path,
            category_index_path=self.category_path,
            attribute_index_path=self.attribute_path,
            model_path=self.data_directory / "missing-hybrid.sqlite3",
            linear_model_path=self.data_directory / "missing-linear.sqlite3",
            ranking_mode="routed",
        )
        self.addCleanup(self.agent.close)

    def _reset(self, session_id: str = "session", profile: dict | None = None) -> None:
        self.agent.reset(session_id, profile or {"summary": "stable profile"})

    def _start_browsing(self, session_id: str = "session", turn: int = 1) -> dict:
        self._reset(session_id)
        return self.agent.respond(
            session_id,
            f"I'm looking for {LARGE_CATEGORY}, but I'm still exploring.",
            turn,
            10,
        )

    def test_initial_parsing_exact_category_and_evaluator_classification(self) -> None:
        self.assertEqual(len(EVALUATOR_ATTRIBUTES), 10)
        self.assertNotIn("category", ROADMAP_ATTRIBUTES)
        self.assertEqual(
            Agent._parse_initial_message(
                f"I'm looking for {LARGE_CATEGORY}. A key requirement is: cotton."
            ),
            ("buying", LARGE_CATEGORY, "cotton"),
        )
        self.assertEqual(
            Agent._parse_initial_message(
                f"I'm looking for {LARGE_CATEGORY}, but I'm still exploring."
            ),
            ("exploring_unknown", LARGE_CATEGORY, None),
        )
        self.assertEqual(
            Agent._parse_initial_message(
                f"I'm looking for {LARGE_CATEGORY}. Actually, ignore my earlier preference."
            ),
            ("intent_override", LARGE_CATEGORY, None),
        )
        self.assertEqual(classify_constraint("gray hat"), "feature")
        self.assertEqual(classify_constraint("color: green"), "color")
        self.assertEqual(classify_constraint("budget around $30"), "budget")

        self._reset()
        self.agent.respond(
            "session",
            f"I'm looking for {LARGE_CATEGORY}. A key requirement is: common option.",
            1,
            10,
        )
        state = self.agent._sessions["session"]
        self.assertEqual(state.coarse_category, LARGE_CATEGORY)
        self.assertEqual(len(state.surviving_candidates), 15)
        self.assertEqual(state.known_constraints["feature"], ["common option"])

    @unittest.skipIf(
        agent_module.rapidfuzz_fuzz is None, "RapidFuzz is not installed"
    )
    def test_fuzzy_intent_fallback_handles_template_wording_variations(self) -> None:
        self.assertEqual(
            Agent._parse_initial_message(
                f"I’m looking for {LARGE_CATEGORY}. A key requirment is: cotton."
            ),
            ("buying", LARGE_CATEGORY, "cotton"),
        )
        self.assertEqual(
            Agent._parse_initial_message(
                f"I'm looking for {LARGE_CATEGORY}, but im still exploreing."
            ),
            ("exploring_unknown", LARGE_CATEGORY, None),
        )

        self._start_browsing("fuzzy-boundary")
        self.agent.respond(
            "fuzzy-boundary",
            "I don't have a preference; please use your judgmet.",
            2,
            10,
        )
        self.assertEqual(
            self.agent._sessions["fuzzy-boundary"].scenario_state, "boundary"
        )

    def test_exact_non_buying_cue_wins_over_buying_words(self) -> None:
        scenario, _, _ = Agent._parse_initial_message(
            f"I'm looking for {LARGE_CATEGORY}; requirements matter, "
            "but I'm still exploring."
        )
        self.assertEqual(scenario, "exploring_unknown")

    def test_routed_mode_uses_one_model_for_questions_and_ranking(self) -> None:
        linear = RecordingModel("linear")
        hybrid = RecordingModel("hybrid")
        self.agent.ranking_mode = "routed"
        self.agent.linear_model = linear  # type: ignore[assignment]
        self.agent.model = hybrid  # type: ignore[assignment]

        buying = SessionState(
            scenario_state="buying",
            coarse_category=LARGE_CATEGORY,
            surviving_candidates={f"A{index:02d}" for index in range(15)},
        )
        self.agent._choose_next_attribute(buying, turn=1)
        self.agent._recommendations(buying, top_k=10, turn=1)
        self.assertEqual(linear.calls, [("posterior", "linear"), ("rank", "linear")])
        self.assertEqual(hybrid.calls, [])

        browsing = SessionState(
            scenario_state="browsing",
            coarse_category=LARGE_CATEGORY,
            surviving_candidates={f"A{index:02d}" for index in range(15)},
        )
        self.agent._choose_next_attribute(browsing, turn=2)
        self.agent._recommendations(browsing, top_k=10, turn=2)
        self.assertEqual(
            hybrid.calls, [("posterior", "hybrid"), ("rank", "hybrid")]
        )

    def test_routed_mode_falls_back_to_the_available_model(self) -> None:
        linear = RecordingModel("linear")
        hybrid = RecordingModel("hybrid")
        self.agent.ranking_mode = "routed"

        self.agent.linear_model = None
        self.agent.model = hybrid  # type: ignore[assignment]
        selected, mode = self.agent._active_model(
            SessionState(scenario_state="buying")
        )
        self.assertIs(selected, hybrid)
        self.assertEqual(mode, "hybrid")

        self.agent.linear_model = linear  # type: ignore[assignment]
        self.agent.model = None
        selected, mode = self.agent._active_model(
            SessionState(scenario_state="browsing")
        )
        self.assertIs(selected, linear)
        self.assertEqual(mode, "linear")

    def test_explicit_model_path_keeps_legacy_single_hybrid_default(self) -> None:
        with Agent(
            self.catalog_path,
            category_index_path=self.category_path,
            attribute_index_path=self.attribute_path,
            model_path=self.data_directory / "missing-explicit.sqlite3",
        ) as explicit_agent:
            self.assertEqual(explicit_agent.ranking_mode, "hybrid")
            self.assertIsNone(explicit_agent.linear_model)

    def test_buying_resumes_after_the_initial_constraints_entire_stage(self) -> None:
        self._reset()
        response = self.agent.respond(
            "session",
            f"I'm looking for {LARGE_CATEGORY}. A key requirement is: common option.",
            1,
            10,
        )
        state = self.agent._sessions["session"]
        self.assertTrue(
            {"use_case", "feature", "style", "material"}.issubset(
                state.exhausted_attributes
            )
        )
        self.assertEqual(state.roadmap_stage, 2)
        self.assertEqual(response["ask_attribute"], "budget")

    def test_browsing_and_boundary_diverge_only_after_boundary_reply(self) -> None:
        self._start_browsing("browse")
        self._start_browsing("boundary")
        self.assertEqual(
            self.agent._sessions["browse"].scenario_state, "exploring_unknown"
        )
        self.assertEqual(
            self.agent._sessions["boundary"].scenario_state, "exploring_unknown"
        )

        self.agent.respond(
            "browse",
            "I don't have an additional preference for use_case.",
            2,
            10,
        )
        self.agent.respond(
            "boundary",
            "I don't have a preference for use_case; please use your judgment.",
            2,
            10,
        )
        self.assertEqual(self.agent._sessions["browse"].scenario_state, "browsing")
        self.assertEqual(
            self.agent._sessions["boundary"].scenario_state, "boundary"
        )

    def test_same_and_different_attribute_values_use_and_with_rollback(self) -> None:
        self._start_browsing()
        state = self.agent._sessions["session"]

        self.assertTrue(
            self.agent._apply_values(
                state, "feature", ["feature-0", "common option"]
            )
        )
        self.assertEqual(state.surviving_candidates, {"A00"})

        self._start_browsing("different")
        state = self.agent._sessions["different"]
        self.assertTrue(self.agent._apply_values(state, "material", ["cotton"]))
        self.assertTrue(self.agent._apply_values(state, "color", ["red"]))
        self.assertEqual(
            state.surviving_candidates,
            {f"A{index:02d}" for index in range(15) if index % 2 == 0 and index not in (2, 4)},
        )

        before = set(state.surviving_candidates)
        self.assertFalse(
            self.agent._apply_values(state, "feature", ["not in the index"])
        )
        self.assertEqual(state.surviving_candidates, before)
        self.assertIn(
            ("feature", normalize_value("not in the index")), state.unindexed_values
        )

    def test_session_exhaustion_does_not_mutate_shared_hashmaps(self) -> None:
        mapping = self.agent._hashmap("feature")
        keys_before = frozenset(mapping)
        posting_before = mapping[normalize_value("common option")]

        self._start_browsing()
        self.agent.respond(
            "session",
            "I don't have an additional preference for use_case.",
            2,
            10,
        )
        self.assertEqual(frozenset(mapping), keys_before)
        self.assertEqual(mapping[normalize_value("common option")], posting_before)
        self.assertIn("use_case", self.agent._sessions["session"].exhausted_attributes)

    def test_adaptive_questions_and_recommendations_are_session_id_independent(self) -> None:
        profile = {"summary": "identical stable content", "tags": ["fit"]}
        for session_id in ("random-id-one", "a-completely-different-id"):
            self._reset(session_id, profile)

        initial_message = (
            f"I'm looking for {LARGE_CATEGORY}, but I'm still exploring."
        )
        first = self.agent.respond("random-id-one", initial_message, 1, 10)
        second = self.agent.respond(
            "a-completely-different-id", initial_message, 1, 10
        )
        self.assertEqual(first["ask_attribute"], "use_case")
        self.assertEqual(len(first["recommendations"]), 10)
        self.assertEqual(first, second)

        reply = "I don't have an additional preference for use_case."
        first = self.agent.respond("random-id-one", reply, 2, 10)
        second = self.agent.respond("a-completely-different-id", reply, 2, 10)
        # All three stage-two attributes have a largest bucket of 15 in this
        # fixture, so the roadmap-order tie break chooses feature.
        self.assertEqual(first["ask_attribute"], "feature")
        self.assertEqual(len(first["recommendations"]), 10)
        self.assertEqual(first, second)

    def test_recommendations_return_ranked_top_ten_on_every_turn(self) -> None:
        response = self._start_browsing()
        state = self.agent._sessions["session"]
        self.assertGreater(len(state.surviving_candidates), 10)
        recommendations = [
            item["parent_asin"] for item in response["recommendations"]
        ]
        self.assertEqual(recommendations, [f"A{index:02d}" for index in range(10)])
        self.assertEqual(len(recommendations), len(set(recommendations)))
        self.assertTrue(set(recommendations).issubset(state.surviving_candidates))

        self.assertTrue(self.agent._apply_values(state, "feature", ["feature-0"]))
        recommendations = [
            item["parent_asin"]
            for item in self.agent._recommendations(state, top_k=10)
        ]
        self.assertEqual(recommendations, ["A00"])
        self.assertEqual(len(recommendations), len(set(recommendations)))
        self.assertTrue(set(recommendations).issubset(state.surviving_candidates))
        self.assertTrue(set(recommendations).issubset(self.agent._catalog_ids()))

    def test_early_stop_returns_all_survivors_and_turn_ten_never_asks(self) -> None:
        self._reset("small")
        response = self.agent.respond(
            "small",
            f"I'm looking for {SMALL_CATEGORY}, but I'm still exploring.",
            1,
            10,
        )
        self.assertIsNone(response["ask_attribute"])
        self.assertEqual(
            [item["parent_asin"] for item in response["recommendations"]],
            ["B00", "B01", "B02"],
        )

        response = self._start_browsing("last-turn", turn=10)
        self.assertIsNone(response["ask_attribute"])
        self.assertEqual(len(response["recommendations"]), 10)

    def test_unknown_initial_template_uses_nonempty_safe_fallback(self) -> None:
        self._reset()
        response = self.agent.respond("session", "Please help me shop.", 1, 10)
        state = self.agent._sessions["session"]
        self.assertEqual(state.scenario_state, "unknown")
        self.assertEqual(state.coarse_category, "clothing item")
        self.assertEqual(len(state.surviving_candidates), 18)
        self.assertEqual(len(response["recommendations"]), 10)

    def test_intent_override_replaces_obsolete_filters_atomically(self) -> None:
        self._reset()
        first = self.agent.respond(
            "session",
            f"I'm looking for {LARGE_CATEGORY}. feature-0.",
            1,
            10,
        )
        state = self.agent._sessions["session"]
        self.assertEqual(state.scenario_state, "provisional_override")
        self.assertEqual(state.surviving_candidates, {"A00"})
        self.assertIsNone(first["ask_attribute"])

        linear = RecordingModel("linear")
        hybrid = RecordingModel("hybrid")
        self.agent.linear_model = linear  # type: ignore[assignment]
        self.agent.model = hybrid  # type: ignore[assignment]
        selected, mode = self.agent._active_model(state)
        self.assertIs(selected, hybrid)
        self.assertEqual(mode, "hybrid")

        response = self.agent.respond(
            "session",
            "Actually, ignore my earlier preference. What I need is: common option.",
            3,
            10,
        )
        self.assertEqual(state.scenario_state, "intent_override")
        self.assertEqual(state.intent_epoch, 1)
        self.assertEqual(state.override_count, 1)
        self.assertEqual(state.known_constraints, {"feature": ["common option"]})
        self.assertEqual(len(state.surviving_candidates), 15)
        self.assertNotIn("feature-0", state.known_constraints["feature"])
        self.assertEqual(len(response["recommendations"]), 10)
        selected, mode = self.agent._active_model(state)
        self.assertIs(selected, hybrid)
        self.assertEqual(mode, "hybrid")
        self.assertIn(("posterior", "hybrid"), hybrid.calls)
        self.assertIn(("rank", "hybrid"), hybrid.calls)

    def test_incomplete_override_preserves_existing_state(self) -> None:
        self._reset()
        self.agent.respond(
            "session",
            f"I'm looking for {LARGE_CATEGORY}. feature-0.",
            1,
            10,
        )
        state = self.agent._sessions["session"]
        before = set(state.surviving_candidates)
        self.agent.respond(
            "session",
            "Actually, I changed my mind.",
            2,
            10,
        )
        self.assertEqual(state.surviving_candidates, before)
        self.assertEqual(state.intent_epoch, 0)

    def test_explicit_override_switches_a_buying_session_to_hybrid(self) -> None:
        linear = RecordingModel("linear")
        hybrid = RecordingModel("hybrid")
        self.agent.linear_model = linear  # type: ignore[assignment]
        self.agent.model = hybrid  # type: ignore[assignment]
        self._reset()

        self.agent.respond(
            "session",
            f"I'm looking for {LARGE_CATEGORY}. A key requirement is: common option.",
            1,
            10,
        )
        self.assertEqual(self.agent._sessions["session"].scenario_state, "buying")
        self.assertIn(("rank", "linear"), linear.calls)

        self.agent.respond(
            "session",
            "Actually, ignore my earlier preference. What I need is: cotton.",
            2,
            10,
        )
        self.assertEqual(
            self.agent._sessions["session"].scenario_state, "intent_override"
        )
        self.assertIn(("rank", "hybrid"), hybrid.calls)

    def test_free_form_opening_applies_category_color_and_numeric_budget(self) -> None:
        self._reset()
        self.agent.respond(
            "session", "I need black belts below $12", 1, 10
        )
        state = self.agent._sessions["session"]
        self.assertTrue(state.free_form_active)
        self.assertEqual(state.scenario_state, "buying")
        self.assertEqual(state.coarse_category, "belts")
        self.assertEqual(state.known_constraints["color"], ["black"])
        self.assertEqual(state.maximum_price, 12.0)
        self.assertEqual(state.surviving_candidates, {"B00", "B01"})

    def test_free_form_category_change_clears_obsolete_constraints(self) -> None:
        self._reset()
        self.agent.respond("session", "I want black belts", 1, 10)
        state = self.agent._sessions["session"]
        self.assertEqual(state.surviving_candidates, {"B00", "B01", "B02"})

        self.agent.respond("session", "Actually make that tunics", 2, 10)
        self.assertEqual(state.scenario_state, "intent_override")
        self.assertEqual(state.coarse_category, "tunics")
        self.assertEqual(state.known_constraints, {})
        self.assertEqual(
            state.surviving_candidates, {f"A{index:02d}" for index in range(15)}
        )

    def test_free_form_removal_and_budget_replacement_rebuild_candidates(self) -> None:
        self._reset("remove")
        self.agent.respond("remove", "I want something black", 1, 10)
        state = self.agent._sessions["remove"]
        self.assertEqual(state.surviving_candidates, {"B00", "B01", "B02"})
        self.agent.respond(
            "remove", "Actually colour doesn't matter anymore", 2, 10
        )
        self.assertEqual(len(state.surviving_candidates), 18)
        self.assertNotIn("color", state.known_constraints)

        self._reset("budget")
        self.agent.respond("budget", "Budget is $30", 1, 10)
        state = self.agent._sessions["budget"]
        self.assertEqual(len(state.surviving_candidates), 14)
        self.agent.respond(
            "budget", "Changed my mind, keep it below $25", 2, 10
        )
        self.assertEqual(state.maximum_price, 25.0)
        self.assertEqual(len(state.surviving_candidates), 8)

    def test_free_form_operator_edits_replace_or_remove_old_preferences(self) -> None:
        self._reset("operator-edits")
        self.agent.respond("operator-edits", "I want something black below $30", 1, 10)
        state = self.agent._sessions["operator-edits"]

        self.agent.respond("operator-edits", "White would be better instead", 2, 10)
        self.assertEqual(state.known_constraints["color"], ["white"])
        self.assertNotIn("black", state.known_constraints["color"])
        self.assertEqual(state.scenario_state, "intent_override")

        self.agent.respond("operator-edits", "There is no budget limit now", 3, 10)
        self.assertIsNone(state.maximum_price)
        self.assertNotIn("budget", state.known_constraints)
        self.assertEqual(state.scenario_state, "intent_override")

        self._reset("budget-operator")
        self.agent.respond("budget-operator", "Shoes under $80", 1, 10)
        self.agent.respond("budget-operator", "Raise the limit to $110", 2, 10)
        budget_state = self.agent._sessions["budget-operator"]
        self.assertEqual(budget_state.maximum_price, 110.0)
        self.assertEqual(budget_state.scenario_state, "intent_override")

    def test_free_form_or_and_exclusion_affect_candidates(self) -> None:
        self._reset("or-filter")
        self.agent.respond("or-filter", "I want red or blue tunics", 1, 10)
        state = self.agent._sessions["or-filter"]
        self.assertEqual(state.alternative_constraints["color"], ["red", "blue"])
        self.assertEqual(len(state.surviving_candidates), 15)

        self.agent.respond("or-filter", "Avoid red", 2, 10)
        self.assertEqual(state.excluded_constraints["color"], ["red"])
        self.assertEqual(
            state.surviving_candidates,
            {f"A{index:02d}" for index in range(15) if index % 2 == 1},
        )

    def test_soft_alternatives_do_not_become_destructive_exact_filters(self) -> None:
        cases = (
            "breathable or lightweight tunics",
            "size 9 or 10 tunics",
            "running or walking tunics",
        )
        for index, message in enumerate(cases):
            session_id = f"soft-or-{index}"
            with self.subTest(message=message):
                self._reset(session_id)
                self.agent.respond(session_id, message, 1, 10)
                state = self.agent._sessions[session_id]
                self.assertEqual(len(state.surviving_candidates), 15)

    def test_free_form_blanket_override_clears_every_old_state_store(self) -> None:
        self._reset("blanket")
        self.agent.respond(
            "blanket", "I want red tunics below $25", 1, 10
        )
        state = self.agent._sessions["blanket"]
        self.assertTrue(state.hard_constraints)
        self.assertIsNotNone(state.maximum_price)
        self.assertTrue(state.semantic_fragments)

        self.agent.respond(
            "blanket",
            "Actually, ignore my earlier preference. What I need is: wool.",
            2,
            10,
        )
        self.assertEqual(state.hard_constraints, {"material": ["wool"]})
        self.assertIsNone(state.maximum_price)
        self.assertNotIn("color", state.known_constraints)
        self.assertNotIn("red", self.agent._semantic_query(state).casefold())

        self.agent.respond("blanket", "make it breathable", 3, 10)
        self.assertEqual(state.hard_constraints, {"material": ["wool"]})
        self.assertIsNone(state.maximum_price)
        self.assertNotIn("red", self.agent._semantic_query(state).casefold())

    def test_positive_and_excluded_values_are_reconciled_both_ways(self) -> None:
        self._reset("exclude-positive")
        self.agent.respond("exclude-positive", "I want red tunics", 1, 10)
        state = self.agent._sessions["exclude-positive"]
        self.agent.respond("exclude-positive", "avoid red", 2, 10)
        self.assertNotIn("color", state.hard_constraints)
        self.assertNotIn("color", state.known_constraints)
        self.assertEqual(state.excluded_constraints["color"], ["red"])
        self.assertEqual(
            state.surviving_candidates,
            {f"A{index:02d}" for index in range(15) if index % 2 == 1},
        )

        self._reset("positive-exclude")
        self.agent.respond("positive-exclude", "tunics but not red", 1, 10)
        state = self.agent._sessions["positive-exclude"]
        self.agent.respond("positive-exclude", "Actually red is fine", 2, 10)
        self.assertNotIn("color", state.excluded_constraints)
        self.assertEqual(state.hard_constraints["color"], ["red"])
        self.assertEqual(
            state.surviving_candidates,
            {f"A{index:02d}" for index in range(15) if index % 2 == 0},
        )

    def test_exclusions_that_remove_every_candidate_ask_for_clarification(self) -> None:
        self._reset("exclude-all")
        response = self.agent.respond(
            "exclude-all", "tunics without red or blue", 1, 10
        )
        state = self.agent._sessions["exclude-all"]
        self.assertEqual(state.surviving_candidates, set())
        self.assertEqual(response["recommendations"], [])
        self.assertIsNotNone(response["ask_attribute"])
        self.assertIn("couldn't find", response["message"])

    def test_inclusive_and_exclusive_price_bounds_differ_at_the_boundary(self) -> None:
        self._reset("exclusive-price")
        self.agent.respond("exclusive-price", "tunics under $20", 1, 10)
        self.assertEqual(
            self.agent._sessions["exclusive-price"].surviving_candidates,
            set(),
        )

        self._reset("inclusive-price")
        self.agent.respond("inclusive-price", "tunics up to $20", 1, 10)
        self.assertEqual(
            self.agent._sessions["inclusive-price"].surviving_candidates,
            {"A00"},
        )

    def test_category_alternative_replacement_updates_semantic_state(self) -> None:
        self._reset("category-or")
        self.agent.respond("category-or", "I want black shoes", 1, 10)
        state = self.agent._sessions["category-or"]
        self.agent.respond(
            "category-or", "sandals or boots instead", 2, 10
        )
        query = self.agent._semantic_query(state)
        self.assertEqual(
            state.alternative_constraints["category"], ["sandals", "boots"]
        )
        self.assertNotIn("color", state.known_constraints)
        self.assertIn("category alternative sandals", query)
        self.assertIn("category alternative boots", query)

    def test_implicit_same_field_replacement_drops_stale_semantic_language(self) -> None:
        self._reset("implicit-replace")
        self.agent.respond(
            "implicit-replace", "I want comfortable black tunics", 1, 10
        )
        state = self.agent._sessions["implicit-replace"]
        self.agent.respond("implicit-replace", "white would work", 2, 10)
        self.assertEqual(state.hard_constraints["color"], ["white"])
        self.assertNotIn("black", self.agent._semantic_query(state).casefold())

    def test_category_refinement_preserves_unrelated_preferences(self) -> None:
        self._reset("refine")
        self.agent.respond("refine", "Something comfortable", 1, 10)
        state = self.agent._sessions["refine"]
        self.agent.respond("refine", "tunics", 2, 10)
        self.assertEqual(state.coarse_category, "tunics")
        self.assertEqual(state.known_constraints["feature"], ["comfort"])

        self._reset("narrow")
        self.agent.respond(
            "narrow", "I want red under $25", 1, 10
        )
        narrow_state = self.agent._sessions["narrow"]
        self.agent.respond("narrow", "tunics", 2, 10)
        self.assertEqual(narrow_state.coarse_category, "tunics")
        self.assertEqual(narrow_state.hard_constraints["color"], ["red"])
        self.assertEqual(narrow_state.maximum_price, 25.0)

    def test_changed_mind_field_edit_is_not_a_blanket_reset(self) -> None:
        self._reset("field-edit")
        self.agent.respond(
            "field-edit", "I want red tunics under $25", 1, 10
        )
        state = self.agent._sessions["field-edit"]
        self.agent.respond(
            "field-edit", "Changed my mind, I want blue", 2, 10
        )
        self.assertEqual(state.coarse_category, "tunics")
        self.assertEqual(state.hard_constraints["color"], ["blue"])
        self.assertEqual(state.maximum_price, 25.0)
        self.assertIn("budget", state.known_constraints)

    def test_short_clarification_answers_use_the_asked_field(self) -> None:
        self._reset("answer-budget")
        self.agent.respond("answer-budget", "I need tunics", 1, 10)
        state = self.agent._sessions["answer-budget"]
        state.last_asked_attribute = "budget"
        self.agent.respond("answer-budget", "$1,200.50", 2, 10)
        self.assertEqual(state.maximum_price, 1200.5)
        self.assertTrue(state.maximum_price_inclusive)

        self._reset("answer-size")
        self.agent.respond("answer-size", "I need tunics", 1, 10)
        state = self.agent._sessions["answer-size"]
        state.last_asked_attribute = "size"
        self.agent.respond("answer-size", "medium", 2, 10)
        self.assertEqual(state.known_constraints["size"], ["medium"])

        self._reset("answer-use")
        self.agent.respond("answer-use", "I need tunics", 1, 10)
        state = self.agent._sessions["answer-use"]
        state.last_asked_attribute = "use_case"
        self.agent.respond("answer-use", "daily commuting", 2, 10)
        self.assertEqual(
            state.known_constraints["use_case"], ["daily commuting"]
        )

        self._reset("answer-brand")
        self.agent.respond("answer-brand", "I need tunics", 1, 10)
        state = self.agent._sessions["answer-brand"]
        state.last_asked_attribute = "brand"
        self.agent.respond("answer-brand", "Brand-0", 2, 10)
        self.assertEqual(state.hard_constraints["brand"], ["Brand-0"])

    def test_contextual_no_preference_removes_the_asked_field(self) -> None:
        self._reset("answer-remove")
        self.agent.respond(
            "answer-remove", "I want comfortable tunics", 1, 10
        )
        state = self.agent._sessions["answer-remove"]
        state.last_asked_attribute = "feature"
        self.agent.respond("answer-remove", "no preference", 2, 10)
        self.assertNotIn("feature", state.known_constraints)

    def test_experimental_gliner_layer_cannot_run_on_official_openings(self) -> None:
        from starter.gliner_parser import GLiNERAugmenter, GLiNERExperimentalAgent

        class FailIfCalledModel:
            def __init__(self) -> None:
                self.calls = 0

            def eval(self) -> "FailIfCalledModel":
                return self

            def predict_entities(self, *_: object, **__: object) -> list[dict]:
                self.calls += 1
                raise AssertionError("structured evaluator path called GLiNER")

        fake = FailIfCalledModel()
        augmenter = GLiNERAugmenter(model=fake)
        experimental = GLiNERExperimentalAgent.create(
            augmenter,
            self.catalog_path,
            category_index_path=self.category_path,
            attribute_index_path=self.attribute_path,
            model_path=self.data_directory / "missing-hybrid.sqlite3",
            linear_model_path=self.data_directory / "missing-linear.sqlite3",
            ranking_mode="routed",
        )
        self.addCleanup(experimental.close)

        experimental.reset("official-buying", {})
        experimental.respond(
            "official-buying",
            f"I'm looking for {LARGE_CATEGORY}. A key requirement is: cotton.",
            1,
            10,
        )
        experimental.reset("official-browsing", {})
        experimental.respond(
            "official-browsing",
            f"I'm looking for {LARGE_CATEGORY}, but I'm still exploring.",
            1,
            10,
        )

        self.assertEqual(fake.calls, 0)
        self.assertEqual(experimental.gliner_augmentations, [])

    def test_semantic_retrieval_cannot_run_on_official_openings(self) -> None:
        class FailIfCalledSemanticIndex:
            def __init__(self) -> None:
                self.calls = 0

            def dense_rank(self, *_: object, **__: object) -> list[str]:
                self.calls += 1
                raise AssertionError("structured evaluator path called dense retrieval")

        fake = FailIfCalledSemanticIndex()
        self.agent.free_form_retrieval_mode = "dense"
        self.agent._semantic_index = fake
        self._reset()
        self.agent.respond(
            "session",
            f"I'm looking for {LARGE_CATEGORY}. A key requirement is: cotton.",
            1,
            10,
        )

        self.assertFalse(self.agent._sessions["session"].free_form_active)
        self.assertEqual(fake.calls, 0)

    def test_vague_free_form_language_activates_dense_retrieval(self) -> None:
        class RecordingSemanticIndex:
            def __init__(self) -> None:
                self.calls: list[tuple[str, set[str], int]] = []

            def dense_rank(
                self, query: str, candidates: set[str], *, limit: int
            ) -> list[str]:
                self.calls.append((query, set(candidates), limit))
                return sorted(candidates, reverse=True)[:limit]

        fake = RecordingSemanticIndex()
        self.agent.free_form_retrieval_mode = "dense"
        self.agent._semantic_index = fake
        self._reset()
        self.agent.respond(
            "session", "Something breathable for summer", 1, 10
        )

        state = self.agent._sessions["session"]
        self.assertTrue(state.free_form_active)
        self.assertEqual(len(fake.calls), 1)
        self.assertIn("breathable for summer", fake.calls[0][0].casefold())
        self.assertEqual(fake.calls[0][1], state.surviving_candidates)

    def test_free_form_hard_constraint_conflict_fails_closed(self) -> None:
        self.agent.free_form_retrieval_mode = "off"
        self._reset()
        response = self.agent.respond(
            "session", "I want black tunics below $1", 1, 10
        )

        self.assertEqual(self.agent._sessions["session"].surviving_candidates, set())
        self.assertEqual(response["recommendations"], [])
        self.assertIsNotNone(response["ask_attribute"])

    def test_operator_turn_removes_stale_raw_semantic_fragment(self) -> None:
        self._reset()
        self.agent.respond(
            "session", "I want comfortable black belts", 1, 10
        )
        state = self.agent._sessions["session"]
        self.assertTrue(state.semantic_fragments)

        self.agent.respond(
            "session", "Actually colour doesn't matter anymore", 2, 10
        )
        self.assertEqual(state.semantic_fragments, [])
        self.assertNotIn("black", self.agent._semantic_query(state).casefold())


if __name__ == "__main__":
    unittest.main()
