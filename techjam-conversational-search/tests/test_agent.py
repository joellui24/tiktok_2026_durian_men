from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from starter.agent import (
    Agent,
    EVALUATOR_ATTRIBUTES,
    ROADMAP_ATTRIBUTES,
    classify_constraint,
)
from starter.attribute_index import build_attribute_database, normalize_value
from starter.category_index import build_category_database


LARGE_CATEGORY = "Tops Tunics"
SMALL_CATEGORY = "Accessories Belts"


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


if __name__ == "__main__":
    unittest.main()
