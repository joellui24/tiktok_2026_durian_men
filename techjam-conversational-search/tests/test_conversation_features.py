from __future__ import annotations

import unittest

from evaluator.local_evaluator import (
    classify_constraint as evaluator_classify_constraint,
)
from starter.agent import Agent, SessionState
from starter.conversation_features import (
    classify_constraint,
    constraint_context_features,
    context_feature_names,
    normalize_constraint,
)


class ConversationFeaturesTest(unittest.TestCase):
    def test_classification_and_normalization_match_evaluator_contract(self) -> None:
        examples = (
            "gray hat",
            "color: green",
            "budget around $30",
            "cotton fabric",
            "wide sizing",
            "relaxed fit",
            "for outdoor work",
        )
        self.assertEqual(
            [classify_constraint(value) for value in examples],
            [evaluator_classify_constraint(value) for value in examples],
        )
        self.assertEqual(
            normalize_constraint("  WATERPROOF\N{NO-BREAK SPACE} shell. "),
            "waterproof shell",
        )

    def test_other_constraint_is_dual_encoded_with_inferred_type(self) -> None:
        self.assertEqual(
            constraint_context_features("other", "  Waterproof shell. "),
            (
                "ctx:answer_source=other",
                "ctx:other=waterproof shell",
                "ctx:feature=waterproof shell",
            ),
        )
        self.assertEqual(
            constraint_context_features("other", "For HIKING"),
            (
                "ctx:answer_source=other",
                "ctx:other=for hiking",
                "ctx:use_case=for hiking",
            ),
        )

    def test_other_reply_has_identical_training_and_runtime_feature_names(self) -> None:
        known_constraints = {
            "other": ["Waterproof shell", "For HIKING"],
            "style": ["Relaxed fit"],
        }
        training_names = context_feature_names(
            coarse_category="Tops Tunics",
            scenario_state="browsing",
            turn=4,
            intent_epoch=1,
            known_constraints=known_constraints,
        )
        agent = object.__new__(Agent)
        agent._attribute_hashmaps = {
            "other": {
                "waterproof shell": ("target",),
                "for hiking": ("target",),
            }
        }
        agent._indexed_products = {}
        runtime_state = SessionState(
            scenario_state="browsing",
            coarse_category="Tops Tunics",
            surviving_candidates={"target", "filtered-out"},
            known_constraints={"style": ["Relaxed fit"]},
            last_asked_attribute="other",
            intent_epoch=1,
        )
        agent._process_reply(
            runtime_state,
            "For that, what matters is: Waterproof shell; For HIKING.",
        )
        runtime_names = agent._context_features(runtime_state, turn=4)

        expected = [
            "ctx:category=tops tunics",
            "ctx:scenario=browsing",
            "ctx:turn=middle",
            "ctx:override=post",
            "ctx:answer_source=other",
            "ctx:other=waterproof shell",
            "ctx:feature=waterproof shell",
            "ctx:other=for hiking",
            "ctx:use_case=for hiking",
            "ctx:style=relaxed fit",
        ]
        self.assertEqual(training_names, expected)
        self.assertEqual(runtime_names, training_names)
        self.assertEqual(runtime_state.surviving_candidates, {"target"})

        # The shared strings are directly ID-ready: the same vocabulary maps
        # both construction paths to exactly the same sparse feature IDs.
        vocabulary = {
            name: feature_id for feature_id, name in enumerate(sorted(expected))
        }
        self.assertEqual(
            tuple(vocabulary[name] for name in runtime_names),
            tuple(vocabulary[name] for name in training_names),
        )
        self.assertEqual(training_names.count("ctx:answer_source=other"), 1)


if __name__ == "__main__":
    unittest.main()
