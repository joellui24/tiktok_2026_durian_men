"""Standalone diagnostics for structured free-form parsing and intent fallback.

This file is intentionally not named ``test_*.py`` so the desired free-form
contract can be evaluated without changing the official/default unit-test
discovery result. Run it directly from ``techjam-conversational-search``:

    py -3.13 tests/free_form_parser_diagnostics.py

The process exits with status 1 while desired free-form cases remain unmet.
It imports production parsing functions but does not modify production state,
the evaluator, the catalog, or the public dataset.
"""

from __future__ import annotations

import json
import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starter import agent as agent_module  # noqa: E402
from starter.agent import Agent  # noqa: E402
from starter.free_form_parser import parse_free_form_message  # noqa: E402


@dataclass(frozen=True)
class ParserCase:
    case_id: str
    group: str
    message: str
    expected: dict[str, Any]
    ambiguous: tuple[str, ...] = ()
    forbidden: dict[str, Any] = field(default_factory=dict)


INITIAL_CASES = (
    ParserCase(
        "A1",
        "direct",
        "I want black running shoes",
        {"intent": "buying", "category": "running shoes", "color": ["black"], "use_case": ["running"]},
    ),
    ParserCase(
        "A2",
        "direct",
        "Nike sneakers under $120",
        {"intent": "buying", "category": "sneakers", "brand": ["Nike"], "maximum_price": 120.0},
    ),
    ParserCase(
        "A3",
        "direct",
        "I need a blue cotton shirt",
        {"intent": "buying", "category": "shirts", "color": ["blue"], "material": ["cotton"]},
    ),
    ParserCase(
        "B1",
        "synonym",
        "I want comfy shoes",
        {"intent": "buying", "category": "shoes", "feature": ["comfort"]},
        ambiguous=("Whether the catalog should normalize 'comfy' to a comfort feature",),
    ),
    ParserCase(
        "B2",
        "synonym",
        "Something cushioned for running",
        {"feature": ["cushioned"], "use_case": ["running"]},
        ambiguous=("Buying versus browsing intent", "Product category"),
    ),
    ParserCase(
        "B3",
        "synonym",
        "dark coloured sneakers",
        {"category": "sneakers"},
        ambiguous=("The color family represented by 'dark coloured'",),
        forbidden={"color": ["black"]},
    ),
    ParserCase(
        "B4",
        "synonym",
        "cheap running shoes",
        {"category": "running shoes", "use_case": ["running"], "maximum_price": None},
        ambiguous=("Qualitative budget preference",),
    ),
    ParserCase(
        "C1",
        "natural",
        "Looking for something black from Nike that I can run in, preferably below 120 dollars",
        {"intent": "buying", "color": ["black"], "brand": ["Nike"], "use_case": ["running"], "maximum_price": 120.0},
        ambiguous=("Exact category",),
    ),
    ParserCase(
        "C2",
        "natural",
        "Need some sneakers for jogging, don't really care about brand",
        {"intent": "buying", "category": "sneakers", "use_case": ["running"], "brand": []},
    ),
    ParserCase(
        "C3",
        "natural",
        "Can you find me something comfortable for walking all day?",
        {"intent": "buying", "feature": ["comfort"], "use_case": ["walking"]},
        ambiguous=("Exact category",),
    ),
    ParserCase(
        "C4",
        "natural",
        "I'm going to the beach and need something suitable",
        {"intent": "browsing", "use_case": ["beach"]},
        ambiguous=("Exact category",),
    ),
    ParserCase(
        "D1",
        "multiple",
        "Black Nike running shoes under $120, size 10",
        {"intent": "buying", "category": "running shoes", "color": ["black"], "brand": ["Nike"], "use_case": ["running"], "maximum_price": 120.0, "size": ["10"]},
    ),
    ParserCase(
        "D2",
        "multiple",
        "Blue or black sneakers below $100",
        {"intent": "buying", "category": "sneakers", "color_any_of": ["blue", "black"], "maximum_price": 100.0},
    ),
    ParserCase(
        "D3",
        "multiple",
        "Leather shoes but not brown",
        {"intent": "buying", "category": "shoes", "material": ["leather"], "excluded_color": ["brown"]},
    ),
    ParserCase(
        "F1",
        "intent",
        "Show me some cool things for a beach holiday",
        {"intent": "browsing", "use_case": ["beach"]},
        ambiguous=("Exact category",),
    ),
    ParserCase(
        "F2",
        "intent",
        "I need black Nike running shoes below $120",
        {"intent": "buying", "category": "running shoes", "color": ["black"], "brand": ["Nike"], "use_case": ["running"], "maximum_price": 120.0},
    ),
    ParserCase(
        "F3",
        "intent",
        "I need shoes",
        {"intent": "unknown", "category": "shoes"},
        ambiguous=("Buying intent is plausible, but the product request is underspecified",),
    ),
)


MULTITURN_CASES = (
    {
        "case_id": "E1",
        "messages": ("I want running shoes", "Actually make that sandals"),
        "expected_after": {"category": "sandals", "intent": "intent_override"},
    },
    {
        "case_id": "E2",
        "messages": ("I want something black", "Actually colour doesn't matter anymore"),
        "expected_after": {"color": [], "intent": "intent_override"},
    },
    {
        "case_id": "E3",
        "messages": ("Budget is $150", "Changed my mind, keep it below $100"),
        "expected_after": {"maximum_price": 100.0, "intent": "intent_override"},
    },
)


DANGEROUS_PAIRS = (
    ("running", "walking"),
    ("black", "dark blue"),
    ("cotton", "polyester"),
    ("nike", "adidas"),
    ("comfort", "style"),
    ("sandals", "running shoes"),
)


END_TO_END_CASES = (
    ("F2", ("I need black Nike running shoes below $120",)),
    ("A3", ("I need a blue cotton shirt",)),
    ("C4", ("I'm going to the beach and need something suitable",)),
    ("D1", ("Black Nike running shoes under $120, size 10",)),
    ("E1", ("I want running shoes", "Actually make that sandals")),
    (
        "E2",
        ("I want something black", "Actually colour doesn't matter anymore"),
    ),
    (
        "E3",
        ("Budget is $150", "Changed my mind, keep it below $100"),
    ),
)


INTENT_GROUND_TRUTH = {
    "A1": "buying",
    "A2": "buying",
    "A3": "buying",
    "B1": "buying",
    "B3": "buying",
    "B4": "buying",
    "C1": "buying",
    "C2": "buying",
    "C3": "buying",
    "C4": "browsing",
    "D1": "buying",
    "D2": "buying",
    "D3": "buying",
    "F1": "browsing",
    "F2": "buying",
    "F3": "unknown",
}


def similarity_scores(message: str) -> dict[str, float]:
    return {
        intent: round(agent_module._fuzzy_intent_score(message, intent), 3)
        for intent in agent_module.INTENT_CUES
    }


def observe_initial(message: str) -> dict[str, Any]:
    parsed = parse_free_form_message(
        message, known_brands=("Nike", "Adidas", "Puma")
    )
    actual: dict[str, Any] = {
        "intent": parsed.intent,
        "category": parsed.category,
        "constraint": None,
        "color": [],
        "material": [],
        "brand": [],
        "feature": [],
        "use_case": [],
        "size": [],
        "maximum_price": parsed.maximum_price,
        "color_any_of": parsed.alternatives.get("color", []),
        "excluded_color": parsed.excluded.get("color", []),
    }
    for attribute, values in parsed.attributes.items():
        if attribute in actual:
            actual[attribute] = list(values)
    return actual


def case_result(case: ParserCase) -> dict[str, Any]:
    actual = observe_initial(case.message)
    mismatches = {
        key: {"expected": expected, "actual": actual.get(key)}
        for key, expected in case.expected.items()
        if actual.get(key) != expected
    }
    forbidden_matches = {
        key: value
        for key, value in case.forbidden.items()
        if actual.get(key) == value
    }
    return {
        "case_id": case.case_id,
        "group": case.group,
        "input": case.message,
        "expected": case.expected,
        "actual": actual,
        "similarity_scores": similarity_scores(case.message),
        "ambiguous": list(case.ambiguous),
        "mismatches": mismatches,
        "forbidden_matches": forbidden_matches,
        "pass": not mismatches and not forbidden_matches,
    }


def classify_at_thresholds(
    message: str, buying_threshold: float, non_buying_threshold: float
) -> str:
    normalized = agent_module._normalize_intent_text(message)
    for intent, scenario in (
        ("browsing", "browsing"),
        ("boundary", "boundary"),
        ("buying", "buying"),
    ):
        cue = agent_module._normalize_intent_text(agent_module.INTENT_CUES[intent])
        if cue in normalized:
            return scenario
    # Preserve the production fuzzy fallback order.
    if agent_module._fuzzy_intent_score(message, "buying") >= buying_threshold:
        return "buying"
    if agent_module._fuzzy_intent_score(message, "browsing") >= non_buying_threshold:
        return "browsing"
    if agent_module._fuzzy_intent_score(message, "boundary") >= non_buying_threshold:
        return "boundary"
    return "unknown"


def threshold_result(
    buying_threshold: float, non_buying_threshold: float
) -> dict[str, Any]:
    by_id = {case.case_id: case for case in INITIAL_CASES}
    true_matches = false_matches = missed_matches = 0
    rows = []
    for case_id, expected in INTENT_GROUND_TRUTH.items():
        predicted = classify_at_thresholds(
            by_id[case_id].message, buying_threshold, non_buying_threshold
        )
        if expected == "unknown":
            false_matches += int(predicted != "unknown")
        elif predicted == expected:
            true_matches += 1
        elif predicted == "unknown":
            missed_matches += 1
        else:
            false_matches += 1
            missed_matches += 1
        rows.append({"case_id": case_id, "expected": expected, "predicted": predicted})
    precision_denominator = true_matches + false_matches
    recall_denominator = true_matches + missed_matches
    precision = true_matches / precision_denominator if precision_denominator else 0.0
    recall = true_matches / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "buying_threshold": buying_threshold,
        "non_buying_threshold": non_buying_threshold,
        "true_matches": true_matches,
        "false_matches": false_matches,
        "missed_matches": missed_matches,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "rows": rows,
    }


def multi_turn_parser_result(case: dict[str, Any]) -> dict[str, Any]:
    first, second = case["messages"]
    first_actual = observe_initial(first)
    parsed_second = parse_free_form_message(
        second, known_brands=("Nike", "Adidas", "Puma")
    )
    second_actual = {
        "intent": "intent_override"
        if parsed_second.category
        or parsed_second.has_constraints
        or parsed_second.remove_attributes
        else first_actual["intent"],
        "replacement": parsed_second.category,
        "category": parsed_second.category or first_actual["category"],
        "color": []
        if "color" in parsed_second.remove_attributes
        else first_actual["color"],
        "maximum_price": parsed_second.maximum_price
        if parsed_second.maximum_price is not None
        else first_actual["maximum_price"],
    }
    expected = case["expected_after"]
    mismatches = {
        key: {"expected": value, "actual": second_actual.get(key)}
        for key, value in expected.items()
        if second_actual.get(key) != value
    }
    return {
        "case_id": case["case_id"],
        "input": list(case["messages"]),
        "expected": expected,
        "actual": second_actual,
        "similarity_scores": [similarity_scores(first), similarity_scores(second)],
        "mismatches": mismatches,
        "pass": not mismatches,
    }


def dangerous_pair_results() -> list[dict[str, Any]]:
    fuzz = agent_module.rapidfuzz_fuzz
    return [
        {
            "left": left,
            "right": right,
            "partial_ratio": None if fuzz is None else round(float(fuzz.partial_ratio(left, right)), 3),
            "used_by_production_attribute_parser": False,
        }
        for left, right in DANGEROUS_PAIRS
    ]


def end_to_end_results() -> list[dict[str, Any]]:
    """Exercise the public Agent.respond API without mutating evaluator data."""

    rows: list[dict[str, Any]] = []
    with Agent(PROJECT_ROOT / "data" / "catalog.jsonl") as shopping_agent:
        for case_id, messages in END_TO_END_CASES:
            session_id = f"free-form-{case_id}"
            shopping_agent.reset(session_id, {})
            turns = []
            for turn, message in enumerate(messages, start=1):
                response = shopping_agent.respond(
                    session_id, message, turn=turn, top_k=10
                )
                state = shopping_agent._sessions[session_id]
                _, active_mode = shopping_agent._active_model(state)
                turns.append(
                    {
                        "message": message,
                        "parsed_state": {
                            "scenario": state.scenario_state,
                            "coarse_category": state.coarse_category,
                            "known_constraints": copy.deepcopy(state.known_constraints),
                            "unindexed_values": sorted(state.unindexed_values),
                            "intent_epoch": state.intent_epoch,
                            "override_count": state.override_count,
                            "alternatives": copy.deepcopy(
                                state.alternative_constraints
                            ),
                            "excluded": copy.deepcopy(state.excluded_constraints),
                            "maximum_price": state.maximum_price,
                        },
                        "candidate_count": len(state.surviving_candidates),
                        "filters_applied": {
                            "hard": copy.deepcopy(state.hard_constraints),
                            "alternatives": copy.deepcopy(
                                state.alternative_constraints
                            ),
                            "excluded": copy.deepcopy(
                                state.excluded_constraints
                            ),
                            "maximum_price": state.maximum_price,
                        },
                        "ranking_mode": active_mode,
                        "ask_attribute": response["ask_attribute"],
                        "top_10": [
                            item["parent_asin"]
                            for item in response["recommendations"]
                        ],
                    }
                )
            rows.append({"case_id": case_id, "turns": turns})
    return rows


def main() -> int:
    cases = [case_result(case) for case in INITIAL_CASES]
    multi_turn = [multi_turn_parser_result(case) for case in MULTITURN_CASES]
    # The 70/75-90/95 band brackets the asymmetric production 80/85 profile.
    # Lower profiles show what would happen if the thresholds were reduced far
    # enough to catch these natural requests.
    thresholds = [
        threshold_result(buying, non_buying)
        for buying, non_buying in (
            (40.0, 45.0),
            (45.0, 50.0),
            (50.0, 55.0),
            (55.0, 60.0),
            (60.0, 65.0),
            (70.0, 75.0),
            (75.0, 80.0),
            (80.0, 85.0),
            (85.0, 90.0),
            (90.0, 95.0),
        )
    ]
    payload = {
        "rapidfuzz_available": agent_module.rapidfuzz_fuzz is not None,
        "production_thresholds": {
            "buying": agent_module.BUYING_FUZZY_THRESHOLD,
            "browsing_boundary": agent_module.NON_BUYING_FUZZY_THRESHOLD,
        },
        "threshold_scope": (
            "legacy evaluator-cue typo matching only; structured free-form hard "
            "fields use exact field-scoped rules"
        ),
        "cases": cases,
        "multi_turn": multi_turn,
        "dangerous_pairs": dangerous_pair_results(),
        "end_to_end": end_to_end_results(),
        "thresholds": thresholds,
        "summary": {
            "case_count": len(cases) + len(multi_turn),
            "passed": sum(row["pass"] for row in cases + multi_turn),
            "failed": sum(not row["pass"] for row in cases + multi_turn),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if payload["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
