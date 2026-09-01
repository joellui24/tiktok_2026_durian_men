"""Metrics and retrieval diagnostics for the fixed free-form corpus.

Run development first and held-out only after analysis/tuning is frozen:

    py -3.13 -B tests/free_form_generalization_diagnostics.py --split dev
    py -3.13 -B tests/free_form_generalization_diagnostics.py --split heldout
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starter.agent import Agent  # noqa: E402
from starter.free_form_parser import parse_free_form_message  # noqa: E402
from tests.free_form_generalization_cases import CASES, CONVERSATIONS, QueryCase  # noqa: E402


KNOWN_BRANDS = (
    "Nike",
    "Adidas",
    "Puma",
    "ASICS",
    "Skechers",
    "Reebok",
    "Columbia",
    "Clarks",
    "Calvin Klein",
    "Cotton On",
    "Under Armour",
    "Coach",
    "Guess",
    "Gap",
)
FIELDS = (
    "intent",
    "category",
    "brand",
    "color",
    "material",
    "size",
    "maximum_price",
    "qualitative_budget",
    "feature",
    "use_case",
    "negation",
    "alternatives",
)


def _expected_slots(case: QueryCase) -> dict[str, set[str]]:
    slots = {field: set() for field in FIELDS}
    slots["intent"] = {case.intent}
    if case.category:
        slots["category"] = {case.category}
    for field in ("brand", "color", "material", "size", "feature", "use_case"):
        slots[field] = set(case.attributes.get(field, ()))
    if case.maximum_price is not None:
        slots["maximum_price"] = {f"{case.maximum_price:g}"}
    if case.qualitative_budget:
        slots["qualitative_budget"] = {case.qualitative_budget}
    slots["negation"] = {
        f"{attribute}:{value}"
        for attribute, values in case.excluded.items()
        for value in values
    }
    slots["alternatives"] = {
        f"{attribute}:{value}"
        for attribute, values in case.alternatives.items()
        for value in values
    }
    return slots


def _actual_slots(message: str) -> tuple[dict[str, set[str]], dict[str, Any]]:
    parsed = parse_free_form_message(message, known_brands=KNOWN_BRANDS)
    slots = {field: set() for field in FIELDS}
    slots["intent"] = {parsed.intent}
    if parsed.category:
        slots["category"] = {parsed.category}
    for field in ("brand", "color", "material", "size", "feature", "use_case"):
        slots[field] = set(parsed.attributes.get(field, ()))
    if parsed.maximum_price is not None:
        slots["maximum_price"] = {f"{parsed.maximum_price:g}"}
    if parsed.qualitative_budget:
        slots["qualitative_budget"] = {parsed.qualitative_budget}
    slots["negation"] = {
        f"{attribute}:{value}"
        for attribute, values in parsed.excluded.items()
        for value in values
    }
    slots["alternatives"] = {
        f"{attribute}:{value}"
        for attribute, values in parsed.alternatives.items()
        for value in values
    }
    detail = {
        "intent": parsed.intent,
        "category": parsed.category,
        "attributes": parsed.attributes,
        "alternatives": parsed.alternatives,
        "excluded": parsed.excluded,
        "remove_attributes": sorted(parsed.remove_attributes),
        "maximum_price": parsed.maximum_price,
        "qualitative_budget": parsed.qualitative_budget,
    }
    return slots, detail


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def parser_report(split: str) -> dict[str, Any]:
    selected = [case for case in CASES if split == "all" or case.split == split]
    counts = {
        field: {"tp": 0, "fp": 0, "fn": 0, "correct_queries": 0}
        for field in FIELDS
    }
    rows = []
    group_counts: dict[str, list[bool]] = defaultdict(list)
    for case in selected:
        expected = _expected_slots(case)
        actual, detail = _actual_slots(case.message)
        mismatches = {}
        for field in FIELDS:
            counts[field]["tp"] += len(expected[field] & actual[field])
            counts[field]["fp"] += len(actual[field] - expected[field])
            counts[field]["fn"] += len(expected[field] - actual[field])
            field_correct = expected[field] == actual[field]
            counts[field]["correct_queries"] += int(field_correct)
            if not field_correct:
                mismatches[field] = {
                    "expected": sorted(expected[field]),
                    "actual": sorted(actual[field]),
                }
        exact = not mismatches
        group_counts[case.group].append(exact)
        rows.append(
            {
                "case_id": case.case_id,
                "group": case.group,
                "message": case.message,
                "expected": {key: sorted(value) for key, value in expected.items()},
                "actual": detail,
                "mismatches": mismatches,
                "exact": exact,
            }
        )

    field_metrics = {}
    for field, values in counts.items():
        precision = _safe_ratio(values["tp"], values["tp"] + values["fp"])
        recall = _safe_ratio(values["tp"], values["tp"] + values["fn"])
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        field_metrics[field] = {
            **values,
            "field_accuracy": round(
                _safe_ratio(values["correct_queries"], len(selected)), 6
            ),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    exact_count = sum(row["exact"] for row in rows)
    return {
        "split": split,
        "case_count": len(selected),
        "exact_query_count": exact_count,
        "exact_query_accuracy": round(_safe_ratio(exact_count, len(selected)), 6),
        "field_metrics": field_metrics,
        "group_exact_accuracy": {
            group: round(_safe_ratio(sum(values), len(values)), 6)
            for group, values in sorted(group_counts.items())
        },
        "failures": [row for row in rows if not row["exact"]],
    }


def conversation_report(split: str) -> dict[str, Any]:
    selected = [
        case for case in CONVERSATIONS if split == "all" or case.split == split
    ]
    rows = []
    with Agent(PROJECT_ROOT / "data" / "catalog.jsonl") as agent:
        for case in selected:
            agent.reset(case.case_id, {})
            responses = []
            for turn, message in enumerate(case.messages, start=1):
                responses.append(
                    agent.respond(case.case_id, message, turn=turn, top_k=10)
                )
            state = agent._sessions[case.case_id]
            mismatches = {}
            if case.expected_category and state.coarse_category != case.expected_category:
                mismatches["category"] = {
                    "expected": case.expected_category,
                    "actual": state.coarse_category,
                }
            for attribute, expected in case.expected_attributes.items():
                actual = tuple(state.known_constraints.get(attribute, ()))
                if set(actual) != set(expected):
                    mismatches[attribute] = {
                        "expected": list(expected),
                        "actual": list(actual),
                    }
            for attribute in case.expected_removed:
                if attribute in state.known_constraints:
                    mismatches[f"removed:{attribute}"] = {
                        "expected": [],
                        "actual": state.known_constraints[attribute],
                    }
            if state.maximum_price != case.expected_maximum_price:
                mismatches["maximum_price"] = {
                    "expected": case.expected_maximum_price,
                    "actual": state.maximum_price,
                }
            if state.scenario_state != case.expected_intent:
                mismatches["intent"] = {
                    "expected": case.expected_intent,
                    "actual": state.scenario_state,
                }
            rows.append(
                {
                    "case_id": case.case_id,
                    "group": case.group,
                    "messages": list(case.messages),
                    "final_state": {
                        "intent": state.scenario_state,
                        "category": state.coarse_category,
                        "known_constraints": state.known_constraints,
                        "maximum_price": state.maximum_price,
                        "candidate_count": len(state.surviving_candidates),
                    },
                    "top_10": [
                        item["parent_asin"]
                        for item in responses[-1]["recommendations"]
                    ],
                    "mismatches": mismatches,
                    "pass": not mismatches,
                }
            )
    return {
        "case_count": len(rows),
        "passed": sum(row["pass"] for row in rows),
        "failed": sum(not row["pass"] for row in rows),
        "rows": rows,
    }


E2E_IDS = (
    "NAT01",
    "NAT02",
    "NAT03",
    "STR02",
    "USE03",
    "USE04",
    "NEG01",
    "NEG02",
    "ALT02",
    "BRW01",
    "BRW02",
    "BRW04",
    "BUY01",
    "BUY05",
    "BUY07",
    "SIZ02",
    "ADV02",
    "UNS02",
    "VAR09",
    "VAR10",
)


def end_to_end_report() -> list[dict[str, Any]]:
    by_id = {case.case_id: case for case in CASES}
    rows = []
    with Agent(PROJECT_ROOT / "data" / "catalog.jsonl") as agent:
        for case_id in E2E_IDS:
            case = by_id[case_id]
            parsed = agent._parse_free_form(case.message)
            _, base_candidates = agent._free_form_candidates(parsed.category)
            agent.reset(case_id, {})
            response = agent.respond(case_id, case.message, turn=1, top_k=10)
            state = agent._sessions[case_id]
            _, mode = agent._active_model(state)
            rows.append(
                {
                    "case_id": case_id,
                    "message": case.message,
                    "parser_exact": not bool(
                        {
                            field: True
                            for field in FIELDS
                            if _expected_slots(case)[field]
                            != _actual_slots(case.message)[0][field]
                        }
                    ),
                    "parsed_fields": _actual_slots(case.message)[1],
                    "candidate_count_before_filters": len(base_candidates),
                    "candidate_count_after_filters": len(state.surviving_candidates),
                    "hard_filters": state.hard_constraints,
                    "alternative_filters": state.alternative_constraints,
                    "excluded_filters": state.excluded_constraints,
                    "maximum_price": state.maximum_price,
                    "ranking_mode": mode,
                    "top_10": [
                        item["parent_asin"]
                        for item in response["recommendations"]
                    ],
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "heldout", "all"), required=True)
    parser.add_argument("--end-to-end", action="store_true")
    args = parser.parse_args()
    payload = {
        "parser": parser_report(args.split),
        "conversations": conversation_report(args.split),
    }
    if args.end_to_end:
        payload["end_to_end"] = end_to_end_report()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
