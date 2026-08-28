from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPOSITORY_ROOT / "techjam-conversational-search"
sys.path.insert(0, str(PROJECT_ROOT))

from evaluator import local_evaluator as evaluator  # noqa: E402
from starter.agent import Agent  # noqa: E402


def analyze_turn_10(
    catalog_path: Path,
    dataset_path: Path,
) -> list[dict[str, object]]:
    """Simulate every public case through turn 10 and record survivor counts.

    The official evaluator stops as soon as it finds the target. This analysis
    deliberately continues those sessions so that every sample has a comparable
    hypothetical turn-10 candidate count. ``actually_reached_turn_10`` records
    whether the unmodified evaluator would really have reached that turn.
    """

    samples = evaluator.load_jsonl(dataset_path)
    catalog_ids, categories, products = evaluator.catalog_index(catalog_path)
    records: list[dict[str, object]] = []

    with Agent(catalog_path) as agent:
        for sample in samples:
            sample_id = str(sample["sample_id"])
            scenario = str(sample["scenario_type"])
            session_id = f"turn10_{sample_id}"
            target = str(sample["ground_truth"]["parent_asin"])
            agent.reset(session_id, sample["user_profile"])

            intent_card, behavior = evaluator.materialize_hidden_fields(
                sample, products
            )
            effective_sample = {
                **sample,
                "intent_card": intent_card,
                "behavior": behavior,
            }
            disclosed: set[str] = set()
            boundary_used = False
            override_applied = scenario != "intent_override"
            first_hit_turn: int | None = None
            user_message = evaluator.initial_message(
                effective_sample,
                evaluator.coarse_category(categories.get(target, [])),
                disclosed,
            )

            for turn in range(1, evaluator.MAX_TURNS + 1):
                response = agent.respond(
                    session_id, user_message, turn, evaluator.TOP_K
                )
                ranked = evaluator.normalize_recommendations(
                    response.get("recommendations"), catalog_ids
                )
                if first_hit_turn is None and override_applied and target in ranked:
                    first_hit_turn = turn

                if turn == evaluator.MAX_TURNS:
                    state = agent._sessions[session_id]
                    survivor_count = len(state.surviving_candidates)
                    recommendation_count = len(ranked)
                    break

                override = effective_sample.get("behavior", {}).get("override") or {}
                if (
                    not override_applied
                    and turn + 1 == int(override.get("turn", 3))
                ):
                    override_applied = True
                    new_value = str(override.get("new_value", ""))
                    if new_value:
                        disclosed.add(new_value)
                    user_message = str(
                        override.get(
                            "message",
                            "Actually, please ignore my earlier preference.",
                        )
                    )
                else:
                    user_message, boundary_used = evaluator.customer_reply(
                        effective_sample,
                        response.get("ask_attribute"),
                        disclosed,
                        boundary_used,
                    )

            actually_reached = (
                first_hit_turn is None or first_hit_turn == evaluator.MAX_TURNS
            )
            records.append(
                {
                    "sample_id": sample_id,
                    "scenario_type": scenario,
                    "in_supported_170": scenario != "intent_override",
                    "surviving_parent_asins_turn_10": survivor_count,
                    "recommendations_returned_turn_10": recommendation_count,
                    "exactly_10_survivors_turn_10": survivor_count == 10,
                    "first_hit_turn": first_hit_turn,
                    "actually_reached_turn_10": actually_reached,
                }
            )

    return records


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    cohorts = {
        "supported_170": [row for row in records if row["in_supported_170"]],
        "official_200": records,
    }
    for cohort, rows in cohorts.items():
        result[cohort] = {
            "sample_count": len(rows),
            "exactly_10_survivors": sum(
                row["exactly_10_survivors_turn_10"] is True for row in rows
            ),
            "exactly_10_survivor_sample_ids": [
                row["sample_id"]
                for row in rows
                if row["exactly_10_survivors_turn_10"] is True
            ],
            "at_most_10_survivors": sum(
                int(row["surviving_parent_asins_turn_10"]) <= 10 for row in rows
            ),
            "more_than_10_survivors": sum(
                int(row["surviving_parent_asins_turn_10"]) > 10 for row in rows
            ),
            "exactly_10_recommendations": sum(
                int(row["recommendations_returned_turn_10"]) == 10 for row in rows
            ),
        }

    reached = [row for row in records if row["actually_reached_turn_10"] is True]
    result["actually_reached_turn_10"] = {
        "sample_count": len(reached),
        "exactly_10_survivors": sum(
            row["exactly_10_survivors_turn_10"] is True for row in reached
        ),
        "survivor_counts": {
            str(row["sample_id"]): row["surviving_parent_asins_turn_10"]
            for row in reached
        },
    }
    result["exactly_10_survivors_by_scenario"] = dict(
        sorted(
            Counter(
                str(row["scenario_type"])
                for row in records
                if row["exactly_10_survivors_turn_10"] is True
            ).items()
        )
    )
    return result


def write_csv(records: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count surviving parent_asin values at turn 10"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "data" / "catalog.jsonl",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "public_set.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("turn10_results.csv"),
    )
    args = parser.parse_args()

    records = analyze_turn_10(args.catalog, args.dataset)
    write_csv(records, args.output)
    print(json.dumps(summarize(records), indent=2))
    print(f"Wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
