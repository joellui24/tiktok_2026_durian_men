from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPOSITORY_ROOT / "techjam-conversational-search"
sys.path.insert(0, str(PROJECT_ROOT))

from evaluator import local_evaluator as evaluator  # noqa: E402
from starter.agent import Agent  # noqa: E402


def rank_of(scores: dict[str, float], target: str) -> int | None:
    target_score = scores.get(target)
    if target_score is None:
        return None
    return 1 + sum(
        score > target_score or (score == target_score and parent_asin < target)
        for parent_asin, score in scores.items()
        if parent_asin != target
    )


def catalog_cross_statistics(model_path: Path) -> dict[tuple[str, str], dict[str, float]]:
    connection = sqlite3.connect(f"file:{model_path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT cf.field, it.field, cw.positive_support, cw.negative_support, cw.weight
            FROM cross_weights cw
            JOIN context_features cf ON cf.feature_id = cw.context_feature_id
            JOIN item_features it ON it.feature_id = cw.item_feature_id
            """
        ).fetchall()
    finally:
        connection.close()
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {
            "cross_count": 0,
            "positive_support": 0,
            "comparable_negative_support": 0,
            "sum_absolute_weight": 0.0,
            "max_absolute_weight": 0.0,
            "positive_weight_count": 0,
            "negative_weight_count": 0,
        }
    )
    for context_field, item_field, positive, negative, weight in rows:
        stats = grouped[(str(context_field), str(item_field))]
        stats["cross_count"] += 1
        stats["positive_support"] += int(positive)
        stats["comparable_negative_support"] += int(negative)
        absolute = abs(float(weight))
        stats["sum_absolute_weight"] += absolute
        stats["max_absolute_weight"] = max(stats["max_absolute_weight"], absolute)
        stats["positive_weight_count"] += int(float(weight) > 0)
        stats["negative_weight_count"] += int(float(weight) < 0)
    return dict(grouped)


def analyze_ranking_ablation(
    catalog_path: Path,
    dataset_path: Path,
) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, object]]:
    samples = evaluator.load_jsonl(dataset_path)
    catalog_ids, categories, products = evaluator.catalog_index(catalog_path)
    accumulator: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {
            "active_state_count": 0,
            "baseline_reciprocal_rank_sum": 0.0,
            "ablated_reciprocal_rank_sum": 0.0,
            "baseline_hit_at_10_sum": 0,
            "ablated_hit_at_10_sum": 0,
            "mean_absolute_active_contribution_sum": 0.0,
        }
    )
    eligible_state_count = 0
    target_retained_state_count = 0
    with Agent(catalog_path, ranking_mode="hybrid") as agent:
        if agent.model is None:
            raise RuntimeError(agent.model_error or "FM model is unavailable")
        for sample in samples:
            sample_id = str(sample["sample_id"])
            session_id = f"interaction_{sample_id}"
            target = str(sample["ground_truth"]["parent_asin"])
            scenario = str(sample["scenario_type"])
            agent.reset(session_id, sample["user_profile"])
            card, behavior = evaluator.materialize_hidden_fields(sample, products)
            effective = {**sample, "intent_card": card, "behavior": behavior}
            disclosed: set[str] = set()
            boundary_used = False
            override_applied = scenario != "intent_override"
            user_message = evaluator.initial_message(
                effective,
                evaluator.coarse_category(categories.get(target, [])),
                disclosed,
            )
            for turn in range(1, evaluator.MAX_TURNS + 1):
                response = agent.respond(session_id, user_message, turn, evaluator.TOP_K)
                state = agent._sessions[session_id]
                if override_applied:
                    eligible_state_count += 1
                    if target in state.surviving_candidates:
                        target_retained_state_count += 1
                        context = agent._context_features(state, turn)
                        scores = agent.model.score_many(
                            state.surviving_candidates,
                            context,
                            mode="hybrid",
                        )
                        baseline_rank = rank_of(scores, target)
                        if baseline_rank is not None:
                            contributions = agent.model.cross_contributions(
                                state.surviving_candidates, context
                            )
                            for field_pair, by_product in contributions.items():
                                ablated_scores = {
                                    parent_asin: score - by_product.get(parent_asin, 0.0)
                                    for parent_asin, score in scores.items()
                                }
                                ablated_rank = rank_of(ablated_scores, target)
                                if ablated_rank is None:
                                    continue
                                stats = accumulator[field_pair]
                                stats["active_state_count"] += 1
                                stats["baseline_reciprocal_rank_sum"] += 1.0 / baseline_rank
                                stats["ablated_reciprocal_rank_sum"] += 1.0 / ablated_rank
                                stats["baseline_hit_at_10_sum"] += int(baseline_rank <= 10)
                                stats["ablated_hit_at_10_sum"] += int(ablated_rank <= 10)
                                stats["mean_absolute_active_contribution_sum"] += (
                                    sum(abs(value) for value in by_product.values())
                                    / max(1, len(by_product))
                                )

                if turn == evaluator.MAX_TURNS:
                    break
                override = effective.get("behavior", {}).get("override") or {}
                if not override_applied and turn + 1 == int(override.get("turn", 3)):
                    override_applied = True
                    new_value = str(override.get("new_value", ""))
                    if new_value:
                        disclosed.add(new_value)
                    user_message = str(override.get("message", ""))
                else:
                    user_message, boundary_used = evaluator.customer_reply(
                        effective,
                        response.get("ask_attribute"),
                        disclosed,
                        boundary_used,
                    )
    return dict(accumulator), {
        "eligible_scoring_states": eligible_state_count,
        "target_retained_states": target_retained_state_count,
    }


def combine_rows(
    catalog: dict[tuple[str, str], dict[str, float]],
    ranking: dict[tuple[str, str], dict[str, float]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for field_pair in sorted(set(catalog) | set(ranking)):
        catalog_stats = catalog.get(field_pair, {})
        ranking_stats = ranking.get(field_pair, {})
        count = int(ranking_stats.get("active_state_count", 0))
        baseline_mrr = (
            float(ranking_stats.get("baseline_reciprocal_rank_sum", 0.0)) / count
            if count
            else 0.0
        )
        ablated_mrr = (
            float(ranking_stats.get("ablated_reciprocal_rank_sum", 0.0)) / count
            if count
            else 0.0
        )
        baseline_hit = (
            float(ranking_stats.get("baseline_hit_at_10_sum", 0.0)) / count
            if count
            else 0.0
        )
        ablated_hit = (
            float(ranking_stats.get("ablated_hit_at_10_sum", 0.0)) / count
            if count
            else 0.0
        )
        cross_count = int(catalog_stats.get("cross_count", 0))
        result.append(
            {
                "context_field": field_pair[0],
                "item_field": field_pair[1],
                "field_pair": f"{field_pair[0]}×{field_pair[1]}",
                "cross_count": cross_count,
                "positive_support": int(catalog_stats.get("positive_support", 0)),
                "comparable_negative_support": int(
                    catalog_stats.get("comparable_negative_support", 0)
                ),
                "mean_absolute_weight": (
                    float(catalog_stats.get("sum_absolute_weight", 0.0)) / cross_count
                    if cross_count
                    else 0.0
                ),
                "max_absolute_weight": float(
                    catalog_stats.get("max_absolute_weight", 0.0)
                ),
                "positive_weight_count": int(
                    catalog_stats.get("positive_weight_count", 0)
                ),
                "negative_weight_count": int(
                    catalog_stats.get("negative_weight_count", 0)
                ),
                "active_public_state_count": count,
                "mean_absolute_active_contribution": (
                    float(
                        ranking_stats.get(
                            "mean_absolute_active_contribution_sum", 0.0
                        )
                    )
                    / count
                    if count
                    else 0.0
                ),
                "baseline_state_mrr": baseline_mrr,
                "ablated_state_mrr": ablated_mrr,
                "mrr_importance": baseline_mrr - ablated_mrr,
                "baseline_state_hit_at_10": baseline_hit,
                "ablated_state_hit_at_10": ablated_hit,
                "hit_at_10_importance": baseline_hit - ablated_hit,
            }
        )
    result.sort(
        key=lambda row: (
            -abs(float(row["mrr_importance"])),
            str(row["field_pair"]),
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure FM interaction importance")
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data/catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/public_set.jsonl")
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("fm_model.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results") / "field_pair_importance.csv")
    args = parser.parse_args()

    catalog = catalog_cross_statistics(args.model)
    ranking, summary = analyze_ranking_ablation(args.catalog, args.dataset)
    rows = combine_rows(catalog, ranking)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({**summary, "field_pair_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
