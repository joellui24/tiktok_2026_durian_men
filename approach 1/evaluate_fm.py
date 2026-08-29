from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPOSITORY_ROOT / "techjam-conversational-search"
sys.path.insert(0, str(PROJECT_ROOT))

from evaluator import local_evaluator as evaluator  # noqa: E402
from starter.agent import Agent  # noqa: E402


def scored_summary(sessions: list[dict]) -> dict[str, object]:
    summary = evaluator.metric_summary(sessions)
    accuracy = float(summary["hit_rate_at_10"])
    mrr = float(summary["mrr"])
    mttc = summary["mttc"]
    efficiency = (
        0.0
        if mttc is None
        else max(0.0, min(1.0, (11.0 - float(mttc)) / 10.0))
    )
    return {
        **summary,
        "correct_answers": sum(int(row["hit"]) for row in sessions),
        "accuracy": round(accuracy, 6),
        "efficiency": round(efficiency, 6),
        "technical_score": round(
            0.50 * accuracy + 0.30 * mrr + 0.20 * efficiency, 6
        ),
    }


def simulate(
    agent: Agent,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    *,
    full_horizon: bool,
) -> tuple[dict, list[dict]]:
    sessions: list[dict] = []
    traces: list[dict] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        session_id = f"fm_{'full' if full_horizon else 'official'}_{sample_id}"
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
        hit_turn: int | None = None
        best_rank: int | None = None
        trace: dict[str, object] = {
            "sample_id": sample_id,
            "scenario_type": scenario,
            "target_parent_asin": target,
        }

        for turn in range(1, evaluator.MAX_TURNS + 1):
            response = agent.respond(session_id, user_message, turn, evaluator.TOP_K)
            ranked = evaluator.normalize_recommendations(
                response.get("recommendations"), catalog_ids
            )
            state = agent._sessions[session_id]
            trace[f"candidate_count_turn_{turn}"] = len(state.surviving_candidates)
            trace[f"recommendation_count_turn_{turn}"] = len(ranked)
            trace[f"ask_attribute_turn_{turn}"] = response.get("ask_attribute") or ""
            trace[f"recommendations_turn_{turn}"] = ";".join(ranked)
            trace[f"target_survives_turn_{turn}"] = target in state.surviving_candidates
            trace[f"intent_epoch_turn_{turn}"] = state.intent_epoch

            if hit_turn is None and override_applied and target in ranked:
                hit_turn = turn
                best_rank = ranked.index(target) + 1
                if not full_horizon:
                    break
            if turn == evaluator.MAX_TURNS:
                break

            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(
                    override.get(
                        "message", "Actually, please ignore my earlier preference."
                    )
                )
            else:
                user_message, boundary_used = evaluator.customer_reply(
                    effective,
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )

        session = {
            "sample_id": sample_id,
            "scenario_type": scenario,
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        }
        sessions.append(session)
        final_count = int(trace.get("candidate_count_turn_10", len(state.surviving_candidates)))
        trace.update(
            {
                "hit": hit_turn is not None,
                "first_hit_turn": hit_turn or "",
                "best_rank": best_rank or "",
                "actually_reached_turn_10": hit_turn is None or hit_turn == 10,
                "surviving_parent_asins_turn_10": final_count,
                "exactly_10_survivors_turn_10": final_count == 10,
                "at_most_10_survivors_turn_10": final_count <= 10,
            }
        )
        traces.append(trace)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in sessions:
        grouped[str(row["scenario_type"])].append(row)
    overall = scored_summary(sessions)
    return (
        {
            **overall,
            "scenario_metrics": {
                scenario: scored_summary(rows)
                for scenario, rows in sorted(grouped.items())
            },
            "sessions": sessions,
        },
        traces,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ablation_rows(
    samples: list[dict],
    catalog_path: Path,
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    model_paths: dict[str, Path],
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    session_rows: list[dict] = []
    for mode in ("linear", "fm", "hybrid"):
        with Agent(
            catalog_path,
            model_path=model_paths[mode],
            ranking_mode="hybrid",
        ) as agent:
            result, _ = simulate(
                agent,
                samples,
                catalog_ids,
                categories,
                products,
                full_horizon=False,
            )
        for session in result["sessions"]:
            hit_turn = session["first_hit_turn"]
            efficiency = (
                0.0
                if hit_turn is None
                else max(0.0, min(1.0, (11.0 - float(hit_turn)) / 10.0))
            )
            session_rows.append(
                {
                    "model": mode,
                    **session,
                    "efficiency": efficiency,
                    "technical_score_contribution": (
                        0.50 * int(session["hit"])
                        + 0.30 * float(session["reciprocal_rank"])
                        + 0.20 * efficiency
                    ),
                }
            )
        for scenario, summary in [("overall", result), *result["scenario_metrics"].items()]:
            rows.append(
                {
                    "model": mode,
                    "scenario": scenario,
                    "sample_count": summary["sample_count"],
                    "correct_answers": summary["correct_answers"],
                    "accuracy": summary["accuracy"],
                    "mrr": summary["mrr"],
                    "mttc": summary["mttc"],
                    "efficiency": summary["efficiency"],
                    "technical_score": summary["technical_score"],
                }
            )
    return rows, session_rows


def bootstrap_rows(session_rows: list[dict], replicates: int = 10_000) -> list[dict]:
    by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in session_rows:
        by_model[str(row["model"])][str(row["sample_id"])] = row
    comparisons = (("fm", "linear"), ("hybrid", "fm"), ("hybrid", "linear"))
    metrics = {
        "accuracy": lambda row: float(bool(row["hit"])),
        "mrr": lambda row: float(row["reciprocal_rank"]),
        "efficiency": lambda row: float(row["efficiency"]),
        "technical_score": lambda row: float(row["technical_score_contribution"]),
    }
    rng = random.Random(2026)
    output: list[dict] = []
    for candidate, baseline in comparisons:
        sample_ids = sorted(set(by_model[candidate]) & set(by_model[baseline]))
        for metric, getter in metrics.items():
            deltas = [
                getter(by_model[candidate][sample_id])
                - getter(by_model[baseline][sample_id])
                for sample_id in sample_ids
            ]
            observed = sum(deltas) / len(deltas)
            draws = sorted(
                sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
                for _ in range(replicates)
            )
            output.append(
                {
                    "comparison": f"{candidate}_minus_{baseline}",
                    "metric": metric,
                    "sample_count": len(deltas),
                    "observed_delta": observed,
                    "bootstrap_replicates": replicates,
                    "ci_95_lower": draws[int(0.025 * replicates)],
                    "ci_95_upper": draws[int(0.975 * replicates)],
                    "ci_excludes_zero": (
                        draws[int(0.025 * replicates)] > 0
                        or draws[int(0.975 * replicates)] < 0
                    ),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Approach 1 FM")
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data/catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/public_set.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument(
        "--linear-model", type=Path, default=Path(__file__).with_name("linear_model.sqlite3")
    )
    parser.add_argument(
        "--fm-model", type=Path, default=Path(__file__).with_name("fm_only_model.sqlite3")
    )
    parser.add_argument(
        "--hybrid-model", type=Path, default=Path(__file__).with_name("fm_model.sqlite3")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = evaluator.load_jsonl(args.dataset)
    supported = [row for row in samples if row.get("scenario_type") != "intent_override"]
    catalog_ids, categories, products = evaluator.catalog_index(args.catalog)

    with Agent(args.catalog, model_path=args.hybrid_model, ranking_mode="hybrid") as agent:
        supported_result, supported_traces = simulate(
            agent,
            supported,
            catalog_ids,
            categories,
            products,
            full_horizon=False,
        )
        official_result, official_traces = simulate(
            agent,
            samples,
            catalog_ids,
            categories,
            products,
            full_horizon=False,
        )
        full_result, full_traces = simulate(
            agent,
            samples,
            catalog_ids,
            categories,
            products,
            full_horizon=True,
        )

    supported_result["cohort"] = "buying_browsing_boundary_170"
    official_result["cohort"] = "official_public_200"
    official_result["intent_override_implemented"] = True
    full_result["cohort"] = "official_public_200_full_horizon_diagnostic"
    for name, payload in (
        ("non_override_170.json", supported_result),
        ("official_200.json", official_result),
        ("full_horizon_200.json", full_result),
    ):
        (args.output_dir / name).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    write_csv(args.output_dir / "non_override_170.csv", supported_traces)
    write_csv(args.output_dir / "official_200.csv", official_traces)
    write_csv(args.output_dir / "full_horizon_200.csv", full_traces)

    ablations, ablation_sessions = ablation_rows(
        samples,
        args.catalog,
        catalog_ids,
        categories,
        products,
        {
            "linear": args.linear_model,
            "fm": args.fm_model,
            "hybrid": args.hybrid_model,
        },
    )
    write_csv(args.output_dir / "model_ablation.csv", ablations)
    write_csv(args.output_dir / "model_ablation_sessions.csv", ablation_sessions)
    write_csv(
        args.output_dir / "model_ablation_bootstrap.csv",
        bootstrap_rows(ablation_sessions),
    )

    summary = {
        "non_override_170": {
            key: value for key, value in supported_result.items()
            if key not in {"sessions", "scenario_metrics"}
        },
        "official_200": {
            key: value for key, value in official_result.items()
            if key not in {"sessions", "scenario_metrics"}
        },
        "full_horizon_turn_10": {
            "exactly_10": sum(
                row["exactly_10_survivors_turn_10"] is True for row in full_traces
            ),
            "at_most_10": sum(
                row["at_most_10_survivors_turn_10"] is True for row in full_traces
            ),
            "more_than_10": sum(
                int(row["surviving_parent_asins_turn_10"]) > 10
                for row in full_traces
            ),
            "naturally_reached_turn_10": sum(
                row["actually_reached_turn_10"] is True for row in full_traces
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
