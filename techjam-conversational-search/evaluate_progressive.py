from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator import local_evaluator as frozen_evaluator
from starter.agent import Agent


def _scored_summary(summary: dict) -> dict:
    """Add the requested labels and derived scores without changing scoring."""

    mttc = summary.get("mttc")
    efficiency = (
        0.0
        if mttc is None
        else max(0.0, min(1.0, (11.0 - float(mttc)) / 10.0))
    )
    accuracy = float(summary.get("hit_rate_at_10", 0.0))
    mrr = float(summary.get("mrr", 0.0))
    technical_score = 0.50 * accuracy + 0.30 * mrr + 0.20 * efficiency
    return {
        **summary,
        "accuracy": round(accuracy, 6),
        "efficiency": round(efficiency, 6),
        "technical_score": round(technical_score, 6),
    }


def _report(result: dict, *, cohort: str, intent_override_note: str) -> dict:
    overall = _scored_summary(
        {
            key: result[key]
            for key in ("sample_count", "hit_rate_at_10", "mrr", "mttc")
        }
    )
    return {
        "cohort": cohort,
        "intent_override_implemented": True,
        "note": intent_override_note,
        **overall,
        "recommended_technical_score": overall["technical_score"],
        "reported_token_usage": result["reported_token_usage"],
        "scenario_metrics": {
            name: _scored_summary(summary)
            for name, summary in result["scenario_metrics"].items()
        },
        "sessions": result["sessions"],
    }


def evaluate_cohorts(
    *,
    catalog_path: str | Path = "data/catalog.jsonl",
    dataset_path: str | Path = "data/public_set.jsonl",
) -> tuple[dict, dict]:
    """Evaluate the in-memory no-override subset and unchanged official set."""

    samples = frozen_evaluator.load_jsonl(dataset_path)
    no_override_samples = [
        sample for sample in samples if sample.get("scenario_type") != "intent_override"
    ]
    catalog_ids, categories, products = frozen_evaluator.catalog_index(catalog_path)
    with Agent(catalog_path) as agent:
        no_override_result = frozen_evaluator.evaluate(
            agent, no_override_samples, catalog_ids, categories, products
        )
        official_result = frozen_evaluator.evaluate(
            agent, samples, catalog_ids, categories, products
        )

    return (
        _report(
            no_override_result,
            cohort="buying_browsing_boundary_170",
            intent_override_note=(
                "Intent Override sessions were filtered in memory; public_set.jsonl "
                "was not modified. The FM agent also supports override state replacement."
            ),
        ),
        _report(
            official_result,
            cohort="official_public_200",
            intent_override_note=(
                "This is the unchanged official 200-session evaluation, including "
                "implemented Intent Override state replacement and reranking."
            ),
        ),
    )


def _print_summary(report: dict) -> None:
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "sessions"},
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the progressive agent on both requested cohorts"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--no-override-output", default="results_no_override.json")
    parser.add_argument("--official-output", default="results_official.json")
    args = parser.parse_args()

    no_override, official = evaluate_cohorts(
        catalog_path=args.catalog, dataset_path=args.dataset
    )
    for path, report in (
        (Path(args.no_override_output), no_override),
        (Path(args.official_output), official),
    ):
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    _print_summary(no_override)
    _print_summary(official)


if __name__ == "__main__":
    main()
