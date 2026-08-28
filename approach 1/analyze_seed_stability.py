from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def load_audits(paths: list[Path]) -> dict[tuple[str, str], list[float]]:
    result: dict[tuple[str, str], list[float]] = defaultdict(list)
    for seed_index, path in enumerate(paths):
        current: dict[tuple[str, str], float] = {}
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                current[(row["context_feature"], row["item_feature"])] = float(
                    row["learned_weight"]
                )
        for relationship in list(result):
            result[relationship].append(current.get(relationship, 0.0))
        for relationship, weight in current.items():
            if relationship not in result:
                result[relationship] = [0.0] * seed_index + [weight]
    return result


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze learned-cross seed stability")
    parser.add_argument("audits", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("results") / "cross_seed_stability.csv",
    )
    parser.add_argument(
        "--field-output",
        type=Path,
        default=Path(__file__).with_name("results") / "field_pair_seed_stability.csv",
    )
    args = parser.parse_args()
    relationships = load_audits(args.audits)
    rows: list[dict[str, object]] = []
    by_field: dict[str, list[dict[str, object]]] = defaultdict(list)
    for (context_feature, item_feature), weights in relationships.items():
        signs = {1 if value > 0 else -1 if value < 0 else 0 for value in weights}
        field_pair = (
            f"{context_feature.split(':', 1)[1].split('=', 1)[0]}×"
            f"{item_feature.split(':', 1)[1].split('=', 1)[0]}"
        )
        row = {
            "context_feature": context_feature,
            "item_feature": item_feature,
            "field_pair": field_pair,
            "seed_count": len(weights),
            "mean_weight": statistics.fmean(weights),
            "weight_stddev": statistics.pstdev(weights),
            "minimum_weight": min(weights),
            "maximum_weight": max(weights),
            "positive_seed_fraction": sum(value > 0 for value in weights) / len(weights),
            "same_nonzero_sign_all_seeds": len(signs) == 1 and 0 not in signs,
        }
        rows.append(row)
        by_field[field_pair].append(row)
    rows.sort(
        key=lambda row: (-abs(float(row["mean_weight"])), str(row["context_feature"]), str(row["item_feature"]))
    )
    field_rows: list[dict[str, object]] = []
    for field_pair, values in sorted(by_field.items()):
        field_rows.append(
            {
                "field_pair": field_pair,
                "cross_count": len(values),
                "same_sign_cross_count": sum(
                    row["same_nonzero_sign_all_seeds"] is True for row in values
                ),
                "same_sign_fraction": sum(
                    row["same_nonzero_sign_all_seeds"] is True for row in values
                )
                / len(values),
                "mean_absolute_weight": statistics.fmean(
                    abs(float(row["mean_weight"])) for row in values
                ),
                "mean_seed_stddev": statistics.fmean(
                    float(row["weight_stddev"]) for row in values
                ),
            }
        )
    write_rows(args.output, rows)
    write_rows(args.field_output, field_rows)
    print(
        f"Wrote {len(rows)} crosses and {len(field_rows)} field pairs "
        f"from {len(args.audits)} seeds"
    )


if __name__ == "__main__":
    main()
