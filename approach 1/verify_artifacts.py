from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPOSITORY_ROOT / "techjam-conversational-search"
EXPECTED_FROZEN = {
    PROJECT_ROOT / "data/public_set.jsonl": (
        "857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579"
    ),
    PROJECT_ROOT / "evaluator/local_evaluator.py": (
        "79a5ea06f9a1b8c5036f30efa85dc1f36b8f6b06eb8feb8f545dfa767bc45564"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_summary(path: Path) -> dict[str, object]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        return {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "model_type": metadata.get("model_type"),
            "product_count": connection.execute(
                "SELECT COUNT(*) FROM products"
            ).fetchone()[0],
            "context_feature_count": connection.execute(
                "SELECT COUNT(*) FROM context_features"
            ).fetchone()[0],
            "item_feature_count": connection.execute(
                "SELECT COUNT(*) FROM item_features"
            ).fetchone()[0],
            "cross_count": connection.execute(
                "SELECT COUNT(*) FROM cross_weights"
            ).fetchone()[0],
        }
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen inputs and FM artifacts")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("results") / "artifact_checksums.json",
    )
    args = parser.parse_args()
    frozen: dict[str, object] = {}
    for path, expected in EXPECTED_FROZEN.items():
        actual = sha256(path)
        frozen[str(path.relative_to(REPOSITORY_ROOT))] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "unchanged": actual == expected,
        }
    models = {
        name: model_summary(Path(__file__).with_name(filename))
        for name, filename in (
            ("linear", "linear_model.sqlite3"),
            ("fm", "fm_only_model.sqlite3"),
            ("hybrid", "fm_model.sqlite3"),
        )
    }
    report = {
        "all_frozen_inputs_unchanged": all(
            bool(value["unchanged"]) for value in frozen.values()
        ),
        "frozen_inputs": frozen,
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_frozen_inputs_unchanged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
