"""Run this submission against an external copy of the official public kit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from agent import Agent  # noqa: E402


def _load_official_evaluator(evaluator_path: Path) -> types.ModuleType:
    """Load the unchanged evaluator while routing its Agent import here."""

    starter_package = types.ModuleType("starter")
    starter_package.__path__ = []  # type: ignore[attr-defined]
    starter_agent = types.ModuleType("starter.agent")
    starter_agent.Agent = Agent  # type: ignore[attr-defined]
    sys.modules["starter"] = starter_package
    sys.modules["starter.agent"] = starter_agent

    specification = importlib.util.spec_from_file_location(
        "official_local_evaluator", evaluator_path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not load official evaluator: {evaluator_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the submission with an external official participant kit"
    )
    parser.add_argument(
        "--kit-root",
        required=True,
        type=Path,
        help="official techjam-conversational-search participant-kit directory",
    )
    parser.add_argument("--output", type=Path, default=Path("public-results.json"))
    args = parser.parse_args()

    kit_root = args.kit_root.expanduser().resolve()
    evaluator_path = kit_root / "evaluator" / "local_evaluator.py"
    catalog_path = kit_root / "data" / "catalog.jsonl"
    dataset_path = kit_root / "data" / "public_set.jsonl"
    required = (evaluator_path, catalog_path, dataset_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing official-kit files: " + ", ".join(missing))

    evaluator = _load_official_evaluator(evaluator_path)
    samples = evaluator.load_jsonl(dataset_path)
    catalog_ids, categories, products = evaluator.catalog_index(catalog_path)
    with Agent(catalog_path) as shopping_agent:
        result = evaluator.evaluate(
            shopping_agent, samples, catalog_ids, categories, products
        )

    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "sessions"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
