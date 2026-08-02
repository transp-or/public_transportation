"""Illustrate gravity estimation, model relaxation, and grouped holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docs.source.examples.simple_gravity_workflow import (
    run_simple_gravity_workflow,
)

EXAMPLE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).parent / "results/gravity_estimation_summary.json"
DEFAULT_CACHE = Path(__file__).parent / "results/gravity_operator_cache"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maximum-iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--operator-cache", type=Path, default=DEFAULT_CACHE)
    arguments = parser.parse_args()
    report = run_simple_gravity_workflow(
        example=EXAMPLE,
        routing_parameter=1.0,
        maximum_iterations=arguments.maximum_iterations,
        operator_cache_directory=arguments.operator_cache,
        include_relaxation_and_holdout=True,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
