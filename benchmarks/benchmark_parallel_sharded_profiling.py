"""Short benchmark for parallel fixed-routing profiling overhead."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.domain import Scenario
from public_transportation.inference.assignment_adapter import build_assignment_inputs
from public_transportation.inference.sharded_fixed_routing import (
    FixedRoutingPreparationConfig,
    load_fixed_routing_shard,
    prepare_fixed_routing_sharded,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/source/examples/simple_example_02"


def _run(*, inputs, directory: Path, workers: int, profiling: bool):
    config = FixedRoutingPreparationConfig(
        maximum_groups_per_shard=2,
        construction_workers=workers,
        detailed_profiling=profiling,
        cache_directory=directory / "cache",
        checkpoint_directory=directory / "checkpoint",
        durable_progress=False,
    )
    started = perf_counter()
    result = prepare_fixed_routing_sharded(inputs=inputs, theta=1.0, config=config)
    elapsed = perf_counter() - started
    masks = []
    probabilities = []
    for descriptor in result.plan.descriptors:
        shard = load_fixed_routing_shard(
            routing=result.routing, descriptor=descriptor
        )
        masks.append(shard.effective_group_link_mask)
        probabilities.append(shard.group_link_probability)
    return result, elapsed, np.concatenate(masks), np.concatenate(probabilities)


def run_benchmark(*, work_directory: Path) -> dict[str, object]:
    scenario_directory = work_directory / "scenario"
    shutil.copytree(EXAMPLE / "data", scenario_directory)
    shutil.copy2(
        EXAMPLE / "pre_processing/results/demand.csv",
        scenario_directory / "demand.csv",
    )
    scenario = Scenario.from_folder(scenario_directory, strict=True)
    artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
    inputs = build_assignment_inputs(artifacts=artifacts)
    runs = {}
    arrays = {}
    modes = (
        ("serial_unprofiled", 1, False),
        ("parallel_unprofiled", 2, False),
        ("parallel_profiled", 2, True),
    )
    for name, workers, profiling in modes:
        result, elapsed, mask, probability = _run(
            inputs=inputs,
            directory=work_directory / name,
            workers=workers,
            profiling=profiling,
        )
        arrays[name] = (mask, probability)
        runs[name] = {
            "elapsed_seconds": elapsed,
            "peak_rss_bytes": result.peak_rss_bytes,
            "compilation_count": result.compilation_count,
            "diagnostic_count": len(result.shard_diagnostics),
            "diagnostic_order": [
                item.shard_index for item in result.shard_diagnostics
            ],
            "shard_seconds": [
                item.total_shard_seconds for item in result.shard_diagnostics
            ],
            "diagnostics": [
                {
                    key: value
                    for key, value in asdict(item).items()
                    if key != "device_memory_stats"
                }
                for item in result.shard_diagnostics
            ],
        }
    reference_mask, reference_probability = arrays["serial_unprofiled"]
    equivalence = {}
    for name, (mask, probability) in arrays.items():
        equivalence[name] = {
            "mask_equal": bool(np.array_equal(mask, reference_mask)),
            "maximum_probability_difference": float(
                np.max(np.abs(probability - reference_probability), initial=0.0)
            ),
        }
    unprofiled = float(runs["parallel_unprofiled"]["elapsed_seconds"])
    profiled = float(runs["parallel_profiled"]["elapsed_seconds"])
    return {
        "schema_version": 1,
        "destination_groups": int(inputs.group_dest_node.shape[0]),
        "links": inputs.graph.num_links,
        "runs": runs,
        "numerical_equivalence": equivalence,
        "parallel_profiling_overhead_seconds": profiled - unprofiled,
        "parallel_profiling_overhead_fraction": (
            (profiled - unprofiled) / unprofiled if unprofiled else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="parallel-routing-profile-") as temporary:
        report = run_benchmark(work_directory=Path(temporary))
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
