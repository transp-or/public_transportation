"""Deterministic public benchmark for bounded routing shards."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp
import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.domain import Scenario
from public_transportation.inference.assignment_adapter import (
    build_assignment_inputs,
    prepare_fixed_routing,
)
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout
from public_transportation.inference.sharded_fixed_routing import (
    FixedRoutingPreparationConfig,
    load_fixed_routing_shard,
    prepare_fixed_routing_sharded,
)
from public_transportation.inference.sharded_matrix_free_operator import (
    ShardedMatrixFreeFixedRoutingMeasurementOperator,
)
from public_transportation.measurement.mapping import AggregationSpec

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/source/examples/simple_example_02"


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
    num_od = int(inputs.od_origin_node.shape[0])
    layout = ODParameterLayout(
        num_od_total=num_od,
        od_keys=tuple((f"o{i}", "d", "t") for i in range(num_od)),
        free_od_indices=tuple(range(num_od)),
        fixed_od_indices=(),
        fixed_od_values=(),
        free_baseline_values=tuple(1.0 for _ in range(num_od)),
        fixed_zero_indices=(),
        fixed_positive_indices=(),
    )
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    links = np.arange(min(8, inputs.graph.num_links), dtype=np.int32)
    spec = AggregationSpec(
        num_measurements=3,
        measurement_index=np.arange(len(links), dtype=np.int32) % 3,
        link_index=links,
    )
    started = perf_counter()
    complete = prepare_fixed_routing(inputs=inputs, theta=1.0)
    complete_seconds = perf_counter() - started
    config = FixedRoutingPreparationConfig(
        maximum_groups_per_shard=2,
        cache_directory=work_directory / "cache",
        checkpoint_directory=work_directory / "checkpoint",
        detailed_profiling=True,
    )
    first = prepare_fixed_routing_sharded(inputs=inputs, theta=1.0, config=config)
    reload_started = perf_counter()
    reload = prepare_fixed_routing_sharded(inputs=inputs, theta=1.0, config=config)
    reload_seconds = perf_counter() - reload_started
    masks = []
    probabilities = []
    for descriptor in first.plan.descriptors:
        shard = load_fixed_routing_shard(routing=first.routing, descriptor=descriptor)
        masks.append(shard.effective_group_link_mask)
        probabilities.append(shard.group_link_probability)
    mask = np.concatenate(masks) if masks else np.empty_like(complete.effective_group_link_mask)
    probability = np.concatenate(probabilities) if probabilities else np.empty_like(complete.group_link_probability)
    operator = ShardedMatrixFreeFixedRoutingMeasurementOperator(
        inputs=inputs,
        routing=first.routing,
        spec=spec,
        compact_layout=compact,
    )
    demand = jnp.linspace(0.1, 5.0, num_od, dtype=jnp.float32)
    forward_started = perf_counter()
    forward = operator.matvec(demand)
    forward_seconds = perf_counter() - forward_started
    transpose_started = perf_counter()
    transpose = operator.rmatvec(np.ones(3, dtype=np.float32))
    transpose_seconds = perf_counter() - transpose_started
    return {
        "schema_version": 1,
        "destination_groups": first.routing.num_destination_groups,
        "links": first.routing.num_links,
        "shards": first.routing.num_shards,
        "groups_per_shard": first.plan.groups_per_full_shard,
        "predicted_shard_bytes": first.plan.retained_bytes_per_group
        * first.plan.groups_per_full_shard,
        "cache_bytes": first.retained_cache_bytes,
        "complete_seconds": complete_seconds,
        "sharded_seconds": first.elapsed_seconds,
        "cache_reload_seconds": reload_seconds,
        "cache_hits_on_reload": reload.cache_hits,
        "compilation_count": first.compilation_count,
        "tracing_seconds": first.tracing_seconds,
        "lowering_seconds": first.lowering_seconds,
        "compilation_seconds": first.compilation_seconds,
        "peak_rss_bytes": first.peak_rss_bytes,
        "forward_seconds": forward_seconds,
        "transpose_seconds": transpose_seconds,
        "forward_norm": float(np.linalg.norm(forward)),
        "transpose_norm": float(np.linalg.norm(transpose)),
        "maximum_mask_difference": int(np.max(mask != np.asarray(complete.effective_group_link_mask), initial=0)),
        "maximum_probability_difference": float(np.max(np.abs(probability - np.asarray(complete.group_link_probability)), initial=0.0)),
        "global_measurement_matrix_constructed": False,
        "construction_workers": config.construction_workers,
        "threads_per_worker": config.threads_per_worker,
        "warm_shard_diagnostics": (
            None
            if not first.shard_diagnostics
            else {
                name: value
                for name, value in asdict(first.shard_diagnostics[-1]).items()
                if name != "device_memory_stats"
            }
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="sharded-routing-benchmark-") as temporary:
        report = run_benchmark(work_directory=Path(temporary))
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
