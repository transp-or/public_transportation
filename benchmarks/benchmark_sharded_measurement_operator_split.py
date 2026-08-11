"""Bounded benchmark for the split sharded measurement construction path.

This benchmark intentionally uses the public ``simple_example_02`` fixture and
small shard limits.  It measures the same construction path used by a
full-network run without committing generated cache artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.domain import Scenario
from public_transportation.inference.assignment_adapter import build_assignment_inputs
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.fixed_routing_sharded_builder import (
    ShardedConstructionConfig,
    prepare_sharded_fixed_routing_measurement_operator,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout
from public_transportation.inference.sharded_fixed_routing import (
    FixedRoutingPreparationConfig,
    prepare_fixed_routing_sharded,
)
from public_transportation.measurement.mapping import AggregationSpec

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/source/examples/simple_example_02"


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.uname().sysname == "Darwin" else value * 1024


def run(*, workers: int, chunk_size: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="split-sharded-benchmark-") as raw:
        work = Path(raw)
        scenario_directory = work / "scenario"
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
        routing = prepare_fixed_routing_sharded(
            inputs=inputs,
            theta=1.0,
            config=FixedRoutingPreparationConfig(
                maximum_groups_per_shard=2,
                cache_directory=work / "routing-cache",
                checkpoint_directory=work / "routing-checkpoint",
            ),
        ).routing
        links = np.arange(min(24, inputs.graph.num_links), dtype=np.int32)
        spec = AggregationSpec(
            num_measurements=8,
            measurement_index=np.arange(links.size, dtype=np.int32) % 8,
            link_index=links,
        )
        config = ShardedConstructionConfig(
            od_chunk_size=chunk_size,
            measurement_block_size=2,
            workers=workers,
            maximum_resident_shards=max(2, workers),
            worker_memory_budget_bytes=1_000_000_000,
            maximum_storage_shards=10_000,
            maximum_manifest_bytes=100_000_000,
            maximum_filesystem_operations=100_000,
            maximum_sparse_calls_per_product=10_000,
            maximum_construction_dispatches=1_000_000,
            target_nonzeros_per_storage_shard=64,
            maximum_nonzeros_per_storage_shard=4_096,
        )
        started = perf_counter()
        result = prepare_sharded_fixed_routing_measurement_operator(
            directory=work / "operator",
            inputs=inputs,
            routing=routing,
            spec=spec,
            compact_layout=compact,
            assignment_fingerprint="split-kernel-public-benchmark",
            od_layout_fingerprint=layout.fingerprint,
            config=config,
        )
        total_seconds = perf_counter() - started
        stored_bytes = sum(
            path.stat().st_size for path in (work / "operator" / "shards").glob("*.npz")
        )
        return {
            "workers": workers,
            "od_chunk_size": chunk_size,
            "construction_seconds": result.total_seconds,
            "wall_seconds": total_seconds,
            "peak_rss_bytes": _peak_rss_bytes(),
            "stored_bytes": stored_bytes,
            "stored_nonzeros": result.manifest.aggregate_nonzeros,
            "reachability_evaluations": getattr(
                result, "reachability_evaluations", None
            ),
            "edge_gather_evaluations": getattr(result, "edge_gather_evaluations", None),
            "compilation_count": getattr(result, "compilation_count", None),
            "jax_execution_seconds": getattr(
                result, "jax_execution_seconds", result.dispatch_seconds
            ),
            "synchronization_seconds": result.synchronization_seconds,
            "host_transfer_seconds": result.transfer_seconds,
            "sparse_assembly_seconds": result.zero_filtering_seconds,
            "persistence_seconds": result.shard_persistence_seconds,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args()
    report = [run(workers=value, chunk_size=args.chunk_size) for value in args.workers]
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
