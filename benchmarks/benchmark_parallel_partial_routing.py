"""Public scaling benchmark for the persistent partial-routing executor."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import jax

from benchmarks.benchmark_sharded_gravity_operator import _inputs, _layout, _problem
from public_transportation.inference.gravity import gravity_value_and_gradient_adjoint
from public_transportation.inference.parallel_exact_gate import (
    assess_parallel_exact_gate,
)
from public_transportation.inference.parallel_partial_execution import (
    build_balanced_microshard_plan,
    routing_group_work_units,
)
from public_transportation.inference.parallel_routing_executor import (
    ParallelExactRoutingOperator,
    ParallelRoutingExecutorConfig,
    PersistentParallelRoutingExecutor,
)
from public_transportation.inference.sharded_fixed_routing import (
    FixedRoutingPreparationConfig,
    prepare_fixed_routing_sharded,
)
from public_transportation.inference.sharded_matrix_free_operator import (
    ShardedMatrixFreeFixedRoutingMeasurementOperator,
)
from public_transportation.measurement.mapping import AggregationSpec


def _median_product(function, repetitions: int):
    times = []
    result = None
    for _ in range(repetitions):
        started = perf_counter()
        result = function()
        times.append(perf_counter() - started)
    return result, float(np.median(times)), times


def _median_gravity(function, repetitions: int):
    times = []
    result = None
    for _ in range(repetitions):
        started = perf_counter()
        result = function()
        jax.block_until_ready(result)
        times.append(perf_counter() - started)
    return result, float(np.median(times)), times


def run(args) -> dict[str, object]:
    inputs = _inputs(
        nodes=args.nodes,
        degree=args.maximum_out_degree,
        groups=args.destination_groups,
        od_cells=args.od_cells,
    )
    spec = AggregationSpec(
        num_measurements=args.measurements,
        measurement_index=np.arange(args.measurements, dtype=np.int32),
        link_index=np.arange(args.measurements, dtype=np.int32) % inputs.graph.num_links,
    )
    with tempfile.TemporaryDirectory(prefix="parallel-partial-routing-") as temporary:
        root = Path(temporary)
        prepared = prepare_fixed_routing_sharded(
            inputs=inputs,
            theta=1.0,
            config=FixedRoutingPreparationConfig(
                maximum_groups_per_shard=args.groups_per_routing_shard,
                cache_directory=root / "routing",
                checkpoint_directory=root / "checkpoints",
                resident_shard_limit=args.resident_shard_limit,
            ),
        )
        operator = ShardedMatrixFreeFixedRoutingMeasurementOperator(
            inputs=inputs,
            routing=prepared.routing,
            spec=spec,
            compact_layout=_layout(args.od_cells),
            resident_shard_limit=args.resident_shard_limit,
            operator_shards_per_batch=args.exact_shards_per_batch,
        )
        plan = build_balanced_microshard_plan(
            routing_group_work_units(operator),
            target_microshards=args.microshards,
            problem_fingerprint=operator.assignment_fingerprint,
        )
        demand = np.linspace(0.1, 2.0, operator.num_free_od, dtype=np.float32)
        cotangent = np.linspace(
            -1.0, 1.0, operator.num_measurements, dtype=np.float32
        )
        operator.matvec(demand)
        operator.rmatvec(cotangent)
        exact_forward, exact_forward_median, exact_forward_times = _median_product(
            lambda: operator.matvec(demand), args.repetitions
        )
        exact_reverse, exact_reverse_median, exact_reverse_times = _median_product(
            lambda: operator.rmatvec(cotangent), args.repetitions
        )
        problem, raw = _problem(operator)
        gravity_value_and_gradient_adjoint(raw, problem=problem)
        exact_gravity, exact_gravity_median, exact_gravity_times = _median_gravity(
            lambda: gravity_value_and_gradient_adjoint(raw, problem=problem),
            args.repetitions,
        )

        cases = []
        for workers in args.workers:
            config = ParallelRoutingExecutorConfig(
                worker_count=workers,
                threads_per_worker=args.threads_per_worker,
                supported_group_batch_sizes=tuple(args.group_batch_sizes),
            )
            with PersistentParallelRoutingExecutor(
                operator=operator, microshard_plan=plan, config=config
            ) as executor:
                executor.execute("matvec", demand)
                executor.execute("rmatvec", cotangent)
                forward, forward_median, forward_times = _median_product(
                    lambda: executor.execute("matvec", demand), args.repetitions
                )
                reverse, reverse_median, reverse_times = _median_product(
                    lambda: executor.execute("rmatvec", cotangent), args.repetitions
                )
                parallel_operator = ParallelExactRoutingOperator(operator, executor)
                parallel_problem = replace(problem, operator=parallel_operator)
                gravity_value_and_gradient_adjoint(raw, problem=parallel_problem)
                parallel_gravity, parallel_gravity_median, parallel_gravity_times = (
                    _median_gravity(
                        lambda: gravity_value_and_gradient_adjoint(
                            raw, problem=parallel_problem
                        ),
                        args.repetitions,
                    )
                )
            forward_error = float(
                np.max(np.abs(np.asarray(forward.value) - exact_forward))
            )
            reverse_error = float(
                np.max(np.abs(np.asarray(reverse.value) - exact_reverse))
            )
            partial_total = forward_median + reverse_median
            exact_total = exact_forward_median + exact_reverse_median
            gate = assess_parallel_exact_gate(
                reference_forward=exact_forward,
                parallel_forward=forward.value,
                reference_reverse=exact_reverse,
                parallel_reverse=reverse.value,
                reference_objective=float(exact_gravity[0].objective),
                parallel_objective=float(parallel_gravity[0].objective),
                reference_gradient=exact_gravity[1],
                parallel_gradient=parallel_gravity[1],
                existing_exact_seconds=exact_gravity_median,
                parallel_exact_seconds=parallel_gravity_median,
                requested_workers=workers,
                observed_worker_lanes=min(
                    len(forward.worker_thread_ids), len(reverse.worker_thread_ids)
                ),
            )
            cases.append(
                {
                    "requested_workers": workers,
                    "effective_workers": forward.worker_count,
                    "forward_median_seconds": forward_median,
                    "reverse_median_seconds": reverse_median,
                    "combined_median_seconds": partial_total,
                    "speedup_over_existing_exact": exact_total / partial_total,
                    "forward_wall_seconds": forward_times,
                    "reverse_wall_seconds": reverse_times,
                    "forward_worker_lanes_used": len(forward.worker_thread_ids),
                    "reverse_worker_lanes_used": len(reverse.worker_thread_ids),
                    "forward_max_abs_error": forward_error,
                    "reverse_max_abs_error": reverse_error,
                    "gravity_objective_median_seconds": parallel_gravity_median,
                    "gravity_objective_wall_seconds": parallel_gravity_times,
                    "promotion_gate": gate.to_dict(),
                }
            )
        return {
            "schema_version": 1,
            "problem": {
                "nodes": args.nodes,
                "links": inputs.graph.num_links,
                "destination_groups": args.destination_groups,
                "od_cells": args.od_cells,
                "measurements": args.measurements,
                "routing_shards": prepared.routing.num_shards,
                "microshards": len(plan.microshards),
            },
            "existing_exact": {
                "forward_median_seconds": exact_forward_median,
                "reverse_median_seconds": exact_reverse_median,
                "combined_median_seconds": exact_forward_median + exact_reverse_median,
                "forward_wall_seconds": exact_forward_times,
                "reverse_wall_seconds": exact_reverse_times,
                "gravity_objective_median_seconds": exact_gravity_median,
                "gravity_objective_wall_seconds": exact_gravity_times,
            },
            "cases": cases,
            "repetitions": args.repetitions,
            "notes": [
                "Compilation warm-up is excluded from steady-state timings.",
                "Every parallel case executes 100% of routing work.",
                "This is a public synthetic CPU benchmark, not a TPG result.",
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--maximum-out-degree", type=int, default=2)
    parser.add_argument("--destination-groups", type=int, default=64)
    parser.add_argument("--groups-per-routing-shard", type=int, default=2)
    parser.add_argument("--od-cells", type=int, default=1024)
    parser.add_argument("--measurements", type=int, default=128)
    parser.add_argument("--resident-shard-limit", type=int, default=8)
    parser.add_argument("--exact-shards-per-batch", type=int, default=4)
    parser.add_argument("--microshards", type=int, default=64)
    parser.add_argument("--workers", nargs="+", type=int, default=(1, 2, 4, 8))
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument(
        "--group-batch-sizes", nargs="+", type=int, default=(1, 2, 4, 8, 16)
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
