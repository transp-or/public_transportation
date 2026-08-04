"""Public accuracy/runtime benchmark for parallel fixed-budget gravity."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import tempfile
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from benchmarks.benchmark_sharded_gravity_operator import (
    _inputs,
    _layout,
    _problem,
)
from public_transportation.inference.gravity import gravity_value_and_gradient_adjoint
from public_transportation.inference.parallel_partial_execution import (
    build_balanced_microshard_plan,
    plan_fixed_budget_routing_selection,
    routing_group_work_units,
)
from public_transportation.inference.parallel_gravity_anchor import (
    create_parallel_gravity_anchor,
    parallel_anchored_value_and_gradient,
)
from public_transportation.inference.parallel_routing_executor import (
    ParallelApproximateRoutingOperator,
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


def _measure(function, repetitions: int):
    values = []
    result = None
    for _ in range(repetitions):
        started = perf_counter()
        result = function()
        jax.block_until_ready(result)
        values.append(perf_counter() - started)
    return result, float(np.median(values)), values


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
    with tempfile.TemporaryDirectory(prefix="parallel-partial-gravity-") as temporary:
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
        microshards = build_balanced_microshard_plan(
            routing_group_work_units(operator),
            target_microshards=args.microshards,
            problem_fingerprint=operator.assignment_fingerprint,
        )
        problem, raw = _problem(operator)
        anchor = (
            create_parallel_gravity_anchor(raw, problem=problem)
            if args.anchor_comparison
            else None
        )
        evaluation_raw = (
            np.asarray(raw) + np.asarray(args.parameter_perturbation)
            if args.anchor_comparison
            else raw
        )
        gravity_value_and_gradient_adjoint(evaluation_raw, problem=problem)
        exact, exact_median, exact_times = _measure(
            lambda: gravity_value_and_gradient_adjoint(
                evaluation_raw, problem=problem
            ),
            args.repetitions,
        )
        exact_mean = np.asarray(exact[0].measurement_mean)
        exact_gradient = np.asarray(exact[1])
        mean_norm = max(np.linalg.norm(exact_mean), np.finfo(float).eps)
        gradient_norm = max(np.linalg.norm(exact_gradient), np.finfo(float).eps)

        rows = []
        config = ParallelRoutingExecutorConfig(
            worker_count=args.workers,
            threads_per_worker=args.threads_per_worker,
            supported_group_batch_sizes=tuple(args.group_batch_sizes),
            maximum_retained_batch_bytes=args.maximum_retained_batch_bytes,
        )
        with PersistentParallelRoutingExecutor(
            operator=operator, microshard_plan=microshards, config=config
        ) as executor:
            for effort in args.efforts:
                selection = plan_fixed_budget_routing_selection(
                    microshards, effort_percent=effort, seed=args.seed
                )
                approximate_operator = ParallelApproximateRoutingOperator(
                    operator, executor, selection
                )
                approximate_problem = replace(problem, operator=approximate_operator)
                gravity_value_and_gradient_adjoint(
                    evaluation_raw, problem=approximate_problem
                )
                result, median, times = _measure(
                    lambda: gravity_value_and_gradient_adjoint(
                        evaluation_raw, problem=approximate_problem
                    ),
                    args.repetitions,
                )
                mean = np.asarray(result[0].measurement_mean)
                gradient = np.asarray(result[1])
                denominator = np.linalg.norm(gradient) * np.linalg.norm(exact_gradient)
                row = {
                        "requested_effort_percent": effort,
                        "realized_effort_percent": selection.realized_effort_percent,
                        "selected_microshards": len(selection.selected_work_ids),
                        "total_microshards": len(microshards.microshards),
                        "median_wall_seconds": median,
                        "wall_seconds": times,
                        "speedup_over_exact": exact_median / median,
                        "objective_absolute_error": abs(
                            float(result[0].objective) - float(exact[0].objective)
                        ),
                        "gradient_relative_norm_error": float(
                            np.linalg.norm(gradient - exact_gradient) / gradient_norm
                        ),
                        "gradient_cosine_similarity": (
                            None
                            if denominator <= np.finfo(float).eps
                            else float(np.dot(gradient, exact_gradient) / denominator)
                        ),
                        "predicted_count_relative_error": float(
                            np.linalg.norm(mean - exact_mean) / mean_norm
                        ),
                    }
                if anchor is not None:
                    parallel_anchored_value_and_gradient(
                        evaluation_raw,
                        problem=problem,
                        executor=executor,
                        selection=selection,
                        anchor=anchor,
                    )
                    anchored, anchored_median, anchored_times = _measure(
                        lambda: parallel_anchored_value_and_gradient(
                            evaluation_raw,
                            problem=problem,
                            executor=executor,
                            selection=selection,
                            anchor=anchor,
                        ),
                        args.repetitions,
                    )
                    anchored_mean = np.asarray(anchored[0].measurement_mean)
                    anchored_gradient = np.asarray(anchored[1])
                    row["anchored"] = {
                        "median_wall_seconds": anchored_median,
                        "wall_seconds": anchored_times,
                        "speedup_over_exact": exact_median / anchored_median,
                        "objective_absolute_error": abs(
                            float(anchored[0].objective) - float(exact[0].objective)
                        ),
                        "gradient_relative_norm_error": float(
                            np.linalg.norm(anchored_gradient - exact_gradient)
                            / gradient_norm
                        ),
                        "predicted_count_relative_error": float(
                            np.linalg.norm(anchored_mean - exact_mean) / mean_norm
                        ),
                    }
                rows.append(row)
        return {
            "schema_version": 1,
            "problem": {
                "nodes": args.nodes,
                "links": inputs.graph.num_links,
                "destination_groups": args.destination_groups,
                "od_cells": args.od_cells,
                "measurements": args.measurements,
                "routing_shards": prepared.routing.num_shards,
                "microshards": len(microshards.microshards),
            },
            "execution": {
                "workers": args.workers,
                "group_batch_sizes": args.group_batch_sizes,
                "maximum_retained_batch_bytes": args.maximum_retained_batch_bytes,
            },
            "exact": {
                "median_wall_seconds": exact_median,
                "wall_seconds": exact_times,
            },
            "results": rows,
            "repetitions": args.repetitions,
            "anchor_comparison": args.anchor_comparison,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=2048)
    parser.add_argument("--maximum-out-degree", type=int, default=2)
    parser.add_argument("--destination-groups", type=int, default=256)
    parser.add_argument("--groups-per-routing-shard", type=int, default=2)
    parser.add_argument("--od-cells", type=int, default=8192)
    parser.add_argument("--measurements", type=int, default=512)
    parser.add_argument("--resident-shard-limit", type=int, default=64)
    parser.add_argument("--exact-shards-per-batch", type=int, default=4)
    parser.add_argument("--microshards", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument(
        "--group-batch-sizes",
        nargs="+",
        type=int,
        default=(1, 2, 4, 8, 16, 32, 64),
    )
    parser.add_argument(
        "--maximum-retained-batch-bytes", type=int, default=512 * 1024 * 1024
    )
    parser.add_argument("--efforts", nargs="+", type=float, default=(10, 25, 50, 75))
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--anchor-comparison", action="store_true")
    parser.add_argument(
        "--parameter-perturbation",
        nargs=3,
        type=float,
        default=(0.03, -0.02, 0.01),
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
