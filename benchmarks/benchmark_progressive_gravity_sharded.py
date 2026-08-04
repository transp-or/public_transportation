"""Large persisted-shard benchmark for progressive-fidelity gravity evaluation."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from benchmarks.benchmark_sharded_gravity_operator import _inputs, _layout, _problem
from public_transportation.inference.gravity import (
    GravityFidelityRequest,
    build_gravity_fidelity_context,
    gravity_value_and_gradient_progressive,
)
from public_transportation.inference.sharded_fixed_routing import (
    FixedRoutingPreparationConfig,
    prepare_fixed_routing_sharded,
)
from public_transportation.inference.sharded_matrix_free_operator import (
    ShardedMatrixFreeFixedRoutingMeasurementOperator,
)
from public_transportation.measurement.mapping import AggregationSpec


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _evaluate(raw, *, problem, context, effort: float, seed: int):
    started = perf_counter()
    result = gravity_value_and_gradient_progressive(
        raw,
        problem=problem,
        fidelity=GravityFidelityRequest(
            effort_percent=effort,
            seed=seed,
            quality_groups=4,
        ),
        context=context,
    )
    jax.block_until_ready((result.evaluation.objective, result.gradient))
    return result, perf_counter() - started


def run(args) -> dict[str, object]:
    inputs = _inputs(
        nodes=args.nodes,
        degree=args.maximum_out_degree,
        groups=args.destination_groups,
        od_cells=args.od_cells,
    )
    compact = _layout(args.od_cells)
    spec = AggregationSpec(
        num_measurements=args.measurements,
        measurement_index=np.arange(args.measurements, dtype=np.int32),
        link_index=(
            np.arange(args.measurements, dtype=np.int32) % inputs.graph.num_links
        ),
    )
    with tempfile.TemporaryDirectory(
        prefix="progressive-gravity-sharded-"
    ) as temporary:
        root = Path(temporary)
        preparation_started = perf_counter()
        prepared = prepare_fixed_routing_sharded(
            inputs=inputs,
            theta=1.0,
            config=FixedRoutingPreparationConfig(
                maximum_groups_per_shard=args.groups_per_shard,
                cache_directory=root / "routing",
                checkpoint_directory=root / "checkpoints",
                resident_shard_limit=args.resident_shard_limit,
            ),
        )
        preparation_seconds = perf_counter() - preparation_started
        operator = ShardedMatrixFreeFixedRoutingMeasurementOperator(
            inputs=inputs,
            routing=prepared.routing,
            spec=spec,
            compact_layout=compact,
            resident_shard_limit=args.resident_shard_limit,
            operator_shards_per_batch=args.operator_shards_per_batch,
            group_execution_strategy=args.group_execution_strategy,
            shard_execution_strategy=args.shard_execution_strategy,
            operator_concurrency=args.operator_concurrency,
        )
        problem, raw = _problem(operator)
        context = build_gravity_fidelity_context(problem)

        # Compile every path before measuring steady-state execution. The warm-up
        # uses the benchmark seed so the same nested selection is timed below.
        warmup_seconds: dict[str, float] = {}
        for effort in (*args.efforts, 100.0):
            _, elapsed = _evaluate(
                raw,
                problem=problem,
                context=context,
                effort=effort,
                seed=args.seed,
            )
            warmup_seconds[str(effort)] = elapsed

        exact_times = []
        exact = None
        for _ in range(args.repetitions):
            exact, elapsed = _evaluate(
                raw,
                problem=problem,
                context=context,
                effort=100.0,
                seed=args.seed,
            )
            exact_times.append(elapsed)
        assert exact is not None
        exact_gradient = np.asarray(exact.gradient)
        exact_mean = np.asarray(exact.evaluation.measurement_mean)
        exact_objective = float(exact.evaluation.objective)
        exact_gradient_norm = max(
            float(np.linalg.norm(exact_gradient)), np.finfo(float).eps
        )
        exact_mean_norm = max(float(np.linalg.norm(exact_mean)), np.finfo(float).eps)
        exact_median = float(np.median(exact_times))

        rows = []
        for effort in args.efforts:
            times = []
            results = []
            for _ in range(args.repetitions):
                result, elapsed = _evaluate(
                    raw,
                    problem=problem,
                    context=context,
                    effort=effort,
                    seed=args.seed,
                )
                results.append(result)
                times.append(elapsed)
            representative = results[0]
            gradient_errors = [
                float(np.linalg.norm(np.asarray(item.gradient) - exact_gradient))
                / exact_gradient_norm
                for item in results
            ]
            count_errors = [
                float(
                    np.linalg.norm(
                        np.asarray(item.evaluation.measurement_mean) - exact_mean
                    )
                )
                / exact_mean_norm
                for item in results
            ]
            objective_errors = [
                abs(float(item.evaluation.objective) - exact_objective)
                for item in results
            ]
            cosines = []
            for item in results:
                gradient = np.asarray(item.gradient)
                denominator = np.linalg.norm(gradient) * np.linalg.norm(exact_gradient)
                cosines.append(
                    None
                    if denominator <= np.finfo(float).eps
                    else float(np.dot(gradient, exact_gradient) / denominator)
                )
            median = float(np.median(times))
            rows.append(
                {
                    "requested_effort_percent": effort,
                    "effective_effort_percent": (
                        representative.fidelity.effective_effort_percent
                    ),
                    "selected_shards": representative.fidelity.selected_shard_count,
                    "total_shards": representative.fidelity.total_shard_count,
                    "median_wall_seconds": median,
                    "speedup_over_exact": exact_median / median,
                    "gradient_relative_norm_error_median": float(
                        np.median(gradient_errors)
                    ),
                    "gradient_cosine_similarity_median": float(
                        np.median([value for value in cosines if value is not None])
                    ),
                    "predicted_count_relative_error_median": float(
                        np.median(count_errors)
                    ),
                    "objective_absolute_error_median": float(
                        np.median(objective_errors)
                    ),
                    "quality_score": representative.quality.quality_score,
                }
            )

        return {
            "schema_version": 1,
            "backend": jax.default_backend(),
            "problem": {
                "nodes": args.nodes,
                "links": inputs.graph.num_links,
                "destination_groups": args.destination_groups,
                "od_cells": args.od_cells,
                "measurements": args.measurements,
                "routing_shards": prepared.routing.num_shards,
                "resident_shard_limit": args.resident_shard_limit,
            },
            "execution": {
                "group_execution_strategy": args.group_execution_strategy,
                "shard_execution_strategy": args.shard_execution_strategy,
                "operator_shards_per_batch": args.operator_shards_per_batch,
                "operator_concurrency": args.operator_concurrency,
            },
            "preparation_seconds": preparation_seconds,
            "warmup_seconds": warmup_seconds,
            "repetitions": args.repetitions,
            "exact": {
                "median_wall_seconds": exact_median,
                "wall_seconds": exact_times,
            },
            "results": rows,
            "peak_rss_bytes": _peak_rss_bytes(),
            "notes": [
                "Preparation and compilation warm-up are excluded from steady-state timings.",
                "Approximation errors are measured against the exact adjoint at identical parameters.",
                "Synthetic topology is public and reproducible; it is not a claim about TPG runtime.",
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=512)
    parser.add_argument("--maximum-out-degree", type=int, default=2)
    parser.add_argument("--destination-groups", type=int, default=64)
    parser.add_argument("--groups-per-shard", type=int, default=2)
    parser.add_argument("--od-cells", type=int, default=2048)
    parser.add_argument("--measurements", type=int, default=256)
    parser.add_argument("--resident-shard-limit", type=int, default=2)
    parser.add_argument("--operator-shards-per-batch", type=int, default=4)
    parser.add_argument("--group-execution-strategy", default="scan")
    parser.add_argument("--shard-execution-strategy", default="aggregate")
    parser.add_argument("--operator-concurrency", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--efforts", nargs="+", type=float, default=(10.0, 25.0, 50.0, 75.0)
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")


if __name__ == "__main__":
    main()
