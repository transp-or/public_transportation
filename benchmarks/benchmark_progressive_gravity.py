"""Bounded public benchmark for progressive-fidelity gravity evaluation."""

from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.inference.fixed_routing_measurement_operator import (
    FixedRoutingMeasurementOperator,
    MeasurementOperatorMetrics,
)
from public_transportation.inference.gravity import (
    GravityFeatures,
    GravityFidelityRequest,
    GravityLikelihood,
    GravityModelSpecification,
    GravityObjectiveProblem,
    GravityParameterLayout,
    build_gravity_fidelity_context,
    gravity_value_and_gradient_progressive,
)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _problem(cells: int, measurements: int, seed: int) -> GravityObjectiveProblem:
    rng = np.random.default_rng(seed)
    origins = max(2, int(np.ceil(cells / 10)))
    origin_index = np.arange(cells) // 10
    destination_index = np.arange(cells) % 10
    features = GravityFeatures(
        canonical_od_index=np.arange(cells),
        origin_index=origin_index,
        destination_index=destination_index,
        departure_time_index=np.zeros(cells, dtype=int),
        origin_time_group_index=origin_index,
        journey_time=rng.uniform(5.0, 50.0, cells),
        transfer_count=rng.integers(0, 3, cells),
        structural_feasible=np.ones(cells, dtype=bool),
        origin_time_totals=np.full(origins, 100.0),
        destination_attractiveness=rng.uniform(0.5, 2.0, cells),
        num_origins=origins,
        num_destinations=10,
        num_departure_times=1,
        od_layout_fingerprint="progressive-benchmark",
        journey_time_scale=20.0,
    )
    matrix = rng.uniform(0.0, 1.0, (measurements, cells))
    matrix[rng.random(matrix.shape) < 0.9] = 0.0
    matrix = jnp.asarray(matrix)
    operator = FixedRoutingMeasurementOperator(
        matrix=matrix,
        fixed_measurement_offset=jnp.full(measurements, 1.0),
        representation="dense",
        num_active_od=cells,
        num_free_od=cells,
        num_measurements=measurements,
        od_layout_fingerprint="progressive-benchmark",
        compact_layout_fingerprint="progressive-benchmark",
        assignment_fingerprint="progressive-benchmark-assignment",
        graph_fingerprint="progressive-benchmark-graph",
        mapping_fingerprint="progressive-benchmark-mapping",
        theta=1.0,
        dtype=str(matrix.dtype),
        metrics=MeasurementOperatorMetrics(
            construction_seconds=0.0,
            dense_bytes=matrix.nbytes,
            stored_bytes=matrix.nbytes,
            peak_construction_bytes=matrix.nbytes,
            nonzero_entries=int(np.count_nonzero(matrix)),
            total_entries=matrix.size,
            density=float(np.count_nonzero(matrix) / matrix.size),
            chunk_size=cells,
            cache_hit=False,
        ),
    )
    layout = GravityParameterLayout(GravityModelSpecification())
    provisional = GravityObjectiveProblem(
        features=features,
        parameter_layout=layout,
        operator=operator,
        observations=np.ones(measurements),
        likelihood=GravityLikelihood.NEGATIVE_BINOMIAL,
    )
    truth = gravity_value_and_gradient_progressive(
        np.zeros(3),
        problem=provisional,
        fidelity=GravityFidelityRequest(),
    )
    observations = np.maximum(0.0, np.rint(np.asarray(truth.evaluation.measurement_mean)))
    return GravityObjectiveProblem(
        features=features,
        parameter_layout=layout,
        operator=operator,
        observations=observations,
        likelihood=GravityLikelihood.NEGATIVE_BINOMIAL,
    )


def run(args) -> dict[str, object]:
    item = _problem(args.cells, args.measurements, args.seed)
    context = build_gravity_fidelity_context(
        item, maximum_dense_shards=args.shards
    )
    raw = np.asarray((0.1, -0.2, 1.0))
    exact = gravity_value_and_gradient_progressive(
        raw,
        problem=item,
        fidelity=GravityFidelityRequest(effort_percent=100, seed=args.seed),
        context=context,
    )
    exact_objective = float(exact.evaluation.objective)
    exact_gradient = np.asarray(exact.gradient)
    exact_mean = np.asarray(exact.evaluation.measurement_mean)
    rows = []
    for effort in args.efforts:
        times = []
        result = None
        for repetition in range(args.repetitions):
            started = perf_counter()
            result = gravity_value_and_gradient_progressive(
                raw,
                problem=item,
                fidelity=GravityFidelityRequest(
                    effort_percent=effort,
                    seed=args.seed + repetition,
                    quality_groups=args.quality_groups,
                ),
                context=context,
            )
            jax.block_until_ready((result.evaluation.objective, result.gradient))
            times.append(perf_counter() - started)
        assert result is not None
        gradient = np.asarray(result.gradient)
        gradient_error = np.linalg.norm(gradient - exact_gradient)
        gradient_norm = max(np.linalg.norm(exact_gradient), np.finfo(float).eps)
        cosine_denominator = np.linalg.norm(gradient) * np.linalg.norm(exact_gradient)
        rows.append(
            {
                "requested_effort_percent": effort,
                "effective_effort_percent": result.fidelity.effective_effort_percent,
                "selected_shards": result.fidelity.selected_shard_count,
                "total_shards": result.fidelity.total_shard_count,
                "selected_support_entries": result.fidelity.selected_support_entries,
                "total_support_entries": result.fidelity.total_support_entries,
                "median_wall_seconds": float(np.median(times)),
                "peak_rss_bytes": _peak_rss_bytes(),
                "objective_absolute_error": abs(float(result.evaluation.objective) - exact_objective),
                "gradient_relative_norm_error": float(gradient_error / gradient_norm),
                "gradient_cosine_similarity": (
                    None
                    if cosine_denominator == 0.0
                    else float(np.dot(gradient, exact_gradient) / cosine_denominator)
                ),
                "predicted_count_relative_error": float(
                    np.linalg.norm(np.asarray(result.evaluation.measurement_mean) - exact_mean)
                    / max(np.linalg.norm(exact_mean), np.finfo(float).eps)
                ),
                "quality_score": result.quality.quality_score,
                "objective_standard_error_estimate": result.quality.objective_standard_error,
                "gradient_relative_error_estimate": result.quality.gradient_relative_error_estimate,
            }
        )
    return {
        "schema_version": 1,
        "public_synthetic_problem": {
            "cells": args.cells,
            "measurements": args.measurements,
            "shards": len(context.shards),
            "seed": args.seed,
        },
        "repetitions": args.repetitions,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=200)
    parser.add_argument("--measurements", type=int, default=60)
    parser.add_argument("--shards", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--quality-groups", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--efforts", type=float, nargs="+", default=(1, 2, 5, 10, 25, 50, 75, 100)
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
