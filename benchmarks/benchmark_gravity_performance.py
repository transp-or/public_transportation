"""Public synthetic benchmark for reduced-dimensional gravity estimation."""

from __future__ import annotations

import argparse
import json
import resource
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse as jsparse

from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)
from public_transportation.inference.fixed_routing_measurement_operator import (
    FixedRoutingMeasurementOperator,
    MeasurementOperatorMetrics,
)
from public_transportation.inference.gravity import (
    GravityEffectScope,
    GravityFeatures,
    GravityGradientStrategy,
    GravityLikelihood,
    GravityModelSpecification,
    GravityObjectiveProblem,
    GravityParameterLayout,
    gravity_value_and_gradient,
)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if value > 10_000_000 else value * 1024


def _timed(call):
    started = perf_counter()
    value = call()
    jax.block_until_ready(value)
    return perf_counter() - started, value


def _warm_median(call, repeats: int) -> float:
    return median(_timed(call)[0] for _ in range(repeats))


def _problem(
    *, parameter_count: int, representation: str, od_cells: int, measurements: int
) -> tuple[GravityObjectiveProblem, np.ndarray]:
    if parameter_count != 3 and parameter_count < 4:
        raise ValueError("parameter_count must be three or at least four.")
    groups = 16
    if od_cells % groups:
        raise ValueError("od_cells must be divisible by 16.")
    alternatives = od_cells // groups
    compact = CompactODAssignmentLayout(
        od_cells,
        tuple(range(od_cells)),
        (),
        tuple(range(od_cells)),
        tuple(range(od_cells)),
        tuple(range(od_cells)),
        tuple(1.0 for _ in range(od_cells)),
        (),
        (),
    )
    zone_count = 0 if parameter_count == 3 else parameter_count - 2
    destination_zones = None
    specification = GravityModelSpecification()
    if zone_count:
        if zone_count > od_cells:
            raise ValueError("parameter_count is too large for the requested OD cells.")
        destination_zones = np.arange(od_cells) % zone_count
        specification = replace(
            specification,
            destination_attractiveness_scope=GravityEffectScope.DESTINATION_ZONE,
            destination_zone_count=zone_count,
        )
    features = GravityFeatures(
        canonical_od_index=np.arange(od_cells),
        origin_index=np.repeat(np.arange(groups), alternatives),
        destination_index=np.tile(np.arange(alternatives), groups),
        departure_time_index=np.repeat(np.arange(groups) % 4, alternatives),
        origin_time_group_index=np.repeat(np.arange(groups), alternatives),
        journey_time=np.linspace(2.0, 60.0, od_cells, dtype=np.float32),
        transfer_count=np.arange(od_cells) % 4,
        structural_feasible=np.ones(od_cells, dtype=bool),
        origin_time_totals=np.linspace(80.0, 180.0, groups, dtype=np.float32),
        destination_attractiveness=np.linspace(0.5, 2.0, od_cells, dtype=np.float32),
        num_origins=groups,
        num_destinations=alternatives,
        num_departure_times=4,
        od_layout_fingerprint=compact.fingerprint,
        journey_time_scale=30.0,
        destination_zone_index=destination_zones,
    )
    rng = np.random.default_rng(20260801)
    dense_numpy = rng.uniform(0.0, 1.0, (measurements, od_cells)).astype(np.float32)
    dense_numpy[rng.uniform(size=dense_numpy.shape) > 0.08] = 0.0
    dense_numpy[:, 0] += 0.05
    dense = jnp.asarray(dense_numpy)
    matrix = dense if representation == "dense" else jsparse.BCOO.fromdense(dense)
    nonzero = int(np.count_nonzero(dense_numpy))
    stored = int(dense.nbytes) if representation == "dense" else nonzero * 12
    operator = FixedRoutingMeasurementOperator(
        matrix,
        jnp.full(measurements, 0.25, dtype=jnp.float32),
        representation,
        od_cells,
        od_cells,
        measurements,
        compact.fingerprint,
        compact.fingerprint,
        "synthetic-assignment",
        "synthetic-graph",
        "synthetic-mapping",
        1.0,
        "float32",
        MeasurementOperatorMetrics(
            0.0,
            int(dense.nbytes),
            stored,
            0,
            nonzero,
            int(dense.size),
            nonzero / dense.size,
            od_cells,
        ),
    )
    layout = GravityParameterLayout(specification)
    raw = np.zeros(layout.size, dtype=np.float32)
    seed = GravityObjectiveProblem(
        features,
        layout,
        operator,
        np.ones(measurements, dtype=np.float32),
        GravityLikelihood.NEGATIVE_BINOMIAL,
    )
    from public_transportation.inference.gravity import predict_gravity_measurements

    observations = np.maximum(
        0.0, np.rint(np.asarray(predict_gravity_measurements(raw, problem=seed)[0]))
    )
    return replace(seed, observations=observations), raw


def _strategy_metrics(
    problem: GravityObjectiveProblem,
    raw: np.ndarray,
    strategy: GravityGradientStrategy,
    repeats: int,
) -> tuple[dict[str, object], np.ndarray]:
    function = jax.jit(
        lambda value: gravity_value_and_gradient(
            value, problem=problem, strategy=strategy
        )
    )
    raw_jax = jnp.asarray(raw)
    rss_before = _peak_rss_bytes()
    started = perf_counter()
    traced = function.trace(raw_jax)
    tracing = perf_counter() - started
    started = perf_counter()
    lowered = traced.lower()
    lowering = perf_counter() - started
    lowered_bytes = len(lowered.as_text().encode("utf-8"))
    started = perf_counter()
    compiled = lowered.compile()
    compilation = perf_counter() - started
    first_seconds, first = _timed(lambda: compiled(raw_jax))
    warm_seconds = _warm_median(lambda: compiled(raw_jax), repeats)
    changed_seconds, changed = _timed(lambda: compiled(raw_jax + 0.01))
    rss_after = _peak_rss_bytes()
    first_gradient = np.asarray(first[1])
    changed_gradient = np.asarray(changed[1])
    return (
        {
            "strategy": strategy.value,
            "tracing_seconds": tracing,
            "lowering_seconds": lowering,
            "compilation_seconds": compilation,
            "first_execution_seconds": first_seconds,
            "warm_execution_seconds": warm_seconds,
            "changed_parameter_execution_seconds": changed_seconds,
            "lowered_text_bytes": lowered_bytes,
            "peak_host_rss_bytes": rss_after,
            "peak_host_rss_growth_bytes": max(0, rss_after - rss_before),
            "compiled_kernel_cache_misses": 1,
            "compiled_kernel_cache_hits": repeats + 1,
            "parameter_value_reuse_verified": bool(
                first_gradient.shape == changed_gradient.shape
                and not np.array_equal(first_gradient, changed_gradient)
            ),
            "persistent_cache_hit_status": "requires_fresh_process_protocol",
        },
        first_gradient,
    )


def _routing_metrics(problem: GravityObjectiveProblem, repeats: int) -> dict[str, float]:
    demand = jnp.ones(problem.operator.num_free_od, dtype=jnp.float32)
    cotangent = jnp.ones(problem.operator.num_measurements, dtype=jnp.float32)
    forward = jax.jit(problem.operator.jax_matvec)
    transpose = jax.jit(problem.operator.jax_rmatvec)
    first_forward, _ = _timed(lambda: forward(demand))
    first_transpose, _ = _timed(lambda: transpose(cotangent))
    return {
        "forward_routing_first_seconds": first_forward,
        "forward_routing_warm_seconds": _warm_median(lambda: forward(demand), repeats),
        "transpose_routing_first_seconds": first_transpose,
        "transpose_routing_warm_seconds": _warm_median(
            lambda: transpose(cotangent), repeats
        ),
    }


def run_benchmark(
    *,
    parameter_counts: tuple[int, ...] = (3, 7, 15, 31),
    representations: tuple[str, ...] = ("dense", "bcoo"),
    od_cells: int = 128,
    measurements: int = 96,
    repeats: int = 5,
) -> dict[str, object]:
    if not parameter_counts or len(set(parameter_counts)) != len(parameter_counts):
        raise ValueError("parameter_counts must be nonempty and unique.")
    if any(value < 3 for value in parameter_counts):
        raise ValueError("parameter counts must be at least three.")
    if not representations or any(value not in ("dense", "bcoo") for value in representations):
        raise ValueError("representations must contain dense and/or bcoo.")
    if measurements <= 0 or repeats <= 0:
        raise ValueError("measurements and repeats must be positive.")
    cases = []
    for representation in representations:
        for parameter_count in parameter_counts:
            problem, raw = _problem(
                parameter_count=parameter_count,
                representation=representation,
                od_cells=od_cells,
                measurements=measurements,
            )
            forward, forward_gradient = _strategy_metrics(
                problem, raw, GravityGradientStrategy.BATCHED_FORWARD, repeats
            )
            adjoint, adjoint_gradient = _strategy_metrics(
                problem, raw, GravityGradientStrategy.ADJOINT, repeats
            )
            np.testing.assert_allclose(
                forward_gradient, adjoint_gradient, rtol=3e-4, atol=3e-4
            )
            strategies = (forward, adjoint)
            fastest = min(
                strategies, key=lambda item: float(item["warm_execution_seconds"])
            )
            cases.append(
                {
                    "representation": representation,
                    "parameter_count": parameter_count,
                    "od_cell_count": od_cells,
                    "measurement_count": measurements,
                    "operator_stored_bytes": problem.operator.metrics.stored_bytes,
                    **_routing_metrics(problem, repeats),
                    "strategies": strategies,
                    "maximum_strategy_gradient_difference": float(
                        np.max(np.abs(forward_gradient - adjoint_gradient), initial=0.0)
                    ),
                    "fastest_warm_strategy": fastest["strategy"],
                }
            )
    return {
        "schema_version": 1,
        "backend": jax.default_backend(),
        "dtype": "float32",
        "parameter_counts": parameter_counts,
        "representations": representations,
        "repeats": repeats,
        "persistent_cache_note": (
            "Persistent-cache hits and misses require separate fresh processes; "
            "this benchmark reports in-process compiled-handle reuse explicitly."
        ),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-counts", nargs="+", type=int, default=[3, 7, 15, 31])
    parser.add_argument("--representations", nargs="+", default=["dense", "bcoo"])
    parser.add_argument("--od-cells", type=int, default=128)
    parser.add_argument("--measurements", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_benchmark(
        parameter_counts=tuple(arguments.parameter_counts),
        representations=tuple(arguments.representations),
        od_cells=arguments.od_cells,
        measurements=arguments.measurements,
        repeats=arguments.repeats,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(rendered)
    else:
        arguments.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
