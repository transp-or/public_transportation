"""Scalable public benchmark for persisted sharded gravity products."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter, process_time

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.inference.assignment_adapter import AssignmentInputs
from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)
from public_transportation.inference.gravity import (
    GravityFeatures,
    GravityLikelihood,
    GravityModelSpecification,
    GravityObjectiveProblem,
    GravityParameterLayout,
    GravityPreflightPhase,
    run_gravity_preflight,
)
from public_transportation.inference.gravity.demand import generate_gravity_demand
from public_transportation.inference.gravity.objective import _evaluation_from_mean
from public_transportation.inference.sharded_fixed_routing import (
    FixedRoutingPreparationConfig,
    prepare_fixed_routing_sharded,
)
from public_transportation.inference.sharded_matrix_free_operator import (
    ShardedMatrixFreeFixedRoutingMeasurementOperator,
)
from public_transportation.measurement.mapping import AggregationSpec

from benchmarks.benchmark_fixed_routing_scaling import _dag


def _inputs(*, nodes: int, degree: int, groups: int, od_cells: int) -> AssignmentInputs:
    if od_cells < groups:
        raise ValueError("od_cells must be at least destination_groups.")
    graph = _dag(num_nodes=nodes, maximum_out_degree=degree)
    group = np.arange(od_cells) % groups
    width = int(np.max(np.bincount(group)))
    indices = np.zeros((groups, width), dtype=np.int32)
    valid = np.zeros((groups, width), dtype=bool)
    for group_index in range(groups):
        cells = np.flatnonzero(group == group_index)
        indices[group_index, : cells.size] = cells
        valid[group_index, : cells.size] = True
    return AssignmentInputs(
        graph=graph,
        base_link_cost=jnp.ones(graph.num_links, dtype=jnp.float32),
        group_dest_node=jnp.full(groups, nodes - 1, dtype=jnp.int32),
        group_link_mask=jnp.ones((groups, graph.num_links), dtype=bool),
        od_origin_node=jnp.asarray(np.arange(od_cells) % max(1, nodes // 2)),
        group_od_index_padded=jnp.asarray(indices),
        group_od_mask=jnp.asarray(valid),
    )


def _layout(od_cells: int) -> CompactODAssignmentLayout:
    indices = tuple(range(od_cells))
    return CompactODAssignmentLayout(
        od_cells, indices, (), indices, indices, indices, (1.0,) * od_cells, (), ()
    )


def _problem(
    operator: ShardedMatrixFreeFixedRoutingMeasurementOperator,
) -> tuple[GravityObjectiveProblem, np.ndarray]:
    cells = operator.num_free_od
    origin_groups = min(16, cells)
    group = np.arange(cells) % origin_groups
    features = GravityFeatures(
        canonical_od_index=np.arange(cells),
        origin_index=group,
        destination_index=np.arange(cells),
        departure_time_index=group % 4,
        origin_time_group_index=group,
        journey_time=np.linspace(2.0, 60.0, cells, dtype=np.float32),
        transfer_count=np.arange(cells) % 4,
        structural_feasible=np.ones(cells, dtype=bool),
        origin_time_totals=np.linspace(80.0, 180.0, origin_groups, dtype=np.float32),
        destination_attractiveness=np.linspace(0.5, 2.0, cells, dtype=np.float32),
        num_origins=origin_groups,
        num_destinations=cells,
        num_departure_times=4,
        od_layout_fingerprint=operator.compact_layout_fingerprint,
        journey_time_scale=30.0,
    )
    parameters = GravityParameterLayout(GravityModelSpecification())
    raw = np.zeros(parameters.size, dtype=np.float32)
    observations = np.ones(operator.num_measurements, dtype=np.float32)
    return (
        GravityObjectiveProblem(
            features=features,
            parameter_layout=parameters,
            operator=operator,
            observations=observations,
            likelihood=GravityLikelihood.NEGATIVE_BINOMIAL,
        ),
        raw,
    )


def _delta(before: object, after: object) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in asdict(after).items():
        previous = getattr(before, name)
        result[name] = value if value is None or previous is None else value - previous
    return result


def _measure(operator, operation: str, value: np.ndarray) -> dict[str, object]:
    before = operator.metrics
    wall_started = perf_counter()
    cpu_started = process_time()
    result = getattr(operator, operation)(value)
    elapsed = perf_counter() - wall_started
    cpu = process_time() - cpu_started
    metrics = _delta(before, operator.metrics)
    metrics["peak_rss_bytes"] = operator.metrics.peak_rss_bytes
    metrics["resident_routing_bytes"] = operator.metrics.resident_routing_bytes
    metrics["effective_cpu_cores"] = operator.metrics.effective_cpu_cores
    shards = operator.routing.num_shards
    metrics.update(
        total_elapsed_seconds=elapsed,
        process_cpu_seconds_measured=cpu,
        effective_cpu_cores_measured=cpu / max(elapsed, np.finfo(float).eps),
        shards_per_second=shards / max(elapsed, np.finfo(float).eps),
        groups_per_second=operator.routing.num_destination_groups
        / max(elapsed, np.finfo(float).eps),
        result_norm=float(np.linalg.norm(result)),
    )
    return metrics


def _objective_decomposition(problem, raw: np.ndarray) -> dict[str, float]:
    warm_demand = generate_gravity_demand(
        raw,
        features=problem.features,
        parameter_layout=problem.parameter_layout,
    ).demand
    warm_routed = problem.operator.jax_matvec(warm_demand)
    warm_mean = problem.rho * (
        warm_routed + jnp.asarray(problem.operator.fixed_measurement_offset)
    )
    warm_evaluation = _evaluation_from_mean(
        jnp.asarray(raw),
        mean=warm_mean,
        demand=warm_demand,
        problem=problem,
    )
    jax.block_until_ready(warm_evaluation)
    started = perf_counter()
    demand = generate_gravity_demand(
        raw,
        features=problem.features,
        parameter_layout=problem.parameter_layout,
    ).demand
    jax.block_until_ready(demand)
    demand_seconds = perf_counter() - started
    started = perf_counter()
    routed = problem.operator.jax_matvec(demand)
    jax.block_until_ready(routed)
    forward_seconds = perf_counter() - started
    mean = problem.rho * (
        routed + jnp.asarray(problem.operator.fixed_measurement_offset)
    )
    started = perf_counter()
    evaluation = _evaluation_from_mean(
        jnp.asarray(raw), mean=mean, demand=demand, problem=problem
    )
    jax.block_until_ready(evaluation)
    likelihood_seconds = perf_counter() - started
    return {
        "demand_generation_seconds": demand_seconds,
        "forward_operator_seconds": forward_seconds,
        "likelihood_and_regularization_seconds": likelihood_seconds,
    }


def run_benchmark(
    *,
    nodes: int = 128,
    maximum_out_degree: int = 2,
    destination_groups: int = 16,
    groups_per_shard: int = 2,
    od_cells: int = 128,
    measurements: int = 64,
    resident_shard_limit: int = 2,
    operator_batch_sizes: tuple[int, ...] = (1, 2, 4, 8),
) -> dict[str, object]:
    inputs = _inputs(
        nodes=nodes,
        degree=maximum_out_degree,
        groups=destination_groups,
        od_cells=od_cells,
    )
    compact = _layout(od_cells)
    spec = AggregationSpec(
        num_measurements=measurements,
        measurement_index=np.arange(measurements, dtype=np.int32),
        link_index=np.arange(measurements, dtype=np.int32) % inputs.graph.num_links,
    )
    with tempfile.TemporaryDirectory(prefix="sharded-gravity-benchmark-") as temporary:
        root = Path(temporary)
        prepared = prepare_fixed_routing_sharded(
            inputs=inputs,
            theta=1.0,
            config=FixedRoutingPreparationConfig(
                maximum_groups_per_shard=groups_per_shard,
                cache_directory=root / "routing",
                checkpoint_directory=root / "checkpoints",
                resident_shard_limit=resident_shard_limit,
            ),
        )
        cases = []
        for batch_size in operator_batch_sizes:
            operator = ShardedMatrixFreeFixedRoutingMeasurementOperator(
                inputs=inputs,
                routing=prepared.routing,
                spec=spec,
                compact_layout=compact,
                resident_shard_limit=resident_shard_limit,
                operator_shards_per_batch=batch_size,
            )
            demand = np.linspace(0.1, 2.0, od_cells, dtype=np.float32)
            cotangent = np.linspace(-1.0, 1.0, measurements, dtype=np.float32)
            jacobian = np.column_stack((demand, 0.5 * demand, -demand))
            operator.matvec(demand)
            operator.rmatvec(cotangent)
            operator._host_matmat(jacobian)
            products = {
                "matvec": _measure(operator, "matvec", demand),
                "rmatvec": _measure(operator, "rmatvec", cotangent),
                "matmat": _measure(operator, "_host_matmat", jacobian),
            }
            problem, raw = _problem(operator)
            decomposition = _objective_decomposition(problem, raw)
            preflight = run_gravity_preflight(
                problem=problem,
                raw_parameters=raw,
                stop_after=GravityPreflightPhase.RECOMMENDATION,
            )
            assert preflight.recommendation is not None
            recommendation = asdict(preflight.recommendation)
            recommendation["gradient_strategy"] = (
                preflight.recommendation.gradient_strategy.value
            )
            cases.append(
                {
                    "operator_shards_per_batch": batch_size,
                    "products": products,
                    "objective_gradient": {
                        "timings_seconds": dict(preflight.timings_seconds),
                        "decomposition_seconds": {
                            **decomposition,
                            "reverse_operator_seconds": products["rmatvec"][
                                "total_elapsed_seconds"
                            ],
                            "jacobian_operator_seconds": products["matmat"][
                                "total_elapsed_seconds"
                            ],
                        },
                        "gradient_max_abs_difference": (
                            preflight.gradient_max_abs_difference
                        ),
                        "recommendation": recommendation,
                    },
                }
            )
    fastest = min(
        cases,
        key=lambda case: sum(
            product["total_elapsed_seconds"]
            for product in case["products"].values()
        ),
    )
    return {
        "schema_version": 1,
        "backend": jax.default_backend(),
        "nodes": nodes,
        "links": inputs.graph.num_links,
        "destination_groups": destination_groups,
        "groups_per_shard": groups_per_shard,
        "routing_shards": prepared.routing.num_shards,
        "od_cells": od_cells,
        "measurements": measurements,
        "resident_shard_limit": resident_shard_limit,
        "cases": cases,
        "best_operator_shards_per_batch": fastest["operator_shards_per_batch"],
        "dense_measurement_od_constructed": False,
        "complete_routing_array_materialized": False,
        "measurement_aggregation_timing": "fused into compiled forward execution",
        "archive_decompression_timing": "zero because routing shards use uncompressed npz",
        "persistent_compilation_cache_note": (
            "JAX exposes no authoritative public persistent-cache hit counter; "
            "compilation counts and times are reported without inferring hits."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=128)
    parser.add_argument("--maximum-out-degree", type=int, default=2)
    parser.add_argument("--destination-groups", type=int, default=16)
    parser.add_argument("--groups-per-shard", type=int, default=2)
    parser.add_argument("--od-cells", type=int, default=128)
    parser.add_argument("--measurements", type=int, default=64)
    parser.add_argument("--resident-shard-limit", type=int, default=2)
    parser.add_argument("--operator-batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_benchmark(
        nodes=arguments.nodes,
        maximum_out_degree=arguments.maximum_out_degree,
        destination_groups=arguments.destination_groups,
        groups_per_shard=arguments.groups_per_shard,
        od_cells=arguments.od_cells,
        measurements=arguments.measurements,
        resident_shard_limit=arguments.resident_shard_limit,
        operator_batch_sizes=tuple(arguments.operator_batch_sizes),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
