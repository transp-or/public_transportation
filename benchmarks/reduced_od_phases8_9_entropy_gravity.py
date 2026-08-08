"""Scaling benchmark for sparse entropy and minimal JAX gravity."""

from __future__ import annotations

import argparse
import json
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.inference.reduced_od import (
    ConditionalGravityFeatures,
    EntropyConfig,
    EntropySupport,
    JourneyMarginals,
    MinimalGravityParameterLayout,
    MinimalGravityProblem,
    MinimalGravitySpecification,
    build_reduced_response_operator_from_coo,
    default_minimal_gravity_raw_parameters,
    estimate_entropy_transport,
    evaluate_minimal_gravity_objective,
)
from public_transportation.preprocessing.reduced_od import ResponseCellKey


def _entropy(cells: int) -> dict[str, int | float | str]:
    origins = max(1, int(np.sqrt(cells)))
    destinations = max(1, cells // origins)
    cells = origins * destinations
    origin_index = np.repeat(np.arange(origins), destinations)
    destination_index = np.tile(np.arange(destinations), origins)
    cost = (
        np.abs(origin_index / max(origins, 1) - destination_index / max(destinations, 1))
        + (origin_index * 17 + destination_index * 13) % 11 / 20.0
    )
    support = EntropySupport(
        origin_index=origin_index,
        destination_index=destination_index,
        generalized_cost=cost,
        number_of_origins=origins,
        number_of_destinations=destinations,
    )
    kernel = np.exp(-cost / 0.5)
    plan = (
        kernel
        * np.exp(np.linspace(-0.4, 0.4, origins))[origin_index]
        * np.exp(np.linspace(0.3, -0.3, destinations))[destination_index]
    )
    marginals = JourneyMarginals(
        origin=np.bincount(origin_index, weights=plan, minlength=origins),
        destination=np.bincount(
            destination_index, weights=plan, minlength=destinations
        ),
    )
    started = time.perf_counter()
    result = estimate_entropy_transport(
        support,
        marginals,
        config=EntropyConfig(epsilon=0.5, tolerance=1.0e-9),
    )
    elapsed = time.perf_counter() - started
    return {
        "kind": "balanced_entropy",
        "cells": cells,
        "origins": origins,
        "destinations": destinations,
        "iterations": result.diagnostics.iterations,
        "seconds": elapsed,
        "cells_per_second": cells / elapsed,
        "maximum_residual": max(
            result.diagnostics.maximum_origin_residual,
            result.diagnostics.maximum_destination_residual,
        ),
    }


def _gravity(cells: int, repeats: int) -> dict[str, int | float | str]:
    cells_per_group = 20
    groups = max(1, cells // cells_per_group)
    cells = groups * cells_per_group
    measurements = 2000
    group_index = np.repeat(np.arange(groups), cells_per_group)
    destination_index = np.tile(np.arange(cells_per_group), groups)
    cell_keys = tuple(
        ResponseCellKey(f"O{group:06d}", f"D{destination:04d}", "P")
        for group in range(groups)
        for destination in range(cells_per_group)
    )
    features = ConditionalGravityFeatures(
        cell_keys=cell_keys,
        origin_time_group_index=group_index,
        destination_index=destination_index,
        journey_time_seconds=300.0 + (np.arange(cells) * 37 % 3600),
        transfer_count=(np.arange(cells) * 7 % 4).astype(float),
        destination_attractiveness=1.0 + destination_index / cells_per_group,
        baseline_productions=np.full(groups, 100.0),
        origin_time_group_keys=tuple((f"O{group:06d}", "P") for group in range(groups)),
        destination_ids=tuple(f"D{destination:04d}" for destination in range(cells_per_group)),
    )
    columns = np.repeat(np.arange(cells), 3)
    offsets = np.tile(np.arange(3), cells)
    rows = (columns * 31 + offsets * 101) % measurements
    values = 0.5 + (columns + offsets) % 5 / 4.0
    operator = build_reduced_response_operator_from_coo(
        number_of_measurements=measurements,
        number_of_free_cells=cells,
        measurement_index=rows,
        free_cell_index=columns,
        response_values=values,
    )
    specification = MinimalGravitySpecification(likelihood="poisson")
    layout = MinimalGravityParameterLayout(specification)
    raw = jnp.asarray(default_minimal_gravity_raw_parameters(layout))
    provisional = MinimalGravityProblem(
        features=features,
        parameter_layout=layout,
        response_operator=operator,
        observations=np.ones(measurements),
    )
    observations = np.asarray(
        evaluate_minimal_gravity_objective(raw, problem=provisional).measurement_mean
    )
    problem = MinimalGravityProblem(
        features=features,
        parameter_layout=layout,
        response_operator=operator,
        observations=observations,
    )
    compiled = jax.jit(
        jax.value_and_grad(
            lambda parameters: evaluate_minimal_gravity_objective(
                parameters, problem=problem
            ).objective
        )
    )
    cold_started = time.perf_counter()
    value, gradient = compiled(raw)
    jax.block_until_ready(value)
    jax.block_until_ready(gradient)
    cold_seconds = time.perf_counter() - cold_started
    warm_started = time.perf_counter()
    for _ in range(repeats):
        value, gradient = compiled(raw)
        jax.block_until_ready(value)
        jax.block_until_ready(gradient)
    warm_seconds = (time.perf_counter() - warm_started) / repeats
    return {
        "kind": "minimal_gravity",
        "cells": cells,
        "groups": groups,
        "measurements": measurements,
        "parameters": layout.size,
        "operator_nnz": operator.original_nnz,
        "cold_value_gradient_seconds": cold_seconds,
        "warm_value_gradient_seconds": warm_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, nargs="+", default=[1000, 5000, 20000])
    parser.add_argument("--repeats", type=int, default=20)
    arguments = parser.parse_args()
    if any(value < 20 for value in arguments.cells) or arguments.repeats < 1:
        parser.error("cells must be at least 20 and repeats must be positive")
    results = [_entropy(cells) for cells in arguments.cells]
    results.extend(_gravity(cells, arguments.repeats) for cells in arguments.cells)
    results.append(
        {
            "kind": "process",
            "peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
        }
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
