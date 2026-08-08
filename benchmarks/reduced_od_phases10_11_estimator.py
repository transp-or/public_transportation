"""Operational benchmark for reduced-OD fitting and explicit reconstruction."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from public_transportation.inference.reduced_od import (
    ConditionalGravityFeatures,
    JourneyODTimeKey,
    MinimalGravityParameterLayout,
    MinimalGravityProblem,
    MinimalGravitySpecification,
    ReducedODFitConfig,
    ReducedODProblemContract,
    build_reduced_response_operator_from_coo,
    default_minimal_gravity_raw_parameters,
    estimate_minimal_gravity,
    evaluate_minimal_gravity_objective,
    reconstruct_full_od,
)
from public_transportation.preprocessing.reduced_od import ResponseCellKey


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=20_000)
    parser.add_argument("--maximum-iterations", type=int, default=100)
    arguments = parser.parse_args()
    cells_per_group = 20
    if arguments.cells < cells_per_group:
        parser.error("cells must be at least 20")
    groups = arguments.cells // cells_per_group
    cells = groups * cells_per_group
    group_index = np.repeat(np.arange(groups), cells_per_group)
    destination_index = np.tile(np.arange(cells_per_group), groups)
    response_keys = tuple(
        ResponseCellKey(f"O{group:06d}", f"D{destination:04d}", "P")
        for group in range(groups)
        for destination in range(cells_per_group)
    )
    features = ConditionalGravityFeatures(
        cell_keys=response_keys,
        origin_time_group_index=group_index,
        destination_index=destination_index,
        journey_time_seconds=300.0 + (np.arange(cells) * 37 % 3600),
        transfer_count=(np.arange(cells) * 7 % 4).astype(float),
        destination_attractiveness=1.0 + destination_index / cells_per_group,
        baseline_productions=np.full(groups, 100.0),
        origin_time_group_keys=tuple((f"O{group:06d}", "P") for group in range(groups)),
        destination_ids=tuple(
            f"D{destination:04d}" for destination in range(cells_per_group)
        ),
    )
    measurements = 2000
    columns = np.repeat(np.arange(cells), 3)
    offsets = np.tile(np.arange(3), cells)
    operator = build_reduced_response_operator_from_coo(
        number_of_measurements=measurements,
        number_of_free_cells=cells,
        measurement_index=(columns * 31 + offsets * 101) % measurements,
        free_cell_index=columns,
        response_values=0.5 + (columns + offsets) % 5 / 4.0,
    )
    layout = MinimalGravityParameterLayout(MinimalGravitySpecification())
    truth = default_minimal_gravity_raw_parameters(
        layout, beta_time=0.8, beta_transfer=1.2
    )
    provisional = MinimalGravityProblem(
        features=features,
        parameter_layout=layout,
        response_operator=operator,
        observations=np.ones(measurements),
    )
    observations = np.asarray(
        evaluate_minimal_gravity_objective(
            truth, problem=provisional
        ).measurement_mean
    )
    problem = MinimalGravityProblem(
        features=features,
        parameter_layout=layout,
        response_operator=operator,
        observations=observations,
    )
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "fit.json"
        fitted = estimate_minimal_gravity(
            problem=problem,
            initial_raw_parameters=np.asarray([-0.5, -0.5]),
            model_fingerprint="benchmark",
            config=ReducedODFitConfig(
                maximum_iterations=arguments.maximum_iterations,
                checkpoint_every_iterations=1,
            ),
            checkpoint_path=checkpoint,
        )
        checkpoint_bytes = checkpoint.stat().st_size
    free_demand = np.asarray(
        evaluate_minimal_gravity_objective(
            fitted.raw_parameters, problem=problem
        ).demand
    )
    contract = ReducedODProblemContract(
        configuration_fingerprint="benchmark-config",
        timetable_artifact_fingerprint="benchmark-timetable",
        response_artifact_fingerprint="benchmark-response",
        od_keys=tuple(JourneyODTimeKey(*key.tuple) for key in response_keys),
        free_od_indices=np.arange(cells),
        fixed_od_indices=np.asarray([], dtype=np.int64),
        fixed_od_values=np.asarray([], dtype=np.float64),
    )
    reconstruction_started = time.perf_counter()
    reconstructed = reconstruct_full_od(
        contract=contract,
        free_cell_keys=response_keys,
        free_demand=free_demand,
    )
    reconstruction_seconds = time.perf_counter() - reconstruction_started
    print(
        json.dumps(
            {
                "cells": cells,
                "measurements": measurements,
                "parameters": layout.size,
                "status": fitted.status,
                "iterations": fitted.iterations,
                "evaluations": fitted.evaluations,
                "compile_seconds": fitted.compile_seconds,
                "optimization_seconds": fitted.optimization_seconds,
                "checkpoint_bytes": checkpoint_bytes,
                "reconstruction_seconds": reconstruction_seconds,
                "reconstructed_bytes": reconstructed.demand.nbytes,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
