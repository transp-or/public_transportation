"""Benchmark reduced-OD adequacy, grouped holdout, and advisory scoring."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from public_transportation.inference.reduced_od import (
    ConditionalGravityFeatures,
    MinimalGravityParameterLayout,
    MinimalGravityProblem,
    MinimalGravitySpecification,
    ReducedODHoldoutConfig,
    ReducedODMeasurementMetadata,
    build_reduced_od_holdout_split,
    build_reduced_response_operator_from_coo,
    default_minimal_gravity_raw_parameters,
    diagnose_reduced_od_adequacy,
    estimate_minimal_gravity,
    evaluate_minimal_gravity_objective,
    recommend_reduced_od_relaxations,
    validate_reduced_od_holdout,
)
from public_transportation.preprocessing.reduced_od import ResponseCellKey


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=20_000)
    arguments = parser.parse_args()
    cells_per_group = 20
    groups = arguments.cells // cells_per_group
    if groups < 2:
        parser.error("cells must be at least 40")
    cells = groups * cells_per_group
    group_index = np.repeat(np.arange(groups), cells_per_group)
    destination_index = np.tile(np.arange(cells_per_group), groups)
    features = ConditionalGravityFeatures(
        cell_keys=tuple(
            ResponseCellKey(f"O{group:06d}", f"D{destination:04d}", "P")
            for group in range(groups)
            for destination in range(cells_per_group)
        ),
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
    metadata = ReducedODMeasurementMetadata(
        number_of_measurements=measurements,
        measurement_type=np.tile(np.asarray(["boarding", "alighting"]), measurements // 2),
        line=np.asarray([f"L{index % 20:02d}" for index in range(measurements)]),
        time_period=np.asarray([f"P{index % 4}" for index in range(measurements)]),
        vehicle_journey=np.asarray([f"V{index // 10:04d}" for index in range(measurements)]),
        origin_zone=np.asarray([f"OZ{index % 10}" for index in range(measurements)]),
        destination_zone=np.asarray([f"DZ{index % 10}" for index in range(measurements)]),
        transfer_place=np.asarray([f"T{index % 8}" for index in range(measurements)]),
    )
    fitted = estimate_minimal_gravity(
        problem=problem,
        initial_raw_parameters=np.asarray([-0.5, -0.5]),
        model_fingerprint="phase12-benchmark",
    )
    started = time.perf_counter()
    adequacy = diagnose_reduced_od_adequacy(
        fit=fitted, problem=problem, metadata=metadata
    )
    cold_adequacy_seconds = time.perf_counter() - started
    started = time.perf_counter()
    adequacy = diagnose_reduced_od_adequacy(
        fit=fitted, problem=problem, metadata=metadata
    )
    warm_adequacy_seconds = time.perf_counter() - started
    started = time.perf_counter()
    recommendations = recommend_reduced_od_relaxations(
        adequacy=adequacy, metadata=metadata
    )
    recommendation_seconds = time.perf_counter() - started
    split = build_reduced_od_holdout_split(
        metadata=metadata,
        measurement_identity="phase12-benchmark",
        config=ReducedODHoldoutConfig(
            unit="vehicle_journey", fraction=0.2, seed=0
        ),
    )
    started = time.perf_counter()
    holdout = validate_reduced_od_holdout(
        problem=problem,
        initial_raw_parameters=fitted.raw_parameters,
        model_fingerprint="phase12-benchmark",
        metadata=metadata,
        split=split,
    )
    holdout_seconds = time.perf_counter() - started
    print(
        json.dumps(
            {
                "cells": cells,
                "measurements": measurements,
                "cold_adequacy_seconds": cold_adequacy_seconds,
                "warm_adequacy_seconds": warm_adequacy_seconds,
                "group_summaries": len(adequacy.grouped_summaries),
                "recommendation_seconds": recommendation_seconds,
                "recommendation_candidates": len(recommendations.candidates),
                "holdout_refit_seconds": holdout_seconds,
                "holdout_measurements": holdout.holdout.measurements,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
