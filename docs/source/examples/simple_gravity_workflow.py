"""Reusable gravity-estimation workflow for the two synthetic examples.

This module deliberately keeps the orchestration visible: it prepares gravity
features, builds the fixed-routing measurement operator, estimates the minimal
model, checks adequacy, and optionally validates one model relaxation on a
grouped holdout sample.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.domain import Scenario, read_fixed_demand_csv
from public_transportation.inference.assignment_adapter import (
    build_assignment_inputs,
    prepare_fixed_routing,
)
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.fixed_routing_measurement_operator import (
    load_or_prepare_fixed_routing_measurement_operator,
)
from public_transportation.inference.gravity import (
    GravityEffectScope,
    GravityEstimatorConfig,
    GravityExecutionPolicy,
    GravityFeatures,
    GravityHoldoutSplitConfig,
    GravityLikelihood,
    GravityModelLineage,
    GravityModelSpecification,
    GravityObjectiveProblem,
    GravityParameterLayout,
    GravityValidationMetadata,
    build_gravity_holdout_split,
    create_gravity_model_node,
    estimate_and_validate_gravity_holdout,
    estimate_gravity_model,
    gravity_measurement_identity,
    progress_gravity_model_lineage,
    validate_full_data_gravity_adequacy,
)
from public_transportation.inference.od_parameter_layout import (
    build_od_parameter_layout,
)
from public_transportation.measurement import (
    build_mapping_spec_strict,
    read_measurements_csv,
)
from public_transportation.preprocessing import (
    ODTimeKey,
    build_structural_zero_topology,
    compute_od_path_metrics,
)
from public_transportation.preprocessing.structural_zeros.config import (
    StructuralZeroAssignmentConfig,
)


def _features(scenario, od_layout, compact) -> GravityFeatures:
    """Prepare the externally defined totals, attractiveness, and path metrics."""
    free_indices = np.asarray(od_layout.free_od_indices, dtype=np.int64)
    keys = [od_layout.od_keys[index] for index in free_indices]
    topology = build_structural_zero_topology(
        scenario, StructuralZeroAssignmentConfig()
    )
    records = compute_od_path_metrics(
        topology, keys=tuple(ODTimeKey(*key) for key in keys)
    )
    metrics = {record.key.tuple: record.metrics for record in records}
    if any(not metrics[key].feasible for key in keys):
        raise ValueError("Every free gravity cell must be scheduled-feasible.")

    origins = sorted({key[0] for key in keys})
    destinations = sorted({key[1] for key in keys})
    periods = sorted({key[2] for key in keys})
    origin_lookup = {value: index for index, value in enumerate(origins)}
    destination_lookup = {value: index for index, value in enumerate(destinations)}
    period_lookup = {value: index for index, value in enumerate(periods)}
    group_pairs = sorted({(key[0], key[2]) for key in keys})
    group_lookup = {value: index for index, value in enumerate(group_pairs)}

    baseline = np.asarray(od_layout.free_baseline_values, dtype=np.float32)
    totals = np.zeros(len(group_pairs), dtype=np.float32)
    destination_totals: dict[tuple[str, str], float] = {}
    for key, value in zip(keys, baseline, strict=True):
        totals[group_lookup[(key[0], key[2])]] += value
        destination_key = (key[1], key[2])
        destination_totals[destination_key] = (
            destination_totals.get(destination_key, 0.0) + float(value)
        )

    return GravityFeatures(
        canonical_od_index=free_indices,
        origin_index=np.asarray([origin_lookup[key[0]] for key in keys]),
        destination_index=np.asarray(
            [destination_lookup[key[1]] for key in keys]
        ),
        departure_time_index=np.asarray([period_lookup[key[2]] for key in keys]),
        origin_time_group_index=np.asarray(
            [group_lookup[(key[0], key[2])] for key in keys]
        ),
        journey_time=np.asarray(
            [metrics[key].minimum_journey_time_minutes for key in keys],
            dtype=np.float32,
        ),
        transfer_count=np.asarray(
            [metrics[key].minimum_transfers for key in keys], dtype=np.int64
        ),
        structural_feasible=np.ones(len(keys), dtype=bool),
        origin_time_totals=totals,
        destination_attractiveness=np.asarray(
            [destination_totals[(key[1], key[2])] for key in keys],
            dtype=np.float32,
        ),
        num_origins=len(origins),
        num_destinations=len(destinations),
        num_departure_times=len(periods),
        od_layout_fingerprint=compact.fingerprint,
        journey_time_scale=30.0,
        time_period_index=np.asarray(
            [0 if period_lookup[key[2]] < len(periods) / 2 else 1 for key in keys]
        ),
    )


def _metadata(measurement_path: Path) -> GravityValidationMetadata:
    with measurement_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    trips = [str(row["trip_id"]) for row in rows]
    hours = [int(str(row["time"]).split(":", 1)[0]) for row in rows]
    split_hour = int(np.median(hours))
    return GravityValidationMetadata(
        len(rows),
        measurement_type=np.asarray([row["measurement_type"] for row in rows]),
        line=np.asarray([trip.split("_", 1)[0] for trip in trips]),
        stop=np.asarray([row["stop_id"] for row in rows]),
        time_period=np.asarray(
            ["early" if hour <= split_hour else "late" for hour in hours]
        ),
        vehicle_journey=np.asarray(trips),
    )


def _true_demand(path: Path, od_keys) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as stream:
        values = {
            (row["origin_stop_id"], row["dest_stop_id"], row["time_bin_id"]): float(
                row["flow"]
            )
            for row in csv.DictReader(stream)
        }
    return np.asarray([values[key] for key in od_keys], dtype=np.float64)


def run_simple_gravity_workflow(
    *,
    example: Path,
    routing_parameter: float,
    maximum_iterations: int,
    operator_cache_directory: Path,
    include_relaxation_and_holdout: bool,
) -> dict[str, object]:
    """Run and summarize an end-to-end synthetic gravity estimation."""
    label = example.name

    def stage(message: str) -> None:
        print(f"[{label}-gravity] {message}", flush=True)

    data = example / "data"
    measurements = (
        example / "pre_processing/results/measurements_boarding_alighting.csv"
    )
    stage("load scenario and frozen-cell layout")
    scenario = Scenario.from_folder(
        data, strict=True, demand_file=data / "prior_demand.csv"
    )
    od_layout = build_od_parameter_layout(
        scenario=scenario,
        fixed_demand=read_fixed_demand_csv(
            data / "fixed_demand.csv", scenario=scenario
        ),
    )
    compact = build_compact_od_assignment_layout(parameter_layout=od_layout)
    features = _features(scenario, od_layout, compact)

    stage("prepare assignment, observations, and cached measurement operator")
    artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
    inputs = build_assignment_inputs(artifacts=artifacts, compact_layout=compact)
    routing = prepare_fixed_routing(inputs=inputs, theta=routing_parameter)
    id_manager = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)
    mapped = build_mapping_spec_strict(
        id_manager=id_manager,
        table=read_measurements_csv(measurements),
        include_link_lists_for_report=False,
    )
    operator = load_or_prepare_fixed_routing_measurement_operator(
        cache_directory=operator_cache_directory,
        inputs=inputs,
        routing=routing,
        spec=mapped.spec,
        assignment_fingerprint=str(id_manager.fingerprint),
        compact_layout=compact,
        od_layout_fingerprint=od_layout.fingerprint,
        representation="bcoo",
        chunk_size=min(16, compact.num_free),
    )

    parameter_layout = GravityParameterLayout(GravityModelSpecification())
    problem = GravityObjectiveProblem(
        features=features,
        parameter_layout=parameter_layout,
        operator=operator,
        observations=np.asarray(mapped.y_obs),
        likelihood=GravityLikelihood.NEGATIVE_BINOMIAL,
    )
    initial = parameter_layout.raw_from_physical((0.5, 1.0, 10.0))
    config = GravityEstimatorConfig(maximum_iterations=maximum_iterations)
    execution = GravityExecutionPolicy(gradient_strategy="adjoint")
    stage("estimate minimal gravity model")
    result = estimate_gravity_model(
        problem=problem,
        compact_layout=compact,
        initial_raw_parameters=initial,
        config=config,
        execution=execution,
    )
    metadata = _metadata(measurements)
    adequacy = validate_full_data_gravity_adequacy(
        result=result,
        problem=problem,
        compact_layout=compact,
        metadata=metadata,
    )
    truth = _true_demand(data / "true_demand.csv", od_layout.od_keys)
    error = np.asarray(result.full_od_demand) - truth
    report: dict[str, object] = {
        "schema_version": 1,
        "example": label,
        "backend": jax.default_backend(),
        "num_od_total": compact.num_od_total,
        "num_free_od": compact.num_free,
        "num_fixed_zero": compact.num_removed_zero,
        "num_fixed_positive": compact.num_fixed_positive,
        "num_measurements": operator.num_measurements,
        "operator": {
            "representation": operator.representation,
            "cache_hit": operator.metrics.cache_hit,
            "density": operator.metrics.density,
            "stored_bytes": operator.metrics.stored_bytes,
        },
        "minimal_model": {
            "status": result.status,
            "iterations": result.iterations,
            "physical_parameters": result.physical_parameters.tolist(),
            "objective": result.objective,
            "negative_binomial_deviance": adequacy.negative_binomial_deviance,
            "measurement_rmse": adequacy.rmse,
            "demand_rmse": float(np.sqrt(np.mean(np.square(error)))),
            "fixed_cell_maximum_error": float(
                np.max(
                    np.abs(error[np.asarray(od_layout.fixed_od_indices)]),
                    initial=0.0,
                )
            ),
        },
    }
    if not include_relaxation_and_holdout:
        return report

    stage("relax the temporal specification and validate grouped holdout")
    identity = gravity_measurement_identity(
        measurement_indices=np.arange(operator.num_measurements),
        label=f"{label} synthetic boarding/alighting rows v1",
    )
    node = create_gravity_model_node(
        result=result,
        problem=problem,
        compact_layout=compact,
        adequacy_report=adequacy,
        calibration_measurement_identity=identity,
        validation_measurement_identity=identity,
    )
    progression = progress_gravity_model_lineage(
        lineage=GravityModelLineage((node,)),
        parent_identifier=node.model_identifier,
        selected_relaxation=GravityEffectScope.TIME_PERIOD,
        problem=problem,
        compact_layout=compact,
        parent_adequacy_report=adequacy,
        calibration_measurement_identity=identity,
        validation_measurement_identity=identity,
        estimator_config=config,
        execution_policy=execution,
        validation_metadata=metadata,
    )
    child_problem = replace(
        problem, parameter_layout=progression.warm_start.child_parameter_layout
    )
    split = build_gravity_holdout_split(
        metadata=metadata,
        measurement_identity=identity,
        config=GravityHoldoutSplitConfig(
            unit="vehicle_journey",
            holdout_fraction=0.2,
            seed=731,
            stratify_by=("measurement_type", "line", "time_period"),
        ),
    )
    holdout = estimate_and_validate_gravity_holdout(
        problem=child_problem,
        compact_layout=compact,
        split=split,
        measurement_identity=identity,
        initial_raw_parameters=progression.child.raw_parameter_estimates,
        estimator_config=config,
        execution_policy=execution,
    )
    report["relaxation"] = {
        "selected": progression.child.relaxation_applied.value,
        "lineage_nodes": len(progression.lineage.nodes),
        "warm_start_maximum_prediction_difference": (
            progression.warm_start.maximum_parent_prediction_difference
        ),
        "objective_change": progression.comparison.objective_change,
    }
    report["holdout"] = {
        "unit": "vehicle_journey",
        "calibration_measurements": holdout.calibration.measurements,
        "holdout_measurements": holdout.holdout.measurements,
        "calibration_rmse": holdout.calibration.rmse,
        "holdout_rmse": holdout.holdout.rmse,
        "calibration_negative_binomial_deviance": (
            holdout.calibration.negative_binomial_deviance
        ),
        "holdout_negative_binomial_deviance": (
            holdout.holdout.negative_binomial_deviance
        ),
    }
    return report
