"""Bounded end-to-end gravity validation on the public Geneva snapshot."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse as jsparse

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.domain import Scenario, read_fixed_demand_csv
from public_transportation.inference.assignment_adapter import (
    assign_link_flow_fixed_routing,
    build_assignment_inputs,
    prepare_fixed_routing,
)
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.fixed_routing_measurement_operator import (
    FixedRoutingMeasurementOperator,
    MeasurementOperatorMetrics,
    assignment_inputs_fingerprint,
    measurement_mapping_fingerprint,
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
from public_transportation.measurement.likelihood_jax import (
    predict_measurements_from_link_flow,
)
from public_transportation.preprocessing import (
    ODTimeKey,
    build_structural_zero_topology,
    compute_od_path_metrics,
    load_structural_zero_config,
)

EXAMPLE = Path(__file__).resolve().parents[1]
DATA = EXAMPLE / "data"
MEASUREMENTS = EXAMPLE / "pre_processing/results/measurements_boarding_alighting.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results/gravity_validation_summary.json"


def _gravity_features(scenario, layout, compact) -> GravityFeatures:
    free_indices = np.asarray(layout.free_od_indices, dtype=np.int64)
    keys = [layout.od_keys[index] for index in free_indices]
    topology_config = load_structural_zero_config(EXAMPLE / "structural_zeros.toml")
    topology = build_structural_zero_topology(scenario, topology_config.assignment)
    metric_records = compute_od_path_metrics(
        topology,
        keys=tuple(ODTimeKey(*key) for key in keys),
    )
    metrics = {record.key.tuple: record.metrics for record in metric_records}
    if any(not metrics[key].feasible for key in keys):
        raise ValueError("Every free Geneva gravity cell must be scheduled-feasible.")
    origins = sorted({key[0] for key in keys})
    destinations = sorted({key[1] for key in keys})
    periods = sorted({key[2] for key in keys})
    origin_lookup = {value: index for index, value in enumerate(origins)}
    destination_lookup = {value: index for index, value in enumerate(destinations)}
    period_lookup = {value: index for index, value in enumerate(periods)}
    group_pairs = sorted({(key[0], key[2]) for key in keys})
    group_lookup = {value: index for index, value in enumerate(group_pairs)}
    baselines = np.asarray(layout.free_baseline_values, dtype=np.float32)
    totals = np.zeros(len(group_pairs), dtype=np.float32)
    destination_totals: dict[tuple[str, str], float] = {}
    for key, baseline in zip(keys, baselines, strict=True):
        totals[group_lookup[(key[0], key[2])]] += baseline
        destination_key = (key[1], key[2])
        destination_totals[destination_key] = (
            destination_totals.get(destination_key, 0.0) + float(baseline)
        )
    time_period = np.asarray(
        [0 if period_lookup[key[2]] < len(periods) / 2 else 1 for key in keys]
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
        time_period_index=time_period,
    )


def _validation_metadata() -> GravityValidationMetadata:
    rows = []
    with MEASUREMENTS.open(encoding="utf-8-sig", newline="") as stream:
        rows.extend(csv.DictReader(stream))
    trips = [str(row["trip_id"]) for row in rows]
    return GravityValidationMetadata(
        len(rows),
        measurement_type=np.asarray([row["measurement_type"] for row in rows]),
        line=np.asarray([trip.split("_", 1)[0] for trip in trips]),
        stop=np.asarray([row["stop_id"] for row in rows]),
        time_period=np.asarray(
            ["early" if int(row["time"].split(":", 1)[0]) < 8 else "late" for row in rows]
        ),
        vehicle_journey=np.asarray(trips),
    )


def _prepare_scalar_bcoo_operator(*, inputs, routing, spec, compact, od_layout):
    """Build Geneva's sparse operator without a node-by-measurement DP state."""
    if compact.num_fixed_positive:
        raise ValueError("The public Geneva scalar builder expects zero fixed offsets.")

    def forward(demand):
        link_flow = assign_link_flow_fixed_routing(
            inputs=inputs, routing=routing, f=demand
        )
        return predict_measurements_from_link_flow(
            link_flow,
            spec_num_measurements=spec.num_measurements,
            spec_measurement_index=jnp.asarray(spec.measurement_index),
            spec_link_index=jnp.asarray(spec.link_index),
        )

    compiled = jax.jit(forward)
    data: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    started = perf_counter()
    for column in range(compact.num_free):
        basis = jnp.zeros(compact.num_free, dtype=inputs.base_link_cost.dtype).at[
            column
        ].set(1)
        values = np.asarray(compiled(basis))
        rows = np.flatnonzero(values)
        if rows.size:
            data.append(values[rows])
            indices.append(
                np.column_stack(
                    (rows, np.full(rows.size, column, dtype=np.int64))
                )
            )
    construction = perf_counter() - started
    assembled_data = (
        np.concatenate(data)
        if data
        else np.empty(0, dtype=np.dtype(inputs.base_link_cost.dtype))
    )
    assembled_indices = (
        np.concatenate(indices)
        if indices
        else np.empty((0, 2), dtype=np.int64)
    )
    matrix = jsparse.BCOO(
        (jnp.asarray(assembled_data), jnp.asarray(assembled_indices)),
        shape=(spec.num_measurements, compact.num_free),
        unique_indices=True,
    )
    nonzero = int(assembled_data.size)
    stored = int(assembled_data.nbytes + assembled_indices.nbytes)
    return FixedRoutingMeasurementOperator(
        matrix=matrix,
        fixed_measurement_offset=jnp.zeros(
            spec.num_measurements, dtype=inputs.base_link_cost.dtype
        ),
        representation="bcoo",
        num_active_od=compact.num_active,
        num_free_od=compact.num_free,
        num_measurements=spec.num_measurements,
        od_layout_fingerprint=od_layout.fingerprint,
        compact_layout_fingerprint=compact.fingerprint,
        assignment_fingerprint=assignment_inputs_fingerprint(inputs),
        graph_fingerprint=assignment_inputs_fingerprint(inputs),
        mapping_fingerprint=measurement_mapping_fingerprint(spec),
        theta=5.0,
        dtype=str(np.dtype(inputs.base_link_cost.dtype)),
        metrics=MeasurementOperatorMetrics(
            construction_seconds=construction,
            dense_bytes=spec.num_measurements
            * compact.num_free
            * np.dtype(inputs.base_link_cost.dtype).itemsize,
            stored_bytes=stored,
            peak_construction_bytes=stored,
            nonzero_entries=nonzero,
            total_entries=spec.num_measurements * compact.num_free,
            density=nonzero / (spec.num_measurements * compact.num_free),
            chunk_size=1,
            compilation_count=1,
            num_chunks=compact.num_free,
            chunk_shape=(1, spec.num_measurements),
        ),
    )


def run_validation(
    *,
    maximum_iterations: int = 2,
    holdout_iterations: int = 2,
) -> dict[str, object]:
    def stage(message: str) -> None:
        print(f"[geneva-gravity] {message}", flush=True)

    stage("load scenario and compact layout")
    scenario = Scenario.from_folder(
        DATA, strict=True, demand_file=DATA / "prior_demand.csv"
    )
    od_layout = build_od_parameter_layout(
        scenario=scenario,
        fixed_demand=read_fixed_demand_csv(DATA / "fixed_demand.csv", scenario=scenario),
    )
    compact = build_compact_od_assignment_layout(parameter_layout=od_layout)
    stage("prepare gravity features from scheduled paths")
    features = _gravity_features(scenario, od_layout, compact)
    gc.collect()
    stage("prepare assignment and measurement mapping")
    artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
    inputs = build_assignment_inputs(artifacts=artifacts, compact_layout=compact)
    routing = prepare_fixed_routing(inputs=inputs, theta=5.0)
    id_manager = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)
    mapped = build_mapping_spec_strict(
        id_manager=id_manager,
        table=read_measurements_csv(MEASUREMENTS),
        include_link_lists_for_report=False,
    )
    stage("construct BCOO fixed-routing measurement operator")
    operator = _prepare_scalar_bcoo_operator(
        inputs=inputs,
        routing=routing,
        spec=mapped.spec,
        compact=compact,
        od_layout=od_layout,
    )
    gc.collect()
    parameter_layout = GravityParameterLayout(GravityModelSpecification())
    problem = GravityObjectiveProblem(
        features=features,
        parameter_layout=parameter_layout,
        operator=operator,
        observations=np.asarray(mapped.y_obs),
        likelihood=GravityLikelihood.NEGATIVE_BINOMIAL,
    )
    initial = parameter_layout.raw_from_physical((0.5, 1.0, 10.0))
    estimator_config = GravityEstimatorConfig(maximum_iterations=maximum_iterations)
    execution = GravityExecutionPolicy(gradient_strategy="adjoint")
    stage("estimate minimal full-data model")
    parent_result = estimate_gravity_model(
        problem=problem,
        compact_layout=compact,
        initial_raw_parameters=initial,
        config=estimator_config,
        execution=execution,
    )
    metadata = _validation_metadata()
    stage("compute full-data adequacy")
    parent_adequacy = validate_full_data_gravity_adequacy(
        result=parent_result,
        problem=problem,
        compact_layout=compact,
        metadata=metadata,
    )
    measurement_identity = gravity_measurement_identity(
        measurement_indices=np.arange(operator.num_measurements),
        label="public Geneva boarding/alighting rows v1",
    )
    parent_node = create_gravity_model_node(
        result=parent_result,
        problem=problem,
        compact_layout=compact,
        adequacy_report=parent_adequacy,
        calibration_measurement_identity=measurement_identity,
        validation_measurement_identity=measurement_identity,
    )
    stage("score recommendations and estimate selected time-period child")
    progression = progress_gravity_model_lineage(
        lineage=GravityModelLineage((parent_node,)),
        parent_identifier=parent_node.model_identifier,
        selected_relaxation=GravityEffectScope.TIME_PERIOD,
        problem=problem,
        compact_layout=compact,
        parent_adequacy_report=parent_adequacy,
        calibration_measurement_identity=measurement_identity,
        validation_measurement_identity=measurement_identity,
        estimator_config=estimator_config,
        execution_policy=execution,
        validation_metadata=metadata,
    )
    child_problem = replace(
        problem, parameter_layout=progression.warm_start.child_parameter_layout
    )
    split = build_gravity_holdout_split(
        metadata=metadata,
        measurement_identity=measurement_identity,
        config=GravityHoldoutSplitConfig(
            unit="vehicle_journey",
            holdout_fraction=0.1,
            seed=731,
            stratify_by=("measurement_type", "line", "time_period"),
        ),
    )
    stage("re-estimate selected child under grouped holdout")
    holdout = estimate_and_validate_gravity_holdout(
        problem=child_problem,
        compact_layout=compact,
        split=split,
        measurement_identity=measurement_identity,
        initial_raw_parameters=progression.child.raw_parameter_estimates,
        estimator_config=GravityEstimatorConfig(maximum_iterations=holdout_iterations),
        execution_policy=execution,
    )
    candidate_rows = [
        {
            "candidate": item.candidate_name,
            "applicable": item.applicable,
            "added_parameters": item.added_parameters,
            "strength": item.recommendation_strength,
            "approximate_gain": item.approximate_gain,
        }
        for item in progression.recommendations.candidates
    ]
    stage("assemble validation report")
    return {
        "schema_version": 1,
        "example": "geneva_gtfs",
        "backend": jax.default_backend(),
        "dtype": str(features.dtype),
        "num_od_total": compact.num_od_total,
        "num_free_od": compact.num_free,
        "num_measurements": operator.num_measurements,
        "operator": {
            "representation": operator.representation,
            "density": operator.metrics.density,
            "stored_bytes": operator.metrics.stored_bytes,
            "construction_seconds": operator.metrics.construction_seconds,
        },
        "minimal": {
            "status": parent_result.status,
            "iterations": parent_result.iterations,
            "objective": parent_result.objective,
            "negative_binomial_deviance": parent_adequacy.negative_binomial_deviance,
            "rmse": parent_adequacy.rmse,
        },
        "recommendations": candidate_rows,
        "lineage": {
            "nodes": len(progression.lineage.nodes),
            "selected_relaxation": progression.child.relaxation_applied.value,
            "warm_start_maximum_prediction_difference": progression.warm_start.maximum_parent_prediction_difference,
            "child_status": progression.child.estimation_result.status,
            "objective_change": progression.comparison.objective_change,
        },
        "holdout": {
            "split_fingerprint": split.split_fingerprint,
            "calibration_measurements": holdout.calibration.measurements,
            "holdout_measurements": holdout.holdout.measurements,
            "calibration_rmse": holdout.calibration.rmse,
            "holdout_rmse": holdout.holdout.rmse,
            "calibration_nb_deviance": holdout.calibration.negative_binomial_deviance,
            "holdout_nb_deviance": holdout.holdout.negative_binomial_deviance,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maximum-iterations", type=int, default=2)
    parser.add_argument("--holdout-iterations", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = run_validation(
        maximum_iterations=arguments.maximum_iterations,
        holdout_iterations=arguments.holdout_iterations,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
