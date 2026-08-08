from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import jax

from public_transportation.inference.reduced_od import (
    ConditionalGravityFeatures,
    MinimalGravityParameterLayout,
    MinimalGravityProblem,
    MinimalGravitySpecification,
    ReducedODAdequacyConfig,
    ReducedODHoldoutConfig,
    ReducedODMeasurementMetadata,
    ReducedODModelLineage,
    build_reduced_od_holdout_split,
    build_reduced_response_operator_from_coo,
    construct_reduced_od_child_warm_start,
    create_reduced_od_advisory_child,
    create_reduced_od_root_node,
    default_minimal_gravity_raw_parameters,
    diagnose_reduced_od_adequacy,
    estimate_minimal_gravity,
    evaluate_minimal_gravity_objective,
    recommend_reduced_od_relaxations,
    validate_reduced_od_holdout,
)
from public_transportation.preprocessing.reduced_od import ResponseCellKey


@pytest.fixture(autouse=True)
def _enable_required_precision():
    with jax.enable_x64():
        yield


def _problem(*, degenerate: bool = False) -> tuple[MinimalGravityProblem, np.ndarray]:
    groups = 4
    cells_per_group = 2
    cells = groups * cells_per_group
    keys = tuple(
        ResponseCellKey(f"O{group}", f"D{destination}", "P")
        for group in range(groups)
        for destination in range(cells_per_group)
    )
    if degenerate:
        times = np.full(cells, 600.0)
        transfers = np.zeros(cells)
    else:
        times = np.tile(np.asarray([300.0, 1200.0]), groups)
        transfers = np.tile(np.asarray([0.0, 1.0]), groups)
    features = ConditionalGravityFeatures(
        cell_keys=keys,
        origin_time_group_index=np.repeat(np.arange(groups), cells_per_group),
        destination_index=np.tile(np.arange(cells_per_group), groups),
        journey_time_seconds=times,
        transfer_count=transfers,
        destination_attractiveness=np.tile(np.asarray([1.0, 1.5]), groups),
        baseline_productions=np.full(groups, 100.0),
        origin_time_group_keys=tuple((f"O{group}", "P") for group in range(groups)),
        destination_ids=("D0", "D1"),
    )
    operator = build_reduced_response_operator_from_coo(
        number_of_measurements=cells,
        number_of_free_cells=cells,
        measurement_index=np.arange(cells),
        free_cell_index=np.arange(cells),
        response_values=np.ones(cells),
    )
    layout = MinimalGravityParameterLayout(MinimalGravitySpecification())
    truth = default_minimal_gravity_raw_parameters(
        layout, beta_time=0.8, beta_transfer=1.2
    )
    provisional = MinimalGravityProblem(
        features=features,
        parameter_layout=layout,
        response_operator=operator,
        observations=np.ones(cells),
    )
    observations = np.asarray(
        evaluate_minimal_gravity_objective(
            truth, problem=provisional
        ).measurement_mean
    )
    return replace(provisional, observations=observations), truth


def _metadata() -> ReducedODMeasurementMetadata:
    return ReducedODMeasurementMetadata(
        number_of_measurements=8,
        measurement_type=np.tile(np.asarray(["boarding", "alighting"]), 4),
        line=np.repeat(np.asarray(["L1", "L2"]), 4),
        direction=np.repeat(np.asarray(["out", "back"]), 4),
        stop=np.tile(np.asarray(["S1", "S2"]), 4),
        time_period=np.repeat(np.asarray(["AM", "PM"]), 4),
        vehicle_journey=np.repeat(np.asarray(["V1", "V2", "V3", "V4"]), 2),
        origin_zone=np.repeat(np.asarray(["OZ1", "OZ2"]), 4),
        destination_zone=np.tile(np.asarray(["DZ1", "DZ2"]), 4),
        transfer_place=np.tile(np.asarray(["none", "hub"]), 4),
    )


def test_grouped_holdout_keeps_complete_vehicle_journeys_together() -> None:
    metadata = _metadata()
    split = build_reduced_od_holdout_split(
        metadata=metadata,
        measurement_identity="counts-v1",
        config=ReducedODHoldoutConfig(
            unit="vehicle_journey", fraction=0.25, seed=17
        ),
    )
    for journey in np.unique(metadata.vehicle_journey):
        indices = np.flatnonzero(metadata.vehicle_journey == journey)
        assert np.all(split.holdout_mask[indices]) or np.all(
            split.calibration_mask[indices]
        )
    assert not set(split.calibration_groups) & set(split.holdout_groups)


def test_holdout_observations_never_affect_calibration_fit() -> None:
    problem, _ = _problem()
    metadata = _metadata()
    split = build_reduced_od_holdout_split(
        metadata=metadata,
        measurement_identity="counts-v1",
        config=ReducedODHoldoutConfig(unit="vehicle_journey", fraction=0.25),
    )
    first = validate_reduced_od_holdout(
        problem=problem,
        initial_raw_parameters=np.asarray([-0.5, -0.5]),
        model_fingerprint="J0",
        metadata=metadata,
        split=split,
    )
    changed_observations = np.array(problem.observations, copy=True)
    changed_observations[split.holdout_mask] += 10_000.0
    second = validate_reduced_od_holdout(
        problem=replace(problem, observations=changed_observations),
        initial_raw_parameters=np.asarray([-0.5, -0.5]),
        model_fingerprint="J0",
        metadata=metadata,
        split=split,
    )
    np.testing.assert_array_equal(first.fit.raw_parameters, second.fit.raw_parameters)
    assert first.calibration == second.calibration
    assert first.holdout != second.holdout


def test_full_data_adequacy_has_route_transfer_summaries() -> None:
    problem, _ = _problem()
    fitted = estimate_minimal_gravity(
        problem=problem,
        initial_raw_parameters=np.asarray([-0.5, -0.5]),
        model_fingerprint="J0",
    )
    report = diagnose_reduced_od_adequacy(
        fit=fitted, problem=problem, metadata=_metadata()
    )
    groupings = {item.grouping for item in report.grouped_summaries}
    assert {"line", "vehicle_journey", "transfer_place"} <= groupings
    assert report.calibration_adequacy_only
    assert report.rmse < 1.0e-3


def test_adequacy_warns_about_unidentified_parameters() -> None:
    problem, _ = _problem(degenerate=True)
    fitted = estimate_minimal_gravity(
        problem=problem,
        initial_raw_parameters=np.zeros(2),
        model_fingerprint="degenerate-J0",
    )
    report = diagnose_reduced_od_adequacy(
        fit=fitted,
        problem=problem,
        metadata=_metadata(),
        config=ReducedODAdequacyConfig(weak_identification_condition=100.0),
    )
    assert report.identification.weakly_identified
    assert report.identification.warnings


def test_recommendations_are_advisory_and_report_missing_metadata() -> None:
    problem, _ = _problem()
    noisy = np.array(problem.observations, copy=True)
    noisy[np.asarray(_metadata().time_period) == "PM"] += 30.0
    noisy_problem = replace(problem, observations=noisy)
    fitted = estimate_minimal_gravity(
        problem=noisy_problem,
        initial_raw_parameters=np.asarray([-0.5, -0.5]),
        model_fingerprint="noisy-J0",
    )
    metadata = replace(_metadata(), destination_zone=None)
    adequacy = diagnose_reduced_od_adequacy(
        fit=fitted, problem=noisy_problem, metadata=metadata
    )
    recommendations = recommend_reduced_od_relaxations(
        adequacy=adequacy, metadata=metadata
    )
    assert recommendations.advisory_only
    by_stage = {item.stage: item for item in recommendations.candidates}
    assert by_stage["J1"].applicable
    assert by_stage["J1"].strength != "none"
    assert not by_stage["J2"].applicable


@pytest.mark.parametrize("stage", ["J1", "J2", "J3", "J4"])
def test_every_advisory_child_has_verified_parent_and_warm_start(stage) -> None:
    root = create_reduced_od_root_node(
        model_fingerprint="J0",
        parameter_names=("beta_time", "beta_transfer"),
        raw_parameters=np.asarray([0.1, 0.2]),
    )
    prediction = np.asarray([10.0, 20.0])
    warm = construct_reduced_od_child_warm_start(
        parent=root,
        stage=stage,
        added_parameter_names=(f"{stage}_deviation[0]",),
        parent_prediction=prediction,
        child_prediction_at_warm_start=prediction.copy(),
    )
    child = create_reduced_od_advisory_child(
        parent=root, warm_start=warm, model_fingerprint=f"{stage}-candidate"
    )
    lineage = ReducedODModelLineage((root,)).append(child)
    assert lineage.nodes[-1].parent_identifier == root.identifier
    assert not lineage.nodes[-1].fitted
    np.testing.assert_array_equal(warm.raw_parameters[:2], root.raw_parameters)
    np.testing.assert_array_equal(warm.raw_parameters[2:], [0.0])
    assert warm.maximum_parent_prediction_difference == 0.0


def test_warm_start_rejects_prediction_change() -> None:
    root = create_reduced_od_root_node(
        model_fingerprint="J0",
        parameter_names=("beta_time", "beta_transfer"),
        raw_parameters=np.asarray([0.1, 0.2]),
    )
    with pytest.raises(ValueError, match="does not reproduce"):
        construct_reduced_od_child_warm_start(
            parent=root,
            stage="J1",
            added_parameter_names=("period[0]",),
            parent_prediction=np.asarray([1.0]),
            child_prediction_at_warm_start=np.asarray([1.1]),
        )


def test_adequacy_rejects_holdout_problem_to_keep_labels_distinct() -> None:
    problem, _ = _problem()
    masked = replace(
        problem,
        calibration_mask=np.asarray([True, True, True, True, False, False, False, False]),
    )
    fitted = estimate_minimal_gravity(
        problem=masked,
        initial_raw_parameters=np.asarray([-0.5, -0.5]),
        model_fingerprint="holdout-fit",
    )
    with pytest.raises(ValueError, match="full-data adequacy"):
        diagnose_reduced_od_adequacy(fit=fitted, problem=masked)
