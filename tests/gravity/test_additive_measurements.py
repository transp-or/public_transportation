from __future__ import annotations

from dataclasses import replace

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import json

from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)
from public_transportation.inference.fixed_routing_measurement_operator import (
    FixedRoutingMeasurementOperator,
    MeasurementOperatorMetrics,
)
from public_transportation.inference.gravity import (
    DirectNonnegativeFlowModel,
    GravityAdditiveFlowBlock,
    GravityDenseLinearMeasurementOperator,
    GravityFeatures,
    GravityGradientStrategy,
    GravityJointParameterLayout,
    GravityLikelihood,
    LinearNonnegativeFlowModel,
    GravityModelSpecification,
    GravityObservationModel,
    GravityObjectiveProblem,
    GravityParameterLayout,
    GravityStrategySelection,
    GravityValidationMetadata,
    GravityEstimatorConfig,
    GravityExecutionPolicy,
    GravityEstimationResult,
    build_gravity_run_manifest,
    load_gravity_boundary_artifact,
    write_gravity_boundary_artifact,
    evaluate_gravity_objective,
    gravity_value_and_gradient,
    write_gravity_detailed_report,
    GravityMeasurementOperatorLinearAdapter,
)
from public_transportation.inference.gravity.estimator import estimate_gravity_model
from public_transportation.inference.od_parameter_layout import ODParameterLayout


def _features() -> GravityFeatures:
    return GravityFeatures(
        canonical_od_index=np.arange(4),
        origin_index=np.asarray((0, 0, 1, 1)),
        destination_index=np.asarray((0, 1, 0, 1)),
        departure_time_index=np.zeros(4, dtype=int),
        origin_time_group_index=np.asarray((0, 0, 1, 1)),
        journey_time=np.asarray((5.0, 15.0, 12.0, 7.0)),
        transfer_count=np.asarray((0, 1, 2, 0)),
        structural_feasible=np.ones(4, dtype=bool),
        origin_time_totals=np.asarray((20.0, 30.0)),
        destination_attractiveness=np.asarray((1.0, 3.0, 2.0, 1.0)),
        num_origins=2,
        num_destinations=2,
        num_departure_times=1,
        od_layout_fingerprint="additive-fixture-layout",
        journey_time_scale=10.0,
    )


def _operator() -> FixedRoutingMeasurementOperator:
    matrix = jnp.asarray(((1.0, 0.0, 0.5, 0.0), (0.0, 1.0, 0.0, 1.0)))
    return FixedRoutingMeasurementOperator(
        matrix=matrix,
        fixed_measurement_offset=jnp.asarray((2.0, 1.0)),
        representation="dense",
        num_active_od=4,
        num_free_od=4,
        num_measurements=2,
        od_layout_fingerprint="additive-fixture-layout",
        compact_layout_fingerprint="additive-fixture-layout",
        assignment_fingerprint="additive-fixture-assignment",
        graph_fingerprint="additive-fixture-graph",
        mapping_fingerprint="additive-fixture-mapping",
        theta=1.0,
        dtype="float32",
        metrics=MeasurementOperatorMetrics(
            construction_seconds=0.0,
            dense_bytes=int(matrix.size * 4),
            stored_bytes=int(matrix.size * 4),
            peak_construction_bytes=0,
            nonzero_entries=4,
            total_entries=8,
            density=0.5,
            chunk_size=4,
            cache_hit=False,
        ),
    )


def _layout() -> GravityJointParameterLayout:
    initial = GravityAdditiveFlowBlock(
        name="initial_boarding",
        operator=GravityDenseLinearMeasurementOperator(np.asarray(((1.0,), (0.0,)))),
        latent_flow_model=DirectNonnegativeFlowModel(1, prefix="initial"),
        regularization_strength=0.2,
    )
    terminal = GravityAdditiveFlowBlock(
        name="terminal_alighting",
        operator=GravityDenseLinearMeasurementOperator(np.asarray(((0.0,), (1.0,)))),
        latent_flow_model=DirectNonnegativeFlowModel(1, prefix="terminal"),
    )
    return GravityJointParameterLayout(
        GravityParameterLayout(GravityModelSpecification()),
        (initial, terminal),
    )


def _problem() -> GravityObjectiveProblem:
    return GravityObjectiveProblem(
        features=_features(),
        parameter_layout=_layout(),
        operator=_operator(),
        observations=np.asarray((25.0, 32.0)),
        likelihood=GravityLikelihood.POISSON,
        rho=2.0,
        observation_model=GravityObservationModel(
            measurement_types=("boarding", "alighting"),
            boarding_scale=3.0,
            alighting_scale=4.0,
        ),
    )


def test_additive_blocks_and_fixed_observation_scales_are_explicitly_composed():
    with jax.enable_x64():
        problem = _problem()
        raw = jnp.asarray((0.1, -0.2, 1.0, 0.3, -0.4), dtype=jnp.float64)
        evaluation = evaluate_gravity_objective(raw, problem=problem)
        assert problem.parameter_layout.names == (
            "beta_time",
            "beta_transfer",
            "dispersion",
            "initial_boarding.initial[0]",
            "terminal_alighting.terminal[0]",
        )
        assert len(evaluation.additive_flows) == 2
        latent = (
            np.asarray(evaluation.od_measurement)
            + np.asarray(evaluation.additive_measurements[0])
            + np.asarray(evaluation.additive_measurements[1])
            + np.asarray(problem.operator.fixed_measurement_offset)
        )
        expected = np.asarray((6.0, 8.0)) * latent
        np.testing.assert_allclose(evaluation.latent_measurement, latent)
        np.testing.assert_allclose(evaluation.observation_scales, (6.0, 8.0))
        np.testing.assert_allclose(evaluation.measurement_mean, expected)


def test_existing_od_operator_can_be_used_through_generic_linear_adapter():
    adapter = GravityMeasurementOperatorLinearAdapter(_operator())
    assert (adapter.num_rows, adapter.num_columns) == (2, 4)
    vector = jnp.ones(4)
    np.testing.assert_allclose(adapter.jax_matvec(vector), (1.5, 2.0))


def test_linear_flow_specification_is_json_serializable():
    block = GravityAdditiveFlowBlock(
        name="initial_onboard",
        operator=GravityDenseLinearMeasurementOperator(np.ones((2, 2))),
        latent_flow_model=LinearNonnegativeFlowModel(
            basis=np.asarray(((1.0, 0.0), (0.5, 1.0))),
        ),
    )
    layout = GravityJointParameterLayout(
        GravityParameterLayout(GravityModelSpecification()), (block,)
    )
    payload = layout.specification.to_dict()
    json.dumps(payload)
    assert payload["additive_flow_blocks"][0]["latent_flow_model"]["basis"] == [
        [1.0, 0.0],
        [0.5, 1.0],
    ]


def test_additive_branch_has_consistent_gradients_for_both_public_strategies():
    with jax.enable_x64():
        problem = _problem()
        raw = np.asarray((0.1, -0.2, 1.0, 0.3, -0.4))
        forward, forward_gradient = gravity_value_and_gradient(
            raw, problem=problem, strategy=GravityGradientStrategy.BATCHED_FORWARD
        )
        adjoint, adjoint_gradient = gravity_value_and_gradient(
            raw, problem=problem, strategy=GravityGradientStrategy.ADJOINT
        )
        np.testing.assert_allclose(forward.objective, adjoint.objective)
        np.testing.assert_allclose(forward_gradient, adjoint_gradient, rtol=1e-10, atol=1e-10)
        assert np.all(np.isfinite(np.asarray(forward_gradient)))
        numerical = np.empty_like(raw)
        for index in range(raw.size):
            delta = np.zeros_like(raw)
            delta[index] = 1.0e-4
            plus = evaluate_gravity_objective(raw + delta, problem=problem).objective
            minus = evaluate_gravity_objective(raw - delta, problem=problem).objective
            numerical[index] = (float(plus) - float(minus)) / (2.0e-4)
        np.testing.assert_allclose(forward_gradient, numerical, rtol=2.0e-4, atol=2.0e-4)


def test_joint_layout_round_trips_with_caller_supplied_operators():
    layout = _layout()
    payload = layout.to_dict()
    operators = {
        block.name: block.operator for block in layout.additive_flow_blocks
    }
    restored = GravityJointParameterLayout.from_dict(payload, operators=operators)
    assert restored.names == layout.names
    assert restored.fingerprint == layout.fingerprint


def test_linear_operator_and_flow_models_reject_incompatible_dimensions():
    try:
        GravityAdditiveFlowBlock(
            name="bad",
            operator=GravityDenseLinearMeasurementOperator(np.ones((2, 2))),
            latent_flow_model=DirectNonnegativeFlowModel(1),
        )
    except ValueError as error:
        assert "columns" in str(error)
    else:
        raise AssertionError("dimension mismatch must be rejected")
    short_block = GravityAdditiveFlowBlock(
        name="short",
        operator=GravityDenseLinearMeasurementOperator(np.ones((1, 1))),
        latent_flow_model=DirectNonnegativeFlowModel(1),
    )
    short_layout = GravityJointParameterLayout(
        GravityParameterLayout(GravityModelSpecification()), (short_block,)
    )
    with pytest.raises(ValueError, match="primary measurement dimension"):
        GravityObjectiveProblem(
            features=_features(),
            parameter_layout=short_layout,
            operator=_operator(),
            observations=np.asarray((1.0, 1.0)),
            likelihood=GravityLikelihood.POISSON,
        )


def test_observation_model_requires_row_types_for_separate_scales():
    with pytest.raises(ValueError, match="measurement_types"):
        GravityObservationModel(boarding_scale=2.0)
    with pytest.raises(ValueError, match="finite and positive"):
        GravityObservationModel(
            measurement_types=("boarding",), boarding_scale=float("nan")
        )
    model = GravityObservationModel(
        measurement_types=("boarding", "alighting"),
        boarding_scale=2.0,
        alighting_scale=3.0,
    )
    assert GravityObservationModel.from_dict(model.to_dict()) == model


def test_joint_model_manifest_records_additive_provenance_and_support():
    problem = _problem()
    compact = CompactODAssignmentLayout(
        num_od_total=4,
        active_full_indices=(0, 1, 2, 3),
        removed_zero_full_indices=(),
        full_to_compact=(0, 1, 2, 3),
        free_full_indices=(0, 1, 2, 3),
        free_compact_indices=(0, 1, 2, 3),
        free_baseline_values=(1.0, 1.0, 1.0, 1.0),
        fixed_compact_indices=(),
        fixed_compact_values=(),
    )
    manifest = build_gravity_run_manifest(
        problem=problem,
        compact_layout=compact,
        estimator_config=GravityEstimatorConfig(),
        execution=GravityExecutionPolicy(),
        repository_revision="fixture",
    )
    assert manifest["observation_model"]["measurement_types"] == [
        "boarding",
        "alighting",
    ]
    assert manifest["fingerprints"]["observation_model"]
    assert set(manifest["fingerprints"]["additive_flow_operators"]) == {
        "initial_boarding",
        "terminal_alighting",
    }
    support = manifest["measurement_support"]
    assert support["rows_supported_by_od_flow"] == 2
    assert support["rows_supported_by_any_active_flow"] == 2
    assert support["positive_rows_without_any_active_support"] == 0


def test_boundary_artifact_round_trip_and_fingerprint_validation(tmp_path):
    operator = GravityDenseLinearMeasurementOperator(np.asarray(((1.0,), (0.0,))))
    artifact = write_gravity_boundary_artifact(
        tmp_path / "initial_onboard",
        name="initial_onboard",
        operator=operator,
        row_order_fingerprint="rows-v1",
        boundary_group_ids=("cohort-a",),
        parameter_layout_fingerprint="layout-v1",
        model_specification_fingerprint="model-v1",
        construction_configuration={"source": "synthetic"},
    )
    restored = load_gravity_boundary_artifact(
        tmp_path / "initial_onboard",
        expected_row_order_fingerprint="rows-v1",
        expected_parameter_layout_fingerprint="layout-v1",
        expected_model_specification_fingerprint="model-v1",
    )
    assert restored.content_sha256 == artifact.content_sha256
    np.testing.assert_array_equal(restored.matrix, operator.matrix)
    with pytest.raises(ValueError, match="row-order"):
        load_gravity_boundary_artifact(
            tmp_path / "initial_onboard",
            expected_row_order_fingerprint="changed",
        )


def test_joint_layout_runs_through_checkpointed_estimator(tmp_path):
    problem = _problem()
    compact = CompactODAssignmentLayout(
        num_od_total=4,
        active_full_indices=(0, 1, 2, 3),
        removed_zero_full_indices=(),
        full_to_compact=(0, 1, 2, 3),
        free_full_indices=(0, 1, 2, 3),
        free_compact_indices=(0, 1, 2, 3),
        free_baseline_values=(1.0, 1.0, 1.0, 1.0),
        fixed_compact_indices=(),
        fixed_compact_values=(),
    )
    problem = replace(
        problem,
        operator=replace(
            problem.operator,
            compact_layout_fingerprint=compact.fingerprint,
        ),
        features=replace(
            problem.features,
            od_layout_fingerprint=compact.fingerprint,
        ),
    )
    result = estimate_gravity_model(
        problem=problem,
        compact_layout=compact,
        initial_raw_parameters=np.zeros(problem.parameter_layout.size),
        config=GravityEstimatorConfig(maximum_iterations=1),
        execution=GravityExecutionPolicy(
            gradient_strategy="adjoint",
            checkpoint_path=tmp_path / "gravity.json",
        ),
    )
    assert result.raw_parameters.size == problem.parameter_layout.size
    assert result.latent_measurements is not None
    assert len(result.additive_measurement_contributions) == 2
    manifest = build_gravity_run_manifest(
        problem=problem,
        compact_layout=compact,
        estimator_config=GravityEstimatorConfig(maximum_iterations=1),
        execution=GravityExecutionPolicy(gradient_strategy="adjoint"),
        repository_revision="fixture",
        result=result,
    )
    assert manifest["specification_fingerprint"] == result.specification_fingerprint


def test_detailed_report_persists_separate_additive_contributions(tmp_path):
    problem = _problem()
    evaluation = evaluate_gravity_objective(
        np.zeros(problem.parameter_layout.size), problem=problem
    )
    result = GravityEstimationResult(
        schema_version=4,
        status="converged",
        success=True,
        message="converged",
        raw_parameters=np.zeros(problem.parameter_layout.size),
        physical_parameters=np.ones(problem.parameter_layout.size),
        free_od_demand=np.asarray(evaluation.demand),
        active_od_demand=np.asarray(evaluation.demand),
        full_od_demand=np.asarray(evaluation.demand),
        predicted_measurements=np.asarray(evaluation.measurement_mean),
        objective=float(evaluation.objective),
        data_log_likelihood=float(evaluation.data_log_likelihood),
        gradient=np.zeros(problem.parameter_layout.size),
        iterations=1,
        elapsed_seconds=0.1,
        model_fingerprint="additive-report-model",
        strategy_selection=GravityStrategySelection(
            requested="adjoint",
            selected="adjoint",
            reason="test",
            candidates=(),
            persistent_compilation_cache_enabled=False,
            persistent_compilation_cache_directory=None,
        ),
        resumed=False,
        checkpoint_path=None,
        specification_fingerprint=problem.parameter_layout.specification.fingerprint,
        model_specification=problem.parameter_layout.specification.to_dict(),
        parameter_names=problem.parameter_layout.names,
        latent_measurements=np.asarray(evaluation.latent_measurement),
        od_measurement_contribution=np.asarray(evaluation.od_measurement),
        additive_measurement_contributions=tuple(
            np.asarray(value) for value in evaluation.additive_measurements
        ),
        additive_flows=tuple(np.asarray(value) for value in evaluation.additive_flows),
        observation_scales=np.asarray(evaluation.observation_scales),
    )
    report = write_gravity_detailed_report(
        result=result,
        observations=np.asarray(problem.observations),
        predicted_measurements=np.asarray(evaluation.measurement_mean),
        od_layout=ODParameterLayout(
            num_od_total=4,
            od_keys=(("o0", "d0", "t0"), ("o0", "d1", "t0"), ("o1", "d0", "t0"), ("o1", "d1", "t0")),
            free_od_indices=(0, 1, 2, 3),
            fixed_od_indices=(),
            fixed_od_values=(),
            free_baseline_values=(1.0, 1.0, 1.0, 1.0),
            fixed_zero_indices=(),
            fixed_positive_indices=(),
        ),
        metadata=GravityValidationMetadata(
            2, measurement_type=np.asarray(("boarding", "alighting"))
        ),
        likelihood=GravityLikelihood.POISSON,
        output_directory=tmp_path / "report",
    )
    assert report.files["measurement_contributions.csv"].is_file()
    summary = json.loads(report.files["report.json"].read_text())
    assert summary["measurement_contributions"]["available"] is True
