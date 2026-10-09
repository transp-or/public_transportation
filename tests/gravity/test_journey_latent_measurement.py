from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.inference.fixed_routing_measurement_operator import (
    FixedRoutingMeasurementOperator,
    MeasurementOperatorMetrics,
)
from public_transportation.inference.gravity import (
    GravityFeatures,
    GravityGradientStrategy,
    GravityJourneyLatentClassModel,
    GravityLikelihood,
    GravityMeasurementParameterLayout,
    GravityModelSpecification,
    GravityObjectiveProblem,
    GravityParameterLayout,
    evaluate_gravity_objective,
    gravity_model_fingerprint,
    gravity_value_and_gradient,
    predict_gravity_measurements,
)
from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)


def _features() -> GravityFeatures:
    return GravityFeatures(
        canonical_od_index=np.arange(3),
        origin_index=np.asarray((0, 0, 1)),
        destination_index=np.asarray((0, 1, 0)),
        departure_time_index=np.asarray((0, 0, 0)),
        origin_time_group_index=np.asarray((0, 0, 1)),
        journey_time=np.asarray((5.0, 15.0, 12.0), dtype=np.float32),
        transfer_count=np.asarray((0, 1, 2)),
        structural_feasible=np.asarray((True, True, True)),
        origin_time_totals=np.asarray((20.0, 30.0), dtype=np.float32),
        destination_attractiveness=np.asarray((1.0, 3.0, 2.0), dtype=np.float32),
        num_origins=2,
        num_destinations=2,
        num_departure_times=1,
        od_layout_fingerprint="latent-layout",
        journey_time_scale=10.0,
    )


def _operator() -> FixedRoutingMeasurementOperator:
    matrix = jnp.asarray(((1.0, 0.5, 0.0), (0.0, 1.0, 2.0)), dtype=jnp.float32)
    return FixedRoutingMeasurementOperator(
        matrix=matrix,
        fixed_measurement_offset=jnp.asarray((2.0, 1.0), dtype=jnp.float32),
        representation="dense",
        num_active_od=3,
        num_free_od=3,
        num_measurements=2,
        od_layout_fingerprint="latent-layout",
        compact_layout_fingerprint="latent-layout",
        assignment_fingerprint="assignment-latent",
        graph_fingerprint="graph-latent",
        mapping_fingerprint="mapping-latent",
        theta=1.0,
        dtype="float32",
        metrics=MeasurementOperatorMetrics(
            construction_seconds=0.0,
            dense_bytes=int(matrix.size * 4),
            stored_bytes=int(matrix.size * 4),
            peak_construction_bytes=0,
            nonzero_entries=4,
            total_entries=6,
            density=4 / 6,
            chunk_size=3,
            cache_hit=True,
        ),
    )


def _problem(model: GravityJourneyLatentClassModel) -> GravityObjectiveProblem:
    base = GravityParameterLayout(GravityModelSpecification())
    layout = GravityMeasurementParameterLayout(base, model)
    return GravityObjectiveProblem(
        features=_features(),
        parameter_layout=layout,
        operator=_operator(),
        observations=np.asarray((18.0, 27.0), dtype=np.float32),
        likelihood=GravityLikelihood.NEGATIVE_BINOMIAL,
        measurement_model=model,
    )


def test_class_probabilities_are_normalized_and_effects_are_bounded() -> None:
    model = GravityJourneyLatentClassModel(num_classes=3, max_log_effect=1.5)
    probabilities, effects = model.class_parameters(
        np.asarray((100.0, -100.0, 20.0, -20.0))
    )
    assert float(jnp.sum(probabilities)) == pytest.approx(1.0, abs=1.0e-6)
    assert np.all(np.asarray(probabilities) > 0.0)
    assert np.all(np.asarray(effects) > 0.0)
    assert np.all(np.asarray(effects) <= np.exp(1.5) + 1.0e-6)
    assert float(effects[model.reference_class]) == pytest.approx(1.0)


def test_measurement_parameters_round_trip_through_physical_coordinates() -> None:
    model = GravityJourneyLatentClassModel(num_classes=3)
    raw = np.asarray((0.2, -0.4, 0.3, -0.1), dtype=np.float64)
    physical = np.asarray(model.physical_free_parameters(raw))
    restored = model.raw_from_physical(physical)
    np.testing.assert_allclose(restored, raw, rtol=1.0e-6, atol=1.0e-6)
    assert (
        model.fingerprint
        == GravityJourneyLatentClassModel.from_dict(model.to_dict()).fingerprint
    )


def test_float32_latent_parameters_agree_with_float64_reference() -> None:
    model = GravityJourneyLatentClassModel(num_classes=3)
    raw = np.asarray((0.7, -0.3, 0.2, -0.5), dtype=np.float64)
    with jax.enable_x64():
        probabilities_32, effects_32 = model.class_parameters(
            jnp.asarray(raw, dtype=jnp.float32), dtype=jnp.float32
        )
        probabilities_64, effects_64 = model.class_parameters(
            jnp.asarray(raw, dtype=jnp.float64), dtype=jnp.float64
        )
    np.testing.assert_allclose(
        probabilities_32, probabilities_64, rtol=2.0e-6, atol=2.0e-6
    )
    np.testing.assert_allclose(effects_32, effects_64, rtol=2.0e-6, atol=2.0e-6)


def test_invalid_models_are_rejected() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        GravityJourneyLatentClassModel(
            class_probabilities=(0.2, 0.2), class_effects=(1.0, 2.0)
        )
    with pytest.raises(ValueError, match="reference class effect"):
        GravityJourneyLatentClassModel(class_effects=(2.0, 1.0))
    with pytest.raises(ValueError, match="only 'independent_journey_aggregate'"):
        GravityJourneyLatentClassModel(aggregation_unit="shared_joint_journey")


def test_latent_model_changes_assigned_flow_but_not_fixed_offset() -> None:
    model = GravityJourneyLatentClassModel(
        class_probabilities=(0.25, 0.75), class_effects=(1.0, 2.0)
    )
    problem = _problem(model)
    mean, _ = predict_gravity_measurements(
        np.asarray((-0.2, 0.4, 1.5), dtype=np.float32), problem=problem
    )
    base_problem = replace(
        problem,
        measurement_model=None,
        parameter_layout=GravityParameterLayout(GravityModelSpecification()),
    )
    base_mean, _ = predict_gravity_measurements(
        np.asarray((-0.2, 0.4, 1.5), dtype=np.float32), problem=base_problem
    )
    expected = (
        problem.operator.fixed_measurement_offset
        + (
            np.asarray(base_mean)
            - np.asarray(problem.operator.fixed_measurement_offset)
        )
        * 1.75
    )
    np.testing.assert_allclose(mean, expected, rtol=1.0e-5, atol=1.0e-5)


def test_latent_objective_and_gradients_are_finite_and_strategy_consistent() -> None:
    model = GravityJourneyLatentClassModel(
        num_classes=3, regularization_strength=1.0e-4
    )
    problem = _problem(model)
    raw = np.asarray((-0.2, 0.4, 1.5, 2.0, -2.0, 0.3, -0.4), dtype=np.float32)
    with jax.enable_x64(False):
        evaluation = evaluate_gravity_objective(raw, problem=problem)
        forward, forward_gradient = gravity_value_and_gradient(
            raw, problem=problem, strategy=GravityGradientStrategy.BATCHED_FORWARD
        )
        adjoint, adjoint_gradient = gravity_value_and_gradient(
            raw, problem=problem, strategy=GravityGradientStrategy.ADJOINT
        )
    assert np.isfinite(float(evaluation.objective))
    assert np.all(np.isfinite(np.asarray(forward_gradient)))
    np.testing.assert_allclose(
        forward.objective, adjoint.objective, rtol=2.0e-5, atol=2.0e-5
    )
    np.testing.assert_allclose(
        forward_gradient, adjoint_gradient, rtol=2.0e-5, atol=2.0e-5
    )
    np.testing.assert_allclose(forward.measurement_mean, evaluation.measurement_mean)
    finite_difference = np.empty_like(raw)
    for index in range(raw.size):
        delta = np.zeros_like(raw)
        delta[index] = 1.0e-3
        plus = float(evaluate_gravity_objective(raw + delta, problem=problem).objective)
        minus = float(
            evaluate_gravity_objective(raw - delta, problem=problem).objective
        )
        finite_difference[index] = (plus - minus) / 2.0e-3
    np.testing.assert_allclose(
        forward_gradient, finite_difference, rtol=5.0e-3, atol=5.0e-3
    )


def test_measurement_model_changes_fit_fingerprint_but_not_operator_identity() -> None:
    model = GravityJourneyLatentClassModel(
        class_probabilities=(0.5, 0.5), class_effects=(1.0, 1.5)
    )
    problem = _problem(model)
    compact = CompactODAssignmentLayout(
        num_od_total=3,
        active_full_indices=(0, 1, 2),
        removed_zero_full_indices=(),
        full_to_compact=(0, 1, 2),
        free_full_indices=(0, 1, 2),
        free_compact_indices=(0, 1, 2),
        free_baseline_values=(1.0, 1.0, 1.0),
        fixed_compact_indices=(),
        fixed_compact_values=(),
    )
    plain = replace(
        problem,
        measurement_model=None,
        parameter_layout=GravityParameterLayout(GravityModelSpecification()),
    )
    assert gravity_model_fingerprint(problem, compact) != gravity_model_fingerprint(
        plain, compact
    )
    assert (
        problem.operator.assignment_fingerprint == plain.operator.assignment_fingerprint
    )
    assert problem.operator.graph_fingerprint == plain.operator.graph_fingerprint
    assert problem.operator.mapping_fingerprint == plain.operator.mapping_fingerprint
