from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)
from public_transportation.inference.fixed_routing_measurement_operator import (
    FixedRoutingMeasurementOperator,
    MeasurementOperatorMetrics,
)
from public_transportation.inference.gravity import (
    GravityAdequacyConfig,
    GravityEffectScope,
    GravityEstimatorConfig,
    GravityExecutionPolicy,
    GravityFeatures,
    GravityLikelihood,
    GravityModelSpecification,
    GravityObjectiveProblem,
    GravityParameterLayout,
    GravityRecommendationConfig,
    GravityValidationMetadata,
    add_gravity_relaxation,
    estimate_gravity_model,
    generate_gravity_demand,
    predict_gravity_measurements,
    recommend_gravity_relaxations,
    validate_full_data_gravity_adequacy,
)


def recommendation_case(*, destination_effect: float = 0.0, shift=None):
    cells = 8
    compact = CompactODAssignmentLayout(
        cells,
        tuple(range(cells)),
        (),
        tuple(range(cells)),
        tuple(range(cells)),
        tuple(range(cells)),
        tuple(1.0 for _ in range(cells)),
        (),
        (),
    )
    features = GravityFeatures(
        canonical_od_index=np.arange(cells),
        origin_index=np.asarray((0, 0, 0, 0, 1, 1, 1, 1)),
        destination_index=np.asarray((0, 1, 0, 1, 0, 1, 0, 1)),
        departure_time_index=np.asarray((0, 0, 1, 1, 0, 0, 1, 1)),
        origin_time_group_index=np.asarray((0, 0, 1, 1, 2, 2, 3, 3)),
        journey_time=np.asarray((5, 15, 7, 12, 20, 8, 18, 6), dtype=float),
        transfer_count=np.asarray((0, 1, 0, 1, 1, 0, 1, 0)),
        structural_feasible=np.ones(cells, dtype=bool),
        origin_time_totals=np.asarray((30, 40, 50, 60), dtype=float),
        destination_attractiveness=np.ones(cells),
        num_origins=2,
        num_destinations=2,
        num_departure_times=2,
        od_layout_fingerprint=compact.fingerprint,
        journey_time_scale=10,
        origin_zone_index=np.asarray((0, 0, 0, 0, 1, 1, 1, 1)),
        destination_zone_index=np.asarray((0, 1, 0, 1, 0, 1, 0, 1)),
        time_period_index=np.asarray((0, 0, 1, 1, 0, 0, 1, 1)),
    )
    matrix = jnp.eye(cells, dtype=jnp.float64)
    operator = FixedRoutingMeasurementOperator(
        matrix,
        jnp.zeros(cells),
        "dense",
        cells,
        cells,
        cells,
        compact.fingerprint,
        compact.fingerprint,
        "assignment",
        "graph",
        "mapping",
        1.0,
        "float64",
        MeasurementOperatorMetrics(
            0, matrix.nbytes, matrix.nbytes, 0, cells, cells**2, 1 / cells, cells
        ),
    )
    parent_layout = GravityParameterLayout(GravityModelSpecification())
    parent_raw = parent_layout.raw_from_physical((0.6, 0.9, 10.0))
    child_specification, _ = add_gravity_relaxation(
        parent_layout.specification,
        features=features,
        scope=GravityEffectScope.DESTINATION_ZONE,
        ridge=1.0,
    )
    child_layout = GravityParameterLayout(child_specification)
    truth = np.concatenate((parent_raw, np.asarray((destination_effect,))))
    observations = np.asarray(
        generate_gravity_demand(
            truth, features=features, parameter_layout=child_layout
        ).demand
    )
    if shift is not None:
        observations = observations + np.asarray(shift)
    problem = GravityObjectiveProblem(
        features,
        parent_layout,
        operator,
        observations,
        GravityLikelihood.NEGATIVE_BINOMIAL,
    )
    result = estimate_gravity_model(
        problem=problem,
        compact_layout=compact,
        initial_raw_parameters=parent_raw,
        config=GravityEstimatorConfig(maximum_iterations=12),
        execution=GravityExecutionPolicy(gradient_strategy="adjoint"),
    )
    return problem, compact, result


def metadata() -> GravityValidationMetadata:
    return GravityValidationMetadata(
        8,
        destination_zone=np.asarray(("a", "b") * 4),
        origin_zone=np.asarray(("west",) * 4 + ("east",) * 4),
        time_period=np.asarray(("am", "am", "pm", "pm") * 2),
        vehicle_journey=np.asarray(("j1",) * 4 + ("j2",) * 4),
    )


def test_catalog_scores_atomic_children_without_changing_parent():
    with jax.enable_x64():
        problem, compact, result = recommendation_case(destination_effect=0.55)
        report = validate_full_data_gravity_adequacy(
            result=result, problem=problem, compact_layout=compact, metadata=metadata()
        )
        raw_before = result.raw_parameters.copy()
        recommendations = recommend_gravity_relaxations(
            result=result,
            problem=problem,
            compact_layout=compact,
            adequacy_report=report,
            metadata=metadata(),
        )
        assert recommendations.advisory_only
        assert len(recommendations.candidates) == 5
        destination = next(
            item
            for item in recommendations.candidates
            if item.candidate_name == "destination_zone_attractiveness"
        )
        assert destination.applicable
        assert destination.added_parameters == 1
        assert destination.approximate_gain is not None
        assert destination.approximate_gain > 0
        assert destination.expected_calibration_deviance_improvement == pytest.approx(
            2 * destination.approximate_gain
        )
        assert destination.observation_support == 8
        assert destination.support_groups == 2
        assert destination.explanation
        np.testing.assert_array_equal(result.raw_parameters, raw_before)
        assert problem.parameter_layout.specification == GravityModelSpecification()


def test_perfect_parent_has_negligible_atomic_scores():
    with jax.enable_x64():
        problem, compact, result = recommendation_case()
        # Use the exact parent prediction to remove optimizer stopping noise.
        exact = np.asarray(
            predict_gravity_measurements(result.raw_parameters, problem=problem)[0]
        )
        exact_problem = replace(problem, observations=exact)
        exact_result = replace(
            result,
            model_fingerprint="placeholder",
            predicted_measurements=exact,
        )
        from public_transportation.inference.gravity import gravity_model_fingerprint

        exact_result = replace(
            exact_result,
            model_fingerprint=gravity_model_fingerprint(exact_problem, compact),
        )
        report = validate_full_data_gravity_adequacy(
            result=exact_result, problem=exact_problem, compact_layout=compact
        )
        recommendations = recommend_gravity_relaxations(
            result=exact_result,
            problem=exact_problem,
            compact_layout=compact,
            adequacy_report=report,
        )
        atomic = recommendations.candidates[:3]
        assert all(item.recommendation_strength == "none" for item in atomic)
        assert all((item.approximate_gain or 0) < 1e-10 for item in atomic)


def test_unavailable_mapping_is_catalogued_and_data_warnings_are_advisory():
    with jax.enable_x64():
        problem, compact, result = recommendation_case(
            shift=(20, 15, 10, 5, -1, -2, -3, -4)
        )
        no_zones = replace(
            problem.features,
            destination_zone_index=None,
            time_period_index=None,
            origin_zone_index=None,
        )
        changed_problem = replace(problem, features=no_zones)
        from public_transportation.inference.gravity import gravity_model_fingerprint

        changed_result = replace(
            result,
            model_fingerprint=gravity_model_fingerprint(changed_problem, compact),
        )
        predicted = np.asarray(
            predict_gravity_measurements(
                changed_result.raw_parameters, problem=changed_problem
            )[0]
        )
        changed_result = replace(changed_result, predicted_measurements=predicted)
        report = validate_full_data_gravity_adequacy(
            result=changed_result,
            problem=changed_problem,
            compact_layout=compact,
            metadata=metadata(),
            config=GravityAdequacyConfig(
                standardized_residual_thresholds=(0.1, 0.2),
                systematic_group_mean_threshold=0.01,
                journey_correlation_threshold=0.0,
            ),
        )
        recommendations = recommend_gravity_relaxations(
            result=changed_result,
            problem=changed_problem,
            compact_layout=compact,
            adequacy_report=report,
            metadata=metadata(),
            config=GravityRecommendationConfig(
                feature_pattern_correlation_threshold=0.0
            ),
        )
        assert all(not item.applicable for item in recommendations.candidates[:3])
        assert all(item.weak_identification_warnings for item in recommendations.candidates)
        assert recommendations.diagnostic_warnings


def test_rejects_mismatched_result_and_adequacy_report():
    with jax.enable_x64():
        problem, compact, result = recommendation_case()
        report = validate_full_data_gravity_adequacy(
            result=result, problem=problem, compact_layout=compact
        )
        with pytest.raises(ValueError, match="result"):
            recommend_gravity_relaxations(
                result=replace(result, model_fingerprint="wrong"),
                problem=problem,
                compact_layout=compact,
                adequacy_report=report,
            )
