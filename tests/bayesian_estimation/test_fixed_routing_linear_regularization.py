from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
    LinearRegularizationBlock,
)
from public_transportation.inference.fixed_routing_linear_regularization import (
    build_augmented_linear_least_squares_system,
    evaluate_linear_least_squares,
    evaluate_regularization_block,
    ridge_to_prior,
    scaled_ridge_to_prior,
)


def problem(**overrides) -> FixedRoutingLinearProblem:
    values = {
        "measurement_operator": np.array([[1.0, 0.5], [0.0, 2.0], [0.25, 0.75]]),
        "fixed_measurement_offset": np.array([2.0, 0.0, 1.0]),
        "observations": np.array([6.0, 5.0, 3.0]),
        "observation_weights": np.array([1.0, 4.0, 0.25]),
        "prior_demand": np.array([1.0, 3.0]),
        "lower_bounds": np.zeros(2),
        "upper_bounds": np.full(2, np.inf),
        "provenance": FixedRoutingLinearProvenance(
            od_layout_fingerprint="od",
            assignment_fingerprint="assignment",
            mapping_fingerprint="mapping",
            routing_parameter=1.0,
        ),
        "regularization_selection": "none",
    }
    values.update(overrides)
    return FixedRoutingLinearProblem(**values)


def configured_problem(*blocks) -> FixedRoutingLinearProblem:
    return problem(
        regularization_selection="configured",
        regularization_blocks=tuple(blocks),
    )


def test_ridge_to_prior_is_zero_at_prior_and_has_expected_value():
    block = ridge_to_prior([2.0, 4.0], strength=3.0)
    at_prior = evaluate_regularization_block(block, [2.0, 4.0])
    np.testing.assert_array_equal(at_prior.residual, np.zeros(2))
    np.testing.assert_array_equal(at_prior.gradient, np.zeros(2))
    assert at_prior.objective == 0.0

    demand = np.array([3.0, 2.0])
    evaluation = evaluate_regularization_block(block, demand)
    difference = demand - np.array([2.0, 4.0])
    np.testing.assert_allclose(evaluation.residual, np.sqrt(3.0) * difference)
    assert evaluation.objective == pytest.approx(1.5 * difference @ difference)
    np.testing.assert_allclose(evaluation.gradient, 3.0 * difference)


def test_scaled_ridge_uses_declared_scales_and_supports_zero_prior():
    block = scaled_ridge_to_prior(
        [0.0, 10.0], [2.0, 5.0], strength=4.0
    )
    demand = np.array([4.0, 5.0])
    expected_unscaled = np.array([2.0, -1.0])
    evaluation = evaluate_regularization_block(block, demand)
    np.testing.assert_allclose(evaluation.residual, 2.0 * expected_unscaled)
    assert evaluation.objective == pytest.approx(
        0.5 * (2.0 * expected_unscaled) @ (2.0 * expected_unscaled)
    )
    np.testing.assert_allclose(
        evaluation.gradient, np.array([4.0, -0.8])
    )


def test_zero_strength_block_has_zero_residual_objective_and_gradient():
    block = ridge_to_prior([1.0, 2.0], strength=0.0)
    evaluation = evaluate_regularization_block(block, [10.0, 20.0])
    np.testing.assert_array_equal(evaluation.residual, np.zeros(2))
    np.testing.assert_array_equal(evaluation.gradient, np.zeros(2))
    assert evaluation.objective == 0.0


def test_none_selection_augments_only_the_data_residual():
    instance = problem()
    demand = np.array([2.0, 1.0])
    system = build_augmented_linear_least_squares_system(instance)
    evaluation = evaluate_linear_least_squares(instance, demand)

    assert system.operator.shape == (
        instance.num_measurements,
        instance.num_free_od,
    )
    np.testing.assert_allclose(
        system.operator.matvec(demand) - system.target,
        evaluation.data_fit.weighted_residual,
    )
    assert evaluation.regularization == ()
    assert evaluation.objective == pytest.approx(evaluation.data_fit.objective)
    np.testing.assert_array_equal(evaluation.gradient, evaluation.data_fit.gradient)


def test_multiple_blocks_equal_augmented_residual_objective_and_gradient():
    first = ridge_to_prior([1.0, 3.0], strength=2.0, name="absolute")
    second = scaled_ridge_to_prior(
        [1.0, 3.0], [2.0, 4.0], strength=5.0, name="scaled"
    )
    instance = configured_problem(first, second)
    demand = np.array([2.5, 1.25])
    system = build_augmented_linear_least_squares_system(instance)
    evaluation = evaluate_linear_least_squares(instance, demand)
    augmented_residual = system.operator.matvec(demand) - system.target

    np.testing.assert_allclose(augmented_residual, evaluation.augmented_residual)
    assert evaluation.objective == pytest.approx(
        0.5 * augmented_residual @ augmented_residual
    )
    np.testing.assert_allclose(
        evaluation.gradient,
        system.operator.rmatvec(augmented_residual),
    )
    assert [item.name for item in evaluation.regularization] == [
        "absolute",
        "scaled",
    ]
    assert system.regularization_slices == (
        ("absolute", slice(3, 5)),
        ("scaled", slice(5, 7)),
    )


def test_augmented_operator_satisfies_adjoint_identity():
    instance = configured_problem(
        ridge_to_prior([1.0, 3.0], strength=2.0),
        LinearRegularizationBlock(
            "difference", [[1.0, -1.0]], [0.0], 0.75
        ),
    )
    system = build_augmented_linear_least_squares_system(instance)
    x = np.array([2.0, -0.5])
    v = np.linspace(-1.0, 2.0, system.operator.shape[0])
    assert np.vdot(v, system.operator.matvec(x)) == pytest.approx(
        np.vdot(x, system.operator.rmatvec(v)), rel=1e-13, abs=1e-13
    )


def test_complete_gradient_matches_central_finite_differences():
    instance = configured_problem(
        ridge_to_prior([1.0, 3.0], strength=2.0),
        LinearRegularizationBlock(
            "difference", [[1.0, -1.0]], [0.0], 0.75
        ),
    )
    demand = np.array([1.4, 2.1])
    step = 1.0e-6
    finite_difference = np.empty(2)
    for index in range(2):
        direction = np.zeros(2)
        direction[index] = step
        finite_difference[index] = (
            evaluate_linear_least_squares(instance, demand + direction).objective
            - evaluate_linear_least_squares(instance, demand - direction).objective
        ) / (2.0 * step)
    np.testing.assert_allclose(
        evaluate_linear_least_squares(instance, demand).gradient,
        finite_difference,
        rtol=2e-9,
        atol=2e-9,
    )


def test_unspecified_selection_cannot_be_evaluated_or_augmented():
    instance = replace(problem(), regularization_selection="unspecified")
    with pytest.raises(ValueError, match="explicitly select 'none'"):
        build_augmented_linear_least_squares_system(instance)
    with pytest.raises(ValueError, match="explicitly select 'none'"):
        evaluate_linear_least_squares(instance, instance.prior_demand)


@pytest.mark.parametrize(
    ("prior", "scales", "message"),
    [
        ([1.0, 2.0], [1.0], "shape \\(2,\\)"),
        ([1.0, 2.0], [1.0, 0.0], "strictly positive"),
        ([1.0, 2.0], [1.0, np.inf], "must be finite"),
    ],
)
def test_scaled_ridge_rejects_invalid_scales(prior, scales, message):
    with pytest.raises(ValueError, match=message):
        scaled_ridge_to_prior(prior, scales, strength=1.0)
