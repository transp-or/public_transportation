from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from public_transportation.inference.fixed_routing_linear_objective import (
    evaluate_linear_data_fit,
    linear_data_gradient,
    linear_data_objective,
    predict_linear_measurements,
    raw_linear_residual,
    weighted_linear_residual,
)
from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
)
from public_transportation.inference.linear_operator import SparseLinearOperator


def problem(**overrides) -> FixedRoutingLinearProblem:
    values = {
        "measurement_operator": np.array(
            [[1.0, 0.5, 0.0], [0.0, 2.0, 1.0], [0.25, 0.0, 0.75]]
        ),
        "fixed_measurement_offset": np.array([2.0, 0.0, 1.0]),
        "observations": np.array([6.0, 5.0, 3.0]),
        "observation_weights": np.array([1.0, 4.0, 0.25]),
        "prior_demand": np.array([1.0, 2.0, 3.0]),
        "lower_bounds": np.zeros(3),
        "upper_bounds": np.full(3, np.inf),
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


def test_prediction_residuals_objective_and_gradient_match_direct_numpy():
    instance = problem()
    demand = np.array([1.5, 0.5, 2.0])
    matrix = instance.measurement_operator.matrix
    expected_prediction = matrix @ demand + instance.fixed_measurement_offset
    expected_raw = expected_prediction - instance.observations
    expected_weighted = np.sqrt(instance.observation_weights) * expected_raw
    expected_objective = 0.5 * expected_weighted @ expected_weighted
    expected_gradient = matrix.T @ (instance.observation_weights * expected_raw)

    np.testing.assert_allclose(
        predict_linear_measurements(instance, demand), expected_prediction
    )
    np.testing.assert_allclose(raw_linear_residual(instance, demand), expected_raw)
    np.testing.assert_allclose(
        weighted_linear_residual(instance, demand), expected_weighted
    )
    assert linear_data_objective(instance, demand) == pytest.approx(
        expected_objective
    )
    np.testing.assert_allclose(
        linear_data_gradient(instance, demand), expected_gradient
    )


def test_complete_evaluation_is_consistent_and_immutable():
    instance = problem()
    demand = np.array([0.5, 1.0, 1.5])
    evaluation = evaluate_linear_data_fit(instance, demand)

    np.testing.assert_allclose(
        evaluation.prediction, predict_linear_measurements(instance, demand)
    )
    np.testing.assert_allclose(
        evaluation.raw_residual, raw_linear_residual(instance, demand)
    )
    np.testing.assert_allclose(
        evaluation.weighted_residual, weighted_linear_residual(instance, demand)
    )
    assert evaluation.objective == pytest.approx(
        linear_data_objective(instance, demand)
    )
    np.testing.assert_allclose(
        evaluation.gradient, linear_data_gradient(instance, demand)
    )
    for array in (
        evaluation.prediction,
        evaluation.raw_residual,
        evaluation.weighted_residual,
        evaluation.gradient,
    ):
        assert not array.flags.writeable


def test_fixed_offset_participates_in_prediction_residual_and_gradient():
    instance = problem()
    zero = np.zeros(instance.num_free_od)
    np.testing.assert_array_equal(
        predict_linear_measurements(instance, zero),
        instance.fixed_measurement_offset,
    )
    expected_raw = instance.fixed_measurement_offset - instance.observations
    np.testing.assert_array_equal(raw_linear_residual(instance, zero), expected_raw)
    np.testing.assert_allclose(
        linear_data_gradient(instance, zero),
        instance.measurement_operator.matrix.T
        @ (instance.observation_weights * expected_raw),
    )


def test_unit_weights_reduce_to_ordinary_least_squares():
    instance = problem(observation_weights=np.ones(3))
    demand = np.array([1.0, 2.0, 1.0])
    raw = raw_linear_residual(instance, demand)
    np.testing.assert_array_equal(weighted_linear_residual(instance, demand), raw)
    assert linear_data_objective(instance, demand) == pytest.approx(
        0.5 * raw @ raw
    )


def test_analytic_gradient_matches_central_finite_differences():
    instance = problem()
    demand = np.array([1.2, 2.3, 0.7])
    step = 1.0e-6
    finite_difference = np.empty(instance.num_free_od)
    for index in range(instance.num_free_od):
        direction = np.zeros(instance.num_free_od)
        direction[index] = step
        finite_difference[index] = (
            linear_data_objective(instance, demand + direction)
            - linear_data_objective(instance, demand - direction)
        ) / (2.0 * step)

    np.testing.assert_allclose(
        linear_data_gradient(instance, demand),
        finite_difference,
        rtol=2e-9,
        atol=2e-9,
    )


def test_dense_and_sparse_evaluations_are_identical():
    dense_problem = problem()
    sparse_problem = replace(
        dense_problem,
        measurement_operator=SparseLinearOperator(
            dense_problem.measurement_operator.matrix
        ),
    )
    demand = np.array([3.0, 0.25, 1.75])
    dense = evaluate_linear_data_fit(dense_problem, demand)
    sparse = evaluate_linear_data_fit(sparse_problem, demand)

    np.testing.assert_allclose(sparse.prediction, dense.prediction)
    np.testing.assert_allclose(sparse.raw_residual, dense.raw_residual)
    np.testing.assert_allclose(sparse.weighted_residual, dense.weighted_residual)
    assert sparse.objective == pytest.approx(dense.objective)
    np.testing.assert_allclose(sparse.gradient, dense.gradient)


@pytest.mark.parametrize(
    "demand",
    [np.ones(2), np.ones((3, 1)), [1.0, np.nan, 2.0]],
)
def test_calculations_reject_invalid_demand(demand):
    with pytest.raises(ValueError):
        evaluate_linear_data_fit(problem(), demand)
