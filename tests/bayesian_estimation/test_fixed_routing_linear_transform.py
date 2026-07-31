from __future__ import annotations

import numpy as np
import pytest

from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
)
from public_transportation.inference.fixed_routing_linear_regularization import (
    evaluate_linear_least_squares,
    ridge_to_prior,
    scaled_ridge_to_prior,
)
from public_transportation.inference.fixed_routing_linear_transform import (
    ColumnScaledLinearOperator,
    PhysicalDemandTransform,
    build_solver_variable_least_squares_system,
)
from public_transportation.inference.linear_operator import DenseLinearOperator


def problem(**overrides) -> FixedRoutingLinearProblem:
    prior = np.array([0.0, 4.0, 10.0])
    blocks = (
        ridge_to_prior(prior, strength=0.5),
        scaled_ridge_to_prior(prior, [2.0, 4.0, 5.0], strength=1.5),
    )
    values = {
        "measurement_operator": np.array(
            [[1.0, 0.5, 0.0], [0.0, 2.0, 1.0], [0.25, 0.0, 0.75]]
        ),
        "fixed_measurement_offset": np.array([2.0, 0.0, 1.0]),
        "observations": np.array([6.0, 5.0, 3.0]),
        "observation_weights": np.array([1.0, 4.0, 0.25]),
        "prior_demand": prior,
        "lower_bounds": np.array([0.0, 1.0, -np.inf]),
        "upper_bounds": np.array([8.0, np.inf, 20.0]),
        "variable_scales": np.array([2.0, 4.0, 5.0]),
        "provenance": FixedRoutingLinearProvenance(
            od_layout_fingerprint="od",
            assignment_fingerprint="assignment",
            mapping_fingerprint="mapping",
            routing_parameter=1.0,
        ),
        "regularization_selection": "configured",
        "regularization_blocks": blocks,
    }
    values.update(overrides)
    return FixedRoutingLinearProblem(**values)


def test_transform_round_trips_and_places_prior_at_zero():
    transform = PhysicalDemandTransform.from_problem(problem())
    demand = np.array([3.0, 8.0, 5.0])
    solver_variable = transform.solver_variable_from_demand(demand)

    np.testing.assert_array_equal(
        transform.solver_variable_from_demand(transform.prior_demand),
        np.zeros(3),
    )
    np.testing.assert_allclose(
        transform.demand_from_solver_variable(solver_variable), demand
    )
    np.testing.assert_allclose(
        transform.solver_variable_from_demand(
            transform.demand_from_solver_variable(solver_variable)
        ),
        solver_variable,
    )


def test_transform_converts_heterogeneous_and_infinite_bounds():
    transform = PhysicalDemandTransform.from_problem(problem())
    np.testing.assert_array_equal(
        transform.solver_lower_bounds, np.array([0.0, -0.75, -np.inf])
    )
    np.testing.assert_array_equal(
        transform.solver_upper_bounds, np.array([4.0, np.inf, 2.0])
    )
    zero = np.zeros(3)
    assert np.all(zero >= transform.solver_lower_bounds)
    assert np.all(zero <= transform.solver_upper_bounds)


def test_zero_prior_entry_remains_estimable_with_positive_scale():
    transform = PhysicalDemandTransform.from_problem(problem())
    demand = transform.demand_from_solver_variable([1.5, 0.0, 0.0])
    assert demand[0] == pytest.approx(3.0)


def test_column_scaled_operator_matches_explicit_product_and_adjoint():
    matrix = np.array([[1.0, 2.0, 0.0], [0.0, -1.0, 3.0]])
    scales = np.array([2.0, 4.0, 0.5])
    operator = ColumnScaledLinearOperator(DenseLinearOperator(matrix), scales)
    x = np.array([1.0, -2.0, 3.0])
    v = np.array([0.25, -1.5])

    np.testing.assert_allclose(operator.matvec(x), (matrix * scales) @ x)
    np.testing.assert_allclose(operator.rmatvec(v), (matrix * scales).T @ v)
    assert np.vdot(v, operator.matvec(x)) == pytest.approx(
        np.vdot(x, operator.rmatvec(v)), rel=1e-13, abs=1e-13
    )


def test_solver_system_residual_objective_and_gradient_match_physical_problem():
    instance = problem()
    system = build_solver_variable_least_squares_system(instance)
    demand = np.array([3.0, 8.0, 5.0])
    solver_variable = system.transform.solver_variable_from_demand(demand)
    physical = evaluate_linear_least_squares(instance, demand)
    solver_residual = system.operator.matvec(solver_variable) - system.target

    np.testing.assert_allclose(solver_residual, physical.augmented_residual)
    assert 0.5 * solver_residual @ solver_residual == pytest.approx(
        physical.objective
    )
    np.testing.assert_allclose(
        system.operator.rmatvec(solver_residual),
        system.transform.solver_gradient_from_physical(physical.gradient),
    )


def test_gradient_chain_rule_matches_solver_coordinate_finite_differences():
    instance = problem()
    system = build_solver_variable_least_squares_system(instance)
    solver_variable = np.array([1.0, -0.5, 0.25])
    demand = system.transform.demand_from_solver_variable(solver_variable)
    expected = system.transform.solver_gradient_from_physical(
        evaluate_linear_least_squares(instance, demand).gradient
    )
    step = 1.0e-6
    finite_difference = np.empty(3)
    for index in range(3):
        direction = np.zeros(3)
        direction[index] = step

        def objective(value):
            residual = system.operator.matvec(value) - system.target
            return 0.5 * residual @ residual

        finite_difference[index] = (
            objective(solver_variable + direction)
            - objective(solver_variable - direction)
        ) / (2.0 * step)
    np.testing.assert_allclose(expected, finite_difference, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scales", [1.0, 0.0], "strictly positive"),
        ("scales", [1.0, np.inf], "must be finite"),
        ("lower_bounds", [0.0], "shape \\(2,\\)"),
        ("upper_bounds", [1.0, -np.inf], "not NaN or -inf"),
    ],
)
def test_transform_rejects_invalid_inputs(field, value, message):
    values = {
        "prior_demand": [1.0, 2.0],
        "scales": [1.0, 2.0],
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [np.inf, np.inf],
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        PhysicalDemandTransform(**values)
