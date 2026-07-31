from __future__ import annotations

import numpy as np
import pytest

from public_transportation.inference.fixed_routing_linear_dense_solver import (
    evaluate_bound_kkt,
    solve_dense_reference,
)
from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
)
from public_transportation.inference.fixed_routing_linear_regularization import (
    ridge_to_prior,
)


def problem(
    matrix,
    target,
    *,
    lower=None,
    upper=None,
    prior=None,
    offset=None,
    blocks=(),
) -> FixedRoutingLinearProblem:
    matrix = np.asarray(matrix, dtype=float)
    num_variables = matrix.shape[1]
    prior = np.zeros(num_variables) if prior is None else np.asarray(prior, dtype=float)
    return FixedRoutingLinearProblem(
        measurement_operator=matrix,
        fixed_measurement_offset=(
            np.zeros(matrix.shape[0]) if offset is None else np.asarray(offset)
        ),
        observations=np.asarray(target, dtype=float),
        observation_weights=np.ones(matrix.shape[0]),
        prior_demand=prior,
        lower_bounds=(
            np.full(num_variables, -np.inf) if lower is None else np.asarray(lower)
        ),
        upper_bounds=(
            np.full(num_variables, np.inf) if upper is None else np.asarray(upper)
        ),
        provenance=FixedRoutingLinearProvenance("od", "assignment", "mapping", 1.0),
        regularization_selection="configured" if blocks else "none",
        regularization_blocks=tuple(blocks),
    )


def assert_kkt(result, tolerance=1.0e-8):
    assert result.kkt.feasibility_inf_norm <= tolerance
    assert result.kkt.projected_gradient_inf_norm <= tolerance


def test_unconstrained_full_rank_solution_uses_svd_reference():
    instance = problem([[1.0, 0.0], [0.0, 2.0]], [3.0, 8.0])
    result = solve_dense_reference(instance)

    assert result.method == "svd_lstsq"
    assert result.success
    np.testing.assert_allclose(result.demand, [3.0, 4.0], atol=1e-12)
    assert result.evaluation.objective == pytest.approx(0.0, abs=1e-24)
    assert result.numerical_rank == 2
    assert_kkt(result, 1e-12)


def test_rank_deficient_solution_is_minimum_norm_and_reported():
    instance = problem([[1.0, 1.0], [2.0, 2.0]], [3.0, 6.0])
    result = solve_dense_reference(instance)

    np.testing.assert_allclose(result.demand, [1.5, 1.5], atol=1e-12)
    assert result.numerical_rank == 1
    assert result.evaluation.objective == pytest.approx(0.0, abs=1e-24)
    assert_kkt(result, 1e-12)


def test_bvls_finds_lower_and_upper_active_solution():
    instance = problem(
        np.eye(2),
        [0.0, 8.0],
        lower=[0.0, 0.0],
        upper=[10.0, 5.0],
        prior=[0.0, 2.0],
        offset=[2.0, 0.0],
    )
    result = solve_dense_reference(instance)

    assert result.method == "bvls"
    assert result.success
    np.testing.assert_allclose(result.demand, [0.0, 5.0], atol=1e-12)
    np.testing.assert_array_equal(result.kkt.lower_active, [True, False])
    np.testing.assert_array_equal(result.kkt.upper_active, [False, True])
    np.testing.assert_allclose(result.kkt.lower_multipliers, [2.0, 0.0])
    np.testing.assert_allclose(result.kkt.upper_multipliers, [0.0, 3.0])
    assert_kkt(result, 1e-12)


def test_equal_bounds_are_eliminated_before_bvls():
    instance = problem(
        np.eye(2),
        [10.0, 4.0],
        lower=[2.0, 0.0],
        upper=[2.0, 5.0],
        prior=[2.0, 1.0],
    )
    result = solve_dense_reference(instance)

    np.testing.assert_allclose(result.demand, [2.0, 4.0], atol=1e-12)
    np.testing.assert_array_equal(result.kkt.fixed_by_bounds, [True, False])
    assert result.kkt.upper_multipliers[0] == pytest.approx(8.0)
    assert_kkt(result, 1e-12)


def test_all_equal_bounds_require_no_optimizer_iterations():
    instance = problem(
        np.eye(2),
        [10.0, 4.0],
        lower=[2.0, 3.0],
        upper=[2.0, 3.0],
        prior=[2.0, 3.0],
    )
    result = solve_dense_reference(instance)

    assert result.method == "fixed_bounds"
    assert result.iterations == 0
    np.testing.assert_array_equal(result.demand, [2.0, 3.0])
    assert_kkt(result, 0.0)


def test_ridge_regularized_solution_matches_closed_form():
    prior = np.array([2.0, 4.0])
    strength = 3.0
    instance = problem(
        np.eye(2),
        [10.0, 0.0],
        prior=prior,
        blocks=(ridge_to_prior(prior, strength=strength),),
    )
    result = solve_dense_reference(instance)
    expected = (np.array([10.0, 0.0]) + strength * prior) / (1.0 + strength)

    np.testing.assert_allclose(result.demand, expected, atol=1e-12)
    assert_kkt(result, 1e-12)


def test_kkt_detects_infeasibility_and_wrong_bound_gradient_direction():
    diagnostics = evaluate_bound_kkt(
        demand=[-0.5, 2.0],
        gradient=[-3.0, 4.0],
        lower_bounds=[0.0, 0.0],
        upper_bounds=[5.0, 2.0],
    )
    assert diagnostics.feasibility_inf_norm == pytest.approx(0.5)
    assert diagnostics.projected_gradient_inf_norm == pytest.approx(4.0)


def test_dense_materialization_limit_is_enforced():
    instance = problem(np.eye(3), np.ones(3))
    with pytest.raises(ValueError, match="materialization limit"):
        solve_dense_reference(instance, max_materialized_entries=8)
