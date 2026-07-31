from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from public_transportation.inference.fixed_routing_linear_dense_solver import (
    solve_dense_reference,
)
from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
)
from public_transportation.inference.fixed_routing_linear_regularization import (
    ridge_to_prior,
)
from public_transportation.inference.fixed_routing_linear_trf_solver import (
    TRFLSMRConfig,
    solve_trf_lsmr,
)
from public_transportation.inference.linear_operator import SparseLinearOperator


def problem(
    matrix,
    observations,
    *,
    offset=None,
    lower=None,
    upper=None,
    prior=None,
    scales=None,
    blocks=(),
) -> FixedRoutingLinearProblem:
    matrix = np.asarray(matrix, dtype=float)
    size = matrix.shape[1]
    prior = np.zeros(size) if prior is None else np.asarray(prior, dtype=float)
    return FixedRoutingLinearProblem(
        measurement_operator=matrix,
        fixed_measurement_offset=(
            np.zeros(matrix.shape[0]) if offset is None else np.asarray(offset)
        ),
        observations=np.asarray(observations, dtype=float),
        observation_weights=np.ones(matrix.shape[0]),
        prior_demand=prior,
        lower_bounds=(np.full(size, -np.inf) if lower is None else np.asarray(lower)),
        upper_bounds=(np.full(size, np.inf) if upper is None else np.asarray(upper)),
        variable_scales=np.ones(size) if scales is None else np.asarray(scales),
        provenance=FixedRoutingLinearProvenance("od", "assignment", "mapping", 1.0),
        regularization_selection="configured" if blocks else "none",
        regularization_blocks=tuple(blocks),
    )


def compare_with_reference(instance, *, demand_tolerance=1.0e-7):
    reference = solve_dense_reference(instance)
    iterative = solve_trf_lsmr(
        instance,
        config=TRFLSMRConfig(
            tolerance=1e-10,
            lsmr_tolerance=1e-12,
            active_tolerance=1e-7,
        ),
    )
    assert iterative.success
    assert iterative.matvec_count > 0
    assert iterative.rmatvec_count > 0
    np.testing.assert_allclose(
        iterative.demand,
        reference.demand,
        rtol=demand_tolerance,
        atol=demand_tolerance,
    )
    assert iterative.evaluation.objective == pytest.approx(
        reference.evaluation.objective, rel=1e-9, abs=1e-9
    )
    assert iterative.kkt.feasibility_inf_norm <= 1e-9
    assert iterative.kkt.projected_gradient_inf_norm <= 1e-6
    return iterative


def test_trf_lsmr_matches_full_rank_unconstrained_reference():
    result = compare_with_reference(
        problem([[1.0, 0.0], [0.0, 2.0]], [3.0, 8.0])
    )
    np.testing.assert_allclose(result.demand, [3.0, 4.0], atol=1e-8)


def test_trf_lsmr_matches_simultaneous_lower_and_upper_bounds():
    instance = problem(
        np.eye(2),
        [0.0, 8.0],
        offset=[2.0, 0.0],
        lower=[0.0, 0.0],
        upper=[10.0, 5.0],
        prior=[0.0, 2.0],
        scales=[2.0, 4.0],
    )
    result = compare_with_reference(instance)
    np.testing.assert_allclose(result.demand, [0.0, 5.0], atol=1e-7)
    np.testing.assert_array_equal(result.kkt.lower_active, [True, False])
    np.testing.assert_array_equal(result.kkt.upper_active, [False, True])


def test_trf_lsmr_handles_equal_bounds_by_reduction():
    instance = problem(
        np.eye(2),
        [10.0, 4.0],
        lower=[2.0, 0.0],
        upper=[2.0, 5.0],
        prior=[2.0, 1.0],
    )
    result = compare_with_reference(instance)
    np.testing.assert_allclose(result.demand, [2.0, 4.0], atol=1e-7)
    np.testing.assert_array_equal(result.kkt.fixed_by_bounds, [True, False])


def test_trf_lsmr_handles_all_variables_fixed_without_products():
    instance = problem(
        np.eye(2),
        [10.0, 4.0],
        lower=[2.0, 3.0],
        upper=[2.0, 3.0],
        prior=[2.0, 3.0],
    )
    result = solve_trf_lsmr(instance)
    assert result.success
    assert result.iterations == 0
    assert result.matvec_count == 0
    assert result.rmatvec_count == 0
    np.testing.assert_array_equal(result.demand, [2.0, 3.0])


def test_trf_lsmr_matches_regularized_rank_deficient_reference():
    prior = np.array([1.0, 3.0])
    instance = problem(
        [[1.0, 1.0], [2.0, 2.0]],
        [3.0, 6.0],
        lower=[0.0, 0.0],
        upper=[np.inf, np.inf],
        prior=prior,
        scales=[2.0, 4.0],
        blocks=(ridge_to_prior(prior, strength=0.25),),
    )
    compare_with_reference(instance, demand_tolerance=2e-7)


def test_unregularized_rank_deficiency_compares_predictions_not_demand_vectors():
    instance = problem(
        [[1.0, 1.0], [2.0, 2.0]],
        [3.0, 6.0],
        lower=[0.0, 0.0],
        upper=[np.inf, np.inf],
    )
    reference = solve_dense_reference(instance)
    iterative = solve_trf_lsmr(
        instance,
        config=TRFLSMRConfig(tolerance=1e-10, lsmr_tolerance=1e-12),
    )

    assert iterative.success
    np.testing.assert_allclose(
        iterative.evaluation.data_fit.prediction,
        reference.evaluation.data_fit.prediction,
        atol=1e-8,
    )
    assert iterative.evaluation.objective == pytest.approx(
        reference.evaluation.objective, abs=1e-12
    )
    assert iterative.kkt.projected_gradient_inf_norm <= 1e-7


def test_trf_lsmr_uses_sparse_measurement_operator_without_changing_solution():
    dense = problem(
        [[1.0, 0.0], [0.5, 2.0]],
        [3.0, 8.0],
        lower=[0.0, 0.0],
        upper=[np.inf, np.inf],
    )
    sparse = replace(
        dense,
        measurement_operator=SparseLinearOperator(
            dense.measurement_operator.matrix
        ),
    )
    dense_result = solve_trf_lsmr(dense)
    sparse_result = solve_trf_lsmr(sparse)
    np.testing.assert_allclose(sparse_result.demand, dense_result.demand)
    assert sparse_result.evaluation.objective == pytest.approx(
        dense_result.evaluation.objective
    )


def test_diagonal_preconditioner_preserves_sparse_solution():
    instance = problem(
        [[1.0e-3, 0.0], [0.0, 1.0e3]],
        [2.0e-3, 3.0e3],
        lower=[0.0, 0.0],
        upper=[np.inf, np.inf],
    )
    instance = replace(
        instance,
        measurement_operator=SparseLinearOperator(
            instance.measurement_operator.matrix
        ),
    )
    reference = solve_dense_reference(instance)
    scaled = solve_trf_lsmr(
        instance,
        config=TRFLSMRConfig(
            diagonal_preconditioner=True,
            tolerance=1.0e-10,
            lsmr_tolerance=1.0e-12,
        ),
    )
    np.testing.assert_allclose(
        scaled.demand, reference.demand, rtol=1e-7, atol=1e-7
    )
    assert scaled.preconditioner_seconds >= 0.0
    assert scaled.preparation_matvec_count == 1
    assert scaled.final_matvec_count == 1
    assert scaled.final_rmatvec_count == 1


def test_strict_kkt_success_policy_rejects_relative_cost_only_success():
    instance = problem(
        [[1.0e-3, 0.0], [0.0, 1.0e3]],
        [2.0e-3, 3.0e3],
        lower=[0.0, 0.0],
        upper=[np.inf, np.inf],
    )
    result = solve_trf_lsmr(
        instance,
        config=TRFLSMRConfig(
            success_policy="both", kkt_tolerance=1.0e-12
        ),
    )
    assert not result.success
    assert "policy=both" in result.stopping_condition
    assert "Rejected" in result.message


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tolerance": 0.0},
        {"lsmr_tolerance": 0.0},
        {"lsmr_tolerance": "invalid"},
        {"max_iterations": 0},
        {"lsmr_max_iterations": 0},
        {"active_tolerance": -1.0},
        {"verbose": 3},
        {"success_policy": "invalid"},
        {"kkt_tolerance": 0.0},
    ],
)
def test_trf_lsmr_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        TRFLSMRConfig(**kwargs)
