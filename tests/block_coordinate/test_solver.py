"""Tests for bounded local solves and atomic update acceptance."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from public_transportation.inference.block_coordinate import (
    BlockSolverConfig,
    BlockSolverResult,
    BlockUpdatePolicy,
    ColumnSelectedLinearOperator,
    ODBlock,
    build_conditional_block_objective,
    decide_block_update,
    initialize_incremental_state,
    prepare_separable_quadratic_prior,
    solve_and_decide_block_update,
    solve_conditional_block,
    validate_incremental_prediction,
)
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


def _problem(
    matrix: np.ndarray,
    observations: np.ndarray,
    *,
    prior: np.ndarray | None = None,
    strength: float = 0.0,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> FixedRoutingLinearProblem:
    variables = matrix.shape[1]
    prior = np.ones(variables) if prior is None else prior
    blocks = () if strength == 0.0 else (ridge_to_prior(prior, strength=strength),)
    return FixedRoutingLinearProblem(
        measurement_operator=matrix,
        fixed_measurement_offset=np.zeros(matrix.shape[0]),
        observations=observations,
        observation_weights=np.ones(matrix.shape[0]),
        prior_demand=prior,
        lower_bounds=np.zeros(variables) if lower is None else lower,
        upper_bounds=np.full(variables, np.inf) if upper is None else upper,
        provenance=FixedRoutingLinearProvenance("od", "assignment", "mapping", 1.0),
        regularization_selection="none" if not blocks else "configured",
        regularization_blocks=blocks,
    )


def _block(columns: tuple[int, ...]) -> ODBlock:
    return ODBlock(
        block_id="block",
        free_column_indices=columns,
        active_od_indices=columns,
        destination_group_indices=(0,),
        time_bin_ids=("t0",),
    )


def _conditional(problem: FixedRoutingLinearProblem, flow: np.ndarray, columns):
    state = initialize_incremental_state(
        problem.measurement_operator, flow, problem.fixed_measurement_offset
    )
    block = _block(columns)
    operator = ColumnSelectedLinearOperator(problem.measurement_operator, columns)
    objective = build_conditional_block_objective(
        problem,
        prepare_separable_quadratic_prior(problem),
        state,
        block,
        operator,
    )
    return state, block, operator, objective


def test_single_block_solution_matches_dense_bounded_reference() -> None:
    rng = np.random.default_rng(2026)
    matrix = rng.uniform(0.1, 1.0, size=(9, 4))
    observations = rng.uniform(1.0, 6.0, size=9)
    problem = _problem(
        matrix,
        observations,
        prior=np.array([1.0, 2.0, 1.5, 3.0]),
        strength=0.7,
        upper=np.array([4.0, 4.0, 4.0, 4.0]),
    )
    initial = np.array([1.0, 2.0, 1.5, 3.0])
    state, block, operator, objective = _conditional(problem, initial, (0, 1, 2, 3))
    result = solve_conditional_block(
        objective,
        initial,
        config=BlockSolverConfig(maximum_iterations=200, tolerance=1.0e-12),
    )
    reference = solve_dense_reference(problem, tolerance=1.0e-12)

    assert result.success
    np.testing.assert_allclose(
        result.candidate_local_flow, reference.demand, rtol=2.0e-6, atol=2.0e-6
    )
    decision = decide_block_update(state, block, operator, objective, result)
    assert decision.accepted
    assert decision.objective_improvement >= 0.0
    assert validate_incremental_prediction(decision.state, problem.measurement_operator).within_tolerance


def test_subblock_update_decreases_global_objective_and_preserves_other_columns() -> None:
    matrix = np.array([[1.0, 0.2, 0.4], [0.0, 1.0, 0.5], [0.5, 0.0, 1.0]])
    problem = _problem(matrix, np.array([3.0, 2.0, 4.0]))
    initial = np.array([0.5, 0.5, 0.5])
    state, block, operator, objective = _conditional(problem, initial, (0, 2))

    decision = solve_and_decide_block_update(
        state,
        block,
        operator,
        objective,
        solver_config=BlockSolverConfig(maximum_iterations=100, tolerance=1.0e-12),
    )
    assert decision.accepted
    assert decision.accepted_evaluation.objective <= decision.solver_result.initial_evaluation.objective
    assert decision.state.free_flow[1] == initial[1]
    np.testing.assert_allclose(
        decision.state.prediction,
        matrix @ decision.state.free_flow,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_backtracking_accepts_overshoot_only_after_objective_decreases() -> None:
    problem = _problem(np.array([[1.0]]), np.array([1.0]), upper=np.array([10.0]))
    state, block, operator, objective = _conditional(problem, np.array([0.0]), (0,))
    initial_evaluation = objective.evaluate([0.0])
    overshoot_evaluation = objective.evaluate([3.0])
    solver_result = BlockSolverResult(
        initial_local_flow=[0.0],
        candidate_local_flow=[3.0],
        initial_evaluation=initial_evaluation,
        candidate_evaluation=overshoot_evaluation,
        success=True,
        status=0,
        message="synthetic overshoot",
        iterations=1,
        function_evaluations=1,
        gradient_evaluations=1,
        elapsed_seconds=0.0,
    )
    decision = decide_block_update(
        state,
        block,
        operator,
        objective,
        solver_result,
        policy=BlockUpdatePolicy(maximum_backtracking_steps=3),
    )
    assert decision.accepted
    assert decision.backtracking_steps == 1
    assert decision.applied_damping == 0.5
    np.testing.assert_allclose(decision.state.free_flow, [1.5])
    assert decision.objective_improvement > 0.0


def test_failed_and_increasing_candidates_return_exact_original_state() -> None:
    problem = _problem(np.array([[1.0]]), np.array([1.0]), upper=np.array([10.0]))
    state, block, operator, objective = _conditional(problem, np.array([0.0]), (0,))
    solved = solve_conditional_block(objective, [0.0])

    failed = replace(solved, success=False, status=2, message="synthetic failure")
    failure_decision = decide_block_update(state, block, operator, objective, failed)
    assert not failure_decision.accepted
    assert failure_decision.reason == "solver_failure"
    assert failure_decision.state is state
    assert failure_decision.proposal is None

    increasing = replace(
        solved,
        candidate_local_flow=np.array([3.0]),
        candidate_evaluation=objective.evaluate([3.0]),
    )
    increase_decision = decide_block_update(
        state,
        block,
        operator,
        objective,
        increasing,
        policy=BlockUpdatePolicy(maximum_backtracking_steps=0),
    )
    assert not increase_decision.accepted
    assert increase_decision.reason == "objective_increase"
    assert increase_decision.state is state
    np.testing.assert_array_equal(state.free_flow, [0.0])
    np.testing.assert_array_equal(state.prediction, [0.0])

    outside_bounds = replace(
        solved,
        candidate_local_flow=np.array([11.0]),
        candidate_evaluation=solved.initial_evaluation,
    )
    bounds_decision = decide_block_update(
        state, block, operator, objective, outside_bounds
    )
    assert not bounds_decision.accepted
    assert bounds_decision.reason == "bound_violation"
    assert bounds_decision.state is state


def test_all_fixed_local_variables_produce_successful_no_change() -> None:
    problem = _problem(
        np.array([[1.0]]),
        np.array([4.0]),
        prior=np.array([2.0]),
        lower=np.array([2.0]),
        upper=np.array([2.0]),
    )
    state, block, operator, objective = _conditional(problem, np.array([2.0]), (0,))
    result = solve_conditional_block(objective, [2.0])
    decision = decide_block_update(state, block, operator, objective, result)
    assert result.success
    assert result.iterations == 0
    assert not decision.accepted
    assert decision.reason == "no_flow_change"
    assert decision.state is state


@pytest.mark.parametrize(
    "policy",
    [
        {"update_damping": 0.0},
        {"backtracking_factor": 1.0},
        {"maximum_backtracking_steps": -1},
        {"minimum_damping": 0.0},
        {"absolute_objective_tolerance": -1.0},
    ],
)
def test_update_policy_rejects_invalid_controls(policy) -> None:
    with pytest.raises(ValueError):
        BlockUpdatePolicy(**policy)
