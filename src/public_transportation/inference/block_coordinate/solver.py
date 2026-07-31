"""Bounded local solves and atomic monotone block-update decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from scipy.optimize import minimize

from .blocks import ODBlock
from .incremental import (
    BlockUpdateProposal,
    IncrementalLinearState,
    apply_incremental_update,
    propose_incremental_update,
)
from .objective import BlockObjectiveEvaluation, ConditionalBlockObjective
from .operator import BlockLinearOperatorProtocol

Array = np.ndarray
BlockUpdateReason = Literal[
    "accepted",
    "no_flow_change",
    "solver_failure",
    "objective_increase",
    "nonfinite_candidate",
    "bound_violation",
]


def _immutable_vector(value: object, *, name: str, size: int) -> Array:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")
    array = np.array(array, dtype=np.float64, copy=True)
    if array.ndim != 1 or array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class BlockSolverConfig:
    """Numerical controls for one bounded conditional solve."""

    maximum_iterations: int = 50
    tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive.")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and strictly positive.")


@dataclass(frozen=True, slots=True)
class BlockUpdatePolicy:
    """Monotonicity and backtracking controls for an atomic block update."""

    update_damping: float = 1.0
    backtracking_factor: float = 0.5
    maximum_backtracking_steps: int = 12
    minimum_damping: float = 1.0e-8
    absolute_objective_tolerance: float = 1.0e-10
    relative_objective_tolerance: float = 1.0e-12
    require_solver_success: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.update_damping <= 1.0:
            raise ValueError("update_damping must be in (0, 1].")
        if not 0.0 < self.backtracking_factor < 1.0:
            raise ValueError("backtracking_factor must be in (0, 1).")
        if self.maximum_backtracking_steps < 0:
            raise ValueError("maximum_backtracking_steps must be non-negative.")
        if not math.isfinite(self.minimum_damping) or not 0.0 < self.minimum_damping <= 1.0:
            raise ValueError("minimum_damping must be finite and in (0, 1].")
        for name in ("absolute_objective_tolerance", "relative_objective_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class BlockSolverResult:
    """Candidate returned by one local bounded optimization."""

    initial_local_flow: Array
    candidate_local_flow: Array
    initial_evaluation: BlockObjectiveEvaluation
    candidate_evaluation: BlockObjectiveEvaluation
    success: bool
    status: int
    message: str
    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        size = np.asarray(self.initial_local_flow).size
        initial = _immutable_vector(
            self.initial_local_flow, name="initial_local_flow", size=size
        )
        candidate = _immutable_vector(
            self.candidate_local_flow, name="candidate_local_flow", size=size
        )
        if not self.message.strip():
            raise ValueError("solver message must be nonempty.")
        for name in ("iterations", "function_evaluations", "gradient_evaluations"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative.")
        object.__setattr__(self, "initial_local_flow", initial)
        object.__setattr__(self, "candidate_local_flow", candidate)


@dataclass(frozen=True, slots=True)
class BlockUpdateDecision:
    """Accepted new state or an immutable rejection of a local candidate."""

    accepted: bool
    reason: BlockUpdateReason
    state: IncrementalLinearState
    solver_result: BlockSolverResult
    accepted_evaluation: BlockObjectiveEvaluation
    proposal: BlockUpdateProposal | None
    applied_damping: float
    backtracking_steps: int
    objective_improvement: float
    maximum_flow_change: float

    def __post_init__(self) -> None:
        if self.reason not in {
            "accepted",
            "no_flow_change",
            "solver_failure",
            "objective_increase",
            "nonfinite_candidate",
            "bound_violation",
        }:
            raise ValueError("invalid block-update reason.")
        if self.accepted != (self.reason == "accepted"):
            raise ValueError("accepted flag and reason are inconsistent.")
        if self.accepted != (self.proposal is not None):
            raise ValueError("only accepted decisions may contain a proposal.")
        if not math.isfinite(self.applied_damping) or self.applied_damping < 0.0:
            raise ValueError("applied_damping must be finite and non-negative.")
        if self.backtracking_steps < 0:
            raise ValueError("backtracking_steps must be non-negative.")
        if not math.isfinite(self.objective_improvement):
            raise ValueError("objective_improvement must be finite.")
        if not math.isfinite(self.maximum_flow_change) or self.maximum_flow_change < 0.0:
            raise ValueError("maximum_flow_change must be finite and non-negative.")


def solve_conditional_block(
    objective: ConditionalBlockObjective,
    initial_local_flow: object,
    *,
    config: BlockSolverConfig | None = None,
) -> BlockSolverResult:
    """Solve one smooth box-constrained conditional quadratic with L-BFGS-B."""
    config = BlockSolverConfig() if config is None else config
    size = objective.block.num_free_variables
    initial = _immutable_vector(
        initial_local_flow, name="initial_local_flow", size=size
    )
    initial_evaluation = objective.evaluate(initial)
    fixed = np.isfinite(objective.lower_bounds) & np.isfinite(
        objective.upper_bounds
    ) & (objective.lower_bounds == objective.upper_bounds)
    started = perf_counter()
    if np.all(fixed):
        candidate = np.asarray(objective.lower_bounds, dtype=float)
        success = True
        status = 0
        message = "All local variables are fixed by bounds."
        iterations = function_evaluations = gradient_evaluations = 0
    else:
        def value_and_gradient(value: Array) -> tuple[float, Array]:
            evaluation = objective.evaluate(value)
            return evaluation.objective, np.asarray(evaluation.gradient)

        optimized = minimize(
            value_and_gradient,
            np.asarray(initial, dtype=float),
            method="L-BFGS-B",
            jac=True,
            bounds=list(zip(objective.lower_bounds, objective.upper_bounds, strict=True)),
            options={
                "maxiter": config.maximum_iterations,
                "ftol": config.tolerance,
                "gtol": config.tolerance,
            },
        )
        candidate = np.asarray(optimized.x, dtype=float)
        success = bool(optimized.success)
        status = int(optimized.status)
        message = str(optimized.message)
        iterations = int(optimized.nit)
        function_evaluations = int(optimized.nfev)
        gradient_evaluations = int(optimized.njev)
    elapsed = perf_counter() - started
    candidate_evaluation = objective.evaluate(candidate)
    return BlockSolverResult(
        initial_local_flow=initial,
        candidate_local_flow=candidate,
        initial_evaluation=initial_evaluation,
        candidate_evaluation=candidate_evaluation,
        success=success,
        status=status,
        message=message,
        iterations=iterations,
        function_evaluations=function_evaluations,
        gradient_evaluations=gradient_evaluations,
        elapsed_seconds=elapsed,
    )


def _rejected_decision(
    *,
    reason: BlockUpdateReason,
    state: IncrementalLinearState,
    solver_result: BlockSolverResult,
    current_evaluation: BlockObjectiveEvaluation,
) -> BlockUpdateDecision:
    return BlockUpdateDecision(
        accepted=False,
        reason=reason,
        state=state,
        solver_result=solver_result,
        accepted_evaluation=current_evaluation,
        proposal=None,
        applied_damping=0.0,
        backtracking_steps=0,
        objective_improvement=0.0,
        maximum_flow_change=0.0,
    )


def decide_block_update(
    state: IncrementalLinearState,
    block: ODBlock,
    block_operator: BlockLinearOperatorProtocol,
    objective: ConditionalBlockObjective,
    solver_result: BlockSolverResult,
    *,
    policy: BlockUpdatePolicy | None = None,
) -> BlockUpdateDecision:
    """Apply damping/backtracking and atomically accept or reject a candidate."""
    policy = BlockUpdatePolicy() if policy is None else policy
    columns = np.asarray(block.free_column_indices, dtype=np.intp)
    current = state.free_flow[columns]
    if not np.array_equal(current, solver_result.initial_local_flow):
        raise ValueError("solver result is stale for the current block state.")
    if objective.block.fingerprint != block.fingerprint:
        raise ValueError("conditional objective belongs to a different block.")
    current_evaluation = objective.evaluate(current)
    if not np.allclose(
        current_evaluation.prediction,
        state.prediction,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("conditional objective is stale for the current prediction.")
    if policy.require_solver_success and not solver_result.success:
        return _rejected_decision(
            reason="solver_failure",
            state=state,
            solver_result=solver_result,
            current_evaluation=current_evaluation,
        )
    candidate = np.asarray(solver_result.candidate_local_flow)
    if not np.all(np.isfinite(candidate)):
        return _rejected_decision(
            reason="nonfinite_candidate",
            state=state,
            solver_result=solver_result,
            current_evaluation=current_evaluation,
        )
    if np.any(candidate < objective.lower_bounds) or np.any(
        candidate > objective.upper_bounds
    ):
        return _rejected_decision(
            reason="bound_violation",
            state=state,
            solver_result=solver_result,
            current_evaluation=current_evaluation,
        )
    direction = candidate - current
    if not np.any(direction):
        return _rejected_decision(
            reason="no_flow_change",
            state=state,
            solver_result=solver_result,
            current_evaluation=current_evaluation,
        )
    current_objective = current_evaluation.objective
    acceptance_tolerance = policy.absolute_objective_tolerance + (
        policy.relative_objective_tolerance * abs(current_objective)
    )
    damping = policy.update_damping
    for step in range(policy.maximum_backtracking_steps + 1):
        if damping < policy.minimum_damping:
            break
        trial = current + damping * direction
        evaluation = objective.evaluate(trial)
        if evaluation.objective <= current_objective + acceptance_tolerance:
            proposal = propose_incremental_update(
                state,
                block,
                block_operator,
                trial,
                lower_bounds=objective.lower_bounds,
                upper_bounds=objective.upper_bounds,
            )
            if not np.allclose(
                proposal.trial_prediction,
                evaluation.prediction,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    "conditional and incremental trial predictions are inconsistent."
                )
            new_state = apply_incremental_update(state, proposal)
            improvement = current_objective - evaluation.objective
            return BlockUpdateDecision(
                accepted=True,
                reason="accepted",
                state=new_state,
                solver_result=solver_result,
                accepted_evaluation=evaluation,
                proposal=proposal,
                applied_damping=damping,
                backtracking_steps=step,
                objective_improvement=float(improvement),
                maximum_flow_change=float(np.max(np.abs(trial - current), initial=0.0)),
            )
        damping *= policy.backtracking_factor
    return _rejected_decision(
        reason="objective_increase",
        state=state,
        solver_result=solver_result,
        current_evaluation=current_evaluation,
    )


def solve_and_decide_block_update(
    state: IncrementalLinearState,
    block: ODBlock,
    block_operator: BlockLinearOperatorProtocol,
    objective: ConditionalBlockObjective,
    *,
    solver_config: BlockSolverConfig | None = None,
    update_policy: BlockUpdatePolicy | None = None,
) -> BlockUpdateDecision:
    """Solve one conditional problem and execute its atomic update decision."""
    columns = np.asarray(block.free_column_indices, dtype=np.intp)
    result = solve_conditional_block(
        objective, state.free_flow[columns], config=solver_config
    )
    return decide_block_update(
        state,
        block,
        block_operator,
        objective,
        result,
        policy=update_policy,
    )
