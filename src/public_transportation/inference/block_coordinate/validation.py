"""Optional global validation separated from bounded estimator execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
)
from public_transportation.inference.linear_operator import LinearOperatorProtocol

from .objective import prepare_separable_quadratic_prior, projected_gradient
from .results import BlockCoordinateState

ValidationStatus = Literal["computed", "deferred", "failed", "unavailable"]


@dataclass(frozen=True, slots=True)
class BlockCoordinateValidationResult:
    prediction_status: ValidationStatus
    objective_status: ValidationStatus
    gradient_status: ValidationStatus
    bounds_status: ValidationStatus
    checkpoint_status: ValidationStatus
    prediction_maximum_absolute_error: float | None = None
    objective_absolute_error: float | None = None
    exact_projected_gradient_norm: float | None = None
    messages: tuple[str, ...] = ()


def validate_block_coordinate_result(
    *,
    problem: FixedRoutingLinearProblem,
    state: BlockCoordinateState,
    exact_prediction: bool = True,
    exact_objective: bool = True,
    exact_gradient: bool = False,
) -> BlockCoordinateValidationResult:
    """Run explicitly requested complete-operator checks, possibly in a fresh process."""
    messages: list[str] = []
    bounds_ok = bool(
        np.all(state.current_free_flow >= problem.lower_bounds)
        and np.all(state.current_free_flow <= problem.upper_bounds)
    )
    prediction_status: ValidationStatus = "deferred"
    objective_status: ValidationStatus = "deferred"
    gradient_status: ValidationStatus = "deferred"
    prediction_error = None
    objective_error = None
    gradient_norm = None
    recomputed_prediction: np.ndarray | None = None
    operator = cast(LinearOperatorProtocol, problem.measurement_operator)
    if exact_prediction or exact_objective or exact_gradient:
        recomputed_prediction = (
            np.asarray(operator.matvec(state.current_free_flow))
            + problem.fixed_measurement_offset
        )
    if exact_prediction:
        assert recomputed_prediction is not None
        prediction_error = float(
            np.max(np.abs(recomputed_prediction - state.current_prediction))
        )
        prediction_status = "computed" if np.allclose(
            recomputed_prediction, state.current_prediction, rtol=1.0e-10, atol=1.0e-10
        ) else "failed"
        if prediction_status == "failed":
            messages.append("stored prediction differs from exact recomputation")
    prior = prepare_separable_quadratic_prior(problem)
    if exact_objective:
        assert recomputed_prediction is not None
        residual = recomputed_prediction - problem.observations
        objective = float(
            0.5 * np.dot(problem.observation_weights, residual * residual)
            + prior.objective(state.current_free_flow)
        )
        objective_error = abs(objective - state.current_objective)
        objective_status = "computed" if np.isclose(
            objective, state.current_objective, rtol=1.0e-10, atol=1.0e-10
        ) else "failed"
        if objective_status == "failed":
            messages.append("stored objective differs from exact recomputation")
    if exact_gradient:
        assert recomputed_prediction is not None
        residual = recomputed_prediction - problem.observations
        gradient = operator.rmatvec(
            problem.observation_weights * residual
        ) + prior.gradient(state.current_free_flow)
        gradient_norm = float(
            np.linalg.norm(
                projected_gradient(
                    state.current_free_flow,
                    gradient,
                    problem.lower_bounds,
                    problem.upper_bounds,
                )
            )
        )
        gradient_status = "computed"
    return BlockCoordinateValidationResult(
        prediction_status=prediction_status,
        objective_status=objective_status,
        gradient_status=gradient_status,
        bounds_status="computed" if bounds_ok else "failed",
        checkpoint_status="computed",
        prediction_maximum_absolute_error=prediction_error,
        objective_absolute_error=objective_error,
        exact_projected_gradient_norm=gradient_norm,
        messages=tuple(messages),
    )
