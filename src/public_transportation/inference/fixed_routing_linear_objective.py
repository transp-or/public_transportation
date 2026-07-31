"""Data-fit calculations for fixed-routing linear least squares."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fixed_routing_linear_problem import FixedRoutingLinearProblem

Array = np.ndarray


def _immutable_vector(value: object) -> Array:
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class LinearDataFitEvaluation:
    """Complete unregularized least-squares evaluation at one OD vector."""

    prediction: Array
    raw_residual: Array
    weighted_residual: Array
    objective: float
    gradient: Array


def predict_linear_measurements(
    problem: FixedRoutingLinearProblem, demand: object
) -> Array:
    """Return ``A x + c`` in canonical measurement order."""
    prediction = (
        problem.measurement_operator.matvec(demand)
        + problem.fixed_measurement_offset
    )
    return np.asarray(prediction)


def raw_linear_residual(
    problem: FixedRoutingLinearProblem, demand: object
) -> Array:
    """Return the unweighted residual ``A x + c - y``."""
    return predict_linear_measurements(problem, demand) - problem.observations


def weighted_linear_residual(
    problem: FixedRoutingLinearProblem, demand: object
) -> Array:
    """Return ``sqrt(W) (A x + c - y)``."""
    return np.sqrt(problem.observation_weights) * raw_linear_residual(
        problem, demand
    )


def linear_data_objective(
    problem: FixedRoutingLinearProblem, demand: object
) -> float:
    """Return ``0.5 * ||sqrt(W) (A x + c - y)||**2``."""
    residual = weighted_linear_residual(problem, demand)
    return float(0.5 * np.vdot(residual, residual))


def linear_data_gradient(
    problem: FixedRoutingLinearProblem, demand: object
) -> Array:
    """Return ``A.T W (A x + c - y)`` in canonical free-OD order."""
    residual = raw_linear_residual(problem, demand)
    return problem.measurement_operator.rmatvec(
        problem.observation_weights * residual
    )


def evaluate_linear_data_fit(
    problem: FixedRoutingLinearProblem, demand: object
) -> LinearDataFitEvaluation:
    """Evaluate prediction, both residuals, objective, and analytic gradient."""
    prediction = predict_linear_measurements(problem, demand)
    raw_residual = prediction - problem.observations
    weighted_residual = np.sqrt(problem.observation_weights) * raw_residual
    objective = float(0.5 * np.vdot(weighted_residual, weighted_residual))
    gradient = problem.measurement_operator.rmatvec(
        problem.observation_weights * raw_residual
    )
    return LinearDataFitEvaluation(
        prediction=_immutable_vector(prediction),
        raw_residual=_immutable_vector(raw_residual),
        weighted_residual=_immutable_vector(weighted_residual),
        objective=objective,
        gradient=_immutable_vector(gradient),
    )
