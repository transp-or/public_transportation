"""Controlled correctness checks for small fixed-routing linear models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping

import numpy as np

from .fixed_routing_linear_dense_solver import DenseReferenceResult, solve_dense_reference
from .fixed_routing_linear_objective import predict_linear_measurements
from .fixed_routing_linear_problem import FixedRoutingLinearProblem
from .fixed_routing_linear_quality import LinearEstimateQuality, analyze_linear_estimate_quality
from .linear_operator import materialize_linear_operator

Array = np.ndarray


def _immutable(value: object) -> Array:
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


def _demand_vector(problem: FixedRoutingLinearProblem, value: object, *, name: str) -> Array:
    demand = np.asarray(value, dtype=np.float64)
    if demand.shape != (problem.num_free_od,):
        raise ValueError(
            f"{name} must have shape ({problem.num_free_od},), got {demand.shape}."
        )
    if not np.all(np.isfinite(demand)):
        raise ValueError(f"{name} must be finite.")
    if np.any(demand < problem.lower_bounds) or np.any(demand > problem.upper_bounds):
        raise ValueError(f"{name} must satisfy the demand bounds.")
    return demand


@dataclass(frozen=True, slots=True)
class ForwardEquivalenceCase:
    """Difference between the explicit linear map and assignment prediction."""

    name: str
    demand: Array
    linear_prediction: Array
    assignment_prediction: Array
    difference: Array
    max_abs_difference: float
    rms_difference: float


@dataclass(frozen=True, slots=True)
class ForwardEquivalenceValidation:
    """Forward-equivalence results for a collection of feasible OD vectors."""

    cases: tuple[ForwardEquivalenceCase, ...]
    absolute_tolerance: float
    relative_tolerance: float
    passed: bool
    worst_max_abs_difference: float


@dataclass(frozen=True, slots=True)
class NoiseFreeRecoveryValidation:
    """Noise-free recovery and observable/null-space error decomposition."""

    problem: FixedRoutingLinearProblem
    true_demand: Array
    synthetic_observations: Array
    result: DenseReferenceResult
    quality: LinearEstimateQuality
    estimation_error: Array
    identifiable_error: Array
    null_space_error: Array
    measurement_rank: int
    measurement_nullity: int
    rank_tolerance: float
    measurement_residual_inf_norm: float
    estimation_error_norm: float
    identifiable_error_norm: float
    null_space_error_norm: float


def validate_fixed_routing_forward_equivalence(
    problem: FixedRoutingLinearProblem,
    demand_cases: Mapping[str, object],
    assignment_predictor: Callable[[Array], object],
    *,
    absolute_tolerance: float = 5.0e-5,
    relative_tolerance: float = 5.0e-5,
) -> ForwardEquivalenceValidation:
    """Compare ``A x + c`` with an independent fixed-routing assignment path."""
    if not demand_cases:
        raise ValueError("at least one demand case is required.")
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("equivalence tolerances must be non-negative.")

    results: list[ForwardEquivalenceCase] = []
    passed = True
    for name, value in demand_cases.items():
        if not str(name).strip():
            raise ValueError("demand case names must be nonempty.")
        demand = _demand_vector(problem, value, name=f"demand case {name!r}")
        linear = np.asarray(predict_linear_measurements(problem, demand), dtype=np.float64)
        assignment = np.asarray(assignment_predictor(demand), dtype=np.float64)
        if assignment.shape != linear.shape:
            raise ValueError(
                f"assignment prediction for {name!r} must have shape {linear.shape}, "
                f"got {assignment.shape}."
            )
        if not np.all(np.isfinite(assignment)):
            raise ValueError(f"assignment prediction for {name!r} must be finite.")
        difference = linear - assignment
        maximum = float(np.max(np.abs(difference), initial=0.0))
        rms = float(np.sqrt(np.mean(np.square(difference))))
        passed = passed and bool(
            np.allclose(linear, assignment, rtol=relative_tolerance, atol=absolute_tolerance)
        )
        results.append(
            ForwardEquivalenceCase(
                name=str(name),
                demand=_immutable(demand),
                linear_prediction=_immutable(linear),
                assignment_prediction=_immutable(assignment),
                difference=_immutable(difference),
                max_abs_difference=maximum,
                rms_difference=rms,
            )
        )
    return ForwardEquivalenceValidation(
        cases=tuple(results),
        absolute_tolerance=float(absolute_tolerance),
        relative_tolerance=float(relative_tolerance),
        passed=passed,
        worst_max_abs_difference=max(case.max_abs_difference for case in results),
    )


def validate_noise_free_linear_recovery(
    problem: FixedRoutingLinearProblem,
    true_demand: object,
    *,
    solver_tolerance: float = 1.0e-12,
    max_materialized_entries: int = 10_000_000,
) -> NoiseFreeRecoveryValidation:
    """Solve observations generated by the operator and decompose demand error.

    Regularization is deliberately removed. The row-space component of the
    estimation error is observable through the weighted measurement operator;
    the remaining component lies in its numerical null space.
    """
    truth = _demand_vector(problem, true_demand, name="true_demand")
    observations = predict_linear_measurements(problem, truth)
    synthetic_problem = replace(
        problem,
        observations=observations,
        regularization_selection="none",
        regularization_blocks=(),
    )
    result = solve_dense_reference(
        synthetic_problem,
        tolerance=solver_tolerance,
        max_materialized_entries=max_materialized_entries,
    )
    quality = analyze_linear_estimate_quality(
        synthetic_problem,
        result.demand,
        active_tolerance=max(1.0e-9, 10.0 * solver_tolerance),
        max_materialized_entries=max_materialized_entries,
        exclude_active_bounds=False,
    )

    matrix = materialize_linear_operator(
        synthetic_problem.measurement_operator,
        max_entries=max_materialized_entries,
    ).astype(np.float64, copy=False)
    weighted = np.sqrt(synthetic_problem.observation_weights)[:, None] * matrix
    _, singular_values, right_vectors = np.linalg.svd(weighted, full_matrices=True)
    rank_tolerance = (
        0.0
        if singular_values.size == 0
        else max(weighted.shape) * np.finfo(np.float64).eps * singular_values[0]
    )
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    row_basis = right_vectors[:rank, :]
    error = np.asarray(result.demand) - truth
    identifiable_error = row_basis.T @ (row_basis @ error)
    null_error = error - identifiable_error
    residual = np.asarray(result.evaluation.data_fit.raw_residual)
    return NoiseFreeRecoveryValidation(
        problem=synthetic_problem,
        true_demand=_immutable(truth),
        synthetic_observations=_immutable(observations),
        result=result,
        quality=quality,
        estimation_error=_immutable(error),
        identifiable_error=_immutable(identifiable_error),
        null_space_error=_immutable(null_error),
        measurement_rank=rank,
        measurement_nullity=problem.num_free_od - rank,
        rank_tolerance=float(rank_tolerance),
        measurement_residual_inf_norm=float(np.max(np.abs(residual), initial=0.0)),
        estimation_error_norm=float(np.linalg.norm(error)),
        identifiable_error_norm=float(np.linalg.norm(identifiable_error)),
        null_space_error_norm=float(np.linalg.norm(null_error)),
    )
