"""Dense reference solver and independent KKT checks for small problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import lsq_linear

from .fixed_routing_linear_problem import FixedRoutingLinearProblem
from .fixed_routing_linear_regularization import (
    LinearLeastSquaresEvaluation,
    build_augmented_linear_least_squares_system,
    evaluate_linear_least_squares,
)
from .fixed_routing_linear_transform import PhysicalDemandTransform
from .linear_operator import materialize_linear_operator

Array = np.ndarray
DenseReferenceMethod = Literal["svd_lstsq", "bvls", "fixed_bounds"]


def _immutable(value: object) -> Array:
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class BoundKKTDiagnostics:
    """Independent first-order and feasibility diagnostics in physical units."""

    lower_active: Array
    upper_active: Array
    fixed_by_bounds: Array
    lower_multipliers: Array
    upper_multipliers: Array
    projected_gradient: Array
    projected_gradient_inf_norm: float
    feasibility_inf_norm: float


@dataclass(frozen=True, slots=True)
class DenseReferenceResult:
    """Reference solution and diagnostics for a small explicit problem."""

    demand: Array
    solver_variable: Array
    evaluation: LinearLeastSquaresEvaluation
    kkt: BoundKKTDiagnostics
    method: DenseReferenceMethod
    success: bool
    status: int
    message: str
    iterations: int
    numerical_rank: int
    singular_values: Array


def evaluate_bound_kkt(
    *,
    demand: object,
    gradient: object,
    lower_bounds: object,
    upper_bounds: object,
    active_tolerance: float = 1.0e-8,
) -> BoundKKTDiagnostics:
    """Evaluate box feasibility and projected-gradient KKT conditions."""
    if not np.isfinite(active_tolerance) or active_tolerance < 0.0:
        raise ValueError("active_tolerance must be finite and non-negative.")
    demand = np.asarray(demand, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    if demand.ndim != 1:
        raise ValueError("demand must be one-dimensional.")
    expected = demand.shape
    for name, value in (
        ("gradient", gradient),
        ("lower_bounds", lower),
        ("upper_bounds", upper),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}.")
    if not np.all(np.isfinite(demand)) or not np.all(np.isfinite(gradient)):
        raise ValueError("demand and gradient must be finite.")

    fixed = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
    lower_active = np.isfinite(lower) & (demand <= lower + active_tolerance)
    upper_active = np.isfinite(upper) & (demand >= upper - active_tolerance)
    projected = np.array(gradient, copy=True)
    projected[fixed] = 0.0
    projected[lower_active & (gradient >= 0.0)] = 0.0
    projected[upper_active & (gradient <= 0.0)] = 0.0

    lower_multipliers = np.zeros_like(gradient)
    upper_multipliers = np.zeros_like(gradient)
    lower_multipliers[lower_active & ~fixed] = np.maximum(
        gradient[lower_active & ~fixed], 0.0
    )
    upper_multipliers[upper_active & ~fixed] = np.maximum(
        -gradient[upper_active & ~fixed], 0.0
    )
    # At an equality bound, use a minimal one-sided decomposition of the
    # stationarity equation g - lambda_lower + lambda_upper = 0.
    lower_multipliers[fixed] = np.maximum(gradient[fixed], 0.0)
    upper_multipliers[fixed] = np.maximum(-gradient[fixed], 0.0)

    lower_violation = np.maximum(lower - demand, 0.0)
    upper_violation = np.maximum(demand - upper, 0.0)
    feasibility = float(
        max(
            np.max(lower_violation, initial=0.0),
            np.max(upper_violation, initial=0.0),
        )
    )
    return BoundKKTDiagnostics(
        lower_active=_immutable(lower_active),
        upper_active=_immutable(upper_active),
        fixed_by_bounds=_immutable(fixed),
        lower_multipliers=_immutable(lower_multipliers),
        upper_multipliers=_immutable(upper_multipliers),
        projected_gradient=_immutable(projected),
        projected_gradient_inf_norm=float(np.max(np.abs(projected), initial=0.0)),
        feasibility_inf_norm=feasibility,
    )


def solve_dense_reference(
    problem: FixedRoutingLinearProblem,
    *,
    tolerance: float = 1.0e-10,
    active_tolerance: float = 1.0e-8,
    max_iterations: int | None = None,
    max_materialized_entries: int = 10_000_000,
) -> DenseReferenceResult:
    """Solve a small problem by SVD least squares or bounded-variable LS."""
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and strictly positive.")
    if max_iterations is not None and max_iterations <= 0:
        raise ValueError("max_iterations must be strictly positive when provided.")

    system = build_augmented_linear_least_squares_system(problem)
    matrix = materialize_linear_operator(
        system.operator, max_entries=max_materialized_entries
    ).astype(np.float64, copy=False)
    target = np.asarray(system.target, dtype=np.float64)
    lower = np.asarray(problem.lower_bounds, dtype=np.float64)
    upper = np.asarray(problem.upper_bounds, dtype=np.float64)
    fixed = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
    free = ~fixed
    demand = np.empty(problem.num_free_od, dtype=np.float64)
    demand[fixed] = lower[fixed]
    reduced_target = target - matrix[:, fixed] @ demand[fixed]
    reduced_matrix = matrix[:, free]

    if not np.any(free):
        method: DenseReferenceMethod = "fixed_bounds"
        success, status, message, iterations = True, 3, "All variables fixed by bounds.", 0
    elif not np.any(np.isfinite(lower[free])) and not np.any(
        np.isfinite(upper[free])
    ):
        method = "svd_lstsq"
        solution, _, _, _ = np.linalg.lstsq(
            reduced_matrix, reduced_target, rcond=None
        )
        demand[free] = solution
        success, status, message, iterations = (
            True,
            3,
            "Unconstrained SVD least-squares solution.",
            1,
        )
    else:
        method = "bvls"
        optimized = lsq_linear(
            reduced_matrix,
            reduced_target,
            bounds=(lower[free], upper[free]),
            method="bvls",
            tol=tolerance,
            max_iter=max_iterations,
        )
        demand[free] = optimized.x
        success = bool(optimized.success)
        status = int(optimized.status)
        message = str(optimized.message)
        iterations = int(optimized.nit)

    evaluation = evaluate_linear_least_squares(problem, demand)
    kkt = evaluate_bound_kkt(
        demand=demand,
        gradient=evaluation.gradient,
        lower_bounds=lower,
        upper_bounds=upper,
        active_tolerance=active_tolerance,
    )
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank_tolerance = (
        0.0
        if singular_values.size == 0
        else np.max(matrix.shape) * np.finfo(matrix.dtype).eps * singular_values[0]
    )
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    transform = PhysicalDemandTransform.from_problem(problem)
    return DenseReferenceResult(
        demand=_immutable(demand),
        solver_variable=_immutable(transform.solver_variable_from_demand(demand)),
        evaluation=evaluation,
        kkt=kkt,
        method=method,
        success=success,
        status=status,
        message=message,
        iterations=iterations,
        numerical_rank=rank,
        singular_values=_immutable(singular_values),
    )
