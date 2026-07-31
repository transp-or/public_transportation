"""TRF/LSMR solver for fixed-routing bound-constrained least squares."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator

from .fixed_routing_linear_dense_solver import (
    BoundKKTDiagnostics,
    evaluate_bound_kkt,
)
from .fixed_routing_linear_problem import FixedRoutingLinearProblem
from .fixed_routing_linear_regularization import (
    LinearLeastSquaresEvaluation,
    evaluate_linear_least_squares,
)
from .fixed_routing_linear_transform import (
    SolverVariableLeastSquaresSystem,
    build_solver_variable_least_squares_system,
)
from .linear_operator import DenseLinearOperator, LinearOperatorProtocol, SparseLinearOperator

Array = np.ndarray
LSMRTolerance = float | Literal["auto"] | None
SuccessPolicy = Literal["scipy", "kkt", "both"]


def _immutable(value: object) -> Array:
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class TRFLSMRConfig:
    tolerance: float = 1.0e-8
    lsmr_tolerance: LSMRTolerance = None
    max_iterations: int | None = None
    lsmr_max_iterations: int | None = None
    active_tolerance: float = 1.0e-7
    verbose: int = 0
    diagonal_preconditioner: bool = False
    preconditioner_floor: float = 1.0e-12
    success_policy: SuccessPolicy = "scipy"
    kkt_tolerance: float = 1.0e-6

    def __post_init__(self) -> None:
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and strictly positive.")
        if self.lsmr_tolerance not in (None, "auto"):
            if not isinstance(self.lsmr_tolerance, (int, float)) or not math.isfinite(
                self.lsmr_tolerance
            ) or self.lsmr_tolerance <= 0.0:
                raise ValueError(
                    "lsmr_tolerance must be None, 'auto', or a finite positive number."
                )
        if self.max_iterations is not None and self.max_iterations <= 0:
            raise ValueError("max_iterations must be strictly positive when provided.")
        if self.lsmr_max_iterations is not None and self.lsmr_max_iterations <= 0:
            raise ValueError(
                "lsmr_max_iterations must be strictly positive when provided."
            )
        if not math.isfinite(self.active_tolerance) or self.active_tolerance < 0.0:
            raise ValueError("active_tolerance must be finite and non-negative.")
        if self.verbose not in (0, 1, 2):
            raise ValueError("verbose must be 0, 1, or 2.")
        if not math.isfinite(self.preconditioner_floor) or self.preconditioner_floor <= 0:
            raise ValueError("preconditioner_floor must be finite and positive.")
        if self.success_policy not in {"scipy", "kkt", "both"}:
            raise ValueError("success_policy must be 'scipy', 'kkt', or 'both'.")
        if not math.isfinite(self.kkt_tolerance) or self.kkt_tolerance <= 0:
            raise ValueError("kkt_tolerance must be finite and strictly positive.")


@dataclass(frozen=True, slots=True)
class TRFLSMRResult:
    demand: Array
    solver_variable: Array
    evaluation: LinearLeastSquaresEvaluation
    kkt: BoundKKTDiagnostics
    success: bool
    status: int
    message: str
    iterations: int
    solver_cost: float
    solver_optimality: float
    matvec_count: int
    rmatvec_count: int
    elapsed_seconds: float
    preconditioner_seconds: float = 0.0
    preparation_matvec_count: int = 0
    final_matvec_count: int = 0
    final_rmatvec_count: int = 0
    stopping_condition: str = "unknown"


@dataclass(frozen=True, slots=True)
class _RestrictedOperator:
    base: LinearOperatorProtocol
    free_indices: Array

    @property
    def shape(self) -> tuple[int, int]:
        return self.base.shape[0], int(self.free_indices.size)

    @property
    def dtype(self) -> np.dtype:
        return self.base.dtype

    def matvec(self, vector: object) -> Array:
        value = np.asarray(vector)
        if value.shape != (self.shape[1],):
            raise ValueError(
                f"restricted vector must have shape ({self.shape[1]},), "
                f"got {value.shape}."
            )
        full = np.zeros(self.base.shape[1], dtype=np.result_type(value, self.dtype))
        full[self.free_indices] = value
        return self.base.matvec(full)

    def rmatvec(self, vector: object) -> Array:
        return np.asarray(self.base.rmatvec(vector))[self.free_indices]


@dataclass(slots=True)
class _ProductCounter:
    operator: LinearOperatorProtocol
    matvec_count: int = 0
    rmatvec_count: int = 0

    def matvec(self, vector: object) -> Array:
        self.matvec_count += 1
        return self.operator.matvec(vector)

    def rmatvec(self, vector: object) -> Array:
        self.rmatvec_count += 1
        return self.operator.rmatvec(vector)


def _column_squared_norms(operator: LinearOperatorProtocol) -> Array:
    """Return exact column squared norms without forming a normal matrix."""
    if isinstance(operator, SparseLinearOperator):
        return np.asarray(operator.matrix.power(2).sum(axis=0)).reshape(-1)
    if isinstance(operator, DenseLinearOperator):
        return np.einsum("ij,ij->j", operator.matrix, operator.matrix)
    raise TypeError(
        "diagonal preconditioning requires explicit dense or sparse measurement "
        "and regularization operators"
    )


def _diagonal_preconditioner_scales(
    problem: FixedRoutingLinearProblem, *, floor: float
) -> Array:
    norms = np.asarray(
        problem.measurement_operator.matrix.power(2).T
        @ problem.observation_weights
        if isinstance(problem.measurement_operator, SparseLinearOperator)
        else np.einsum(
            "ij,i,ij->j",
            problem.measurement_operator.matrix,
            problem.observation_weights,
            problem.measurement_operator.matrix,
        )
    ).reshape(-1)
    for block in problem.regularization_blocks:
        norms += block.strength * _column_squared_norms(block.operator)
    base = np.asarray(problem.variable_scales, dtype=float)
    scaled_norms = norms * np.square(base)
    adjustment = 1.0 / np.sqrt(np.maximum(scaled_norms, floor))
    # A structurally zero column must not receive an extreme artificial scale.
    adjustment[scaled_norms < floor] = 1.0
    return base * adjustment


def _reduced_solver_system(
    system: SolverVariableLeastSquaresSystem,
) -> tuple[_RestrictedOperator, Array, Array, Array, Array, Array]:
    lower = np.asarray(system.lower_bounds, dtype=float)
    upper = np.asarray(system.upper_bounds, dtype=float)
    fixed = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
    free_indices = np.flatnonzero(~fixed)
    fixed_indices = np.flatnonzero(fixed)
    fixed_values = np.zeros(lower.size, dtype=float)
    fixed_values[fixed_indices] = lower[fixed_indices]
    target = np.asarray(system.target, dtype=float)
    if fixed_indices.size:
        target = target - system.operator.matvec(fixed_values)
    return (
        _RestrictedOperator(system.operator, free_indices),
        target,
        lower[free_indices],
        upper[free_indices],
        fixed_indices,
        fixed_values,
    )


def solve_trf_lsmr(
    problem: FixedRoutingLinearProblem,
    *,
    config: TRFLSMRConfig | None = None,
) -> TRFLSMRResult:
    """Solve through forward/transpose products and return physical diagnostics."""
    config = TRFLSMRConfig() if config is None else config
    preconditioner_start = perf_counter()
    scales = (
        _diagonal_preconditioner_scales(
            problem, floor=config.preconditioner_floor
        )
        if config.diagonal_preconditioner
        else None
    )
    preconditioner_seconds = perf_counter() - preconditioner_start
    system = build_solver_variable_least_squares_system(
        problem, variable_scales=scales
    )
    (
        restricted,
        target,
        lower,
        upper,
        fixed_indices,
        fixed_values,
    ) = _reduced_solver_system(system)
    counter = _ProductCounter(restricted)
    start = perf_counter()

    if restricted.shape[1] == 0:
        solver_variable = fixed_values
        success, status, message, iterations = (
            True,
            3,
            "All variables fixed by bounds.",
            0,
        )
        residual = system.operator.matvec(solver_variable) - system.target
        solver_cost = float(0.5 * np.vdot(residual, residual))
        solver_optimality = 0.0
    else:
        scipy_operator = ScipyLinearOperator(
            shape=restricted.shape,
            matvec=counter.matvec,
            rmatvec=counter.rmatvec,
            dtype=restricted.dtype,
        )
        optimized = lsq_linear(
            scipy_operator,
            target,
            bounds=(lower, upper),
            method="trf",
            lsq_solver="lsmr",
            tol=config.tolerance,
            lsmr_tol=config.lsmr_tolerance,
            max_iter=config.max_iterations,
            verbose=config.verbose,
            lsmr_maxiter=config.lsmr_max_iterations,
        )
        solver_variable = fixed_values
        solver_variable[np.asarray(restricted.free_indices)] = optimized.x
        success = bool(optimized.success)
        status = int(optimized.status)
        message = str(optimized.message)
        iterations = int(optimized.nit)
        solver_cost = float(optimized.cost)
        solver_optimality = float(optimized.optimality)

    elapsed = perf_counter() - start
    demand = system.transform.demand_from_solver_variable(solver_variable)
    evaluation = evaluate_linear_least_squares(problem, demand)
    kkt = evaluate_bound_kkt(
        demand=demand,
        gradient=evaluation.gradient,
        lower_bounds=problem.lower_bounds,
        upper_bounds=problem.upper_bounds,
        active_tolerance=config.active_tolerance,
    )
    kkt_success = (
        kkt.feasibility_inf_norm <= config.kkt_tolerance
        and kkt.projected_gradient_inf_norm <= config.kkt_tolerance
    )
    scipy_success = success
    if config.success_policy == "kkt":
        success = kkt_success
    elif config.success_policy == "both":
        success = scipy_success and kkt_success
    stopping_condition = (
        f"scipy_status_{status}; policy={config.success_policy}; "
        f"kkt_satisfied={kkt_success}"
    )
    if not success and scipy_success and config.success_policy != "scipy":
        message = f"{message} Rejected by {config.success_policy} success policy."
    return TRFLSMRResult(
        demand=_immutable(demand),
        solver_variable=_immutable(solver_variable),
        evaluation=evaluation,
        kkt=kkt,
        success=success,
        status=status,
        message=message,
        iterations=iterations,
        solver_cost=solver_cost,
        solver_optimality=solver_optimality,
        matvec_count=counter.matvec_count,
        rmatvec_count=counter.rmatvec_count,
        elapsed_seconds=float(elapsed),
        preconditioner_seconds=float(preconditioner_seconds),
        preparation_matvec_count=1,
        final_matvec_count=1,
        final_rmatvec_count=1,
        stopping_condition=stopping_condition,
    )
