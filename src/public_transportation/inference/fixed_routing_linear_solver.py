"""Solver-neutral interface and comparison tools for fixed-routing least squares."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

import numpy as np

from .fixed_routing_linear_dense_solver import (
    BoundKKTDiagnostics,
    DenseReferenceResult,
    solve_dense_reference,
)
from .fixed_routing_linear_problem import FixedRoutingLinearProblem
from .fixed_routing_linear_regularization import LinearLeastSquaresEvaluation
from .fixed_routing_linear_trf_solver import (
    TRFLSMRConfig,
    TRFLSMRResult,
    solve_trf_lsmr,
)

Array = np.ndarray
LinearSolverBackend = Literal["trf_lsmr", "dense_reference"]
REGISTERED_LINEAR_SOLVER_BACKENDS: tuple[LinearSolverBackend, ...] = (
    "trf_lsmr",
    "dense_reference",
)


def _immutable(value: object) -> Array:
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class FixedRoutingLinearSolverConfig:
    """Configuration for one registered solver backend."""

    backend: LinearSolverBackend = "trf_lsmr"
    trf_lsmr: TRFLSMRConfig = field(default_factory=TRFLSMRConfig)
    dense_tolerance: float = 1.0e-10
    dense_active_tolerance: float = 1.0e-8
    dense_max_iterations: int | None = None
    dense_max_materialized_entries: int = 10_000_000

    def __post_init__(self) -> None:
        if self.backend not in REGISTERED_LINEAR_SOLVER_BACKENDS:
            raise ValueError(f"unknown linear solver backend: {self.backend!r}.")
        if not math.isfinite(self.dense_tolerance) or self.dense_tolerance <= 0.0:
            raise ValueError("dense_tolerance must be finite and strictly positive.")
        if (
            not math.isfinite(self.dense_active_tolerance)
            or self.dense_active_tolerance < 0.0
        ):
            raise ValueError("dense_active_tolerance must be finite and non-negative.")
        if self.dense_max_iterations is not None and self.dense_max_iterations <= 0:
            raise ValueError("dense_max_iterations must be strictly positive.")
        if self.dense_max_materialized_entries <= 0:
            raise ValueError(
                "dense_max_materialized_entries must be strictly positive."
            )


@dataclass(frozen=True, slots=True)
class FixedRoutingLinearSolverResult:
    """Common result fields supplied by every registered backend."""

    backend: LinearSolverBackend
    demand: Array
    solver_variable: Array
    evaluation: LinearLeastSquaresEvaluation
    kkt: BoundKKTDiagnostics
    success: bool
    status: int
    message: str
    iterations: int
    elapsed_seconds: float
    matvec_count: int | None
    rmatvec_count: int | None
    solver_optimality: float
    numerical_rank: int | None
    singular_values: Array
    native_result: TRFLSMRResult | DenseReferenceResult

    def __post_init__(self) -> None:
        object.__setattr__(self, "demand", _immutable(self.demand))
        object.__setattr__(self, "solver_variable", _immutable(self.solver_variable))
        object.__setattr__(self, "singular_values", _immutable(self.singular_values))


@dataclass(frozen=True, slots=True)
class LinearSolverBenchmarkRecord:
    """Comparable accuracy and work metrics for one backend run."""

    backend: LinearSolverBackend
    success: bool
    objective: float
    data_residual_norm: float
    weighted_residual_norm: float
    projected_gradient_inf_norm: float
    feasibility_inf_norm: float
    iterations: int
    elapsed_seconds: float
    matvec_count: int | None
    rmatvec_count: int | None
    objective_difference_from_best: float


def solve_fixed_routing_linear(
    problem: FixedRoutingLinearProblem,
    *,
    config: FixedRoutingLinearSolverConfig | None = None,
) -> FixedRoutingLinearSolverResult:
    """Solve through one backend and normalize its result fields."""
    config = FixedRoutingLinearSolverConfig() if config is None else config
    start = perf_counter()
    if config.backend == "trf_lsmr":
        native = solve_trf_lsmr(problem, config=config.trf_lsmr)
        return FixedRoutingLinearSolverResult(
            backend=config.backend,
            demand=native.demand,
            solver_variable=native.solver_variable,
            evaluation=native.evaluation,
            kkt=native.kkt,
            success=native.success,
            status=native.status,
            message=native.message,
            iterations=native.iterations,
            elapsed_seconds=native.elapsed_seconds,
            matvec_count=native.matvec_count,
            rmatvec_count=native.rmatvec_count,
            solver_optimality=native.solver_optimality,
            numerical_rank=None,
            singular_values=np.empty(0),
            native_result=native,
        )

    native = solve_dense_reference(
        problem,
        tolerance=config.dense_tolerance,
        active_tolerance=config.dense_active_tolerance,
        max_iterations=config.dense_max_iterations,
        max_materialized_entries=config.dense_max_materialized_entries,
    )
    elapsed = perf_counter() - start
    return FixedRoutingLinearSolverResult(
        backend=config.backend,
        demand=native.demand,
        solver_variable=native.solver_variable,
        evaluation=native.evaluation,
        kkt=native.kkt,
        success=native.success,
        status=native.status,
        message=native.message,
        iterations=native.iterations,
        elapsed_seconds=float(elapsed),
        matvec_count=None,
        rmatvec_count=None,
        solver_optimality=native.kkt.projected_gradient_inf_norm,
        numerical_rank=native.numerical_rank,
        singular_values=native.singular_values,
        native_result=native,
    )


def benchmark_fixed_routing_linear_solvers(
    problem: FixedRoutingLinearProblem,
    *,
    configs: tuple[FixedRoutingLinearSolverConfig, ...] | None = None,
) -> tuple[LinearSolverBenchmarkRecord, ...]:
    """Run registered backends and return directly comparable diagnostics."""
    if configs is None:
        configs = (
            FixedRoutingLinearSolverConfig(backend="dense_reference"),
            FixedRoutingLinearSolverConfig(backend="trf_lsmr"),
        )
    if not configs:
        raise ValueError("at least one solver configuration is required.")
    backends = [config.backend for config in configs]
    if len(backends) != len(set(backends)):
        raise ValueError("benchmark solver backends must be unique.")
    solved = tuple(solve_fixed_routing_linear(problem, config=item) for item in configs)
    successful_objectives = [
        item.evaluation.objective for item in solved if item.success
    ]
    if not successful_objectives:
        raise RuntimeError("no benchmark solver backend succeeded.")
    best = min(successful_objectives)
    return tuple(
        LinearSolverBenchmarkRecord(
            backend=item.backend,
            success=item.success,
            objective=item.evaluation.objective,
            data_residual_norm=float(
                np.linalg.norm(item.evaluation.data_fit.raw_residual)
            ),
            weighted_residual_norm=float(
                np.linalg.norm(item.evaluation.data_fit.weighted_residual)
            ),
            projected_gradient_inf_norm=item.kkt.projected_gradient_inf_norm,
            feasibility_inf_norm=item.kkt.feasibility_inf_norm,
            iterations=item.iterations,
            elapsed_seconds=item.elapsed_seconds,
            matvec_count=item.matvec_count,
            rmatvec_count=item.rmatvec_count,
            objective_difference_from_best=item.evaluation.objective - best,
        )
        for item in solved
    )
