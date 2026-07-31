"""High-level workflow for fixed-routing linear OD estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np

from .fixed_routing_linear_dense_solver import (
    DenseReferenceResult,
)
from .fixed_routing_linear_problem import FixedRoutingLinearProblem
from .fixed_routing_linear_quality import (
    LinearEstimateQuality,
    analyze_linear_estimate_quality,
)
from .fixed_routing_linear_recommendation import (
    RegularizationRecommendation,
    recommend_linear_regularization,
)
from .fixed_routing_linear_regularization import (
    ridge_to_prior,
    scaled_ridge_to_prior,
)
from .fixed_routing_linear_results import (
    FixedRoutingLinearResult,
    build_fixed_routing_linear_result,
    save_fixed_routing_linear_result,
)
from .fixed_routing_linear_solver import (
    FixedRoutingLinearSolverConfig,
    FixedRoutingLinearSolverResult,
    solve_fixed_routing_linear,
)
from .fixed_routing_linear_scalable_quality import (
    ScalableLinearEstimateQuality,
    ScalableQualityConfig,
    analyze_linear_estimate_quality_scalable,
)
from .fixed_routing_linear_trf_solver import (
    TRFLSMRConfig,
    TRFLSMRResult,
)

RegularizationChoice = Literal[
    "none",
    "ridge_to_prior",
    "scaled_ridge_to_prior",
]


@dataclass(frozen=True, slots=True)
class FixedRoutingLinearEstimationConfig:
    """Explicit modeling, solver, and verification choices for one fit."""

    regularization: RegularizationChoice
    regularization_strength: float | None = None
    trf_lsmr: TRFLSMRConfig = field(default_factory=TRFLSMRConfig)
    dense_reference_tolerance: float = 1.0e-10
    quality_active_tolerance: float = 1.0e-7
    verification_relative_tolerance: float = 3.0e-5
    verification_absolute_tolerance: float = 3.0e-5
    verify_prediction: bool = True

    def __post_init__(self) -> None:
        if self.regularization not in {
            "none",
            "ridge_to_prior",
            "scaled_ridge_to_prior",
        }:
            raise ValueError(f"unknown regularization choice: {self.regularization!r}.")
        if self.regularization == "none":
            if self.regularization_strength is not None:
                raise ValueError(
                    "regularization_strength must be omitted when regularization='none'."
                )
        elif self.regularization_strength is None:
            raise ValueError(
                f"regularization_strength is required for {self.regularization!r}."
            )
        elif not math.isfinite(self.regularization_strength) or (
            self.regularization_strength < 0.0
        ):
            raise ValueError("regularization_strength must be finite and non-negative.")
        for name in (
            "dense_reference_tolerance",
            "quality_active_tolerance",
            "verification_relative_tolerance",
            "verification_absolute_tolerance",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.dense_reference_tolerance == 0.0:
            raise ValueError("dense_reference_tolerance must be strictly positive.")


@dataclass(frozen=True, slots=True)
class FixedRoutingLinearEstimationRun:
    """Complete in-memory and optional persisted output of one workflow run."""

    problem: FixedRoutingLinearProblem
    recommendation: RegularizationRecommendation
    dense_reference: DenseReferenceResult
    iterative_result: TRFLSMRResult
    quality: LinearEstimateQuality
    result: FixedRoutingLinearResult
    output_path: Path | None


@dataclass(frozen=True, slots=True)
class ScalableFixedRoutingLinearEstimationRun:
    """Operator-only fit and approximate diagnostics without materialization."""

    problem: FixedRoutingLinearProblem
    solver_result: FixedRoutingLinearSolverResult
    quality: ScalableLinearEstimateQuality


def configure_fixed_routing_linear_regularization(
    problem: FixedRoutingLinearProblem,
    config: FixedRoutingLinearEstimationConfig,
) -> FixedRoutingLinearProblem:
    """Apply one explicit regularization choice to an unconfigured base problem."""
    if problem.regularization_selection != "unspecified":
        raise ValueError(
            "the workflow requires a base problem with regularization_selection="
            "'unspecified'; configure regularization through its workflow config."
        )
    if config.regularization == "none":
        return replace(
            problem,
            regularization_selection="none",
            regularization_blocks=(),
        )
    strength = config.regularization_strength
    assert strength is not None
    if config.regularization == "ridge_to_prior":
        block = ridge_to_prior(problem.prior_demand, strength=strength)
    else:
        block = scaled_ridge_to_prior(
            problem.prior_demand,
            problem.variable_scales,
            strength=strength,
        )
    return replace(
        problem,
        regularization_selection="configured",
        regularization_blocks=(block,),
    )


def run_fixed_routing_linear_estimation(
    problem: FixedRoutingLinearProblem,
    *,
    config: FixedRoutingLinearEstimationConfig,
    output_path: str | Path | None = None,
) -> FixedRoutingLinearEstimationRun:
    """Configure, solve, verify, diagnose, and optionally persist one fit."""
    recommendation = recommend_linear_regularization(problem)
    configured = configure_fixed_routing_linear_regularization(problem, config)
    dense_solve = solve_fixed_routing_linear(
        configured,
        config=FixedRoutingLinearSolverConfig(
            backend="dense_reference",
            dense_tolerance=config.dense_reference_tolerance,
        ),
    )
    iterative_solve = solve_fixed_routing_linear(
        configured,
        config=FixedRoutingLinearSolverConfig(
            backend="trf_lsmr",
            trf_lsmr=config.trf_lsmr,
        ),
    )
    if not isinstance(dense_solve.native_result, DenseReferenceResult):
        raise TypeError("dense backend returned an incompatible native result.")
    if not isinstance(iterative_solve.native_result, TRFLSMRResult):
        raise TypeError("TRF/LSMR backend returned an incompatible native result.")
    dense = dense_solve.native_result
    iterative = iterative_solve.native_result
    if not dense.success:
        raise RuntimeError(f"dense reference solver failed: {dense.message}")
    if not iterative.success:
        raise RuntimeError(f"TRF/LSMR failed: {iterative.message}")
    if not np.isclose(
        iterative.evaluation.objective,
        dense.evaluation.objective,
        rtol=config.verification_relative_tolerance,
        atol=config.verification_absolute_tolerance,
    ):
        raise RuntimeError("dense-reference and TRF/LSMR objective values disagree.")
    if config.verify_prediction and not np.allclose(
        iterative.evaluation.data_fit.prediction,
        dense.evaluation.data_fit.prediction,
        rtol=config.verification_relative_tolerance,
        atol=config.verification_absolute_tolerance,
    ):
        raise RuntimeError("dense-reference and TRF/LSMR predictions disagree.")
    quality = analyze_linear_estimate_quality(
        configured,
        iterative.demand,
        active_tolerance=config.quality_active_tolerance,
    )
    result = build_fixed_routing_linear_result(configured, iterative, quality)
    destination = None if output_path is None else Path(output_path)
    if destination is not None:
        save_fixed_routing_linear_result(result, destination)
    return FixedRoutingLinearEstimationRun(
        problem=configured,
        recommendation=recommendation,
        dense_reference=dense,
        iterative_result=iterative,
        quality=quality,
        result=result,
        output_path=destination,
    )


def run_fixed_routing_linear_estimation_scalable(
    problem: FixedRoutingLinearProblem,
    *,
    config: FixedRoutingLinearEstimationConfig,
    quality_config: ScalableQualityConfig | None = None,
) -> ScalableFixedRoutingLinearEstimationRun:
    """Configure, solve, and diagnose using products only, without persistence."""
    configured = configure_fixed_routing_linear_regularization(problem, config)
    solved = solve_fixed_routing_linear(
        configured,
        config=FixedRoutingLinearSolverConfig(
            backend="trf_lsmr",
            trf_lsmr=config.trf_lsmr,
        ),
    )
    if not solved.success:
        raise RuntimeError(f"TRF/LSMR failed: {solved.message}")
    quality = analyze_linear_estimate_quality_scalable(
        configured,
        solved.demand,
        config=quality_config,
    )
    return ScalableFixedRoutingLinearEstimationRun(
        problem=configured,
        solver_result=solved,
        quality=quality,
    )
