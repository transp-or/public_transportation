"""Operator-only approximate quality diagnostics for large linear estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import (
    ArpackError,
    ArpackNoConvergence,
    LinearOperator as ScipyLinearOperator,
    cg,
    svds,
)

from .fixed_routing_linear_dense_solver import BoundKKTDiagnostics, evaluate_bound_kkt
from .fixed_routing_linear_problem import FixedRoutingLinearProblem
from .fixed_routing_linear_regularization import evaluate_linear_least_squares
from .fixed_routing_linear_quality import EstimateClassification

Array = np.ndarray


def _immutable(value: object) -> Array:
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ScalableQualityConfig:
    """Accuracy and reproducibility controls for approximate diagnostics."""

    smallest_singular_values: int = 6
    rank_relative_tolerance: float = 1.0e-8
    spectral_max_iterations: int | None = None
    resolution_samples: int = 32
    random_seed: int = 1729
    linear_solve_relative_tolerance: float = 1.0e-6
    linear_solve_max_iterations: int | None = None
    active_tolerance: float = 1.0e-7
    data_informed_threshold: float = 0.8
    regularization_dominated_threshold: float = 0.2
    classification_standard_error_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.smallest_singular_values <= 0:
            raise ValueError("smallest_singular_values must be strictly positive.")
        if self.resolution_samples <= 0:
            raise ValueError("resolution_samples must be strictly positive.")
        if self.linear_solve_max_iterations is not None and (
            self.linear_solve_max_iterations <= 0
        ):
            raise ValueError("linear_solve_max_iterations must be strictly positive.")
        if self.spectral_max_iterations is not None and self.spectral_max_iterations <= 0:
            raise ValueError("spectral_max_iterations must be strictly positive.")
        for name in (
            "rank_relative_tolerance",
            "linear_solve_relative_tolerance",
            "active_tolerance",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive.")
        if not (
            0.0
            <= self.regularization_dominated_threshold
            <= self.data_informed_threshold
            <= 1.0
        ):
            raise ValueError(
                "resolution thresholds must satisfy 0 <= regularization <= data <= 1."
            )
        if (
            not math.isfinite(self.classification_standard_error_multiplier)
            or self.classification_standard_error_multiplier < 0.0
        ):
            raise ValueError(
                "classification_standard_error_multiplier must be finite and non-negative."
            )


@dataclass(frozen=True, slots=True)
class ScalableLinearEstimateQuality:
    """Sampled spectral and resolution diagnostics with explicit uncertainty."""

    kkt: BoundKKTDiagnostics
    free_indices: Array
    largest_singular_value_estimate: float
    smallest_singular_value_estimates: Array
    rank_tolerance: float
    estimated_rank_upper_bound: int
    estimated_nullity_lower_bound: int
    condition_estimate: float
    spectral_converged: bool
    spectral_message: str
    resolution_samples: int
    resolution_converged_samples: int
    resolution_failed_samples: int
    random_seed: int
    data_resolution_score_estimate: Array
    data_resolution_standard_error: Array
    regularization_reliance_score_estimate: Array
    effective_data_degrees_of_freedom_estimate: float
    effective_data_degrees_of_freedom_standard_error: float
    classifications: tuple[EstimateClassification, ...]


@dataclass(frozen=True, slots=True)
class _RestrictedWeightedOperator:
    problem: FixedRoutingLinearProblem
    free_indices: Array

    @property
    def shape(self) -> tuple[int, int]:
        return self.problem.num_measurements, int(self.free_indices.size)

    def matvec(self, vector: object) -> Array:
        full = np.zeros(self.problem.num_free_od, dtype=float)
        full[self.free_indices] = np.asarray(vector).reshape(-1)
        return np.sqrt(self.problem.observation_weights) * (
            self.problem.measurement_operator.matvec(full)
        )

    def rmatvec(self, vector: object) -> Array:
        full = self.problem.measurement_operator.rmatvec(
            np.sqrt(self.problem.observation_weights) * np.asarray(vector).reshape(-1)
        )
        return np.asarray(full)[self.free_indices]


def _scipy_operator(operator: _RestrictedWeightedOperator) -> ScipyLinearOperator:
    return ScipyLinearOperator(
        operator.shape,
        matvec=operator.matvec,
        rmatvec=operator.rmatvec,
        dtype=np.float64,
    )


def _spectral_diagnostics(
    operator: _RestrictedWeightedOperator,
    config: ScalableQualityConfig,
) -> tuple[float, Array, float, int, int, float, bool, str]:
    rows, columns = operator.shape
    if columns == 0:
        return math.nan, np.empty(0), 0.0, 0, 0, math.nan, True, "empty free set"
    scipy_operator = _scipy_operator(operator)
    try:
        if min(rows, columns) == 1:
            if columns == 1:
                singular = float(np.linalg.norm(operator.matvec(np.ones(1))))
            else:
                singular = float(np.linalg.norm(operator.rmatvec(np.ones(1))))
            smallest = np.asarray([singular])
            largest = singular
        else:
            largest = float(
                svds(
                    scipy_operator,
                    k=1,
                    which="LM",
                    return_singular_vectors=False,
                    random_state=config.random_seed,
                    maxiter=config.spectral_max_iterations,
                )[0]
            )
            count = min(config.smallest_singular_values, min(rows, columns) - 1)
            smallest = np.sort(
                svds(
                    scipy_operator,
                    k=count,
                    which="SM",
                    return_singular_vectors=False,
                    random_state=config.random_seed,
                    maxiter=config.spectral_max_iterations,
                )
            )
        converged = True
        message = "converged"
    except (ArpackError, ArpackNoConvergence, TypeError, ValueError) as error:
        largest, smallest, converged = math.nan, np.empty(0), False
        message = f"{type(error).__name__}: {error}"
    tolerance = (
        math.nan
        if not math.isfinite(largest)
        else config.rank_relative_tolerance * largest
    )
    nullity_lower = (
        0
        if smallest.size == 0 or not math.isfinite(tolerance)
        else int(np.count_nonzero(smallest <= tolerance))
    )
    rank_upper = columns - nullity_lower
    if not smallest.size or not math.isfinite(largest):
        condition = math.nan
    elif smallest[0] <= tolerance:
        condition = math.inf
    else:
        condition = largest / float(smallest[0])
    return (
        largest,
        smallest,
        tolerance,
        rank_upper,
        nullity_lower,
        condition,
        converged,
        message,
    )


def _hessian_products(
    problem: FixedRoutingLinearProblem,
    weighted: _RestrictedWeightedOperator,
):
    free = weighted.free_indices

    def data_product(vector: object) -> Array:
        return weighted.rmatvec(weighted.matvec(vector))

    def regularization_product(vector: object) -> Array:
        full = np.zeros(problem.num_free_od, dtype=float)
        full[free] = np.asarray(vector)
        result = np.zeros(free.size, dtype=float)
        for block in problem.regularization_blocks:
            applied = block.operator.matvec(full)
            transpose = block.operator.rmatvec(applied)
            result += block.strength * np.asarray(transpose)[free]
        return result

    def combined_product(vector: object) -> Array:
        return data_product(vector) + regularization_product(vector)

    return data_product, regularization_product, combined_product


def analyze_linear_estimate_quality_scalable(
    problem: FixedRoutingLinearProblem,
    demand: object,
    *,
    config: ScalableQualityConfig | None = None,
) -> ScalableLinearEstimateQuality:
    """Estimate identifiability using only operator products and bounded memory."""
    config = ScalableQualityConfig() if config is None else config
    evaluation = evaluate_linear_least_squares(problem, demand)
    kkt = evaluate_bound_kkt(
        demand=demand,
        gradient=evaluation.gradient,
        lower_bounds=problem.lower_bounds,
        upper_bounds=problem.upper_bounds,
        active_tolerance=config.active_tolerance,
    )
    active = kkt.lower_active | kkt.upper_active | kkt.fixed_by_bounds
    free = np.flatnonzero(~active)
    weighted = _RestrictedWeightedOperator(problem, free)
    (
        largest,
        smallest,
        rank_tolerance,
        rank_upper,
        nullity_lower,
        condition,
        spectral_converged,
        spectral_message,
    ) = _spectral_diagnostics(weighted, config)

    data_product, _, combined_product = _hessian_products(problem, weighted)
    num_free = free.size
    samples: list[Array] = []
    rng = np.random.default_rng(config.random_seed)
    unregularized_sampled_null_space = nullity_lower > 0 and not any(
        block.strength > 0.0 for block in problem.regularization_blocks
    )
    if num_free and not unregularized_sampled_null_space:
        hessian = ScipyLinearOperator(
            (num_free, num_free),
            matvec=combined_product,
            rmatvec=combined_product,
            dtype=np.float64,
        )
        for _ in range(config.resolution_samples):
            probe = rng.choice(np.asarray([-1.0, 1.0]), size=num_free)
            solution, status = cg(
                hessian,
                data_product(probe),
                rtol=config.linear_solve_relative_tolerance,
                atol=0.0,
                maxiter=config.linear_solve_max_iterations,
            )
            if status == 0 and np.all(np.isfinite(solution)):
                samples.append(probe * solution)
    if samples:
        sample_array = np.asarray(samples)
        data_free = np.mean(sample_array, axis=0)
        standard_error_free = (
            np.zeros(num_free)
            if len(samples) == 1
            else np.std(sample_array, axis=0, ddof=1) / math.sqrt(len(samples))
        )
        trace_samples = np.sum(sample_array, axis=1)
        trace_standard_error = (
            0.0
            if len(samples) == 1
            else float(np.std(trace_samples, ddof=1) / math.sqrt(len(samples)))
        )
    else:
        data_free = np.full(num_free, np.nan)
        standard_error_free = np.full(num_free, np.nan)
        trace_standard_error = math.nan
    regularization_free = 1.0 - data_free
    data_scores = np.full(problem.num_free_od, np.nan)
    standard_errors = np.full(problem.num_free_od, np.nan)
    regularization_scores = np.full(problem.num_free_od, np.nan)
    data_scores[free] = data_free
    standard_errors[free] = standard_error_free
    regularization_scores[free] = regularization_free

    classifications: list[EstimateClassification] = []
    for index in range(problem.num_free_od):
        if kkt.fixed_by_bounds[index]:
            classification: EstimateClassification = "fixed_by_bounds"
        elif kkt.lower_active[index]:
            classification = "lower_bound_active"
        elif kkt.upper_active[index]:
            classification = "upper_bound_active"
        elif not np.isfinite(data_scores[index]):
            classification = "weakly_identified"
        else:
            uncertainty = (
                config.classification_standard_error_multiplier * standard_errors[index]
            )
            lower_score = data_scores[index] - uncertainty
            upper_score = data_scores[index] + uncertainty
            if lower_score >= config.data_informed_threshold:
                classification = "data_informed"
            elif upper_score <= config.regularization_dominated_threshold:
                classification = "regularization_dominated"
            else:
                classification = "mixed_data_and_regularization"
        classifications.append(classification)

    return ScalableLinearEstimateQuality(
        kkt=kkt,
        free_indices=_immutable(free),
        largest_singular_value_estimate=largest,
        smallest_singular_value_estimates=_immutable(smallest),
        rank_tolerance=rank_tolerance,
        estimated_rank_upper_bound=rank_upper,
        estimated_nullity_lower_bound=nullity_lower,
        condition_estimate=condition,
        spectral_converged=spectral_converged,
        spectral_message=spectral_message,
        resolution_samples=config.resolution_samples,
        resolution_converged_samples=len(samples),
        resolution_failed_samples=config.resolution_samples - len(samples),
        random_seed=config.random_seed,
        data_resolution_score_estimate=_immutable(data_scores),
        data_resolution_standard_error=_immutable(standard_errors),
        regularization_reliance_score_estimate=_immutable(regularization_scores),
        effective_data_degrees_of_freedom_estimate=(
            float(np.sum(data_free)) if samples else math.nan
        ),
        effective_data_degrees_of_freedom_standard_error=trace_standard_error,
        classifications=tuple(classifications),
    )
