"""Exact estimate-quality diagnostics for small fixed-routing problems."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .fixed_routing_linear_dense_solver import BoundKKTDiagnostics, evaluate_bound_kkt
from .fixed_routing_linear_objective import evaluate_linear_data_fit
from .fixed_routing_linear_problem import FixedRoutingLinearProblem
from .fixed_routing_linear_regularization import evaluate_linear_least_squares
from .linear_operator import materialize_linear_operator

Array = np.ndarray
EstimateClassification = Literal[
    "data_informed",
    "mixed_data_and_regularization",
    "regularization_dominated",
    "weakly_identified",
    "lower_bound_active",
    "upper_bound_active",
    "fixed_by_bounds",
]


def _immutable(value: object) -> Array:
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class LinearEstimateQuality:
    """Exact local resolution and identifiability report for a small problem."""

    kkt: BoundKKTDiagnostics
    free_indices: Array
    measurement_singular_values: Array
    measurement_rank: int
    measurement_nullity: int
    measurement_rank_tolerance: float
    measurement_condition_estimate: float
    combined_rank: int
    combined_nullity: int
    data_hessian: Array
    regularization_hessian: Array
    combined_hessian: Array
    data_resolution: Array
    regularization_resolution: Array
    resolution_closure_inf_norm: float
    effective_data_degrees_of_freedom: float
    data_mode_fractions: Array
    data_modes: Array
    data_resolution_score: Array
    regularization_reliance_score: Array
    null_space_participation: Array
    classifications: tuple[EstimateClassification, ...]


def _rank_tolerance(matrix: Array, singular_values: Array) -> float:
    if singular_values.size == 0:
        return 0.0
    return float(
        max(matrix.shape)
        * np.finfo(matrix.dtype).eps
        * float(singular_values[0])
    )


def _quality_gradient(problem: FixedRoutingLinearProblem, demand: object) -> Array:
    if problem.regularization_selection == "configured":
        return np.asarray(evaluate_linear_least_squares(problem, demand).gradient)
    return np.asarray(evaluate_linear_data_fit(problem, demand).gradient)


def analyze_linear_estimate_quality(
    problem: FixedRoutingLinearProblem,
    demand: object,
    *,
    active_tolerance: float = 1.0e-7,
    data_informed_threshold: float = 0.8,
    regularization_dominated_threshold: float = 0.2,
    null_participation_tolerance: float = 1.0e-8,
    max_materialized_entries: int = 10_000_000,
    exclude_active_bounds: bool = True,
) -> LinearEstimateQuality:
    """Compute exact rank and resolution diagnostics on the non-active set."""
    if not 0.0 <= regularization_dominated_threshold <= data_informed_threshold <= 1.0:
        raise ValueError(
            "resolution thresholds must satisfy 0 <= regularization <= data <= 1."
        )
    if not math.isfinite(null_participation_tolerance) or null_participation_tolerance < 0.0:
        raise ValueError(
            "null_participation_tolerance must be finite and non-negative."
        )
    gradient = _quality_gradient(problem, demand)
    kkt = evaluate_bound_kkt(
        demand=demand,
        gradient=gradient,
        lower_bounds=problem.lower_bounds,
        upper_bounds=problem.upper_bounds,
        active_tolerance=active_tolerance,
    )
    active_lower = (
        kkt.lower_active
        if exclude_active_bounds
        else np.zeros(problem.num_free_od, dtype=bool)
    )
    active_upper = (
        kkt.upper_active
        if exclude_active_bounds
        else np.zeros(problem.num_free_od, dtype=bool)
    )
    active = active_lower | active_upper | kkt.fixed_by_bounds
    free_indices = np.flatnonzero(~active)
    num_free = int(free_indices.size)

    measurement = materialize_linear_operator(
        problem.measurement_operator,
        max_entries=max_materialized_entries,
    ).astype(np.float64, copy=False)
    weighted = np.sqrt(problem.observation_weights)[:, None] * measurement
    weighted_free = weighted[:, free_indices]
    singular_values = np.linalg.svd(weighted_free, compute_uv=False)
    rank_tolerance = _rank_tolerance(weighted_free, singular_values)
    measurement_rank = int(np.count_nonzero(singular_values > rank_tolerance))
    measurement_nullity = num_free - measurement_rank
    if singular_values.size == 0:
        condition = math.nan
    elif measurement_rank < num_free:
        condition = math.inf
    elif singular_values[-1] <= rank_tolerance:
        condition = math.inf
    else:
        condition = float(singular_values[0] / singular_values[-1])

    data_hessian = weighted_free.T @ weighted_free
    regularization_hessian = np.zeros((num_free, num_free), dtype=np.float64)
    if problem.regularization_selection == "configured":
        for block in problem.regularization_blocks:
            operator = materialize_linear_operator(
                block.operator,
                max_entries=max_materialized_entries,
            ).astype(np.float64, copy=False)
            restricted = operator[:, free_indices]
            regularization_hessian += block.strength * (restricted.T @ restricted)
    combined_hessian = data_hessian + regularization_hessian

    eigenvalues, eigenvectors = np.linalg.eigh(combined_hessian)
    combined_scale = float(np.max(np.abs(eigenvalues), initial=0.0))
    combined_tolerance = (
        max(1, num_free) * np.finfo(np.float64).eps * combined_scale
    )
    positive = eigenvalues > combined_tolerance
    combined_rank = int(np.count_nonzero(positive))
    combined_nullity = num_free - combined_rank
    if combined_rank:
        positive_vectors = eigenvectors[:, positive]
        inverse = (
            positive_vectors
            * (1.0 / eigenvalues[positive])[None, :]
        ) @ positive_vectors.T
        inverse_sqrt = (
            positive_vectors
            * (1.0 / np.sqrt(eigenvalues[positive]))[None, :]
        ) @ positive_vectors.T
    else:
        inverse = np.zeros_like(combined_hessian)
        inverse_sqrt = np.zeros_like(combined_hessian)

    data_resolution = inverse @ data_hessian
    regularization_resolution = inverse @ regularization_hessian
    identifiable_projector = inverse @ combined_hessian
    closure = float(
        np.linalg.norm(
            data_resolution + regularization_resolution - identifiable_projector,
            ord=np.inf,
        )
        if num_free
        else 0.0
    )
    null_projector = np.eye(num_free) - identifiable_projector
    null_participation_free = np.clip(np.diag(null_projector), 0.0, 1.0)

    whitened_data = inverse_sqrt @ data_hessian @ inverse_sqrt
    if num_free:
        fractions, whitened_modes = np.linalg.eigh(
            0.5 * (whitened_data + whitened_data.T)
        )
        order = np.argsort(fractions)[::-1]
        fractions = np.clip(fractions[order], 0.0, 1.0)
        data_modes = inverse_sqrt @ whitened_modes[:, order]
    else:
        fractions = np.empty(0)
        data_modes = np.empty((0, 0))

    data_scores = np.full(problem.num_free_od, np.nan)
    regularization_scores = np.full(problem.num_free_od, np.nan)
    null_scores = np.full(problem.num_free_od, np.nan)
    data_scores[free_indices] = np.diag(data_resolution)
    regularization_scores[free_indices] = np.diag(regularization_resolution)
    null_scores[free_indices] = null_participation_free

    classifications: list[EstimateClassification] = []
    free_position = {int(index): position for position, index in enumerate(free_indices)}
    for index in range(problem.num_free_od):
        if kkt.fixed_by_bounds[index]:
            classification: EstimateClassification = "fixed_by_bounds"
        elif active_lower[index]:
            classification = "lower_bound_active"
        elif active_upper[index]:
            classification = "upper_bound_active"
        else:
            position = free_position[index]
            if null_participation_free[position] > null_participation_tolerance:
                classification = "weakly_identified"
            elif data_scores[index] >= data_informed_threshold:
                classification = "data_informed"
            elif data_scores[index] <= regularization_dominated_threshold:
                classification = "regularization_dominated"
            else:
                classification = "mixed_data_and_regularization"
        classifications.append(classification)

    return LinearEstimateQuality(
        kkt=kkt,
        free_indices=_immutable(free_indices),
        measurement_singular_values=_immutable(singular_values),
        measurement_rank=measurement_rank,
        measurement_nullity=measurement_nullity,
        measurement_rank_tolerance=rank_tolerance,
        measurement_condition_estimate=condition,
        combined_rank=combined_rank,
        combined_nullity=combined_nullity,
        data_hessian=_immutable(data_hessian),
        regularization_hessian=_immutable(regularization_hessian),
        combined_hessian=_immutable(combined_hessian),
        data_resolution=_immutable(data_resolution),
        regularization_resolution=_immutable(regularization_resolution),
        resolution_closure_inf_norm=closure,
        effective_data_degrees_of_freedom=float(np.trace(data_resolution)),
        data_mode_fractions=_immutable(fractions),
        data_modes=_immutable(data_modes),
        data_resolution_score=_immutable(data_scores),
        regularization_reliance_score=_immutable(regularization_scores),
        null_space_participation=_immutable(null_scores),
        classifications=tuple(classifications),
    )
