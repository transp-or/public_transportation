"""Transparent route-level IPF baseline for boarding and alighting counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


TotalReconciliationPolicy = Literal["error", "average_observed_totals"]


def _text(value: str, name: str) -> str:
    parsed = str(value).strip()
    if not parsed:
        raise ValueError(f"{name} must be a non-empty string.")
    return parsed


def _immutable_float_vector(value: Any, name: str, size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numbers.")
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values.")
    result.setflags(write=False)
    return result


def _immutable_bool_vector(value: Any, name: str, size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    if not np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must contain Boolean values.")
    result = np.array(array, dtype=bool, copy=True, order="C")
    result.setflags(write=False)
    return result


def _immutable_float_array(value: Any) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class RouteLevelCounts:
    """Boarding/alighting marginals for one route pattern and service class."""

    route_pattern_id: str
    service_period_id: str
    stop_ids: tuple[str, ...]
    boarding_counts: np.ndarray
    alighting_counts: np.ndarray
    boarding_observed: np.ndarray
    alighting_observed: np.ndarray

    def __post_init__(self) -> None:
        _text(self.route_pattern_id, "route_pattern_id")
        _text(self.service_period_id, "service_period_id")
        if len(self.stop_ids) < 2:
            raise ValueError("stop_ids must contain at least two ordered stops.")
        if any(not str(stop_id).strip() for stop_id in self.stop_ids):
            raise ValueError("stop_ids must contain non-empty identifiers.")
        size = len(self.stop_ids)
        boarding = _immutable_float_vector(
            self.boarding_counts, "boarding_counts", size
        )
        alighting = _immutable_float_vector(
            self.alighting_counts, "alighting_counts", size
        )
        boarding_mask = _immutable_bool_vector(
            self.boarding_observed, "boarding_observed", size
        )
        alighting_mask = _immutable_bool_vector(
            self.alighting_observed, "alighting_observed", size
        )
        if np.any(boarding[~boarding_mask] != 0.0):
            raise ValueError(
                "unobserved boarding_counts entries must use zero placeholders."
            )
        if np.any(alighting[~alighting_mask] != 0.0):
            raise ValueError(
                "unobserved alighting_counts entries must use zero placeholders."
            )
        object.__setattr__(self, "boarding_counts", boarding)
        object.__setattr__(self, "alighting_counts", alighting)
        object.__setattr__(self, "boarding_observed", boarding_mask)
        object.__setattr__(self, "alighting_observed", alighting_mask)


@dataclass(frozen=True, slots=True)
class RouteLevelIPFConfig:
    """Numerical and explicit marginal-reconciliation policy."""

    tolerance: float = 1.0e-10
    max_iterations: int = 10_000
    total_reconciliation: TotalReconciliationPolicy = "error"

    def __post_init__(self) -> None:
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite.")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer.")
        if self.total_reconciliation not in {
            "error",
            "average_observed_totals",
        }:
            raise ValueError("total_reconciliation is unsupported.")


@dataclass(frozen=True, slots=True)
class RouteLevelDataQualityReport:
    """Audit of observed totals, reconciliation, coverage, and limitations."""

    observed_boarding_total: float
    observed_alighting_total: float
    reconciled_boarding_total: float
    reconciled_alighting_total: float
    original_total_imbalance: float
    maximum_marginal_adjustment: float
    all_boardings_observed: bool
    all_alightings_observed: bool
    underdetermined: bool
    total_reconciliation: TotalReconciliationPolicy
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RouteLevelIPFDiagnostics:
    """Convergence information for one route-level fit."""

    converged: bool
    iterations: int
    maximum_absolute_residual: float
    maximum_relative_residual: float
    structural_support_size: int
    data_quality: RouteLevelDataQualityReport


@dataclass(frozen=True, slots=True)
class RouteLevelIPFResult:
    """Leg-level result; it is deliberately not labeled as journey OD."""

    route_pattern_id: str
    service_period_id: str
    stop_ids: tuple[str, ...]
    leg_od_matrix: np.ndarray
    alighting_probabilities: np.ndarray
    fitted_boarding_counts: np.ndarray
    fitted_alighting_counts: np.ndarray
    reconciled_boarding_targets: np.ndarray
    reconciled_alighting_targets: np.ndarray
    diagnostics: RouteLevelIPFDiagnostics
    level: Literal["leg_level"] = "leg_level"
    boarding_semantics: Literal["leg_boarding_unclassified"] = (
        "leg_boarding_unclassified"
    )
    journey_od_compatible: Literal[False] = False

    def __post_init__(self) -> None:
        size = len(self.stop_ids)
        for name in (
            "leg_od_matrix",
            "alighting_probabilities",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != (size, size):
                raise ValueError(f"{name} must have shape ({size}, {size}).")
            object.__setattr__(self, name, _immutable_float_array(value))
        for name in (
            "fitted_boarding_counts",
            "fitted_alighting_counts",
            "reconciled_boarding_targets",
            "reconciled_alighting_targets",
        ):
            object.__setattr__(
                self,
                name,
                _immutable_float_vector(getattr(self, name), name, size),
            )


class RouteLevelInfeasibleError(ValueError):
    """The declared route-level constraints cannot be satisfied."""

    def __init__(
        self,
        message: str,
        *,
        data_quality: RouteLevelDataQualityReport,
    ) -> None:
        super().__init__(message)
        self.data_quality = data_quality


def _quality_report(
    counts: RouteLevelCounts,
    *,
    boarding_targets: np.ndarray,
    alighting_targets: np.ndarray,
    policy: TotalReconciliationPolicy,
    warnings: tuple[str, ...],
) -> RouteLevelDataQualityReport:
    all_boarding = bool(np.all(counts.boarding_observed))
    all_alighting = bool(np.all(counts.alighting_observed))
    observed_boarding_total = float(
        np.sum(counts.boarding_counts[counts.boarding_observed])
    )
    observed_alighting_total = float(
        np.sum(counts.alighting_counts[counts.alighting_observed])
    )
    adjustments = np.concatenate(
        (
            np.abs(boarding_targets - counts.boarding_counts),
            np.abs(alighting_targets - counts.alighting_counts),
        )
    )
    return RouteLevelDataQualityReport(
        observed_boarding_total=observed_boarding_total,
        observed_alighting_total=observed_alighting_total,
        reconciled_boarding_total=float(
            np.sum(boarding_targets[counts.boarding_observed])
        ),
        reconciled_alighting_total=float(
            np.sum(alighting_targets[counts.alighting_observed])
        ),
        original_total_imbalance=(observed_boarding_total - observed_alighting_total),
        maximum_marginal_adjustment=float(np.max(adjustments, initial=0.0)),
        all_boardings_observed=all_boarding,
        all_alightings_observed=all_alighting,
        underdetermined=not (all_boarding and all_alighting),
        total_reconciliation=policy,
        warnings=warnings,
    )


def _reconcile_targets(
    counts: RouteLevelCounts, config: RouteLevelIPFConfig
) -> tuple[np.ndarray, np.ndarray, RouteLevelDataQualityReport]:
    boarding = np.array(counts.boarding_counts, copy=True)
    alighting = np.array(counts.alighting_counts, copy=True)
    all_boarding = bool(np.all(counts.boarding_observed))
    all_alighting = bool(np.all(counts.alighting_observed))
    warnings: list[str] = []
    if not (all_boarding and all_alighting):
        warnings.append(
            "Incomplete marginals: unconstrained rows or columns remain "
            "seed-dependent and are not journey totals."
        )
    if all_boarding and all_alighting:
        boarding_total = float(np.sum(boarding))
        alighting_total = float(np.sum(alighting))
        scale = max(1.0, boarding_total, alighting_total)
        if abs(boarding_total - alighting_total) > config.tolerance * scale:
            if config.total_reconciliation == "error":
                report = _quality_report(
                    counts,
                    boarding_targets=boarding,
                    alighting_targets=alighting,
                    policy=config.total_reconciliation,
                    warnings=tuple(warnings),
                )
                raise RouteLevelInfeasibleError(
                    "complete boarding and alighting totals differ; select an "
                    "explicit reconciliation policy or correct the data.",
                    data_quality=report,
                )
            if boarding_total <= 0.0 or alighting_total <= 0.0:
                report = _quality_report(
                    counts,
                    boarding_targets=boarding,
                    alighting_targets=alighting,
                    policy=config.total_reconciliation,
                    warnings=tuple(warnings),
                )
                raise RouteLevelInfeasibleError(
                    "a positive total cannot be reconciled with a zero total.",
                    data_quality=report,
                )
            common_total = 0.5 * (boarding_total + alighting_total)
            boarding *= common_total / boarding_total
            alighting *= common_total / alighting_total
            warnings.append(
                "Complete totals were explicitly rescaled to their arithmetic "
                "mean; fitted marginals are reconciled rather than raw counts."
            )
    report = _quality_report(
        counts,
        boarding_targets=boarding,
        alighting_targets=alighting,
        policy=config.total_reconciliation,
        warnings=tuple(warnings),
    )
    return boarding, alighting, report


def _check_complete_triangular_feasibility(
    boarding: np.ndarray,
    alighting: np.ndarray,
    *,
    tolerance: float,
    report: RouteLevelDataQualityReport,
) -> None:
    size = boarding.size
    scale = max(1.0, float(np.sum(boarding)), float(np.sum(alighting)))
    threshold = tolerance * scale
    if boarding[-1] > threshold:
        raise RouteLevelInfeasibleError(
            "positive boarding at the final stop has no downstream destination.",
            data_quality=report,
        )
    if alighting[0] > threshold:
        raise RouteLevelInfeasibleError(
            "positive alighting at the first stop has no upstream origin.",
            data_quality=report,
        )
    for destination in range(size):
        available_upstream = float(np.sum(boarding[:destination]))
        required_prefix = float(np.sum(alighting[: destination + 1]))
        if required_prefix - available_upstream > threshold:
            raise RouteLevelInfeasibleError(
                "triangular support is infeasible: cumulative alightings through "
                f"stop {destination} exceed upstream boardings.",
                data_quality=report,
            )


def _residuals(
    matrix: np.ndarray,
    boarding_targets: np.ndarray,
    alighting_targets: np.ndarray,
    boarding_observed: np.ndarray,
    alighting_observed: np.ndarray,
) -> tuple[float, float]:
    boarding_residual = (
        np.sum(matrix, axis=1)[boarding_observed] - boarding_targets[boarding_observed]
    )
    alighting_residual = (
        np.sum(matrix, axis=0)[alighting_observed]
        - alighting_targets[alighting_observed]
    )
    residual = np.concatenate((boarding_residual, alighting_residual))
    target = np.concatenate(
        (
            boarding_targets[boarding_observed],
            alighting_targets[alighting_observed],
        )
    )
    absolute = float(np.max(np.abs(residual), initial=0.0))
    relative = float(np.max(np.abs(residual) / np.maximum(1.0, target), initial=0.0))
    return absolute, relative


def estimate_route_level_ipf(
    counts: RouteLevelCounts,
    *,
    config: RouteLevelIPFConfig = RouteLevelIPFConfig(),
    seed_matrix: np.ndarray | None = None,
) -> RouteLevelIPFResult:
    """Estimate an upper-triangular vehicle-leg OD matrix by IPF."""
    size = len(counts.stop_ids)
    boarding_targets, alighting_targets, quality = _reconcile_targets(counts, config)
    if bool(np.all(counts.boarding_observed)) and bool(
        np.all(counts.alighting_observed)
    ):
        _check_complete_triangular_feasibility(
            boarding_targets,
            alighting_targets,
            tolerance=config.tolerance,
            report=quality,
        )

    support = np.triu(np.ones((size, size), dtype=bool), k=1)
    if seed_matrix is None:
        matrix = support.astype(np.float64)
    else:
        seed = np.asarray(seed_matrix)
        if seed.shape != (size, size):
            raise ValueError(f"seed_matrix must have shape ({size}, {size}).")
        if not np.issubdtype(seed.dtype, np.number):
            raise TypeError("seed_matrix must contain numbers.")
        matrix = np.array(seed, dtype=np.float64, copy=True, order="C")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
            raise ValueError("seed_matrix must be finite and non-negative.")
        if np.any(matrix[~support] != 0.0):
            raise ValueError("seed_matrix must be zero outside i < j support.")
        if np.any(matrix[support] <= 0.0):
            raise ValueError("seed_matrix must be positive on every i < j cell.")

    converged = False
    absolute = float("inf")
    relative = float("inf")
    iterations = 0
    for iterations in range(1, config.max_iterations + 1):
        row_sums = np.sum(matrix, axis=1)
        for origin in np.flatnonzero(counts.boarding_observed):
            target = boarding_targets[origin]
            if target == 0.0:
                matrix[origin, :] = 0.0
            elif row_sums[origin] <= 0.0:
                raise RouteLevelInfeasibleError(
                    f"boarding target at stop {origin} has no remaining support.",
                    data_quality=quality,
                )
            else:
                matrix[origin, :] *= target / row_sums[origin]

        column_sums = np.sum(matrix, axis=0)
        for destination in np.flatnonzero(counts.alighting_observed):
            target = alighting_targets[destination]
            if target == 0.0:
                matrix[:, destination] = 0.0
            elif column_sums[destination] <= 0.0:
                raise RouteLevelInfeasibleError(
                    f"alighting target at stop {destination} has no remaining support.",
                    data_quality=quality,
                )
            else:
                matrix[:, destination] *= target / column_sums[destination]

        absolute, relative = _residuals(
            matrix,
            boarding_targets,
            alighting_targets,
            counts.boarding_observed,
            counts.alighting_observed,
        )
        if relative <= config.tolerance:
            converged = True
            break

    if not converged:
        raise RouteLevelInfeasibleError(
            "IPF did not converge within "
            f"{config.max_iterations} iterations; maximum relative residual "
            f"is {relative:.6g}.",
            data_quality=quality,
        )

    fitted_boarding = np.sum(matrix, axis=1)
    fitted_alighting = np.sum(matrix, axis=0)
    probabilities = np.zeros_like(matrix)
    positive_rows = fitted_boarding > 0.0
    probabilities[positive_rows] = (
        matrix[positive_rows] / fitted_boarding[positive_rows, None]
    )
    diagnostics = RouteLevelIPFDiagnostics(
        converged=True,
        iterations=iterations,
        maximum_absolute_residual=absolute,
        maximum_relative_residual=relative,
        structural_support_size=int(np.count_nonzero(support)),
        data_quality=quality,
    )
    return RouteLevelIPFResult(
        route_pattern_id=counts.route_pattern_id,
        service_period_id=counts.service_period_id,
        stop_ids=counts.stop_ids,
        leg_od_matrix=matrix,
        alighting_probabilities=probabilities,
        fitted_boarding_counts=fitted_boarding,
        fitted_alighting_counts=fitted_alighting,
        reconciled_boarding_targets=boarding_targets,
        reconciled_alighting_targets=alighting_targets,
        diagnostics=diagnostics,
    )
