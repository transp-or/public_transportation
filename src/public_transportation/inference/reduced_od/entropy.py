"""Balanced and unbalanced sparse entropic journey-transport baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


EntropyMode = Literal["balanced", "unbalanced"]
MarginalSemantics = Literal["journey"]


class EntropyInfeasibleError(ValueError):
    """The declared journey marginals cannot be fitted on the support."""


def _immutable(value: object, dtype: np.dtype, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    result = np.array(array, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class EntropySupport:
    """Canonical sparse origin-destination support and generalized costs."""

    origin_index: np.ndarray
    destination_index: np.ndarray
    generalized_cost: np.ndarray
    number_of_origins: int
    number_of_destinations: int

    def __post_init__(self) -> None:
        origins = _immutable(self.origin_index, np.dtype(np.int64), "origin_index")
        destinations = _immutable(
            self.destination_index, np.dtype(np.int64), "destination_index"
        )
        costs = _immutable(
            self.generalized_cost, np.dtype(np.float64), "generalized_cost"
        )
        if not (origins.size == destinations.size == costs.size):
            raise ValueError("support arrays must have equal length.")
        if self.number_of_origins <= 0 or self.number_of_destinations <= 0:
            raise ValueError("support dimensions must be positive.")
        if origins.size and (
            np.any(origins < 0)
            or np.any(origins >= self.number_of_origins)
            or np.any(destinations < 0)
            or np.any(destinations >= self.number_of_destinations)
        ):
            raise ValueError("support indices are outside declared dimensions.")
        if not np.all(np.isfinite(costs)):
            raise ValueError("generalized costs must be finite.")
        pairs = list(zip(origins.tolist(), destinations.tolist(), strict=True))
        if len(set(pairs)) != len(pairs):
            raise ValueError("origin-destination support cells must be unique.")
        if pairs != sorted(pairs):
            raise ValueError("support cells must be sorted by origin and destination.")
        object.__setattr__(self, "origin_index", origins)
        object.__setattr__(self, "destination_index", destinations)
        object.__setattr__(self, "generalized_cost", costs)

    @property
    def number_of_cells(self) -> int:
        return int(self.origin_index.size)


@dataclass(frozen=True, slots=True)
class JourneyMarginals:
    """Externally justified passenger-journey marginals, never raw APC totals."""

    origin: np.ndarray
    destination: np.ndarray
    semantics: MarginalSemantics = "journey"

    def __post_init__(self) -> None:
        if self.semantics != "journey":
            raise ValueError(
                "entropy marginals must have journey semantics; raw APC boardings "
                "and alightings are not valid journey marginals."
            )
        origin = _immutable(self.origin, np.dtype(np.float64), "origin")
        destination = _immutable(self.destination, np.dtype(np.float64), "destination")
        if (
            not np.all(np.isfinite(origin))
            or not np.all(np.isfinite(destination))
            or np.any(origin < 0.0)
            or np.any(destination < 0.0)
        ):
            raise ValueError("journey marginals must be finite and non-negative.")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "destination", destination)


@dataclass(frozen=True, slots=True)
class EntropyConfig:
    """Numerically explicit balanced or KL-unbalanced Sinkhorn configuration."""

    mode: EntropyMode = "balanced"
    epsilon: float = 1.0
    origin_penalty: float = 10.0
    destination_penalty: float = 10.0
    tolerance: float = 1.0e-9
    maximum_iterations: int = 10_000

    def __post_init__(self) -> None:
        if self.mode not in {"balanced", "unbalanced"}:
            raise ValueError("entropy mode must be balanced or unbalanced.")
        for value, name in (
            (self.epsilon, "epsilon"),
            (self.tolerance, "tolerance"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for value, name in (
            (self.origin_penalty, "origin_penalty"),
            (self.destination_penalty, "destination_penalty"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if (
            isinstance(self.maximum_iterations, bool)
            or not isinstance(self.maximum_iterations, int)
            or self.maximum_iterations <= 0
        ):
            raise ValueError("maximum_iterations must be positive.")


@dataclass(frozen=True, slots=True)
class EntropyDiagnostics:
    iterations: int
    converged: bool
    maximum_origin_residual: float
    maximum_destination_residual: float
    origin_target_total: float
    destination_target_total: float
    fitted_total: float
    support_cells: int
    mode: EntropyMode


@dataclass(frozen=True, slots=True)
class EntropyResult:
    cell_flow: np.ndarray
    fitted_origin: np.ndarray
    fitted_destination: np.ndarray
    log_origin_scaling: np.ndarray
    log_destination_scaling: np.ndarray
    diagnostics: EntropyDiagnostics

    def __post_init__(self) -> None:
        for name in (
            "cell_flow",
            "fitted_origin",
            "fitted_destination",
            "log_origin_scaling",
            "log_destination_scaling",
        ):
            object.__setattr__(
                self,
                name,
                _immutable(getattr(self, name), np.dtype(np.float64), name),
            )


def _segment_logsumexp(
    values: np.ndarray, indices: np.ndarray, segments: int
) -> np.ndarray:
    maximum = np.full(segments, -np.inf, dtype=np.float64)
    np.maximum.at(maximum, indices, values)
    finite = np.isfinite(values) & np.isfinite(maximum[indices])
    totals = np.zeros(segments, dtype=np.float64)
    np.add.at(
        totals,
        indices[finite],
        np.exp(values[finite] - maximum[indices[finite]]),
    )
    result = np.full(segments, -np.inf, dtype=np.float64)
    represented = totals > 0.0
    result[represented] = maximum[represented] + np.log(totals[represented])
    return result


def _log_marginal(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, -np.inf, dtype=np.float64)
    positive = values > 0.0
    result[positive] = np.log(values[positive])
    return result


def _validate_problem(
    support: EntropySupport,
    marginals: JourneyMarginals,
    config: EntropyConfig,
) -> None:
    if marginals.origin.shape != (support.number_of_origins,):
        raise ValueError("origin marginals do not match support dimensions.")
    if marginals.destination.shape != (support.number_of_destinations,):
        raise ValueError("destination marginals do not match support dimensions.")
    if config.mode == "balanced" and not math.isclose(
        float(marginals.origin.sum()),
        float(marginals.destination.sum()),
        rel_tol=config.tolerance,
        abs_tol=config.tolerance,
    ):
        raise EntropyInfeasibleError("balanced journey marginal totals must agree.")
    unsupported_origins = np.flatnonzero(
        (marginals.origin > 0.0)
        & (np.bincount(support.origin_index, minlength=support.number_of_origins) == 0)
    )
    unsupported_destinations = np.flatnonzero(
        (marginals.destination > 0.0)
        & (
            np.bincount(
                support.destination_index,
                minlength=support.number_of_destinations,
            )
            == 0
        )
    )
    if config.mode == "balanced" and (
        unsupported_origins.size or unsupported_destinations.size
    ):
        raise EntropyInfeasibleError(
            "balanced positive marginals contain unsupported origins or destinations."
        )


def estimate_entropy_transport(
    support: EntropySupport,
    marginals: JourneyMarginals,
    *,
    config: EntropyConfig = EntropyConfig(),
    initial_log_origin_scaling: np.ndarray | None = None,
    initial_log_destination_scaling: np.ndarray | None = None,
) -> EntropyResult:
    """Fit a sparse balanced or KL-unbalanced entropic transport plan."""
    _validate_problem(support, marginals, config)
    log_kernel = -support.generalized_cost / config.epsilon
    log_origin_target = _log_marginal(marginals.origin)
    log_destination_target = _log_marginal(marginals.destination)
    log_u = (
        np.zeros(support.number_of_origins, dtype=np.float64)
        if initial_log_origin_scaling is None
        else np.array(initial_log_origin_scaling, dtype=np.float64, copy=True)
    )
    log_v = (
        np.zeros(support.number_of_destinations, dtype=np.float64)
        if initial_log_destination_scaling is None
        else np.array(initial_log_destination_scaling, dtype=np.float64, copy=True)
    )
    if log_u.shape != (support.number_of_origins,) or log_v.shape != (
        support.number_of_destinations,
    ):
        raise ValueError("initial entropy scaling arrays have invalid shapes.")
    if not np.all(np.isfinite(log_u)) or not np.all(np.isfinite(log_v)):
        raise ValueError("initial entropy scaling arrays must be finite.")
    origin_power = (
        1.0
        if config.mode == "balanced"
        else config.origin_penalty / (config.origin_penalty + config.epsilon)
    )
    destination_power = (
        1.0
        if config.mode == "balanced"
        else config.destination_penalty / (config.destination_penalty + config.epsilon)
    )
    converged = False
    iterations = 0
    for iterations in range(1, config.maximum_iterations + 1):
        previous_u = log_u.copy()
        previous_v = log_v.copy()
        row_logsum = _segment_logsumexp(
            log_kernel + log_v[support.destination_index],
            support.origin_index,
            support.number_of_origins,
        )
        valid_rows = np.isfinite(row_logsum) & np.isfinite(log_origin_target)
        log_u[:] = -np.inf
        log_u[valid_rows] = origin_power * (
            log_origin_target[valid_rows] - row_logsum[valid_rows]
        )
        column_logsum = _segment_logsumexp(
            log_kernel + log_u[support.origin_index],
            support.destination_index,
            support.number_of_destinations,
        )
        valid_columns = np.isfinite(column_logsum) & np.isfinite(log_destination_target)
        log_v[:] = -np.inf
        log_v[valid_columns] = destination_power * (
            log_destination_target[valid_columns] - column_logsum[valid_columns]
        )
        if config.mode == "balanced":
            log_plan = (
                log_kernel
                + log_u[support.origin_index]
                + log_v[support.destination_index]
            )
            plan = np.exp(log_plan)
            fitted_origin = np.bincount(
                support.origin_index,
                weights=plan,
                minlength=support.number_of_origins,
            )
            fitted_destination = np.bincount(
                support.destination_index,
                weights=plan,
                minlength=support.number_of_destinations,
            )
            residual = max(
                float(np.max(np.abs(fitted_origin - marginals.origin))),
                float(np.max(np.abs(fitted_destination - marginals.destination))),
            )
            scale = max(
                1.0,
                float(np.max(marginals.origin)),
                float(np.max(marginals.destination)),
            )
            converged = residual <= config.tolerance * scale
        else:
            finite_u = np.isfinite(log_u) & np.isfinite(previous_u)
            finite_v = np.isfinite(log_v) & np.isfinite(previous_v)
            delta_u = (
                float(np.max(np.abs(log_u[finite_u] - previous_u[finite_u])))
                if np.any(finite_u)
                else 0.0
            )
            delta_v = (
                float(np.max(np.abs(log_v[finite_v] - previous_v[finite_v])))
                if np.any(finite_v)
                else 0.0
            )
            converged = max(delta_u, delta_v) <= config.tolerance
        if converged:
            break
    log_plan = (
        log_kernel + log_u[support.origin_index] + log_v[support.destination_index]
    )
    plan = np.exp(log_plan)
    fitted_origin = np.bincount(
        support.origin_index,
        weights=plan,
        minlength=support.number_of_origins,
    )
    fitted_destination = np.bincount(
        support.destination_index,
        weights=plan,
        minlength=support.number_of_destinations,
    )
    origin_residual = float(np.max(np.abs(fitted_origin - marginals.origin)))
    destination_residual = float(
        np.max(np.abs(fitted_destination - marginals.destination))
    )
    if not converged:
        raise EntropyInfeasibleError(
            f"{config.mode} entropy did not converge in {iterations} iterations; "
            f"origin_residual={origin_residual:.6g}, "
            f"destination_residual={destination_residual:.6g}."
        )
    return EntropyResult(
        cell_flow=plan,
        fitted_origin=fitted_origin,
        fitted_destination=fitted_destination,
        log_origin_scaling=log_u,
        log_destination_scaling=log_v,
        diagnostics=EntropyDiagnostics(
            iterations=iterations,
            converged=True,
            maximum_origin_residual=origin_residual,
            maximum_destination_residual=destination_residual,
            origin_target_total=float(marginals.origin.sum()),
            destination_target_total=float(marginals.destination.sum()),
            fitted_total=float(plan.sum()),
            support_cells=support.number_of_cells,
            mode=config.mode,
        ),
    )
