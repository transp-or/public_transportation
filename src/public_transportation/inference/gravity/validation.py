"""Full-data model-adequacy diagnostics for fitted gravity models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import gammaln, xlogy  # type: ignore[import-untyped]

from public_transportation.inference.block_coordinate._canonical import fingerprint
from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)

from .estimator import GravityEstimationResult, gravity_model_fingerprint
from .objective import GravityObjectiveProblem, predict_gravity_measurements


def _immutable(value: object, *, name: str, length: int) -> np.ndarray:
    array = np.array(value, copy=True)
    if array.ndim != 1 or array.size != length:
        raise ValueError(f"{name} must have shape ({length},).")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class GravityValidationMetadata:
    """Optional generic grouping labels aligned with measurement rows."""

    num_measurements: int
    measurement_type: np.ndarray | None = None
    line: np.ndarray | None = None
    direction: np.ndarray | None = None
    stop: np.ndarray | None = None
    time_period: np.ndarray | None = None
    origin_zone: np.ndarray | None = None
    destination_zone: np.ndarray | None = None
    vehicle_journey: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.num_measurements <= 0:
            raise ValueError("num_measurements must be positive.")
        for name in (
            "measurement_type",
            "line",
            "direction",
            "stop",
            "time_period",
            "origin_zone",
            "destination_zone",
            "vehicle_journey",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _immutable(value, name=name, length=self.num_measurements),
                )


@dataclass(frozen=True, slots=True)
class GravityAdequacyConfig:
    standardized_residual_thresholds: tuple[float, ...] = (2.0, 3.0)
    systematic_group_mean_threshold: float = 1.0
    journey_correlation_threshold: float = 0.5

    def __post_init__(self) -> None:
        thresholds = self.standardized_residual_thresholds
        if not thresholds or any(
            not np.isfinite(value) or value <= 0 for value in thresholds
        ):
            raise ValueError("residual thresholds must be finite and positive.")
        if tuple(sorted(set(thresholds))) != thresholds:
            raise ValueError("residual thresholds must be unique and ascending.")
        if self.systematic_group_mean_threshold <= 0:
            raise ValueError("systematic_group_mean_threshold must be positive.")
        if not 0 <= self.journey_correlation_threshold <= 1:
            raise ValueError("journey_correlation_threshold must lie in [0, 1].")


@dataclass(frozen=True, slots=True)
class GravityGroupedResidualSummary:
    grouping: str
    label: str
    measurements: int
    observed_total: float
    modeled_total: float
    mean_residual: float
    mae: float
    rmse: float
    weighted_rmse: float
    mean_standardized_residual: float
    maximum_absolute_standardized_residual: float


@dataclass(frozen=True, slots=True)
class GravityJourneyCorrelationSummary:
    journey: str
    measurements: int
    adjacent_residual_correlation: float | None


@dataclass(frozen=True, slots=True)
class GravityAdequacyFindings:
    demand_restriction_pattern: bool
    possible_routing_or_timing_pattern: bool
    isolated_suspect_observations: bool
    systematic_overdispersion: bool
    messages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GravityAdequacyReport:
    schema_version: int
    model_fingerprint: str
    report_fingerprint: str
    measurements: int
    observed_total: float
    modeled_total: float
    negative_binomial_deviance: float
    poisson_deviance: float
    mae: float
    rmse: float
    weighted_rmse: float
    residual: np.ndarray
    standardized_nb_residual: np.ndarray
    threshold_counts: tuple[tuple[float, int, float], ...]
    observed_predicted_quantiles: tuple[tuple[float, float, float], ...]
    grouped_summaries: tuple[GravityGroupedResidualSummary, ...]
    journey_correlations: tuple[GravityJourneyCorrelationSummary, ...]
    findings: GravityAdequacyFindings

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported gravity adequacy schema version.")
        for name in ("residual", "standardized_nb_residual"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def _nb_logpmf(y: np.ndarray, mu: np.ndarray, dispersion: float) -> np.ndarray:
    return (
        gammaln(y + dispersion)
        - gammaln(dispersion)
        - gammaln(y + 1)
        - dispersion * np.log1p(mu / dispersion)
        + xlogy(y, mu)
        - xlogy(y, dispersion + mu)
    )


def _deviances(
    observed: np.ndarray, modeled: np.ndarray, dispersion: float
) -> tuple[float, float]:
    poisson = 2 * np.sum(xlogy(observed, observed / modeled) - (observed - modeled))
    saturated_mean = np.maximum(observed, np.finfo(modeled.dtype).tiny)
    nb = 2 * np.sum(
        _nb_logpmf(observed, saturated_mean, dispersion)
        - _nb_logpmf(observed, modeled, dispersion)
    )
    return float(max(0.0, nb)), float(max(0.0, poisson))


def _summary(
    grouping: str,
    label: str,
    indices: np.ndarray,
    observed: np.ndarray,
    modeled: np.ndarray,
    residual: np.ndarray,
    standardized: np.ndarray,
    variance: np.ndarray,
) -> GravityGroupedResidualSummary:
    local_residual = residual[indices]
    weights = 1 / variance[indices]
    return GravityGroupedResidualSummary(
        grouping,
        label,
        int(indices.size),
        float(np.sum(observed[indices])),
        float(np.sum(modeled[indices])),
        float(np.mean(local_residual)),
        float(np.mean(np.abs(local_residual))),
        float(np.sqrt(np.mean(local_residual**2))),
        float(np.sqrt(np.sum(weights * local_residual**2) / np.sum(weights))),
        float(np.mean(standardized[indices])),
        float(np.max(np.abs(standardized[indices]), initial=0.0)),
    )


def _groups(
    metadata: GravityValidationMetadata,
    observed: np.ndarray,
    modeled: np.ndarray,
    residual: np.ndarray,
    standardized: np.ndarray,
    variance: np.ndarray,
) -> tuple[GravityGroupedResidualSummary, ...]:
    summaries = []
    for name in (
        "measurement_type",
        "line",
        "direction",
        "stop",
        "time_period",
        "origin_zone",
        "destination_zone",
    ):
        labels = getattr(metadata, name)
        if labels is None:
            continue
        for label in sorted(np.unique(labels).tolist(), key=str):
            indices = np.flatnonzero(labels == label)
            summaries.append(
                _summary(
                    name,
                    str(label),
                    indices,
                    observed,
                    modeled,
                    residual,
                    standardized,
                    variance,
                )
            )
    return tuple(summaries)


def _journeys(
    labels: np.ndarray | None, standardized: np.ndarray
) -> tuple[GravityJourneyCorrelationSummary, ...]:
    if labels is None:
        return ()
    result = []
    for label in sorted(np.unique(labels).tolist(), key=str):
        values = standardized[labels == label]
        correlation = None
        if values.size >= 3 and np.std(values[:-1]) > 0 and np.std(values[1:]) > 0:
            correlation = float(np.corrcoef(values[:-1], values[1:])[0, 1])
        result.append(
            GravityJourneyCorrelationSummary(str(label), int(values.size), correlation)
        )
    return tuple(result)


def validate_full_data_gravity_adequacy(
    *,
    result: GravityEstimationResult,
    problem: GravityObjectiveProblem,
    compact_layout: CompactODAssignmentLayout,
    metadata: GravityValidationMetadata | None = None,
    config: GravityAdequacyConfig = GravityAdequacyConfig(),
) -> GravityAdequacyReport:
    """Evaluate calibration fit on all measurements; this is not holdout validation."""
    mask = problem.calibration_mask
    assert mask is not None
    if not np.all(mask):
        raise ValueError(
            "full-data adequacy requires every measurement in calibration."
        )
    if result.model_fingerprint != gravity_model_fingerprint(problem, compact_layout):
        raise ValueError("gravity result and validation problem fingerprints differ.")
    predicted, demand = predict_gravity_measurements(
        result.raw_parameters, problem=problem
    )
    modeled = np.asarray(predicted, dtype=np.float64)
    tolerance = 5.0e-6 if problem.features.dtype == np.dtype(np.float32) else 1.0e-8
    np.testing.assert_allclose(
        modeled, result.predicted_measurements, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        demand, result.free_od_demand, rtol=tolerance, atol=tolerance
    )
    observed = np.asarray(problem.observations, dtype=np.float64)
    dispersion = float(result.physical_parameters[2])
    variance = modeled + modeled**2 / dispersion
    residual = observed - modeled
    standardized = residual / np.sqrt(variance)
    nb_deviance, poisson_deviance = _deviances(observed, modeled, dispersion)
    weights = 1 / variance
    selected_metadata = metadata or GravityValidationMetadata(observed.size)
    if selected_metadata.num_measurements != observed.size:
        raise ValueError("validation metadata has the wrong measurement dimension.")
    grouped = _groups(
        selected_metadata,
        observed,
        modeled,
        residual,
        standardized,
        variance,
    )
    journeys = _journeys(selected_metadata.vehicle_journey, standardized)
    threshold_counts = tuple(
        (
            threshold,
            int(np.count_nonzero(np.abs(standardized) > threshold)),
            float(np.mean(np.abs(standardized) > threshold)),
        )
        for threshold in config.standardized_residual_thresholds
    )
    quantiles = tuple(
        (float(q), float(np.quantile(observed, q)), float(np.quantile(modeled, q)))
        for q in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    systematic_groups = tuple(
        item
        for item in grouped
        if abs(item.mean_standardized_residual)
        >= config.systematic_group_mean_threshold
    )
    coherent_journeys = tuple(
        item
        for item in journeys
        if item.adjacent_residual_correlation is not None
        and abs(item.adjacent_residual_correlation)
        >= config.journey_correlation_threshold
    )
    isolated = bool(
        threshold_counts[-1][1] > 0
        and threshold_counts[-1][2] < 0.05
        and not systematic_groups
    )
    overdispersion = bool(poisson_deviance > 1.25 * max(nb_deviance, 1e-12))
    messages = []
    if systematic_groups:
        messages.append(
            "Systematic grouped residuals may be explainable by gravity-demand restrictions."
        )
    if coherent_journeys:
        messages.append(
            "Coherent within-journey residuals may indicate routing or timing discrepancies."
        )
    if isolated:
        messages.append("A small number of isolated observations are suspect.")
    if overdispersion:
        messages.append("Poisson variation is insufficient relative to the NB model.")
    if not messages:
        messages.append("No configured structural discrepancy flag was triggered.")
    findings = GravityAdequacyFindings(
        bool(systematic_groups),
        bool(coherent_journeys),
        isolated,
        overdispersion,
        tuple(messages),
    )
    report_payload: dict[str, Any] = {
        "schema_version": 1,
        "model_fingerprint": result.model_fingerprint,
        "observed": observed,
        "modeled": modeled,
        "dispersion": dispersion,
        "metadata": {
            name: getattr(selected_metadata, name)
            for name in selected_metadata.__dataclass_fields__
        },
        "config": config,
    }
    return GravityAdequacyReport(
        1,
        result.model_fingerprint,
        fingerprint(report_payload),
        observed.size,
        float(np.sum(observed)),
        float(np.sum(modeled)),
        nb_deviance,
        poisson_deviance,
        float(np.mean(np.abs(residual))),
        float(np.sqrt(np.mean(residual**2))),
        float(np.sqrt(np.sum(weights * residual**2) / np.sum(weights))),
        residual,
        standardized,
        threshold_counts,
        quantiles,
        grouped,
        journeys,
        findings,
    )
