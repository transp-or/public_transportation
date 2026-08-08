"""Full-data adequacy and advisory reduced-model relaxation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from .estimator import ReducedODFitResult
from .objective import MinimalGravityProblem, evaluate_minimal_gravity_objective
from .parameters import transform_minimal_gravity_parameters
from .validation import ReducedODMeasurementMetadata


RecommendationStrength = Literal["none", "weak", "moderate", "strong"]
ReducedODModelStage = Literal["J0", "J1", "J2", "J3", "J4"]


@dataclass(frozen=True, slots=True)
class ReducedODAdequacyConfig:
    standardized_residual_thresholds: tuple[float, ...] = (2.0, 3.0)
    systematic_group_threshold: float = 1.0
    weak_identification_condition: float = 1.0e8
    curvature_floor: float = 1.0e-8
    minimum_observations_per_parameter: float = 5.0

    def __post_init__(self) -> None:
        if (
            not self.standardized_residual_thresholds
            or tuple(sorted(set(self.standardized_residual_thresholds)))
            != self.standardized_residual_thresholds
            or any(value <= 0.0 for value in self.standardized_residual_thresholds)
        ):
            raise ValueError("residual thresholds must be positive, unique and sorted.")
        if self.systematic_group_threshold <= 0.0:
            raise ValueError("systematic_group_threshold must be positive.")
        if self.weak_identification_condition <= 1.0:
            raise ValueError("weak_identification_condition must exceed one.")
        if self.curvature_floor <= 0.0:
            raise ValueError("curvature_floor must be positive.")
        if self.minimum_observations_per_parameter <= 0.0:
            raise ValueError("minimum_observations_per_parameter must be positive.")


@dataclass(frozen=True, slots=True)
class ReducedODGroupedResidualSummary:
    grouping: str
    label: str
    measurements: int
    observed_total: float
    modeled_total: float
    mean_residual: float
    rmse: float
    mean_standardized_residual: float
    maximum_absolute_standardized_residual: float


@dataclass(frozen=True, slots=True)
class ReducedODIdentificationDiagnostics:
    parameter_count: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float
    weakly_identified: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReducedODAdequacyReport:
    model_fingerprint: str
    measurements: int
    residual: np.ndarray
    standardized_residual: np.ndarray
    rmse: float
    weighted_rmse: float
    threshold_counts: tuple[tuple[float, int, float], ...]
    grouped_summaries: tuple[ReducedODGroupedResidualSummary, ...]
    identification: ReducedODIdentificationDiagnostics
    calibration_adequacy_only: bool = True

    def __post_init__(self) -> None:
        if not self.calibration_adequacy_only:
            raise ValueError("adequacy must not be labeled predictive validation.")
        for name in ("residual", "standardized_residual"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def _summaries(
    *,
    metadata: ReducedODMeasurementMetadata | None,
    observed: np.ndarray,
    modeled: np.ndarray,
    residual: np.ndarray,
    standardized: np.ndarray,
) -> tuple[ReducedODGroupedResidualSummary, ...]:
    if metadata is None:
        return ()
    result: list[ReducedODGroupedResidualSummary] = []
    for grouping in (
        "measurement_type",
        "line",
        "direction",
        "stop",
        "time_period",
        "vehicle_journey",
        "origin_zone",
        "destination_zone",
        "transfer_place",
    ):
        labels = getattr(metadata, grouping)
        if labels is None:
            continue
        for label in sorted(np.unique(labels), key=str):
            indices = np.flatnonzero(labels == label)
            local = residual[indices]
            local_standardized = standardized[indices]
            result.append(
                ReducedODGroupedResidualSummary(
                    grouping=grouping,
                    label=str(label),
                    measurements=int(indices.size),
                    observed_total=float(np.sum(observed[indices])),
                    modeled_total=float(np.sum(modeled[indices])),
                    mean_residual=float(np.mean(local)),
                    rmse=float(np.sqrt(np.mean(local * local))),
                    mean_standardized_residual=float(np.mean(local_standardized)),
                    maximum_absolute_standardized_residual=float(
                        np.max(np.abs(local_standardized), initial=0.0)
                    ),
                )
            )
    return tuple(result)


def diagnose_reduced_od_adequacy(
    *,
    fit: ReducedODFitResult,
    problem: MinimalGravityProblem,
    metadata: ReducedODMeasurementMetadata | None = None,
    config: ReducedODAdequacyConfig = ReducedODAdequacyConfig(),
) -> ReducedODAdequacyReport:
    """Describe in-sample adequacy; never present it as prediction evidence."""
    if problem.calibration_mask is not None:
        raise ValueError("full-data adequacy does not accept a calibration mask.")
    if fit.raw_parameters.shape != (problem.parameter_layout.size,):
        raise ValueError("fit parameters do not match the problem.")
    if metadata is not None and (
        metadata.number_of_measurements != problem.observations.size
    ):
        raise ValueError("adequacy metadata has the wrong measurement dimension.")
    evaluation = evaluate_minimal_gravity_objective(fit.raw_parameters, problem=problem)
    modeled = np.asarray(evaluation.measurement_mean, dtype=np.float64)
    observed = problem.observations
    transformed = transform_minimal_gravity_parameters(
        fit.raw_parameters, layout=problem.parameter_layout
    )
    dispersion = (
        np.inf if transformed.dispersion is None else float(transformed.dispersion)
    )
    variance = (
        modeled if np.isinf(dispersion) else modeled + modeled * modeled / dispersion
    )
    residual = observed - modeled
    standardized = residual / np.sqrt(variance)
    absolute = np.abs(standardized)
    threshold_counts = tuple(
        (
            threshold,
            int(np.count_nonzero(absolute >= threshold)),
            float(np.mean(absolute >= threshold)),
        )
        for threshold in config.standardized_residual_thresholds
    )

    def objective(raw: jax.Array) -> jax.Array:
        return evaluate_minimal_gravity_objective(raw, problem=problem).objective

    hessian = np.asarray(
        jax.hessian(objective)(jnp.asarray(fit.raw_parameters)), dtype=np.float64
    )
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues = np.linalg.eigvalsh(hessian)
    stabilized = np.maximum(eigenvalues, config.curvature_floor)
    condition = float(stabilized[-1] / stabilized[0])
    warnings: list[str] = []
    if eigenvalues[0] <= config.curvature_floor:
        warnings.append("Objective curvature is singular or nearly singular.")
    if condition >= config.weak_identification_condition:
        warnings.append(f"Curvature condition number is {condition:.3g}.")
    if (
        observed.size / fit.raw_parameters.size
        < config.minimum_observations_per_parameter
    ):
        warnings.append("Observation support per fitted parameter is weak.")
    identification = ReducedODIdentificationDiagnostics(
        parameter_count=int(fit.raw_parameters.size),
        minimum_eigenvalue=float(eigenvalues[0]),
        maximum_eigenvalue=float(eigenvalues[-1]),
        condition_number=condition,
        weakly_identified=bool(warnings),
        warnings=tuple(warnings),
    )
    return ReducedODAdequacyReport(
        model_fingerprint=fit.manifest.model_fingerprint,
        measurements=int(observed.size),
        residual=residual,
        standardized_residual=standardized,
        rmse=float(np.sqrt(np.mean(residual * residual))),
        weighted_rmse=float(np.sqrt(np.mean(residual * residual / variance))),
        threshold_counts=threshold_counts,
        grouped_summaries=_summaries(
            metadata=metadata,
            observed=observed,
            modeled=modeled,
            residual=residual,
            standardized=standardized,
        ),
        identification=identification,
    )


@dataclass(frozen=True, slots=True)
class ReducedODRelaxationRecommendation:
    stage: ReducedODModelStage
    name: str
    grouping: str
    applicable: bool
    added_parameters: int | None
    maximum_group_signal: float | None
    strength: RecommendationStrength
    weak_identification_warnings: tuple[str, ...]
    explanation: str


@dataclass(frozen=True, slots=True)
class ReducedODRecommendationReport:
    parent_model_fingerprint: str
    candidates: tuple[ReducedODRelaxationRecommendation, ...]
    advisory_only: bool = True

    def __post_init__(self) -> None:
        if not self.advisory_only:
            raise ValueError("relaxation recommendations must remain advisory.")


def recommend_reduced_od_relaxations(
    *,
    adequacy: ReducedODAdequacyReport,
    metadata: ReducedODMeasurementMetadata,
    config: ReducedODAdequacyConfig = ReducedODAdequacyConfig(),
) -> ReducedODRecommendationReport:
    """Recommend at most one-step children; do not construct or fit them."""
    definitions = (
        ("J1", "broad_period_cost", "time_period"),
        ("J2", "destination_zone_attractiveness", "destination_zone"),
        ("J3", "origin_zone_production", "origin_zone"),
        ("J4", "transfer_place_correction", "transfer_place"),
    )
    candidates: list[ReducedODRelaxationRecommendation] = []
    for stage, name, grouping in definitions:
        labels = getattr(metadata, grouping)
        summaries = tuple(
            item for item in adequacy.grouped_summaries if item.grouping == grouping
        )
        if labels is None or np.unique(labels).size < 2 or not summaries:
            candidates.append(
                ReducedODRelaxationRecommendation(
                    stage=stage,  # type: ignore[arg-type]
                    name=name,
                    grouping=grouping,
                    applicable=False,
                    added_parameters=None,
                    maximum_group_signal=None,
                    strength="none",
                    weak_identification_warnings=(
                        f"metadata.{grouping} needs at least two groups.",
                    ),
                    explanation="No child was created or fitted.",
                )
            )
            continue
        added = int(np.unique(labels).size - 1)
        signal = max(abs(item.mean_standardized_residual) for item in summaries)
        strength: RecommendationStrength = (
            "strong"
            if signal >= 3.0
            else "moderate"
            if signal >= 2.0
            else "weak"
            if signal >= config.systematic_group_threshold
            else "none"
        )
        warnings: list[str] = []
        if (
            metadata.number_of_measurements / added
            < config.minimum_observations_per_parameter
        ):
            warnings.append("Observation support per added parameter is weak.")
            if strength == "strong":
                strength = "moderate"
        candidates.append(
            ReducedODRelaxationRecommendation(
                stage=stage,  # type: ignore[arg-type]
                name=name,
                grouping=grouping,
                applicable=True,
                added_parameters=added,
                maximum_group_signal=float(signal),
                strength=strength,
                weak_identification_warnings=tuple(warnings),
                explanation=(
                    f"Largest absolute grouped mean standardized residual is {signal:.4g}. "
                    "This score is advisory; the parent fit is unchanged."
                ),
            )
        )
    return ReducedODRecommendationReport(
        parent_model_fingerprint=adequacy.model_fingerprint,
        candidates=tuple(candidates),
    )
