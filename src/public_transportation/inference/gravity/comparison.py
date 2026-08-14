"""Auditable held-out comparison records for explicit gravity specifications."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .estimator import GravityEstimationResult
from .holdout import GravityHoldoutValidationReport
from .objective import GravityObjectiveProblem


@dataclass(frozen=True, slots=True)
class GravityModelFitSummary:
    model_name: str
    specification_fingerprint: str
    parameter_count: int
    in_sample_objective: float
    held_out_data_log_likelihood: float | None
    held_out_rmse: float | None
    convergence_status: str
    success: bool
    gradient_inf_norm: float
    regularization_contribution: float
    calibration_measurements: int
    excluded_measurements: int
    observed_count_mass: float
    predicted_count_mass: float
    unsupported_rows: int


def summarize_gravity_model_fit(
    *,
    result: GravityEstimationResult,
    problem: GravityObjectiveProblem,
    holdout: GravityHoldoutValidationReport | None = None,
    unsupported_measurement_mask: object | None = None,
) -> GravityModelFitSummary:
    """Combine fit, gradient, support, count-mass, and optional holdout audits."""
    specification = problem.parameter_layout.specification
    if result.specification_fingerprint and (
        result.specification_fingerprint != specification.fingerprint
    ):
        raise ValueError("result and comparison specification fingerprints differ.")
    unsupported = (
        np.zeros(problem.observations.size, dtype=bool)
        if unsupported_measurement_mask is None
        else np.asarray(unsupported_measurement_mask, dtype=bool)
    )
    if unsupported.shape != problem.observations.shape:
        raise ValueError("unsupported_measurement_mask must match observations.")
    calibration = np.asarray(problem.calibration_mask, dtype=bool)
    if np.any(unsupported & calibration):
        raise ValueError("unsupported rows must be excluded from calibration.")
    if holdout is not None and (
        holdout.selected_specification_fingerprint != specification.fingerprint
    ):
        raise ValueError("holdout report belongs to a different specification.")
    return GravityModelFitSummary(
        model_name=specification.model_name,
        specification_fingerprint=specification.fingerprint,
        parameter_count=problem.parameter_layout.size,
        in_sample_objective=result.objective,
        held_out_data_log_likelihood=(
            None if holdout is None else holdout.holdout.data_log_likelihood
        ),
        held_out_rmse=None if holdout is None else holdout.holdout.rmse,
        convergence_status=result.status,
        success=result.success,
        gradient_inf_norm=float(np.max(np.abs(result.gradient), initial=0.0)),
        regularization_contribution=result.regularization_contribution,
        calibration_measurements=int(np.count_nonzero(calibration)),
        excluded_measurements=int(calibration.size - np.count_nonzero(calibration)),
        observed_count_mass=float(np.sum(problem.observations[calibration])),
        predicted_count_mass=float(np.sum(result.predicted_measurements[calibration])),
        unsupported_rows=int(np.count_nonzero(unsupported)),
    )


def rank_gravity_model_summaries(
    summaries: object,
) -> tuple[GravityModelFitSummary, ...]:
    """Rank only by held-out evidence, then predictive error and parsimony."""
    values = tuple(summaries)  # type: ignore[arg-type]
    if not values:
        raise ValueError("at least one gravity-model summary is required.")
    if any(
        item.held_out_data_log_likelihood is None or item.held_out_rmse is None
        for item in values
    ):
        raise ValueError("model selection requires held-out metrics for every model.")
    return tuple(
        sorted(
            values,
            key=lambda item: (
                -float(item.held_out_data_log_likelihood),
                float(item.held_out_rmse),
                item.parameter_count,
            ),
        )
    )
