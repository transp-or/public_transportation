"""Advisory score diagnostics for candidate gravity-model relaxations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.inference.block_coordinate._canonical import fingerprint
from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)

from .estimator import GravityEstimationResult, gravity_model_fingerprint
from .objective import (
    GravityObjectiveProblem,
    evaluate_gravity_objective,
    predict_gravity_measurements,
)
from .parameters import GravityParameterLayout, warm_start_gravity_parameters
from .relaxations import add_gravity_relaxation
from .specification import GravityEffectScope
from .validation import (
    GravityAdequacyReport,
    GravityValidationMetadata,
)

RecommendationStrength = Literal["none", "weak", "moderate", "strong"]


@dataclass(frozen=True, slots=True)
class GravityRecommendationConfig:
    curvature_floor: float = 1.0e-8
    weak_score_threshold: float = 2.0
    moderate_score_threshold: float = 6.0
    strong_score_threshold: float = 10.0
    weak_identification_condition: float = 1.0e8
    minimum_observations_per_parameter: float = 5.0
    feature_pattern_correlation_threshold: float = 0.3
    grouped_residual_pattern_threshold: float = 1.0
    parent_gradient_tolerance: float = 1.0e-3

    def __post_init__(self) -> None:
        thresholds = (
            self.weak_score_threshold,
            self.moderate_score_threshold,
            self.strong_score_threshold,
        )
        if not 0 <= thresholds[0] <= thresholds[1] <= thresholds[2]:
            raise ValueError("recommendation score thresholds must be ascending.")
        if self.curvature_floor <= 0:
            raise ValueError("curvature_floor must be positive.")
        if self.weak_identification_condition <= 1:
            raise ValueError("weak_identification_condition must exceed one.")
        if self.minimum_observations_per_parameter <= 0:
            raise ValueError("minimum_observations_per_parameter must be positive.")
        if not 0 <= self.feature_pattern_correlation_threshold <= 1:
            raise ValueError("feature_pattern_correlation_threshold must lie in [0, 1].")
        if self.grouped_residual_pattern_threshold <= 0:
            raise ValueError("grouped_residual_pattern_threshold must be positive.")
        if self.parent_gradient_tolerance <= 0:
            raise ValueError("parent_gradient_tolerance must be positive.")


@dataclass(frozen=True, slots=True)
class GravityRelaxationRecommendation:
    candidate_name: str
    discrepancy_addressed: str
    applicable: bool
    added_parameters: int | None
    score_test_statistic: float | None
    approximate_gain: float | None
    observation_support: int
    support_groups: int | None
    expected_calibration_deviance_improvement: float | None
    estimated_computational_cost: str
    weak_identification_warnings: tuple[str, ...]
    recommendation_strength: RecommendationStrength
    explanation: str


@dataclass(frozen=True, slots=True)
class GravityRelaxationRecommendationReport:
    schema_version: int
    model_fingerprint: str
    adequacy_report_fingerprint: str
    report_fingerprint: str
    candidates: tuple[GravityRelaxationRecommendation, ...]
    diagnostic_warnings: tuple[str, ...]
    advisory_only: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported gravity recommendation schema version.")
        if not self.advisory_only:
            raise ValueError("gravity relaxation recommendations must remain advisory.")


_ATOMIC_CANDIDATES = (
    (
        GravityEffectScope.DESTINATION_ZONE,
        "destination_zone_attractiveness",
        "Systematic residual scale differences between destination zones.",
        "destination_zone",
    ),
    (
        GravityEffectScope.TIME_PERIOD,
        "broad_time_period",
        "Systematic fit differences between broad time periods.",
        "time_period",
    ),
    (
        GravityEffectScope.ORIGIN_ZONE,
        "origin_zone_production",
        "Systematic production-scale differences between origin zones.",
        "origin_zone",
    ),
)


def _strength(score: float, config: GravityRecommendationConfig) -> RecommendationStrength:
    if score >= config.strong_score_threshold:
        return "strong"
    if score >= config.moderate_score_threshold:
        return "moderate"
    if score >= config.weak_score_threshold:
        return "weak"
    return "none"


def _support(
    metadata: GravityValidationMetadata | None,
    grouping: str,
    measurements: int,
) -> tuple[int, int | None]:
    if metadata is None:
        return measurements, None
    labels = getattr(metadata, grouping)
    if labels is None:
        return measurements, None
    return int(labels.size), int(np.unique(labels).size)


def _score_atomic_candidate(
    *,
    scope: GravityEffectScope,
    name: str,
    discrepancy: str,
    grouping: str,
    result: GravityEstimationResult,
    problem: GravityObjectiveProblem,
    metadata: GravityValidationMetadata | None,
    adequacy_report: GravityAdequacyReport,
    config: GravityRecommendationConfig,
) -> GravityRelaxationRecommendation:
    parent = problem.parameter_layout
    support, support_groups = _support(metadata, grouping, problem.calibration_measurements)
    try:
        child_specification, info = add_gravity_relaxation(
            parent.specification, features=problem.features, scope=scope
        )
    except ValueError as error:
        return GravityRelaxationRecommendation(
            name,
            discrepancy,
            False,
            None,
            None,
            None,
            support,
            support_groups,
            None,
            "No child evaluation: the required user-supplied mapping is unavailable.",
            (str(error),),
            "none",
            f"{name} cannot be scored until its compact grouping map is supplied.",
        )
    child = GravityParameterLayout(child_specification, parent.positivity_floor)
    warm = warm_start_gravity_parameters(parent, child, result.raw_parameters)
    child_problem = replace(problem, parameter_layout=child)
    new_indices = np.asarray(
        [index for index, parameter_name in enumerate(child.names) if parameter_name not in parent.names],
        dtype=np.int64,
    )

    def objective(raw: jax.Array) -> jax.Array:
        return evaluate_gravity_objective(raw, problem=child_problem).objective

    raw = jnp.asarray(warm)
    gradient = np.asarray(jax.grad(objective)(raw), dtype=np.float64)[new_indices]
    hessian = np.asarray(jax.hessian(objective)(raw), dtype=np.float64)
    curvature = hessian[np.ix_(new_indices, new_indices)]
    curvature = 0.5 * (curvature + curvature.T)
    eigenvalues, eigenvectors = np.linalg.eigh(curvature)
    warnings: list[str] = []
    parent_gradient_norm = float(np.max(np.abs(result.gradient), initial=0.0))
    if parent_gradient_norm > config.parent_gradient_tolerance:
        warnings.append(
            f"The parent gradient infinity norm is {parent_gradient_norm:.3g}; local scores assume a converged optimum."
        )
    if eigenvalues[0] <= config.curvature_floor:
        warnings.append(
            "The local child curvature is non-positive or nearly singular; the gain uses a floored curvature."
        )
    stabilized = np.maximum(eigenvalues, config.curvature_floor)
    condition = float(stabilized[-1] / stabilized[0])
    if condition >= config.weak_identification_condition:
        warnings.append(f"The regularized child curvature condition number is {condition:.3g}.")
    if support / info.added_parameter_count < config.minimum_observations_per_parameter:
        warnings.append("Observation support per added parameter is weak.")
    inverse_gradient = eigenvectors @ ((eigenvectors.T @ gradient) / stabilized)
    gain = max(0.0, 0.5 * float(gradient @ inverse_gradient))
    statistic = 2.0 * gain
    strength = _strength(statistic, config)
    grouped_signal = max(
        (
            abs(item.mean_standardized_residual)
            for item in adequacy_report.grouped_summaries
            if item.grouping == grouping
        ),
        default=0.0,
    )
    if (
        grouped_signal >= config.grouped_residual_pattern_threshold
        and strength == "none"
    ):
        strength = "weak"
    if warnings and strength == "strong":
        strength = "moderate"
    explanation = (
        f"At the parent optimum, the {info.added_parameter_count}-parameter centered child "
        f"has a regularized quadratic gain of {gain:.4g} "
        f"(score statistic {statistic:.4g}); the largest absolute grouped mean "
        f"standardized residual is {grouped_signal:.4g}. This is advisory and the "
        "parent model is unchanged."
    )
    return GravityRelaxationRecommendation(
        name,
        discrepancy,
        True,
        info.added_parameter_count,
        statistic,
        gain,
        support,
        support_groups,
        statistic,
        f"{info.execution_impact} Scoring evaluates one gradient and one Hessian block.",
        tuple(warnings),
        strength,
        explanation,
    )


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 3 or np.std(first) == 0 or np.std(second) == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _future_feature_candidate(
    *,
    name: str,
    discrepancy: str,
    sensitivity: np.ndarray,
    standardized_residual: np.ndarray,
    config: GravityRecommendationConfig,
) -> GravityRelaxationRecommendation:
    correlation = _safe_correlation(sensitivity, standardized_residual)
    statistic = standardized_residual.size * correlation**2
    pattern = abs(correlation) >= config.feature_pattern_correlation_threshold
    return GravityRelaxationRecommendation(
        name,
        discrepancy,
        False,
        None,
        statistic,
        None,
        int(standardized_residual.size),
        None,
        None,
        "Not estimated: a centered child parameterization and grouping map must be selected first.",
        ("This candidate is diagnostic only and is not implemented as a Phase-5 child model.",),
        _strength(statistic, config) if pattern else "none",
        (
            f"The standardized residual correlation with the current {name} sensitivity is "
            f"{correlation:.3g}. "
            + ("This supports designing a child relaxation." if pattern else "No configured feature pattern was triggered.")
        ),
    )


def recommend_gravity_relaxations(
    *,
    result: GravityEstimationResult,
    problem: GravityObjectiveProblem,
    compact_layout: CompactODAssignmentLayout,
    adequacy_report: GravityAdequacyReport,
    metadata: GravityValidationMetadata | None = None,
    config: GravityRecommendationConfig = GravityRecommendationConfig(),
) -> GravityRelaxationRecommendationReport:
    """Score nested children locally without fitting or applying any candidate."""
    expected = gravity_model_fingerprint(problem, compact_layout)
    if result.model_fingerprint != expected:
        raise ValueError("gravity result and recommendation problem fingerprints differ.")
    if adequacy_report.model_fingerprint != expected:
        raise ValueError("adequacy report and recommendation problem fingerprints differ.")
    if metadata is not None and metadata.num_measurements != problem.observations.size:
        raise ValueError("recommendation metadata has the wrong measurement dimension.")
    candidates = [
        _score_atomic_candidate(
            scope=scope,
            name=name,
            discrepancy=discrepancy,
            grouping=grouping,
            result=result,
            problem=problem,
            metadata=metadata,
            adequacy_report=adequacy_report,
            config=config,
        )
        for scope, name, discrepancy, grouping in _ATOMIC_CANDIDATES
    ]
    raw = jnp.asarray(result.raw_parameters)
    mean_jacobian = np.asarray(
        jax.jacfwd(lambda value: predict_gravity_measurements(value, problem=problem)[0])(raw)
    )
    candidates.extend(
        (
            _future_feature_candidate(
                name="journey_time_impedance",
                discrepancy="Residual structure associated with journey-time impedance.",
                sensitivity=mean_jacobian[:, 0],
                standardized_residual=adequacy_report.standardized_nb_residual,
                config=config,
            ),
            _future_feature_candidate(
                name="transfer_penalty",
                discrepancy="Residual structure associated with transfer penalties.",
                sensitivity=mean_jacobian[:, 1],
                standardized_residual=adequacy_report.standardized_nb_residual,
                config=config,
            ),
        )
    )
    warnings: list[str] = []
    if adequacy_report.findings.possible_routing_or_timing_pattern:
        warnings.append(
            "Coherent residuals along vehicle journeys may indicate routing or timing error rather than demand complexity."
        )
    if adequacy_report.findings.isolated_suspect_observations:
        warnings.append(
            "Isolated extreme observations should be checked as possible data problems before relaxing demand."
        )
    payload = {
        "schema_version": 1,
        "model_fingerprint": expected,
        "adequacy_report_fingerprint": adequacy_report.report_fingerprint,
        "config": asdict(config),
        "candidates": [asdict(candidate) for candidate in candidates],
        "warnings": warnings,
    }
    return GravityRelaxationRecommendationReport(
        1,
        expected,
        adequacy_report.report_fingerprint,
        fingerprint(payload),
        tuple(candidates),
        tuple(warnings),
    )
