"""Structured regularization recommendations without implicit model changes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .fixed_routing_linear_problem import FixedRoutingLinearProblem
from .fixed_routing_linear_quality import (
    LinearEstimateQuality,
    analyze_linear_estimate_quality,
)

RecommendationStatus = Literal[
    "none_is_reasonable",
    "regularization_recommended",
    "regularization_required_for_uniqueness",
    "scaling_recommended",
    "bounds_are_dominant",
]
RegularizationOptionName = Literal[
    "none",
    "ridge_to_prior",
    "scaled_ridge_to_prior",
]


@dataclass(frozen=True, slots=True)
class RegularizationOptionRecommendation:
    name: RegularizationOptionName
    recommended: bool
    requires_strength: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RegularizationRecommendation:
    status: RecommendationStatus
    reasons: tuple[str, ...]
    options: tuple[RegularizationOptionRecommendation, ...]
    quality: LinearEstimateQuality
    automatic_selection_applied: bool = False

    @property
    def recommended_options(self) -> tuple[RegularizationOptionName, ...]:
        return tuple(option.name for option in self.options if option.recommended)


class RegularizationSelectionRequiredError(ValueError):
    """Raised when estimation is requested before an explicit selection."""

    def __init__(self, recommendation: RegularizationRecommendation):
        self.recommendation = recommendation
        choices = ", ".join(recommendation.recommended_options)
        super().__init__(
            "Regularization selection is required before estimation. "
            f"Diagnostic status: {recommendation.status}. "
            f"Recommended options: {choices or 'review all explicit options'}."
        )


def recommend_linear_regularization(
    problem: FixedRoutingLinearProblem,
    demand: object | None = None,
    *,
    condition_threshold: float = 1.0e8,
    scale_ratio_threshold: float = 100.0,
    bound_dominance_fraction: float = 0.5,
) -> RegularizationRecommendation:
    """Analyze a small problem and return explicit, non-binding choices."""
    if not math.isfinite(condition_threshold) or condition_threshold <= 1.0:
        raise ValueError("condition_threshold must be finite and greater than one.")
    if not math.isfinite(scale_ratio_threshold) or scale_ratio_threshold <= 1.0:
        raise ValueError("scale_ratio_threshold must be finite and greater than one.")
    if not 0.0 <= bound_dominance_fraction <= 1.0:
        raise ValueError("bound_dominance_fraction must lie between zero and one.")

    analysis_demand = problem.prior_demand if demand is None else demand
    quality = analyze_linear_estimate_quality(
        problem,
        analysis_demand,
        exclude_active_bounds=demand is not None,
    )
    active_count = int(
        np.count_nonzero(
            quality.kkt.lower_active
            | quality.kkt.upper_active
            | quality.kkt.fixed_by_bounds
        )
    )
    active_fraction = active_count / problem.num_free_od

    positive_prior = problem.prior_demand[problem.prior_demand > 0.0]
    has_zero_and_positive_prior = bool(
        positive_prior.size and np.any(problem.prior_demand == 0.0)
    )
    scale_ratio = (
        1.0
        if positive_prior.size < 2
        else float(np.max(positive_prior) / np.min(positive_prior))
    )
    heterogeneous_scale = (
        has_zero_and_positive_prior or scale_ratio >= scale_ratio_threshold
    )
    ill_conditioned = (
        math.isinf(quality.measurement_condition_estimate)
        or (
            math.isfinite(quality.measurement_condition_estimate)
            and quality.measurement_condition_estimate >= condition_threshold
        )
    )

    reasons: list[str] = []
    if demand is not None and active_fraction >= bound_dominance_fraction:
        status: RecommendationStatus = "bounds_are_dominant"
        reasons.append(
            f"{active_count} of {problem.num_free_od} OD variables are active at bounds; "
            "interpret resolution scores on the remaining free set."
        )
    elif quality.measurement_nullity > 0:
        status = "regularization_required_for_uniqueness"
        reasons.append(
            f"The weighted measurement operator has nullity "
            f"{quality.measurement_nullity} on the current free set."
        )
    elif ill_conditioned:
        status = "regularization_recommended"
        reasons.append(
            "The weighted measurement operator is poorly conditioned "
            f"(condition estimate {quality.measurement_condition_estimate:.6g})."
        )
    elif heterogeneous_scale:
        status = "scaling_recommended"
        reasons.append(
            "Prior OD magnitudes are heterogeneous; penalizing standardized "
            "deviations is more comparable across cells."
        )
    else:
        status = "none_is_reasonable"
        reasons.append(
            "The weighted measurement operator is full column rank and its "
            "condition estimate is below the configured threshold."
        )

    needs_stabilization = status in {
        "regularization_required_for_uniqueness",
        "regularization_recommended",
    }
    prefer_scaled = heterogeneous_scale and status != "none_is_reasonable"
    none_reasons = (
        "Leaves the data objective unchanged and exposes any nonuniqueness directly.",
    )
    ridge_reasons = (
        "Anchors weakly observed directions to the prior OD demand.",
    )
    scaled_reasons = (
        "Anchors standardized deviations and supports zero prior entries through "
        "positive user-supplied scales.",
    )
    options = (
        RegularizationOptionRecommendation(
            name="none",
            recommended=status == "none_is_reasonable",
            requires_strength=False,
            reasons=none_reasons,
        ),
        RegularizationOptionRecommendation(
            name="ridge_to_prior",
            recommended=needs_stabilization and not prefer_scaled,
            requires_strength=True,
            reasons=ridge_reasons,
        ),
        RegularizationOptionRecommendation(
            name="scaled_ridge_to_prior",
            recommended=(needs_stabilization and prefer_scaled)
            or status == "scaling_recommended",
            requires_strength=True,
            reasons=scaled_reasons,
        ),
    )
    return RegularizationRecommendation(
        status=status,
        reasons=tuple(reasons),
        options=options,
        quality=quality,
        automatic_selection_applied=False,
    )


def require_explicit_regularization_selection(
    problem: FixedRoutingLinearProblem,
    demand: object | None = None,
) -> FixedRoutingLinearProblem:
    """Return an explicitly configured problem or raise with recommendations."""
    if problem.regularization_selection != "unspecified":
        return problem
    raise RegularizationSelectionRequiredError(
        recommend_linear_regularization(problem, demand)
    )
