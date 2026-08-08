"""Transparent data-support and model-assumption diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np

SupportLabel = Literal[
    "data_supported", "mixed", "assumption_dominated", "unidentified"
]


@dataclass(frozen=True, slots=True)
class ParameterBlockEvidence:
    block: str
    parameters: int
    associated_observations: int
    sensitivity_norm: float
    gradient_norm: float
    curvature_minimum: float | None
    penalty_contribution: float
    likelihood_improvement: float | None
    held_out_improvement: float | None
    on_bound: bool
    support: SupportLabel
    explanation: str


@dataclass(frozen=True, slots=True)
class EvidenceAssumptionReport:
    observations_used: int
    measurement_types: tuple[str, ...]
    periods: tuple[str, ...]
    blocks: tuple[ParameterBlockEvidence, ...]
    assumptions: tuple[str, ...]
    structural_zero_count: int
    observed_zero_count: int
    expected_zero_count: float
    excess_zero_count: float


def build_evidence_assumption_report(
    *,
    block_jacobians: Mapping[str, object],
    block_gradients: Mapping[str, object],
    block_penalties: Mapping[str, float],
    observations: object,
    expected_zero_probabilities: object,
    measurement_types: Sequence[str] = (),
    periods: Sequence[str] = (),
    likelihood_improvements: Mapping[str, float] = {},
    held_out_improvements: Mapping[str, float] = {},
    structural_zero_count: int = 0,
    assumptions: Sequence[str] = (),
) -> EvidenceAssumptionReport:
    observed = np.asarray(observations, dtype=np.float64)
    zero = np.asarray(expected_zero_probabilities, dtype=np.float64)
    if (
        observed.ndim != 1
        or zero.shape != observed.shape
        or np.any(observed < 0)
        or np.any((zero < 0) | (zero > 1))
    ):
        raise ValueError(
            "observations and zero probabilities must be aligned and valid."
        )
    blocks: list[ParameterBlockEvidence] = []
    for name in sorted(block_jacobians):
        jacobian = np.asarray(block_jacobians[name], dtype=np.float64)
        gradient = np.asarray(block_gradients.get(name, ()), dtype=np.float64)
        parameters = jacobian.shape[1] if jacobian.ndim == 2 else gradient.size
        sensitivity = float(np.linalg.norm(jacobian))
        gradient_norm = float(np.linalg.norm(gradient))
        associated = (
            int(np.count_nonzero(np.any(np.abs(jacobian) > 1.0e-12, axis=1)))
            if jacobian.ndim == 2
            else 0
        )
        penalty = float(block_penalties.get(name, 0.0))
        if parameters == 0 or sensitivity <= 1.0e-12:
            support: SupportLabel = "unidentified"
            explanation = "The observation response is insensitive to this block."
        elif associated < parameters:
            support = "assumption_dominated"
            explanation = "Fewer sensitive observations than parameters make regularization decisive."
        elif penalty > max(abs(float(likelihood_improvements.get(name, 0.0))), 1.0e-12):
            support = "mixed"
            explanation = "Both likelihood information and the declared penalty materially control this block."
        else:
            support = "data_supported"
            explanation = (
                "The block has nonzero sensitivity across enough retained observations."
            )
        blocks.append(
            ParameterBlockEvidence(
                name,
                parameters,
                associated,
                sensitivity,
                gradient_norm,
                None,
                penalty,
                likelihood_improvements.get(name),
                held_out_improvements.get(name),
                False,
                support,
                explanation,
            )
        )
    expected = float(np.sum(zero))
    observed_zeros = int(np.count_nonzero(observed == 0))
    return EvidenceAssumptionReport(
        len(observed),
        tuple(sorted(set(measurement_types))),
        tuple(sorted(set(periods))),
        tuple(blocks),
        tuple(assumptions),
        structural_zero_count,
        observed_zeros,
        expected,
        observed_zeros - expected,
    )


@dataclass(frozen=True, slots=True)
class DemandRelaxationRecommendation:
    code: str
    severity: Literal["info", "warning", "strong_warning"]
    evidence: Mapping[str, float]
    suggestion: str


def recommend_demand_relaxations(
    *,
    period_residual_ratio: float = 0.0,
    origin_group_residual_ratio: float = 0.0,
    destination_group_residual_ratio: float = 0.0,
    od_pattern_residual_ratio: float = 0.0,
    variance_to_mean_ratio: float = 1.0,
    excess_zero_fraction_after_nb: float = 0.0,
    held_out_improvement: float | None = None,
) -> tuple[DemandRelaxationRecommendation, ...]:
    recommendations: list[DemandRelaxationRecommendation] = []
    checks = (
        (
            period_residual_ratio,
            0.1,
            "period_effects",
            "Add period-specific production or impedance coefficients.",
        ),
        (
            origin_group_residual_ratio,
            0.1,
            "origin_group_effects",
            "Add origin-group production effects.",
        ),
        (
            destination_group_residual_ratio,
            0.1,
            "destination_group_effects",
            "Add destination-group attraction effects.",
        ),
        (
            od_pattern_residual_ratio,
            0.1,
            "low_rank_interaction",
            "Test rank one after margin effects have been assessed.",
        ),
        (
            variance_to_mean_ratio,
            1.5,
            "negative_binomial",
            "Compare negative binomial before zero inflation.",
        ),
        (
            excess_zero_fraction_after_nb,
            0.05,
            "zero_inflation_sensitivity",
            "Test ZIP or ZINB only as a separate zero-process sensitivity.",
        ),
    )
    for value, threshold, code, suggestion in checks:
        if value > threshold:
            recommendations.append(
                DemandRelaxationRecommendation(
                    code,
                    "warning",
                    {"value": value, "threshold": threshold},
                    suggestion,
                )
            )
    if held_out_improvement is not None and held_out_improvement <= 0.0:
        recommendations.append(
            DemandRelaxationRecommendation(
                "stop_complexity_growth",
                "strong_warning",
                {"held_out_improvement": held_out_improvement},
                "Do not add complexity without held-out predictive improvement.",
            )
        )
    return tuple(recommendations)
