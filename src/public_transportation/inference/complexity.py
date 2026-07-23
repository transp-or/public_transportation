"""Structural complexity accounting for reduced OD estimation problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from public_transportation.estimation.bayesian.guides import resolve_lowrank_rank
from public_transportation.inference.od_parameter_layout import ODParameterLayout


GuideName = Literal["auto_diag", "auto_lowrank", "auto_mvn", "auto_normal"]


@dataclass(frozen=True, slots=True)
class ODParameterComplexity:
    """Exact statistical state sizes implied by a reduced OD layout."""

    num_od_total: int
    num_free_od: int
    num_fixed_od: int
    num_fixed_zero_od: int
    num_fixed_positive_od: int
    estimate_theta: bool
    statistical_dim: int
    guide: GuideName
    effective_lowrank_rank: int | None
    guide_parameter_count: int
    optimizer_vector_size: int
    gradient_size: int
    hessian_element_count: int
    covariance_element_count: int
    assignment_od_vector_size: int


def build_od_parameter_complexity(
    *,
    layout: ODParameterLayout,
    estimate_theta: bool,
    guide: GuideName = "auto_diag",
    lowrank_rank: int | None = None,
    compute_hessian: bool = False,
) -> ODParameterComplexity:
    """Compute state sizes without constructing an estimator or assignment.

    Frozen cells affect only the reported full-vector counts. Every statistical
    state size below is a function of ``layout.num_free`` and optional theta.
    """
    dim = layout.parameter_dim(estimate_theta=estimate_theta)
    effective_rank: int | None = None
    if guide in ("auto_diag", "auto_normal"):
        guide_parameter_count = 2 * dim
    elif guide == "auto_lowrank":
        effective_rank = resolve_lowrank_rank(dim=dim, lowrank_rank=lowrank_rank)
        rank = 0 if effective_rank is None else effective_rank
        guide_parameter_count = 2 * dim + dim * rank
    elif guide == "auto_mvn":
        guide_parameter_count = dim + dim * (dim + 1) // 2
    else:
        raise ValueError(f"Unknown guide: {guide!r}")

    matrix_elements = dim * dim if compute_hessian else 0
    return ODParameterComplexity(
        num_od_total=layout.num_od_total,
        num_free_od=layout.num_free,
        num_fixed_od=layout.num_fixed,
        num_fixed_zero_od=layout.num_fixed_zero,
        num_fixed_positive_od=layout.num_fixed_positive,
        estimate_theta=bool(estimate_theta),
        statistical_dim=dim,
        guide=guide,
        effective_lowrank_rank=effective_rank,
        guide_parameter_count=guide_parameter_count,
        optimizer_vector_size=dim,
        gradient_size=dim,
        hessian_element_count=matrix_elements,
        covariance_element_count=matrix_elements,
        assignment_od_vector_size=layout.num_free + layout.num_fixed_positive,
    )
