from __future__ import annotations

import pytest

from public_transportation.inference.complexity import build_od_parameter_complexity
from public_transportation.inference.od_parameter_layout import ODParameterLayout


def _layout(*, total: int, free: int) -> ODParameterLayout:
    free_indices = tuple(range(free))
    fixed_indices = tuple(range(free, total))
    return ODParameterLayout(
        num_od_total=total,
        od_keys=tuple((f"o{i}", f"d{i}", "t") for i in range(total)),
        free_od_indices=free_indices,
        fixed_od_indices=fixed_indices,
        fixed_od_values=(0.0,) * (total - free),
        free_baseline_values=(1.0,) * free,
        fixed_zero_indices=fixed_indices,
        fixed_positive_indices=(),
    )


@pytest.mark.parametrize("estimate_theta", [False, True])
@pytest.mark.parametrize("guide", ["auto_diag", "auto_normal", "auto_lowrank", "auto_mvn"])
def test_statistical_complexity_is_independent_of_total_frozen_cells(estimate_theta, guide):
    small_total = build_od_parameter_complexity(
        layout=_layout(total=20, free=10),
        estimate_theta=estimate_theta,
        guide=guide,
        lowrank_rank=20,
        compute_hessian=True,
    )
    large_total = build_od_parameter_complexity(
        layout=_layout(total=1000, free=10),
        estimate_theta=estimate_theta,
        guide=guide,
        lowrank_rank=20,
        compute_hessian=True,
    )

    assert small_total.num_fixed_od == 10
    assert large_total.num_fixed_od == 990
    assert small_total.num_fixed_zero_od == 10
    assert large_total.num_fixed_zero_od == 990
    assert small_total.num_fixed_positive_od == 0
    assert large_total.num_fixed_positive_od == 0
    assert small_total.statistical_dim == large_total.statistical_dim
    assert small_total.guide_parameter_count == large_total.guide_parameter_count
    assert small_total.optimizer_vector_size == large_total.optimizer_vector_size
    assert small_total.gradient_size == large_total.gradient_size
    assert small_total.hessian_element_count == large_total.hessian_element_count
    assert small_total.covariance_element_count == large_total.covariance_element_count
    # Frozen-zero cells are absent from the compact assignment vector.
    assert small_total.assignment_od_vector_size == 10
    assert large_total.assignment_od_vector_size == 10


def test_complexity_counts_for_ten_free_cells_and_estimated_theta():
    layout = _layout(total=1000, free=10)
    diagonal = build_od_parameter_complexity(
        layout=layout,
        estimate_theta=True,
        guide="auto_diag",
        compute_hessian=True,
    )
    lowrank = build_od_parameter_complexity(
        layout=layout,
        estimate_theta=True,
        guide="auto_lowrank",
        lowrank_rank=50,
        compute_hessian=True,
    )
    full = build_od_parameter_complexity(
        layout=layout,
        estimate_theta=True,
        guide="auto_mvn",
        compute_hessian=True,
    )

    assert diagonal.statistical_dim == 11
    assert diagonal.guide_parameter_count == 22
    assert diagonal.hessian_element_count == 121
    assert lowrank.effective_lowrank_rank == 11
    assert lowrank.guide_parameter_count == 143
    assert full.guide_parameter_count == 77  # 11 locations + 66 Cholesky entries


def test_all_frozen_fixed_theta_has_no_statistical_state():
    profile = build_od_parameter_complexity(
        layout=_layout(total=1000, free=0),
        estimate_theta=False,
        guide="auto_lowrank",
        lowrank_rank=20,
        compute_hessian=True,
    )
    assert profile.statistical_dim == 0
    assert profile.effective_lowrank_rank is None
    assert profile.guide_parameter_count == 0
    assert profile.optimizer_vector_size == 0
    assert profile.gradient_size == 0
    assert profile.hessian_element_count == 0
    assert profile.covariance_element_count == 0
    assert profile.assignment_od_vector_size == 0


def test_complexity_rejects_unknown_guide():
    with pytest.raises(ValueError, match="Unknown guide"):
        build_od_parameter_complexity(
            layout=_layout(total=2, free=1),
            estimate_theta=False,
            guide="unknown",  # type: ignore[arg-type]
        )
