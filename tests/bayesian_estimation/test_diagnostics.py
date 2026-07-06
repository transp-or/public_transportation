# tests/estimation/test_diagnostics.py
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from public_transportation.estimation.bayesian.diagnostics import (
    _describe_guide,
    _recent_window_size,
    _safe_float,
    compute_all_diagnostics,
    compute_optimization_diagnostics,
    compute_posterior_diagnostics,
)


def _mk_result(
    *,
    losses=None,
    samples=None,
    mean=None,
    sd=None,
    q05=None,
    q50=None,
    q95=None,
    guide="auto_diag",
    lowrank_rank=None,
):
    if losses is None:
        losses = np.asarray([10.0, 8.0, 6.0, 5.0], dtype=float)

    if mean is None:
        mean = np.asarray([1.0, -2.0, 0.5], dtype=float)
    else:
        mean = np.asarray(mean, dtype=float)

    if sd is None:
        sd = np.asarray([0.1, 0.5, 0.2], dtype=float)
    else:
        sd = np.asarray(sd, dtype=float)

    if q05 is None:
        q05 = mean - 1.0
    else:
        q05 = np.asarray(q05, dtype=float)

    if q50 is None:
        q50 = mean
    else:
        q50 = np.asarray(q50, dtype=float)

    if q95 is None:
        q95 = mean + 1.0
    else:
        q95 = np.asarray(q95, dtype=float)
    if samples is None:
        samples = np.zeros((100, len(mean)), dtype=float)

    return SimpleNamespace(
        losses=np.asarray(losses, dtype=float),
        posterior_samples_theta=np.asarray(samples, dtype=float),
        posterior_mean=np.asarray(mean, dtype=float),
        posterior_sd=np.asarray(sd, dtype=float),
        posterior_q05=np.asarray(q05, dtype=float),
        posterior_q50=np.asarray(q50, dtype=float),
        posterior_q95=np.asarray(q95, dtype=float),
        dim=len(mean),
        guide=guide,
        lowrank_rank=lowrank_rank,
        seed=123,
        num_steps=len(losses),
        learning_rate=0.01,
        num_posterior_draws=np.asarray(samples).shape[0],
        runtime_seconds=1.25,
        timestamp="2026-01-01T00:00:00",
        use_base_normal_correction=True,
    )


def test_safe_float_accepts_scalar_like_values():
    assert _safe_float(3) == pytest.approx(3.0)
    assert _safe_float(3.5) == pytest.approx(3.5)
    assert _safe_float(np.asarray(2.25)) == pytest.approx(2.25)


def test_recent_window_size_handles_empty_and_small_sequences():
    assert _recent_window_size(0) == 0
    assert _recent_window_size(5, fraction=0.1, minimum=20) == 5
    assert _recent_window_size(100, fraction=0.1, minimum=20) == 20
    assert _recent_window_size(1000, fraction=0.1, minimum=20) == 100


def test_optimization_diagnostics_empty_losses():
    result = _mk_result(losses=[])

    diagnostics = compute_optimization_diagnostics(result)

    assert diagnostics["num_steps"] == 0
    assert diagnostics["has_losses"] is False
    assert diagnostics["all_finite"] is True
    assert diagnostics["contains_nan"] is False
    assert diagnostics["contains_inf"] is False
    assert diagnostics["converged"] is False
    assert "No loss values" in diagnostics["message"]


def test_optimization_diagnostics_basic_decreasing_losses():
    result = _mk_result(losses=[10.0, 8.0, 6.0, 5.0])

    diagnostics = compute_optimization_diagnostics(
        result,
        recent_fraction=0.5,
        minimum_recent_window=2,
    )

    assert diagnostics["num_steps"] == 4
    assert diagnostics["has_losses"] is True
    assert diagnostics["all_finite"] is True
    assert diagnostics["contains_nan"] is False
    assert diagnostics["contains_inf"] is False
    assert diagnostics["initial_loss"] == pytest.approx(10.0)
    assert diagnostics["final_loss"] == pytest.approx(5.0)
    assert diagnostics["best_loss"] == pytest.approx(5.0)
    assert diagnostics["best_loss_step"] == 3
    assert diagnostics["absolute_improvement"] == pytest.approx(5.0)
    assert diagnostics["relative_improvement"] == pytest.approx(0.5)
    assert diagnostics["recent_window_size"] == 2
    assert diagnostics["recent_mean"] == pytest.approx(5.5)
    assert diagnostics["recent_min"] == pytest.approx(5.0)
    assert diagnostics["recent_max"] == pytest.approx(6.0)
    assert diagnostics["monotonic_fraction"] == pytest.approx(1.0)


def test_optimization_diagnostics_detects_nonfinite_losses():
    result = _mk_result(losses=[10.0, np.nan, np.inf, 5.0])

    diagnostics = compute_optimization_diagnostics(result)

    assert diagnostics["all_finite"] is False
    assert diagnostics["contains_nan"] is True
    assert diagnostics["contains_inf"] is True
    assert diagnostics["converged"] is False
    assert "NaN or infinite" in diagnostics["message"]


def test_optimization_diagnostics_converged_when_recent_window_stable():
    result = _mk_result(losses=[5.0, 4.0, 3.0, 2.0001, 2.00005, 2.00004])

    diagnostics = compute_optimization_diagnostics(
        result,
        recent_fraction=0.5,
        minimum_recent_window=3,
        convergence_rtol=1e-3,
        oscillation_ratio_threshold=1e-3,
    )

    assert diagnostics["all_finite"] is True
    assert diagnostics["converged"] is True
    assert "stabilized" in diagnostics["message"]


def test_optimization_diagnostics_not_converged_when_recent_losses_still_change():
    result = _mk_result(losses=[10.0, 8.0, 6.0, 4.0, 2.0])

    diagnostics = compute_optimization_diagnostics(
        result,
        recent_fraction=0.4,
        minimum_recent_window=2,
        convergence_rtol=1e-6,
        oscillation_ratio_threshold=1e-6,
    )

    assert diagnostics["converged"] is False
    assert "uncertain" in diagnostics["message"]


def test_optimization_diagnostics_monotonic_fraction_with_increases():
    result = _mk_result(losses=[10.0, 8.0, 9.0, 7.0, 7.5])

    diagnostics = compute_optimization_diagnostics(
        result,
        recent_fraction=0.4,
        minimum_recent_window=2,
    )

    # Differences: -2, +1, -2, +0.5. Two out of four are non-positive.
    assert diagnostics["monotonic_fraction"] == pytest.approx(0.5)


def test_posterior_diagnostics_basic_summary_and_top_uncertain():
    result = _mk_result(
        mean=[1.0, -2.0, 0.5],
        sd=[0.1, 0.5, 0.2],
        q05=[0.5, -3.0, -0.1],
        q50=[1.0, -2.0, 0.5],
        q95=[1.5, -1.0, 1.1],
        samples=np.zeros((50, 3)),
        guide="auto_diag",
    )

    diagnostics = compute_posterior_diagnostics(
        result,
        top_k=2,
        parameter_names=["alpha", "beta", "gamma"],
    )

    assert diagnostics["guide"] == "auto_diag"
    assert diagnostics["dim"] == 3
    assert diagnostics["num_draws"] == 50
    assert diagnostics["average_sd"] == pytest.approx(np.mean([0.1, 0.5, 0.2]))
    assert diagnostics["median_sd"] == pytest.approx(0.2)
    assert diagnostics["min_sd"] == pytest.approx(0.1)
    assert diagnostics["max_sd"] == pytest.approx(0.5)
    assert diagnostics["average_interval_width_90"] == pytest.approx(np.mean([1.0, 2.0, 1.2]))
    assert diagnostics["median_interval_width_90"] == pytest.approx(1.2)
    assert diagnostics["fraction_90_intervals_containing_zero"] == pytest.approx(1.0 / 3.0)

    top = diagnostics["top_uncertain_parameters"]
    assert len(top) == 2
    assert top[0]["index"] == 1
    assert top[0]["name"] == "beta"
    assert top[0]["sd"] == pytest.approx(0.5)
    assert top[1]["index"] == 2
    assert top[1]["name"] == "gamma"


def test_posterior_diagnostics_uses_default_parameter_names_when_missing():
    result = _mk_result(mean=[1.0, 2.0], sd=[0.1, 0.2])

    diagnostics = compute_posterior_diagnostics(result, top_k=5)

    assert len(diagnostics["top_uncertain_parameters"]) == 2
    assert diagnostics["top_uncertain_parameters"][0]["name"] == "param_1"
    assert diagnostics["top_uncertain_parameters"][1]["name"] == "param_0"


def test_posterior_diagnostics_top_k_is_capped_by_dimension():
    result = _mk_result(mean=[1.0, 2.0], sd=[0.1, 0.2])

    diagnostics = compute_posterior_diagnostics(result, top_k=10)

    assert len(diagnostics["top_uncertain_parameters"]) == 2


@pytest.mark.parametrize(
    ("guide", "expected_fragment"),
    [
        ("auto_diag", "diagonal Gaussian"),
        ("auto_normal", "independent normal"),
        ("auto_lowrank", "low-rank Gaussian"),
        ("auto_mvn", "full-covariance Gaussian"),
        ("custom_guide", "automatic guide"),
    ],
)
def test_posterior_diagnostics_guide_specific_warning(guide, expected_fragment):
    result = _mk_result(guide=guide, lowrank_rank=2)

    diagnostics = compute_posterior_diagnostics(result)

    assert expected_fragment in diagnostics["approximation_warning"]


@pytest.mark.parametrize(
    ("guide", "rank", "expected"),
    [
        ("auto_diag", None, "Mean-field Gaussian"),
        ("auto_normal", None, "Automatic Normal guide"),
        ("auto_lowrank", None, "Low-rank multivariate Gaussian"),
        ("auto_lowrank", 3, "with rank 3"),
        ("auto_mvn", None, "Full-covariance multivariate Gaussian"),
        ("something_else", None, "Automatic guide 'something_else'"),
    ],
)
def test_describe_guide(guide, rank, expected):
    assert expected in _describe_guide(guide, rank)


def test_compute_all_diagnostics_combines_sections_and_metadata():
    result = _mk_result(
        losses=[3.0, 2.0, 1.0],
        guide="auto_lowrank",
        lowrank_rank=2,
        samples=np.zeros((25, 3)),
    )

    diagnostics = compute_all_diagnostics(
        result,
        top_k=2,
        parameter_names=["a", "b", "c"],
        recent_fraction=0.5,
        minimum_recent_window=2,
    )

    assert set(diagnostics) == {"metadata", "optimization", "posterior"}

    metadata = diagnostics["metadata"]
    assert metadata["guide"] == "auto_lowrank"
    assert metadata["dim"] == 3
    assert metadata["seed"] == 123
    assert metadata["num_steps"] == 3
    assert metadata["learning_rate"] == pytest.approx(0.01)
    assert metadata["lowrank_rank"] == 2
    assert metadata["num_posterior_draws"] == 25
    assert metadata["runtime_seconds"] == pytest.approx(1.25)
    assert metadata["timestamp"] == "2026-01-01T00:00:00"
    assert metadata["use_base_normal_correction"] is True

    assert diagnostics["optimization"]["num_steps"] == 3
    assert diagnostics["posterior"]["guide"] == "auto_lowrank"
    assert len(diagnostics["posterior"]["top_uncertain_parameters"]) == 2


def test_posterior_diagnostics_handles_one_dimensional_samples():
    result = _mk_result(
        mean=[1.0],
        sd=[0.3],
        q05=[0.1],
        q50=[1.0],
        q95=[1.9],
        samples=np.asarray([0.9, 1.0, 1.1]),
    )

    diagnostics = compute_posterior_diagnostics(result)

    assert diagnostics["dim"] == 1
    assert diagnostics["num_draws"] == 3
    assert diagnostics["average_sd"] == pytest.approx(0.3)
    assert len(diagnostics["top_uncertain_parameters"]) == 1