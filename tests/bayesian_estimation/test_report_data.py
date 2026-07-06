from __future__ import annotations

from typing import Any

import pytest

from public_transportation.estimation.bayesian.report_data import (
    _build_executive_summary,
    _build_optimization_section,
    _build_posterior_section,
    _build_recommendations,
    _fmt_float,
    _yes_no,
    build_vi_report_data,
)


def _diagnostics(
    *,
    guide: str = "auto_diag",
    converged: bool = True,
    all_finite: bool = True,
    contains_nan: bool = False,
    contains_inf: bool = False,
    oscillation_ratio: float = 0.01,
    recent_relative_change: float = 1e-4,
    fraction_zero: float = 0.25,
    max_sd: float = 2.0,
    median_sd: float = 1.0,
    lowrank_rank: int | None = None,
    dim: int = 10,
) -> dict[str, Any]:
    return {
        "metadata": {
            "guide": guide,
            "dim": dim,
            "seed": 123,
            "num_steps": 1000,
            "learning_rate": 0.01,
            "lowrank_rank": lowrank_rank,
            "num_posterior_draws": 500,
            "runtime_seconds": 12.34567,
            "timestamp": "2026-01-01T12:00:00",
            "use_base_normal_correction": True,
        },
        "optimization": {
            "converged": converged,
            "all_finite": all_finite,
            "contains_nan": contains_nan,
            "contains_inf": contains_inf,
            "has_losses": True,
            "message": "Optimization stabilized.",
            "num_steps": 1000,
            "initial_loss": 100.0,
            "final_loss": 10.0,
            "best_loss": 9.5,
            "best_loss_step": 900,
            "absolute_improvement": 90.0,
            "relative_improvement": 0.9,
            "recent_window_size": 100,
            "recent_mean": 10.1,
            "recent_std": 0.05,
            "recent_relative_change": recent_relative_change,
            "oscillation_ratio": oscillation_ratio,
            "monotonic_fraction": 0.8,
        },
        "posterior": {
            "guide": guide,
            "dim": dim,
            "num_draws": 500,
            "posterior_family_description": "Diagonal Gaussian variational posterior.",
            "approximation_warning": "This approximation ignores posterior correlations.",
            "average_sd": 1.2,
            "median_sd": median_sd,
            "min_sd": 0.2,
            "max_sd": max_sd,
            "average_interval_width_90": 3.4,
            "median_interval_width_90": 3.0,
            "fraction_90_intervals_containing_zero": fraction_zero,
            "top_uncertain_parameters": [
                {
                    "name": "beta_time",
                    "mean": -1.0,
                    "sd": 2.0,
                    "q05": -4.0,
                    "median": -1.0,
                    "q95": 2.0,
                    "interval_width": 6.0,
                    "contains_zero": True,
                },
                {
                    "name": "beta_cost",
                    "mean": -0.5,
                    "sd": 1.0,
                    "q05": -2.0,
                    "median": -0.5,
                    "q95": 1.0,
                    "interval_width": 3.0,
                    "contains_zero": True,
                },
            ],
        },
    }


def test_build_vi_report_data_uses_precomputed_diagnostics():
    diagnostics = _diagnostics()
    figures = {
        "loss_curve": "loss.png",
        "loss_curve_recent": "loss_recent.png",
        "posterior_sd_rank": "sd_rank.png",
        "posterior_intervals_top": "intervals.png",
    }

    report = build_vi_report_data(
        result=object(),
        diagnostics=diagnostics,
        figure_files=figures,
    )

    assert report["title"] == "Variational Inference Diagnostic Report"
    assert report["subtitle"] == "Guide: auto_diag | Dimension: 10"
    assert len(report["metadata"]) == 10
    assert report["metadata"][0] == ("Guide", "auto_diag")
    assert report["metadata"][1] == ("Dimension", 10)
    assert report["metadata"][7] == ("Runtime (seconds)", 12.346)
    assert report["metadata"][9] == ("Base normal correction", "Yes")

    assert len(report["executive_summary"]) == 4
    assert len(report["recommendations"]) >= 1
    assert [section["id"] for section in report["sections"]] == [
        "optimization",
        "posterior",
    ]

    optimization_section = report["sections"][0]
    posterior_section = report["sections"][1]

    assert optimization_section["figures"][0]["file"] == "loss.png"
    assert optimization_section["figures"][1]["file"] == "loss_recent.png"
    assert posterior_section["figures"][0]["file"] == "sd_rank.png"
    assert posterior_section["figures"][1]["file"] == "intervals.png"


def test_build_vi_report_data_defaults_figure_files_to_none_entries():
    report = build_vi_report_data(
        result=object(),
        diagnostics=_diagnostics(),
        figure_files=None,
    )

    figures = report["sections"][0]["figures"] + report["sections"][1]["figures"]
    assert all(fig["file"] is None for fig in figures)


def test_build_vi_report_data_computes_diagnostics_when_omitted(monkeypatch):
    diagnostics = _diagnostics()
    seen = {}

    def _compute_all_diagnostics(result):
        seen["result"] = result
        return diagnostics

    monkeypatch.setattr(
        "public_transportation.estimation.bayesian.report_data.compute_all_diagnostics",
        _compute_all_diagnostics,
    )

    result = object()
    report = build_vi_report_data(result=result)

    assert seen["result"] is result
    assert report["subtitle"] == "Guide: auto_diag | Dimension: 10"


def test_executive_summary_for_converged_finite_run():
    summary = _build_executive_summary(
        optimization=_diagnostics()["optimization"],
        posterior=_diagnostics()["posterior"],
    )

    assert "appears to have converged" in summary[0]
    assert "No NaN or infinite values" in summary[1]
    assert summary[2] == "Diagonal Gaussian variational posterior."
    assert summary[3] == "This approximation ignores posterior correlations."


def test_executive_summary_for_unstable_run():
    diagnostics = _diagnostics(
        converged=False,
        all_finite=False,
        contains_nan=True,
    )

    summary = _build_executive_summary(
        optimization=diagnostics["optimization"],
        posterior=diagnostics["posterior"],
    )

    assert "does not appear fully stabilized" in summary[0]
    assert "contains non-finite values" in summary[1]


def test_optimization_section_contains_expected_table_rows_and_figures():
    diagnostics = _diagnostics()
    section = _build_optimization_section(
        diagnostics["optimization"],
        {
            "loss_curve": "loss.png",
            "loss_curve_recent": "recent.png",
        },
    )

    assert section["id"] == "optimization"
    assert section["title"] == "Optimization diagnostics"
    assert section["summary"] == "Optimization stabilized."
    assert len(section["interpretation"]) == 4

    table = section["tables"][0]
    assert table["title"] == "Optimization summary"
    assert table["columns"] == ["Metric", "Value"]

    row_names = [row[0] for row in table["rows"]]
    assert "Number of steps" in row_names
    assert "All losses finite" in row_names
    assert "Converged" in row_names

    assert section["figures"][0]["id"] == "loss_curve"
    assert section["figures"][0]["file"] == "loss.png"
    assert section["figures"][1]["id"] == "loss_curve_recent"
    assert section["figures"][1]["file"] == "recent.png"


def test_optimization_section_without_losses_has_short_interpretation():
    diagnostics = _diagnostics()
    diagnostics["optimization"]["has_losses"] = False

    section = _build_optimization_section(diagnostics["optimization"], {})

    assert section["interpretation"] == ["Optimization stabilized."]


def test_posterior_section_contains_expected_tables_and_figures():
    diagnostics = _diagnostics()
    section = _build_posterior_section(
        diagnostics["posterior"],
        {
            "posterior_sd_rank": "sd.png",
            "posterior_intervals_top": "intervals.png",
        },
    )

    assert section["id"] == "posterior"
    assert section["title"] == "Variational posterior diagnostics"
    assert section["summary"] == "This approximation ignores posterior correlations."

    posterior_summary = section["tables"][0]
    uncertain_table = section["tables"][1]

    assert posterior_summary["title"] == "Posterior summary"
    assert ["Guide", "auto_diag"] in posterior_summary["rows"]
    assert ["Dimension", 10] in posterior_summary["rows"]
    assert ["Posterior draws", 500] in posterior_summary["rows"]

    assert uncertain_table["title"] == "Most uncertain parameters"
    assert uncertain_table["columns"] == [
        "Parameter",
        "Mean",
        "SD",
        "Q05",
        "Median",
        "Q95",
        "Interval width",
        "Contains zero",
    ]
    assert uncertain_table["rows"][0][0] == "beta_time"
    assert uncertain_table["rows"][0][-1] == "Yes"

    assert section["figures"][0]["file"] == "sd.png"
    assert section["figures"][1]["file"] == "intervals.png"


def test_recommendations_detect_numerical_instability():
    diagnostics = _diagnostics(
        all_finite=False,
        contains_nan=True,
        converged=False,
    )

    recommendations = _build_recommendations(
        metadata=diagnostics["metadata"],
        optimization=diagnostics["optimization"],
        posterior=diagnostics["posterior"],
    )

    titles = [item["title"] for item in recommendations]
    severities = {item["title"]: item["severity"] for item in recommendations}

    assert "Numerical instability detected" in titles
    assert severities["Numerical instability detected"] == "critical"
    assert "Optimization may not have converged" in titles


def test_recommendations_detect_non_convergence():
    diagnostics = _diagnostics(converged=False)

    recommendations = _build_recommendations(
        metadata=diagnostics["metadata"],
        optimization=diagnostics["optimization"],
        posterior=diagnostics["posterior"],
    )

    assert any(
        item["title"] == "Optimization may not have converged"
        and item["severity"] == "warning"
        for item in recommendations
    )


def test_recommendations_detect_large_oscillations():
    diagnostics = _diagnostics(oscillation_ratio=0.2)

    recommendations = _build_recommendations(
        metadata=diagnostics["metadata"],
        optimization=diagnostics["optimization"],
        posterior=diagnostics["posterior"],
    )

    assert any(item["title"] == "Loss oscillations are non-negligible" for item in recommendations)


def test_recommendations_detect_recent_drift():
    diagnostics = _diagnostics(recent_relative_change=0.02, all_finite=True)

    recommendations = _build_recommendations(
        metadata=diagnostics["metadata"],
        optimization=diagnostics["optimization"],
        posterior=diagnostics["posterior"],
    )

    assert any(item["title"] == "Loss is still drifting near the end of optimization" for item in recommendations)


@pytest.mark.parametrize(
    ("guide", "expected_title"),
    [
        ("auto_diag", "Check whether posterior correlation matters"),
        ("auto_normal", "Consider a richer posterior family"),
    ],
)
def test_recommendations_for_independent_guides(guide: str, expected_title: str):
    diagnostics = _diagnostics(guide=guide)
    diagnostics["metadata"]["guide"] = guide
    diagnostics["posterior"]["guide"] = guide

    recommendations = _build_recommendations(
        metadata=diagnostics["metadata"],
        optimization=diagnostics["optimization"],
        posterior=diagnostics["posterior"],
    )

    assert any(item["title"] == expected_title for item in recommendations)


def test_recommendations_for_lowrank_without_rank():
    diagnostics = _diagnostics(guide="auto_lowrank", lowrank_rank=None)

    recommendations = _build_recommendations(
        metadata=diagnostics["metadata"],
        optimization=diagnostics["optimization"],
        posterior=diagnostics["posterior"],
    )

    assert any(item["title"] == "Document the low-rank structure" for item in recommendations)


def test_recommendations_for_lowrank_small_rank():
    diagnostics = _diagnostics(
        guide="auto_lowrank",
        lowrank_rank=2,
        dim=100,
    )

    recommendations = _build_recommendations(
        metadata=diagnostics["metadata"],
        optimization=diagnostics["optimization"],
        posterior=diagnostics["posterior"],
    )

    assert any(item["title"] == "Low-rank approximation may be restrictive" for item in recommendations)


def test_recommendations_for_high_dimensional_full_mvn():
    diagnostics = _diagnostics(guide="auto_mvn", dim=201)

    recommendations = _build_recommendations(
        metadata=diagnostics["metadata"],
        optimization=diagnostics["optimization"],
        posterior=diagnostics["posterior"],
    )

    assert any(item["title"] == "Full covariance may be expensive in high dimension" for item in recommendations)


def test_recommendations_for_many_intervals_containing_zero():
    diagnostics = _diagnostics(fraction_zero=0.95)

    recommendations = _build_recommendations(
        metadata=diagnostics["metadata"],
        optimization=diagnostics["optimization"],
        posterior=diagnostics["posterior"],
    )

    assert any(item["title"] == "Most intervals contain zero" for item in recommendations)


def test_recommendations_for_highly_uneven_uncertainty():
    diagnostics = _diagnostics(max_sd=20.0, median_sd=1.0)

    recommendations = _build_recommendations(
        metadata=diagnostics["metadata"],
        optimization=diagnostics["optimization"],
        posterior=diagnostics["posterior"],
    )

    assert any(item["title"] == "Some parameters are much more uncertain than the others" for item in recommendations)


def test_recommendations_always_include_general_robustness_advice():
    diagnostics = _diagnostics()

    recommendations = _build_recommendations(
        metadata=diagnostics["metadata"],
        optimization=diagnostics["optimization"],
        posterior=diagnostics["posterior"],
    )

    titles = [item["title"] for item in recommendations]
    assert "Assess approximation quality beyond the ELBO" in titles
    assert "Check robustness across independent runs" in titles


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "Yes"),
        (False, "No"),
    ],
)
def test_yes_no(value: bool, expected: str):
    assert _yes_no(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ""),
        (1.23456789, "1.23457"),
        (1000000.0, "1e+06"),
        ("abc", "abc"),
    ],
)
def test_fmt_float(value: Any, expected: str):
    assert _fmt_float(value) == expected