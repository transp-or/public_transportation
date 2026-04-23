from __future__ import annotations

from typing import Any

from .diagnostics import compute_all_diagnostics
from .results import VIResult


def build_vi_report_data(
    result: VIResult,
    diagnostics: dict[str, Any] | None = None,
    figure_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Build a structured, human-readable report data dictionary from a VIResult.

    Parameters
    ----------
    result:
        Variational inference result.
    diagnostics:
        Optional precomputed diagnostics. If omitted, they are computed internally.

    Returns
    -------
    dict
        Structured report content ready for HTML rendering.
    """
    if diagnostics is None:
        diagnostics = compute_all_diagnostics(result)
    if figure_files is None:
        figure_files = {}

    metadata = diagnostics["metadata"]
    optimization = diagnostics["optimization"]
    posterior = diagnostics["posterior"]

    executive_summary = _build_executive_summary(
        optimization=optimization,
        posterior=posterior,
    )

    optimization_section = _build_optimization_section(
        optimization,
        figure_files,
    )
    posterior_section = _build_posterior_section(
        posterior,
        figure_files,
    )
    recommendations = _build_recommendations(
        metadata=metadata,
        optimization=optimization,
        posterior=posterior,
    )

    return {
        "title": "Variational Inference Diagnostic Report",
        "subtitle": f"Guide: {metadata['guide']} | Dimension: {metadata['dim']}",
        "metadata": [
            ("Guide", metadata["guide"]),
            ("Dimension", metadata["dim"]),
            ("Seed", metadata["seed"]),
            ("Number of optimization steps", metadata["num_steps"]),
            ("Learning rate", metadata["learning_rate"]),
            ("Low-rank rank", metadata["lowrank_rank"]),
            ("Posterior draws", metadata["num_posterior_draws"]),
            ("Runtime (seconds)", round(metadata["runtime_seconds"], 3)),
            ("Timestamp", metadata["timestamp"]),
            (
                "Base normal correction",
                "Yes" if metadata["use_base_normal_correction"] else "No",
            ),
        ],
        "executive_summary": executive_summary,
        "recommendations": recommendations,
        "sections": [
            optimization_section,
            posterior_section,
        ],
    }


def _build_executive_summary(
    *,
    optimization: dict[str, Any],
    posterior: dict[str, Any],
) -> list[str]:
    """
    Build a short list of high-level conclusions for the report header.
    """
    summary: list[str] = []

    if optimization["converged"]:
        summary.append(
            "The optimization appears to have converged: the loss stabilized over the final iterations."
        )
    else:
        summary.append(
            "The optimization does not appear fully stabilized; convergence remains uncertain."
        )

    if optimization["all_finite"]:
        summary.append("No NaN or infinite values were detected in the loss sequence.")
    else:
        summary.append(
            "The loss sequence contains non-finite values, which indicates numerical instability."
        )

    summary.append(posterior["posterior_family_description"])
    summary.append(posterior["approximation_warning"])

    return summary


def _build_optimization_section(
    optimization: dict[str, Any],
    figure_files: dict[str, str],
) -> dict[str, Any]:
    """
    Build the optimization section of the report.
    """
    interpretation = [optimization["message"]]

    if optimization["has_losses"]:
        interpretation.append(
            f"The initial loss was {optimization['initial_loss']:.6g}, "
            f"and the final loss was {optimization['final_loss']:.6g}."
        )
        interpretation.append(
            f"The best loss was {optimization['best_loss']:.6g}, "
            f"reached at iteration {optimization['best_loss_step']}."
        )
        interpretation.append(
            f"Over the most recent {optimization['recent_window_size']} iterations, "
            f"the relative change in loss was {optimization['recent_relative_change']:.3g} "
            f"and the oscillation ratio was {optimization['oscillation_ratio']:.3g}."
        )

    return {
        "id": "optimization",
        "title": "Optimization diagnostics",
        "summary": optimization["message"],
        "interpretation": interpretation,
        "tables": [
            {
                "title": "Optimization summary",
                "columns": ["Metric", "Value"],
                "rows": [
                    ["Number of steps", optimization["num_steps"]],
                    ["All losses finite", _yes_no(optimization["all_finite"])],
                    ["Contains NaN", _yes_no(optimization["contains_nan"])],
                    ["Contains inf", _yes_no(optimization["contains_inf"])],
                    ["Initial loss", _fmt_float(optimization.get("initial_loss"))],
                    ["Final loss", _fmt_float(optimization.get("final_loss"))],
                    ["Best loss", _fmt_float(optimization.get("best_loss"))],
                    ["Best-loss iteration", optimization.get("best_loss_step")],
                    [
                        "Absolute improvement",
                        _fmt_float(optimization.get("absolute_improvement")),
                    ],
                    [
                        "Relative improvement",
                        _fmt_float(optimization.get("relative_improvement")),
                    ],
                    [
                        "Recent window size",
                        optimization.get("recent_window_size"),
                    ],
                    ["Recent mean", _fmt_float(optimization.get("recent_mean"))],
                    ["Recent std", _fmt_float(optimization.get("recent_std"))],
                    [
                        "Recent relative change",
                        _fmt_float(optimization.get("recent_relative_change")),
                    ],
                    [
                        "Oscillation ratio",
                        _fmt_float(optimization.get("oscillation_ratio")),
                    ],
                    [
                        "Fraction of decreasing steps",
                        _fmt_float(optimization.get("monotonic_fraction")),
                    ],
                    ["Converged", _yes_no(optimization["converged"])],
                ],
            }
        ],
        "figures": [
            {
                "id": "loss_curve",
                "title": "Loss trajectory",
                "kind": "loss_curve",
                "description": "Evolution of the ELBO loss over optimization iterations.",
                "file": figure_files.get("loss_curve"),
            },
            {
                "id": "loss_curve_recent",
                "title": "Loss trajectory (final iterations)",
                "kind": "loss_curve_recent",
                "description": "Zoom on the most recent part of the optimization to assess stabilization.",
                "file": figure_files.get("loss_curve_recent"),
            },
        ],
    }


def _build_posterior_section(
    posterior: dict[str, Any],
    figure_files: dict[str, str],
) -> dict[str, Any]:
    """
    Build the variational posterior section of the report.
    """
    interpretation = [
        posterior["posterior_family_description"],
        posterior["approximation_warning"],
        (
            f"The average posterior standard deviation is "
            f"{posterior['average_sd']:.6g}, with median {posterior['median_sd']:.6g}."
        ),
        (
            f"The average width of the 90% central interval is "
            f"{posterior['average_interval_width_90']:.6g}."
        ),
        (
            f"The fraction of 90% intervals containing zero is "
            f"{posterior['fraction_90_intervals_containing_zero']:.3f}."
        ),
    ]

    top_uncertain_rows = []
    for item in posterior["top_uncertain_parameters"]:
        top_uncertain_rows.append(
            [
                item["name"],
                _fmt_float(item["mean"]),
                _fmt_float(item["sd"]),
                _fmt_float(item["q05"]),
                _fmt_float(item["median"]),
                _fmt_float(item["q95"]),
                _fmt_float(item["interval_width"]),
                _yes_no(item["contains_zero"]),
            ]
        )

    return {
        "id": "posterior",
        "title": "Variational posterior diagnostics",
        "summary": posterior["approximation_warning"],
        "interpretation": interpretation,
        "tables": [
            {
                "title": "Posterior summary",
                "columns": ["Metric", "Value"],
                "rows": [
                    ["Guide", posterior["guide"]],
                    [
                        "Posterior family",
                        posterior["posterior_family_description"],
                    ],
                    ["Dimension", posterior["dim"]],
                    ["Posterior draws", posterior["num_draws"]],
                    ["Average posterior sd", _fmt_float(posterior["average_sd"])],
                    ["Median posterior sd", _fmt_float(posterior["median_sd"])],
                    ["Minimum posterior sd", _fmt_float(posterior["min_sd"])],
                    ["Maximum posterior sd", _fmt_float(posterior["max_sd"])],
                    [
                        "Average 90% interval width",
                        _fmt_float(posterior["average_interval_width_90"]),
                    ],
                    [
                        "Median 90% interval width",
                        _fmt_float(posterior["median_interval_width_90"]),
                    ],
                    [
                        "Fraction of 90% intervals containing zero",
                        _fmt_float(
                            posterior["fraction_90_intervals_containing_zero"]
                        ),
                    ],
                ],
            },
            {
                "title": "Most uncertain parameters",
                "columns": [
                    "Parameter",
                    "Mean",
                    "SD",
                    "Q05",
                    "Median",
                    "Q95",
                    "Interval width",
                    "Contains zero",
                ],
                "rows": top_uncertain_rows,
            },
        ],
        "figures": [
            {
                "id": "posterior_sd_rank",
                "title": "Posterior uncertainty ranking",
                "kind": "posterior_sd_rank",
                "description": "Posterior standard deviations sorted from largest to smallest.",
                "file": figure_files.get("posterior_sd_rank"),
            },
            {
                "id": "posterior_intervals_top",
                "title": "Intervals for most uncertain parameters",
                "kind": "posterior_intervals_top",
                "description": "Posterior medians and 90% intervals for the most uncertain parameters.",
                "file": figure_files.get("posterior_intervals_top"),
            },
        ],
    }


def _build_recommendations(
    *,
    metadata: dict[str, Any],
    optimization: dict[str, Any],
    posterior: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Build actionable recommendations based on simple diagnostic heuristics.

    Returns
    -------
    list of dict
        Each item contains:
        - severity: one of "info", "warning", "critical"
        - title: short heading
        - message: human-readable explanation and suggested next steps
    """
    recommendations: list[dict[str, str]] = []

    if not optimization["all_finite"]:
        recommendations.append(
            {
                "severity": "critical",
                "title": "Numerical instability detected",
                "message": (
                    "The loss sequence contains NaN or infinite values. "
                    "Try a smaller learning rate, inspect the numerical stability "
                    "of the likelihood and transformations, and verify that priors "
                    "and constraints keep the computation in a valid range."
                ),
            }
        )

    if not optimization["converged"]:
        recommendations.append(
            {
                "severity": "warning",
                "title": "Optimization may not have converged",
                "message": (
                    "The loss does not appear fully stabilized over the final iterations. "
                    "Consider increasing the number of optimization steps. If the recent "
                    "loss still oscillates, also try a smaller learning rate."
                ),
            }
        )

    if optimization["has_losses"] and optimization["oscillation_ratio"] > 0.1:
        recommendations.append(
            {
                "severity": "warning",
                "title": "Loss oscillations are non-negligible",
                "message": (
                    "The recent loss variability is relatively large compared with its mean. "
                    "A smaller learning rate, gradient clipping, or a better initialization "
                    "may improve stability."
                ),
            }
        )

    if (
        optimization["has_losses"]
        and optimization["recent_relative_change"] > 1e-2
        and optimization["all_finite"]
    ):
        recommendations.append(
            {
                "severity": "warning",
                "title": "Loss is still drifting near the end of optimization",
                "message": (
                    "The loss is still changing noticeably in the final iterations. "
                    "Try more optimization steps before interpreting the posterior."
                ),
            }
        )

    guide = metadata["guide"]
    dim = int(metadata["dim"])
    lowrank_rank = metadata["lowrank_rank"]

    if guide == "auto_diag":
        recommendations.append(
            {
                "severity": "info",
                "title": "Check whether posterior correlation matters",
                "message": (
                    "The current guide is diagonal Gaussian, so it cannot represent "
                    "posterior correlations. If correlated parameters are expected, "
                    "try `auto_lowrank` or `auto_mvn` and compare the results."
                ),
            }
        )

    if guide == "auto_normal":
        recommendations.append(
            {
                "severity": "info",
                "title": "Consider a richer posterior family",
                "message": (
                    "The current guide uses independent latent-site normal factors. "
                    "If dependencies between parameters matter, try `auto_lowrank` "
                    "or `auto_mvn` and compare posterior summaries and loss values."
                ),
            }
        )

    if guide == "auto_lowrank":
        if lowrank_rank is None:
            recommendations.append(
                {
                    "severity": "info",
                    "title": "Document the low-rank structure",
                    "message": (
                        "The guide is low-rank Gaussian. Record and report the chosen rank, "
                        "because it directly controls how much correlation structure can be represented."
                    ),
                }
            )
        else:
            if lowrank_rank <= max(1, dim // 20):
                recommendations.append(
                    {
                        "severity": "info",
                        "title": "Low-rank approximation may be restrictive",
                        "message": (
                            f"The low-rank guide uses rank {lowrank_rank} for dimension {dim}. "
                            "If posterior dependence seems important, try a larger rank and "
                            "check whether the loss or posterior summaries change materially."
                        ),
                    }
                )

    if guide == "auto_mvn" and dim > 200:
        recommendations.append(
            {
                "severity": "info",
                "title": "Full covariance may be expensive in high dimension",
                "message": (
                    "The guide is full-covariance Gaussian in a high-dimensional space. "
                    "If runtime or memory becomes problematic, compare with `auto_lowrank` "
                    "as a cheaper alternative."
                ),
            }
        )

    frac_zero = posterior["fraction_90_intervals_containing_zero"]
    if frac_zero > 0.9:
        recommendations.append(
            {
                "severity": "info",
                "title": "Most intervals contain zero",
                "message": (
                    "A very large fraction of 90% posterior intervals contain zero. "
                    "This may indicate weak information in the data, strong shrinkage, "
                    "or poor identifiability. Review priors, scaling, and parameterization."
                ),
            }
        )

    if posterior["max_sd"] > 10 * max(posterior["median_sd"], 1e-12):
        recommendations.append(
            {
                "severity": "warning",
                "title": "Some parameters are much more uncertain than the others",
                "message": (
                    "The posterior uncertainty is highly uneven across parameters. "
                    "Inspect the most uncertain parameters individually: they may suffer "
                    "from weak identification, poor scaling, or strong posterior dependence "
                    "that the guide does not capture well."
                ),
            }
        )

    recommendations.append(
        {
            "severity": "info",
            "title": "Assess approximation quality beyond the ELBO",
            "message": (
                "ELBO stabilization does not by itself prove that the variational posterior "
                "is accurate. A useful next step is to compare several seeds, compare guide "
                "families, and run posterior predictive checks. For smaller problems, a "
                "reference MCMC run can provide a stronger benchmark."
            ),
        }
    )

    recommendations.append(
        {
            "severity": "info",
            "title": "Check robustness across independent runs",
            "message": (
                "Repeat the optimization with several random seeds. If posterior means, "
                "uncertainties, or final loss values vary materially, the variational fit "
                "is not yet robust."
            ),
        }
    )

    return recommendations


def _yes_no(value: bool) -> str:
    """Format a boolean as Yes/No."""
    return "Yes" if value else "No"


def _fmt_float(value: Any) -> str:
    """Format a numeric value for display in the report."""
    if value is None:
        return ""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)