from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import compute_all_diagnostics
from .results import MLResult


def generate_ml_report_plots(
    result: MLResult,
    output_dir: str | Path,
    *,
    diagnostics: dict[str, Any] | None = None,
    parameter_names: list[str] | None = None,
    top_k: int = 10,
) -> dict[str, str]:
    """Generate plot files for the ML HTML report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if diagnostics is None:
        diagnostics = compute_all_diagnostics(
            result,
            parameter_names=parameter_names,
            top_k=top_k,
        )

    relative_prefix = output_path.name
    produced: dict[str, str] = {}

    objective_path = output_path / "objective_trace.png"
    _plot_objective_trace(result=result, output_file=objective_path)
    produced["objective_trace"] = f"{relative_prefix}/{objective_path.name}"

    gradient_path = output_path / "gradient_norm_trace.png"
    _plot_gradient_norm_trace(result=result, output_file=gradient_path)
    produced["gradient_norm_trace"] = f"{relative_prefix}/{gradient_path.name}"

    parameter_path = output_path / "parameter_estimates_top.png"
    _plot_parameter_estimates_top(
        diagnostics=diagnostics,
        output_file=parameter_path,
        top_k=top_k,
    )
    produced["parameter_estimates_top"] = f"{relative_prefix}/{parameter_path.name}"

    correlation_path = output_path / "parameter_correlation_matrix.png"
    _plot_parameter_correlation_matrix(result=result, output_file=correlation_path)
    produced["parameter_correlation_matrix"] = f"{relative_prefix}/{correlation_path.name}"

    return produced


def _plot_objective_trace(result: MLResult, output_file: Path) -> None:
    """Plot objective value across optimizer iterations."""
    trace = np.asarray(result.optimization_trace, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    if trace.size > 0 and trace.ndim == 2 and trace.shape[1] >= 2:
        ax.plot(trace[:, 0], trace[:, 1])
    else:
        ax.plot([0], [result.objective_value], marker="o")
    ax.set_title("Objective trace")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def _plot_gradient_norm_trace(result: MLResult, output_file: Path) -> None:
    """Plot gradient norm across optimizer iterations."""
    trace = np.asarray(result.optimization_trace, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    if trace.size > 0 and trace.ndim == 2 and trace.shape[1] >= 3:
        ax.plot(trace[:, 0], trace[:, 2])
    else:
        ax.plot([0], [result.gradient_norm], marker="o")
    ax.set_title("Gradient norm trace")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient norm")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def _plot_parameter_estimates_top(
    diagnostics: dict[str, Any],
    output_file: Path,
    *,
    top_k: int = 10,
) -> None:
    """Plot top parameter estimates with approximate 95% confidence intervals."""
    rows = diagnostics["parameters"].get("top_uncertain_parameters", [])[: max(top_k, 0)]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * max(len(rows), 1))))
    if not rows:
        ax.text(0.5, 0.5, "No standard errors available", ha="center", va="center")
        ax.set_axis_off()
    else:
        names = [str(row["name"]) for row in rows]
        estimate = np.asarray([row["estimate"] for row in rows], dtype=float)
        se = np.asarray([row["standard_error"] for row in rows], dtype=float)
        y = np.arange(len(rows))
        ax.errorbar(estimate, y, xerr=1.96 * se, fmt="o", capsize=3)
        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_title("Parameter estimates: most uncertain")
        ax.set_xlabel("Estimate with approximate 95% CI")
        ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def _plot_parameter_correlation_matrix(result: MLResult, output_file: Path) -> None:
    """Plot parameter correlation matrix derived from covariance matrix."""
    cov = result.covariance_matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    if cov is None or cov.size == 0:
        ax.text(0.5, 0.5, "No covariance matrix available", ha="center", va="center")
        ax.set_axis_off()
    else:
        cov = np.asarray(cov, dtype=float)
        sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
        denom = np.outer(sd, sd)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 0, cov / denom, np.nan)
        im = ax.imshow(corr, vmin=-1.0, vmax=1.0)
        ax.set_title("Parameter correlation matrix")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)
