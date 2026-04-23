from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import compute_all_diagnostics
from .results import VIResult


def generate_vi_report_plots(
    result: VIResult,
    output_dir: str | Path,
    *,
    diagnostics: dict[str, Any] | None = None,
    parameter_names: list[str] | None = None,
    top_k: int = 10,
) -> dict[str, str]:
    """
    Generate plot files for the VI HTML report.

    Parameters
    ----------
    result:
        Variational inference result.
    output_dir:
        Directory where the plots will be written.
    diagnostics:
        Optional precomputed diagnostics. If omitted, they are computed internally.
    parameter_names:
        Optional parameter names used for the interval plot.
    top_k:
        Number of most uncertain parameters to plot.

    Returns
    -------
    dict
        Mapping from figure id to relative filename.
    """
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

    loss_curve_path = output_path / "loss_curve.png"
    _plot_loss_curve(result=result, output_file=loss_curve_path)
    produced["loss_curve"] = f"{relative_prefix}/{loss_curve_path.name}"

    loss_curve_recent_path = output_path / "loss_curve_recent.png"
    _plot_loss_curve_recent(
        result=result,
        diagnostics=diagnostics,
        output_file=loss_curve_recent_path,
    )
    produced["loss_curve_recent"] = f"{relative_prefix}/{loss_curve_recent_path.name}"

    posterior_sd_rank_path = output_path / "posterior_sd_rank.png"
    _plot_posterior_sd_rank(result=result, output_file=posterior_sd_rank_path)
    produced["posterior_sd_rank"] = f"{relative_prefix}/{posterior_sd_rank_path.name}"

    posterior_intervals_top_path = output_path / "posterior_intervals_top.png"
    _plot_posterior_intervals_top(
        diagnostics=diagnostics,
        output_file=posterior_intervals_top_path,
        top_k=top_k,
    )
    produced["posterior_intervals_top"] = f"{relative_prefix}/{posterior_intervals_top_path.name}"

    return produced


def _plot_loss_curve(result: VIResult, output_file: Path) -> None:
    """Plot the full optimization loss trajectory."""
    losses = np.asarray(result.losses, dtype=float)
    steps = np.arange(1, losses.size + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, losses)
    ax.set_title("Loss trajectory")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("ELBO loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_loss_curve_recent(
    result: VIResult,
    diagnostics: dict[str, Any],
    output_file: Path,
) -> None:
    """Plot the recent tail of the loss trajectory."""
    losses = np.asarray(result.losses, dtype=float)
    recent_k = int(diagnostics["optimization"]["recent_window_size"])
    recent_k = max(1, min(recent_k, losses.size))

    recent_losses = losses[-recent_k:]
    recent_steps = np.arange(losses.size - recent_k + 1, losses.size + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(recent_steps, recent_losses)
    ax.set_title("Loss trajectory (final iterations)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("ELBO loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_posterior_sd_rank(result: VIResult, output_file: Path) -> None:
    """Plot posterior standard deviations sorted from largest to smallest."""
    sd = np.asarray(result.posterior_sd, dtype=float)
    sd_sorted = np.sort(sd)[::-1]
    ranks = np.arange(1, sd_sorted.size + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ranks, sd_sorted)
    ax.set_title("Posterior uncertainty ranking")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Posterior standard deviation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_posterior_intervals_top(
    diagnostics: dict[str, Any],
    output_file: Path,
    *,
    top_k: int,
) -> None:
    """Plot medians and 90% intervals for the most uncertain parameters."""
    items = diagnostics["posterior"]["top_uncertain_parameters"][:top_k]
    if not items:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.set_title("Intervals for most uncertain parameters")
        ax.text(0.5, 0.5, "No posterior data available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    names = [str(item["name"]) for item in items]
    medians = np.asarray([item["median"] for item in items], dtype=float)
    q05 = np.asarray([item["q05"] for item in items], dtype=float)
    q95 = np.asarray([item["q95"] for item in items], dtype=float)

    lower = medians - q05
    upper = q95 - medians
    y = np.arange(len(items))

    fig_height = max(4.5, 0.45 * len(items) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.errorbar(
        medians,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_title("Intervals for most uncertain parameters")
    ax.set_xlabel("Parameter value")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)