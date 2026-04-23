from __future__ import annotations

from typing import Any
from icecream import ic
import numpy as np

from .results import VIResult


def _safe_float(value: np.ndarray | float | int) -> float:
    """Convert a scalar-like value to float."""
    return float(np.asarray(value).item())


def _recent_window_size(n: int, fraction: float = 0.1, minimum: int = 20) -> int:
    """Determine a reasonable size for the recent-iterations window."""
    if n <= 0:
        return 0
    return min(n, max(minimum, int(round(fraction * n))))


def compute_optimization_diagnostics(
    result: VIResult,
    *,
    recent_fraction: float = 0.1,
    minimum_recent_window: int = 20,
    convergence_rtol: float = 1e-3,
    oscillation_ratio_threshold: float = 0.1,
) -> dict[str, Any]:
    """
    Compute diagnostics related to the VI optimization process.

    Parameters
    ----------
    result:
        Variational inference result object.
    recent_fraction:
        Fraction of iterations considered as "recent" for stability checks.
    minimum_recent_window:
        Minimum size of the recent window.
    convergence_rtol:
        Relative tolerance used in a simple convergence heuristic.
    oscillation_ratio_threshold:
        Threshold on recent std / |recent mean| used to flag oscillations.

    Returns
    -------
    dict
        Dictionary of optimization diagnostics.
    """
    losses = np.asarray(result.losses, dtype=float)
    n = losses.size

    if n == 0:
        return {
            "num_steps": 0,
            "has_losses": False,
            "all_finite": True,
            "contains_nan": False,
            "contains_inf": False,
            "converged": False,
            "message": "No loss values are available.",
        }

    finite_mask = np.isfinite(losses)
    all_finite = bool(np.all(finite_mask))
    contains_nan = bool(np.any(np.isnan(losses)))
    contains_inf = bool(np.any(np.isinf(losses)))

    initial_loss = _safe_float(losses[0])
    final_loss = _safe_float(losses[-1])

    best_index = int(np.argmin(losses))
    best_loss = _safe_float(losses[best_index])

    absolute_improvement = initial_loss - final_loss
    relative_improvement = absolute_improvement / max(abs(initial_loss), 1e-12)

    recent_k = _recent_window_size(
        n=n,
        fraction=recent_fraction,
        minimum=minimum_recent_window,
    )
    recent_losses = losses[-recent_k:]

    recent_mean = _safe_float(np.mean(recent_losses))
    recent_std = _safe_float(np.std(recent_losses, ddof=0))
    recent_min = _safe_float(np.min(recent_losses))
    recent_max = _safe_float(np.max(recent_losses))

    if recent_k >= 2:
        recent_start = _safe_float(recent_losses[0])
        recent_end = _safe_float(recent_losses[-1])
        recent_absolute_change = recent_end - recent_start
        recent_relative_change = abs(recent_absolute_change) / max(
            abs(recent_start), 1e-12
        )
    else:
        recent_absolute_change = 0.0
        recent_relative_change = 0.0

    oscillation_ratio = recent_std / max(abs(recent_mean), 1e-12)

    monotonic_fraction = _safe_float(np.mean(np.diff(losses) <= 0.0)) if n >= 2 else 1.0

    converged = bool(
        all_finite
        and recent_relative_change <= convergence_rtol
        and oscillation_ratio <= oscillation_ratio_threshold
    )

    if not all_finite:
        message = "The loss sequence contains NaN or infinite values."
    elif converged:
        message = (
            "The loss appears to have stabilized over the final iterations, "
            "suggesting convergence."
        )
    else:
        message = (
            "The loss does not appear fully stabilized yet; convergence is uncertain."
        )

    return {
        "num_steps": int(n),
        "has_losses": True,
        "all_finite": all_finite,
        "contains_nan": contains_nan,
        "contains_inf": contains_inf,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss": best_loss,
        "best_loss_step": best_index,
        "absolute_improvement": float(absolute_improvement),
        "relative_improvement": float(relative_improvement),
        "recent_window_size": int(recent_k),
        "recent_mean": recent_mean,
        "recent_std": recent_std,
        "recent_min": recent_min,
        "recent_max": recent_max,
        "recent_absolute_change": float(recent_absolute_change),
        "recent_relative_change": float(recent_relative_change),
        "oscillation_ratio": float(oscillation_ratio),
        "monotonic_fraction": float(monotonic_fraction),
        "converged": converged,
        "message": message,
    }


def compute_posterior_diagnostics(
    result: VIResult,
    *,
    top_k: int = 10,
    parameter_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute diagnostics summarizing the fitted variational posterior.

    Parameters
    ----------
    result:
        Variational inference result object.
    top_k:
        Number of most uncertain parameters to list explicitly.

    Returns
    -------
    dict
        Dictionary of posterior diagnostics.
    """
    samples = np.asarray(result.posterior_samples_theta, dtype=float)
    mean = np.asarray(result.posterior_mean, dtype=float)
    sd = np.asarray(result.posterior_sd, dtype=float)
    q05 = np.asarray(result.posterior_q05, dtype=float)
    q50 = np.asarray(result.posterior_q50, dtype=float)
    q95 = np.asarray(result.posterior_q95, dtype=float)

    num_draws = samples.shape[0] if samples.ndim >= 1 else 0
    dim = int(result.dim)

    avg_sd = _safe_float(np.mean(sd))
    median_sd = _safe_float(np.median(sd))
    min_sd = _safe_float(np.min(sd))
    max_sd = _safe_float(np.max(sd))

    interval_width = q95 - q05
    avg_interval_width = _safe_float(np.mean(interval_width))
    median_interval_width = _safe_float(np.median(interval_width))

    near_zero_mask = (q05 <= 0.0) & (q95 >= 0.0)
    fraction_intervals_containing_zero = _safe_float(np.mean(near_zero_mask))

    uncertainty_order = np.argsort(-sd)
    top_k = min(top_k, dim)
    top_uncertain = []
    for idx in uncertainty_order[:top_k]:
        if parameter_names is not None and idx < len(parameter_names):
            name = str(parameter_names[idx])
        else:
            name = f"param_{idx}"

        top_uncertain.append(
            {
                "index": int(idx),
                "name": name,
                "mean": _safe_float(mean[idx]),
                "sd": _safe_float(sd[idx]),
                "q05": _safe_float(q05[idx]),
                "median": _safe_float(q50[idx]),
                "q95": _safe_float(q95[idx]),
                "interval_width": _safe_float(interval_width[idx]),
                "contains_zero": bool(near_zero_mask[idx]),
            }
        )

    posterior_family_description = _describe_guide(
        guide=result.guide,
        lowrank_rank=result.lowrank_rank,
    )

    if result.guide == "auto_diag":
        approximation_warning = (
            "The variational posterior is diagonal Gaussian, so posterior "
            "correlations are not represented."
        )
    elif result.guide == "auto_normal":
        approximation_warning = (
            "The variational posterior is based on independent normal latent-site "
            "approximations, so global posterior correlation may be poorly captured."
        )
    elif result.guide == "auto_lowrank":
        approximation_warning = (
            "The variational posterior is low-rank Gaussian. It can represent part "
            "of the correlation structure, but only through a restricted low-rank form."
        )
    elif result.guide == "auto_mvn":
        approximation_warning = (
            "The variational posterior is full-covariance Gaussian. Correlation is "
            "represented, but non-Gaussian structure such as multimodality is not."
        )
    else:
        approximation_warning = (
            "The variational posterior is based on an automatic guide. Its quality "
            "depends on how well the selected family matches the true posterior."
        )

    return {
        "guide": result.guide,
        "posterior_family_description": posterior_family_description,
        "approximation_warning": approximation_warning,
        "dim": dim,
        "num_draws": int(num_draws),
        "average_sd": avg_sd,
        "median_sd": median_sd,
        "min_sd": min_sd,
        "max_sd": max_sd,
        "average_interval_width_90": avg_interval_width,
        "median_interval_width_90": median_interval_width,
        "fraction_90_intervals_containing_zero": fraction_intervals_containing_zero,
        "top_uncertain_parameters": top_uncertain,
    }


def compute_all_diagnostics(
    result: VIResult,
    *,
    top_k: int = 10,
    parameter_names: list[str] | None = None,
    recent_fraction: float = 0.1,
    minimum_recent_window: int = 20,
    convergence_rtol: float = 1e-3,
    oscillation_ratio_threshold: float = 0.1,
) -> dict[str, Any]:
    """
    Compute all currently available diagnostics for a VIResult.

    Returns
    -------
    dict
        Dictionary with optimization and posterior sections.
    """
    ic(parameter_names)
    optimization = compute_optimization_diagnostics(
        result=result,
        recent_fraction=recent_fraction,
        minimum_recent_window=minimum_recent_window,
        convergence_rtol=convergence_rtol,
        oscillation_ratio_threshold=oscillation_ratio_threshold,
    )
    posterior = compute_posterior_diagnostics(
        result=result,
        top_k=top_k,
        parameter_names=parameter_names,
    )

    return {
        "metadata": {
            "guide": result.guide,
            "dim": int(result.dim),
            "seed": int(result.seed),
            "num_steps": int(result.num_steps),
            "learning_rate": float(result.learning_rate),
            "lowrank_rank": result.lowrank_rank,
            "num_posterior_draws": int(result.num_posterior_draws),
            "runtime_seconds": float(result.runtime_seconds),
            "timestamp": str(result.timestamp),
            "use_base_normal_correction": bool(result.use_base_normal_correction),
        },
        "optimization": optimization,
        "posterior": posterior,
    }


def _describe_guide(guide: str, lowrank_rank: int | None) -> str:
    """Return a human-readable description of the guide."""
    if guide == "auto_diag":
        return "Mean-field Gaussian variational posterior with diagonal covariance."
    if guide == "auto_normal":
        return "Automatic Normal guide with independent latent-site normal factors."
    if guide == "auto_lowrank":
        if lowrank_rank is None:
            return "Low-rank multivariate Gaussian variational posterior."
        return (
            "Low-rank multivariate Gaussian variational posterior "
            f"with rank {lowrank_rank}."
        )
    if guide == "auto_mvn":
        return "Full-covariance multivariate Gaussian variational posterior."
    return f"Automatic guide '{guide}'."