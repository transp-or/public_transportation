"""Numerically safe residual calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import ResidualAuditConfig


def _finite_numeric(frame: pd.DataFrame, column: str, *, context: str) -> np.ndarray:
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} column {column!r} must contain numeric values.") from error
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{context} column {column!r} must contain finite values.")
    return values


def poisson_deviance_residual(observed: object, predicted: object) -> np.ndarray:
    """Return signed Poisson deviance residuals with correct boundary limits."""
    y = np.asarray(observed, dtype=np.float64)
    mu = np.asarray(predicted, dtype=np.float64)
    if y.shape != mu.shape:
        raise ValueError("observed and predicted arrays must have the same shape.")
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(mu)):
        raise ValueError("observed and predicted values must be finite.")
    if np.any(y < 0.0) or np.any(mu < 0.0):
        raise ValueError("Poisson observed and predicted values must be non-negative.")

    deviance = np.zeros_like(y)
    positive_y = y > 0.0
    positive_mu = mu > 0.0
    both_positive = positive_y & positive_mu
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        deviance[both_positive] = 2.0 * (
            y[both_positive] * np.log(y[both_positive] / mu[both_positive])
            - (y[both_positive] - mu[both_positive])
        )
    # y=0 has the limit D=2*mu, including y=mu=0 -> D=0.
    zero_y = ~positive_y
    deviance[zero_y] = 2.0 * mu[zero_y]
    # mu=0 and y>0 has infinite deviance; retain the mathematically correct
    # limit so that the support-failure flag explains the exceptional row.
    deviance[positive_y & ~positive_mu] = np.inf
    deviance = np.maximum(deviance, 0.0)
    sign = np.sign(y - mu)
    return sign * np.sqrt(deviance)


def compute_residual_diagnostics(
    observations: pd.DataFrame,
    *,
    config: ResidualAuditConfig = ResidualAuditConfig(),
) -> pd.DataFrame:
    """Compute row-level residual diagnostics from a normalized table."""
    required = {"observation_id", "observed_value", "predicted_value"}
    missing = sorted(required - set(observations.columns))
    if missing:
        raise ValueError("observation table is missing required column(s): " + ", ".join(missing))
    if observations["observation_id"].isna().any() or observations["observation_id"].duplicated().any():
        raise ValueError("observation table contains duplicate or missing observation_id values.")
    frame = observations.copy()
    observed = _finite_numeric(frame, "observed_value", context="observation")
    predicted = _finite_numeric(frame, "predicted_value", context="observation")
    if np.any(observed < 0.0) or np.any(predicted < 0.0):
        raise ValueError("observed_value and predicted_value must be non-negative count values.")

    if "variance" in frame.columns:
        variance = _finite_numeric(frame, "variance", context="observation")
        if np.any(variance < 0.0):
            raise ValueError("variance must be non-negative.")
    else:
        variance = predicted.copy()

    raw = observed - predicted
    relative = np.full(observed.shape, np.nan, dtype=np.float64)
    valid_relative = np.abs(observed) > config.relative_observation_threshold
    relative[valid_relative] = raw[valid_relative] / observed[valid_relative]
    pearson = raw / np.sqrt(np.maximum(variance, config.variance_floor))
    deviance = poisson_deviance_residual(observed, predicted)
    positive = observed > config.positive_observation_threshold
    near_zero = predicted <= config.predicted_mean_floor
    support_failure = positive & near_zero

    frame["observed_value"] = observed
    frame["predicted_value"] = predicted
    frame["raw_residual"] = raw
    frame["relative_residual"] = relative
    frame["variance"] = variance
    frame["pearson_residual"] = pearson
    frame["poisson_deviance_residual"] = deviance
    frame["support_failure"] = support_failure
    frame["near_zero_prediction"] = near_zero
    frame["positive_observation"] = positive
    return frame


def metric_value(values: np.ndarray, *, reducer: str) -> float | None:
    """Reduce finite values, returning ``None`` when no denominator exists."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    if reducer == "mean":
        return float(np.mean(finite))
    if reducer == "sum":
        return float(np.sum(finite))
    if reducer == "mae":
        return float(np.mean(np.abs(finite)))
    if reducer == "rmse":
        return float(np.sqrt(np.mean(finite * finite)))
    raise ValueError(f"unsupported metric reducer {reducer!r}.")


compute_residual_metrics = compute_residual_diagnostics
