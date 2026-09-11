"""Generic grouping and aggregation for residual diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .metrics import metric_value
from .model import ResidualAuditConfig


GROUP_METRIC_COLUMNS = (
    "observation_count",
    "observed_total",
    "predicted_total",
    "raw_residual_total",
    "mean_raw_residual",
    "mean_predicted_value",
    "mae",
    "rmse",
    "weighted_rmse",
    "mean_pearson_residual",
    "fraction_abs_pearson_above_threshold",
    "support_failure_count",
    "support_failure_fraction",
)


def _group_columns(frame: pd.DataFrame, group_by: Sequence[str]) -> tuple[str, ...]:
    columns = tuple(str(column) for column in group_by)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError("grouping column(s) are missing: " + ", ".join(missing))
    return columns


def _group_label(values: tuple[object, ...]) -> str:
    return "all" if not values else "|".join(str(value) for value in values)


def _metric_row(frame: pd.DataFrame, *, config: ResidualAuditConfig, group_values: tuple[object, ...], group_by: tuple[str, ...]) -> dict[str, object]:
    row: dict[str, object] = {column: value for column, value in zip(group_by, group_values, strict=True)}
    row["group_key"] = _group_label(group_values)
    count = int(len(frame))
    raw = frame["raw_residual"].to_numpy(dtype=np.float64)
    predicted = frame["predicted_value"].to_numpy(dtype=np.float64)
    observed = frame["observed_value"].to_numpy(dtype=np.float64)
    pearson = frame["pearson_residual"].to_numpy(dtype=np.float64)
    variance = frame["variance"].to_numpy(dtype=np.float64)
    support = frame["support_failure"].to_numpy(dtype=bool)
    row["observation_count"] = count
    row["observed_total"] = metric_value(observed, reducer="sum")
    row["predicted_total"] = metric_value(predicted, reducer="sum")
    row["raw_residual_total"] = metric_value(raw, reducer="sum")
    row["mean_raw_residual"] = metric_value(raw, reducer="mean")
    row["mean_predicted_value"] = metric_value(predicted, reducer="mean")
    row["mae"] = metric_value(raw, reducer="mae")
    row["rmse"] = metric_value(raw, reducer="rmse")
    weighted_valid = np.isfinite(raw) & np.isfinite(variance) & (variance >= 0.0)
    if np.any(weighted_valid):
        weighted = raw[weighted_valid] / np.sqrt(
            np.maximum(variance[weighted_valid], config.variance_floor)
        )
        row["weighted_rmse"] = metric_value(weighted, reducer="rmse")
    else:
        row["weighted_rmse"] = None
    row["mean_pearson_residual"] = metric_value(pearson, reducer="mean")
    finite_pearson = pearson[np.isfinite(pearson)]
    row["fraction_abs_pearson_above_threshold"] = (
        None
        if finite_pearson.size == 0
        else float(np.mean(np.abs(finite_pearson) > config.pearson_threshold))
    )
    row["support_failure_count"] = int(np.sum(support))
    row["support_failure_fraction"] = None if count == 0 else float(np.mean(support))
    return row


def summarize_by(
    observations: pd.DataFrame,
    group_by: Sequence[str] = (),
    *,
    config: ResidualAuditConfig = ResidualAuditConfig(),
) -> pd.DataFrame:
    """Summarize diagnostics by arbitrary columns.

    The input is normally the output of
    :func:`~public_transportation.inference.residual_audit.metrics.compute_residual_diagnostics`.
    If diagnostics are absent, callers receive a clear error rather than a
    silently incomplete report.
    """
    required = {
        "observed_value",
        "predicted_value",
        "raw_residual",
        "variance",
        "pearson_residual",
        "support_failure",
    }
    missing = sorted(required - set(observations.columns))
    if missing:
        if {"observed_value", "predicted_value"}.issubset(observations.columns):
            from .metrics import compute_residual_diagnostics

            observations = compute_residual_diagnostics(observations, config=config)
        else:
            raise ValueError("residual diagnostics are missing column(s): " + ", ".join(missing))
    grouping = _group_columns(observations, group_by)
    if not grouping:
        rows = [_metric_row(observations, config=config, group_values=(), group_by=())]
    else:
        rows = []
        grouped = observations.groupby(list(grouping), dropna=False, sort=True, observed=True)
        for values, frame in grouped:
            if not isinstance(values, tuple):
                values = (values,)
            normalized = tuple("<missing>" if pd.isna(value) else value for value in values)
            rows.append(_metric_row(frame, config=config, group_values=normalized, group_by=grouping))
    columns = [*grouping, "group_key", *GROUP_METRIC_COLUMNS]
    return pd.DataFrame(rows, columns=columns)


def summarize_comparison_by(
    comparison: pd.DataFrame,
    group_by: Sequence[str] = (),
) -> pd.DataFrame:
    """Summarize prediction and residual changes by arbitrary metadata."""
    grouping = _group_columns(comparison, group_by)
    rows: list[dict[str, object]] = []
    if grouping:
        grouped = comparison.groupby(list(grouping), dropna=False, sort=True, observed=True)
        groups = ((values if isinstance(values, tuple) else (values,), frame) for values, frame in grouped)
    else:
        groups = (((), comparison),)
    for values, frame in groups:
        values = tuple("<missing>" if pd.isna(value) else value for value in values)
        prediction = frame["prediction_difference"].to_numpy(dtype=np.float64)
        residual = frame["residual_difference"].to_numpy(dtype=np.float64)
        row: dict[str, object] = {column: value for column, value in zip(grouping, values, strict=True)}
        row["group_key"] = _group_label(values)
        row.update(
            {
                "observation_count": int(len(frame)),
                "mean_prediction_difference": metric_value(prediction, reducer="mean"),
                "mae_prediction_difference": metric_value(prediction, reducer="mae"),
                "rmse_prediction_difference": metric_value(prediction, reducer="rmse"),
                "mean_residual_difference": metric_value(residual, reducer="mean"),
                "mae_residual_difference": metric_value(residual, reducer="mae"),
                "rmse_residual_difference": metric_value(residual, reducer="rmse"),
                "support_failure_changes": int(np.sum(frame["support_failure_run_a"] != frame["support_failure_run_b"])),
            }
        )
        rows.append(row)
    columns = [
        *grouping,
        "group_key",
        "observation_count",
        "mean_prediction_difference",
        "mae_prediction_difference",
        "rmse_prediction_difference",
        "mean_residual_difference",
        "mae_residual_difference",
        "rmse_residual_difference",
        "support_failure_changes",
    ]
    return pd.DataFrame(rows, columns=columns)
