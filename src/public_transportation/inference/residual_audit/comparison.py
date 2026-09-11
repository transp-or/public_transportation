"""Cross-run residual and prediction comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .grouping import summarize_comparison_by
from .io import _id_key, normalize_observation_table
from .metrics import compute_residual_diagnostics
from .model import ResidualAuditConfig


@dataclass(frozen=True, slots=True)
class ResidualRunComparison:
    """Row-level and grouped comparison results."""

    rows: pd.DataFrame
    grouped_metrics: pd.DataFrame


def compare_runs(
    run_a: pd.DataFrame,
    run_b: pd.DataFrame,
    *,
    group_by: Sequence[str] = (),
    allow_subset_comparison: bool = False,
    config: ResidualAuditConfig = ResidualAuditConfig(),
) -> ResidualRunComparison:
    """Compare two normalized or persisted observation tables by ID."""
    left = compute_residual_diagnostics(
        normalize_observation_table(run_a, table_name="run_a"), config=config
    )
    right = compute_residual_diagnostics(
        normalize_observation_table(run_b, table_name="run_b"), config=config
    )
    left = left.copy()
    right = right.copy()
    left["__comparison_id"] = left["observation_id"].map(_id_key)
    right["__comparison_id"] = right["observation_id"].map(_id_key)
    left_ids = set(left["__comparison_id"])
    right_ids = set(right["__comparison_id"])
    if not allow_subset_comparison and left_ids != right_ids:
        raise ValueError("compared runs contain different observation IDs; use allow_subset_comparison explicitly for a subset comparison.")
    common = left_ids & right_ids
    if not common:
        raise ValueError("compared runs contain no common observation IDs.")
    left = left[left["__comparison_id"].isin(common)].copy()
    right = right[right["__comparison_id"].isin(common)].copy()
    right = right.set_index("__comparison_id", drop=False)
    observed_left = pd.to_numeric(left["observed_value"], errors="raise").to_numpy(dtype=np.float64)
    observed_right = np.asarray([float(right.loc[key, "observed_value"]) for key in left["__comparison_id"]])
    if not np.array_equal(observed_left, observed_right):
        raise ValueError("compared runs contain different observed values.")
    rows: list[dict[str, object]] = []
    for _, current in left.iterrows():
        key = current["__comparison_id"]
        other = right.loc[key]
        prediction_a = float(current["predicted_value"])
        prediction_b = float(other["predicted_value"])
        residual_a = float(current.get("raw_residual", float(current["observed_value"]) - prediction_a))
        residual_b = float(other.get("raw_residual", float(other["observed_value"]) - prediction_b))
        row: dict[str, object] = {
            "observation_id": current["observation_id"],
            "predicted_value_run_a": prediction_a,
            "predicted_value_run_b": prediction_b,
            "prediction_difference": prediction_b - prediction_a,
            "absolute_prediction_difference": abs(prediction_b - prediction_a),
            "raw_residual_run_a": residual_a,
            "raw_residual_run_b": residual_b,
            "residual_difference": residual_b - residual_a,
            "support_failure_run_a": bool(current.get("support_failure", False)),
            "support_failure_run_b": bool(other.get("support_failure", False)),
        }
        for column in group_by:
            if column not in left.columns:
                raise ValueError(f"grouping column(s) are missing: {column}")
            row[column] = current[column]
        rows.append(row)
    frame = pd.DataFrame(rows)
    grouped = summarize_comparison_by(frame, group_by)
    return ResidualRunComparison(rows=frame, grouped_metrics=grouped)
