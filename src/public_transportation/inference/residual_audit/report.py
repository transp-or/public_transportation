"""Audit orchestration and deterministic report writing."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from public_transportation.version import __version__

from .comparison import ResidualRunComparison, compare_runs
from .grouping import summarize_by
from .io import (
    join_metadata,
    normalize_contribution_table,
    normalize_metadata_table,
    normalize_observation_table,
)
from .metrics import compute_residual_diagnostics
from .model import ResidualAuditConfig, ResidualAuditResult, ResidualAuditRun


RESIDUAL_AUDIT_SCHEMA_VERSION = 1
_EMPTY_SUPPORT_COLUMNS = (
    "observation_id",
    "positive_observation",
    "near_zero_prediction",
    "support_failure",
    "observed_value",
    "predicted_value",
    "raw_residual",
)
_COMPARISON_COLUMNS = (
    "observation_id",
    "predicted_value_run_a",
    "predicted_value_run_b",
    "prediction_difference",
    "absolute_prediction_difference",
    "raw_residual_run_a",
    "raw_residual_run_b",
    "residual_difference",
    "support_failure_run_a",
    "support_failure_run_b",
)


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _dataframe_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        records.append({str(key): _json_value(value) for key, value in record.items()})
    return records


def _write_csv(path: Path, frame: pd.DataFrame, *, columns: Sequence[str] | None = None) -> None:
    selected = frame if columns is None else frame.reindex(columns=list(columns))
    temporary = path.with_name(f".{path.name}.tmp")
    selected.to_csv(temporary, index=False, na_rep="", lineterminator="\n")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _numeric_component_columns(contributions: pd.DataFrame) -> list[str]:
    excluded = {
        "observation_id",
        "__audit_id",
        "observation_scale",
        "scaled_mean",
        "modeled_mean",
        "latent_total",
        "active_components",
    }
    columns: list[str] = []
    for column in contributions.columns:
        if column in excluded:
            continue
        if column == "measurement_type" or contributions[column].dtype == object:
            continue
        try:
            values = pd.to_numeric(contributions[column], errors="raise").to_numpy(dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if np.all(np.isfinite(values)):
            columns.append(column)
    return columns


def _contribution_analysis(
    diagnostics: pd.DataFrame,
    contributions: pd.DataFrame | None,
    *,
    group_by: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    if contributions is None:
        return pd.DataFrame(), {
            "available": False,
            "message": "No contribution table was supplied.",
        }
    contribution_ids = set(contributions["__audit_id"])
    observation_ids = set(diagnostics["__audit_id"])
    if contribution_ids != observation_ids:
        raise ValueError("contribution table cannot be joined: observation IDs differ.")
    components = _numeric_component_columns(contributions)
    if not components:
        return pd.DataFrame(), {
            "available": False,
            "message": "Contribution table contains no numeric contribution columns.",
        }
    diagnostic_columns = ["__audit_id", "observation_id", "predicted_value"]
    diagnostic_columns.extend(
        column
        for column in {"measurement_type", *map(str, group_by)}
        if column in diagnostics.columns and column not in diagnostic_columns
    )
    merged = diagnostics[diagnostic_columns].merge(
        contributions.drop(columns=["observation_id"], errors="ignore"),
        on="__audit_id",
        how="left",
        validate="one_to_one",
    )
    # Metadata are already on diagnostics.  Add only requested grouping columns.
    for column in group_by:
        if column not in diagnostics.columns:
            raise ValueError(f"grouping column(s) are missing: {column}")
        merged[column] = diagnostics.set_index("__audit_id").loc[merged["__audit_id"], column].to_numpy()
    expected_column = "scaled_mean" if "scaled_mean" in merged.columns else (
        "modeled_mean" if "modeled_mean" in merged.columns else None
    )
    component_sum = merged[components].sum(axis=1)
    expected = merged[expected_column] if expected_column is not None else merged["predicted_value"]
    difference = expected.to_numpy(dtype=np.float64) - component_sum.to_numpy(dtype=np.float64)
    check = {
        "performed": True,
        "component_columns": components,
        "expected_column": expected_column or "predicted_value",
        "rows_checked": int(len(merged)),
        "rows_not_matching": int(np.sum(~np.isclose(difference, 0.0, rtol=1.0e-6, atol=1.0e-8))),
        "maximum_absolute_difference": float(np.max(np.abs(difference))) if len(difference) else 0.0,
    }
    boundary_columns = [
        column
        for column in components
        if "initial_onboard" in column or "terminal_outflow" in column
    ]
    od_columns = [column for column in components if column in {"od_contribution", "od_flow"}]
    boundary_total = merged[boundary_columns].sum(axis=1) if boundary_columns else pd.Series(0.0, index=merged.index)
    od_total = merged[od_columns].sum(axis=1) if od_columns else pd.Series(0.0, index=merged.index)
    merged["__boundary_nonzero"] = np.abs(boundary_total) > 0.0
    merged["__od_zero_prediction_positive"] = (np.abs(od_total) <= 0.0) & (merged["predicted_value"] > 0.0)
    merged["__boundary_dominates"] = np.abs(boundary_total) > np.abs(od_total)
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("all", merged)]
    if "measurement_type" in merged.columns and "measurement_type" not in group_by:
        for value, frame in merged.groupby("measurement_type", dropna=False, sort=True, observed=True):
            scopes.append((f"measurement_type={value}", frame))
    if len(group_by) > 1:
        for values, frame in merged.groupby(list(group_by), dropna=False, sort=True, observed=True):
            if not isinstance(values, tuple):
                values = (values,)
            scopes.append(("|".join(f"{column}={value}" for column, value in zip(group_by, values, strict=True)), frame))
    for column in group_by:
        if column == "measurement_type":
            continue
        for value, frame in merged.groupby(column, dropna=False, sort=True, observed=True):
            scopes.append((f"{column}={value}", frame))
    for scope, frame in scopes:
        for component in components:
            values = frame[component].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "scope": scope,
                    "component": component,
                    "row_count": int(len(frame)),
                    "total_contribution": float(np.sum(values)),
                    "nonzero_rows": int(np.sum(np.abs(values) > 0.0)),
                    "boundary_nonzero_rows": int(np.sum(frame["__boundary_nonzero"])),
                    "od_zero_prediction_positive_rows": int(np.sum(frame["__od_zero_prediction_positive"])),
                    "boundary_dominates_rows": int(np.sum(frame["__boundary_dominates"])),
                }
            )
    return pd.DataFrame(rows), {
        "available": True,
        "component_columns": components,
        "boundary_component_columns": boundary_columns,
        "contribution_check": check,
    }


def _summary_markdown(
    diagnostics: pd.DataFrame,
    grouped: pd.DataFrame,
    support: pd.DataFrame,
    contribution_info: Mapping[str, object],
    comparison: pd.DataFrame,
    *,
    is_holdout_evaluation: bool,
) -> str:
    observed = diagnostics["observed_value"].to_numpy(dtype=np.float64)
    predicted = diagnostics["predicted_value"].to_numpy(dtype=np.float64)
    raw = diagnostics["raw_residual"].to_numpy(dtype=np.float64)
    finite_raw = raw[np.isfinite(raw)]
    mae = float(np.mean(np.abs(finite_raw))) if finite_raw.size else None
    rmse = float(np.sqrt(np.mean(finite_raw * finite_raw))) if finite_raw.size else None
    variance = diagnostics["variance"].to_numpy(dtype=np.float64)
    valid = np.isfinite(raw) & np.isfinite(variance)
    weighted = raw[valid] / np.sqrt(np.maximum(variance[valid], 1.0e-8))
    weighted_rmse = float(np.sqrt(np.mean(weighted * weighted))) if weighted.size else None
    lines = [
        "# Residual audit",
        "",
        f"- Observation count: {len(diagnostics):,}",
        f"- Observed total: {float(np.sum(observed)):g}",
        f"- Predicted total: {float(np.sum(predicted)):g}",
        f"- MAE: {'' if mae is None else f'{mae:.6g}'}",
        f"- RMSE: {'' if rmse is None else f'{rmse:.6g}'}",
        f"- Weighted RMSE: {'' if weighted_rmse is None else f'{weighted_rmse:.6g}'}",
        f"- Support failures: {len(support):,}",
        f"- Support-failure share: {len(support) / len(diagnostics):.6%}" if len(diagnostics) else "- Support-failure share: 0%",
        "",
        "## Interpretation",
        "",
        "No observations were automatically removed by this audit.",
        "Analysis of fit observations is not independent validation unless an explicit holdout was supplied.",
        f"This run was marked as an independent holdout evaluation: {bool(is_holdout_evaluation)}.",
    ]
    if len(support):
        lines.extend(
            [
                "",
                "## Support failures",
                "",
                "Positive observations with near-zero predictions are reported separately and should not be interpreted as ordinary residuals.",
            ]
        )
    if not grouped.empty:
        largest_positive = grouped.sort_values("raw_residual_total", ascending=False).iloc[0]
        largest_negative = grouped.sort_values("raw_residual_total", ascending=True).iloc[0]
        highest_support = grouped.sort_values("support_failure_fraction", ascending=False, na_position="last").iloc[0]
        lines.extend(
            [
                "",
                "## Largest grouped mismatches",
                "",
                f"- Largest positive residual group: {largest_positive['group_key']} ({largest_positive['raw_residual_total']:.6g}).",
                f"- Largest negative residual group: {largest_negative['group_key']} ({largest_negative['raw_residual_total']:.6g}).",
                f"- Highest support-failure fraction: {highest_support['group_key']} ({highest_support['support_failure_fraction']:.6%}).",
            ]
        )
    lines.extend(["", "## Contribution analysis", ""])
    if contribution_info.get("available"):
        lines.append(f"Contribution components: {', '.join(map(str, contribution_info.get('component_columns', [])))}.")
        check = contribution_info.get("contribution_check", {})
        if isinstance(check, Mapping):
            lines.append(f"Rows failing contribution-total check: {int(check.get('rows_not_matching', 0)):,}.")
        lines.append("Boundary contributions are additive measurement components; they are not interpreted as conserved OD flows.")
    else:
        lines.append(str(contribution_info.get("message", "No contribution analysis was available.")))
    if len(comparison):
        difference = comparison["absolute_prediction_difference"].to_numpy(dtype=np.float64)
        lines.extend(
            [
                "",
                "## Cross-run comparison",
                "",
                f"Compared observations: {len(comparison):,}.",
                f"Maximum absolute prediction difference: {float(np.max(difference)):.6g}.",
                f"Mean absolute prediction difference: {float(np.mean(difference)):.6g}.",
            ]
        )
    else:
        lines.extend(["", "## Cross-run comparison", "", "No second run was supplied."])
    return "\n".join(lines) + "\n"


def audit_run(
    run: ResidualAuditRun,
    *,
    group_by: Sequence[str] = (),
    config: ResidualAuditConfig = ResidualAuditConfig(),
    comparison_run: ResidualAuditRun | None = None,
    allow_subset_comparison: bool = False,
    is_holdout_evaluation: bool = False,
) -> ResidualAuditResult:
    """Analyze one run and optionally compare it with a second run."""
    observations = (
        run.observations
        if "__audit_id" in run.observations.columns
        else normalize_observation_table(run.observations)
    )
    if run.metadata is not None:
        metadata = (
            run.metadata
            if "__audit_id" in run.metadata.columns
            else normalize_metadata_table(run.metadata)
        )
        if len(metadata):
            observations = join_metadata(observations, metadata)
    contributions = run.contributions
    if contributions is not None and "__audit_id" not in contributions.columns:
        contributions = normalize_contribution_table(contributions)
    diagnostics = compute_residual_diagnostics(observations, config=config)
    grouped = summarize_by(diagnostics, group_by, config=config)
    support_columns = [
        column
        for column in (
            "observation_id",
            "measurement_id",
            "source_row_index",
            "measurement_type",
            "line",
            "direction",
            "stop",
            "trip",
            "time_bin",
            "observation_time",
            "journey_position",
            "boundary_role",
            "observed_value",
            "predicted_value",
            "raw_residual",
            "positive_observation",
            "near_zero_prediction",
            "support_failure",
        )
        if column in diagnostics.columns
    ]
    support = diagnostics.loc[diagnostics["support_failure"], support_columns].copy()
    contribution_summary, contribution_info = _contribution_analysis(
        diagnostics, contributions, group_by=group_by
    )
    comparison_frame = pd.DataFrame(columns=list(_COMPARISON_COLUMNS))
    comparison_grouped = pd.DataFrame()
    comparison_info: dict[str, object] = {"available": False}
    if comparison_run is not None:
        comparison_result: ResidualRunComparison = compare_runs(
            run.observations,
            comparison_run.observations,
            group_by=group_by,
            allow_subset_comparison=allow_subset_comparison,
            config=config,
        )
        comparison_frame = comparison_result.rows
        comparison_grouped = comparison_result.grouped_metrics
        comparison_info = {
            "available": True,
            "observation_count": int(len(comparison_frame)),
            "allow_subset_comparison": bool(allow_subset_comparison),
            "model_fingerprints_a": list(run.model_fingerprints),
            "model_fingerprints_b": list(comparison_run.model_fingerprints),
            "specification_fingerprints_a": list(run.specification_fingerprints),
            "specification_fingerprints_b": list(comparison_run.specification_fingerprints),
        }
    public_diagnostics = diagnostics.drop(columns=["__audit_id"], errors="ignore")
    manifest: dict[str, object] = {
        "schema_version": RESIDUAL_AUDIT_SCHEMA_VERSION,
        "tool_version": __version__,
        "input_paths": {
            **run.input_paths,
            **({f"run_b_{key}": value for key, value in comparison_run.input_paths.items()} if comparison_run is not None else {}),
        },
        "input_fingerprints": {
            **run.input_fingerprints,
            **({f"run_b_{key}": value for key, value in comparison_run.input_fingerprints.items()} if comparison_run is not None else {}),
        },
        "model_fingerprints": list(dict.fromkeys((*run.model_fingerprints, *(comparison_run.model_fingerprints if comparison_run else ())))),
        "specification_fingerprints": list(dict.fromkeys((*run.specification_fingerprints, *(comparison_run.specification_fingerprints if comparison_run else ())))),
        "observation_count": int(len(diagnostics)),
        "groupings": [str(value) for value in group_by],
        "thresholds": config.to_dict(),
        "is_holdout_evaluation": bool(is_holdout_evaluation),
        "contribution_analysis": contribution_info,
        "comparison": comparison_info,
        "warnings": [
            "Fit-observation analysis is not independent validation unless marked as a holdout evaluation.",
            "No observations were automatically removed.",
        ],
    }
    summary = _summary_markdown(
        diagnostics,
        grouped,
        support,
        contribution_info,
        comparison_frame,
        is_holdout_evaluation=is_holdout_evaluation,
    )
    return ResidualAuditResult(
        row_diagnostics=public_diagnostics,
        grouped_metrics=grouped,
        support_failures=support,
        contribution_summary=contribution_summary,
        comparison=comparison_frame,
        comparison_grouped_metrics=comparison_grouped,
        manifest=manifest,
        summary_markdown=summary,
    )


def write_residual_audit(
    run: ResidualAuditRun,
    *,
    output_directory: str | Path,
    group_by: Sequence[str] = (),
    config: ResidualAuditConfig = ResidualAuditConfig(),
    comparison_run: ResidualAuditRun | None = None,
    allow_subset_comparison: bool = False,
    is_holdout_evaluation: bool = False,
    force: bool = False,
) -> ResidualAuditResult:
    """Run an audit and persist all documented report files."""
    result = audit_run(
        run,
        group_by=group_by,
        config=config,
        comparison_run=comparison_run,
        allow_subset_comparison=allow_subset_comparison,
        is_holdout_evaluation=is_holdout_evaluation,
    )
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = (
        "audit_manifest.json",
        "summary.md",
        "row_diagnostics.csv",
        "grouped_metrics.csv",
        "support_failures.csv",
        "contribution_summary.csv",
        "run_comparison.csv",
        "comparison_grouped_metrics.csv",
    )
    existing = [output / name for name in names if (output / name).exists()]
    if existing and not force:
        raise FileExistsError(
            "residual-audit output already exists; choose a new directory or use force=True: "
            + ", ".join(str(path) for path in existing)
        )
    paths = {name: output / name for name in names}
    _write_csv(paths["row_diagnostics.csv"], result.row_diagnostics)
    _write_csv(paths["grouped_metrics.csv"], result.grouped_metrics)
    _write_csv(paths["support_failures.csv"], result.support_failures, columns=_EMPTY_SUPPORT_COLUMNS if result.support_failures.empty else None)
    _write_csv(paths["contribution_summary.csv"], result.contribution_summary, columns=(
        [
            "scope",
            "component",
            "row_count",
            "total_contribution",
            "nonzero_rows",
            "boundary_nonzero_rows",
            "od_zero_prediction_positive_rows",
            "boundary_dominates_rows",
        ] if result.contribution_summary.empty else None
    ))
    _write_csv(paths["run_comparison.csv"], result.comparison, columns=_COMPARISON_COLUMNS if result.comparison.empty else None)
    if result.comparison_grouped_metrics.empty:
        _write_csv(paths["comparison_grouped_metrics.csv"], result.comparison_grouped_metrics, columns=(
            [
                *group_by,
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
        ))
    else:
        _write_csv(paths["comparison_grouped_metrics.csv"], result.comparison_grouped_metrics)
    result.manifest["files"] = {name: str(path) for name, path in paths.items()}
    _write_json(paths["audit_manifest.json"], result.manifest)
    temporary = paths["summary.md"].with_name(".summary.md.tmp")
    temporary.write_text(result.summary_markdown, encoding="utf-8")
    temporary.replace(paths["summary.md"])
    return ResidualAuditResult(
        row_diagnostics=result.row_diagnostics,
        grouped_metrics=result.grouped_metrics,
        support_failures=result.support_failures,
        contribution_summary=result.contribution_summary,
        comparison=result.comparison,
        comparison_grouped_metrics=result.comparison_grouped_metrics,
        manifest=result.manifest,
        summary_markdown=result.summary_markdown,
        output_files=paths,
    )


# Descriptive aliases make the reusable API easy to discover while retaining
# the short ``audit_run`` name used internally.
audit_residuals = audit_run
write_audit = write_residual_audit
