"""Persistent detailed and executive reports for gravity-model estimates."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from public_transportation.inference.od_parameter_layout import ODParameterLayout

from .estimator import GravityEstimationResult
from .objective import GravityLikelihood
from .validation import (
    GravityAdequacyConfig,
    GravityAdequacyReport,
    GravityValidationMetadata,
    build_gravity_adequacy_report,
)


GRAVITY_DETAILED_REPORT_SCHEMA_VERSION = 1
_REPORT_FILES = (
    "parameters.csv",
    "full_od.csv",
    "predicted_measurements.csv",
    "residuals.csv",
    "grouped_residuals.csv",
    "report.json",
    "report.md",
    "executive_summary.md",
)
_METADATA_FIELDS = (
    "method_id",
    "measurement_type",
    "line",
    "direction",
    "stop",
    "observation_time",
    "time_period",
    "origin_zone",
    "destination_zone",
    "vehicle_journey",
)
_PARAMETER_FIELDS = (
    "parameter_index",
    "parameter_name",
    "raw_value",
    "physical_value",
)
_OD_FIELDS = (
    "od_index",
    "origin_stop_id",
    "destination_stop_id",
    "departure_time_bin",
    "estimated_demand",
    "cell_status",
    "inference_basis",
    "inference_score",
    "fixed_value",
    "prior_demand",
    "estimated_to_prior_ratio",
)


@dataclass(frozen=True, slots=True)
class GravityDetailedReport:
    """Summary and paths produced by :func:`write_gravity_detailed_report`."""

    schema_version: int
    report_fingerprint: str
    output_directory: Path
    files: Mapping[str, Path]
    adequacy: GravityAdequacyReport
    estimated_od_cells: int
    fixed_od_cells: int
    structural_zero_cells: int
    executive_messages: tuple[str, ...]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _metadata_rows(
    observations: np.ndarray, metadata: GravityValidationMetadata
) -> list[dict[str, object]]:
    rows = []
    for index, observed in enumerate(observations):
        row: dict[str, object] = {"row_index": index, "observed": float(observed)}
        for field in _METADATA_FIELDS:
            labels = getattr(metadata, field)
            row[field] = "" if labels is None else str(labels[index])
        rows.append(row)
    return rows


def _parameter_rows(result: GravityEstimationResult) -> list[dict[str, object]]:
    raw = np.asarray(result.raw_parameters, dtype=np.float64)
    physical = np.asarray(result.physical_parameters, dtype=np.float64)
    names = tuple(result.parameter_names)
    if raw.shape != physical.shape:
        raise ValueError("raw and physical parameter vectors have different shapes.")
    if names and len(names) != raw.size:
        raise ValueError("parameter_names does not match the parameter vector.")
    labels = names or tuple(f"parameter_{index}" for index in range(raw.size))
    if np.any(~np.isfinite(raw)) or np.any(~np.isfinite(physical)):
        raise ValueError("gravity parameter vectors must be finite.")
    return [
        {
            "parameter_index": index,
            "parameter_name": str(name),
            "raw_value": float(raw[index]),
            "physical_value": float(physical[index]),
        }
        for index, name in enumerate(labels)
    ]


def _od_rows(result: GravityEstimationResult, layout: ODParameterLayout) -> list[dict[str, object]]:
    demand = np.asarray(result.full_od_demand, dtype=np.float64)
    if demand.shape != (layout.num_od_total,) or np.any(~np.isfinite(demand)):
        raise ValueError("full_od_demand does not match the OD parameter layout.")
    free_indices = set(layout.free_od_indices)
    fixed_values = dict(zip(layout.fixed_od_indices, layout.fixed_od_values, strict=True))
    baselines = dict(zip(layout.free_od_indices, layout.free_baseline_values, strict=True))
    rows = []
    for index, key in enumerate(layout.od_keys):
        origin, destination, time_period = key
        value = float(demand[index])
        if index in free_indices:
            baseline = float(baselines[index])
            rows.append(
                {
                    "od_index": index,
                    "origin_stop_id": str(origin),
                    "destination_stop_id": str(destination),
                    "departure_time_bin": str(time_period),
                    "estimated_demand": value,
                    "cell_status": "estimated_free",
                    "inference_basis": "observed_flows_and_model_regularization",
                    "inference_score": 1.0,
                    "fixed_value": "",
                    "prior_demand": baseline,
                    "estimated_to_prior_ratio": value / baseline,
                }
            )
        else:
            fixed = float(fixed_values[index])
            rows.append(
                {
                    "od_index": index,
                    "origin_stop_id": str(origin),
                    "destination_stop_id": str(destination),
                    "departure_time_bin": str(time_period),
                    "estimated_demand": value,
                    "cell_status": "fixed_zero" if fixed == 0.0 else "fixed_positive",
                    "inference_basis": "fixed_input_or_structural_zero_policy",
                    "inference_score": 0.0,
                    "fixed_value": fixed,
                    "prior_demand": "",
                    "estimated_to_prior_ratio": "",
                }
            )
    return rows


def _dispersion(result: GravityEstimationResult, likelihood: str) -> float | None:
    if likelihood == "poisson":
        return None
    names = tuple(result.parameter_names)
    try:
        index = names.index("dispersion")
    except ValueError as error:
        raise ValueError("negative-binomial result has no dispersion parameter.") from error
    value = float(result.physical_parameters[index])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("negative-binomial dispersion must be finite and positive.")
    return value


def _adequacy_json(report: GravityAdequacyReport) -> dict[str, object]:
    return {
        "schema_version": report.schema_version,
        "model_fingerprint": report.model_fingerprint,
        "report_fingerprint": report.report_fingerprint,
        "measurements": report.measurements,
        "observed_total": report.observed_total,
        "modeled_total": report.modeled_total,
        "negative_binomial_deviance": report.negative_binomial_deviance,
        "poisson_deviance": report.poisson_deviance,
        "mae": report.mae,
        "rmse": report.rmse,
        "weighted_rmse": report.weighted_rmse,
        "threshold_counts": [list(item) for item in report.threshold_counts],
        "observed_predicted_quantiles": [list(item) for item in report.observed_predicted_quantiles],
        "grouped_summaries": [asdict(item) for item in report.grouped_summaries],
        "journey_correlations": [asdict(item) for item in report.journey_correlations],
        "findings": asdict(report.findings),
    }


def _executive_messages(
    *,
    result: GravityEstimationResult,
    adequacy: GravityAdequacyReport,
    estimated_od_cells: int,
    fixed_od_cells: int,
    structural_zero_cells: int,
) -> tuple[str, ...]:
    messages = [
        (
            f"The fit is {result.status} with acceptance={result.acceptance}; "
            f"the optimizer used {result.iterations} iterations."
        ),
        (
            f"Observed flow totals {adequacy.observed_total:.6g}, while modeled flows total "
            f"{adequacy.modeled_total:.6g}; RMSE is {adequacy.rmse:.6g} "
            f"and MAE is {adequacy.mae:.6g}."
        ),
        (
            f"The OD table contains {estimated_od_cells} free cells estimated from the "
            f"observed flows and {fixed_od_cells} fixed cells, including "
            f"{structural_zero_cells} structural zeros."
        ),
    ]
    threshold_three = next(
        (count for threshold, count, _ in adequacy.threshold_counts if threshold == 3.0),
        None,
    )
    if threshold_three is not None:
        messages.append(
            f"{threshold_three} observations have an absolute standardized residual above 3."
        )
    messages.extend(adequacy.findings.messages)
    messages.append(
        "These are full-data adequacy diagnostics, not an independent holdout validation."
    )
    return tuple(messages)


def _executive_markdown(
    *,
    result: GravityEstimationResult,
    adequacy: GravityAdequacyReport,
    messages: tuple[str, ...],
) -> str:
    lines = [
        "# Executive summary",
        "",
        "## Main take-home messages",
        "",
    ]
    lines.extend(f"- {message}" for message in messages)
    lines.extend(
        [
            "",
            "## Key figures",
            "",
            "| Figure | Value |",
            "|---|---:|",
            f"| Fit status | `{result.status}` |",
            f"| Acceptance | `{result.acceptance}` |",
            f"| Objective | {result.objective:.9g} |",
            f"| Measurements | {adequacy.measurements} |",
            f"| Observed total | {adequacy.observed_total:.9g} |",
            f"| Modeled total | {adequacy.modeled_total:.9g} |",
            f"| RMSE | {adequacy.rmse:.9g} |",
            f"| MAE | {adequacy.mae:.9g} |",
            "",
            "The complete OD and measurement-level tables are in the accompanying CSV files.",
            "",
        ]
    )
    return "\n".join(lines)


def _detailed_markdown(
    *,
    result: GravityEstimationResult,
    adequacy: GravityAdequacyReport,
    summary: Mapping[str, object],
) -> str:
    lines = [
        "# Detailed gravity-fit report",
        "",
        "This report evaluates the fitted model on all calibration observations. It is not a holdout evaluation.",
        "",
        "## Fit and convergence",
        "",
        f"- Status: `{result.status}`; success: `{result.success}`; acceptance: `{result.acceptance}`.",
        f"- Optimizer: `{result.optimizer}`; iterations: `{result.iterations}`; evaluations: `{result.optimizer_evaluations}`.",
        f"- Objective: `{result.objective:.9g}`; scaled gradient infinity norm: `{result.scaled_gradient_inf_norm}`.",
        f"- Model fingerprint: `{result.model_fingerprint}`.",
        "",
        "## Adequacy metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Measurements | {adequacy.measurements} |",
        f"| Observed total | {adequacy.observed_total:.9g} |",
        f"| Modeled total | {adequacy.modeled_total:.9g} |",
        f"| MAE | {adequacy.mae:.9g} |",
        f"| RMSE | {adequacy.rmse:.9g} |",
        f"| Weighted RMSE | {adequacy.weighted_rmse:.9g} |",
        f"| Poisson deviance | {adequacy.poisson_deviance:.9g} |",
        f"| Negative-binomial deviance | {adequacy.negative_binomial_deviance} |",
        "",
        "## Inference indicator",
        "",
        "In `full_od.csv`, `inference_score=0` identifies cells fixed by the demand input or structural-zero policy, while `inference_score=1` identifies free cells estimated using the observed flows and model regularization. This is a structural classification, not a standard error or posterior probability.",
        "",
        "## Findings",
        "",
    ]
    lines.extend(f"- {message}" for message in summary["executive_messages"])
    lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `parameters.csv`: fitted raw and physical values for every model parameter.",
            "- `full_od.csv`: every canonical origin/destination/departure-time cell.",
            "- `predicted_measurements.csv`: observed and modeled value for every measurement.",
            "- `residuals.csv`: residual, variance, standardized residual, and relative residual.",
            "- `grouped_residuals.csv`: residual diagnostics by measurement attribute.",
            "- `executive_summary.md`: short decision-oriented summary.",
            "- `report.json`: machine-readable comprehensive summary.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_gravity_detailed_report(
    *,
    result: GravityEstimationResult,
    observations: object,
    predicted_measurements: object,
    od_layout: ODParameterLayout,
    metadata: GravityValidationMetadata | None = None,
    likelihood: GravityLikelihood | str = GravityLikelihood.NEGATIVE_BINOMIAL,
    output_directory: str | Path,
    adequacy_config: GravityAdequacyConfig = GravityAdequacyConfig(),
    provenance: Mapping[str, object] | None = None,
    force: bool = False,
) -> GravityDetailedReport:
    """Write CSV, JSON, Markdown, and executive reports for a gravity fit.

    The function consumes persisted fit and prediction vectors. It does not
    activate an assignment operator, rerun an optimizer, or change the
    objective function.
    """
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = [output / name for name in _REPORT_FILES if (output / name).exists()]
    if existing and not force:
        raise FileExistsError(
            "report output already exists; choose a new directory or use force=True: "
            + ", ".join(str(path) for path in existing)
        )

    observed = np.asarray(observations, dtype=np.float64)
    modeled = np.asarray(predicted_measurements, dtype=np.float64)
    persisted_modeled = np.asarray(result.predicted_measurements, dtype=np.float64)
    if persisted_modeled.shape != modeled.shape:
        raise ValueError(
            "predicted_measurements does not match the fit result dimension."
        )
    np.testing.assert_allclose(
        persisted_modeled,
        modeled,
        rtol=5.0e-6,
        atol=5.0e-6,
        err_msg="persisted validation predictions do not match the persisted fit result",
    )
    selected_metadata = metadata or GravityValidationMetadata(observed.size)
    try:
        likelihood_value = str(likelihood.value)
    except AttributeError:
        likelihood_value = str(likelihood)
    adequacy = build_gravity_adequacy_report(
        observations=observed,
        modeled=modeled,
        model_fingerprint=result.model_fingerprint,
        likelihood=likelihood_value,
        dispersion=_dispersion(result, likelihood_value),
        metadata=selected_metadata,
        config=adequacy_config,
    )

    od_rows = _od_rows(result, od_layout)
    measurement_rows = _metadata_rows(observed, selected_metadata)
    residual = adequacy.residual
    standardized = adequacy.standardized_nb_residual
    variance = np.maximum(modeled, np.finfo(np.float64).tiny)
    if likelihood_value == "negative_binomial":
        dispersion = _dispersion(result, likelihood_value)
        assert dispersion is not None
        variance = modeled + modeled**2 / dispersion
    for row, prediction, residual_value, standardized_value, variance_value in zip(
        measurement_rows,
        modeled,
        residual,
        standardized,
        variance,
        strict=True,
    ):
        row.update(
            {
                "modeled": float(prediction),
                "residual": float(residual_value),
                "variance": float(variance_value),
                "standardized_residual": float(standardized_value),
                "absolute_residual": float(abs(residual_value)),
                "relative_residual": (
                    ""
                    if float(row["observed"]) == 0.0
                    else float(residual_value) / float(row["observed"])
                ),
            }
        )

    prediction_fields = ["row_index", *_METADATA_FIELDS, "observed", "modeled"]
    residual_fields = [
        *prediction_fields,
        "residual",
        "variance",
        "standardized_residual",
        "absolute_residual",
        "relative_residual",
    ]
    parameter_rows = _parameter_rows(result)
    _write_csv(output / "parameters.csv", list(_PARAMETER_FIELDS), parameter_rows)
    _write_csv(output / "full_od.csv", list(_OD_FIELDS), od_rows)
    _write_csv(
        output / "predicted_measurements.csv",
        prediction_fields,
        [{field: row[field] for field in prediction_fields} for row in measurement_rows],
    )
    _write_csv(
        output / "residuals.csv",
        residual_fields,
        [{field: row[field] for field in residual_fields} for row in measurement_rows],
    )
    grouped_rows = [asdict(item) for item in adequacy.grouped_summaries]
    _write_csv(
        output / "grouped_residuals.csv",
        list(grouped_rows[0]) if grouped_rows else ["grouping", "label"],
        grouped_rows,
    )

    free_cells = len(od_layout.free_od_indices)
    fixed_cells = len(od_layout.fixed_od_indices)
    structural_zero_cells = len(od_layout.fixed_zero_indices)
    executive_messages = _executive_messages(
        result=result,
        adequacy=adequacy,
        estimated_od_cells=free_cells,
        fixed_od_cells=fixed_cells,
        structural_zero_cells=structural_zero_cells,
    )
    summary: dict[str, object] = {
        "schema_version": GRAVITY_DETAILED_REPORT_SCHEMA_VERSION,
        "status": "completed",
        "model_fingerprint": result.model_fingerprint,
        "fit": {
            "status": result.status,
            "success": result.success,
            "acceptance": result.acceptance,
            "message": result.message,
            "optimizer": result.optimizer,
            "iterations": result.iterations,
            "evaluations": result.optimizer_evaluations,
            "objective": result.objective,
            "scaled_gradient_inf_norm": result.scaled_gradient_inf_norm,
        },
        "od": {
            "total_cells": od_layout.num_od_total,
            "estimated_free_cells": free_cells,
            "fixed_cells": fixed_cells,
            "structural_zero_cells": structural_zero_cells,
            "inference_indicator": {
                "definition": "0=fixed by input/structural policy; 1=estimated free cell",
                "limitation": "structural classification, not a standard error or posterior probability",
            },
        },
        "adequacy": _adequacy_json(adequacy),
        "executive_messages": list(executive_messages),
        "provenance": dict(provenance or {}),
        "files": {name: str(output / name) for name in _REPORT_FILES},
    }
    report_hash = hashlib.sha256(
        json.dumps(summary, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    summary["report_fingerprint"] = report_hash
    _write_json(output / "report.json", summary)
    _write_text(
        output / "executive_summary.md",
        _executive_markdown(result=result, adequacy=adequacy, messages=executive_messages),
    )
    _write_text(output / "report.md", _detailed_markdown(result=result, adequacy=adequacy, summary={"executive_messages": executive_messages}))
    return GravityDetailedReport(
        GRAVITY_DETAILED_REPORT_SCHEMA_VERSION,
        report_hash,
        output,
        {name: output / name for name in _REPORT_FILES},
        adequacy,
        free_cells,
        fixed_cells,
        structural_zero_cells,
        executive_messages,
    )
