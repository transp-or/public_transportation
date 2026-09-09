"""Portable, validated gravity-result bundles for independent viewers.

The bundle API deliberately stops at data access.  It does not import or
activate an assignment operator and it never recomputes the fitted model.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from public_transportation.inference.block_coordinate._canonical import canonical_json
from public_transportation.inference.od_parameter_layout import ODParameterLayout

from .estimator import GravityEstimationResult
from .identifiability import (
    GravityODIdentifiability,
    read_gravity_od_identifiability,
    write_gravity_od_identifiability,
)
from .objective import GravityLikelihood
from .reporting import (
    GRAVITY_REPORT_PROVENANCE_FIELDS,
    GravityDetailedReport,
)
from .specification import GravityModelSpecification
from .validation import GravityAdequacyReport, GravityValidationMetadata


GRAVITY_VIEWER_BUNDLE_SCHEMA_VERSION = 1
GRAVITY_VIEWER_BUNDLE_TYPE = "gravity_viewer_bundle"
_MODEL_SPECIFICATION_SCHEMA_VERSION = 1
_MODEL_SPECIFICATION_FILE = "model_specification.json"
_IDENTIFIABILITY_MESSAGE = (
    "OD identifiability diagnostics were not available; inference_score remains "
    "only a structural fixed/free indicator."
)
_TABLE_FILES = (
    "full_od.csv",
    "predicted_measurements.csv",
    "residuals.csv",
    "grouped_residuals.csv",
    "parameters.csv",
)
_OPTIONAL_DERIVED_FILES = ("od_map.csv", "stop_summary.csv")
_REPORT_FILES = ("report.json", "executive_summary.md")
_OPTIONAL_REPORT_FILES = ("report.md",)
_FULL_OD_COLUMNS = (
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
    "count_information_share",
    "regularization_information_share",
    "auxiliary_information_share",
    "local_information_variance",
    "identifiability_class",
    "identifiability_reason",
)
_PREDICTED_COLUMNS = (
    "row_index",
    "measurement_type",
    "line",
    "direction",
    "stop",
    "observation_time",
    "time_period",
    "observed",
    "modeled",
)
_RESIDUAL_COLUMNS = (*_PREDICTED_COLUMNS, "residual", "variance", "standardized_residual")
_PARAMETER_COLUMNS = ("parameter_index", "parameter_name", "raw_value", "physical_value")
_OD_MAP_COLUMNS = (
    "origin_stop_id",
    "destination_stop_id",
    "departure_time_bin",
    "observed_demand",
    "modeled_demand",
    "residual",
    "fixed_free_status",
    "identifiability_class",
)
_STOP_SUMMARY_COLUMNS = (
    "stop_id",
    "observed_total",
    "modeled_total",
    "residual",
    "incoming_total",
    "outgoing_total",
)
_NETWORK_FILES = ("stops.csv", "lines.csv", "trips.csv", "stop_times.csv")
_CANONICAL_PROVENANCE = tuple(GRAVITY_REPORT_PROVENANCE_FIELDS)


def _json_safe(value: object) -> object:
    """Convert public result/config values to strict JSON-compatible values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("bundle metadata must not contain non-finite numbers.")
        return value
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {path} has no header.")
        return list(reader.fieldnames), list(reader)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    """Write a small deterministic CSV atomically."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _csv_number(value: object) -> float | None:
    """Parse an optional finite number used by display-only summaries."""
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _csv_number_text(value: float | None) -> str:
    """Return a deterministic CSV representation for an optional number."""
    return "" if value is None else format(value, ".17g")


def _observed_od_values(
    full_od_rows: Sequence[Mapping[str, str]],
    report_payload: Mapping[str, object],
) -> list[float | None]:
    """Return optional observed OD values without pretending counts are OD data.

    The core reports normally contain only modeled OD demand.  A case-specific
    report may additionally persist ``observed_demand`` (or ``observed``) in
    ``full_od.csv`` or an aligned ``observed_od_demand`` array in ``report.json``.
    When neither is present, the display field remains empty rather than
    substituting modeled or prior demand.
    """
    report_values = report_payload.get("observed_od_demand")
    if isinstance(report_values, Sequence) and not isinstance(report_values, (str, bytes)):
        if len(report_values) == len(full_od_rows):
            return [_csv_number(value) for value in report_values]
    values: list[float | None] = []
    for row in full_od_rows:
        value = row.get("observed_demand")
        if value is None or str(value).strip() == "":
            value = row.get("observed")
        values.append(_csv_number(value))
    return values


def _fixed_free_status(value: object) -> str:
    """Normalize report cell status to the viewer's fixed/free vocabulary."""
    status = str(value or "").strip()
    if status.lower().startswith("fixed"):
        return "fixed"
    if status.lower() == "free" or status.lower().endswith("_free"):
        return "free"
    return status


def _write_od_map(
    destination: Path,
    *,
    full_od_rows: Sequence[Mapping[str, str]],
    report_payload: Mapping[str, object],
) -> int:
    """Write the optional display-oriented OD aggregation."""
    required = {"origin_stop_id", "destination_stop_id", "departure_time_bin", "estimated_demand"}
    if not required.issubset(full_od_rows[0].keys() if full_od_rows else ()):
        return 0
    observed_values = _observed_od_values(full_od_rows, report_payload)
    rows: list[dict[str, object]] = []
    for index, source in enumerate(full_od_rows):
        origin = str(source.get("origin_stop_id", ""))
        destination_id = str(source.get("destination_stop_id", ""))
        if not origin or not destination_id:
            continue
        modeled = _csv_number(source.get("estimated_demand"))
        if modeled is None:
            continue
        observed = observed_values[index]
        residual = None if observed is None else observed - modeled
        rows.append(
            {
                "origin_stop_id": origin,
                "destination_stop_id": destination_id,
                "departure_time_bin": str(source.get("departure_time_bin", "")),
                "observed_demand": _csv_number_text(observed),
                "modeled_demand": _csv_number_text(modeled),
                "residual": _csv_number_text(residual),
                "fixed_free_status": _fixed_free_status(source.get("cell_status")),
                "identifiability_class": str(source.get("identifiability_class", "")),
            }
        )
    if not rows:
        return 0
    _write_csv(destination, list(_OD_MAP_COLUMNS), rows)
    return len(rows)


def _write_stop_summary(
    destination: Path,
    *,
    predicted_fields: Sequence[str],
    predicted_rows: Sequence[Mapping[str, str]],
    network_stops: Sequence[Mapping[str, str]] = (),
) -> int:
    """Write display-oriented observed/modelled stop totals."""
    stop_field = "stop" if "stop" in predicted_fields else "stop_id" if "stop_id" in predicted_fields else None
    if stop_field is None:
        return 0
    totals: dict[str, dict[str, float]] = {}
    for row in predicted_rows:
        stop = str(row.get(stop_field, "")).strip()
        if not stop:
            continue
        observed = _csv_number(row.get("observed"))
        modeled = _csv_number(row.get("modeled"))
        if observed is None or modeled is None:
            continue
        item = totals.setdefault(
            stop,
            {"observed_total": 0.0, "modeled_total": 0.0, "incoming_total": 0.0, "outgoing_total": 0.0},
        )
        item["observed_total"] += observed
        item["modeled_total"] += modeled
        measurement_type = str(row.get("measurement_type", "")).strip().lower()
        if measurement_type in {"alighting", "incoming"}:
            item["incoming_total"] += observed
        elif measurement_type in {"boarding", "outgoing"}:
            item["outgoing_total"] += observed
    for stop in network_stops:
        stop_id = str(stop.get("stop_id", "")).strip()
        if stop_id:
            totals.setdefault(
                stop_id,
                {"observed_total": 0.0, "modeled_total": 0.0, "incoming_total": 0.0, "outgoing_total": 0.0},
            )
    if not totals:
        return 0
    rows = []
    for stop_id in sorted(totals):
        item = totals[stop_id]
        rows.append(
            {
                "stop_id": stop_id,
                "observed_total": _csv_number_text(item["observed_total"]),
                "modeled_total": _csv_number_text(item["modeled_total"]),
                "residual": _csv_number_text(item["observed_total"] - item["modeled_total"]),
                "incoming_total": _csv_number_text(item["incoming_total"]),
                "outgoing_total": _csv_number_text(item["outgoing_total"]),
            }
        )
    _write_csv(destination, list(_STOP_SUMMARY_COLUMNS), rows)
    return len(rows)


def _require_columns(path: Path, fieldnames: Sequence[str], required: Sequence[str]) -> None:
    missing = [name for name in required if name not in fieldnames]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}.")


def _float_column(rows: Sequence[Mapping[str, str]], name: str, path: Path) -> np.ndarray:
    values: list[float] = []
    for row_number, row in enumerate(rows, start=2):
        raw = row.get(name, "")
        if raw == "":
            values.append(np.nan)
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}:{row_number} has a non-numeric {name!r}.") from error
    return np.asarray(values, dtype=np.float64)


def _integer_column(rows: Sequence[Mapping[str, str]], name: str, path: Path) -> np.ndarray:
    values = _float_column(rows, name, path)
    if np.any(~np.isfinite(values)) or np.any(values != np.floor(values)):
        raise ValueError(f"{path} column {name!r} must contain finite integers.")
    return values.astype(np.int64)


def _nonempty_values(rows: Sequence[Mapping[str, str]], name: str) -> set[str]:
    return {str(row.get(name, "")) for row in rows if str(row.get(name, "")) != ""}


def _first_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> str | None:
    return next((name for name in candidates if name in fieldnames), None)


def _report_path(report: GravityDetailedReport, name: str) -> Path:
    try:
        candidate = report.files[name]
    except (KeyError, TypeError):
        candidate = report.output_directory / name
    path = Path(candidate).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"gravity report file is missing: {path}")
    return path


def _result_specification(
    fit_result: GravityEstimationResult,
    explicit: GravityModelSpecification | Mapping[str, object] | None,
) -> tuple[dict[str, object], str]:
    fit_raw = fit_result.model_specification
    raw = explicit if explicit is not None else fit_raw
    if raw is None:
        raise ValueError(
            "fit_result does not contain an exact model specification; supply "
            "model_specification explicitly before exporting a viewer bundle."
        )
    if isinstance(raw, GravityModelSpecification):
        specification = raw.to_dict()
        computed_fingerprint = raw.fingerprint
    elif isinstance(raw, Mapping):
        payload = dict(raw)
        try:
            parsed = GravityModelSpecification.from_dict(payload)
        except (TypeError, ValueError) as error:
            raise ValueError("model_specification is not a valid gravity specification.") from error
        specification = parsed.to_dict()
        if canonical_json(payload) != canonical_json(specification):
            raise ValueError(
                "model_specification is not the canonical serialized specification."
            )
        computed_fingerprint = parsed.fingerprint
    else:
        raise TypeError("model_specification must be a GravityModelSpecification or mapping.")
    if explicit is not None and fit_raw is not None:
        if isinstance(fit_raw, GravityModelSpecification):
            fit_specification = fit_raw.to_dict()
            fit_fingerprint = fit_raw.fingerprint
        elif isinstance(fit_raw, Mapping):
            fit_payload = dict(fit_raw)
            try:
                fit_parsed = GravityModelSpecification.from_dict(fit_payload)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "fit_result model_specification is not a valid gravity specification."
                ) from error
            fit_specification = fit_parsed.to_dict()
            if canonical_json(fit_payload) != canonical_json(fit_specification):
                raise ValueError(
                    "fit_result model_specification is not the canonical serialized specification."
                )
            fit_fingerprint = fit_parsed.fingerprint
        else:
            raise TypeError(
                "fit_result model_specification must be a GravityModelSpecification or mapping."
            )
        if fit_specification != specification or fit_fingerprint != computed_fingerprint:
            raise ValueError(
                "explicit model_specification differs from the fitted result."
            )
    stored = str(fit_result.specification_fingerprint or "")
    if stored and stored != computed_fingerprint:
        raise ValueError(
            "model specification fingerprint differs from the fitted result: "
            f"result={stored!r}, specification={computed_fingerprint!r}."
        )
    return specification, stored or computed_fingerprint


def _source_provenance(
    report_payload: Mapping[str, object],
    fit_result: GravityEstimationResult,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    report_provenance = report_payload.get("provenance", {})
    if not isinstance(report_provenance, Mapping):
        raise ValueError("report.json provenance must be a mapping.")
    metadata_provenance = metadata.get("provenance", {})
    if not isinstance(metadata_provenance, Mapping):
        raise ValueError("bundle metadata provenance must be a mapping.")
    result_fallbacks: dict[str, object] = {
        "model_fingerprint": fit_result.model_fingerprint,
        "specification_fingerprint": fit_result.specification_fingerprint or None,
        "artifact_identity_fingerprint": (
            fit_result.direct_operator_artifact_fingerprint or None
        ),
    }
    result: dict[str, object] = {}
    for field in _CANONICAL_PROVENANCE:
        result[field] = report_provenance.get(
            field, metadata_provenance.get(field, None)
        )
    for field, value in result_fallbacks.items():
        if value is not None:
            result[field] = value
    result["model_fingerprint"] = fit_result.model_fingerprint
    result["specification_fingerprint"] = fit_result.specification_fingerprint or result.get(
        "specification_fingerprint"
    )
    return result


def _validation_provenance(validation_result: object) -> dict[str, object]:
    if isinstance(validation_result, GravityAdequacyReport):
        return {
            "model_fingerprint": validation_result.model_fingerprint,
            "specification_fingerprint": validation_result.specification_fingerprint,
            "model_specification": validation_result.model_specification,
            "report_fingerprint": validation_result.report_fingerprint,
        }
    if isinstance(validation_result, Mapping):
        adequacy = validation_result.get("adequacy")
        if isinstance(adequacy, Mapping):
            merged = dict(validation_result)
            merged.update(adequacy)
            validation_result = merged
        provenance = validation_result.get("provenance", {})
        if provenance is None:
            provenance = {}
        if not isinstance(provenance, Mapping):
            raise ValueError("validation_result provenance must be a mapping.")
        provenance = dict(provenance)
        for name in (
            "model_fingerprint",
            "specification_fingerprint",
            "model_specification",
            "report_fingerprint",
        ):
            if name in validation_result:
                provenance[name] = validation_result[name]
        return dict(provenance)
    values: dict[str, object] = {}
    for name in (
        "model_fingerprint",
        "specification_fingerprint",
        "model_specification",
        "report_fingerprint",
    ):
        value = getattr(validation_result, name, None)
        if value is not None:
            values[name] = value
    return values


def _validate_table_inputs(
    staging: Path,
    fit_result: GravityEstimationResult,
    report_payload: Mapping[str, object],
    validation_result: object,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    fields, full_od = _read_csv(staging / "full_od.csv")
    _require_columns(staging / "full_od.csv", fields, _FULL_OD_COLUMNS)
    od_indices = _integer_column(full_od, "od_index", staging / "full_od.csv")
    if len(np.unique(od_indices)) != len(od_indices):
        raise ValueError("full_od.csv contains duplicate od_index values.")
    estimated = _float_column(full_od, "estimated_demand", staging / "full_od.csv")
    if estimated.size != fit_result.full_od_demand.size:
        raise ValueError("full_od.csv row count does not match fitted full OD demand.")
    np.testing.assert_allclose(estimated, np.asarray(fit_result.full_od_demand), rtol=1e-8, atol=1e-8)
    counts["full_od.csv"] = len(full_od)

    fields, predictions = _read_csv(staging / "predicted_measurements.csv")
    _require_columns(staging / "predicted_measurements.csv", fields, _PREDICTED_COLUMNS)
    row_indices = _integer_column(
        predictions, "row_index", staging / "predicted_measurements.csv"
    )
    if len(np.unique(row_indices)) != len(row_indices):
        raise ValueError("predicted_measurements.csv contains duplicate row_index values.")
    modeled = _float_column(predictions, "modeled", staging / "predicted_measurements.csv")
    if modeled.shape != np.asarray(fit_result.predicted_measurements).shape:
        raise ValueError("predicted_measurements.csv length does not match fitted predictions.")
    np.testing.assert_allclose(
        modeled, np.asarray(fit_result.predicted_measurements), rtol=1e-8, atol=1e-8
    )
    counts["predicted_measurements.csv"] = len(predictions)

    fields, residuals = _read_csv(staging / "residuals.csv")
    _require_columns(staging / "residuals.csv", fields, _RESIDUAL_COLUMNS)
    residual_row_indices = _integer_column(residuals, "row_index", staging / "residuals.csv")
    if not np.array_equal(residual_row_indices, row_indices):
        raise ValueError("residuals.csv row_index values differ from predicted_measurements.csv.")
    residual_values = _float_column(residuals, "residual", staging / "residuals.csv")
    observed = _float_column(residuals, "observed", staging / "residuals.csv")
    residual_modeled = _float_column(residuals, "modeled", staging / "residuals.csv")
    np.testing.assert_allclose(residual_values, observed - residual_modeled, rtol=1e-8, atol=1e-8)
    counts["residuals.csv"] = len(residuals)

    fields, grouped = _read_csv(staging / "grouped_residuals.csv")
    if not grouped:
        raise ValueError("grouped_residuals.csv must contain its header and at least one row.")
    counts["grouped_residuals.csv"] = len(grouped)

    fields, parameters = _read_csv(staging / "parameters.csv")
    _require_columns(staging / "parameters.csv", fields, _PARAMETER_COLUMNS)
    parameter_indices = _integer_column(parameters, "parameter_index", staging / "parameters.csv")
    if not np.array_equal(parameter_indices, np.arange(parameter_indices.size)):
        raise ValueError("parameters.csv parameter_index values must be consecutive.")
    raw = _float_column(parameters, "raw_value", staging / "parameters.csv")
    physical = _float_column(parameters, "physical_value", staging / "parameters.csv")
    np.testing.assert_allclose(raw, np.asarray(fit_result.raw_parameters), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(
        physical, np.asarray(fit_result.physical_parameters), rtol=1e-8, atol=1e-8
    )
    counts["parameters.csv"] = len(parameters)

    report_model = report_payload.get("model_fingerprint")
    if report_model != fit_result.model_fingerprint:
        raise ValueError(
            "report.json model_fingerprint differs from the fitted result: "
            f"report={report_model!r}, fit={fit_result.model_fingerprint!r}."
        )
    validation_provenance = _validation_provenance(validation_result)
    validation_model = validation_provenance.get("model_fingerprint")
    if validation_model is not None and validation_model != fit_result.model_fingerprint:
        raise ValueError(
            "validation provenance model_fingerprint differs from the fitted result: "
            f"validation={validation_model!r}, fit={fit_result.model_fingerprint!r}."
        )
    fit_specification, fit_specification_fingerprint = _result_specification(
        fit_result, None
    )
    report_specification = report_payload.get("model_specification")
    if not isinstance(report_specification, Mapping):
        raise ValueError("report.json is missing model_specification.")
    try:
        parsed_report_specification = GravityModelSpecification.from_dict(
            dict(report_specification)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("report.json model_specification is invalid.") from error
    if canonical_json(dict(report_specification)) != canonical_json(
        parsed_report_specification.to_dict()
    ):
        raise ValueError("report.json model_specification is not canonical.")
    report_specification_fingerprint = report_payload.get(
        "specification_fingerprint"
    )
    if report_specification_fingerprint != parsed_report_specification.fingerprint:
        raise ValueError(
            "report.json specification_fingerprint does not match its "
            "serialized model_specification."
        )
    if (
        report_specification_fingerprint != fit_specification_fingerprint
        or canonical_json(dict(report_specification))
        != canonical_json(fit_specification)
    ):
        raise ValueError(
            "report.json model specification differs from the fitted result."
        )
    validation_specification = validation_provenance.get("model_specification")
    validation_specification_fingerprint = validation_provenance.get(
        "specification_fingerprint"
    )
    if not isinstance(validation_specification, Mapping):
        raise ValueError("validation provenance is missing model_specification.")
    if validation_specification_fingerprint != fit_specification_fingerprint:
        raise ValueError(
            "validation provenance specification_fingerprint differs from the "
            "fitted result."
        )
    if canonical_json(dict(validation_specification)) != canonical_json(
        fit_specification
    ):
        raise ValueError(
            "validation provenance model_specification differs from the fitted result."
        )
    return counts


def _validate_network(
    staging: Path,
    table_counts: Mapping[str, int],
) -> dict[str, int]:
    network_dir = staging / "network"
    if not network_dir.exists():
        return {}
    files = sorted(path.name for path in network_dir.iterdir() if path.is_file())
    unknown = [name for name in files if name not in _NETWORK_FILES]
    if unknown:
        raise ValueError(f"unsupported network snapshot files: {', '.join(unknown)}.")
    if "stops.csv" not in files:
        raise ValueError("network snapshot must include stops.csv.")
    stop_fields, stops = _read_csv(network_dir / "stops.csv")
    _require_columns(network_dir / "stops.csv", stop_fields, ("stop_id", "name", "lat", "lon"))
    stop_ids = [str(row["stop_id"]) for row in stops]
    if len(set(stop_ids)) != len(stop_ids):
        raise ValueError("network/stops.csv contains duplicate stop_id values.")
    lat = _float_column(stops, "lat", network_dir / "stops.csv")
    lon = _float_column(stops, "lon", network_dir / "stops.csv")
    if (
        np.any(~np.isfinite(lat))
        or np.any(~np.isfinite(lon))
        or np.any((lat < -90) | (lat > 90))
        or np.any((lon < -180) | (lon > 180))
    ):
        raise ValueError("network/stops.csv contains invalid latitude/longitude values.")
    stop_set = set(stop_ids)

    od_fields, od_rows = _read_csv(staging / "full_od.csv")
    report_stops = _nonempty_values(od_rows, "origin_stop_id") | _nonempty_values(
        od_rows, "destination_stop_id"
    )
    pred_fields, pred_rows = _read_csv(staging / "predicted_measurements.csv")
    report_stops |= _nonempty_values(pred_rows, "stop")
    missing_stops = sorted(report_stops - stop_set)
    if missing_stops:
        raise ValueError(
            "report references stop IDs absent from network/stops.csv: "
            + ", ".join(missing_stops[:10])
        )

    counts = {"network/stops.csv": len(stops)}
    line_ids: set[str] = set()
    if "lines.csv" in files:
        fields, rows = _read_csv(network_dir / "lines.csv")
        line_column = _first_column(fields, ("line_id", "route_id", "line", "id"))
        if line_column is None:
            raise ValueError("network/lines.csv must contain line_id or route_id.")
        line_ids = _nonempty_values(rows, line_column)
        if len(line_ids) != len(rows):
            raise ValueError("network/lines.csv contains duplicate or empty line identifiers.")
        counts["network/lines.csv"] = len(rows)
        measurement_lines = _nonempty_values(pred_rows, "line")
        missing_lines = sorted(measurement_lines - line_ids)
        if missing_lines:
            raise ValueError(
                "predicted measurements reference line IDs absent from network/lines.csv: "
                + ", ".join(missing_lines[:10])
            )
    trip_ids: set[str] = set()
    if "trips.csv" in files:
        fields, rows = _read_csv(network_dir / "trips.csv")
        trip_column = _first_column(fields, ("trip_id", "trip", "id"))
        if trip_column is None:
            raise ValueError("network/trips.csv must contain trip_id.")
        trip_ids = _nonempty_values(rows, trip_column)
        if len(trip_ids) != len(rows):
            raise ValueError("network/trips.csv contains duplicate or empty trip identifiers.")
        if line_ids:
            line_column = _first_column(fields, ("line_id", "route_id", "line"))
            if line_column is not None:
                missing = sorted(_nonempty_values(rows, line_column) - line_ids)
                if missing:
                    raise ValueError(
                        "network/trips.csv references line IDs absent from lines.csv: "
                        + ", ".join(missing[:10])
                    )
        counts["network/trips.csv"] = len(rows)
    if "stop_times.csv" in files:
        fields, rows = _read_csv(network_dir / "stop_times.csv")
        trip_column = _first_column(fields, ("trip_id", "trip"))
        stop_column = _first_column(fields, ("stop_id", "stop"))
        if trip_column is None or stop_column is None:
            raise ValueError("network/stop_times.csv must contain trip_id and stop_id.")
        if trip_ids:
            missing = sorted(_nonempty_values(rows, trip_column) - trip_ids)
            if missing:
                raise ValueError(
                    "network/stop_times.csv references unknown trip IDs: "
                    + ", ".join(missing[:10])
                )
        missing = sorted(_nonempty_values(rows, stop_column) - stop_set)
        if missing:
            raise ValueError(
                "network/stop_times.csv references unknown stop IDs: "
                + ", ".join(missing[:10])
            )
        counts["network/stop_times.csv"] = len(rows)
    return counts


def _file_records(root: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "bundle_manifest.json":
            continue
        record: dict[str, object] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                record["row_count"] = max(0, sum(1 for _ in stream) - 1)
        records[relative] = record
    return records


def _validate_bundle_directory(root: Path, manifest: Mapping[str, object]) -> None:
    if manifest.get("schema_version") != GRAVITY_VIEWER_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported gravity viewer bundle schema version.")
    if manifest.get("bundle_type") != GRAVITY_VIEWER_BUNDLE_TYPE:
        raise ValueError("unexpected gravity viewer bundle type.")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("bundle manifest files are missing or invalid.")
    required = (*_REPORT_FILES, *_TABLE_FILES, _MODEL_SPECIFICATION_FILE)
    for name in required:
        if name not in files:
            raise ValueError(f"bundle manifest is missing required file {name!r}.")
    for name, record in files.items():
        if not isinstance(name, str) or not isinstance(record, Mapping):
            raise ValueError("bundle manifest file records are invalid.")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"bundle manifest contains unsafe file path {name!r}.")
        path = root / relative
        if not path.is_file():
            raise ValueError(f"bundle file is missing: {name}")
        expected_size = record.get("size_bytes")
        if expected_size != path.stat().st_size:
            raise ValueError(
                f"bundle file size mismatch for {name!r}: "
                f"manifest={expected_size!r}, actual={path.stat().st_size!r}."
            )
        expected_digest = record.get("sha256")
        actual_digest = _sha256(path)
        if expected_digest != actual_digest:
            raise ValueError(
                f"bundle file checksum mismatch for {name!r}: "
                f"manifest={expected_digest!r}, actual={actual_digest!r}."
            )
        if "row_count" in record and path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                actual_rows = max(0, sum(1 for _ in stream) - 1)
            if record.get("row_count") != actual_rows:
                raise ValueError(
                    f"bundle file row count mismatch for {name!r}: "
                    f"manifest={record.get('row_count')!r}, actual={actual_rows!r}."
                )
    report_payload = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if not isinstance(report_payload, Mapping):
        raise ValueError("bundle report.json must contain an object.")
    spec_payload = json.loads((root / _MODEL_SPECIFICATION_FILE).read_text(encoding="utf-8"))
    if not isinstance(spec_payload, Mapping) or spec_payload.get("artifact_type") != "gravity_model_specification":
        raise ValueError("bundle model_specification.json has an unexpected artifact type.")
    specification = spec_payload.get("specification")
    if not isinstance(specification, Mapping):
        raise ValueError("bundle model specification is missing.")
    try:
        parsed_spec = GravityModelSpecification.from_dict(dict(specification))
    except (TypeError, ValueError) as error:
        raise ValueError("bundle model specification is invalid.") from error
    spec_fingerprint = parsed_spec.fingerprint
    if spec_payload.get("specification_fingerprint") != spec_fingerprint:
        raise ValueError("bundle model specification fingerprint mismatch.")
    manifest_spec = manifest.get("model_specification")
    if not isinstance(manifest_spec, Mapping) or manifest_spec.get("fingerprint") != spec_fingerprint:
        raise ValueError("bundle manifest model specification provenance mismatch.")
    if manifest.get("specification_fingerprint") != spec_fingerprint:
        raise ValueError("bundle specification_fingerprint does not match model specification.")
    report_specification = report_payload.get("model_specification")
    if not isinstance(report_specification, Mapping):
        raise ValueError("bundle report.json is missing model_specification.")
    try:
        parsed_report_specification = GravityModelSpecification.from_dict(
            dict(report_specification)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("bundle report model specification is invalid.") from error
    if canonical_json(dict(report_specification)) != canonical_json(
        parsed_report_specification.to_dict()
    ):
        raise ValueError("bundle report model specification is not canonical.")
    if report_payload.get("specification_fingerprint") != spec_fingerprint:
        raise ValueError("bundle report specification provenance mismatch.")
    if parsed_report_specification.fingerprint != spec_fingerprint:
        raise ValueError("bundle report model specification fingerprint mismatch.")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("bundle provenance is missing or invalid.")
    missing_provenance = [
        field for field in _CANONICAL_PROVENANCE if field not in provenance
    ]
    if missing_provenance:
        raise ValueError(
            "bundle provenance is missing fields: "
            + ", ".join(missing_provenance)
        )
    report_model = report_payload.get("model_fingerprint")
    if report_model != provenance.get("model_fingerprint"):
        raise ValueError(
            "bundle report/model provenance mismatch: "
            f"report={report_model!r}, bundle={provenance.get('model_fingerprint')!r}."
        )
    report_provenance = manifest.get("report_provenance", {})
    if not isinstance(report_provenance, Mapping):
        raise ValueError("bundle report provenance is missing or invalid.")
    if (
        report_provenance.get("report_fingerprint") is not None
        and report_payload.get("report_fingerprint") != report_provenance.get("report_fingerprint")
    ):
        raise ValueError(
            "bundle report_fingerprint mismatch: "
            f"report={report_payload.get('report_fingerprint')!r}, "
            f"manifest={report_provenance.get('report_fingerprint')!r}."
        )
    fit_provenance = manifest.get("fit_provenance", {})
    if not isinstance(fit_provenance, Mapping):
        raise ValueError("bundle fit provenance is missing or invalid.")
    if fit_provenance.get("model_fingerprint") != provenance.get("model_fingerprint"):
        raise ValueError(
            "bundle fit/model provenance mismatch: "
            f"fit={fit_provenance.get('model_fingerprint')!r}, "
            f"bundle={provenance.get('model_fingerprint')!r}."
        )
    if fit_provenance.get("specification_fingerprint") != spec_fingerprint:
        raise ValueError(
            "bundle fit/specification provenance mismatch: "
            f"fit={fit_provenance.get('specification_fingerprint')!r}, "
            f"specification={spec_fingerprint!r}."
        )
    validation_provenance = manifest.get("validation_provenance", {})
    if not isinstance(validation_provenance, Mapping):
        raise ValueError("bundle validation provenance is missing or invalid.")
    validation_model = validation_provenance.get("model_fingerprint")
    if validation_model is not None and validation_model != provenance.get("model_fingerprint"):
        raise ValueError(
            "bundle validation/model provenance mismatch: "
            f"validation={validation_model!r}, bundle={provenance.get('model_fingerprint')!r}."
        )
    if validation_provenance.get("specification_fingerprint") != spec_fingerprint:
        raise ValueError(
            "bundle validation/specification provenance mismatch: "
            f"validation={validation_provenance.get('specification_fingerprint')!r}, "
            f"specification={spec_fingerprint!r}."
        )
    validation_specification = validation_provenance.get("model_specification")
    if not isinstance(validation_specification, Mapping) or dict(
        validation_specification
    ) != dict(specification):
        raise ValueError("bundle validation model specification provenance mismatch.")
    report_model_provenance = report_provenance.get("model_fingerprint")
    if report_model_provenance is not None and report_model_provenance != provenance.get("model_fingerprint"):
        raise ValueError(
            "bundle report/model provenance mismatch: "
            f"report={report_model_provenance!r}, bundle={provenance.get('model_fingerprint')!r}."
        )
    if report_provenance.get("specification_fingerprint") != spec_fingerprint:
        raise ValueError("bundle report/specification provenance mismatch.")
    report_specification_provenance = report_provenance.get("model_specification")
    if not isinstance(report_specification_provenance, Mapping) or dict(
        report_specification_provenance
    ) != dict(specification):
        raise ValueError("bundle report model specification provenance mismatch.")
    counts = manifest.get("row_counts", {})
    if not isinstance(counts, Mapping):
        raise ValueError("bundle row_counts are missing or invalid.")
    for name in _TABLE_FILES:
        fields, rows = _read_csv(root / name)
        expected = counts.get(name)
        if expected != len(rows):
            raise ValueError(
                f"bundle row count mismatch for {name!r}: manifest={expected!r}, actual={len(rows)!r}."
            )
        if name == "full_od.csv":
            _require_columns(root / name, fields, _FULL_OD_COLUMNS)
        elif name == "predicted_measurements.csv":
            _require_columns(root / name, fields, _PREDICTED_COLUMNS)
        elif name == "residuals.csv":
            _require_columns(root / name, fields, _RESIDUAL_COLUMNS)
        elif name == "parameters.csv":
            _require_columns(root / name, fields, _PARAMETER_COLUMNS)
    for name, columns in (
        ("od_map.csv", _OD_MAP_COLUMNS),
        ("stop_summary.csv", _STOP_SUMMARY_COLUMNS),
    ):
        if not (root / name).is_file():
            continue
        fields, rows = _read_csv(root / name)
        expected = counts.get(name)
        if expected != len(rows):
            raise ValueError(
                f"bundle row count mismatch for {name!r}: "
                f"manifest={expected!r}, actual={len(rows)!r}."
            )
        _require_columns(root / name, fields, columns)
    identifiability = manifest.get("identifiability", {})
    if not isinstance(identifiability, Mapping):
        raise ValueError("bundle identifiability metadata is invalid.")
    if bool(identifiability.get("available")):
        ident_dir = root / "identifiability"
        if not (ident_dir / "identifiability.json").is_file() or not (ident_dir / "identifiability.npz").is_file():
            raise ValueError("bundle identifiability metadata claims an absent artifact.")
        diagnostic = read_gravity_od_identifiability(ident_dir)
        if diagnostic.provenance.get("model_fingerprint") != provenance.get("model_fingerprint"):
            raise ValueError("bundle identifiability/model provenance mismatch.")
    network_counts = _validate_network(root, counts)
    for name, count in network_counts.items():
        if counts.get(name) != count:
            raise ValueError(
                f"bundle row count mismatch for {name!r}: manifest={counts.get(name)!r}, actual={count!r}."
            )


@dataclass(frozen=True, slots=True)
class GravityViewerBundle:
    """Validated bundle handle exposed to an independent graphical viewer."""

    path: Path
    manifest: Mapping[str, object]

    def validate(self) -> None:
        """Revalidate checksums, joins, provenance, and row counts."""
        _validate_bundle_directory(self.path, self.manifest)

    @property
    def model_specification(self) -> Mapping[str, object]:
        payload = json.loads(
            (self.path / _MODEL_SPECIFICATION_FILE).read_text(encoding="utf-8")
        )
        specification = payload.get("specification")
        if not isinstance(specification, Mapping):
            raise ValueError("bundle model specification is missing.")
        return specification

    @property
    def table_names(self) -> tuple[str, ...]:
        return (*_TABLE_FILES, *tuple(name for name in _OPTIONAL_DERIVED_FILES if (self.path / name).is_file()))

    def table_path(self, name: str) -> Path:
        if name not in _TABLE_FILES and name not in _OPTIONAL_DERIVED_FILES:
            raise KeyError(f"unknown gravity viewer table {name!r}.")
        path = self.path / name
        if not path.is_file():
            raise FileNotFoundError(f"bundle table is not present: {name}")
        return path

    def network_path(self, name: str) -> Path:
        if name not in _NETWORK_FILES:
            raise KeyError(f"unknown network snapshot file {name!r}.")
        path = self.path / "network" / name
        if not path.is_file():
            raise FileNotFoundError(f"network snapshot file is not present: {name}")
        return path

    def iter_table(
        self,
        name: str,
        *,
        columns: Sequence[str] | None = None,
        chunksize: int = 100_000,
        filters: Mapping[str, object] | None = None,
    ) -> Iterator[Any]:
        """Yield pandas chunks, optionally filtered without loading the full table."""
        if chunksize <= 0:
            raise ValueError("chunksize must be positive.")
        path = self.table_path(name)
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover - pandas is a core dependency
            raise RuntimeError("pandas is required for chunked viewer access.") from error
        filters = {} if filters is None else dict(filters)
        for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
            for field, expected in filters.items():
                if field not in chunk.columns:
                    raise ValueError(f"table {name!r} has no filter column {field!r}.")
                if isinstance(expected, (set, frozenset, tuple, list)):
                    chunk = chunk[chunk[field].isin(expected)]
                else:
                    chunk = chunk[chunk[field] == expected]
            if not chunk.empty:
                yield chunk

    def read_table(
        self,
        name: str,
        *,
        columns: Sequence[str] | None = None,
        filters: Mapping[str, object] | None = None,
        chunksize: int | None = None,
    ) -> Any:
        """Read a table; filtered reads default to bounded chunks."""
        if chunksize is not None or filters:
            size = 100_000 if chunksize is None else chunksize
            chunks = list(self.iter_table(name, columns=columns, chunksize=size, filters=filters))
            try:
                import pandas as pd
            except ImportError as error:  # pragma: no cover
                raise RuntimeError("pandas is required for viewer access.") from error
            return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=columns)
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("pandas is required for viewer access.") from error
        return pd.read_csv(self.table_path(name), usecols=columns)

    def query_od(
        self,
        *,
        origin: str | None = None,
        destination: str | None = None,
        departure_time_bin: str | None = None,
        identifiability_class: str | None = None,
        minimum_flow: float | None = None,
        top_n: int | None = None,
        chunksize: int = 100_000,
    ) -> Any:
        """Filter the OD table and optionally return its top-N flows."""
        filters: dict[str, object] = {}
        if origin is not None:
            filters["origin_stop_id"] = origin
        if destination is not None:
            filters["destination_stop_id"] = destination
        if departure_time_bin is not None:
            filters["departure_time_bin"] = departure_time_bin
        if identifiability_class is not None:
            filters["identifiability_class"] = identifiability_class
        result = self.read_table("full_od.csv", filters=filters, chunksize=chunksize)
        if minimum_flow is not None:
            result = result[result["estimated_demand"] >= minimum_flow]
        if top_n is not None:
            if top_n <= 0:
                raise ValueError("top_n must be positive when provided.")
            result = result.nlargest(top_n, "estimated_demand")
        return result

    def aggregate_measurements(
        self,
        by: str | Sequence[str] = "line",
        *,
        chunksize: int = 100_000,
    ) -> Any:
        """Aggregate observed/modelled measurements by one or more columns."""
        group_by = (by,) if isinstance(by, str) else tuple(by)
        if not group_by:
            raise ValueError("at least one aggregation column is required.")
        chunks = list(
            self.iter_table(
                "predicted_measurements.csv",
                columns=(*group_by, "observed", "modeled"),
                chunksize=chunksize,
            )
        )
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("pandas is required for viewer access.") from error
        if not chunks:
            return pd.DataFrame(columns=[*group_by, "observed", "modeled"])
        combined = pd.concat(chunks, ignore_index=True)
        return (
            combined.groupby(list(group_by), dropna=False, sort=True)[["observed", "modeled"]]
            .sum()
            .reset_index()
        )


def write_gravity_viewer_bundle(
    *,
    output_directory: str | Path,
    fit_result: GravityEstimationResult,
    validation_result: object,
    report: GravityDetailedReport,
    identifiability: GravityODIdentifiability | None = None,
    network_files: Mapping[str, str | Path] | None = None,
    metadata: Mapping[str, object] | None = None,
    require_identifiability: bool = False,
    model_specification: GravityModelSpecification | Mapping[str, object] | None = None,
) -> GravityViewerBundle:
    """Export validated fit/report data for an independent graphical viewer."""
    if not isinstance(fit_result, GravityEstimationResult):
        raise TypeError("fit_result must be a GravityEstimationResult.")
    if not isinstance(report, GravityDetailedReport):
        raise TypeError("report must be a GravityDetailedReport.")
    if validation_result is None:
        raise ValueError("validation_result is required for a viewer bundle.")
    if require_identifiability and identifiability is None:
        raise ValueError("OD identifiability diagnostics are required for this bundle.")
    metadata = {} if metadata is None else dict(metadata)
    specification, specification_fingerprint = _result_specification(
        fit_result, model_specification
    )
    destination = Path(output_directory).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing viewer bundle directory: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for name in (*_REPORT_FILES, *_TABLE_FILES):
            shutil.copy2(_report_path(report, name), staging / name)
        for name in _OPTIONAL_REPORT_FILES:
            source = report.files.get(name) if isinstance(report.files, Mapping) else None
            if source is not None and Path(source).is_file():
                shutil.copy2(source, staging / name)

        report_payload = json.loads((staging / "report.json").read_text(encoding="utf-8"))
        if not isinstance(report_payload, Mapping):
            raise ValueError("report.json must contain an object.")
        table_counts = _validate_table_inputs(
            staging, fit_result, report_payload, validation_result
        )

        _write_json(
            staging / _MODEL_SPECIFICATION_FILE,
            {
                "schema_version": _MODEL_SPECIFICATION_SCHEMA_VERSION,
                "artifact_type": "gravity_model_specification",
                "specification_fingerprint": specification_fingerprint,
                "specification": specification,
            },
        )

        identifiability_provenance: Mapping[str, object] | None = None
        if identifiability is not None:
            if identifiability.num_cells != table_counts["full_od.csv"]:
                raise ValueError("identifiability length does not match full_od.csv.")
            if identifiability.total_hessian_dimension != fit_result.raw_parameters.size:
                raise ValueError("identifiability parameter dimension does not match fit_result.")
            if identifiability.provenance.get("model_fingerprint") != fit_result.model_fingerprint:
                raise ValueError("identifiability model_fingerprint does not match fit_result.")
            write_gravity_od_identifiability(identifiability, staging / "identifiability")
            restored = read_gravity_od_identifiability(staging / "identifiability")
            identifiability_provenance = dict(restored.provenance)

        network_counts: dict[str, int] = {}
        if network_files is not None:
            if not isinstance(network_files, Mapping):
                raise TypeError("network_files must be a mapping of names to paths.")
            if "stops.csv" not in network_files:
                raise ValueError("network_files must explicitly include stops.csv.")
            network_dir = staging / "network"
            network_dir.mkdir()
            for name, source_value in network_files.items():
                if name not in _NETWORK_FILES:
                    raise ValueError(f"unsupported network snapshot file {name!r}.")
                source = Path(source_value).expanduser().resolve()
                if not source.is_file():
                    raise FileNotFoundError(f"network snapshot file is missing: {source}")
                shutil.copy2(source, network_dir / name)
            network_counts = _validate_network(staging, table_counts)
            table_counts.update(network_counts)

        # These are intentionally derived, display-oriented summaries.  They
        # never participate in model identity and are omitted when the source
        # report does not contain enough information to construct them.
        full_od_fields, full_od_rows = _read_csv(staging / "full_od.csv")
        del full_od_fields
        if _write_od_map(
            staging / "od_map.csv",
            full_od_rows=full_od_rows,
            report_payload=report_payload,
        ):
            table_counts["od_map.csv"] = len(_read_csv(staging / "od_map.csv")[1])

        predicted_fields, predicted_rows = _read_csv(staging / "predicted_measurements.csv")
        network_stops: list[dict[str, str]] = []
        stops_path = staging / "network" / "stops.csv"
        if stops_path.is_file():
            _, network_stops = _read_csv(stops_path)
        if _write_stop_summary(
            staging / "stop_summary.csv",
            predicted_fields=predicted_fields,
            predicted_rows=predicted_rows,
            network_stops=network_stops,
        ):
            table_counts["stop_summary.csv"] = len(
                _read_csv(staging / "stop_summary.csv")[1]
            )

        provenance = _source_provenance(report_payload, fit_result, metadata)
        provenance["specification_fingerprint"] = specification_fingerprint
        validation_prov = _validation_provenance(validation_result)
        for field, value in validation_prov.items():
            if field in provenance and provenance[field] is not None and value is not None and provenance[field] != value:
                raise ValueError(
                    f"validation provenance field {field!r} differs from fit/report: "
                    f"validation={value!r}, fit={provenance[field]!r}."
                )

        ident_summary: dict[str, object]
        if identifiability is None:
            ident_summary = {"available": False, "message": _IDENTIFIABILITY_MESSAGE}
        else:
            ident_summary = {
                "available": True,
                "path": "identifiability",
                "schema_version": 1,
                "provenance": dict(identifiability_provenance or {}),
                "effective_hessian_rank": identifiability.effective_hessian_rank,
                "total_hessian_dimension": identifiability.total_hessian_dimension,
                "classification_counts": identifiability.classification_counts,
            }
        bundle_metadata = dict(metadata)
        bundle_metadata.pop("provenance", None)
        coordinate_convention = {
            "coordinate_system": str(
                bundle_metadata.pop("coordinate_system", "latitude_longitude")
            ),
            "x_column": str(bundle_metadata.pop("x_column", "lon")),
            "y_column": str(bundle_metadata.pop("y_column", "lat")),
        }
        time_zone = str(bundle_metadata.pop("time_zone", "unspecified"))
        manifest: dict[str, object] = {
            "schema_version": GRAVITY_VIEWER_BUNDLE_SCHEMA_VERSION,
            "bundle_type": GRAVITY_VIEWER_BUNDLE_TYPE,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_specification": {
                "available": True,
                "path": _MODEL_SPECIFICATION_FILE,
                "fingerprint": specification_fingerprint,
            },
            "provenance": provenance,
            "model_fingerprint": fit_result.model_fingerprint,
            "specification_fingerprint": specification_fingerprint,
            "fit_provenance": {
                "model_fingerprint": fit_result.model_fingerprint,
                "specification_fingerprint": specification_fingerprint,
                "optimizer": fit_result.optimizer,
                "status": fit_result.status,
                "acceptance": fit_result.acceptance,
            },
            "validation_provenance": dict(validation_prov),
            "report_provenance": {
                "report_fingerprint": report_payload.get("report_fingerprint", report.report_fingerprint),
                "model_fingerprint": report_payload.get("model_fingerprint"),
                "specification_fingerprint": report_payload.get("specification_fingerprint"),
                "model_specification": report_payload.get("model_specification"),
                "provenance": report_payload.get("provenance", {}),
            },
            "identifiability": ident_summary,
            "artifact_identity_fingerprint": provenance.get("artifact_identity_fingerprint"),
            "assignment_fingerprint": provenance.get("assignment_fingerprint"),
            "binding_fingerprint": provenance.get("binding_fingerprint"),
            "canonical_index_fingerprint": provenance.get("canonical_index_fingerprint"),
            "od_layout_fingerprint": provenance.get("od_layout_fingerprint"),
            "compact_layout_fingerprint": provenance.get("compact_layout_fingerprint"),
            "gravity_features_fingerprint": provenance.get("gravity_features_fingerprint"),
            "package_revision": provenance.get("package_revision"),
            "time_zone": time_zone,
            "coordinate_convention": coordinate_convention,
            "metadata": bundle_metadata,
            "row_counts": table_counts,
        }
        manifest["files"] = _file_records(staging)
        _write_json(staging / "bundle_manifest.json", manifest)
        # Validate the staged bundle before it becomes visible to another process.
        staged_manifest = json.loads((staging / "bundle_manifest.json").read_text(encoding="utf-8"))
        _validate_bundle_directory(staging, staged_manifest)
        os.replace(staging, destination)
        return GravityViewerBundle(destination, staged_manifest)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def write_persisted_gravity_viewer_bundle(
    *,
    output_directory: str | Path,
    fit_manifest: Mapping[str, object],
    validation_manifest: Mapping[str, object],
    observations: object,
    od_layout: ODParameterLayout,
    metadata: GravityValidationMetadata | None = None,
    likelihood: GravityLikelihood | str = "negative_binomial",
    identifiability: GravityODIdentifiability | None = None,
    network_files: Mapping[str, str | Path] | None = None,
    bundle_metadata: Mapping[str, object] | None = None,
    require_identifiability: bool = False,
    model_specification: GravityModelSpecification | Mapping[str, object] | None = None,
    report_output_directory: str | Path | None = None,
) -> GravityViewerBundle:
    """Export a viewer bundle directly from persisted fit/validation artifacts.

    This convenience entry point restores the typed fit result, generates the
    detailed report through the persisted-data reporting contract, and passes
    the report to :func:`write_gravity_viewer_bundle`. It never loads a case
    context, activates routing, evaluates an objective, or invokes an
    optimizer. Unless ``report_output_directory`` is supplied, the temporary
    detailed report is removed after the bundle is written.
    """
    if not isinstance(fit_manifest, Mapping):
        raise TypeError("fit_manifest must be a mapping.")
    if not isinstance(validation_manifest, Mapping):
        raise TypeError("validation_manifest must be a mapping.")
    raw_result = fit_manifest.get("result")
    if not isinstance(raw_result, Mapping):
        raise ValueError("fit_manifest['result'] must be a mapping.")

    from .reporting import write_persisted_gravity_detailed_report

    result = GravityEstimationResult.from_dict(raw_result)
    destination = Path(output_directory).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_report: Path | None = None
    if report_output_directory is None:
        report_root = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.report-",
                dir=destination.parent,
            )
        )
        temporary_report = report_root
    else:
        report_root = Path(report_output_directory).expanduser().resolve()

    try:
        report = write_persisted_gravity_detailed_report(
            fit_manifest=fit_manifest,
            validation_manifest=validation_manifest,
            observations=observations,
            od_layout=od_layout,
            metadata=metadata,
            likelihood=likelihood,
            output_directory=report_root,
            identifiability=identifiability,
            require_identifiability=require_identifiability,
        )
        return write_gravity_viewer_bundle(
            output_directory=destination,
            fit_result=result,
            validation_result=validation_manifest,
            report=report,
            identifiability=identifiability,
            network_files=network_files,
            metadata=bundle_metadata,
            require_identifiability=require_identifiability,
            model_specification=model_specification,
        )
    finally:
        if temporary_report is not None:
            shutil.rmtree(temporary_report, ignore_errors=True)


def read_gravity_viewer_bundle(path: str | Path) -> GravityViewerBundle:
    """Read and validate a self-contained gravity viewer bundle."""
    root = Path(path).expanduser().resolve()
    manifest_path = root / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"gravity viewer bundle manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("gravity viewer bundle manifest must contain an object.")
    bundle = GravityViewerBundle(root, dict(payload))
    bundle.validate()
    return bundle


__all__ = [
    "GRAVITY_VIEWER_BUNDLE_SCHEMA_VERSION",
    "GRAVITY_VIEWER_BUNDLE_TYPE",
    "GravityViewerBundle",
    "read_gravity_viewer_bundle",
    "write_gravity_viewer_bundle",
    "write_persisted_gravity_viewer_bundle",
]
