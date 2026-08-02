"""Deterministic, atomic persistence for structural-zero analysis artifacts."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import StructuralZeroConfig
from .progress import ProgressEmitter, StructuralZeroProgress, emit_phase
from .reconciliation import FixedDemandReconciliationResult
from .types import StructuralZeroAnalysisResult


FIXED_DEMAND_FILENAME = "fixed_demand.csv"
AUDIT_FILENAME = "structural_zero_audit.csv"
SUMMARY_FILENAME = "structural_zero_summary.json"
FINGERPRINTS_FILENAME = "fingerprints.json"
RESOLVED_CONFIG_FILENAME = "resolved_config.toml"


@dataclass(frozen=True, slots=True)
class StructuralZeroOutputPaths:
    folder: Path
    fixed_demand: Path
    audit: Path
    summary: Path
    fingerprints: Path
    resolved_config: Path
    artifact_sha256: tuple[tuple[str, str], ...]


def write_structural_zero_outputs(
    analysis: StructuralZeroAnalysisResult,
    reconciliation: FixedDemandReconciliationResult,
    config: StructuralZeroConfig,
    *,
    progress: Callable[[StructuralZeroProgress], None] | None = None,
) -> StructuralZeroOutputPaths:
    """Validate, render, and atomically replace every output artifact.

    All payloads are rendered in memory before the output folder is created or
    any destination file is replaced. Each file replacement is atomic on the
    destination filesystem.
    """
    _validate_consistency(analysis, reconciliation, config)
    include_retained = config.output.include_retained_cells_in_report

    emit_phase(
        progress, "render_fixed_demand", completed=0, message=FIXED_DEMAND_FILENAME
    )
    fixed_payload = _fixed_demand_csv(reconciliation).encode("utf-8")
    emit_phase(
        progress, "render_fixed_demand", completed=1, message=FIXED_DEMAND_FILENAME
    )
    audit_payload = _audit_csv(
        analysis, include_retained=include_retained, progress=progress
    ).encode("utf-8")
    emit_phase(progress, "render_summary", completed=0, message=SUMMARY_FILENAME)
    summary_payload = _json_bytes(
        _summary_payload(
            analysis,
            reconciliation,
            include_retained=include_retained,
        )
    )
    emit_phase(progress, "render_summary", completed=1, message=SUMMARY_FILENAME)
    payloads: dict[str, bytes] = {
        FIXED_DEMAND_FILENAME: fixed_payload,
        AUDIT_FILENAME: audit_payload,
        RESOLVED_CONFIG_FILENAME: config.to_resolved_toml().encode("utf-8"),
        SUMMARY_FILENAME: summary_payload,
    }
    artifact_hashes = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
    }
    payloads[FINGERPRINTS_FILENAME] = _json_bytes(
        {
            "schema_version": 1,
            "scenario_fingerprint": analysis.scenario_fingerprint,
            "graph_fingerprint": analysis.graph_fingerprint,
            "configuration_fingerprint": analysis.configuration_fingerprint,
            "configuration_fingerprint_payload_json": config.fingerprint_payload_json,
            "artifact_sha256": artifact_hashes,
        }
    )

    folder = config.output.folder
    folder.mkdir(parents=True, exist_ok=True)
    write_progress = ProgressEmitter(
        progress, phase="write_outputs", total=len(payloads)
    )
    write_progress.start()
    for index, (name, payload) in enumerate(payloads.items()):
        _atomic_write_bytes(folder / name, payload)
        write_progress.update(index + 1)

    return StructuralZeroOutputPaths(
        folder=folder,
        fixed_demand=folder / FIXED_DEMAND_FILENAME,
        audit=folder / AUDIT_FILENAME,
        summary=folder / SUMMARY_FILENAME,
        fingerprints=folder / FINGERPRINTS_FILENAME,
        resolved_config=folder / RESOLVED_CONFIG_FILENAME,
        artifact_sha256=tuple(sorted(artifact_hashes.items())),
    )


def _validate_consistency(
    analysis: StructuralZeroAnalysisResult,
    reconciliation: FixedDemandReconciliationResult,
    config: StructuralZeroConfig,
) -> None:
    if analysis.configuration_fingerprint != config.fingerprint:
        raise ValueError(
            "Analysis configuration fingerprint does not match the configuration "
            "being persisted."
        )
    if reconciliation.num_structural_zero != analysis.num_structural_zero:
        raise ValueError(
            "Reconciliation structural-zero count does not match the analysis."
        )
    fixed_by_key = reconciliation.fixed_demand.as_dict()
    for record in analysis.records:
        if not record.is_structural_zero:
            continue
        key = record.key.tuple
        if key not in fixed_by_key or fixed_by_key[key] != 0.0:
            raise ValueError(
                f"Merged fixed demand must contain structural-zero key {key!r} at zero."
            )


def _fixed_demand_csv(reconciliation: FixedDemandReconciliationResult) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("origin_stop_id", "dest_stop_id", "time_bin_id", "fixed_flow"))
    for record in reconciliation.fixed_demand.records:
        writer.writerow((*record.key, _float_text(record.fixed_flow)))
    return stream.getvalue()


def _audit_csv(
    analysis: StructuralZeroAnalysisResult,
    *,
    include_retained: bool,
    progress: Callable[[StructuralZeroProgress], None] | None = None,
) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "origin_stop_id",
            "dest_stop_id",
            "time_bin_id",
            "is_structural_zero",
            "primary_reason",
            "triggered_rules",
            "feasible",
            "minimum_transfers",
            "minimum_initial_wait_minutes",
            "minimum_journey_time_minutes",
            "feasible_departure_count",
            "earliest_arrival_seconds",
        )
    )
    report_records = tuple(
        record
        for record in analysis.records
        if include_retained or record.is_structural_zero
    )
    render_progress = ProgressEmitter(
        progress,
        phase="render_outputs",
        total=len(report_records),
        message=AUDIT_FILENAME,
    )
    render_progress.start()
    for index, record in enumerate(report_records):
        metrics = record.metrics
        writer.writerow(
            (
                *record.key.tuple,
                _bool_text(record.is_structural_zero),
                "" if record.primary_reason is None else record.primary_reason.value,
                ";".join(reason.value for reason in record.triggered_rules),
                _bool_text(metrics.feasible),
                _optional_text(metrics.minimum_transfers),
                _optional_text(metrics.minimum_initial_wait_minutes),
                _optional_text(metrics.minimum_journey_time_minutes),
                str(metrics.feasible_departure_count),
                _optional_text(metrics.earliest_arrival_seconds),
            )
        )
        render_progress.update(index + 1)
    return stream.getvalue()


def _summary_payload(
    analysis: StructuralZeroAnalysisResult,
    reconciliation: FixedDemandReconciliationResult,
    *,
    include_retained: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "num_cells": analysis.num_cells,
        "num_structural_zero": analysis.num_structural_zero,
        "num_retained": analysis.num_retained,
        "primary_reason_counts": analysis.reason_counts,
        "audit_includes_retained_cells": include_retained,
        "reconciliation": {
            "num_existing": reconciliation.num_existing,
            "num_existing_structural_zero": reconciliation.num_existing_structural_zero,
            "num_added_structural_zero": reconciliation.num_added_structural_zero,
            "num_merged": reconciliation.num_merged,
        },
    }


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _float_text(value: float) -> str:
    return format(float(value), ".17g")


def _optional_text(value: int | float | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return _float_text(value)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
