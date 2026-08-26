"""Fail-closed persistence for canonical sparse temporal assignment blocks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .assignment_contract import (
    AssignmentArtifactIdentity,
    AssignmentCompatibilityError,
    CanonicalAssignmentIndex,
    CanonicalMeasurement,
    CanonicalODTimeCell,
    CanonicalTimeInterval,
    assert_assignment_artifact_compatible,
)
from .temporal_assignment_blocks import (
    TemporalBlockAssignmentOperator,
    TemporalBlockConstructionDiagnostics,
    TemporalBlockKey,
    TemporalSparseBlock,
)
from .construction_control import (
    ConstructionDeadline,
    ConstructionPhase,
    ConstructionProgressReporter,
    deadline_stop,
)

TEMPORAL_BLOCK_ARTIFACT_SCHEMA_VERSION = 1


def temporal_block_cache_path(
    root: str | Path, identity: AssignmentArtifactIdentity
) -> Path:
    """Return the immutable content-addressed artifact directory."""
    return Path(root) / identity.fingerprint


def _block_content_hash(
    rows: np.ndarray, columns: np.ndarray, values: np.ndarray
) -> str:
    digest = hashlib.sha256()
    for item in (rows, columns, values):
        array = np.ascontiguousarray(item)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _array_content_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_index_payload(index: CanonicalAssignmentIndex) -> dict[str, object]:
    return {
        "schema_version": index.schema_version,
        "time_intervals": [asdict(item) for item in index.time_intervals],
        "demand_cells": [asdict(item) for item in index.demand_cells],
        "measurements": [asdict(item) for item in index.measurements],
        "source_od_layout_fingerprint": index.source_od_layout_fingerprint,
        "source_compact_layout_fingerprint": (
            index.source_compact_layout_fingerprint
        ),
        "artifact_fingerprint": index.artifact_fingerprint,
        "binding_fingerprint": index.binding_fingerprint,
    }


def _decode_canonical_index(payload: object) -> CanonicalAssignmentIndex:
    if not isinstance(payload, dict):
        raise ValueError("canonical index payload must be an object.")
    try:
        index = CanonicalAssignmentIndex(
            time_intervals=tuple(
                CanonicalTimeInterval(**item) for item in payload["time_intervals"]
            ),
            demand_cells=tuple(
                CanonicalODTimeCell(**item) for item in payload["demand_cells"]
            ),
            measurements=tuple(
                CanonicalMeasurement(**item) for item in payload["measurements"]
            ),
            source_od_layout_fingerprint=payload.get("source_od_layout_fingerprint"),
            source_compact_layout_fingerprint=payload.get(
                "source_compact_layout_fingerprint"
            ),
            schema_version=int(payload["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("canonical index payload is invalid.") from error
    if payload.get("artifact_fingerprint") != index.artifact_fingerprint:
        raise ValueError("canonical physical-index fingerprint is invalid.")
    if payload.get("binding_fingerprint") != index.binding_fingerprint:
        raise ValueError("canonical binding fingerprint is invalid.")
    return index


def save_temporal_block_operator(
    directory: str | Path,
    operator: TemporalBlockAssignmentOperator,
    *,
    deadline: ConstructionDeadline | None = None,
    reporter: ConstructionProgressReporter | None = None,
    checkpoint_location: str | Path | None = None,
) -> Path:
    """Publish a complete immutable artifact atomically."""
    destination = Path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"temporal block artifact already exists: {destination}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        blocks_directory = staging / "blocks"
        blocks_directory.mkdir()
        block_manifest = []
        for block_index, block in enumerate(operator.blocks):
            if deadline is not None and not deadline.may_start():
                raise deadline_stop(
                    deadline,
                    phase=ConstructionPhase.PERSISTENCE,
                    reason="deadline reached while staging temporal blocks",
                    completed_units=block_index,
                    total_units=len(operator.blocks),
                    next_resumable_position=f"block-{block_index:06d}",
                    checkpoint_location=(
                        None
                        if checkpoint_location is None
                        else str(checkpoint_location)
                    ),
                    artifact_location=str(destination),
                    checkpoint_reusable=checkpoint_location is not None,
                )
            filename = f"block-{block_index:06d}.npz"
            content_hash = _block_content_hash(
                block.row_indices, block.column_indices, block.values
            )
            np.savez(
                blocks_directory / filename,
                row_indices=block.row_indices,
                column_indices=block.column_indices,
                values=block.values,
            )
            block_manifest.append(
                {
                    "filename": filename,
                    "key": asdict(block.key),
                    "nonzero_entries": block.nonzero_entries,
                    "content_hash": content_hash,
                }
            )
            if reporter is not None:
                reporter.emit(
                    phase=ConstructionPhase.PERSISTENCE,
                    status="running",
                    completed_units=block_index + 1,
                    total_units=len(operator.blocks),
                    current_unit=filename,
                    checkpoint_location=(
                        None
                        if checkpoint_location is None
                        else str(checkpoint_location)
                    ),
                )
        np.save(staging / "fixed_measurement_offset.npy", operator.fixed_measurement_offset)
        manifest = {
            "schema_version": TEMPORAL_BLOCK_ARTIFACT_SCHEMA_VERSION,
            "complete": True,
            "identity": asdict(operator.identity),
            "identity_fingerprint": operator.identity.fingerprint,
            "canonical_index": _canonical_index_payload(operator.canonical_index),
            "number_of_demand_cells": operator.number_of_demand_cells,
            "number_of_measurements": operator.number_of_measurements,
            "nonzero_entries": operator.diagnostics.nonzero_entries,
            "fixed_measurement_offset_hash": _array_content_hash(
                operator.fixed_measurement_offset
            ),
            "diagnostics": asdict(operator.diagnostics),
            "blocks": block_manifest,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def load_temporal_block_operator(
    directory: str | Path,
    *,
    expected_identity: AssignmentArtifactIdentity,
    expected_canonical_index: CanonicalAssignmentIndex,
    reporter: ConstructionProgressReporter | None = None,
) -> TemporalBlockAssignmentOperator:
    """Validate every identity and payload before accepting an artifact.

    When supplied, ``reporter`` receives throttled cache-validation progress;
    omitting it preserves the original silent loading behavior.
    """
    source = Path(directory)
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("temporal block manifest is missing or corrupt.") from error
    if manifest.get("schema_version") != TEMPORAL_BLOCK_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("temporal block artifact schema is incompatible.")
    if manifest.get("complete") is not True:
        raise ValueError("temporal block artifact is incomplete.")
    block_items = manifest.get("blocks", [])
    total_blocks = len(block_items) if isinstance(block_items, list) else 0
    try:
        actual_identity = AssignmentArtifactIdentity(**manifest["identity"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("temporal block artifact identity is invalid.") from error
    if manifest.get("identity_fingerprint") != actual_identity.fingerprint:
        raise ValueError("temporal block artifact identity fingerprint is invalid.")
    assert_assignment_artifact_compatible(
        expected=expected_identity, actual=actual_identity
    )
    actual_index = _decode_canonical_index(manifest.get("canonical_index"))
    if actual_index.artifact_fingerprint != expected_canonical_index.artifact_fingerprint:
        raise AssignmentCompatibilityError(
            "temporal block physical canonical index is incompatible."
        )
    if actual_index.binding_fingerprint != expected_canonical_index.binding_fingerprint:
        raise AssignmentCompatibilityError(
            "temporal block canonical binding is incompatible."
        )
    if reporter is not None:
        first_filename = (
            block_items[0].get("filename")
            if total_blocks and isinstance(block_items[0], dict)
            else None
        )
        reporter.emit(
            phase=ConstructionPhase.CACHE_VALIDATION,
            status="running",
            force=True,
            completed_units=0,
            total_units=total_blocks,
            current_unit=first_filename,
            checkpoint_location=str(source),
            details={"cache_validation_stage": "temporal_block_load"},
        )
    blocks = []
    for block_position, item in enumerate(block_items):
        try:
            with np.load(source / "blocks" / item["filename"], allow_pickle=False) as data:
                rows = data["row_indices"]
                columns = data["column_indices"]
                values = data["values"]
            if _block_content_hash(rows, columns, values) != item["content_hash"]:
                raise ValueError("temporal block content hash is invalid.")
            block = TemporalSparseBlock(
                key=TemporalBlockKey(**item["key"]),
                row_indices=rows,
                column_indices=columns,
                values=values,
                number_of_measurements=actual_index.number_of_measurements,
                number_of_demand_cells=actual_index.number_of_demand_cells,
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise ValueError("temporal block payload is missing or corrupt.") from error
        if block.nonzero_entries != item.get("nonzero_entries"):
            raise ValueError("temporal block nonzero count is invalid.")
        blocks.append(block)
        if reporter is not None:
            reporter.emit(
                phase=ConstructionPhase.CACHE_VALIDATION,
                status="running",
                completed_units=block_position + 1,
                total_units=total_blocks,
                current_unit=item.get("filename"),
                checkpoint_location=str(source),
                details={"cache_validation_stage": "temporal_block_load"},
            )
    diagnostics = TemporalBlockConstructionDiagnostics(**manifest["diagnostics"])
    if sum(block.nonzero_entries for block in blocks) != manifest.get("nonzero_entries"):
        raise ValueError("temporal block aggregate nonzero count is invalid.")
    try:
        offset = np.load(source / "fixed_measurement_offset.npy", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("fixed measurement offset is missing or corrupt.") from error
    if _array_content_hash(offset) != manifest.get("fixed_measurement_offset_hash"):
        raise ValueError("fixed measurement offset content hash is invalid.")
    operator = TemporalBlockAssignmentOperator(
        canonical_index=actual_index,
        identity=actual_identity,
        blocks=tuple(blocks),
        fixed_measurement_offset=offset,
        diagnostics=diagnostics,
    )
    if reporter is not None:
        reporter.emit(
            phase=ConstructionPhase.CACHE_VALIDATION,
            status="completed",
            force=True,
            completed_units=total_blocks,
            total_units=total_blocks,
            checkpoint_location=str(source),
            details={"cache_validation_stage": "temporal_block_load"},
        )
    return operator


def reuse_or_build_temporal_block_operator(
    *,
    cache_root: str | Path,
    reference,
    zero_tolerance: float = 0.0,
    progress=None,
) -> tuple[TemporalBlockAssignmentOperator, bool]:
    """Reuse a compatible content-addressed artifact or construct it once."""
    from .temporal_assignment_blocks import build_exact_temporal_block_operator

    path = temporal_block_cache_path(cache_root, reference.identity)
    if path.exists():
        return (
            load_temporal_block_operator(
                path,
                expected_identity=reference.identity,
                expected_canonical_index=reference.canonical_index,
            ),
            True,
        )
    operator = build_exact_temporal_block_operator(
        reference=reference,
        zero_tolerance=zero_tolerance,
        progress=progress,
    )
    save_temporal_block_operator(path, operator)
    return operator, False
