"""Fail-closed persistence for canonical sparse temporal assignment blocks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import deque
from dataclasses import asdict
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

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
    estimate_completed_unit_eta,
)

TEMPORAL_BLOCK_ARTIFACT_SCHEMA_VERSION = 1
TEMPORAL_OPERATOR_CACHE_SCHEMA_VERSION = 1
TEMPORAL_OPERATOR_VALIDATOR_VERSION = 1
PREFLIGHT_ADOPTION_SCHEMA_VERSION = 1


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


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: str | Path, payload: Mapping[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _identity_checkpoint_directory(
    checkpoint_directory: str | Path, identity: AssignmentArtifactIdentity
) -> Path:
    """Accept either a checkpoint root or an identity-specific directory."""
    path = Path(checkpoint_directory)
    return path if path.name == identity.fingerprint else path / identity.fingerprint


def _nested_value(payload: object, key: str) -> object | None:
    """Find a metadata value in the few nested manifest layouts in use."""
    if isinstance(payload, Mapping):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _nested_value(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _nested_value(value, key)
            if found is not None:
                return found
    return None


def _read_json(path: str | Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"JSON metadata is missing or corrupt: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON metadata must be an object: {path}")
    return payload


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
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return destination


def _preflight_payload_and_hash(
    preflight_manifest: str | Path | Mapping[str, object],
) -> tuple[dict[str, object], str]:
    if isinstance(preflight_manifest, (str, Path)):
        path = Path(preflight_manifest)
        return _read_json(path), _file_sha256(path)
    payload = dict(preflight_manifest)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return payload, hashlib.sha256(encoded).hexdigest()


def adopt_completed_preflight(
    *,
    preflight_manifest: str | Path | Mapping[str, object],
    artifact_directory: str | Path,
    expected_identity: AssignmentArtifactIdentity,
    expected_canonical_index: CanonicalAssignmentIndex,
    checkpoint_directory: str | Path,
) -> Path:
    """Adopt completed validation evidence without opening source blocks.

    This operation reads only the preflight and artifact manifests.  It creates
    a small identity-bound certificate that permits the one-time packed-cache
    consolidation to skip semantic validation already recorded by preflight.
    """
    preflight, preflight_hash = _preflight_payload_and_hash(preflight_manifest)
    result = preflight.get("result")
    if preflight.get("status") != "completed" or not isinstance(result, Mapping):
        raise ValueError("completed preflight adoption requires status='completed'.")
    if result.get("completed_phase") != 6:
        raise ValueError("completed preflight adoption requires completed_phase=6.")
    if "recommendation" not in result:
        raise ValueError("completed preflight adoption requires a recommendation.")

    artifact = Path(artifact_directory)
    artifact_manifest_path = artifact / "manifest.json"
    artifact_manifest = _read_json(artifact_manifest_path)
    if artifact_manifest.get("complete") is not True:
        raise ValueError("cannot adopt an incomplete temporal artifact.")
    try:
        artifact_identity = AssignmentArtifactIdentity(
            **artifact_manifest["identity"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("temporal artifact identity is invalid.") from error
    assert_assignment_artifact_compatible(
        expected=expected_identity, actual=artifact_identity
    )
    actual_index = _decode_canonical_index(artifact_manifest.get("canonical_index"))
    if actual_index.artifact_fingerprint != expected_canonical_index.artifact_fingerprint:
        raise AssignmentCompatibilityError(
            "preflight adoption canonical-index fingerprint is incompatible."
        )
    if actual_index.binding_fingerprint != expected_canonical_index.binding_fingerprint:
        raise AssignmentCompatibilityError(
            "preflight adoption canonical binding is incompatible."
        )

    artifact_hash = _file_sha256(artifact_manifest_path)
    # Legacy reports do not all expose every identity component.  Compare every
    # field that is present, while requiring the artifact and canonical-index
    # identities above.  This keeps adoption compatible with the completed
    # legacy report without weakening current artifact checks.
    expected_values: dict[str, object] = {
        "artifact_identity_fingerprint": expected_identity.fingerprint,
        "assignment_fingerprint": expected_identity.timetable_fingerprint,
        "binding_fingerprint": expected_canonical_index.binding_fingerprint,
        "canonical_index_fingerprint": expected_canonical_index.artifact_fingerprint,
        "compact_layout_fingerprint": expected_canonical_index.source_compact_layout_fingerprint,
        "mapping_fingerprint": expected_identity.measurement_mapping_fingerprint,
        "od_layout_fingerprint": expected_canonical_index.source_od_layout_fingerprint,
    }
    for key, expected in expected_values.items():
        observed = _nested_value(preflight, key)
        if observed is not None and expected is not None and observed != expected:
            raise AssignmentCompatibilityError(
                f"completed preflight {key} is incompatible "
                f"(expected={expected!r}, got={observed!r})."
            )
    for key, expected in (
        ("artifact_manifest_sha256", artifact_hash),
        ("number_of_blocks", len(artifact_manifest.get("blocks", []))),
        ("fixed_measurement_offset_hash", artifact_manifest.get("fixed_measurement_offset_hash")),
    ):
        observed = _nested_value(preflight, key)
        if observed is not None and observed != expected:
            raise AssignmentCompatibilityError(
                f"completed preflight {key} is incompatible "
                f"(expected={expected!r}, got={observed!r})."
            )
    checkpoint_root = _identity_checkpoint_directory(
        checkpoint_directory, expected_identity
    )
    certificate_path = checkpoint_root / "temporal_operator_cache" / "validation_certificate.json"
    certificate = {
        "complete": True,
        "schema_version": PREFLIGHT_ADOPTION_SCHEMA_VERSION,
        "validator_version": TEMPORAL_OPERATOR_VALIDATOR_VERSION,
        "provenance": {
            "source": "completed_preflight_manifest",
            "preflight_manifest_sha256": preflight_hash,
            "preflight_completed_phase": 6,
            "legacy_completed_preflight": True,
        },
        "artifact_identity_fingerprint": expected_identity.fingerprint,
        "artifact_manifest_sha256": artifact_hash,
        "canonical_index_fingerprint": expected_canonical_index.artifact_fingerprint,
        "binding_fingerprint": expected_canonical_index.binding_fingerprint,
        "number_of_blocks": len(artifact_manifest.get("blocks", [])),
        "fixed_measurement_offset_hash": artifact_manifest.get(
            "fixed_measurement_offset_hash"
        ),
    }
    return _atomic_json_write(certificate_path, certificate)


def _certificate_is_compatible(
    certificate: Mapping[str, object],
    *,
    identity: AssignmentArtifactIdentity,
    canonical_index: CanonicalAssignmentIndex,
    artifact_manifest_sha256: str,
) -> bool:
    return (
        certificate.get("complete") is True
        and certificate.get("schema_version") == PREFLIGHT_ADOPTION_SCHEMA_VERSION
        and certificate.get("validator_version") == TEMPORAL_OPERATOR_VALIDATOR_VERSION
        and certificate.get("artifact_identity_fingerprint") == identity.fingerprint
        and certificate.get("artifact_manifest_sha256") == artifact_manifest_sha256
        and certificate.get("canonical_index_fingerprint")
        == canonical_index.artifact_fingerprint
        and certificate.get("binding_fingerprint") == canonical_index.binding_fingerprint
    )


def materialize_operator_cache_from_adopted_preflight(
    *,
    artifact_directory: str | Path,
    certificate_path: str | Path,
    cache_directory: str | Path,
    reporter: ConstructionProgressReporter | None = None,
) -> Path:
    """Pack validated source blocks once, without repeating semantic checks."""
    artifact = Path(artifact_directory)
    cache = Path(cache_directory)
    certificate = _read_json(certificate_path)
    artifact_manifest_path = artifact / "manifest.json"
    artifact_manifest = _read_json(artifact_manifest_path)
    artifact_hash = _file_sha256(artifact_manifest_path)
    identity_payload = artifact_manifest.get("identity")
    if not isinstance(identity_payload, Mapping):
        raise ValueError("temporal artifact identity is invalid.")
    try:
        identity = AssignmentArtifactIdentity(**identity_payload)
        canonical_index = _decode_canonical_index(artifact_manifest["canonical_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("temporal artifact metadata is invalid.") from error
    if not _certificate_is_compatible(
        certificate,
        identity=identity,
        canonical_index=canonical_index,
        artifact_manifest_sha256=artifact_hash,
    ):
        raise AssignmentCompatibilityError(
            "validation certificate is incompatible with the temporal artifact."
        )
    if artifact_manifest.get("complete") is not True:
        raise ValueError("cannot materialize an incomplete temporal artifact.")

    block_items = artifact_manifest.get("blocks")
    if not isinstance(block_items, list):
        raise ValueError("temporal artifact block manifest is invalid.")
    total_blocks = len(block_items)
    total_nonzeros = sum(int(item.get("nonzero_entries", 0)) for item in block_items)
    dtype = np.dtype(identity.numeric_dtype)
    cache.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{cache.name}.materializing-", dir=cache.parent))
    recent_durations: deque[float] = deque(maxlen=16)
    started = perf_counter()
    reporting_enabled = reporter is not None and reporter.sink is not None
    try:
        if reporting_enabled:
            reporter.emit(
                phase=ConstructionPhase.VALIDATED_OPERATOR_CACHE_PERSISTENCE,
                status="running",
                force=True,
                completed_units=0,
                total_units=total_blocks,
                current_unit=(
                    None
                    if not block_items
                    else block_items[0].get("filename")
                ),
                checkpoint_location=str(cache),
                details={"cache_validation_stage": "validated_operator_cache_persistence"},
            )
        row_path = staging / "row_indices.npy"
        column_path = staging / "column_indices.npy"
        value_path = staging / "values.npy"
        offset_path = staging / "offsets.npy"
        fixed_offset_path = staging / "fixed_measurement_offset.npy"
        rows = np.lib.format.open_memmap(row_path, mode="w+", dtype=np.int32, shape=(total_nonzeros,))
        columns = np.lib.format.open_memmap(column_path, mode="w+", dtype=np.int32, shape=(total_nonzeros,))
        values = np.lib.format.open_memmap(value_path, mode="w+", dtype=dtype, shape=(total_nonzeros,))
        offsets = np.lib.format.open_memmap(offset_path, mode="w+", dtype=np.int64, shape=(total_blocks + 1,))
        offsets[0] = 0
        keys: list[dict[str, object]] = []
        cursor = 0
        for position, item in enumerate(block_items):
            filename = item.get("filename")
            key = item.get("key")
            if not isinstance(filename, str) or not isinstance(key, Mapping):
                raise ValueError("temporal block manifest entry is invalid.")
            block_started = perf_counter()
            # This is deliberately a consolidation read.  Content hashes and
            # semantic support were already validated by the adopted preflight.
            with np.load(artifact / "blocks" / filename, allow_pickle=False) as data:
                block_rows = np.asarray(data["row_indices"], dtype=np.int32)
                block_columns = np.asarray(data["column_indices"], dtype=np.int32)
                block_values = np.asarray(data["values"], dtype=dtype)
            length = int(block_values.size)
            if length != int(item.get("nonzero_entries", -1)):
                raise ValueError("temporal block size differs from its manifest.")
            rows[cursor : cursor + length] = block_rows
            columns[cursor : cursor + length] = block_columns
            values[cursor : cursor + length] = block_values
            cursor += length
            offsets[position + 1] = cursor
            keys.append(dict(key))
            if reporting_enabled:
                duration = max(0.0, perf_counter() - block_started)
                recent_durations.append(duration)
                eta = estimate_completed_unit_eta(
                    recent_durations,
                    completed_units=position + 1,
                    total_units=total_blocks,
                    parallelism=1,
                    elapsed_seconds=max(0.0, perf_counter() - started),
                )
                reporter.emit(
                    phase=ConstructionPhase.VALIDATED_OPERATOR_CACHE_PERSISTENCE,
                    status="running",
                    completed_units=position + 1,
                    total_units=total_blocks,
                    current_unit=filename,
                    checkpoint_location=str(cache),
                    recent_unit_seconds=duration,
                    predicted_remaining_seconds=eta.predicted_remaining_seconds,
                    eta_confidence=eta.eta_confidence,
                    estimated_completion_at_utc=eta.estimated_completion_at_utc,
                    eta_reason=eta.eta_reason,
                    eta_lower_seconds=eta.eta_lower_seconds,
                    eta_upper_seconds=eta.eta_upper_seconds,
                    throughput_units_per_second=eta.throughput_units_per_second,
                    details={"cache_validation_stage": "validated_operator_cache_persistence"},
                )
        fixed_offset = np.load(
            artifact / "fixed_measurement_offset.npy", allow_pickle=False
        )
        np.save(fixed_offset_path, np.asarray(fixed_offset))
        for array in (rows, columns, values, offsets):
            array.flush()
        del rows, columns, values, offsets
        keys_path = staging / "keys.json"
        keys_path.write_text(json.dumps(keys, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        packed_names = (
            "row_indices.npy",
            "column_indices.npy",
            "values.npy",
            "offsets.npy",
            "fixed_measurement_offset.npy",
            "keys.json",
        )
        hashes = {
            name: _file_sha256(staging / name) for name in packed_names
        }
        cache_manifest = {
            "complete": True,
            "schema_version": TEMPORAL_OPERATOR_CACHE_SCHEMA_VERSION,
            "validator_version": TEMPORAL_OPERATOR_VALIDATOR_VERSION,
            "identity": asdict(identity),
            "artifact_identity_fingerprint": identity.fingerprint,
            "artifact_manifest_sha256": artifact_hash,
            "canonical_index_fingerprint": canonical_index.artifact_fingerprint,
            "binding_fingerprint": canonical_index.binding_fingerprint,
            "number_of_blocks": total_blocks,
            "number_of_demand_cells": canonical_index.number_of_demand_cells,
            "number_of_measurements": canonical_index.number_of_measurements,
            "nonzero_entries": total_nonzeros,
            "fixed_measurement_offset_hash": artifact_manifest.get(
                "fixed_measurement_offset_hash"
            ),
            "packed_array_hashes": hashes,
            "diagnostics": artifact_manifest.get("diagnostics", {}),
            "keys": "keys.json",
            "provenance": {
                "source": "adopted_preflight",
                "validation_certificate": str(certificate_path),
            },
        }
        cache.mkdir(parents=True, exist_ok=True)
        previous_manifest = cache / "manifest.json"
        if previous_manifest.exists():
            previous_manifest.rename(cache / ".manifest.previous")
        for name in packed_names:
            os.replace(staging / name, cache / name)
        _atomic_json_write(cache / "manifest.json", cache_manifest)
        if reporting_enabled:
            eta = estimate_completed_unit_eta(
                recent_durations,
                completed_units=total_blocks,
                total_units=total_blocks,
                parallelism=1,
                elapsed_seconds=max(0.0, perf_counter() - started),
            )
            reporter.emit(
                phase=ConstructionPhase.VALIDATED_OPERATOR_CACHE_PERSISTENCE,
                status="completed",
                force=True,
                completed_units=total_blocks,
                total_units=total_blocks,
                current_unit=(None if not block_items else block_items[-1].get("filename")),
                checkpoint_location=str(cache),
                recent_unit_seconds=(recent_durations[-1] if recent_durations else None),
                predicted_remaining_seconds=0.0,
                eta_confidence="high",
                estimated_completion_at_utc=eta.estimated_completion_at_utc,
                eta_reason=eta.eta_reason,
                eta_lower_seconds=0.0,
                eta_upper_seconds=0.0,
                throughput_units_per_second=eta.throughput_units_per_second,
                details={"cache_validation_stage": "validated_operator_cache_persistence"},
            )
        return cache
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _materialize_operator_cache_from_operator(
    *,
    operator: TemporalBlockAssignmentOperator,
    cache_directory: str | Path,
    artifact_manifest_sha256: str,
    certificate_path: str | Path,
    provenance_source: str = "full_validation",
    reporter: ConstructionProgressReporter | None = None,
) -> Path:
    """Persist a packed cache from an already validated in-memory operator."""
    cache = Path(cache_directory)
    cache.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{cache.name}.materializing-", dir=cache.parent)
    )
    blocks = operator.blocks
    total_nonzeros = sum(block.nonzero_entries for block in blocks)
    reporting_enabled = reporter is not None and reporter.sink is not None
    started = perf_counter()
    recent_durations: deque[float] = deque(maxlen=16)
    if reporting_enabled:
        reporter.emit(
            phase=ConstructionPhase.VALIDATED_OPERATOR_CACHE_PERSISTENCE,
            status="running",
            force=True,
            completed_units=0,
            total_units=len(blocks),
            current_unit=(None if not blocks else "block-000000.npz"),
            checkpoint_location=str(cache),
            details={"cache_validation_stage": "validated_operator_cache_persistence"},
        )
    try:
        rows = np.lib.format.open_memmap(
            staging / "row_indices.npy", mode="w+", dtype=np.int32, shape=(total_nonzeros,)
        )
        columns = np.lib.format.open_memmap(
            staging / "column_indices.npy", mode="w+", dtype=np.int32, shape=(total_nonzeros,)
        )
        values = np.lib.format.open_memmap(
            staging / "values.npy", mode="w+", dtype=operator.dtype, shape=(total_nonzeros,)
        )
        offsets = np.lib.format.open_memmap(
            staging / "offsets.npy", mode="w+", dtype=np.int64, shape=(len(blocks) + 1,)
        )
        offsets[0] = 0
        keys = []
        cursor = 0
        for position, block in enumerate(blocks):
            block_started = perf_counter()
            length = block.nonzero_entries
            rows[cursor : cursor + length] = block.row_indices
            columns[cursor : cursor + length] = block.column_indices
            values[cursor : cursor + length] = block.values
            cursor += length
            offsets[position + 1] = cursor
            keys.append(asdict(block.key))
            if reporting_enabled:
                duration = max(0.0, perf_counter() - block_started)
                recent_durations.append(duration)
                eta = estimate_completed_unit_eta(
                    recent_durations,
                    completed_units=position + 1,
                    total_units=len(blocks),
                    parallelism=1,
                    elapsed_seconds=max(0.0, perf_counter() - started),
                )
                reporter.emit(
                    phase=ConstructionPhase.VALIDATED_OPERATOR_CACHE_PERSISTENCE,
                    status="running",
                    completed_units=position + 1,
                    total_units=len(blocks),
                    current_unit=f"block-{position:06d}.npz",
                    checkpoint_location=str(cache),
                    recent_unit_seconds=duration,
                    predicted_remaining_seconds=eta.predicted_remaining_seconds,
                    eta_confidence=eta.eta_confidence,
                    estimated_completion_at_utc=eta.estimated_completion_at_utc,
                    eta_reason=eta.eta_reason,
                    eta_lower_seconds=eta.eta_lower_seconds,
                    eta_upper_seconds=eta.eta_upper_seconds,
                    throughput_units_per_second=eta.throughput_units_per_second,
                    details={"cache_validation_stage": "validated_operator_cache_persistence"},
                )
        for array in (rows, columns, values, offsets):
            array.flush()
        del rows, columns, values, offsets
        np.save(staging / "fixed_measurement_offset.npy", operator.fixed_measurement_offset)
        (staging / "keys.json").write_text(
            json.dumps(keys, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        names = (
            "row_indices.npy",
            "column_indices.npy",
            "values.npy",
            "offsets.npy",
            "fixed_measurement_offset.npy",
            "keys.json",
        )
        hashes = {name: _file_sha256(staging / name) for name in names}
        manifest = {
            "complete": True,
            "schema_version": TEMPORAL_OPERATOR_CACHE_SCHEMA_VERSION,
            "validator_version": TEMPORAL_OPERATOR_VALIDATOR_VERSION,
            "identity": asdict(operator.identity),
            "artifact_identity_fingerprint": operator.identity.fingerprint,
            "artifact_manifest_sha256": artifact_manifest_sha256,
            "canonical_index_fingerprint": operator.canonical_index.artifact_fingerprint,
            "binding_fingerprint": operator.canonical_index.binding_fingerprint,
            "number_of_blocks": len(blocks),
            "number_of_demand_cells": operator.number_of_demand_cells,
            "number_of_measurements": operator.number_of_measurements,
            "nonzero_entries": total_nonzeros,
            "fixed_measurement_offset_hash": _array_content_hash(
                operator.fixed_measurement_offset
            ),
            "packed_array_hashes": hashes,
            "diagnostics": asdict(operator.diagnostics),
            "keys": "keys.json",
            "provenance": {
                "source": provenance_source,
                "validation_certificate": str(certificate_path),
            },
        }
        cache.mkdir(parents=True, exist_ok=True)
        previous_manifest = cache / "manifest.json"
        if previous_manifest.exists():
            previous_manifest.rename(cache / ".manifest.previous")
        for name in names:
            os.replace(staging / name, cache / name)
        _atomic_json_write(cache / "manifest.json", manifest)
        if reporting_enabled:
            reporter.emit(
                phase=ConstructionPhase.VALIDATED_OPERATOR_CACHE_PERSISTENCE,
                status="completed",
                force=True,
                completed_units=len(blocks),
                total_units=len(blocks),
                current_unit=(None if not blocks else f"block-{len(blocks) - 1:06d}.npz"),
                checkpoint_location=str(cache),
                recent_unit_seconds=(recent_durations[-1] if recent_durations else None),
                predicted_remaining_seconds=0.0,
                eta_confidence="high",
                eta_lower_seconds=0.0,
                eta_upper_seconds=0.0,
                throughput_units_per_second=(
                    len(blocks) / max(perf_counter() - started, np.finfo(float).eps)
                ),
                details={"cache_validation_stage": "validated_operator_cache_persistence"},
            )
        return cache
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _load_validated_operator_cache(
    *,
    cache_directory: str | Path,
    expected_identity: AssignmentArtifactIdentity,
    expected_canonical_index: CanonicalAssignmentIndex,
    artifact_manifest_sha256: str,
) -> TemporalBlockAssignmentOperator | None:
    """Load packed arrays after validating their metadata and hashes."""
    cache = Path(cache_directory)
    manifest_path = cache / "manifest.json"
    certificate_path = cache / "validation_certificate.json"
    if not manifest_path.is_file() or not certificate_path.is_file():
        return None
    try:
        manifest = _read_json(manifest_path)
        certificate = _read_json(certificate_path)
        if manifest.get("complete") is not True:
            return None
        if manifest.get("schema_version") != TEMPORAL_OPERATOR_CACHE_SCHEMA_VERSION:
            return None
        if manifest.get("validator_version") != TEMPORAL_OPERATOR_VALIDATOR_VERSION:
            return None
        if not _certificate_is_compatible(
            certificate,
            identity=expected_identity,
            canonical_index=expected_canonical_index,
            artifact_manifest_sha256=artifact_manifest_sha256,
        ):
            return None
        if manifest.get("artifact_identity_fingerprint") != expected_identity.fingerprint:
            return None
        if manifest.get("artifact_manifest_sha256") != artifact_manifest_sha256:
            return None
        if manifest.get("canonical_index_fingerprint") != expected_canonical_index.artifact_fingerprint:
            return None
        if manifest.get("binding_fingerprint") != expected_canonical_index.binding_fingerprint:
            return None
        block_count = int(manifest["number_of_blocks"])
        number_of_measurements = int(manifest["number_of_measurements"])
        number_of_demand_cells = int(manifest["number_of_demand_cells"])
        total_nonzeros = int(manifest["nonzero_entries"])
        hashes = manifest.get("packed_array_hashes")
        if not isinstance(hashes, Mapping):
            return None
        filenames = (
            "row_indices.npy",
            "column_indices.npy",
            "values.npy",
            "offsets.npy",
            "fixed_measurement_offset.npy",
            "keys.json",
        )
        for filename in filenames:
            path = cache / filename
            if not path.is_file() or hashes.get(filename) != _file_sha256(path):
                return None
        rows = np.load(cache / "row_indices.npy", mmap_mode="r", allow_pickle=False)
        columns = np.load(cache / "column_indices.npy", mmap_mode="r", allow_pickle=False)
        values = np.load(cache / "values.npy", mmap_mode="r", allow_pickle=False)
        offsets = np.load(cache / "offsets.npy", mmap_mode="r", allow_pickle=False)
        fixed_offset = np.load(
            cache / "fixed_measurement_offset.npy", mmap_mode="r", allow_pickle=False
        )
        keys_payload = json.loads((cache / "keys.json").read_text(encoding="utf-8"))
        if (
            rows.shape != (total_nonzeros,)
            or columns.shape != (total_nonzeros,)
            or values.shape != (total_nonzeros,)
            or offsets.shape != (block_count + 1,)
            or fixed_offset.shape != (number_of_measurements,)
            or not isinstance(keys_payload, list)
            or len(keys_payload) != block_count
            or int(offsets[0]) != 0
            or int(offsets[-1]) != total_nonzeros
            or np.any(np.diff(offsets) < 0)
        ):
            return None
        blocks: list[TemporalSparseBlock] = []
        for position, key in enumerate(keys_payload):
            if not isinstance(key, Mapping):
                return None
            start, stop = int(offsets[position]), int(offsets[position + 1])
            blocks.append(
                TemporalSparseBlock(
                    key=TemporalBlockKey(**key),
                    row_indices=np.asarray(rows[start:stop]),
                    column_indices=np.asarray(columns[start:stop]),
                    values=np.asarray(values[start:stop]),
                    number_of_measurements=number_of_measurements,
                    number_of_demand_cells=number_of_demand_cells,
                )
            )
        diagnostics = TemporalBlockConstructionDiagnostics(
            **dict(manifest.get("diagnostics", {}))
        )
        if _array_content_hash(fixed_offset) != manifest.get("fixed_measurement_offset_hash"):
            return None
        return TemporalBlockAssignmentOperator(
            canonical_index=expected_canonical_index,
            identity=expected_identity,
            blocks=tuple(blocks),
            fixed_measurement_offset=np.asarray(fixed_offset),
            diagnostics=diagnostics,
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def load_temporal_block_operator(
    directory: str | Path,
    *,
    expected_identity: AssignmentArtifactIdentity,
    expected_canonical_index: CanonicalAssignmentIndex,
    reporter: ConstructionProgressReporter | None = None,
    validated_cache_directory: str | Path | None = None,
    preflight_manifest: str | Path | Mapping[str, object] | None = None,
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
    artifact_manifest_sha256 = _file_sha256(source / "manifest.json")
    if validated_cache_directory is not None:
        cache_directory = Path(validated_cache_directory)
        cached = _load_validated_operator_cache(
            cache_directory=cache_directory,
            expected_identity=expected_identity,
            expected_canonical_index=expected_canonical_index,
            artifact_manifest_sha256=artifact_manifest_sha256,
        )
        if cached is None and preflight_manifest is not None:
            try:
                certificate_path = adopt_completed_preflight(
                    preflight_manifest=preflight_manifest,
                    artifact_directory=source,
                    expected_identity=expected_identity,
                    expected_canonical_index=expected_canonical_index,
                    checkpoint_directory=cache_directory.parent.parent,
                )
                materialize_operator_cache_from_adopted_preflight(
                    artifact_directory=source,
                    certificate_path=certificate_path,
                    cache_directory=cache_directory,
                    reporter=reporter,
                )
                cached = _load_validated_operator_cache(
                    cache_directory=cache_directory,
                    expected_identity=expected_identity,
                    expected_canonical_index=expected_canonical_index,
                    artifact_manifest_sha256=artifact_manifest_sha256,
                )
            except (AssignmentCompatibilityError, OSError, TypeError, ValueError):
                # Adoption is an optimization.  If evidence is stale or
                # malformed, retain the safe full-validation fallback rather
                # than quarantining a still-valid source artifact.
                cached = None
        if cached is not None:
            if reporter is not None and reporter.sink is not None:
                reporter.emit(
                    phase=ConstructionPhase.CACHE_VALIDATION,
                    status="completed",
                    force=True,
                    completed_units=1,
                    total_units=1,
                    current_unit="validated_operator_cache",
                    checkpoint_location=str(cache_directory),
                    predicted_remaining_seconds=0.0,
                    eta_confidence="high",
                    eta_lower_seconds=0.0,
                    eta_upper_seconds=0.0,
                    throughput_units_per_second=1.0,
                    cache_hits=1,
                    cache_misses=0,
                    # Keep the historical temporal-block stage name for
                    # consumers that classify cache-validation events by
                    # substage, while recording that this was the packed-cache
                    # fast path and did not open source block files.
                    details={
                        "cache_validation_stage": "temporal_block_load",
                        "operator_cache_hit": True,
                    },
                )
            return cached
    reporting_enabled = reporter is not None and reporter.sink is not None
    recent_durations: deque[float] = deque(maxlen=16)
    phase_started_at = perf_counter() if reporting_enabled else None
    if reporting_enabled:
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
        block_started_at = perf_counter() if reporting_enabled else None
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
        if reporting_enabled:
            assert block_started_at is not None
            assert phase_started_at is not None
            completed_units = block_position + 1
            duration = max(0.0, perf_counter() - block_started_at)
            recent_durations.append(duration)
            phase_elapsed_seconds = max(
                np.finfo(float).eps, perf_counter() - phase_started_at
            )
            eta = estimate_completed_unit_eta(
                recent_durations,
                completed_units=completed_units,
                total_units=total_blocks,
                parallelism=1,
                elapsed_seconds=phase_elapsed_seconds,
            )
            reporter.emit(
                phase=ConstructionPhase.CACHE_VALIDATION,
                status="running",
                completed_units=completed_units,
                total_units=total_blocks,
                current_unit=item.get("filename"),
                checkpoint_location=str(source),
                recent_unit_seconds=duration,
                predicted_remaining_seconds=eta.predicted_remaining_seconds,
                eta_confidence=eta.eta_confidence,
                estimated_completion_at_utc=eta.estimated_completion_at_utc,
                eta_reason=eta.eta_reason,
                eta_lower_seconds=eta.eta_lower_seconds,
                eta_upper_seconds=eta.eta_upper_seconds,
                throughput_units_per_second=eta.throughput_units_per_second,
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
    if reporting_enabled:
        assert phase_started_at is not None
        final_eta = estimate_completed_unit_eta(
            recent_durations,
            completed_units=total_blocks,
            total_units=total_blocks,
            parallelism=1,
            elapsed_seconds=max(0.0, perf_counter() - phase_started_at),
        )
        reporter.emit(
            phase=ConstructionPhase.CACHE_VALIDATION,
            status="completed",
            force=True,
            completed_units=total_blocks,
            total_units=total_blocks,
            recent_unit_seconds=(
                recent_durations[-1] if recent_durations else None
            ),
            predicted_remaining_seconds=final_eta.predicted_remaining_seconds,
            eta_confidence=final_eta.eta_confidence,
            estimated_completion_at_utc=final_eta.estimated_completion_at_utc,
            eta_reason=final_eta.eta_reason,
            eta_lower_seconds=final_eta.eta_lower_seconds,
            eta_upper_seconds=final_eta.eta_upper_seconds,
            throughput_units_per_second=final_eta.throughput_units_per_second,
            checkpoint_location=str(source),
            details={"cache_validation_stage": "temporal_block_load"},
        )
    if validated_cache_directory is not None:
        cache_directory = Path(validated_cache_directory)
        certificate_path = cache_directory / "validation_certificate.json"
        certificate = {
            "complete": True,
            "schema_version": PREFLIGHT_ADOPTION_SCHEMA_VERSION,
            "validator_version": TEMPORAL_OPERATOR_VALIDATOR_VERSION,
            "provenance": {
                "source": "full_validation",
                "preflight_manifest_sha256": None,
                "preflight_completed_phase": None,
                "legacy_completed_preflight": False,
            },
            "artifact_identity_fingerprint": expected_identity.fingerprint,
            "artifact_manifest_sha256": artifact_manifest_sha256,
            "canonical_index_fingerprint": expected_canonical_index.artifact_fingerprint,
            "binding_fingerprint": expected_canonical_index.binding_fingerprint,
            "number_of_blocks": total_blocks,
            "fixed_measurement_offset_hash": manifest.get(
                "fixed_measurement_offset_hash"
            ),
        }
        _atomic_json_write(certificate_path, certificate)
        _materialize_operator_cache_from_operator(
            operator=operator,
            cache_directory=cache_directory,
            artifact_manifest_sha256=artifact_manifest_sha256,
            certificate_path=certificate_path,
            provenance_source="full_validation",
            reporter=reporter,
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
