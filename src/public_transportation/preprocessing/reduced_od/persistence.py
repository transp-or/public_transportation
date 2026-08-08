"""Atomic fail-closed persistence for reduced measurement responses."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from public_transportation.measurement.schema import MeasurementType

from .artifacts import canonical_json
from .equivalence import ResponseEquivalence
from .response_atoms import (
    MeasurementResponseArtifact,
    ResolvedMeasurement,
    ResponseCellKey,
)


MEASUREMENT_RESPONSE_CACHE_SCHEMA_VERSION = 1


class MeasurementResponseCacheError(ValueError):
    """A response cache is corrupt, incomplete, or has the wrong identity."""


_ARRAY_NAMES = {
    "class_by_cell",
    "fixed_offset",
    "free_cell_index",
    "measurement_index",
    "member_cell_indices",
    "member_indptr",
    "metadata_json",
    "observed_values",
    "representative_cell_indices",
    "response_values",
}


def _metadata(artifact: MeasurementResponseArtifact) -> dict[str, object]:
    return {
        "artifact_fingerprint": artifact.fingerprint,
        "configuration_fingerprint": artifact.configuration_fingerprint,
        "fixed_cell_keys": [list(key.tuple) for key in artifact.fixed_cell_keys],
        "free_cell_keys": [list(key.tuple) for key in artifact.free_cell_keys],
        "journey_choice_fingerprint": artifact.journey_choice_fingerprint,
        "measurement_fingerprint": artifact.measurement_fingerprint,
        "resolved_measurements": [
            [
                item.row_index,
                item.method_id,
                item.measurement_type.value,
                item.scenario_stop_id,
                item.physical_stop_id,
                item.seconds,
                item.trip_id,
                item.line_id,
                item.observed_value,
            ]
            for item in artifact.resolved_measurements
        ],
        "schema_version": MEASUREMENT_RESPONSE_CACHE_SCHEMA_VERSION,
        "timetable_fingerprint": artifact.timetable_fingerprint,
    }


def save_measurement_response_cache(
    path: str | Path, artifact: MeasurementResponseArtifact
) -> None:
    """Atomically save one compressed response cache without pickle payloads."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(
                stream,
                metadata_json=np.asarray(canonical_json(_metadata(artifact))),
                observed_values=artifact.observed_values,
                measurement_index=artifact.measurement_index,
                free_cell_index=artifact.free_cell_index,
                response_values=artifact.response_values,
                fixed_offset=artifact.fixed_offset,
                class_by_cell=artifact.equivalence.class_by_cell,
                representative_cell_indices=(
                    artifact.equivalence.representative_cell_indices
                ),
                member_indptr=artifact.equivalence.member_indptr,
                member_cell_indices=artifact.equivalence.member_cell_indices,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _text(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise MeasurementResponseCacheError(f"{location} must be non-empty text.")
    return value


def _keys(value: object, location: str) -> tuple[ResponseCellKey, ...]:
    if not isinstance(value, list):
        raise MeasurementResponseCacheError(f"{location} must be a list.")
    try:
        result = tuple(ResponseCellKey(*item) for item in value)
    except (TypeError, ValueError) as error:
        raise MeasurementResponseCacheError(f"{location} is invalid.") from error
    return result


def _resolved(value: object) -> tuple[ResolvedMeasurement, ...]:
    if not isinstance(value, list):
        raise MeasurementResponseCacheError("resolved_measurements must be a list.")
    result: list[ResolvedMeasurement] = []
    try:
        for item in value:
            if not isinstance(item, list) or len(item) != 9:
                raise ValueError("invalid resolved measurement row")
            result.append(
                ResolvedMeasurement(
                    row_index=int(item[0]),
                    method_id=str(item[1]),
                    measurement_type=MeasurementType(str(item[2])),
                    scenario_stop_id=str(item[3]),
                    physical_stop_id=str(item[4]),
                    seconds=int(item[5]),
                    trip_id=str(item[6]),
                    line_id=str(item[7]),
                    observed_value=float(item[8]),
                )
            )
    except (TypeError, ValueError) as error:
        raise MeasurementResponseCacheError(
            "resolved_measurements contains invalid values."
        ) from error
    return tuple(result)


def load_measurement_response_cache(
    path: str | Path,
    *,
    expected_configuration_fingerprint: str | None = None,
    expected_timetable_fingerprint: str | None = None,
    expected_journey_choice_fingerprint: str | None = None,
    expected_measurement_fingerprint: str | None = None,
) -> MeasurementResponseArtifact:
    """Load and fully validate one cache, including optional expected identity."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise MeasurementResponseCacheError(f"response cache does not exist: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != _ARRAY_NAMES:
                raise MeasurementResponseCacheError(
                    "response cache members do not match the schema."
                )
            metadata_value = archive["metadata_json"]
            if metadata_value.shape != () or metadata_value.dtype.kind not in {"U", "S"}:
                raise MeasurementResponseCacheError("metadata_json is not scalar text.")
            metadata = json.loads(str(metadata_value.item()))
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in _ARRAY_NAMES - {"metadata_json"}
            }
    except MeasurementResponseCacheError:
        raise
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise MeasurementResponseCacheError(
            f"response cache cannot be decoded: {source}"
        ) from error
    if not isinstance(metadata, dict):
        raise MeasurementResponseCacheError("cache metadata must be a JSON object.")
    if metadata.get("schema_version") != MEASUREMENT_RESPONSE_CACHE_SCHEMA_VERSION:
        raise MeasurementResponseCacheError("unsupported response cache schema version.")
    artifact = MeasurementResponseArtifact(
        configuration_fingerprint=_text(
            metadata.get("configuration_fingerprint"),
            "configuration_fingerprint",
        ),
        timetable_fingerprint=_text(
            metadata.get("timetable_fingerprint"), "timetable_fingerprint"
        ),
        journey_choice_fingerprint=_text(
            metadata.get("journey_choice_fingerprint"),
            "journey_choice_fingerprint",
        ),
        measurement_fingerprint=_text(
            metadata.get("measurement_fingerprint"), "measurement_fingerprint"
        ),
        free_cell_keys=_keys(metadata.get("free_cell_keys"), "free_cell_keys"),
        fixed_cell_keys=_keys(metadata.get("fixed_cell_keys"), "fixed_cell_keys"),
        resolved_measurements=_resolved(metadata.get("resolved_measurements")),
        observed_values=arrays["observed_values"],
        measurement_index=arrays["measurement_index"],
        free_cell_index=arrays["free_cell_index"],
        response_values=arrays["response_values"],
        fixed_offset=arrays["fixed_offset"],
        equivalence=ResponseEquivalence(
            class_by_cell=arrays["class_by_cell"],
            representative_cell_indices=arrays[
                "representative_cell_indices"
            ],
            member_indptr=arrays["member_indptr"],
            member_cell_indices=arrays["member_cell_indices"],
        ),
    )
    stored_fingerprint = _text(
        metadata.get("artifact_fingerprint"), "artifact_fingerprint"
    )
    if artifact.fingerprint != stored_fingerprint:
        raise MeasurementResponseCacheError(
            "response cache content does not match its stored fingerprint."
        )
    expectations = (
        (
            expected_configuration_fingerprint,
            artifact.configuration_fingerprint,
            "configuration",
        ),
        (expected_timetable_fingerprint, artifact.timetable_fingerprint, "timetable"),
        (
            expected_journey_choice_fingerprint,
            artifact.journey_choice_fingerprint,
            "journey choices",
        ),
        (
            expected_measurement_fingerprint,
            artifact.measurement_fingerprint,
            "measurements",
        ),
    )
    for expected, actual, name in expectations:
        if expected is not None and expected != actual:
            raise MeasurementResponseCacheError(
                f"response cache {name} fingerprint does not match the expected value."
            )
    return artifact
