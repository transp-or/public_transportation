"""Immutable, fingerprinted Phase-1 contracts for reduced-OD artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np


REDUCED_OD_ARTIFACT_SCHEMA_VERSION = 1
REDUCED_OD_IMPLEMENTATION_VERSION = "reduced-od-contracts-v1"


class ReducedODArtifactKind(str, Enum):
    """Artifact categories whose numerical payloads arrive in later phases."""

    TIMETABLE = "timetable"
    JOURNEY_CHOICES = "journey_choices"
    MEASUREMENT_RESPONSE = "measurement_response"


def canonical_json(value: Any) -> str:
    """Return deterministic compact JSON for a JSON-compatible value."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def fingerprint_json(value: Any) -> str:
    """Hash a JSON-compatible value using the package's canonical encoding."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _sorted_unique_pairs(
    values: tuple[tuple[str, str], ...], name: str
) -> tuple[tuple[str, str], ...]:
    parsed = tuple(
        (_required_text(key, f"{name} key"), _required_text(value, f"{name} value"))
        for key, value in values
    )
    if len({key for key, _ in parsed}) != len(parsed):
        raise ValueError(f"{name} keys must be unique.")
    if parsed != tuple(sorted(parsed)):
        raise ValueError(f"{name} must be sorted by key.")
    return parsed


def _sorted_unique_dimensions(
    values: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, int], ...]:
    parsed: list[tuple[str, int]] = []
    for key, value in values:
        _required_text(key, "dimension key")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"dimension {key!r} must be a non-negative integer."
            )
        parsed.append((key, value))
    result = tuple(parsed)
    if len({key for key, _ in result}) != len(result):
        raise ValueError("dimension keys must be unique.")
    if result != tuple(sorted(result)):
        raise ValueError("dimensions must be sorted by key.")
    return result


@dataclass(frozen=True, slots=True)
class ReducedODArtifactManifest:
    """Small immutable identity shared by all persisted reduced-OD artifacts."""

    artifact_kind: ReducedODArtifactKind
    configuration_fingerprint: str
    source_fingerprints: tuple[tuple[str, str], ...]
    dimensions: tuple[tuple[str, int], ...] = ()
    schema_version: int = REDUCED_OD_ARTIFACT_SCHEMA_VERSION
    implementation_version: str = REDUCED_OD_IMPLEMENTATION_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_kind, ReducedODArtifactKind):
            raise TypeError("artifact_kind must be a ReducedODArtifactKind.")
        if self.schema_version != REDUCED_OD_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {REDUCED_OD_ARTIFACT_SCHEMA_VERSION}."
            )
        _required_text(self.implementation_version, "implementation_version")
        _required_text(
            self.configuration_fingerprint, "configuration_fingerprint"
        )
        object.__setattr__(
            self,
            "source_fingerprints",
            _sorted_unique_pairs(
                self.source_fingerprints, "source_fingerprints"
            ),
        )
        object.__setattr__(
            self, "dimensions", _sorted_unique_dimensions(self.dimensions)
        )

    @property
    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind.value,
            "configuration_fingerprint": self.configuration_fingerprint,
            "dimensions": [list(item) for item in self.dimensions],
            "implementation_version": self.implementation_version,
            "schema_version": self.schema_version,
            "source_fingerprints": [
                list(item) for item in self.source_fingerprints
            ],
        }

    @property
    def fingerprint_payload_json(self) -> str:
        return canonical_json(self.fingerprint_payload)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            self.fingerprint_payload_json.encode("utf-8")
        ).hexdigest()


def _immutable_array(value: Any) -> np.ndarray:
    array = np.array(value, copy=True, order="C")
    if array.dtype.hasobject:
        raise TypeError("artifact arrays must not use object dtype.")
    array.setflags(write=False)
    return array


def _array_digest(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NamedImmutableArray:
    """One named, owned, C-contiguous, read-only artifact array."""

    name: str
    values: np.ndarray

    def __post_init__(self) -> None:
        _required_text(self.name, "array name")
        object.__setattr__(self, "values", _immutable_array(self.values))

    @property
    def descriptor(self) -> dict[str, Any]:
        return {
            "digest": _array_digest(self.values),
            "dtype": self.values.dtype.str,
            "name": self.name,
            "shape": list(self.values.shape),
        }


@dataclass(frozen=True, slots=True)
class ReducedODArrayArtifact:
    """A manifest and canonical tuple of immutable numerical arrays."""

    manifest: ReducedODArtifactManifest
    arrays: tuple[NamedImmutableArray, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ReducedODArtifactManifest):
            raise TypeError("manifest must be a ReducedODArtifactManifest.")
        if any(not isinstance(item, NamedImmutableArray) for item in self.arrays):
            raise TypeError("arrays must contain NamedImmutableArray values.")
        names = tuple(item.name for item in self.arrays)
        if len(set(names)) != len(names):
            raise ValueError("array names must be unique.")
        if names != tuple(sorted(names)):
            raise ValueError("arrays must be sorted by name.")

    @property
    def fingerprint_payload(self) -> Mapping[str, Any]:
        return {
            "arrays": [item.descriptor for item in self.arrays],
            "manifest_fingerprint": self.manifest.fingerprint,
        }

    @property
    def fingerprint_payload_json(self) -> str:
        return canonical_json(self.fingerprint_payload)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            self.fingerprint_payload_json.encode("utf-8")
        ).hexdigest()

    @property
    def retained_bytes(self) -> int:
        return sum(int(item.values.nbytes) for item in self.arrays)
