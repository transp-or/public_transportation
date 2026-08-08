"""Canonical Phase-1 problem identities for reduced-dimensional OD inference."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from public_transportation.preprocessing.reduced_od.artifacts import canonical_json


REDUCED_OD_PROBLEM_SCHEMA_VERSION = 1


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _immutable_indices(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers.")
    result = np.array(array, dtype=np.int64, copy=True, order="C")
    if result.size and (np.any(result < 0) or np.any(result[1:] <= result[:-1])):
        raise ValueError(f"{name} must be strictly increasing and non-negative.")
    result.setflags(write=False)
    return result


def _immutable_values(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numbers.")
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    if np.any(result < 0.0):
        raise ValueError(f"{name} must contain non-negative values.")
    result.setflags(write=False)
    return result


def _array_descriptor(array: np.ndarray) -> dict[str, Any]:
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return {
        "digest": digest.hexdigest(),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }


@dataclass(frozen=True, slots=True, order=True)
class JourneyODTimeKey:
    """Canonical external key for one passenger-journey demand cell."""

    origin_stop_id: str
    dest_stop_id: str
    time_bin_id: str

    def __post_init__(self) -> None:
        _required_text(self.origin_stop_id, "origin_stop_id")
        _required_text(self.dest_stop_id, "dest_stop_id")
        _required_text(self.time_bin_id, "time_bin_id")

    @property
    def tuple(self) -> tuple[str, str, str]:
        return (self.origin_stop_id, self.dest_stop_id, self.time_bin_id)


@dataclass(frozen=True, slots=True)
class ReducedODProblemContract:
    """Immutable identity and exact statistical partition for one problem."""

    configuration_fingerprint: str
    timetable_artifact_fingerprint: str
    response_artifact_fingerprint: str
    od_keys: tuple[JourneyODTimeKey, ...]
    free_od_indices: np.ndarray
    fixed_od_indices: np.ndarray
    fixed_od_values: np.ndarray
    schema_version: int = REDUCED_OD_PROBLEM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REDUCED_OD_PROBLEM_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {REDUCED_OD_PROBLEM_SCHEMA_VERSION}."
            )
        _required_text(self.configuration_fingerprint, "configuration_fingerprint")
        _required_text(
            self.timetable_artifact_fingerprint,
            "timetable_artifact_fingerprint",
        )
        _required_text(
            self.response_artifact_fingerprint, "response_artifact_fingerprint"
        )
        if any(not isinstance(key, JourneyODTimeKey) for key in self.od_keys):
            raise TypeError("od_keys must contain JourneyODTimeKey values.")
        if len(set(self.od_keys)) != len(self.od_keys):
            raise ValueError("od_keys must be unique.")
        if self.od_keys != tuple(sorted(self.od_keys)):
            raise ValueError("od_keys must use canonical sorted order.")

        free = _immutable_indices(self.free_od_indices, "free_od_indices")
        fixed = _immutable_indices(self.fixed_od_indices, "fixed_od_indices")
        values = _immutable_values(self.fixed_od_values, "fixed_od_values")
        if fixed.size != values.size:
            raise ValueError(
                "fixed_od_indices and fixed_od_values must have equal length."
            )
        num_od = len(self.od_keys)
        if (free.size and free[-1] >= num_od) or (fixed.size and fixed[-1] >= num_od):
            raise ValueError("free/fixed OD indices are outside od_keys.")
        if np.intersect1d(free, fixed, assume_unique=True).size:
            raise ValueError("free and fixed OD indices must be disjoint.")
        combined = np.sort(np.concatenate((free, fixed)))
        if not np.array_equal(combined, np.arange(num_od, dtype=np.int64)):
            raise ValueError("free and fixed OD indices must partition every OD key.")
        object.__setattr__(self, "free_od_indices", free)
        object.__setattr__(self, "fixed_od_indices", fixed)
        object.__setattr__(self, "fixed_od_values", values)

    @property
    def num_od(self) -> int:
        return len(self.od_keys)

    @property
    def num_free_od(self) -> int:
        return int(self.free_od_indices.size)

    @property
    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "configuration_fingerprint": self.configuration_fingerprint,
            "fixed_od_indices": _array_descriptor(self.fixed_od_indices),
            "fixed_od_values": _array_descriptor(self.fixed_od_values),
            "free_od_indices": _array_descriptor(self.free_od_indices),
            "od_keys": [list(key.tuple) for key in self.od_keys],
            "response_artifact_fingerprint": self.response_artifact_fingerprint,
            "schema_version": self.schema_version,
            "timetable_artifact_fingerprint": (self.timetable_artifact_fingerprint),
        }

    @property
    def fingerprint_payload_json(self) -> str:
        return canonical_json(self.fingerprint_payload)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.fingerprint_payload_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ReducedODModelContract:
    """Small statistical-model identity layered on a problem contract."""

    problem_fingerprint: str
    model_name: str
    production_mode: Literal["provided", "estimated_basis"]
    likelihood: Literal["poisson", "negative_binomial"]
    estimated_parameters: tuple[str, ...]

    def __post_init__(self) -> None:
        _required_text(self.problem_fingerprint, "problem_fingerprint")
        _required_text(self.model_name, "model_name")
        if self.production_mode not in {"provided", "estimated_basis"}:
            raise ValueError("production_mode is unsupported.")
        if self.likelihood not in {"poisson", "negative_binomial"}:
            raise ValueError("likelihood is unsupported.")
        for parameter in self.estimated_parameters:
            _required_text(parameter, "estimated parameter")
        if len(set(self.estimated_parameters)) != len(self.estimated_parameters):
            raise ValueError("estimated_parameters must be unique.")
        if self.estimated_parameters != tuple(sorted(self.estimated_parameters)):
            raise ValueError("estimated_parameters must be sorted.")

    @property
    def fingerprint_payload_json(self) -> str:
        return canonical_json(
            {
                "estimated_parameters": list(self.estimated_parameters),
                "likelihood": self.likelihood,
                "model_name": self.model_name,
                "problem_fingerprint": self.problem_fingerprint,
                "production_mode": self.production_mode,
            }
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.fingerprint_payload_json.encode("utf-8")).hexdigest()
