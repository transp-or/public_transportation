"""Persistence for named non-OD measurement operators.

Boundary-flow operators are intentionally stored independently from the OD
routing artifact.  This keeps an unchanged assignment artifact reusable while
making the row order and model contract of each companion operator explicit.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from public_transportation.inference.block_coordinate._canonical import fingerprint

from .additive import GravityDenseLinearMeasurementOperator

GRAVITY_BOUNDARY_ARTIFACT_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True, slots=True)
class GravityBoundaryArtifact:
    """Validated persisted boundary-flow matrix and its provenance."""

    name: str
    matrix: np.ndarray
    row_order_fingerprint: str
    boundary_group_ids: tuple[str, ...]
    parameter_layout_fingerprint: str
    model_specification_fingerprint: str
    construction_configuration: Mapping[str, object]
    schema_version: int = GRAVITY_BOUNDARY_ARTIFACT_SCHEMA_VERSION
    source_path: Path | None = None
    file_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != GRAVITY_BOUNDARY_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported boundary-artifact schema version.")
        if not self.name:
            raise ValueError("boundary artifact name must be non-empty.")
        matrix = np.asarray(self.matrix)
        if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
            raise ValueError("boundary artifact matrix must be non-empty and two-dimensional.")
        if matrix.dtype.kind not in "fiu" or not np.all(np.isfinite(matrix)):
            raise ValueError("boundary artifact matrix must be finite and numeric.")
        matrix = np.asarray(matrix, dtype=np.result_type(matrix.dtype, np.float32))
        matrix = np.array(matrix, copy=True)
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        groups = tuple(str(value) for value in self.boundary_group_ids)
        if groups and len(groups) != matrix.shape[1]:
            raise ValueError(
                "boundary_group_ids must be empty or contain one identifier per column."
            )
        object.__setattr__(self, "boundary_group_ids", groups)
        object.__setattr__(self, "construction_configuration", dict(self.construction_configuration))
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path).resolve())

    @property
    def matrix_fingerprint(self) -> str:
        return fingerprint({"matrix": self.matrix, "dtype": str(self.matrix.dtype)})

    @property
    def content_sha256(self) -> str:
        return fingerprint(self.identity_payload())

    @property
    def operator(self) -> GravityDenseLinearMeasurementOperator:
        return GravityDenseLinearMeasurementOperator(self.matrix)

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "shape": list(self.matrix.shape),
            "dtype": str(self.matrix.dtype),
            "matrix_fingerprint": self.matrix_fingerprint,
            "row_order_fingerprint": self.row_order_fingerprint,
            "boundary_group_ids": list(self.boundary_group_ids),
            "parameter_layout_fingerprint": self.parameter_layout_fingerprint,
            "model_specification_fingerprint": self.model_specification_fingerprint,
            "construction_configuration": dict(self.construction_configuration),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "matrix_file": "matrix.npy",
            "file_sha256": self.file_sha256,
            "complete": True,
            "content_sha256": self.content_sha256,
        }


def write_gravity_boundary_artifact(
    directory: Path,
    *,
    name: str,
    operator: GravityDenseLinearMeasurementOperator,
    row_order_fingerprint: str,
    boundary_group_ids: Sequence[str] = (),
    parameter_layout_fingerprint: str,
    model_specification_fingerprint: str,
    construction_configuration: Mapping[str, object] | None = None,
) -> GravityBoundaryArtifact:
    """Write a complete boundary artifact and return its validated metadata."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    matrix_path = destination / "matrix.npy"
    temporary = matrix_path.with_name(f".{matrix_path.name}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(operator.matrix), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, matrix_path)
    artifact = GravityBoundaryArtifact(
        name=name,
        matrix=np.asarray(operator.matrix),
        row_order_fingerprint=row_order_fingerprint,
        boundary_group_ids=tuple(boundary_group_ids),
        parameter_layout_fingerprint=parameter_layout_fingerprint,
        model_specification_fingerprint=model_specification_fingerprint,
        construction_configuration=(
            {} if construction_configuration is None else construction_configuration
        ),
        source_path=destination,
        file_sha256=_sha256_file(matrix_path),
    )
    _atomic_json(destination / "manifest.json", artifact.to_dict())
    return artifact


def load_gravity_boundary_artifact(
    directory: Path,
    *,
    expected_row_order_fingerprint: str | None = None,
    expected_parameter_layout_fingerprint: str | None = None,
    expected_model_specification_fingerprint: str | None = None,
) -> GravityBoundaryArtifact:
    """Load and validate a boundary artifact without touching OD artifacts."""
    source = Path(directory)
    payload = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != GRAVITY_BOUNDARY_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported boundary-artifact schema version.")
    if payload.get("complete") is not True:
        raise ValueError("boundary artifact is not complete.")
    matrix_path = source / str(payload.get("matrix_file", "matrix.npy"))
    if not matrix_path.is_file():
        raise ValueError("boundary artifact matrix is missing.")
    actual_file_hash = _sha256_file(matrix_path)
    if actual_file_hash != payload.get("file_sha256"):
        raise ValueError("boundary artifact matrix file fingerprint mismatch.")
    matrix = np.load(matrix_path, allow_pickle=False)
    artifact = GravityBoundaryArtifact(
        name=str(payload["name"]),
        matrix=matrix,
        row_order_fingerprint=str(payload["row_order_fingerprint"]),
        boundary_group_ids=tuple(payload.get("boundary_group_ids", ())),
        parameter_layout_fingerprint=str(payload["parameter_layout_fingerprint"]),
        model_specification_fingerprint=str(payload["model_specification_fingerprint"]),
        construction_configuration=dict(payload.get("construction_configuration", {})),
        source_path=source,
        file_sha256=actual_file_hash,
    )
    if artifact.matrix_fingerprint != payload.get("matrix_fingerprint"):
        raise ValueError("boundary artifact matrix content fingerprint mismatch.")
    if artifact.content_sha256 != payload.get("content_sha256"):
        raise ValueError("boundary artifact identity fingerprint mismatch.")
    checks = (
        (expected_row_order_fingerprint, artifact.row_order_fingerprint, "row-order"),
        (
            expected_parameter_layout_fingerprint,
            artifact.parameter_layout_fingerprint,
            "parameter-layout",
        ),
        (
            expected_model_specification_fingerprint,
            artifact.model_specification_fingerprint,
            "model-specification",
        ),
    )
    for expected, actual, label in checks:
        if expected is not None and expected != actual:
            raise ValueError(f"boundary artifact {label} fingerprint mismatch.")
    return artifact
