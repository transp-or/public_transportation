"""Parent-aware artifacts for scheduled temporal assignment.

The hierarchy in this module deliberately separates scientific identity from
execution controls.  It is small enough to use for examples and complete
enough to serve as the persistence contract for production builders.

The numerical payload of :class:`EstimationAssignmentMapping` may be dense or
SciPy sparse.  Production scheduled operators can also use a companion
payload (for example, a validated block-sharded operator) through the same
manifest contract without materialising a dense full-network matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
from time import monotonic
from typing import Callable, Literal, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse

from .measurement_operator_protocol import GravityOperatorCapabilities
from .checkpoint_policy import CheckpointPolicy, normalize_checkpoint_policy


HIERARCHICAL_ARTIFACT_SCHEMA_VERSION = 3
HIERARCHICAL_PROGRESS_SCHEMA_VERSION = 1

ArtifactLayer = Literal[
    "scenario_base",
    "canonical_od_time_universe",
    "feasibility_support",
    "route_choice_basis",
    "route_choice_materialization",
    "full_od_assignment_mapping",
    "observation_projection",
    "estimation_assignment_mapping",
]

ARTIFACT_LAYERS: tuple[ArtifactLayer, ...] = (
    "scenario_base",
    "canonical_od_time_universe",
    "feasibility_support",
    "route_choice_basis",
    "route_choice_materialization",
    "full_od_assignment_mapping",
    "observation_projection",
    "estimation_assignment_mapping",
)

# The layers are topologically ordered, but their identities are not a
# linear chain.  Each manifest records only the scientific parents that it
# actually consumes.  In particular, L7 is a join of the full assignment map
# and the observation projection; changing observation mapping metadata does
# not invalidate L5.
DAG_PARENT_LAYERS: Mapping[ArtifactLayer, tuple[ArtifactLayer, ...]] = {
    "scenario_base": (),
    "canonical_od_time_universe": ("scenario_base",),
    "feasibility_support": ("canonical_od_time_universe",),
    "route_choice_basis": ("feasibility_support",),
    "route_choice_materialization": ("route_choice_basis",),
    "full_od_assignment_mapping": ("route_choice_materialization",),
    "observation_projection": ("full_od_assignment_mapping",),
    "estimation_assignment_mapping": (
        "full_od_assignment_mapping",
        "observation_projection",
    ),
}

_LAYER_SET = frozenset(ARTIFACT_LAYERS)
_VALID_STATUSES = frozenset(
    {"started", "running", "heartbeat", "reused", "completed", "failed", "interrupted"}
)


def package_revision() -> str:
    """Return the package revision recorded in new artifact manifests.

    Deployments can provide an exact source revision through the environment;
    installed distributions otherwise contribute their package version.  The
    value is metadata, but it is checked before a persisted numerical payload
    is accepted.
    """

    explicit = os.environ.get("PUBLIC_TRANSPORTATION_GIT_REVISION")
    if explicit and explicit.strip():
        return explicit.strip()
    try:
        return importlib.metadata.version("public-transportation")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _json_default(value: object) -> object:
    """Encode common scientific metadata without accepting non-canonical text."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: object) -> str:
    """Return the canonical JSON representation used for scientific hashes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )


def fingerprint(value: object) -> str:
    """Hash canonical JSON metadata."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file in bounded memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ObsoleteArtifactFormatError(RuntimeError):
    """Raised when a pre-hierarchy artifact is encountered."""

    def __init__(self, path: str | Path, *, artifact_layer: str | None = None):
        self.path = str(path)
        self.artifact_layer = artifact_layer
        layer = "" if artifact_layer is None else f" for layer {artifact_layer!r}"
        super().__init__(
            f"artifact{layer} at {self.path} uses the obsolete monolithic format; "
            "regenerate the hierarchical artifact explicitly."
        )


class HierarchicalArtifactUnavailableError(RuntimeError):
    """Structured failure for strict reuse-only activation."""

    def __init__(
        self,
        *,
        artifact_layer: str,
        expected_fingerprint: str,
        artifact_path: str | Path,
        reason_code: str,
        details: Mapping[str, object] | None = None,
        recommended_command: str | None = None,
    ) -> None:
        self.artifact_layer = str(artifact_layer)
        self.expected_fingerprint = str(expected_fingerprint)
        self.artifact_path = str(artifact_path)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})
        self.recommended_command = recommended_command
        self.actual_fingerprint = self.details.get("actual_fingerprint")
        self.expected_parent_fingerprints = self.details.get(
            "expected_parent_fingerprints"
        )
        self.actual_parent_fingerprints = self.details.get(
            "actual_parent_fingerprints"
        )
        self.first_incompatible_field = self.details.get("first_incompatible_field")
        message = (
            f"hierarchical artifact unavailable ({self.artifact_layer}): "
            f"{self.reason_code}; expected fingerprint={self.expected_fingerprint}; "
            f"path={self.artifact_path}"
        )
        if self.details:
            message += f"; details={self.details}"
        if recommended_command:
            message += f"; run={recommended_command}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class HierarchicalArtifactManifest:
    """Validated metadata for one completed hierarchy layer."""

    artifact_layer: ArtifactLayer
    artifact_schema_version: int
    fingerprint: str
    parent_fingerprints: Mapping[str, str]
    scientific_parameters: Mapping[str, object]
    input_fingerprints: Mapping[str, str]
    execution_parameters: Mapping[str, object]
    status: str
    checksums: Mapping[str, str] = field(default_factory=dict)
    payload: Mapping[str, object] = field(default_factory=dict)
    file_sizes: Mapping[str, int] = field(default_factory=dict)
    package_revision: str = field(default_factory=package_revision)
    identity_basis_fingerprint: str = ""
    configuration_fingerprint: str = ""
    parent_identity_fingerprint: str | None = None

    @property
    def artifact_type(self) -> str:
        return self.artifact_layer

    @property
    def identity_payload(self) -> dict[str, object]:
        """The execution-independent identity payload."""

        return {
            "artifact_type": self.artifact_layer,
            "artifact_schema_version": self.artifact_schema_version,
            "parent_fingerprints": dict(self.parent_fingerprints),
            "scientific_parameters": dict(self.scientific_parameters),
            "input_fingerprints": dict(self.input_fingerprints),
            "package_revision": self.package_revision,
        }

    @property
    def identity_basis_payload(self) -> dict[str, object]:
        """Identity without the package revision, for cheap diagnostics."""

        return {
            "artifact_type": self.artifact_layer,
            "artifact_schema_version": self.artifact_schema_version,
            "parent_fingerprints": dict(self.parent_fingerprints),
            "scientific_parameters": dict(self.scientific_parameters),
            "input_fingerprints": dict(self.input_fingerprints),
        }

    @property
    def computed_fingerprint(self) -> str:
        return fingerprint(self.identity_payload)

    def __post_init__(self) -> None:
        if self.artifact_layer not in _LAYER_SET:
            raise ValueError(f"unknown hierarchy layer: {self.artifact_layer!r}")
        if self.artifact_schema_version != HIERARCHICAL_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported hierarchical artifact schema version")
        if self.status not in {"completed", "started", "running", "failed", "interrupted"}:
            raise ValueError(f"unsupported artifact status: {self.status!r}")
        if self.fingerprint != self.computed_fingerprint:
            raise ValueError(
                "hierarchical artifact fingerprint does not match canonical identity"
            )
        expected_basis = fingerprint(self.identity_basis_payload)
        if self.identity_basis_fingerprint and self.identity_basis_fingerprint != expected_basis:
            raise ValueError("hierarchical identity basis fingerprint is invalid")
        if not self.package_revision.strip():
            raise ValueError("package revision must be nonempty")
        expected_configuration = fingerprint(self.execution_parameters)
        if self.configuration_fingerprint and self.configuration_fingerprint != expected_configuration:
            raise ValueError("hierarchical configuration fingerprint is invalid")
        for name, values in (
            ("parent_fingerprints", self.parent_fingerprints),
            ("input_fingerprints", self.input_fingerprints),
        ):
            if any(not isinstance(key, str) or not isinstance(value, str) for key, value in values.items()):
                raise ValueError(f"{name} must contain string keys and values")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_type": self.artifact_layer,
            "artifact_layer": self.artifact_layer,
            "artifact_schema_version": self.artifact_schema_version,
            "fingerprint": self.fingerprint,
            "parent_fingerprints": dict(self.parent_fingerprints),
            "scientific_parameters": dict(self.scientific_parameters),
            "input_fingerprints": dict(self.input_fingerprints),
            "execution_parameters": dict(self.execution_parameters),
            "status": self.status,
            "checksums": dict(self.checksums),
            "payload": dict(self.payload),
            "file_sizes": {key: int(value) for key, value in self.file_sizes.items()},
            "package_revision": self.package_revision,
            "identity_basis_fingerprint": (
                self.identity_basis_fingerprint
                or fingerprint(self.identity_basis_payload)
            ),
            "configuration_fingerprint": (
                self.configuration_fingerprint
                or fingerprint(self.execution_parameters)
            ),
            "parent_identity_fingerprint": self.parent_identity_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "HierarchicalArtifactManifest":
        layer = value.get("artifact_layer", value.get("artifact_type"))
        if not isinstance(layer, str) or layer not in _LAYER_SET:
            raise ObsoleteArtifactFormatError("manifest", artifact_layer=str(layer))
        if value.get("artifact_schema_version") != HIERARCHICAL_ARTIFACT_SCHEMA_VERSION:
            raise ObsoleteArtifactFormatError("manifest", artifact_layer=layer)
        try:
            return cls(
                artifact_layer=layer,  # type: ignore[arg-type]
                artifact_schema_version=int(value["artifact_schema_version"]),
                fingerprint=str(value["fingerprint"]),
                parent_fingerprints=dict(value.get("parent_fingerprints", {})),
                scientific_parameters=dict(value.get("scientific_parameters", {})),
                input_fingerprints=dict(value.get("input_fingerprints", {})),
                execution_parameters=dict(value.get("execution_parameters", {})),
                status=str(value["status"]),
                checksums=dict(value.get("checksums", {})),
                payload=dict(value.get("payload", {})),
                file_sizes={
                    str(key): int(item)
                    for key, item in dict(value.get("file_sizes", {})).items()
                },
                package_revision=str(value.get("package_revision", "unknown")),
                identity_basis_fingerprint=str(
                    value.get("identity_basis_fingerprint", "")
                ),
                configuration_fingerprint=str(
                    value.get("configuration_fingerprint", "")
                ),
                parent_identity_fingerprint=(
                    None
                    if value.get("parent_identity_fingerprint") is None
                    else str(value["parent_identity_fingerprint"])
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("hierarchical artifact manifest is invalid") from error


def make_manifest(
    *,
    artifact_layer: ArtifactLayer,
    parent_fingerprints: Mapping[str, str],
    scientific_parameters: Mapping[str, object] | None = None,
    input_fingerprints: Mapping[str, str] | None = None,
    execution_parameters: Mapping[str, object] | None = None,
    status: str = "completed",
    checksums: Mapping[str, str] | None = None,
    payload: Mapping[str, object] | None = None,
    package_revision_value: str | None = None,
) -> HierarchicalArtifactManifest:
    """Create a manifest whose identity excludes execution-only settings."""

    revision = package_revision_value or package_revision()
    identity_basis = {
        "artifact_type": artifact_layer,
        "artifact_schema_version": HIERARCHICAL_ARTIFACT_SCHEMA_VERSION,
        "parent_fingerprints": dict(parent_fingerprints),
        "scientific_parameters": dict(scientific_parameters or {}),
        "input_fingerprints": dict(input_fingerprints or {}),
    }
    identity = {**identity_basis, "package_revision": revision}
    execution = dict(execution_parameters or {})
    return HierarchicalArtifactManifest(
        artifact_layer=artifact_layer,
        artifact_schema_version=HIERARCHICAL_ARTIFACT_SCHEMA_VERSION,
        fingerprint=fingerprint(identity),
        parent_fingerprints=dict(parent_fingerprints),
        scientific_parameters=dict(scientific_parameters or {}),
        input_fingerprints=dict(input_fingerprints or {}),
        execution_parameters=execution,
        status=status,
        checksums=dict(checksums or {}),
        payload=dict(payload or {}),
        package_revision=revision,
        identity_basis_fingerprint=fingerprint(identity_basis),
        configuration_fingerprint=fingerprint(execution),
        parent_identity_fingerprint=(
            None if not parent_fingerprints else fingerprint(dict(parent_fingerprints))
        ),
    )


def derive_layer_fingerprints(
    specifications: Mapping[str, Mapping[str, object]],
    *,
    package_revision_value: str | None = None,
) -> dict[str, str]:
    """Derive deterministic fingerprints from the explicit DAG parent map.

    ``specifications`` may omit execution parameters.  Each node identity
    contains only its direct scientific parents; callers can recover the full
    transitive lineage by following those parent manifests.
    """

    revision = package_revision_value or package_revision()
    result: dict[str, str] = {}
    for layer in ARTIFACT_LAYERS:
        raw = specifications.get(layer, {})
        try:
            parent = {name: result[name] for name in DAG_PARENT_LAYERS[layer]}
        except KeyError as error:  # pragma: no cover - protects future graph edits
            raise ValueError(
                f"hierarchy parent {error.args[0]!r} is not topologically ordered"
            ) from error
        result[layer] = fingerprint(
            {
                "artifact_type": layer,
                "artifact_schema_version": HIERARCHICAL_ARTIFACT_SCHEMA_VERSION,
                "parent_fingerprints": parent,
                "scientific_parameters": dict(raw.get("scientific_parameters", {})),
                "input_fingerprints": dict(raw.get("input_fingerprints", {})),
                "package_revision": revision,
            }
        )
    return result


def _file_checksums(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not root.exists():
        return result
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in {"manifest.json", "COMPLETE.json"}:
            continue
        result[str(path.relative_to(root))] = file_sha256(path)
    return result


def _file_sizes(root: Path) -> dict[str, int]:
    """Return declared payload sizes without reading payload contents."""

    result: dict[str, int] = {}
    if not root.exists():
        return result
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in {"manifest.json", "COMPLETE.json"}:
            continue
        result[str(path.relative_to(root))] = int(path.stat().st_size)
    return result


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reason_code_from_error(message: str, fallback: str) -> str:
    """Map fast metadata failures to stable diagnostics."""

    for prefix, reason in (
        ("completion_marker_missing", "completion_marker_missing"),
        ("completion_marker_invalid", "completion_marker_invalid"),
        ("output_missing:", "output_missing"),
        ("output_size_mismatch:", "output_size_mismatch"),
        ("checkpoint_incomplete", "checkpoint_incomplete"),
        ("hierarchical artifact payload checksum", "payload_corrupt"),
        ("package revision", "package_revision_mismatch"),
    ):
        if prefix in message:
            return reason
    return fallback


class HierarchicalArtifactStore:
    """Atomic, checksum-validating storage for the hierarchy."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def artifact_path(self, layer: ArtifactLayer | str, layer_fingerprint: str) -> Path:
        return self.root / "artifacts" / str(layer) / str(layer_fingerprint)

    def checkpoint_path(self, layer: ArtifactLayer | str, layer_fingerprint: str) -> Path:
        return self.root / "checkpoints" / str(layer) / str(layer_fingerprint)

    def manifest_path(self, layer: ArtifactLayer | str, layer_fingerprint: str) -> Path:
        return self.artifact_path(layer, layer_fingerprint) / "manifest.json"

    def checkpoint_manifest_path(
        self, layer: ArtifactLayer | str, layer_fingerprint: str
    ) -> Path:
        return self.checkpoint_path(layer, layer_fingerprint) / "manifest.json"

    def load(
        self,
        layer: ArtifactLayer,
        layer_fingerprint: str,
        *,
        expected_parents: Mapping[str, str] | None = None,
        verify: Literal["fast", "full"] = "fast",
        validate_checksums: bool | None = None,
        _namespace: Literal["artifacts", "checkpoints"] = "artifacts",
    ) -> HierarchicalArtifactManifest:
        if validate_checksums is not None:
            verify = "full" if validate_checksums else "fast"
        if verify not in {"fast", "full"}:
            raise ValueError("verify must be 'fast' or 'full'")
        path = (
            self.artifact_path(layer, layer_fingerprint)
            if _namespace == "artifacts"
            else self.checkpoint_path(layer, layer_fingerprint)
        ) / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            manifest = HierarchicalArtifactManifest.from_dict(payload)
        except ObsoleteArtifactFormatError:
            raise ObsoleteArtifactFormatError(path, artifact_layer=layer)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid hierarchical manifest: {path}") from error
        if manifest.artifact_layer != layer or manifest.fingerprint != layer_fingerprint:
            raise ValueError("hierarchical artifact layer or fingerprint mismatch")
        if manifest.status != "completed":
            raise ValueError("checkpoint_incomplete")
        marker_path = path.parent / "COMPLETE.json"
        if not marker_path.is_file():
            raise ValueError("completion_marker_missing")
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError) as error:
            raise ValueError("completion_marker_missing") from error
        if (
            marker.get("identity_fingerprint") != manifest.fingerprint
            or marker.get("manifest_sha256") != file_sha256(path)
        ):
            raise ValueError("completion_marker_invalid")
        # Empty payloads are valid for hierarchy layers whose scientific
        # state is represented entirely by their manifest.  When files are
        # declared, however, their existence and recorded sizes are part of
        # the cheap reuse contract.
        sizes = dict(manifest.file_sizes)
        if not sizes and manifest.checksums:
            raise ValueError("output_size_metadata_missing")
        for relative, expected_size in sizes.items():
            output = path.parent / relative
            if not output.is_file():
                raise ValueError(f"output_missing:{relative}")
            if output.stat().st_size != int(expected_size):
                raise ValueError(f"output_size_mismatch:{relative}")
        if expected_parents is not None:
            mismatches = {
                key: (expected, manifest.parent_fingerprints.get(key))
                for key, expected in expected_parents.items()
                if manifest.parent_fingerprints.get(key) != expected
            }
            if mismatches:
                raise ValueError(f"hierarchical parent fingerprints mismatch: {mismatches}")
        if verify == "full":
            actual = _file_checksums(path.parent)
            if actual != dict(manifest.checksums):
                raise ValueError("hierarchical artifact payload checksum mismatch")
        return manifest

    def load_checkpoint(
        self,
        layer: ArtifactLayer,
        layer_fingerprint: str,
        *,
        expected_parents: Mapping[str, str] | None = None,
        verify: Literal["fast", "full"] = "fast",
    ) -> HierarchicalArtifactManifest:
        """Load and validate a checkpoint manifest."""
        path = self.checkpoint_manifest_path(layer, layer_fingerprint)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            manifest = HierarchicalArtifactManifest.from_dict(payload)
        except FileNotFoundError:
            raise
        except ObsoleteArtifactFormatError:
            raise
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid hierarchical checkpoint: {path}") from error
        if manifest.artifact_layer != layer or manifest.fingerprint != layer_fingerprint:
            raise ValueError("hierarchical checkpoint layer or fingerprint mismatch")
        if expected_parents is not None and any(
            manifest.parent_fingerprints.get(key) != expected
            for key, expected in expected_parents.items()
        ):
            raise ValueError("hierarchical parent fingerprints mismatch")
        if manifest.status != "completed":
            return manifest
        return self.load(
            layer,
            layer_fingerprint,
            expected_parents=expected_parents,
            verify=verify,
            _namespace="checkpoints",
        )

    def require(
        self,
        layer: ArtifactLayer,
        layer_fingerprint: str,
        *,
        expected_parents: Mapping[str, str] | None = None,
        verify: Literal["fast", "full"] = "fast",
        recommended_command: str | None = None,
    ) -> HierarchicalArtifactManifest:
        """Load one layer or fail with an actionable strict-reuse error."""

        return self._require_namespace(
            layer=layer,
            layer_fingerprint=layer_fingerprint,
            expected_parents=expected_parents,
            verify=verify,
            recommended_command=recommended_command,
            namespace="artifacts",
        )

    def require_checkpoint(
        self,
        layer: ArtifactLayer,
        layer_fingerprint: str,
        *,
        expected_parents: Mapping[str, str] | None = None,
        verify: Literal["fast", "full"] = "fast",
        recommended_command: str | None = None,
    ) -> HierarchicalArtifactManifest:
        """Load one checkpoint or fail with the same strict diagnostics."""

        return self._require_namespace(
            layer=layer,
            layer_fingerprint=layer_fingerprint,
            expected_parents=expected_parents,
            verify=verify,
            recommended_command=recommended_command,
            namespace="checkpoints",
        )

    def _require_namespace(
        self,
        *,
        layer: ArtifactLayer,
        layer_fingerprint: str,
        expected_parents: Mapping[str, str] | None,
        verify: Literal["fast", "full"],
        recommended_command: str | None,
        namespace: Literal["artifacts", "checkpoints"],
    ) -> HierarchicalArtifactManifest:
        path = (
            self.manifest_path(layer, layer_fingerprint)
            if namespace == "artifacts"
            else self.checkpoint_manifest_path(layer, layer_fingerprint)
        )
        try:
            return self.load(
                layer,
                layer_fingerprint,
                expected_parents=expected_parents,
                verify=verify,
                _namespace=namespace,
            )
        except ObsoleteArtifactFormatError:
            raise
        except FileNotFoundError as error:
            raise HierarchicalArtifactUnavailableError(
                artifact_layer=layer,
                expected_fingerprint=layer_fingerprint,
                artifact_path=path,
                reason_code=(
                    "artifact_missing" if namespace == "artifacts" else "checkpoint_missing"
                ),
                recommended_command=recommended_command,
            ) from error
        except (OSError, TypeError, ValueError) as error:
            actual = None
            actual_parents: Mapping[str, object] | None = None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                actual = payload.get("fingerprint")
                raw_parents = payload.get("parent_fingerprints")
                if isinstance(raw_parents, Mapping):
                    actual_parents = dict(raw_parents)
            except (OSError, TypeError, json.JSONDecodeError):
                pass
            expected_parent_values = (
                None if expected_parents is None else dict(expected_parents)
            )
            first_incompatible_field = "manifest"
            if actual is not None and actual != layer_fingerprint:
                first_incompatible_field = "fingerprint"
            elif expected_parent_values is not None:
                for parent_name, expected_parent in expected_parent_values.items():
                    actual_parent = (
                        None
                        if actual_parents is None
                        else actual_parents.get(parent_name)
                    )
                    if actual_parent != expected_parent:
                        first_incompatible_field = (
                            f"parent_fingerprints.{parent_name}"
                        )
                        break
            raise HierarchicalArtifactUnavailableError(
                artifact_layer=layer,
                expected_fingerprint=layer_fingerprint,
                artifact_path=path,
                reason_code=(
                    "artifact_incompatible"
                    if namespace == "artifacts"
                    else "checkpoint_incompatible"
                ),
                details={
                    "actual_fingerprint": actual,
                    "expected_parent_fingerprints": expected_parent_values,
                    "actual_parent_fingerprints": actual_parents,
                    "first_incompatible_field": first_incompatible_field,
                    "validation_error": str(error),
                },
                recommended_command=recommended_command,
            ) from error

    def write(
        self,
        manifest: HierarchicalArtifactManifest,
        *,
        payload_writer: Callable[[Path], Mapping[str, object] | None] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Atomically publish a completed layer and its checksums."""

        return self._write_to_namespace(
            manifest,
            namespace="artifacts",
            payload_writer=payload_writer,
            overwrite=overwrite,
        )

    def write_checkpoint(
        self,
        manifest: HierarchicalArtifactManifest,
        *,
        payload_writer: Callable[[Path], Mapping[str, object] | None] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Atomically publish a resumable checkpoint namespace."""

        return self._write_to_namespace(
            manifest,
            namespace="checkpoints",
            payload_writer=payload_writer,
            overwrite=overwrite,
        )

    def _write_to_namespace(
        self,
        manifest: HierarchicalArtifactManifest,
        *,
        namespace: Literal["artifacts", "checkpoints"],
        payload_writer: Callable[[Path], Mapping[str, object] | None] | None,
        overwrite: bool,
    ) -> Path:
        destination = self.root / namespace / manifest.artifact_layer / manifest.fingerprint
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            existing_path = destination / "manifest.json"
            existing = HierarchicalArtifactManifest.from_dict(
                json.loads(existing_path.read_text(encoding="utf-8"))
            )
            if existing.to_dict() != manifest.to_dict():
                raise ValueError(f"artifact already exists with different metadata: {destination}")
            return destination
        staging = Path(tempfile.mkdtemp(prefix=f".{manifest.fingerprint}.", dir=destination.parent))
        try:
            payload = dict(manifest.payload)
            if payload_writer is not None:
                payload.update(dict(payload_writer(staging) or {}))
            checksums = _file_checksums(staging)
            file_sizes = _file_sizes(staging)
            published_status = (
                "completed" if namespace == "artifacts" else manifest.status
            )
            completed = HierarchicalArtifactManifest(
                artifact_layer=manifest.artifact_layer,
                artifact_schema_version=manifest.artifact_schema_version,
                fingerprint=manifest.fingerprint,
                parent_fingerprints=manifest.parent_fingerprints,
                scientific_parameters=manifest.scientific_parameters,
                input_fingerprints=manifest.input_fingerprints,
                execution_parameters=manifest.execution_parameters,
                status=published_status,
                checksums=checksums,
                payload=payload,
                file_sizes=file_sizes,
                package_revision=manifest.package_revision,
                identity_basis_fingerprint=manifest.identity_basis_fingerprint,
                configuration_fingerprint=manifest.configuration_fingerprint,
                parent_identity_fingerprint=manifest.parent_identity_fingerprint,
            )
            manifest_path = staging / "manifest.json"
            manifest_path.write_text(
                json.dumps(completed.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if published_status == "completed":
                _atomic_json(
                    staging / "COMPLETE.json",
                    {
                        "identity_fingerprint": completed.fingerprint,
                        "identity_basis_fingerprint": completed.identity_basis_fingerprint,
                        "manifest_sha256": file_sha256(manifest_path),
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            if destination.exists():
                if not overwrite:
                    raise FileExistsError(destination)
                shutil.rmtree(destination)
            os.replace(staging, destination)
            return destination
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


@dataclass(frozen=True, slots=True)
class HierarchicalProgressEvent:
    """Durable per-layer progress event."""

    artifact_layer: ArtifactLayer
    artifact_fingerprint: str
    parent_fingerprint: str | None
    activation_policy: str
    phase: str
    status: str
    current_unit: str | None
    completed_units: int
    total_units: int | None
    elapsed_seconds: float
    estimated_remaining_seconds: float | None
    estimated_completion_at_utc: str | None
    eta_confidence: str
    reused: bool
    rebuild: bool
    action: str = "build"
    schema_version: int = HIERARCHICAL_PROGRESS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"unsupported progress status: {self.status!r}")
        if self.completed_units < 0:
            raise ValueError("completed_units must be nonnegative")
        if self.total_units is not None and self.total_units < self.completed_units:
            raise ValueError("completed_units cannot exceed total_units")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_layer": self.artifact_layer,
            "artifact_fingerprint": self.artifact_fingerprint,
            "parent_fingerprint": self.parent_fingerprint,
            "activation_policy": self.activation_policy,
            "checkpoint_policy": self.activation_policy,
            "phase": self.phase,
            "status": self.status,
            "current_unit": self.current_unit,
            "completed_units": self.completed_units,
            "total_units": self.total_units,
            "elapsed_seconds": self.elapsed_seconds,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "estimated_completion_at_utc": self.estimated_completion_at_utc,
            "eta_confidence": self.eta_confidence,
            "reused": self.reused,
            "rebuild": self.rebuild,
            "stage": "prepare",
            "layer": self.artifact_layer,
            "action": self.action,
        }


class HierarchicalProgressReporter:
    """Append per-layer and campaign-level JSONL progress events."""

    def __init__(
        self,
        path: str | Path,
        *,
        campaign_path: str | Path | None = None,
        interval_seconds: float = 5.0,
    ) -> None:
        if interval_seconds < 0.0:
            raise ValueError("interval_seconds must be nonnegative")
        self.path = Path(path)
        self.campaign_path = None if campaign_path is None else Path(campaign_path)
        self.interval_seconds = float(interval_seconds)
        self._started: dict[str, float] = {}
        self._last_emit: dict[str, float] = {}
        self._durations: dict[str, list[float]] = {}

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def emit(
        self,
        *,
        artifact_layer: ArtifactLayer,
        artifact_fingerprint: str,
        parent_fingerprint: str | None,
        activation_policy: str,
        phase: str,
        status: str,
        current_unit: str | None = None,
        completed_units: int = 0,
        total_units: int | None = None,
        elapsed_seconds: float | None = None,
        estimated_remaining_seconds: float | None = None,
        eta_confidence: str = "unavailable",
        reused: bool = False,
        rebuild: bool = False,
        action: str | None = None,
        force: bool = False,
    ) -> HierarchicalProgressEvent | None:
        now = monotonic()
        started = self._started.setdefault(artifact_layer, now)
        if not force and status in {"running", "heartbeat"}:
            last = self._last_emit.get(artifact_layer, float("-inf"))
            if now - last < self.interval_seconds:
                return None
        elapsed = max(0.0, now - started) if elapsed_seconds is None else max(0.0, elapsed_seconds)
        if estimated_remaining_seconds is not None:
            estimated_remaining_seconds = max(0.0, float(estimated_remaining_seconds))
            eta_at = datetime.now(timezone.utc).timestamp() + estimated_remaining_seconds
            estimated_completion = datetime.fromtimestamp(eta_at, timezone.utc).isoformat()
        else:
            estimated_completion = None
        event = HierarchicalProgressEvent(
            artifact_layer=artifact_layer,
            artifact_fingerprint=artifact_fingerprint,
            parent_fingerprint=parent_fingerprint,
            activation_policy=activation_policy,
            phase=phase,
            status=status,
            current_unit=current_unit,
            completed_units=int(completed_units),
            total_units=None if total_units is None else int(total_units),
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=estimated_remaining_seconds,
            estimated_completion_at_utc=estimated_completion,
            eta_confidence=eta_confidence,
            reused=bool(reused),
            rebuild=bool(rebuild),
            action=(
                action
                or ("reuse" if reused else "rebuild" if rebuild else "build")
            ),
        )
        encoded = json.dumps(event.to_dict(), sort_keys=True) + "\n"
        for target in (self.path, self.campaign_path):
            if target is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
        self._last_emit[artifact_layer] = now
        return event


def _immutable_array(value: object, *, dtype: np.dtype | None = None, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class EstimationAssignmentMapping:
    """Final L7 linear assignment mapping consumed by estimation."""

    A_free: object
    fixed_offset: np.ndarray
    free_column_index: np.ndarray
    measurement_row_index: np.ndarray
    full_to_compact: np.ndarray
    compact_to_full: np.ndarray
    full_od_fingerprint: str
    active_od_fingerprint: str
    parent_fingerprints: Mapping[str, str]
    fingerprint: str
    fixed_positive_full_index: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    fixed_positive_values: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    ancestor_fingerprints: Mapping[str, str] = field(default_factory=dict)
    fixed_offset_fingerprint: str | None = None
    companion_fingerprints: Mapping[str, str] = field(default_factory=dict)
    artifact_schema_version: int = HIERARCHICAL_ARTIFACT_SCHEMA_VERSION
    _jax_matrix: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        matrix = self.A_free
        if sparse.issparse(matrix):
            if len(matrix.shape) != 2:
                raise ValueError("A_free must be two-dimensional")
            if np.any(~np.isfinite(matrix.data)):
                raise ValueError("A_free must be finite")
        elif isinstance(matrix, (np.ndarray, np.matrix)):
            matrix = np.asarray(matrix)
            if matrix.ndim != 2:
                raise ValueError("A_free must be two-dimensional")
            if not np.all(np.isfinite(matrix)):
                raise ValueError("A_free must be finite")
            object.__setattr__(self, "A_free", np.array(matrix, copy=True))
        else:
            shape = getattr(matrix, "shape", None)
            if shape is None:
                shape = (
                    getattr(matrix, "num_measurements", None),
                    getattr(matrix, "num_free_od", None),
                )
            if (
                not isinstance(shape, tuple)
                or len(shape) != 2
                or any(value is None for value in shape)
            ):
                raise ValueError("A_free must expose a two-dimensional shape")
        rows, columns = map(int, matrix.shape)
        offset = _immutable_array(self.fixed_offset, dtype=np.dtype(matrix.dtype), name="fixed_offset")
        if offset.shape != (rows,):
            raise ValueError("fixed_offset shape does not match A_free")
        free = _immutable_array(self.free_column_index, dtype=np.int64, name="free_column_index")
        measurement = _immutable_array(self.measurement_row_index, dtype=np.int64, name="measurement_row_index")
        full_to_compact = _immutable_array(self.full_to_compact, dtype=np.int64, name="full_to_compact")
        compact_to_full = _immutable_array(self.compact_to_full, dtype=np.int64, name="compact_to_full")
        fixed_positive_index = _immutable_array(
            self.fixed_positive_full_index, dtype=np.int64, name="fixed_positive_full_index"
        )
        fixed_positive_values = _immutable_array(
            self.fixed_positive_values, dtype=np.dtype(matrix.dtype), name="fixed_positive_values"
        )
        if fixed_positive_index.shape != fixed_positive_values.shape:
            raise ValueError(
                "fixed_positive_full_index and fixed_positive_values must have the same shape"
            )
        if np.any(fixed_positive_index < 0) or np.any(fixed_positive_values <= 0):
            raise ValueError("fixed-positive cells must have positive indices and values")
        if free.shape != (columns,) or measurement.shape != (rows,):
            raise ValueError("mapping coordinate arrays do not match A_free")
        if compact_to_full.shape != (columns,) or np.any(compact_to_full < 0):
            raise ValueError("compact_to_full is invalid")
        if full_to_compact.size and np.any(full_to_compact < -1):
            raise ValueError("full_to_compact is invalid")
        object.__setattr__(self, "fixed_offset", offset)
        object.__setattr__(self, "free_column_index", free)
        object.__setattr__(self, "measurement_row_index", measurement)
        object.__setattr__(self, "full_to_compact", full_to_compact)
        object.__setattr__(self, "compact_to_full", compact_to_full)
        object.__setattr__(self, "fixed_positive_full_index", fixed_positive_index)
        object.__setattr__(self, "fixed_positive_values", fixed_positive_values)
        object.__setattr__(self, "fixed_offset_fingerprint", self.fixed_offset_fingerprint or fingerprint(offset.tolist()))
        if not self.fingerprint:
            raise ValueError("mapping fingerprint must be nonempty")

    @property
    def fixed_measurement_offset(self) -> np.ndarray:
        return self.fixed_offset

    @property
    def compact_layout_fingerprint(self) -> str | None:
        return self.active_od_fingerprint

    @property
    def artifact_layer(self) -> str:
        return "estimation_assignment_mapping"

    @property
    def representation(self) -> str:
        return "hierarchical_estimation_assignment_mapping"

    @property
    def is_matrix_free(self) -> bool:
        return not (sparse.issparse(self.A_free) or isinstance(self.A_free, np.ndarray))

    @property
    def assignment_fingerprint(self) -> str:
        return self.parent_fingerprints.get(
            "full_od_assignment_mapping",
            self.ancestor_fingerprints.get("full_od_assignment_mapping", ""),
        )

    @property
    def graph_fingerprint(self) -> str:
        return self.parent_fingerprints.get(
            "scenario_base", self.ancestor_fingerprints.get("scenario_base", "")
        )

    @property
    def mapping_fingerprint(self) -> str:
        return self.parent_fingerprints.get(
            "observation_projection",
            self.ancestor_fingerprints.get("observation_projection", ""),
        )

    @property
    def theta(self) -> float:
        # L7 is a numerical mapping and does not need theta to execute.  The
        # route-choice theta, when relevant, is represented by an L4 parent.
        return 1.0

    @property
    def metrics(self) -> object:
        class _Metrics:
            stored_bytes = int(
                self.fixed_offset.nbytes
                + self.free_column_index.nbytes
                + self.measurement_row_index.nbytes
                + self.full_to_compact.nbytes
                + self.compact_to_full.nbytes
                + self.fixed_positive_full_index.nbytes
                + self.fixed_positive_values.nbytes
                + (
                    self.A_free.data.nbytes
                    + self.A_free.indices.nbytes
                    + self.A_free.indptr.nbytes
                    if sparse.issparse(self.A_free)
                    else getattr(self.A_free, "nbytes", 0)
                )
            )
            peak_construction_bytes = 0

        return _Metrics()

    @property
    def product_capabilities(self) -> GravityOperatorCapabilities:
        return GravityOperatorCapabilities(matmat=True)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.A_free.shape)

    @property
    def num_measurements(self) -> int:
        return self.shape[0]

    @property
    def num_free_od(self) -> int:
        return self.shape[1]

    @property
    def number_of_measurements(self) -> int:
        return self.num_measurements

    @property
    def number_of_demand_cells(self) -> int:
        return self.num_free_od

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.A_free.dtype)

    @property
    def artifact_fingerprint(self) -> str:
        return self.fingerprint

    def matvec(self, demand: object) -> np.ndarray:
        value = np.asarray(demand)
        if value.shape != (self.num_free_od,):
            raise ValueError(f"demand must have shape ({self.num_free_od},)")
        if hasattr(self.A_free, "matvec"):
            return np.asarray(self.A_free.matvec(value))
        return np.asarray(self.A_free @ value)

    def rmatvec(self, residual: object) -> np.ndarray:
        value = np.asarray(residual)
        if value.shape != (self.num_measurements,):
            raise ValueError(f"residual must have shape ({self.num_measurements},)")
        if hasattr(self.A_free, "rmatvec"):
            return np.asarray(self.A_free.rmatvec(value))
        return np.asarray(self.A_free.T @ value)

    def matmat(self, demand: object) -> np.ndarray:
        value = np.asarray(demand)
        if value.ndim != 2 or value.shape[0] != self.num_free_od:
            raise ValueError("matmat input has an incompatible shape")
        if hasattr(self.A_free, "matmat"):
            return np.asarray(self.A_free.matmat(value))
        return np.asarray(self.A_free @ value)

    def _jax_operator(self) -> object:
        cached = self._jax_matrix
        if cached is not None:
            return cached
        if hasattr(self.A_free, "jax_matvec"):
            return self.A_free
        if sparse.issparse(self.A_free):
            try:
                from jax.experimental import sparse as jsparse

                cached = jsparse.BCOO.from_scipy_sparse(self.A_free.tocoo())
            except (AttributeError, ImportError, TypeError, ValueError):
                cached = jnp.asarray(self.A_free.toarray())
        else:
            cached = jnp.asarray(self.A_free)
        object.__setattr__(self, "_jax_matrix", cached)
        return cached

    def jax_matvec(self, demand: jax.Array) -> jax.Array:
        operator = self._jax_operator()
        if hasattr(operator, "jax_matvec"):
            return operator.jax_matvec(demand)
        return operator @ demand

    def jax_rmatvec(self, residual: jax.Array) -> jax.Array:
        operator = self._jax_operator()
        if hasattr(operator, "jax_rmatvec"):
            return operator.jax_rmatvec(residual)
        return operator.T @ residual

    def jax_matmat(self, demand: jax.Array) -> jax.Array:
        operator = self._jax_operator()
        if hasattr(operator, "jax_matmat"):
            return operator.jax_matmat(demand)
        return operator @ demand

    def to_payload(self, directory: str | Path) -> Mapping[str, object]:
        """Persist the numerical L7 payload into ``directory``."""

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        if hasattr(self.A_free, "to_payload"):
            self.A_free.to_payload(target)
            matrix_format = "companion_operator"
        elif sparse.issparse(self.A_free):
            sparse.save_npz(target / "A_free.npz", self.A_free.tocsr())
            matrix_format = "scipy_csr"
        else:
            np.save(target / "A_free.npy", np.asarray(self.A_free))
            matrix_format = "numpy_dense"
        for name, value in (
            ("fixed_offset", self.fixed_offset),
            ("free_column_index", self.free_column_index),
            ("measurement_row_index", self.measurement_row_index),
            ("full_to_compact", self.full_to_compact),
            ("compact_to_full", self.compact_to_full),
            ("fixed_positive_full_index", self.fixed_positive_full_index),
            ("fixed_positive_values", self.fixed_positive_values),
        ):
            np.save(target / f"{name}.npy", value)
        metadata = {
            "matrix_format": matrix_format,
            "full_od_fingerprint": self.full_od_fingerprint,
            "active_od_fingerprint": self.active_od_fingerprint,
            "parent_fingerprints": dict(self.parent_fingerprints),
            "ancestor_fingerprints": dict(self.ancestor_fingerprints),
            "fingerprint": self.fingerprint,
            "fixed_offset_fingerprint": self.fixed_offset_fingerprint,
            "companion_fingerprints": dict(self.companion_fingerprints),
        }
        (target / "mapping.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return metadata

    @classmethod
    def from_payload(
        cls,
        directory: str | Path,
        *,
        companion_loader: Callable[[Path, Mapping[str, object]], object] | None = None,
    ) -> "EstimationAssignmentMapping":
        source = Path(directory)
        metadata = json.loads((source / "mapping.json").read_text(encoding="utf-8"))
        matrix_format = metadata.get("matrix_format")
        if matrix_format == "scipy_csr":
            matrix = sparse.load_npz(source / "A_free.npz").tocsr()
        elif matrix_format == "numpy_dense":
            matrix = np.load(source / "A_free.npy", allow_pickle=False)
        elif matrix_format == "companion_operator":
            if companion_loader is None:
                raise ValueError(
                    "companion_operator payload requires an explicit companion_loader"
                )
            matrix = companion_loader(source, metadata)
        else:
            raise ValueError("unsupported estimation mapping matrix format")
        fixed_offset = np.load(source / "fixed_offset.npy", allow_pickle=False)
        stored_offset_fingerprint = metadata.get("fixed_offset_fingerprint")
        if stored_offset_fingerprint is not None and str(stored_offset_fingerprint) != fingerprint(
            np.asarray(fixed_offset).tolist()
        ):
            raise ValueError("fixed offset fingerprint is invalid")
        result = cls(
            A_free=matrix,
            fixed_offset=fixed_offset,
            free_column_index=np.load(source / "free_column_index.npy", allow_pickle=False),
            measurement_row_index=np.load(source / "measurement_row_index.npy", allow_pickle=False),
            full_to_compact=np.load(source / "full_to_compact.npy", allow_pickle=False),
            compact_to_full=np.load(source / "compact_to_full.npy", allow_pickle=False),
            fixed_positive_full_index=np.load(
                source / "fixed_positive_full_index.npy", allow_pickle=False
            )
            if (source / "fixed_positive_full_index.npy").is_file()
            else np.empty(0, dtype=np.int64),
            fixed_positive_values=np.load(
                source / "fixed_positive_values.npy", allow_pickle=False
            )
            if (source / "fixed_positive_values.npy").is_file()
            else np.empty(0, dtype=np.float64),
            full_od_fingerprint=str(metadata["full_od_fingerprint"]),
            active_od_fingerprint=str(metadata["active_od_fingerprint"]),
            parent_fingerprints=dict(metadata["parent_fingerprints"]),
            fingerprint=str(metadata["fingerprint"]),
            fixed_offset_fingerprint=str(metadata.get("fixed_offset_fingerprint", "")),
            companion_fingerprints=dict(metadata.get("companion_fingerprints", {})),
            ancestor_fingerprints=dict(metadata.get("ancestor_fingerprints", {})),
        )
        if result.fingerprint != str(metadata.get("fingerprint", "")):
            raise ValueError("estimation mapping fingerprint is missing or invalid")
        return result


@dataclass(frozen=True, slots=True)
class HierarchicalPreparationResult:
    """Result of preparing or reusing the complete hierarchy."""

    mapping: EstimationAssignmentMapping
    manifests: Mapping[str, HierarchicalArtifactManifest]
    reused_layers: tuple[str, ...]
    rebuilt_layers: tuple[str, ...]
    actions: Mapping[str, str] = field(default_factory=dict)
    verification_mode: str = "fast"
    checkpoint_reused: bool = False
    rebuild_performed: bool = False


def _write_preparation_manifest(
    root: str | Path,
    *,
    policy: CheckpointPolicy,
    verify: Literal["fast", "full"],
    checkpoint_reused: bool,
    rebuilt: tuple[str, ...] | list[str],
    actions: Mapping[str, str],
    fingerprints: Mapping[str, str],
) -> None:
    """Write the small campaign summary after a successful preparation."""

    rebuilt_tuple = tuple(rebuilt)
    _atomic_json(
        Path(root) / "prepare-manifest.json",
        {
            "status": "completed",
            "checkpoint_policy": policy,
            "verification_mode": verify,
            "checkpoint_reused": checkpoint_reused,
            "rebuild_performed": bool(rebuilt_tuple),
            "layers": {
                layer: {
                    "action": actions.get(layer, "reuse"),
                    "identity_fingerprint": fingerprints[layer],
                    "parent_identity_fingerprint": (
                        None
                        if not DAG_PARENT_LAYERS[layer]
                        else fingerprints[DAG_PARENT_LAYERS[layer][0]]
                    ),
                }
                for layer in ARTIFACT_LAYERS
            },
        },
    )


def prepare_hierarchical_assignment_mapping(
    *,
    root: str | Path,
    specifications: Mapping[str, Mapping[str, object]],
    mapping_builder: Callable[[], EstimationAssignmentMapping] | None = None,
    payload_builders: Mapping[str, Callable[[Path], Mapping[str, object] | None]] | None = None,
    checkpoint_policy: CheckpointPolicy = "reuse_or_build",
    activation_policy: Literal["reuse_only", "build_or_reuse", "force_rebuild"] | None = None,
    verify: Literal["fast", "full"] = "fast",
    execution_parameters: Mapping[str, object] | None = None,
    expected_fixed_offset_fingerprint: str | None = None,
    progress: HierarchicalProgressReporter | None = None,
    recommended_command: str | None = None,
) -> HierarchicalPreparationResult:
    """Prepare the hierarchy with strict, parent-aware reuse semantics.

    ``specifications`` contains scientific ``scientific_parameters`` and
    ``input_fingerprints`` for each layer.  Execution controls belong in
    ``execution_parameters`` and never change layer fingerprints.
    """

    policy = normalize_checkpoint_policy(
        None if activation_policy is not None else checkpoint_policy,
        legacy_policy=activation_policy,
    )
    if verify not in {"fast", "full"}:
        raise ValueError("verify must be 'fast' or 'full'")
    layer_fingerprints = derive_layer_fingerprints(specifications)
    store = HierarchicalArtifactStore(root)
    manifests: dict[str, HierarchicalArtifactManifest] = {}
    reused: list[str] = []
    rebuilt: list[str] = []
    actions: dict[str, str] = {}
    checkpoint_reused = False
    builders = dict(payload_builders or {})
    for position, layer in enumerate(ARTIFACT_LAYERS[:-1]):
        layer_checkpoint_reused = False
        expected = layer_fingerprints[layer]
        parents = {
            name: layer_fingerprints[name] for name in DAG_PARENT_LAYERS[layer]
        }
        parent = (
            None
            if not DAG_PARENT_LAYERS[layer]
            else layer_fingerprints[DAG_PARENT_LAYERS[layer][0]]
        )
        if policy != "rebuild":
            try:
                manifest = store.load(
                    layer, expected, expected_parents=parents, verify=verify
                )
            except FileNotFoundError:
                manifest = None
            except ObsoleteArtifactFormatError:
                raise
            except (OSError, TypeError, ValueError) as error:
                raise HierarchicalArtifactUnavailableError(
                    artifact_layer=layer,
                    expected_fingerprint=expected,
                    artifact_path=store.artifact_path(layer, expected),
                    reason_code=_reason_code_from_error(
                        str(error), "artifact_incompatible"
                    ),
                    details={"validation_error": str(error)},
                    recommended_command=recommended_command,
                ) from error
            if manifest is not None:
                manifests[layer] = manifest
                reused.append(layer)
                checkpoint_reused = True
                actions[layer] = "reuse"
                if progress is not None:
                    progress.emit(
                        artifact_layer=layer,
                        artifact_fingerprint=expected,
                        parent_fingerprint=parent,
                        activation_policy=policy,
                        phase="activation",
                        status="reused",
                        completed_units=1,
                        total_units=1,
                        estimated_remaining_seconds=0.0,
                        eta_confidence="high",
                        reused=True,
                        rebuild=False,
                        action="reuse",
                        force=True,
                    )
                continue
        if policy == "reuse_only":
            raise HierarchicalArtifactUnavailableError(
                artifact_layer=layer,
                expected_fingerprint=expected,
                artifact_path=store.artifact_path(layer, expected),
                reason_code="artifact_missing",
                recommended_command=recommended_command,
            )
        if policy != "rebuild":
            try:
                store.load_checkpoint(
                    layer, expected, expected_parents=parents, verify="fast"
                )
            except FileNotFoundError:
                pass
            except (ObsoleteArtifactFormatError, OSError, TypeError, ValueError) as error:
                raise HierarchicalArtifactUnavailableError(
                    artifact_layer=layer,
                    expected_fingerprint=expected,
                    artifact_path=store.checkpoint_path(layer, expected),
                    reason_code=_reason_code_from_error(
                        str(error), "checkpoint_incompatible"
                    ),
                    details={"validation_error": str(error)},
                    recommended_command=recommended_command,
                ) from error
            else:
                layer_checkpoint_reused = True
                checkpoint_reused = True
        manifest = make_manifest(
            artifact_layer=layer,
            parent_fingerprints=parents,
            scientific_parameters=dict(specifications.get(layer, {}).get("scientific_parameters", {})),
            input_fingerprints=dict(specifications.get(layer, {}).get("input_fingerprints", {})),
            execution_parameters=execution_parameters,
            payload=dict(specifications.get(layer, {}).get("payload", {})),
            status="started",
        )
        if progress is not None:
            progress.emit(
                artifact_layer=layer,
                artifact_fingerprint=expected,
                parent_fingerprint=parent,
                activation_policy=policy,
                phase="construction",
                status="started",
                completed_units=0,
                total_units=1,
                eta_confidence="unavailable",
                reused=False,
                rebuild=True,
                action="rebuild" if policy == "rebuild" else (
                    "resume" if layer_checkpoint_reused else "build"
                ),
                force=True,
            )
        # Checkpoint metadata is published separately from the completed
        # artifact payload.  A checkpoint can therefore be inspected/resumed
        # without making a partial artifact visible to consumers.
        try:
            store.write_checkpoint(manifest, overwrite=True)
            store.write(manifest, payload_writer=builders.get(layer), overwrite=True)
        except BaseException:
            if progress is not None:
                progress.emit(
                    artifact_layer=layer,
                    artifact_fingerprint=expected,
                    parent_fingerprint=parent,
                    activation_policy=policy,
                    phase="construction",
                    status="failed",
                    completed_units=0,
                    total_units=1,
                    eta_confidence="unavailable",
                    reused=False,
                    rebuild=True,
                    action="rebuild" if policy == "rebuild" else (
                        "resume" if layer_checkpoint_reused else "build"
                    ),
                    force=True,
                )
            raise
        manifests[layer] = store.load(layer, expected, verify=verify)
        rebuilt.append(layer)
        actions[layer] = "rebuild" if policy == "rebuild" else (
            "resume" if layer_checkpoint_reused else "build"
        )
        if progress is not None:
            progress.emit(
                artifact_layer=layer,
                artifact_fingerprint=expected,
                parent_fingerprint=parent,
                activation_policy=policy,
                phase="construction",
                status="completed",
                completed_units=1,
                total_units=1,
                estimated_remaining_seconds=0.0,
                eta_confidence="high",
                reused=False,
                rebuild=True,
                action=actions[layer],
                force=True,
            )

    final_layer: ArtifactLayer = "estimation_assignment_mapping"
    final_expected = layer_fingerprints[final_layer]
    final_parents = {
        name: layer_fingerprints[name]
        for name in DAG_PARENT_LAYERS[final_layer]
    }
    final_parent = layer_fingerprints[DAG_PARENT_LAYERS[final_layer][0]]
    final_manifest: HierarchicalArtifactManifest | None = None
    final_checkpoint_reused = False
    if policy != "rebuild":
        try:
            final_manifest = store.load(
                final_layer,
                final_expected,
                expected_parents=final_parents,
                verify=verify,
            )
        except FileNotFoundError:
            final_manifest = None
        except ObsoleteArtifactFormatError:
            raise
        except (OSError, TypeError, ValueError) as error:
            raise HierarchicalArtifactUnavailableError(
                artifact_layer=final_layer,
                expected_fingerprint=final_expected,
                artifact_path=store.artifact_path(final_layer, final_expected),
                reason_code=_reason_code_from_error(
                    str(error), "artifact_incompatible"
                ),
                details={"validation_error": str(error)},
                recommended_command=recommended_command,
            ) from error
    if final_manifest is not None:
        payload_directory = store.artifact_path(final_layer, final_expected)
        try:
            mapping = EstimationAssignmentMapping.from_payload(payload_directory)
        except (OSError, TypeError, ValueError, KeyError) as error:
            if policy == "reuse_only":
                raise HierarchicalArtifactUnavailableError(
                    artifact_layer=final_layer,
                    expected_fingerprint=final_expected,
                    artifact_path=payload_directory,
                    reason_code="payload_incompatible",
                    details={"validation_error": str(error)},
                    recommended_command=recommended_command,
                ) from error
            final_manifest = None
        else:
            if (
                expected_fixed_offset_fingerprint is not None
                and mapping.fixed_offset_fingerprint
                != expected_fixed_offset_fingerprint
            ):
                final_manifest = None
            else:
                manifests[final_layer] = final_manifest
                reused.append(final_layer)
                checkpoint_reused = True
                actions[final_layer] = "reuse"
                if progress is not None:
                    progress.emit(
                        artifact_layer=final_layer,
                        artifact_fingerprint=final_expected,
                        parent_fingerprint=final_parent,
                        activation_policy=policy,
                        phase="activation",
                        status="reused",
                        completed_units=1,
                        total_units=1,
                        estimated_remaining_seconds=0.0,
                        eta_confidence="high",
                        reused=True,
                        rebuild=False,
                        action="reuse",
                        force=True,
                    )
                _write_preparation_manifest(
                    root,
                    policy=policy,
                    verify=verify,
                    checkpoint_reused=checkpoint_reused,
                    rebuilt=rebuilt,
                    actions=actions,
                    fingerprints=layer_fingerprints,
                )
                return HierarchicalPreparationResult(
                    mapping,
                    manifests,
                    tuple(reused),
                    tuple(rebuilt),
                    actions,
                    verify,
                    checkpoint_reused,
                    bool(rebuilt),
                )
    if policy == "reuse_only":
        raise HierarchicalArtifactUnavailableError(
            artifact_layer=final_layer,
            expected_fingerprint=final_expected,
            artifact_path=store.artifact_path(final_layer, final_expected),
            reason_code="artifact_missing",
            recommended_command=recommended_command,
        )
    if policy != "rebuild":
        try:
            store.load_checkpoint(
                final_layer, final_expected, expected_parents=final_parents, verify="fast"
            )
        except FileNotFoundError:
            pass
        except (ObsoleteArtifactFormatError, OSError, TypeError, ValueError) as error:
            raise HierarchicalArtifactUnavailableError(
                artifact_layer=final_layer,
                expected_fingerprint=final_expected,
                artifact_path=store.checkpoint_path(final_layer, final_expected),
                reason_code=_reason_code_from_error(
                    str(error), "checkpoint_incompatible"
                ),
                details={"validation_error": str(error)},
                recommended_command=recommended_command,
            ) from error
        else:
            final_checkpoint_reused = True
            checkpoint_reused = True
    if mapping_builder is None:
        raise ValueError("mapping_builder is required when L7 must be constructed")
    if progress is not None:
        progress.emit(
            artifact_layer=final_layer,
            artifact_fingerprint=final_expected,
            parent_fingerprint=final_parent,
            activation_policy=policy,
            phase="construction",
            status="started",
            completed_units=0,
            total_units=1,
            eta_confidence="unavailable",
            reused=False,
            rebuild=True,
            action="rebuild" if policy == "rebuild" else (
                "resume" if checkpoint_reused else "build"
            ),
            force=True,
        )
    try:
        mapping = mapping_builder()
    except BaseException:
        if progress is not None:
            progress.emit(
                artifact_layer=final_layer,
                artifact_fingerprint=final_expected,
                parent_fingerprint=final_parent,
                activation_policy=policy,
                phase="construction",
                status="failed",
                completed_units=0,
                total_units=1,
                eta_confidence="unavailable",
                reused=False,
                rebuild=True,
                action="rebuild" if policy == "rebuild" else (
                    "resume" if checkpoint_reused else "build"
                ),
                force=True,
            )
        raise
    if mapping.fingerprint != final_expected:
        raise ValueError(
            "mapping_builder returned a fingerprint different from the expected L7 identity"
        )
    if (
        expected_fixed_offset_fingerprint is not None
        and mapping.fixed_offset_fingerprint != expected_fixed_offset_fingerprint
    ):
        raise ValueError(
            "mapping_builder returned an unexpected fixed-offset fingerprint"
        )

    def write_mapping(directory: Path) -> Mapping[str, object]:
        return mapping.to_payload(directory)

    final_manifest = make_manifest(
        artifact_layer=final_layer,
        parent_fingerprints=final_parents,
        scientific_parameters=dict(specifications.get(final_layer, {}).get("scientific_parameters", {})),
        input_fingerprints=dict(specifications.get(final_layer, {}).get("input_fingerprints", {})),
        execution_parameters=execution_parameters,
        payload=dict(specifications.get(final_layer, {}).get("payload", {})),
        status="started",
    )
    try:
        store.write_checkpoint(final_manifest, overwrite=True)
        store.write(final_manifest, payload_writer=write_mapping, overwrite=True)
    except BaseException:
        if progress is not None:
            progress.emit(
                artifact_layer=final_layer,
                artifact_fingerprint=final_expected,
                parent_fingerprint=final_parent,
                activation_policy=policy,
                phase="construction",
                status="failed",
                completed_units=0,
                total_units=1,
                eta_confidence="unavailable",
                reused=False,
                rebuild=True,
                force=True,
            )
        raise
    manifests[final_layer] = store.load(final_layer, final_expected, verify=verify)
    rebuilt.append(final_layer)
    actions[final_layer] = "rebuild" if policy == "rebuild" else (
        "resume" if final_checkpoint_reused else "build"
    )
    if progress is not None:
        progress.emit(
            artifact_layer=final_layer,
            artifact_fingerprint=final_expected,
            parent_fingerprint=final_parent,
            activation_policy=policy,
            phase="construction",
            status="completed",
            completed_units=1,
            total_units=1,
            estimated_remaining_seconds=0.0,
            eta_confidence="high",
            reused=False,
            rebuild=True,
            action=actions[final_layer],
            force=True,
        )
    result = HierarchicalPreparationResult(
        mapping,
        manifests,
        tuple(reused),
        tuple(rebuilt),
        actions,
        verify,
        checkpoint_reused,
        bool(rebuilt),
    )
    _write_preparation_manifest(
        root,
        policy=policy,
        verify=verify,
        checkpoint_reused=checkpoint_reused,
        rebuilt=rebuilt,
        actions=actions,
        fingerprints=layer_fingerprints,
    )
    return result


__all__ = [
    "ARTIFACT_LAYERS",
    "ArtifactLayer",
    "DAG_PARENT_LAYERS",
    "CheckpointPolicy",
    "EstimationAssignmentMapping",
    "HIERARCHICAL_ARTIFACT_SCHEMA_VERSION",
    "HIERARCHICAL_PROGRESS_SCHEMA_VERSION",
    "HierarchicalArtifactManifest",
    "HierarchicalArtifactStore",
    "HierarchicalArtifactUnavailableError",
    "HierarchicalPreparationResult",
    "HierarchicalProgressEvent",
    "HierarchicalProgressReporter",
    "ObsoleteArtifactFormatError",
    "canonical_json",
    "derive_layer_fingerprints",
    "file_sha256",
    "fingerprint",
    "make_manifest",
    "package_revision",
    "prepare_hierarchical_assignment_mapping",
]
