"""Independent, fail-closed storage for reduced-OD phase artifacts."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import tempfile
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from public_transportation.version import __version__

from .artifacts import canonical_json
from .progress import ReducedODProgress, ReducedODProgressEmitter


REDUCED_OD_STORE_SCHEMA_VERSION = 1


class ReducedODArtifactStoreError(ValueError):
    """A persisted phase is missing, corrupt, or identity-incompatible."""


def _class_name(value: type[Any]) -> str:
    return f"{value.__module__}:{value.__qualname__}"


def _resolve_class(name: str) -> type[Any]:
    module_name, separator, qualname = name.partition(":")
    if separator != ":" or not module_name.startswith("public_transportation."):
        raise ReducedODArtifactStoreError(f"unsupported persisted type {name!r}.")
    value: Any = importlib.import_module(module_name)
    try:
        for component in qualname.split("."):
            value = getattr(value, component)
    except AttributeError as error:
        raise ReducedODArtifactStoreError(
            f"persisted type {name!r} is unavailable in the installed library."
        ) from error
    if not isinstance(value, type):
        raise ReducedODArtifactStoreError(f"persisted type {name!r} is invalid.")
    return value


def _encode(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(value, np.ndarray):
        name = f"array_{len(arrays):06d}.npy"
        array = np.array(value, copy=True, order="C")
        if array.dtype.hasobject:
            raise TypeError("persisted reduced-OD arrays cannot use object dtype.")
        arrays[name] = array
        digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
        return {
            "__array__": name,
            "digest": digest,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
    if isinstance(value, Enum):
        return {"__enum__": _class_name(type(value)), "value": value.value}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": _class_name(type(value)),
            "fields": {
                item.name: _encode(getattr(value, item.name), arrays)
                for item in fields(value)
            },
        }
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(item, arrays) for item in value]}
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if isinstance(value, Mapping):
        encoded = [
            (_encode(key, arrays), _encode(item, arrays)) for key, item in value.items()
        ]
        encoded.sort(key=lambda item: canonical_json(item[0]))
        return {"__mapping__": encoded}
    if isinstance(value, list):
        return [_encode(item, arrays) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported persisted reduced-OD value {type(value)!r}.")


def _decode(value: Any, directory: Path) -> Any:
    if isinstance(value, list):
        return [_decode(item, directory) for item in value]
    if not isinstance(value, dict):
        return value
    if "__array__" in value:
        path = directory / str(value["__array__"])
        if not path.is_file():
            raise ReducedODArtifactStoreError(f"missing artifact array {path.name}.")
        array = np.load(path, allow_pickle=False)
        expected_shape = tuple(int(item) for item in value["shape"])
        if array.dtype.str != value["dtype"] or array.shape != expected_shape:
            raise ReducedODArtifactStoreError(
                f"array descriptor mismatch for {path.name}."
            )
        digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
        if digest != value["digest"]:
            raise ReducedODArtifactStoreError(f"array digest mismatch for {path.name}.")
        array.setflags(write=False)
        return array
    if "__tuple__" in value:
        return tuple(_decode(item, directory) for item in value["__tuple__"])
    if "__path__" in value:
        return Path(value["__path__"])
    if "__mapping__" in value:
        return {
            _decode(key, directory): _decode(item, directory)
            for key, item in value["__mapping__"]
        }
    if "__enum__" in value:
        try:
            return _resolve_class(value["__enum__"])(value["value"])
        except ReducedODArtifactStoreError:
            raise
        except (TypeError, ValueError) as error:
            raise ReducedODArtifactStoreError(
                f"persisted enum {value['__enum__']!r} is incompatible with "
                "the installed library schema."
            ) from error
    if "__dataclass__" in value:
        cls = _resolve_class(value["__dataclass__"])
        encoded_fields = value.get("fields")
        if not isinstance(encoded_fields, dict):
            raise ReducedODArtifactStoreError(
                f"persisted dataclass {_class_name(cls)!r} has invalid fields."
            )
        decoded = {
            key: _decode(item, directory) for key, item in encoded_fields.items()
        }
        try:
            return cls(**decoded)
        except (TypeError, ValueError) as error:
            raise ReducedODArtifactStoreError(
                f"persisted dataclass {_class_name(cls)!r} is incompatible with "
                "the installed library schema."
            ) from error
    raise ReducedODArtifactStoreError("unrecognized artifact payload object.")


def _descriptor_fingerprint(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


def save_reduced_od_phase_artifact(
    directory: str | Path,
    *,
    phase: str,
    payload: Any,
    configuration_fingerprint: str,
    upstream_fingerprints: Mapping[str, str] = {},
    dimensions: Mapping[str, int] = {},
    semantic_conventions: Mapping[str, str] = {},
    progress: ReducedODProgress | None = None,
) -> str:
    """Atomically publish one self-contained phase directory."""
    target = Path(directory)
    arrays: dict[str, np.ndarray] = {}
    encoded = _encode(payload, arrays)
    descriptor = {
        "array_descriptors": {
            name: {
                "digest": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            }
            for name, array in sorted(arrays.items())
        },
        "configuration_fingerprint": configuration_fingerprint,
        "creation_library_version": __version__,
        "dimensions": dict(sorted(dimensions.items())),
        "payload": encoded,
        "phase": phase,
        "schema_version": REDUCED_OD_STORE_SCHEMA_VERSION,
        "semantic_conventions": dict(sorted(semantic_conventions.items())),
        "upstream_fingerprints": dict(sorted(upstream_fingerprints.items())),
    }
    fingerprint = _descriptor_fingerprint(descriptor)
    document = dict(descriptor, content_fingerprint=fingerprint)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    backup: Path | None = None
    persistence_progress = ReducedODProgressEmitter(
        progress,
        phase="reduced_od_artifact_persistence",
        total=len(arrays) + 1,
    )
    persistence_progress.start(details={"artifact_phase": phase})
    try:
        for array_position, (name, array) in enumerate(arrays.items(), start=1):
            with (temporary / name).open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            persistence_progress.update(
                array_position,
                current_unit=name,
                details={"artifact_phase": phase},
            )
        manifest = temporary / "manifest.json"
        manifest.write_text(canonical_json(document), encoding="utf-8")
        if target.exists():
            backup = target.with_name(f".{target.name}.previous")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(target, backup)
        os.replace(temporary, target)
        if backup is not None:
            shutil.rmtree(backup)
        persistence_progress.update(
            len(arrays) + 1,
            current_unit="manifest.json",
            details={"artifact_phase": phase},
        )
    except Exception:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return fingerprint


def load_reduced_od_phase_artifact(
    directory: str | Path,
    *,
    expected_phase: str,
    expected_configuration_fingerprint: str,
    expected_upstream_fingerprints: Mapping[str, str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load one phase only after validating every recorded identity."""
    target = Path(directory)
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file():
        raise ReducedODArtifactStoreError(f"missing artifact manifest: {manifest_path}")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReducedODArtifactStoreError(
            f"invalid artifact manifest: {manifest_path}"
        ) from error
    fingerprint = document.pop("content_fingerprint", None)
    if fingerprint != _descriptor_fingerprint(document):
        raise ReducedODArtifactStoreError(f"content fingerprint mismatch: {target}")
    if document.get("schema_version") != REDUCED_OD_STORE_SCHEMA_VERSION:
        raise ReducedODArtifactStoreError(f"unsupported artifact schema: {target}")
    if document.get("phase") != expected_phase:
        raise ReducedODArtifactStoreError(f"artifact phase mismatch: {target}")
    if document.get("configuration_fingerprint") != expected_configuration_fingerprint:
        raise ReducedODArtifactStoreError(
            f"configuration fingerprint mismatch: {target}"
        )
    if expected_upstream_fingerprints is not None and document.get(
        "upstream_fingerprints"
    ) != dict(sorted(expected_upstream_fingerprints.items())):
        raise ReducedODArtifactStoreError(f"upstream fingerprint mismatch: {target}")
    payload = _decode(document["payload"], target)
    return payload, dict(document, content_fingerprint=fingerprint)
