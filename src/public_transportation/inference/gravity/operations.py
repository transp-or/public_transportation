"""Durable operational records for long-running gravity estimation."""

from __future__ import annotations

import json
import os
import platform
import tempfile
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Mapping

import jax

from public_transportation import __version__
from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)

from .estimator import (
    GravityExecutionPolicy,
    GravityEstimatorConfig,
    gravity_model_fingerprint,
)
from .objective import GravityObjectiveProblem

GRAVITY_RUN_MANIFEST_SCHEMA_VERSION = 1
GRAVITY_PROGRESS_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: object) -> object:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(slots=True)
class GravityJSONLProgressSink:
    """Thread-safe append-only progress sink with optional durable flushes."""

    path: Path
    durable: bool = True
    context: Mapping[str, object] | None = None
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: object) -> None:
        payload = {
            "schema_version": GRAVITY_PROGRESS_SCHEMA_VERSION,
            "recorded_at_utc": _utc_now(),
            "event_type": type(event).__name__,
            **({} if self.context is None else dict(self.context)),
            "event": _json_value(event),
        }
        rendered = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            if self.durable:
                os.fsync(stream.fileno())


def build_gravity_run_manifest(
    *,
    problem: GravityObjectiveProblem,
    compact_layout: CompactODAssignmentLayout,
    estimator_config: GravityEstimatorConfig,
    execution: GravityExecutionPolicy,
    repository_revision: str,
    preflight: object | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a serializable immutable-identity and execution record."""
    operator = problem.operator
    environment_names = (
        "JAX_ENABLE_X64",
        "JAX_PLATFORM_NAME",
        "OMP_NUM_THREADS",
        "XLA_FLAGS",
    )
    return {
        "schema_version": GRAVITY_RUN_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "repository_revision": str(repository_revision),
        "package_version": __version__,
        "model_fingerprint": gravity_model_fingerprint(problem, compact_layout),
        "fingerprints": {
            "compact_layout": compact_layout.fingerprint,
            "assignment": operator.assignment_fingerprint,
            "graph": operator.graph_fingerprint,
            "mapping": operator.mapping_fingerprint,
        },
        "operator": {
            "representation": operator.representation,
            "num_free_od": operator.num_free_od,
            "num_measurements": operator.num_measurements,
            "dtype": str(operator.dtype),
            "theta": operator.theta,
            "operator_shards_per_batch": getattr(
                operator, "operator_shards_per_batch", None
            ),
            "group_execution_strategy": getattr(
                operator, "group_execution_strategy", None
            ),
            "shard_execution_strategy": getattr(
                operator, "shard_execution_strategy", None
            ),
            "operator_concurrency": getattr(operator, "operator_concurrency", None),
            "effective_operator_concurrency": getattr(
                operator, "effective_operator_concurrency", None
            ),
            "maximum_concurrent_routing_bytes": getattr(
                operator, "maximum_concurrent_routing_bytes", None
            ),
            "resident_shard_limit": getattr(operator, "resident_shard_limit", None),
            "initial_predicted_batch_seconds": getattr(
                operator, "initial_predicted_batch_seconds", None
            ),
            "deadline_safety_margin_seconds": getattr(
                operator, "deadline_safety_margin_seconds", None
            ),
        },
        "estimator_config": _json_value(estimator_config),
        "execution": _json_value(execution),
        "jax": {
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "environment": {
                name: os.environ[name]
                for name in environment_names
                if name in os.environ
            },
            "python": platform.python_version(),
        },
        "preflight": _json_value(preflight),
        "extra": _json_value({} if extra is None else extra),
    }


def write_gravity_run_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    """Atomically publish a run manifest without exposing a partial file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        _json_value(manifest), indent=2, sort_keys=True, ensure_ascii=True
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
