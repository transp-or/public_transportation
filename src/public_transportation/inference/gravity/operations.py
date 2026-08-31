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
import numpy as np

from public_transportation import __version__
from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)

from .estimator import (
    GravityEstimationResult,
    GravityExecutionPolicy,
    GravityEstimatorConfig,
    gravity_model_fingerprint,
)
from .objective import GravityObjectiveProblem
from public_transportation.inference.block_coordinate._canonical import fingerprint
from public_transportation.inference.construction_control import (
    normalize_progress_event,
)

GRAVITY_RUN_MANIFEST_SCHEMA_VERSION = 3
GRAVITY_PROGRESS_SCHEMA_VERSION = 1


def _append_progress_line(path: Path, rendered: str, *, durable: bool) -> None:
    """Append one complete JSONL record with an atomic append operation."""

    payload = rendered.encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("progress append wrote no bytes")
            offset += written
        if durable:
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
            # Normalize only at the reporting boundary.  The event object and
            # all numerical code remain untouched, while legacy event classes
            # gain the common hierarchical fields in JSONL output.
            "event": _json_value(normalize_progress_event(event)),
        }
        rendered = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock:
            _append_progress_line(self.path, rendered, durable=self.durable)


def build_gravity_run_manifest(
    *,
    problem: GravityObjectiveProblem,
    compact_layout: CompactODAssignmentLayout,
    estimator_config: GravityEstimatorConfig,
    execution: GravityExecutionPolicy,
    repository_revision: str,
    result: GravityEstimationResult | None = None,
    preflight: object | None = None,
    holdout_mask: object | None = None,
    unsupported_measurement_mask: object | None = None,
    structural_zero_fingerprint: str | None = None,
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
    calibration = problem.calibration_mask
    assert calibration is not None
    holdout = (
        None if holdout_mask is None else jax.numpy.asarray(holdout_mask, dtype=bool)
    )
    unsupported = (
        None
        if unsupported_measurement_mask is None
        else jax.numpy.asarray(unsupported_measurement_mask, dtype=bool)
    )
    for name, mask in (("holdout", holdout), ("unsupported", unsupported)):
        if mask is not None and mask.shape != calibration.shape:
            raise ValueError(f"{name}_measurement_mask must match observations.")
    if unsupported is not None and bool(jax.numpy.any(unsupported & calibration)):
        raise ValueError(
            "unsupported measurement rows must be excluded from calibration."
        )
    artifact_fingerprint = getattr(
        operator, "artifact_fingerprint", None
    ) or fingerprint(
        {
            "schema_version": 1,
            "assignment": operator.assignment_fingerprint,
            "graph": operator.graph_fingerprint,
            "mapping": operator.mapping_fingerprint,
            "layout": operator.compact_layout_fingerprint,
            "theta": operator.theta,
            "representation": operator.representation,
            "dtype": str(operator.dtype),
        }
    )
    specification = problem.parameter_layout.specification
    manifest: dict[str, object] = {
        "schema_version": GRAVITY_RUN_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "repository_revision": str(repository_revision),
        "package_version": __version__,
        "model_fingerprint": gravity_model_fingerprint(problem, compact_layout),
        "model_specification": specification.to_dict(),
        "specification_fingerprint": specification.fingerprint,
        "parameter_layout": problem.parameter_layout.to_dict(),
        "parameter_names": list(problem.parameter_layout.names),
        "parameter_blocks": [
            block.to_dict() for block in problem.parameter_layout.blocks
        ],
        "fingerprints": {
            "compact_layout": compact_layout.fingerprint,
            "assignment": operator.assignment_fingerprint,
            "graph": operator.graph_fingerprint,
            "mapping": operator.mapping_fingerprint,
            "features": problem.features.fingerprint,
            "direct_operator_artifact": artifact_fingerprint,
            "structural_zeros": structural_zero_fingerprint,
        },
        "observation_masks": {
            "calibration": {
                "policy": specification.likelihood.calibration_mask,
                "included": int(calibration.sum()),
                "excluded": int(calibration.size - calibration.sum()),
                "total": int(calibration.size),
            },
            "holdout": {
                "included": None if holdout is None else int(holdout.sum()),
                "total": None if holdout is None else int(holdout.size),
            },
            "unsupported": {
                "excluded": None if unsupported is None else int(unsupported.sum()),
                "total": None if unsupported is None else int(unsupported.size),
            },
        },
        "regularization": [
            {
                "component": block.component,
                "type": block.regularization_type.value,
                "strength": block.regularization_strength,
            }
            for block in problem.parameter_layout.blocks
            if block.regularization_strength > 0
        ],
        "time_discretization": asdict(specification.time),
        "destination_attractiveness_provenance": specification.component(
            "destination_attractiveness"
        ).source,
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
        "convergence_diagnostics": {
            "initial_objective": (
                None if result is None else result.initial_objective
            ),
            "objective": None if result is None else result.objective,
            "gradient_inf_norm": (
                None if result is None else result.gradient_inf_norm
            ),
            "scaled_gradient_inf_norm": (
                None if result is None else result.scaled_gradient_inf_norm
            ),
            "scaled_gradient_tolerance": estimator_config.scaled_gradient_tolerance,
            "typical_objective_scale": estimator_config.typical_objective_scale,
            "typical_parameter_scales": _json_value(
                result.typical_parameter_scales
                if result is not None
                else estimator_config.typical_parameter_scales
            ),
            "typical_objective_scale_provenance": (
                "configured fixed lower bound; verify against the initial objective"
                if result is None
                else result.typical_objective_scale_provenance
            ),
            "typical_parameter_scales_provenance": (
                (
                    "generic default unit scales"
                    if estimator_config.typical_parameter_scales is None
                    else (
                        "configured scalar expanded to every parameter"
                        if np.isscalar(estimator_config.typical_parameter_scales)
                        else "configured per-parameter vector"
                    )
                )
                if result is None
                else result.typical_parameter_scales_provenance
            ),
            "typical_objective_scale_selection": (
                "fixed case-specific typf; recommended rule is "
                "max(abs(initial_objective), objective_floor)"
                if result is None
                else result.typical_objective_scale_selection
            ),
            "objective_dtype": None if result is None else result.objective_dtype,
            "gradient_dtype": None if result is None else result.gradient_dtype,
            "objective_spacing": (
                None if result is None else result.objective_spacing
            ),
            "objective_reduction": (
                None if result is None else result.objective_reduction
            ),
            "objective_tolerance_below_precision": (
                None
                if result is None
                else result.objective_tolerance_below_precision
            ),
            "termination_message": None if result is None else result.message,
            "status": None if result is None else result.status,
            "success": None if result is None else result.success,
        },
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
    if problem.auxiliary_observations.enabled:
        manifest["auxiliary_observations"] = _json_value(
            problem.auxiliary_observations.to_dict()
        )
    return manifest


def write_gravity_run_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    """Atomically publish a run manifest without exposing a partial file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(_json_value(manifest), indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    )
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
