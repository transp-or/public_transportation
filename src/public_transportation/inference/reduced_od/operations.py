"""Operational contracts and atomic persistence for reduced-OD fitting."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from public_transportation.preprocessing.reduced_od.artifacts import canonical_json


REDUCED_OD_FIT_SCHEMA_VERSION = 1
ReducedODEstimationMethod = Literal["ml", "map"]
ReducedODFitStatus = Literal["complete", "deadline", "interrupted", "failed"]
ReducedODPrecisionPolicy = Literal[
    "float64_required", "float64_preferred", "allow_float32"
]


@dataclass(frozen=True, slots=True)
class ReducedODNumericalConfig:
    """Explicit dtype and optimizer-resolution contract."""

    precision: ReducedODPrecisionPolicy = "float64_required"
    requested_dtype: Literal["float32", "float64"] = "float64"
    reject_unresolved_function_tolerance: bool = True

    def __post_init__(self) -> None:
        if self.precision not in {
            "float64_required",
            "float64_preferred",
            "allow_float32",
        }:
            raise ValueError("unsupported reduced-OD precision policy.")
        if self.requested_dtype not in {"float32", "float64"}:
            raise ValueError("requested_dtype must be float32 or float64.")


@dataclass(frozen=True, slots=True)
class ReducedODRawParameterBounds:
    """Optional independent bounds in the optimizer's raw coordinates."""

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.array(self.lower, dtype=np.float64, copy=True)
        upper = np.array(self.upper, dtype=np.float64, copy=True)
        if lower.ndim != 1 or upper.shape != lower.shape:
            raise ValueError("raw parameter bounds must be aligned vectors.")
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
            raise ValueError("raw parameter bounds cannot contain NaN.")
        if np.any(lower >= upper):
            raise ValueError("every raw lower bound must be below its upper bound.")
        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True)
class ReducedODNamedRawParameterBounds:
    """Bounds keyed by canonical raw parameter name."""

    bounds: Mapping[str, tuple[float, float]]
    require_complete: bool = True

    def __post_init__(self) -> None:
        copied: dict[str, tuple[float, float]] = {}
        for name, interval in self.bounds.items():
            if not isinstance(name, str) or not name:
                raise ValueError("raw bound names must be non-empty strings.")
            if len(interval) != 2:
                raise ValueError(f"raw bounds for {name!r} must contain two values.")
            lower, upper = (float(interval[0]), float(interval[1]))
            if np.isnan(lower) or np.isnan(upper) or lower >= upper:
                raise ValueError(
                    f"raw lower bound for {name!r} must be below its upper bound."
                )
            copied[name] = (lower, upper)
        object.__setattr__(self, "bounds", copied)

    def resolve(self, parameter_names: Sequence[str]) -> ReducedODRawParameterBounds:
        """Resolve named intervals in the supplied canonical ordering."""
        names = tuple(parameter_names)
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique when resolving bounds.")
        unknown = sorted(set(self.bounds) - set(names))
        if unknown:
            raise ValueError(f"unknown raw parameter bounds: {unknown}.")
        missing = [name for name in names if name not in self.bounds]
        if self.require_complete and missing:
            raise ValueError(f"missing required raw parameter bounds: {missing}.")
        lower = np.asarray(
            [self.bounds.get(name, (-np.inf, np.inf))[0] for name in names]
        )
        upper = np.asarray(
            [self.bounds.get(name, (-np.inf, np.inf))[1] for name in names]
        )
        return ReducedODRawParameterBounds(lower=lower, upper=upper)


@dataclass(frozen=True, slots=True)
class ReducedODFitConfig:
    """Optimizer and restart policy for one reduced gravity fit."""

    method: ReducedODEstimationMethod = "ml"
    maximum_iterations: int = 200
    gradient_tolerance: float = 1.0e-6
    gradient_convergence_factor: float = 10.0
    function_tolerance: float = 1.0e-10
    deadline_seconds: float | None = None
    checkpoint_every_iterations: int = 10
    numerical: ReducedODNumericalConfig = ReducedODNumericalConfig()
    raw_parameter_bounds: ReducedODRawParameterBounds | None = None
    named_raw_parameter_bounds: ReducedODNamedRawParameterBounds | None = None
    checkpoint_path: str | Path | None = None
    resume: bool = False
    progress_interval_iterations: int = 1
    progress_interval_seconds: float = 5.0
    identification_condition_threshold: float = 1.0e8
    identification_eigenvalue_tolerance: float = 1.0e-8
    boundary_relative_tolerance: float = 1.0e-6
    dispersion_floor_warning_factor: float = 100.0
    production_multiplier_minimum_warning: float = 1.0e-6
    production_multiplier_maximum_warning: float = 1.0e6
    production_total_ratio_warning: tuple[float, float] = (0.1, 10.0)

    def __post_init__(self) -> None:
        if self.method not in {"ml", "map"}:
            raise ValueError("method must be 'ml' or 'map'.")
        if self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive.")
        if self.gradient_tolerance <= 0.0 or self.function_tolerance <= 0.0:
            raise ValueError("optimizer tolerances must be positive.")
        if self.gradient_convergence_factor < 1.0:
            raise ValueError("gradient_convergence_factor must be at least one.")
        if self.deadline_seconds is not None and self.deadline_seconds <= 0.0:
            raise ValueError("deadline_seconds must be positive when provided.")
        if self.checkpoint_every_iterations <= 0:
            raise ValueError("checkpoint_every_iterations must be positive.")
        if (
            self.raw_parameter_bounds is not None
            and self.named_raw_parameter_bounds is not None
        ):
            raise ValueError("provide vector or named raw bounds, not both.")
        if self.resume and self.checkpoint_path is None:
            raise ValueError("resume requires checkpoint_path.")
        if self.progress_interval_iterations <= 0:
            raise ValueError("progress_interval_iterations must be positive.")
        if self.progress_interval_seconds <= 0.0:
            raise ValueError("progress_interval_seconds must be positive.")
        if self.identification_condition_threshold <= 1.0:
            raise ValueError("identification condition threshold must exceed one.")
        if self.identification_eigenvalue_tolerance <= 0.0:
            raise ValueError("identification eigenvalue tolerance must be positive.")
        if self.boundary_relative_tolerance <= 0.0:
            raise ValueError("boundary_relative_tolerance must be positive.")
        if self.dispersion_floor_warning_factor <= 1.0:
            raise ValueError("dispersion floor warning factor must exceed one.")
        if self.production_multiplier_minimum_warning <= 0.0:
            raise ValueError("production multiplier minimum must be positive.")
        if (
            self.production_multiplier_maximum_warning
            <= self.production_multiplier_minimum_warning
        ):
            raise ValueError("production multiplier warning thresholds are invalid.")
        low, high = self.production_total_ratio_warning
        if low <= 0.0 or high <= low:
            raise ValueError("production total ratio warning thresholds are invalid.")


@dataclass(frozen=True, slots=True)
class GaussianRawParameterPrior:
    """Independent Gaussian prior on the unconstrained parameter vector."""

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = np.array(self.mean, dtype=np.float64, copy=True)
        scale = np.array(self.scale, dtype=np.float64, copy=True)
        if mean.ndim != 1 or scale.shape != mean.shape:
            raise ValueError("prior mean and scale must be aligned vectors.")
        if not np.all(np.isfinite(mean)) or np.any(np.isnan(scale)):
            raise ValueError("prior means must be finite and scales cannot be NaN.")
        if np.any(scale <= 0.0):
            raise ValueError("prior scales must be positive; infinity denotes flat.")
        mean.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)


@dataclass(frozen=True, slots=True)
class ReducedODFitManifest:
    model_fingerprint: str
    method: ReducedODEstimationMethod
    parameter_count: int
    schema_version: int = REDUCED_OD_FIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.model_fingerprint:
            raise ValueError("model_fingerprint must be non-empty.")
        if self.method not in {"ml", "map"}:
            raise ValueError("unsupported estimation method.")
        if self.parameter_count <= 0:
            raise ValueError("parameter_count must be positive.")
        if self.schema_version != REDUCED_OD_FIT_SCHEMA_VERSION:
            raise ValueError("unsupported fit schema version.")


@dataclass(frozen=True, slots=True)
class ReducedODCheckpoint:
    manifest: ReducedODFitManifest
    raw_parameters: np.ndarray
    iteration: int
    objective: float
    elapsed_seconds: float

    def __post_init__(self) -> None:
        raw = np.array(self.raw_parameters, dtype=np.float64, copy=True)
        if raw.shape != (self.manifest.parameter_count,):
            raise ValueError("checkpoint parameter dimension is incompatible.")
        if not np.all(np.isfinite(raw)) or not np.isfinite(self.objective):
            raise ValueError("checkpoint numerical values must be finite.")
        if self.iteration < 0 or self.elapsed_seconds < 0.0:
            raise ValueError("checkpoint counters must be non-negative.")
        raw.setflags(write=False)
        object.__setattr__(self, "raw_parameters", raw)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_reduced_od_checkpoint(
    path: str | Path, checkpoint: ReducedODCheckpoint
) -> None:
    document = {
        "elapsed_seconds": checkpoint.elapsed_seconds,
        "iteration": checkpoint.iteration,
        "manifest": asdict(checkpoint.manifest),
        "objective": checkpoint.objective,
        "raw_parameters": checkpoint.raw_parameters.tolist(),
    }
    _atomic_write(Path(path), canonical_json(document).encode("utf-8"))


def load_reduced_od_checkpoint(
    path: str | Path, *, expected_manifest: ReducedODFitManifest
) -> ReducedODCheckpoint:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = ReducedODFitManifest(**document["manifest"])
    if manifest != expected_manifest:
        raise ValueError("checkpoint manifest is incompatible with this fit.")
    return ReducedODCheckpoint(
        manifest=manifest,
        raw_parameters=np.asarray(document["raw_parameters"], dtype=np.float64),
        iteration=int(document["iteration"]),
        objective=float(document["objective"]),
        elapsed_seconds=float(document["elapsed_seconds"]),
    )


def save_reduced_od_fit_result(
    path: str | Path,
    *,
    manifest: ReducedODFitManifest,
    status: ReducedODFitStatus,
    raw_parameters: np.ndarray,
    summary: Mapping[str, Any],
) -> None:
    """Publish a self-describing result; callers choose the final filename."""
    document = {
        "manifest": asdict(manifest),
        "raw_parameters": np.asarray(raw_parameters, dtype=float).tolist(),
        "saved_unix_seconds": time.time(),
        "schema_version": REDUCED_OD_FIT_SCHEMA_VERSION,
        "status": status,
        "summary": dict(summary),
    }
    _atomic_write(Path(path), canonical_json(document).encode("utf-8"))
