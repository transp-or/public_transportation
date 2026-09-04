"""Minimal checkpointed L-BFGS estimator for the gravity model."""

from __future__ import annotations

import json
import os
import tempfile
from collections import deque
from dataclasses import dataclass, fields, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

from public_transportation.compilation_cache import configure_jax_compilation_cache
from public_transportation.inference.block_coordinate._canonical import fingerprint
from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)
from public_transportation.inference.construction_control import (
    estimate_completed_unit_eta,
)

from .objective import (
    GravityGradientStrategy,
    GravityObjectiveEvaluation,
    GravityObjectiveProblem,
    gravity_value_and_gradient,
)

GRAVITY_CHECKPOINT_SCHEMA_VERSION = 1
GRAVITY_RESULT_SCHEMA_VERSION = 4


def _validate_positive_finite_scale(name: str, value: object) -> float:
    """Return one validated positive finite scale."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and strictly positive.") from error
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return numeric


def _is_scalar_scale(value: object) -> bool:
    """Return whether a scale is scalar, including a zero-dimensional array."""
    if np.isscalar(value):
        return True
    try:
        return np.asarray(value).ndim == 0
    except (TypeError, ValueError):
        return False


def _resolve_typical_parameter_scales(
    parameter_count: int, scales: object | None
) -> np.ndarray:
    """Resolve per-parameter Dennis--Schnabel ``typx`` values."""
    if scales is None:
        return np.ones(parameter_count, dtype=np.float64)
    if _is_scalar_scale(scales):
        value = _validate_positive_finite_scale(
            "typical_parameter_scales", scales
        )
        return np.full(parameter_count, value, dtype=np.float64)
    try:
        resolved = np.asarray(tuple(scales), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "typical_parameter_scales must contain finite positive values."
        ) from error
    if resolved.shape != (parameter_count,):
        raise ValueError(
            "typical_parameter_scales must have one value per parameter."
        )
    if not np.all(np.isfinite(resolved)) or np.any(resolved <= 0.0):
        raise ValueError(
            "typical_parameter_scales must contain finite positive values."
        )
    return resolved


def scaled_gradient_inf_norm(
    parameters: object,
    gradient: object,
    objective: float,
    *,
    typical_objective_scale: float = 1.0,
    typical_parameter_scales: object | None = None,
) -> float:
    """Compute the Dennis--Schnabel scaled-gradient infinity norm."""
    parameter_array = np.asarray(parameters, dtype=np.float64)
    gradient_array = np.asarray(gradient, dtype=np.float64)
    if parameter_array.ndim != 1 or gradient_array.shape != parameter_array.shape:
        raise ValueError("parameters and gradient must be one-dimensional and match.")
    if not np.all(np.isfinite(parameter_array)) or not np.all(
        np.isfinite(gradient_array)
    ):
        raise ValueError("parameters and gradient must contain finite values.")
    typf = _validate_positive_finite_scale(
        "typical_objective_scale", typical_objective_scale
    )
    typx = _resolve_typical_parameter_scales(
        parameter_array.size, typical_parameter_scales
    )
    try:
        objective_value = float(objective)
    except (TypeError, ValueError) as error:
        raise ValueError("objective must be finite.") from error
    if not np.isfinite(objective_value):
        raise ValueError("objective must be finite.")
    denominator = max(abs(objective_value), typf)
    components = (
        np.abs(gradient_array)
        * np.maximum(np.abs(parameter_array), typx)
        / denominator
    )
    return float(np.max(components, initial=0.0))


@dataclass(frozen=True, slots=True)
class GravityEstimatorConfig:
    maximum_iterations: int = 100
    gradient_tolerance: float = 1.0e-6
    objective_tolerance: float = 1.0e-9
    optimizer_maxls: int = 20
    scaled_gradient_tolerance: float = 1.0e-4
    typical_objective_scale: float = 1.0
    typical_parameter_scales: float | tuple[float, ...] | None = None
    optimizer: Literal["scipy", "biogeme_tr_bfgs"] = "scipy"

    def __post_init__(self) -> None:
        if self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive.")
        for name in (
            "gradient_tolerance",
            "objective_tolerance",
            "scaled_gradient_tolerance",
            "typical_objective_scale",
        ):
            object.__setattr__(
                self,
                name,
                _validate_positive_finite_scale(name, getattr(self, name)),
            )
        if self.optimizer_maxls <= 0:
            raise ValueError("optimizer_maxls must be positive.")
        if self.optimizer not in ("scipy", "biogeme_tr_bfgs"):
            raise ValueError(
                "optimizer must be 'scipy' or 'biogeme_tr_bfgs'."
            )
        if self.typical_parameter_scales is not None:
            if _is_scalar_scale(self.typical_parameter_scales):
                object.__setattr__(
                    self,
                    "typical_parameter_scales",
                    _validate_positive_finite_scale(
                        "typical_parameter_scales", self.typical_parameter_scales
                    ),
                )
            else:
                normalized = tuple(self.typical_parameter_scales)
                if not normalized:
                    raise ValueError(
                        "typical_parameter_scales must contain finite positive values."
                    )
                validated = _resolve_typical_parameter_scales(
                    len(normalized), normalized
                )
                object.__setattr__(
                    self,
                    "typical_parameter_scales",
                    tuple(float(value) for value in validated),
                )


@dataclass(frozen=True, slots=True)
class GravityExecutionPolicy:
    gradient_strategy: Literal["auto", "batched_forward", "adjoint"] = "auto"
    automatic_forward_parameter_limit: int = 8
    wall_time_seconds: float | None = None
    checkpoint_path: Path | None = None
    progress_interval: int = 1
    jax_compilation_cache_directory: Path | None = None

    def __post_init__(self) -> None:
        if self.gradient_strategy not in ("auto", "batched_forward", "adjoint"):
            raise ValueError("unsupported gradient_strategy.")
        if self.automatic_forward_parameter_limit <= 0:
            raise ValueError("automatic_forward_parameter_limit must be positive.")
        if self.wall_time_seconds is not None and self.wall_time_seconds <= 0:
            raise ValueError("wall_time_seconds must be positive when provided.")
        if self.progress_interval <= 0:
            raise ValueError("progress_interval must be positive.")


@dataclass(frozen=True, slots=True)
class GravityCompilationDiagnostics:
    strategy: str
    tracing_seconds: float
    lowering_seconds: float
    compilation_seconds: float
    first_execution_seconds: float
    warm_execution_seconds: float
    lowered_text_bytes: int | None


@dataclass(frozen=True, slots=True)
class GravityStrategySelection:
    requested: str
    selected: str
    reason: str
    candidates: tuple[GravityCompilationDiagnostics, ...]
    persistent_compilation_cache_enabled: bool
    persistent_compilation_cache_directory: str | None


@dataclass(frozen=True, slots=True)
class GravityEstimatorProgress:
    iteration: int
    objective: float
    gradient_inf_norm: float
    elapsed_seconds: float
    checkpoint_written: bool
    scaled_gradient_inf_norm: float | None = None
    scaled_gradient_tolerance: float | None = None
    typical_objective_scale: float | None = None
    typical_parameter_scales: float | tuple[float, ...] | None = None
    status: str = "running"
    termination_message: str | None = None
    phase: str = "optimizer_iterations"
    phase_elapsed_seconds: float | None = None
    completed_units: int | None = None
    total_units: int | None = None
    predicted_remaining_seconds: float | None = None
    eta_confidence: str = "unavailable"
    eta_reason: str | None = None
    estimated_completion_at_utc: str | None = None
    work_stack: tuple[dict[str, object], ...] = ()
    active_units: tuple[str, ...] = ()
    queued_units: int | None = None
    active_workers: int | None = None
    requested_workers: int | None = None
    completed_weight: float | None = None
    total_weight: float | None = None
    weighted_fraction: float | None = None
    checkpoint_location: str | None = None
    checkpoint_reusable: bool | None = None
    reused_units: int | None = None
    rebuilt_units: int | None = None
    next_resumable_position: str | None = None
    deadline_remaining_seconds: float | None = None
    deadline_margin_seconds: float | None = None
    will_finish_before_deadline: bool | None = None
    job_elapsed_seconds: float | None = None
    eta_lower_seconds: float | None = None
    eta_upper_seconds: float | None = None
    predicted_job_remaining_seconds: float | None = None
    job_eta_confidence: str = "unavailable"
    job_eta_reason: str | None = None
    estimated_job_completion_at_utc: str | None = None
    initial_objective: float | None = None
    typical_objective_scale_provenance: str | None = None
    typical_parameter_scales_provenance: str | None = None
    typical_objective_scale_selection: str | None = None
    current_unit: str | None = None
    schema_version: int = 1
    optimizer: Literal["scipy", "biogeme_tr_bfgs"] = "scipy"


@dataclass(frozen=True, slots=True)
class GravityEstimationResult:
    schema_version: int
    status: str
    success: bool
    message: str
    raw_parameters: np.ndarray
    physical_parameters: np.ndarray
    free_od_demand: np.ndarray
    active_od_demand: np.ndarray
    full_od_demand: np.ndarray
    predicted_measurements: np.ndarray
    objective: float
    data_log_likelihood: float
    gradient: np.ndarray
    iterations: int
    elapsed_seconds: float
    model_fingerprint: str
    strategy_selection: GravityStrategySelection
    resumed: bool
    checkpoint_path: Path | None
    deadline_phase: str | None = None
    specification_fingerprint: str = ""
    model_specification: dict[str, object] | None = None
    parameter_names: tuple[str, ...] = ()
    parameter_blocks: tuple[dict[str, object], ...] = ()
    feature_cache_fingerprint: str = ""
    direct_operator_artifact_fingerprint: str = ""
    regularization_contribution: float = 0.0
    calibration_measurements: int = 0
    excluded_measurements: int = 0
    time_discretization: dict[str, object] | None = None
    destination_attractiveness_provenance: str | None = None
    count_log_likelihood: float = 0.0
    auxiliary_log_likelihood: float = 0.0
    auxiliary_channel_log_likelihoods: tuple[float, ...] = ()
    auxiliary_observations: dict[str, object] | None = None
    gradient_inf_norm: float | None = None
    scaled_gradient_inf_norm: float | None = None
    scaled_gradient_tolerance: float | None = None
    typical_objective_scale: float | None = None
    typical_parameter_scales: float | tuple[float, ...] | None = None
    objective_dtype: str | None = None
    gradient_dtype: str | None = None
    objective_spacing: float | None = None
    objective_reduction: float | None = None
    objective_tolerance_below_precision: bool | None = None
    initial_objective: float | None = None
    typical_objective_scale_provenance: str | None = None
    typical_parameter_scales_provenance: str | None = None
    typical_objective_scale_selection: str | None = None
    optimizer: Literal["scipy", "biogeme_tr_bfgs"] = "scipy"
    optimizer_message: str | None = None
    optimizer_iterations: int | None = None
    optimizer_evaluations: int | None = None
    optimizer_options: dict[str, object] | None = None
    acceptance: str | None = None
    convergence_reclassification: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != GRAVITY_RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported gravity result schema version.")
        if self.optimizer not in ("scipy", "biogeme_tr_bfgs"):
            raise ValueError("unsupported gravity optimizer metadata.")
        if self.acceptance is None:
            object.__setattr__(
                self,
                "acceptance",
                "accepted" if self.success else "not_accepted",
            )
        for name in (
            "raw_parameters",
            "physical_parameters",
            "free_od_demand",
            "active_od_demand",
            "full_od_demand",
            "predicted_measurements",
            "gradient",
        ):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if (
            self.parameter_names
            and len(self.parameter_names) != self.raw_parameters.size
        ):
            raise ValueError(
                "gravity result parameter names do not match the estimates."
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GravityEstimationResult":
        """Restore a persisted JSON result for diagnostics and reporting.

        The estimator itself continues to use its typed in-memory result. This
        loader is intentionally limited to the JSON representation emitted by
        the gravity driver and validates it through the normal dataclass
        invariants.
        """
        values = dict(payload)
        array_fields = (
            "raw_parameters",
            "physical_parameters",
            "free_od_demand",
            "active_od_demand",
            "full_od_demand",
            "predicted_measurements",
            "gradient",
        )
        for name in array_fields:
            if name not in values:
                raise ValueError(f"persisted gravity result is missing {name!r}.")
            values[name] = np.asarray(values[name])

        selection = values.get("strategy_selection")
        if not isinstance(selection, Mapping):
            raise ValueError("persisted gravity result has no strategy selection.")
        candidates = []
        for candidate in selection.get("candidates", ()):
            if not isinstance(candidate, Mapping):
                raise ValueError("invalid gravity compilation diagnostic.")
            candidates.append(GravityCompilationDiagnostics(**dict(candidate)))
        values["strategy_selection"] = GravityStrategySelection(
            requested=str(selection.get("requested", "auto")),
            selected=str(selection.get("selected", "")),
            reason=str(selection.get("reason", "")),
            candidates=tuple(candidates),
            persistent_compilation_cache_enabled=bool(
                selection.get("persistent_compilation_cache_enabled", False)
            ),
            persistent_compilation_cache_directory=(
                None
                if selection.get("persistent_compilation_cache_directory") is None
                else str(selection["persistent_compilation_cache_directory"])
            ),
        )
        for name in ("parameter_names", "auxiliary_channel_log_likelihoods"):
            if name in values and values[name] is not None:
                values[name] = tuple(values[name])
        if "parameter_blocks" in values and values["parameter_blocks"] is not None:
            values["parameter_blocks"] = tuple(values["parameter_blocks"])
        if values.get("checkpoint_path") is not None:
            values["checkpoint_path"] = Path(str(values["checkpoint_path"]))
        allowed = {field.name for field in fields(cls)}
        return cls(**{name: value for name, value in values.items() if name in allowed})


def _json_safe_result_payload(result: GravityEstimationResult) -> dict[str, object]:
    """Return a small JSON-safe record for a reclassified result."""
    return {
        "schema_version": result.schema_version,
        "status": result.status,
        "success": result.success,
        "acceptance": result.acceptance,
        "message": result.message,
        "raw_parameters": result.raw_parameters.tolist(),
        "objective": result.objective,
        "gradient": result.gradient.tolist(),
        "scaled_gradient_inf_norm": result.scaled_gradient_inf_norm,
        "scaled_gradient_tolerance": result.scaled_gradient_tolerance,
        "model_fingerprint": result.model_fingerprint,
        "direct_operator_artifact_fingerprint": result.direct_operator_artifact_fingerprint,
        "convergence_reclassification": result.convergence_reclassification,
    }


def reclassify_gravity_result(
    result: GravityEstimationResult,
    *,
    scaled_gradient_tolerance: float,
    expected_model_fingerprint: str,
    expected_operator_fingerprint: str,
    output_path: Path | None = None,
) -> GravityEstimationResult:
    """Reclassify a completed fit using only a new convergence tolerance.

    No objective, gradient, parameter, prediction, or scientific identity is
    recomputed.  The caller must supply the model and operator fingerprints
    from the run being reviewed.  If ``output_path`` is supplied it must not
    already exist; the original result is therefore never overwritten.
    """
    tolerance = _validate_positive_finite_scale(
        "scaled_gradient_tolerance", scaled_gradient_tolerance
    )
    if not expected_model_fingerprint or expected_model_fingerprint != result.model_fingerprint:
        raise ValueError("model fingerprint does not match the stored gravity result.")
    if not expected_operator_fingerprint or expected_operator_fingerprint != result.direct_operator_artifact_fingerprint:
        raise ValueError("operator fingerprint does not match the stored gravity result.")
    if result.typical_objective_scale is None:
        raise ValueError("stored result has no typical objective scale.")
    scaled = scaled_gradient_inf_norm(
        result.raw_parameters,
        result.gradient,
        result.objective,
        typical_objective_scale=result.typical_objective_scale,
        typical_parameter_scales=result.typical_parameter_scales,
    )
    message_lower = result.message.lower()
    optimizer_success = result.status == "converged" or (
        result.optimizer in {"scipy", "biogeme_tr_bfgs"}
        and any(token in message_lower for token in ("converg", "relative gradient"))
        and "failed" not in message_lower
    )
    accepted = bool(optimizer_success and scaled <= tolerance)
    metadata = {
        "previous_status": result.status,
        "previous_success": result.success,
        "previous_scaled_gradient_tolerance": result.scaled_gradient_tolerance,
        "new_scaled_gradient_tolerance": tolerance,
        "scaled_gradient_inf_norm": scaled,
        "model_fingerprint": result.model_fingerprint,
        "operator_fingerprint": result.direct_operator_artifact_fingerprint,
    }
    reclassified = replace(
        result,
        status="converged" if accepted else "iteration_limit",
        success=accepted,
        scaled_gradient_inf_norm=scaled,
        scaled_gradient_tolerance=tolerance,
        acceptance="accepted" if accepted else "not_accepted",
        convergence_reclassification=metadata,
    )
    if output_path is not None:
        destination = Path(output_path)
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite existing gravity result: {destination}"
            )
        _atomic_checkpoint(destination, _json_safe_result_payload(reclassified))
    return reclassified


def gravity_model_fingerprint(
    problem: GravityObjectiveProblem, compact_layout: CompactODAssignmentLayout
) -> str:
    operator = problem.operator
    payload: dict[str, object] = {
        "schema_version": 1,
        "features": problem.features.fingerprint,
        "specification": problem.parameter_layout.specification.to_dict(),
        "parameter_layout": problem.parameter_layout.fingerprint,
        "compact_layout": compact_layout.fingerprint,
        "assignment": operator.assignment_fingerprint,
        "graph": operator.graph_fingerprint,
        "mapping": operator.mapping_fingerprint,
        "routing_theta": operator.theta,
        "operator_dtype": str(operator.dtype),
        "likelihood": problem.likelihood.value,
        "rho": problem.rho,
        "mean_floor": problem.mean_floor,
        "observations": problem.observations,
        "calibration_mask": problem.calibration_mask,
    }
    observations = problem.auxiliary_observations
    if observations is not None and observations.enabled:
        payload["auxiliary_observations"] = observations.identity_payload()
    return fingerprint(payload)


def _atomic_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path = Path(path).expanduser()
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


def _write_checkpoint(
    path: Path,
    *,
    model_fingerprint: str,
    raw_parameters: np.ndarray,
    iterations: int,
    elapsed_seconds: float,
    auxiliary_observations: dict[str, object] | None = None,
    optimizer: Literal["scipy", "biogeme_tr_bfgs"] = "scipy",
) -> None:
    payload: dict[str, object] = {
        "schema_version": GRAVITY_CHECKPOINT_SCHEMA_VERSION,
        "model_fingerprint": model_fingerprint,
        "raw_parameters": raw_parameters.tolist(),
        "iterations": iterations,
        "elapsed_seconds": elapsed_seconds,
        "optimizer": optimizer,
    }
    if auxiliary_observations is not None:
        payload["auxiliary_observations"] = auxiliary_observations
    _atomic_checkpoint(path, payload)


def _load_checkpoint(
    path: Path,
    model_fingerprint: str,
    parameter_count: int = 3,
    expected_optimizer: Literal["scipy", "biogeme_tr_bfgs"] = "scipy",
) -> tuple[np.ndarray, int, float]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read gravity checkpoint {path}.") from error
    if payload.get("schema_version") != GRAVITY_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("incompatible gravity checkpoint schema.")
    if payload.get("model_fingerprint") != model_fingerprint:
        raise ValueError("gravity checkpoint model fingerprint mismatch.")
    checkpoint_optimizer = payload.get("optimizer")
    if checkpoint_optimizer is not None and checkpoint_optimizer != expected_optimizer:
        raise ValueError(
            "gravity checkpoint optimizer mismatch: "
            f"checkpoint={checkpoint_optimizer!r}, expected={expected_optimizer!r}."
        )
    if checkpoint_optimizer is None and expected_optimizer != "scipy":
        raise ValueError(
            "gravity checkpoint does not identify an optimizer; "
            "start a separate Biogeme checkpoint."
        )
    raw = np.asarray(payload.get("raw_parameters"), dtype=np.float64)
    if raw.shape != (parameter_count,) or not np.all(np.isfinite(raw)):
        raise ValueError("gravity checkpoint parameters are invalid.")
    return (
        raw,
        int(payload.get("iterations", 0)),
        float(payload.get("elapsed_seconds", 0)),
    )


def _compile_strategy(
    strategy: GravityGradientStrategy,
    raw: jax.Array,
    problem: GravityObjectiveProblem,
    clock: Callable[[], float],
) -> tuple[Any, GravityCompilationDiagnostics]:
    def kernel(value: jax.Array):
        return gravity_value_and_gradient(value, problem=problem, strategy=strategy)

    jitted = jax.jit(kernel)
    started = clock()
    traced = jitted.trace(raw)
    tracing = clock() - started
    started = clock()
    lowered = traced.lower()
    lowering = clock() - started
    try:
        lowered_bytes = len(lowered.as_text().encode("utf-8"))
    except (AttributeError, TypeError, ValueError):
        lowered_bytes = None
    started = clock()
    compiled = lowered.compile()
    compilation = clock() - started
    started = clock()
    first = compiled(raw)
    jax.block_until_ready(first)
    first_execution = clock() - started
    started = clock()
    warm = compiled(raw)
    jax.block_until_ready(warm)
    warm_execution = clock() - started
    return compiled, GravityCompilationDiagnostics(
        strategy.value,
        tracing,
        lowering,
        compilation,
        first_execution,
        warm_execution,
        lowered_bytes,
    )


def _prepare_kernel(
    requested: str,
    raw: jax.Array,
    problem: GravityObjectiveProblem,
    policy: GravityExecutionPolicy,
    clock: Callable[[], float],
) -> tuple[Any, GravityStrategySelection]:
    if requested != "auto":
        strategy = GravityGradientStrategy(requested)
        compiled, explicit_metrics = _compile_strategy(strategy, raw, problem, clock)
        return compiled, GravityStrategySelection(
            requested,
            requested,
            "explicit user selection",
            (explicit_metrics,),
            policy.jax_compilation_cache_directory is not None,
            (
                None
                if policy.jax_compilation_cache_directory is None
                else str(policy.jax_compilation_cache_directory.expanduser().resolve())
            ),
        )
    if problem.operator.is_matrix_free:
        strategy = GravityGradientStrategy.ADJOINT
        compiled, metrics = _compile_strategy(strategy, raw, problem, clock)
        return compiled, GravityStrategySelection(
            requested="auto",
            selected=strategy.value,
            reason=(
                "matrix-free operator defaults to adjoint without benchmarking "
                "multiple cold full-network kernels"
            ),
            candidates=(metrics,),
            persistent_compilation_cache_enabled=(
                policy.jax_compilation_cache_directory is not None
            ),
            persistent_compilation_cache_directory=(
                None
                if policy.jax_compilation_cache_directory is None
                else str(policy.jax_compilation_cache_directory.expanduser().resolve())
            ),
        )
    candidates = (
        GravityGradientStrategy.BATCHED_FORWARD,
        GravityGradientStrategy.ADJOINT,
    )
    compiled_candidates: list[Any] = []
    candidate_metrics: list[GravityCompilationDiagnostics] = []
    for candidate in candidates:
        compiled, item = _compile_strategy(candidate, raw, problem, clock)
        compiled_candidates.append(compiled)
        candidate_metrics.append(item)
    parameter_count = problem.parameter_layout.size
    allowed = (
        range(2)
        if parameter_count <= policy.automatic_forward_parameter_limit
        else range(1, 2)
    )
    selected_index = min(
        allowed, key=lambda index: candidate_metrics[index].warm_execution_seconds
    )
    selected = candidates[selected_index]
    reason = (
        f"minimum measured warm time among strategies allowed for "
        f"{parameter_count} parameters"
    )
    return compiled_candidates[selected_index], GravityStrategySelection(
        "auto",
        selected.value,
        reason,
        tuple(candidate_metrics),
        policy.jax_compilation_cache_directory is not None,
        (
            None
            if policy.jax_compilation_cache_directory is None
            else str(policy.jax_compilation_cache_directory.expanduser().resolve())
        ),
    )


class _DeadlineStop(RuntimeError):
    def __init__(self, phase: str) -> None:
        super().__init__(phase)
        self.phase = phase


@dataclass(frozen=True, slots=True)
class _OptimizerRun:
    """Normalized optimizer output used by the common estimator finalization."""

    raw_parameters: np.ndarray
    success: bool
    message: str
    iterations: int
    evaluations: int
    options: dict[str, object]


def _run_scipy_lbfgsb(
    *,
    evaluate: Callable[[np.ndarray], tuple[float, np.ndarray]],
    initial: np.ndarray,
    callback: Callable[[np.ndarray], None],
    remaining_iterations: int,
    config: GravityEstimatorConfig,
) -> _OptimizerRun:
    """Run the existing SciPy path with its historical options unchanged."""
    options: dict[str, object] = {
        "maxiter": remaining_iterations,
        "gtol": config.gradient_tolerance,
        "ftol": config.objective_tolerance,
        "maxls": config.optimizer_maxls,
    }
    solver = minimize(
        evaluate,
        initial,
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options=options,
    )
    return _OptimizerRun(
        raw_parameters=np.asarray(solver.x, dtype=np.float64),
        success=bool(solver.success),
        message=str(solver.message),
        iterations=int(getattr(solver, "nit", 0) or 0),
        evaluations=int(getattr(solver, "nfev", 0) or 0),
        options=options,
    )


def _biogeme_messages(value: object) -> dict[str, object]:
    """Normalize Biogeme's version-dependent termination metadata."""
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if value is None:
        return {}
    try:
        return {str(key): item for key, item in vars(value).items()}
    except TypeError:
        return {"value": value}


def _biogeme_iteration_count(
    messages: dict[str, object], *, configured_limit: int
) -> int:
    """Recover the iteration count when Biogeme omits it at its limit."""
    for key in ("Number of iterations", "iterations"):
        value = messages.get(key)
        if value is not None:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                break
    termination = str(messages.get("Cause of termination", ""))
    if "maximum number of iterations" in termination.lower():
        return configured_limit
    return 0


def _run_biogeme_tr_bfgs(
    *,
    evaluate: Callable[[np.ndarray], tuple[float, np.ndarray]],
    initial: np.ndarray,
    parameter_names: tuple[str, ...],
    remaining_iterations: int,
    config: GravityEstimatorConfig,
    typical_parameter_scales: np.ndarray | None = None,
    on_evaluation: Callable[[int, float, np.ndarray], None] | None = None,
) -> _OptimizerRun:
    """Run Biogeme TR-BFGS against the already compiled common callback."""
    try:
        from biogeme.optimization import bfgs_trust_region_for_biogeme
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "The Biogeme optimizer requires a separately installed, verified "
            "Biogeme environment. Install the optional 'biogeme' and "
            "'biogeme-optimization' packages before selecting "
            "optimizer = 'biogeme_tr_bfgs'."
        ) from error

    # Reuse the adapter from the isolated pilot.  Importing this module does
    # not import Biogeme; only the optimizer selection above is optional.
    from .biogeme_pilot import _BiogemeObjective

    resolved_parameter_scales = (
        _resolve_typical_parameter_scales(
            initial.size, config.typical_parameter_scales
        )
        if typical_parameter_scales is None
        else np.asarray(typical_parameter_scales, dtype=np.float64)
    )
    objective = _BiogemeObjective(
        evaluate,
        on_evaluation=on_evaluation,
        epsilon=config.gradient_tolerance,
        typical_objective_scale=config.typical_objective_scale,
        typical_parameter_scales=resolved_parameter_scales,
    )
    objective.set_variables(initial)
    options: dict[str, object] = {
        "maxiter": remaining_iterations,
        "tolerance": config.gradient_tolerance,
        "objective_tolerance": config.objective_tolerance,
    }
    bounds = [(None, None) for _ in range(initial.size)]
    optimization_result = bfgs_trust_region_for_biogeme(
        objective,
        initial,
        bounds,
        list(parameter_names),
        options,
    )
    if hasattr(optimization_result, "solution"):
        solution = np.asarray(optimization_result.solution, dtype=np.float64)
        messages = _biogeme_messages(
            getattr(optimization_result, "messages", None)
        )
        optimizer_success = bool(
            getattr(optimization_result, "convergence", False)
        )
    else:
        try:
            solution = np.asarray(optimization_result[0], dtype=np.float64)
            messages = _biogeme_messages(optimization_result[1])
        except (IndexError, TypeError, ValueError) as error:
            raise RuntimeError(
                "Biogeme TR-BFGS returned an unsupported result."
            ) from error
        optimizer_success = bool(messages.get("convergence", False))
    if solution.shape != initial.shape or not np.all(np.isfinite(solution)):
        raise RuntimeError("Biogeme TR-BFGS returned invalid parameters.")
    message = messages.get(
        "Cause of termination", messages.get("message", "")
    )
    if not message:
        message = "Biogeme TR-BFGS did not provide a termination message."
    iterations = _biogeme_iteration_count(
        messages, configured_limit=remaining_iterations
    )
    return _OptimizerRun(
        raw_parameters=solution,
        success=optimizer_success,
        message=str(message),
        iterations=iterations,
        evaluations=len(objective.evaluations),
        options=options,
    )


def estimate_gravity_model(
    *,
    problem: GravityObjectiveProblem,
    compact_layout: CompactODAssignmentLayout,
    initial_raw_parameters: object,
    config: GravityEstimatorConfig = GravityEstimatorConfig(),
    execution: GravityExecutionPolicy = GravityExecutionPolicy(),
    resume: bool = False,
    progress: Callable[[GravityEstimatorProgress], None] | None = None,
    clock: Callable[[], float] = perf_counter,
) -> GravityEstimationResult:
    """Estimate the minimal model and checkpoint only at valid iterate boundaries."""
    problem.features.validate_compact_layout(compact_layout)
    typical_parameter_scales = _resolve_typical_parameter_scales(
        problem.parameter_layout.size, config.typical_parameter_scales
    )
    typical_parameter_scales_tuple = tuple(
        float(value) for value in typical_parameter_scales
    )
    if execution.jax_compilation_cache_directory is not None:
        configure_jax_compilation_cache(execution.jax_compilation_cache_directory)
    model_fingerprint = gravity_model_fingerprint(problem, compact_layout)
    auxiliary_metadata = (
        None
        if not problem.auxiliary_observations.enabled
        else problem.auxiliary_observations.to_dict()
    )
    started = clock()
    checkpoint = execution.checkpoint_path
    resumed_elapsed = 0.0
    completed_iterations = 0
    if resume:
        if checkpoint is None:
            raise ValueError("resume requires checkpoint_path.")
        raw_numpy, completed_iterations, resumed_elapsed = _load_checkpoint(
            checkpoint,
            model_fingerprint,
            problem.parameter_layout.size,
            expected_optimizer=config.optimizer,
        )
        resumed = True
    else:
        raw_numpy = np.asarray(initial_raw_parameters, dtype=np.float64)
        if raw_numpy.shape != (problem.parameter_layout.size,) or not np.all(
            np.isfinite(raw_numpy)
        ):
            raise ValueError("initial_raw_parameters have the wrong shape or values.")
        resumed = False
        if checkpoint is not None:
            if checkpoint.exists():
                raise FileExistsError(
                    f"gravity checkpoint already exists at {checkpoint}; "
                    "resume it or choose another path."
                )
            _write_checkpoint(
                checkpoint,
                model_fingerprint=model_fingerprint,
                raw_parameters=raw_numpy,
                iterations=0,
                elapsed_seconds=0.0,
                auxiliary_observations=auxiliary_metadata,
                optimizer=config.optimizer,
            )
    deadline = (
        None
        if execution.wall_time_seconds is None
        else started + execution.wall_time_seconds
    )
    evaluation_deadline = deadline
    raw_jax = jnp.asarray(raw_numpy, dtype=problem.features.dtype)
    compiled, selection = _prepare_kernel(
        execution.gradient_strategy, raw_jax, problem, execution, clock
    )
    latest_raw = raw_numpy.copy()
    latest_evaluation: GravityObjectiveEvaluation | None = None
    latest_gradient = np.zeros_like(raw_numpy)
    valid_raw = raw_numpy.copy()
    valid_evaluation: GravityObjectiveEvaluation | None = None
    valid_gradient = np.zeros_like(raw_numpy)
    deadline_phase: str | None = None
    iteration_durations: deque[float] = deque(maxlen=32)
    accepted_objectives: list[float] = []
    initial_objective: float | None = None
    latest_objective_dtype: str | None = None
    latest_gradient_dtype: str | None = None
    typical_objective_scale_provenance = (
        "configured fixed lower bound; verify against the initial objective"
    )
    typical_parameter_scales_provenance = (
        "generic default unit scales"
        if config.typical_parameter_scales is None
        else (
            "configured scalar expanded to every parameter"
            if _is_scalar_scale(config.typical_parameter_scales)
            else "configured per-parameter vector"
        )
    )
    typical_objective_scale_selection = (
        "fixed case-specific typf; recommended rule is "
        "max(abs(initial_objective), objective_floor)"
    )
    last_iteration_at = started

    def emit_progress(event: GravityEstimatorProgress) -> None:
        if progress is None:
            return
        try:
            progress(event)
        except Exception:
            # Progress is observability only.  Keep the optimizer and its
            # checkpoint semantics unchanged if a sink is unavailable.
            return

    def evaluate(value: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal latest_raw, latest_evaluation, latest_gradient
        nonlocal valid_evaluation, valid_gradient
        nonlocal latest_objective_dtype, latest_gradient_dtype
        nonlocal initial_objective
        if evaluation_deadline is not None and clock() >= evaluation_deadline:
            raise _DeadlineStop("before objective-and-gradient evaluation")
        if hasattr(problem.operator, "absolute_deadline"):
            problem.operator.absolute_deadline = evaluation_deadline
        try:
            evaluation, gradient = compiled(jnp.asarray(value, dtype=raw_jax.dtype))
            jax.block_until_ready((evaluation, gradient))
        except Exception as error:
            message = str(error)
            if "operator product stopped before a shard batch" in message:
                if "rmatvec operator product" in message:
                    phase = "during reverse operator product"
                elif "matvec operator product" in message or (
                    "matmat operator product" in message
                ):
                    phase = "during forward operator product"
                else:
                    phase = "during operator product"
                raise _DeadlineStop(phase) from error
            raise
        latest_raw = np.asarray(value, dtype=np.float64).copy()
        latest_evaluation = evaluation
        if initial_objective is None:
            initial_objective = float(evaluation.objective)
        latest_objective_dtype = str(np.asarray(evaluation.objective).dtype)
        latest_gradient_dtype = str(np.asarray(gradient).dtype)
        latest_gradient = np.asarray(gradient, dtype=np.float64)
        if np.array_equal(latest_raw, valid_raw):
            valid_evaluation = evaluation
            valid_gradient = latest_gradient.copy()
        return float(evaluation.objective), latest_gradient

    def callback(value: np.ndarray) -> None:
        nonlocal completed_iterations, valid_raw, valid_evaluation, valid_gradient
        nonlocal last_iteration_at
        completed_iterations += 1
        valid_raw = np.asarray(value, dtype=np.float64).copy()
        if np.array_equal(latest_raw, valid_raw):
            valid_evaluation = latest_evaluation
            valid_gradient = latest_gradient.copy()
        now = clock()
        elapsed = resumed_elapsed + now - started
        iteration_durations.append(max(0.0, now - last_iteration_at))
        last_iteration_at = now
        assert latest_evaluation is not None
        accepted_objectives.append(float(latest_evaluation.objective))
        if checkpoint is not None:
            _write_checkpoint(
                checkpoint,
                model_fingerprint=model_fingerprint,
                raw_parameters=np.asarray(value),
                iterations=completed_iterations,
                elapsed_seconds=elapsed,
                auxiliary_observations=auxiliary_metadata,
                optimizer=config.optimizer,
            )
        if (
            progress is not None
            and completed_iterations % execution.progress_interval == 0
        ):
            current_scaled_gradient = scaled_gradient_inf_norm(
                latest_raw,
                latest_gradient,
                float(latest_evaluation.objective),
                typical_objective_scale=config.typical_objective_scale,
                typical_parameter_scales=typical_parameter_scales,
            )
            eta = estimate_completed_unit_eta(
                iteration_durations,
                completed_units=completed_iterations,
                total_units=config.maximum_iterations,
                elapsed_seconds=max(0.0, elapsed),
            )
            emit_progress(
                GravityEstimatorProgress(
                    completed_iterations,
                    float(latest_evaluation.objective),
                    float(np.max(np.abs(latest_gradient), initial=0.0)),
                    elapsed,
                    checkpoint is not None,
                    scaled_gradient_inf_norm=current_scaled_gradient,
                    scaled_gradient_tolerance=config.scaled_gradient_tolerance,
                    typical_objective_scale=config.typical_objective_scale,
                    typical_parameter_scales=typical_parameter_scales_tuple,
                    initial_objective=initial_objective,
                    typical_objective_scale_provenance=(
                        typical_objective_scale_provenance
                    ),
                    typical_parameter_scales_provenance=(
                        typical_parameter_scales_provenance
                    ),
                    typical_objective_scale_selection=(
                        typical_objective_scale_selection
                    ),
                    completed_units=completed_iterations,
                    total_units=config.maximum_iterations,
                    predicted_remaining_seconds=eta.predicted_remaining_seconds,
                    eta_confidence=eta.eta_confidence,
                    eta_reason=eta.eta_reason,
                    estimated_completion_at_utc=eta.estimated_completion_at_utc,
                    work_stack=(
                        {
                            "name": "optimizer_iterations",
                            "completed_units": completed_iterations,
                            "total_units": config.maximum_iterations,
                            "current_unit": f"iteration-{completed_iterations:06d}",
                            "status": "running",
                        },
                    ),
                    active_units=(f"iteration-{completed_iterations:06d}",),
                    requested_workers=1,
                    completed_weight=float(completed_iterations),
                    total_weight=float(config.maximum_iterations),
                    weighted_fraction=(
                        completed_iterations / config.maximum_iterations
                    ),
                    checkpoint_location=(
                        None if checkpoint is None else str(checkpoint)
                    ),
                    checkpoint_reusable=checkpoint is not None,
                    reused_units=(completed_iterations if resumed else 0),
                    rebuilt_units=(0 if resumed else completed_iterations),
                    next_resumable_position=(
                        f"iteration-{completed_iterations + 1:06d}"
                        if completed_iterations < config.maximum_iterations
                        else None
                    ),
                    deadline_remaining_seconds=(
                        None if deadline is None else max(0.0, deadline - now)
                    ),
                    deadline_margin_seconds=(
                        None
                        if deadline is None or eta.predicted_remaining_seconds is None
                        else max(0.0, deadline - now) - eta.predicted_remaining_seconds
                    ),
                    will_finish_before_deadline=(
                        None
                        if deadline is None or eta.predicted_remaining_seconds is None
                        else eta.predicted_remaining_seconds <= max(0.0, deadline - now)
                    ),
                    job_elapsed_seconds=elapsed,
                    eta_lower_seconds=eta.eta_lower_seconds,
                    eta_upper_seconds=eta.eta_upper_seconds,
                    predicted_job_remaining_seconds=eta.predicted_remaining_seconds,
                    job_eta_confidence=eta.eta_confidence,
                    job_eta_reason=eta.eta_reason,
                    estimated_job_completion_at_utc=eta.estimated_completion_at_utc,
                    optimizer=config.optimizer,
                )
            )
        if deadline is not None and clock() >= deadline:
            raise _DeadlineStop("after a completed optimizer iteration")

    def biogeme_evaluation_progress(
        evaluation_count: int, objective: float, gradient: np.ndarray
    ) -> None:
        """Report Biogeme evaluations when no accepted-iterate callback exists.

        Biogeme's public TR-BFGS entry point does not expose accepted
        trust-region iterates.  These events are therefore deliberately labeled
        as evaluations, do not claim resumable iteration checkpoints, and do
        not invent an ETA for the opaque optimizer call.
        """
        if progress is None or evaluation_count % execution.progress_interval != 0:
            return
        now = clock()
        elapsed = resumed_elapsed + now - started
        emit_progress(
            GravityEstimatorProgress(
                iteration=evaluation_count,
                objective=float(objective),
                gradient_inf_norm=float(np.max(np.abs(gradient), initial=0.0)),
                elapsed_seconds=elapsed,
                checkpoint_written=False,
                scaled_gradient_inf_norm=scaled_gradient_inf_norm(
                    latest_raw,
                    gradient,
                    float(objective),
                    typical_objective_scale=config.typical_objective_scale,
                    typical_parameter_scales=typical_parameter_scales,
                ),
                scaled_gradient_tolerance=config.scaled_gradient_tolerance,
                typical_objective_scale=config.typical_objective_scale,
                typical_parameter_scales=typical_parameter_scales_tuple,
                initial_objective=initial_objective,
                typical_objective_scale_provenance=typical_objective_scale_provenance,
                typical_parameter_scales_provenance=typical_parameter_scales_provenance,
                typical_objective_scale_selection=typical_objective_scale_selection,
                status="running",
                phase="optimizer_evaluations",
                completed_units=evaluation_count,
                total_units=None,
                current_unit=f"evaluation-{evaluation_count:06d}",
                eta_confidence="unavailable",
                eta_reason=(
                    "Biogeme exposes objective evaluations but not accepted "
                    "trust-region iterates"
                ),
                checkpoint_location=(None if checkpoint is None else str(checkpoint)),
                checkpoint_reusable=False,
                job_elapsed_seconds=elapsed,
                optimizer="biogeme_tr_bfgs",
            )
        )

    status = "completed"
    success = False
    message = ""
    optimizer_run: _OptimizerRun | None = None
    optimizer_start_iterations = completed_iterations
    if deadline is not None and clock() >= deadline:
        status, message = (
            "stopped_by_time_budget",
            "wall-time budget reached before optimization",
        )
        # No evaluation can safely start. The checkpoint remains the only valid
        # resumable state; callers should resume with a larger budget.
        if valid_evaluation is None:
            evaluation_deadline = None
            if hasattr(problem.operator, "absolute_deadline"):
                problem.operator.absolute_deadline = None
            evaluate(raw_numpy)
            deadline_phase = "before objective-and-gradient evaluation"
    else:
        remaining = max(0, config.maximum_iterations - completed_iterations)
        if remaining == 0:
            evaluate(raw_numpy)
            status, message = (
                "iteration_limit",
                "configured iteration limit already reached",
            )
        else:
            try:
                if config.optimizer == "scipy":
                    optimizer_run = _run_scipy_lbfgsb(
                        evaluate=evaluate,
                        initial=raw_numpy,
                        callback=callback,
                        remaining_iterations=remaining,
                        config=config,
                    )
                else:
                    optimizer_run = _run_biogeme_tr_bfgs(
                        evaluate=evaluate,
                        initial=raw_numpy,
                        parameter_names=problem.parameter_layout.names,
                        remaining_iterations=remaining,
                        config=config,
                        typical_parameter_scales=typical_parameter_scales,
                        on_evaluation=biogeme_evaluation_progress,
                    )
            except _DeadlineStop as stop:
                deadline_phase = stop.phase
                status, message = (
                    "stopped_by_time_budget",
                    f"wall-time budget reached {stop.phase}",
                )
                if valid_evaluation is None:
                    evaluation_deadline = None
                    if hasattr(problem.operator, "absolute_deadline"):
                        problem.operator.absolute_deadline = None
                    evaluate(raw_numpy)
                else:
                    latest_raw = valid_raw.copy()
                    latest_evaluation = valid_evaluation
                    latest_gradient = valid_gradient.copy()
            else:
                assert optimizer_run is not None
                solver_raw = optimizer_run.raw_parameters
                if not np.array_equal(latest_raw, solver_raw):
                    evaluate(solver_raw)
                    assert latest_evaluation is not None
                    accepted_objectives.append(float(latest_evaluation.objective))
                latest_raw = solver_raw
                if config.optimizer == "biogeme_tr_bfgs":
                    completed_iterations += optimizer_run.iterations
                    if checkpoint is not None:
                        _write_checkpoint(
                            checkpoint,
                            model_fingerprint=model_fingerprint,
                            raw_parameters=solver_raw,
                            iterations=completed_iterations,
                            elapsed_seconds=resumed_elapsed + clock() - started,
                            auxiliary_observations=auxiliary_metadata,
                            optimizer=config.optimizer,
                        )
                success = optimizer_run.success
                status = "converged" if success else "iteration_limit"
                message = optimizer_run.message
                if deadline is not None and clock() >= deadline:
                    status = "stopped_by_time_budget"
                    success = False
                    deadline_phase = "after optimizer returned"
    optimizer_message = message
    optimizer_iterations = (
        completed_iterations
        if optimizer_run is None
        else (
            optimizer_run.iterations
            if optimizer_run.iterations > 0
            else max(0, completed_iterations - optimizer_start_iterations)
        )
    )
    optimizer_evaluations = (
        0 if optimizer_run is None else optimizer_run.evaluations
    )
    optimizer_options = {} if optimizer_run is None else optimizer_run.options
    assert latest_evaluation is not None
    objective_value = float(latest_evaluation.objective)
    gradient_inf_norm = float(np.max(np.abs(latest_gradient), initial=0.0))
    scaled_gradient = scaled_gradient_inf_norm(
        latest_raw,
        latest_gradient,
        objective_value,
        typical_objective_scale=config.typical_objective_scale,
        typical_parameter_scales=typical_parameter_scales,
    )
    if status == "converged" and scaled_gradient > config.scaled_gradient_tolerance:
        success = False
        status = "iteration_limit"
    objective_array = np.asarray(latest_evaluation.objective)
    objective_spacing = float(
        np.spacing(objective_array.dtype.type(objective_value))
    )
    objective_reduction = (
        None
        if len(accepted_objectives) < 2
        else accepted_objectives[-2] - accepted_objectives[-1]
    )
    objective_tolerance_below_precision = config.objective_tolerance < abs(
        objective_spacing
    )
    final_elapsed = resumed_elapsed + clock() - started
    if progress is not None:
        final_eta = estimate_completed_unit_eta(
            iteration_durations,
            completed_units=completed_iterations,
            total_units=config.maximum_iterations,
            elapsed_seconds=max(0.0, final_elapsed),
        )
        emit_progress(
            GravityEstimatorProgress(
                iteration=completed_iterations,
                objective=objective_value,
                gradient_inf_norm=gradient_inf_norm,
                elapsed_seconds=final_elapsed,
                checkpoint_written=checkpoint is not None,
                scaled_gradient_inf_norm=scaled_gradient,
                scaled_gradient_tolerance=config.scaled_gradient_tolerance,
                typical_objective_scale=config.typical_objective_scale,
                typical_parameter_scales=typical_parameter_scales_tuple,
                initial_objective=initial_objective,
                typical_objective_scale_provenance=typical_objective_scale_provenance,
                typical_parameter_scales_provenance=typical_parameter_scales_provenance,
                typical_objective_scale_selection=typical_objective_scale_selection,
                status=status,
                termination_message=message,
                completed_units=completed_iterations,
                total_units=config.maximum_iterations,
                predicted_remaining_seconds=final_eta.predicted_remaining_seconds,
                eta_confidence=final_eta.eta_confidence,
                eta_reason=final_eta.eta_reason,
                estimated_completion_at_utc=final_eta.estimated_completion_at_utc,
                work_stack=(
                    {
                        "name": "optimizer_iterations",
                        "completed_units": completed_iterations,
                        "total_units": config.maximum_iterations,
                        "current_unit": (
                            None
                            if status == "converged"
                            else f"iteration-{completed_iterations:06d}"
                        ),
                        "status": status,
                    },
                ),
                checkpoint_location=None if checkpoint is None else str(checkpoint),
                checkpoint_reusable=checkpoint is not None,
                reused_units=(completed_iterations if resumed else 0),
                rebuilt_units=(0 if resumed else completed_iterations),
                optimizer=config.optimizer,
            )
        )
    free_demand = np.asarray(latest_evaluation.demand)
    active = np.zeros(compact_layout.num_active, dtype=free_demand.dtype)
    active[np.asarray(compact_layout.free_compact_indices, dtype=np.int64)] = (
        free_demand
    )
    active[np.asarray(compact_layout.fixed_compact_indices, dtype=np.int64)] = (
        np.asarray(compact_layout.fixed_compact_values)
    )
    full = np.zeros(compact_layout.num_od_total, dtype=free_demand.dtype)
    full[np.asarray(compact_layout.active_full_indices, dtype=np.int64)] = active
    physical = np.asarray(problem.parameter_layout.physical_vector(latest_raw))
    elapsed = resumed_elapsed + clock() - started
    artifact_fingerprint = getattr(problem.operator, "artifact_fingerprint", None)
    if artifact_fingerprint is None:
        artifact_fingerprint = fingerprint(
            {
                "schema_version": 1,
                "assignment": problem.operator.assignment_fingerprint,
                "graph": problem.operator.graph_fingerprint,
                "mapping": problem.operator.mapping_fingerprint,
                "layout": problem.operator.compact_layout_fingerprint,
                "theta": problem.operator.theta,
                "representation": problem.operator.representation,
                "dtype": str(problem.operator.dtype),
            }
        )
    return GravityEstimationResult(
        GRAVITY_RESULT_SCHEMA_VERSION,
        status,
        success,
        message,
        latest_raw,
        physical,
        free_demand,
        active,
        full,
        np.asarray(latest_evaluation.measurement_mean),
        objective_value,
        float(latest_evaluation.data_log_likelihood),
        latest_gradient,
        completed_iterations,
        elapsed,
        model_fingerprint,
        selection,
        resumed,
        checkpoint,
        deadline_phase,
        problem.parameter_layout.specification.fingerprint,
        problem.parameter_layout.specification.to_dict(),
        problem.parameter_layout.names,
        tuple(block.to_dict() for block in problem.parameter_layout.blocks),
        problem.features.fingerprint,
        str(artifact_fingerprint),
        float(latest_evaluation.regularization),
        int(latest_evaluation.calibration_measurements),
        int(latest_evaluation.excluded_measurements),
        {
            "units": problem.parameter_layout.specification.time.units,
            "interpretation": problem.parameter_layout.specification.time.interpretation,
            "bin_labels": list(problem.parameter_layout.specification.time.bin_labels),
            "smooth_basis_name": problem.parameter_layout.specification.time.smooth_basis_name,
        },
        problem.parameter_layout.specification.component(
            "destination_attractiveness"
        ).source,
        count_log_likelihood=float(latest_evaluation.count_log_likelihood),
        auxiliary_log_likelihood=float(latest_evaluation.auxiliary_log_likelihood),
        auxiliary_channel_log_likelihoods=tuple(
            float(value)
            for value in latest_evaluation.auxiliary_channel_log_likelihoods
        ),
        auxiliary_observations=auxiliary_metadata,
        gradient_inf_norm=gradient_inf_norm,
        scaled_gradient_inf_norm=scaled_gradient,
        scaled_gradient_tolerance=config.scaled_gradient_tolerance,
        typical_objective_scale=config.typical_objective_scale,
        typical_parameter_scales=typical_parameter_scales_tuple,
        objective_dtype=str(objective_array.dtype),
        gradient_dtype=(
            latest_gradient_dtype
            if latest_gradient_dtype is not None
            else str(np.asarray(latest_gradient).dtype)
        ),
        objective_spacing=objective_spacing,
        objective_reduction=objective_reduction,
        objective_tolerance_below_precision=objective_tolerance_below_precision,
        initial_objective=initial_objective,
        typical_objective_scale_provenance=typical_objective_scale_provenance,
        typical_parameter_scales_provenance=typical_parameter_scales_provenance,
        typical_objective_scale_selection=typical_objective_scale_selection,
        optimizer=config.optimizer,
        optimizer_message=optimizer_message,
        optimizer_iterations=optimizer_iterations,
        optimizer_evaluations=optimizer_evaluations,
        optimizer_options=optimizer_options,
    )
