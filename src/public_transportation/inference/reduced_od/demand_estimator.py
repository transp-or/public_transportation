"""Checkpointed optimizer for the resolved generic demand family."""

from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

from public_transportation.preprocessing.reduced_od.artifacts import canonical_json

from .demand_model import DemandModelProblem, evaluate_demand_model
from .operations import (
    GaussianRawParameterPrior,
    ReducedODCheckpoint,
    ReducedODFitConfig,
    ReducedODFitManifest,
    ReducedODFitStatus,
    ReducedODRawParameterBounds,
    load_reduced_od_checkpoint,
    save_reduced_od_checkpoint,
)

ProgressCallback = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class DemandFitIdentity:
    specification_fingerprint: str
    parameter_layout_fingerprint: str
    feature_fingerprint: str
    grouping_fingerprint: str
    operator_fingerprint: str
    data_fingerprint: str
    software_schema: str = "generic-demand-fit-v1"

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class DemandFitResult:
    raw_parameters: np.ndarray
    objective: float
    log_likelihood: float
    iterations: int
    evaluations: int
    success: bool
    message: str
    compile_seconds: float
    optimization_seconds: float
    identity: DemandFitIdentity
    status: ReducedODFitStatus = "complete"
    resumed_from_iteration: int = 0
    gradient_infinity_norm: float = float("nan")
    checkpoint_path: str | None = None
    average_iteration_seconds: float | None = None
    estimated_remaining_seconds: float | None = None

    def __post_init__(self) -> None:
        value = np.array(self.raw_parameters, dtype=np.float64, copy=True)
        value.setflags(write=False)
        object.__setattr__(self, "raw_parameters", value)


def estimate_demand_model(
    *,
    problem: DemandModelProblem,
    initial_raw_parameters: object,
    identity: DemandFitIdentity,
    config: ReducedODFitConfig = ReducedODFitConfig(),
    prior: GaussianRawParameterPrior | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    progress: ProgressCallback | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> DemandFitResult:
    """Fit one resolved kernel; checkpoints reject every identity mismatch."""
    layout = problem.parameter_layout
    if config.method == "map" and prior is None:
        raise ValueError("MAP estimation requires a prior.")
    if prior is not None and prior.mean.shape != (layout.size,):
        raise ValueError("prior dimension does not match the parameter layout.")
    raw = np.asarray(initial_raw_parameters, dtype=np.float64)
    if raw.shape != (problem.parameter_layout.size,) or not np.all(np.isfinite(raw)):
        raise ValueError("initial parameters are invalid.")
    fit_contract = {
        "identity": identity.fingerprint,
        "prior": None
        if prior is None
        else {"mean": prior.mean.tolist(), "scale": prior.scale.tolist()},
        "raw_bounds": None
        if config.raw_parameter_bounds is None
        else {
            "lower": config.raw_parameter_bounds.lower.tolist(),
            "upper": config.raw_parameter_bounds.upper.tolist(),
        },
        "named_bounds": None
        if config.named_raw_parameter_bounds is None
        else {
            "bounds": dict(sorted(config.named_raw_parameter_bounds.bounds.items())),
            "require_complete": config.named_raw_parameter_bounds.require_complete,
        },
    }
    manifest = ReducedODFitManifest(
        hashlib.sha256(canonical_json(fit_contract).encode()).hexdigest(),
        config.method,
        raw.size,
    )
    if checkpoint_path is not None and config.checkpoint_path is not None:
        if Path(checkpoint_path) != Path(config.checkpoint_path):
            raise ValueError("checkpoint_path conflicts with the fit configuration.")
    effective_checkpoint_path = checkpoint_path or config.checkpoint_path
    effective_resume = resume or config.resume
    if config.deadline_seconds is not None and effective_checkpoint_path is None:
        raise ValueError("deadline_seconds requires a checkpoint_path.")
    iteration = 0
    resumed_iteration = 0
    if effective_resume:
        if effective_checkpoint_path is None or not Path(effective_checkpoint_path).is_file():
            raise ValueError("resume requires an existing checkpoint_path.")
        checkpoint = load_reduced_od_checkpoint(
            effective_checkpoint_path, expected_manifest=manifest
        )
        raw = np.asarray(checkpoint.raw_parameters)
        iteration = checkpoint.iteration
        resumed_iteration = iteration
    bounds: ReducedODRawParameterBounds | None = config.raw_parameter_bounds
    if config.named_raw_parameter_bounds is not None:
        bounds = config.named_raw_parameter_bounds.resolve(layout.raw_parameter_names)
    scipy_bounds = None
    if bounds is not None:
        if bounds.lower.shape != (layout.size,):
            raise ValueError("raw parameter bounds do not match the parameter layout.")
        if np.any(raw < bounds.lower) or np.any(raw > bounds.upper):
            raise ValueError("initial parameters must lie inside raw bounds.")
        scipy_bounds = tuple(zip(bounds.lower, bounds.upper, strict=True))
    prior_mean = None if prior is None else jnp.asarray(prior.mean)
    prior_scale = None if prior is None else jnp.asarray(prior.scale)

    def total_objective(value: jax.Array) -> jax.Array:
        result = evaluate_demand_model(value, problem=problem).objective
        if config.method == "map":
            assert prior_mean is not None and prior_scale is not None
            standardized = (value - prior_mean) / prior_scale
            result += 0.5 * jnp.sum(standardized * standardized)
        return result

    started = clock()
    compiled = jax.jit(
        jax.value_and_grad(total_objective)
    )
    compile_started = clock()
    first = compiled(jnp.asarray(raw))
    jax.block_until_ready(first)
    compile_seconds = clock() - compile_started
    first_objective, first_gradient = first
    evaluations = 1

    def emit(status: str, **values: object) -> None:
        if progress is not None:
            progress(
                {
                    "phase": "generic_demand_fit",
                    "status": status,
                    "model_fingerprint": identity.specification_fingerprint,
                    "elapsed_seconds": max(0.0, clock() - started),
                    **values,
                }
            )

    emit(
        "started",
        operation="optimization",
        completed_iterations=iteration,
        completed_iterations_this_run=0,
        total_iterations=resumed_iteration + config.maximum_iterations,
    )

    optimization_started = clock()
    previous_iteration_completed = optimization_started
    iteration_durations: deque[float] = deque(maxlen=10)
    last_checkpoint_time: float | None = None
    latest = raw.copy()
    latest_objective = float(first_objective)
    latest_gradient = np.asarray(first_gradient, dtype=np.float64)
    evaluated = raw.copy()

    class DeadlineReached(Exception):
        pass

    def deadline_reached() -> bool:
        return config.deadline_seconds is not None and clock() - optimization_started >= config.deadline_seconds

    def evaluate(value: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluations, evaluated, latest_objective, latest_gradient
        candidate = np.asarray(value, dtype=np.float64)
        if np.array_equal(candidate, evaluated):
            return latest_objective, latest_gradient.copy()
        if deadline_reached():
            raise DeadlineReached
        objective, gradient = compiled(jnp.asarray(value))
        jax.block_until_ready((objective, gradient))
        evaluations += 1
        evaluated = candidate.copy()
        latest_objective = float(objective)
        latest_gradient = np.asarray(gradient, dtype=np.float64)
        if not np.isfinite(latest_objective) or not np.all(np.isfinite(latest_gradient)):
            raise FloatingPointError("generic demand objective or gradient became nonfinite.")
        if deadline_reached():
            raise DeadlineReached
        return latest_objective, latest_gradient.copy()

    def save_checkpoint() -> None:
        nonlocal last_checkpoint_time
        if effective_checkpoint_path is not None:
            save_reduced_od_checkpoint(
                effective_checkpoint_path,
                ReducedODCheckpoint(
                    manifest, latest, iteration, latest_objective, clock() - optimization_started
                ),
            )
            last_checkpoint_time = clock()

    def callback(value: np.ndarray) -> None:
        nonlocal iteration, latest, previous_iteration_completed
        now = clock()
        iteration_seconds = max(0.0, now - previous_iteration_completed)
        previous_iteration_completed = now
        iteration_durations.append(iteration_seconds)
        iteration += 1
        latest = np.asarray(value).copy()
        objective = latest_objective
        if (
            effective_checkpoint_path is not None
            and iteration % config.checkpoint_every_iterations == 0
        ):
            save_checkpoint()
        completed_this_run = iteration - resumed_iteration
        remaining_iterations = max(
            0, config.maximum_iterations - completed_this_run
        )
        rolling_average = float(np.mean(iteration_durations))
        remaining_seconds = rolling_average * remaining_iterations
        completion_time = datetime.now(timezone.utc) + timedelta(
            seconds=remaining_seconds
        )
        emit(
            "in_progress",
            completed_iterations=iteration,
            completed_iterations_this_run=completed_this_run,
            total_iterations=resumed_iteration + config.maximum_iterations,
            remaining_iterations=remaining_iterations,
            objective=objective,
            iteration_seconds=iteration_seconds,
            rolling_average_iteration_seconds=rolling_average,
            estimated_remaining_seconds=remaining_seconds,
            expected_completion_utc=completion_time.isoformat(),
            checkpoint_path=(
                None
                if effective_checkpoint_path is None
                else str(effective_checkpoint_path)
            ),
            seconds_since_checkpoint=(
                None
                if last_checkpoint_time is None
                else max(0.0, now - last_checkpoint_time)
            ),
        )

    status: ReducedODFitStatus = "complete"
    try:
        optimized = minimize(
            evaluate, raw, jac=True, method="L-BFGS-B", callback=callback,
            bounds=scipy_bounds,
            options={"maxiter": config.maximum_iterations, "gtol": config.gradient_tolerance, "ftol": config.function_tolerance},
        )
        latest = np.asarray(optimized.x, dtype=np.float64)
        latest_objective, latest_gradient = evaluate(latest)
        success, message = bool(optimized.success), str(optimized.message)
    except DeadlineReached:
        status, success, message = "deadline", False, "Time budget reached; checkpoint saved."
    except KeyboardInterrupt:
        status, success, message = (
            "interrupted",
            False,
            "Interrupted by user; latest accepted iterate checkpointed.",
        )
    save_checkpoint()
    evaluation = evaluate_demand_model(latest, problem=problem)
    emit(
        status,
        completed_iterations=iteration,
        completed_iterations_this_run=iteration - resumed_iteration,
        total_iterations=resumed_iteration + config.maximum_iterations,
        objective=latest_objective,
    )
    return DemandFitResult(
        latest,
        latest_objective,
        float(evaluation.log_likelihood),
        iteration,
        evaluations,
        success,
        message,
        compile_seconds,
        clock() - optimization_started,
        identity,
        status,
        resumed_iteration,
        float(np.max(np.abs(latest_gradient), initial=0.0)),
        None if effective_checkpoint_path is None else str(effective_checkpoint_path),
        (
            None
            if not iteration_durations
            else float(np.mean(iteration_durations))
        ),
        (
            None
            if not iteration_durations
            else float(np.mean(iteration_durations))
            * max(0, config.maximum_iterations - (iteration - resumed_iteration))
        ),
    )
