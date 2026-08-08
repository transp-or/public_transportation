"""ML and MAP estimation for the minimal reduced gravity model."""

from __future__ import annotations

import time
import resource
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

from .objective import MinimalGravityProblem, evaluate_minimal_gravity_objective
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
from .parameters import transform_minimal_gravity_parameters


ProgressCallback = Callable[[Mapping[str, object]], None]


class _DeadlineReached(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ReducedODPrecisionDiagnostics:
    x64_enabled: bool
    requested_numpy_dtype: str
    actual_jax_input_dtype: str
    compiled_objective_dtype: str
    compiled_gradient_dtype: str
    machine_epsilon: float
    objective_scale: float
    numerical_absolute_resolution: float
    requested_absolute_function_resolution: float
    tolerance_resolvable: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReducedODConvergenceDiagnostics:
    optimizer_success: bool
    optimizer_status: int | None
    optimizer_message: str
    gradient_norm: float
    projected_gradient_norm: float | None
    objective: float
    objective_scale: float
    requested_gradient_tolerance: float
    requested_function_tolerance: float
    active_dtype: str
    tolerance_resolvable: bool
    finite: bool
    stationary: bool
    optimizer_terminated: bool
    numerically_converged: bool
    scientifically_admissible: bool | None
    transformed_parameters_plausible: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReducedODProductionDiagnostics:
    baseline_total: float
    fitted_total: float
    minimum: float
    maximum: float
    median: float
    quantiles: tuple[tuple[float, float], ...]
    zero_count: int
    nonfinite_count: int
    minimum_multiplier: float
    maximum_multiplier: float
    totals_by_basis_group: tuple[tuple[str, float], ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReducedODTransformedParameterDiagnostics:
    beta_time: float
    beta_transfer: float
    dispersion: float | None
    production_coefficients: tuple[float, ...]
    parameter_names: tuple[str, ...]
    parameter_values: tuple[float, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReducedODPostFitIdentificationDiagnostics:
    hessian: np.ndarray
    eigenvalues: np.ndarray
    rank: int
    condition_number: float
    weak_direction_loadings: tuple[tuple[str, float], ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("hessian", "eigenvalues"):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class ReducedODMapDiagnostics:
    negative_log_likelihood: float
    negative_log_prior: float
    total_map_objective: float
    per_parameter_prior_contribution: tuple[float, ...]
    standardized_distance: tuple[float, ...]
    prior_dominated: bool


def _parameter_names(problem: MinimalGravityProblem) -> tuple[str, ...]:
    return problem.parameter_layout.raw_parameter_names(problem.production_basis_labels)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


@dataclass(frozen=True, slots=True)
class ReducedODFitResult:
    manifest: ReducedODFitManifest
    status: ReducedODFitStatus
    success: bool
    message: str
    raw_parameters: np.ndarray
    objective: float
    log_likelihood: float
    log_prior: float
    gradient_norm: float
    iterations: int
    evaluations: int
    compile_seconds: float
    optimization_seconds: float
    resumed_from_iteration: int
    optimizer_success: bool = False
    precision: ReducedODPrecisionDiagnostics | None = None
    convergence: ReducedODConvergenceDiagnostics | None = None
    transformed_parameters: ReducedODTransformedParameterDiagnostics | None = None
    production: ReducedODProductionDiagnostics | None = None
    identification: ReducedODPostFitIdentificationDiagnostics | None = None
    map_diagnostics: ReducedODMapDiagnostics | None = None
    active_bound_parameters: tuple[str, ...] = ()
    peak_rss_bytes: int | None = None
    checkpoint_path: str | None = None

    def __post_init__(self) -> None:
        raw = np.array(self.raw_parameters, dtype=np.float64, copy=True)
        raw.setflags(write=False)
        object.__setattr__(self, "raw_parameters", raw)


def estimate_minimal_gravity(
    *,
    problem: MinimalGravityProblem,
    initial_raw_parameters: object,
    model_fingerprint: str,
    config: ReducedODFitConfig = ReducedODFitConfig(),
    prior: GaussianRawParameterPrior | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    progress: ProgressCallback | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ReducedODFitResult:
    """Fit the compact model without reconstructing OD or running assignment."""
    layout = problem.parameter_layout
    numerical = config.numerical
    if numerical.precision == "float64_required" and not jax.config.x64_enabled:
        raise RuntimeError(
            "Reduced-OD estimation requires JAX float64, but x64 is disabled. "
            "Set JAX_ENABLE_X64=true before starting Python, or call "
            "jax.config.update('jax_enable_x64', True) before constructing JAX "
            "arrays and compiled functions."
        )
    precision_warnings: list[str] = []
    if numerical.precision == "float64_preferred" and not jax.config.x64_enabled:
        precision_warnings.append(
            "JAX x64 is disabled; requested float64 values may be demoted."
        )
    requested_dtype = np.dtype(numerical.requested_dtype)
    basis_warnings: list[str] = []
    if problem.production_basis is not None:
        basis_rank = int(np.linalg.matrix_rank(problem.production_basis))
        if basis_rank != problem.production_basis.shape[1]:
            raise ValueError(
                "production_basis must have full column rank before fitting."
            )
        basis_condition = float(np.linalg.cond(problem.production_basis))
        if basis_condition >= config.identification_condition_threshold:
            basis_warnings.append(
                f"Production basis condition number is {basis_condition:.3g}."
            )
    manifest = ReducedODFitManifest(
        model_fingerprint=model_fingerprint,
        method=config.method,
        parameter_count=layout.size,
    )
    if config.method == "map" and prior is None:
        raise ValueError("MAP estimation requires a prior.")
    if prior is not None and prior.mean.shape != (layout.size,):
        raise ValueError("prior dimension does not match the parameter layout.")
    raw = np.asarray(initial_raw_parameters, dtype=np.float64)
    if raw.shape != (layout.size,) or not np.all(np.isfinite(raw)):
        raise ValueError("initial_raw_parameters has an invalid shape or values.")
    if checkpoint_path is not None and config.checkpoint_path is not None:
        if Path(checkpoint_path) != Path(config.checkpoint_path):
            raise ValueError("checkpoint_path conflicts with the fit configuration.")
    effective_checkpoint_path = checkpoint_path or config.checkpoint_path
    effective_resume = resume or config.resume
    if config.deadline_seconds is not None and effective_checkpoint_path is None:
        raise ValueError("deadline_seconds requires a checkpoint_path.")
    resumed_iteration = 0
    if effective_resume:
        if (
            effective_checkpoint_path is None
            or not Path(effective_checkpoint_path).is_file()
        ):
            raise ValueError("resume requires an existing checkpoint_path.")
        checkpoint = load_reduced_od_checkpoint(
            effective_checkpoint_path, expected_manifest=manifest
        )
        raw = np.asarray(checkpoint.raw_parameters)
        resumed_iteration = checkpoint.iteration

    parameter_names = _parameter_names(problem)
    bounds: ReducedODRawParameterBounds | None = config.raw_parameter_bounds
    if config.named_raw_parameter_bounds is not None:
        bounds = config.named_raw_parameter_bounds.resolve(parameter_names)
    scipy_bounds = None
    if bounds is not None:
        if bounds.lower.shape != (layout.size,):
            raise ValueError("raw parameter bounds do not match the parameter layout.")
        if np.any(raw < bounds.lower) or np.any(raw > bounds.upper):
            raise ValueError("initial_raw_parameters must lie inside raw bounds.")
        scipy_bounds = tuple(
            (float(lower), float(upper))
            for lower, upper in zip(bounds.lower, bounds.upper, strict=True)
        )

    prior_mean = (
        None if prior is None else jnp.asarray(prior.mean, dtype=requested_dtype)
    )
    prior_scale = (
        None if prior is None else jnp.asarray(prior.scale, dtype=requested_dtype)
    )

    def total_objective(value: jax.Array) -> jax.Array:
        result = evaluate_minimal_gravity_objective(value, problem=problem).objective
        if config.method == "map":
            assert prior_mean is not None and prior_scale is not None
            standardized = (value - prior_mean) / prior_scale
            result = result + 0.5 * jnp.sum(standardized * standardized)
        return result

    progress_started = clock()
    last_progress_time = progress_started

    def emit_progress(status: str, **payload: object) -> None:
        if progress is None:
            return
        progress(
            {
                "phase": "reduced_od_fit",
                "status": status,
                "method": config.method,
                "elapsed_seconds": max(0.0, clock() - progress_started),
                "peak_rss_bytes": _peak_rss_bytes(),
                **payload,
            }
        )

    compiled = jax.jit(jax.value_and_grad(total_objective))
    emit_progress("started", operation="compilation")
    compile_start = clock()
    jax_initial = jnp.asarray(raw, dtype=requested_dtype)
    first_value, first_gradient = compiled(jax_initial)
    jax.block_until_ready((first_value, first_gradient))
    compile_seconds = clock() - compile_start
    emit_progress("completed", operation="compilation", compile_seconds=compile_seconds)
    active_dtype = np.dtype(first_value.dtype)
    machine_epsilon = float(np.finfo(active_dtype).eps)
    objective_scale = max(1.0, abs(float(first_value)))
    numerical_resolution = machine_epsilon * objective_scale
    requested_resolution = config.function_tolerance * objective_scale
    tolerance_resolvable = requested_resolution >= 8.0 * numerical_resolution
    if not tolerance_resolvable:
        warning = (
            "Configured function_tolerance cannot be resolved reliably at the "
            f"current objective scale in {active_dtype.name}: requested absolute "
            f"resolution {requested_resolution:.6g}, numerical resolution "
            f"{numerical_resolution:.6g}."
        )
        if numerical.reject_unresolved_function_tolerance:
            raise ValueError(warning)
        precision_warnings.append(warning)
    precision_diagnostics = ReducedODPrecisionDiagnostics(
        x64_enabled=bool(jax.config.x64_enabled),
        requested_numpy_dtype=requested_dtype.name,
        actual_jax_input_dtype=np.dtype(jax_initial.dtype).name,
        compiled_objective_dtype=active_dtype.name,
        compiled_gradient_dtype=np.dtype(first_gradient.dtype).name,
        machine_epsilon=machine_epsilon,
        objective_scale=objective_scale,
        numerical_absolute_resolution=numerical_resolution,
        requested_absolute_function_resolution=requested_resolution,
        tolerance_resolvable=tolerance_resolvable,
        warnings=tuple(precision_warnings),
    )
    start = clock()
    latest = raw.copy()
    latest_objective = float(first_value)
    latest_gradient = np.asarray(first_gradient, dtype=np.float64)
    if not np.isfinite(latest_objective) or not np.all(np.isfinite(latest_gradient)):
        raise FloatingPointError(
            "initial reduced-OD objective or gradient is nonfinite."
        )
    evaluated = raw.copy()
    evaluated_objective = latest_objective
    evaluated_gradient = latest_gradient.copy()
    iterations = resumed_iteration
    evaluations = 1

    def deadline_reached() -> bool:
        return (
            config.deadline_seconds is not None
            and clock() - start >= config.deadline_seconds
        )

    def evaluate(value: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluated, evaluated_objective, evaluated_gradient, evaluations
        candidate = np.asarray(value, dtype=np.float64)
        if np.array_equal(candidate, evaluated):
            return evaluated_objective, evaluated_gradient.copy()
        if deadline_reached():
            raise _DeadlineReached
        objective, gradient = compiled(jnp.asarray(value, dtype=requested_dtype))
        jax.block_until_ready((objective, gradient))
        evaluated = candidate.copy()
        evaluated_objective = float(objective)
        evaluated_gradient = np.asarray(gradient, dtype=np.float64)
        if not np.isfinite(evaluated_objective) or not np.all(
            np.isfinite(evaluated_gradient)
        ):
            raise FloatingPointError(
                "reduced-OD objective or gradient became nonfinite."
            )
        evaluations += 1
        if deadline_reached():
            raise _DeadlineReached
        return evaluated_objective, evaluated_gradient.copy()

    def emit_checkpoint() -> None:
        checkpoint = ReducedODCheckpoint(
            manifest=manifest,
            raw_parameters=latest,
            iteration=iterations,
            objective=latest_objective,
            elapsed_seconds=clock() - start,
        )
        if effective_checkpoint_path is not None:
            save_reduced_od_checkpoint(effective_checkpoint_path, checkpoint)
            emit_progress(
                "completed",
                operation="checkpoint",
                iteration=iterations,
                checkpoint_path=str(effective_checkpoint_path),
            )

    def callback(value: np.ndarray) -> None:
        nonlocal iterations, latest, latest_objective, latest_gradient
        nonlocal last_progress_time
        iterations += 1
        latest = np.asarray(value, dtype=np.float64).copy()
        if not np.array_equal(latest, evaluated):
            raise RuntimeError("optimizer callback did not follow an evaluation.")
        latest_objective = evaluated_objective
        latest_gradient = evaluated_gradient.copy()
        if iterations % config.checkpoint_every_iterations == 0:
            emit_checkpoint()
        now = clock()
        if (
            iterations % config.progress_interval_iterations == 0
            or now - last_progress_time >= config.progress_interval_seconds
        ):
            elapsed = max(0.0, now - start)
            completed = max(0, iterations - resumed_iteration)
            remaining = max(0, config.maximum_iterations - completed)
            progress_gradient = latest_gradient.copy()
            if bounds is not None:
                progress_gradient[
                    (latest <= bounds.lower) & (progress_gradient > 0.0)
                ] = 0.0
                progress_gradient[
                    (latest >= bounds.upper) & (progress_gradient < 0.0)
                ] = 0.0
            emit_progress(
                "in_progress",
                operation="optimization",
                iteration=iterations,
                maximum_iterations=config.maximum_iterations,
                objective=latest_objective,
                gradient_infinity_norm=float(
                    np.max(np.abs(latest_gradient), initial=0.0)
                ),
                projected_gradient_infinity_norm=float(
                    np.max(np.abs(progress_gradient), initial=0.0)
                ),
                predicted_remaining_seconds=(
                    elapsed * remaining / completed if completed else None
                ),
            )
            last_progress_time = now
        if deadline_reached():
            emit_checkpoint()
            raise _DeadlineReached

    status: ReducedODFitStatus
    optimizer_success = False
    optimizer_returned = False
    optimizer_status: int | None = None
    optimizer_message = ""
    emit_progress(
        "started",
        operation="optimization",
        iteration=resumed_iteration,
        maximum_iterations=config.maximum_iterations,
    )
    try:
        optimized = minimize(
            evaluate,
            raw,
            method="L-BFGS-B",
            jac=True,
            bounds=scipy_bounds,
            callback=callback,
            options={
                "maxiter": config.maximum_iterations,
                "gtol": config.gradient_tolerance,
                "ftol": config.function_tolerance,
            },
        )
        optimizer_returned = True
        optimized_raw = np.asarray(optimized.x, dtype=np.float64)
        latest_objective = float(optimized.fun)
        if np.array_equal(optimized_raw, evaluated):
            latest = optimized_raw
            latest_gradient = evaluated_gradient.copy()
        elif np.array_equal(optimized_raw, latest):
            # A terminating line search may leave a rejected trial evaluation
            # after the last accepted callback point.
            latest_objective = float(optimized.fun)
        else:
            # SciPy normally returns its last evaluated point.  Fail closed if
            # a future optimizer version violates that accepted-point contract.
            raise RuntimeError("optimizer returned an unevaluated final point.")
        iterations = resumed_iteration + int(optimized.nit)
        status = "complete" if optimized.success else "failed"
        optimizer_success = bool(optimized.success)
        optimizer_status = int(optimized.status)
        optimizer_message = str(optimized.message)
    except _DeadlineReached:
        status = "deadline"
        optimizer_message = "deadline reached; resumable checkpoint saved"
        emit_checkpoint()
        emit_progress("deadline", operation="optimization", iteration=iterations)
    optimization_seconds = clock() - start
    final_evaluation = evaluate_minimal_gravity_objective(latest, problem=problem)
    log_likelihood = float(final_evaluation.log_likelihood)
    log_prior = 0.0
    map_diagnostics = None
    if config.method == "map":
        assert prior is not None
        standardized = (latest - prior.mean) / prior.scale
        contributions = 0.5 * standardized * standardized
        log_prior = float(-np.sum(contributions))
        negative_log_likelihood = -log_likelihood
        negative_log_prior = -log_prior
        map_diagnostics = ReducedODMapDiagnostics(
            negative_log_likelihood=negative_log_likelihood,
            negative_log_prior=negative_log_prior,
            total_map_objective=negative_log_likelihood + negative_log_prior,
            per_parameter_prior_contribution=tuple(
                float(value) for value in contributions
            ),
            standardized_distance=tuple(float(value) for value in standardized),
            prior_dominated=negative_log_prior > max(1.0, negative_log_likelihood),
        )

    transformed = transform_minimal_gravity_parameters(latest, layout=layout)
    beta_time = float(transformed.beta_time)
    beta_transfer = float(transformed.beta_transfer)
    dispersion = (
        None if transformed.dispersion is None else float(transformed.dispersion)
    )
    production_coefficients = tuple(
        float(value) for value in np.asarray(transformed.production_coefficients)
    )
    transformed_values = (beta_time, beta_transfer) + (
        (() if dispersion is None else (dispersion,)) + production_coefficients
    )
    transformed_warnings: list[str] = []
    if not np.all(np.isfinite(transformed_values)):
        transformed_warnings.append("A transformed parameter is nonfinite.")
    floor = layout.specification.positivity_floor
    if (
        dispersion is not None
        and dispersion <= floor * config.dispersion_floor_warning_factor
    ):
        transformed_warnings.append("Dispersion is near its positivity floor.")
    transformed_diagnostics = ReducedODTransformedParameterDiagnostics(
        beta_time=beta_time,
        beta_transfer=beta_transfer,
        dispersion=dispersion,
        production_coefficients=production_coefficients,
        parameter_names=parameter_names,
        parameter_values=transformed_values,
        warnings=tuple(transformed_warnings),
    )

    baseline = problem.features.baseline_productions
    productions = np.asarray(final_evaluation.productions, dtype=np.float64)
    multipliers = np.divide(
        productions,
        baseline,
        out=np.ones_like(productions),
        where=baseline != 0.0,
    )
    production_warnings: list[str] = []
    multiplier_minimum = float(np.min(multipliers, initial=np.inf))
    multiplier_maximum = float(np.max(multipliers, initial=0.0))
    baseline_total = float(np.sum(baseline))
    fitted_total = float(np.sum(productions))
    if multiplier_minimum <= config.production_multiplier_minimum_warning:
        production_warnings.append("A production multiplier is near zero.")
    if multiplier_maximum >= config.production_multiplier_maximum_warning:
        production_warnings.append("A production multiplier is extremely large.")
    if baseline_total > 0.0:
        total_ratio = fitted_total / baseline_total
        low, high = config.production_total_ratio_warning
        if total_ratio < low or total_ratio > high:
            production_warnings.append(
                "Fitted production total differs implausibly from baseline."
            )
    nonfinite_productions = int(np.count_nonzero(~np.isfinite(productions)))
    if nonfinite_productions:
        production_warnings.append("Fitted productions contain nonfinite values.")
    totals_by_basis_group: tuple[tuple[str, float], ...] = ()
    if problem.production_basis is not None:
        labels = problem.production_basis_labels or tuple(
            f"basis_{index}" for index in range(problem.production_basis.shape[1])
        )
        totals_by_basis_group = tuple(
            (
                label,
                float(np.sum(productions[problem.production_basis[:, index] != 0.0])),
            )
            for index, label in enumerate(labels)
        )
    production_diagnostics = ReducedODProductionDiagnostics(
        baseline_total=baseline_total,
        fitted_total=fitted_total,
        minimum=float(np.min(productions, initial=np.inf)),
        maximum=float(np.max(productions, initial=0.0)),
        median=float(np.median(productions)),
        quantiles=tuple(
            (quantile, float(np.quantile(productions, quantile)))
            for quantile in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        ),
        zero_count=int(np.count_nonzero(productions == 0.0)),
        nonfinite_count=nonfinite_productions,
        minimum_multiplier=multiplier_minimum,
        maximum_multiplier=multiplier_maximum,
        totals_by_basis_group=totals_by_basis_group,
        warnings=tuple(production_warnings),
    )

    def hessian_objective(value: jax.Array) -> jax.Array:
        return total_objective(value)

    emit_progress("started", operation="identification")
    hessian_device = jax.hessian(hessian_objective)(
        jnp.asarray(latest, dtype=requested_dtype)
    )
    jax.block_until_ready(hessian_device)
    hessian = np.asarray(hessian_device, dtype=np.float64)
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    tolerance = config.identification_eigenvalue_tolerance
    rank = int(np.linalg.matrix_rank(hessian, tol=tolerance))
    positive = np.abs(eigenvalues)
    condition = float(
        np.max(positive) / np.min(positive) if np.min(positive) > 0.0 else np.inf
    )
    identification_warnings: list[str] = []
    identification_warnings.extend(basis_warnings)
    if np.min(eigenvalues) < -tolerance:
        identification_warnings.append("Objective curvature is materially negative.")
    if np.min(np.abs(eigenvalues)) <= tolerance:
        identification_warnings.append("Objective curvature is singular or near zero.")
    if condition >= config.identification_condition_threshold:
        identification_warnings.append(f"Hessian condition number is {condition:.3g}.")
    weak_vector = eigenvectors[:, int(np.argmin(np.abs(eigenvalues)))]
    weak_loadings = tuple(
        sorted(
            zip(parameter_names, (float(value) for value in weak_vector), strict=True),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
    )
    identification = ReducedODPostFitIdentificationDiagnostics(
        hessian=hessian,
        eigenvalues=eigenvalues,
        rank=rank,
        condition_number=condition,
        weak_direction_loadings=weak_loadings,
        warnings=tuple(identification_warnings),
    )
    emit_progress(
        "completed",
        operation="identification",
        hessian_rank=rank,
        hessian_condition_number=condition,
    )

    active_bound_parameters: tuple[str, ...] = ()
    projected_gradient = latest_gradient.copy()
    if bounds is not None:
        scale = np.maximum(1.0, np.abs(latest))
        near_lower = np.abs(latest - bounds.lower) <= (
            config.boundary_relative_tolerance * scale
        )
        near_upper = np.abs(latest - bounds.upper) <= (
            config.boundary_relative_tolerance * scale
        )
        projected_gradient[near_lower & (latest_gradient > 0.0)] = 0.0
        projected_gradient[near_upper & (latest_gradient < 0.0)] = 0.0
        active_bound_parameters = tuple(
            parameter_names[index] for index in np.flatnonzero(near_lower | near_upper)
        )
    gradient_norm = float(np.linalg.norm(latest_gradient))
    projected_gradient_norm = float(np.max(np.abs(projected_gradient), initial=0.0))
    finite = bool(
        np.isfinite(latest_objective)
        and np.all(np.isfinite(latest_gradient))
        and np.all(np.isfinite(transformed_values))
    )
    stationary = projected_gradient_norm <= (
        config.gradient_tolerance * config.gradient_convergence_factor
    )
    numerically_converged = bool(
        optimizer_returned and finite and stationary and tolerance_resolvable
    )
    convergence_warnings = list(precision_warnings)
    convergence_warnings.extend(transformed_warnings)
    convergence_warnings.extend(production_warnings)
    convergence_warnings.extend(identification_warnings)
    if optimizer_success and not stationary:
        convergence_warnings.append(
            "Optimizer terminated successfully but the projected gradient exceeds "
            "the requested tolerance."
        )
    convergence = ReducedODConvergenceDiagnostics(
        optimizer_success=optimizer_success,
        optimizer_status=optimizer_status,
        optimizer_message=optimizer_message,
        gradient_norm=gradient_norm,
        projected_gradient_norm=projected_gradient_norm,
        objective=latest_objective,
        objective_scale=max(1.0, abs(latest_objective)),
        requested_gradient_tolerance=config.gradient_tolerance,
        requested_function_tolerance=config.function_tolerance,
        active_dtype=active_dtype.name,
        tolerance_resolvable=tolerance_resolvable,
        finite=finite,
        stationary=stationary,
        optimizer_terminated=optimizer_returned,
        numerically_converged=numerically_converged,
        scientifically_admissible=None,
        transformed_parameters_plausible=not (
            transformed_warnings or production_warnings
        ),
        warnings=tuple(convergence_warnings),
    )
    if status == "complete" and effective_checkpoint_path is not None:
        emit_checkpoint()
    emit_progress(
        "completed" if status == "complete" else status,
        operation="fit",
        iteration=iterations,
        objective=latest_objective,
        numerical_convergence=numerically_converged,
    )
    return ReducedODFitResult(
        manifest=manifest,
        status=status,
        success=numerically_converged,
        message=optimizer_message,
        raw_parameters=latest,
        objective=latest_objective,
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        gradient_norm=gradient_norm,
        iterations=iterations,
        evaluations=evaluations,
        compile_seconds=compile_seconds,
        optimization_seconds=optimization_seconds,
        resumed_from_iteration=resumed_iteration,
        optimizer_success=optimizer_success,
        precision=precision_diagnostics,
        convergence=convergence,
        transformed_parameters=transformed_diagnostics,
        production=production_diagnostics,
        identification=identification,
        map_diagnostics=map_diagnostics,
        active_bound_parameters=active_bound_parameters,
        peak_rss_bytes=_peak_rss_bytes(),
        checkpoint_path=(
            None
            if effective_checkpoint_path is None
            else str(effective_checkpoint_path)
        ),
    )
