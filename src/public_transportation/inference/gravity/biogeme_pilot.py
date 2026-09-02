"""Optional Biogeme TR-BFGS comparison pilot.

This module is deliberately isolated from the production estimator.  Biogeme
is imported only when :func:`run_biogeme_tr_bfgs_pilot` is called, so the
default package has no Biogeme dependency and importing the gravity API stays
cheap.  The pilot accepts the same objective/gradient callback used by the
production estimator and applies the same Dennis--Schnabel convergence audit.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from .estimator import (
    GravityEstimatorConfig,
    GravityEstimatorProgress,
    _resolve_typical_parameter_scales,
    scaled_gradient_inf_norm,
)


@dataclass(frozen=True, slots=True)
class GravityBiogemePilotResult:
    """Diagnostics returned by the optional TR-BFGS comparison."""

    optimizer: str
    status: str
    success: bool
    message: str
    raw_parameters: np.ndarray
    objective: float
    gradient: np.ndarray
    iterations: int
    elapsed_seconds: float
    gradient_inf_norm: float
    scaled_gradient_inf_norm: float
    scaled_gradient_tolerance: float
    typical_objective_scale: float
    typical_parameter_scales: tuple[float, ...]
    objective_dtype: str
    gradient_dtype: str
    objective_spacing: float
    objective_reduction: float | None
    objective_tolerance_below_precision: bool
    optimizer_messages: dict[str, object] = field(default_factory=dict)
    initial_objective: float | None = None
    typical_objective_scale_provenance: str | None = None
    typical_parameter_scales_provenance: str | None = None
    typical_objective_scale_selection: str | None = None
    optimizer_options: dict[str, object] = field(default_factory=dict)
    optimizer_evaluations: int | None = None

    def __post_init__(self) -> None:
        raw = np.array(self.raw_parameters, dtype=np.float64, copy=True)
        gradient = np.array(self.gradient, copy=True)
        raw.setflags(write=False)
        gradient.setflags(write=False)
        object.__setattr__(self, "raw_parameters", raw)
        object.__setattr__(self, "gradient", gradient)


class _BiogemeObjective:
    """Biogeme ``FunctionToMinimize`` adapter around a value/gradient callback.

    Biogeme's public TR-BFGS entry point is typed against
    ``biogeme_optimization.function.FunctionToMinimize``.  The implementation
    intentionally uses the same protocol without importing Biogeme at module
    import time: the package remains optional, while the adapter is usable with
    the current and older Biogeme releases.  The base class' numerical
    stopping test is reproduced here so the optional pilot can set the same
    Dennis--Schnabel scales as the public estimator.
    """

    def __init__(
        self,
        objective_and_gradient: Callable[[np.ndarray], tuple[object, object]],
        on_evaluation: Callable[[int, float, np.ndarray], None] | None = None,
        *,
        epsilon: float = 1.0e-6,
        typical_objective_scale: float = 1.0,
        typical_parameter_scales: object | None = None,
    ) -> None:
        self._objective_and_gradient = objective_and_gradient
        self._on_evaluation = on_evaluation
        self._variables: np.ndarray | None = None
        self._dimension: int | None = None
        self._cached_key: bytes | None = None
        self._cached_evaluation: tuple[float, np.ndarray] | None = None
        self.epsilon = float(epsilon)
        self.steptol = 1.0e-5
        self.x: np.ndarray | None = None
        self.typf = float(typical_objective_scale)
        self.typx: np.ndarray | None = (
            None
            if typical_parameter_scales is None
            else np.asarray(typical_parameter_scales, dtype=np.float64).copy()
        )
        self.relative_gradient_norm: float | None = None
        self.messages: dict[str, object] = {}
        self.number_of_functions = 0
        self.number_of_gradients = 0
        self.number_of_hessians = 0
        self.evaluations: list[tuple[np.ndarray, float]] = []

    def set_variables(self, variables: object) -> None:
        resolved = np.asarray(variables, dtype=np.float64)
        if resolved.ndim != 1:
            raise ValueError("Biogeme variables must be a one-dimensional vector.")
        if self._dimension is not None and resolved.size != self._dimension:
            raise ValueError("Biogeme variables changed dimension during optimization.")
        self._dimension = int(resolved.size)
        self._variables = resolved.copy()
        self.x = self._variables
        self._cached_key = None
        self._cached_evaluation = None

    def dimension(self) -> int:
        """Return the number of optimization variables expected by Biogeme."""
        if self._dimension is None:
            raise RuntimeError("Biogeme requested the dimension before set_variables.")
        return self._dimension

    def _evaluate(self) -> tuple[float, np.ndarray]:
        if self._variables is None:
            raise RuntimeError("Biogeme requested an objective before set_variables.")
        key = self._variables.tobytes()
        if key == self._cached_key and self._cached_evaluation is not None:
            objective, gradient = self._cached_evaluation
            return objective, gradient.copy()
        objective, gradient = self._objective_and_gradient(self._variables)
        objective_array = np.asarray(objective)
        gradient_array = np.asarray(gradient)
        if objective_array.ndim != 0:
            raise ValueError("objective callback must return a scalar objective.")
        if gradient_array.shape != self._variables.shape:
            raise ValueError("objective callback returned a gradient with the wrong shape.")
        if not np.isfinite(objective_array).all() or not np.all(
            np.isfinite(gradient_array)
        ):
            raise ValueError("objective callback returned non-finite values.")
        objective_value = float(objective_array)
        gradient_value = np.asarray(gradient_array)
        self.evaluations.append((self._variables.copy(), objective_value))
        self._cached_key = key
        self._cached_evaluation = (objective_value, gradient_value.copy())
        if self._on_evaluation is not None:
            try:
                self._on_evaluation(
                    len(self.evaluations), objective_value, gradient_value
                )
            except Exception:
                # Progress is observability only and must never affect a
                # numerical optimizer call.
                pass
        return objective_value, gradient_value

    def f(self, batch: object | None = None) -> float:
        self.number_of_functions += 1
        return self._evaluate()[0]

    def f_g(self, batch: object | None = None) -> SimpleNamespace:
        self.number_of_functions += 1
        self.number_of_gradients += 1
        objective, gradient = self._evaluate()
        return SimpleNamespace(function=objective, gradient=gradient)

    def f_g_h(self, batch: object | None = None) -> SimpleNamespace:
        """Expose the complete protocol; TR-BFGS does not request a Hessian."""
        self.number_of_functions += 1
        self.number_of_gradients += 1
        self.number_of_hessians += 1
        objective, gradient = self._evaluate()
        return SimpleNamespace(function=objective, gradient=gradient, hessian=None)

    def check_optimality(self, bounds: object | None = None) -> bool:
        """Apply Biogeme's scaled-gradient stopping test with our scales."""
        del bounds  # TR-BFGS does not support bound constraints.
        objective, gradient = self._evaluate()
        if self.typx is None:
            self.typx = np.ones(self.dimension(), dtype=np.float64)
        if self.typx.shape != (self.dimension(),):
            raise ValueError("typical_parameter_scales must match the dimension.")
        self.relative_gradient_norm = scaled_gradient_inf_norm(
            self._variables,
            gradient,
            objective,
            typical_objective_scale=self.typf,
            typical_parameter_scales=self.typx,
        )
        if self.relative_gradient_norm <= self.epsilon:
            self.messages["Relative gradient"] = self.relative_gradient_norm
            self.messages["Cause of termination"] = (
                f"Relative gradient = {self.relative_gradient_norm:.2g} "
                f"<= {self.epsilon:.2g}"
            )
            self.messages["Number of function evaluations"] = (
                self.number_of_functions
            )
            self.messages["Number of gradient evaluations"] = (
                self.number_of_gradients
            )
            self.messages["Number of hessian evaluations"] = (
                self.number_of_hessians
            )
            return True
        return False

    def nbr_function_evaluations(self) -> int:
        return self.number_of_functions

    def nbr_gradient_evaluations(self) -> int:
        return self.number_of_gradients

    def nbr_hessian_evaluations(self) -> int:
        return self.number_of_hessians


def _messages_as_dict(value: object) -> dict[str, object]:
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


def run_biogeme_tr_bfgs_pilot(
    *,
    objective_and_gradient: Callable[[np.ndarray], tuple[object, object]],
    initial_raw_parameters: object,
    config: GravityEstimatorConfig = GravityEstimatorConfig(),
    variable_names: Sequence[str] | None = None,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    progress: Callable[[GravityEstimatorProgress], None] | None = None,
) -> GravityBiogemePilotResult:
    """Run Biogeme's optional trust-region BFGS algorithm on one objective.

    ``objective_and_gradient`` must return ``(objective, gradient)`` for the
    supplied NumPy parameter vector.  No model or data are recreated here: a
    caller should close over the exact objective, initial vector, data, and
    scientific configuration used by the production estimator.  The returned
    status is certified with the same scaled-gradient rule as
    :func:`estimate_gravity_model`.

    Biogeme is an optional dependency.  Calling this function without the
    optional package installed raises an actionable :class:`ImportError`.
    """
    try:
        from biogeme.optimization import bfgs_trust_region_for_biogeme
    except ImportError as error:  # pragma: no cover - exercised without extra
        raise ImportError(
            "The Biogeme TR-BFGS pilot requires the optional 'biogeme' "
            "and 'biogeme-optimization' packages."
        ) from error

    initial = np.asarray(initial_raw_parameters, dtype=np.float64)
    if initial.ndim != 1 or not np.all(np.isfinite(initial)):
        raise ValueError("initial_raw_parameters must be a finite one-dimensional vector.")
    parameter_scales = _resolve_typical_parameter_scales(
        initial.size, config.typical_parameter_scales
    )
    names = list(variable_names or (f"parameter_{index}" for index in range(initial.size)))
    if len(names) != initial.size:
        raise ValueError("variable_names must have one entry per parameter.")
    optimizer_bounds = list(bounds or ((None, None) for _ in range(initial.size)))
    if len(optimizer_bounds) != initial.size:
        raise ValueError("bounds must have one entry per parameter.")

    objective = _BiogemeObjective(
        objective_and_gradient,
        epsilon=config.gradient_tolerance,
        typical_objective_scale=config.typical_objective_scale,
        typical_parameter_scales=parameter_scales,
    )
    objective.set_variables(initial)
    initial_objective, _ = objective._evaluate()
    typical_objective_scale_provenance = (
        "configured fixed lower bound; verify against the initial objective"
    )
    typical_parameter_scales_provenance = (
        "generic default unit scales"
        if config.typical_parameter_scales is None
        else (
            "configured scalar expanded to every parameter"
            if np.isscalar(config.typical_parameter_scales)
            else "configured per-parameter vector"
        )
    )
    typical_objective_scale_selection = (
        "fixed case-specific typf; recommended rule is "
        "max(abs(initial_objective), objective_floor)"
    )
    started = perf_counter()
    optimization_result = bfgs_trust_region_for_biogeme(
        objective,
        initial,
        optimizer_bounds,
        names,
        {
            "maxiter": config.maximum_iterations,
            "tolerance": config.gradient_tolerance,
            # The objective adapter sets Biogeme's relative-gradient epsilon
            # explicitly. Keep both public tolerances in the parameter payload
            # for an auditable pilot; current Biogeme TR-BFGS consumes maxiter
            # but does not forward these tolerance entries itself.
            "objective_tolerance": config.objective_tolerance,
        },
    )
    elapsed = perf_counter() - started

    if hasattr(optimization_result, "solution"):
        solution = np.asarray(optimization_result.solution, dtype=np.float64)
        messages = _messages_as_dict(getattr(optimization_result, "messages", None))
        optimizer_success = bool(getattr(optimization_result, "convergence", False))
    else:
        try:
            solution = np.asarray(optimization_result[0], dtype=np.float64)
            messages = _messages_as_dict(optimization_result[1])
        except (IndexError, TypeError, ValueError) as error:
            raise RuntimeError("Biogeme TR-BFGS returned an unsupported result.") from error
        optimizer_success = bool(messages.get("convergence", False))
    if solution.shape != initial.shape or not np.all(np.isfinite(solution)):
        raise RuntimeError("Biogeme TR-BFGS returned invalid parameters.")

    final_objective, final_gradient = objective_and_gradient(solution)
    objective_array = np.asarray(final_objective)
    gradient_array = np.asarray(final_gradient)
    if objective_array.ndim != 0 or gradient_array.shape != solution.shape:
        raise ValueError("objective callback returned invalid final values.")
    objective_value = float(objective_array)
    gradient_inf_norm = float(np.max(np.abs(gradient_array), initial=0.0))
    scaled_gradient = scaled_gradient_inf_norm(
        solution,
        gradient_array,
        objective_value,
        typical_objective_scale=config.typical_objective_scale,
        typical_parameter_scales=parameter_scales,
    )
    success = optimizer_success and scaled_gradient <= config.scaled_gradient_tolerance
    status = "converged" if success else "iteration_limit"
    termination_message = messages.get("Cause of termination", messages.get("message", ""))
    message = str(termination_message)
    if not message:
        message = "Biogeme TR-BFGS did not provide a termination message."
    objective_spacing = float(np.spacing(objective_array.dtype.type(objective_value)))
    reductions = [item[1] for item in objective.evaluations]
    objective_reduction = None if len(reductions) < 2 else reductions[-2] - reductions[-1]
    iterations = _biogeme_iteration_count(
        messages, configured_limit=config.maximum_iterations
    )
    result = GravityBiogemePilotResult(
        optimizer="biogeme_tr_bfgs",
        status=status,
        success=success,
        message=message,
        raw_parameters=solution,
        objective=objective_value,
        gradient=gradient_array,
        iterations=iterations,
        elapsed_seconds=elapsed,
        gradient_inf_norm=gradient_inf_norm,
        scaled_gradient_inf_norm=scaled_gradient,
        scaled_gradient_tolerance=config.scaled_gradient_tolerance,
        typical_objective_scale=config.typical_objective_scale,
        typical_parameter_scales=tuple(float(value) for value in parameter_scales),
        objective_dtype=str(objective_array.dtype),
        gradient_dtype=str(gradient_array.dtype),
        objective_spacing=objective_spacing,
        objective_reduction=objective_reduction,
        objective_tolerance_below_precision=config.objective_tolerance
        < abs(objective_spacing),
        optimizer_messages=messages,
        optimizer_options={
            "maxiter": config.maximum_iterations,
            "tolerance": config.gradient_tolerance,
            "objective_tolerance": config.objective_tolerance,
        },
        optimizer_evaluations=len(objective.evaluations),
        initial_objective=initial_objective,
        typical_objective_scale_provenance=typical_objective_scale_provenance,
        typical_parameter_scales_provenance=typical_parameter_scales_provenance,
        typical_objective_scale_selection=typical_objective_scale_selection,
    )
    if progress is not None:
        try:
            progress(
                GravityEstimatorProgress(
                    iteration=iterations,
                    objective=result.objective,
                    gradient_inf_norm=result.gradient_inf_norm,
                    elapsed_seconds=result.elapsed_seconds,
                    checkpoint_written=False,
                    scaled_gradient_inf_norm=result.scaled_gradient_inf_norm,
                    scaled_gradient_tolerance=result.scaled_gradient_tolerance,
                    typical_objective_scale=result.typical_objective_scale,
                    typical_parameter_scales=result.typical_parameter_scales,
                    initial_objective=result.initial_objective,
                    typical_objective_scale_provenance=(
                        result.typical_objective_scale_provenance
                    ),
                    typical_parameter_scales_provenance=(
                        result.typical_parameter_scales_provenance
                    ),
                    typical_objective_scale_selection=(
                        result.typical_objective_scale_selection
                    ),
                    status=result.status,
                    termination_message=result.message,
                    completed_units=iterations,
                    total_units=config.maximum_iterations,
                    optimizer="biogeme_tr_bfgs",
                )
            )
        except OSError:
            # The optional pilot must retain its numerical result when a
            # progress sink cannot be written.
            pass
    return result


@dataclass(frozen=True, slots=True)
class GravityOptimizerRunSummary:
    """Comparable diagnostics for one optimizer in a controlled pilot."""

    optimizer: str
    status: str
    success: bool
    message: str
    raw_parameters: np.ndarray
    objective: float
    gradient_inf_norm: float
    scaled_gradient_inf_norm: float
    scaled_gradient_tolerance: float
    iterations: int
    evaluations: int
    elapsed_optimizer_seconds: float
    objective_dtype: str
    gradient_dtype: str
    objective_spacing: float
    objective_reduction: float | None
    objective_tolerance_below_precision: bool
    optimizer_options: dict[str, object]
    model_fingerprint: str | None = None
    operator_fingerprint: str | None = None
    elapsed_total_seconds: float | None = None
    initial_objective: float | None = None
    operator_activation_seconds: float = 0.0

    def __post_init__(self) -> None:
        parameters = np.array(self.raw_parameters, dtype=np.float64, copy=True)
        parameters.setflags(write=False)
        object.__setattr__(self, "raw_parameters", parameters)


@dataclass(frozen=True, slots=True)
class GravityOptimizerComparisonResult:
    """Result of running SciPy and Biogeme independently from one start."""

    scipy: GravityOptimizerRunSummary
    biogeme_tr_bfgs: GravityOptimizerRunSummary
    final_parameter_distance: float
    final_objective_difference: float
    same_initial_parameters: bool
    scipy_checkpoint_path: str | None = None
    biogeme_checkpoint_path: str | None = None
    scipy_result_path: str | None = None
    biogeme_result_path: str | None = None
    model_fingerprint: str | None = None
    operator_fingerprint: str | None = None
    operator_activation_seconds: float = 0.0


def _write_comparison_record(path: Path, summary: GravityOptimizerRunSummary) -> None:
    """Write a small, independent comparison record atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "optimizer": summary.optimizer,
        "status": summary.status,
        "success": summary.success,
        "message": summary.message,
        "raw_parameters": summary.raw_parameters.tolist(),
        "objective": summary.objective,
        "gradient_inf_norm": summary.gradient_inf_norm,
        "scaled_gradient_inf_norm": summary.scaled_gradient_inf_norm,
        "scaled_gradient_tolerance": summary.scaled_gradient_tolerance,
        "iterations": summary.iterations,
        "evaluations": summary.evaluations,
        "elapsed_optimizer_seconds": summary.elapsed_optimizer_seconds,
        "elapsed_total_seconds": summary.elapsed_total_seconds,
        "operator_activation_seconds": summary.operator_activation_seconds,
        "initial_objective": summary.initial_objective,
        "objective_dtype": summary.objective_dtype,
        "gradient_dtype": summary.gradient_dtype,
        "objective_spacing": summary.objective_spacing,
        "objective_reduction": summary.objective_reduction,
        "objective_tolerance_below_precision": summary.objective_tolerance_below_precision,
        "optimizer_options": summary.optimizer_options,
        "model_fingerprint": summary.model_fingerprint,
        "operator_fingerprint": summary.operator_fingerprint,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _summary_from_biogeme(
    result: GravityBiogemePilotResult,
    *,
    model_fingerprint: str | None,
    operator_fingerprint: str | None,
    operator_activation_seconds: float = 0.0,
) -> GravityOptimizerRunSummary:
    return GravityOptimizerRunSummary(
        optimizer="biogeme_tr_bfgs",
        status=result.status,
        success=result.success,
        message=result.message,
        raw_parameters=result.raw_parameters,
        objective=result.objective,
        gradient_inf_norm=result.gradient_inf_norm,
        scaled_gradient_inf_norm=result.scaled_gradient_inf_norm,
        scaled_gradient_tolerance=result.scaled_gradient_tolerance,
        iterations=result.iterations,
        evaluations=(
            len(result.optimizer_messages)
            if result.optimizer_evaluations is None
            else result.optimizer_evaluations
        ),
        elapsed_optimizer_seconds=result.elapsed_seconds,
        objective_dtype=result.objective_dtype,
        gradient_dtype=result.gradient_dtype,
        objective_spacing=result.objective_spacing,
        objective_reduction=result.objective_reduction,
        objective_tolerance_below_precision=result.objective_tolerance_below_precision,
        optimizer_options=result.optimizer_options,
        model_fingerprint=model_fingerprint,
        operator_fingerprint=operator_fingerprint,
        elapsed_total_seconds=result.elapsed_seconds + operator_activation_seconds,
        initial_objective=result.initial_objective,
        operator_activation_seconds=operator_activation_seconds,
    )


def compare_gravity_optimizers(
    *,
    objective_and_gradient: Callable[[np.ndarray], tuple[object, object]],
    initial_raw_parameters: object,
    config: GravityEstimatorConfig = GravityEstimatorConfig(),
    variable_names: Sequence[str] | None = None,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    scipy_checkpoint_path: Path | None = None,
    biogeme_checkpoint_path: Path | None = None,
    scipy_result_path: Path | None = None,
    biogeme_result_path: Path | None = None,
    model_fingerprint: str | None = None,
    operator_fingerprint: str | None = None,
    operator_activation_seconds: float = 0.0,
) -> GravityOptimizerComparisonResult:
    """Compare both optimizers from an identical callback and initial point.

    This is a diagnostic utility, not a replacement for the checkpointed
    production estimator.  Each optimizer receives an independent copy of the
    same initial vector and callback.  Optional checkpoint/result paths are
    deliberately separate; records contain optimizer metadata and never allow
    one algorithm's state to be resumed as the other algorithm.
    """
    initial = np.asarray(initial_raw_parameters, dtype=np.float64)
    if initial.ndim != 1 or not np.all(np.isfinite(initial)):
        raise ValueError("initial_raw_parameters must be a finite one-dimensional vector.")
    if not np.isfinite(operator_activation_seconds) or operator_activation_seconds < 0.0:
        raise ValueError("operator_activation_seconds must be finite and nonnegative.")
    scales = _resolve_typical_parameter_scales(initial.size, config.typical_parameter_scales)
    names = list(variable_names or (f"parameter_{index}" for index in range(initial.size)))
    if len(names) != initial.size:
        raise ValueError("variable_names must have one entry per parameter.")
    optimizer_bounds = list(bounds or ((None, None) for _ in range(initial.size)))
    if len(optimizer_bounds) != initial.size:
        raise ValueError("bounds must have one entry per parameter.")

    scipy_evaluations: list[tuple[np.ndarray, float, np.ndarray]] = []
    scipy_accepted: list[float] = []
    scipy_objective_dtype = "float64"
    scipy_gradient_dtype = "float64"

    def scipy_callback(value: np.ndarray) -> None:
        candidate = np.asarray(value, dtype=np.float64).copy()
        for item in reversed(scipy_evaluations):
            if np.array_equal(item[0], candidate):
                scipy_accepted.append(item[1])
                break

    def scipy_objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal scipy_objective_dtype, scipy_gradient_dtype
        candidate = np.asarray(value, dtype=np.float64).copy()
        objective, gradient = objective_and_gradient(candidate)
        objective_array = np.asarray(objective)
        gradient_array = np.asarray(gradient)
        if objective_array.ndim != 0 or gradient_array.shape != candidate.shape:
            raise ValueError("objective callback returned invalid values.")
        objective_value = float(objective_array)
        gradient_value = np.asarray(gradient_array, dtype=np.float64)
        scipy_objective_dtype = str(objective_array.dtype)
        scipy_gradient_dtype = str(gradient_array.dtype)
        if not np.isfinite(objective_value) or not np.all(np.isfinite(gradient_value)):
            raise ValueError("objective callback returned non-finite values.")
        scipy_evaluations.append((candidate, objective_value, gradient_value))
        return objective_value, gradient_value

    scipy_started = perf_counter()
    scipy_objective(initial.copy())
    scipy_options = {
        "maxiter": config.maximum_iterations,
        "gtol": config.gradient_tolerance,
        "ftol": config.objective_tolerance,
        "maxls": config.optimizer_maxls,
    }
    scipy_result = minimize(
        scipy_objective,
        initial.copy(),
        method="L-BFGS-B",
        jac=True,
        callback=scipy_callback,
        bounds=optimizer_bounds,
        options=scipy_options,
    )
    scipy_elapsed = perf_counter() - scipy_started
    scipy_solution = np.asarray(scipy_result.x, dtype=np.float64)
    scipy_objective_value, scipy_gradient = scipy_objective(scipy_solution)
    scipy_scaled = scaled_gradient_inf_norm(
        scipy_solution,
        scipy_gradient,
        scipy_objective_value,
        typical_objective_scale=config.typical_objective_scale,
        typical_parameter_scales=scales,
    )
    scipy_success = bool(scipy_result.success) and scipy_scaled <= config.scaled_gradient_tolerance
    scipy_summary = GravityOptimizerRunSummary(
        optimizer="scipy",
        status="converged" if scipy_success else "iteration_limit",
        success=scipy_success,
        message=str(scipy_result.message),
        raw_parameters=scipy_solution,
        objective=scipy_objective_value,
        gradient_inf_norm=float(np.max(np.abs(scipy_gradient), initial=0.0)),
        scaled_gradient_inf_norm=scipy_scaled,
        scaled_gradient_tolerance=config.scaled_gradient_tolerance,
        iterations=int(getattr(scipy_result, "nit", 0) or 0),
        evaluations=len(scipy_evaluations),
        elapsed_optimizer_seconds=scipy_elapsed,
        objective_dtype=scipy_objective_dtype,
        gradient_dtype=scipy_gradient_dtype,
        objective_spacing=float(
            np.spacing(np.dtype(scipy_objective_dtype).type(scipy_objective_value))
        ),
        objective_reduction=(
            None if len(scipy_accepted) < 2 else scipy_accepted[-2] - scipy_accepted[-1]
        ),
        objective_tolerance_below_precision=(
            config.objective_tolerance
            < abs(
                float(
                    np.spacing(
                        np.dtype(scipy_objective_dtype).type(scipy_objective_value)
                    )
                )
            )
        ),
        optimizer_options=scipy_options,
        model_fingerprint=model_fingerprint,
        operator_fingerprint=operator_fingerprint,
        elapsed_total_seconds=scipy_elapsed + operator_activation_seconds,
        initial_objective=float(scipy_evaluations[0][1]),
        operator_activation_seconds=operator_activation_seconds,
    )
    if scipy_checkpoint_path is not None:
        _write_comparison_record(Path(scipy_checkpoint_path), scipy_summary)
    if scipy_result_path is not None:
        _write_comparison_record(Path(scipy_result_path), scipy_summary)

    biogeme_pilot = run_biogeme_tr_bfgs_pilot(
        objective_and_gradient=objective_and_gradient,
        initial_raw_parameters=initial.copy(),
        config=config,
        variable_names=names,
        bounds=optimizer_bounds,
    )
    biogeme_summary = _summary_from_biogeme(
        biogeme_pilot,
        model_fingerprint=model_fingerprint,
        operator_fingerprint=operator_fingerprint,
        operator_activation_seconds=operator_activation_seconds,
    )
    if biogeme_checkpoint_path is not None:
        _write_comparison_record(Path(biogeme_checkpoint_path), biogeme_summary)
    if biogeme_result_path is not None:
        _write_comparison_record(Path(biogeme_result_path), biogeme_summary)

    return GravityOptimizerComparisonResult(
        scipy=scipy_summary,
        biogeme_tr_bfgs=biogeme_summary,
        final_parameter_distance=float(
            np.linalg.norm(scipy_summary.raw_parameters - biogeme_summary.raw_parameters)
        ),
        final_objective_difference=float(
            scipy_summary.objective - biogeme_summary.objective
        ),
        same_initial_parameters=True,
        scipy_checkpoint_path=(
            None if scipy_checkpoint_path is None else str(scipy_checkpoint_path)
        ),
        biogeme_checkpoint_path=(
            None if biogeme_checkpoint_path is None else str(biogeme_checkpoint_path)
        ),
        scipy_result_path=None if scipy_result_path is None else str(scipy_result_path),
        biogeme_result_path=None if biogeme_result_path is None else str(biogeme_result_path),
        model_fingerprint=model_fingerprint,
        operator_fingerprint=operator_fingerprint,
        operator_activation_seconds=operator_activation_seconds,
    )
