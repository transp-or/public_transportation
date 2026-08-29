"""Optional Biogeme TR-BFGS comparison pilot.

This module is deliberately isolated from the production estimator.  Biogeme
is imported only when :func:`run_biogeme_tr_bfgs_pilot` is called, so the
default package has no Biogeme dependency and importing the gravity API stays
cheap.  The pilot accepts the same objective/gradient callback used by the
production estimator and applies the same Dennis--Schnabel convergence audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np

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

    def __post_init__(self) -> None:
        raw = np.array(self.raw_parameters, dtype=np.float64, copy=True)
        gradient = np.array(self.gradient, copy=True)
        raw.setflags(write=False)
        gradient.setflags(write=False)
        object.__setattr__(self, "raw_parameters", raw)
        object.__setattr__(self, "gradient", gradient)


class _BiogemeObjective:
    """Duck-typed Biogeme objective wrapper around a value/gradient callback."""

    def __init__(
        self,
        objective_and_gradient: Callable[[np.ndarray], tuple[object, object]],
    ) -> None:
        self._objective_and_gradient = objective_and_gradient
        self._variables: np.ndarray | None = None
        self.evaluations: list[tuple[np.ndarray, float]] = []

    def set_variables(self, variables: object) -> None:
        self._variables = np.asarray(variables, dtype=np.float64).copy()

    def _evaluate(self) -> tuple[float, np.ndarray]:
        if self._variables is None:
            raise RuntimeError("Biogeme requested an objective before set_variables.")
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
        return objective_value, gradient_value

    def f(self, batch: object | None = None) -> float:
        return self._evaluate()[0]

    def f_g(self, batch: object | None = None) -> SimpleNamespace:
        objective, gradient = self._evaluate()
        return SimpleNamespace(function=objective, gradient=gradient)


def _messages_as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if value is None:
        return {}
    try:
        return {str(key): item for key, item in vars(value).items()}
    except TypeError:
        return {"value": value}


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

    objective = _BiogemeObjective(objective_and_gradient)
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
            # The current Biogeme adapter uses ``tolerance`` for its
            # relative-gradient stop.  Keep the production objective
            # tolerance in the same parameter payload for an auditable pilot;
            # Biogeme versions that do not consume it simply ignore it.
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
    iterations = int(messages.get("Number of iterations", messages.get("iterations", 0)))
    result = GravityBiogemePilotResult(
        optimizer="TR-BFGS",
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
        initial_objective=initial_objective,
        typical_objective_scale_provenance=typical_objective_scale_provenance,
        typical_parameter_scales_provenance=typical_parameter_scales_provenance,
        typical_objective_scale_selection=typical_objective_scale_selection,
    )
    if progress is not None:
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
            )
        )
    return result
