from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from ..common.model_blackbox import Array, LogLikFn, LogPriorFn, _as_scalar
from .config import MLConfig
from .results import MLResult


@dataclass(frozen=True, slots=True)
class MLCompilationMetrics:
    tracing_seconds: float
    lowering_seconds: float
    compilation_seconds: float
    executable_loading_seconds: float
    first_execution_seconds: float
    lowered_text_bytes: int | None


@dataclass(frozen=True, slots=True)
class PreparedMLObjective:
    """Stable value/diagnostics/gradient kernel with dynamic data arguments."""

    jitted: Any
    theta_example: Array
    data: Any


@dataclass(frozen=True, slots=True)
class CompiledMLObjective:
    """Ahead-of-time compiled objective and its separated phase timings."""

    callable: Any
    metrics: MLCompilationMetrics


def prepare_ml_objective(
    *,
    theta_example: Array,
    data: Any,
    loglik: LogLikFn,
    logprior: LogPriorFn | None = None,
    prior_weight: float = 0.0,
) -> PreparedMLObjective:
    """Prepare a stable objective whose numerical data remain dynamic arguments."""
    if prior_weight < 0:
        raise ValueError("prior_weight must be non-negative.")
    lp_fun = _zero_logprior if logprior is None else logprior
    weight = float(prior_weight)

    def objective(theta: Array, dynamic_data: Any):
        ll = _as_scalar(loglik(theta, dynamic_data))
        lp = _as_scalar(lp_fun(theta))
        return -(ll + weight * lp), (ll, lp)

    jitted = jax.jit(jax.value_and_grad(objective, argnums=0, has_aux=True))
    return PreparedMLObjective(jitted, jnp.asarray(theta_example), data)


def compile_ml_objective(
    prepared: PreparedMLObjective, *, execute_first: bool = True
) -> CompiledMLObjective:
    """Trace, lower, compile, and optionally execute while timing each phase."""
    started = perf_counter()
    traced = prepared.jitted.trace(prepared.theta_example, prepared.data)
    tracing_seconds = perf_counter() - started
    started = perf_counter()
    lowered = traced.lower()
    lowering_seconds = perf_counter() - started
    try:
        lowered_text_bytes = len(lowered.as_text().encode("utf-8"))
    except (AttributeError, TypeError, ValueError):
        lowered_text_bytes = None
    started = perf_counter()
    compiled_callable = lowered.compile()
    compilation_seconds = perf_counter() - started
    started = perf_counter()
    compiled_callable.runtime_executable()
    executable_loading_seconds = perf_counter() - started
    first_execution_seconds = 0.0
    if execute_first:
        started = perf_counter()
        first = compiled_callable(prepared.theta_example, prepared.data)
        jax.block_until_ready(first)
        first_execution_seconds = perf_counter() - started
    return CompiledMLObjective(
        compiled_callable,
        MLCompilationMetrics(
            tracing_seconds,
            lowering_seconds,
            compilation_seconds,
            executable_loading_seconds,
            first_execution_seconds,
            lowered_text_bytes,
        ),
    )


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string."""
    return str(timedelta(seconds=int(round(max(seconds, 0.0)))))


def _zero_logprior(theta: Array) -> Array:
    """Default log-prior: no prior penalty."""
    return jnp.asarray(0.0, dtype=theta.dtype)


def make_ml_objective(
    *,
    loglik: LogLikFn,
    logprior: LogPriorFn | None,
    data: Any,
    prior_weight: float = 0.0,
) -> Callable[[Array], Array]:
    """
    Build the scalar minimized objective.

    The same `logprior` used by Bayesian estimation is reused here. The ML
    objective is

        -loglik(theta, data) - prior_weight * logprior(theta).

    Therefore:
      - prior_weight = 0 gives pure maximum likelihood;
      - prior_weight = 1 gives the usual MAP / penalized-ML objective induced
        by the same prior.

    Additive constants in logprior do not affect the optimizer.

    :param loglik: JAX-compatible log-likelihood.
    :param logprior: JAX-compatible absolute log-prior. If None, no penalty is used.
    :param data: Data passed to loglik.
    :param prior_weight: Non-negative scaling of the prior penalty.
    :return: JAX-compatible objective theta -> scalar.
    """
    if prior_weight < 0:
        raise ValueError("prior_weight must be non-negative.")
    objective_with_aux = _make_ml_objective_with_aux(
        loglik=loglik,
        logprior=logprior,
        data=data,
        prior_weight=prior_weight,
    )

    def objective(theta: Array) -> Array:
        value, _ = objective_with_aux(theta)
        return value

    return objective


def _make_ml_objective_with_aux(
    *,
    loglik: LogLikFn,
    logprior: LogPriorFn | None,
    data: Any,
    prior_weight: float,
) -> Callable[[Array], tuple[Array, tuple[Array, Array]]]:
    """Build the objective together with likelihood and prior diagnostics."""
    if prior_weight < 0:
        raise ValueError("prior_weight must be non-negative.")
    lp_fun = _zero_logprior if logprior is None else logprior

    def objective(theta: Array) -> tuple[Array, tuple[Array, Array]]:
        ll = _as_scalar(loglik(theta, data))
        lp = _as_scalar(lp_fun(theta))
        return -(ll + prior_weight * lp), (ll, lp)

    return objective


@dataclass(frozen=True)
class _CachedEvaluation:
    """One compiled objective/gradient evaluation at an exact parameter vector."""

    theta: np.ndarray
    objective: float
    gradient: np.ndarray
    loglikelihood: float
    logprior: float


def _extract_inverse_hessian(scipy_result: Any, dim: int) -> np.ndarray | None:
    """Extract an inverse-Hessian approximation from scipy.optimize results."""
    hess_inv = getattr(scipy_result, "hess_inv", None)
    if hess_inv is None:
        return None
    try:
        if hasattr(hess_inv, "todense"):
            arr = np.asarray(hess_inv.todense(), dtype=float)
        else:
            arr = np.asarray(hess_inv, dtype=float)
        if arr.shape == (dim, dim):
            return arr
    except Exception:
        return None
    return None


def _safe_inverse(matrix: np.ndarray) -> np.ndarray | None:
    """Invert a matrix, returning None if it is singular or invalid."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.all(np.isfinite(matrix)):
        return None
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        try:
            return np.linalg.pinv(matrix)
        except np.linalg.LinAlgError:
            return None


def _standard_errors(covariance: np.ndarray | None) -> np.ndarray | None:
    """Compute standard errors from a covariance matrix."""
    if covariance is None:
        return None
    diag = np.diag(covariance)
    if np.any(diag < 0):
        return np.full_like(diag, np.nan, dtype=float)
    return np.sqrt(diag)


def run_ml(
    *,
    dim: int,
    data: Any,
    loglik: LogLikFn,
    logprior: LogPriorFn | None = None,
    theta0: Array | np.ndarray | None = None,
    config: MLConfig | None = None,
    method: str | None = None,
    maxiter: int | None = None,
    gtol: float | None = None,
    prior_weight: float | None = None,
    compute_hessian: bool | None = None,
    compiled_objective: CompiledMLObjective | Any | None = None,
    logger: Any | None = None,
) -> MLResult:
    """
    Estimate parameters by maximum likelihood or penalized maximum likelihood.

    The statistical model is the same as for Bayesian estimation: the user
    supplies a black-box log-likelihood and, optionally, the same log-prior.
    The difference is only the estimation engine.

    Parameters
    ----------
    dim:
        Dimension of theta.

    data:
        Data passed to loglik(theta, data).

    loglik:
        JAX-compatible differentiable log-likelihood.

    logprior:
        JAX-compatible absolute log-prior. Ignored if prior_weight=0.

    theta0:
        Initial parameter vector. Defaults to zeros.

    config:
        Optional MLConfig. Explicit keyword arguments override it.

    prior_weight:
        Zero for pure ML. One for MAP/penalized ML corresponding to the
        Bayesian prior. Other non-negative values scale the prior penalty.

    Returns
    -------
    MLResult
        Optimization result and inference diagnostics.
    """
    if dim < 0:
        raise ValueError("dim must be non-negative.")

    cfg = MLConfig() if config is None else config
    cfg.validate()

    method = cfg.method if method is None else method
    maxiter = cfg.maxiter if maxiter is None else int(maxiter)
    gtol = cfg.gtol if gtol is None else float(gtol)
    prior_weight = cfg.prior_weight if prior_weight is None else float(prior_weight)
    compute_hessian = (
        cfg.compute_hessian if compute_hessian is None else bool(compute_hessian)
    )

    if maxiter <= 0:
        raise ValueError("maxiter must be positive.")
    if gtol <= 0:
        raise ValueError("gtol must be positive.")
    if prior_weight < 0:
        raise ValueError("prior_weight must be non-negative.")

    if dim == 0:
        empty = jnp.empty((0,), dtype=float)
        ll = float(np.asarray(_as_scalar(loglik(empty, data))))
        lp = 0.0 if logprior is None else float(np.asarray(_as_scalar(logprior(empty))))
        objective_value = -(ll + prior_weight * lp)
        return MLResult(
            dim=0,
            theta_hat=np.empty((0,), dtype=float),
            objective_value=float(objective_value),
            loglikelihood=ll,
            logprior=lp,
            prior_weight=float(prior_weight),
            gradient=np.empty((0,), dtype=float),
            gradient_norm=0.0,
            hessian=(np.empty((0, 0), dtype=float) if compute_hessian else None),
            covariance_matrix=(
                np.empty((0, 0), dtype=float) if compute_hessian else None
            ),
            standard_errors=(np.empty((0,), dtype=float) if compute_hessian else None),
            z_values=(np.empty((0,), dtype=float) if compute_hessian else None),
            success=True,
            message="No estimable parameters; optimizer bypassed.",
            method=str(method),
            num_iterations=0,
            num_function_evaluations=1,
            num_gradient_evaluations=0,
            runtime_seconds=0.0,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            optimization_trace=np.empty((0, 3), dtype=float),
            num_compiled_evaluations=0,
            scipy_result=None,
        )

    if theta0 is None:
        theta0_np = np.zeros(dim, dtype=float)
    else:
        theta0_np = np.asarray(theta0, dtype=float).reshape((dim,))

    if isinstance(compiled_objective, CompiledMLObjective):
        value_and_grad = compiled_objective.callable
    elif compiled_objective is not None:
        value_and_grad = compiled_objective
    else:
        value_and_grad = prepare_ml_objective(
            theta_example=jnp.asarray(theta0_np),
            data=data,
            loglik=loglik,
            logprior=logprior,
            prior_weight=prior_weight,
        ).jitted
    objective = make_ml_objective(
        loglik=loglik,
        logprior=logprior,
        data=data,
        prior_weight=prior_weight,
    )

    trace: list[tuple[int, float, float]] = []
    start_time = perf_counter()
    cached_evaluation: _CachedEvaluation | None = None
    num_compiled_evaluations = 0

    def scipy_fun(theta_np: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal cached_evaluation, num_compiled_evaluations
        theta = jnp.asarray(theta_np)
        (value, (ll, lp)), grad = value_and_grad(theta, data)
        value_np = float(np.asarray(value))
        grad_np = np.array(grad, dtype=float, copy=True)
        cached_evaluation = _CachedEvaluation(
            theta=np.array(theta_np, dtype=float, copy=True),
            objective=value_np,
            gradient=grad_np,
            loglikelihood=float(np.asarray(ll)),
            logprior=float(np.asarray(lp)),
        )
        num_compiled_evaluations += 1
        return value_np, grad_np

    def evaluation_at(theta_np: np.ndarray) -> _CachedEvaluation:
        nonlocal cached_evaluation
        candidate = np.asarray(theta_np, dtype=float)
        if cached_evaluation is None or not np.array_equal(
            candidate, cached_evaluation.theta
        ):
            scipy_fun(candidate)
        assert cached_evaluation is not None
        return cached_evaluation

    iteration = {"k": 0}

    def callback(theta_np: np.ndarray) -> None:
        evaluation = evaluation_at(theta_np)
        grad_norm = float(np.linalg.norm(evaluation.gradient))
        trace.append((iteration["k"], evaluation.objective, grad_norm))

        if logger is not None and iteration["k"] % cfg.log_every == 0:
            now = perf_counter()
            logger.info(
                "ML iteration %d — objective: %.6f — gradient norm: %.6g — elapsed: %s",
                iteration["k"],
                evaluation.objective,
                grad_norm,
                _format_duration(now - start_time),
            )
            nonlocal_last = now  # makes the logging intent explicit for linters
            _ = nonlocal_last

        iteration["k"] += 1

    options = {"maxiter": maxiter, "gtol": gtol}
    scipy_result = minimize(
        fun=lambda x: scipy_fun(x),
        x0=theta0_np,
        jac=True,
        method=method,
        callback=callback,
        options=options,
    )

    theta_hat = np.array(scipy_result.x, dtype=float, copy=True)
    final_evaluation = evaluation_at(theta_hat)
    final_objective = final_evaluation.objective
    final_gradient = final_evaluation.gradient
    final_gradient_norm = float(np.linalg.norm(final_gradient))

    theta_jax = jnp.asarray(theta_hat)
    loglikelihood = final_evaluation.loglikelihood
    logprior_value = final_evaluation.logprior

    hessian_np: np.ndarray | None = None
    covariance_np: np.ndarray | None = None

    if compute_hessian:
        hessian = jax.jit(jax.hessian(objective))(theta_jax)
        hessian_np = np.asarray(hessian, dtype=float)
        covariance_np = _safe_inverse(hessian_np)

    if covariance_np is None:
        covariance_np = _extract_inverse_hessian(scipy_result, dim)

    se = _standard_errors(covariance_np)
    z_values = None
    if se is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            z_values = theta_hat / se

    runtime_seconds = perf_counter() - start_time
    timestamp = datetime.now().isoformat(timespec="seconds")

    if not trace:
        trace.append((0, final_objective, final_gradient_norm))

    return MLResult(
        dim=dim,
        theta_hat=theta_hat,
        objective_value=float(final_objective),
        loglikelihood=float(loglikelihood),
        logprior=float(logprior_value),
        prior_weight=float(prior_weight),
        gradient=np.asarray(final_gradient, dtype=float),
        gradient_norm=float(final_gradient_norm),
        hessian=hessian_np,
        covariance_matrix=covariance_np,
        standard_errors=se,
        z_values=z_values,
        success=bool(scipy_result.success),
        message=str(scipy_result.message),
        method=str(method),
        num_iterations=int(getattr(scipy_result, "nit", iteration["k"])),
        num_function_evaluations=int(getattr(scipy_result, "nfev", -1)),
        num_gradient_evaluations=(
            None
            if not hasattr(scipy_result, "njev")
            else int(getattr(scipy_result, "njev"))
        ),
        runtime_seconds=float(runtime_seconds),
        timestamp=timestamp,
        optimization_trace=np.asarray(trace, dtype=float),
        num_compiled_evaluations=num_compiled_evaluations,
        scipy_result=scipy_result,
    )
