from __future__ import annotations

from collections.abc import Callable
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
    lp_fun = _zero_logprior if logprior is None else logprior

    def objective(theta: Array) -> Array:
        ll = _as_scalar(loglik(theta, data))
        lp = _as_scalar(lp_fun(theta))
        return -(ll + prior_weight * lp)

    return objective


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
    compute_hessian = cfg.compute_hessian if compute_hessian is None else bool(compute_hessian)

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
            covariance_matrix=(np.empty((0, 0), dtype=float) if compute_hessian else None),
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
            scipy_result=None,
        )

    if theta0 is None:
        theta0_np = np.zeros(dim, dtype=float)
    else:
        theta0_np = np.asarray(theta0, dtype=float).reshape((dim,))

    objective = make_ml_objective(
        loglik=loglik,
        logprior=logprior,
        data=data,
        prior_weight=prior_weight,
    )
    value_and_grad = jax.jit(jax.value_and_grad(objective))

    trace: list[tuple[int, float, float]] = []
    start_time = perf_counter()
    last_log_time = start_time

    def scipy_fun(theta_np: np.ndarray) -> tuple[float, np.ndarray]:
        theta = jnp.asarray(theta_np)
        value, grad = value_and_grad(theta)
        value_np = float(np.asarray(value))
        grad_np = np.asarray(grad, dtype=float)
        return value_np, grad_np

    iteration = {"k": 0}

    def callback(theta_np: np.ndarray) -> None:
        value_np, grad_np = scipy_fun(theta_np)
        grad_norm = float(np.linalg.norm(grad_np))
        trace.append((iteration["k"], value_np, grad_norm))

        if logger is not None and iteration["k"] % cfg.log_every == 0:
            now = perf_counter()
            logger.info(
                "ML iteration %d — objective: %.6f — gradient norm: %.6g — elapsed: %s",
                iteration["k"],
                value_np,
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

    theta_hat = np.asarray(scipy_result.x, dtype=float)
    final_objective, final_gradient = scipy_fun(theta_hat)
    final_gradient_norm = float(np.linalg.norm(final_gradient))

    theta_jax = jnp.asarray(theta_hat)
    loglikelihood = float(np.asarray(_as_scalar(loglik(theta_jax, data))))
    logprior_value = (
        0.0
        if logprior is None
        else float(np.asarray(_as_scalar(logprior(theta_jax))))
    )

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
        scipy_result=scipy_result,
    )
