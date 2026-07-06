from __future__ import annotations

from typing import Any

import numpy as np

from .results import MLResult


def _safe_float(value: np.ndarray | float | int) -> float:
    """Convert a scalar-like value to float."""
    return float(np.asarray(value).item())


def compute_optimization_diagnostics(
    result: MLResult,
    *,
    gradient_norm_threshold: float = 1e-5,
) -> dict[str, Any]:
    """Compute optimizer diagnostics for an MLResult."""
    trace = np.asarray(result.optimization_trace, dtype=float)
    has_trace = trace.size > 0

    objective_initial = None
    objective_final = result.objective_value
    objective_best = result.objective_value
    if has_trace and trace.ndim == 2 and trace.shape[1] >= 3:
        objective_initial = _safe_float(trace[0, 1])
        objective_best = _safe_float(np.min(trace[:, 1]))

    return {
        "method": result.method,
        "success": bool(result.success),
        "message": result.message,
        "num_iterations": int(result.num_iterations),
        "num_function_evaluations": int(result.num_function_evaluations),
        "num_gradient_evaluations": result.num_gradient_evaluations,
        "objective_initial": objective_initial,
        "objective_final": float(objective_final),
        "objective_best": float(objective_best),
        "gradient_norm": float(result.gradient_norm),
        "gradient_norm_threshold": float(gradient_norm_threshold),
        "small_gradient": bool(result.gradient_norm <= gradient_norm_threshold),
        "has_trace": bool(has_trace),
    }


def compute_covariance_diagnostics(result: MLResult) -> dict[str, Any]:
    """Compute diagnostics for Hessian/covariance based inference."""
    hessian = result.hessian
    covariance = result.covariance_matrix
    se = result.standard_errors

    hessian_available = hessian is not None
    covariance_available = covariance is not None
    standard_errors_available = se is not None

    hessian_symmetric = None
    hessian_condition_number = None
    hessian_min_eigenvalue = None
    hessian_positive_definite = None

    if hessian_available:
        h = np.asarray(hessian, dtype=float)
        hessian_symmetric = bool(np.allclose(h, h.T, rtol=1e-6, atol=1e-8))
        if np.all(np.isfinite(h)):
            try:
                eig = np.linalg.eigvalsh((h + h.T) / 2.0)
                hessian_min_eigenvalue = float(np.min(eig))
                hessian_positive_definite = bool(np.all(eig > 0))
                hessian_condition_number = float(np.linalg.cond(h))
            except np.linalg.LinAlgError:
                hessian_min_eigenvalue = None
                hessian_positive_definite = False
                hessian_condition_number = None

    invalid_standard_errors = None
    if standard_errors_available:
        invalid_standard_errors = bool(np.any(~np.isfinite(se)) or np.any(se < 0))

    return {
        "hessian_available": hessian_available,
        "hessian_symmetric": hessian_symmetric,
        "hessian_min_eigenvalue": hessian_min_eigenvalue,
        "hessian_positive_definite": hessian_positive_definite,
        "hessian_condition_number": hessian_condition_number,
        "covariance_available": covariance_available,
        "standard_errors_available": standard_errors_available,
        "invalid_standard_errors": invalid_standard_errors,
    }


def compute_parameter_diagnostics(
    result: MLResult,
    *,
    parameter_names: list[str] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Compute parameter-summary diagnostics."""
    theta = np.asarray(result.theta_hat, dtype=float)
    dim = theta.size
    names = (
        [f"theta[{i}]" for i in range(dim)]
        if parameter_names is None
        else list(parameter_names)
    )
    if len(names) != dim:
        raise ValueError(f"parameter_names must have length {dim}, got {len(names)}.")

    se = result.standard_errors
    z = result.z_values

    rows: list[dict[str, Any]] = []
    for i in range(dim):
        rows.append(
            {
                "index": i,
                "name": names[i],
                "estimate": float(theta[i]),
                "standard_error": None if se is None else float(se[i]),
                "z_value": None if z is None else float(z[i]),
                "abs_z_value": None if z is None else float(abs(z[i])),
            }
        )

    if se is not None:
        order = np.argsort(-np.asarray(se, dtype=float))
        top_uncertain = [rows[int(i)] for i in order[: max(int(top_k), 0)]]
    else:
        top_uncertain = []

    return {
        "dim": dim,
        "has_standard_errors": se is not None,
        "parameter_summary": rows,
        "top_uncertain_parameters": top_uncertain,
    }


def compute_all_diagnostics(
    result: MLResult,
    *,
    parameter_names: list[str] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Compute all ML diagnostics."""
    return {
        "metadata": {
            "method": result.method,
            "dim": result.dim,
            "prior_weight": result.prior_weight,
            "runtime_seconds": result.runtime_seconds,
            "timestamp": result.timestamp,
        },
        "optimization": compute_optimization_diagnostics(result),
        "covariance": compute_covariance_diagnostics(result),
        "parameters": compute_parameter_diagnostics(
            result,
            parameter_names=parameter_names,
            top_k=top_k,
        ),
    }
