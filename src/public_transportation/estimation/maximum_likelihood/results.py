from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MLResult:
    """
    Container for maximum-likelihood or penalized-maximum-likelihood results.

    Parameters
    ----------
    theta_hat:
        Estimated parameter vector.

    objective_value:
        Final minimized objective value. For pure ML this is -loglik(theta_hat).
        With prior_weight > 0 it is -loglik(theta_hat) - prior_weight*logprior(theta_hat).

    loglikelihood:
        Log-likelihood value at theta_hat.

    logprior:
        Log-prior value at theta_hat.

    prior_weight:
        Global prior penalty weight. Zero means pure ML.

    gradient:
        Gradient of the minimized objective at theta_hat.

    hessian:
        Hessian of the minimized objective at theta_hat, if computed.

    covariance_matrix:
        Inverse Hessian approximation or inverse exact Hessian, if available.

    standard_errors:
        Square root of the covariance diagonal, if available.

    z_values:
        theta_hat / standard_errors, if available.

    success, message:
        Optimizer termination information.
    """

    dim: int
    theta_hat: np.ndarray
    objective_value: float
    loglikelihood: float
    logprior: float
    prior_weight: float
    gradient: np.ndarray
    gradient_norm: float
    hessian: np.ndarray | None
    covariance_matrix: np.ndarray | None
    standard_errors: np.ndarray | None
    z_values: np.ndarray | None
    success: bool
    message: str
    method: str
    num_iterations: int
    num_function_evaluations: int
    num_gradient_evaluations: int | None
    runtime_seconds: float
    timestamp: str
    optimization_trace: np.ndarray
    scipy_result: Any | None = None
