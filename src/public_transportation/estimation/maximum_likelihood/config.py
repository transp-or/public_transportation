from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class MLConfig:
    """
    Configuration for the maximum-likelihood / penalized-maximum-likelihood engine.

    The statistical model specification is intentionally shared with Bayesian
    estimation: the same log-likelihood and the same log-prior are used.

    Parameters
    ----------
    method:
        SciPy optimization method. Recommended defaults are "BFGS" for smooth
        unconstrained problems and "L-BFGS-B" when bounds are needed.

    maxiter:
        Maximum number of optimizer iterations.

    gtol:
        Gradient-norm tolerance passed to SciPy when supported.

    prior_weight:
        Global scaling factor applied to the prior penalty. Use 0.0 for pure
        maximum likelihood. Use 1.0 for the MAP/penalized-ML objective induced
        by the same prior used in Bayesian estimation.

    compute_hessian:
        If True, compute the Hessian of the negative objective at the optimum
        and attempt to invert it to obtain an asymptotic covariance matrix.

    finite_difference_check:
        Reserved for optional future gradient checks.
    """

    method: Literal["BFGS", "L-BFGS-B", "CG", "Newton-CG", "trust-ncg"] = "BFGS"
    maxiter: int = 1_000
    gtol: float = 1e-6
    prior_weight: float = 0.0
    seed: int = 0
    compute_hessian: bool = True
    finite_difference_check: bool = False
    log_every: int = 1

    def validate(self) -> None:
        """Validate configuration values."""
        if self.maxiter <= 0:
            raise ValueError("maxiter must be positive.")
        if self.gtol <= 0:
            raise ValueError("gtol must be positive.")
        if self.prior_weight < 0:
            raise ValueError("prior_weight must be non-negative.")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive.")
