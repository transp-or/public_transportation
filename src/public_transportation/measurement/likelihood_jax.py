# src/public_transportation/measurement/likelihood_jax.py
"""
JAX-compatible measurement equation and likelihood.

This module provides pure-JAX functions that:
  1) aggregate assignment link flows into predicted measurement quantities, and
  2) compute a (properly normalized) Negative Binomial log-likelihood.

Key design choices
------------------
- The mapping from measurements to assignment links is represented by integer
  index arrays ("AggregationSpec") that are built ONCE in Python (strict mapper),
  and then used inside JAX without any Python logic.
- No "methods" are modeled here: a single scalar rho is used to scale predicted
  continuous flows into expected counts.
- The assignment itself remains the sole owner of its indexing conventions.
  The mapper's output is only (measurement_index, link_index, num_measurements).

All functions in this file are pure JAX (JIT-safe) and are suitable for use inside
PyMC/JAX graphs. This module intentionally contains no PyMC code, no IO, and no
assignment-wiring logic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, xlog1py, xlogy

Array = jnp.ndarray


@jax.jit(static_argnames=("spec_num_measurements",))
def predict_measurements_from_link_flow(
    link_flow: jnp.ndarray,
    spec_num_measurements: int,
    spec_measurement_index: jnp.ndarray,
    spec_link_index: jnp.ndarray,
) -> jnp.ndarray:
    """Aggregate assignment link flows into predicted measurement quantities.

    Parameters
    ----------
    link_flow:
        Assignment link flow vector, shape (num_links,).

    spec_*:
        Aggregation spec components produced by the strict mapper (Python side) and
        passed as JAX arrays/constants.
        Typically:
          spec_num_measurements = spec.num_measurements
          spec_measurement_index = spec.measurement_index
          spec_link_index = spec.link_index

    Returns
    -------
    y_pred:
        Predicted measurement vector, shape (M,).
    """
    # Gather the contributing link flows
    contrib = link_flow[spec_link_index]  # (K,)
    y_pred = jnp.zeros((spec_num_measurements,), dtype=link_flow.dtype)
    y_pred = y_pred.at[spec_measurement_index].add(contrib)
    return y_pred


def _nb_logpmf_mu_r(y: Array, mu: Array, r: Array) -> Array:
    """Negative Binomial log pmf with mean mu and dispersion r (shape/size).

    Parameterization:
      Var(Y) = mu + mu^2 / r
      p = r / (r + mu)

    log pmf:
      log Γ(y+r) - log Γ(r) - log Γ(y+1)
      + r log p + y log(1-p)

    All operations are JAX-compatible.
    """
    # p in (0,1)
    p = r / (r + mu)
    # Use gammaln for stability and to avoid factorials.
    # Use xlogy/xlog1py to avoid NaN gradients from 0*log(0) or 0*(-inf).
    return (
        gammaln(y + r)
        - gammaln(r)
        - gammaln(y + 1.0)
        + xlogy(r, p)
        + xlog1py(y, -p)
    )


@jax.jit
def negbinom_loglikelihood(
    *,
    y_obs: Array,
    mu: Array,
    r: Array,
) -> Array:
    """Return the total NB log-likelihood sum_m log p(y_obs[m] | mu[m], r).

    y_obs should be nonnegative (integer-valued in principle).
    """
    return jnp.sum(_nb_logpmf_mu_r(y_obs, mu, r))


@jax.jit(static_argnames=("spec_num_measurements",))
def measurement_loglik_from_link_flow(
    *,
    link_flow: Array,
    y_obs: Array,
    theta: Array,
    rho: Array,
    r: Array,
    # Aggregation spec components (passed explicitly for JIT-friendliness)
    spec_num_measurements: int,
    spec_measurement_index: Array,
    spec_link_index: Array,
    eps_mu: float = 1e-9,
) -> Array:
    """Compute log p(y_obs | link_flow, theta, rho, r) under NB.

    Notes
    -----
    - theta appears explicitly in the signature for clarity and to match the
      intended statistical model. It is not used here directly because theta
      acts through link_flow (produced by the assignment). Keeping it in the
      signature is useful for PyMC graphs and for later extensions.
    - rho is a *single scalar* scaling continuous predicted flows into expected
      counts: mu = rho * y_pred.
    - eps_mu avoids mu=0 which would create log(0) issues.
    """
    # Predicted continuous quantities in measurement space
    y_pred = predict_measurements_from_link_flow(
        link_flow=link_flow,
        spec_num_measurements=spec_num_measurements,
        spec_measurement_index=spec_measurement_index,
        spec_link_index=spec_link_index,
    )

    # Mean of counts
    mu = rho * y_pred
    mu = jnp.maximum(mu, jnp.asarray(eps_mu, dtype=mu.dtype))
    eps = jnp.asarray(1e-6, dtype=mu.dtype)
    mu = mu + eps

    # NB loglik
    _ = theta  # theta influences link_flow; included for explicitness.
    return negbinom_loglikelihood(y_obs=y_obs, mu=mu, r=r)

