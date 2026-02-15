"""
Generic Variational Inference (VI) with NumPyro (JAX) for a black-box log-posterior.

Design goals
------------
This module is intentionally **efficiency-first** rather than fully general.
It targets the common high-dimensional calibration setting where:

- Parameters live in an **unconstrained** vector space (R^d).
- The user provides a **JAX-compatible**, **differentiable** log-likelihood and log-prior.
- We run SVI (stochastic variational inference) with NumPyro autoguides.

Core contract (IMPORTANT)
-------------------------
You must implement:

1) loglik(theta, data) -> scalar log-likelihood
2) logprior(theta)     -> scalar log-prior

with the following assumptions:

A. JAX-traceable
   - Use `jax.numpy as jnp` (and optionally `jax.scipy`) for all computations.
   - Do NOT call `numpy`, `pandas`, non-JAX SciPy, or arbitrary Python side effects
     inside these functions.

B. Differentiable w.r.t. theta
   - The ELBO gradient uses pathwise/reparameterization gradients, which require
     differentiability of loglik and logprior w.r.t. theta (almost everywhere).
   - Hard discontinuities (argmin/argmax, discrete route choice, link removal, etc.)
     should be smoothed/relaxed for best results.

C. Deterministic
   - Same (theta, data) must yield the same value. If you need randomness, handle it
     explicitly with JAX PRNG keys and a different modeling interface (not provided here).

D. Scalar output and numerical stability
   - Return a scalar (shape ()) JAX array or a Python float.
   - Avoid NaNs. Return `-jnp.inf` for invalid theta regions.

Data must be JAX-friendly
-------------------------
`data` should be a PyTree of JAX arrays (nested dict/list/tuple of `jnp.ndarray`),
prepared outside the inference loop.

Parameter constraints
---------------------
This module assumes theta is **already unconstrained**. If your natural parameters are
constrained (e.g., positivity), you should reparameterize externally, for example:

    z in R^d  (unconstrained)
    theta = exp(z)   (positive)

and then implement `loglik_z(z, data) = loglik(exp(z), data)` and
`logprior_z(z) = logprior_theta(exp(z)) + sum(z)` if you need the Jacobian.

Efficiency note: prefer defining priors directly in the unconstrained space when possible.

What this module does
---------------------
We define a NumPyro model:

    theta ~ Normal(0, I)              (a convenient latent site for autoguides)
    factor("target", loglik(theta,data) + logprior(theta))

This means the *target posterior* is proportional to:

    exp( loglik(theta,data) + logprior(theta) ) * Normal(theta;0,I)

So: **your logprior must be specified relative to this base** if you want the exact
posterior you intend.

Recommended practice (IMPORTANT)
--------------------------------
Define logprior as an *increment* relative to the base Normal(0,I), i.e.,

    logprior(theta) = log p(theta) - log Normal(theta;0,I)

Then the target becomes exactly exp(loglik + log p(theta)).

This avoids subtracting the base density inside the model (saves work) and keeps the model fast.

If you instead want to specify logprior(theta) = log p(theta) directly, then set
`use_base_normal_correction=True` in `make_blackbox_model` or `run_vi` to subtract the base
Normal log-density (slightly slower but conceptually simpler).

Dependencies
------------
- jax
- numpyro
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import (
    AutoDiagonalNormal,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
)
from numpyro.optim import Adam


Array = jnp.ndarray
PRNGKey = Array

LogLikFn = Callable[[Array, Any], Array | float]
LogPriorFn = Callable[[Array], Array | float]


@dataclass(frozen=True)
class VIResult:
    """
    Container for variational inference results.

    :param guide: Name of the guide used.
    :param dim: Dimension of the parameter vector theta.
    :param use_base_normal_correction: Whether the base Normal(0,I) log-density was subtracted.
    :param svi_state: Final SVI state (NumPyro object).
    :param params: Learned variational parameters (PyTree).
    :param losses: ELBO losses per optimization step, shape (num_steps,).
    :param posterior_samples_theta: Samples from the variational posterior over theta,
        shape (num_draws, dim).
    """
    guide: str
    dim: int
    use_base_normal_correction: bool
    svi_state: Any
    params: Any
    losses: np.ndarray
    posterior_samples_theta: np.ndarray


def _as_scalar(x: Array | float) -> Array:
    """
    Convert input to a scalar JAX array.

    :param x: A Python float or JAX array representing a scalar value.
    :return: Scalar JAX array of shape ().
    """
    return jnp.asarray(x).reshape(())


def base_normal_logpdf(theta: Array) -> Array:
    """
    Compute log N(theta; 0, I) for theta in R^d.

    :param theta: Parameter vector of shape (d,).
    :return: Scalar log-density.
    """
    return dist.Normal(0.0, 1.0).log_prob(theta).sum()


def make_blackbox_model(
    *,
    dim: int,
    loglik: LogLikFn,
    logprior: LogPriorFn,
    data: Any,
    use_base_normal_correction: bool = False,
) -> Callable[[], None]:
    """
    Build a NumPyro model for black-box log-likelihood and log-prior.

    Model form
    ----------
    We introduce a latent site for autoguides:

        theta ~ Normal(0, I)

    and add a factor term to define the target log density.

    Two modes exist:

    1) Efficiency-first (default): `use_base_normal_correction=False`
       We set:
           factor = loglik(theta,data) + logprior(theta)

       In this case, the implied target is proportional to:
           Normal(theta;0,I) * exp(loglik + logprior)

       Recommended practice: define
           logprior(theta) = log p(theta) - log Normal(theta;0,I)
       so that the target becomes exactly exp(loglik + log p(theta)).

    2) Exact-prior mode: `use_base_normal_correction=True`
       We set:
           factor = loglik(theta,data) + logprior(theta) - log Normal(theta;0,I)

       In this case, if you provide logprior(theta) = log p(theta), then the target is
       exactly proportional to exp(loglik + log p(theta)).

    :param dim: Dimension of theta.
    :param loglik: JAX-compatible log-likelihood. Signature: loglik(theta, data) -> scalar.
    :param logprior: JAX-compatible log-prior. Signature: logprior(theta) -> scalar.
        Interpretation depends on `use_base_normal_correction` (see above).
    :param data: User data passed to loglik. Should be a JAX PyTree for best performance.
    :param use_base_normal_correction: If True, subtract log N(theta;0,I) inside the model,
        allowing logprior to be the absolute log prior log p(theta).
    :return: A zero-argument NumPyro model callable.
    """
    if dim <= 0:
        raise ValueError("dim must be a positive integer.")

    def model() -> None:
        theta = numpyro.sample("theta", dist.Normal(0.0, 1.0).expand([dim]).to_event(1))

        ll = _as_scalar(loglik(theta, data))
        lp = _as_scalar(logprior(theta))

        if use_base_normal_correction:
            lp = lp - base_normal_logpdf(theta)

        numpyro.factor("target", ll + lp)

    return model


def make_autoguide(
    *,
    model: Callable[[], None],
    guide: Literal["auto_diag", "auto_lowrank", "auto_mvn", "auto_normal"] = "auto_diag",
    lowrank_rank: int | None = None,
) -> Any:
    """
    Create a NumPyro autoguide for the given model.

    :param model: NumPyro model callable.
    :param guide: Choice of autoguide:
        - "auto_diag": AutoDiagonalNormal (mean-field Gaussian). Best default for very high d.
        - "auto_lowrank": Low-rank + diagonal Gaussian. Captures correlations at moderate cost.
        - "auto_mvn": Full-covariance Gaussian. Usually infeasible for thousands of parameters.
        - "auto_normal": AutoNormal (kept for completeness; often similar spirit).
    :param lowrank_rank: Rank for "auto_lowrank". If None, a conservative default is used.
    :return: A NumPyro autoguide instance.
    """
    if guide == "auto_diag":
        return AutoDiagonalNormal(model)
    if guide == "auto_lowrank":
        rank = 20 if lowrank_rank is None else int(lowrank_rank)
        if rank <= 0:
            raise ValueError("lowrank_rank must be a positive integer.")
        return AutoLowRankMultivariateNormal(model, rank=rank)
    if guide == "auto_mvn":
        return AutoMultivariateNormal(model)
    if guide == "auto_normal":
        return AutoNormal(model)
    raise ValueError(f"Unknown guide: {guide!r}")


def run_vi(
    *,
    dim: int,
    data: Any,
    loglik: LogLikFn,
    logprior: LogPriorFn,
    guide: Literal["auto_diag", "auto_lowrank", "auto_mvn", "auto_normal"] = "auto_diag",
    lowrank_rank: int | None = None,
    use_base_normal_correction: bool = False,
    num_steps: int = 5_000,
    learning_rate: float = 1e-2,
    seed: int = 0,
    num_posterior_draws: int = 1_000,
    logger: Any | None = None,
    log_every: int = 100,
) -> VIResult:
    """
    Run variational inference (SVI) for a user-defined posterior target.

    :param dim: Dimension of theta (unconstrained parameter vector).
    :param data: Arbitrary user data passed to loglik(theta, data). Prefer JAX PyTrees.
    :param loglik: JAX-compatible, differentiable log-likelihood. Must return a scalar.
    :param logprior: JAX-compatible, differentiable log-prior term. Must return a scalar.
        Interpretation:
          - If `use_base_normal_correction=False` (default), best practice is:
                logprior(theta) = log p(theta) - log N(theta;0,I).
          - If `use_base_normal_correction=True`, then:
                logprior(theta) = log p(theta).
    :param guide: Autoguide choice. For thousands of parameters, prefer:
        - "auto_diag" (fast) or
        - "auto_lowrank" (captures correlations).
    :param lowrank_rank: Rank for "auto_lowrank".
    :param use_base_normal_correction: If True, subtract log N(theta;0,I) inside the model
        so `logprior` can be an absolute prior log-density.
    :param num_steps: Number of SVI optimization steps.
    :param learning_rate: Adam learning rate.
    :param seed: Random seed for initialization and posterior sampling.
    :param num_posterior_draws: Number of posterior samples to draw from the variational guide.
    :param logger: Optional logger used to report progress. It must provide a method
        `info(msg, *args)` like Python's standard `logging.Logger`. If None, no logging
        is performed.
    :param log_every: If logger is provided, emit a log message every `log_every` steps,
        and also at the last step. Must be a positive integer.
    :return: VIResult including learned parameters and posterior samples.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if num_posterior_draws <= 0:
        raise ValueError("num_posterior_draws must be positive.")
    if log_every <= 0:
        raise ValueError("log_every must be positive.")

    model = make_blackbox_model(
        dim=dim,
        loglik=loglik,
        logprior=logprior,
        data=data,
        use_base_normal_correction=use_base_normal_correction,
    )
    guide_obj = make_autoguide(model=model, guide=guide, lowrank_rank=lowrank_rank)

    optimizer = Adam(learning_rate)
    svi = SVI(model=model, guide=guide_obj, optim=optimizer, loss=Trace_ELBO())

    key = jax.random.PRNGKey(seed)

    svi_state = svi.init(key)

    losses: list[float] = []
    for step in range(num_steps):
        svi_state, loss = svi.update(svi_state)
        loss_val = float(loss)
        losses.append(loss_val)

        if logger is not None and (step % log_every == 0 or step == num_steps - 1):
            logger.info(
                "VI step %d/%d — ELBO loss: %.6f",
                step + 1,
                num_steps,
                loss_val,
            )

    params = svi.get_params(svi_state)

    key, subkey = jax.random.split(key)
    theta_samples = guide_obj.sample_posterior(
        subkey,
        params,
        sample_shape=(num_posterior_draws,),
    )["theta"]

    return VIResult(
        guide=guide,
        dim=dim,
        use_base_normal_correction=use_base_normal_correction,
        svi_state=svi_state,
        params=params,
        losses=np.asarray(losses, dtype=float),
        posterior_samples_theta=np.asarray(theta_samples),
    )


def recommend_vi_defaults(dim: int) -> dict[str, Any]:
    """
    Recommend conservative VI defaults for large-dimensional problems.

    :param dim: Parameter dimension.
    :return: Dictionary of suggested settings for `run_vi`.
    """
    if dim <= 0:
        raise ValueError("dim must be positive.")

    # Heuristics: keep it simple and stable for very large dim.
    if dim >= 5000:
        return {
            "guide": "auto_diag",
            "learning_rate": 5e-3,
            "num_steps": 10_000,
            "num_posterior_draws": 2000,
        }
    if dim >= 1000:
        return {
            "guide": "auto_lowrank",
            "lowrank_rank": 50,
            "learning_rate": 1e-2,
            "num_steps": 8_000,
            "num_posterior_draws": 2000,
        }
    return {
        "guide": "auto_lowrank",
        "lowrank_rank": 20,
        "learning_rate": 1e-2,
        "num_steps": 5_000,
        "num_posterior_draws": 2000,
    }


# ----------------------------
# Minimal usage sketch (no execution)
# ----------------------------
if __name__ == "__main__":
    """
    Usage patterns
    --------------

    Preferred (efficiency-first) prior specification:
        logprior(theta) = log p(theta) - log N(theta;0,I)

    then run with:
        use_base_normal_correction=False

    Alternative (absolute prior):
        logprior(theta) = log p(theta)

    then run with:
        use_base_normal_correction=True
    """

    def loglik(theta: Array, data: Any) -> Array:
        # Must be JAX-compatible and differentiable w.r.t theta.
        raise NotImplementedError

    def logprior_increment(theta: Array) -> Array:
        # Example: if you want a Normal(0, sigma^2 I) prior:
        # log p(theta) - log N(theta;0,I) = sum_i [log N(theta_i;0,sigma) - log N(theta_i;0,1)]
        raise NotImplementedError

    # defaults = recommend_vi_defaults(dim=5000)
    # res = run_vi(
    #     dim=5000,
    #     data=your_data_pytree,
    #     loglik=loglik,
    #     logprior=logprior_increment,
    #     use_base_normal_correction=False,
    #     **defaults,
    # )
    pass