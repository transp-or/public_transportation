from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

Array = jnp.ndarray


LogLikFn = Callable[[Array, Any], Array | float]
LogPriorFn = Callable[[Array], Array | float]

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



def negative_log_prior_penalty(logprior: LogPriorFn, theta: Array) -> Array:
    """
    Return the negative log-prior contribution used by penalized ML/MAP.

    Additive constants in `logprior` are harmless for optimization. For a
    Gaussian prior N(mu, sigma^2), this is equivalent, up to constants, to

        0.5 * ((theta - mu) / sigma) ** 2

    summed over parameters.

    :param logprior: JAX-compatible absolute log-prior function.
    :param theta: Parameter vector.
    :return: Scalar penalty -logprior(theta).
    """
    return -_as_scalar(logprior(theta))
