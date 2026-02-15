"""
Example: Bayesian least-squares estimation of a linear model using VI (NumPyro),
reusing the generic utilities from `bayesian_estimation.py`.

This example does NOT re-implement the VI machinery. It imports:
- run_vi
- recommend_vi_defaults (optional)
- Array (type alias)

Model
-----
We generate synthetic data from:
    y_i = a + b x_i + ε_i,    ε_i ~ Normal(0, σ^2)

Then we estimate θ = [a, b] via variational inference targeting:
    p(θ | data) ∝ exp( loglik(θ; data) + logprior_increment(θ) ) * Normal(θ; 0, I)

Because `bayesian_estimation.run_vi` uses the efficiency-first default
(use_base_normal_correction=False), the prior must be provided as an increment:

    logprior_increment(θ) = log p(θ) - log Normal(θ; 0, I)

In this example we choose:
    p(θ) = Normal(0, 10^2 I)  (weakly informative)

So:
    logprior_increment(θ) = sum_k [log N(θ_k; 0, 10) - log N(θ_k; 0, 1)]

Assumptions
-----------
- loglik and logprior_increment are implemented in JAX (jax.numpy / numpyro distributions).
- They are differentiable w.r.t. θ (almost everywhere).
- They return scalar log-densities.

Run
---
python examples/linear_least_squares_vi.py
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
import numpyro.distributions as dist

from public_transportation.bayesian_estimation import run_vi, recommend_vi_defaults, Array

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("public_transportation.vi")

def generate_synthetic_data(
    *,
    n: int,
    a_true: float,
    b_true: float,
    sigma: float,
    seed: int = 42,
) -> dict[str, Array]:
    """
    Generate synthetic data for y = a + b x + noise.

    :param n: Number of observations.
    :param a_true: True intercept.
    :param b_true: True slope.
    :param sigma: Noise standard deviation (known in this example).
    :param seed: RNG seed (NumPy).
    :return: Data PyTree with keys 'x', 'y', 'sigma' as JAX arrays.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=n)
    y = a_true + b_true * x + rng.normal(0.0, sigma, size=n)
    return {
        "x": jnp.asarray(x),
        "y": jnp.asarray(y),
        "sigma": jnp.asarray(sigma),
    }


def loglik_linear(theta: Array, data: dict[str, Array]) -> Array:
    """
    Log-likelihood for linear regression with known sigma.

    Model:
        y_i ~ Normal(a + b x_i, sigma)

    :param theta: Parameter vector [a, b], shape (2,).
    :param data: Dict with keys 'x', 'y', 'sigma' as JAX arrays.
    :return: Scalar log-likelihood log p(y | theta).
    """
    a = theta[0]
    b = theta[1]
    mu = a + b * data["x"]
    return dist.Normal(mu, data["sigma"]).log_prob(data["y"]).sum()


def logprior_increment_normal_scale(theta: Array, *, prior_sd: float = 10.0) -> Array:
    """
    Prior increment for an independent Normal(0, prior_sd^2) prior on theta,
    expressed relative to the base Normal(0, I) used by `bayesian_estimation.py`.

    We want the target posterior to use:
        log p(theta)  with  p(theta) = Normal(0, prior_sd^2 I)

    But the VI engine defines (up to proportionality):
        Normal(theta;0,I) * exp(loglik + logprior_increment)

    Therefore, to obtain:
        exp(loglik + log p(theta))
    we must provide:
        logprior_increment(theta) = log p(theta) - log Normal(theta;0,I).

    :param theta: Parameter vector, shape (d,).
    :param prior_sd: Standard deviation of the intended prior p(theta).
    :return: Scalar logprior increment.
    """
    # log p(theta) under N(0, prior_sd^2 I)
    lp_target = dist.Normal(0.0, prior_sd).log_prob(theta).sum()
    # log base density under N(0, I)
    lp_base = dist.Normal(0.0, 1.0).log_prob(theta).sum()
    return lp_target - lp_base


def summarize_posterior(samples: np.ndarray) -> dict[str, Any]:
    """
    Compute mean, sd, and 95% intervals for a 2D parameter vector.

    :param samples: Posterior samples, shape (n_draws, 2).
    :return: Summary dict with entries for 'a' and 'b'.
    """
    a = samples[:, 0]
    b = samples[:, 1]
    return {
        "a": {
            "mean": float(np.mean(a)),
            "sd": float(np.std(a, ddof=1)),
            "ci95": (float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))),
        },
        "b": {
            "mean": float(np.mean(b)),
            "sd": float(np.std(b, ddof=1)),
            "ci95": (float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))),
        },
    }


def main() -> None:
    """
    Run the full example: generate data, run VI, print and plot results.
    """
    # True parameters and data
    a_true = 1.25
    b_true = -0.80
    sigma = 0.40
    data = generate_synthetic_data(n=120, a_true=a_true, b_true=b_true, sigma=sigma, seed=42)

    # VI settings (dim=2, so defaults are simple; shown here for consistency)
    defaults = recommend_vi_defaults(dim=2)
    # For a tiny model like this, reduce steps a bit to keep it quick.
    defaults["num_steps"] = 4000
    defaults["learning_rate"] = 1e-2
    defaults["num_posterior_draws"] = 5000

    # Run VI using the generic routine from bayesian_estimation.py
    res = run_vi(
        dim=2,
        data=data,
        loglik=loglik_linear,
        logprior=lambda th: logprior_increment_normal_scale(th, prior_sd=10.0),
        use_base_normal_correction=False,  # efficiency-first: expects prior increment
        seed=123,
        logger=logger,
        log_every=100,
        **defaults,
    )

    samples = res.posterior_samples_theta
    summary = summarize_posterior(samples)

    print("=== VI (NumPyro SVI) — Bayesian least squares linear regression ===")
    print(f"True: a={a_true:.3f}, b={b_true:.3f}, sigma={sigma:.3f}")
    print(
        f"a: mean={summary['a']['mean']:.3f}, sd={summary['a']['sd']:.3f}, "
        f"95% CI=[{summary['a']['ci95'][0]:.3f}, {summary['a']['ci95'][1]:.3f}]"
    )
    print(
        f"b: mean={summary['b']['mean']:.3f}, sd={summary['b']['sd']:.3f}, "
        f"95% CI=[{summary['b']['ci95'][0]:.3f}, {summary['b']['ci95'][1]:.3f}]"
    )
    print(f"Guide: {res.guide} | Final ELBO loss: {res.losses[-1]:.3f}")

    # Plot: data and posterior mean fit
    x_np = np.asarray(data["x"])
    y_np = np.asarray(data["y"])
    x_grid = np.linspace(x_np.min(), x_np.max(), 200)
    y_fit = summary["a"]["mean"] + summary["b"]["mean"] * x_grid

    plt.figure()
    plt.plot(x_np, y_np, "o", label="data")
    plt.plot(x_grid, y_fit, label="VI posterior mean fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear regression: data and VI posterior mean fit")
    plt.legend()
    plt.show()

    # Plot: ELBO loss trace
    plt.figure()
    plt.plot(res.losses)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss")
    plt.title("SVI optimization trace (ELBO loss)")
    plt.show()


if __name__ == "__main__":
    main()