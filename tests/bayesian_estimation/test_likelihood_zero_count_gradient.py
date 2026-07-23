from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.measurement.likelihood_jax import negbinom_loglikelihood


def test_negative_binomial_zero_count_has_finite_float32_gradient() -> None:
    y_obs = jnp.asarray([0.0, 0.0, 3.0], dtype=jnp.float32)
    r = jnp.asarray(100.0, dtype=jnp.float32)

    def objective(mu: jnp.ndarray) -> jnp.ndarray:
        return negbinom_loglikelihood(y_obs=y_obs, mu=mu, r=r)

    mu = jnp.asarray([1.0e-9, 1.0e-6, 2.5], dtype=jnp.float32)
    value, gradient = jax.value_and_grad(objective)(mu)

    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradient)))
