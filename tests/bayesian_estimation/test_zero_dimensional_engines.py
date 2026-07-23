from __future__ import annotations

import jax.numpy as jnp

from public_transportation.estimation.bayesian import run_vi
from public_transportation.estimation.maximum_likelihood import MLConfig, run_ml


def test_vi_bypasses_svi_for_zero_dimensional_problem():
    calls = {"likelihood": 0, "prior": 0}

    def loglik(theta, data):
        calls["likelihood"] += 1
        assert theta.shape == (0,)
        return jnp.asarray(data["value"])

    def logprior(theta):
        calls["prior"] += 1
        assert theta.shape == (0,)
        return jnp.asarray(0.0)

    result = run_vi(
        dim=0,
        data={"value": -2.0},
        loglik=loglik,
        logprior=logprior,
        num_steps=100,
        num_posterior_draws=3,
    )

    assert calls == {"likelihood": 1, "prior": 1}
    assert result.dim == 0
    assert result.num_steps == 0
    assert result.svi_state is None
    assert result.posterior_samples_theta.shape == (3, 0)
    assert result.losses.shape == (0,)


def test_ml_bypasses_scipy_for_zero_dimensional_problem():
    calls = {"likelihood": 0, "prior": 0}

    def loglik(theta, data):
        calls["likelihood"] += 1
        assert theta.shape == (0,)
        return jnp.asarray(data["value"])

    def logprior(theta):
        calls["prior"] += 1
        assert theta.shape == (0,)
        return jnp.asarray(-1.5)

    result = run_ml(
        dim=0,
        data={"value": -2.0},
        loglik=loglik,
        logprior=logprior,
        config=MLConfig(prior_weight=1.0, compute_hessian=True),
    )

    assert calls == {"likelihood": 1, "prior": 1}
    assert result.success is True
    assert result.dim == 0
    assert result.theta_hat.shape == (0,)
    assert result.objective_value == 3.5
    assert result.hessian is not None and result.hessian.shape == (0, 0)
    assert "optimizer bypassed" in result.message


def test_engines_reject_negative_dimension():
    def loglik(theta, data):
        return jnp.asarray(0.0)

    def logprior(theta):
        return jnp.asarray(0.0)

    for runner in (run_vi, run_ml):
        kwargs = dict(dim=-1, data={}, loglik=loglik, logprior=logprior)
        try:
            runner(**kwargs)
        except ValueError as error:
            assert "non-negative" in str(error)
        else:
            raise AssertionError("negative dimension was accepted")
