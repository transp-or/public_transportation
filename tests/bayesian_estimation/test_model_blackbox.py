# tests/inference/test_model_blackbox.py
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

from public_transportation.estimation.bayesian.model_blackbox import (
    base_normal_logpdf,
    make_blackbox_model,
)


def _run_model(model):
    return trace(seed(model, jax.random.PRNGKey(0))).get_trace()


def test_base_normal_logpdf_matches_numpyro_distribution():
    theta = jnp.asarray([0.0, 1.0, -2.0], dtype=jnp.float32)

    actual = base_normal_logpdf(theta)
    expected = dist.Normal(0.0, 1.0).log_prob(theta).sum()

    assert actual.shape == ()
    assert np.isclose(float(actual), float(expected))


def test_base_normal_logpdf_scalar_dimension():
    theta = jnp.asarray([1.5], dtype=jnp.float32)

    actual = base_normal_logpdf(theta)
    expected = dist.Normal(0.0, 1.0).log_prob(theta).sum()

    assert actual.shape == ()
    assert np.isclose(float(actual), float(expected))


@pytest.mark.parametrize("dim", [1, 2, 5])
def test_make_blackbox_model_creates_theta_site_with_expected_shape(dim: int):
    def loglik(theta, data):
        assert theta.shape == (dim,)
        assert data["scale"] == 2.0
        return -jnp.sum(theta**2)

    def logprior(theta):
        assert theta.shape == (dim,)
        return -0.5 * jnp.sum(theta**2)

    model = make_blackbox_model(
        dim=dim,
        loglik=loglik,
        logprior=logprior,
        data={"scale": 2.0},
    )

    tr = _run_model(model)

    assert "theta" in tr
    assert "target" in tr
    assert tr["theta"]["value"].shape == (dim,)
    assert tr["target"]["type"] == "sample"
    assert tr["target"]["fn"].log_factor.shape == ()


@pytest.mark.parametrize("dim", [0, -1, -5])
def test_make_blackbox_model_rejects_nonpositive_dimension(dim: int):
    with pytest.raises(ValueError, match="dim must be a positive integer"):
        make_blackbox_model(
            dim=dim,
            loglik=lambda theta, data: 0.0,
            logprior=lambda theta: 0.0,
            data=None,
        )


def test_model_without_base_normal_correction_uses_loglik_plus_logprior():
    dim = 3

    def loglik(theta, data):
        return data["offset"] + jnp.sum(theta)

    def logprior(theta):
        return -0.25 * jnp.sum(theta**2)

    data = {"offset": jnp.asarray(7.0)}
    model = make_blackbox_model(
        dim=dim,
        loglik=loglik,
        logprior=logprior,
        data=data,
        use_base_normal_correction=False,
    )

    tr = _run_model(model)
    theta = tr["theta"]["value"]

    expected = loglik(theta, data) + logprior(theta)
    actual = tr["target"]["fn"].log_factor

    assert np.isclose(float(actual), float(expected))


def test_model_with_base_normal_correction_subtracts_base_normal_logpdf():
    dim = 4

    def loglik(theta, data):
        return data["offset"] - 0.1 * jnp.sum(theta)

    def logprior(theta):
        return -0.5 * jnp.sum((theta - 2.0) ** 2)

    data = {"offset": jnp.asarray(3.0)}
    model = make_blackbox_model(
        dim=dim,
        loglik=loglik,
        logprior=logprior,
        data=data,
        use_base_normal_correction=True,
    )

    tr = _run_model(model)
    theta = tr["theta"]["value"]

    expected = loglik(theta, data) + logprior(theta) - base_normal_logpdf(theta)
    actual = tr["target"]["fn"].log_factor

    assert np.isclose(float(actual), float(expected))


def test_model_accepts_python_float_loglik_and_logprior():
    model = make_blackbox_model(
        dim=2,
        loglik=lambda theta, data: 1.25,
        logprior=lambda theta: -0.75,
        data=None,
    )

    tr = _run_model(model)
    actual = tr["target"]["fn"].log_factor

    assert actual.shape == ()
    assert np.isclose(float(actual), 0.5)


def test_model_accepts_singleton_array_loglik_and_logprior():
    model = make_blackbox_model(
        dim=2,
        loglik=lambda theta, data: jnp.asarray([1.25]),
        logprior=lambda theta: jnp.asarray([-0.75]),
        data=None,
    )

    tr = _run_model(model)
    actual = tr["target"]["fn"].log_factor

    assert actual.shape == ()
    assert np.isclose(float(actual), 0.5)


def test_model_rejects_vector_loglik_output_when_executed():
    model = make_blackbox_model(
        dim=2,
        loglik=lambda theta, data: jnp.asarray([1.0, 2.0]),
        logprior=lambda theta: 0.0,
        data=None,
    )

    with pytest.raises(TypeError):
        _run_model(model)


def test_model_rejects_vector_logprior_output_when_executed():
    model = make_blackbox_model(
        dim=2,
        loglik=lambda theta, data: 0.0,
        logprior=lambda theta: jnp.asarray([1.0, 2.0]),
        data=None,
    )

    with pytest.raises(TypeError):
        _run_model(model)


def test_model_passes_data_object_to_loglik():
    seen = {}

    def loglik(theta, data):
        seen["data"] = data
        return data["constant"]

    def logprior(theta):
        return 0.0

    data = {"constant": jnp.asarray(2.5)}
    model = make_blackbox_model(
        dim=1,
        loglik=loglik,
        logprior=logprior,
        data=data,
    )

    tr = _run_model(model)

    assert seen["data"] is data
    assert np.isclose(float(tr["target"]["fn"].log_factor), 2.5)


def test_model_is_reproducible_for_fixed_seed():
    model = make_blackbox_model(
        dim=3,
        loglik=lambda theta, data: jnp.sum(theta),
        logprior=lambda theta: -jnp.sum(theta**2),
        data=None,
    )

    tr1 = _run_model(model)
    tr2 = _run_model(model)

    assert np.allclose(np.asarray(tr1["theta"]["value"]), np.asarray(tr2["theta"]["value"]))
    assert np.isclose(
        float(tr1["target"]["fn"].log_factor),
        float(tr2["target"]["fn"].log_factor),
    )


def test_model_can_be_used_with_numpyro_log_density():
    from numpyro.infer.util import log_density

    dim = 2
    data = {"y": jnp.asarray([1.0, -1.0], dtype=jnp.float32)}

    def loglik(theta, data):
        return -0.5 * jnp.sum((theta - data["y"]) ** 2)

    def logprior(theta):
        return -0.5 * jnp.sum(theta**2)

    model = make_blackbox_model(
        dim=dim,
        loglik=loglik,
        logprior=logprior,
        data=data,
        use_base_normal_correction=True,
    )

    theta_value = jnp.asarray([0.2, -0.4], dtype=jnp.float32)
    log_joint, _ = log_density(
        model,
        model_args=(),
        model_kwargs={},
        params={"theta": theta_value},
    )

    # With base-normal correction, the Normal(0,I) sample-site contribution
    # cancels out and the log density equals loglik + logprior.
    expected = loglik(theta_value, data) + logprior(theta_value)

    assert np.isclose(float(log_joint), float(expected), atol=1e-6)


def test_model_log_density_without_correction_includes_base_normal_twice_if_prior_is_absolute():
    from numpyro.infer.util import log_density

    dim = 2

    def loglik(theta, data):
        return -0.5 * jnp.sum(theta**2)

    def absolute_logprior(theta):
        return -0.25 * jnp.sum(theta**2)

    model = make_blackbox_model(
        dim=dim,
        loglik=loglik,
        logprior=absolute_logprior,
        data=None,
        use_base_normal_correction=False,
    )

    theta_value = jnp.asarray([0.3, -0.7], dtype=jnp.float32)
    log_joint, _ = log_density(
        model,
        model_args=(),
        model_kwargs={},
        params={"theta": theta_value},
    )

    expected = (
        base_normal_logpdf(theta_value)
        + loglik(theta_value, None)
        + absolute_logprior(theta_value)
    )

    assert np.isclose(float(log_joint), float(expected), atol=1e-6)


def test_model_log_density_without_correction_is_exact_when_prior_is_corrected():
    from numpyro.infer.util import log_density

    dim = 2

    def loglik(theta, data):
        return -0.5 * jnp.sum(theta**2)

    def absolute_logprior(theta):
        return -0.25 * jnp.sum(theta**2)

    def corrected_logprior(theta):
        return absolute_logprior(theta) - base_normal_logpdf(theta)

    model = make_blackbox_model(
        dim=dim,
        loglik=loglik,
        logprior=corrected_logprior,
        data=None,
        use_base_normal_correction=False,
    )

    theta_value = jnp.asarray([0.3, -0.7], dtype=jnp.float32)
    log_joint, _ = log_density(
        model,
        model_args=(),
        model_kwargs={},
        params={"theta": theta_value},
    )

    expected = loglik(theta_value, None) + absolute_logprior(theta_value)

    assert np.isclose(float(log_joint), float(expected), atol=1e-6)


def test_target_factor_is_finite_for_finite_loglik_and_logprior():
    model = make_blackbox_model(
        dim=2,
        loglik=lambda theta, data: -jnp.sum(theta**2),
        logprior=lambda theta: -jnp.sum(theta**2),
        data=None,
    )

    tr = _run_model(model)

    assert np.isfinite(float(tr["target"]["fn"].log_factor))


def test_model_allows_infinite_factor_values():
    model = make_blackbox_model(
        dim=1,
        loglik=lambda theta, data: -jnp.inf,
        logprior=lambda theta: 0.0,
        data=None,
    )

    tr = _run_model(model)

    assert np.isneginf(float(tr["target"]["fn"].log_factor))