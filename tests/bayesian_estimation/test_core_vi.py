# tests/test_core_vi.py
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.estimation.bayesian.core_vi import run_vi


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_steps": 0}, "num_steps must be positive"),
        ({"num_steps": -1}, "num_steps must be positive"),
        ({"learning_rate": 0.0}, "learning_rate must be positive"),
        ({"learning_rate": -1.0}, "learning_rate must be positive"),
        ({"num_posterior_draws": 0}, "num_posterior_draws must be positive"),
        ({"num_posterior_draws": -3}, "num_posterior_draws must be positive"),
        ({"log_every": 0}, "log_every must be positive"),
        ({"log_every": -10}, "log_every must be positive"),
    ],
)
def test_run_vi_validates_positive_arguments(kwargs, message):
    def loglik(theta, data):
        return -jnp.sum(theta**2)

    def logprior(theta):
        return 0.0

    call_kwargs = dict(
        dim=2,
        data={},
        loglik=loglik,
        logprior=logprior,
        num_steps=2,
        learning_rate=1e-2,
        num_posterior_draws=3,
        log_every=1,
    )
    call_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        run_vi(**call_kwargs)


def test_run_vi_returns_result_with_expected_shapes():
    def loglik(theta, data):
        target = data["target"]
        return -0.5 * jnp.sum((theta - target) ** 2)

    def logprior(theta):
        return -0.5 * jnp.sum(theta**2)

    result = run_vi(
        dim=2,
        data={"target": jnp.asarray([1.0, -1.0])},
        loglik=loglik,
        logprior=logprior,
        guide="auto_diag",
        num_steps=5,
        learning_rate=1e-2,
        seed=123,
        num_posterior_draws=7,
        log_every=2,
    )

    assert result.guide == "auto_diag"
    assert result.dim == 2
    assert result.seed == 123
    assert result.num_steps == 5
    assert result.learning_rate == pytest.approx(1e-2)
    assert result.num_posterior_draws == 7

    assert result.losses.shape == (5,)
    assert result.posterior_samples_theta.shape == (7, 2)
    assert result.posterior_mean.shape == (2,)
    assert result.posterior_sd.shape == (2,)
    assert result.posterior_q05.shape == (2,)
    assert result.posterior_q50.shape == (2,)
    assert result.posterior_q95.shape == (2,)

    assert np.all(np.isfinite(result.losses))
    assert np.all(np.isfinite(result.posterior_samples_theta))
    assert np.all(np.isfinite(result.posterior_mean))
    assert np.all(np.isfinite(result.posterior_sd))
    assert result.runtime_seconds >= 0.0
    assert isinstance(result.timestamp, str)
    assert len(result.timestamp) >= 19


def test_run_vi_is_reproducible_for_same_seed():
    def loglik(theta, data):
        return -0.5 * jnp.sum((theta - data["target"]) ** 2)

    def logprior(theta):
        return -0.5 * jnp.sum(theta**2)

    kwargs = dict(
        dim=2,
        data={"target": jnp.asarray([0.5, -0.25])},
        loglik=loglik,
        logprior=logprior,
        guide="auto_diag",
        num_steps=5,
        learning_rate=1e-2,
        seed=77,
        num_posterior_draws=6,
    )

    result_1 = run_vi(**kwargs)
    result_2 = run_vi(**kwargs)

    assert np.allclose(result_1.losses, result_2.losses)
    assert np.allclose(result_1.posterior_samples_theta, result_2.posterior_samples_theta)
    assert np.allclose(result_1.posterior_mean, result_2.posterior_mean)
    assert np.allclose(result_1.posterior_sd, result_2.posterior_sd)


def test_run_vi_different_seeds_give_different_posterior_draws():
    def loglik(theta, data):
        return -0.5 * jnp.sum(theta**2)

    def logprior(theta):
        return 0.0

    common = dict(
        dim=2,
        data={},
        loglik=loglik,
        logprior=logprior,
        guide="auto_diag",
        num_steps=4,
        learning_rate=1e-2,
        num_posterior_draws=10,
    )

    result_1 = run_vi(**common, seed=1)
    result_2 = run_vi(**common, seed=2)

    assert not np.allclose(
        result_1.posterior_samples_theta,
        result_2.posterior_samples_theta,
    )


def test_run_vi_computes_summary_statistics_from_samples():
    def loglik(theta, data):
        return -0.5 * jnp.sum(theta**2)

    def logprior(theta):
        return 0.0

    result = run_vi(
        dim=3,
        data={},
        loglik=loglik,
        logprior=logprior,
        guide="auto_diag",
        num_steps=4,
        learning_rate=1e-2,
        seed=12,
        num_posterior_draws=20,
    )

    samples = result.posterior_samples_theta

    assert np.allclose(result.posterior_mean, np.mean(samples, axis=0))
    assert np.allclose(result.posterior_sd, np.std(samples, axis=0, ddof=1))
    assert np.allclose(result.posterior_q05, np.quantile(samples, 0.05, axis=0))
    assert np.allclose(result.posterior_q50, np.quantile(samples, 0.50, axis=0))
    assert np.allclose(result.posterior_q95, np.quantile(samples, 0.95, axis=0))


def test_run_vi_accepts_base_normal_correction_flag():
    def loglik(theta, data):
        return -0.5 * jnp.sum(theta**2)

    def logprior(theta):
        return -0.5 * jnp.sum(theta**2)

    result = run_vi(
        dim=2,
        data={},
        loglik=loglik,
        logprior=logprior,
        use_base_normal_correction=True,
        num_steps=3,
        learning_rate=1e-2,
        num_posterior_draws=5,
    )

    assert result.use_base_normal_correction is True


def test_run_vi_supports_auto_normal_guide():
    def loglik(theta, data):
        return -0.5 * jnp.sum(theta**2)

    def logprior(theta):
        return 0.0

    result = run_vi(
        dim=2,
        data={},
        loglik=loglik,
        logprior=logprior,
        guide="auto_normal",
        num_steps=3,
        learning_rate=1e-2,
        num_posterior_draws=5,
    )

    assert result.guide == "auto_normal"
    assert result.posterior_samples_theta.shape == (5, 2)


def test_run_vi_supports_auto_lowrank_guide():
    def loglik(theta, data):
        return -0.5 * jnp.sum(theta**2)

    def logprior(theta):
        return 0.0

    result = run_vi(
        dim=3,
        data={},
        loglik=loglik,
        logprior=logprior,
        guide="auto_lowrank",
        lowrank_rank=1,
        num_steps=3,
        learning_rate=1e-2,
        num_posterior_draws=5,
    )

    assert result.guide == "auto_lowrank"
    assert result.lowrank_rank == 1
    assert result.posterior_samples_theta.shape == (5, 3)


class _Logger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, tuple[object, ...]]] = []

    def info(self, msg: str, *args: object) -> None:
        self.messages.append((msg, args))


def test_run_vi_logs_progress_at_requested_frequency():
    def loglik(theta, data):
        return -0.5 * jnp.sum(theta**2)

    def logprior(theta):
        return 0.0

    logger = _Logger()

    run_vi(
        dim=2,
        data={},
        loglik=loglik,
        logprior=logprior,
        guide="auto_diag",
        num_steps=5,
        learning_rate=1e-2,
        num_posterior_draws=4,
        logger=logger,
        log_every=2,
    )

    # Logs at steps 1, 3, 5 because step is zero-based and the last step is always logged.
    assert len(logger.messages) == 3

    logged_completed_steps = [args[0] for _, args in logger.messages]
    assert logged_completed_steps == [1, 3, 5]

    for msg, args in logger.messages:
        assert "VI step %d/%d" in msg
        assert args[1] == 5
        assert isinstance(args[2], float)


def test_run_vi_logs_last_step_even_when_not_multiple_of_log_every():
    def loglik(theta, data):
        return -0.5 * jnp.sum(theta**2)

    def logprior(theta):
        return 0.0

    logger = _Logger()

    run_vi(
        dim=1,
        data={},
        loglik=loglik,
        logprior=logprior,
        num_steps=4,
        learning_rate=1e-2,
        num_posterior_draws=3,
        logger=logger,
        log_every=10,
    )

    assert len(logger.messages) == 2
    assert [args[0] for _, args in logger.messages] == [1, 4]


def test_run_vi_without_logger_does_not_fail():
    def loglik(theta, data):
        return -0.5 * jnp.sum(theta**2)

    def logprior(theta):
        return 0.0

    result = run_vi(
        dim=1,
        data={},
        loglik=loglik,
        logprior=logprior,
        num_steps=3,
        learning_rate=1e-2,
        num_posterior_draws=3,
        logger=None,
    )

    assert result.losses.shape == (3,)


def test_run_vi_passes_data_to_loglik():
    def loglik(theta, data):
        return -0.5 * jnp.sum((theta - data["target"]) ** 2)

    def logprior(theta):
        return 0.0

    result = run_vi(
        dim=1,
        data={"target": jnp.asarray([2.0])},
        loglik=loglik,
        logprior=logprior,
        num_steps=5,
        learning_rate=1e-2,
        num_posterior_draws=5,
    )

    assert result.posterior_samples_theta.shape == (5, 1)


def test_run_vi_rejects_invalid_guide_name():
    def loglik(theta, data):
        return -0.5 * jnp.sum(theta**2)

    def logprior(theta):
        return 0.0

    with pytest.raises((ValueError, KeyError, NotImplementedError)):
        run_vi(
            dim=1,
            data={},
            loglik=loglik,
            logprior=logprior,
            guide="not_a_guide",  # type: ignore[arg-type]
            num_steps=2,
            num_posterior_draws=2,
        )