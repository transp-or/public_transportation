from __future__ import annotations

import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import public_transportation.estimation.maximum_likelihood.core as ml_core
from public_transportation.estimation.maximum_likelihood import (
    compile_ml_objective,
    prepare_ml_objective,
    run_ml,
)
from public_transportation.estimation.maximum_likelihood.persistence import (
    load_ml_result,
    save_ml_result,
)


def _loglik(theta, data):
    residual = theta - jnp.asarray(data["target"])
    return -0.5 * jnp.sum(residual**2)


def _logprior(theta):
    return -0.5 * jnp.sum(theta**2)


def _result(x, *, nit=1, nfev=1, njev=1):
    return SimpleNamespace(
        x=np.array(x, dtype=float, copy=True),
        success=True,
        message="test optimizer",
        nit=nit,
        nfev=nfev,
        njev=njev,
        hess_inv=np.eye(len(x)),
    )


def test_callback_and_final_diagnostics_reuse_cached_evaluation_and_copy_theta(
    monkeypatch,
):
    def fake_minimize(*, fun, x0, callback, **kwargs):
        del x0, kwargs
        optimizer_buffer = np.asarray([1.0])
        fun(optimizer_buffer)
        optimizer_buffer[:] = 9.0
        callback(np.asarray([1.0]))
        return _result([1.0])

    monkeypatch.setattr(ml_core, "minimize", fake_minimize)
    result = run_ml(dim=1, data={"target": [2.0]}, loglik=_loglik)

    assert result.num_compiled_evaluations == 1
    assert result.objective_value == pytest.approx(0.5)
    assert result.loglikelihood == pytest.approx(-0.5)
    assert result.logprior == pytest.approx(0.0)
    assert np.allclose(result.gradient, [-1.0])
    assert np.allclose(result.optimization_trace, [[0.0, 0.5, 1.0]])


def test_callback_cache_miss_evaluates_once(monkeypatch):
    def fake_minimize(*, fun, callback, **kwargs):
        del kwargs
        fun(np.asarray([0.0]))
        callback(np.asarray([1.0]))
        return _result([1.0], nfev=1)

    monkeypatch.setattr(ml_core, "minimize", fake_minimize)
    result = run_ml(dim=1, data={"target": [2.0]}, loglik=_loglik)

    assert result.num_compiled_evaluations == 2


def test_final_result_cache_miss_evaluates_once(monkeypatch):
    def fake_minimize(*, fun, **kwargs):
        del kwargs
        fun(np.asarray([0.0]))
        return _result([1.0], nfev=1)

    monkeypatch.setattr(ml_core, "minimize", fake_minimize)
    result = run_ml(dim=1, data={"target": [2.0]}, loglik=_loglik)

    assert result.num_compiled_evaluations == 2
    assert result.objective_value == pytest.approx(0.5)


@pytest.mark.parametrize("prior_weight", [0.0, 0.25, 1.0, 2.0])
def test_auxiliary_diagnostics_and_objective_respect_prior_weight(
    monkeypatch, prior_weight
):
    def fake_minimize(*, fun, **kwargs):
        del kwargs
        fun(np.asarray([2.0]))
        return _result([2.0])

    monkeypatch.setattr(ml_core, "minimize", fake_minimize)
    result = run_ml(
        dim=1,
        data={"target": [3.0]},
        loglik=_loglik,
        logprior=_logprior,
        prior_weight=prior_weight,
    )

    assert result.loglikelihood == pytest.approx(-0.5)
    assert result.logprior == pytest.approx(-2.0)
    assert result.objective_value == pytest.approx(0.5 + 2.0 * prior_weight)
    assert np.allclose(result.gradient, [-1.0 + 2.0 * prior_weight])


def test_logprior_none_reports_zero_auxiliary_value(monkeypatch):
    def fake_minimize(*, fun, **kwargs):
        del kwargs
        fun(np.asarray([2.0]))
        return _result([2.0])

    monkeypatch.setattr(ml_core, "minimize", fake_minimize)
    result = run_ml(
        dim=1,
        data={"target": [3.0]},
        loglik=_loglik,
        logprior=None,
        prior_weight=4.0,
    )

    assert result.logprior == 0.0
    assert result.objective_value == pytest.approx(0.5)


def test_hessian_behavior_is_preserved():
    result = run_ml(
        dim=2,
        data={"target": [1.0, -2.0]},
        loglik=_loglik,
        logprior=_logprior,
        prior_weight=0.5,
        compute_hessian=True,
        maxiter=50,
    )

    assert result.success
    assert np.allclose(result.theta_hat, [2.0 / 3.0, -4.0 / 3.0], atol=1e-5)
    assert np.allclose(result.hessian, 1.5 * np.eye(2), atol=1e-6)
    assert np.allclose(result.covariance_matrix, np.eye(2) / 1.5, atol=1e-6)
    assert result.num_compiled_evaluations <= result.num_function_evaluations + 1


def test_compiled_evaluation_count_survives_persistence(tmp_path):
    result = run_ml(
        dim=1,
        data={"target": [1.0]},
        loglik=_loglik,
        maxiter=20,
    )
    save_ml_result(result, tmp_path)

    loaded = load_ml_result(tmp_path)
    assert loaded.num_compiled_evaluations == result.num_compiled_evaluations


def test_loader_defaults_old_results_to_scipy_function_count(tmp_path):
    result = run_ml(dim=0, data={"target": []}, loglik=_loglik)
    save_ml_result(result, tmp_path)
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("num_compiled_evaluations")
    metadata_path.write_text(json.dumps(metadata))

    loaded = load_ml_result(tmp_path)
    assert loaded.num_compiled_evaluations == result.num_function_evaluations


def test_precompiled_objective_is_reused_by_run_ml(monkeypatch):
    prepared = prepare_ml_objective(
        theta_example=jnp.zeros((1,)),
        data={"target": jnp.asarray([2.0])},
        loglik=_loglik,
    )
    compiled = compile_ml_objective(prepared)
    calls = 0

    def counted(parameter, data):
        nonlocal calls
        calls += 1
        return compiled.callable(parameter, data)

    def fake_minimize(*, fun, callback, **kwargs):
        del kwargs
        fun(np.asarray([0.0]))
        callback(np.asarray([1.0]))
        return _result([1.0])

    monkeypatch.setattr(ml_core, "minimize", fake_minimize)
    result = run_ml(
        dim=1,
        data=prepared.data,
        loglik=_loglik,
        compiled_objective=counted,
    )

    assert calls == result.num_compiled_evaluations == 2


def test_dynamic_values_and_optimizer_settings_do_not_retrace(monkeypatch):
    traces = 0

    def tracing_loglik(theta, data):
        nonlocal traces
        traces += 1
        return -0.5 * jnp.sum((theta - data["target"]) ** 2)

    prepared = prepare_ml_objective(
        theta_example=jnp.zeros((2,)),
        data={"target": jnp.asarray([1.0, 2.0])},
        loglik=tracing_loglik,
    )
    compiled = compile_ml_objective(prepared, execute_first=False)
    first = compiled.callable(
        jnp.asarray([0.0, 0.0]), {"target": jnp.asarray([1.0, 2.0])}
    )
    second = compiled.callable(
        jnp.asarray([3.0, 4.0]), {"target": jnp.asarray([2.0, 1.0])}
    )
    jax.block_until_ready((first, second))

    def fake_minimize(*, fun, x0, **kwargs):
        del kwargs
        fun(x0)
        return _result(x0)

    monkeypatch.setattr(ml_core, "minimize", fake_minimize)
    for maxiter in (1, 5, 20):
        run_ml(
            dim=2,
            data=prepared.data,
            loglik=tracing_loglik,
            maxiter=maxiter,
            compute_hessian=False,
            compiled_objective=compiled,
        )

    assert traces == 1
