from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

import public_transportation.inference.pipeline as pipeline
from public_transportation.estimation.bayesian import VIConfig
from public_transportation.inference.od_parameter_layout import ODParameterLayout
from public_transportation.inference.model import ForwardModelOutputs
from public_transportation.measurement.mapping import AggregationSpec


def _spec(num_measurements: int = 1) -> AggregationSpec:
    return AggregationSpec(
        num_measurements=num_measurements,
        measurement_index=np.arange(num_measurements, dtype=np.int32),
        link_index=np.arange(num_measurements, dtype=np.int32),
    )


def _request(**overrides: Any) -> pipeline.ODThetaEstimationRequest:
    values: dict[str, Any] = {
        "fingerprint": "test-fingerprint",
        "f0": jnp.asarray([10.0, 20.0]),
        "y_obs": jnp.asarray([3.0]),
        "mapping_spec": _spec(),
        "baseline_theta": 2.0,
        "assignment_artifacts": object(),
        "vi": VIConfig(num_steps=3, num_posterior_draws=2, log_every=1),
    }
    values.update(overrides)
    return pipeline.ODThetaEstimationRequest(**values)


def _reduced_layout() -> ODParameterLayout:
    return ODParameterLayout(
        num_od_total=3,
        od_keys=(("a", "b", "t"), ("b", "c", "t"), ("c", "d", "t")),
        free_od_indices=(1,),
        fixed_od_indices=(0, 2),
        fixed_od_values=(0.0, 7.5),
        free_baseline_values=(20.0,),
        fixed_zero_indices=(0,),
        fixed_positive_indices=(2,),
    )


def test_forward_outputs_name_compact_demand_explicitly_with_compatibility_alias():
    demand = jnp.asarray([2.0, 7.5])
    outputs = ForwardModelOutputs(
        assignment_demand=demand,
        link_flow=jnp.asarray([3.0]),
        lambda_m=jnp.asarray([3.0]),
        mu_m=jnp.asarray([3.0]),
    )
    assert outputs.assignment_demand is demand
    assert outputs.f is demand


def _install_pipeline_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    samples: np.ndarray,
    likelihood: float = -1.25,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_build_assignment_inputs(*, artifacts: Any, compact_layout: Any = None) -> str:
        captured["artifacts"] = artifacts
        captured["compact_layout"] = compact_layout
        return "assignment-inputs"

    def fake_make_forward_inputs(*, f0: Any, spec: Any) -> str:
        captured["forward_f0"] = np.asarray(f0)
        captured["forward_spec"] = spec
        return "forward-inputs"

    def fake_forward_model(**kwargs: Any) -> SimpleNamespace:
        captured.setdefault("forward_calls", []).append(kwargs)
        return SimpleNamespace(link_flow=jnp.asarray([7.0]))

    def fake_loglikelihood_from_link_flow(**kwargs: Any) -> jnp.ndarray:
        captured.setdefault("likelihood_calls", []).append(kwargs)
        return jnp.asarray(likelihood)

    def fake_run_vi(**kwargs: Any) -> SimpleNamespace:
        captured["run_vi"] = kwargs
        return SimpleNamespace(posterior_samples_theta=np.asarray(samples, dtype=float))

    monkeypatch.setattr(pipeline, "build_assignment_inputs", fake_build_assignment_inputs)
    monkeypatch.setattr(
        pipeline,
        "prepare_fixed_routing",
        lambda **kwargs: captured.setdefault("fixed_routing", kwargs) or "routing",
    )
    monkeypatch.setattr(
        pipeline,
        "build_od_assignment_runtime_profile",
        lambda **_: SimpleNamespace(assignment_active_od=2),
    )
    monkeypatch.setattr(pipeline, "make_forward_inputs", fake_make_forward_inputs)
    monkeypatch.setattr(pipeline, "forward_model_from_demand", fake_forward_model)
    monkeypatch.setattr(pipeline, "loglikelihood_from_link_flow", fake_loglikelihood_from_link_flow)
    monkeypatch.setattr(pipeline, "run_vi", fake_run_vi)
    return captured


@pytest.mark.parametrize("bound", [0.5, 1.0, 6.0])
def test_smooth_bound_stays_within_bound_and_is_odd(bound: float):
    x = jnp.asarray([-100.0, -2.0, 0.0, 2.0, 100.0])
    actual = np.asarray(pipeline._smooth_bound(x, bound))

    # Mathematically tanh maps to an open interval. At extreme inputs, finite
    # precision can round the result exactly to either boundary.
    assert np.all(actual >= -bound)
    assert np.all(actual <= bound)
    assert actual[2] == pytest.approx(0.0)
    assert actual[0] == pytest.approx(-actual[-1])
    assert actual[1] == pytest.approx(-actual[-2])


def test_smooth_bound_matches_formula_and_numpy_implementation():
    x = np.asarray([-7.0, -1.0, 0.0, 1.0, 7.0])
    bound = 6.0
    expected = bound * np.tanh(x / bound)

    assert np.allclose(np.asarray(pipeline._smooth_bound(jnp.asarray(x), bound)), expected)
    assert np.allclose(pipeline._smooth_bound_numpy(x, bound), expected)


def test_smooth_bound_has_nonzero_gradient_beyond_old_clip_boundary():
    derivative = jax.grad(lambda value: pipeline._smooth_bound(value, 6.0))(7.0)
    assert float(derivative) > 0.0


def test_estimated_theta_uses_smooth_bounds_in_forward_and_postprocessing(monkeypatch):
    samples = np.asarray([[7.0, -7.0, 7.0], [-2.0, 3.0, -7.0]])
    captured = _install_pipeline_stubs(monkeypatch, samples=samples)

    result = pipeline.estimate_od_theta_vi(_request(z_clip=6.0, u_clip=6.0))
    run_args = captured["run_vi"]
    probe = jnp.asarray(samples[0])
    assert float(run_args["loglik"](probe, run_args["data"])) == pytest.approx(-1.25)

    call = captured["forward_calls"][0]
    expected_z = 6.0 * np.tanh(samples[0, :2] / 6.0)
    expected_u = 6.0 * np.tanh(samples[0, 2] / 6.0)
    assert np.allclose(
        np.asarray(call["f"]),
        np.asarray([10.0, 20.0]) * np.exp(expected_z),
    )
    assert float(call["theta"]) == pytest.approx(np.exp(expected_u))

    expected_z_samples = 6.0 * np.tanh(samples[:, :2] / 6.0)
    expected_theta = np.exp(6.0 * np.tanh(samples[:, 2] / 6.0))
    expected_f = np.asarray([10.0, 20.0])[None, :] * np.exp(expected_z_samples)
    assert np.allclose(result.theta_samples, expected_theta)
    assert np.allclose(result.f_samples, expected_f)
    assert result.theta_mean == pytest.approx(expected_theta.mean())
    assert result.theta_sd == pytest.approx(expected_theta.std(ddof=0))
    assert np.allclose(result.f_mean, expected_f.mean(axis=0))


def test_fixed_theta_is_not_transformed(monkeypatch):
    samples = np.asarray([[7.0, -7.0], [1.0, -1.0]])
    captured = _install_pipeline_stubs(monkeypatch, samples=samples)
    request = _request(estimate_theta=False, fixed_theta=3.5)

    result = pipeline.estimate_od_theta_vi(request)
    run_args = captured["run_vi"]
    run_args["loglik"](jnp.asarray(samples[0]), run_args["data"])

    assert run_args["dim"] == 2
    assert float(captured["forward_calls"][0]["theta"]) == pytest.approx(3.5)
    assert captured["fixed_routing"] == {
        "inputs": "assignment-inputs",
        "theta": 3.5,
    }
    assert captured["forward_calls"][0]["fixed_routing"] == captured["fixed_routing"]
    assert np.all(result.theta_samples == 3.5)
    assert result.theta_mean == pytest.approx(3.5)
    assert result.theta_sd == pytest.approx(0.0)
    assert result.fixed_theta == pytest.approx(3.5)
    assert result.estimate_theta is False


def test_reduced_layout_controls_dimension_forward_reconstruction_and_samples(monkeypatch):
    samples = np.asarray([[0.0, 0.25], [6.0, -0.5]])
    captured = _install_pipeline_stubs(monkeypatch, samples=samples)
    layout = _reduced_layout()
    result = pipeline.estimate_od_theta_vi(
        _request(
            f0=jnp.asarray([11.0, 20.0, 33.0]),
            od_layout=layout,
            estimate_theta=True,
        )
    )

    args = captured["run_vi"]
    assert args["dim"] == 2
    args["loglik"](jnp.asarray(samples[0]), args["data"])
    assert np.allclose(captured["forward_calls"][0]["f"], [20.0, 7.5])
    assert captured["compact_layout"].active_full_indices == (1, 2)

    effective_z = 6.0 * np.tanh(samples[:, :1] / 6.0)
    assert np.all(result.f_samples[:, 0] == 0.0)
    assert np.all(result.f_samples[:, 2] == 7.5)
    assert np.allclose(result.f_samples[:, 1], 20.0 * np.exp(effective_z[:, 0]))
    assert result.num_od == 3
    assert result.num_free_od == 1
    assert result.num_fixed_od == 2
    assert result.runtime_profile.assignment_active_od == 2
    assert result.od_layout_fingerprint == layout.fingerprint
    assert result.od_layout_payload_json == layout.fingerprint_payload_json
    assert result.compact_layout_fingerprint is not None
    assert result.compact_layout_payload_json is not None


def test_all_frozen_fixed_theta_has_zero_dimensional_estimation_problem(monkeypatch):
    layout = ODParameterLayout(
        num_od_total=2,
        od_keys=(("a", "b", "t"), ("b", "c", "t")),
        free_od_indices=(),
        fixed_od_indices=(0, 1),
        fixed_od_values=(0.0, 4.0),
        free_baseline_values=(),
        fixed_zero_indices=(0,),
        fixed_positive_indices=(1,),
    )
    captured = _install_pipeline_stubs(monkeypatch, samples=np.empty((2, 0)))
    result = pipeline.estimate_od_theta_vi(
        _request(
            f0=jnp.asarray([5.0, 6.0]),
            od_layout=layout,
            estimate_theta=False,
            fixed_theta=2.0,
        )
    )

    assert captured["run_vi"]["dim"] == 0
    captured["run_vi"]["loglik"](jnp.empty((0,)), captured["run_vi"]["data"])
    assert np.allclose(captured["forward_calls"][0]["f"], [4.0])
    assert np.allclose(result.f_samples, [[0.0, 4.0], [0.0, 4.0]])


def test_reduced_layout_must_match_full_size_and_free_baselines(monkeypatch):
    _install_pipeline_stubs(monkeypatch, samples=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="num_od_total"):
        pipeline.estimate_od_theta_vi(_request(od_layout=_reduced_layout()))

    with pytest.raises(ValueError, match="free baselines"):
        pipeline.estimate_od_theta_vi(
            _request(f0=jnp.asarray([11.0, 99.0, 33.0]), od_layout=_reduced_layout())
        )


def test_gaussian_prior_is_evaluated_on_raw_variables(monkeypatch):
    samples = np.zeros((2, 3))
    captured = _install_pipeline_stubs(monkeypatch, samples=samples)
    request = _request(
        sigma_z=1.7,
        sigma_u=0.4,
        baseline_theta=2.5,
        u_clip=6.0,
        vi=VIConfig(use_base_normal_correction=True, num_steps=1, num_posterior_draws=2),
    )
    pipeline.estimate_od_theta_vi(request)

    raw = jnp.asarray([8.0, -9.0, 0.25])
    effective_center = np.log(2.5)
    raw_center = 6.0 * np.arctanh(effective_center / 6.0)
    expected = dist.Normal(0.0, 1.7).log_prob(raw[:2]).sum()
    expected += dist.Normal(raw_center, 0.4).log_prob(raw[2])

    actual = captured["run_vi"]["logprior"](raw)
    assert float(actual) == pytest.approx(float(expected))


def test_theta_raw_prior_center_maps_exactly_to_baseline(monkeypatch):
    captured = _install_pipeline_stubs(monkeypatch, samples=np.zeros((2, 3)))
    baseline = 4.0
    bound = 6.0
    pipeline.estimate_od_theta_vi(_request(baseline_theta=baseline, u_clip=bound))

    raw_center = bound * np.arctanh(np.log(baseline) / bound)
    theta_vec = jnp.asarray([0.0, 0.0, raw_center])
    args = captured["run_vi"]
    args["loglik"](theta_vec, args["data"])

    assert float(captured["forward_calls"][0]["theta"]) == pytest.approx(baseline)


def test_logprior_returns_increment_when_base_correction_is_disabled(monkeypatch):
    captured = _install_pipeline_stubs(monkeypatch, samples=np.zeros((2, 2)))
    request = _request(
        estimate_theta=False,
        fixed_theta=2.0,
        sigma_z=2.0,
        vi=VIConfig(use_base_normal_correction=False, num_steps=1, num_posterior_draws=2),
    )
    pipeline.estimate_od_theta_vi(request)

    raw_z = jnp.asarray([1.5, -2.5])
    expected = dist.Normal(0.0, 2.0).log_prob(raw_z).sum()
    expected -= dist.Normal(0.0, 1.0).log_prob(raw_z).sum()
    actual = captured["run_vi"]["logprior"](raw_z)
    assert float(actual) == pytest.approx(float(expected))


def test_nonfinite_likelihood_is_converted_to_negative_infinity(monkeypatch):
    captured = _install_pipeline_stubs(
        monkeypatch,
        samples=np.zeros((2, 2)),
        likelihood=np.nan,
    )
    pipeline.estimate_od_theta_vi(_request(estimate_theta=False, fixed_theta=2.0))
    args = captured["run_vi"]
    actual = args["loglik"](jnp.zeros(2), args["data"])
    assert np.isneginf(float(actual))


def test_pipeline_forwards_vi_configuration_and_metadata(monkeypatch):
    samples = np.zeros((4, 2))
    captured = _install_pipeline_stubs(monkeypatch, samples=samples)
    logger = object()
    config = VIConfig(
        guide="auto_lowrank",
        lowrank_rank=2,
        use_base_normal_correction=True,
        num_steps=17,
        learning_rate=0.003,
        seed=42,
        num_posterior_draws=4,
        log_every=5,
    )
    request = _request(
        estimate_theta=False,
        fixed_theta=2.0,
        vi=config,
        logger=logger,
        fingerprint_payload_json='{"case":"test"}',
    )

    result = pipeline.estimate_od_theta_vi(request)
    args = captured["run_vi"]
    assert args["guide"] == "auto_lowrank"
    assert args["lowrank_rank"] == 2
    assert args["use_base_normal_correction"] is True
    assert args["num_steps"] == 17
    assert args["learning_rate"] == pytest.approx(0.003)
    assert args["seed"] == 42
    assert args["num_posterior_draws"] == 4
    assert args["logger"] is logger
    assert args["log_every"] == 5
    assert result.fingerprint == "test-fingerprint"
    assert result.fingerprint_payload_json == '{"case":"test"}'
    assert result.vi.posterior_samples_theta.shape == (4, 2)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"assignment_artifacts": None}, "assignment_artifacts must be provided"),
        ({"f0": jnp.ones((1, 2))}, "f0 must be 1D"),
        ({"y_obs": jnp.ones((1, 1))}, "y_obs must be 1D"),
        ({"y_obs": jnp.asarray([1.0, 2.0])}, "does not match spec.num_measurements"),
        ({"estimate_theta": True, "fixed_theta": 2.0}, "fixed_theta must be None"),
        ({"estimate_theta": False, "fixed_theta": None}, "fixed_theta must be provided"),
        ({"estimate_theta": False, "fixed_theta": 0.0}, "fixed_theta must be positive and finite"),
        ({"estimate_theta": False, "fixed_theta": np.inf}, "fixed_theta must be positive and finite"),
        ({"baseline_theta": 0.0}, "baseline_theta must be positive and finite"),
        ({"mu_u_strategy": "invalid"}, "Unknown mu_u_strategy"),
        ({"mu_u_strategy": "fixed", "mu_u_fixed": None}, "mu_u_fixed must be provided"),
        ({"sigma_z": 0.0}, "sigma_z must be positive and finite"),
        ({"sigma_u": np.nan}, "sigma_u must be positive and finite"),
        ({"z_clip": 0.0}, "z_clip must be positive and finite"),
        ({"u_clip": np.inf}, "u_clip must be positive and finite"),
        ({"baseline_theta": np.exp(6.0), "u_clip": 6.0}, "strictly inside"),
        ({"mu_u_strategy": "fixed", "mu_u_fixed": -6.0, "u_clip": 6.0}, "strictly inside"),
    ],
)
def test_request_validation(monkeypatch, overrides: dict[str, Any], message: str):
    _install_pipeline_stubs(monkeypatch, samples=np.zeros((2, 3)))
    with pytest.raises(ValueError, match=message):
        pipeline.estimate_od_theta_vi(_request(**overrides))


def test_rejects_unexpected_posterior_sample_shape(monkeypatch):
    _install_pipeline_stubs(monkeypatch, samples=np.zeros((2, 2)))
    with pytest.raises(RuntimeError, match="Unexpected posterior sample shape"):
        pipeline.estimate_od_theta_vi(_request(estimate_theta=True))


def test_zero_baseline_cells_remain_zero_after_postprocessing(monkeypatch):
    samples = np.asarray([[100.0, 100.0], [-100.0, -100.0]])
    _install_pipeline_stubs(monkeypatch, samples=samples)
    request = _request(
        f0=jnp.asarray([0.0, 10.0]),
        estimate_theta=False,
        fixed_theta=2.0,
    )
    result = pipeline.estimate_od_theta_vi(request)

    assert np.all(result.f_samples[:, 0] == 0.0)
    assert np.all(np.isfinite(result.f_samples))
    assert np.all(result.f_samples[:, 1] > 0.0)
