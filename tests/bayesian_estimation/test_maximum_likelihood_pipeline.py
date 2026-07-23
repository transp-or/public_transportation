from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import public_transportation.inference.maximum_likelihood_pipeline as ml_pipeline
from public_transportation.inference.od_parameter_layout import ODParameterLayout
from public_transportation.inference.pipeline import ODThetaEstimationRequest
from public_transportation.measurement.mapping import AggregationSpec


def _layout() -> ODParameterLayout:
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


def _request(**overrides: Any) -> ODThetaEstimationRequest:
    values = dict(
        fingerprint="test",
        f0=jnp.asarray([11.0, 20.0, 33.0]),
        y_obs=jnp.asarray([3.0]),
        mapping_spec=AggregationSpec(
            num_measurements=1,
            measurement_index=np.asarray([0], dtype=np.int32),
            link_index=np.asarray([0], dtype=np.int32),
        ),
        baseline_theta=2.0,
        od_layout=_layout(),
        assignment_artifacts=object(),
    )
    values.update(overrides)
    return ODThetaEstimationRequest(**values)


def _stub_forward_dependencies(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(ml_pipeline, "build_assignment_inputs", lambda **_: "assignment")
    monkeypatch.setattr(ml_pipeline, "make_forward_inputs", lambda **_: SimpleNamespace())
    monkeypatch.setattr(ml_pipeline, "prepare_likelihood_inputs", lambda **_: "prepared")
    monkeypatch.setattr(
        ml_pipeline,
        "build_od_assignment_runtime_profile",
        lambda **_: SimpleNamespace(assignment_active_od=2),
    )

    def forward(**kwargs: Any) -> SimpleNamespace:
        captured["forward"] = kwargs
        return SimpleNamespace(link_flow=jnp.asarray([4.0]))

    monkeypatch.setattr(ml_pipeline, "forward_model_from_demand", forward)
    monkeypatch.setattr(
        ml_pipeline,
        "loglikelihood_from_link_flow",
        lambda **_: jnp.asarray(-1.25),
    )
    return captured


@pytest.mark.parametrize("estimate_theta", [False, True])
def test_problem_dimension_depends_only_on_free_cells_and_theta(monkeypatch, estimate_theta):
    _stub_forward_dependencies(monkeypatch)
    problem = ml_pipeline.build_od_theta_ml_problem(
        _request(
            estimate_theta=estimate_theta,
            fixed_theta=(None if estimate_theta else 3.0),
        )
    )
    assert problem.dim == 1 + int(estimate_theta)
    assert problem.theta0.shape == (problem.dim,)
    assert problem.num_free_od == 1
    assert problem.num_fixed_od == 2
    assert problem.runtime_profile.assignment_active_od == 2
    assert problem.od_layout_fingerprint == _layout().fingerprint
    assert problem.od_layout_payload_json == _layout().fingerprint_payload_json
    assert problem.compact_layout_fingerprint is not None
    assert problem.compact_layout_payload_json is not None


def test_forward_and_decode_keep_frozen_values_exact(monkeypatch):
    captured = _stub_forward_dependencies(monkeypatch)
    problem = ml_pipeline.build_od_theta_ml_problem(
        _request(estimate_theta=False, fixed_theta=3.0)
    )
    parameter = jnp.asarray([6.0])
    assert float(problem.loglik(parameter, problem.data)) == pytest.approx(-1.25)
    expected_free = 20.0 * np.exp(6.0 * np.tanh(1.0))
    assert np.allclose(captured["forward"]["f"], [expected_free, 7.5])

    f_hat, theta_hat = problem.decode(np.asarray([6.0]))
    assert np.allclose(f_hat, [0.0, expected_free, 7.5])
    assert theta_hat == pytest.approx(3.0)


def test_prior_contains_only_free_od_and_optional_theta(monkeypatch):
    _stub_forward_dependencies(monkeypatch)
    fixed_theta_problem = ml_pipeline.build_od_theta_ml_problem(
        _request(estimate_theta=False, fixed_theta=3.0, sigma_z=2.0)
    )
    assert np.ndim(fixed_theta_problem.logprior(jnp.asarray([0.5]))) == 0


def test_rejects_layout_baseline_mismatch(monkeypatch):
    _stub_forward_dependencies(monkeypatch)
    with pytest.raises(ValueError, match="free baselines"):
        ml_pipeline.build_od_theta_ml_problem(
            _request(f0=jnp.asarray([11.0, 99.0, 33.0]))
        )
