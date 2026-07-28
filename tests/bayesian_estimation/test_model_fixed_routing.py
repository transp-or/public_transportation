from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

import public_transportation.inference.model as model
from public_transportation.inference.model import ForwardModelInputs


def _forward_inputs() -> ForwardModelInputs:
    return ForwardModelInputs(
        f0=jnp.asarray([2.0]),
        num_measurements=1,
        measurement_index=jnp.asarray([0], dtype=jnp.int32),
        link_index=jnp.asarray([0], dtype=jnp.int32),
    )


def test_forward_model_dispatches_to_fixed_routing_when_provided(monkeypatch):
    captured = {}
    assignment_inputs = SimpleNamespace(od_origin_node=jnp.asarray([0]))
    routing = object()

    def cached(**kwargs):
        captured["cached"] = kwargs
        return jnp.asarray([4.0])

    monkeypatch.setattr(model, "assign_link_flow_fixed_routing", cached)
    monkeypatch.setattr(
        model,
        "assign_link_flow",
        lambda **_: (_ for _ in ()).throw(AssertionError("dynamic path called")),
    )

    result = model.forward_model_from_demand(
        inputs=_forward_inputs(),
        f=jnp.asarray([3.0]),
        theta=jnp.asarray(2.0),
        rho=jnp.asarray(0.5),
        assignment_inputs=assignment_inputs,
        fixed_routing=routing,
    )

    assert captured["cached"]["inputs"] is assignment_inputs
    assert captured["cached"]["routing"] is routing
    assert np.allclose(captured["cached"]["f"], [3.0])
    assert np.allclose(result.link_flow, [4.0])
    assert np.allclose(result.lambda_m, [4.0])
    assert np.allclose(result.mu_m, [2.0])


def test_forward_model_keeps_dynamic_path_without_fixed_routing(monkeypatch):
    captured = {}
    assignment_inputs = SimpleNamespace(od_origin_node=jnp.asarray([0]))

    def dynamic(**kwargs):
        captured["dynamic"] = kwargs
        return jnp.asarray([5.0])

    monkeypatch.setattr(model, "assign_link_flow", dynamic)
    monkeypatch.setattr(
        model,
        "assign_link_flow_fixed_routing",
        lambda **_: (_ for _ in ()).throw(AssertionError("cached path called")),
    )

    result = model.forward_model_from_demand(
        inputs=_forward_inputs(),
        f=jnp.asarray([3.0]),
        theta=jnp.asarray(2.0),
        rho=jnp.asarray(1.0),
        assignment_inputs=assignment_inputs,
    )

    assert captured["dynamic"]["inputs"] is assignment_inputs
    assert float(captured["dynamic"]["theta"]) == 2.0
    assert np.allclose(result.link_flow, [5.0])
