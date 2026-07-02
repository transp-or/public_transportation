# tests/assignment/test_factory.py
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.assignment.assign import (
    AssignmentArtifacts,
    AssignmentResult,
)
from public_transportation.assignment.config import AssignmentConfig
from public_transportation.assignment.factory import (
    AssignmentFactory,
    build_assignment_factory,
)


@dataclass(frozen=True)
class _DummyGraph:
    num_nodes: int
    num_links: int


@dataclass(frozen=True)
class _DummyCostParts:
    pass


@dataclass(frozen=True)
class _DummyODGroups:
    group_dest_node: jnp.ndarray
    group_link_mask: jnp.ndarray
    od_origin_node: jnp.ndarray
    group_od_index_padded: jnp.ndarray
    group_od_mask: jnp.ndarray


@dataclass(frozen=True)
class _DummyIDManager:
    marker: str = "id-manager"


def _mk_artifacts(*, config: AssignmentConfig | None = None) -> AssignmentArtifacts:
    graph = _DummyGraph(num_nodes=4, num_links=3)
    od_groups = _DummyODGroups(
        group_dest_node=jnp.asarray([2, 3], dtype=jnp.int32),
        group_link_mask=jnp.asarray(
            [
                [True, True, False],
                [True, False, True],
            ],
            dtype=jnp.bool_,
        ),
        od_origin_node=jnp.asarray([0, 1], dtype=jnp.int32),
        group_od_index_padded=jnp.asarray([[0], [1]], dtype=jnp.int32),
        group_od_mask=jnp.asarray([[True], [True]], dtype=jnp.bool_),
    )
    return AssignmentArtifacts(
        graph=graph,  # type: ignore[arg-type]
        od_groups=od_groups,  # type: ignore[arg-type]
        cost_parts=_DummyCostParts(),  # type: ignore[arg-type]
        config=config or AssignmentConfig(theta_default=5.0),
    )


def test_factory_run_delegates_to_assign(monkeypatch):
    artifacts = _mk_artifacts()
    id_manager = _DummyIDManager()
    base_link_cost = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)

    seen: dict[str, Any] = {}

    def _assign(*, od_values, artifacts, theta, return_group_link_flows):
        seen["od_values"] = od_values
        seen["artifacts"] = artifacts
        seen["theta"] = theta
        seen["return_group_link_flows"] = return_group_link_flows
        return AssignmentResult(
            theta=7.0,
            link_flow=jnp.asarray([10.0, 20.0, 30.0], dtype=jnp.float32),
            link_cost=base_link_cost,
            group_link_flow=None,
        )

    monkeypatch.setattr("public_transportation.assignment.factory.assign", _assign)

    factory = AssignmentFactory(
        artifacts=artifacts,
        id_manager=id_manager,  # type: ignore[arg-type]
        base_link_cost=base_link_cost,
        link_flow_fn=lambda od_values, theta: jnp.asarray([0.0]),
        link_flow_and_group_fn=lambda od_values, theta: (
            jnp.asarray([0.0]),
            jnp.asarray([[0.0]]),
        ),
    )

    result = factory.run(
        od_values=[1.0, 2.0],
        theta=3.5,
        return_group_link_flows=True,
    )

    assert seen["artifacts"] is artifacts
    assert seen["theta"] == pytest.approx(3.5)
    assert seen["return_group_link_flows"] is True
    assert np.allclose(np.asarray(seen["od_values"]), [1.0, 2.0])
    assert result.theta == pytest.approx(7.0)
    assert np.allclose(np.asarray(result.link_flow), [10.0, 20.0, 30.0])


def test_build_assignment_factory_constructs_artifacts_id_manager_and_base_cost(monkeypatch):
    scenario = SimpleNamespace(name="scenario")
    config = AssignmentConfig(theta_default=4.0)
    artifacts = _mk_artifacts(config=config)
    id_manager = _DummyIDManager()
    base_cost = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)

    seen: dict[str, Any] = {}

    def _prepare_assignment(scenario, config):
        seen["prepare_scenario"] = scenario
        seen["prepare_config"] = config
        return artifacts

    class _AssignmentIDManager:
        @staticmethod
        def build(*, scenario, graph):
            seen["id_manager_scenario"] = scenario
            seen["id_manager_graph"] = graph
            return id_manager

    def _link_costs(*, graph, cost_parts, config):
        seen["link_costs_graph"] = graph
        seen["link_costs_cost_parts"] = cost_parts
        seen["link_costs_config"] = config
        return base_cost

    monkeypatch.setattr(
        "public_transportation.assignment.factory.prepare_assignment",
        _prepare_assignment,
    )
    monkeypatch.setattr(
        "public_transportation.assignment.factory.AssignmentIDManager",
        _AssignmentIDManager,
    )
    monkeypatch.setattr(
        "public_transportation.assignment.factory.link_costs",
        _link_costs,
    )
    monkeypatch.setattr(
        "public_transportation.assignment.factory._assign_core",
        lambda **kw: (jnp.zeros((artifacts.graph.num_links,)), None),
    )

    factory = build_assignment_factory(scenario=scenario, config=config)

    assert factory.artifacts is artifacts
    assert factory.id_manager is id_manager
    assert np.allclose(np.asarray(factory.base_link_cost), np.asarray(base_cost))

    assert seen["prepare_scenario"] is scenario
    assert seen["prepare_config"] is config
    assert seen["id_manager_scenario"] is scenario
    assert seen["id_manager_graph"] is artifacts.graph
    assert seen["link_costs_graph"] is artifacts.graph
    assert seen["link_costs_cost_parts"] is artifacts.cost_parts
    assert seen["link_costs_config"] is artifacts.config


def test_link_flow_fn_calls_assign_core_with_expected_arrays(monkeypatch):
    scenario = SimpleNamespace(name="scenario")
    config = AssignmentConfig(theta_default=4.0)
    artifacts = _mk_artifacts(config=config)
    base_cost = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)

    seen: dict[str, Any] = {}

    def _assign_core(**kw):
        seen.update(kw)
        od_values = kw["od_values"]
        theta = kw["theta"]
        total = kw["base_link_cost"] + jnp.sum(od_values) + theta
        return total, None

    monkeypatch.setattr(
        "public_transportation.assignment.factory.prepare_assignment",
        lambda scenario, config: artifacts,
    )
    monkeypatch.setattr(
        "public_transportation.assignment.factory.AssignmentIDManager",
        SimpleNamespace(build=lambda *, scenario, graph: _DummyIDManager()),
    )
    monkeypatch.setattr(
        "public_transportation.assignment.factory.link_costs",
        lambda *, graph, cost_parts, config: base_cost,
    )
    monkeypatch.setattr(
        "public_transportation.assignment.factory._assign_core",
        _assign_core,
    )

    factory = build_assignment_factory(scenario=scenario, config=config)

    od_values = jnp.asarray([5.0, 7.0], dtype=jnp.float32)
    result = factory.link_flow_fn(od_values, theta=2.0)

    assert np.allclose(np.asarray(result), [15.0, 16.0, 17.0])
    assert seen["graph"] is artifacts.graph
    assert seen["od_values"].shape == (2,)
    assert seen["base_link_cost"].shape == (3,)
    assert seen["theta"].shape == ()
    assert np.array_equal(
        np.asarray(seen["group_dest_node"]),
        np.asarray(artifacts.od_groups.group_dest_node),
    )
    assert np.array_equal(
        np.asarray(seen["group_link_mask"]),
        np.asarray(artifacts.od_groups.group_link_mask),
    )
    assert np.array_equal(
        np.asarray(seen["od_origin_node"]),
        np.asarray(artifacts.od_groups.od_origin_node),
    )
    assert np.array_equal(
        np.asarray(seen["group_od_index_padded"]),
        np.asarray(artifacts.od_groups.group_od_index_padded),
    )
    assert np.array_equal(
        np.asarray(seen["group_od_mask"]),
        np.asarray(artifacts.od_groups.group_od_mask),
    )
    assert seen["return_group_link_flows"] is False


def test_link_flow_and_group_fn_returns_total_and_per_group(monkeypatch):
    scenario = SimpleNamespace(name="scenario")
    config = AssignmentConfig(theta_default=4.0)
    artifacts = _mk_artifacts(config=config)
    base_cost = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)

    total = jnp.asarray([10.0, 20.0, 30.0], dtype=jnp.float32)
    per_group = jnp.asarray(
        [
            [1.0, 2.0, 3.0],
            [9.0, 18.0, 27.0],
        ],
        dtype=jnp.float32,
    )
    seen: dict[str, Any] = {}

    def _assign_core(**kw):
        seen.update(kw)
        return total, per_group

    monkeypatch.setattr(
        "public_transportation.assignment.factory.prepare_assignment",
        lambda scenario, config: artifacts,
    )
    monkeypatch.setattr(
        "public_transportation.assignment.factory.AssignmentIDManager",
        SimpleNamespace(build=lambda *, scenario, graph: _DummyIDManager()),
    )
    monkeypatch.setattr(
        "public_transportation.assignment.factory.link_costs",
        lambda *, graph, cost_parts, config: base_cost,
    )
    monkeypatch.setattr(
        "public_transportation.assignment.factory._assign_core",
        _assign_core,
    )

    factory = build_assignment_factory(scenario=scenario, config=config)

    result_total, result_per_group = factory.link_flow_and_group_fn(
        jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        theta=3.0,
    )

    assert np.allclose(np.asarray(result_total), np.asarray(total))
    assert np.allclose(np.asarray(result_per_group), np.asarray(per_group))
    assert seen["return_group_link_flows"] is True


def test_factory_wrappers_convert_python_inputs_to_jax_arrays(monkeypatch):
    scenario = SimpleNamespace(name="scenario")
    config = AssignmentConfig(theta_default=4.0)
    artifacts = _mk_artifacts(config=config)
    base_cost = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)

    seen: dict[str, Any] = {}

    def _assign_core(**kw):
        seen["od_values_dtype"] = kw["od_values"].dtype
        seen["theta_dtype"] = kw["theta"].dtype
        return jnp.asarray([1.0, 2.0, 3.0], dtype=kw["od_values"].dtype), None

    monkeypatch.setattr(
        "public_transportation.assignment.factory.prepare_assignment",
        lambda scenario, config: artifacts,
    )
    monkeypatch.setattr(
        "public_transportation.assignment.factory.AssignmentIDManager",
        SimpleNamespace(build=lambda *, scenario, graph: _DummyIDManager()),
    )
    monkeypatch.setattr(
        "public_transportation.assignment.factory.link_costs",
        lambda *, graph, cost_parts, config: base_cost,
    )
    monkeypatch.setattr(
        "public_transportation.assignment.factory._assign_core",
        _assign_core,
    )

    factory = build_assignment_factory(scenario=scenario, config=config)

    result = factory.link_flow_fn([1.0, 2.0], theta=3.0)

    assert np.allclose(np.asarray(result), [1.0, 2.0, 3.0])
    assert seen["od_values_dtype"] == jnp.asarray([1.0]).dtype
    assert seen["theta_dtype"] == seen["od_values_dtype"]