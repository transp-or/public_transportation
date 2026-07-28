from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import pytest

from public_transportation.assignment.assign import (
    assign,
    assign_from_scenario,
    prepare_assignment,
)
from public_transportation.assignment.config import AssignmentConfig
from public_transportation.assignment.graph_sentinels import NODE_KIND_CENTROID_OUT


@dataclass(frozen=True)
class _DummyGraph:
    num_nodes: int
    num_links: int
    node_kind: jnp.ndarray


class _DummyODGroups:
    def __init__(
        self,
        *,
        group_dest_node: list[int],
        group_link_mask: list[list[bool]],
        od_origin_node: list[int],
        group_od_index_padded: list[list[int]],
        group_od_mask: list[list[bool]],
    ) -> None:
        self.group_dest_node = jnp.asarray(group_dest_node, dtype=jnp.int32)
        self.group_link_mask = jnp.asarray(group_link_mask, dtype=jnp.bool_)
        self.od_origin_node = jnp.asarray(od_origin_node, dtype=jnp.int32)
        self.group_od_index_padded = jnp.asarray(group_od_index_padded, dtype=jnp.int32)
        self.group_od_mask = jnp.asarray(group_od_mask, dtype=jnp.bool_)


def _scenario_with_timetable() -> Any:
    return SimpleNamespace(timetable=object())


def _scenario_without_timetable() -> Any:
    return SimpleNamespace(timetable=None)


def _centroid_out_graph(
    *,
    num_nodes: int,
    num_links: int,
    dest_nodes: list[int],
) -> _DummyGraph:
    node_kind = jnp.zeros((num_nodes,), dtype=jnp.int32)
    for node in dest_nodes:
        node_kind = node_kind.at[node].set(NODE_KIND_CENTROID_OUT)
    return _DummyGraph(num_nodes=num_nodes, num_links=num_links, node_kind=node_kind)


def _valid_od_groups(*, num_links: int) -> _DummyODGroups:
    return _DummyODGroups(
        group_dest_node=[1],
        group_link_mask=[[True] * num_links],
        od_origin_node=[0],
        group_od_index_padded=[[0]],
        group_od_mask=[[True]],
    )


def test_prepare_assignment_requires_timetable(monkeypatch):
    monkeypatch.setattr(
        "public_transportation.assignment.assign.build_time_expanded_graph",
        lambda **kw: None,
    )
    monkeypatch.setattr(
        "public_transportation.assignment.assign.build_od_groups",
        lambda **kw: None,
    )
    monkeypatch.setattr(
        "public_transportation.assignment.assign.precompute_base_costs",
        lambda *a, **k: None,
    )

    with pytest.raises(ValueError, match="Scenario\\.timetable is required"):
        prepare_assignment(_scenario_without_timetable(), AssignmentConfig())


def test_prepare_assignment_calls_builders_and_returns_artifacts(monkeypatch):
    graph = _centroid_out_graph(num_nodes=3, num_links=4, dest_nodes=[1, 2])
    od_groups = _DummyODGroups(
        group_dest_node=[1, 2],
        group_link_mask=[
            [True, True, True, True],
            [True, True, True, True],
        ],
        od_origin_node=[0, 0],
        group_od_index_padded=[[0], [1]],
        group_od_mask=[[True], [True]],
    )
    cost_parts = SimpleNamespace(name="cost_parts")
    calls = {"graph": 0, "groups": 0, "cost_parts": 0}

    def _build_graph(*, scenario, config, profile):
        calls["graph"] += 1
        assert scenario.timetable is not None
        assert isinstance(config, AssignmentConfig)
        return graph

    def _build_groups(*, scenario, graph: _DummyGraph, profile):
        calls["groups"] += 1
        assert scenario.timetable is not None
        return od_groups

    def _precompute(graph_in, config_in):
        calls["cost_parts"] += 1
        assert graph_in is graph
        assert isinstance(config_in, AssignmentConfig)
        return cost_parts

    monkeypatch.setattr(
        "public_transportation.assignment.assign.build_time_expanded_graph",
        _build_graph,
    )
    monkeypatch.setattr(
        "public_transportation.assignment.assign.build_od_groups",
        _build_groups,
    )
    monkeypatch.setattr(
        "public_transportation.assignment.assign.precompute_base_costs",
        _precompute,
    )

    cfg = AssignmentConfig()
    artifacts = prepare_assignment(_scenario_with_timetable(), cfg)

    assert artifacts.graph is graph
    assert artifacts.od_groups is od_groups
    assert artifacts.cost_parts is cost_parts
    assert artifacts.config is cfg
    assert calls == {"graph": 1, "groups": 1, "cost_parts": 1}


def test_assign_rejects_nonpositive_theta(monkeypatch):
    graph = _centroid_out_graph(num_nodes=2, num_links=3, dest_nodes=[1])
    od_groups = _valid_od_groups(num_links=graph.num_links)
    artifacts = SimpleNamespace(
        graph=graph,
        od_groups=od_groups,
        cost_parts=SimpleNamespace(),
        config=AssignmentConfig(),
    )

    monkeypatch.setattr(
        "public_transportation.assignment.assign.link_costs",
        lambda **kw: jnp.zeros((graph.num_links,), dtype=jnp.float32),
    )

    with pytest.raises(ValueError, match="theta must be positive"):
        assign(jnp.asarray([1.0], dtype=jnp.float32), artifacts, theta=0.0)


def test_assign_uses_default_theta_when_none(monkeypatch):
    graph = _centroid_out_graph(num_nodes=2, num_links=2, dest_nodes=[1])
    od_groups = _valid_od_groups(num_links=graph.num_links)
    cost_parts = SimpleNamespace()
    seen: dict[str, Any] = {}

    def _link_costs(*, graph, cost_parts, config):
        seen["link_costs_graph"] = graph
        seen["link_costs_cost_parts"] = cost_parts
        seen["link_costs_config"] = config
        return jnp.zeros((graph.num_links,), dtype=jnp.float32)

    def _assign_core(**kw):
        seen.update(kw)
        return jnp.ones((graph.num_links,), dtype=jnp.float32), None

    monkeypatch.setattr("public_transportation.assignment.assign.link_costs", _link_costs)
    monkeypatch.setattr("public_transportation.assignment.assign._assign_core", _assign_core)

    cfg = AssignmentConfig(theta_default=7.5)
    artifacts = SimpleNamespace(
        graph=graph,
        od_groups=od_groups,
        cost_parts=cost_parts,
        config=cfg,
    )

    result = assign(jnp.asarray([3.0], dtype=jnp.float32), artifacts)

    assert result.theta == pytest.approx(7.5)
    assert seen["theta"] == pytest.approx(7.5)
    assert seen["link_costs_graph"] is graph
    assert seen["link_costs_cost_parts"] is cost_parts
    assert seen["link_costs_config"] is cfg
    assert jnp.all(result.link_flow == 1.0)
    assert result.group_link_flow is None


def test_assign_clamps_theta_to_theta_min(monkeypatch):
    graph = _centroid_out_graph(num_nodes=2, num_links=2, dest_nodes=[1])
    od_groups = _valid_od_groups(num_links=graph.num_links)
    seen: dict[str, Any] = {}

    monkeypatch.setattr(
        "public_transportation.assignment.assign.link_costs",
        lambda **kw: jnp.zeros((graph.num_links,), dtype=jnp.float32),
    )

    def _assign_core(**kw):
        seen.update(kw)
        return jnp.zeros((graph.num_links,), dtype=jnp.float32), None

    monkeypatch.setattr("public_transportation.assignment.assign._assign_core", _assign_core)

    cfg = AssignmentConfig(theta_default=5.0, theta_min=0.5)
    artifacts = SimpleNamespace(
        graph=graph,
        od_groups=od_groups,
        cost_parts=SimpleNamespace(),
        config=cfg,
    )

    result = assign(jnp.asarray([1.0], dtype=jnp.float32), artifacts, theta=0.1)

    assert result.theta == pytest.approx(0.5)
    assert seen["theta"] == pytest.approx(0.5)


def test_assign_passes_current_od_group_arrays_to_core(monkeypatch):
    graph = _centroid_out_graph(num_nodes=3, num_links=4, dest_nodes=[1, 2])
    od_groups = _DummyODGroups(
        group_dest_node=[1, 2],
        group_link_mask=[
            [True, False, True, True],
            [False, True, True, True],
        ],
        od_origin_node=[0, 0, 0],
        group_od_index_padded=[[0, 1], [2, 0]],
        group_od_mask=[[True, True], [True, False]],
    )
    seen: dict[str, Any] = {}
    base_cost = jnp.arange(graph.num_links, dtype=jnp.float32)

    monkeypatch.setattr(
        "public_transportation.assignment.assign.link_costs",
        lambda **kw: base_cost,
    )

    def _assign_core(**kw):
        seen.update(kw)
        return jnp.arange(graph.num_links, dtype=jnp.float32), None

    monkeypatch.setattr("public_transportation.assignment.assign._assign_core", _assign_core)

    artifacts = SimpleNamespace(
        graph=graph,
        od_groups=od_groups,
        cost_parts=SimpleNamespace(),
        config=AssignmentConfig(),
    )

    result = assign(jnp.asarray([10.0, 20.0, 30.0], dtype=jnp.float32), artifacts)

    assert result.link_cost.shape == (graph.num_links,)
    assert jnp.allclose(result.link_cost, base_cost)
    assert seen["group_dest_node"].shape == (2,)
    assert seen["group_link_mask"].shape == (2, graph.num_links)
    assert seen["od_origin_node"].shape == (3,)
    assert seen["group_od_index_padded"].shape == (2, 2)
    assert seen["group_od_mask"].shape == (2, 2)
    assert bool(seen["return_group_link_flows"]) is False


def test_assign_can_return_group_link_flows(monkeypatch):
    graph = _centroid_out_graph(num_nodes=3, num_links=4, dest_nodes=[1, 2])
    od_groups = _DummyODGroups(
        group_dest_node=[1, 2],
        group_link_mask=[
            [True, True, True, True],
            [True, True, True, True],
        ],
        od_origin_node=[0, 0],
        group_od_index_padded=[[0], [1]],
        group_od_mask=[[True], [True]],
    )
    base = jnp.arange(1, graph.num_links + 1, dtype=jnp.float32)
    group_link_flow = jnp.vstack([base, 2.0 * base])
    total_link_flow = jnp.sum(group_link_flow, axis=0)

    monkeypatch.setattr(
        "public_transportation.assignment.assign.link_costs",
        lambda **kw: jnp.zeros((graph.num_links,), dtype=jnp.float32),
    )

    def _assign_core(**kw):
        assert bool(kw["return_group_link_flows"])
        return total_link_flow, group_link_flow

    monkeypatch.setattr("public_transportation.assignment.assign._assign_core", _assign_core)

    artifacts = SimpleNamespace(
        graph=graph,
        od_groups=od_groups,
        cost_parts=SimpleNamespace(),
        config=AssignmentConfig(),
    )

    result = assign(
        jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        artifacts,
        return_group_link_flows=True,
    )

    assert jnp.allclose(result.link_flow, total_link_flow)
    assert result.group_link_flow is not None
    assert result.group_link_flow.shape == (2, graph.num_links)
    assert jnp.allclose(result.group_link_flow, group_link_flow)


def test_assign_rejects_non_vector_od_values(monkeypatch):
    graph = _centroid_out_graph(num_nodes=2, num_links=2, dest_nodes=[1])
    od_groups = _valid_od_groups(num_links=graph.num_links)

    monkeypatch.setattr(
        "public_transportation.assignment.assign.link_costs",
        lambda **kw: jnp.zeros((graph.num_links,), dtype=jnp.float32),
    )

    artifacts = SimpleNamespace(
        graph=graph,
        od_groups=od_groups,
        cost_parts=SimpleNamespace(),
        config=AssignmentConfig(),
    )

    with pytest.raises(ValueError, match="od_values must be a 1D array"):
        assign(jnp.ones((1, 1), dtype=jnp.float32), artifacts)


def test_assign_rejects_missing_group_link_mask(monkeypatch):
    graph = _centroid_out_graph(num_nodes=2, num_links=1, dest_nodes=[1])

    class _BadGroups:
        group_dest_node = jnp.asarray([1], dtype=jnp.int32)

    monkeypatch.setattr(
        "public_transportation.assignment.assign.link_costs",
        lambda **kw: jnp.zeros((graph.num_links,), dtype=jnp.float32),
    )

    artifacts = SimpleNamespace(
        graph=graph,
        od_groups=_BadGroups(),
        cost_parts=SimpleNamespace(),
        config=AssignmentConfig(),
    )

    with pytest.raises(ValueError, match="group_link_mask"):
        assign(jnp.asarray([1.0], dtype=jnp.float32), artifacts)


@pytest.mark.parametrize(
    "missing_field, expected_message",
    [
        ("od_origin_node", "od_origin_node"),
        ("group_od_index_padded", "group_od_index_padded"),
        ("group_od_mask", "group_od_mask"),
    ],
)
def test_assign_rejects_missing_current_od_group_fields(
    monkeypatch,
    missing_field: str,
    expected_message: str,
):
    graph = _centroid_out_graph(num_nodes=2, num_links=1, dest_nodes=[1])
    od_groups = _valid_od_groups(num_links=graph.num_links)
    delattr(od_groups, missing_field)

    monkeypatch.setattr(
        "public_transportation.assignment.assign.link_costs",
        lambda **kw: jnp.zeros((graph.num_links,), dtype=jnp.float32),
    )

    artifacts = SimpleNamespace(
        graph=graph,
        od_groups=od_groups,
        cost_parts=SimpleNamespace(),
        config=AssignmentConfig(),
    )

    with pytest.raises(ValueError, match=expected_message):
        assign(jnp.asarray([1.0], dtype=jnp.float32), artifacts)


def test_assign_rejects_invalid_destination_node_index(monkeypatch):
    graph = _centroid_out_graph(num_nodes=2, num_links=1, dest_nodes=[1])
    od_groups = _DummyODGroups(
        group_dest_node=[2],
        group_link_mask=[[True]],
        od_origin_node=[0],
        group_od_index_padded=[[0]],
        group_od_mask=[[True]],
    )

    monkeypatch.setattr(
        "public_transportation.assignment.assign.link_costs",
        lambda **kw: jnp.zeros((graph.num_links,), dtype=jnp.float32),
    )

    artifacts = SimpleNamespace(
        graph=graph,
        od_groups=od_groups,
        cost_parts=SimpleNamespace(),
        config=AssignmentConfig(),
    )

    with pytest.raises(ValueError, match="group_dest_node contains invalid node indices"):
        assign(jnp.asarray([1.0], dtype=jnp.float32), artifacts)


def test_assign_rejects_destination_that_is_not_centroid_out(monkeypatch):
    graph = _centroid_out_graph(num_nodes=2, num_links=1, dest_nodes=[])
    od_groups = _valid_od_groups(num_links=graph.num_links)

    monkeypatch.setattr(
        "public_transportation.assignment.assign.link_costs",
        lambda **kw: jnp.zeros((graph.num_links,), dtype=jnp.float32),
    )

    artifacts = SimpleNamespace(
        graph=graph,
        od_groups=od_groups,
        cost_parts=SimpleNamespace(),
        config=AssignmentConfig(),
    )

    with pytest.raises(ValueError, match="must be a destination centroid-out node"):
        assign(jnp.asarray([1.0], dtype=jnp.float32), artifacts)


def test_assign_rejects_group_link_mask_wrong_shape(monkeypatch):
    graph = _centroid_out_graph(num_nodes=2, num_links=3, dest_nodes=[1])
    od_groups = _DummyODGroups(
        group_dest_node=[1],
        group_link_mask=[[True, True]],
        od_origin_node=[0],
        group_od_index_padded=[[0]],
        group_od_mask=[[True]],
    )

    monkeypatch.setattr(
        "public_transportation.assignment.assign.link_costs",
        lambda **kw: jnp.zeros((graph.num_links,), dtype=jnp.float32),
    )

    artifacts = SimpleNamespace(
        graph=graph,
        od_groups=od_groups,
        cost_parts=SimpleNamespace(),
        config=AssignmentConfig(),
    )

    with pytest.raises(ValueError, match="group_link_mask has wrong shape"):
        assign(jnp.asarray([1.0], dtype=jnp.float32), artifacts)


def test_assign_rejects_group_od_index_and_mask_shape_mismatch(monkeypatch):
    graph = _centroid_out_graph(num_nodes=2, num_links=1, dest_nodes=[1])
    od_groups = _DummyODGroups(
        group_dest_node=[1],
        group_link_mask=[[True]],
        od_origin_node=[0],
        group_od_index_padded=[[0, 0]],
        group_od_mask=[[True]],
    )

    monkeypatch.setattr(
        "public_transportation.assignment.assign.link_costs",
        lambda **kw: jnp.zeros((graph.num_links,), dtype=jnp.float32),
    )

    artifacts = SimpleNamespace(
        graph=graph,
        od_groups=od_groups,
        cost_parts=SimpleNamespace(),
        config=AssignmentConfig(),
    )

    with pytest.raises(ValueError, match="must have the same shape"):
        assign(jnp.asarray([1.0], dtype=jnp.float32), artifacts)


def test_assign_rejects_link_cost_wrong_shape(monkeypatch):
    graph = _centroid_out_graph(num_nodes=2, num_links=3, dest_nodes=[1])
    od_groups = _valid_od_groups(num_links=graph.num_links)

    monkeypatch.setattr(
        "public_transportation.assignment.assign.link_costs",
        lambda **kw: jnp.zeros((2,), dtype=jnp.float32),
    )

    artifacts = SimpleNamespace(
        graph=graph,
        od_groups=od_groups,
        cost_parts=SimpleNamespace(),
        config=AssignmentConfig(),
    )

    with pytest.raises(ValueError, match="link_costs returned shape"):
        assign(jnp.asarray([1.0], dtype=jnp.float32), artifacts)


def test_assign_from_scenario_calls_prepare_and_assign(monkeypatch):
    scenario = _scenario_with_timetable()
    cfg = AssignmentConfig()
    expected = SimpleNamespace(theta=5.0, link_flow=jnp.asarray([1.0]), group_link_flow=None)

    def _prepare(scenario_in, config_in):
        assert scenario_in is scenario
        assert config_in is cfg
        return "ARTIFACTS"

    def _assign(od_values, artifacts, **kwargs):
        assert artifacts == "ARTIFACTS"
        assert jnp.allclose(od_values, jnp.asarray([2.0]))
        assert kwargs == {"theta": None, "return_group_link_flows": False}
        return expected

    monkeypatch.setattr("public_transportation.assignment.assign.prepare_assignment", _prepare)
    monkeypatch.setattr("public_transportation.assignment.assign.assign", _assign)

    result = assign_from_scenario(scenario, jnp.asarray([2.0]), cfg)

    assert result is expected
