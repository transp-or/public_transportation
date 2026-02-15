from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import jax.numpy as jnp

from public_transportation.assignment.assign import (
    prepare_assignment,
    assign,
    assign_from_scenario,
)
from public_transportation.assignment.config import AssignmentConfig


# -------------------------
# Tiny stubs for isolation
# -------------------------


@dataclass(frozen=True)
class _DummyGraph:
    num_nodes: int
    num_links: int


class _DummyODGroups:
    def __init__(self, *, num_groups: int, dest_node: list[int], a_min: list[float], b_min: list[float]) -> None:
        self.num_groups = int(num_groups)
        self.dest_node = jnp.asarray(dest_node, dtype=jnp.int32)
        self.a_min = jnp.asarray(a_min, dtype=jnp.float32)
        self.b_min = jnp.asarray(b_min, dtype=jnp.float32)

    def make_initial_node_flow(self, *, group_index: int, od_values: jnp.ndarray, num_nodes: int) -> jnp.ndarray:
        # Inject a single scalar per group into node 0 (centroid), purely for testing.
        y0 = jnp.zeros((num_nodes,), dtype=jnp.asarray(od_values).dtype)
        return y0.at[0].set(od_values[group_index])


@dataclass(frozen=True)
class _DummyDialResult:
    link_flow: jnp.ndarray


# -------------------------
# Helpers
# -------------------------


def _scenario_with_timetable() -> Any:
    # We only need scenario.timetable non-None for prepare_assignment() to pass the guard.
    return SimpleNamespace(timetable=object())


def _scenario_without_timetable() -> Any:
    return SimpleNamespace(timetable=None)


# -------------------------
# prepare_assignment
# -------------------------


def test_prepare_assignment_requires_timetable(monkeypatch):
    # Avoid any accidental calls into real builders
    monkeypatch.setattr("public_transportation.assignment.assign.build_time_expanded_graph", lambda **kw: None)
    monkeypatch.setattr("public_transportation.assignment.assign.build_od_groups", lambda **kw: None)
    monkeypatch.setattr("public_transportation.assignment.assign.precompute_base_costs", lambda *a, **k: None)

    cfg = AssignmentConfig()
    with pytest.raises(ValueError, match="Scenario\\.timetable is required"):
        prepare_assignment(_scenario_without_timetable(), cfg)


def test_prepare_assignment_calls_builders_and_returns_artifacts(monkeypatch):
    graph = _DummyGraph(num_nodes=3, num_links=4)
    od_groups = _DummyODGroups(num_groups=2, dest_node=[1, 2], a_min=[0.0, 10.0], b_min=[5.0, 20.0])
    cost_parts = SimpleNamespace(name="cost_parts")

    called = {"graph": 0, "groups": 0, "cost_parts": 0}

    def _build_graph(*, scenario, config):
        called["graph"] += 1
        return graph

    def _build_groups(*, scenario, graph):
        called["groups"] += 1
        assert graph is graph  # same object
        return od_groups

    def _precompute(graph_in, config_in):
        called["cost_parts"] += 1
        assert graph_in is graph
        return cost_parts

    monkeypatch.setattr("public_transportation.assignment.assign.build_time_expanded_graph", _build_graph)
    monkeypatch.setattr("public_transportation.assignment.assign.build_od_groups", _build_groups)
    monkeypatch.setattr("public_transportation.assignment.assign.precompute_base_costs", _precompute)

    cfg = AssignmentConfig()
    arts = prepare_assignment(_scenario_with_timetable(), cfg)

    assert arts.graph is graph
    assert arts.od_groups is od_groups
    assert arts.cost_parts is cost_parts
    assert arts.config is cfg
    assert called == {"graph": 1, "groups": 1, "cost_parts": 1}


# -------------------------
# assign: theta logic
# -------------------------


def test_assign_rejects_nonpositive_theta(monkeypatch):
    # Minimal artifacts
    graph = _DummyGraph(num_nodes=2, num_links=3)
    od_groups = _DummyODGroups(num_groups=1, dest_node=[1], a_min=[0.0], b_min=[1.0])
    cost_parts = SimpleNamespace()

    # Stubs
    monkeypatch.setattr("public_transportation.assignment.assign.link_costs_for_group", lambda **kw: jnp.zeros((3,)))
    monkeypatch.setattr(
        "public_transportation.assignment.assign.run_dial_for_destination",
        lambda **kw: _DummyDialResult(link_flow=jnp.zeros((3,))),
    )

    cfg = AssignmentConfig()
    # Some projects have theta_min; if it doesn't exist, we still test theta<=0 rejection.
    arts = SimpleNamespace(graph=graph, od_groups=od_groups, cost_parts=cost_parts, config=cfg)

    with pytest.raises(ValueError, match="theta must be positive"):
        assign(jnp.array([1.0]), arts, theta=0.0)


def test_assign_uses_default_theta_when_none(monkeypatch):
    graph = _DummyGraph(num_nodes=2, num_links=2)
    od_groups = _DummyODGroups(num_groups=1, dest_node=[1], a_min=[0.0], b_min=[1.0])
    cost_parts = SimpleNamespace()

    seen: dict[str, float] = {}

    def _costs_for_group(*, theta, **kw):
        seen["theta_in_costs"] = float(theta)
        return jnp.zeros((graph.num_links,), dtype=jnp.float32)

    def _dial(*, theta, **kw):
        seen["theta_in_dial"] = float(theta)
        return _DummyDialResult(link_flow=jnp.ones((graph.num_links,), dtype=jnp.float32))

    monkeypatch.setattr("public_transportation.assignment.assign.link_costs_for_group", _costs_for_group)
    monkeypatch.setattr("public_transportation.assignment.assign.run_dial_for_destination", _dial)

    cfg = AssignmentConfig(theta_default=7.5)
    arts = SimpleNamespace(graph=graph, od_groups=od_groups, cost_parts=cost_parts, config=cfg)

    res = assign(jnp.array([3.0], dtype=jnp.float32), arts, theta=None)

    assert res.theta == pytest.approx(7.5)
    assert seen["theta_in_costs"] == pytest.approx(7.5)
    assert seen["theta_in_dial"] == pytest.approx(7.5)
    assert res.link_flow.shape == (graph.num_links,)
    assert jnp.all(res.link_flow == 1.0)
    assert res.group_link_flow is None


# -------------------------
# assign: aggregation over groups + return_group_link_flows
# -------------------------


def test_assign_sums_group_link_flows_and_can_return_per_group(monkeypatch):
    graph = _DummyGraph(num_nodes=3, num_links=4)
    od_groups = _DummyODGroups(num_groups=2, dest_node=[1, 2], a_min=[0.0, 10.0], b_min=[5.0, 20.0])
    cost_parts = SimpleNamespace()

    # Make group g produce link_flow = (g+1) * [1,2,3,4]
    base = jnp.arange(1, graph.num_links + 1, dtype=jnp.float32)

    def _costs_for_group(*, a_min, b_min, **kw):
        # Verify a_min/b_min pass-through from ODGroups
        assert (a_min, b_min) in {(0.0, 5.0), (10.0, 20.0)}
        return jnp.zeros((graph.num_links,), dtype=jnp.float32)

    def _dial(*, initial_node_flow, dest_node, **kw):
        # Determine g from the injected demand: we inject od_values[g] into node 0.
        g_val = float(initial_node_flow[0])
        # In this test, od_values is [0.0, 1.0] so g_val identifies group.
        g = 0 if g_val == 0.0 else 1
        return _DummyDialResult(link_flow=(g + 1) * base)

    monkeypatch.setattr("public_transportation.assignment.assign.link_costs_for_group", _costs_for_group)
    monkeypatch.setattr("public_transportation.assignment.assign.run_dial_for_destination", _dial)

    cfg = AssignmentConfig(theta_default=5.0)
    arts = SimpleNamespace(graph=graph, od_groups=od_groups, cost_parts=cost_parts, config=cfg)

    od_values = jnp.array([0.0, 1.0], dtype=jnp.float32)

    res = assign(od_values, arts, return_group_link_flows=True)

    # total = 1*base + 2*base = 3*base
    assert jnp.allclose(res.link_flow, 3.0 * base)
    assert res.group_link_flow is not None
    assert res.group_link_flow.shape == (od_groups.num_groups, graph.num_links)
    assert jnp.allclose(res.group_link_flow[0], 1.0 * base)
    assert jnp.allclose(res.group_link_flow[1], 2.0 * base)


# -------------------------
# assign: contract checks for ODGroups API
# -------------------------


def test_assign_requires_od_groups_dest_node_and_time_bounds(monkeypatch):
    graph = _DummyGraph(num_nodes=2, num_links=1)
    cfg = AssignmentConfig()
    cost_parts = SimpleNamespace()

    class _BadGroups:
        num_groups = 1

        def make_initial_node_flow(self, *, group_index, od_values, num_nodes):
            return jnp.zeros((num_nodes,))

    arts = SimpleNamespace(graph=graph, od_groups=_BadGroups(), cost_parts=cost_parts, config=cfg)

    monkeypatch.setattr("public_transportation.assignment.assign.link_costs_for_group", lambda **kw: jnp.zeros((1,)))
    monkeypatch.setattr(
        "public_transportation.assignment.assign.run_dial_for_destination",
        lambda **kw: _DummyDialResult(link_flow=jnp.zeros((1,))),
    )

    with pytest.raises(ValueError, match="ODGroups must provide `dest_node`"):
        assign(jnp.array([1.0]), arts)


def test_assign_requires_make_initial_node_flow(monkeypatch):
    graph = _DummyGraph(num_nodes=2, num_links=1)
    cfg = AssignmentConfig()
    cost_parts = SimpleNamespace()

    class _GroupsNoMethod:
        num_groups = 1
        dest_node = jnp.asarray([1], dtype=jnp.int32)
        a_min = jnp.asarray([0.0], dtype=jnp.float32)
        b_min = jnp.asarray([1.0], dtype=jnp.float32)

    arts = SimpleNamespace(graph=graph, od_groups=_GroupsNoMethod(), cost_parts=cost_parts, config=cfg)

    monkeypatch.setattr("public_transportation.assignment.assign.link_costs_for_group", lambda **kw: jnp.zeros((1,)))
    monkeypatch.setattr(
        "public_transportation.assignment.assign.run_dial_for_destination",
        lambda **kw: _DummyDialResult(link_flow=jnp.zeros((1,))),
    )

    with pytest.raises(ValueError, match=r"ODGroups is missing method `make_initial_node_flow"):
        assign(jnp.array([1.0]), arts)


# -------------------------
# assign_from_scenario
# -------------------------


def test_assign_from_scenario_calls_prepare_and_assign(monkeypatch):
    scenario = _scenario_with_timetable()
    cfg = AssignmentConfig()

    dummy_res = SimpleNamespace(theta=5.0, link_flow=jnp.array([1.0]), group_link_flow=None)

    def _prep(scenario_in, config_in):
        assert scenario_in is scenario
        assert config_in is cfg
        return "ARTS"

    def _assign(od_values, artifacts, **kwargs):
        assert artifacts == "ARTS"
        assert jnp.allclose(od_values, jnp.array([2.0]))
        return dummy_res

    monkeypatch.setattr("public_transportation.assignment.assign.prepare_assignment", _prep)
    monkeypatch.setattr("public_transportation.assignment.assign.assign", _assign)

    res = assign_from_scenario(scenario, jnp.array([2.0]), cfg)
    assert res is dummy_res