# tests/assignment/test_costs.py
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.assignment.config import AssignmentConfig
from public_transportation.assignment.costs import (
    CostParts,
    link_costs,
    link_costs_for_group,
    link_costs_for_interval,
    precompute_base_costs,
    stable_logit_transition_logits,
    typical_cost_scale_from_assignment,
)
from public_transportation.assignment.graph_sentinels import (
    LINK_TYPE_ACCESS,
    LINK_TYPE_DWELL,
    LINK_TYPE_EGRESS,
    LINK_TYPE_RIDE,
    LINK_TYPE_TRANSFER,
)


@dataclass(frozen=True)
class _Graph:
    num_nodes: int
    num_links: int
    tail: jnp.ndarray
    head: jnp.ndarray
    link_type: jnp.ndarray
    travel_time: jnp.ndarray
    node_time: jnp.ndarray | None = None
    node_bin_start_min: jnp.ndarray | None = None
    node_bin_end_min: jnp.ndarray | None = None
    capacity: jnp.ndarray | None = None


def _as_np(x):
    return np.asarray(x)


def _mk_graph_with_all_link_types() -> _Graph:
    # Nodes:
    # 0: centroid-in, bin [10, 20]
    # 1: departure event at 5
    # 2: departure event at 15
    # 3: departure/arrival event at 25
    # 4: centroid-out
    #
    # Links:
    # 0: access 0 -> 1, early by 5
    # 1: access 0 -> 2, inside bin
    # 2: access 0 -> 3, late by 5
    # 3: ride
    # 4: transfer
    # 5: egress
    # 6: dwell
    return _Graph(
        num_nodes=5,
        num_links=7,
        tail=jnp.asarray([0, 0, 0, 1, 2, 3, 3], dtype=jnp.int32),
        head=jnp.asarray([1, 2, 3, 3, 3, 4, 2], dtype=jnp.int32),
        link_type=jnp.asarray(
            [
                LINK_TYPE_ACCESS,
                LINK_TYPE_ACCESS,
                LINK_TYPE_ACCESS,
                LINK_TYPE_RIDE,
                LINK_TYPE_TRANSFER,
                LINK_TYPE_EGRESS,
                LINK_TYPE_DWELL,
            ],
            dtype=jnp.int32,
        ),
        travel_time=jnp.asarray([0.0, 0.0, 0.0, 12.0, 4.0, 0.0, 1.5], dtype=jnp.float32),
        node_time=jnp.asarray([10.0, 5.0, 15.0, 25.0, 25.0], dtype=jnp.float32),
        node_bin_start_min=jnp.asarray([10.0, np.nan, np.nan, np.nan, np.nan], dtype=jnp.float32),
        node_bin_end_min=jnp.asarray([20.0, np.nan, np.nan, np.nan, np.nan], dtype=jnp.float32),
        capacity=jnp.asarray([100.0] * 7, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------
# precompute_base_costs
# ---------------------------------------------------------------------


def test_precompute_base_costs_identifies_link_types_and_base_costs():
    cfg = AssignmentConfig(beta_transfer=2.5)
    graph = _mk_graph_with_all_link_types()

    parts = precompute_base_costs(graph, cfg)

    assert isinstance(parts, CostParts)
    assert np.array_equal(_as_np(parts.is_access), [True, True, True, False, False, False, False])
    assert np.array_equal(_as_np(parts.is_ride), [False, False, False, True, False, False, False])
    assert np.array_equal(_as_np(parts.is_transfer), [False, False, False, False, True, False, False])
    assert np.array_equal(_as_np(parts.is_egress), [False, False, False, False, False, True, False])
    assert np.array_equal(_as_np(parts.is_dwell), [False, False, False, False, False, False, True])

    # access = 0, ride = travel_time, transfer = beta_transfer * travel_time,
    # egress = 0, dwell = travel_time
    assert np.allclose(_as_np(parts.base_cost), [0.0, 0.0, 0.0, 12.0, 10.0, 0.0, 1.5])


def test_precompute_base_costs_validates_config():
    cfg = AssignmentConfig(beta_transfer=-1.0)
    graph = _mk_graph_with_all_link_types()

    with pytest.raises(ValueError, match="beta_transfer"):
        precompute_base_costs(graph, cfg)


def test_precompute_base_costs_returns_jax_arrays_for_valid_config():
    cfg = AssignmentConfig(beta_transfer=2.0)
    graph = _mk_graph_with_all_link_types()

    parts = precompute_base_costs(graph, cfg)

    assert isinstance(parts.base_cost, jnp.ndarray)
    assert np.isclose(float(jnp.sum(parts.base_cost)), 12.0 + 8.0 + 1.5)


# ---------------------------------------------------------------------
# link_costs and access penalties
# ---------------------------------------------------------------------


def test_link_costs_adds_access_early_late_penalties_and_base_costs():
    cfg = AssignmentConfig(beta_transfer=2.5, beta_early=2.0, beta_late=3.0)
    graph = _mk_graph_with_all_link_types()
    parts = precompute_base_costs(graph, cfg)

    costs = link_costs(graph=graph, cost_parts=parts, config=cfg)

    # access links:
    # tau=5, bin [10,20]  -> early 5 -> 10
    # tau=15, bin [10,20] -> 0
    # tau=25, bin [10,20] -> late 5 -> 15
    # ride -> 12
    # transfer -> 2.5 * 4 = 10
    # egress -> 0
    # dwell -> 1.5
    assert np.allclose(_as_np(costs), [10.0, 0.0, 15.0, 12.0, 10.0, 0.0, 1.5])


def test_link_costs_zero_access_penalty_inside_interval():
    cfg = AssignmentConfig(beta_early=100.0, beta_late=100.0)
    graph = _Graph(
        num_nodes=2,
        num_links=1,
        tail=jnp.asarray([0], dtype=jnp.int32),
        head=jnp.asarray([1], dtype=jnp.int32),
        link_type=jnp.asarray([LINK_TYPE_ACCESS], dtype=jnp.int32),
        travel_time=jnp.asarray([0.0], dtype=jnp.float32),
        node_time=jnp.asarray([10.0, 15.0], dtype=jnp.float32),
        node_bin_start_min=jnp.asarray([10.0, np.nan], dtype=jnp.float32),
        node_bin_end_min=jnp.asarray([20.0, np.nan], dtype=jnp.float32),
    )
    parts = precompute_base_costs(graph, cfg)

    costs = link_costs(graph=graph, cost_parts=parts, config=cfg)

    assert np.allclose(_as_np(costs), [0.0])


def test_link_costs_handles_no_access_links_without_bin_bounds():
    cfg = AssignmentConfig(beta_transfer=2.0)
    graph = _Graph(
        num_nodes=3,
        num_links=3,
        tail=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        head=jnp.asarray([1, 2, 0], dtype=jnp.int32),
        link_type=jnp.asarray(
            [LINK_TYPE_RIDE, LINK_TYPE_TRANSFER, LINK_TYPE_DWELL],
            dtype=jnp.int32,
        ),
        travel_time=jnp.asarray([8.0, 3.0, 0.5], dtype=jnp.float32),
        node_time=jnp.asarray([0.0, 8.0, 11.0], dtype=jnp.float32),
        node_bin_start_min=jnp.asarray([np.nan, np.nan, np.nan], dtype=jnp.float32),
        node_bin_end_min=jnp.asarray([np.nan, np.nan, np.nan], dtype=jnp.float32),
    )
    parts = precompute_base_costs(graph, cfg)

    costs = link_costs(graph=graph, cost_parts=parts, config=cfg)

    assert np.allclose(_as_np(costs), [8.0, 6.0, 0.5])


def test_link_costs_requires_bin_fields_when_access_links_exist():
    cfg = AssignmentConfig()
    graph = _Graph(
        num_nodes=2,
        num_links=1,
        tail=jnp.asarray([0], dtype=jnp.int32),
        head=jnp.asarray([1], dtype=jnp.int32),
        link_type=jnp.asarray([LINK_TYPE_ACCESS], dtype=jnp.int32),
        travel_time=jnp.asarray([0.0], dtype=jnp.float32),
        node_time=jnp.asarray([0.0, 5.0], dtype=jnp.float32),
        node_bin_start_min=None,
        node_bin_end_min=None,
    )
    parts = CostParts(
        base_cost=jnp.asarray([0.0], dtype=jnp.float32),
        is_access=jnp.asarray([True]),
        is_ride=jnp.asarray([False]),
        is_transfer=jnp.asarray([False]),
        is_egress=jnp.asarray([False]),
        is_dwell=jnp.asarray([False]),
    )

    with pytest.raises((RuntimeError, TypeError, AttributeError)):
        link_costs(graph=graph, cost_parts=parts, config=cfg)


def test_link_costs_validates_config():
    cfg = AssignmentConfig(beta_late=-1.0)
    graph = _mk_graph_with_all_link_types()
    parts = precompute_base_costs(graph, AssignmentConfig())

    with pytest.raises(ValueError, match="beta_late"):
        link_costs(graph=graph, cost_parts=parts, config=cfg)


def test_link_costs_for_group_is_backward_compatible_alias():
    cfg = AssignmentConfig(beta_transfer=2.5, beta_early=2.0, beta_late=3.0)
    graph = _mk_graph_with_all_link_types()
    parts = precompute_base_costs(graph, cfg)

    direct = link_costs(graph=graph, cost_parts=parts, config=cfg)
    alias = link_costs_for_group(
        graph=graph,
        cost_parts=parts,
        config=cfg,
        od_groups=object(),
        group_index=0,
    )

    assert np.allclose(_as_np(alias), _as_np(direct))


def test_link_costs_for_interval_uses_encoded_node_bins_when_present():
    cfg = AssignmentConfig(beta_transfer=2.5, beta_early=2.0, beta_late=3.0)
    graph = _mk_graph_with_all_link_types()
    parts = precompute_base_costs(graph, cfg)

    direct = link_costs(graph=graph, cost_parts=parts, config=cfg)
    interval = link_costs_for_interval(
        graph=graph,
        cost_parts=parts,
        config=cfg,
        bin_start_min=999.0,
        bin_end_min=1000.0,
    )

    assert np.allclose(_as_np(interval), _as_np(direct))


def test_link_costs_for_interval_requires_encoded_node_bins_for_access_links():
    cfg = AssignmentConfig(beta_early=2.0, beta_late=3.0)
    graph = _Graph(
        num_nodes=3,
        num_links=2,
        tail=jnp.asarray([0, 0], dtype=jnp.int32),
        head=jnp.asarray([1, 2], dtype=jnp.int32),
        link_type=jnp.asarray([LINK_TYPE_ACCESS, LINK_TYPE_ACCESS], dtype=jnp.int32),
        travel_time=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        node_time=jnp.asarray([0.0, 5.0, 25.0], dtype=jnp.float32),
    )
    parts = precompute_base_costs(graph, cfg)

    with pytest.raises(TypeError):
        link_costs_for_interval(
            graph=graph,
            cost_parts=parts,
            config=cfg,
            bin_start_min=10.0,
            bin_end_min=20.0,
        )


# ---------------------------------------------------------------------
# stable_logit_transition_logits
# ---------------------------------------------------------------------


def test_stable_logit_transition_logits_computes_expected_values():
    link_cost = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    v_head = jnp.asarray([4.0, 5.0, 6.0], dtype=jnp.float32)

    logits = stable_logit_transition_logits(
        link_cost=link_cost,
        v_head=v_head,
        theta=2.0,
    )

    assert np.allclose(_as_np(logits), [-2.5, -3.5, -4.5])


def test_stable_logit_transition_logits_is_jittable():
    link_cost = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    v_head = jnp.asarray([3.0, 5.0], dtype=jnp.float32)

    logits = stable_logit_transition_logits(
        link_cost=link_cost,
        v_head=v_head,
        theta=2.0,
    )

    assert np.allclose(_as_np(logits), [-2.0, -3.5])


def test_stable_logit_transition_logits_allows_array_theta():
    link_cost = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    v_head = jnp.asarray([3.0, 5.0], dtype=jnp.float32)
    theta = jnp.asarray(2.0, dtype=jnp.float32)

    logits = stable_logit_transition_logits(
        link_cost=link_cost,
        v_head=v_head,
        theta=theta,
    )

    assert np.allclose(_as_np(logits), [-2.0, -3.5])


# ---------------------------------------------------------------------
# typical_cost_scale_from_assignment
# ---------------------------------------------------------------------


def test_typical_cost_scale_uses_access_flow_as_default_demand_total():
    graph = _Graph(
        num_nodes=3,
        num_links=4,
        tail=jnp.asarray([0, 1, 1, 2], dtype=jnp.int32),
        head=jnp.asarray([1, 2, 0, 0], dtype=jnp.int32),
        link_type=jnp.asarray(
            [LINK_TYPE_ACCESS, LINK_TYPE_RIDE, LINK_TYPE_TRANSFER, LINK_TYPE_EGRESS],
            dtype=jnp.int32,
        ),
        travel_time=jnp.zeros((4,), dtype=jnp.float32),
    )
    link_flow = jnp.asarray([10.0, 10.0, 5.0, 10.0], dtype=jnp.float32)
    link_cost = jnp.asarray([1.0, 8.0, 2.0, 0.0], dtype=jnp.float32)

    scale = typical_cost_scale_from_assignment(
        graph=graph,
        link_flow=link_flow,
        link_cost=link_cost,
    )

    # total cost = 10*1 + 10*8 + 5*2 = 100
    # default demand = access flow = 10
    assert scale == pytest.approx(10.0)


def test_typical_cost_scale_uses_explicit_demand_total_when_provided():
    graph = _Graph(
        num_nodes=3,
        num_links=2,
        tail=jnp.asarray([0, 1], dtype=jnp.int32),
        head=jnp.asarray([1, 2], dtype=jnp.int32),
        link_type=jnp.asarray([LINK_TYPE_RIDE, LINK_TYPE_TRANSFER], dtype=jnp.int32),
        travel_time=jnp.zeros((2,), dtype=jnp.float32),
    )
    link_flow = jnp.asarray([5.0, 5.0], dtype=jnp.float32)
    link_cost = jnp.asarray([4.0, 6.0], dtype=jnp.float32)

    scale = typical_cost_scale_from_assignment(
        graph=graph,
        link_flow=link_flow,
        link_cost=link_cost,
        demand_total=10.0,
    )

    assert scale == pytest.approx(5.0)


def test_typical_cost_scale_rejects_non_1d_inputs():
    graph = _Graph(
        num_nodes=1,
        num_links=1,
        tail=jnp.asarray([0], dtype=jnp.int32),
        head=jnp.asarray([0], dtype=jnp.int32),
        link_type=jnp.asarray([LINK_TYPE_ACCESS], dtype=jnp.int32),
        travel_time=jnp.zeros((1,), dtype=jnp.float32),
    )

    with pytest.raises(ValueError, match="must be 1D"):
        typical_cost_scale_from_assignment(
            graph=graph,
            link_flow=jnp.asarray([[1.0]], dtype=jnp.float32),
            link_cost=jnp.asarray([1.0], dtype=jnp.float32),
        )


def test_typical_cost_scale_rejects_wrong_lengths():
    graph = _Graph(
        num_nodes=1,
        num_links=2,
        tail=jnp.asarray([0, 0], dtype=jnp.int32),
        head=jnp.asarray([0, 0], dtype=jnp.int32),
        link_type=jnp.asarray([LINK_TYPE_ACCESS, LINK_TYPE_RIDE], dtype=jnp.int32),
        travel_time=jnp.zeros((2,), dtype=jnp.float32),
    )

    with pytest.raises(ValueError, match="graph.num_links"):
        typical_cost_scale_from_assignment(
            graph=graph,
            link_flow=jnp.asarray([1.0], dtype=jnp.float32),
            link_cost=jnp.asarray([1.0], dtype=jnp.float32),
        )


def test_typical_cost_scale_rejects_nonpositive_scale():
    graph = _Graph(
        num_nodes=1,
        num_links=1,
        tail=jnp.asarray([0], dtype=jnp.int32),
        head=jnp.asarray([0], dtype=jnp.int32),
        link_type=jnp.asarray([LINK_TYPE_ACCESS], dtype=jnp.int32),
        travel_time=jnp.zeros((1,), dtype=jnp.float32),
    )

    with pytest.raises(ValueError, match="non-positive"):
        typical_cost_scale_from_assignment(
            graph=graph,
            link_flow=jnp.asarray([0.0], dtype=jnp.float32),
            link_cost=jnp.asarray([1.0], dtype=jnp.float32),
        )