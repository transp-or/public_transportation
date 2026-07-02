from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.assignment.dial_dp import (
    DialResult,
    compute_link_probabilities,
    compute_value_function,
    load_flows,
    run_dial_for_destination,
)
from public_transportation.assignment.jax_graph_types import JaxGraph


def _as_np(x):
    return np.asarray(x)


def _mk_jax_graph(
    *,
    num_nodes: int,
    tail: list[int],
    head: list[int],
    topo_order: list[int],
    out_links: list[list[int]],
    out_mask: list[list[bool]],
) -> JaxGraph:
    num_links = len(tail)

    return JaxGraph(
        num_nodes=num_nodes,
        num_links=num_links,
        tail=jnp.asarray(tail, dtype=jnp.int32),
        head=jnp.asarray(head, dtype=jnp.int32),
        topo_order=jnp.asarray(topo_order, dtype=jnp.int32),
        topo_order_rev=jnp.asarray(list(reversed(topo_order)), dtype=jnp.int32),
        node_time=jnp.zeros((num_nodes,), dtype=jnp.float32),
        node_stop_index=jnp.arange(num_nodes, dtype=jnp.int32),
        node_time_s=jnp.zeros((num_nodes,), dtype=jnp.int32),
        node_kind=jnp.zeros((num_nodes,), dtype=jnp.int32),
        node_trip_index=jnp.full((num_nodes,), -1, dtype=jnp.int32),
        out_start=jnp.zeros((num_nodes + 1,), dtype=jnp.int32),
        out_links_csr=jnp.arange(num_links, dtype=jnp.int32),
        out_links=jnp.asarray(out_links, dtype=jnp.int32),
        out_mask=jnp.asarray(out_mask, dtype=jnp.bool_),
        link_type=jnp.zeros((num_links,), dtype=jnp.int32),
        travel_time=jnp.ones((num_links,), dtype=jnp.float32),
        capacity=jnp.full((num_links,), jnp.inf, dtype=jnp.float32),
        link_trip_index=jnp.full((num_links,), -1, dtype=jnp.int32),
    )


def _mk_chain_graph() -> JaxGraph:
    # 0 -> 1
    return _mk_jax_graph(
        num_nodes=2,
        tail=[0],
        head=[1],
        topo_order=[0, 1],
        out_links=[
            [0],
            [-1],
        ],
        out_mask=[
            [True],
            [False],
        ],
    )


def _mk_diamond_graph() -> JaxGraph:
    #      1
    #    /   \
    #  0       3
    #    \   /
    #      2
    #
    # links:
    # 0: 0 -> 1
    # 1: 0 -> 2
    # 2: 1 -> 3
    # 3: 2 -> 3
    return _mk_jax_graph(
        num_nodes=4,
        tail=[0, 0, 1, 2],
        head=[1, 2, 3, 3],
        topo_order=[0, 1, 2, 3],
        out_links=[
            [0, 1],
            [2, -1],
            [3, -1],
            [-1, -1],
        ],
        out_mask=[
            [True, True],
            [True, False],
            [True, False],
            [False, False],
        ],
    )


def _mk_two_successor_graph() -> JaxGraph:
    # 0 -> 1
    # 0 -> 2
    return _mk_jax_graph(
        num_nodes=3,
        tail=[0, 0],
        head=[1, 2],
        topo_order=[0, 1, 2],
        out_links=[
            [0, 1],
            [-1, -1],
            [-1, -1],
        ],
        out_mask=[
            [True, True],
            [False, False],
            [False, False],
        ],
    )


def test_missing_padded_adjacency_is_rejected():
    graph = _mk_chain_graph()
    graph_without_adjacency = replace(
        graph,
        out_links=None,
        out_mask=None,
    )

    link_cost = jnp.asarray([1.0], dtype=jnp.float32)
    enabled = jnp.asarray([True])

    with pytest.raises(ValueError, match="padded adjacency arrays"):
        compute_value_function.__wrapped__(
            graph_without_adjacency,
            link_cost,
            enabled,
            dest_node=1,
            theta=1.0,
        )

    with pytest.raises(ValueError, match="padded adjacency arrays"):
        compute_link_probabilities.__wrapped__(
            graph_without_adjacency,
            link_cost,
            enabled,
            value=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
            theta=1.0,
        )

    with pytest.raises(ValueError, match="padded adjacency arrays"):
        load_flows.__wrapped__(
            graph_without_adjacency,
            link_prob=jnp.ones((1,), dtype=jnp.float32),
            enabled_link_mask=enabled,
            initial_node_flow=jnp.ones((2,), dtype=jnp.float32),
        )


def test_compute_value_function_chain():
    graph = _mk_chain_graph()
    link_cost = jnp.asarray([4.0], dtype=jnp.float32)
    enabled = jnp.asarray([True])

    value = compute_value_function.__wrapped__(
        graph,
        link_cost,
        enabled,
        dest_node=1,
        theta=1.0,
    )

    assert np.allclose(_as_np(value), [4.0, 0.0], atol=1e-6)


def test_compute_value_function_destination_is_zero():
    graph = _mk_diamond_graph()
    link_cost = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    enabled = jnp.asarray([True, True, True, True])

    value = compute_value_function.__wrapped__(
        graph,
        link_cost,
        enabled,
        dest_node=3,
        theta=1.0,
    )

    assert np.isclose(float(value[3]), 0.0, atol=1e-6)


def test_compute_value_function_diamond_matches_manual_logsumexp():
    graph = _mk_diamond_graph()
    theta = 2.0
    link_cost = jnp.asarray([1.0, 2.0, 3.0, 5.0], dtype=jnp.float32)
    enabled = jnp.asarray([True, True, True, True])

    value = compute_value_function.__wrapped__(
        graph,
        link_cost,
        enabled,
        dest_node=3,
        theta=theta,
    )

    expected_v1 = 3.0
    expected_v2 = 5.0
    expected_v0 = -theta * np.log(
        np.exp(-(1.0 + expected_v1) / theta)
        + np.exp(-(2.0 + expected_v2) / theta)
    )

    assert np.allclose(
        _as_np(value),
        [expected_v0, expected_v1, expected_v2, 0.0],
        atol=1e-6,
    )


def test_compute_value_function_ignores_disabled_links():
    graph = _mk_diamond_graph()
    theta = 1.0
    link_cost = jnp.asarray([1.0, 100.0, 2.0, 3.0], dtype=jnp.float32)
    enabled = jnp.asarray([True, False, True, True])

    value = compute_value_function.__wrapped__(
        graph,
        link_cost,
        enabled,
        dest_node=3,
        theta=theta,
    )

    # Only path 0 -> 1 -> 3 is enabled from node 0.
    assert np.allclose(float(value[0]), 3.0, atol=1e-6)


def test_compute_value_function_unreachable_node_gets_large_value():
    graph = _mk_chain_graph()
    link_cost = jnp.asarray([1.0], dtype=jnp.float32)
    enabled = jnp.asarray([False])

    value = compute_value_function.__wrapped__(
        graph,
        link_cost,
        enabled,
        dest_node=1,
        theta=1.0,
    )

    assert np.isclose(float(value[1]), 0.0, atol=1e-6)
    assert float(value[0]) == pytest.approx(1.0e6)


def test_compute_value_function_clips_small_theta_to_positive_floor():
    graph = _mk_chain_graph()
    link_cost = jnp.asarray([2.0], dtype=jnp.float32)
    enabled = jnp.asarray([True])

    value = compute_value_function.__wrapped__(
        graph,
        link_cost,
        enabled,
        dest_node=1,
        theta=0.0,
    )

    assert np.all(np.isfinite(_as_np(value)))
    assert np.allclose(_as_np(value), [2.0, 0.0], atol=1e-6)


def test_link_probabilities_chain_are_one():
    graph = _mk_chain_graph()
    link_cost = jnp.asarray([4.0], dtype=jnp.float32)
    enabled = jnp.asarray([True])
    value = jnp.asarray([4.0, 0.0], dtype=jnp.float32)

    prob = compute_link_probabilities.__wrapped__(
        graph,
        link_cost,
        enabled,
        value,
        theta=1.0,
    )

    assert np.allclose(_as_np(prob), [1.0], atol=1e-6)


def test_link_probabilities_two_successors_match_logit_formula():
    graph = _mk_two_successor_graph()
    link_cost = jnp.asarray([1.0, 3.0], dtype=jnp.float32)
    enabled = jnp.asarray([True, True])
    value = jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32)
    theta = 2.0

    prob = compute_link_probabilities.__wrapped__(
        graph,
        link_cost,
        enabled,
        value,
        theta=theta,
    )

    logits = np.asarray([-1.0 / theta, -3.0 / theta])
    expected = np.exp(logits) / np.exp(logits).sum()

    assert np.allclose(_as_np(prob), expected, atol=1e-6)
    assert np.isclose(float(prob.sum()), 1.0, atol=1e-6)


def test_link_probabilities_disabled_links_are_zero():
    graph = _mk_two_successor_graph()
    link_cost = jnp.asarray([1.0, 3.0], dtype=jnp.float32)
    enabled = jnp.asarray([True, False])
    value = jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32)

    prob = compute_link_probabilities.__wrapped__(
        graph,
        link_cost,
        enabled,
        value,
        theta=1.0,
    )

    assert np.allclose(_as_np(prob), [1.0, 0.0], atol=1e-6)


def test_link_probabilities_are_zero_when_no_enabled_outgoing_link():
    graph = _mk_chain_graph()
    link_cost = jnp.asarray([1.0], dtype=jnp.float32)
    enabled = jnp.asarray([False])
    value = jnp.asarray([1.0e6, 0.0], dtype=jnp.float32)

    prob = compute_link_probabilities.__wrapped__(
        graph,
        link_cost,
        enabled,
        value,
        theta=1.0,
    )

    assert np.allclose(_as_np(prob), [0.0], atol=1e-6)


def test_link_probabilities_are_finite_with_large_value_sentinel():
    graph = _mk_diamond_graph()
    link_cost = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    enabled = jnp.asarray([True, True, True, True])
    value = jnp.asarray([1.0e6, 3.0, 4.0, 0.0], dtype=jnp.float32)

    prob = compute_link_probabilities.__wrapped__(
        graph,
        link_cost,
        enabled,
        value,
        theta=1.0,
    )

    assert np.all(np.isfinite(_as_np(prob)))
    assert np.all(_as_np(prob) >= 0.0)


def test_load_flows_chain():
    graph = _mk_chain_graph()
    link_prob = jnp.asarray([1.0], dtype=jnp.float32)
    enabled = jnp.asarray([True])
    initial = jnp.asarray([10.0, 0.0], dtype=jnp.float32)

    node_flow, link_flow = load_flows.__wrapped__(
        graph,
        link_prob,
        enabled,
        initial,
    )

    assert np.allclose(_as_np(link_flow), [10.0], atol=1e-6)
    assert np.allclose(_as_np(node_flow), [10.0, 10.0], atol=1e-6)


def test_load_flows_diamond_splits_flow():
    graph = _mk_diamond_graph()
    link_prob = jnp.asarray([0.25, 0.75, 1.0, 1.0], dtype=jnp.float32)
    enabled = jnp.asarray([True, True, True, True])
    initial = jnp.asarray([20.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    node_flow, link_flow = load_flows.__wrapped__(
        graph,
        link_prob,
        enabled,
        initial,
    )

    assert np.allclose(_as_np(link_flow), [5.0, 15.0, 5.0, 15.0], atol=1e-6)
    assert np.allclose(_as_np(node_flow), [20.0, 5.0, 15.0, 20.0], atol=1e-6)


def test_load_flows_ignores_disabled_links_even_if_probability_positive():
    graph = _mk_two_successor_graph()
    link_prob = jnp.asarray([0.2, 0.8], dtype=jnp.float32)
    enabled = jnp.asarray([True, False])
    initial = jnp.asarray([10.0, 0.0, 0.0], dtype=jnp.float32)

    node_flow, link_flow = load_flows.__wrapped__(
        graph,
        link_prob,
        enabled,
        initial,
    )

    assert np.allclose(_as_np(link_flow), [2.0, 0.0], atol=1e-6)
    assert np.allclose(_as_np(node_flow), [10.0, 2.0, 0.0], atol=1e-6)


def test_run_dial_rejects_nonpositive_python_theta():
    graph = _mk_chain_graph()
    with pytest.raises(ValueError, match="theta must be positive"):
        run_dial_for_destination(
            graph=graph,
            link_cost=jnp.asarray([1.0], dtype=jnp.float32),
            enabled_link_mask=jnp.asarray([True]),
            dest_node=1,
            theta=0.0,
            initial_node_flow=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        )


def test_run_dial_for_destination_chain():
    graph = _mk_chain_graph()

    result = run_dial_for_destination(
        graph=graph,
        link_cost=jnp.asarray([2.5], dtype=jnp.float32),
        enabled_link_mask=jnp.asarray([True]),
        dest_node=1,
        theta=1.0,
        initial_node_flow=jnp.asarray([7.0, 0.0], dtype=jnp.float32),
    )

    assert isinstance(result, DialResult)
    assert np.allclose(_as_np(result.value), [2.5, 0.0], atol=1e-6)
    assert np.allclose(_as_np(result.link_prob), [1.0], atol=1e-6)
    assert np.allclose(_as_np(result.link_flow), [7.0], atol=1e-6)
    assert np.allclose(_as_np(result.node_flow), [7.0, 7.0], atol=1e-6)


def test_run_dial_for_destination_diamond_flow_conservation():
    graph = _mk_diamond_graph()
    result = run_dial_for_destination(
        graph=graph,
        link_cost=jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
        enabled_link_mask=jnp.asarray([True, True, True, True]),
        dest_node=3,
        theta=1.5,
        initial_node_flow=jnp.asarray([100.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
    )

    assert np.isclose(float(result.node_flow[0]), 100.0, atol=1e-5)
    assert np.isclose(float(result.node_flow[3]), 100.0, atol=1e-5)
    assert np.isclose(float(result.link_flow[:2].sum()), 100.0, atol=1e-5)
    assert np.isclose(float(result.link_flow[2:].sum()), 100.0, atol=1e-5)


def test_jitted_functions_match_eager_wrapped_versions():
    graph = _mk_diamond_graph()
    link_cost = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    enabled = jnp.asarray([True, True, True, True])
    theta = 2.0
    initial = jnp.asarray([10.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    value_eager = compute_value_function.__wrapped__(
        graph,
        link_cost,
        enabled,
        dest_node=3,
        theta=theta,
    )
    value_jit = compute_value_function(
        graph,
        link_cost,
        enabled,
        dest_node=3,
        theta=theta,
    )

    prob_eager = compute_link_probabilities.__wrapped__(
        graph,
        link_cost,
        enabled,
        value_eager,
        theta,
    )
    prob_jit = compute_link_probabilities(
        graph,
        link_cost,
        enabled,
        value_eager,
        theta,
    )

    node_flow_eager, link_flow_eager = load_flows.__wrapped__(
        graph,
        prob_eager,
        enabled,
        initial,
    )
    node_flow_jit, link_flow_jit = load_flows(
        graph,
        prob_eager,
        enabled,
        initial,
    )

    assert np.allclose(_as_np(value_jit), _as_np(value_eager), atol=1e-6)
    assert np.allclose(_as_np(prob_jit), _as_np(prob_eager), atol=1e-6)
    assert np.allclose(_as_np(node_flow_jit), _as_np(node_flow_eager), atol=1e-6)
    assert np.allclose(_as_np(link_flow_jit), _as_np(link_flow_eager), atol=1e-6)


def test_dial_result_is_pytree_round_trippable():
    result = DialResult(
        dest_node=jnp.asarray(1, dtype=jnp.int32),
        theta=jnp.asarray(2.0, dtype=jnp.float32),
        value=jnp.asarray([2.0, 0.0], dtype=jnp.float32),
        link_prob=jnp.asarray([1.0], dtype=jnp.float32),
        node_flow=jnp.asarray([5.0, 5.0], dtype=jnp.float32),
        link_flow=jnp.asarray([5.0], dtype=jnp.float32),
    )

    children, treedef = jax.tree_util.tree_flatten(result)
    rebuilt = jax.tree_util.tree_unflatten(treedef, children)

    assert isinstance(rebuilt, DialResult)
    assert np.allclose(_as_np(rebuilt.value), _as_np(result.value))
    assert np.allclose(_as_np(rebuilt.link_prob), _as_np(result.link_prob))
    assert np.allclose(_as_np(rebuilt.node_flow), _as_np(result.node_flow))
    assert np.allclose(_as_np(rebuilt.link_flow), _as_np(result.link_flow))