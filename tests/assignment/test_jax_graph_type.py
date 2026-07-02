from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.assignment.jax_graph_types import (
    JaxGraph,
    JaxOD,
    ReferenceFlows,
)


def _mk_graph(
    *,
    with_time_bin: bool = True,
    with_bin_bounds: bool = True,
) -> JaxGraph:
    num_nodes = 3
    num_links = 2

    return JaxGraph(
        num_nodes=num_nodes,
        num_links=num_links,
        tail=jnp.asarray([0, 1], dtype=jnp.int32),
        head=jnp.asarray([1, 2], dtype=jnp.int32),
        topo_order=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        topo_order_rev=jnp.asarray([2, 1, 0], dtype=jnp.int32),
        node_time=jnp.asarray([-1.0e9, 10.0, 1.0e9], dtype=jnp.float32),
        node_stop_index=jnp.asarray([0, 1, 1], dtype=jnp.int32),
        node_time_s=jnp.asarray([-1, 600, -1], dtype=jnp.int32),
        node_kind=jnp.asarray([0, 2, 3], dtype=jnp.int32),
        node_trip_index=jnp.asarray([-1, 0, -1], dtype=jnp.int32),
        out_start=jnp.asarray([0, 1, 2, 2], dtype=jnp.int32),
        out_links_csr=jnp.asarray([0, 1], dtype=jnp.int32),
        out_links=jnp.asarray([[0], [1], [-1]], dtype=jnp.int32),
        out_mask=jnp.asarray([[True], [True], [False]], dtype=jnp.bool_),
        link_type=jnp.asarray([1, 4], dtype=jnp.int32),
        travel_time=jnp.asarray([10.0, 0.0], dtype=jnp.float32),
        capacity=jnp.asarray([100.0, jnp.inf], dtype=jnp.float32),
        link_trip_index=jnp.asarray([0, -1], dtype=jnp.int32),
        node_time_bin_index=(
            jnp.asarray([0, -1, -1], dtype=jnp.int32) if with_time_bin else None
        ),
        node_bin_start_min=(
            jnp.asarray([480.0, jnp.nan, jnp.nan], dtype=jnp.float32)
            if with_bin_bounds
            else None
        ),
        node_bin_end_min=(
            jnp.asarray([495.0, jnp.nan, jnp.nan], dtype=jnp.float32)
            if with_bin_bounds
            else None
        ),
        node_stop_id=("A", "B"),
        node_stop_name=("Stop A", "Stop B"),
        trip_id=("T1",),
        trip_line_ref=("L1",),
    )


def _assert_graph_arrays_equal(g1: JaxGraph, g2: JaxGraph) -> None:
    assert g2.num_nodes == g1.num_nodes
    assert g2.num_links == g1.num_links

    for field in (
        "tail",
        "head",
        "topo_order",
        "topo_order_rev",
        "node_time",
        "node_stop_index",
        "node_time_s",
        "node_kind",
        "node_trip_index",
        "out_start",
        "out_links_csr",
        "out_links",
        "out_mask",
        "link_type",
        "travel_time",
        "capacity",
        "link_trip_index",
    ):
        assert np.array_equal(np.asarray(getattr(g2, field)), np.asarray(getattr(g1, field)))

    assert g2.node_stop_id == g1.node_stop_id
    assert g2.node_stop_name == g1.node_stop_name
    assert g2.trip_id == g1.trip_id
    assert g2.trip_line_ref == g1.trip_line_ref


def test_jaxgraph_pytree_round_trip_preserves_required_fields():
    graph = _mk_graph()

    children, treedef = jax.tree_util.tree_flatten(graph)
    rebuilt = jax.tree_util.tree_unflatten(treedef, children)

    _assert_graph_arrays_equal(graph, rebuilt)


def test_jaxgraph_pytree_round_trip_preserves_optional_time_bin_fields():
    graph = _mk_graph(with_time_bin=True, with_bin_bounds=True)

    children, treedef = jax.tree_util.tree_flatten(graph)
    rebuilt = jax.tree_util.tree_unflatten(treedef, children)

    assert rebuilt.node_time_bin_index is not None
    assert rebuilt.node_bin_start_min is not None
    assert rebuilt.node_bin_end_min is not None

    assert np.array_equal(
        np.asarray(rebuilt.node_time_bin_index),
        np.asarray(graph.node_time_bin_index),
    )
    assert np.allclose(
        np.asarray(rebuilt.node_bin_start_min),
        np.asarray(graph.node_bin_start_min),
        equal_nan=True,
    )
    assert np.allclose(
        np.asarray(rebuilt.node_bin_end_min),
        np.asarray(graph.node_bin_end_min),
        equal_nan=True,
    )


def test_jaxgraph_pytree_round_trip_preserves_missing_optional_fields():
    graph = _mk_graph(with_time_bin=False, with_bin_bounds=False)

    children, treedef = jax.tree_util.tree_flatten(graph)
    rebuilt = jax.tree_util.tree_unflatten(treedef, children)

    assert rebuilt.node_time_bin_index is None
    assert rebuilt.node_bin_start_min is None
    assert rebuilt.node_bin_end_min is None


def test_jaxgraph_can_be_passed_to_jitted_function():
    graph = _mk_graph()

    @jax.jit
    def total_travel_time(g: JaxGraph):
        return jnp.sum(g.travel_time)

    assert np.isclose(float(total_travel_time(graph)), 10.0)


def test_jaxgraph_static_metadata_survives_jitted_use():
    graph = _mk_graph()

    @jax.jit
    def number_of_links(g: JaxGraph):
        return g.num_links + jnp.sum(g.out_mask)

    assert int(number_of_links(graph)) == 4


def test_jaxgraph_is_frozen():
    graph = _mk_graph()

    with pytest.raises(Exception):
        graph.num_nodes = 10


def test_jaxgraph_slots_prevent_unknown_attributes():
    graph = _mk_graph()

    with pytest.raises(Exception):
        graph.new_attribute = 1


def test_jaxod_pytree_round_trip():
    od = JaxOD(
        origin_node=jnp.asarray([0, 1], dtype=jnp.int32),
        dest_node=jnp.asarray([2, 3], dtype=jnp.int32),
        desired_time=jnp.asarray([480.0, 540.0], dtype=jnp.float32),
    )

    children, treedef = jax.tree_util.tree_flatten(od)
    rebuilt = jax.tree_util.tree_unflatten(treedef, children)

    assert np.array_equal(np.asarray(rebuilt.origin_node), np.asarray(od.origin_node))
    assert np.array_equal(np.asarray(rebuilt.dest_node), np.asarray(od.dest_node))
    assert np.allclose(np.asarray(rebuilt.desired_time), np.asarray(od.desired_time))


def test_jaxod_can_be_passed_to_jitted_function():
    od = JaxOD(
        origin_node=jnp.asarray([0, 1], dtype=jnp.int32),
        dest_node=jnp.asarray([2, 3], dtype=jnp.int32),
        desired_time=jnp.asarray([480.0, 540.0], dtype=jnp.float32),
    )

    @jax.jit
    def mean_desired_time(x: JaxOD):
        return jnp.mean(x.desired_time)

    assert np.isclose(float(mean_desired_time(od)), 510.0)


def test_jaxod_is_frozen():
    od = JaxOD(
        origin_node=jnp.asarray([0], dtype=jnp.int32),
        dest_node=jnp.asarray([1], dtype=jnp.int32),
        desired_time=jnp.asarray([480.0], dtype=jnp.float32),
    )

    with pytest.raises(Exception):
        od.origin_node = jnp.asarray([1], dtype=jnp.int32)


def test_reference_flows_pytree_round_trip():
    reference = ReferenceFlows(
        flow=jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
    )

    children, treedef = jax.tree_util.tree_flatten(reference)
    rebuilt = jax.tree_util.tree_unflatten(treedef, children)

    assert np.allclose(np.asarray(rebuilt.flow), np.asarray(reference.flow))


def test_reference_flows_can_be_passed_to_jitted_function():
    reference = ReferenceFlows(
        flow=jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
    )

    @jax.jit
    def total_flow(rf: ReferenceFlows):
        return jnp.sum(rf.flow)

    assert np.isclose(float(total_flow(reference)), 6.0)


def test_reference_flows_is_frozen():
    reference = ReferenceFlows(
        flow=jnp.asarray([1.0], dtype=jnp.float32),
    )

    with pytest.raises(Exception):
        reference.flow = jnp.asarray([2.0], dtype=jnp.float32)