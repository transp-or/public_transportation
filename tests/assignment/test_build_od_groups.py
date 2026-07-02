from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.assignment.build_od_groups import ODGroups, build_od_groups
from public_transportation.assignment.graph_sentinels import (
    LINK_TYPE_EGRESS,
    LINK_TYPE_RIDE,
    NODE_KIND_CENTROID_IN,
    NODE_KIND_CENTROID_OUT,
)


def _as_np(x):
    return np.asarray(x)


@dataclass(frozen=True)
class _TimeBin:
    bin_id: str | None = None


@dataclass(frozen=True)
class _DemandRecord:
    origin_stop_id: str
    dest_stop_id: str
    time_bin_index: int | None = None
    time_bin_id: str | None = None
    demand: float = 1.0


@dataclass(frozen=True)
class _Demand:
    records: list[_DemandRecord]


@dataclass(frozen=True)
class _Graph:
    num_nodes: int
    num_links: int
    node_stop_id: tuple[str, ...]
    node_kind: jnp.ndarray
    node_stop_index: jnp.ndarray
    link_type: jnp.ndarray
    head: jnp.ndarray


def _scenario(
    *,
    records: list[_DemandRecord],
    time_bins: list[_TimeBin] | None = None,
):
    if time_bins is None:
        time_bins = [_TimeBin("morning"), _TimeBin("midday")]
    return SimpleNamespace(
        demand=_Demand(records),
        time_bins=time_bins,
    )


def _graph(
    *,
    stop_ids: tuple[str, ...] = ("A", "B", "C"),
    num_time_bins: int = 2,
    egress_heads: list[int] | None = None,
) -> _Graph:
    """
    Minimal graph consistent with build_od_groups.

    Centroid-in nodes are ordered as expected by build_od_groups:
        for stop in sorted/declared stop ids:
            for time bin:
                centroid-in node

    Centroid-out nodes follow immediately after the centroid-in block.
    """
    num_stops = len(stop_ids)
    num_centroid_in = num_stops * num_time_bins
    num_centroid_out = num_stops
    num_nodes = num_centroid_in + num_centroid_out

    centroid_in_stop_indices = [
        stop_index
        for stop_index in range(num_stops)
        for _ in range(num_time_bins)
    ]
    centroid_out_stop_indices = list(range(num_stops))

    node_stop_index = np.asarray(
        centroid_in_stop_indices + centroid_out_stop_indices,
        dtype=int,
    )

    node_kind = np.full((num_nodes,), NODE_KIND_CENTROID_IN, dtype=int)
    node_kind[num_centroid_in:] = NODE_KIND_CENTROID_OUT

    if egress_heads is None:
        # One egress link to each centroid-out node, plus one non-egress link.
        egress_heads = list(range(num_centroid_in, num_centroid_in + num_centroid_out))

    link_type = [LINK_TYPE_EGRESS for _ in egress_heads] + [LINK_TYPE_RIDE]
    head = egress_heads + [0]

    return _Graph(
        num_nodes=num_nodes,
        num_links=len(head),
        node_stop_id=tuple(stop_ids),
        node_kind=jnp.asarray(node_kind, dtype=jnp.int32),
        node_stop_index=jnp.asarray(node_stop_index, dtype=jnp.int32),
        link_type=jnp.asarray(link_type, dtype=jnp.int32),
        head=jnp.asarray(head, dtype=jnp.int32),
    )


def test_build_od_groups_basic_shapes_and_alignment():
    graph = _graph()
    scenario = _scenario(
        records=[
            _DemandRecord("A", "C", time_bin_index=0, demand=10.0),
            _DemandRecord("B", "C", time_bin_index=1, demand=20.0),
            _DemandRecord("A", "B", time_bin_index=1, demand=30.0),
            _DemandRecord("C", "A", time_bin_index=0, demand=40.0),
        ],
    )

    groups = build_od_groups(scenario, graph=graph)

    assert isinstance(groups, ODGroups)
    assert groups.num_od == 4

    od_origin_node = _as_np(groups.od_origin_node)
    od_dest_node = _as_np(groups.od_dest_node)

    # stop_ids=("A","B","C"), two time bins:
    # A0=0, A1=1, B0=2, B1=3, C0=4, C1=5
    # centroid-out: A=6, B=7, C=8
    assert od_origin_node.tolist() == [0, 3, 1, 4]
    assert od_dest_node.tolist() == [8, 8, 7, 6]

    assert groups.group_start.ndim == 1
    assert groups.group_dest_node.ndim == 1
    assert groups.group_od_index.ndim == 1
    assert groups.group_link_mask.ndim == 2
    assert groups.group_link_mask.shape[1] == graph.num_links


def test_groups_are_destination_only_and_deterministic():
    graph = _graph()
    records = [
        _DemandRecord("A", "C", time_bin_index=0),
        _DemandRecord("B", "C", time_bin_index=0),
        _DemandRecord("A", "B", time_bin_index=1),
        _DemandRecord("B", "C", time_bin_index=1),
        _DemandRecord("C", "A", time_bin_index=0),
        _DemandRecord("A", "B", time_bin_index=0),
    ]
    scenario = _scenario(records=records)

    g1 = build_od_groups(scenario, graph=graph)
    g2 = build_od_groups(scenario, graph=graph)

    assert np.array_equal(_as_np(g1.group_start), _as_np(g2.group_start))
    assert np.array_equal(_as_np(g1.group_dest_node), _as_np(g2.group_dest_node))
    assert np.array_equal(_as_np(g1.group_od_index), _as_np(g2.group_od_index))
    assert np.array_equal(_as_np(g1.group_od_index_padded), _as_np(g2.group_od_index_padded))
    assert np.array_equal(_as_np(g1.group_od_mask), _as_np(g2.group_od_mask))
    assert np.array_equal(_as_np(g1.group_link_mask), _as_np(g2.group_link_mask))

    # Destination-only grouping: A=6, B=7, C=8.
    assert _as_np(g1.group_dest_node).tolist() == [6, 7, 8]

    od_dest_node = _as_np(g1.od_dest_node)
    group_start = _as_np(g1.group_start)
    group_od_index = _as_np(g1.group_od_index)
    group_dest_node = _as_np(g1.group_dest_node)

    for group_index in range(group_dest_node.size):
        sl = group_od_index[group_start[group_index] : group_start[group_index + 1]]
        assert sl.size > 0
        assert np.all(od_dest_node[sl] == group_dest_node[group_index])


def test_group_od_index_is_a_permutation_and_padded_representation_matches_csr():
    graph = _graph()
    scenario = _scenario(
        records=[
            _DemandRecord("A", "C", time_bin_index=0),
            _DemandRecord("B", "C", time_bin_index=1),
            _DemandRecord("A", "B", time_bin_index=1),
            _DemandRecord("C", "A", time_bin_index=0),
            _DemandRecord("B", "C", time_bin_index=0),
        ],
    )

    groups = build_od_groups(scenario, graph=graph)

    group_start = _as_np(groups.group_start)
    group_od_index = _as_np(groups.group_od_index)
    group_od_index_padded = _as_np(groups.group_od_index_padded)
    group_od_mask = _as_np(groups.group_od_mask)

    assert np.array_equal(np.sort(group_od_index), np.arange(groups.num_od))

    assert group_od_index_padded.ndim == 2
    assert group_od_mask.shape == group_od_index_padded.shape
    assert np.array_equal(
        np.sort(group_od_index_padded[group_od_mask]),
        np.arange(groups.num_od),
    )

    for group_index in range(group_start.size - 1):
        csr_slice = group_od_index[group_start[group_index] : group_start[group_index + 1]]
        padded_slice = group_od_index_padded[group_index][group_od_mask[group_index]]
        assert np.array_equal(padded_slice, csr_slice)


def test_enabled_link_mask_method_returns_one_group_mask():
    graph = _graph()
    scenario = _scenario(
        records=[
            _DemandRecord("A", "B", time_bin_index=0),
            _DemandRecord("A", "C", time_bin_index=0),
        ],
    )

    groups = build_od_groups(scenario, graph=graph)

    for group_index in range(groups.group_dest_node.shape[0]):
        mask = groups.enabled_link_mask(group_index, graph.num_links)
        assert mask.shape == (graph.num_links,)
        assert mask.dtype == jnp.bool_ or mask.dtype == bool


def test_enabled_link_mask_rejects_wrong_num_links():
    graph = _graph()
    scenario = _scenario(records=[_DemandRecord("A", "B", time_bin_index=0)])
    groups = build_od_groups(scenario, graph=graph)

    with pytest.raises(ValueError, match="group_link_mask"):
        groups.enabled_link_mask(0, graph.num_links + 1)


def test_group_link_mask_disables_egress_links_to_wrong_destinations_only():
    graph = _graph()
    scenario = _scenario(
        records=[
            _DemandRecord("A", "A", time_bin_index=0),
            _DemandRecord("A", "B", time_bin_index=0),
            _DemandRecord("A", "C", time_bin_index=0),
        ],
    )

    groups = build_od_groups(scenario, graph=graph)

    group_dest_node = _as_np(groups.group_dest_node)
    group_link_mask = _as_np(groups.group_link_mask)
    link_type = _as_np(graph.link_type)
    head = _as_np(graph.head)

    for group_index, dest_node in enumerate(group_dest_node):
        mask = group_link_mask[group_index]

        for link_index in range(graph.num_links):
            if link_type[link_index] == LINK_TYPE_EGRESS:
                assert mask[link_index] == (head[link_index] == dest_node)
            else:
                assert mask[link_index]


def test_make_initial_node_flow_injects_od_values_at_origin_nodes():
    graph = _graph()
    scenario = _scenario(
        records=[
            _DemandRecord("A", "C", time_bin_index=0),  # origin node 0
            _DemandRecord("B", "C", time_bin_index=1),  # origin node 3
            _DemandRecord("A", "B", time_bin_index=1),  # origin node 1
        ],
    )
    groups = build_od_groups(scenario, graph=graph)
    od_values = jnp.asarray([10.0, 20.0, 30.0], dtype=jnp.float32)

    # Find destination C group.
    group_dest_node = _as_np(groups.group_dest_node)
    c_group = int(np.where(group_dest_node == 8)[0][0])

    y0 = groups.make_initial_node_flow(c_group, od_values, graph.num_nodes)

    expected = np.zeros((graph.num_nodes,), dtype=float)
    expected[0] = 10.0
    expected[3] = 20.0

    assert np.allclose(_as_np(y0), expected)


def test_make_initial_node_flow_accumulates_multiple_od_records_same_origin():
    graph = _graph()
    scenario = _scenario(
        records=[
            _DemandRecord("A", "C", time_bin_index=0),
            _DemandRecord("A", "C", time_bin_index=0),
        ],
    )
    groups = build_od_groups(scenario, graph=graph)
    od_values = jnp.asarray([10.0, 15.0], dtype=jnp.float32)

    y0 = groups.make_initial_node_flow(0, od_values, graph.num_nodes)

    expected = np.zeros((graph.num_nodes,), dtype=float)
    expected[0] = 25.0
    assert np.allclose(_as_np(y0), expected)


def test_make_initial_node_flow_is_jittable():
    graph = _graph()
    scenario = _scenario(
        records=[
            _DemandRecord("A", "C", time_bin_index=0),
            _DemandRecord("B", "C", time_bin_index=1),
        ],
    )
    groups = build_od_groups(scenario, graph=graph)
    od_values = jnp.asarray([10.0, 20.0], dtype=jnp.float32)

    @jax.jit
    def build_flow(group_index):
        return groups.make_initial_node_flow(group_index, od_values, graph.num_nodes)

    y0 = build_flow(jnp.asarray(0, dtype=jnp.int32))

    expected = np.zeros((graph.num_nodes,), dtype=float)
    expected[0] = 10.0
    expected[3] = 20.0
    assert np.allclose(_as_np(y0), expected)


def test_odgroups_pytree_round_trip():
    graph = _graph()
    scenario = _scenario(
        records=[
            _DemandRecord("A", "B", time_bin_index=0),
            _DemandRecord("C", "B", time_bin_index=1),
        ],
    )
    groups = build_od_groups(scenario, graph=graph)

    children, treedef = jax.tree_util.tree_flatten(groups)
    rebuilt = jax.tree_util.tree_unflatten(treedef, children)

    assert rebuilt.num_od == groups.num_od
    assert np.array_equal(_as_np(rebuilt.od_origin_node), _as_np(groups.od_origin_node))
    assert np.array_equal(_as_np(rebuilt.od_dest_node), _as_np(groups.od_dest_node))
    assert np.array_equal(_as_np(rebuilt.group_start), _as_np(groups.group_start))
    assert np.array_equal(_as_np(rebuilt.group_dest_node), _as_np(groups.group_dest_node))
    assert np.array_equal(_as_np(rebuilt.group_od_index), _as_np(groups.group_od_index))
    assert np.array_equal(_as_np(rebuilt.group_od_index_padded), _as_np(groups.group_od_index_padded))
    assert np.array_equal(_as_np(rebuilt.group_od_mask), _as_np(groups.group_od_mask))
    assert np.array_equal(_as_np(rebuilt.group_link_mask), _as_np(groups.group_link_mask))


def test_time_bin_id_lookup_is_supported():
    graph = _graph()
    scenario = _scenario(
        time_bins=[_TimeBin("am"), _TimeBin("pm")],
        records=[
            _DemandRecord("A", "B", time_bin_id="am"),
            _DemandRecord("A", "B", time_bin_id="pm"),
        ],
    )

    groups = build_od_groups(scenario, graph=graph)

    assert _as_np(groups.od_origin_node).tolist() == [0, 1]


def test_time_bin_index_takes_precedence_over_time_bin_id():
    graph = _graph()
    scenario = _scenario(
        time_bins=[_TimeBin("am"), _TimeBin("pm")],
        records=[
            # Conflicting information: index=1 should be used, not id="am".
            _DemandRecord("A", "B", time_bin_index=1, time_bin_id="am"),
        ],
    )

    groups = build_od_groups(scenario, graph=graph)

    assert int(groups.od_origin_node[0]) == 1


def test_missing_demand_is_rejected():
    graph = _graph()
    scenario = SimpleNamespace(demand=None, time_bins=[_TimeBin("am")])

    with pytest.raises(ValueError, match="no demand"):
        build_od_groups(scenario, graph=graph)


def test_empty_demand_records_are_rejected():
    graph = _graph()
    scenario = _scenario(records=[])

    with pytest.raises(ValueError, match="zero records"):
        build_od_groups(scenario, graph=graph)


def test_missing_time_bins_are_rejected():
    graph = _graph()
    scenario = SimpleNamespace(
        demand=_Demand([_DemandRecord("A", "B", time_bin_index=0)]),
        time_bins=[],
    )

    with pytest.raises(ValueError, match="no time bins"):
        build_od_groups(scenario, graph=graph)


def test_missing_node_stop_id_metadata_is_rejected():
    graph = _graph()
    graph = _Graph(
        num_nodes=graph.num_nodes,
        num_links=graph.num_links,
        node_stop_id=(),
        node_kind=graph.node_kind,
        node_stop_index=graph.node_stop_index,
        link_type=graph.link_type,
        head=graph.head,
    )
    scenario = _scenario(records=[_DemandRecord("A", "B", time_bin_index=0)])

    with pytest.raises(ValueError, match="node_stop_id"):
        build_od_groups(scenario, graph=graph)


def test_graph_with_too_few_nodes_is_rejected():
    graph = _graph()
    graph = _Graph(
        num_nodes=2,
        num_links=graph.num_links,
        node_stop_id=graph.node_stop_id,
        node_kind=graph.node_kind[:2],
        node_stop_index=graph.node_stop_index[:2],
        link_type=graph.link_type,
        head=graph.head,
    )
    scenario = _scenario(records=[_DemandRecord("A", "B", time_bin_index=0)])

    with pytest.raises(ValueError, match="fewer nodes"):
        build_od_groups(scenario, graph=graph)


def test_inconsistent_centroid_in_kind_is_rejected():
    graph = _graph()
    node_kind = np.asarray(graph.node_kind).copy()
    node_kind[0] = NODE_KIND_CENTROID_OUT
    bad_graph = _Graph(
        num_nodes=graph.num_nodes,
        num_links=graph.num_links,
        node_stop_id=graph.node_stop_id,
        node_kind=jnp.asarray(node_kind, dtype=jnp.int32),
        node_stop_index=graph.node_stop_index,
        link_type=graph.link_type,
        head=graph.head,
    )
    scenario = _scenario(records=[_DemandRecord("A", "B", time_bin_index=0)])

    with pytest.raises(ValueError, match="centroid-in block"):
        build_od_groups(scenario, graph=bad_graph)


def test_inconsistent_centroid_in_stop_index_is_rejected():
    graph = _graph()
    node_stop_index = np.asarray(graph.node_stop_index).copy()
    node_stop_index[0] = 1
    bad_graph = _Graph(
        num_nodes=graph.num_nodes,
        num_links=graph.num_links,
        node_stop_id=graph.node_stop_id,
        node_kind=graph.node_kind,
        node_stop_index=jnp.asarray(node_stop_index, dtype=jnp.int32),
        link_type=graph.link_type,
        head=graph.head,
    )
    scenario = _scenario(records=[_DemandRecord("A", "B", time_bin_index=0)])

    with pytest.raises(ValueError, match="node_stop_index"):
        build_od_groups(scenario, graph=bad_graph)


def test_missing_centroid_out_nodes_are_rejected():
    graph = _graph()
    node_kind = np.asarray(graph.node_kind).copy()
    node_kind[node_kind == NODE_KIND_CENTROID_OUT] = NODE_KIND_CENTROID_IN
    bad_graph = _Graph(
        num_nodes=graph.num_nodes,
        num_links=graph.num_links,
        node_stop_id=graph.node_stop_id,
        node_kind=jnp.asarray(node_kind, dtype=jnp.int32),
        node_stop_index=graph.node_stop_index,
        link_type=graph.link_type,
        head=graph.head,
    )
    scenario = _scenario(records=[_DemandRecord("A", "B", time_bin_index=0)])

    with pytest.raises(ValueError, match="centroid-out"):
        build_od_groups(scenario, graph=bad_graph)


def test_invalid_centroid_out_stop_index_is_rejected():
    graph = _graph()
    node_stop_index = np.asarray(graph.node_stop_index).copy()
    node_stop_index[6] = 99
    bad_graph = _Graph(
        num_nodes=graph.num_nodes,
        num_links=graph.num_links,
        node_stop_id=graph.node_stop_id,
        node_kind=graph.node_kind,
        node_stop_index=jnp.asarray(node_stop_index, dtype=jnp.int32),
        link_type=graph.link_type,
        head=graph.head,
    )
    scenario = _scenario(records=[_DemandRecord("A", "B", time_bin_index=0)])

    with pytest.raises(ValueError, match="Invalid node_stop_index"):
        build_od_groups(scenario, graph=bad_graph)


def test_missing_time_bin_on_demand_record_is_rejected():
    graph = _graph()
    scenario = _scenario(records=[_DemandRecord("A", "B")])

    with pytest.raises(ValueError, match="missing time_bin"):
        build_od_groups(scenario, graph=graph)


def test_unknown_time_bin_index_is_rejected():
    graph = _graph()
    scenario = _scenario(records=[_DemandRecord("A", "B", time_bin_index=99)])

    with pytest.raises(ValueError, match="Unknown time bin index"):
        build_od_groups(scenario, graph=graph)


def test_unknown_time_bin_id_is_rejected():
    graph = _graph()
    scenario = _scenario(
        time_bins=[_TimeBin("am"), _TimeBin("pm")],
        records=[_DemandRecord("A", "B", time_bin_id="night")],
    )

    with pytest.raises(ValueError, match="Unknown time bin id"):
        build_od_groups(scenario, graph=graph)


def test_unknown_destination_stop_is_rejected():
    graph = _graph()
    scenario = _scenario(records=[_DemandRecord("A", "Z", time_bin_index=0)])

    with pytest.raises(ValueError, match="Unknown dest_stop_id"):
        build_od_groups(scenario, graph=graph)


def test_unknown_origin_stop_is_rejected():
    graph = _graph()
    scenario = _scenario(records=[_DemandRecord("Z", "B", time_bin_index=0)])

    with pytest.raises(ValueError, match="Unknown origin/time-bin combination"):
        build_od_groups(scenario, graph=graph)


def test_group_destination_must_be_centroid_out_node():
    graph = _graph()
    # Make B's centroid-out node look like a centroid-in node. This is caught
    # either while deriving centroid-out indices or during the final group sanity check.
    node_kind = np.asarray(graph.node_kind).copy()
    node_kind[7] = NODE_KIND_CENTROID_IN
    bad_graph = _Graph(
        num_nodes=graph.num_nodes,
        num_links=graph.num_links,
        node_stop_id=graph.node_stop_id,
        node_kind=jnp.asarray(node_kind, dtype=jnp.int32),
        node_stop_index=graph.node_stop_index,
        link_type=graph.link_type,
        head=graph.head,
    )
    scenario = _scenario(records=[_DemandRecord("A", "B", time_bin_index=0)])

    with pytest.raises(ValueError, match="Unknown dest_stop_id|centroid-out"):
        build_od_groups(scenario, graph=bad_graph)