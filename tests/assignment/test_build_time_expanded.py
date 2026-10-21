"""
Comprehensive tests for public_transportation.assignment.build_time_expanded.

The tests are written against the public contract of build_jax_graph and the
graph semantics, not against fragile implementation details. They should remain
useful if JaxGraph later receives additional optional metadata, for example
topological layers for a level-based JAX implementation.

Suggested location in the repository:
    tests/assignment/test_build_time_expanded_comprehensive.py
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np
import pytest

from public_transportation.assignment.build_time_expanded import build_jax_graph
from public_transportation.assignment.config import AssignmentConfig
from public_transportation.assignment.costs import precompute_base_costs
from public_transportation.assignment.graph_sentinels import (CENTROID_TIME_S, LINK_TYPE_ACCESS, LINK_TYPE_DWELL,
                                                              LINK_TYPE_EGRESS, LINK_TYPE_RIDE, LINK_TYPE_TRANSFER,
                                                              NODE_KIND_CENTROID_IN, NODE_KIND_CENTROID_OUT,
                                                              NODE_KIND_EVENT_ARR, NODE_KIND_EVENT_DEP)
from public_transportation.assignment.graph_sentinels import LINK_TYPE_HUB_CONNECTOR
from public_transportation.domain.transfer_geometry import TransferGeometry


# ---------------------------------------------------------------------------
# Lightweight domain stubs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Stop:
    stop_id: str
    name: str | None = None


@dataclass(frozen=True, slots=True)
class _Trip:
    trip_id: str
    line_ref: str
    capacity: float | None = None


@dataclass(frozen=True, slots=True)
class _StopTime:
    trip_id: str
    stop_id: str
    stop_sequence: int
    arrival_time: int
    departure_time: int


@dataclass(frozen=True, slots=True)
class _TimeBin:
    start_s: int
    end_s: int


@dataclass(frozen=True, slots=True)
class _Timetable:
    trips: list[_Trip]
    stop_times: list[_StopTime]


@dataclass(frozen=True, slots=True)
class _Scenario:
    stops: list[_Stop] | dict[str, _Stop]
    timetable: _Timetable | None
    time_bins: list[_TimeBin]
    transfer_geometry: TransferGeometry | None = None


def _cfg(**kwargs: Any) -> AssignmentConfig:
    values = dict(
        max_access_deviation_min=10.0,
        max_transfer_wait_min=10.0,
        min_dwell_s=1,
    )
    values.update(kwargs)
    return AssignmentConfig(**values)


def _scenario(
    *,
    stops: list[str] = ("A", "B"),
    time_bins: list[_TimeBin] | None = None,
    trips: list[_Trip] | None = None,
    stop_times: list[_StopTime] | None = None,
    stops_as_dict: bool = False,
    transfer_geometry: TransferGeometry | None = None,
) -> _Scenario:
    stop_objects = [_Stop(s) for s in stops]
    stop_container: list[_Stop] | dict[str, _Stop]
    if stops_as_dict:
        stop_container = {s.stop_id: s for s in stop_objects}
    else:
        stop_container = stop_objects

    return _Scenario(
        stops=stop_container,
        timetable=_Timetable(
            trips=trips
            if trips is not None
            else [_Trip("T1", line_ref="L1", capacity=33.0)],
            stop_times=stop_times
            if stop_times is not None
            else [
                _StopTime("T1", "A", 1, arrival_time=0, departure_time=1),
                _StopTime("T1", "B", 2, arrival_time=300, departure_time=301),
            ],
        ),
        time_bins=time_bins if time_bins is not None else [_TimeBin(0, 600)],
        transfer_geometry=transfer_geometry,
    )


def _as_np(x):
    return np.asarray(x)


def _link_indices_by_type(graph, link_type: int) -> np.ndarray:
    return np.flatnonzero(_as_np(graph.link_type).astype(int) == int(link_type))


def _node_indices_by_kind(graph, node_kind: int) -> np.ndarray:
    return np.flatnonzero(_as_np(graph.node_kind).astype(int) == int(node_kind))


def _stop_index_by_id(graph) -> dict[str, int]:
    return {stop_id: i for i, stop_id in enumerate(graph.node_stop_id)}


def _event_node(
    graph,
    *,
    stop_id: str,
    time_s: int,
    kind: int,
    trip_index: int = 0,
) -> int:
    stop_index = _stop_index_by_id(graph)[stop_id]
    node_kind = _as_np(graph.node_kind).astype(int)
    node_stop_index = _as_np(graph.node_stop_index).astype(int)
    node_time_s = _as_np(graph.node_time_s).astype(int)
    node_trip_index = _as_np(graph.node_trip_index).astype(int)

    matches = np.flatnonzero(
        (node_kind == int(kind))
        & (node_stop_index == int(stop_index))
        & (node_time_s == int(time_s))
        & (node_trip_index == int(trip_index))
    )
    assert matches.size == 1
    return int(matches[0])


def _centroid_in_node(graph, *, stop_id: str, time_bin_index: int) -> int:
    stop_index = _stop_index_by_id(graph)[stop_id]
    node_kind = _as_np(graph.node_kind).astype(int)
    node_stop_index = _as_np(graph.node_stop_index).astype(int)
    node_time_bin_index = _as_np(graph.node_time_bin_index).astype(int)

    matches = np.flatnonzero(
        (node_kind == NODE_KIND_CENTROID_IN)
        & (node_stop_index == stop_index)
        & (node_time_bin_index == int(time_bin_index))
    )
    assert matches.size == 1
    return int(matches[0])


def _centroid_out_node(graph, *, stop_id: str) -> int:
    stop_index = _stop_index_by_id(graph)[stop_id]
    node_kind = _as_np(graph.node_kind).astype(int)
    node_stop_index = _as_np(graph.node_stop_index).astype(int)

    matches = np.flatnonzero(
        (node_kind == NODE_KIND_CENTROID_OUT)
        & (node_stop_index == stop_index)
    )
    assert matches.size == 1
    return int(matches[0])


def _has_path(graph, source: int, target: int) -> bool:
    tail = _as_np(graph.tail).astype(int)
    head = _as_np(graph.head).astype(int)
    outgoing: list[list[int]] = [[] for _ in range(graph.num_nodes)]
    for link, node in enumerate(tail):
        outgoing[int(node)].append(link)
    pending = [source]
    visited = {source}
    while pending:
        node = pending.pop()
        if node == target:
            return True
        for link in outgoing[node]:
            successor = int(head[link])
            if successor not in visited:
                visited.add(successor)
                pending.append(successor)
    return False


def _path_links(graph, source: int, target: int) -> list[int]:
    tail = _as_np(graph.tail).astype(int)
    head = _as_np(graph.head).astype(int)
    outgoing: list[list[int]] = [[] for _ in range(graph.num_nodes)]
    for link, node in enumerate(tail):
        outgoing[int(node)].append(link)
    pending = [source]
    parent: dict[int, tuple[int, int]] = {}
    visited = {source}
    while pending:
        node = pending.pop(0)
        if node == target:
            break
        for link in outgoing[node]:
            successor = int(head[link])
            if successor not in visited:
                visited.add(successor)
                parent[successor] = (node, link)
                pending.append(successor)
    if target not in visited:
        raise AssertionError(f"No path from node {source} to {target}.")
    result: list[int] = []
    node = target
    while node != source:
        previous, link = parent[node]
        result.append(link)
        node = previous
    return list(reversed(result))


def _hub_transfer_scenario(
    *,
    x2_station: str = "H",
    l2_arrival_s: int = 500,
    l1_cross_departure_s: int = 540,
    distance_x1_m: float = 30.0,
    distance_x2_m: float = 60.0,
) -> _Scenario:
    stops = ["A", "X1", "X2", "D"]
    trips = [
        _Trip("T1", "L1"),
        _Trip("T2", "L2"),
        _Trip("T3", "L1"),
        _Trip("T4", "L2"),
    ]
    stop_times = [
        _StopTime("T1", "A", 1, 50, 60),
        _StopTime("T1", "X1", 2, 300, 301),
        _StopTime("T2", "X2", 1, l2_arrival_s - 1, l2_arrival_s),
        _StopTime("T2", "D", 2, l2_arrival_s + 200, l2_arrival_s + 201),
        _StopTime("T3", "X2", 1, l1_cross_departure_s - 1, l1_cross_departure_s),
        _StopTime("T3", "D", 2, l1_cross_departure_s + 200, l1_cross_departure_s + 201),
        _StopTime("T4", "X1", 1, 390, 391),
        _StopTime("T4", "D", 2, 590, 591),
    ]
    membership = {"A": "A", "X1": "H", "X2": x2_station, "D": "D"}
    distances = {("H", "X1"): distance_x1_m}
    distances[(x2_station, "X2")] = distance_x2_m
    return _scenario(
        stops=stops,
        time_bins=[_TimeBin(0, 1200)],
        trips=trips,
        stop_times=stop_times,
        transfer_geometry=TransferGeometry(membership, distances),
    )


# ---------------------------------------------------------------------------
# Structural graph invariants
# ---------------------------------------------------------------------------


def test_build_jax_graph_has_expected_basic_shapes_and_metadata():
    g = build_jax_graph(_scenario(), config=_cfg())

    assert g.num_nodes == int(_as_np(g.node_kind).shape[0])
    assert g.num_nodes == int(_as_np(g.node_time).shape[0])
    assert g.num_nodes == int(_as_np(g.node_time_s).shape[0])
    assert g.num_nodes == int(_as_np(g.node_stop_index).shape[0])
    assert g.num_nodes == int(_as_np(g.node_trip_index).shape[0])
    assert g.num_nodes == int(_as_np(g.node_time_bin_index).shape[0])
    assert g.num_nodes == int(_as_np(g.node_bin_start_min).shape[0])
    assert g.num_nodes == int(_as_np(g.node_bin_end_min).shape[0])

    assert g.num_links == int(_as_np(g.tail).shape[0])
    assert g.num_links == int(_as_np(g.head).shape[0])
    assert g.num_links == int(_as_np(g.link_type).shape[0])
    assert g.num_links == int(_as_np(g.travel_time).shape[0])
    assert g.num_links == int(_as_np(g.capacity).shape[0])
    assert g.num_links == int(_as_np(g.link_trip_index).shape[0])

    assert tuple(g.node_stop_id) == ("A", "B")
    assert tuple(g.trip_id) == ("T1",)
    assert tuple(g.trip_line_ref) == ("L1",)
    assert len(g.node_station_id) == g.num_nodes
    assert g.transfer_wait_time.shape == (g.num_links,)
    assert g.transfer_walking_time.shape == (g.num_links,)
    assert g.assignment_semantics_fingerprint


def test_graph_arrays_have_valid_index_ranges():
    g = build_jax_graph(_scenario(), config=_cfg())

    tail = _as_np(g.tail).astype(int)
    head = _as_np(g.head).astype(int)
    node_stop_index = _as_np(g.node_stop_index).astype(int)
    node_time_bin_index = _as_np(g.node_time_bin_index).astype(int)
    node_trip_index = _as_np(g.node_trip_index).astype(int)
    link_trip_index = _as_np(g.link_trip_index).astype(int)

    assert np.all((0 <= tail) & (tail < g.num_nodes))
    assert np.all((0 <= head) & (head < g.num_nodes))
    assert np.all((0 <= node_stop_index) & (node_stop_index < len(g.node_stop_id)))

    valid_time_bin = node_time_bin_index >= 0
    assert np.all(node_time_bin_index[~valid_time_bin] == -1)
    assert np.all(node_time_bin_index[valid_time_bin] < len(_scenario().time_bins))

    valid_node_trip = node_trip_index >= 0
    assert np.all(node_trip_index[~valid_node_trip] == -1)
    assert np.all(node_trip_index[valid_node_trip] < len(g.trip_id))

    valid_link_trip = link_trip_index >= 0
    assert np.all(link_trip_index[~valid_link_trip] == -1)
    assert np.all(link_trip_index[valid_link_trip] < len(g.trip_id))


def test_topological_order_is_a_permutation_and_respects_all_links():
    g = build_jax_graph(_scenario(), config=_cfg())

    topo = _as_np(g.topo_order).astype(int)
    topo_rev = _as_np(g.topo_order_rev).astype(int)

    assert np.array_equal(np.sort(topo), np.arange(g.num_nodes))
    assert np.array_equal(topo_rev, topo[::-1])

    position = np.empty(g.num_nodes, dtype=int)
    position[topo] = np.arange(g.num_nodes)

    tail = _as_np(g.tail).astype(int)
    head = _as_np(g.head).astype(int)

    assert np.all(position[tail] < position[head])


def test_jaxgraph_is_registered_as_a_pytree():
    g = build_jax_graph(_scenario(), config=_cfg())

    leaves, treedef = jax.tree_util.tree_flatten(g)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert rebuilt.num_nodes == g.num_nodes
    assert rebuilt.num_links == g.num_links
    assert tuple(rebuilt.node_stop_id) == tuple(g.node_stop_id)
    assert tuple(rebuilt.trip_id) == tuple(g.trip_id)
    assert tuple(rebuilt.trip_line_ref) == tuple(g.trip_line_ref)
    assert np.array_equal(_as_np(rebuilt.tail), _as_np(g.tail))
    assert np.array_equal(_as_np(rebuilt.head), _as_np(g.head))
    assert rebuilt.node_station_id == g.node_station_id
    assert (
        rebuilt.assignment_semantics_fingerprint
        == g.assignment_semantics_fingerprint
    )


def test_interplatform_transfer_is_enabled_and_cost_is_decomposed():
    scenario = _hub_transfer_scenario()
    config = _cfg(beta_transfer=1.5, beta_walk=1.0, walking_speed_m_per_s=1.0)
    graph = build_jax_graph(scenario, config=config)
    tail = _event_node(
        graph,
        stop_id="X1",
        time_s=300,
        kind=NODE_KIND_EVENT_ARR,
        trip_index=0,
    )
    head = _event_node(
        graph,
        stop_id="X2",
        time_s=500,
        kind=NODE_KIND_EVENT_DEP,
        trip_index=1,
    )
    assert _has_path(graph, tail, head)
    links = _path_links(graph, tail, head)
    link_types = _as_np(graph.link_type).astype(int)
    assert np.count_nonzero(link_types[links] == LINK_TYPE_TRANSFER) == 1
    assert np.count_nonzero(link_types[links] == LINK_TYPE_HUB_CONNECTOR) > 0

    walking = float(np.sum(_as_np(graph.transfer_walking_time)[links]))
    waiting = float(np.sum(_as_np(graph.transfer_wait_time)[links]))
    assert walking == pytest.approx(1.5)
    assert waiting == pytest.approx(200.0 / 60.0 - 1.5)
    parts = precompute_base_costs(graph, config)
    path_cost = float(np.sum(_as_np(parts.base_cost)[links]))
    assert path_cost == pytest.approx(1.5 * (200.0 / 60.0 - 1.5) + 1.5)
    assert graph.node_station_id[tail] == "H"
    assert graph.node_station_id[head] == "H"


def test_interplatform_transfer_requires_same_station_and_different_line():
    graph = build_jax_graph(
        _hub_transfer_scenario(x2_station="J"), config=_cfg()
    )
    arrival = _event_node(
        graph, stop_id="X1", time_s=300, kind=NODE_KIND_EVENT_ARR, trip_index=0
    )
    different_line_departure = _event_node(
        graph, stop_id="X2", time_s=500, kind=NODE_KIND_EVENT_DEP, trip_index=1
    )
    same_line_departure = _event_node(
        graph, stop_id="X2", time_s=540, kind=NODE_KIND_EVENT_DEP, trip_index=2
    )
    same_platform_other_line = _event_node(
        graph, stop_id="X1", time_s=391, kind=NODE_KIND_EVENT_DEP, trip_index=3
    )
    assert not _has_path(graph, arrival, different_line_departure)
    # Different trip IDs on the same line are not transfer eligibility.
    assert not _has_path(graph, arrival, same_line_departure)
    # Same platform, different line remains a valid transfer.
    assert _has_path(graph, arrival, same_platform_other_line)
    links = _path_links(graph, arrival, same_platform_other_line)
    assert float(np.sum(_as_np(graph.transfer_walking_time)[links])) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("departure_s", "maximum_wait_min", "expected"),
    [(330, 10.0, True), (500, 2.0, False), (300, 10.0, False)],
)
def test_transfer_timetable_window_and_short_walking_gap(
    departure_s: int, maximum_wait_min: float, expected: bool
):
    graph = build_jax_graph(
        _hub_transfer_scenario(l2_arrival_s=departure_s),
        config=_cfg(
            max_transfer_wait_min=maximum_wait_min,
            walking_speed_m_per_s=1.0,
        ),
    )
    arrival = _event_node(
        graph, stop_id="X1", time_s=300, kind=NODE_KIND_EVENT_ARR, trip_index=0
    )
    departure = _event_node(
        graph,
        stop_id="X2",
        time_s=departure_s,
        kind=NODE_KIND_EVENT_DEP,
        trip_index=1,
    )
    assert _has_path(graph, arrival, departure) is expected
    if departure_s == 330:
        links = _path_links(graph, arrival, departure)
        assert float(np.sum(_as_np(graph.transfer_wait_time)[links])) == pytest.approx(0.0)
        assert float(np.sum(_as_np(graph.transfer_walking_time)[links])) == pytest.approx(1.5)


def test_cross_platform_candidate_requires_both_hub_distances():
    scenario = _hub_transfer_scenario()
    scenario = replace(
        scenario,
        transfer_geometry=TransferGeometry(
            scenario.transfer_geometry.platform_to_station,
        {("H", "X1"): 30.0},
        ),
    )
    with pytest.raises(ValueError, match="Missing hub distance"):
        build_jax_graph(scenario, config=_cfg())


def test_hub_transfer_graph_growth_is_subquadratic_in_departure_events():
    count = 20
    stops = ["X1", "X2"]
    trips: list[_Trip] = []
    stop_times: list[_StopTime] = []
    membership = {"X1": "H", "X2": "H"}
    for index in range(count):
        origin = f"O{index}"
        destination = f"D{index}"
        first_trip = f"A{index}"
        second_trip = f"B{index}"
        stops.extend((origin, destination))
        membership[origin] = origin
        membership[destination] = destination
        trips.extend((_Trip(first_trip, "L1"), _Trip(second_trip, "L2")))
        arrival_s = 100 + index * 2
        departure_s = 500 + index * 2
        stop_times.extend(
            (
                _StopTime(first_trip, origin, 1, 20, 21),
                _StopTime(first_trip, "X1", 2, arrival_s, arrival_s + 1),
                _StopTime(second_trip, "X2", 1, departure_s - 1, departure_s),
                _StopTime(second_trip, destination, 2, departure_s + 100, departure_s + 101),
            )
        )
    scenario = _scenario(
        stops=stops,
        time_bins=[_TimeBin(0, 1000)],
        trips=trips,
        stop_times=stop_times,
        transfer_geometry=TransferGeometry(
            membership, {("H", "X1"): 0.0, ("H", "X2"): 0.0}
        ),
    )
    graph = build_jax_graph(scenario, config=_cfg(max_transfer_wait_min=30.0))
    link_types = _as_np(graph.link_type).astype(int)
    hub_links = np.count_nonzero(
        (link_types == LINK_TYPE_TRANSFER)
        | (link_types == LINK_TYPE_HUB_CONNECTOR)
    )
    assert hub_links < count * count


# ---------------------------------------------------------------------------
# Node construction
# ---------------------------------------------------------------------------


def test_node_counts_and_kinds_for_single_trip_two_stops():
    g = build_jax_graph(_scenario(), config=_cfg())

    kinds = _as_np(g.node_kind).astype(int)

    # 2 stops * 1 time bin = 2 centroid-in nodes
    assert int((kinds == NODE_KIND_CENTROID_IN).sum()) == 2

    # 2 stop_times, each split into ARR and DEP
    assert int((kinds == NODE_KIND_EVENT_ARR).sum()) == 2
    assert int((kinds == NODE_KIND_EVENT_DEP).sum()) == 2

    # 2 centroid-out nodes
    assert int((kinds == NODE_KIND_CENTROID_OUT).sum()) == 2


def test_centroid_in_nodes_store_time_bin_metadata_only_on_centroid_in_nodes():
    scenario = _scenario(
        stops=["A", "B"],
        time_bins=[_TimeBin(0, 600), _TimeBin(3600, 4200)],
    )
    g = build_jax_graph(scenario, config=_cfg())

    kind = _as_np(g.node_kind).astype(int)
    tbi = _as_np(g.node_time_bin_index).astype(int)
    bstart = _as_np(g.node_bin_start_min).astype(float)
    bend = _as_np(g.node_bin_end_min).astype(float)
    time_s = _as_np(g.node_time_s).astype(int)

    centroid_in = kind == NODE_KIND_CENTROID_IN
    assert int(centroid_in.sum()) == 4
    assert set(tbi[centroid_in].tolist()) == {0, 1}
    assert np.all(np.isfinite(bstart[centroid_in]))
    assert np.all(np.isfinite(bend[centroid_in]))
    assert np.all(time_s[centroid_in] == CENTROID_TIME_S)

    non_centroid_in = ~centroid_in
    assert np.all(tbi[non_centroid_in] == -1)
    assert np.all(np.isnan(bstart[non_centroid_in]))
    assert np.all(np.isnan(bend[non_centroid_in]))


def test_event_nodes_are_trip_specific_even_when_stop_and_time_coincide():
    scenario = _scenario(
        stops=["A", "B"],
        trips=[
            _Trip("T1", line_ref="L1"),
            _Trip("T2", line_ref="L2"),
        ],
        stop_times=[
            _StopTime("T1", "A", 1, arrival_time=0, departure_time=1),
            _StopTime("T1", "B", 2, arrival_time=300, departure_time=301),
            _StopTime("T2", "A", 1, arrival_time=0, departure_time=1),
            _StopTime("T2", "B", 2, arrival_time=300, departure_time=301),
        ],
    )

    g = build_jax_graph(scenario, config=_cfg())

    kinds = _as_np(g.node_kind).astype(int)
    assert int((kinds == NODE_KIND_EVENT_ARR).sum()) == 4
    assert int((kinds == NODE_KIND_EVENT_DEP).sum()) == 4

    # Same stop and same time, but different trips.
    assert _event_node(g, stop_id="A", time_s=0, kind=NODE_KIND_EVENT_ARR, trip_index=0) != _event_node(
        g, stop_id="A", time_s=0, kind=NODE_KIND_EVENT_ARR, trip_index=1
    )
    assert _event_node(g, stop_id="A", time_s=1, kind=NODE_KIND_EVENT_DEP, trip_index=0) != _event_node(
        g, stop_id="A", time_s=1, kind=NODE_KIND_EVENT_DEP, trip_index=1
    )


def test_stops_can_be_provided_as_dict_keys():
    scenario = _scenario(stops=["B", "A"], stops_as_dict=True)
    g = build_jax_graph(scenario, config=_cfg())

    # The builder uses sorted stop ids.
    assert tuple(g.node_stop_id) == ("A", "B")


# ---------------------------------------------------------------------------
# Link construction semantics
# ---------------------------------------------------------------------------


def test_ride_link_connects_departure_to_next_arrival_with_capacity_and_trip_index():
    g = build_jax_graph(_scenario(), config=_cfg())

    dep_a = _event_node(g, stop_id="A", time_s=1, kind=NODE_KIND_EVENT_DEP)
    arr_b = _event_node(g, stop_id="B", time_s=300, kind=NODE_KIND_EVENT_ARR)

    ride_idx = _link_indices_by_type(g, LINK_TYPE_RIDE)
    assert ride_idx.size == 1

    i = int(ride_idx[0])
    assert int(g.tail[i]) == dep_a
    assert int(g.head[i]) == arr_b
    assert np.isclose(float(g.travel_time[i]), (300 - 1) / 60.0)
    assert np.isclose(float(g.capacity[i]), 33.0)
    assert int(g.link_trip_index[i]) == 0


def test_dwell_link_connects_arrival_to_departure_same_stop_time():
    g = build_jax_graph(_scenario(), config=_cfg())

    arr_a = _event_node(g, stop_id="A", time_s=0, kind=NODE_KIND_EVENT_ARR)
    dep_a = _event_node(g, stop_id="A", time_s=1, kind=NODE_KIND_EVENT_DEP)

    dwell_idx = _link_indices_by_type(g, LINK_TYPE_DWELL)
    dwell_edges = {(int(g.tail[i]), int(g.head[i])) for i in dwell_idx.tolist()}

    assert (arr_a, dep_a) in dwell_edges
    assert np.all(_as_np(g.travel_time)[dwell_idx] > 0.0)


def test_egress_links_go_from_arrival_events_to_same_stop_centroid_out():
    g = build_jax_graph(_scenario(), config=_cfg())

    egress_idx = _link_indices_by_type(g, LINK_TYPE_EGRESS)
    assert egress_idx.size == 2

    node_kind = _as_np(g.node_kind).astype(int)
    node_stop_index = _as_np(g.node_stop_index).astype(int)

    for i in egress_idx:
        tail = int(g.tail[i])
        head = int(g.head[i])
        assert node_kind[tail] == NODE_KIND_EVENT_ARR
        assert node_kind[head] == NODE_KIND_CENTROID_OUT
        assert node_stop_index[tail] == node_stop_index[head]
        assert float(g.travel_time[i]) == 0.0
        assert int(g.link_trip_index[i]) == -1


def test_access_links_only_leave_centroid_in_nodes_and_enter_departure_events_at_same_stop():
    scenario = _scenario(
        stops=["A", "B"],
        time_bins=[_TimeBin(0, 600)],
        trips=[_Trip("T1", line_ref="L1")],
        stop_times=[
            _StopTime("T1", "A", 1, arrival_time=0, departure_time=60),
            _StopTime("T1", "B", 2, arrival_time=300, departure_time=301),
        ],
    )
    g = build_jax_graph(scenario, config=_cfg(max_access_deviation_min=1.0))

    access_idx = _link_indices_by_type(g, LINK_TYPE_ACCESS)
    assert access_idx.size >= 1

    kind = _as_np(g.node_kind).astype(int)
    node_stop_index = _as_np(g.node_stop_index).astype(int)

    for i in access_idx:
        tail = int(g.tail[i])
        head = int(g.head[i])
        assert kind[tail] == NODE_KIND_CENTROID_IN
        assert kind[head] == NODE_KIND_EVENT_DEP
        assert node_stop_index[tail] == node_stop_index[head]
        assert float(g.travel_time[i]) == 0.0
        assert int(g.link_trip_index[i]) >= 0


def test_access_window_controls_which_departures_are_reachable():
    scenario = _scenario(
        stops=["A", "B"],
        time_bins=[_TimeBin(600, 600)],  # exactly 10:00 minutes
        trips=[_Trip("T1", line_ref="L1"), _Trip("T2", line_ref="L1")],
        stop_times=[
            _StopTime("T1", "A", 1, arrival_time=0, departure_time=300),   # too early if tolerance is 1 min
            _StopTime("T1", "B", 2, arrival_time=360, departure_time=361),
            _StopTime("T2", "A", 1, arrival_time=0, departure_time=600),   # exactly in window
            _StopTime("T2", "B", 2, arrival_time=660, departure_time=661),
        ],
    )
    g = build_jax_graph(scenario, config=_cfg(max_access_deviation_min=1.0))

    c_in_a = _centroid_in_node(g, stop_id="A", time_bin_index=0)
    dep_a_t1 = _event_node(g, stop_id="A", time_s=300, kind=NODE_KIND_EVENT_DEP, trip_index=0)
    dep_a_t2 = _event_node(g, stop_id="A", time_s=600, kind=NODE_KIND_EVENT_DEP, trip_index=1)

    access_edges = {
        (int(g.tail[i]), int(g.head[i]))
        for i in _link_indices_by_type(g, LINK_TYPE_ACCESS).tolist()
    }

    assert (c_in_a, dep_a_t1) not in access_edges
    assert (c_in_a, dep_a_t2) in access_edges


def test_transfer_links_are_inter_line_only_and_within_wait_window():
    scenario = _scenario(
        stops=["A", "B", "C"],
        trips=[
            _Trip("T1", line_ref="L1"),
            _Trip("T2", line_ref="L2"),
            _Trip("T3", line_ref="L1"),  # same line as T1: should not receive transfer from T1
        ],
        stop_times=[
            _StopTime("T1", "A", 1, arrival_time=0, departure_time=1),
            _StopTime("T1", "B", 2, arrival_time=600, departure_time=601),
            _StopTime("T2", "B", 1, arrival_time=700, departure_time=900),  # 5 min after arr at B
            _StopTime("T2", "C", 2, arrival_time=1200, departure_time=1201),
            _StopTime("T3", "B", 1, arrival_time=700, departure_time=900),  # same line as T1
            _StopTime("T3", "C", 2, arrival_time=1200, departure_time=1201),
        ],
    )
    g = build_jax_graph(scenario, config=_cfg(max_transfer_wait_min=10.0))

    arr_b_t1 = _event_node(g, stop_id="B", time_s=600, kind=NODE_KIND_EVENT_ARR, trip_index=0)
    dep_b_t2 = _event_node(g, stop_id="B", time_s=900, kind=NODE_KIND_EVENT_DEP, trip_index=1)
    dep_b_t3 = _event_node(g, stop_id="B", time_s=900, kind=NODE_KIND_EVENT_DEP, trip_index=2)

    transfer_edges = {
        (int(g.tail[i]), int(g.head[i]))
        for i in _link_indices_by_type(g, LINK_TYPE_TRANSFER).tolist()
    }

    assert (arr_b_t1, dep_b_t2) in transfer_edges
    assert (arr_b_t1, dep_b_t3) not in transfer_edges


def test_no_transfer_link_when_wait_exceeds_threshold():
    scenario = _scenario(
        stops=["A", "B", "C"],
        trips=[
            _Trip("T1", line_ref="L1"),
            _Trip("T2", line_ref="L2"),
        ],
        stop_times=[
            _StopTime("T1", "A", 1, arrival_time=0, departure_time=1),
            _StopTime("T1", "B", 2, arrival_time=600, departure_time=601),
            _StopTime("T2", "B", 1, arrival_time=700, departure_time=1800),
            _StopTime("T2", "C", 2, arrival_time=2100, departure_time=2101),
        ],
    )
    g = build_jax_graph(scenario, config=_cfg(max_transfer_wait_min=10.0))

    arr_b_t1 = _event_node(g, stop_id="B", time_s=600, kind=NODE_KIND_EVENT_ARR, trip_index=0)
    dep_b_t2 = _event_node(g, stop_id="B", time_s=1800, kind=NODE_KIND_EVENT_DEP, trip_index=1)

    transfer_edges = {
        (int(g.tail[i]), int(g.head[i]))
        for i in _link_indices_by_type(g, LINK_TYPE_TRANSFER).tolist()
    }

    assert (arr_b_t1, dep_b_t2) not in transfer_edges


# ---------------------------------------------------------------------------
# CSR and padded outgoing adjacency
# ---------------------------------------------------------------------------


def test_csr_outgoing_representation_is_consistent_with_tail_array():
    g = build_jax_graph(_scenario(), config=_cfg())

    out_start = _as_np(g.out_start).astype(int)
    out_links_csr = _as_np(g.out_links_csr).astype(int)
    tail = _as_np(g.tail).astype(int)

    assert out_start.shape == (g.num_nodes + 1,)
    assert out_start[0] == 0
    assert out_start[-1] == g.num_links
    assert np.all(out_start[1:] >= out_start[:-1])
    assert np.array_equal(np.sort(out_links_csr), np.arange(g.num_links))

    for node in range(g.num_nodes):
        sl = out_links_csr[out_start[node] : out_start[node + 1]]
        assert np.all(tail[sl] == node)


def test_padded_outgoing_representation_matches_csr():
    g = build_jax_graph(_scenario(), config=_cfg())

    out_start = _as_np(g.out_start).astype(int)
    out_links_csr = _as_np(g.out_links_csr).astype(int)
    out_links = _as_np(g.out_links).astype(int)
    out_mask = _as_np(g.out_mask).astype(bool)

    assert out_links.shape == out_mask.shape
    assert out_links.shape[0] == g.num_nodes

    for node in range(g.num_nodes):
        csr_links = out_links_csr[out_start[node] : out_start[node + 1]]
        padded_links = out_links[node]
        mask = out_mask[node]

        assert np.array_equal(padded_links[mask], csr_links)
        assert np.all(padded_links[~mask] == -1)


# ---------------------------------------------------------------------------
# Error handling and validation
# ---------------------------------------------------------------------------


def test_build_jax_graph_requires_timetable():
    scenario = _Scenario(
        stops=[_Stop("A")],
        timetable=None,
        time_bins=[_TimeBin(0, 600)],
    )

    with pytest.raises(ValueError, match="Scenario has no timetable"):
        build_jax_graph(scenario, config=_cfg())


def test_build_jax_graph_requires_time_bins():
    scenario = _scenario(time_bins=[])

    with pytest.raises(ValueError, match="no time bins"):
        build_jax_graph(scenario, config=_cfg())


def test_build_jax_graph_rejects_unknown_stop_in_stop_times():
    scenario = _scenario(
        stops=["A"],
        stop_times=[
            _StopTime("T1", "A", 1, arrival_time=0, departure_time=1),
            _StopTime("T1", "B", 2, arrival_time=300, departure_time=301),
        ],
    )

    with pytest.raises(ValueError, match="Unknown stop_id"):
        build_jax_graph(scenario, config=_cfg())


def test_build_jax_graph_rejects_unknown_trip_in_stop_times():
    scenario = _scenario(
        stops=["A", "B"],
        trips=[_Trip("T1", line_ref="L1")],
        stop_times=[
            _StopTime("UNKNOWN", "A", 1, arrival_time=0, departure_time=1),
            _StopTime("T1", "B", 2, arrival_time=300, departure_time=301),
        ],
    )

    with pytest.raises(ValueError, match="Unknown trip_id"):
        build_jax_graph(scenario, config=_cfg())


def test_build_jax_graph_rejects_non_increasing_ride_times():
    scenario = _scenario(
        stops=["A", "B"],
        stop_times=[
            _StopTime("T1", "A", 1, arrival_time=0, departure_time=400),
            _StopTime("T1", "B", 2, arrival_time=300, departure_time=301),
        ],
    )

    with pytest.raises(ValueError, match="Non-increasing ride times"):
        build_jax_graph(scenario, config=_cfg())


def test_equal_arrival_departure_is_regularized_and_warns():
    scenario = _scenario(
        stops=["A", "B"],
        stop_times=[
            _StopTime("T1", "A", 1, arrival_time=0, departure_time=0),
            _StopTime("T1", "B", 2, arrival_time=300, departure_time=300),
        ],
    )

    with pytest.warns(RuntimeWarning, match="arrival == departure"):
        g = build_jax_graph(scenario, config=_cfg(min_dwell_s=2))

    dep_a = _event_node(g, stop_id="A", time_s=2, kind=NODE_KIND_EVENT_DEP)
    arr_a = _event_node(g, stop_id="A", time_s=0, kind=NODE_KIND_EVENT_ARR)
    dwell_edges = {
        (int(g.tail[i]), int(g.head[i]))
        for i in _link_indices_by_type(g, LINK_TYPE_DWELL).tolist()
    }
    assert (arr_a, dep_a) in dwell_edges


def test_trip_line_ref_is_required_for_transfer_logic():
    bad_trip = SimpleNamespace(trip_id="T1", line_ref="")
    scenario = _scenario(trips=[bad_trip])

    with pytest.raises(ValueError, match="line_ref is empty"):
        build_jax_graph(scenario, config=_cfg())


def test_config_validation_is_called():
    # AssignmentConfig.validate should reject this before graph construction.
    with pytest.raises(ValueError, match="max_transfer_wait_min"):
        build_jax_graph(_scenario(), config=_cfg(max_transfer_wait_min=-1.0))


# ---------------------------------------------------------------------------
# Future-proofing tests for optional JaxGraph extensions
# ---------------------------------------------------------------------------


def test_graph_may_have_additional_optional_metadata_without_breaking_core_contract():
    g = build_jax_graph(_scenario(), config=_cfg())

    # These are the fields currently required by the assignment code. Future
    # versions may add optional fields such as topological levels. This test
    # deliberately checks the required contract rather than exact dataclass
    # field equality.
    required_fields = [
        "num_nodes",
        "num_links",
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
        "node_stop_id",
        "trip_id",
        "trip_line_ref",
    ]

    for field in required_fields:
        assert hasattr(g, field), field

    # If future layer metadata exists, it should be internally shape-compatible.
    if hasattr(g, "node_level") and getattr(g, "node_level") is not None:
        assert _as_np(g.node_level).shape == (g.num_nodes,)
    if hasattr(g, "level_nodes") and getattr(g, "level_nodes") is not None:
        assert _as_np(g.level_nodes).ndim >= 1
    if hasattr(g, "level_mask") and getattr(g, "level_mask") is not None:
        assert _as_np(g.level_mask).dtype == np.bool_
