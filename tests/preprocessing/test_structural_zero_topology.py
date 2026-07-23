from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from public_transportation.assignment.build_time_expanded import build_jax_graph
from public_transportation.assignment.config import AssignmentConfig
from public_transportation.assignment.graph_sentinels import LINK_TYPE_TRANSFER
from public_transportation.preprocessing import (
    build_structural_zero_topology,
)
from public_transportation.preprocessing.structural_zeros.config import (
    StructuralZeroAssignmentConfig,
)


@dataclass(frozen=True, slots=True)
class _Stop:
    stop_id: str
    name: str


@dataclass(frozen=True, slots=True)
class _Trip:
    trip_id: str
    line_ref: str
    capacity: float = 40.0


@dataclass(frozen=True, slots=True)
class _StopTime:
    trip_id: str
    stop_id: str
    stop_sequence: int
    arrival_time: int
    departure_time: int


@dataclass(frozen=True, slots=True)
class _TimeBin:
    bin_id: str
    start_s: int
    end_s: int


@dataclass(frozen=True, slots=True)
class _Timetable:
    trips: list[_Trip]
    stop_times: list[_StopTime]


@dataclass(frozen=True, slots=True)
class _Scenario:
    stops: list[_Stop]
    time_bins: list[_TimeBin]
    timetable: _Timetable | None


def _transfer_scenario() -> _Scenario:
    return _Scenario(
        stops=[_Stop("A", "A"), _Stop("X", "X"), _Stop("B", "B")],
        time_bins=[_TimeBin("morning", 28_800, 29_700)],
        timetable=_Timetable(
            trips=[_Trip("T1", "L1"), _Trip("T2", "L2")],
            stop_times=[
                _StopTime("T1", "A", 1, 28_860, 28_861),
                _StopTime("T1", "X", 2, 29_160, 29_161),
                _StopTime("T2", "X", 1, 29_280, 29_281),
                _StopTime("T2", "B", 2, 29_580, 29_581),
            ],
        ),
    )


def test_topology_is_exact_adapter_of_assignment_graph() -> None:
    scenario = _transfer_scenario()
    config = StructuralZeroAssignmentConfig(
        max_access_deviation_minutes=12.0,
        max_transfer_wait_minutes=7.0,
        minimum_dwell_seconds=2,
    )
    topology = build_structural_zero_topology(scenario, config)
    assignment_graph = build_jax_graph(
        scenario,
        config=AssignmentConfig(
            max_access_deviation_min=12.0,
            max_transfer_wait_min=7.0,
            min_dwell_s=2,
        ),
    )

    for name in (
        "tail",
        "head",
        "node_kind",
        "node_time_s",
        "node_time_bin_index",
        "link_type",
        "travel_time",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(topology.graph, name)),
            np.asarray(getattr(assignment_graph, name)),
        )


def test_topology_indexes_centroids_adjacency_and_link_types() -> None:
    topology = build_structural_zero_topology(
        _transfer_scenario(), StructuralZeroAssignmentConfig()
    )

    assert topology.stop_ids == ("A", "B", "X")
    assert topology.time_bin_ids == ("morning",)
    origin = topology.origin_node("A", "morning")
    destination = topology.destination_node("B")
    assert origin != destination
    assert topology.outgoing_links[origin]
    assert topology.incoming_links[destination]
    assert len(topology.transfer_links) == 1
    assert all(
        int(topology.graph.link_type[link]) == LINK_TYPE_TRANSFER
        for link in topology.transfer_links
    )

    for link, (tail, head) in enumerate(
        zip(topology.graph.tail, topology.graph.head, strict=True)
    ):
        assert link in topology.outgoing_links[int(tail)]
        assert link in topology.incoming_links[int(head)]


def test_topology_fingerprint_is_deterministic_and_configuration_sensitive() -> None:
    scenario = _transfer_scenario()
    first = build_structural_zero_topology(
        scenario, StructuralZeroAssignmentConfig(max_transfer_wait_minutes=3.0)
    )
    again = build_structural_zero_topology(
        scenario, StructuralZeroAssignmentConfig(max_transfer_wait_minutes=3.0)
    )
    tighter = build_structural_zero_topology(
        scenario, StructuralZeroAssignmentConfig(max_transfer_wait_minutes=1.0)
    )

    assert first.fingerprint == again.fingerprint
    assert first.fingerprint != tighter.fingerprint
    assert first.transfer_links
    assert not tighter.transfer_links


def test_topology_lookup_rejects_unknown_identifiers() -> None:
    topology = build_structural_zero_topology(
        _transfer_scenario(), StructuralZeroAssignmentConfig()
    )

    with pytest.raises(KeyError, match="Unknown stop_id"):
        topology.destination_node("missing")
    with pytest.raises(KeyError, match="Unknown time_bin_id"):
        topology.origin_node("A", "missing")


def test_topology_requires_timetable_and_named_time_bins() -> None:
    scenario = _transfer_scenario()
    without_timetable = _Scenario(
        stops=scenario.stops, time_bins=scenario.time_bins, timetable=None
    )
    with pytest.raises(ValueError, match="timetable is required"):
        build_structural_zero_topology(
            without_timetable, StructuralZeroAssignmentConfig()
        )

    unnamed = _Scenario(
        stops=scenario.stops,
        time_bins=[_TimeBin("", 28_800, 29_700)],
        timetable=scenario.timetable,
    )
    with pytest.raises(ValueError, match="no non-empty bin_id"):
        build_structural_zero_topology(unnamed, StructuralZeroAssignmentConfig())
