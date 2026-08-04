"""Behavioral contract for timetable-based dynamic and fixed routing."""

from __future__ import annotations

import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.assignment.graph_sentinels import (
    LINK_TYPE_ACCESS,
    LINK_TYPE_RIDE,
)
from public_transportation.domain.demand import ODDemand, ODRecord
from public_transportation.domain.line import Line
from public_transportation.domain.metadata import Metadata
from public_transportation.domain.scenario import Scenario
from public_transportation.domain.stop import Stop
from public_transportation.domain.stop_time import StopTime
from public_transportation.domain.time_bin import TimeBin
from public_transportation.domain.time_of_day import TimeOfDay
from public_transportation.domain.timetable import Timetable
from public_transportation.domain.trip import Trip
from public_transportation.inference.assignment_adapter import (
    assign_link_flow,
    assign_link_flow_fixed_routing,
    build_assignment_inputs,
    prepare_fixed_routing,
)


def _scenario(
    *,
    trips: list[Trip],
    stop_times: list[StopTime],
    bins: list[tuple[str, str, str]],
    demand: list[float] | None = None,
) -> Scenario:
    values = demand or [10.0] * len(bins)
    return Scenario(
        metadata=Metadata(title="fixed-routing semantics"),
        stops=[
            Stop("A", "Origin", 46.20, 6.10),
            Stop("B", "Destination", 46.21, 6.11),
        ],
        lines=[Line(line_id=line) for line in sorted({trip.line_ref for trip in trips})],
        time_bins=[TimeBin(bin_id, start, end) for bin_id, start, end in bins],
        demand=ODDemand(
            [
                ODRecord("A", "B", bin_id, float(value))
                for (bin_id, _, _), value in zip(bins, values, strict=True)
            ]
        ),
        timetable=Timetable(trips=trips, stop_times=stop_times),
    )


def _direct_trip(trip_id: str, line: str, departure: str, arrival: str, *, capacity=50.0):
    departure_s = TimeOfDay.parse(departure).seconds_from_midnight
    arrival_s = TimeOfDay.parse(arrival).seconds_from_midnight
    return (
        Trip(trip_id=trip_id, line_ref=line, capacity=capacity),
        [
            StopTime(trip_id, "A", 1, departure_s - 1, departure_s),
            StopTime(trip_id, "B", 2, arrival_s, arrival_s + 1),
        ],
    )


def _prepare(scenario: Scenario):
    artifacts = prepare_assignment(
        scenario,
        AssignmentConfig(max_access_deviation_min=2.0),
    )
    inputs = build_assignment_inputs(artifacts=artifacts)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    return artifacts, inputs, routing


def _accessible_trip_ids(inputs, od_index: int) -> set[str]:
    graph = inputs.graph
    origin = int(np.asarray(inputs.od_origin_node)[od_index])
    tail = np.asarray(graph.tail)
    link_type = np.asarray(graph.link_type)
    trip_index = np.asarray(graph.link_trip_index)
    selected = (tail == origin) & (link_type == LINK_TYPE_ACCESS)
    return {graph.trip_id[index] for index in trip_index[selected] if index >= 0}


def _access_probability_by_trip(inputs, routing, od_index: int) -> dict[str, float]:
    graph = inputs.graph
    origin = int(np.asarray(inputs.od_origin_node)[od_index])
    tail = np.asarray(graph.tail)
    link_type = np.asarray(graph.link_type)
    trip_index = np.asarray(graph.link_trip_index)
    probabilities = np.asarray(routing.group_link_probability)[0]
    selected = np.flatnonzero((tail == origin) & (link_type == LINK_TYPE_ACCESS))
    return {
        graph.trip_id[int(trip_index[link])]: float(probabilities[link])
        for link in selected
    }


def _assert_dynamic_and_fixed_agree(inputs, routing, demand):
    dynamic = np.asarray(assign_link_flow(inputs=inputs, f=demand, theta=1.0))
    fixed = np.asarray(
        assign_link_flow_fixed_routing(inputs=inputs, routing=routing, f=demand)
    )
    np.testing.assert_allclose(fixed, dynamic, rtol=2.0e-6, atol=2.0e-6)
    return dynamic


def test_time_bins_can_expose_different_lines_and_cached_probabilities():
    morning, morning_times = _direct_trip("M", "morning-line", "08:05", "08:15")
    afternoon, afternoon_times = _direct_trip("A", "afternoon-line", "17:05", "17:15")
    scenario = _scenario(
        trips=[morning, afternoon],
        stop_times=morning_times + afternoon_times,
        bins=[("morning", "08:00", "08:10"), ("afternoon", "17:00", "17:10")],
    )
    _, inputs, routing = _prepare(scenario)

    assert _accessible_trip_ids(inputs, 0) == {"M"}
    assert _accessible_trip_ids(inputs, 1) == {"A"}
    assert _access_probability_by_trip(inputs, routing, 0) == {"M": 1.0}
    assert _access_probability_by_trip(inputs, routing, 1) == {"A": 1.0}
    _assert_dynamic_and_fixed_agree(inputs, routing, np.asarray([10.0, 12.0]))


def test_same_line_has_different_scheduled_trips_in_different_bins():
    morning, morning_times = _direct_trip("L1-0805", "L1", "08:05", "08:15")
    afternoon, afternoon_times = _direct_trip("L1-1705", "L1", "17:05", "17:15")
    scenario = _scenario(
        trips=[morning, afternoon],
        stop_times=morning_times + afternoon_times,
        bins=[("morning", "08:00", "08:10"), ("afternoon", "17:00", "17:10")],
    )
    _, inputs, routing = _prepare(scenario)

    assert _accessible_trip_ids(inputs, 0) == {"L1-0805"}
    assert _accessible_trip_ids(inputs, 1) == {"L1-1705"}
    _assert_dynamic_and_fixed_agree(inputs, routing, np.asarray([7.0, 9.0]))


def test_simultaneous_route_alternatives_split_flow_and_ignore_demand_level():
    first, first_times = _direct_trip("T1", "L1", "08:05", "08:15")
    second, second_times = _direct_trip("T2", "L2", "08:05", "08:15")
    scenario = _scenario(
        trips=[first, second],
        stop_times=first_times + second_times,
        bins=[("morning", "08:00", "08:10")],
    )
    _, inputs, routing = _prepare(scenario)

    probability = _access_probability_by_trip(inputs, routing, 0)
    assert probability.keys() == {"T1", "T2"}
    np.testing.assert_allclose(list(probability.values()), [0.5, 0.5], atol=1.0e-6)

    low = _assert_dynamic_and_fixed_agree(inputs, routing, np.asarray([10.0]))
    high = _assert_dynamic_and_fixed_agree(inputs, routing, np.asarray([30.0]))
    np.testing.assert_allclose(high, 3.0 * low, rtol=2.0e-6, atol=2.0e-6)
    unchanged = prepare_fixed_routing(inputs=inputs, theta=1.0)
    np.testing.assert_array_equal(
        unchanged.group_link_probability,
        routing.group_link_probability,
    )


def test_capacity_is_metadata_and_does_not_limit_boarding_or_change_routing():
    trip, stop_times = _direct_trip(
        "small-bus", "L1", "08:05", "08:15", capacity=5.0
    )
    scenario = _scenario(
        trips=[trip],
        stop_times=stop_times,
        bins=[("morning", "08:00", "08:10")],
        demand=[20.0],
    )
    artifacts, inputs, routing = _prepare(scenario)
    dynamic = _assert_dynamic_and_fixed_agree(inputs, routing, np.asarray([20.0]))

    graph = artifacts.graph
    ride = np.asarray(graph.link_type) == LINK_TYPE_RIDE
    assert np.asarray(graph.capacity)[ride].item() == 5.0
    assert dynamic[ride].item() == 20.0
    assert _access_probability_by_trip(inputs, routing, 0) == {"small-bus": 1.0}
