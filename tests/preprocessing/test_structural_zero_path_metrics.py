from __future__ import annotations

from dataclasses import dataclass

import pytest

from public_transportation.preprocessing import (
    ODTimeKey,
    build_structural_zero_topology,
    compute_od_path_metrics,
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
    timetable: _Timetable


def _scenario_with_transfer() -> _Scenario:
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


def _scenario_with_two_direct_departures() -> _Scenario:
    return _Scenario(
        stops=[_Stop("A", "A"), _Stop("B", "B")],
        time_bins=[_TimeBin("t0", 0, 300)],
        timetable=_Timetable(
            trips=[_Trip("slow", "L1"), _Trip("fast", "L2")],
            stop_times=[
                _StopTime("slow", "A", 1, 59, 60),
                _StopTime("slow", "B", 2, 300, 301),
                _StopTime("fast", "A", 1, 119, 120),
                _StopTime("fast", "B", 2, 240, 241),
            ],
        ),
    )


def _metrics_by_key(scenario: _Scenario):
    topology = build_structural_zero_topology(
        scenario, StructuralZeroAssignmentConfig()
    )
    return {record.key: record.metrics for record in compute_od_path_metrics(topology)}


def test_transfer_path_metrics_and_destination_absorption() -> None:
    metrics = _metrics_by_key(_scenario_with_transfer())

    to_b = metrics[ODTimeKey("A", "B", "morning")]
    assert to_b.feasible
    assert to_b.minimum_transfers == 1
    assert to_b.minimum_initial_wait_minutes == pytest.approx(61.0 / 60.0)
    assert to_b.minimum_journey_time_minutes == pytest.approx(719.0 / 60.0)
    assert to_b.feasible_departure_count == 1
    assert to_b.earliest_arrival_seconds == 29_580

    # Arrival at X is absorbing when X is the requested destination. The path
    # must not continue to B and return a later arrival at X.
    to_x = metrics[ODTimeKey("A", "X", "morning")]
    assert to_x.feasible
    assert to_x.minimum_transfers == 0
    assert to_x.earliest_arrival_seconds == 29_160


def test_disconnected_direction_and_same_stop_are_unreachable_topologically() -> None:
    metrics = _metrics_by_key(_scenario_with_transfer())

    assert not metrics[ODTimeKey("B", "A", "morning")].feasible
    # The same-stop structural-zero rule is applied during classification; the
    # scheduled path engine does not invent a zero-length transit path.
    assert not metrics[ODTimeKey("A", "A", "morning")].feasible


def test_multiple_departures_are_counted_and_metric_minima_are_independent() -> None:
    metrics = _metrics_by_key(_scenario_with_two_direct_departures())[
        ODTimeKey("A", "B", "t0")
    ]

    assert metrics.feasible
    assert metrics.minimum_transfers == 0
    assert metrics.feasible_departure_count == 2
    assert metrics.minimum_initial_wait_minutes == 1.0  # slow service
    assert metrics.minimum_journey_time_minutes == 2.0  # fast service
    assert metrics.earliest_arrival_seconds == 240


def test_full_cartesian_product_is_sorted_and_unique() -> None:
    topology = build_structural_zero_topology(
        _scenario_with_transfer(), StructuralZeroAssignmentConfig()
    )
    records = compute_od_path_metrics(topology)
    keys = tuple(record.key for record in records)

    assert len(records) == 3 * 3 * 1
    assert keys == tuple(sorted(keys))
    assert len(keys) == len(set(keys))


def test_destination_profile_progress_is_complete_and_monotonic() -> None:
    topology = build_structural_zero_topology(
        _scenario_with_transfer(), StructuralZeroAssignmentConfig()
    )
    events = []

    compute_od_path_metrics(topology, progress=events.append)

    profile_events = [
        event for event in events if event.phase == "destination_profiles"
    ]
    assert [event.completed for event in profile_events] == [0, 1, 2, 3]
    assert all(event.total == 3 for event in profile_events)
    assert all(event.elapsed_seconds >= 0 for event in profile_events)
    materialization_events = [
        event for event in events if event.phase == "materialize_od_metrics"
    ]
    assert materialization_events[0].completed == 0
    assert materialization_events[-1].completed == materialization_events[-1].total
    assert materialization_events[-1].estimated_remaining_seconds == pytest.approx(0.0)


def test_destination_progress_callback_failure_is_observability_only() -> None:
    topology = build_structural_zero_topology(
        _scenario_with_transfer(), StructuralZeroAssignmentConfig()
    )

    def fail(event) -> None:
        if event.completed == 1:
            raise RuntimeError("stop requested")

    # A broken progress sink must not change the scientific calculation.
    records = compute_od_path_metrics(topology, progress=fail)
    assert len(records) == 9
