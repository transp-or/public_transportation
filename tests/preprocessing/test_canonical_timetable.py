from __future__ import annotations

from public_transportation.preprocessing.canonical_timetable import (
    build_canonical_timetable_index,
)

from public_transportation.domain import (
    Metadata,
    ODDemand,
    Scenario,
    Stop,
    StopTime,
    TimeBin,
    TimeOfDay,
    Timetable,
    Trip,
)
from public_transportation.domain.line import Line


def _scenario() -> Scenario:
    return Scenario(
        metadata=Metadata("canonical-index-test"),
        stops=[
            Stop("A", "A", 0.0, 0.0),
            Stop("B", "B", 0.0, 0.1),
            Stop("C", "C", 0.0, 0.2),
            Stop("X", "inactive", 0.0, 0.3),
        ],
        lines=[Line("L1")],
        time_bins=[TimeBin("morning", TimeOfDay(8 * 3600), TimeOfDay(9 * 3600))],
        demand=ODDemand([]),
        timetable=Timetable(
            trips=[Trip("T1", "L1")],
            stop_times=[
                StopTime("T1", "A", 1, TimeOfDay(8 * 3600), TimeOfDay(8 * 3600)),
                StopTime(
                    "T1",
                    "B",
                    2,
                    TimeOfDay(8 * 3600 + 600),
                    TimeOfDay(8 * 3600 + 600),
                ),
                StopTime(
                    "T1",
                    "C",
                    3,
                    TimeOfDay(8 * 3600 + 1200),
                    TimeOfDay(8 * 3600 + 1200),
                ),
            ],
        ),
    )


def test_canonical_index_is_immutable_and_deterministic() -> None:
    first = build_canonical_timetable_index(_scenario())
    second = build_canonical_timetable_index(_scenario())

    assert first.fingerprint == second.fingerprint
    assert first.stop_ids == ("A", "B", "C", "X")
    assert tuple(first.trip_sequences) == ("T1",)
    assert tuple(record.stop_id for record in first.trip_sequences["T1"]) == (
        "A",
        "B",
        "C",
    )
    assert first.departure_seconds_by_stop["A"] == (8 * 3600,)
    assert first.arrival_seconds_by_stop["C"] == (8 * 3600 + 1200,)
    assert first.route_patterns["L1"] == (("A", "B", "C"),)


def test_canonical_index_applies_dwell_policy_without_mutating_source() -> None:
    index = build_canonical_timetable_index(_scenario())
    raw = index.trip_sequences["T1"][0]
    regularized = index.regularized_stop_times(minimum_dwell_seconds=1)

    assert raw.arrival_s == raw.departure_s
    assert regularized[0].arrival_s == raw.arrival_s
    assert regularized[0].departure_s == raw.arrival_s + 1
    assert index.trip_sequences["T1"][0].departure_s == raw.departure_s
