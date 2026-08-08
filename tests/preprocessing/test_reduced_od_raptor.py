from __future__ import annotations

import pytest

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
from public_transportation.preprocessing.reduced_od import (
    Footpath,
    RaptorQuery,
    StructuralZeroReason,
    prepare_reduced_od_timetable,
    run_raptor_query,
    run_raptor_range_query,
)


def _scenario(*, reverse_input: bool = False) -> Scenario:
    stop_ids = ("A", "B", "C", "D", "E", "F", "G", "X")
    stops = [
        Stop(stop_id, stop_id, 46.0 + index / 100, 6.0)
        for index, stop_id in enumerate(stop_ids)
    ]
    trips = [
        Trip("ABCD", "DIRECT", service_id="weekday", direction_id=0),
        Trip("CD", "L2", service_id="weekday", direction_id=0),
        Trip("DE", "L3", service_id="weekday", direction_id=0),
        Trip("EG", "L4", service_id="weekday", direction_id=0),
        Trip("L1_EARLY", "L1", service_id="weekday", direction_id=0),
        Trip("L1_LATE", "L1", service_id="weekday", direction_id=0),
    ]
    schedules = {
        "ABCD": (("A", 8 * 3600 + 3 * 60), ("D", 8 * 3600 + 25 * 60)),
        "CD": (("C", 8 * 3600 + 12 * 60), ("D", 8 * 3600 + 20 * 60)),
        "DE": (("D", 8 * 3600 + 22 * 60), ("E", 8 * 3600 + 30 * 60)),
        "EG": (("E", 8 * 3600 + 32 * 60), ("G", 8 * 3600 + 40 * 60)),
        "L1_EARLY": (
            ("A", 8 * 3600),
            ("B", 8 * 3600 + 5 * 60),
            ("C", 8 * 3600 + 10 * 60),
        ),
        "L1_LATE": (
            ("A", 9 * 3600),
            ("B", 9 * 3600 + 5 * 60),
            ("C", 9 * 3600 + 10 * 60),
        ),
    }
    stop_times = [
        StopTime(
            trip_id=trip_id,
            stop_id=stop_id,
            sequence=sequence,
            arrival=seconds,
            departure=seconds,
        )
        for trip_id, rows in schedules.items()
        for sequence, (stop_id, seconds) in enumerate(rows, start=1)
    ]
    if reverse_input:
        stops.reverse()
        trips.reverse()
        stop_times.reverse()
    return Scenario(
        metadata=Metadata(title="RAPTOR", created_at="2026-01-01T00:00:00"),
        stops=stops,
        lines=[Line("L4"), Line("L3"), Line("DIRECT"), Line("L1"), Line("L2")],
        time_bins=[TimeBin("day", TimeOfDay(0), TimeOfDay(30 * 3600))],
        demand=ODDemand(records=[]),
        timetable=Timetable(trips=trips, stop_times=stop_times),
    )


def _index(*, reverse_input: bool = False):
    return prepare_reduced_od_timetable(
        _scenario(reverse_input=reverse_input),
        configuration_fingerprint="phase-4",
    )


def test_one_line_and_mandatory_transfer_features_match_enumeration() -> None:
    result = run_raptor_query(
        _index(), RaptorQuery("A", 8 * 3600, maximum_transfers=1)
    )

    c = result.destination("C").earliest
    assert (c.arrival_seconds, c.travel_seconds, c.wait_seconds) == (
        8 * 3600 + 10 * 60,
        10 * 60,
        0,
    )
    assert c.transfers == 0
    assert c.in_vehicle_seconds == 10 * 60
    assert tuple(leg.trip_id for leg in c.transit_legs) == ("L1_EARLY",)

    d = result.destination("D").earliest
    assert d.arrival_seconds == 8 * 3600 + 20 * 60
    assert d.wait_seconds == 2 * 60
    assert d.in_vehicle_seconds == 18 * 60
    assert d.transfers == 1
    assert tuple(leg.trip_id for leg in d.transit_legs) == ("L1_EARLY", "CD")


def test_raptor_progress_has_terminal_eta_for_every_long_loop() -> None:
    events: list[dict[str, object]] = []
    run_raptor_query(
        _index(),
        RaptorQuery("A", 8 * 3600, maximum_transfers=1),
        progress=events.append,
    )

    phases = {event["phase"] for event in events}
    assert phases == {
        "raptor_rounds",
        "raptor_destinations",
        "raptor_structural_zeros",
    }
    for phase in phases:
        terminal = [event for event in events if event["phase"] == phase][-1]
        assert terminal["status"] == "completed"
        assert terminal["estimated_remaining_seconds"] == pytest.approx(0.0)


def test_multiple_routes_retain_transfer_time_tradeoff() -> None:
    labels = run_raptor_query(
        _index(), RaptorQuery("A", 8 * 3600, maximum_transfers=1)
    ).destination("D").labels

    signatures = {
        tuple(leg.trip_id for leg in label.transit_legs): (
            label.arrival_seconds,
            label.transfers,
        )
        for label in labels
    }
    assert signatures[("L1_EARLY", "CD")] == (8 * 3600 + 20 * 60, 1)
    assert signatures[("ABCD",)] == (8 * 3600 + 25 * 60, 0)


def test_footpaths_are_additive_and_do_not_consume_a_transfer_round() -> None:
    result = run_raptor_query(
        _index(),
        RaptorQuery("A", 8 * 3600, maximum_transfers=0),
        footpaths=(Footpath("B", "F", 120),),
    )
    label = result.destination("F").earliest
    assert label.arrival_seconds == 8 * 3600 + 7 * 60
    assert label.walk_seconds == 120
    assert label.in_vehicle_seconds == 5 * 60
    assert label.transfers == 0


def test_structural_zero_reasons_separate_topology_transfer_and_schedule() -> None:
    limited = run_raptor_query(
        _index(), RaptorQuery("A", 8 * 3600, maximum_transfers=1)
    )
    assert limited.structural_zero_reason("A") is StructuralZeroReason.ORIGIN
    assert (
        limited.structural_zero_reason("G")
        is StructuralZeroReason.EXCEEDS_TRANSFER_LIMIT
    )
    assert (
        limited.structural_zero_reason("X")
        is StructuralZeroReason.NO_TOPOLOGICAL_PATH
    )
    assert (
        limited.structural_zero_reason("E")
        is StructuralZeroReason.NO_TIMETABLE_FEASIBLE_JOURNEY
    )

    late = run_raptor_query(
        _index(), RaptorQuery("A", 10 * 3600, maximum_transfers=2)
    )
    assert (
        late.structural_zero_reason("C")
        is StructuralZeroReason.NO_TIMETABLE_FEASIBLE_JOURNEY
    )
    assert (
        late.structural_zero_reason("E")
        is StructuralZeroReason.NO_TIMETABLE_FEASIBLE_JOURNEY
    )


def test_transfer_bound_is_exact() -> None:
    result = run_raptor_query(
        _index(), RaptorQuery("A", 8 * 3600, maximum_transfers=2)
    )
    e = result.destination("E").earliest
    assert e.arrival_seconds == 8 * 3600 + 30 * 60
    assert e.transfers == 2
    assert tuple(leg.trip_id for leg in e.transit_legs) == (
        "L1_EARLY",
        "CD",
        "DE",
    )
    assert result.diagnostics.rounds == 3


def test_departure_time_and_after_midnight_queries_are_supported() -> None:
    missed = run_raptor_query(
        _index(), RaptorQuery("A", 8 * 3600 + 1, maximum_transfers=0)
    )
    assert missed.destination("C").earliest.arrival_seconds == 9 * 3600 + 10 * 60
    after_service = run_raptor_query(
        _index(), RaptorQuery("A", 25 * 3600, maximum_transfers=0)
    )
    assert (
        after_service.structural_zero_reason("B")
        is StructuralZeroReason.NO_TIMETABLE_FEASIBLE_JOURNEY
    )


def test_range_queries_are_sorted_deduplicated_and_independent() -> None:
    result = run_raptor_range_query(
        _index(),
        origin_physical_stop_id="A",
        departure_seconds=(8 * 3600 + 1, 8 * 3600, 8 * 3600),
        maximum_transfers=0,
    )
    assert tuple(item.query.departure_seconds for item in result.results) == (
        8 * 3600,
        8 * 3600 + 1,
    )
    assert result.results[0].destination("C").earliest.arrival_seconds < (
        result.results[1].destination("C").earliest.arrival_seconds
    )


def test_results_are_deterministic_under_domain_and_footpath_order() -> None:
    paths = (Footpath("B", "F", 120), Footpath("C", "F", 60))
    query = RaptorQuery("A", 8 * 3600, maximum_transfers=2)
    first = run_raptor_query(_index(), query, footpaths=paths)
    reordered = run_raptor_query(
        _index(reverse_input=True), query, footpaths=reversed(paths)
    )
    assert first == reordered


@pytest.mark.parametrize(
    "footpath",
    [
        Footpath("A", "B", 1),
    ],
)
def test_invalid_query_inputs_fail_closed(footpath: Footpath) -> None:
    with pytest.raises(ValueError, match="unknown query origin"):
        run_raptor_query(_index(), RaptorQuery("UNKNOWN", 0, 0))
    with pytest.raises(ValueError, match="unknown stop"):
        run_raptor_query(
            _index(), RaptorQuery("A", 0, 0), footpaths=(Footpath("A", "Z", 1),)
        )
    with pytest.raises(ValueError, match="duplicates"):
        run_raptor_query(
            _index(), RaptorQuery("A", 0, 0), footpaths=(footpath, footpath)
        )


def test_empty_range_and_invalid_scalar_contracts_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        run_raptor_range_query(
            _index(),
            origin_physical_stop_id="A",
            departure_seconds=(),
            maximum_transfers=0,
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        RaptorQuery("A", -1, 0)
    with pytest.raises(ValueError, match="positive integer"):
        Footpath("A", "B", 0)
