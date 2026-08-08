from __future__ import annotations

import math

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
    JourneyChoicePolicy,
    JourneyEventKind,
    JourneyTimePeriod,
    RaptorQuery,
    build_journey_choices,
    prepare_reduced_od_timetable,
    run_raptor_query,
)


def _scenario(*, reverse_input: bool = False) -> Scenario:
    stops = [
        Stop(stop_id, stop_id, 46.0 + index * 0.001, 6.0)
        for index, stop_id in enumerate(("A", "B", "C", "D"))
    ]
    trips = [
        Trip("DIRECT", "LD", service_id="day", direction_id=0),
        Trip("FIRST", "L1", service_id="day", direction_id=0),
        Trip("SECOND", "L2", service_id="day", direction_id=0),
    ]
    schedules = {
        "DIRECT": (("A", 8 * 3600 + 5 * 60), ("D", 8 * 3600 + 30 * 60)),
        "FIRST": (("A", 8 * 3600), ("B", 8 * 3600 + 10 * 60)),
        "SECOND": (("C", 8 * 3600 + 15 * 60), ("D", 8 * 3600 + 25 * 60)),
    }
    stop_times = [
        StopTime(trip_id, stop_id, sequence, seconds, seconds)
        for trip_id, rows in schedules.items()
        for sequence, (stop_id, seconds) in enumerate(rows, start=1)
    ]
    if reverse_input:
        stops.reverse()
        trips.reverse()
        stop_times.reverse()
    return Scenario(
        metadata=Metadata(title="Choices", created_at="2026-01-01T00:00:00"),
        stops=stops,
        lines=[Line("L2"), Line("LD"), Line("L1")],
        time_bins=[TimeBin("day", TimeOfDay(0), TimeOfDay(30 * 3600))],
        demand=ODDemand(records=[]),
        timetable=Timetable(trips=trips, stop_times=stop_times),
    )


def _inputs(*, reverse_input: bool = False):
    timetable = prepare_reduced_od_timetable(
        _scenario(reverse_input=reverse_input),
        configuration_fingerprint="phase-5",
    )
    paths = (Footpath("B", "C", 120),)
    result = run_raptor_query(
        timetable,
        RaptorQuery("A", 8 * 3600, maximum_transfers=1),
        footpaths=paths,
    )
    return timetable, result


def _destination(result, destination: str, period: str = "unclassified"):
    return next(
        item
        for item in result.choice_sets
        if item.destination_physical_stop_id == destination
        and item.origin_time_period_id == period
    )


def test_complete_journey_events_conserve_boardings_and_alightings() -> None:
    timetable, raptor = _inputs()
    result = build_journey_choices(timetable, raptor)
    choice = _destination(result, "D")

    assert len(choice.alternatives) == 2
    for alternative in choice.alternatives:
        assert len(alternative.boarding_events) == len(alternative.alighting_events)
        assert len(alternative.boarding_events) == len(alternative.transit_legs)
        assert alternative.events[0].event_kind is JourneyEventKind.FIRST_BOARDING
        assert alternative.events[-1].event_kind is JourneyEventKind.FINAL_ALIGHTING
        assert alternative.travel_seconds == (
            alternative.wait_seconds
            + alternative.walk_seconds
            + alternative.in_vehicle_seconds
        )


def test_internal_platform_transfer_has_paired_events() -> None:
    timetable, raptor = _inputs()
    choice = _destination(build_journey_choices(timetable, raptor), "D")
    transfer = next(item for item in choice.alternatives if item.transfers == 1)

    assert tuple(event.event_kind for event in transfer.events) == (
        JourneyEventKind.FIRST_BOARDING,
        JourneyEventKind.TRANSFER_ALIGHTING,
        JourneyEventKind.TRANSFER_BOARDING,
        JourneyEventKind.FINAL_ALIGHTING,
    )
    assert transfer.events[1].physical_stop_id == "B"
    assert transfer.events[2].physical_stop_id == "C"
    assert transfer.events[2].seconds >= transfer.events[1].seconds


def test_route_alternatives_and_deterministic_pruning_are_reported() -> None:
    timetable, raptor = _inputs()
    policy = JourneyChoicePolicy(
        maximum_alternatives_per_cell=1,
        transfer_penalty_seconds=600.0,
    )
    result = build_journey_choices(timetable, raptor, policy=policy)
    choice = _destination(result, "D")

    assert len(choice.alternatives) == 1
    assert choice.alternatives[0].transfers == 0
    assert result.diagnostics.pruned_alternatives == 1
    assert result.diagnostics.maximum_candidates_in_cell == 2
    assert all(
        len(item.alternatives) <= policy.maximum_alternatives_per_cell
        for item in result.choice_sets
    )


def test_route_level_weights_produce_fixed_normalized_initial_shares() -> None:
    timetable, raptor = _inputs()
    pattern_by_trip = {
        trip_id: timetable.route_patterns.patterns[
            int(timetable.route_patterns.trip_to_pattern_index[index])
        ].pattern_id
        for index, trip_id in enumerate(timetable.trip_ids)
    }
    weights = {
        pattern_by_trip["DIRECT"]: 3.0,
        pattern_by_trip["FIRST"]: 1.0,
    }
    policy = JourneyChoicePolicy(transfer_penalty_seconds=300.0)
    choice = _destination(
        build_journey_choices(
            timetable,
            raptor,
            policy=policy,
            route_pattern_initial_weights=weights,
        ),
        "D",
    )
    shares = {
        alternative.transfers: share
        for alternative, share in zip(
            choice.alternatives, choice.initial_shares, strict=True
        )
    }
    assert shares[0] == pytest.approx(0.75)
    assert shares[1] == pytest.approx(0.25)
    assert math.isclose(sum(choice.initial_shares), 1.0)


def test_each_event_is_labeled_when_a_journey_crosses_periods() -> None:
    timetable, raptor = _inputs()
    periods = (
        JourneyTimePeriod("early", 0, 8 * 3600 + 12 * 60),
        JourneyTimePeriod("peak", 8 * 3600 + 12 * 60, 30 * 3600),
    )
    choice = _destination(
        build_journey_choices(timetable, raptor, time_periods=periods),
        "D",
        "early",
    )
    transfer = next(item for item in choice.alternatives if item.transfers == 1)
    assert tuple(event.time_period_id for event in transfer.events) == (
        "early",
        "early",
        "peak",
        "peak",
    )
    assert transfer.origin_time_period_id == "early"


def test_explicit_demand_period_jointly_groups_cross_period_boardings() -> None:
    timetable, raptor = _inputs()
    periods = (
        JourneyTimePeriod("t0", 0, 8 * 3600 + 3 * 60),
        JourneyTimePeriod("t1", 8 * 3600 + 3 * 60, 30 * 3600),
    )
    result = build_journey_choices(
        timetable,
        raptor,
        time_periods=periods,
        desired_departure_time_period_id="t0",
    )
    choice = _destination(result, "D", "t0")

    assert len(choice.alternatives) == 2
    assert choice.demand_time_period_id == "t0"
    assert {
        alternative.first_boarding_time_period_id
        for alternative in choice.alternatives
    } == {"t0", "t1"}
    assert all(
        alternative.desired_departure_time_period_id == "t0"
        for alternative in choice.alternatives
    )
    assert sum(choice.initial_shares) == pytest.approx(1.0)
    assert result.diagnostics.multi_first_boarding_period_choice_sets >= 1
    assert result.diagnostics.cross_period_alternatives >= 1


def test_legacy_construction_still_groups_by_first_boarding_period() -> None:
    timetable, raptor = _inputs()
    periods = (
        JourneyTimePeriod("t0", 0, 8 * 3600 + 3 * 60),
        JourneyTimePeriod("t1", 8 * 3600 + 3 * 60, 30 * 3600),
    )
    result = build_journey_choices(timetable, raptor, time_periods=periods)
    destination_sets = [
        choice
        for choice in result.choice_sets
        if choice.destination_physical_stop_id == "D"
    ]
    assert {choice.demand_time_period_id for choice in destination_sets} == {
        "t0",
        "t1",
    }
    assert result.diagnostics.legacy_period_semantics_choice_sets > 0


def test_output_is_deterministic_and_every_feasible_cell_is_populated() -> None:
    timetable, raptor = _inputs()
    first = build_journey_choices(timetable, raptor)
    reordered_timetable, reordered_raptor = _inputs(reverse_input=True)
    reordered = build_journey_choices(reordered_timetable, reordered_raptor)

    assert first.fingerprint == reordered.fingerprint
    assert first.choice_sets == reordered.choice_sets
    assert all(item.alternatives for item in first.choice_sets)
    assert first.diagnostics.choice_cells == len(first.choice_sets)
    assert first.diagnostics.estimated_payload_bytes > 0


def test_invalid_periods_weights_and_policy_fail_closed() -> None:
    timetable, raptor = _inputs()
    with pytest.raises(ValueError, match="sorted"):
        build_journey_choices(
            timetable,
            raptor,
            time_periods=(
                JourneyTimePeriod("late", 10, 20),
                JourneyTimePeriod("early", 0, 10),
            ),
        )
    with pytest.raises(ValueError, match="exactly one"):
        build_journey_choices(
            timetable,
            raptor,
            time_periods=(JourneyTimePeriod("too_short", 0, 100),),
        )
    with pytest.raises(ValueError, match="unknown keys"):
        build_journey_choices(
            timetable,
            raptor,
            route_pattern_initial_weights={"unknown": 1.0},
        )
    with pytest.raises(ValueError, match="must be positive"):
        JourneyChoicePolicy(maximum_alternatives_per_cell=0)
