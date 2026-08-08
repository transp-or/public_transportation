from __future__ import annotations

from dataclasses import replace

import numpy as np
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
    DEFAULT_SERVICE_ID,
    build_physical_stop_index,
    build_route_pattern_index,
    build_service_period_index,
    prepare_reduced_od_timetable,
)


def _scenario(*, reverse_input: bool = False) -> Scenario:
    stops = [
        Stop("A1", "A platform 1", 46.0000, 6.0000),
        Stop("A2", "A platform 2", 46.0002, 6.0002),
        Stop("B", "B", 46.0100, 6.0100),
        Stop("C", "C", 46.0200, 6.0200),
    ]
    trips = [
        Trip("T1", "L1", service_id="weekday", direction_id=0),
        Trip("T2", "L1", service_id="weekday", direction_id=0),
        Trip("T3", "L1", service_id="weekend", direction_id=1),
        Trip("T4", "L1", service_id="weekday", direction_id=0),
        Trip("T5", "L1", service_id=None, direction_id=0),
        Trip("T6", "L2", service_id="weekday", direction_id=0),
    ]
    specifications = {
        "T1": (8 * 3600, ("A1", "B", "C")),
        "T2": (25 * 3600, ("A2", "B", "C")),
        "T3": (9 * 3600, ("C", "B", "A1")),
        "T4": (10 * 3600, ("A2", "C")),
        "T5": (11 * 3600, ("A1", "B", "A2", "C")),
        "T6": (12 * 3600, ("A1", "B", "C")),
    }
    stop_times: list[StopTime] = []
    for trip_id, (start, stop_ids) in specifications.items():
        for offset, stop_id in enumerate(stop_ids):
            arrival = start + offset * 300
            stop_times.append(
                StopTime(
                    trip_id=trip_id,
                    stop_id=stop_id,
                    sequence=offset + 1,
                    arrival=arrival,
                    departure=arrival + 20,
                )
            )
    if reverse_input:
        stops.reverse()
        trips.reverse()
        stop_times.reverse()
    return Scenario(
        metadata=Metadata(title="Phase 2", created_at="2026-01-01T00:00:00"),
        stops=stops,
        lines=[Line("L2"), Line("L1")],
        time_bins=[
            TimeBin("T1", TimeOfDay(8 * 3600), TimeOfDay(26 * 3600))
        ],
        demand=ODDemand(records=[]),
        timetable=Timetable(trips=trips, stop_times=stop_times),
    )


def _mapping() -> dict[str, str]:
    return {"A1": "A", "A2": "A", "B": "B", "C": "C"}


def test_physical_stop_normalization_is_exact_and_deterministic() -> None:
    first = build_physical_stop_index(
        _scenario(),
        mapping=_mapping(),
        mapping_policy="authoritative",
    )
    reordered = build_physical_stop_index(
        _scenario(reverse_input=True),
        mapping=dict(reversed(tuple(_mapping().items()))),
        mapping_policy="authoritative",
    )

    assert first.fingerprint == reordered.fingerprint
    assert first.scenario_stop_ids == ("A1", "A2", "B", "C")
    assert tuple(place.physical_stop_id for place in first.places) == (
        "A",
        "B",
        "C",
    )
    assert first.places[0].member_stop_ids == ("A1", "A2")
    assert first.places[0].latitude == pytest.approx(46.0001)
    assert first.stop_to_physical_index.tolist() == [0, 0, 1, 2]
    assert not first.stop_to_physical_index.flags.writeable


def test_identity_mapping_and_mapping_validation() -> None:
    identity = build_physical_stop_index(_scenario())
    assert identity.mapping_policy == "identity"
    assert len(identity.places) == 4

    with pytest.raises(ValueError, match="cover exactly"):
        build_physical_stop_index(
            _scenario(),
            mapping={"A1": "A", "B": "B", "C": "C"},
            mapping_policy="authoritative",
        )
    with pytest.raises(ValueError, match="requires mapping_policy"):
        build_physical_stop_index(_scenario(), mapping=_mapping())


def test_patterns_preserve_direction_express_line_and_repeated_stops() -> None:
    scenario = _scenario()
    physical = build_physical_stop_index(
        scenario, mapping=_mapping(), mapping_policy="authoritative"
    )
    patterns = build_route_pattern_index(scenario, physical)

    assert len(patterns.patterns) == 5
    by_trips = {pattern.trip_ids: pattern for pattern in patterns.patterns}
    assert by_trips[("T1", "T2")].physical_stop_ids == ("A", "B", "C")
    assert by_trips[("T3",)].physical_stop_ids == ("C", "B", "A")
    assert by_trips[("T4",)].physical_stop_ids == ("A", "C")
    assert by_trips[("T5",)].physical_stop_ids == ("A", "B", "A", "C")
    assert by_trips[("T6",)].line_id == "L2"
    assert not patterns.trip_to_pattern_index.flags.writeable


def test_service_periods_use_declared_service_id_only() -> None:
    periods = build_service_period_index(_scenario())
    by_id = {
        period.service_period_id: period.trip_ids
        for period in periods.service_periods
    }

    assert by_id[DEFAULT_SERVICE_ID] == ("T5",)
    assert by_id["weekday"] == ("T1", "T2", "T4", "T6")
    assert by_id["weekend"] == ("T3",)
    assert not periods.trip_to_service_period_index.flags.writeable


def test_compact_timetable_index_maps_every_record_once() -> None:
    index = prepare_reduced_od_timetable(
        _scenario(),
        configuration_fingerprint="configuration",
        physical_stop_mapping=_mapping(),
        mapping_policy="authoritative",
    )

    assert index.trip_ids == ("T1", "T2", "T3", "T4", "T5", "T6")
    assert index.line_ids == ("L1", "L2")
    assert index.physical_stop_ids == ("A", "B", "C")
    assert index.array("trip_stop_time_indptr").tolist() == [
        0,
        3,
        6,
        9,
        11,
        15,
        18,
    ]
    assert index.array("stop_time_trip_index").size == 18
    assert index.array("arrival_seconds").max() == 25 * 3600 + 2 * 300
    assert all(not item.values.flags.writeable for item in index.arrays.arrays)
    assert index.retained_bytes == sum(
        item.values.nbytes for item in index.arrays.arrays
    )
    assert len(index.fingerprint) == 64


def test_index_is_independent_of_domain_input_order() -> None:
    kwargs = {
        "configuration_fingerprint": "configuration",
        "physical_stop_mapping": _mapping(),
        "mapping_policy": "authoritative",
    }
    first = prepare_reduced_od_timetable(_scenario(), **kwargs)
    reordered = prepare_reduced_od_timetable(
        _scenario(reverse_input=True), **kwargs
    )
    assert first.fingerprint == reordered.fingerprint
    for item in first.arrays.arrays:
        np.testing.assert_array_equal(
            item.values, reordered.array(item.name)
        )


def test_invalid_timetable_fails_before_array_publication() -> None:
    scenario = _scenario()
    assert scenario.timetable is not None
    bad = replace(
        scenario.timetable.stop_times[1],
        arrival=7 * 3600,
        departure=7 * 3600 + 1,
    )
    scenario.timetable.stop_times[1] = bad
    with pytest.raises(ValueError, match="nonmonotone"):
        prepare_reduced_od_timetable(
            scenario,
            configuration_fingerprint="configuration",
            physical_stop_mapping=_mapping(),
            mapping_policy="authoritative",
        )


def test_missing_trip_stop_times_are_rejected() -> None:
    scenario = _scenario()
    assert scenario.timetable is not None
    scenario.timetable.stop_times = [
        row for row in scenario.timetable.stop_times if row.trip_id != "T4"
    ]
    physical = build_physical_stop_index(
        scenario, mapping=_mapping(), mapping_policy="authoritative"
    )
    with pytest.raises(ValueError, match="missing_stop_times"):
        build_route_pattern_index(scenario, physical)
