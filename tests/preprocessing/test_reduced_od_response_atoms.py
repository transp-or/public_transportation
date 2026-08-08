from __future__ import annotations

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
from public_transportation.measurement import (
    MeasurementRecord,
    MeasurementTable,
    MeasurementType,
)
from public_transportation.preprocessing.reduced_od import (
    Footpath,
    JourneyTimePeriod,
    MeasurementResponseCacheError,
    RaptorQuery,
    ResponseCellKey,
    build_journey_choices,
    build_measurement_response,
    load_measurement_response_cache,
    prepare_reduced_od_timetable,
    run_raptor_query,
    save_measurement_response_cache,
)


def _scenario() -> Scenario:
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
    return Scenario(
        metadata=Metadata(title="Responses", created_at="2026-01-01T00:00:00"),
        stops=stops,
        lines=[Line("L1"), Line("L2"), Line("LD")],
        time_bins=[TimeBin("day", TimeOfDay(0), TimeOfDay(30 * 3600))],
        demand=ODDemand(records=[]),
        timetable=Timetable(trips=trips, stop_times=stop_times),
    )


def _inputs():
    timetable = prepare_reduced_od_timetable(
        _scenario(), configuration_fingerprint="phase-6"
    )
    raptor = run_raptor_query(
        timetable,
        RaptorQuery("A", 8 * 3600, 1),
        footpaths=(Footpath("B", "C", 120),),
    )
    return timetable, build_journey_choices(timetable, raptor)


def _record(
    measurement_type: MeasurementType,
    stop: str,
    seconds: int,
    trip: str,
    *,
    method: str = "apc",
    value: float = 1.0,
) -> MeasurementRecord:
    return MeasurementRecord(
        method_id=method,
        measurement_type=measurement_type,
        stop_id=stop,
        time=TimeOfDay(seconds),
        value=value,
        trip_id=trip,
    )


def _measurements() -> MeasurementTable:
    return MeasurementTable.from_records(
        (
            _record(MeasurementType.BOARDING, "A", 8 * 3600, "FIRST", value=10),
            _record(MeasurementType.ALIGHTING, "B", 8 * 3600 + 600, "FIRST", value=9),
            _record(MeasurementType.BOARDING, "C", 8 * 3600 + 900, "SECOND", value=8),
            _record(MeasurementType.ALIGHTING, "D", 8 * 3600 + 1500, "SECOND", value=7),
            _record(MeasurementType.BOARDING, "A", 8 * 3600 + 300, "DIRECT", value=6),
            _record(MeasurementType.ALIGHTING, "D", 8 * 3600 + 1800, "DIRECT", value=5),
        )
    )


def _dense_reference(artifact, choices) -> np.ndarray:
    by_key = {
        ResponseCellKey(
            item.origin_physical_stop_id,
            item.destination_physical_stop_id,
            item.origin_time_period_id,
        ): item
        for item in choices.choice_sets
    }
    result = np.zeros(
        (artifact.number_of_measurements, artifact.number_of_free_cells),
        dtype=float,
    )
    for column, key in enumerate(artifact.free_cell_keys):
        choice = by_key[key]
        for alternative, share in zip(
            choice.alternatives, choice.initial_shares, strict=True
        ):
            for event in alternative.events:
                leg = alternative.transit_legs[event.leg_index]
                kind = (
                    "boarding"
                    if "boarding" in event.event_kind.value
                    else "alighting"
                )
                event_key = (leg.trip_id, kind, event.physical_stop_id, event.seconds)
                for row, measurement in enumerate(artifact.resolved_measurements):
                    if measurement.event_key == event_key:
                        result[row, column] += share
    return result


def test_sparse_response_matches_explicit_event_enumeration() -> None:
    timetable, choices = _inputs()
    events: list[dict[str, object]] = []
    artifact = build_measurement_response(
        timetable=timetable,
        journey_choices=choices,
        measurements=_measurements(),
        configuration_fingerprint="configuration",
        progress=events.append,
    )
    dense = np.zeros(
        (artifact.number_of_measurements, artifact.number_of_free_cells)
    )
    np.add.at(
        dense,
        (artifact.measurement_index, artifact.free_cell_index),
        artifact.response_values,
    )
    np.testing.assert_allclose(dense, _dense_reference(artifact, choices))
    demand = np.arange(1, artifact.number_of_free_cells + 1, dtype=float)
    np.testing.assert_allclose(artifact.predict(demand), dense @ demand)
    np.testing.assert_array_equal(artifact.observed_values, [10, 9, 8, 7, 6, 5])
    assert artifact.nnz < dense.size
    assert all(
        not array.flags.writeable
        for array in (
            artifact.observed_values,
            artifact.measurement_index,
            artifact.free_cell_index,
            artifact.response_values,
            artifact.fixed_offset,
        )
    )
    completed_phases = {
        event["phase"]
        for event in events
        if event["status"] == "completed"
    }
    assert completed_phases == {
        "measurement_response_index",
        "measurement_response_coefficients",
        "measurement_response_sparse_rows",
        "measurement_response_fixed_offset",
    }


def test_demand_period_cell_maps_to_actual_later_event_measurements() -> None:
    timetable = prepare_reduced_od_timetable(
        _scenario(), configuration_fingerprint="cross-period-response"
    )
    raptor = run_raptor_query(
        timetable,
        RaptorQuery("A", 8 * 3600, 1),
        footpaths=(Footpath("B", "C", 120),),
    )
    periods = (
        JourneyTimePeriod("t0", 0, 8 * 3600 + 3 * 60),
        JourneyTimePeriod("t1", 8 * 3600 + 3 * 60, 30 * 3600),
    )
    choices = build_journey_choices(
        timetable,
        raptor,
        time_periods=periods,
        desired_departure_time_period_id="t0",
    )
    artifact = build_measurement_response(
        timetable=timetable,
        journey_choices=choices,
        measurements=_measurements(),
        configuration_fingerprint="cross-period-response",
    )
    key = ResponseCellKey("A", "D", "t0")
    column = artifact.free_cell_keys.index(key)
    rows = artifact.measurement_index[artifact.free_cell_index == column]

    assert any(
        artifact.resolved_measurements[int(row)].trip_id == "DIRECT"
        and artifact.resolved_measurements[int(row)].seconds >= periods[1].start_seconds
        for row in rows
    )
    choice = next(
        item
        for item in choices.choice_sets
        if item.demand_time_period_id == "t0"
        and item.destination_physical_stop_id == "D"
    )
    assert any(
        event.time_period_id == "t1"
        for alternative in choice.alternatives
        for event in alternative.events
    )


def test_fixed_positive_moves_to_offset_and_fixed_zero_disappears() -> None:
    timetable, choices = _inputs()
    key_d = ResponseCellKey("A", "D", "unclassified")
    key_b = ResponseCellKey("A", "B", "unclassified")
    baseline = build_measurement_response(
        timetable=timetable,
        journey_choices=choices,
        measurements=_measurements(),
        configuration_fingerprint="configuration",
    )
    dense = _dense_reference(baseline, choices)
    column_by_key = {key: index for index, key in enumerate(baseline.free_cell_keys)}
    structural_zero = ResponseCellKey("A", "UNREACHABLE", "unclassified")
    artifact = build_measurement_response(
        timetable=timetable,
        journey_choices=choices,
        measurements=_measurements(),
        configuration_fingerprint="configuration",
        fixed_demand={key_d: 10.0, key_b: 0.0, structural_zero: 0.0},
    )
    assert key_d not in artifact.free_cell_keys
    assert key_b not in artifact.free_cell_keys
    assert artifact.fixed_cell_keys == tuple(sorted((key_b, key_d, structural_zero)))
    np.testing.assert_allclose(
        artifact.fixed_offset, 10.0 * dense[:, column_by_key[key_d]]
    )
    with pytest.raises(ValueError, match="positive fixed demand"):
        build_measurement_response(
            timetable=timetable,
            journey_choices=choices,
            measurements=_measurements(),
            configuration_fingerprint="configuration",
            fixed_demand={structural_zero: 1.0},
        )


def test_missing_sensor_is_absent_and_explicit_zero_is_retained() -> None:
    timetable, choices = _inputs()
    table = MeasurementTable.from_records(
        (
            _record(
                MeasurementType.BOARDING,
                "A",
                8 * 3600,
                "FIRST",
                value=0.0,
            ),
        )
    )
    artifact = build_measurement_response(
        timetable=timetable,
        journey_choices=choices,
        measurements=table,
        configuration_fingerprint="configuration",
    )
    assert artifact.number_of_measurements == 1
    assert artifact.observed_values.tolist() == [0.0]
    assert all(item.seconds == 8 * 3600 for item in artifact.resolved_measurements)


def test_distinct_methods_may_map_to_same_event_without_silent_grouping() -> None:
    timetable, choices = _inputs()
    table = MeasurementTable.from_records(
        (
            _record(MeasurementType.BOARDING, "A", 8 * 3600, "FIRST", method="a"),
            _record(MeasurementType.BOARDING, "A", 8 * 3600, "FIRST", method="b"),
        )
    )
    artifact = build_measurement_response(
        timetable=timetable,
        journey_choices=choices,
        measurements=table,
        configuration_fingerprint="configuration",
    )
    prediction = artifact.predict(np.ones(artifact.number_of_free_cells))
    assert artifact.number_of_measurements == 2
    assert prediction[0] == pytest.approx(prediction[1])
    assert artifact.resolved_measurements[0].method_id == "a"
    assert artifact.resolved_measurements[1].method_id == "b"


def test_exact_response_equivalence_compresses_identical_columns() -> None:
    timetable, choices = _inputs()
    table = MeasurementTable.from_records(
        (_record(MeasurementType.BOARDING, "A", 8 * 3600, "FIRST"),)
    )
    artifact = build_measurement_response(
        timetable=timetable,
        journey_choices=choices,
        measurements=table,
        configuration_fingerprint="configuration",
    )
    key_b = artifact.free_cell_keys.index(ResponseCellKey("A", "B", "unclassified"))
    key_c = artifact.free_cell_keys.index(ResponseCellKey("A", "C", "unclassified"))
    assert artifact.equivalence.class_by_cell[key_b] == (
        artifact.equivalence.class_by_cell[key_c]
    )
    assert artifact.equivalence.number_of_classes < artifact.number_of_free_cells


def test_cache_round_trip_and_expected_identities(tmp_path) -> None:
    timetable, choices = _inputs()
    artifact = build_measurement_response(
        timetable=timetable,
        journey_choices=choices,
        measurements=_measurements(),
        configuration_fingerprint="configuration",
    )
    path = tmp_path / "response.npz"
    save_measurement_response_cache(path, artifact)
    loaded = load_measurement_response_cache(
        path,
        expected_configuration_fingerprint=artifact.configuration_fingerprint,
        expected_timetable_fingerprint=artifact.timetable_fingerprint,
        expected_journey_choice_fingerprint=artifact.journey_choice_fingerprint,
        expected_measurement_fingerprint=artifact.measurement_fingerprint,
    )
    assert loaded.fingerprint == artifact.fingerprint
    assert loaded.free_cell_keys == artifact.free_cell_keys
    np.testing.assert_array_equal(loaded.response_values, artifact.response_values)
    with pytest.raises(MeasurementResponseCacheError, match="configuration"):
        load_measurement_response_cache(
            path, expected_configuration_fingerprint="wrong"
        )


def test_corrupt_cache_and_invalid_measurements_fail_closed(tmp_path) -> None:
    path = tmp_path / "corrupt.npz"
    path.write_bytes(b"not an npz cache")
    with pytest.raises(MeasurementResponseCacheError, match="cannot be decoded"):
        load_measurement_response_cache(path)

    timetable, choices = _inputs()
    load_table = MeasurementTable.from_records(
        (_record(MeasurementType.LOAD, "A", 8 * 3600, "FIRST"),)
    )
    with pytest.raises(ValueError, match="unsupported type"):
        build_measurement_response(
            timetable=timetable,
            journey_choices=choices,
            measurements=load_table,
            configuration_fingerprint="configuration",
        )
    unmatched = MeasurementTable.from_records(
        (_record(MeasurementType.BOARDING, "A", 7 * 3600, "FIRST"),)
    )
    with pytest.raises(ValueError, match="does not match"):
        build_measurement_response(
            timetable=timetable,
            journey_choices=choices,
            measurements=unmatched,
            configuration_fingerprint="configuration",
        )
