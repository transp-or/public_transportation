"""Compact immutable array index for route-based timetable preprocessing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from public_transportation.domain import Scenario, StopTime, TimeOfDay, Trip

from .artifacts import (
    NamedImmutableArray,
    ReducedODArrayArtifact,
    ReducedODArtifactKind,
    ReducedODArtifactManifest,
    canonical_json,
)
from .physical_stops import (
    PhysicalStopIndex,
    PhysicalStopMappingPolicy,
    build_physical_stop_index,
)
from .progress import ReducedODProgress, ReducedODProgressEmitter
from .route_patterns import RoutePatternIndex, build_route_pattern_index
from .service_periods import ServicePeriodIndex, build_service_period_index


def _seconds(value: TimeOfDay | int | str, name: str) -> int:
    if not isinstance(value, TimeOfDay):
        raise TypeError(f"normalized {name} must be a TimeOfDay instance.")
    return int(value.seconds_from_midnight)


def _source_payload(scenario: Scenario) -> dict[str, Any]:
    if scenario.timetable is None:
        raise ValueError("scenario.timetable is required.")
    return {
        "lines": sorted(str(line.line_id) for line in scenario.lines),
        "stop_times": sorted(
            (
                str(stop_time.trip_id),
                int(stop_time.sequence),
                str(stop_time.stop_id),
                _seconds(stop_time.arrival, "arrival"),
                _seconds(stop_time.departure, "departure"),
            )
            for stop_time in scenario.timetable.stop_times
        ),
        "stops": sorted(
            (
                str(stop.stop_id),
                str(stop.name),
                float(stop.lat),
                float(stop.lon),
            )
            for stop in scenario.stops
        ),
        "trips": sorted(
            (
                str(trip.trip_id),
                str(trip.line_ref),
                trip.service_id,
                trip.direction_id,
            )
            for trip in scenario.timetable.trips
        ),
    }


def timetable_source_fingerprint(scenario: Scenario) -> str:
    """Fingerprint timetable-relevant domain content, excluding OD demand."""
    return hashlib.sha256(
        canonical_json(_source_payload(scenario)).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class TimetableIndex:
    """String dictionaries and immutable arrays used by later RAPTOR phases."""

    stop_ids: tuple[str, ...]
    physical_stop_ids: tuple[str, ...]
    trip_ids: tuple[str, ...]
    line_ids: tuple[str, ...]
    physical_stops: PhysicalStopIndex
    route_patterns: RoutePatternIndex
    service_periods: ServicePeriodIndex
    arrays: ReducedODArrayArtifact

    def __post_init__(self) -> None:
        for name, values in (
            ("stop_ids", self.stop_ids),
            ("physical_stop_ids", self.physical_stop_ids),
            ("trip_ids", self.trip_ids),
            ("line_ids", self.line_ids),
        ):
            if values != tuple(sorted(values)) or len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique and sorted.")
        if self.trip_ids != self.route_patterns.trip_ids:
            raise ValueError("route-pattern trip order does not match timetable.")
        if self.trip_ids != self.service_periods.trip_ids:
            raise ValueError("service-period trip order does not match timetable.")
        expected_names = {
            "arrival_seconds",
            "departure_seconds",
            "stop_sequence",
            "stop_time_physical_stop_index",
            "stop_time_stop_index",
            "stop_time_trip_index",
            "trip_direction_id",
            "trip_line_index",
            "trip_route_pattern_index",
            "trip_service_period_index",
            "trip_stop_time_indptr",
        }
        actual_names = {item.name for item in self.arrays.arrays}
        if actual_names != expected_names:
            raise ValueError(
                f"timetable arrays mismatch; missing={sorted(expected_names - actual_names)}, "
                f"extra={sorted(actual_names - expected_names)}."
            )

    def array(self, name: str) -> np.ndarray:
        """Return one named read-only array."""
        for item in self.arrays.arrays:
            if item.name == name:
                return item.values
        raise KeyError(name)

    @property
    def fingerprint(self) -> str:
        return self.arrays.fingerprint

    @property
    def retained_bytes(self) -> int:
        return self.arrays.retained_bytes


def build_timetable_index(
    scenario: Scenario,
    *,
    configuration_fingerprint: str,
    physical_stops: PhysicalStopIndex,
    route_patterns: RoutePatternIndex,
    service_periods: ServicePeriodIndex,
    progress: ReducedODProgress | None = None,
) -> TimetableIndex:
    """Compile domain timetable records into deterministic host arrays."""
    if scenario.timetable is None:
        raise ValueError("scenario.timetable is required.")
    if not configuration_fingerprint:
        raise ValueError("configuration_fingerprint must be non-empty.")

    stop_ids = tuple(sorted(str(stop.stop_id) for stop in scenario.stops))
    if len(set(stop_ids)) != len(stop_ids):
        raise ValueError("scenario stop identifiers must be unique.")
    if stop_ids != physical_stops.scenario_stop_ids:
        raise ValueError("physical-stop index does not match scenario stops.")
    stop_index = {stop_id: index for index, stop_id in enumerate(stop_ids)}

    physical_stop_ids = tuple(
        sorted(place.physical_stop_id for place in physical_stops.places)
    )
    physical_index = {
        physical_id: index for index, physical_id in enumerate(physical_stop_ids)
    }
    scenario_to_physical = dict(
        zip(
            physical_stops.scenario_stop_ids,
            physical_stops.physical_stop_ids_by_scenario_stop,
            strict=True,
        )
    )

    line_ids = tuple(sorted(str(line.line_id) for line in scenario.lines))
    if len(set(line_ids)) != len(line_ids):
        raise ValueError("scenario line identifiers must be unique.")
    line_index = {line_id: index for index, line_id in enumerate(line_ids)}

    trips_by_id: dict[str, Trip] = {}
    trip_progress = ReducedODProgressEmitter(
        progress,
        phase="timetable_trip_validation",
        total=len(scenario.timetable.trips),
    )
    trip_progress.start()
    for trip_position, trip in enumerate(scenario.timetable.trips, start=1):
        trip_id = str(trip.trip_id)
        if trip_id in trips_by_id:
            raise ValueError(f"duplicate trip identifier {trip_id!r}.")
        if str(trip.line_ref) not in line_index:
            raise ValueError(
                f"trip {trip_id!r} references unknown line {trip.line_ref!r}."
            )
        trips_by_id[trip_id] = trip
        trip_progress.update(trip_position, current_unit=trip_id)
    trip_ids = tuple(sorted(trips_by_id))
    if trip_ids != route_patterns.trip_ids or trip_ids != service_periods.trip_ids:
        raise ValueError("trip classifications do not match scenario trips.")
    trip_index = {trip_id: index for index, trip_id in enumerate(trip_ids)}

    stop_times_by_trip: dict[str, list[StopTime]] = {
        trip_id: [] for trip_id in trip_ids
    }
    stop_time_progress = ReducedODProgressEmitter(
        progress,
        phase="timetable_stop_time_index",
        total=len(scenario.timetable.stop_times),
    )
    stop_time_progress.start()
    for stop_time_position, stop_time in enumerate(
        scenario.timetable.stop_times, start=1
    ):
        trip_id = str(stop_time.trip_id)
        if trip_id not in trip_index:
            raise ValueError(
                f"stop time references unknown trip {trip_id!r}."
            )
        stop_times_by_trip[trip_id].append(stop_time)
        stop_time_progress.update(stop_time_position, current_unit=trip_id)

    trip_stop_time_indptr = [0]
    stop_time_trip_index: list[int] = []
    stop_time_stop_index: list[int] = []
    stop_time_physical_stop_index: list[int] = []
    stop_sequence: list[int] = []
    arrival_seconds: list[int] = []
    departure_seconds: list[int] = []
    array_progress = ReducedODProgressEmitter(
        progress, phase="timetable_array_materialization", total=len(trip_ids)
    )
    array_progress.start()
    for trip_position, trip_id in enumerate(trip_ids, start=1):
        rows = stop_times_by_trip[trip_id]
        if len(rows) < 2:
            raise ValueError(f"trip {trip_id!r} must contain at least two stop times.")
        if any(
            not isinstance(stop_time.sequence, int) or stop_time.sequence <= 0
            for stop_time in rows
        ):
            raise ValueError(f"trip {trip_id!r} contains an invalid stop sequence.")
        ordered = sorted(rows, key=lambda stop_time: int(stop_time.sequence))
        sequences = [int(stop_time.sequence) for stop_time in ordered]
        if len(set(sequences)) != len(sequences):
            raise ValueError(f"trip {trip_id!r} has duplicate stop sequences.")
        previous_departure: int | None = None
        for stop_time in ordered:
            stop_id = str(stop_time.stop_id)
            if stop_id not in stop_index:
                raise ValueError(
                    f"trip {trip_id!r} references unknown stop {stop_id!r}."
                )
            arrival = _seconds(stop_time.arrival, "arrival")
            departure = _seconds(stop_time.departure, "departure")
            if arrival < 0 or departure < arrival:
                raise ValueError(
                    f"trip {trip_id!r} has invalid arrival/departure times."
                )
            if previous_departure is not None and arrival < previous_departure:
                raise ValueError(f"trip {trip_id!r} has nonmonotone times.")
            previous_departure = departure
            stop_time_trip_index.append(trip_index[trip_id])
            stop_time_stop_index.append(stop_index[stop_id])
            stop_time_physical_stop_index.append(
                physical_index[scenario_to_physical[stop_id]]
            )
            stop_sequence.append(int(stop_time.sequence))
            arrival_seconds.append(arrival)
            departure_seconds.append(departure)
        trip_stop_time_indptr.append(len(stop_time_trip_index))
        array_progress.update(trip_position, current_unit=trip_id)

    trip_line_index = [
        line_index[str(getattr(trips_by_id[trip_id], "line_ref"))]
        for trip_id in trip_ids
    ]
    trip_direction_id = [
        -1
        if getattr(trips_by_id[trip_id], "direction_id") is None
        else int(getattr(trips_by_id[trip_id], "direction_id"))
        for trip_id in trip_ids
    ]

    numerical: dict[str, np.ndarray] = {
        "arrival_seconds": np.asarray(arrival_seconds, dtype=np.int64),
        "departure_seconds": np.asarray(departure_seconds, dtype=np.int64),
        "stop_sequence": np.asarray(stop_sequence, dtype=np.int32),
        "stop_time_physical_stop_index": np.asarray(
            stop_time_physical_stop_index, dtype=np.int32
        ),
        "stop_time_stop_index": np.asarray(stop_time_stop_index, dtype=np.int32),
        "stop_time_trip_index": np.asarray(stop_time_trip_index, dtype=np.int32),
        "trip_direction_id": np.asarray(trip_direction_id, dtype=np.int32),
        "trip_line_index": np.asarray(trip_line_index, dtype=np.int32),
        "trip_route_pattern_index": route_patterns.trip_to_pattern_index,
        "trip_service_period_index": (
            service_periods.trip_to_service_period_index
        ),
        "trip_stop_time_indptr": np.asarray(
            trip_stop_time_indptr, dtype=np.int64
        ),
    }
    arrays = tuple(
        NamedImmutableArray(name=name, values=numerical[name])
        for name in sorted(numerical)
    )
    source_fingerprints = (
        ("physical_stops", physical_stops.fingerprint),
        ("route_patterns", route_patterns.fingerprint),
        ("service_periods", service_periods.fingerprint),
        ("timetable_source", timetable_source_fingerprint(scenario)),
    )
    manifest = ReducedODArtifactManifest(
        artifact_kind=ReducedODArtifactKind.TIMETABLE,
        configuration_fingerprint=configuration_fingerprint,
        source_fingerprints=source_fingerprints,
        dimensions=(
            ("num_lines", len(line_ids)),
            ("num_physical_stops", len(physical_stop_ids)),
            ("num_route_patterns", len(route_patterns.patterns)),
            ("num_service_periods", len(service_periods.service_periods)),
            ("num_stop_times", len(stop_time_trip_index)),
            ("num_stops", len(stop_ids)),
            ("num_trips", len(trip_ids)),
        ),
    )
    return TimetableIndex(
        stop_ids=stop_ids,
        physical_stop_ids=physical_stop_ids,
        trip_ids=trip_ids,
        line_ids=line_ids,
        physical_stops=physical_stops,
        route_patterns=route_patterns,
        service_periods=service_periods,
        arrays=ReducedODArrayArtifact(manifest=manifest, arrays=arrays),
    )


def prepare_reduced_od_timetable(
    scenario: Scenario,
    *,
    configuration_fingerprint: str,
    physical_stop_mapping: Mapping[str, str] | None = None,
    mapping_policy: PhysicalStopMappingPolicy | None = None,
    progress: ReducedODProgress | None = None,
) -> TimetableIndex:
    """Build all Phase-2 timetable classifications and the compact index."""
    physical_stops = build_physical_stop_index(
        scenario,
        mapping=physical_stop_mapping,
        mapping_policy=mapping_policy,
        progress=progress,
    )
    route_patterns = build_route_pattern_index(
        scenario, physical_stops, progress=progress
    )
    service_periods = build_service_period_index(scenario, progress=progress)
    return build_timetable_index(
        scenario,
        configuration_fingerprint=configuration_fingerprint,
        physical_stops=physical_stops,
        route_patterns=route_patterns,
        service_periods=service_periods,
        progress=progress,
    )
