"""Immutable, reusable indexes for a validated timetable.

The assignment graph, OD-universe generation, and structural-zero analysis all
need the same normalized timetable facts.  This module computes those facts
once per explicitly supplied scenario/index pair.  It deliberately contains
no routing algorithm and no process-global cache.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

from public_transportation.domain import Scenario


CANONICAL_TIMETABLE_SCHEMA_VERSION = 1
CANONICAL_TIMETABLE_ALGORITHM_VERSION = 1


@dataclass(frozen=True, slots=True, order=True)
class CanonicalStopTime:
    """Normalized stop-time record with integer seconds and identifiers."""

    trip_id: str
    stop_id: str
    sequence: int
    arrival_s: int
    departure_s: int


@dataclass(frozen=True, slots=True, order=True)
class CanonicalTrip:
    """Trip metadata in the source timetable's stable trip order."""

    trip_id: str
    line_ref: str
    capacity: float | None = None
    service_id: str | None = None
    headsign: str | None = None
    direction_id: int | None = None


@dataclass(frozen=True, slots=True)
class CanonicalTimetableIndex:
    """Immutable timetable/event index shared by preprocessing consumers."""

    schema_version: int
    algorithm_version: int
    source_fingerprint: str
    stop_ids: tuple[str, ...]
    time_bins: tuple[tuple[str, int, int], ...]
    trips: tuple[CanonicalTrip, ...]
    stop_times: tuple[CanonicalStopTime, ...]
    trip_sequences: Mapping[str, tuple[CanonicalStopTime, ...]]
    trip_index_by_id: Mapping[str, int]
    departures_by_stop: Mapping[str, tuple[tuple[int, str, int], ...]]
    arrivals_by_stop: Mapping[str, tuple[tuple[int, str, int], ...]]
    departure_seconds_by_stop: Mapping[str, tuple[int, ...]]
    arrival_seconds_by_stop: Mapping[str, tuple[int, ...]]
    event_index_by_stop_time_trip: Mapping[tuple[str, int, str, str], int]
    route_patterns: Mapping[str, tuple[tuple[str, ...], ...]]
    service_stops: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        """Stable identity of normalized scientific timetable content."""
        payload = {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "source_fingerprint": self.source_fingerprint,
            "stop_ids": list(self.stop_ids),
            "time_bins": [list(item) for item in self.time_bins],
            "trips": [_trip_payload(item) for item in self.trips],
            "stop_times": [_stop_time_payload(item) for item in self.stop_times],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @property
    def num_stop_times(self) -> int:
        return len(self.stop_times)

    def regularized_stop_times(self, *, minimum_dwell_seconds: int) -> tuple[CanonicalStopTime, ...]:
        """Return records with the assignment dwell policy applied."""
        minimum = int(minimum_dwell_seconds)
        if minimum <= 0:
            raise ValueError("minimum_dwell_seconds must be positive")
        return tuple(
            record
            if record.departure_s != record.arrival_s
            else CanonicalStopTime(
                trip_id=record.trip_id,
                stop_id=record.stop_id,
                sequence=record.sequence,
                arrival_s=record.arrival_s,
                departure_s=record.arrival_s + minimum,
            )
            for record in self.stop_times
        )


def build_canonical_timetable_index(scenario: Scenario) -> CanonicalTimetableIndex:
    """Build and validate one immutable index from a scenario timetable."""
    timetable = scenario.timetable
    if timetable is None:
        raise ValueError("Scenario has no timetable.")

    if isinstance(scenario.stops, dict):
        stop_ids = tuple(sorted(str(stop_id) for stop_id in scenario.stops))
    else:
        stop_ids = tuple(
            sorted(
                str(getattr(stop, "stop_id", getattr(stop, "id", "")))
                for stop in scenario.stops
            )
        )
    if len(stop_ids) != len(set(stop_ids)):
        raise ValueError("Scenario stop identifiers must be unique.")
    stop_id_set = set(stop_ids)

    trips: list[CanonicalTrip] = []
    trip_index_by_id: dict[str, int] = {}
    for trip in timetable.trips:
        trip_id = _required_text(getattr(trip, "trip_id", None), "trip_id")
        if trip_id in trip_index_by_id:
            raise ValueError(f"Duplicate trip identifier: {trip_id!r}")
        raw_line_ref = _first_attr(trip, "line_ref", "line_id")
        if raw_line_ref is None or not str(raw_line_ref).strip():
            raise ValueError("line_ref is empty")
        line_ref = str(raw_line_ref).strip()
        trip_index_by_id[trip_id] = len(trips)
        trips.append(
            CanonicalTrip(
                trip_id=trip_id,
                line_ref=line_ref,
                capacity=getattr(trip, "capacity", None),
                service_id=getattr(trip, "service_id", None),
                headsign=getattr(trip, "headsign", None),
                direction_id=getattr(trip, "direction_id", None),
            )
        )

    raw_records: list[CanonicalStopTime] = []
    for record in timetable.stop_times:
        trip_id = _required_text(getattr(record, "trip_id", None), "stop_time.trip_id")
        stop_id = _required_text(getattr(record, "stop_id", None), "stop_time.stop_id")
        if trip_id not in trip_index_by_id:
            raise ValueError(f"Unknown trip_id in stop_times: {trip_id!r}")
        if stop_id not in stop_id_set:
            raise ValueError(f"Unknown stop_id in stop_times: {stop_id!r}")
        sequence = _integer(
            _first_attr(record, "sequence", "stop_sequence"),
            "stop_time.sequence",
        )
        arrival_s = _seconds(
            _first_attr(record, "arrival", "arrival_time", "arrival_s"),
            "stop_time.arrival",
        )
        departure_s = _seconds(
            _first_attr(record, "departure", "departure_time", "departure_s"),
            "stop_time.departure",
        )
        raw_records.append(
            CanonicalStopTime(trip_id, stop_id, sequence, arrival_s, departure_s)
        )

    stop_times = tuple(sorted(raw_records, key=lambda item: (item.trip_id, item.sequence, item.stop_id)))
    trip_sequences: dict[str, tuple[CanonicalStopTime, ...]] = {}
    for trip_id in trip_index_by_id:
        sequence = tuple(record for record in stop_times if record.trip_id == trip_id)
        if not sequence:
            raise ValueError(f"Trip {trip_id!r} has no stop-time records.")
        if len({record.sequence for record in sequence}) != len(sequence):
            raise ValueError(f"Trip {trip_id!r} has duplicate stop sequences.")
        trip_sequences[trip_id] = sequence

    departures: dict[str, list[tuple[int, str, int]]] = {}
    arrivals: dict[str, list[tuple[int, str, int]]] = {}
    event_index: dict[tuple[str, int, str, str], int] = {}
    for index, record in enumerate(stop_times):
        departures.setdefault(record.stop_id, []).append(
            (record.departure_s, record.trip_id, record.sequence)
        )
        arrivals.setdefault(record.stop_id, []).append(
            (record.arrival_s, record.trip_id, record.sequence)
        )
        event_index[(record.stop_id, record.arrival_s, record.trip_id, "arrival")] = index
        event_index[(record.stop_id, record.departure_s, record.trip_id, "departure")] = index

    ordered_departures = _ordered_events(departures)
    ordered_arrivals = _ordered_events(arrivals)
    patterns: dict[str, set[tuple[str, ...]]] = {}
    for trip in trips:
        patterns.setdefault(trip.line_ref, set()).add(
            tuple(record.stop_id for record in trip_sequences[trip.trip_id])
        )

    time_bins = tuple(
        (
            str(getattr(item, "bin_id", index)),
            _seconds(
                _first_attr(item, "start", "start_s"), "time_bin.start"
            ),
            _seconds(_first_attr(item, "end", "end_s"), "time_bin.end"),
        )
        for index, item in enumerate(scenario.time_bins)
    )

    source_payload = {
        "stops": list(stop_ids),
        "time_bins": [list(item) for item in time_bins],
        "trips": [_trip_payload(item) for item in trips],
        "stop_times": [_stop_time_payload(item) for item in stop_times],
    }
    source_fingerprint = hashlib.sha256(
        json.dumps(source_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CanonicalTimetableIndex(
        schema_version=CANONICAL_TIMETABLE_SCHEMA_VERSION,
        algorithm_version=CANONICAL_TIMETABLE_ALGORITHM_VERSION,
        source_fingerprint=source_fingerprint,
        stop_ids=stop_ids,
        time_bins=time_bins,
        trips=tuple(trips),
        stop_times=stop_times,
        trip_sequences=MappingProxyType(trip_sequences),
        trip_index_by_id=MappingProxyType(dict(trip_index_by_id)),
        departures_by_stop=MappingProxyType(ordered_departures),
        arrivals_by_stop=MappingProxyType(ordered_arrivals),
        departure_seconds_by_stop=MappingProxyType(
            {stop: tuple(item[0] for item in values) for stop, values in ordered_departures.items()}
        ),
        arrival_seconds_by_stop=MappingProxyType(
            {stop: tuple(item[0] for item in values) for stop, values in ordered_arrivals.items()}
        ),
        event_index_by_stop_time_trip=MappingProxyType(event_index),
        route_patterns=MappingProxyType(
            {line: tuple(sorted(values)) for line, values in sorted(patterns.items())}
        ),
        service_stops=tuple(sorted(set(departures) | set(arrivals))),
    )


def _ordered_events(
    values: Mapping[str, list[tuple[int, str, int]]]
) -> dict[str, tuple[tuple[int, str, int], ...]]:
    return {
        stop: tuple(sorted(items, key=lambda item: (item[0], item[1], item[2])))
        for stop, items in sorted(values.items())
    }


def _required_text(value: Any, name: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty identifier.")
    return str(value).strip()


def _first_attr(value: Any, *names: str) -> Any:
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _integer(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer.") from error
    return result


def _seconds(value: Any, name: str) -> int:
    if hasattr(value, "seconds_from_midnight"):
        return int(value.seconds_from_midnight)
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must expose seconds_from_midnight or be an integer.") from error


def _trip_payload(value: CanonicalTrip) -> dict[str, Any]:
    return {
        "trip_id": value.trip_id,
        "line_ref": value.line_ref,
        "capacity": value.capacity,
        "service_id": value.service_id,
        "headsign": value.headsign,
        "direction_id": value.direction_id,
    }


def _stop_time_payload(value: CanonicalStopTime) -> list[Any]:
    return [value.trip_id, value.stop_id, value.sequence, value.arrival_s, value.departure_s]


__all__ = [
    "CANONICAL_TIMETABLE_ALGORITHM_VERSION",
    "CANONICAL_TIMETABLE_SCHEMA_VERSION",
    "CanonicalStopTime",
    "CanonicalTimetableIndex",
    "CanonicalTrip",
    "build_canonical_timetable_index",
]
