"""Deterministic ordered stop-pattern equivalence classes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from public_transportation.domain import Scenario, Trip

from .artifacts import canonical_json
from .physical_stops import PhysicalStopIndex
from .progress import ReducedODProgress, ReducedODProgressEmitter


@dataclass(frozen=True, slots=True)
class RoutePattern:
    """Trips sharing line, direction, and complete ordered physical-stop list."""

    pattern_id: str
    line_id: str
    direction_id: int | None
    physical_stop_ids: tuple[str, ...]
    trip_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.pattern_id or not self.line_id:
            raise ValueError("pattern_id and line_id must be non-empty.")
        if len(self.physical_stop_ids) < 2:
            raise ValueError("a route pattern must contain at least two stops.")
        if not self.trip_ids or self.trip_ids != tuple(sorted(self.trip_ids)):
            raise ValueError("trip_ids must be non-empty and sorted.")
        if len(set(self.trip_ids)) != len(self.trip_ids):
            raise ValueError("trip_ids must be unique.")


@dataclass(frozen=True, slots=True)
class RoutePatternIndex:
    """Canonical patterns plus one pattern index for every sorted trip."""

    trip_ids: tuple[str, ...]
    trip_to_pattern_index: np.ndarray
    patterns: tuple[RoutePattern, ...]

    def __post_init__(self) -> None:
        if self.trip_ids != tuple(sorted(self.trip_ids)):
            raise ValueError("trip_ids must be sorted.")
        if len(set(self.trip_ids)) != len(self.trip_ids):
            raise ValueError("trip_ids must be unique.")
        indices = np.asarray(self.trip_to_pattern_index)
        if indices.shape != (len(self.trip_ids),):
            raise ValueError("trip_to_pattern_index has an invalid shape.")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("trip_to_pattern_index must contain integers.")
        immutable = np.array(indices, dtype=np.int32, copy=True, order="C")
        if immutable.size and (
            np.any(immutable < 0) or np.any(immutable >= len(self.patterns))
        ):
            raise ValueError("trip_to_pattern_index contains invalid indices.")
        pattern_ids = tuple(pattern.pattern_id for pattern in self.patterns)
        if pattern_ids != tuple(sorted(pattern_ids)):
            raise ValueError("patterns must be sorted by pattern_id.")
        covered = sorted(
            trip_id for pattern in self.patterns for trip_id in pattern.trip_ids
        )
        if covered != list(self.trip_ids):
            raise ValueError("every trip must occur in exactly one route pattern.")
        for trip_index, trip_id in enumerate(self.trip_ids):
            pattern = self.patterns[int(immutable[trip_index])]
            if trip_id not in pattern.trip_ids:
                raise ValueError("trip-to-pattern index is inconsistent.")
        immutable.setflags(write=False)
        object.__setattr__(self, "trip_to_pattern_index", immutable)

    @property
    def fingerprint_payload_json(self) -> str:
        return canonical_json(
            {
                "patterns": [
                    {
                        "direction_id": pattern.direction_id,
                        "line_id": pattern.line_id,
                        "physical_stop_ids": list(pattern.physical_stop_ids),
                        "trip_ids": list(pattern.trip_ids),
                    }
                    for pattern in self.patterns
                ],
                "trip_ids": list(self.trip_ids),
            }
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            self.fingerprint_payload_json.encode("utf-8")
        ).hexdigest()


def _ordered_stop_ids(
    scenario: Scenario, progress: ReducedODProgress | None = None
) -> dict[str, tuple[str, ...]]:
    if scenario.timetable is None:
        raise ValueError("scenario.timetable is required.")
    by_trip: dict[str, list[tuple[int, str]]] = {}
    stop_time_progress = ReducedODProgressEmitter(
        progress,
        phase="route_pattern_stop_times",
        total=len(scenario.timetable.stop_times),
    )
    stop_time_progress.start()
    for position, stop_time in enumerate(scenario.timetable.stop_times, start=1):
        if not isinstance(stop_time.sequence, int) or stop_time.sequence <= 0:
            raise ValueError(
                f"trip {stop_time.trip_id!r} has an invalid stop sequence."
            )
        by_trip.setdefault(str(stop_time.trip_id), []).append(
            (int(stop_time.sequence), str(stop_time.stop_id))
        )
        stop_time_progress.update(position, current_unit=str(stop_time.trip_id))
    result: dict[str, tuple[str, ...]] = {}
    for trip_id, rows in by_trip.items():
        ordered = sorted(rows)
        sequences = [sequence for sequence, _ in ordered]
        if len(set(sequences)) != len(sequences):
            raise ValueError(f"trip {trip_id!r} has duplicate stop sequences.")
        if len(ordered) < 2:
            raise ValueError(f"trip {trip_id!r} must contain at least two stops.")
        result[trip_id] = tuple(stop_id for _, stop_id in ordered)
    return result


def build_route_pattern_index(
    scenario: Scenario,
    physical_stops: PhysicalStopIndex,
    *,
    progress: ReducedODProgress | None = None,
) -> RoutePatternIndex:
    """Group every trip by its exact normalized ordered stop pattern."""
    if scenario.timetable is None:
        raise ValueError("scenario.timetable is required.")
    scenario_to_physical = dict(
        zip(
            physical_stops.scenario_stop_ids,
            physical_stops.physical_stop_ids_by_scenario_stop,
            strict=True,
        )
    )
    ordered_stops = _ordered_stop_ids(scenario, progress)
    trips_by_id: dict[str, Trip] = {}
    trip_progress = ReducedODProgressEmitter(
        progress,
        phase="route_pattern_trips",
        total=len(scenario.timetable.trips),
    )
    trip_progress.start()
    for trip_position, trip in enumerate(scenario.timetable.trips, start=1):
        trip_id = str(trip.trip_id)
        if trip_id in trips_by_id:
            raise ValueError(f"duplicate trip identifier {trip_id!r}.")
        trips_by_id[trip_id] = trip
        trip_progress.update(trip_position, current_unit=trip_id)
    missing_stop_times = sorted(set(trips_by_id) - set(ordered_stops))
    unknown_stop_times = sorted(set(ordered_stops) - set(trips_by_id))
    if missing_stop_times or unknown_stop_times:
        raise ValueError(
            "trips and stop times must align exactly; "
            f"missing_stop_times={missing_stop_times}, "
            f"unknown_trip_stop_times={unknown_stop_times}."
        )

    grouped: dict[
        tuple[str, int | None, tuple[str, ...]], list[str]
    ] = {}
    for trip_id, trip in trips_by_id.items():
        physical_sequence: list[str] = []
        for stop_id in ordered_stops[trip_id]:
            try:
                physical_sequence.append(scenario_to_physical[stop_id])
            except KeyError as error:
                raise ValueError(
                    f"trip {trip_id!r} references unknown stop {stop_id!r}."
                ) from error
        key = (
            str(getattr(trip, "line_ref")),
            getattr(trip, "direction_id"),
            tuple(physical_sequence),
        )
        grouped.setdefault(key, []).append(trip_id)

    canonical_groups = sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            -1 if item[0][1] is None else int(item[0][1]),
            item[0][2],
        ),
    )
    patterns = tuple(
        RoutePattern(
            pattern_id=f"route_pattern_{index:06d}",
            line_id=key[0],
            direction_id=key[1],
            physical_stop_ids=key[2],
            trip_ids=tuple(sorted(trip_ids)),
        )
        for index, (key, trip_ids) in enumerate(canonical_groups)
    )
    pattern_by_trip = {
        trip_id: index
        for index, pattern in enumerate(patterns)
        for trip_id in pattern.trip_ids
    }
    trip_ids = tuple(sorted(trips_by_id))
    indices = np.asarray(
        [pattern_by_trip[trip_id] for trip_id in trip_ids], dtype=np.int32
    )
    indices.setflags(write=False)
    return RoutePatternIndex(
        trip_ids=trip_ids,
        trip_to_pattern_index=indices,
        patterns=patterns,
    )
