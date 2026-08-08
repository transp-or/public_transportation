"""Bounded-round timetable feasibility and feature summaries.

This module deliberately works on :class:`TimetableIndex` arrays.  It does not
construct a time-expanded graph and it does not call an assignment engine.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .progress import ReducedODProgress, ReducedODProgressEmitter
from .timetable_index import TimetableIndex


class StructuralZeroReason(str, Enum):
    """Reason why a destination is absent from one timetable query."""

    ORIGIN = "origin"
    NO_TOPOLOGICAL_PATH = "no_topological_path"
    EXCEEDS_TRANSFER_LIMIT = "exceeds_transfer_limit"
    NO_TIMETABLE_FEASIBLE_JOURNEY = "no_timetable_feasible_journey"


@dataclass(frozen=True, slots=True, order=True)
class Footpath:
    """Directed walking connection between two physical stops."""

    from_physical_stop_id: str
    to_physical_stop_id: str
    duration_seconds: int

    def __post_init__(self) -> None:
        if not self.from_physical_stop_id or not self.to_physical_stop_id:
            raise ValueError("footpath stop identifiers must be non-empty.")
        if self.from_physical_stop_id == self.to_physical_stop_id:
            raise ValueError("footpaths must connect two different stops.")
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, int)
            or self.duration_seconds <= 0
        ):
            raise ValueError("footpath duration_seconds must be a positive integer.")


@dataclass(frozen=True, slots=True, order=True)
class RaptorQuery:
    """One earliest-arrival query from a physical stop and clock time."""

    origin_physical_stop_id: str
    departure_seconds: int
    maximum_transfers: int
    maximum_waiting_seconds: int | None = None
    maximum_journey_seconds: int | None = None

    def __post_init__(self) -> None:
        if not self.origin_physical_stop_id:
            raise ValueError("origin_physical_stop_id must be non-empty.")
        if (
            isinstance(self.departure_seconds, bool)
            or not isinstance(self.departure_seconds, int)
            or self.departure_seconds < 0
        ):
            raise ValueError("departure_seconds must be a non-negative integer.")
        if (
            isinstance(self.maximum_transfers, bool)
            or not isinstance(self.maximum_transfers, int)
            or self.maximum_transfers < 0
        ):
            raise ValueError("maximum_transfers must be a non-negative integer.")
        for value, name in (
            (self.maximum_waiting_seconds, "maximum_waiting_seconds"),
            (self.maximum_journey_seconds, "maximum_journey_seconds"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer when provided.")


@dataclass(frozen=True, slots=True, order=True)
class RaptorTransitLeg:
    """One scheduled vehicle leg retained in a Pareto label."""

    trip_id: str
    board_physical_stop_id: str
    alight_physical_stop_id: str
    board_seconds: int
    alight_seconds: int


@dataclass(frozen=True, slots=True)
class RaptorLabel:
    """Immutable timetable label with additive feature summaries."""

    destination_physical_stop_id: str
    arrival_seconds: int
    travel_seconds: int
    wait_seconds: int
    walk_seconds: int
    in_vehicle_seconds: int
    transfers: int
    transit_legs: tuple[RaptorTransitLeg, ...]

    @property
    def boardings(self) -> int:
        return len(self.transit_legs)


@dataclass(frozen=True, slots=True)
class DestinationFeatureSummary:
    """Pareto labels and a deterministic earliest-arrival representative."""

    destination_physical_stop_id: str
    labels: tuple[RaptorLabel, ...]
    earliest: RaptorLabel

    def __post_init__(self) -> None:
        if not self.labels or self.earliest not in self.labels:
            raise ValueError("earliest must be one of the non-empty Pareto labels.")


@dataclass(frozen=True, slots=True)
class RaptorDiagnostics:
    """Small operational counters for one bounded timetable query."""

    rounds: int
    trip_scans: int
    candidate_labels: int
    retained_labels: int
    footpath_relaxations: int


@dataclass(frozen=True, slots=True)
class RaptorResult:
    """Feasible destinations and fail-closed structural-zero classifications."""

    query: RaptorQuery
    destinations: tuple[DestinationFeatureSummary, ...]
    structural_zeros: tuple[tuple[str, StructuralZeroReason], ...]
    diagnostics: RaptorDiagnostics

    def destination(self, physical_stop_id: str) -> DestinationFeatureSummary:
        for item in self.destinations:
            if item.destination_physical_stop_id == physical_stop_id:
                return item
        raise KeyError(physical_stop_id)

    def structural_zero_reason(self, physical_stop_id: str) -> StructuralZeroReason:
        for stop_id, reason in self.structural_zeros:
            if stop_id == physical_stop_id:
                return reason
        raise KeyError(physical_stop_id)


@dataclass(frozen=True, slots=True)
class RaptorRangeResult:
    """Deterministically ordered results for several departure instants."""

    results: tuple[RaptorResult, ...]


@dataclass(frozen=True, slots=True)
class _State:
    stop: int
    arrival: int
    wait: int
    walk: int
    in_vehicle: int
    legs: tuple[RaptorTransitLeg, ...]

    @property
    def key(self) -> tuple[object, ...]:
        return (
            self.arrival,
            self.wait,
            self.walk,
            tuple(self.legs),
        )


def _dominates_across_rounds(left: _State, right: _State) -> bool:
    values_left = (left.arrival, len(left.legs))
    values_right = (right.arrival, len(right.legs))
    return all(a <= b for a, b in zip(values_left, values_right, strict=True)) and any(
        a < b for a, b in zip(values_left, values_right, strict=True)
    )


def _retain(states: list[_State], candidate: _State) -> tuple[list[_State], bool]:
    """Retain one deterministic earliest-arrival label within one round."""
    if not states or candidate.key < states[0].key:
        return [candidate], True
    return states, False


def _retain_across_rounds(
    states: list[_State], candidate: _State
) -> tuple[list[_State], bool]:
    """Retain the arrival/boardings Pareto frontier across RAPTOR rounds."""
    for current in states:
        if current.key == candidate.key or _dominates_across_rounds(
            current, candidate
        ):
            return states, False
    retained = [
        current
        for current in states
        if not _dominates_across_rounds(candidate, current)
    ]
    retained.append(candidate)
    retained.sort(key=lambda state: (state.arrival, len(state.legs), state.key))
    return retained, True


def _normalized_footpaths(
    timetable: TimetableIndex, footpaths: Iterable[Footpath]
) -> tuple[tuple[tuple[tuple[int, int], ...], ...], tuple[Footpath, ...]]:
    stop_index = {stop_id: i for i, stop_id in enumerate(timetable.physical_stop_ids)}
    parsed = tuple(sorted(footpaths))
    if len(set(parsed)) != len(parsed):
        raise ValueError("footpaths must not contain duplicates.")
    adjacency: list[list[tuple[int, int]]] = [
        [] for _ in timetable.physical_stop_ids
    ]
    for item in parsed:
        if item.from_physical_stop_id not in stop_index:
            raise ValueError(
                f"footpath references unknown stop {item.from_physical_stop_id!r}."
            )
        if item.to_physical_stop_id not in stop_index:
            raise ValueError(
                f"footpath references unknown stop {item.to_physical_stop_id!r}."
            )
        adjacency[stop_index[item.from_physical_stop_id]].append(
            (stop_index[item.to_physical_stop_id], item.duration_seconds)
        )
    return tuple(tuple(values) for values in adjacency), parsed


def _footpath_closure(
    labels: list[list[_State]], adjacency: tuple[tuple[tuple[int, int], ...], ...]
) -> tuple[int, int]:
    queue = deque(
        (stop, state)
        for stop, states in enumerate(labels)
        for state in tuple(states)
    )
    candidates = 0
    retained_count = 0
    while queue:
        stop, state = queue.popleft()
        for destination, duration in adjacency[stop]:
            candidates += 1
            candidate = _State(
                stop=destination,
                arrival=state.arrival + duration,
                wait=state.wait,
                walk=state.walk + duration,
                in_vehicle=state.in_vehicle,
                legs=state.legs,
            )
            retained, added = _retain(labels[destination], candidate)
            labels[destination] = retained
            if added:
                retained_count += 1
                queue.append((destination, candidate))
    return candidates, retained_count


def _trip_rows(timetable: TimetableIndex) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    indptr = timetable.array("trip_stop_time_indptr")
    stops = timetable.array("stop_time_physical_stop_index")
    arrivals = timetable.array("arrival_seconds")
    departures = timetable.array("departure_seconds")
    return tuple(
        tuple(
            (int(stops[row]), int(arrivals[row]), int(departures[row]))
            for row in range(int(indptr[trip]), int(indptr[trip + 1]))
        )
        for trip in range(len(timetable.trip_ids))
    )


def _minimum_boardings(
    timetable: TimetableIndex,
    origin: int,
    footpaths: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[int, ...]:
    """Topology-only 0/1 shortest paths; schedule times are intentionally ignored."""
    transit: list[set[int]] = [set() for _ in timetable.physical_stop_ids]
    for rows in _trip_rows(timetable):
        for position, (source, _, _) in enumerate(rows[:-1]):
            transit[source].update(row[0] for row in rows[position + 1 :])
    infinity = len(timetable.physical_stop_ids) + 1
    distance = [infinity] * len(timetable.physical_stop_ids)
    distance[origin] = 0
    queue: list[tuple[int, int]] = [(0, origin)]
    while queue:
        cost, stop = heapq.heappop(queue)
        if cost != distance[stop]:
            continue
        for destination, _ in footpaths[stop]:
            if cost < distance[destination]:
                distance[destination] = cost
                heapq.heappush(queue, (cost, destination))
        for destination in transit[stop]:
            candidate = cost + 1
            if candidate < distance[destination]:
                distance[destination] = candidate
                heapq.heappush(queue, (candidate, destination))
    return tuple(distance)


def run_raptor_query(
    timetable: TimetableIndex,
    query: RaptorQuery,
    *,
    footpaths: Iterable[Footpath] = (),
    progress: ReducedODProgress | None = None,
) -> RaptorResult:
    """Run a deterministic bounded-round schedule query.

    A round adds exactly one vehicle boarding.  Within a round, nondominated
    arrival/wait/walk labels are retained, so alternatives that matter to the
    later feature model are not collapsed to a single earliest-arrival path.
    """
    stop_index = {stop_id: i for i, stop_id in enumerate(timetable.physical_stop_ids)}
    if query.origin_physical_stop_id not in stop_index:
        raise ValueError(
            f"unknown query origin {query.origin_physical_stop_id!r}."
        )
    adjacency, _ = _normalized_footpaths(timetable, footpaths)
    origin = stop_index[query.origin_physical_stop_id]
    number_of_stops = len(timetable.physical_stop_ids)
    previous: list[list[_State]] = [[] for _ in range(number_of_stops)]
    previous[origin] = [
        _State(origin, query.departure_seconds, 0, 0, 0, ())
    ]
    footpath_candidates, footpath_retained = _footpath_closure(previous, adjacency)
    all_rounds: list[list[list[_State]]] = [previous]
    trip_rows = _trip_rows(timetable)
    candidate_count = footpath_candidates
    retained_count = 1 + footpath_retained
    trip_scans = 0

    round_progress = ReducedODProgressEmitter(
        progress, phase="raptor_rounds", total=query.maximum_transfers + 1
    )
    round_progress.start()
    for _round in range(1, query.maximum_transfers + 2):
        current: list[list[_State]] = [[] for _ in range(number_of_stops)]
        for trip_index, rows in enumerate(trip_rows):
            trip_scans += 1
            onboard: tuple[_State, int, int] | None = None
            onboard_rank: tuple[object, ...] | None = None
            for position, (board_stop, arrival, departure) in enumerate(rows):
                if onboard is not None:
                    prior, boarded_stop, boarded_seconds = onboard
                    candidate_count += 1
                    leg = RaptorTransitLeg(
                        trip_id=timetable.trip_ids[trip_index],
                        board_physical_stop_id=timetable.physical_stop_ids[
                            boarded_stop
                        ],
                        alight_physical_stop_id=timetable.physical_stop_ids[
                            board_stop
                        ],
                        board_seconds=boarded_seconds,
                        alight_seconds=arrival,
                    )
                    candidate = _State(
                        stop=board_stop,
                        arrival=arrival,
                        wait=prior.wait + boarded_seconds - prior.arrival,
                        walk=prior.walk,
                        in_vehicle=(
                            prior.in_vehicle + arrival - boarded_seconds
                        ),
                        legs=prior.legs + (leg,),
                    )
                    retained, added = _retain(current[board_stop], candidate)
                    current[board_stop] = retained
                    retained_count += int(added)
                if position == len(rows) - 1:
                    continue
                for prior in previous[board_stop]:
                    if prior.arrival > departure:
                        continue
                    rank = (
                        prior.wait + departure - prior.arrival,
                        prior.walk,
                        prior.in_vehicle,
                        prior.legs,
                        board_stop,
                    )
                    if onboard_rank is None or rank < onboard_rank:
                        onboard = (prior, board_stop, departure)
                        onboard_rank = rank
        walk_candidates, walk_retained = _footpath_closure(current, adjacency)
        candidate_count += walk_candidates
        footpath_candidates += walk_candidates
        retained_count += walk_retained
        all_rounds.append(current)
        previous = current
        round_progress.update(
            _round,
            current_unit=f"round-{_round}",
            details={"trip_scans": trip_scans, "retained_labels": retained_count},
        )

    feasible: list[DestinationFeatureSummary] = []
    feasible_ids: set[str] = set()
    destination_progress = ReducedODProgressEmitter(
        progress, phase="raptor_destinations", total=number_of_stops
    )
    destination_progress.start()
    for destination, stop_id in enumerate(timetable.physical_stop_ids):
        if destination == origin:
            destination_progress.update(destination + 1, current_unit=stop_id)
            continue
        states: list[_State] = []
        for round_labels in all_rounds:
            for state in round_labels[destination]:
                states, _ = _retain_across_rounds(states, state)
        labels = tuple(
            RaptorLabel(
                destination_physical_stop_id=stop_id,
                arrival_seconds=state.arrival,
                travel_seconds=state.arrival - query.departure_seconds,
                wait_seconds=state.wait,
                walk_seconds=state.walk,
                in_vehicle_seconds=state.in_vehicle,
                transfers=max(0, len(state.legs) - 1),
                transit_legs=state.legs,
            )
            for state in states
            if (
                query.maximum_waiting_seconds is None
                or state.wait <= query.maximum_waiting_seconds
            )
            and (
                query.maximum_journey_seconds is None
                or state.arrival - query.departure_seconds
                <= query.maximum_journey_seconds
            )
        )
        if labels:
            earliest = min(
                labels,
                key=lambda label: (
                    label.arrival_seconds,
                    label.transfers,
                    label.walk_seconds,
                    label.wait_seconds,
                    label.transit_legs,
                ),
            )
            feasible.append(DestinationFeatureSummary(stop_id, labels, earliest))
            feasible_ids.add(stop_id)
        destination_progress.update(destination + 1, current_unit=stop_id)

    minimum_boardings = _minimum_boardings(timetable, origin, adjacency)
    structural_zeros: list[tuple[str, StructuralZeroReason]] = []
    infinity = number_of_stops + 1
    zero_progress = ReducedODProgressEmitter(
        progress, phase="raptor_structural_zeros", total=number_of_stops
    )
    zero_progress.start()
    for destination, stop_id in enumerate(timetable.physical_stop_ids):
        if destination == origin:
            reason = StructuralZeroReason.ORIGIN
        elif stop_id in feasible_ids:
            zero_progress.update(destination + 1, current_unit=stop_id)
            continue
        elif minimum_boardings[destination] == infinity:
            reason = StructuralZeroReason.NO_TOPOLOGICAL_PATH
        elif minimum_boardings[destination] > query.maximum_transfers + 1:
            reason = StructuralZeroReason.EXCEEDS_TRANSFER_LIMIT
        else:
            reason = StructuralZeroReason.NO_TIMETABLE_FEASIBLE_JOURNEY
        structural_zeros.append((stop_id, reason))
        zero_progress.update(destination + 1, current_unit=stop_id)

    retained_final = sum(
        len(summary.labels) for summary in feasible
    )
    return RaptorResult(
        query=query,
        destinations=tuple(feasible),
        structural_zeros=tuple(structural_zeros),
        diagnostics=RaptorDiagnostics(
            rounds=query.maximum_transfers + 1,
            trip_scans=trip_scans,
            candidate_labels=candidate_count,
            retained_labels=retained_final,
            footpath_relaxations=footpath_candidates,
        ),
    )


def run_raptor_range_query(
    timetable: TimetableIndex,
    *,
    origin_physical_stop_id: str,
    departure_seconds: Iterable[int],
    maximum_transfers: int,
    maximum_waiting_seconds: int | None = None,
    maximum_journey_seconds: int | None = None,
    footpaths: Iterable[Footpath] = (),
    progress: ReducedODProgress | None = None,
) -> RaptorRangeResult:
    """Run independent, deterministically ordered timetable queries."""
    times = tuple(sorted(set(departure_seconds)))
    if not times:
        raise ValueError("departure_seconds must contain at least one time.")
    paths = tuple(footpaths)
    query_progress = ReducedODProgressEmitter(
        progress, phase="raptor_range_queries", total=len(times)
    )
    query_progress.start()
    results = []
    for position, time in enumerate(times, start=1):
        results.append(
            run_raptor_query(
                timetable,
                RaptorQuery(
                    origin_physical_stop_id,
                    time,
                    maximum_transfers,
                    maximum_waiting_seconds,
                    maximum_journey_seconds,
                ),
                footpaths=paths,
                progress=progress,
            )
        )
        query_progress.update(position, current_unit=str(time))
    return RaptorRangeResult(results=tuple(results))
