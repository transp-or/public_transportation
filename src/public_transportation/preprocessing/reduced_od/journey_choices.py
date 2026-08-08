"""Deterministic bounded journey choices and transfer-event accounting."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .artifacts import canonical_json
from .progress import ReducedODProgress, ReducedODProgressEmitter
from .raptor import RaptorLabel, RaptorResult, RaptorTransitLeg
from .timetable_index import TimetableIndex


UNCLASSIFIED_TIME_PERIOD_ID = "unclassified"


class JourneyEventKind(str, Enum):
    """Passenger-journey event semantics used by measurement responses."""

    FIRST_BOARDING = "first_boarding"
    TRANSFER_ALIGHTING = "transfer_alighting"
    TRANSFER_BOARDING = "transfer_boarding"
    FINAL_ALIGHTING = "final_alighting"


@dataclass(frozen=True, slots=True, order=True)
class JourneyTimePeriod:
    """Half-open time period used to label every journey event."""

    period_id: str
    start_seconds: int
    end_seconds: int

    def __post_init__(self) -> None:
        if not self.period_id:
            raise ValueError("period_id must be non-empty.")
        for value, name in (
            (self.start_seconds, "start_seconds"),
            (self.end_seconds, "end_seconds"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer.")
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("time periods must have positive half-open duration.")


@dataclass(frozen=True, slots=True, order=True)
class JourneyEvent:
    """One boarding or alighting with explicit journey and time semantics."""

    seconds: int
    event_kind: JourneyEventKind
    physical_stop_id: str
    time_period_id: str
    leg_index: int

    def __post_init__(self) -> None:
        if self.seconds < 0 or not self.physical_stop_id or not self.time_period_id:
            raise ValueError("journey event values must be valid and non-empty.")
        if self.leg_index < 0:
            raise ValueError("leg_index must be non-negative.")


@dataclass(frozen=True, slots=True)
class JourneyAlternative:
    """One complete transit journey and all of its observable leg events."""

    alternative_id: str
    origin_physical_stop_id: str
    destination_physical_stop_id: str
    origin_time_period_id: str
    query_departure_seconds: int
    arrival_seconds: int
    travel_seconds: int
    wait_seconds: int
    walk_seconds: int
    in_vehicle_seconds: int
    transfers: int
    transit_legs: tuple[RaptorTransitLeg, ...]
    events: tuple[JourneyEvent, ...]
    route_pattern_ids: tuple[str, ...]
    desired_departure_time_period_id: str | None = None

    def __post_init__(self) -> None:
        if not self.alternative_id or len(self.alternative_id) != 64:
            raise ValueError("alternative_id must be a SHA-256 hexadecimal digest.")
        if not self.transit_legs:
            raise ValueError("a public-transport journey must contain a transit leg.")
        if len(self.route_pattern_ids) != len(self.transit_legs):
            raise ValueError("route_pattern_ids must align with transit_legs.")
        if self.transfers != len(self.transit_legs) - 1:
            raise ValueError("transfers must equal the number of internal leg pairs.")
        if len(self.events) != 2 * len(self.transit_legs):
            raise ValueError("each transit leg must contribute two events.")
        if self.events[0].event_kind is not JourneyEventKind.FIRST_BOARDING:
            raise ValueError("the first event must be the journey's first boarding.")
        if self.events[-1].event_kind is not JourneyEventKind.FINAL_ALIGHTING:
            raise ValueError("the final event must be the journey's final alighting.")
        if self.origin_time_period_id != self.events[0].time_period_id:
            raise ValueError("origin time period must classify the first boarding.")
        if self.desired_departure_time_period_id == "":
            raise ValueError("desired departure period must be non-empty when set.")
        for transfer in range(self.transfers):
            alighting = self.events[1 + 2 * transfer]
            boarding = self.events[2 + 2 * transfer]
            if (
                alighting.event_kind is not JourneyEventKind.TRANSFER_ALIGHTING
                or boarding.event_kind is not JourneyEventKind.TRANSFER_BOARDING
                or boarding.seconds < alighting.seconds
            ):
                raise ValueError("every internal transfer must be a paired event.")

    @property
    def boarding_events(self) -> tuple[JourneyEvent, ...]:
        return tuple(self.events[0::2])

    @property
    def alighting_events(self) -> tuple[JourneyEvent, ...]:
        return tuple(self.events[1::2])

    @property
    def demand_time_period_id(self) -> str:
        """Desired-departure period, falling back to legacy boarding period."""
        return self.desired_departure_time_period_id or self.origin_time_period_id

    @property
    def first_boarding_time_period_id(self) -> str:
        """Period of the realized first boarding event."""
        return self.events[0].time_period_id


@dataclass(frozen=True, slots=True)
class JourneyChoicePolicy:
    """Explicit deterministic cap, ranking, and fixed-share policy."""

    maximum_alternatives_per_cell: int = 4
    transfer_penalty_seconds: float = 600.0
    walk_time_weight: float = 1.0
    share_temperature_seconds: float = 900.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_alternatives_per_cell, bool)
            or not isinstance(self.maximum_alternatives_per_cell, int)
            or self.maximum_alternatives_per_cell <= 0
        ):
            raise ValueError("maximum_alternatives_per_cell must be positive.")
        for value, name, allow_zero in (
            (self.transfer_penalty_seconds, "transfer_penalty_seconds", True),
            (self.walk_time_weight, "walk_time_weight", True),
            (self.share_temperature_seconds, "share_temperature_seconds", False),
        ):
            if not math.isfinite(value) or value < 0.0 or (not allow_zero and value == 0.0):
                raise ValueError(f"{name} has an invalid value.")


@dataclass(frozen=True, slots=True)
class JourneyChoiceSet:
    """Bounded alternatives and fixed initial shares for one journey OD cell."""

    origin_physical_stop_id: str
    destination_physical_stop_id: str
    origin_time_period_id: str
    alternatives: tuple[JourneyAlternative, ...]
    initial_shares: tuple[float, ...]
    served_time_fraction: float = 1.0

    def __post_init__(self) -> None:
        if not self.alternatives:
            raise ValueError("a feasible journey cell must retain an alternative.")
        if len(self.initial_shares) != len(self.alternatives):
            raise ValueError("initial shares must align with alternatives.")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.initial_shares):
            raise ValueError("initial shares must be finite and strictly positive.")
        if not math.isclose(sum(self.initial_shares), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("initial shares must sum to one.")
        if (
            not math.isfinite(self.served_time_fraction)
            or not 0.0 < self.served_time_fraction <= 1.0
        ):
            raise ValueError("served_time_fraction must lie in (0, 1].")
        for alternative in self.alternatives:
            if (
                alternative.origin_physical_stop_id != self.origin_physical_stop_id
                or alternative.destination_physical_stop_id
                != self.destination_physical_stop_id
                or alternative.demand_time_period_id != self.origin_time_period_id
            ):
                raise ValueError("choice-set identity must match every alternative.")

    @property
    def demand_time_period_id(self) -> str:
        """Canonical demand-cell period (legacy storage name retained)."""
        return self.origin_time_period_id


@dataclass(frozen=True, slots=True)
class JourneyChoiceDiagnostics:
    """Audit counts for deterministic choice construction and pruning."""

    feasible_destinations: int
    candidate_alternatives: int
    retained_alternatives: int
    pruned_alternatives: int
    choice_cells: int
    maximum_candidates_in_cell: int
    route_initialized_alternatives: int
    estimated_payload_bytes: int
    first_boarding_period_distribution: tuple[tuple[str, int], ...] = ()
    later_first_boarding_alternatives: int = 0
    multi_first_boarding_period_choice_sets: int = 0
    cross_period_alternatives: int = 0
    maximum_cross_period_wait_seconds: int = 0
    legacy_period_semantics_choice_sets: int = 0


@dataclass(frozen=True, slots=True)
class JourneyChoiceResult:
    """Canonical collection of choice sets with stable content identity."""

    choice_sets: tuple[JourneyChoiceSet, ...]
    diagnostics: JourneyChoiceDiagnostics

    @property
    def fingerprint_payload_json(self) -> str:
        return canonical_json(_result_payload(self.choice_sets))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.fingerprint_payload_json.encode("utf-8")).hexdigest()


def _validate_periods(periods: tuple[JourneyTimePeriod, ...]) -> None:
    if periods != tuple(sorted(periods, key=lambda item: (item.start_seconds, item.period_id))):
        raise ValueError("time_periods must be sorted by start time and identifier.")
    if len({period.period_id for period in periods}) != len(periods):
        raise ValueError("time-period identifiers must be unique.")
    for left, right in zip(periods, periods[1:]):
        if left.end_seconds > right.start_seconds:
            raise ValueError("time periods must not overlap.")


def _period_id(seconds: int, periods: tuple[JourneyTimePeriod, ...]) -> str:
    if not periods:
        return UNCLASSIFIED_TIME_PERIOD_ID
    matches = [
        period.period_id
        for period in periods
        if period.start_seconds <= seconds < period.end_seconds
    ]
    if len(matches) != 1:
        raise ValueError(
            f"event at {seconds} seconds does not map to exactly one time period."
        )
    return matches[0]


def _trip_patterns(timetable: TimetableIndex) -> dict[str, str]:
    indices = timetable.route_patterns.trip_to_pattern_index
    return {
        trip_id: timetable.route_patterns.patterns[int(indices[index])].pattern_id
        for index, trip_id in enumerate(timetable.trip_ids)
    }


def _events(
    legs: tuple[RaptorTransitLeg, ...], periods: tuple[JourneyTimePeriod, ...]
) -> tuple[JourneyEvent, ...]:
    result: list[JourneyEvent] = []
    for index, leg in enumerate(legs):
        result.append(
            JourneyEvent(
                seconds=leg.board_seconds,
                event_kind=(
                    JourneyEventKind.FIRST_BOARDING
                    if index == 0
                    else JourneyEventKind.TRANSFER_BOARDING
                ),
                physical_stop_id=leg.board_physical_stop_id,
                time_period_id=_period_id(leg.board_seconds, periods),
                leg_index=index,
            )
        )
        result.append(
            JourneyEvent(
                seconds=leg.alight_seconds,
                event_kind=(
                    JourneyEventKind.FINAL_ALIGHTING
                    if index == len(legs) - 1
                    else JourneyEventKind.TRANSFER_ALIGHTING
                ),
                physical_stop_id=leg.alight_physical_stop_id,
                time_period_id=_period_id(leg.alight_seconds, periods),
                leg_index=index,
            )
        )
    return tuple(result)


def _alternative(
    *,
    label: RaptorLabel,
    result: RaptorResult,
    patterns: Mapping[str, str],
    periods: tuple[JourneyTimePeriod, ...],
    desired_departure_time_period_id: str | None,
) -> JourneyAlternative:
    if not label.transit_legs:
        raise ValueError(
            "walking-only feasibility cannot form a public-transport journey choice."
        )
    events = _events(label.transit_legs, periods)
    route_pattern_ids = tuple(patterns[leg.trip_id] for leg in label.transit_legs)
    identity = {
        "destination": label.destination_physical_stop_id,
        "legs": [
            [
                leg.trip_id,
                leg.board_physical_stop_id,
                leg.alight_physical_stop_id,
                leg.board_seconds,
                leg.alight_seconds,
            ]
            for leg in label.transit_legs
        ],
        "origin": result.query.origin_physical_stop_id,
        "desired_departure_time_period": desired_departure_time_period_id,
        "first_boarding_time_period": events[0].time_period_id,
        "query_departure_seconds": result.query.departure_seconds,
    }
    alternative_id = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    return JourneyAlternative(
        alternative_id=alternative_id,
        origin_physical_stop_id=result.query.origin_physical_stop_id,
        destination_physical_stop_id=label.destination_physical_stop_id,
        origin_time_period_id=events[0].time_period_id,
        query_departure_seconds=result.query.departure_seconds,
        arrival_seconds=label.arrival_seconds,
        travel_seconds=label.travel_seconds,
        wait_seconds=label.wait_seconds,
        walk_seconds=label.walk_seconds,
        in_vehicle_seconds=label.in_vehicle_seconds,
        transfers=label.transfers,
        transit_legs=label.transit_legs,
        events=events,
        route_pattern_ids=route_pattern_ids,
        desired_departure_time_period_id=desired_departure_time_period_id,
    )


def _cost(alternative: JourneyAlternative, policy: JourneyChoicePolicy) -> float:
    return (
        float(alternative.travel_seconds)
        + policy.transfer_penalty_seconds * alternative.transfers
        + (policy.walk_time_weight - 1.0) * alternative.walk_seconds
    )


def _result_payload(choice_sets: tuple[JourneyChoiceSet, ...]) -> dict[str, object]:
    return {
        "choice_sets": [
            {
                "alternatives": [
                    {
                        "alternative_id": alternative.alternative_id,
                        "arrival_seconds": alternative.arrival_seconds,
                        "events": [
                            [
                                event.seconds,
                                event.event_kind.value,
                                event.physical_stop_id,
                                event.time_period_id,
                                event.leg_index,
                            ]
                            for event in alternative.events
                        ],
                        "in_vehicle_seconds": alternative.in_vehicle_seconds,
                        "query_departure_seconds": (
                            alternative.query_departure_seconds
                        ),
                        "desired_departure_time_period_id": (
                            alternative.desired_departure_time_period_id
                        ),
                        "first_boarding_time_period_id": (
                            alternative.first_boarding_time_period_id
                        ),
                        "route_pattern_ids": list(
                            alternative.route_pattern_ids
                        ),
                        "transfers": alternative.transfers,
                        "transit_legs": [
                            [
                                leg.trip_id,
                                leg.board_physical_stop_id,
                                leg.alight_physical_stop_id,
                                leg.board_seconds,
                                leg.alight_seconds,
                            ]
                            for leg in alternative.transit_legs
                        ],
                        "travel_seconds": alternative.travel_seconds,
                        "wait_seconds": alternative.wait_seconds,
                        "walk_seconds": alternative.walk_seconds,
                    }
                    for alternative in item.alternatives
                ],
                "destination": item.destination_physical_stop_id,
                "initial_shares": list(item.initial_shares),
                "served_time_fraction": item.served_time_fraction,
                "origin": item.origin_physical_stop_id,
                "demand_time_period_id": item.demand_time_period_id,
            }
            for item in choice_sets
        ]
    }


def build_journey_choices(
    timetable: TimetableIndex,
    raptor_result: RaptorResult,
    *,
    policy: JourneyChoicePolicy = JourneyChoicePolicy(),
    time_periods: tuple[JourneyTimePeriod, ...] = (),
    route_pattern_initial_weights: Mapping[str, float] | None = None,
    desired_departure_time_period_id: str | None = None,
    progress: ReducedODProgress | None = None,
) -> JourneyChoiceResult:
    """Build bounded fixed-share choices from one Phase-4 query result.

    Optional route-pattern weights initialize shares through the first boarded
    route pattern.  They are fixed metadata; they are not estimated here.
    """
    _validate_periods(time_periods)
    if desired_departure_time_period_id == "":
        raise ValueError("desired_departure_time_period_id must be non-empty when set.")
    if desired_departure_time_period_id is not None and time_periods and not any(
        period.period_id == desired_departure_time_period_id for period in time_periods
    ):
        raise ValueError("desired departure period is not in time_periods.")
    patterns = _trip_patterns(timetable)
    supplied_weights = dict(route_pattern_initial_weights or {})
    known_patterns = {pattern.pattern_id for pattern in timetable.route_patterns.patterns}
    unknown = sorted(set(supplied_weights) - known_patterns)
    if unknown:
        raise ValueError(f"route-pattern initial weights contain unknown keys: {unknown}.")
    for pattern_id, value in supplied_weights.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"route-pattern initial weight for {pattern_id!r} must be positive."
            )

    grouped: dict[tuple[str, str, str], list[JourneyAlternative]] = {}
    grouping_progress = ReducedODProgressEmitter(
        progress,
        phase="journey_choice_grouping",
        total=len(raptor_result.destinations),
    )
    grouping_progress.start()
    for destination_index, destination in enumerate(
        raptor_result.destinations, start=1
    ):
        for label in destination.labels:
            alternative = _alternative(
                label=label,
                result=raptor_result,
                patterns=patterns,
                periods=time_periods,
                desired_departure_time_period_id=desired_departure_time_period_id,
            )
            key = (
                alternative.origin_physical_stop_id,
                alternative.destination_physical_stop_id,
                alternative.demand_time_period_id,
            )
            grouped.setdefault(key, []).append(alternative)
        grouping_progress.update(
            destination_index,
            current_unit=destination.destination_physical_stop_id,
        )

    choice_sets: list[JourneyChoiceSet] = []
    pruned = 0
    maximum_candidates = 0
    route_initialized = 0
    ordered_groups = sorted(grouped)
    choice_progress = ReducedODProgressEmitter(
        progress, phase="journey_choice_cells", total=len(ordered_groups)
    )
    choice_progress.start()
    for group_index, key in enumerate(ordered_groups, start=1):
        candidates = grouped[key]
        unique = {item.alternative_id: item for item in candidates}
        ranked = sorted(
            unique.values(),
            key=lambda item: (
                _cost(item, policy),
                item.arrival_seconds,
                item.transfers,
                item.alternative_id,
            ),
        )
        maximum_candidates = max(maximum_candidates, len(ranked))
        retained = tuple(ranked[: policy.maximum_alternatives_per_cell])
        pruned += len(ranked) - len(retained)
        minimum_cost = min(_cost(item, policy) for item in retained)
        raw_shares: list[float] = []
        for item in retained:
            route_weight = supplied_weights.get(item.route_pattern_ids[0], 1.0)
            route_initialized += int(item.route_pattern_ids[0] in supplied_weights)
            raw_shares.append(
                route_weight
                * math.exp(
                    -(_cost(item, policy) - minimum_cost)
                    / policy.share_temperature_seconds
                )
            )
        total = sum(raw_shares)
        shares = tuple(value / total for value in raw_shares)
        choice_sets.append(
            JourneyChoiceSet(
                origin_physical_stop_id=key[0],
                destination_physical_stop_id=key[1],
                origin_time_period_id=key[2],
                alternatives=retained,
                initial_shares=shares,
            )
        )
        choice_progress.update(group_index, current_unit="|".join(key))

    canonical_sets = tuple(choice_sets)
    payload_bytes = len(canonical_json(_result_payload(canonical_sets)).encode("utf-8"))
    retained_count = sum(len(item.alternatives) for item in canonical_sets)
    candidate_count = retained_count + pruned
    boarding_period_counts: dict[str, int] = {}
    later_boarding = 0
    cross_period = 0
    maximum_cross_period_wait = 0
    multi_period_sets = 0
    legacy_sets = 0
    diagnostic_progress = ReducedODProgressEmitter(
        progress,
        phase="journey_choice_diagnostics",
        total=len(canonical_sets),
    )
    diagnostic_progress.start()
    for choice_position, choice in enumerate(canonical_sets, start=1):
        periods_in_choice = {
            alternative.first_boarding_time_period_id
            for alternative in choice.alternatives
        }
        multi_period_sets += int(len(periods_in_choice) > 1)
        legacy_sets += int(
            any(
                alternative.desired_departure_time_period_id is None
                for alternative in choice.alternatives
            )
        )
        for alternative in choice.alternatives:
            boarding_period = alternative.first_boarding_time_period_id
            boarding_period_counts[boarding_period] = (
                boarding_period_counts.get(boarding_period, 0) + 1
            )
            is_cross_period = boarding_period != alternative.demand_time_period_id
            cross_period += int(is_cross_period)
            later_boarding += int(is_cross_period)
            if is_cross_period:
                maximum_cross_period_wait = max(
                    maximum_cross_period_wait, alternative.wait_seconds
                )
        diagnostic_progress.update(
            choice_position,
            current_unit="|".join(
                (
                    choice.origin_physical_stop_id,
                    choice.destination_physical_stop_id,
                    choice.origin_time_period_id,
                )
            ),
        )
    return JourneyChoiceResult(
        choice_sets=canonical_sets,
        diagnostics=JourneyChoiceDiagnostics(
            feasible_destinations=len(raptor_result.destinations),
            candidate_alternatives=candidate_count,
            retained_alternatives=retained_count,
            pruned_alternatives=pruned,
            choice_cells=len(canonical_sets),
            maximum_candidates_in_cell=maximum_candidates,
            route_initialized_alternatives=route_initialized,
            estimated_payload_bytes=payload_bytes,
            first_boarding_period_distribution=tuple(
                sorted(boarding_period_counts.items())
            ),
            later_first_boarding_alternatives=later_boarding,
            multi_first_boarding_period_choice_sets=multi_period_sets,
            cross_period_alternatives=cross_period,
            maximum_cross_period_wait_seconds=maximum_cross_period_wait,
            legacy_period_semantics_choice_sets=legacy_sets,
        ),
    )
