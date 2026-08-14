"""Independent candidate OD universes, OD--time expansion, and priors.

This module deliberately keeps three concepts separate:

* an ordered origin/destination pair universe;
* the approved time-bin expansion of that universe; and
* numerical prior values used by a statistical model.

None of the pair-level fingerprints include ``Scenario.time_bins``.  This is
important when a case owner changes the temporal resolution after reviewing
the count timestamps: the OD universe remains the same and only the later
OD--time expansion is invalidated.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Sequence

from public_transportation.domain import Scenario

from .reduced_od.artifacts import canonical_json


ODUniverseSource = Literal["file", "network_ordered_pairs"]
ODUniverseLevel = Literal["stop", "physical_stop"]
ConnectivityPolicy = Literal["none", "directed_reachable"]
TimetablePolicy = Literal["required", "defer", "none"]


def _text(value: Any, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} must be a non-empty identifier.")
    parsed = str(value).strip()
    if not parsed:
        raise ValueError(f"{name} must be a non-empty identifier.")
    return parsed


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class CandidateODPair:
    """One ordered candidate origin/destination pair."""

    origin_stop_id: str
    destination_stop_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "origin_stop_id", _text(self.origin_stop_id, "origin_stop_id")
        )
        object.__setattr__(
            self,
            "destination_stop_id",
            _text(self.destination_stop_id, "destination_stop_id"),
        )

    @property
    def tuple(self) -> tuple[str, str]:
        return (self.origin_stop_id, self.destination_stop_id)


@dataclass(frozen=True, slots=True)
class ODUniverseExclusion:
    """Audit record for one pair removed by an explicit rule."""

    origin_stop_id: str
    destination_stop_id: str
    reason: str
    detail: str = ""

    @property
    def tuple(self) -> tuple[str, str, str, str]:
        return (
            self.origin_stop_id,
            self.destination_stop_id,
            self.reason,
            self.detail,
        )


@dataclass(frozen=True, slots=True)
class CandidateODUniverse:
    """Validated immutable pair universe and its exclusion audit."""

    pairs: tuple[CandidateODPair, ...]
    exclusions: tuple[ODUniverseExclusion, ...]
    source: ODUniverseSource
    level: ODUniverseLevel
    include_same_stop: bool
    active_service_only: bool
    connectivity_policy: ConnectivityPolicy
    physical_stop_mapping: Mapping[str, str]
    generator_fingerprint: str

    def __post_init__(self) -> None:
        if self.pairs != tuple(sorted(self.pairs)):
            raise ValueError("candidate OD pairs must use canonical sorted order.")
        if len(set(self.pairs)) != len(self.pairs):
            raise ValueError("candidate OD pairs must be unique.")
        if self.source not in {"file", "network_ordered_pairs"}:
            raise ValueError("unsupported OD-universe source.")
        if self.level not in {"stop", "physical_stop"}:
            raise ValueError("unsupported OD-universe level.")
        if self.connectivity_policy not in {"none", "directed_reachable"}:
            raise ValueError("unsupported connectivity policy.")

    @property
    def pair_count(self) -> int:
        return len(self.pairs)

    @property
    def fingerprint(self) -> str:
        return _sha256_payload(
            {
                "generator_fingerprint": self.generator_fingerprint,
                "level": self.level,
                "pairs": [list(pair.tuple) for pair in self.pairs],
            }
        )

    @property
    def audit(self) -> dict[str, object]:
        counts: dict[str, int] = defaultdict(int)
        for exclusion in self.exclusions:
            counts[exclusion.reason] += 1
        return {
            "source": self.source,
            "level": self.level,
            "include_same_stop": self.include_same_stop,
            "active_service_only": self.active_service_only,
            "connectivity_policy": self.connectivity_policy,
            "input_pair_count": self.pair_count + len(self.exclusions),
            "retained_pair_count": self.pair_count,
            "exclusion_counts": dict(sorted(counts.items())),
            "fingerprint": self.fingerprint,
            "generator_fingerprint": self.generator_fingerprint,
        }


@dataclass(frozen=True, slots=True, order=True)
class CandidateODTimeCell:
    """One candidate OD pair assigned to one approved time interval."""

    origin_stop_id: str
    destination_stop_id: str
    time_bin_id: str

    @property
    def tuple(self) -> tuple[str, str, str]:
        return (self.origin_stop_id, self.destination_stop_id, self.time_bin_id)


@dataclass(frozen=True, slots=True)
class ODTimeExclusion:
    """Audit record for one pair/time-bin cell removed by a rule."""

    origin_stop_id: str
    destination_stop_id: str
    time_bin_id: str
    reason: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ODTimeExpansion:
    """Immutable expanded candidate cells and complete exclusion audit."""

    universe_fingerprint: str
    cells: tuple[CandidateODTimeCell, ...]
    exclusions: tuple[ODTimeExclusion, ...]
    time_bins: tuple[tuple[str, int, int], ...]
    policies: Mapping[str, object]
    fingerprint: str

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    @property
    def audit(self) -> dict[str, object]:
        counts: dict[str, int] = defaultdict(int)
        for exclusion in self.exclusions:
            counts[exclusion.reason] += 1
        expanded_count = len(self.cells) + len(self.exclusions)
        pair_count = (
            expanded_count // len(self.time_bins)
            if self.time_bins
            else 0
        )
        return {
            "input_pair_count": pair_count,
            "time_bin_count": len(self.time_bins),
            "expanded_od_time_count": expanded_count,
            "retained_cell_count": len(self.cells),
            "exclusion_counts": dict(sorted(counts.items())),
            "universe_fingerprint": self.universe_fingerprint,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class PriorGenerationResult:
    """Generated prior values and semantic provenance."""

    values: Mapping[CandidateODTimeCell, float]
    source: str
    semantics: str
    parameters: Mapping[str, object]
    generator_fingerprint: str
    fingerprint: str

    @property
    def audit(self) -> dict[str, object]:
        return {
            "source": self.source,
            "semantics": self.semantics,
            "parameters": dict(self.parameters),
            "cell_count": len(self.values),
            "generator_fingerprint": self.generator_fingerprint,
            "fingerprint": self.fingerprint,
        }


def _mapping_for_level(
    scenario: Scenario,
    *,
    level: ODUniverseLevel,
    physical_stop_mapping: Mapping[str, str] | None,
) -> dict[str, str]:
    stop_ids = {str(stop.stop_id) for stop in scenario.stops}
    if level == "stop":
        return {stop_id: stop_id for stop_id in sorted(stop_ids)}
    if physical_stop_mapping is None:
        return {stop_id: stop_id for stop_id in sorted(stop_ids)}
    normalized = {
        _text(stop_id, "physical-stop mapping stop_id"): _text(
            physical_id, "physical-stop mapping physical_stop_id"
        )
        for stop_id, physical_id in physical_stop_mapping.items()
    }
    missing = sorted(stop_ids - set(normalized))
    unknown = sorted(set(normalized) - stop_ids)
    if missing or unknown:
        raise ValueError(
            "physical-stop mapping must cover exactly the scenario stops; "
            f"missing={missing}, unknown={unknown}."
        )
    return normalized


def _trip_sequences(scenario: Scenario, mapping: Mapping[str, str]) -> dict[str, tuple[object, ...]]:
    if scenario.timetable is None:
        return {}
    by_trip: dict[str, list[object]] = defaultdict(list)
    for stop_time in scenario.timetable.stop_times:
        by_trip[str(stop_time.trip_id)].append(stop_time)
    return {
        trip_id: tuple(
            sorted(
                values,
                key=lambda item: int(getattr(item, "sequence")),
            )
        )
        for trip_id, values in by_trip.items()
    }


def _service_activity(
    scenario: Scenario,
    mapping: Mapping[str, str],
) -> tuple[set[str], set[str]]:
    departures: set[str] = set()
    arrivals: set[str] = set()
    if scenario.timetable is None:
        return departures, arrivals
    for stop_time in scenario.timetable.stop_times:
        physical = mapping[str(stop_time.stop_id)]
        departures.add(physical)
        arrivals.add(physical)
    return departures, arrivals


def _directed_reachability(
    scenario: Scenario,
    mapping: Mapping[str, str],
) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for sequence in _trip_sequences(scenario, mapping).values():
        for left, right in zip(sequence, sequence[1:], strict=False):
            adjacency[mapping[str(left.stop_id)]].add(mapping[str(right.stop_id)])
    nodes = set(mapping.values())
    reachable: dict[str, set[str]] = {}
    for origin in sorted(nodes):
        seen = {origin}
        queue: deque[str] = deque([origin])
        while queue:
            current = queue.popleft()
            for destination in sorted(adjacency.get(current, ())):
                if destination not in seen:
                    seen.add(destination)
                    queue.append(destination)
        reachable[origin] = seen
    return reachable


def _read_pair_file(
    path: str | Path,
    *,
    allowed_ids: set[str],
    identifier_label: str,
) -> tuple[CandidateODPair, ...]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"candidate OD-pair file does not exist: {source}")
    pairs: list[CandidateODPair] = []
    with source.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("candidate OD-pair file has no header.")
        required = {"origin_stop_id", "destination_stop_id"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(
                "candidate OD-pair file is missing required columns: "
                f"{missing}. Expected origin_stop_id,destination_stop_id."
            )
        extra = sorted(set(reader.fieldnames) - required)
        if extra:
            raise ValueError(
                "candidate OD-pair file must be pair-only; unexpected columns: "
                f"{extra}. Remove time-bin and flow columns."
            )
        for row_number, row in enumerate(reader, start=2):
            origin = _text(row.get("origin_stop_id"), f"OD pair row {row_number} origin_stop_id")
            destination = _text(
                row.get("destination_stop_id"),
                f"OD pair row {row_number} destination_stop_id",
            )
            if origin not in allowed_ids or destination not in allowed_ids:
                raise ValueError(
                    f"OD pair row {row_number} references an unknown {identifier_label}: "
                    f"{origin!r}, {destination!r}."
                )
            pairs.append(CandidateODPair(origin, destination))
    if len(set(pairs)) != len(pairs):
        raise ValueError("candidate OD-pair file contains duplicate ordered pairs.")
    return tuple(sorted(pairs))


def generate_candidate_od_pairs(
    scenario: Scenario,
    *,
    source: ODUniverseSource = "network_ordered_pairs",
    level: ODUniverseLevel = "stop",
    include_same_stop: bool = False,
    active_service_only: bool = True,
    connectivity_policy: ConnectivityPolicy = "directed_reachable",
    od_pairs_path: str | Path | None = None,
    physical_stop_mapping: Mapping[str, str] | None = None,
) -> CandidateODUniverse:
    """Generate and validate an immutable ordered candidate OD universe.

    The function never reads ``scenario.demand`` or ``scenario.time_bins``.
    ``source='file'`` requires a pair-only CSV with no time-bin membership.
    """
    if source not in {"file", "network_ordered_pairs"}:
        raise ValueError("source must be 'file' or 'network_ordered_pairs'.")
    if connectivity_policy not in {"none", "directed_reachable"}:
        raise ValueError("connectivity_policy must be 'none' or 'directed_reachable'.")
    mapping = _mapping_for_level(
        scenario,
        level=level,
        physical_stop_mapping=physical_stop_mapping,
    )
    if source == "file":
        if od_pairs_path is None:
            raise ValueError("source='file' requires od_pairs_path.")
        raw_pairs = _read_pair_file(
            od_pairs_path,
            allowed_ids=set(mapping.values()),
            identifier_label=("physical stop" if level == "physical_stop" else "network stop"),
        )
    else:
        nodes = sorted(set(mapping.values()))
        raw_pairs = tuple(
            CandidateODPair(origin, destination)
            for origin in nodes
            for destination in nodes
        )
    departures, arrivals = _service_activity(scenario, mapping)
    reachable = _directed_reachability(scenario, mapping)
    retained: list[CandidateODPair] = []
    exclusions: list[ODUniverseExclusion] = []
    for pair in raw_pairs:
        if not include_same_stop and pair.origin_stop_id == pair.destination_stop_id:
            exclusions.append(ODUniverseExclusion(*pair.tuple, "same_node"))
            continue
        if active_service_only and pair.origin_stop_id not in departures:
            exclusions.append(ODUniverseExclusion(*pair.tuple, "inactive_origin"))
            continue
        if active_service_only and pair.destination_stop_id not in arrivals:
            exclusions.append(ODUniverseExclusion(*pair.tuple, "inactive_destination"))
            continue
        if (
            connectivity_policy == "directed_reachable"
            and pair.destination_stop_id not in reachable.get(pair.origin_stop_id, set())
        ):
            exclusions.append(ODUniverseExclusion(*pair.tuple, "static_unreachable"))
            continue
        retained.append(pair)
    generator_fingerprint = _sha256_payload(
        {
            "source": source,
            "level": level,
            "include_same_stop": include_same_stop,
            "active_service_only": active_service_only,
            "connectivity_policy": connectivity_policy,
            "od_pairs_path": None if od_pairs_path is None else str(Path(od_pairs_path).resolve()),
            "mapping": sorted(mapping.items()),
            "network_nodes": sorted(set(mapping.values())),
            "directed_edges": sorted(
                (origin, destination)
                for origin, destinations in _directed_reachability(scenario, mapping).items()
                for destination in destinations
            ),
        }
    )
    return CandidateODUniverse(
        pairs=tuple(sorted(retained)),
        exclusions=tuple(sorted(exclusions, key=lambda item: item.tuple)),
        source=source,
        level=level,
        include_same_stop=include_same_stop,
        active_service_only=active_service_only,
        connectivity_policy=connectivity_policy,
        physical_stop_mapping=MappingProxyType(dict(mapping)),
        generator_fingerprint=generator_fingerprint,
    )


def _period_tuple(period: object) -> tuple[str, int, int]:
    if isinstance(period, (tuple, list)) and len(period) == 3:
        period_id, start, end = period
        start_i, end_i = int(start), int(end)
        if end_i <= start_i:
            raise ValueError(f"time period {period_id!r} must have end > start.")
        return _text(period_id, "time period id"), start_i, end_i
    period_id = getattr(period, "period_id", getattr(period, "bin_id", None))
    start = getattr(period, "start_seconds", None)
    end = getattr(period, "end_seconds", None)
    if start is None:
        start = getattr(getattr(period, "start", None), "seconds_from_midnight", None)
    if end is None:
        end = getattr(getattr(period, "end", None), "seconds_from_midnight", None)
    if period_id is None or start is None or end is None:
        raise ValueError("time periods must expose id/bin_id and start/end seconds.")
    start_i, end_i = int(start), int(end)
    if end_i <= start_i:
        raise ValueError(f"time period {period_id!r} must have end > start.")
    return _text(period_id, "time period id"), start_i, end_i


def _timetable_feasible(
    scenario: Scenario,
    pair: CandidateODPair,
    period: tuple[str, int, int],
    *,
    mapping: Mapping[str, str],
    maximum_transfers: int,
    maximum_initial_wait_seconds: int,
    maximum_journey_seconds: int,
    maximum_waiting_seconds: int,
) -> bool:
    """Small schedule search used by expansion for an explicit feasibility rule."""
    sequences = _trip_sequences(scenario, mapping)
    if not sequences:
        return False
    origin = mapping[pair.origin_stop_id]
    destination = mapping[pair.destination_stop_id]
    start, end = period[1], period[2]
    boardings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for trip_id, sequence in sequences.items():
        for index, stop_time in enumerate(sequence):
            if mapping[str(stop_time.stop_id)] == origin:
                departure = int(stop_time.departure.seconds_from_midnight)
                if start <= departure < end and departure - start <= maximum_initial_wait_seconds:
                    boardings[trip_id].append((index, departure))
    queue: deque[tuple[str, int, int, int]] = deque(
        (trip_id, index, departure, 0)
        for trip_id, entries in boardings.items()
        for index, departure in entries
    )
    visited: set[tuple[str, int, int]] = set()
    while queue:
        trip_id, board_index, first_departure, transfers = queue.popleft()
        state_key = (trip_id, board_index, transfers)
        if state_key in visited:
            continue
        visited.add(state_key)
        sequence = sequences[trip_id]
        for alight_index in range(board_index + 1, len(sequence)):
            alight = sequence[alight_index]
            arrival = int(alight.arrival.seconds_from_midnight)
            if arrival - first_departure > maximum_journey_seconds:
                break
            stop = mapping[str(alight.stop_id)]
            if stop == destination:
                return True
            if transfers >= maximum_transfers:
                continue
            for next_trip_id, next_sequence in sequences.items():
                for next_index, next_stop_time in enumerate(next_sequence):
                    if mapping[str(next_stop_time.stop_id)] != stop:
                        continue
                    departure = int(next_stop_time.departure.seconds_from_midnight)
                    if departure < arrival:
                        continue
                    if departure - arrival > maximum_waiting_seconds:
                        break
                    if next_trip_id == trip_id:
                        continue
                    queue.append((next_trip_id, next_index, first_departure, transfers + 1))
                    break
    return False


def expand_candidate_od_time_cells(
    universe: CandidateODUniverse,
    time_periods: Sequence[object],
    *,
    scenario: Scenario | None = None,
    maximum_transfers: int = 2,
    maximum_initial_wait_seconds: int = 3600,
    maximum_journey_seconds: int = 7200,
    maximum_waiting_seconds: int = 3600,
    timetable_policy: TimetablePolicy = "required",
    timetable_feasibility: Callable[[CandidateODPair, tuple[str, int, int]], bool] | None = None,
) -> ODTimeExpansion:
    """Expand a pair universe across approved bins and audit each exclusion."""
    if maximum_transfers < 0 or maximum_initial_wait_seconds < 0 or maximum_journey_seconds <= 0 or maximum_waiting_seconds < 0:
        raise ValueError("feasibility limits must be non-negative (journey time positive).")
    if timetable_policy not in {"required", "defer", "none"}:
        raise ValueError("unsupported timetable_policy.")
    bins = tuple(_period_tuple(period) for period in time_periods)
    if not bins:
        raise ValueError("at least one approved time period is required.")
    if len({item[0] for item in bins}) != len(bins):
        raise ValueError("approved time period identifiers must be unique.")
    if bins != tuple(sorted(bins, key=lambda item: (item[1], item[2], item[0]))):
        raise ValueError("approved time periods must be sorted by start/end/id.")
    if any(left[2] > right[1] for left, right in zip(bins, bins[1:], strict=False)):
        raise ValueError("approved time periods must not overlap.")
    mapping = universe.physical_stop_mapping
    departures, arrivals = (set(), set()) if scenario is None else _service_activity(scenario, mapping)
    reachable = {} if scenario is None else _directed_reachability(scenario, mapping)
    cells: list[CandidateODTimeCell] = []
    exclusions: list[ODTimeExclusion] = []
    for pair in universe.pairs:
        pair_reason: str | None = None
        if pair.origin_stop_id == pair.destination_stop_id and not universe.include_same_stop:
            pair_reason = "same_node"
        elif universe.active_service_only and scenario is not None and pair.origin_stop_id not in departures:
            pair_reason = "inactive_origin"
        elif universe.active_service_only and scenario is not None and pair.destination_stop_id not in arrivals:
            pair_reason = "inactive_destination"
        elif universe.connectivity_policy == "directed_reachable" and scenario is not None and pair.destination_stop_id not in reachable.get(pair.origin_stop_id, set()):
            pair_reason = "static_unreachable"
        for period in bins:
            if pair_reason is not None:
                exclusions.append(ODTimeExclusion(*pair.tuple, period[0], pair_reason))
                continue
            feasible: bool | None
            if timetable_feasibility is not None:
                feasible = bool(timetable_feasibility(pair, period))
            elif timetable_policy == "none":
                feasible = True
            elif timetable_policy == "defer":
                feasible = None
            elif scenario is None or scenario.timetable is None:
                feasible = False
            else:
                feasible = _timetable_feasible(
                    scenario,
                    pair,
                    period,
                    mapping=mapping,
                    maximum_transfers=maximum_transfers,
                    maximum_initial_wait_seconds=maximum_initial_wait_seconds,
                    maximum_journey_seconds=maximum_journey_seconds,
                    maximum_waiting_seconds=maximum_waiting_seconds,
                )
            if feasible is False:
                exclusions.append(
                    ODTimeExclusion(*pair.tuple, period[0], "timetable_infeasible")
                )
            else:
                cells.append(CandidateODTimeCell(*pair.tuple, period[0]))
    policies = MappingProxyType(
        {
            "maximum_transfers": maximum_transfers,
            "maximum_initial_wait_seconds": maximum_initial_wait_seconds,
            "maximum_journey_seconds": maximum_journey_seconds,
            "maximum_waiting_seconds": maximum_waiting_seconds,
            "timetable_policy": timetable_policy,
        }
    )
    fingerprint = _sha256_payload(
        {
            "universe_fingerprint": universe.fingerprint,
            "time_bins": [list(item) for item in bins],
            "cells": [list(item.tuple) for item in sorted(cells)],
            "exclusions": [
                [item.origin_stop_id, item.destination_stop_id, item.time_bin_id, item.reason]
                for item in sorted(exclusions, key=lambda value: (value.origin_stop_id, value.destination_stop_id, value.time_bin_id, value.reason))
            ],
            "policies": dict(policies),
        }
    )
    return ODTimeExpansion(
        universe_fingerprint=universe.fingerprint,
        cells=tuple(sorted(cells)),
        exclusions=tuple(
            sorted(
                exclusions,
                key=lambda item: (
                    item.origin_stop_id,
                    item.destination_stop_id,
                    item.time_bin_id,
                    item.reason,
                ),
            )
        ),
        time_bins=bins,
        policies=policies,
        fingerprint=fingerprint,
    )


def _read_pair_priors(path: str | Path) -> dict[tuple[str, str], float]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"prior demand file does not exist: {source}")
    result: dict[tuple[str, str], float] = {}
    with source.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("prior demand file has no header.")
        required = {"origin_stop_id", "destination_stop_id", "prior_value"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"prior demand file is missing required columns: {missing}")
        extra = sorted(set(reader.fieldnames) - required)
        if extra:
            raise ValueError(
                "prior demand file must be pair-level and independent of time bins; "
                f"unexpected columns: {extra}."
            )
        for row_number, row in enumerate(reader, start=2):
            key = (
                _text(row.get("origin_stop_id"), f"prior row {row_number} origin_stop_id"),
                _text(row.get("destination_stop_id"), f"prior row {row_number} destination_stop_id"),
            )
            if key in result:
                raise ValueError(f"prior demand file contains duplicate pair {key!r}.")
            try:
                value = float(row.get("prior_value", ""))
            except (TypeError, ValueError) as error:
                raise ValueError(f"prior row {row_number} prior_value must be numeric.") from error
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"prior row {row_number} prior_value must be finite and non-negative.")
            result[key] = value
    return result


def generate_prior_demand(
    expansion: ODTimeExpansion,
    *,
    source: str = "all_ones",
    value: float = 1.0,
    semantics: str = "neutral_seed",
    prior_file: str | Path | None = None,
) -> PriorGenerationResult:
    """Generate prior values only after OD--time expansion.

    The default ``all_ones`` prior is a neutral numerical seed, never an
    observation or a production/attractiveness estimate.
    """
    allowed_sources = {
        "all_ones",
        "external_file",
        "distance_decay",
        "travel_time_decay",
        "gravity_seed",
        "destination_attractiveness_seed",
    }
    if source not in allowed_sources:
        raise ValueError(f"unsupported prior source {source!r}.")
    if not isinstance(semantics, str) or not semantics.strip():
        raise ValueError("prior semantics must be a non-empty string.")
    if source == "all_ones":
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("all_ones value must be finite and positive.")
        pair_values: dict[tuple[str, str], float] = {}
        values = {cell: float(value) for cell in expansion.cells}
        parameters: dict[str, object] = {"value": float(value), "expansion": "one_per_retained_od_time_cell"}
    elif source == "external_file":
        if prior_file is None:
            raise ValueError("source='external_file' requires prior_file.")
        pair_values = _read_pair_priors(prior_file)
        required_pairs = {(cell.origin_stop_id, cell.destination_stop_id) for cell in expansion.cells}
        missing = sorted(required_pairs - set(pair_values))
        extra = sorted(set(pair_values) - required_pairs)
        if missing or extra:
            raise ValueError(f"external prior pairs do not match retained cells; missing={missing}, extra={extra}.")
        values = {cell: pair_values[(cell.origin_stop_id, cell.destination_stop_id)] for cell in expansion.cells}
        parameters = {"prior_file": str(Path(prior_file).expanduser().resolve()), "expansion": "pair_value_repeated_over_retained_bins"}
    else:
        raise NotImplementedError(
            f"prior generator {source!r} is reserved for a future explicit implementation."
        )
    generator_fingerprint = _sha256_payload({"source": source, "semantics": semantics, "parameters": parameters})
    fingerprint = _sha256_payload({"generator_fingerprint": generator_fingerprint, "expansion_fingerprint": expansion.fingerprint, "values": [[*cell.tuple, values[cell]] for cell in sorted(values)]})
    return PriorGenerationResult(
        values=MappingProxyType(dict(sorted(values.items()))),
        source=source,
        semantics=semantics,
        parameters=MappingProxyType(parameters),
        generator_fingerprint=generator_fingerprint,
        fingerprint=fingerprint,
    )
