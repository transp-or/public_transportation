"""Deterministic normalization of scenario stops into physical stop places."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from public_transportation.domain import Scenario

from .artifacts import canonical_json
from .progress import ReducedODProgress, ReducedODProgressEmitter


PhysicalStopMappingPolicy = Literal[
    "identity", "authoritative", "reviewed_generated"
]


def _text(value: str, name: str) -> str:
    parsed = str(value).strip()
    if not parsed:
        raise ValueError(f"{name} must be a non-empty identifier.")
    return parsed


def _immutable_int32(values: list[int]) -> np.ndarray:
    result = np.asarray(values, dtype=np.int32)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class PhysicalStopPlace:
    """One passenger-facing place and its canonical scenario-stop members."""

    physical_stop_id: str
    member_stop_ids: tuple[str, ...]
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        _text(self.physical_stop_id, "physical_stop_id")
        if not self.member_stop_ids:
            raise ValueError("member_stop_ids must not be empty.")
        if self.member_stop_ids != tuple(sorted(self.member_stop_ids)):
            raise ValueError("member_stop_ids must be sorted.")
        if len(set(self.member_stop_ids)) != len(self.member_stop_ids):
            raise ValueError("member_stop_ids must be unique.")
        if not np.isfinite(self.latitude) or not -90.0 <= self.latitude <= 90.0:
            raise ValueError("latitude must be finite and in [-90, 90].")
        if not np.isfinite(self.longitude) or not -180.0 <= self.longitude <= 180.0:
            raise ValueError("longitude must be finite and in [-180, 180].")


@dataclass(frozen=True, slots=True)
class PhysicalStopIndex:
    """Canonical physical-stop mapping used by later timetable preprocessing."""

    mapping_policy: PhysicalStopMappingPolicy
    scenario_stop_ids: tuple[str, ...]
    physical_stop_ids_by_scenario_stop: tuple[str, ...]
    stop_to_physical_index: np.ndarray
    places: tuple[PhysicalStopPlace, ...]

    def __post_init__(self) -> None:
        if self.mapping_policy not in {
            "identity",
            "authoritative",
            "reviewed_generated",
        }:
            raise ValueError("mapping_policy is unsupported.")
        if self.scenario_stop_ids != tuple(sorted(self.scenario_stop_ids)):
            raise ValueError("scenario_stop_ids must be sorted.")
        if len(set(self.scenario_stop_ids)) != len(self.scenario_stop_ids):
            raise ValueError("scenario_stop_ids must be unique.")
        if len(self.physical_stop_ids_by_scenario_stop) != len(
            self.scenario_stop_ids
        ):
            raise ValueError("physical-stop mapping must align with scenario stops.")
        place_ids = tuple(place.physical_stop_id for place in self.places)
        if place_ids != tuple(sorted(place_ids)) or len(set(place_ids)) != len(
            place_ids
        ):
            raise ValueError("places must have unique sorted identifiers.")
        indices = np.asarray(self.stop_to_physical_index)
        if indices.shape != (len(self.scenario_stop_ids),):
            raise ValueError("stop_to_physical_index has an invalid shape.")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("stop_to_physical_index must contain integers.")
        immutable = np.array(indices, dtype=np.int32, copy=True, order="C")
        if immutable.size and (
            np.any(immutable < 0) or np.any(immutable >= len(self.places))
        ):
            raise ValueError("stop_to_physical_index contains invalid indices.")
        expected = tuple(place_ids[int(index)] for index in immutable)
        if expected != self.physical_stop_ids_by_scenario_stop:
            raise ValueError("physical-stop identifiers and indices disagree.")
        immutable.setflags(write=False)
        object.__setattr__(self, "stop_to_physical_index", immutable)

    @property
    def fingerprint_payload_json(self) -> str:
        return canonical_json(
            {
                "mapping_policy": self.mapping_policy,
                "places": [
                    {
                        "latitude": place.latitude,
                        "longitude": place.longitude,
                        "member_stop_ids": list(place.member_stop_ids),
                        "physical_stop_id": place.physical_stop_id,
                    }
                    for place in self.places
                ],
                "scenario_stop_ids": list(self.scenario_stop_ids),
            }
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            self.fingerprint_payload_json.encode("utf-8")
        ).hexdigest()


def build_physical_stop_index(
    scenario: Scenario,
    *,
    mapping: Mapping[str, str] | None = None,
    mapping_policy: PhysicalStopMappingPolicy | None = None,
    progress: ReducedODProgress | None = None,
) -> PhysicalStopIndex:
    """Build an exact, deterministic mapping for every scenario stop.

    When the mapping is omitted, every scenario stop is its own physical place.
    A supplied mapping must contain every scenario stop and no unknown stop.
    """
    stops_by_id: dict[str, object] = {}
    stop_progress = ReducedODProgressEmitter(
        progress, phase="physical_stop_index", total=len(scenario.stops)
    )
    stop_progress.start()
    for stop_position, stop in enumerate(scenario.stops, start=1):
        stop_id = _text(stop.stop_id, "stop_id")
        if stop_id in stops_by_id:
            raise ValueError(f"duplicate scenario stop identifier {stop_id!r}.")
        stops_by_id[stop_id] = stop
        stop_progress.update(stop_position, current_unit=stop_id)
    scenario_stop_ids = tuple(sorted(stops_by_id))

    if mapping is None:
        if mapping_policy not in {None, "identity"}:
            raise ValueError(
                "mapping_policy must be 'identity' when mapping is omitted."
            )
        policy: PhysicalStopMappingPolicy = "identity"
        normalized = {stop_id: stop_id for stop_id in scenario_stop_ids}
    else:
        if mapping_policy not in {"authoritative", "reviewed_generated"}:
            raise ValueError(
                "a supplied mapping requires mapping_policy 'authoritative' "
                "or 'reviewed_generated'."
            )
        policy = mapping_policy
        normalized = {
            _text(stop_id, "mapping stop_id"): _text(
                physical_id, "physical_stop_id"
            )
            for stop_id, physical_id in mapping.items()
        }
        missing = sorted(set(scenario_stop_ids) - set(normalized))
        unknown = sorted(set(normalized) - set(scenario_stop_ids))
        if missing or unknown:
            raise ValueError(
                "physical-stop mapping must cover exactly the scenario stops; "
                f"missing={missing}, unknown={unknown}."
            )

    members: dict[str, list[str]] = {}
    for stop_id in scenario_stop_ids:
        members.setdefault(normalized[stop_id], []).append(stop_id)

    places: list[PhysicalStopPlace] = []
    for physical_id in sorted(members):
        member_ids = tuple(sorted(members[physical_id]))
        member_stops = [stops_by_id[stop_id] for stop_id in member_ids]
        places.append(
            PhysicalStopPlace(
                physical_stop_id=physical_id,
                member_stop_ids=member_ids,
                latitude=float(
                    np.mean([float(getattr(stop, "lat")) for stop in member_stops])
                ),
                longitude=float(
                    np.mean([float(getattr(stop, "lon")) for stop in member_stops])
                ),
            )
        )
    place_index = {
        place.physical_stop_id: index for index, place in enumerate(places)
    }
    physical_ids = tuple(normalized[stop_id] for stop_id in scenario_stop_ids)
    indices = _immutable_int32(
        [place_index[physical_id] for physical_id in physical_ids]
    )
    return PhysicalStopIndex(
        mapping_policy=policy,
        scenario_stop_ids=scenario_stop_ids,
        physical_stop_ids_by_scenario_stop=physical_ids,
        stop_to_physical_index=indices,
        places=tuple(places),
    )
