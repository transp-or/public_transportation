"""Explicit platform-to-station membership and hub-distance inputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class HubPlatformDistance:
    """Distance from a station-hub reference point to one platform, in metres."""

    station_id: str
    platform_id: str
    distance_m: float

    def __post_init__(self) -> None:
        station = str(self.station_id).strip()
        platform = str(self.platform_id).strip()
        distance = float(self.distance_m)
        if not station or not platform:
            raise ValueError("station_id and platform_id must be non-empty.")
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("distance_m must be finite and non-negative.")
        object.__setattr__(self, "station_id", station)
        object.__setattr__(self, "platform_id", platform)
        object.__setattr__(self, "distance_m", distance)


@dataclass(frozen=True, slots=True)
class TransferGeometry:
    """Generic transfer geometry; timetable events remain platform-based.

    ``platform_to_station`` maps each scheduled platform to exactly one hub.
    ``hub_platform_distances_m`` is keyed by ``(station_id, platform_id)``.
    It may also be supplied as a sequence of :class:`HubPlatformDistance`
    records, which allows duplicate CSV rows to be detected during validation.
    """

    platform_to_station: Mapping[str, str]
    hub_platform_distances_m: (
        Mapping[tuple[str, str], float] | Sequence[HubPlatformDistance]
    )

    def __post_init__(self) -> None:
        membership: dict[str, str] = {}
        for platform_id, station_id in self.platform_to_station.items():
            platform = str(platform_id).strip()
            station = str(station_id).strip()
            if not platform or not station:
                raise ValueError("platform_to_station identifiers must be non-empty.")
            if platform in membership:
                raise ValueError(
                    f"Duplicate platform_id in station mapping: {platform!r}."
                )
            membership[platform] = station

        distances: dict[tuple[str, str], float] = {}
        source = self.hub_platform_distances_m
        if isinstance(source, Mapping):
            records: Iterable[tuple[tuple[str, str], float]] = source.items()
            for key, distance_value in records:
                if not isinstance(key, tuple) or len(key) != 2:
                    raise ValueError(
                        "Hub distance keys must be (station_id, platform_id)."
                    )
                station_id, platform_id = key
                record = HubPlatformDistance(station_id, platform_id, distance_value)
                distance_key = (record.station_id, record.platform_id)
                if distance_key in distances:
                    raise ValueError(f"Duplicate hub distance key: {distance_key!r}.")
                distances[distance_key] = record.distance_m
        else:
            for record in source:
                if not isinstance(record, HubPlatformDistance):
                    try:
                        record = HubPlatformDistance(*record)  # type: ignore[arg-type]
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            "Hub distance records must contain station_id, platform_id, and distance_m."
                        ) from error
                distance_key = (record.station_id, record.platform_id)
                if distance_key in distances:
                    raise ValueError(f"Duplicate hub distance key: {distance_key!r}.")
                distances[distance_key] = record.distance_m

        for (station_id, platform_id), _distance in distances.items():
            if membership.get(platform_id) != station_id:
                raise ValueError(
                    "Hub distance station_id does not match platform_to_station for "
                    f"platform {platform_id!r}: {station_id!r}."
                )

        object.__setattr__(
            self,
            "platform_to_station",
            MappingProxyType(dict(sorted(membership.items()))),
        )
        object.__setattr__(
            self,
            "hub_platform_distances_m",
            MappingProxyType(dict(sorted(distances.items()))),
        )

    def validate_scenario(
        self,
        *,
        scenario_platform_ids: Iterable[str],
        event_platform_ids: Iterable[str],
    ) -> None:
        """Validate platform references without rewriting schedule or OD IDs."""
        known = {str(platform) for platform in scenario_platform_ids}
        events = {str(platform) for platform in event_platform_ids}
        unknown = sorted(set(self.platform_to_station) - known)
        missing = sorted(events - set(self.platform_to_station))
        if unknown or missing:
            raise ValueError(
                "Transfer geometry must reference known platforms and cover every "
                f"scheduled platform; unknown={unknown[:10]}, missing={missing[:10]}."
            )

    def station_for_platform(self, platform_id: str) -> str:
        """Return the explicit parent station for a scheduled platform."""
        try:
            return self.platform_to_station[str(platform_id)]
        except KeyError as error:
            raise ValueError(
                f"Scheduled platform {platform_id!r} has no station mapping."
            ) from error

    def distance_m(self, station_id: str, platform_id: str) -> float:
        """Return a validated hub-to-platform distance, raising if absent."""
        key = (str(station_id), str(platform_id))
        try:
            return float(self.hub_platform_distances_m[key])
        except KeyError as error:
            raise ValueError(
                "Missing hub distance for cross-platform transfer: "
                f"station_id={key[0]!r}, platform_id={key[1]!r}."
            ) from error

    def fingerprint_payload(self) -> dict[str, object]:
        """Return deterministic, JSON-compatible transfer input metadata."""
        return {
            "platform_to_station": [
                [platform, station]
                for platform, station in self.platform_to_station.items()
            ],
            "hub_platform_distances_m": [
                [station, platform, distance]
                for (
                    station,
                    platform,
                ), distance in self.hub_platform_distances_m.items()
            ],
        }
