"""Deterministic timetable service-id classes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from public_transportation.domain import Scenario

from .artifacts import canonical_json
from .progress import ReducedODProgress, ReducedODProgressEmitter


DEFAULT_SERVICE_ID = "__default_service__"


@dataclass(frozen=True, slots=True)
class ServicePeriod:
    """Trips sharing one declared timetable service identifier."""

    service_period_id: str
    source_service_id: str | None
    trip_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.service_period_id:
            raise ValueError("service_period_id must be non-empty.")
        if not self.trip_ids or self.trip_ids != tuple(sorted(self.trip_ids)):
            raise ValueError("trip_ids must be non-empty and sorted.")
        if len(set(self.trip_ids)) != len(self.trip_ids):
            raise ValueError("trip_ids must be unique.")


@dataclass(frozen=True, slots=True)
class ServicePeriodIndex:
    """Canonical service classes plus one class index per sorted trip."""

    trip_ids: tuple[str, ...]
    trip_to_service_period_index: np.ndarray
    service_periods: tuple[ServicePeriod, ...]

    def __post_init__(self) -> None:
        if self.trip_ids != tuple(sorted(self.trip_ids)):
            raise ValueError("trip_ids must be sorted.")
        if len(set(self.trip_ids)) != len(self.trip_ids):
            raise ValueError("trip_ids must be unique.")
        indices = np.asarray(self.trip_to_service_period_index)
        if indices.shape != (len(self.trip_ids),):
            raise ValueError("trip_to_service_period_index has an invalid shape.")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("trip_to_service_period_index must contain integers.")
        immutable = np.array(indices, dtype=np.int32, copy=True, order="C")
        if immutable.size and (
            np.any(immutable < 0)
            or np.any(immutable >= len(self.service_periods))
        ):
            raise ValueError("trip_to_service_period_index contains invalid indices.")
        identifiers = tuple(
            period.service_period_id for period in self.service_periods
        )
        if identifiers != tuple(sorted(identifiers)):
            raise ValueError("service_periods must be sorted by identifier.")
        covered = sorted(
            trip_id
            for service_period in self.service_periods
            for trip_id in service_period.trip_ids
        )
        if covered != list(self.trip_ids):
            raise ValueError("every trip must occur in exactly one service period.")
        immutable.setflags(write=False)
        object.__setattr__(self, "trip_to_service_period_index", immutable)

    @property
    def fingerprint_payload_json(self) -> str:
        return canonical_json(
            {
                "service_periods": [
                    {
                        "service_period_id": period.service_period_id,
                        "source_service_id": period.source_service_id,
                        "trip_ids": list(period.trip_ids),
                    }
                    for period in self.service_periods
                ],
                "trip_ids": list(self.trip_ids),
            }
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            self.fingerprint_payload_json.encode("utf-8")
        ).hexdigest()


def build_service_period_index(
    scenario: Scenario, *, progress: ReducedODProgress | None = None
) -> ServicePeriodIndex:
    """Group every trip by service_id without inferring calendar semantics."""
    if scenario.timetable is None:
        raise ValueError("scenario.timetable is required.")
    trips_by_id: dict[str, object] = {}
    grouped: dict[str, list[str]] = {}
    source_by_normalized: dict[str, str | None] = {}
    trip_progress = ReducedODProgressEmitter(
        progress,
        phase="service_period_index",
        total=len(scenario.timetable.trips),
    )
    trip_progress.start()
    for trip_position, trip in enumerate(scenario.timetable.trips, start=1):
        trip_id = str(trip.trip_id)
        if trip_id in trips_by_id:
            raise ValueError(f"duplicate trip identifier {trip_id!r}.")
        trips_by_id[trip_id] = trip
        raw = getattr(trip, "service_id")
        source = None if raw is None or not str(raw).strip() else str(raw).strip()
        normalized = DEFAULT_SERVICE_ID if source is None else source
        if (
            normalized == DEFAULT_SERVICE_ID
            and source is not None
        ):
            raise ValueError(
                f"service_id {DEFAULT_SERVICE_ID!r} is reserved for missing values."
            )
        grouped.setdefault(normalized, []).append(trip_id)
        source_by_normalized[normalized] = source
        trip_progress.update(trip_position, current_unit=trip_id)

    service_periods = tuple(
        ServicePeriod(
            service_period_id=service_id,
            source_service_id=source_by_normalized[service_id],
            trip_ids=tuple(sorted(grouped[service_id])),
        )
        for service_id in sorted(grouped)
    )
    period_by_trip = {
        trip_id: index
        for index, service_period in enumerate(service_periods)
        for trip_id in service_period.trip_ids
    }
    trip_ids = tuple(sorted(trips_by_id))
    indices = np.asarray(
        [period_by_trip[trip_id] for trip_id in trip_ids], dtype=np.int32
    )
    indices.setflags(write=False)
    return ServicePeriodIndex(
        trip_ids=trip_ids,
        trip_to_service_period_index=indices,
        service_periods=service_periods,
    )
