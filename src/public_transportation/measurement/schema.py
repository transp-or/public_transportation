from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

from public_transportation.domain.time_of_day import TimeOfDay


class MeasurementType(str, Enum):
    """Supported measurement types."""

    LOAD = "load"  # instantaneous in-vehicle load at a stop/time for a trip (or line)
    BOARDING = "boarding"  # boardings at a stop/time for a trip (or line)
    ALIGHTING = "alighting"  # alightings at a stop/time for a trip (or line)


@dataclass(frozen=True, slots=True)
class MeasurementRecord:
    """One observed measurement.

    Identification is scenario-facing (no assignment indices):
      - stop_id is required
      - time is required (HH:MM:SS -> TimeOfDay)
      - trip_id is optional, line_id is optional (at least one must be provided)
      - method_id identifies the data collection method/device/protocol

    Duplicates are NOT allowed (checked at MeasurementTable construction / load).
    """

    method_id: str
    measurement_type: MeasurementType
    stop_id: str
    time: TimeOfDay
    value: float

    trip_id: Optional[str] = None
    line_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.method_id or not self.method_id.strip():
            raise ValueError("method_id must be a non-empty string.")
        if not self.stop_id or not self.stop_id.strip():
            raise ValueError("stop_id must be a non-empty string.")
        if self.trip_id is not None and not str(self.trip_id).strip():
            raise ValueError("trip_id must be None or a non-empty string.")
        if self.line_id is not None and not str(self.line_id).strip():
            raise ValueError("line_id must be None or a non-empty string.")
        if self.trip_id is None and self.line_id is None:
            raise ValueError("At least one of trip_id or line_id must be provided.")
        if not isinstance(self.time, TimeOfDay):
            raise TypeError("time must be a TimeOfDay instance.")
        if not (self.value == self.value):  # NaN check
            raise ValueError("value must not be NaN.")

    def key(self) -> tuple:
        """Uniqueness key (duplicates not allowed)."""
        return (
            self.method_id,
            self.measurement_type.value,
            self.stop_id,
            self.time.seconds_from_midnight,
            self.trip_id,
            self.line_id,
        )


@dataclass(frozen=True, slots=True)
class MeasurementTable:
    """A validated collection of measurement records (no duplicates)."""

    records: tuple[MeasurementRecord, ...]

    def __post_init__(self) -> None:
        seen: set[tuple] = set()
        for r in self.records:
            k = r.key()
            if k in seen:
                raise ValueError(
                    "Duplicate measurement record detected for key=" f"{k}. Duplicates are not allowed."
                )
            seen.add(k)

    @staticmethod
    def from_records(records: Iterable[MeasurementRecord]) -> "MeasurementTable":
        return MeasurementTable(records=tuple(records))