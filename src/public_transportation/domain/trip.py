from __future__ import annotations

from dataclasses import dataclass

from .issues import Issue, Severity, ValidationReport


@dataclass(slots=True)
class Trip:
    """
    One scheduled vehicle run (a course).

    This is a lightweight domain object, inspired by GTFS `trips.txt`.
    It can later be extended with route/shape/calendar concepts.

    :param trip_id: Unique trip identifier.
    :param line_id: Optional line/route identifier (e.g., "TPG_12").
    :param capacity: Optional passenger capacity for this trip (vehicle run). Must be non-negative when provided.
    :param service_id: Optional service identifier (calendar applicability).
    :param headsign: Optional passenger-facing destination label.
    :param direction_id: Optional direction indicator (0/1 or similar).
    """
    trip_id: str
    line_id: str | None = None
    capacity: float | None = None
    service_id: str | None = None
    headsign: str | None = None
    direction_id: int | None = None

    def validate(self) -> ValidationReport:
        """
        Validate local fields.

        :return: ValidationReport.
        """
        rep = ValidationReport(issues=[])
        if not self.trip_id:
            rep.add(Issue(
                severity=Severity.ERROR,
                code="TRIP_ID_EMPTY",
                message="Trip id is empty.",
                location="timetable.trips[].trip_id",
                suggestion="Provide a non-empty trip_id.",
            ))
        if self.capacity is not None and self.capacity < 0:
            rep.add(Issue(
                severity=Severity.ERROR,
                code="TRIP_CAPACITY_NEGATIVE",
                message="capacity must be non-negative when provided.",
                location=f"timetable.trips[{self.trip_id}].capacity",
                context={"capacity": self.capacity},
            ))
        if self.capacity is not None and self.capacity == 0:
            rep.add(Issue(
                severity=Severity.WARNING,
                code="TRIP_CAPACITY_ZERO",
                message="capacity is zero; this trip will effectively have no usable capacity in capacity-aware assignment.",
                location=f"timetable.trips[{self.trip_id}].capacity",
                suggestion="Set capacity to a positive value, or leave it empty to use defaults later.",
            ))
        if self.direction_id is not None and self.direction_id < 0:
            rep.add(Issue(
                severity=Severity.ERROR,
                code="TRIP_DIRECTION_INVALID",
                message="direction_id must be non-negative when provided.",
                location=f"timetable.trips[{self.trip_id}].direction_id",
                context={"direction_id": self.direction_id},
            ))
        return rep
