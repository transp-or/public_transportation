from __future__ import annotations

from dataclasses import dataclass

from .issues import Issue, Severity, ValidationReport
from .time_of_day import TimeOfDay


@dataclass(slots=True, frozen=True)
class StopTime:
    """
    One stop time record for a trip.

    Inspired by GTFS `stop_times.txt`.

    :param trip_id: Trip identifier.
    :param stop_id: Stop identifier.
    :param sequence: Order of the stop within the trip (starting at 1 typically).
    :param arrival: Arrival time at the stop.
    :param departure: Departure time at the stop (>= arrival typically).
    """
    trip_id: str
    stop_id: str
    sequence: int
    arrival: TimeOfDay
    departure: TimeOfDay

    def validate(self) -> ValidationReport:
        """
        Validate local fields. Cross-references are validated by Timetable/Scenario.

        :return: ValidationReport.
        """
        rep = ValidationReport(issues=[])

        if not self.trip_id:
            rep.add(Issue(
                severity=Severity.ERROR,
                code="STOPTIME_TRIP_EMPTY",
                message="trip_id is empty.",
                location="timetable.stop_times[].trip_id",
            ))

        if not self.stop_id:
            rep.add(Issue(
                severity=Severity.ERROR,
                code="STOPTIME_STOP_EMPTY",
                message="stop_id is empty.",
                location=f"timetable.stop_times[{self.trip_id},{self.sequence}].stop_id",
            ))

        if self.sequence <= 0:
            rep.add(Issue(
                severity=Severity.ERROR,
                code="STOPTIME_SEQUENCE_NONPOSITIVE",
                message="sequence must be a positive integer.",
                location=f"timetable.stop_times[{self.trip_id},{self.sequence}].sequence",
                context={"sequence": self.sequence},
            ))

        if self.departure.seconds_from_midnight < self.arrival.seconds_from_midnight:
            rep.add(Issue(
                severity=Severity.ERROR,
                code="STOPTIME_DEPART_BEFORE_ARRIVE",
                message="departure time is before arrival time.",
                location=f"timetable.stop_times[{self.trip_id},{self.sequence}]",
                context={
                    "arrival_s": self.arrival.seconds_from_midnight,
                    "departure_s": self.departure.seconds_from_midnight,
                },
                suggestion="Ensure departure >= arrival (or fix the time encoding).",
            ))

        return rep