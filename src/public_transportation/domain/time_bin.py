from __future__ import annotations

from dataclasses import dataclass

from .issues import Issue, Severity, ValidationReport
from .time_of_day import TimeOfDay


@dataclass(slots=True)
class TimeBin:
    """
    Departure time interval.

    :param bin_id: Unique bin identifier.
    :param start: Start time.
    :param end: End time. Must be strictly greater than start.
    """
    bin_id: str
    start: TimeOfDay
    end: TimeOfDay

    def validate(self) -> ValidationReport:
        """
        Validate time bin.

        :return: ValidationReport with issues.
        """
        rep = ValidationReport(issues=[])

        if not self.bin_id:
            rep.add(Issue(
                severity=Severity.ERROR,
                code="TIMEBIN_ID_EMPTY",
                message="Time bin id is empty.",
                location="time_bins[].bin_id",
            ))

        if self.end.seconds_from_midnight <= self.start.seconds_from_midnight:
            rep.add(Issue(
                severity=Severity.ERROR,
                code="TIMEBIN_ORDER",
                message="Time bin end must be strictly after start.",
                location=f"time_bins[{self.bin_id}]",
                suggestion="Ensure end > start in seconds_from_midnight.",
                context={
                    "start": self.start.seconds_from_midnight,
                    "end": self.end.seconds_from_midnight,
                },
            ))

        return rep