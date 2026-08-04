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

    def __post_init__(self) -> None:
        """Allow start/end to be provided as "HH:MM:SS" strings."""
        if isinstance(self.start, str):
            object.__setattr__(self, "start", TimeOfDay.parse(self.start))
        if isinstance(self.end, str):
            object.__setattr__(self, "end", TimeOfDay.parse(self.end))

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


    def midpoint_s(self) -> int:
        """Midpoint of the time bin in seconds-from-midnight.

        :return: Midpoint time (integer seconds-from-midnight).
        """
        a = int(self.start.seconds_from_midnight)
        b = int(self.end.seconds_from_midnight)
        return (a + b) // 2

    def midpoint_min(self) -> float:
        """Midpoint of the time bin in minutes.

        :return: Midpoint time in minutes.
        """
        return float(self.midpoint_s()) / 60.0

    def desired_departure_s(self) -> int:
        """Canonical desired departure time for this bin (baseline specification).

        In the baseline model, the desired departure time is defined as the
        midpoint of the time bin.

        :return: Desired departure time (integer seconds-from-midnight).
        """
        return self.midpoint_s()

    def desired_departure_min(self) -> float:
        """Canonical desired departure time for this bin (baseline specification), in minutes.

        :return: Desired departure time in minutes.
        """
        return float(self.desired_departure_s()) / 60.0
