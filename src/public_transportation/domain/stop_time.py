from __future__ import annotations

from dataclasses import dataclass, field

from .issues import Issue, Severity, ValidationReport
from .time_of_day import TimeOfDay


@dataclass(slots=True, frozen=True)
class StopTime:
    """
    One stop time record for a trip.

    Inspired by GTFS `stop_times.txt`.

    :param trip_id: Trip identifier.
    :param stop_id: Stop identifier.
    :param sequence: Order of the stop within the trip (starting at 1 typically). Accepts int or str, parsed to integer.
    :param arrival: Arrival time at the stop. Accepts TimeOfDay, seconds-from-midnight int, or string "HH:MM"/"HH:MM:SS".
    :param departure: Departure time at the stop (> arrival). Accepts TimeOfDay, seconds-from-midnight int, or string "HH:MM"/"HH:MM:SS".
    """
    trip_id: str
    stop_id: str
    sequence: int | str
    arrival: TimeOfDay | int | str
    departure: TimeOfDay | int | str

    sequence_raw: object = field(init=False, repr=False)
    arrival_raw: object = field(init=False, repr=False)
    departure_raw: object = field(init=False, repr=False)

    sequence_parse_error: str | None = field(init=False, default=None, repr=False)
    arrival_parse_error: str | None = field(init=False, default=None, repr=False)
    departure_parse_error: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        # Keep raw inputs for validation/reporting.
        object.__setattr__(self, "sequence_raw", self.sequence)
        object.__setattr__(self, "arrival_raw", self.arrival)
        object.__setattr__(self, "departure_raw", self.departure)

        # Normalize sequence to int when possible; otherwise keep original and store an error.
        seq = self.sequence
        seq_int: int | str
        err: str | None = None

        if isinstance(seq, bool):
            err = "sequence is a bool (not allowed)"
            seq_int = seq  # keep raw for validate()
        elif isinstance(seq, int):
            seq_int = seq
        elif isinstance(seq, str):
            s = seq.strip()
            if not s:
                err = "sequence is an empty string"
                seq_int = seq
            else:
                try:
                    seq_int = int(s)
                except ValueError:
                    err = f"sequence is not an integer: {seq!r}"
                    seq_int = seq
        else:
            err = f"sequence has invalid type: {type(seq).__name__}"
            seq_int = seq  # keep raw for validate()

        object.__setattr__(self, "sequence", seq_int)
        object.__setattr__(self, "sequence_parse_error", err)

        # Normalize arrival/departure to TimeOfDay when possible; otherwise set a safe default and store an error.
        arr, arr_err = self._try_parse_time_of_day(self.arrival, field_name="arrival")
        dep, dep_err = self._try_parse_time_of_day(self.departure, field_name="departure")

        object.__setattr__(self, "arrival", arr)
        object.__setattr__(self, "departure", dep)
        object.__setattr__(self, "arrival_parse_error", arr_err)
        object.__setattr__(self, "departure_parse_error", dep_err)

    @staticmethod
    def _parse_time_of_day_strict(value: TimeOfDay | int | str, field_name: str) -> TimeOfDay:
        """Parse various time encodings into a TimeOfDay.

        Accepted:
        - TimeOfDay instance
        - int: seconds-from-midnight
        - str: "HH:MM" or "HH:MM:SS" (24h)
        """
        if isinstance(value, TimeOfDay):
            return value

        if isinstance(value, int):
            # Assume seconds-from-midnight.
            return TimeOfDay(seconds_from_midnight=value)

        if isinstance(value, str):
            s = value.strip()
            parts = s.split(":")
            if len(parts) == 2:
                hh_s, mm_s = parts
                ss_s = "0"
            elif len(parts) == 3:
                hh_s, mm_s, ss_s = parts
            else:
                raise ValueError(
                    f"Invalid time format for {field_name}: {value!r}. Expected HH:MM or HH:MM:SS."
                )
            try:
                hh = int(hh_s)
                mm = int(mm_s)
                ss = int(ss_s)
            except ValueError as e:
                raise ValueError(
                    f"Invalid time format for {field_name}: {value!r}. Expected HH:MM or HH:MM:SS."
                ) from e

            if not (0 <= hh <= 47):
                # We allow >24h only if the broader codebase supports it; otherwise keep a generous bound.
                raise ValueError(f"Hour out of range for {field_name}: {hh} (from {value!r}).")
            if not (0 <= mm <= 59):
                raise ValueError(f"Minute out of range for {field_name}: {mm} (from {value!r}).")
            if not (0 <= ss <= 59):
                raise ValueError(f"Second out of range for {field_name}: {ss} (from {value!r}).")

            return TimeOfDay(seconds_from_midnight=hh * 3600 + mm * 60 + ss)

        raise TypeError(
            f"Invalid type for {field_name}: {type(value).__name__}. Expected TimeOfDay, int, or str."
        )

    @staticmethod
    def _try_parse_time_of_day(value: TimeOfDay | int | str, field_name: str) -> tuple[TimeOfDay, str | None]:
        """Best-effort parsing for load-time robustness.

        Returns (parsed_time, error_message). When parsing fails, a safe default
        time (00:00:00) is returned so that validation can report the issue.
        """
        try:
            return StopTime._parse_time_of_day_strict(value, field_name=field_name), None
        except (ValueError, TypeError) as e:
            return TimeOfDay(seconds_from_midnight=0), str(e)

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

        if not isinstance(self.sequence, int):
            rep.add(Issue(
                severity=Severity.ERROR,
                code="STOPTIME_SEQUENCE_NOT_INT",
                message="sequence must be an integer.",
                location=f"timetable.stop_times[{self.trip_id},?].sequence",
                context={
                    "sequence": self.sequence,
                    "sequence_raw": self.sequence_raw,
                    "parse_error": self.sequence_parse_error,
                    "type": type(self.sequence).__name__,
                },
                suggestion=(
                    "Ensure the CSV column `sequence` contains integers (1,2,3,...) for each trip. "
                    "If you see stop IDs here, the `stop_id` and `sequence` columns/values may be swapped in stop_times.csv."
                ),
            ))
        elif self.sequence <= 0:
            rep.add(Issue(
                severity=Severity.ERROR,
                code="STOPTIME_SEQUENCE_NONPOSITIVE",
                message="sequence must be a positive integer.",
                location=f"timetable.stop_times[{self.trip_id},{self.sequence}].sequence",
                context={"sequence": self.sequence},
            ))

        if self.arrival_parse_error is not None:
            rep.add(Issue(
                severity=Severity.ERROR,
                code="STOPTIME_ARRIVAL_PARSE",
                message="arrival time could not be parsed.",
                location=f"timetable.stop_times[{self.trip_id},{self.sequence if isinstance(self.sequence, int) else '?'}].arrival",
                context={"arrival_raw": self.arrival_raw, "error": self.arrival_parse_error},
                suggestion="Use HH:MM or HH:MM:SS strings, or seconds-from-midnight integers.",
            ))

        if self.departure_parse_error is not None:
            rep.add(Issue(
                severity=Severity.ERROR,
                code="STOPTIME_DEPARTURE_PARSE",
                message="departure time could not be parsed.",
                location=f"timetable.stop_times[{self.trip_id},{self.sequence if isinstance(self.sequence, int) else '?'}].departure",
                context={"departure_raw": self.departure_raw, "error": self.departure_parse_error},
                suggestion="Use HH:MM or HH:MM:SS strings, or seconds-from-midnight integers.",
            ))

        if self.arrival_parse_error is None and self.departure_parse_error is None:
            if self.departure.seconds_from_midnight <= self.arrival.seconds_from_midnight:
                rep.add(Issue(
                    severity=Severity.ERROR,
                    code="STOPTIME_DEPART_NOT_AFTER_ARRIVE",
                    message="departure time must be strictly after arrival time.",
                    location=f"timetable.stop_times[{self.trip_id},{self.sequence}]",
                    context={
                        "arrival_s": self.arrival.seconds_from_midnight,
                        "departure_s": self.departure.seconds_from_midnight,
                    },
                    suggestion=(
                        "Ensure departure > arrival. If your input has equal times, the assignment builder may "
                        "apply a minimum dwell time via configuration."
                    ),
                ))

        return rep