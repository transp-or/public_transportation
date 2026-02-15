from __future__ import annotations

from dataclasses import dataclass

from .issues import Issue, Severity, ValidationReport
from .stop_time import StopTime
from .trip import Trip


@dataclass(slots=True)
class Timetable:
    """
    Container for timetable data.

    This is intentionally minimal: trips + stop_times.
    It is domain-level (file/GUI friendly) and will later be compiled
    into an efficient time-expanded network representation.

    :param trips: List of Trip objects.
    :param stop_times: List of StopTime records.
    """
    trips: list[Trip]
    stop_times: list[StopTime]

    def validate(self, *, known_stop_ids: set[str] | None = None) -> ValidationReport:
        """
        Validate internal consistency of the timetable.

        Checks include:
        - unique trip ids
        - stop_times reference existing trips
        - (optionally) stop_times reference existing stops
        - per trip: sequences unique
        - per trip: times nondecreasing with sequence (arrival and departure)

        :param known_stop_ids: Optional set of stop_ids to validate references.
        :return: ValidationReport.
        """
        rep = ValidationReport(issues=[])

        # Local validation
        for t in self.trips:
            rep.extend(t.validate())
        for st in self.stop_times:
            rep.extend(st.validate())

        trip_ids = [t.trip_id for t in self.trips]
        trip_id_set = set(trip_ids)

        # Unique trip ids
        if len(trip_id_set) != len(trip_ids):
            seen: set[str] = set()
            for tid in trip_ids:
                if tid in seen:
                    rep.add(Issue(
                        severity=Severity.ERROR,
                        code="TRIP_ID_DUPLICATE",
                        message=f"Duplicate trip_id: {tid!r}.",
                        location="timetable.trips",
                        suggestion="Ensure trip_id is unique.",
                    ))
                seen.add(tid)

        # stop_times reference trips and stops
        for k, st in enumerate(self.stop_times):
            loc = f"timetable.stop_times[{k}]"
            if st.trip_id not in trip_id_set:
                rep.add(Issue(
                    severity=Severity.ERROR,
                    code="STOPTIME_TRIP_UNKNOWN",
                    message=f"stop_time references unknown trip_id: {st.trip_id!r}.",
                    location=loc,
                ))
            if known_stop_ids is not None and st.stop_id not in known_stop_ids:
                rep.add(Issue(
                    severity=Severity.ERROR,
                    code="STOPTIME_STOP_UNKNOWN",
                    message=f"stop_time references unknown stop_id: {st.stop_id!r}.",
                    location=loc,
                ))

        # Per-trip sequencing and monotonic time checks
        by_trip: dict[str, list[StopTime]] = {}
        for st in self.stop_times:
            by_trip.setdefault(st.trip_id, []).append(st)

        for tid, sts in by_trip.items():
            # Only validate sequencing if the trip exists (otherwise already reported)
            if tid not in trip_id_set:
                continue

            # Sequence uniqueness
            seqs = [s.sequence for s in sts]
            if len(set(seqs)) != len(seqs):
                rep.add(Issue(
                    severity=Severity.ERROR,
                    code="STOPTIME_SEQUENCE_DUPLICATE",
                    message=f"Trip {tid!r} has duplicate sequence values.",
                    location=f"timetable.stop_times.trip[{tid}]",
                    suggestion="Ensure (trip_id, sequence) is unique.",
                ))

            # Sort by sequence and check nondecreasing times
            sts_sorted = sorted(sts, key=lambda s: s.sequence)
            prev_dep = None
            for s in sts_sorted:
                arr = s.arrival.seconds_from_midnight
                dep = s.departure.seconds_from_midnight
                if prev_dep is not None and arr < prev_dep:
                    rep.add(Issue(
                        severity=Severity.ERROR,
                        code="STOPTIME_TIME_NONMONOTONE",
                        message=f"Trip {tid!r} times are not nondecreasing with sequence.",
                        location=f"timetable.stop_times.trip[{tid}]",
                        context={"prev_departure_s": prev_dep, "arrival_s": arr},
                        suggestion="Check stop ordering or time encoding (after-midnight service, etc.).",
                    ))
                prev_dep = dep

            # Minimum length
            if len(sts_sorted) < 2:
                rep.add(Issue(
                    severity=Severity.WARNING,
                    code="TRIP_TOO_SHORT",
                    message=f"Trip {tid!r} has fewer than 2 stop_times.",
                    location=f"timetable.stop_times.trip[{tid}]",
                    suggestion="A trip should have at least 2 stops to generate ride links.",
                ))

        return rep