from __future__ import annotations

from dataclasses import dataclass

from .issues import Issue, Severity, ValidationReport
from .stop_time import StopTime
from .trip import Trip


@dataclass(slots=True)
class Timetable:
    """
    Container for timetable data.

    This is intentionally minimal: trips + stop_times (+ optional line references via Trip.line_ref).
    It is domain-level (file/GUI friendly) and will later be compiled
    into an efficient time-expanded network representation.

    :param trips: List of Trip objects.
    :param stop_times: List of StopTime records.
    """
    trips: list[Trip]
    stop_times: list[StopTime]

    def validate(self, *, known_stop_ids: set[str] | None = None, known_line_ids: set[str] | None = None) -> ValidationReport:
        """
        Validate internal consistency of the timetable.

        Checks include:
        - unique trip ids
        - stop_times reference existing trips
        - (optionally) stop_times reference existing stops
        - (optionally) trips reference existing lines
        - per trip: sequences unique (only for stop_times with valid integer sequence)
        - per trip: times nondecreasing with sequence (only for stop_times with valid parsed times)

        :param known_stop_ids: Optional set of stop_ids to validate references.
        :param known_line_ids: Optional set of line_ids to validate Trip.line_ref references.
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

        # trips reference lines (optional)
        if known_line_ids is not None:
            for t in self.trips:
                if t.line_ref is None:
                    continue
                if t.line_ref not in known_line_ids:
                    rep.add(Issue(
                        severity=Severity.ERROR,
                        code="TRIP_LINE_UNKNOWN",
                        message=f"Trip {t.trip_id!r} references unknown line_id: {t.line_ref!r}.",
                        location=f"timetable.trips[trip_id={t.trip_id!r}].line_ref",
                        suggestion="Ensure line_id exists in scenario metadata, or leave line_ref unset.",
                    ))

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

            # Keep only stop_times with a valid integer sequence for sequence-based checks.
            sts_with_seq: list[StopTime] = []
            sts_missing_seq: list[StopTime] = []
            for s in sts:
                seq = getattr(s, "sequence", None)
                if isinstance(seq, int):
                    sts_with_seq.append(s)
                else:
                    sts_missing_seq.append(s)

            # If any sequences are missing/unparseable, we cannot fully validate ordering.
            if sts_missing_seq:
                rep.add(Issue(
                    severity=Severity.WARNING,
                    code="STOPTIME_SEQUENCE_MISSING",
                    message=(
                        f"Trip {tid!r} has stop_times with missing/unparseable sequence; "
                        "sequence-based consistency checks are partial."
                    ),
                    location=f"timetable.stop_times.trip[{tid}]",
                    suggestion=(
                        "Ensure the CSV 'sequence' column contains integers and is not swapped with 'stop_id'."
                    ),
                    context={"count_missing_sequence": len(sts_missing_seq), "count_total": len(sts)},
                ))

            # Sequence uniqueness (only among stop_times with valid sequence)
            seqs = [s.sequence for s in sts_with_seq]
            if len(set(seqs)) != len(seqs):
                rep.add(Issue(
                    severity=Severity.ERROR,
                    code="STOPTIME_SEQUENCE_DUPLICATE",
                    message=f"Trip {tid!r} has duplicate sequence values.",
                    location=f"timetable.stop_times.trip[{tid}]",
                    suggestion="Ensure (trip_id, sequence) is unique.",
                ))

            # Sort by sequence and check nondecreasing times.
            # This is only meaningful when arrival/departure are successfully parsed.
            sts_sorted = sorted(sts_with_seq, key=lambda s: s.sequence)

            prev_dep: int | None = None
            prev_seq: int | None = None
            for s in sts_sorted:
                arr_obj = getattr(s, "arrival", None)
                dep_obj = getattr(s, "departure", None)

                arr_s = getattr(arr_obj, "seconds_from_midnight", None)
                dep_s = getattr(dep_obj, "seconds_from_midnight", None)

                # If either time is missing/unparseable, skip monotonic checks for this row.
                if not isinstance(arr_s, (int, float)) or not isinstance(dep_s, (int, float)):
                    rep.add(Issue(
                        severity=Severity.WARNING,
                        code="STOPTIME_TIME_MISSING",
                        message=(
                            f"Trip {tid!r} has stop_times with missing/unparseable arrival/departure; "
                            "time monotonicity checks are partial."
                        ),
                        location=f"timetable.stop_times.trip[{tid}]",
                        suggestion="Fix time encoding in stop_times.csv (arrival/departure).",
                        context={
                            "sequence": s.sequence,
                            "arrival": getattr(arr_obj, "__str__", lambda: str(arr_obj))(),
                            "departure": getattr(dep_obj, "__str__", lambda: str(dep_obj))(),
                        },
                    ))
                    prev_seq = s.sequence
                    continue

                arr_i = int(arr_s)
                dep_i = int(dep_s)

                if prev_dep is not None and arr_i < prev_dep:
                    rep.add(Issue(
                        severity=Severity.ERROR,
                        code="STOPTIME_TIME_NONMONOTONE",
                        message=f"Trip {tid!r} times are not nondecreasing with sequence.",
                        location=f"timetable.stop_times.trip[{tid}]",
                        context={
                            "prev_sequence": prev_seq,
                            "sequence": s.sequence,
                            "prev_departure_s": prev_dep,
                            "arrival_s": arr_i,
                        },
                        suggestion="Check stop ordering or time encoding (after-midnight service, etc.).",
                    ))

                prev_dep = dep_i
                prev_seq = s.sequence

            # Minimum length: only consider stop_times with valid sequence.
            if len(sts_sorted) < 2:
                rep.add(Issue(
                    severity=Severity.WARNING,
                    code="TRIP_TOO_SHORT",
                    message=f"Trip {tid!r} has fewer than 2 stop_times with a valid sequence.",
                    location=f"timetable.stop_times.trip[{tid}]",
                    suggestion="A trip should have at least 2 stops to generate ride links.",
                ))

        return rep