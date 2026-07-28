# feasibility.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

from .issues import Issue, Severity, ValidationReport

if TYPE_CHECKING:
    from .scenario import Scenario


@dataclass(frozen=True, slots=True)
class FeasibilityConfig:
    """
    Configuration for timetable feasibility checks.

    :param min_dwell_s: Minimum dwell time at a stop, in seconds. Used to check/flag
        cases where departure is not strictly after arrival.
    :param require_strict_depart_after_arrive: If True, require departure_s > arrival_s
        (strict). If False, allow departure_s == arrival_s (still reported as WARNING
        if min_dwell_s > 0).
    :param require_non_decreasing_times_along_trip: If True, enforce that times along a
        trip are non-decreasing in sequence order (arrival/dep).
    """
    min_dwell_s: int = 1
    require_strict_depart_after_arrive: bool = True
    require_non_decreasing_times_along_trip: bool = True


def validate_timetable_feasibility(
    scenario: Scenario,
    *,
    config: FeasibilityConfig | None = None,
) -> ValidationReport:
    """
    Validate feasibility of every trip in the scenario timetable.

    This is a *cross-element* validation:
    - uses timetable stop_times grouped by trip_id,
    - checks that each trip has a consistent stop sequence,
    - checks local time feasibility at each stop (arrival/departure),
    - checks temporal feasibility along the trip (non-decreasing times).

    The goal is to detect issues early (Scenario-level), and provide actionable
    suggestions (e.g., fix swapped columns, fix non-increasing sequences, add dwell).

    :param scenario: Scenario containing the timetable to validate.
    :param config: FeasibilityConfig controlling which constraints are enforced.
    :return: ValidationReport with ERROR/WARNING issues.
    """
    rep = ValidationReport(issues=[])
    cfg = config or FeasibilityConfig()

    if scenario.timetable is None:
        return rep  # nothing to validate

    tt = scenario.timetable
    known_stop_ids = {s.stop_id for s in scenario.stops}
    known_trip_ids = {t.trip_id for t in tt.trips}

    # ---- Basic referential sanity (stop_times -> trips/stops) ----
    for k, st in enumerate(tt.stop_times):
        loc = f"timetable.stop_times[{getattr(st, 'trip_id', '?')},{k}]"

        if st.trip_id not in known_trip_ids:
            rep.add(Issue(
                severity=Severity.ERROR,
                code="STOPTIME_TRIP_UNKNOWN",
                message=f"stop_times references unknown trip_id: {st.trip_id!r}.",
                location=loc,
                suggestion="Add the trip to trips.csv or fix stop_times.trip_id.",
                context={"trip_id": st.trip_id},
            ))

        if st.stop_id not in known_stop_ids:
            rep.add(Issue(
                severity=Severity.ERROR,
                code="STOPTIME_STOP_UNKNOWN",
                message=f"stop_times references unknown stop_id: {st.stop_id!r}.",
                location=loc,
                suggestion="Add the stop to stops.csv or fix stop_times.stop_id.",
                context={"stop_id": st.stop_id},
            ))

        # ---- Local time feasibility at stop ----
        a = int(st.arrival.seconds_from_midnight)
        d = int(st.departure.seconds_from_midnight)

        if cfg.require_strict_depart_after_arrive:
            if d <= a:
                rep.add(Issue(
                    severity=Severity.ERROR,
                    code="STOPTIME_DEPART_NOT_AFTER_ARRIVE",
                    message="departure time must be strictly after arrival time.",
                    location=loc,
                    suggestion=(
                        f"Ensure departure_s >= arrival_s + {cfg.min_dwell_s}. "
                        "If columns are swapped (stop_id/sequence or arrival/departure), fix the CSV."
                    ),
                    context={"arrival_s": a, "departure_s": d, "min_dwell_s": cfg.min_dwell_s},
                ))
        else:
            if d < a:
                rep.add(Issue(
                    severity=Severity.ERROR,
                    code="STOPTIME_DEPART_BEFORE_ARRIVE",
                    message="departure time must be >= arrival time.",
                    location=loc,
                    suggestion="Ensure departure_s >= arrival_s (or fix swapped columns).",
                    context={"arrival_s": a, "departure_s": d},
                ))
            elif d == a and cfg.min_dwell_s > 0:
                rep.add(Issue(
                    severity=Severity.WARNING,
                    code="STOPTIME_ZERO_DWELL",
                    message="arrival == departure (zero dwell).",
                    location=loc,
                    suggestion=f"Consider setting departure_s = arrival_s + {cfg.min_dwell_s}.",
                    context={"arrival_s": a, "departure_s": d, "min_dwell_s": cfg.min_dwell_s},
                ))

        if st.sequence <= 0:
            rep.add(Issue(
                severity=Severity.ERROR,
                code="STOPTIME_SEQUENCE_NONPOSITIVE",
                message=f"sequence must be a positive integer. Found: {st.sequence}.",
                location=loc,
                suggestion="Fix stop_times.sequence to start at 1 and increase by 1.",
                context={"sequence": st.sequence},
            ))

    # ---- Trip-level sequencing + temporal monotonicity ----
    # Group stop_times by trip_id
    by_trip: dict[str, list] = {}
    for st in tt.stop_times:
        by_trip.setdefault(st.trip_id, []).append(st)

    for trip_id, sts in by_trip.items():
        loc_trip = f"timetable.trip[{trip_id}]"

        # Sort by sequence
        sts_sorted = sorted(sts, key=lambda x: int(x.sequence))
        seqs = [int(x.sequence) for x in sts_sorted]

        # Check duplicate sequences
        if len(seqs) != len(set(seqs)):
            dup = _duplicates(seqs)
            rep.add(Issue(
                severity=Severity.ERROR,
                code="TRIP_SEQUENCE_DUPLICATE",
                message=f"Trip has duplicate sequence values: {sorted(dup)}.",
                location=loc_trip,
                suggestion="Ensure each stop in a trip has a unique increasing sequence.",
                context={"trip_id": trip_id, "sequences": seqs},
            ))

        # Check contiguous 1..N (helps catch swapped columns and messy files)
        if seqs:
            expected = list(range(1, len(seqs) + 1))
            if seqs != expected:
                rep.add(Issue(
                    severity=Severity.WARNING,
                    code="TRIP_SEQUENCE_NOT_CONTIGUOUS",
                    message="Trip sequences are not contiguous 1..N after sorting.",
                    location=loc_trip,
                    suggestion=(
                        "Prefer sequences 1..N without gaps. If stop_id and sequence columns "
                        "were swapped in the CSV, fix stop_times.csv."
                    ),
                    context={"trip_id": trip_id, "sequences": seqs, "expected": expected},
                ))

        if cfg.require_non_decreasing_times_along_trip and len(sts_sorted) >= 2:
            # Enforce that arrival/dep do not go backwards in time along the trip.
            prev_dep = None
            for idx, st in enumerate(sts_sorted):
                a = int(st.arrival.seconds_from_midnight)
                d = int(st.departure.seconds_from_midnight)
                loc = f"timetable.stop_times[{trip_id},{idx}]"

                if prev_dep is not None and a < prev_dep:
                    rep.add(Issue(
                        severity=Severity.ERROR,
                        code="TRIP_TIME_DECREASE",
                        message="Arrival time decreases along the trip (non-feasible chronology).",
                        location=loc,
                        suggestion=(
                            "Ensure stop_times are in the correct order and times are monotone. "
                            "Common cause: sequence column incorrect or swapped with stop_id."
                        ),
                        context={
                            "trip_id": trip_id,
                            "prev_departure_s": prev_dep,
                            "arrival_s": a,
                            "sequence": int(st.sequence),
                        },
                    ))
                prev_dep = d

    return rep


def _duplicates(xs: Iterable[int]) -> set[int]:
    """
    Return duplicated values.

    :param xs: Iterable of integers.
    :return: Set of values that appear at least twice.
    """
    seen: set[int] = set()
    dup: set[int] = set()
    for x in xs:
        if x in seen:
            dup.add(x)
        else:
            seen.add(x)
    return dup


def feasibility_report_for_folder(
    folder: str | Path,
    *,
    strict: bool = False,
    config: FeasibilityConfig | None = None,
) -> ValidationReport:
    """
    Convenience helper: load a Scenario and run timetable feasibility checks.

    :param folder: Scenario folder path (contains metadata.json, stops.csv, etc.).
    :param strict: If True, raise ValueError when ERROR issues are found.
    :param config: FeasibilityConfig controlling checks.
    :return: ValidationReport.
    """
    scen = Scenario.from_folder(folder, strict=False)
    rep = validate_timetable_feasibility(scen, config=config)
    if strict:
        errors = [it for it in rep.issues if it.severity == Severity.ERROR]
        if errors:
            msg = ["Feasibility validation failed (strict=True)."]
            for it in errors:
                msg.append(f"- [{it.code}] {it.message} ({it.location})")
                if getattr(it, "suggestion", None):
                    msg.append(f"    suggestion: {it.suggestion}")
                if getattr(it, "context", None):
                    msg.append(f"    context: {it.context}")
            raise ValueError("\n".join(msg))
    return rep