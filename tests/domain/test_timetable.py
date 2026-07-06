# tests/domain/test_timetable.py
from __future__ import annotations

from public_transportation.domain.issues import Severity
from public_transportation.domain.stop_time import StopTime
from public_transportation.domain.time_of_day import TimeOfDay
from public_transportation.domain.timetable import Timetable
from public_transportation.domain.trip import Trip


def _codes(rep):
    return {iss.code for iss in rep.issues}


def _find(rep, code: str):
    return [iss for iss in rep.issues if iss.code == code]


def _mk_trip(trip_id: str = "TR1", *, headsign: str = "X", line_ref: str = "L1", direction_id: int = 0,
             capacity: int = 50) -> Trip:
    # Trip is validated inside Timetable.validate(), so keep it "obviously valid".
    return Trip(
        trip_id=trip_id,
        line_ref=line_ref,
        direction_id=direction_id,
        headsign=headsign,
        capacity=capacity,
    )


def _mk_st(
    trip_id: str,
    stop_id: str,
    seq: int,
    arr_s: int,
    dep_s: int,
) -> StopTime:
    return StopTime(
        trip_id=trip_id,
        stop_id=stop_id,
        sequence=seq,
        arrival=TimeOfDay(seconds_from_midnight=arr_s),
        departure=TimeOfDay(seconds_from_midnight=dep_s),
    )


# ---------------------------------------------------------
# Happy path
# ---------------------------------------------------------


def test_valid_timetable_has_no_issues():
    trips = [_mk_trip("TR1"), _mk_trip("TR2", direction_id=1)]
    stop_times = [
        _mk_st("TR1", "A", 1, 8 * 3600, 8 * 3600 + 1),
        _mk_st("TR1", "B", 2, 8 * 3600 + 300, 8 * 3600 + 360),
        _mk_st("TR2", "B", 1, 9 * 3600, 9 * 3600 + 1),
        _mk_st("TR2", "A", 2, 9 * 3600 + 300, 9 * 3600 + 330),
    ]
    tt = Timetable(trips=trips, stop_times=stop_times)

    rep = tt.validate(known_stop_ids={"A", "B"})

    assert rep.issues == []


# ---------------------------------------------------------
# Duplicate trip ids
# ---------------------------------------------------------


def test_duplicate_trip_id_is_error_and_reported_per_duplicate():
    tt = Timetable(
        trips=[_mk_trip("TR1"), _mk_trip("TR1")],
        stop_times=[
            _mk_st("TR1", "A", 1, 0, 1),
            _mk_st("TR1", "B", 2, 60, 61),
        ],
    )

    rep = tt.validate(known_stop_ids={"A", "B"})

    issues = _find(rep, "TRIP_ID_DUPLICATE")
    assert len(issues) >= 1
    assert all(iss.severity == Severity.ERROR for iss in issues)
    assert all(iss.location == "timetable.trips" for iss in issues)


# ---------------------------------------------------------
# stop_times reference trips / stops
# ---------------------------------------------------------


def test_stop_time_unknown_trip_is_error():
    tt = Timetable(
        trips=[_mk_trip("TR1")],
        stop_times=[
            _mk_st("TR1", "A", 1, 0, 1),
            _mk_st("NOPE", "B", 1, 60, 61),
        ],
    )

    rep = tt.validate(known_stop_ids={"A", "B"})

    issues = _find(rep, "STOPTIME_TRIP_UNKNOWN")
    assert len(issues) == 1
    assert issues[0].severity == Severity.ERROR
    assert issues[0].location == "timetable.stop_times[1]"


def test_stop_time_unknown_stop_is_error_when_known_stop_ids_provided():
    tt = Timetable(
        trips=[_mk_trip("TR1")],
        stop_times=[
            _mk_st("TR1", "A", 1, 0, 1),
            _mk_st("TR1", "ZZZ", 2, 60, 61),
        ],
    )

    rep = tt.validate(known_stop_ids={"A", "B"})

    issues = _find(rep, "STOPTIME_STOP_UNKNOWN")
    assert len(issues) == 1
    assert issues[0].severity == Severity.ERROR
    assert issues[0].location == "timetable.stop_times[1]"


def test_known_stop_ids_none_skips_stop_reference_check():
    tt = Timetable(
        trips=[_mk_trip("TR1")],
        stop_times=[
            _mk_st("TR1", "A", 1, 0, 1),
            _mk_st("TR1", "UNKNOWN_STOP", 2, 60, 61),
        ],
    )

    rep = tt.validate(known_stop_ids=None)

    assert "STOPTIME_STOP_UNKNOWN" not in _codes(rep)


# ---------------------------------------------------------
# Per-trip sequencing rules
# ---------------------------------------------------------


def test_duplicate_sequence_within_trip_is_error():
    tt = Timetable(
        trips=[_mk_trip("TR1")],
        stop_times=[
            _mk_st("TR1", "A", 1, 0, 1),
            _mk_st("TR1", "B", 1, 60, 61),  # duplicate seq
        ],
    )

    rep = tt.validate(known_stop_ids={"A", "B"})

    issues = _find(rep, "STOPTIME_SEQUENCE_DUPLICATE")
    assert len(issues) == 1
    assert issues[0].severity == Severity.ERROR
    assert issues[0].location == "timetable.stop_times.trip[TR1]"
    assert issues[0].suggestion is not None


def test_time_nonmonotone_with_sequence_is_error_and_contains_context():
    # seq 2 arrival happens before seq 1 departure -> nonmonotone
    tt = Timetable(
        trips=[_mk_trip("TR1")],
        stop_times=[
            _mk_st("TR1", "A", 1, 8 * 3600, 8 * 3600 + 120),      # dep 08:02
            _mk_st("TR1", "B", 2, 8 * 3600 + 60, 8 * 3600 + 90),  # arr 08:01 < prev dep
        ],
    )

    rep = tt.validate(known_stop_ids={"A", "B"})

    issues = _find(rep, "STOPTIME_TIME_NONMONOTONE")
    assert len(issues) == 1
    iss = issues[0]
    assert iss.severity == Severity.ERROR
    assert iss.location == "timetable.stop_times.trip[TR1]"
    assert iss.context is not None
    assert iss.context["prev_departure_s"] == 8 * 3600 + 120
    assert iss.context["arrival_s"] == 8 * 3600 + 60
    assert iss.suggestion is not None


# ---------------------------------------------------------
# Trip length warnings
# ---------------------------------------------------------


def test_trip_too_short_is_warning_for_one_stop_time():
    tt = Timetable(
        trips=[_mk_trip("TR1")],
        stop_times=[
            _mk_st("TR1", "A", 1, 0, 1),
        ],
    )

    rep = tt.validate(known_stop_ids={"A"})

    issues = _find(rep, "TRIP_TOO_SHORT")
    assert len(issues) == 1
    assert issues[0].severity == Severity.WARNING


def test_trip_too_short_not_reported_for_unknown_trip_only():
    # stop_times refer to unknown trip; per-trip checks should be skipped for that tid.
    tt = Timetable(
        trips=[_mk_trip("TR1")],
        stop_times=[
            _mk_st("UNKNOWN", "A", 1, 0, 1),
        ],
    )

    rep = tt.validate(known_stop_ids={"A"})

    assert "STOPTIME_TRIP_UNKNOWN" in _codes(rep)
    assert "TRIP_TOO_SHORT" not in _codes(rep)


# ---------------------------------------------------------
# Aggregation: multiple issues are all reported
# ---------------------------------------------------------


def test_multiple_issues_accumulate():
    tt = Timetable(
        trips=[_mk_trip("TR1"), _mk_trip("TR1")],  # duplicate id
        stop_times=[
            _mk_st("TR1", "A", 1, 8 * 3600, 8 * 3600 + 120),
            _mk_st("TR1", "B", 2, 8 * 3600 + 60, 8 * 3600 + 90),   # nonmonotone
            _mk_st("TR1", "B", 2, 8 * 3600 + 300, 8 * 3600 + 301), # duplicate sequence too
            _mk_st("NOPE", "ZZZ", 1, 0, 1),                        # unknown trip and stop
        ],
    )

    rep = tt.validate(known_stop_ids={"A", "B"})

    codes = _codes(rep)
    assert "TRIP_ID_DUPLICATE" in codes
    assert "STOPTIME_TRIP_UNKNOWN" in codes
    assert "STOPTIME_STOP_UNKNOWN" in codes
    assert "STOPTIME_SEQUENCE_DUPLICATE" in codes
    assert "STOPTIME_TIME_NONMONOTONE" in codes