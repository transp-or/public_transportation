from __future__ import annotations

from public_transportation.domain.issues import Severity
from public_transportation.domain.stop_time import StopTime
from public_transportation.domain.time_of_day import TimeOfDay


def _codes(rep):
    return {iss.code for iss in rep.issues}


def _find(rep, code: str):
    return [iss for iss in rep.issues if iss.code == code]


def _mk_st(
    *,
    trip_id: str = "TR1",
    stop_id: str = "A",
    seq: int = 1,
    arr_s: int = 0,
    dep_s: int = 1,
) -> StopTime:
    return StopTime(
        trip_id=trip_id,
        stop_id=stop_id,
        sequence=seq,
        arrival=TimeOfDay(seconds_from_midnight=arr_s),
        departure=TimeOfDay(seconds_from_midnight=dep_s),
    )


# ---------------------------------------------------------
# Valid case
# ---------------------------------------------------------


def test_valid_stop_time_has_no_issues():
    st = _mk_st(arr_s=100, dep_s=101)
    rep = st.validate()
    assert rep.issues == []


# ---------------------------------------------------------
# trip_id
# ---------------------------------------------------------


def test_empty_trip_id_is_error():
    st = _mk_st(trip_id="")
    rep = st.validate()

    issues = _find(rep, "STOPTIME_TRIP_EMPTY")
    assert len(issues) == 1
    assert issues[0].severity == Severity.ERROR
    assert issues[0].location == "timetable.stop_times[].trip_id"


# ---------------------------------------------------------
# stop_id
# ---------------------------------------------------------


def test_empty_stop_id_is_error():
    st = _mk_st(stop_id="", seq=3, trip_id="TRX")
    rep = st.validate()

    issues = _find(rep, "STOPTIME_STOP_EMPTY")
    assert len(issues) == 1
    iss = issues[0]
    assert iss.severity == Severity.ERROR
    assert iss.location == "timetable.stop_times[TRX,3].stop_id"


# ---------------------------------------------------------
# sequence
# ---------------------------------------------------------


def test_sequence_must_be_positive():
    st = _mk_st(seq=0)
    rep = st.validate()

    issues = _find(rep, "STOPTIME_SEQUENCE_NONPOSITIVE")
    assert len(issues) == 1
    iss = issues[0]
    assert iss.severity == Severity.ERROR
    assert iss.context is not None
    assert iss.context["sequence"] == 0


# ---------------------------------------------------------
# departure not strictly after arrival
# ---------------------------------------------------------


def test_departure_not_after_arrival_is_error():
    st = _mk_st(arr_s=500, dep_s=400, seq=2, trip_id="TR1")
    rep = st.validate()

    issues = _find(rep, "STOPTIME_DEPART_NOT_AFTER_ARRIVE")
    assert len(issues) == 1

    iss = issues[0]
    assert iss.severity == Severity.ERROR
    assert iss.location == "timetable.stop_times[TR1,2]"
    assert iss.context is not None
    assert iss.context["arrival_s"] == 500
    assert iss.context["departure_s"] == 400
    assert iss.suggestion is not None


# ---------------------------------------------------------
# multiple issues accumulate
# ---------------------------------------------------------


def test_multiple_issues_all_reported():
    st = _mk_st(
        trip_id="",
        stop_id="",
        seq=0,
        arr_s=500,
        dep_s=400,
    )

    rep = st.validate()
    codes = _codes(rep)

    assert "STOPTIME_TRIP_EMPTY" in codes
    assert "STOPTIME_STOP_EMPTY" in codes
    assert "STOPTIME_SEQUENCE_NONPOSITIVE" in codes
    assert "STOPTIME_DEPART_NOT_AFTER_ARRIVE" in codes

    # exactly 4 issues expected
    assert len(rep.issues) == 4