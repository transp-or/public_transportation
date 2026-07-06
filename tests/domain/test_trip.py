from __future__ import annotations

from public_transportation.domain.issues import Severity
from public_transportation.domain.trip import Trip


def _codes(rep):
    return {iss.code for iss in rep.issues}


def _find(rep, code: str):
    return [iss for iss in rep.issues if iss.code == code]


# ---------------------------------------------------------
# Happy path
# ---------------------------------------------------------


def test_valid_trip_has_no_issues():
    tr = Trip(
        trip_id="TR1",
        line_ref="L1",
        capacity=50,
        service_id="WKD",
        headsign="Downtown",
        direction_id=0,
    )
    rep = tr.validate()
    assert rep.issues == []


def test_trip_optional_fields_can_be_none():
    tr = Trip(trip_id="TR1", line_ref='L1')
    rep = tr.validate()
    assert rep.issues == []


# ---------------------------------------------------------
# trip_id
# ---------------------------------------------------------


def test_empty_trip_id_is_error():
    tr = Trip(trip_id="", line_ref='L1')
    rep = tr.validate()

    assert "TRIP_ID_EMPTY" in _codes(rep)
    issues = _find(rep, "TRIP_ID_EMPTY")
    assert len(issues) == 1
    assert issues[0].severity == Severity.ERROR
    assert issues[0].location == "timetable.trips[].trip_id"
    assert issues[0].suggestion is not None


# ---------------------------------------------------------
# capacity
# ---------------------------------------------------------


def test_negative_capacity_is_error_with_context():
    tr = Trip(trip_id="TR1", capacity=-1, line_ref='L1')
    rep = tr.validate()

    assert "TRIP_CAPACITY_NEGATIVE" in _codes(rep)
    issues = _find(rep, "TRIP_CAPACITY_NEGATIVE")
    assert len(issues) == 1
    iss = issues[0]
    assert iss.severity == Severity.ERROR
    assert iss.location == "timetable.trips[trip_id=TR1].capacity"
    assert iss.context is not None
    assert iss.context["capacity"] == -1


def test_zero_capacity_is_warning_with_suggestion():
    tr = Trip(trip_id="TR1", capacity=0, line_ref='L1')
    rep = tr.validate()

    assert "TRIP_CAPACITY_ZERO" in _codes(rep)
    issues = _find(rep, "TRIP_CAPACITY_ZERO")
    assert len(issues) == 1
    iss = issues[0]
    assert iss.severity == Severity.WARNING
    assert iss.location == "timetable.trips[trip_id=TR1].capacity"
    assert iss.suggestion is not None


def test_capacity_none_has_no_issues():
    tr = Trip(trip_id="TR1", capacity=None, line_ref='L1')
    rep = tr.validate()
    assert rep.issues == []


# ---------------------------------------------------------
# direction_id
# ---------------------------------------------------------


def test_negative_direction_id_is_error_with_context():
    tr = Trip(trip_id="TR1", direction_id=-1, line_ref='L1')
    rep = tr.validate()

    assert "TRIP_DIRECTION_INVALID" in _codes(rep)
    issues = _find(rep, "TRIP_DIRECTION_INVALID")
    assert len(issues) == 1
    iss = issues[0]
    assert iss.severity == Severity.ERROR
    assert iss.location == "timetable.trips[trip_id=TR1].direction_id"
    assert iss.context is not None
    assert iss.context["direction_id"] == -1


def test_direction_id_zero_or_positive_ok():
    tr0 = Trip(trip_id="TR1", direction_id=0, line_ref='L1')
    tr1 = Trip(trip_id="TR2", direction_id=1, line_ref='L1')
    rep0 = tr0.validate()
    rep1 = tr1.validate()
    assert rep0.issues == []
    assert rep1.issues == []


# ---------------------------------------------------------
# Multiple issues
# ---------------------------------------------------------


def test_multiple_issues_are_all_reported():
    tr = Trip(trip_id="", capacity=-5, direction_id=-2, line_ref='L1')
    rep = tr.validate()
    codes = _codes(rep)

    assert "TRIP_ID_EMPTY" in codes
    assert "TRIP_CAPACITY_NEGATIVE" in codes
    assert "TRIP_DIRECTION_INVALID" in codes
    assert len(rep.issues) == 3