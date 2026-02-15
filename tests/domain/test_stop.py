from __future__ import annotations

import pytest

from public_transportation.domain.stop import Stop
from public_transportation.domain.issues import Severity


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def _codes(report):
    return {iss.code for iss in report.issues}


def _severities(report):
    return {iss.severity for iss in report.issues}


# ---------------------------------------------------------
# Valid stop
# ---------------------------------------------------------


def test_valid_stop_has_no_issues():
    stop = Stop(
        stop_id="S1",
        name="Main Station",
        lat=46.52,
        lon=6.57,
    )

    report = stop.validate()

    assert report.issues == []


# ---------------------------------------------------------
# Stop id
# ---------------------------------------------------------


def test_empty_stop_id_is_error():
    stop = Stop(
        stop_id="",
        name="Main Station",
        lat=46.5,
        lon=6.5,
    )

    report = stop.validate()

    assert "STOP_ID_EMPTY" in _codes(report)
    assert Severity.ERROR in _severities(report)


# ---------------------------------------------------------
# Stop name
# ---------------------------------------------------------


def test_empty_name_is_warning():
    stop = Stop(
        stop_id="S1",
        name="",
        lat=46.5,
        lon=6.5,
    )

    report = stop.validate()

    assert "STOP_NAME_EMPTY" in _codes(report)
    assert Severity.WARNING in _severities(report)


# ---------------------------------------------------------
# Latitude validation
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "lat",
    [-91.0, 91.0, 1000.0],
)
def test_latitude_out_of_range_is_error(lat):
    stop = Stop(
        stop_id="S1",
        name="Stop",
        lat=lat,
        lon=6.5,
    )

    report = stop.validate()

    assert "STOP_LAT_RANGE" in _codes(report)
    assert Severity.ERROR in _severities(report)


def test_latitude_bounds_are_valid():
    for lat in (-90.0, 0.0, 90.0):
        stop = Stop(
            stop_id="S1",
            name="Stop",
            lat=lat,
            lon=6.5,
        )
        report = stop.validate()
        assert "STOP_LAT_RANGE" not in _codes(report)


# ---------------------------------------------------------
# Longitude validation
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "lon",
    [-181.0, 181.0, 500.0],
)
def test_longitude_out_of_range_is_error(lon):
    stop = Stop(
        stop_id="S1",
        name="Stop",
        lat=46.5,
        lon=lon,
    )

    report = stop.validate()

    assert "STOP_LON_RANGE" in _codes(report)
    assert Severity.ERROR in _severities(report)


def test_longitude_bounds_are_valid():
    for lon in (-180.0, 0.0, 180.0):
        stop = Stop(
            stop_id="S1",
            name="Stop",
            lat=46.5,
            lon=lon,
        )
        report = stop.validate()
        assert "STOP_LON_RANGE" not in _codes(report)


# ---------------------------------------------------------
# Multiple issues together
# ---------------------------------------------------------


def test_multiple_issues_are_all_reported():
    stop = Stop(
        stop_id="",
        name="",
        lat=200.0,
        lon=300.0,
    )

    report = stop.validate()
    codes = _codes(report)

    assert "STOP_ID_EMPTY" in codes
    assert "STOP_NAME_EMPTY" in codes
    assert "STOP_LAT_RANGE" in codes
    assert "STOP_LON_RANGE" in codes

    # ensure multiple issues collected
    assert len(report.issues) == 4


# ---------------------------------------------------------
# Report structure
# ---------------------------------------------------------


def test_issue_context_contains_value():
    stop = Stop(
        stop_id="S1",
        name="Stop",
        lat=999.0,
        lon=6.5,
    )

    report = stop.validate()

    issue = next(iss for iss in report.issues if iss.code == "STOP_LAT_RANGE")

    assert issue.context is not None
    assert issue.context["lat"] == 999.0