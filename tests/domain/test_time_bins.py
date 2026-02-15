from __future__ import annotations

import pytest

from public_transportation.domain.issues import Severity
from public_transportation.domain.time_bin import TimeBin
from public_transportation.domain.time_of_day import TimeOfDay


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def _codes(report):
    return {iss.code for iss in report.issues}


def _find_issue(report, code: str):
    return next((iss for iss in report.issues if iss.code == code), None)


# ---------------------------------------------------------
# Valid bins
# ---------------------------------------------------------


def test_valid_time_bin_has_no_issues():
    tb = TimeBin(
        bin_id="T1",
        start=TimeOfDay(seconds_from_midnight=8 * 3600),
        end=TimeOfDay(seconds_from_midnight=8 * 3600 + 15 * 60),
    )

    report = tb.validate()

    assert report.issues == []


def test_end_strictly_after_start_required():
    start = TimeOfDay(seconds_from_midnight=8 * 3600)

    tb_equal = TimeBin(bin_id="T1", start=start, end=start)
    rep_equal = tb_equal.validate()
    assert "TIMEBIN_ORDER" in _codes(rep_equal)

    tb_before = TimeBin(bin_id="T1", start=start, end=TimeOfDay(seconds_from_midnight=8 * 3600 - 1))
    rep_before = tb_before.validate()
    assert "TIMEBIN_ORDER" in _codes(rep_before)


# ---------------------------------------------------------
# bin_id
# ---------------------------------------------------------


def test_empty_bin_id_is_error():
    tb = TimeBin(
        bin_id="",
        start=TimeOfDay(seconds_from_midnight=0),
        end=TimeOfDay(seconds_from_midnight=60),
    )

    report = tb.validate()

    assert "TIMEBIN_ID_EMPTY" in _codes(report)
    assert any(iss.severity == Severity.ERROR for iss in report.issues)


# ---------------------------------------------------------
# Error issue content
# ---------------------------------------------------------


def test_timebin_order_issue_contains_context_and_suggestion():
    tb = TimeBin(
        bin_id="Tbad",
        start=TimeOfDay(seconds_from_midnight=10 * 3600),
        end=TimeOfDay(seconds_from_midnight=9 * 3600),
    )

    report = tb.validate()
    iss = _find_issue(report, "TIMEBIN_ORDER")
    assert iss is not None

    assert iss.severity == Severity.ERROR
    assert iss.suggestion is not None and "end > start" in iss.suggestion
    assert iss.context is not None
    assert iss.context["start"] == 10 * 3600
    assert iss.context["end"] == 9 * 3600


# ---------------------------------------------------------
# Multiple issues
# ---------------------------------------------------------


def test_multiple_issues_are_all_reported():
    tb = TimeBin(
        bin_id="",
        start=TimeOfDay(seconds_from_midnight=100),
        end=TimeOfDay(seconds_from_midnight=50),
    )

    report = tb.validate()
    codes = _codes(report)

    assert "TIMEBIN_ID_EMPTY" in codes
    assert "TIMEBIN_ORDER" in codes
    assert len(report.issues) == 2