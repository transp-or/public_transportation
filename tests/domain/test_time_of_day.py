from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from public_transportation.domain.time_of_day import TimeOfDay


# ---------------------------------------------------------
# Construction + invariants
# ---------------------------------------------------------


def test_seconds_from_midnight_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        TimeOfDay(seconds_from_midnight=-1)


def test_from_hms_basic():
    t = TimeOfDay.from_hms(8, 30, 15)
    assert t.seconds_from_midnight == 8 * 3600 + 30 * 60 + 15


def test_from_hms_allows_hours_beyond_23():
    t = TimeOfDay.from_hms(25, 0, 0)
    assert t.seconds_from_midnight == 25 * 3600
    assert t.to_hms() == (25, 0, 0)


@pytest.mark.parametrize(
    "h,m,s,err_msg",
    [
        (-1, 0, 0, "h must be >= 0"),
        (0, -1, 0, "m must be in 0..59"),
        (0, 60, 0, "m must be in 0..59"),
        (0, 0, -1, "s must be in 0..59"),
        (0, 0, 60, "s must be in 0..59"),
    ],
)
def test_from_hms_rejects_invalid_ranges(h: int, m: int, s: int, err_msg: str):
    with pytest.raises(ValueError, match=err_msg):
        TimeOfDay.from_hms(h, m, s)


# ---------------------------------------------------------
# Parsing
# ---------------------------------------------------------


def test_parse_hhmm():
    t = TimeOfDay.parse("08:05")
    assert t.seconds_from_midnight == 8 * 3600 + 5 * 60
    assert t.to_string(include_seconds=False) == "08:05"


def test_parse_hhmmss():
    t = TimeOfDay.parse("08:05:07")
    assert t.seconds_from_midnight == 8 * 3600 + 5 * 60 + 7
    assert t.to_string(include_seconds=True) == "08:05:07"


def test_parse_allows_hours_beyond_23():
    t = TimeOfDay.parse("26:10")
    assert t.seconds_from_midnight == 26 * 3600 + 10 * 60
    assert t.to_string(include_seconds=False) == "26:10"


@pytest.mark.parametrize("text", ["", "8", "08", "08:05:07:09", "08-05", "08.05"])
def test_parse_rejects_bad_formats(text: str):
    with pytest.raises(ValueError, match="Invalid time format"):
        TimeOfDay.parse(text)


def test_parse_strips_whitespace():
    t = TimeOfDay.parse("  09:00  ")
    assert t.seconds_from_midnight == 9 * 3600


# ---------------------------------------------------------
# Conversions
# ---------------------------------------------------------


def test_to_hms_roundtrip():
    original = TimeOfDay(seconds_from_midnight=27 * 3600 + 2 * 60 + 3)
    h, m, s = original.to_hms()
    rebuilt = TimeOfDay.from_hms(h, m, s)
    assert rebuilt == original


def test_to_string_default_includes_seconds():
    t = TimeOfDay.from_hms(1, 2, 3)
    assert t.to_string() == "01:02:03"


def test_to_string_without_seconds():
    t = TimeOfDay.from_hms(1, 2, 3)
    assert t.to_string(include_seconds=False) == "01:02"


def test_to_string_padding_with_hours_beyond_99_is_not_truncated():
    # sanity check: formatting doesn't truncate large hours
    t = TimeOfDay.from_hms(123, 4, 5)
    assert t.to_string() == "123:04:05"


# ---------------------------------------------------------
# to_datetime (timezone-aware)
# ---------------------------------------------------------


def test_to_datetime_is_timezone_aware():
    d = date(2026, 2, 15)
    t = TimeOfDay.from_hms(8, 0, 0)
    dt = t.to_datetime(d, tz="Europe/Zurich")
    assert isinstance(dt, datetime)
    assert dt.tzinfo is not None


def test_to_datetime_matches_seconds_offset_from_midnight():
    d = date(2026, 2, 15)
    t = TimeOfDay(seconds_from_midnight=3600 + 30)  # 01:00:30
    dt = t.to_datetime(d, tz="Europe/Zurich")
    assert dt.hour == 1
    assert dt.minute == 0
    assert dt.second == 30


def test_to_datetime_supports_after_midnight_service():
    d = date(2026, 2, 15)
    t = TimeOfDay.from_hms(25, 0, 0)  # next day 01:00
    dt = t.to_datetime(d, tz="Europe/Zurich")
    assert dt.date() == date(2026, 2, 16)
    assert dt.hour == 1
    assert dt.minute == 0
    assert dt.second == 0