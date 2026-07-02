from __future__ import annotations

import math

import pytest

from public_transportation.assignment import graph_sentinels as gs


# ---------------------------------------------------------------------------
# Node-time sentinels
# ---------------------------------------------------------------------------


def test_centroid_time_sentinels_are_finite_floats():
    assert isinstance(gs.CENTROID_IN_TIME_MIN, float)
    assert isinstance(gs.CENTROID_OUT_TIME_MIN, float)

    assert math.isfinite(gs.CENTROID_IN_TIME_MIN)
    assert math.isfinite(gs.CENTROID_OUT_TIME_MIN)


def test_centroid_time_sentinels_have_expected_ordering():
    assert gs.CENTROID_IN_TIME_MIN < 0.0
    assert gs.CENTROID_OUT_TIME_MIN > 0.0
    assert gs.CENTROID_IN_TIME_MIN < gs.CENTROID_OUT_TIME_MIN


def test_centroid_time_sentinels_are_far_outside_daily_event_range():
    # Event times are seconds-from-midnight converted to minutes, normally
    # within [0, 1440). The sentinels should be far outside that range.
    assert gs.CENTROID_IN_TIME_MIN < -1.0e6
    assert gs.CENTROID_OUT_TIME_MIN > 1.0e6


def test_centroid_time_sentinel_values_are_stable():
    assert gs.CENTROID_IN_TIME_MIN == pytest.approx(-1.0e12)
    assert gs.CENTROID_OUT_TIME_MIN == pytest.approx(1.0e12)


def test_centroid_time_s_is_minus_one():
    assert isinstance(gs.CENTROID_TIME_S, int)
    assert gs.CENTROID_TIME_S == -1


# ---------------------------------------------------------------------------
# Node-kind codes
# ---------------------------------------------------------------------------


def test_node_kind_codes_are_integers():
    assert isinstance(gs.NODE_KIND_CENTROID_IN, int)
    assert isinstance(gs.NODE_KIND_EVENT_ARR, int)
    assert isinstance(gs.NODE_KIND_EVENT_DEP, int)
    assert isinstance(gs.NODE_KIND_CENTROID_OUT, int)


def test_node_kind_codes_are_unique():
    values = [
        gs.NODE_KIND_CENTROID_IN,
        gs.NODE_KIND_EVENT_ARR,
        gs.NODE_KIND_EVENT_DEP,
        gs.NODE_KIND_CENTROID_OUT,
    ]

    assert len(values) == len(set(values))


def test_node_kind_codes_are_contiguous_from_zero():
    values = sorted(
        [
            gs.NODE_KIND_CENTROID_IN,
            gs.NODE_KIND_EVENT_ARR,
            gs.NODE_KIND_EVENT_DEP,
            gs.NODE_KIND_CENTROID_OUT,
        ]
    )

    assert values == [0, 1, 2, 3]


def test_node_kind_code_values_are_stable():
    assert gs.NODE_KIND_CENTROID_IN == 0
    assert gs.NODE_KIND_EVENT_ARR == 1
    assert gs.NODE_KIND_EVENT_DEP == 2
    assert gs.NODE_KIND_CENTROID_OUT == 3


def test_centroid_in_precedes_event_and_centroid_out_codes():
    assert gs.NODE_KIND_CENTROID_IN < gs.NODE_KIND_EVENT_ARR
    assert gs.NODE_KIND_CENTROID_IN < gs.NODE_KIND_EVENT_DEP
    assert gs.NODE_KIND_CENTROID_IN < gs.NODE_KIND_CENTROID_OUT


def test_centroid_out_is_highest_node_kind_code():
    assert gs.NODE_KIND_CENTROID_OUT == max(
        gs.NODE_KIND_CENTROID_IN,
        gs.NODE_KIND_EVENT_ARR,
        gs.NODE_KIND_EVENT_DEP,
        gs.NODE_KIND_CENTROID_OUT,
    )


# ---------------------------------------------------------------------------
# Link-type codes
# ---------------------------------------------------------------------------


def test_link_type_codes_are_integers():
    assert isinstance(gs.LINK_TYPE_RIDE, int)
    assert isinstance(gs.LINK_TYPE_TRANSFER, int)
    assert isinstance(gs.LINK_TYPE_ACCESS, int)
    assert isinstance(gs.LINK_TYPE_EGRESS, int)
    assert isinstance(gs.LINK_TYPE_DWELL, int)


def test_link_type_codes_are_unique():
    values = [
        gs.LINK_TYPE_RIDE,
        gs.LINK_TYPE_TRANSFER,
        gs.LINK_TYPE_ACCESS,
        gs.LINK_TYPE_EGRESS,
        gs.LINK_TYPE_DWELL,
    ]

    assert len(values) == len(set(values))


def test_link_type_codes_are_contiguous_from_zero():
    values = sorted(
        [
            gs.LINK_TYPE_RIDE,
            gs.LINK_TYPE_TRANSFER,
            gs.LINK_TYPE_ACCESS,
            gs.LINK_TYPE_EGRESS,
            gs.LINK_TYPE_DWELL,
        ]
    )

    assert values == [0, 1, 2, 3, 4]


def test_link_type_code_values_are_stable():
    assert gs.LINK_TYPE_RIDE == 0
    assert gs.LINK_TYPE_TRANSFER == 1
    assert gs.LINK_TYPE_ACCESS == 2
    assert gs.LINK_TYPE_EGRESS == 3
    assert gs.LINK_TYPE_DWELL == 4


# ---------------------------------------------------------------------------
# Cross-family checks
# ---------------------------------------------------------------------------


def test_node_kind_and_link_type_names_do_not_overlap_by_identity():
    node_kind_names = {
        "NODE_KIND_CENTROID_IN",
        "NODE_KIND_EVENT_ARR",
        "NODE_KIND_EVENT_DEP",
        "NODE_KIND_CENTROID_OUT",
    }
    link_type_names = {
        "LINK_TYPE_RIDE",
        "LINK_TYPE_TRANSFER",
        "LINK_TYPE_ACCESS",
        "LINK_TYPE_EGRESS",
        "LINK_TYPE_DWELL",
    }

    assert node_kind_names.isdisjoint(link_type_names)


def test_exported_uppercase_constants_are_expected():
    exported_constants = {
        name
        for name in dir(gs)
        if name.isupper() and not name.startswith("__")
    }

    expected = {
        "CENTROID_IN_TIME_MIN",
        "CENTROID_OUT_TIME_MIN",
        "CENTROID_TIME_S",
        "NODE_KIND_CENTROID_IN",
        "NODE_KIND_EVENT_ARR",
        "NODE_KIND_EVENT_DEP",
        "NODE_KIND_CENTROID_OUT",
        "LINK_TYPE_RIDE",
        "LINK_TYPE_TRANSFER",
        "LINK_TYPE_ACCESS",
        "LINK_TYPE_EGRESS",
        "LINK_TYPE_DWELL",
    }

    assert expected.issubset(exported_constants)


def test_no_negative_node_or_link_type_codes():
    node_kind_values = [
        gs.NODE_KIND_CENTROID_IN,
        gs.NODE_KIND_EVENT_ARR,
        gs.NODE_KIND_EVENT_DEP,
        gs.NODE_KIND_CENTROID_OUT,
    ]
    link_type_values = [
        gs.LINK_TYPE_RIDE,
        gs.LINK_TYPE_TRANSFER,
        gs.LINK_TYPE_ACCESS,
        gs.LINK_TYPE_EGRESS,
        gs.LINK_TYPE_DWELL,
    ]

    assert all(value >= 0 for value in node_kind_values)
    assert all(value >= 0 for value in link_type_values)


def test_sentinel_time_values_are_not_valid_link_or_node_codes():
    code_values = {
        gs.NODE_KIND_CENTROID_IN,
        gs.NODE_KIND_EVENT_ARR,
        gs.NODE_KIND_EVENT_DEP,
        gs.NODE_KIND_CENTROID_OUT,
        gs.LINK_TYPE_RIDE,
        gs.LINK_TYPE_TRANSFER,
        gs.LINK_TYPE_ACCESS,
        gs.LINK_TYPE_EGRESS,
        gs.LINK_TYPE_DWELL,
    }

    assert gs.CENTROID_IN_TIME_MIN not in code_values
    assert gs.CENTROID_OUT_TIME_MIN not in code_values
    assert gs.CENTROID_TIME_S not in code_values


# ---------------------------------------------------------------------------
# Documentation-level invariants
# ---------------------------------------------------------------------------


def test_docstring_mentions_centralized_sentinels():
    assert gs.__doc__ is not None
    assert "Centralized sentinel constants" in gs.__doc__


def test_docstring_mentions_minutes_and_seconds_conventions():
    assert gs.__doc__ is not None
    assert "MINUTES" in gs.__doc__
    assert "seconds-from-midnight" in gs.__doc__