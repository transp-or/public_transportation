from __future__ import annotations

import json

import numpy as np
import pytest

from public_transportation.inference.assignment_contract import (
    CanonicalMeasurement,
    CanonicalTimeInterval,
    build_canonical_assignment_index,
)
from public_transportation.inference.measurement_support_preflight import (
    UnsupportedPositiveBoardingError,
    audit_positive_boarding_support,
    enforce_positive_boarding_support,
    format_positive_boarding_support_failure,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout
from public_transportation.measurement.mapping import MappingEntry, MappingInfo


def _index(*, role: str):
    if role == "free":
        free = (0,)
        fixed = ()
        fixed_values = ()
        fixed_zero = ()
        fixed_positive = ()
        baselines = (1.0,)
    elif role == "fixed_zero":
        free = ()
        fixed = (0,)
        fixed_values = (0.0,)
        fixed_zero = (0,)
        fixed_positive = ()
        baselines = ()
    elif role == "fixed_positive":
        free = ()
        fixed = (0,)
        fixed_values = (2.0,)
        fixed_zero = ()
        fixed_positive = (0,)
        baselines = ()
    else:
        raise AssertionError(role)
    layout = ODParameterLayout(
        num_od_total=1,
        od_keys=(("origin-a", "destination", "morning"),),
        free_od_indices=free,
        fixed_od_indices=fixed,
        fixed_od_values=fixed_values,
        free_baseline_values=baselines,
        fixed_zero_indices=fixed_zero,
        fixed_positive_indices=fixed_positive,
    )
    return build_canonical_assignment_index(
        parameter_layout=layout,
        time_intervals=(CanonicalTimeInterval("morning", 0, 3600),),
        measurements=(
            CanonicalMeasurement(
                0, "boarding-0", "boarding", "origin-a", "morning"
            ),
            CanonicalMeasurement(
                1, "boarding-elsewhere", "boarding", "origin-b", "morning"
            ),
            CanonicalMeasurement(
                2, "alighting-0", "alighting", "destination", "morning"
            ),
        ),
    )


def _mapping_info() -> MappingInfo:
    entries = tuple(
        MappingEntry(
            row_index=index,
            measurement_type="boarding" if index < 2 else "alighting",
            method_id="counter",
            stop_id=("origin-a", "origin-b", "destination")[index],
            time_hms="00:10:00",
            trip_id=f"trip-{index}",
            line_id="line",
            observed_value=(4.0, 3.0, 9.0)[index],
            predicted_value=float("nan"),
            matched_event_node=index,
            matched_link_indices=None,
        )
        for index in range(3)
    )
    return MappingInfo(entries=entries, fingerprint="mapping")


def test_canonical_preflight_reports_absent_and_fixed_zero_origins(tmp_path):
    index = _index(role="fixed_zero")
    report = audit_positive_boarding_support(
        canonical_index=index,
        observations=np.asarray([4.0, 3.0, 9.0]),
        mapping_info=_mapping_info(),
        fixed_zero_reasons_by_full_index={0: "maximum_initial_wait_exceeded"},
    )

    assert not report.safe
    assert report.unsupported_positive_boarding_rows == 2
    assert report.unsupported_positive_boarding_mass == 7.0
    assert [issue.cause for issue in report.issues] == [
        "origin_interval_all_fixed_zero",
        "origin_interval_absent_from_demand",
    ]
    assert report.issues[0].fixed_zero_reason_counts == (
        ("maximum_initial_wait_exceeded", 1),
    )
    assert report.issues[0].trip_id == "trip-0"

    path = tmp_path / "preflight.json"
    with pytest.raises(UnsupportedPositiveBoardingError) as caught:
        enforce_positive_boarding_support(report, report_path=path)
    assert caught.value.report_path == path
    assert caught.value.details["report_path"] == str(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["issues"][0]["row_index"] == 0
    assert payload["issues"][0]["fixed_zero_reason_counts"] == [
        ["maximum_initial_wait_exceeded", 1]
    ]
    assert payload["unsupported_positive_boarding_share"] == pytest.approx(1.0)
    summaries = {
        summary["cause"]: summary for summary in payload["cause_summaries"]
    }
    assert summaries["origin_interval_all_fixed_zero"]["rows"] == 1
    assert summaries["origin_interval_all_fixed_zero"]["observed_mass"] == 4.0

    message = str(caught.value)
    assert "Positive boarding support preflight failed." in message
    assert "Measurement rows: 3" in message
    assert "Unsupported positive boarding rows: 2 (100.0000%)" in message
    assert "source row index 0" in message
    assert "Remediation:" in message
    assert "Preserve the original measurement file." in message


def test_zero_boardings_and_positive_alightings_do_not_trigger_boarding_failure():
    report = audit_positive_boarding_support(
        canonical_index=_index(role="fixed_zero"),
        observations=np.asarray([0.0, 0.0, 9.0]),
    )
    assert report.safe
    assert report.positive_boarding_rows == 0


@pytest.mark.parametrize("role", ["free", "fixed_positive"])
def test_active_origin_passes_cheap_check_but_requires_exact_route_support(role):
    index = _index(role=role)
    observations = np.asarray([4.0, 0.0, 0.0])
    canonical = audit_positive_boarding_support(
        canonical_index=index, observations=observations
    )
    assert canonical.safe

    unsupported = audit_positive_boarding_support(
        canonical_index=index,
        observations=observations,
        supported_measurement_rows=np.asarray([], dtype=np.int64),
        stage="routing_support",
    )
    assert not unsupported.safe
    assert unsupported.issues[0].cause == "no_retained_route_to_boarding_event"

    supported = audit_positive_boarding_support(
        canonical_index=index,
        observations=observations,
        supported_measurement_rows=np.asarray([0], dtype=np.int64),
        stage="routing_support",
    )
    assert supported.safe


def test_large_positive_boarding_report_has_expected_aggregates_and_rows():
    """Regression fixture for the medium-case headline support counts."""
    total_rows = 251_926
    positive_rows = 84_032
    supported_rows = 84_023
    unsupported_rows = 9
    layout = ODParameterLayout(
        num_od_total=1,
        od_keys=(("origin-a", "destination", "t8"),),
        free_od_indices=(0,),
        fixed_od_indices=(),
        fixed_od_values=(),
        free_baseline_values=(1.0,),
        fixed_zero_indices=(),
        fixed_positive_indices=(),
    )
    measurements = tuple(
        CanonicalMeasurement(
            row,
            f"measurement-{row}",
            "boarding" if row < positive_rows else "alighting",
            "origin-a" if row < supported_rows else "missing-origin",
            "t8",
        )
        for row in range(total_rows)
    )
    index = build_canonical_assignment_index(
        parameter_layout=layout,
        time_intervals=(CanonicalTimeInterval("t8", 8 * 3600, 9 * 3600),),
        measurements=measurements,
    )
    observations = np.zeros(total_rows, dtype=np.float64)
    observations[:supported_rows] = 1.0
    observations[supported_rows : supported_rows + 8] = 1.0
    observations[supported_rows + 8] = 2.0

    report = audit_positive_boarding_support(
        canonical_index=index,
        observations=observations,
        mapping_info=None,
    )

    assert report.number_of_measurements == total_rows
    assert report.positive_boarding_rows == positive_rows
    assert report.supported_positive_boarding_rows == supported_rows
    assert report.unsupported_positive_boarding_rows == unsupported_rows
    assert report.unsupported_positive_boarding_mass == 10.0
    assert report.unsupported_positive_boarding_share == pytest.approx(
        9 / 84032
    )
    summary = report.cause_summaries
    assert len(summary) == 1
    assert summary[0].cause == "origin_interval_absent_from_demand"
    assert summary[0].rows == 9
    assert summary[0].observed_mass == 10.0

    message = format_positive_boarding_support_failure(report)
    assert "Measurement rows: 251,926" in message
    assert "Positive boarding rows: 84,032" in message
    assert "Supported positive boarding rows: 84,023" in message
    assert "Unsupported positive boarding rows: 9 (0.0107%)" in message
    assert "Unsupported positive boarding mass: 10" in message
    assert "origin_interval_absent_from_demand: 9 rows, mass 10" in message
    assert "source row index 84023" in message
