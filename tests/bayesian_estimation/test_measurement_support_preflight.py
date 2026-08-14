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
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["issues"][0]["row_index"] == 0
    assert payload["issues"][0]["fixed_zero_reason_counts"] == [
        ["maximum_initial_wait_exceeded", 1]
    ]


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

