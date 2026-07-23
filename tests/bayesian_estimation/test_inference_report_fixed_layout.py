from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import public_transportation.viz.inference_comparison_report as report
from public_transportation.measurement.mapping import AggregationSpec


def test_comparison_report_classifies_free_and_frozen_od(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "build_assignment_inputs", lambda **_: "assignment")
    monkeypatch.setattr(
        report,
        "assign_link_flow",
        lambda *, f, **_: np.asarray([np.sum(f)], dtype=float),
    )
    monkeypatch.setattr(
        report,
        "aggregate_link_flow_to_measurements",
        lambda *, link_flow, **_: np.asarray(link_flow),
    )
    monkeypatch.setattr(
        report,
        "apply_detection_rate",
        lambda *, lambda_m, rho: rho * np.asarray(lambda_m),
    )
    keys = (
        SimpleNamespace(origin_stop_id="A", dest_stop_id="B", time_bin_index=0),
        SimpleNamespace(origin_stop_id="B", dest_stop_id="C", time_bin_index=0),
        SimpleNamespace(origin_stop_id="C", dest_stop_id="D", time_bin_index=0),
    )
    id_manager = SimpleNamespace(fingerprint="fp", od_keys_scenario=keys)
    spec = AggregationSpec(
        num_measurements=1,
        measurement_index=np.asarray([0], dtype=np.int32),
        link_index=np.asarray([0], dtype=np.int32),
    )

    bundle = report.compute_od_and_flow_comparison(
        scenario=SimpleNamespace(),
        assignment_artifacts=SimpleNamespace(
            od_groups=SimpleNamespace(
                group_dest_node=np.asarray([10, 20, 30]),
                od_dest_node=np.asarray([10, 20, 30]),
            )
        ),
        id_manager=id_manager,
        mapping_spec=spec,
        y_obs=np.asarray([30.0]),
        fingerprint_expected="fp",
        fingerprint_results="fp",
        theta_hat=2.0,
        f0=np.asarray([10.0, 20.0, 30.0]),
        f_hat=np.asarray([0.0, 22.0, 7.5]),
        fixed_od_indices=np.asarray([0, 2], dtype=np.int64),
        top_k_od=3,
    )

    assert bundle.num_free_od == 1
    assert bundle.num_fixed_od == 2
    assert bundle.num_fixed_zero_od == 1
    assert bundle.num_fixed_positive_od == 1
    assert bundle.assignment_active_od == 2
    assert bundle.active_destination_groups == 2
    assert bundle.removed_destination_groups == 1
    status_by_origin = {row.origin_stop_id: row.estimation_status for row in bundle.od_top_rows}
    assert status_by_origin == {"A": "frozen", "B": "free", "C": "frozen"}

    path = report.write_od_theta_comparison_report_html(
        bundle=bundle,
        output_path=tmp_path / "report.html",
    )
    html = path.read_text(encoding="utf-8")
    assert "free OD" in html
    assert "frozen OD" in html
    assert "status" in html
    assert "frozen" in html
    assert "assignment-active OD" in html
    assert "groups removed" in html


@pytest.mark.parametrize(
    "indices",
    [np.asarray([0, 0]), np.asarray([-1]), np.asarray([3]), np.asarray([0.0])],
)
def test_comparison_report_rejects_invalid_fixed_indices(monkeypatch, indices):
    monkeypatch.setattr(report, "build_assignment_inputs", lambda **_: "assignment")
    spec = AggregationSpec(
        num_measurements=1,
        measurement_index=np.asarray([0], dtype=np.int32),
        link_index=np.asarray([0], dtype=np.int32),
    )
    keys = tuple(
        SimpleNamespace(origin_stop_id=str(i), dest_stop_id=str(i), time_bin_index=0)
        for i in range(3)
    )
    with pytest.raises(ValueError, match="fixed_od_indices"):
        report.compute_od_and_flow_comparison(
            scenario=SimpleNamespace(),
            assignment_artifacts=object(),
            id_manager=SimpleNamespace(fingerprint="fp", od_keys_scenario=keys),
            mapping_spec=spec,
            y_obs=np.asarray([1.0]),
            fingerprint_expected="fp",
            fingerprint_results="fp",
            theta_hat=1.0,
            f0=np.ones(3),
            f_hat=np.ones(3),
            fixed_od_indices=indices,
        )
