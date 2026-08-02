from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPORT = (
    ROOT
    / "docs/source/examples/geneva_gtfs/estimation/results/gravity_validation_summary.json"
)


def test_committed_geneva_gravity_report_covers_complete_methodology():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["example"] == "geneva_gtfs"
    assert report["backend"] == "cpu"
    assert report["dtype"] == "float32"
    assert report["num_od_total"] == 15_128
    assert report["num_free_od"] == 96
    assert report["num_measurements"] == 8_967
    assert report["operator"]["representation"] == "bcoo"
    assert 0 < report["operator"]["density"] < 0.01
    assert report["operator"]["stored_bytes"] < 100_000
    assert report["minimal"]["iterations"] == 2
    assert report["minimal"]["negative_binomial_deviance"] >= 0
    assert report["lineage"]["nodes"] == 2
    assert report["lineage"]["selected_relaxation"] == "time_period"
    assert report["lineage"]["warm_start_maximum_prediction_difference"] == 0
    assert report["holdout"]["calibration_measurements"] == 8_103
    assert report["holdout"]["holdout_measurements"] == 864
    assert report["holdout"]["calibration_nb_deviance"] >= 0
    assert report["holdout"]["holdout_nb_deviance"] >= 0
    candidates = {item["candidate"]: item for item in report["recommendations"]}
    assert candidates["broad_time_period"]["applicable"]
    assert candidates["broad_time_period"]["added_parameters"] == 1
    assert not candidates["destination_zone_attractiveness"]["applicable"]
    assert not candidates["origin_zone_production"]["applicable"]


@pytest.mark.skipif(
    os.environ.get("RUN_GENEVA_GRAVITY_ACCEPTANCE") != "1",
    reason="Set RUN_GENEVA_GRAVITY_ACCEPTANCE=1 for the bounded live Geneva run.",
)
def test_live_geneva_gravity_workflow_matches_committed_dimensions():
    from docs.source.examples.geneva_gtfs.estimation.run_gravity_validation import (
        run_validation,
    )

    report = run_validation(maximum_iterations=1, holdout_iterations=1)
    assert report["num_od_total"] == 15_128
    assert report["num_free_od"] == 96
    assert report["num_measurements"] == 8_967
    assert report["lineage"]["warm_start_maximum_prediction_difference"] == 0
    assert (
        report["holdout"]["calibration_measurements"]
        + report["holdout"]["holdout_measurements"]
        == 8_967
    )
