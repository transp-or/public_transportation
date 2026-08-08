from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "docs/source/examples"


def _report(example: str) -> dict[str, object]:
    path = EXAMPLES / example / "estimation/results/gravity_estimation_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_simple_example_01_report_demonstrates_frozen_positive_cells():
    report = _report("simple_example_01")
    assert report["schema_version"] == 1
    assert report["num_od_total"] == 6
    assert report["num_free_od"] == 4
    assert report["num_fixed_positive"] == 2
    assert report["num_measurements"] == 12
    assert report["operator"]["representation"] == "bcoo"
    assert report["minimal_model"]["status"] == "converged"
    assert report["minimal_model"]["fixed_cell_maximum_error"] == 0


def test_simple_example_02_report_demonstrates_development_and_holdout():
    report = _report("simple_example_02")
    assert report["schema_version"] == 1
    assert report["num_od_total"] == 72
    assert report["num_free_od"] == 70
    assert report["num_fixed_positive"] == 2
    assert report["num_measurements"] == 270
    assert report["operator"]["representation"] == "bcoo"
    assert report["minimal_model"]["status"] == "converged"
    assert report["minimal_model"]["fixed_cell_maximum_error"] == 0
    assert report["relaxation"]["selected"] == "time_period"
    assert report["relaxation"]["lineage_nodes"] == 2
    assert report["relaxation"]["warm_start_maximum_prediction_difference"] == 0
    assert report["holdout"]["unit"] == "vehicle_journey"
    assert report["holdout"]["calibration_measurements"] == 208
    assert report["holdout"]["holdout_measurements"] == 62


@pytest.mark.skipif(
    os.environ.get("RUN_SIMPLE_GRAVITY_ACCEPTANCE") != "1",
    reason="Set RUN_SIMPLE_GRAVITY_ACCEPTANCE=1 for both live synthetic runs.",
)
def test_live_simple_gravity_workflows(tmp_path):
    from docs.source.examples.simple_gravity_workflow import (
        run_simple_gravity_workflow,
    )

    first = run_simple_gravity_workflow(
        example=EXAMPLES / "simple_example_01",
        routing_parameter=5.0,
        maximum_iterations=2,
        operator_cache_directory=tmp_path / "operator-01",
        include_relaxation_and_holdout=False,
    )
    second = run_simple_gravity_workflow(
        example=EXAMPLES / "simple_example_02",
        routing_parameter=1.0,
        maximum_iterations=2,
        operator_cache_directory=tmp_path / "operator-02",
        include_relaxation_and_holdout=True,
    )
    assert first["minimal_model"]["fixed_cell_maximum_error"] == 0
    assert first["generic_demand_m0"]["operator_materialized"] is False
    assert first["generic_demand_m0"]["warm_value_gradient_seconds"] > 0.0
    assert second["generic_demand_m0"]["operator_materialized"] is False
    assert second["relaxation"]["warm_start_maximum_prediction_difference"] == 0
    assert (
        second["holdout"]["calibration_measurements"]
        + second["holdout"]["holdout_measurements"]
        == second["num_measurements"]
    )
