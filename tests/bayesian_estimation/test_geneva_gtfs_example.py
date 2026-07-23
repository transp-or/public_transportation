from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import assign, prepare_assignment
from public_transportation.domain import Scenario, read_fixed_demand_csv
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.od_parameter_layout import build_od_parameter_layout


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "source" / "examples" / "geneva_gtfs" / "data"
NETWORK_FILES = ("metadata.json", "stops.csv", "lines.csv", "trips.csv", "stop_times.csv", "time_bins.csv")


def _load_scenario(tmp_path: Path, demand_name: str) -> Scenario:
    for name in NETWORK_FILES:
        shutil.copy2(DATA / name, tmp_path / name)
    shutil.copy2(DATA / demand_name, tmp_path / "demand.csv")
    return Scenario.from_folder(tmp_path, strict=True)


def test_geneva_snapshot_provenance_and_dimensions(tmp_path: Path) -> None:
    provenance = json.loads((DATA / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["archive_sha256"] == (
        "c6f06bdad9f20349ed08b45daf2ff6114f116a3c231afdd48abe80608382c5dd"
    )
    assert provenance["agency_id"] == "881"
    assert provenance["selected_lines"] == ["12", "14", "18"]

    scenario = _load_scenario(tmp_path, "prior_demand.csv")
    assert len(scenario.stops) == 62
    assert len(scenario.lines) == 3
    assert len(scenario.timetable.trips) == 173
    assert len(scenario.timetable.stop_times) == 4754
    assert len(scenario.time_bins) == 4
    assert len(scenario.demand.records) == 15128


def test_geneva_frozen_cells_are_absent_from_parameter_and_compact_layout(tmp_path: Path) -> None:
    scenario = _load_scenario(tmp_path, "prior_demand.csv")
    fixed = read_fixed_demand_csv(DATA / "fixed_demand.csv", scenario=scenario)
    layout = build_od_parameter_layout(scenario=scenario, fixed_demand=fixed)
    compact = build_compact_od_assignment_layout(parameter_layout=layout)

    assert layout.num_od_total == 15128
    assert layout.num_free == 96
    assert layout.num_fixed == 15032
    assert layout.num_fixed_zero == 15032
    assert compact.num_active == 96
    assert compact.num_free == 96
    assert np.all(np.asarray(compact.active_full_indices) == np.asarray(layout.free_od_indices))


def test_geneva_true_and_prior_assign_on_identical_network(tmp_path: Path) -> None:
    scenario = _load_scenario(tmp_path, "prior_demand.csv")
    artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
    prior = jnp.asarray([record.flow for record in scenario.demand.records], dtype=jnp.float32)
    prior_result = assign(od_values=prior, artifacts=artifacts, theta=5.0)

    true_by_key = {
        (record.origin_stop_id, record.dest_stop_id, record.time_bin_id): record.flow
        for record in _load_scenario(tmp_path, "true_demand.csv").demand.records
    }
    true = jnp.asarray(
        [
            true_by_key[(record.origin_stop_id, record.dest_stop_id, record.time_bin_id)]
            for record in scenario.demand.records
        ],
        dtype=jnp.float32,
    )
    true_result = assign(od_values=true, artifacts=artifacts, theta=5.0)

    assert prior_result.link_flow.shape == true_result.link_flow.shape
    assert np.all(np.isfinite(np.asarray(prior_result.link_flow)))
    assert np.all(np.isfinite(np.asarray(true_result.link_flow)))
    assert not np.allclose(np.asarray(prior_result.link_flow), np.asarray(true_result.link_flow))


def test_geneva_preprocessing_keeps_integer_zero_and_positive_counts() -> None:
    path = DATA.parent / "pre_processing" / "results" / "measurements_boarding_alighting.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        values = [float(row["value"]) for row in csv.DictReader(stream)]

    assert len(values) == 8967
    assert any(value == 0.0 for value in values)
    assert any(value > 0.0 for value in values)
    assert all(value >= 0.0 and value.is_integer() for value in values)


def test_geneva_method_comparison_contains_all_methods_and_vi_coverage() -> None:
    path = DATA.parent / "post_processing" / "results" / "method_comparison.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    methods = {row["method"]: row for row in payload["methods"]}

    assert set(methods) == {"ML", "MAP", "VI"}
    assert methods["MAP"]["success"] is True
    assert 0.0 <= methods["VI"]["coverage_90_active"] <= 1.0
    assert methods["VI"]["mean_interval_width_90_active"] > 0.0
    assert methods["VI"]["runtime_seconds"] > methods["MAP"]["runtime_seconds"]
