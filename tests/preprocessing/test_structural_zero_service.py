from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from public_transportation.domain.scenario import Scenario
from public_transportation.preprocessing import (
    ODTimeKey,
    fingerprint_scenario,
    run_structural_zero_preprocessing,
)


def _write_scenario(folder: Path) -> None:
    folder.mkdir()
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "title": "Service test",
                "description": "A one-way scheduled service",
                "timezone": "Europe/Zurich",
                "cost_unit": "minutes",
                "created_at": "2026-01-01T00:00:00",
                "sources": [],
                "extra": {},
            }
        ),
        encoding="utf-8",
    )
    (folder / "stops.csv").write_text(
        "stop_id,name,lat,lon\nA,Stop A,46.0,6.0\nB,Stop B,46.1,6.1\n",
        encoding="utf-8",
    )
    (folder / "lines.csv").write_text("line_id,short_name\nL1,1\n", encoding="utf-8")
    (folder / "time_bins.csv").write_text(
        "bin_id,start_s,end_s\nt0,28800,29400\n", encoding="utf-8"
    )
    # Deliberately provide only one candidate cell. The service must not add
    # structural zeros for the other Cartesian-product cells.
    (folder / "demand.csv").write_text(
        "origin_stop_id,dest_stop_id,time_bin_id,flow\nB,A,t0,10\n",
        encoding="utf-8",
    )
    (folder / "trips.csv").write_text("trip_id,line_id\nT1,L1\n", encoding="utf-8")
    (folder / "stop_times.csv").write_text(
        "trip_id,stop_id,sequence,arrival_s,departure_s\n"
        "T1,A,1,28859,28860\n"
        "T1,B,2,29100,29101\n",
        encoding="utf-8",
    )


def _write_config(folder: Path) -> Path:
    path = folder / "structural_zeros.toml"
    path.write_text(
        """\
version = 1

[scenario]
folder = "scenario"

[output]
folder = "outputs"
include_retained_cells_in_report = true

[rules.enabled]
same_stop = false
no_feasible_path = true
maximum_transfers = false
maximum_initial_wait = false
maximum_journey_time = false
minimum_feasible_departures = false

[rules.no_feasible_path]

[assignment]
max_access_deviation_minutes = 15.0
max_transfer_wait_minutes = 30.0
minimum_dwell_seconds = 1
""",
        encoding="utf-8",
    )
    return path


def test_end_to_end_service_uses_scenario_demand_as_candidate_universe(
    tmp_path: Path,
) -> None:
    _write_scenario(tmp_path / "scenario")
    config_path = _write_config(tmp_path)

    result = run_structural_zero_preprocessing(config_path)

    assert len(result.scenario_fingerprint) == 64
    assert result.analysis.scenario_fingerprint == result.scenario_fingerprint
    assert result.analysis.graph_fingerprint == result.topology.fingerprint
    assert result.analysis.configuration_fingerprint == result.config.fingerprint
    assert tuple(record.key for record in result.analysis.records) == (
        ODTimeKey("B", "A", "t0"),
    )
    assert result.analysis.num_structural_zero == 1
    assert result.reconciliation.num_merged == 1
    assert result.outputs.fixed_demand.read_text(encoding="utf-8").splitlines()[1] == (
        "B,A,t0,0"
    )

    repeated = run_structural_zero_preprocessing(config_path)
    assert repeated.scenario_fingerprint == result.scenario_fingerprint
    assert repeated.topology.fingerprint == result.topology.fingerprint
    assert repeated.outputs.artifact_sha256 == result.outputs.artifact_sha256


def test_service_progress_has_deterministic_phases_and_loop_finals(
    tmp_path: Path,
) -> None:
    _write_scenario(tmp_path / "scenario")
    events = []

    result = run_structural_zero_preprocessing(
        _write_config(tmp_path), progress=events.append
    )

    phases = [event.phase for event in events]
    first_positions = {phase: phases.index(phase) for phase in set(phases)}
    expected = (
        "load_scenario",
        "build_topology",
        "destination_profiles",
        "classify_cells",
        "reconcile_fixed_demand",
        "render_fixed_demand",
        "render_outputs",
        "render_summary",
        "write_outputs",
        "complete",
    )
    assert tuple(sorted(expected, key=first_positions.__getitem__)) == expected
    classified = [event for event in events if event.phase == "classify_cells"]
    assert classified[0].completed == 0
    assert classified[-1].completed == result.analysis.num_cells
    assert classified[-1].total == result.analysis.num_cells
    assert events[-1].phase == "complete"
    assert events[-1].completed == events[-1].total == 1


def test_scenario_fingerprint_is_order_independent_and_demand_sensitive(
    tmp_path: Path,
) -> None:
    scenario_folder = tmp_path / "scenario"
    _write_scenario(scenario_folder)
    scenario = Scenario.from_folder(scenario_folder, strict=True)
    baseline = fingerprint_scenario(scenario)
    scenario.metadata.created_at = "2099-12-31T23:59:59"
    assert fingerprint_scenario(scenario) == baseline

    scenario.stops.reverse()
    scenario.timetable.trips.reverse()
    scenario.timetable.stop_times.reverse()
    assert fingerprint_scenario(scenario) == baseline

    scenario.demand.records[0] = replace(scenario.demand.records[0], flow=11.0)
    assert fingerprint_scenario(scenario) != baseline


def test_scenario_loader_accepts_explicit_nonstandard_demand_file(
    tmp_path: Path,
) -> None:
    scenario_folder = tmp_path / "scenario"
    _write_scenario(scenario_folder)
    explicit_demand = tmp_path / "prior_demand.csv"
    (scenario_folder / "demand.csv").rename(explicit_demand)

    scenario = Scenario.from_folder(
        scenario_folder,
        strict=True,
        demand_file=explicit_demand,
    )

    assert len(scenario.demand.records) == 1
    assert scenario.demand.records[0].flow == 10.0
