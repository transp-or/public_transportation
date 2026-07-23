from __future__ import annotations

import os
from pathlib import Path

import pytest

from public_transportation.domain import Scenario, read_fixed_demand_csv
from public_transportation.inference.od_parameter_layout import (
    build_od_parameter_layout,
)
from public_transportation.preprocessing import run_structural_zero_preprocessing


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "docs/source/examples/geneva_gtfs"
DATA = EXAMPLE / "data"


@pytest.mark.skipif(
    os.environ.get("RUN_GENEVA_STRUCTURAL_ZERO_ACCEPTANCE") != "1",
    reason="Set RUN_GENEVA_STRUCTURAL_ZERO_ACCEPTANCE=1 for the Geneva acceptance run.",
)
def test_geneva_structural_zero_service_and_estimation_layout(tmp_path: Path) -> None:
    config = tmp_path / "structural_zeros.toml"
    config.write_text(
        f"""\
version = 1

[scenario]
folder = {str(DATA)!r}
demand_file = {str(DATA / "prior_demand.csv")!r}

[output]
folder = {str(tmp_path / "outputs")!r}
include_retained_cells_in_report = true

[rules.enabled]
same_stop = true
no_feasible_path = true
maximum_transfers = true
maximum_initial_wait = false
maximum_journey_time = false
minimum_feasible_departures = false

[rules.same_stop]

[rules.no_feasible_path]

[rules.maximum_transfers]
max_transfers = 2

[assignment]
max_access_deviation_minutes = 15.0
max_transfer_wait_minutes = 30.0
minimum_dwell_seconds = 1

[existing_fixed_demand]
file = {str(DATA / "fixed_demand.csv")!r}
""",
        encoding="utf-8",
    )

    result = run_structural_zero_preprocessing(config)

    assert result.analysis.num_cells == 15_128
    assert result.analysis.num_structural_zero == 0
    assert result.analysis.num_retained == 15_128
    assert result.reconciliation.num_existing == 15_032
    assert result.reconciliation.num_added_structural_zero == 0
    assert (
        result.outputs.fixed_demand.read_bytes()
        == (DATA / "fixed_demand.csv").read_bytes()
    )

    scenario = Scenario.from_folder(
        DATA,
        strict=True,
        demand_file=DATA / "prior_demand.csv",
    )
    fixed = read_fixed_demand_csv(result.outputs.fixed_demand, scenario=scenario)
    layout = build_od_parameter_layout(scenario=scenario, fixed_demand=fixed)
    assert layout.num_od_total == 15_128
    assert layout.num_fixed == 15_032
    assert layout.num_fixed_zero == 15_032
    assert layout.num_free == 96
