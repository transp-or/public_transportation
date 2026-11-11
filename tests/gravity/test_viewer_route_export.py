from __future__ import annotations

import csv

from public_transportation.inference.gravity import (
    export_gravity_route_flows,
    plan_gravity_route_flow_export,
)


def _row() -> dict[str, object]:
    return {
        "origin": "o1",
        "destination": "d1",
        "departure_time_bin": "am",
        "line": "L1",
        "trip": "T1",
        "assigned_demand": 1.5,
        "path_share": 1.0,
        "transfer_count": 0,
        "flow_semantics": "path_probability",
    }


def test_route_export_plan_fails_closed_on_fingerprint_mismatch(tmp_path):
    plan = plan_gravity_route_flow_export(
        routing_cache=tmp_path / "cache",
        expected_assignment_fingerprint="expected",
        cache_metadata={"assignment_fingerprint": "different"},
    )
    assert not plan.available
    assert "fingerprint" in plan.reason


def test_route_export_streams_rows_and_checkpoints(tmp_path):
    progress: list[int] = []
    output = tmp_path / "route_flows.csv"
    checkpoint = tmp_path / "route_flows.checkpoint.json"
    result = export_gravity_route_flows(
        route_flow_rows=(_row(), _row()),
        output_path=output,
        checkpoint_path=checkpoint,
        checkpoint_every_rows=1,
        progress=progress.append,
    )
    assert result["rows_written"] == 2
    assert progress[-1] == 2
    assert '"status": "completed"' in checkpoint.read_text(encoding="utf-8")
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["flow_semantics"] == "path_probability"
    assert rows[0]["assigned_demand"] == "1.5"
