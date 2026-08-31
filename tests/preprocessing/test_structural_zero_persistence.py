from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from public_transportation.domain.fixed_demand import FixedODDemand, FixedODRecord
from public_transportation.preprocessing import (
    ODPathMetrics,
    ODTimeKey,
    StructuralZeroAnalysisResult,
    StructuralZeroReason,
    StructuralZeroRecord,
    reconcile_fixed_demand,
    write_structural_zero_outputs,
)
from public_transportation.preprocessing.structural_zeros.config import (
    EnabledRulesConfig,
    NoFeasiblePathRuleConfig,
    OutputConfig,
    RulesConfig,
    ScenarioConfig,
    StructuralZeroAssignmentConfig,
    StructuralZeroConfig,
)


def _config(tmp_path: Path, *, include_retained: bool = True) -> StructuralZeroConfig:
    scenario_folder = tmp_path / "scenario"
    scenario_folder.mkdir(exist_ok=True)
    return StructuralZeroConfig(
        version=1,
        source_file=tmp_path / "input.toml",
        scenario=ScenarioConfig(folder=scenario_folder),
        output=OutputConfig(
            folder=tmp_path / "outputs",
            include_retained_cells_in_report=include_retained,
        ),
        rules=RulesConfig(
            enabled=EnabledRulesConfig(
                same_stop=False,
                no_feasible_path=True,
                maximum_transfers=False,
                maximum_initial_wait=False,
                maximum_journey_time=False,
                minimum_feasible_departures=False,
            ),
            same_stop=None,
            no_feasible_path=NoFeasiblePathRuleConfig(),
            maximum_transfers=None,
            maximum_initial_wait=None,
            maximum_journey_time=None,
            minimum_feasible_departures=None,
        ),
        assignment=StructuralZeroAssignmentConfig(),
        existing_fixed_demand=None,
    )


def _analysis(config: StructuralZeroConfig) -> StructuralZeroAnalysisResult:
    zero = StructuralZeroRecord(
        key=ODTimeKey("B", "A", "t0"),
        is_structural_zero=True,
        primary_reason=StructuralZeroReason.NO_FEASIBLE_PATH,
        triggered_rules=(StructuralZeroReason.NO_FEASIBLE_PATH,),
        metrics=ODPathMetrics.unreachable(),
    )
    retained = StructuralZeroRecord(
        key=ODTimeKey("A", "B", "t0"),
        is_structural_zero=False,
        primary_reason=None,
        triggered_rules=(),
        metrics=ODPathMetrics(
            feasible=True,
            minimum_transfers=0,
            minimum_initial_wait_minutes=2.0,
            minimum_journey_time_minutes=10.0,
            feasible_departure_count=3,
            earliest_arrival_seconds=30_000,
        ),
    )
    return StructuralZeroAnalysisResult(
        records=(retained, zero),
        scenario_fingerprint="scenario-fingerprint",
        graph_fingerprint="graph-fingerprint",
        configuration_fingerprint=config.fingerprint,
    )


def _reconciliation(analysis: StructuralZeroAnalysisResult):
    return reconcile_fixed_demand(
        analysis,
        FixedODDemand(records=(FixedODRecord("A", "B", "t0", 12.5),)),
    )


def test_complete_artifact_set_is_written_deterministically(tmp_path: Path) -> None:
    config = _config(tmp_path)
    analysis = _analysis(config)
    result = write_structural_zero_outputs(analysis, _reconciliation(analysis), config)

    assert result.folder == config.output.folder
    paths = (
        result.fixed_demand,
        result.audit,
        result.summary,
        result.fingerprints,
        result.resolved_config,
    )
    assert all(path.is_file() for path in paths)
    assert all(path.stat().st_mode & 0o777 == 0o644 for path in paths)

    with result.fixed_demand.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {
            "origin_stop_id": "A",
            "dest_stop_id": "B",
            "time_bin_id": "t0",
            "fixed_flow": "12.5",
        },
        {
            "origin_stop_id": "B",
            "dest_stop_id": "A",
            "time_bin_id": "t0",
            "fixed_flow": "0",
        },
    ]

    with result.audit.open(encoding="utf-8", newline="") as stream:
        audit_rows = list(csv.DictReader(stream))
    assert len(audit_rows) == 2
    assert audit_rows[0]["is_structural_zero"] == "false"
    assert audit_rows[1]["primary_reason"] == "no_feasible_path"
    assert audit_rows[1]["minimum_transfers"] == ""

    summary = json.loads(result.summary.read_text(encoding="utf-8"))
    assert summary["num_cells"] == 2
    assert summary["num_structural_zero"] == 1
    assert summary["reconciliation"]["num_merged"] == 2

    manifest = json.loads(result.fingerprints.read_text(encoding="utf-8"))
    assert manifest["scenario_fingerprint"] == "scenario-fingerprint"
    assert manifest["configuration_fingerprint"] == config.fingerprint
    for name, expected_hash in result.artifact_sha256:
        assert manifest["artifact_sha256"][name] == expected_hash
        assert (
            hashlib.sha256((result.folder / name).read_bytes()).hexdigest()
            == expected_hash
        )
    assert (
        result.resolved_config.read_text(encoding="utf-8") == config.to_resolved_toml()
    )

    first_payloads = {path.name: path.read_bytes() for path in paths}
    write_structural_zero_outputs(analysis, _reconciliation(analysis), config)
    assert {path.name: path.read_bytes() for path in paths} == first_payloads
    assert not list(result.folder.glob(".*.tmp"))


def test_audit_can_exclude_retained_cells_without_changing_summary(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, include_retained=False)
    analysis = _analysis(config)
    result = write_structural_zero_outputs(analysis, _reconciliation(analysis), config)

    with result.audit.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["is_structural_zero"] == "true"
    summary = json.loads(result.summary.read_text(encoding="utf-8"))
    assert summary["num_cells"] == 2
    assert not summary["audit_includes_retained_cells"]


def test_validation_failure_has_no_filesystem_side_effect(tmp_path: Path) -> None:
    config = _config(tmp_path)
    analysis = replace(
        _analysis(config), configuration_fingerprint="different-configuration"
    )

    with pytest.raises(ValueError, match="configuration fingerprint"):
        write_structural_zero_outputs(analysis, _reconciliation(analysis), config)

    assert not config.output.folder.exists()


def test_audit_render_progress_ends_at_written_row_count(tmp_path: Path) -> None:
    config = _config(tmp_path)
    analysis = _analysis(config)
    events = []

    write_structural_zero_outputs(
        analysis, _reconciliation(analysis), config, progress=events.append
    )

    audit = [
        event
        for event in events
        if event.phase == "render_outputs"
        and event.message == "structural_zero_audit.csv"
    ]
    assert [event.completed for event in audit] == [0, 1, 2]
    assert audit[-1].total == 2


def test_callback_failure_during_render_is_observability_only(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    analysis = _analysis(config)
    original = write_structural_zero_outputs(
        analysis, _reconciliation(analysis), config
    )
    before = original.audit.read_bytes()

    def fail(event) -> None:
        if event.phase == "render_outputs" and event.completed == 1:
            raise RuntimeError("cancel rendering")

    write_structural_zero_outputs(
        analysis, _reconciliation(analysis), config, progress=fail
    )

    assert original.audit.read_bytes() == before
