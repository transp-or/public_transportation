from __future__ import annotations

import json
from pathlib import Path

import pytest

from public_transportation.domain import Metadata, ODDemand, Scenario, Stop, StopTime, TimeBin, TimeOfDay, Timetable, Trip
from public_transportation.domain.line import Line
from public_transportation.preprocessing.od_universe import (
    ODTimeExpansionInterrupted,
    generate_candidate_od_pairs,
    run_candidate_od_time_expansion,
)



def _scenario() -> Scenario:
    stops = [Stop("A", "A", 0.0, 0.0), Stop("B", "B", 0.0, 0.1), Stop("C", "C", 0.0, 0.2)]
    timetable = Timetable(
        trips=[Trip("T1", "L1")],
        stop_times=[
            StopTime("T1", "A", 1, TimeOfDay(8 * 3600), TimeOfDay(8 * 3600)),
            StopTime("T1", "B", 2, TimeOfDay(8 * 3600 + 600), TimeOfDay(8 * 3600 + 600)),
            StopTime("T1", "C", 3, TimeOfDay(8 * 3600 + 1200), TimeOfDay(8 * 3600 + 1200)),
        ],
    )
    return Scenario(
        metadata=Metadata("checkpoint-test"),
        stops=stops,
        lines=[Line("L1")],
        time_bins=[TimeBin("source", TimeOfDay(8 * 3600), TimeOfDay(9 * 3600))],
        demand=ODDemand([]),
        timetable=timetable,
    )


def _inputs() -> tuple[object, object, list[tuple[str, int, int]], dict[str, object]]:
    scenario = _scenario()
    universe = generate_candidate_od_pairs(
        scenario,
        source="network_ordered_pairs",
        active_service_only=True,
        connectivity_policy="directed_reachable",
    )
    bins = [("morning", 8 * 3600, 9 * 3600), ("later", 9 * 3600, 10 * 3600)]
    configuration = {
        "chunk_size_pairs": 1,
        "progress_interval_seconds": 0.001,
        "timetable_policy": "required",
        "maximum_transfers": 2,
        "maximum_initial_wait_seconds": 3600,
        "maximum_journey_seconds": 7200,
        "maximum_waiting_seconds": 3600,
        "package_revision": "test-revision",
    }
    return scenario, universe, bins, configuration


def test_checkpoint_emits_progress_and_completes_atomically(tmp_path: Path) -> None:
    scenario, universe, bins, configuration = _inputs()
    events: list[dict[str, object]] = []
    result = run_candidate_od_time_expansion(
        universe,
        bins,
        scenario=scenario,
        configuration=configuration,
        checkpoint_directory=tmp_path / "checkpoint",
        progress=lambda event: events.append(dict(event)),
    )
    assert result.status == "completed"
    assert result.completed_chunks == result.total_chunks
    assert [event["status"] for event in events][0] == "started"
    assert events[-1]["status"] == "completed"
    assert events[-1]["checkpoint_reusable"] is False
    assert events[-1]["eta_seconds"] is None or events[-1]["eta_seconds"] >= 0
    manifest = json.loads((result.checkpoint_directory / "manifest.json").read_text())
    progress = json.loads((result.checkpoint_directory / "progress.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["semantic_checksum"] == result.semantic_checksum
    assert progress["status"] == "completed"
    assert not list(result.checkpoint_directory.glob("*.tmp"))


def test_sigint_preserves_completed_chunks_and_resume_is_semantically_identical(tmp_path: Path) -> None:
    scenario, universe, bins, configuration = _inputs()
    checkpoint = tmp_path / "interrupted"

    def interrupt(event: dict[str, object]) -> None:
        if event.get("status") == "running":
            raise KeyboardInterrupt

    with pytest.raises(ODTimeExpansionInterrupted):
        run_candidate_od_time_expansion(
            universe,
            bins,
            scenario=scenario,
            configuration=configuration,
            checkpoint_directory=checkpoint,
            progress=interrupt,
        )
    interrupted = json.loads((checkpoint / "manifest.json").read_text())
    assert interrupted["status"] == "interrupted"
    assert interrupted["completed_chunks"]
    resumed = run_candidate_od_time_expansion(
        universe,
        bins,
        scenario=scenario,
        configuration=configuration,
        checkpoint_directory=checkpoint,
        resume=True,
    )
    fresh = run_candidate_od_time_expansion(
        universe,
        bins,
        scenario=scenario,
        configuration=configuration,
        checkpoint_directory=tmp_path / "fresh",
    )
    assert resumed.checkpoint_reused is True
    assert resumed.semantic_checksum == fresh.semantic_checksum
    assert resumed.expansion_fingerprint == fresh.expansion_fingerprint


def test_checkpoint_requires_explicit_resume_and_rejects_corruption(tmp_path: Path) -> None:
    scenario, universe, bins, configuration = _inputs()
    checkpoint = tmp_path / "checkpoint"
    run_candidate_od_time_expansion(
        universe,
        bins,
        scenario=scenario,
        configuration=configuration,
        checkpoint_directory=checkpoint,
    )
    with pytest.raises(FileExistsError, match="--resume"):
        run_candidate_od_time_expansion(
            universe,
            bins,
            scenario=scenario,
            configuration=configuration,
            checkpoint_directory=checkpoint,
        )
    chunk = next(checkpoint.glob("chunk-*.jsonl"))
    chunk.write_text(chunk.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        run_candidate_od_time_expansion(
            universe,
            bins,
            scenario=scenario,
            configuration=configuration,
            checkpoint_directory=checkpoint,
            resume=True,
        )


def test_changed_contract_rejects_resume(tmp_path: Path) -> None:
    scenario, universe, bins, configuration = _inputs()
    checkpoint = tmp_path / "checkpoint"
    run_candidate_od_time_expansion(
        universe,
        bins,
        scenario=scenario,
        configuration=configuration,
        checkpoint_directory=checkpoint,
    )
    changed = {**configuration, "chunk_size_pairs": 2}
    with pytest.raises(ValueError, match="fingerprint"):
        run_candidate_od_time_expansion(
            universe,
            bins,
            scenario=scenario,
            configuration=changed,
            checkpoint_directory=checkpoint,
            resume=True,
        )
