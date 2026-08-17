from __future__ import annotations

import json
from pathlib import Path

from public_transportation.domain import (
    Metadata,
    ODDemand,
    Scenario,
    Stop,
    StopTime,
    TimeBin,
    TimeOfDay,
    Timetable,
    Trip,
)
from public_transportation.domain.line import Line
from public_transportation.preprocessing import (
    CandidateODPair,
    CandidateODUniverse,
    ODTimeKey,
    ScheduledFeasibilityContract,
    TimetableFeasibilityIndex,
    build_structural_zero_topology,
    compute_od_path_metrics,
    run_candidate_od_time_expansion,
)
from public_transportation.preprocessing.structural_zeros.config import (
    StructuralZeroAssignmentConfig,
)


def _late_transfer_scenario() -> Scenario:
    """A later-bin path whose transfer wait is 45 minutes.

    The historical structural-zero topology defaulted to a 30-minute transfer
    wait, while bootstrap-prior allows the configured 60-minute wait.  This is
    the same semantic shape as the medium-case ``t8`` mismatch.
    """
    return Scenario(
        metadata=Metadata("scheduled-feasibility-contract-test"),
        stops=[
            Stop("A", "A", 0.0, 0.0),
            Stop("X", "X", 0.0, 0.1),
            Stop("C", "C", 0.0, 0.2),
        ],
        lines=[Line("L1"), Line("L2")],
        time_bins=[TimeBin("t8", TimeOfDay(8 * 3600), TimeOfDay(9 * 3600))],
        demand=ODDemand([]),
        timetable=Timetable(
            trips=[Trip("T1", "L1"), Trip("T2", "L2")],
            stop_times=[
                StopTime(
                    "T1",
                    "A",
                    1,
                    TimeOfDay(8 * 3600 + 5 * 60),
                    TimeOfDay(8 * 3600 + 5 * 60),
                ),
                StopTime(
                    "T1",
                    "X",
                    2,
                    TimeOfDay(8 * 3600 + 15 * 60),
                    TimeOfDay(8 * 3600 + 15 * 60),
                ),
                StopTime("T2", "X", 1, TimeOfDay(9 * 3600), TimeOfDay(9 * 3600)),
                StopTime(
                    "T2",
                    "C",
                    2,
                    TimeOfDay(9 * 3600 + 10 * 60),
                    TimeOfDay(9 * 3600 + 10 * 60),
                ),
            ],
        ),
    )


def _universe() -> CandidateODUniverse:
    return CandidateODUniverse(
        pairs=(CandidateODPair("A", "C"),),
        exclusions=(),
        source="file",
        level="stop",
        include_same_stop=False,
        active_service_only=False,
        connectivity_policy="none",
        physical_stop_mapping={"A": "A", "X": "X", "C": "C"},
        generator_fingerprint="scheduled-contract-fixture",
    )


def test_bootstrap_and_feature_support_share_late_transfer_semantics(
    tmp_path: Path,
) -> None:
    scenario = _late_transfer_scenario()
    universe = _universe()
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
    expansion = run_candidate_od_time_expansion(
        universe,
        [("t8", 8 * 3600, 9 * 3600)],
        scenario=scenario,
        configuration=configuration,
        checkpoint_directory=tmp_path / "checkpoint",
    )
    assert expansion.retained_cells == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "checkpoint/chunk-000000.jsonl")
        .read_text()
        .splitlines()
    ]
    assert rows[0]["status"] == "retained"

    contract = ScheduledFeasibilityContract.from_mapping(configuration)
    index = TimetableFeasibilityIndex.from_scenario(scenario)
    period = ("t8", 8 * 3600, 9 * 3600)
    metrics = contract.path_metrics(index, origin="A", period=period)
    assert "C" in metrics
    assert metrics["C"].minimum_transfers == 1
    assert metrics["C"].minimum_initial_wait_seconds == 5 * 60

    # This is the old feature-construction behavior: the default graph
    # topology rejects the 45-minute transfer, reproducing the mismatch.
    old_topology = build_structural_zero_topology(
        scenario, StructuralZeroAssignmentConfig()
    )
    old_record = compute_od_path_metrics(
        old_topology, keys=(ODTimeKey("A", "C", "t8"),)
    )[0]
    assert not old_record.metrics.feasible


def test_contract_fingerprint_changes_when_feasibility_semantics_change() -> None:
    base = ScheduledFeasibilityContract()
    changed = ScheduledFeasibilityContract(maximum_waiting_seconds=1800)
    assert base.fingerprint != changed.fingerprint
