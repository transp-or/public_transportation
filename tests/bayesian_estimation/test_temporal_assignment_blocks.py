from __future__ import annotations

import numpy as np
import pytest

from public_transportation.inference.assignment_contract import (
    AssignmentArtifactIdentity,
    CanonicalAssignmentIndex,
    CanonicalMeasurement,
    CanonicalODTimeCell,
    CanonicalTimeInterval,
)
from public_transportation.inference.temporal_assignment_blocks import (
    TemporalBlockAssignmentOperator,
    TemporalBlockConstructionDiagnostics,
    TemporalBlockKey,
    TemporalSparseBlock,
)


def _operator(
    blocks: tuple[TemporalSparseBlock, ...],
    *,
    progress=None,
) -> TemporalBlockAssignmentOperator:
    canonical_index = CanonicalAssignmentIndex(
        time_intervals=(CanonicalTimeInterval("period", 0, 3600),),
        demand_cells=tuple(
            CanonicalODTimeCell(
                full_index=index,
                origin_id=f"origin-{index}",
                destination_id="destination",
                departure_interval_id="period",
                role="free",
                operator_column=index,
            )
            for index in range(2)
        ),
        measurements=tuple(
            CanonicalMeasurement(
                row_index=index,
                measurement_id=f"measurement-{index}",
                event="boarding",
                location_id=f"stop-{index}",
                interval_id="period",
            )
            for index in range(2)
        ),
    )
    identity = AssignmentArtifactIdentity(
        canonical_index_fingerprint=canonical_index.artifact_fingerprint,
        network_fingerprint="network",
        timetable_fingerprint="timetable",
        temporal_discretization_fingerprint="temporal",
        route_choice_fingerprint="route-choice",
        departure_choice_fingerprint="departure-choice",
        feasibility_fingerprint="feasibility",
        measurement_mapping_fingerprint="measurement-mapping",
        coefficient_policy_fingerprint="coefficient-policy",
        numeric_dtype="float32",
    )
    return TemporalBlockAssignmentOperator(
        canonical_index=canonical_index,
        identity=identity,
        blocks=blocks,
        fixed_measurement_offset=np.zeros(2, dtype=np.float32),
        diagnostics=TemporalBlockConstructionDiagnostics(
            construction_seconds=0.0,
            nonzero_entries=0,
            retained_l1_mass=0.0,
            removed_l1_mass=0.0,
            zero_tolerance=0.0,
            columns_processed=0,
        ),
        _construction_progress=progress,
    )


def _block(
    *,
    number_of_measurements: int = 2,
    number_of_demand_cells: int = 2,
    key_index: int = 0,
) -> TemporalSparseBlock:
    return TemporalSparseBlock(
        key=TemporalBlockKey(f"measurement-{key_index}", "period"),
        row_indices=np.empty(0, dtype=np.int32),
        column_indices=np.empty(0, dtype=np.int32),
        values=np.empty(0, dtype=np.float32),
        number_of_measurements=number_of_measurements,
        number_of_demand_cells=number_of_demand_cells,
    )


@pytest.mark.parametrize("reporting", [False, True])
@pytest.mark.parametrize(
    ("dimension", "expected_message"),
    [
        ("measurement", "temporal block measurement dimension differs."),
        ("demand", "temporal block demand dimension differs."),
    ],
)
def test_dimension_mismatches_are_rejected_with_or_without_reporting(
    reporting: bool,
    dimension: str,
    expected_message: str,
) -> None:
    block = _block(
        number_of_measurements=3 if dimension == "measurement" else 2,
        number_of_demand_cells=3 if dimension == "demand" else 2,
    )
    events: list[tuple[str, int | None, int | None]] = []
    progress = None
    if reporting:
        def report(phase, completed, total):
            events.append((phase, completed, total))

        progress = report

    with pytest.raises(ValueError, match=expected_message):
        _operator((block,), progress=progress)

    if reporting:
        assert events == [("validate_temporal_operator_blocks", 0, 1)]
    else:
        assert events == []


@pytest.mark.parametrize("block_count", [1, 8])
def test_expected_dimensions_are_read_once_per_nonempty_validation(
    monkeypatch: pytest.MonkeyPatch,
    block_count: int,
) -> None:
    counts = {"measurements": 0, "demand_cells": 0}
    original_measurements = CanonicalAssignmentIndex.number_of_measurements.fget
    original_demand_cells = CanonicalAssignmentIndex.number_of_demand_cells.fget
    assert original_measurements is not None
    assert original_demand_cells is not None

    def count_measurements(index: CanonicalAssignmentIndex) -> int:
        counts["measurements"] += 1
        return original_measurements(index)

    def count_demand_cells(index: CanonicalAssignmentIndex) -> int:
        counts["demand_cells"] += 1
        return original_demand_cells(index)

    monkeypatch.setattr(
        CanonicalAssignmentIndex,
        "number_of_measurements",
        property(count_measurements),
    )
    monkeypatch.setattr(
        CanonicalAssignmentIndex,
        "number_of_demand_cells",
        property(count_demand_cells),
    )

    _operator(tuple(_block(key_index=index) for index in range(block_count)))

    assert counts == {"measurements": 2, "demand_cells": 1}


def test_reporting_progress_cadence_is_unchanged() -> None:
    events: list[tuple[str, int | None, int | None]] = []
    _operator(
        tuple(_block(key_index=index) for index in range(4097)),
        progress=lambda phase, completed, total: events.append(
            (phase, completed, total)
        ),
    )

    assert events == [
        ("validate_temporal_operator_blocks", 0, 4097),
        ("validate_temporal_operator_blocks", 4096, 4097),
        ("validate_temporal_operator_blocks", 4097, 4097),
        ("copy_and_validate_fixed_measurement_offset", None, None),
    ]


def test_empty_blocks_skip_dimension_reads_but_keep_reporting_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = CanonicalAssignmentIndex.number_of_demand_cells.fget
    assert original is not None

    def count_demand_cells(index: CanonicalAssignmentIndex) -> int:
        nonlocal calls
        calls += 1
        return original(index)

    monkeypatch.setattr(
        CanonicalAssignmentIndex,
        "number_of_demand_cells",
        property(count_demand_cells),
    )
    events: list[tuple[str, int | None, int | None]] = []

    _operator(
        (),
        progress=lambda phase, completed, total: events.append(
            (phase, completed, total)
        ),
    )

    assert calls == 0
    assert events == [
        ("validate_temporal_operator_blocks", 0, 0),
        ("copy_and_validate_fixed_measurement_offset", None, None),
    ]
