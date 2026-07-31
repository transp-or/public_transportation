from __future__ import annotations

from pathlib import Path

import pytest

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.domain import Scenario, Severity
from public_transportation.inference.assignment_adapter import build_assignment_inputs
from public_transportation.inference.block_coordinate import (
    BlockSizingConfig,
    partition_assignment_od_blocks,
    require_measurements_for_block_estimation,
)
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.od_parameter_layout import build_od_parameter_layout


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "docs/source/examples/network_model/data"


def test_network_model_loads_prepares_assignment_and_partitions_without_measurements():
    scenario = Scenario.from_folder(EXAMPLE)
    errors = [
        issue for issue in scenario.validate().issues if issue.severity == Severity.ERROR
    ]
    # This legacy public example uses zero dwell times. Loading and structural
    # partitioning remain applicable, while strict validation reports that
    # known timetable issue instead of silently treating the example as valid.
    assert errors
    assert {issue.code for issue in errors} == {"STOPTIME_DEPART_NOT_AFTER_ARRIVE"}
    layout = build_od_parameter_layout(scenario=scenario)
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
    inputs = build_assignment_inputs(artifacts=artifacts, compact_layout=compact)
    partition = partition_assignment_od_blocks(
        inputs=inputs,
        parameter_layout=layout,
        compact_layout=compact,
        sizing=BlockSizingConfig(
            mode="explicit", maximum_free_variables_per_block=2
        ),
    )
    assert partition.num_free_variables == layout.num_free
    assert all(block.num_free_variables <= 2 for block in partition.blocks)
    assert partition.fingerprint == partition_assignment_od_blocks(
        inputs=inputs,
        parameter_layout=layout,
        compact_layout=compact,
        sizing=BlockSizingConfig(
            mode="explicit", maximum_free_variables_per_block=2
        ),
    ).fingerprint

    with pytest.raises(ValueError, match="requires at least one measurement"):
        require_measurements_for_block_estimation(0)
