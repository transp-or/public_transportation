from __future__ import annotations

import pytest

from public_transportation.inference.block_coordinate import (
    BlockSizingConfig,
    ODBlock,
    partition_od_blocks,
    validate_block_partition,
)


def test_partition_groups_by_destination_and_time_then_splits_deterministically():
    kwargs = {
        "free_to_active_indices": (0, 1, 3, 4, 7, 8),
        "destination_group_by_free_column": (0, 0, 0, 1, 1, 1),
        "time_bin_by_free_column": ("am", "am", "pm", "am", "am", "am"),
        "sizing": BlockSizingConfig(
            mode="explicit", maximum_free_variables_per_block=2
        ),
    }
    first = partition_od_blocks(**kwargs)
    repeated = partition_od_blocks(**kwargs)
    assert first == repeated
    assert first.fingerprint == repeated.fingerprint
    assert [block.free_column_indices for block in first.blocks] == [
        (0, 1),
        (2,),
        (3, 4),
        (5,),
    ]
    assert all(block.num_free_variables <= 2 for block in first.blocks)


def test_nonzero_ceiling_is_hard_and_support_is_aggregated():
    partition = partition_od_blocks(
        free_to_active_indices=(0, 1, 2),
        destination_group_by_free_column=(0, 0, 0),
        time_bin_by_free_column=("am", "am", "am"),
        sizing=BlockSizingConfig(
            mode="explicit",
            maximum_free_variables_per_block=10,
            maximum_operator_nonzeros_per_block=7,
        ),
        estimated_nonzeros_by_free_column=(3, 4, 2),
        measurement_support_by_free_column=((0, 1), (1, 2), (4,)),
    )
    assert [block.estimated_nonzeros for block in partition.blocks] == [7, 2]
    assert partition.blocks[0].measurement_support_indices == (0, 1, 2)
    assert partition.blocks[1].measurement_support_indices == (4,)

    with pytest.raises(ValueError, match="by itself"):
        partition_od_blocks(
            free_to_active_indices=(0,),
            destination_group_by_free_column=(0,),
            time_bin_by_free_column=("am",),
            sizing=BlockSizingConfig(
                mode="explicit", maximum_operator_nonzeros_per_block=2
            ),
            estimated_nonzeros_by_free_column=(3,),
        )


def test_partition_rejects_unenforceable_or_missing_estimates():
    with pytest.raises(ValueError, match="requires per-column estimates"):
        partition_od_blocks(
            free_to_active_indices=(0,),
            destination_group_by_free_column=(0,),
            time_bin_by_free_column=("am",),
            sizing=BlockSizingConfig(
                mode="explicit", maximum_operator_nonzeros_per_block=10
            ),
        )
    with pytest.raises(ValueError, match="memory-only"):
        partition_od_blocks(
            free_to_active_indices=(0,),
            destination_group_by_free_column=(0,),
            time_bin_by_free_column=("am",),
            sizing=BlockSizingConfig(
                mode="explicit", maximum_worker_memory_bytes=1_000
            ),
        )


def test_empty_problem_produces_empty_valid_partition():
    partition = partition_od_blocks(
        free_to_active_indices=(),
        destination_group_by_free_column=(),
        time_bin_by_free_column=(),
        sizing=BlockSizingConfig(
            mode="explicit", maximum_free_variables_per_block=2
        ),
    )
    assert partition.blocks == ()
    assert partition.num_free_variables == 0


def test_user_partition_validation_detects_overlap_missing_and_frozen_cells():
    first = ODBlock("a", (0,), (2,), (0,), ("am",))
    second = ODBlock("b", (1,), (4,), (1,), ("am",))
    valid = validate_block_partition(
        (first, second), free_to_active_indices=(2, 4), frozen_active_indices=(3,)
    )
    assert valid.num_blocks == 2

    overlap = ODBlock("overlap", (0,), (2,), (0,), ("am",))
    with pytest.raises(ValueError, match="more than one"):
        validate_block_partition(
            (first, overlap), free_to_active_indices=(2, 4)
        )
    with pytest.raises(ValueError, match="cover every free column"):
        validate_block_partition((first,), free_to_active_indices=(2, 4))
    with pytest.raises(ValueError, match="disjoint"):
        validate_block_partition(
            (first, second),
            free_to_active_indices=(2, 4),
            frozen_active_indices=(4,),
        )


def test_partition_fingerprint_changes_with_membership_or_order():
    sizing = BlockSizingConfig(mode="explicit", maximum_free_variables_per_block=2)
    one = partition_od_blocks(
        free_to_active_indices=(0, 1, 2),
        destination_group_by_free_column=(0, 0, 0),
        time_bin_by_free_column=("am", "am", "am"),
        sizing=sizing,
    )
    two = partition_od_blocks(
        free_to_active_indices=(0, 1, 2),
        destination_group_by_free_column=(0, 1, 1),
        time_bin_by_free_column=("am", "am", "am"),
        sizing=sizing,
    )
    assert one.fingerprint != two.fingerprint


def test_optional_merging_respects_hard_ceiling():
    partition = partition_od_blocks(
        free_to_active_indices=(0, 1, 2),
        destination_group_by_free_column=(0, 1, 2),
        time_bin_by_free_column=("am", "am", "am"),
        sizing=BlockSizingConfig(
            mode="explicit", maximum_free_variables_per_block=2
        ),
        merge_small_compatible=True,
    )
    assert [block.free_column_indices for block in partition.blocks] == [(0, 1), (2,)]
    assert partition.blocks[0].destination_group_indices == (0, 1)
