from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from public_transportation.inference.block_coordinate import (
    AcceptedBlockResourceProposal,
    BlockCheckpointMetadata,
    BlockCoordinateFingerprints,
    BlockCoordinateMAPConfig,
    BlockResourceRecommendation,
    BlockSizingConfig,
    ODBlock,
)


def fingerprints() -> BlockCoordinateFingerprints:
    return BlockCoordinateFingerprints(
        scenario="scenario",
        assignment_inputs="assignment",
        od_layout="layout",
        fixed_demand="fixed",
        measurements="measurements",
        prior="prior",
        routing="routing",
        partition="partition",
        solver_semantics="solver",
    )


def test_od_block_normalizes_immutable_membership_and_fingerprints():
    block = ODBlock(
        block_id=" block-1 ",
        free_column_indices=[1, 3],
        active_od_indices=[4, 8],
        destination_group_indices=[2],
        time_bin_ids=["morning"],
        estimated_nonzeros=12,
        measurement_support_indices=[0, 5],
    )
    assert block.block_id == "block-1"
    assert block.free_column_indices == (1, 3)
    assert block.num_free_variables == 2
    assert len(block.fingerprint) == 64
    with pytest.raises(FrozenInstanceError):
        block.block_id = "changed"

    changed = ODBlock(
        block_id="block-1",
        free_column_indices=(1, 4),
        active_od_indices=(4, 9),
        destination_group_indices=(2,),
        time_bin_ids=("morning",),
        estimated_nonzeros=12,
        measurement_support_indices=(0, 5),
    )
    assert changed.fingerprint != block.fingerprint


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"block_id": ""}, "nonempty"),
        ({"free_column_indices": ()}, "at least one"),
        ({"free_column_indices": (2, 1)}, "unique and ascending"),
        ({"free_column_indices": (0, 1.5)}, "iterable of integers"),
        ({"active_od_indices": (4,)}, "one-to-one"),
        ({"destination_group_indices": (-1,)}, "non-negative"),
        ({"time_bin_ids": ("",)}, "nonempty and unique"),
        ({"estimated_nonzeros": -1}, "non-negative"),
        ({"measurement_support_indices": (2, 2)}, "unique and ascending"),
    ],
)
def test_od_block_rejects_invalid_contracts(overrides, message):
    values = {
        "block_id": "block",
        "free_column_indices": (0, 1),
        "active_od_indices": (3, 4),
        "destination_group_indices": (1,),
        "time_bin_ids": ("am",),
    }
    values.update(overrides)
    with pytest.raises((TypeError, ValueError), match=message):
        ODBlock(**values)


def test_block_sizing_requires_explicit_ceiling_but_auto_is_a_proposal():
    with pytest.raises(ValueError, match="hard ceiling"):
        BlockSizingConfig(mode="explicit")
    explicit = BlockSizingConfig(
        mode="explicit", maximum_free_variables_per_block=50
    )
    automatic = BlockSizingConfig(mode="auto")
    assert explicit.maximum_free_variables_per_block == 50
    assert automatic.mode == "auto"
    with pytest.raises(ValueError, match="positive"):
        BlockSizingConfig(mode="auto", maximum_worker_memory_bytes=0)


def test_map_config_validates_and_fingerprints_solver_semantics(tmp_path):
    config = BlockCoordinateMAPConfig(checkpoint_directory=tmp_path / "checkpoints")
    assert config.checkpoint_directory == tmp_path / "checkpoints"
    assert len(config.fingerprint) == 64
    assert config.fingerprint == BlockCoordinateMAPConfig(
        checkpoint_directory=tmp_path / "checkpoints"
    ).fingerprint
    assert config.fingerprint != BlockCoordinateMAPConfig(
        checkpoint_directory=tmp_path / "checkpoints", update_damping=0.5
    ).fingerprint

    with pytest.raises(ValueError, match="update_damping"):
        BlockCoordinateMAPConfig(update_damping=0.0)
    with pytest.raises(ValueError, match="maximum_sweeps"):
        BlockCoordinateMAPConfig(maximum_sweeps=0)
    with pytest.raises(ValueError, match="strictly positive"):
        BlockCoordinateMAPConfig(block_solver_tolerance=0.0)
    with pytest.raises(ValueError, match="solver_workers"):
        BlockCoordinateMAPConfig(solver_workers=0)


def test_checkpoint_and_resource_proposal_contracts():
    identity = fingerprints()
    metadata = BlockCheckpointMetadata(
        fingerprints=identity,
        checkpoint_sequence=2,
        journal_sequence=5,
        sweep=1,
        schedule_position=3,
        committed=True,
    )
    assert metadata.schema_version == 1
    assert len(identity.fingerprint) == 64

    recommendation = BlockResourceRecommendation(
        resource_profile="laptop",
        maximum_variables_per_block=100,
        maximum_nonzeros_per_block=1_000,
        block_count=10,
        worker_count=2,
        threads_per_worker=2,
        expected_peak_memory_bytes=1_000_000,
        expected_cache_bytes=500_000,
        estimated_first_sweep_seconds=20.0,
        estimated_cache_hit_sweep_seconds=5.0,
        uncertainty_fraction=0.25,
        reason="memory-limited proposal",
    )
    accepted = AcceptedBlockResourceProposal(
        recommendation_fingerprint=recommendation.fingerprint,
        accepted=True,
    )
    assert accepted.accepted
    with pytest.raises(ValueError, match="explicitly accepted"):
        AcceptedBlockResourceProposal(
            recommendation_fingerprint=recommendation.fingerprint,
            accepted=False,
        )


def test_resource_contract_rejects_invalid_values():
    with pytest.raises(ValueError, match="worker_count"):
        BlockResourceRecommendation(
            resource_profile="server",
            maximum_variables_per_block=1,
            maximum_nonzeros_per_block=1,
            block_count=1,
            worker_count=0,
            threads_per_worker=1,
            expected_peak_memory_bytes=1,
            expected_cache_bytes=1,
            estimated_first_sweep_seconds=0.0,
            estimated_cache_hit_sweep_seconds=0.0,
            uncertainty_fraction=0.0,
            reason="test",
        )


def test_public_exports_are_available():
    from public_transportation.inference import ODBlock as ExportedODBlock

    assert ExportedODBlock is ODBlock
    assert isinstance(Path("checkpoint"), Path)
    assert np.dtype(float).kind == "f"
