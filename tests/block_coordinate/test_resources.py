from __future__ import annotations

import numpy as np
import pytest

from public_transportation.inference.block_coordinate import (
    AcceptedBlockResourceProposal,
    BlockPreflightSample,
    DenseBlockLinearOperator,
    MachineResourceSnapshot,
    ODBlock,
    ResourcePreflightConfig,
    BlockCoordinateMAPConfig,
    apply_accepted_resource_recommendation,
    detect_machine_resources,
    measure_representative_blocks,
    recommend_block_resources,
    select_representative_blocks,
    validate_resource_acceptance,
)


def _block(sequence: int, variables: int, nonzeros: int) -> ODBlock:
    columns = tuple(range(sequence * 10, sequence * 10 + variables))
    return ODBlock(
        block_id=f"block-{sequence}",
        free_column_indices=columns,
        active_od_indices=columns,
        destination_group_indices=(sequence,),
        time_bin_ids=("morning",),
        estimated_nonzeros=nonzeros,
        measurement_support_indices=tuple(range(sequence * 10, sequence * 10 + 3)),
    )


def _sample(
    block_id: str = "sample", *, variables: int = 100, worker_bytes: int = 2_000
) -> BlockPreflightSample:
    return BlockPreflightSample(
        block_id=block_id,
        variables=variables,
        nonzeros=variables * 10,
        operator_memory_bytes=worker_bytes // 2,
        local_solver_memory_bytes=worker_bytes - worker_bytes // 2,
        construction_seconds=2.0,
        matvec_seconds=0.2,
        rmatvec_seconds=0.3,
        checkpoint_bytes=variables * 32,
        cache_bytes=variables * 20,
    )


def _machine(
    *, memory: int = 1_000_000, logical: int = 8, physical: int = 4
) -> MachineResourceSnapshot:
    return MachineResourceSnapshot(
        available_memory_bytes=memory,
        logical_cpu_count=logical,
        physical_cpu_count=physical,
        available_cache_bytes=10_000_000,
        coordinator_rss_bytes=10_000,
        assignment_rss_bytes=20_000,
    )


def test_representative_selection_is_bounded_deterministic_and_spans_sizes() -> None:
    blocks = tuple(_block(index, index + 1, (index + 1) * 10) for index in range(7))

    selected = select_representative_blocks(blocks, maximum_samples=3)

    assert tuple(block.block_id for block in selected) == (
        "block-0",
        "block-3",
        "block-6",
    )
    assert selected == select_representative_blocks(tuple(reversed(blocks)), maximum_samples=3)


def test_bounded_measurement_constructs_only_selected_blocks_and_releases() -> None:
    blocks = tuple(_block(index, index + 1, (index + 1) * 10) for index in range(5))
    constructed: list[str] = []

    class ReleasableDense:
        def __init__(self, matrix: np.ndarray) -> None:
            self._operator = DenseBlockLinearOperator(matrix)
            self.released = False

        @property
        def dtype(self) -> np.dtype:
            return self._operator.dtype

        @property
        def num_local_variables(self) -> int:
            return self._operator.num_local_variables

        @property
        def num_measurements(self) -> int:
            return self._operator.num_measurements

        @property
        def retained_bytes(self) -> int:
            return 800

        def matvec(self, value: object) -> np.ndarray:
            return self._operator.matvec(value)

        def rmatvec(self, value: object) -> np.ndarray:
            return self._operator.rmatvec(value)

        def release(self) -> None:
            self.released = True

    operators: list[ReleasableDense] = []

    def factory(block: ODBlock) -> ReleasableDense:
        constructed.append(block.block_id)
        operator = ReleasableDense(np.ones((8, block.num_free_variables)))
        operators.append(operator)
        return operator

    samples = measure_representative_blocks(
        blocks,
        operator_factory=factory,
        maximum_samples=2,
        local_solver_memory_estimator=lambda block: block.num_free_variables * 64,
    )

    assert constructed == ["block-0", "block-4"]
    assert len(samples) == 2
    assert all(sample.operator_memory_bytes == 800 for sample in samples)
    assert all(operator.released for operator in operators)


@pytest.mark.parametrize(
    ("profile", "memory", "logical", "physical"),
    [
        ("laptop", 200_000, 4, 2),
        ("workstation", 1_000_000, 16, 8),
        ("server", 5_000_000, 32, 16),
        ("auto", 1_000_000, 8, 4),
    ],
)
def test_recommendation_respects_cpu_memory_and_storage_policy(
    profile: str, memory: int, logical: int, physical: int
) -> None:
    machine = _machine(memory=memory, logical=logical, physical=physical)
    recommendation = recommend_block_resources(
        samples=(_sample(), _sample("large", variables=200, worker_bytes=5_000)),
        machine=machine,
        total_variables=10_000,
        total_nonzeros=100_000,
        config=ResourcePreflightConfig(resource_profile=profile),
    )

    assert recommendation.worker_count * recommendation.threads_per_worker <= logical
    assert recommendation.expected_peak_memory_bytes <= int(
        memory * {"laptop": 0.55, "workstation": 0.68, "server": 0.78}.get(profile, 0.55)
    )
    assert recommendation.expected_cache_bytes <= machine.available_cache_bytes
    assert recommendation.block_count >= 1


def test_memory_constrained_many_core_machine_throttles_workers() -> None:
    recommendation = recommend_block_resources(
        samples=(_sample(worker_bytes=20_000),),
        machine=_machine(memory=120_000, logical=64, physical=32),
        total_variables=1_000,
        total_nonzeros=10_000,
        config=ResourcePreflightConfig(
            resource_profile="laptop",
            requested_workers=32,
            requested_threads_per_worker=1,
        ),
    )

    assert recommendation.worker_count < 32
    assert recommendation.worker_count >= 1


def test_cpu_constrained_high_memory_machine_is_cpu_bounded() -> None:
    recommendation = recommend_block_resources(
        samples=(_sample(),),
        machine=_machine(memory=100_000_000, logical=2, physical=2),
        total_variables=1_000,
        total_nonzeros=10_000,
        config=ResourcePreflightConfig(requested_workers=100),
    )
    assert recommendation.worker_count == 2


def test_preflight_rejects_insufficient_memory_and_storage() -> None:
    with pytest.raises(MemoryError, match="fixed process memory"):
        recommend_block_resources(
            samples=(_sample(),),
            machine=_machine(memory=40_000),
            total_variables=100,
            total_nonzeros=1_000,
            config=ResourcePreflightConfig(resource_profile="laptop"),
        )

    no_storage = MachineResourceSnapshot(
        available_memory_bytes=1_000_000,
        logical_cpu_count=4,
        physical_cpu_count=4,
        available_cache_bytes=1,
    )
    with pytest.raises(OSError, match="cache exceeds"):
        recommend_block_resources(
            samples=(_sample(),),
            machine=no_storage,
            total_variables=1_000,
            total_nonzeros=10_000,
        )


def test_acceptance_must_match_exact_recommendation() -> None:
    recommendation = recommend_block_resources(
        samples=(_sample(),),
        machine=_machine(),
        total_variables=1_000,
        total_nonzeros=10_000,
    )
    accepted = AcceptedBlockResourceProposal(recommendation.fingerprint, True)
    validate_resource_acceptance(recommendation, accepted)
    sizing, execution = apply_accepted_resource_recommendation(
        recommendation,
        accepted,
        map_config=BlockCoordinateMAPConfig(),
    )
    assert sizing.maximum_free_variables_per_block == recommendation.maximum_variables_per_block
    assert sizing.maximum_operator_nonzeros_per_block == recommendation.maximum_nonzeros_per_block
    assert execution.construction_workers == recommendation.worker_count
    assert execution.solver_workers == recommendation.worker_count
    assert execution.threads_per_worker == recommendation.threads_per_worker

    stale = AcceptedBlockResourceProposal("stale", True)
    with pytest.raises(ValueError, match="does not match"):
        validate_resource_acceptance(recommendation, stale)


def test_invalid_machine_snapshot_and_preflight_overrides_are_rejected() -> None:
    with pytest.raises(ValueError, match="physical_cpu_count"):
        MachineResourceSnapshot(1, 2, 3, 1)
    with pytest.raises(ValueError, match="requested_workers"):
        ResourcePreflightConfig(requested_workers=0)
    with pytest.raises(ValueError, match="at least one"):
        ResourcePreflightConfig(memory_safety_factor=0.99)


def test_machine_detection_supports_nonexistent_cache_directory(tmp_path) -> None:
    snapshot = detect_machine_resources(cache_directory=tmp_path / "future" / "cache")
    assert snapshot.available_memory_bytes > 0
    assert 1 <= snapshot.physical_cpu_count <= snapshot.logical_cpu_count
    assert snapshot.available_cache_bytes > 0
