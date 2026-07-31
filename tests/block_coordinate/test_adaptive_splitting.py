from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from public_transportation.inference.block_coordinate import (
    AdaptiveBlockSplitConfig,
    BlockCoordinateFingerprints,
    BlockCoordinateMAPConfig,
    BlockPreflightSample,
    BlockResourceCostModel,
    BlockResourceGuardError,
    ODBlock,
    fingerprints_for_adapted_partition,
    resume_block_coordinate_map,
    run_block_coordinate_map,
    split_partition_for_resource_limits,
    validate_block_partition,
)
from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
)
from public_transportation.inference.fixed_routing_linear_dense_solver import (
    solve_dense_reference,
)
from public_transportation.inference.fixed_routing_linear_regularization import (
    ridge_to_prior,
)


def _partition(size: int = 8):
    block = ODBlock(
        "original",
        tuple(range(size)),
        tuple(range(size)),
        (0,),
        ("morning",),
        estimated_nonzeros=size * 10,
        measurement_support_indices=tuple(range(size)),
    )
    return validate_block_partition((block,), free_to_active_indices=tuple(range(size)))


def _model(*, bytes_per_variable: float = 100.0, seconds_per_variable: float = 0.1):
    return BlockResourceCostModel(
        worker_bytes_per_variable=bytes_per_variable,
        runtime_seconds_per_variable=seconds_per_variable,
        uncertainty_factor=1.0,
    )


def _fingerprints(config: BlockCoordinateMAPConfig, partition) -> BlockCoordinateFingerprints:
    return BlockCoordinateFingerprints(
        scenario="scenario",
        assignment_inputs="assignment",
        od_layout="layout",
        fixed_demand="fixed",
        measurements="measurements",
        prior="prior",
        routing="routing",
        partition=partition.fingerprint,
        solver_semantics=config.fingerprint,
    )


def _problem() -> FixedRoutingLinearProblem:
    matrix = np.array(
        [
            [1.0, 0.2, 0.0, 0.3],
            [0.0, 1.2, 0.4, 0.0],
            [0.5, 0.0, 1.0, 0.2],
            [0.1, 0.4, 0.0, 1.0],
        ]
    )
    prior = np.array([1.0, 2.0, 1.5, 2.5])
    return FixedRoutingLinearProblem(
        measurement_operator=matrix,
        fixed_measurement_offset=np.zeros(4),
        observations=np.array([3.0, 2.5, 4.0, 3.5]),
        observation_weights=np.ones(4),
        prior_demand=prior,
        lower_bounds=np.zeros(4),
        upper_bounds=np.full(4, 8.0),
        provenance=FixedRoutingLinearProvenance("od", "assignment", "mapping", 1.0),
        regularization_selection="configured",
        regularization_blocks=(ridge_to_prior(prior, strength=0.4),),
    )


def test_memory_triggered_split_preserves_coverage_metadata_and_fingerprint() -> None:
    original = _partition()
    result = split_partition_for_resource_limits(
        original,
        cost_model=_model(),
        config=AdaptiveBlockSplitConfig(maximum_worker_memory_bytes=250),
    )

    assert result.changed
    assert len(result.partition.blocks) == 4
    assert result.partition.fingerprint != original.fingerprint
    assert result.maximum_estimated_worker_memory_bytes <= 250
    assert sum(block.estimated_nonzeros or 0 for block in result.partition.blocks) == 80
    assert [
        column for block in result.partition.blocks for column in block.free_column_indices
    ] == list(range(8))
    assert all(block.measurement_support_indices == tuple(range(8)) for block in result.partition.blocks)
    assert {record.trigger for record in result.records} == {"memory"}


def test_runtime_triggered_split_is_deterministic() -> None:
    original = _partition()
    config = AdaptiveBlockSplitConfig(maximum_block_runtime_seconds=0.21)

    first = split_partition_for_resource_limits(
        original, cost_model=_model(), config=config
    )
    second = split_partition_for_resource_limits(
        original, cost_model=_model(), config=config
    )

    assert first == second
    assert len(first.partition.blocks) == 4
    assert first.maximum_estimated_block_runtime_seconds <= 0.21
    assert {record.trigger for record in first.records} == {"runtime"}


def test_cost_model_uses_conservative_preflight_ratios() -> None:
    samples = (
        BlockPreflightSample("small", 2, 10, 100, 100, 0.2, 0.1, 0.1, 10, 20),
        BlockPreflightSample("large", 4, 30, 600, 200, 1.2, 0.2, 0.2, 20, 40),
    )
    model = BlockResourceCostModel.from_preflight_samples(
        samples, uncertainty_factor=1.5
    )
    memory, runtime = model.estimate(_partition(2).blocks[0])
    assert memory == 600
    assert runtime == pytest.approx(1.2)


def test_indivisible_unsafe_block_triggers_resource_guard() -> None:
    with pytest.raises(BlockResourceGuardError, match="remains unsafe"):
        split_partition_for_resource_limits(
            _partition(1),
            cost_model=_model(),
            config=AdaptiveBlockSplitConfig(maximum_worker_memory_bytes=50),
        )


def test_adapted_partition_checkpoint_resumes_and_obsolete_identity_is_rejected(
    tmp_path,
) -> None:
    problem = _problem()
    original = _partition(4)
    split = split_partition_for_resource_limits(
        original,
        cost_model=_model(),
        config=AdaptiveBlockSplitConfig(maximum_worker_memory_bytes=200),
    )
    config = BlockCoordinateMAPConfig(
        maximum_sweeps=2,
        maximum_block_updates=3,
        global_projected_gradient_tolerance=None,
        relative_sweep_objective_tolerance=None,
        checkpoint_directory=tmp_path / "checkpoint",
    )
    original_identity = _fingerprints(config, original)
    identity = fingerprints_for_adapted_partition(original_identity, split)
    callbacks = 0

    def interrupt_once(_event) -> bool:
        nonlocal callbacks
        callbacks += 1
        return callbacks == 1

    interrupted = run_block_coordinate_map(
        problem=problem,
        partition=split.partition,
        config=config,
        fingerprints=identity,
        stop_callback=interrupt_once,
    )
    assert interrupted.status == "interrupted_with_approximate_solution"
    resumed = resume_block_coordinate_map(
        problem=problem,
        partition=split.partition,
        config=config,
        fingerprints=identity,
    )
    assert resumed.status == "stopped_by_update_budget"
    assert resumed.state.accepted_updates + resumed.state.rejected_updates == 3

    with pytest.raises(ValueError, match="partition fingerprint"):
        resume_block_coordinate_map(
            problem=problem,
            partition=split.partition,
            config=config,
            fingerprints=original_identity,
        )

    obsolete_config = replace(config, checkpoint_directory=tmp_path / "obsolete")
    obsolete_identity = _fingerprints(obsolete_config, original)
    run_block_coordinate_map(
        problem=problem,
        partition=original,
        config=obsolete_config,
        fingerprints=obsolete_identity,
        stop_callback=lambda _event: True,
    )
    adapted_obsolete_identity = replace(
        obsolete_identity, partition=split.partition.fingerprint
    )
    with pytest.raises(ValueError, match="fingerprints"):
        resume_block_coordinate_map(
            problem=problem,
            partition=split.partition,
            config=obsolete_config,
            fingerprints=adapted_obsolete_identity,
        )


def test_fingerprint_adapter_rejects_unrelated_original_partition() -> None:
    original = _partition()
    split = split_partition_for_resource_limits(
        original,
        cost_model=_model(),
        config=AdaptiveBlockSplitConfig(maximum_worker_memory_bytes=250),
    )
    config = BlockCoordinateMAPConfig()
    unrelated = replace(_fingerprints(config, original), partition="unrelated")
    with pytest.raises(ValueError, match="original partition"):
        fingerprints_for_adapted_partition(unrelated, split)


def test_adaptive_split_solution_matches_unsplit_and_dense_reference(tmp_path) -> None:
    problem = _problem()
    original = _partition(4)
    split = split_partition_for_resource_limits(
        original,
        cost_model=_model(),
        config=AdaptiveBlockSplitConfig(maximum_worker_memory_bytes=200),
    )
    common = {
        "maximum_sweeps": 100,
        "block_solver_max_iterations": 100,
        "block_solver_tolerance": 1.0e-12,
        "global_projected_gradient_tolerance": 1.0e-8,
        "relative_sweep_objective_tolerance": 1.0e-14,
    }
    original_config = BlockCoordinateMAPConfig(
        **common, checkpoint_directory=tmp_path / "original"
    )
    split_config = BlockCoordinateMAPConfig(
        **common, checkpoint_directory=tmp_path / "split"
    )
    original_result = run_block_coordinate_map(
        problem=problem,
        partition=original,
        config=original_config,
        fingerprints=_fingerprints(original_config, original),
    )
    split_result = run_block_coordinate_map(
        problem=problem,
        partition=split.partition,
        config=split_config,
        fingerprints=_fingerprints(split_config, split.partition),
    )
    reference = solve_dense_reference(problem, tolerance=1.0e-12)

    assert original_result.status == split_result.status == "converged"
    np.testing.assert_allclose(split_result.latest_free_flow, reference.demand, atol=1.0e-7)
    np.testing.assert_allclose(
        split_result.latest_free_flow, original_result.latest_free_flow, atol=1.0e-7
    )
    assert split_result.state.current_objective == pytest.approx(
        original_result.state.current_objective, abs=1.0e-12
    )
