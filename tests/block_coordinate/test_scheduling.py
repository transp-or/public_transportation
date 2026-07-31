"""Tests for deterministic conflict scheduling and parallel batch algebra."""

from __future__ import annotations

from threading import Barrier, Lock, get_ident

import numpy as np
import pytest

from public_transportation.inference.block_coordinate import (
    BlockCoordinateFingerprints,
    BlockCoordinateMAPConfig,
    BlockSolverConfig,
    BlockUpdatePolicy,
    ColumnSelectedLinearOperator,
    ODBlock,
    ParallelBlockExecutionConfig,
    build_block_conflict_graph,
    color_block_conflict_graph,
    construct_block_operators,
    initialize_incremental_state,
    prepare_separable_quadratic_prior,
    solve_conflict_free_batch,
    resume_block_coordinate_map,
    run_block_coordinate_map,
    validate_block_partition,
    validate_incremental_prediction,
)
from public_transportation.inference.fixed_routing_linear_dense_solver import (
    solve_dense_reference,
)
from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
)


def _block(identifier: str, column: int, support: tuple[int, ...]) -> ODBlock:
    return ODBlock(
        block_id=identifier,
        free_column_indices=(column,),
        active_od_indices=(column,),
        destination_group_indices=(column,),
        time_bin_ids=("t0",),
        measurement_support_indices=support,
    )


def test_conflict_graph_and_coloring_are_deterministic() -> None:
    blocks = (
        _block("a", 0, (0, 2)),
        _block("b", 1, (1,)),
        _block("c", 2, (0, 3)),
        _block("d", 3, (4,)),
    )
    graph = build_block_conflict_graph(blocks)
    schedule = color_block_conflict_graph(graph)

    assert graph.edges == ((0, 2),)
    assert schedule.color_by_block == (0, 0, 1, 0)
    assert tuple(tuple(block.block_id for block in batch) for batch in schedule.batches) == (
        ("a", "b", "d"),
        ("c",),
    )
    repeated = color_block_conflict_graph(build_block_conflict_graph(blocks))
    assert repeated.fingerprint == schedule.fingerprint

    coupled = build_block_conflict_graph(blocks, additional_couplings=(("b", "d"),))
    assert coupled.edges == ((0, 2), (1, 3))


def test_graph_rejects_missing_support_and_unknown_coupling() -> None:
    missing = ODBlock("missing", (0,), (0,), (0,), ("t0",))
    with pytest.raises(ValueError, match="exact measurement_support"):
        build_block_conflict_graph((missing,))
    with pytest.raises(ValueError, match="unknown block"):
        build_block_conflict_graph(
            (_block("a", 0, (0,)),), additional_couplings=(("a", "absent"),)
        )


def test_parallel_construction_uses_bounded_workers_and_preserves_order() -> None:
    blocks = (_block("a", 0, (0,)), _block("b", 1, (1,)))
    barrier = Barrier(2)
    lock = Lock()
    threads: set[int] = set()

    def factory(block):
        with lock:
            threads.add(get_ident())
        barrier.wait(timeout=2.0)
        return ColumnSelectedLinearOperator(np_operator, block.free_column_indices)

    from public_transportation.inference.linear_operator import DenseLinearOperator

    np_operator = DenseLinearOperator(np.eye(2))
    operators = construct_block_operators(blocks, factory, workers=2)
    assert len(threads) == 2
    np.testing.assert_array_equal(operators[0].matvec([2.0]), [2.0, 0.0])
    np.testing.assert_array_equal(operators[1].matvec([3.0]), [0.0, 3.0])


def test_conflict_free_parallel_batch_matches_dense_reference() -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.5],
            [0.0, 0.0, 0.25],
        ]
    )
    problem = FixedRoutingLinearProblem(
        measurement_operator=matrix,
        fixed_measurement_offset=np.zeros(5),
        observations=np.array([2.0, 1.0, 6.0, 3.0, 0.5]),
        observation_weights=np.ones(5),
        prior_demand=np.ones(3),
        lower_bounds=np.zeros(3),
        upper_bounds=np.full(3, 10.0),
        provenance=FixedRoutingLinearProvenance("od", "assignment", "mapping", 1.0),
        regularization_selection="none",
    )
    blocks = (
        _block("a", 0, (0, 1)),
        _block("b", 1, (2,)),
        _block("c", 2, (3, 4)),
    )
    state = initialize_incremental_state(problem.measurement_operator, np.ones(3), np.zeros(5))
    decision = solve_conflict_free_batch(
        problem=problem,
        prior=prepare_separable_quadratic_prior(problem),
        state=state,
        blocks=blocks,
        operator_factory=lambda block: ColumnSelectedLinearOperator(
            problem.measurement_operator,
            block.free_column_indices,
            measurement_support_indices=block.measurement_support_indices,
        ),
        solver_config=BlockSolverConfig(maximum_iterations=100, tolerance=1.0e-12),
        update_policy=BlockUpdatePolicy(),
        parallel_config=ParallelBlockExecutionConfig(
            construction_workers=2,
            solver_workers=2,
            threads_per_worker=1,
            available_cpus=2,
        ),
    )
    reference = solve_dense_reference(problem, tolerance=1.0e-12)

    assert decision.accepted_blocks == 3
    assert decision.rejected_blocks == 0
    np.testing.assert_allclose(decision.state.free_flow, reference.demand, atol=1.0e-7)
    assert decision.objective_improvement > 0.0
    assert validate_incremental_prediction(decision.state, problem.measurement_operator).within_tolerance


def test_parallel_batch_rejects_overlap_and_oversubscription() -> None:
    with pytest.raises(ValueError, match="exceeds available CPUs"):
        ParallelBlockExecutionConfig(
            construction_workers=3,
            solver_workers=2,
            threads_per_worker=2,
            available_cpus=4,
        )

    problem = FixedRoutingLinearProblem(
        measurement_operator=np.ones((1, 2)),
        fixed_measurement_offset=[0.0],
        observations=[1.0],
        observation_weights=[1.0],
        prior_demand=[1.0, 1.0],
        lower_bounds=[0.0, 0.0],
        upper_bounds=[2.0, 2.0],
        provenance=FixedRoutingLinearProvenance("od", "assignment", "mapping", 1.0),
        regularization_selection="none",
    )
    state = initialize_incremental_state(problem.measurement_operator, [1.0, 1.0], [0.0])
    blocks = (_block("a", 0, (0,)), _block("b", 1, (0,)))
    with pytest.raises(ValueError, match="overlapping"):
        solve_conflict_free_batch(
            problem=problem,
            prior=prepare_separable_quadratic_prior(problem),
            state=state,
            blocks=blocks,
            operator_factory=lambda block: ColumnSelectedLinearOperator(
                problem.measurement_operator, block.free_column_indices
            ),
            solver_config=BlockSolverConfig(),
            update_policy=BlockUpdatePolicy(),
            parallel_config=ParallelBlockExecutionConfig(available_cpus=1),
        )


def test_interleaved_estimator_batch_resume_matches_uninterrupted(tmp_path) -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.5],
            [0.0, 0.0, 0.25],
        ]
    )
    problem = FixedRoutingLinearProblem(
        measurement_operator=matrix,
        fixed_measurement_offset=np.zeros(5),
        observations=np.array([2.0, 1.0, 6.0, 3.0, 0.5]),
        observation_weights=np.ones(5),
        prior_demand=np.ones(3),
        lower_bounds=np.zeros(3),
        upper_bounds=np.full(3, 10.0),
        provenance=FixedRoutingLinearProvenance("od", "assignment", "mapping", 1.0),
        regularization_selection="none",
    )
    blocks = (
        _block("a", 0, (0, 1)),
        _block("b", 1, (2,)),
        _block("c", 2, (3, 4)),
    )
    partition = validate_block_partition(blocks, free_to_active_indices=(0, 1, 2))

    def configuration(directory):
        return BlockCoordinateMAPConfig(
            maximum_sweeps=2,
            global_projected_gradient_tolerance=None,
            relative_sweep_objective_tolerance=None,
            block_order="interleaved",
            construction_workers=2,
            solver_workers=2,
            threads_per_worker=1,
            checkpoint_directory=directory,
        )

    def identity(config):
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

    uninterrupted_config = configuration(tmp_path / "uninterrupted")
    uninterrupted_events = []
    uninterrupted = run_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=uninterrupted_config,
        fingerprints=identity(uninterrupted_config),
        progress_callback=uninterrupted_events.append,
    )
    assert len(uninterrupted_events) == 2
    assert all(event.blocks_completed_in_sweep == 3 for event in uninterrupted_events)
    assert all(event.block_or_batch.startswith("batch-") for event in uninterrupted_events)

    resumed_config = configuration(tmp_path / "resumed")
    resumed_identity = identity(resumed_config)
    interrupted = run_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=resumed_config,
        fingerprints=resumed_identity,
        stop_callback=lambda _event: True,
    )
    assert interrupted.status == "interrupted_with_approximate_solution"
    assert interrupted.state.accepted_updates == 3
    resumed = resume_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=resumed_config,
        fingerprints=resumed_identity,
    )
    assert resumed.status == uninterrupted.status == "stopped_by_sweep_budget"
    np.testing.assert_array_equal(resumed.latest_free_flow, uninterrupted.latest_free_flow)
    np.testing.assert_array_equal(
        resumed.state.current_prediction, uninterrupted.state.current_prediction
    )
    assert resumed.state.current_objective == uninterrupted.state.current_objective
    assert resumed.state.accepted_updates == uninterrupted.state.accepted_updates
    assert resumed.state.rejected_updates == uninterrupted.state.rejected_updates
