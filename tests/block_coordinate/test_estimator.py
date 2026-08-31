"""Tests for the sequential anytime block-coordinate estimator."""

from __future__ import annotations

import numpy as np
import pytest

from public_transportation.inference.block_coordinate import (
    BlockCoordinateFingerprints,
    BlockCoordinateMAPConfig,
    BlockCoordinateMAPEstimator,
    GlobalProductPolicy,
    DenseBlockLinearOperator,
    SelectedBlockConstructionDeadlineError,
    SelectedBlockDeadlineDiagnostics,
    ODBlock,
    BlockCheckpointStore,
    resume_block_coordinate_map,
    run_block_coordinate_map,
    validate_block_partition,
)
from public_transportation.inference.fixed_routing_linear_dense_solver import (
    solve_dense_reference,
)
from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
)
from public_transportation.inference.fixed_routing_linear_regularization import (
    ridge_to_prior,
)


def _problem() -> FixedRoutingLinearProblem:
    matrix = np.array(
        [
            [1.0, 0.2, 0.0, 0.3],
            [0.0, 1.2, 0.4, 0.0],
            [0.5, 0.0, 1.0, 0.2],
            [0.1, 0.4, 0.0, 1.0],
            [0.3, 0.2, 0.7, 0.1],
        ]
    )
    prior = np.array([1.0, 2.0, 1.5, 2.5])
    return FixedRoutingLinearProblem(
        measurement_operator=matrix,
        fixed_measurement_offset=np.array([0.2, 0.0, 0.5, 0.1, 0.3]),
        observations=np.array([3.0, 2.5, 4.0, 3.5, 2.0]),
        observation_weights=np.array([1.0, 2.0, 0.5, 1.5, 1.0]),
        prior_demand=prior,
        lower_bounds=np.zeros(4),
        upper_bounds=np.full(4, 8.0),
        provenance=FixedRoutingLinearProvenance("od", "assignment", "mapping", 1.0),
        regularization_selection="configured",
        regularization_blocks=(ridge_to_prior(prior, strength=0.4),),
    )


class CountingOperator:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)
        self.shape = self.matrix.shape
        self.dtype = self.matrix.dtype
        self.forward_count = 0
        self.transpose_count = 0

    def matvec(self, vector):
        self.forward_count += 1
        return self.matrix @ vector

    def rmatvec(self, vector):
        self.transpose_count += 1
        return self.matrix.T @ vector


def _counting_problem():
    base = _problem()
    operator = CountingOperator(base.measurement_operator.matrix)
    problem = FixedRoutingLinearProblem(
        measurement_operator=operator,
        fixed_measurement_offset=base.fixed_measurement_offset,
        observations=base.observations,
        observation_weights=base.observation_weights,
        prior_demand=base.prior_demand,
        lower_bounds=base.lower_bounds,
        upper_bounds=base.upper_bounds,
        provenance=base.provenance,
        regularization_selection=base.regularization_selection,
        regularization_blocks=base.regularization_blocks,
    )
    return problem, operator


def _local_factory(operator):
    return lambda block: DenseBlockLinearOperator(
        operator.matrix[:, block.free_column_indices]
    )


def test_deferred_initial_diagnostic_avoids_global_transpose(tmp_path) -> None:
    problem, operator = _counting_problem()
    partition = _partition()
    config = _config(
        tmp_path,
        maximum_block_updates=1,
        exact_global_diagnostic_every_sweeps=None,
        global_product_policy=GlobalProductPolicy(initial_exact_gradient=False),
    )
    result = run_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=config,
        fingerprints=_fingerprints(config, partition),
        block_operator_factory=_local_factory(operator),
    )
    assert operator.forward_count == 1
    assert operator.transpose_count == 0
    assert result.state.diagnostics.exact_global_projected_gradient.kind == "deferred"
    assert result.work.global_transpose_count == 0


def test_supplied_prediction_avoids_all_initial_global_products(tmp_path) -> None:
    problem, operator = _counting_problem()
    partition = _partition()
    policy = GlobalProductPolicy(
        initial_prediction_mode="provided",
        initial_exact_gradient=False,
        resume_prediction_validation="deferred",
    )
    config = _config(
        tmp_path,
        maximum_block_updates=1,
        exact_global_diagnostic_every_sweeps=None,
        global_product_policy=policy,
    )
    identity = _fingerprints(config, partition)
    initial = np.array(problem.prior_demand, copy=True)
    prediction = operator.matrix @ initial + problem.fixed_measurement_offset
    result = run_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=config,
        fingerprints=identity,
        initial_free_flow=initial,
        initial_prediction=prediction,
        fixed_measurement_offset=problem.fixed_measurement_offset,
        initial_prediction_fingerprint=identity.fingerprint,
        block_operator_factory=_local_factory(operator),
    )
    assert operator.forward_count == operator.transpose_count == 0
    assert result.work.global_forward_count == 0


def test_deferred_resume_extends_update_budget_without_global_products(tmp_path) -> None:
    problem, operator = _counting_problem()
    partition = _partition()
    policy = GlobalProductPolicy(
        initial_exact_gradient=False,
        resume_prediction_validation="deferred",
    )
    first_config = _config(
        tmp_path,
        maximum_block_updates=1,
        exact_global_diagnostic_every_sweeps=None,
        global_product_policy=policy,
    )
    identity = _fingerprints(first_config, partition)
    first = run_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=first_config,
        fingerprints=identity,
        block_operator_factory=_local_factory(operator),
    )
    operator.forward_count = operator.transpose_count = 0
    extended = _config(
        tmp_path,
        maximum_block_updates=2,
        exact_global_diagnostic_every_sweeps=None,
        global_product_policy=policy,
    )
    assert extended.fingerprint == first_config.fingerprint
    resumed = resume_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=extended,
        fingerprints=identity,
        block_operator_factory=_local_factory(operator),
    )
    assert first.state.schedule_position == 1
    assert resumed.state.accepted_updates + resumed.state.rejected_updates == 2
    assert operator.forward_count == operator.transpose_count == 0
    assert resumed.work.resume_prediction_validation == "deferred"


def test_construction_deadline_preserves_pending_block_and_resume_retries_it(
    tmp_path,
) -> None:
    problem = _problem()
    partition = _partition()
    config = _config(tmp_path, maximum_block_updates=1)
    identity = _fingerprints(config, partition)
    deadline = 1.0e30

    class DeadlineFactory:
        supports_absolute_deadline = True

        def build_result(self, block, *, absolute_deadline=None):
            assert absolute_deadline == deadline
            raise SelectedBlockConstructionDeadlineError(
                SelectedBlockDeadlineDiagnostics(
                    block_id=block.block_id,
                    phase="od_batch_preparation",
                    elapsed_construction_seconds=2.0,
                    absolute_deadline=deadline,
                    deadline_overshoot_seconds=0.25,
                    indivisible_operation_overshoot=False,
                    completed_od_batches=0,
                    total_od_batches=2,
                    completed_mapping_passes=0,
                    total_mapping_passes=4,
                    candidate_contributions_examined=0,
                    accepted_nonzeros_accumulated=0,
                    support_cache_hit=True,
                    numerical_cache_persistence_completed=False,
                    valid_warm_cache_exists=False,
                    partial_work_discarded=True,
                    current_temporary_memory_estimate=1024,
                )
            )

    stopped = run_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=config,
        fingerprints=identity,
        block_operator_factory=DeadlineFactory(),
        absolute_deadline=deadline,
    )
    assert stopped.status == "stopped_by_time_budget"
    assert stopped.state.schedule_position == 0
    assert stopped.state.accepted_updates == stopped.state.rejected_updates == 0
    assert stopped.work.selected_block_construction_attempts == 1
    assert stopped.work.selected_block_constructions_completed == 0
    assert stopped.work.selected_block_constructions_deadline_stopped == 1
    assert not stopped.work.solver_started
    assert stopped.work.checkpoint_preserved
    assert stopped.work.scheduled_block_not_attempted_by_solver == "first"

    attempted = []

    def successful_factory(block):
        attempted.append(block.block_id)
        return DenseBlockLinearOperator(
            problem.measurement_operator.matrix[:, block.free_column_indices]
        )

    resumed = resume_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=config,
        fingerprints=identity,
        block_operator_factory=successful_factory,
    )
    assert attempted == ["first"]
    assert resumed.state.schedule_position == 1
    assert resumed.state.accepted_updates + resumed.state.rejected_updates == 1


def test_completed_cache_before_deadline_stop_is_reused_for_pending_update(
    tmp_path,
) -> None:
    problem = _problem()
    partition = _partition()
    config = _config(tmp_path, maximum_block_updates=1)
    identity = _fingerprints(config, partition)

    class Clock:
        value = 0.0

        def __call__(self):
            return self.value

    clock = Clock()

    class Result:
        cache_hit = True

        def __init__(self, operator):
            self.operator = operator

    class CompletingFactory:
        supports_absolute_deadline = True

        def build_result(self, block, *, absolute_deadline=None):
            assert absolute_deadline == 10.0
            operator = DenseBlockLinearOperator(
                problem.measurement_operator.matrix[:, block.free_column_indices]
            )
            clock.value = 10.0
            return Result(operator)

    stopped = BlockCoordinateMAPEstimator(
        problem=problem,
        partition=partition,
        config=config,
        fingerprints=identity,
        block_operator_factory=CompletingFactory(),
        clock=clock,
        absolute_deadline=10.0,
    ).run()
    assert stopped.status == "stopped_by_time_budget"
    assert stopped.state.schedule_position == 0
    assert stopped.state.accepted_updates == stopped.state.rejected_updates == 0
    assert stopped.work.selected_block_cache_hits == 1
    assert stopped.work.selected_block_constructions_completed == 1
    assert not stopped.work.solver_started
    assert stopped.work.checkpoint_preserved

    attempted = []

    def warm_factory(block):
        attempted.append(block.block_id)
        return DenseBlockLinearOperator(
            problem.measurement_operator.matrix[:, block.free_column_indices]
        )

    resumed = resume_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=config,
        fingerprints=identity,
        block_operator_factory=warm_factory,
    )
    assert attempted == ["first"]
    assert resumed.state.schedule_position == 1


def _partition():
    blocks = (
        ODBlock("first", (0, 2), (0, 2), (0,), ("t0",)),
        ODBlock("second", (1, 3), (1, 3), (1,), ("t0",)),
    )
    return validate_block_partition(blocks, free_to_active_indices=(0, 1, 2, 3))


def _config(tmp_path, **overrides) -> BlockCoordinateMAPConfig:
    values = {
        "maximum_sweeps": 50,
        "block_solver_max_iterations": 100,
        "block_solver_tolerance": 1.0e-12,
        "global_projected_gradient_tolerance": 1.0e-7,
        "relative_sweep_objective_tolerance": 1.0e-13,
        "checkpoint_directory": tmp_path / "checkpoint",
    }
    values.update(overrides)
    return BlockCoordinateMAPConfig(**values)


def _fingerprints(config, partition) -> BlockCoordinateFingerprints:
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


def test_estimator_converges_to_dense_reference_with_monotone_progress(tmp_path) -> None:
    problem = _problem()
    partition = _partition()
    config = _config(tmp_path)
    events = []
    result = run_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=config,
        fingerprints=_fingerprints(config, partition),
        progress_callback=events.append,
    )
    reference = solve_dense_reference(problem, tolerance=1.0e-12)

    assert result.status == "converged"
    np.testing.assert_allclose(
        result.latest_free_flow, reference.demand, rtol=2.0e-5, atol=2.0e-5
    )
    np.testing.assert_allclose(
        result.state.current_prediction,
        problem.measurement_operator.matvec(result.latest_free_flow)
        + problem.fixed_measurement_offset,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    objectives = [event.current_objective for event in events]
    assert all(after <= before + 1.0e-10 for before, after in zip(objectives, objectives[1:]))
    assert events
    assert any(event.checkpoint_committed for event in events)
    assert events[-1].exact_global_projected_gradient.kind == "exact"
    assert events[-1].to_json_line().endswith("\n")
    assert result.state.best_objective <= result.state.current_objective


def test_reporting_does_not_change_deterministic_result_or_checkpoint_bytes(tmp_path) -> None:
    problem = _problem()
    partition = _partition()

    def run(directory, *, reporting):
        config = _config(directory)
        events = []
        estimator = BlockCoordinateMAPEstimator(
            problem=problem,
            partition=partition,
            config=config,
            fingerprints=_fingerprints(config, partition),
            progress_callback=events.append if reporting else None,
            # Freeze elapsed-time bookkeeping so this test compares the
            # scientific checkpoint payload byte-for-byte.
            clock=lambda: 0.0,
        )
        result = estimator.run()
        files = {
            path.relative_to(directory): path.read_bytes()
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        }
        return result, files, events

    disabled, disabled_files, disabled_events = run(
        tmp_path / "reporting-disabled", reporting=False
    )
    enabled, enabled_files, enabled_events = run(
        tmp_path / "reporting-enabled", reporting=True
    )

    assert disabled_events == []
    assert enabled_events
    assert disabled.status == enabled.status
    np.testing.assert_array_equal(disabled.latest_free_flow, enabled.latest_free_flow)
    np.testing.assert_array_equal(
        disabled.state.current_prediction, enabled.state.current_prediction
    )
    assert disabled.state.current_objective == enabled.state.current_objective
    assert disabled.state.best_objective == enabled.state.best_objective
    assert disabled_files == enabled_files


def test_update_budget_returns_valid_anytime_solution_after_one_block(tmp_path) -> None:
    problem = _problem()
    partition = _partition()
    config = _config(tmp_path, maximum_block_updates=1)
    result = run_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=config,
        fingerprints=_fingerprints(config, partition),
    )

    assert result.status == "stopped_by_update_budget"
    assert result.state.accepted_updates + result.state.rejected_updates == 1
    assert result.state.sweep == 0
    assert result.state.schedule_position == 1
    np.testing.assert_allclose(
        result.state.current_prediction,
        problem.measurement_operator.matvec(result.latest_free_flow)
        + problem.fixed_measurement_offset,
    )


def test_shuffled_schedules_are_deterministic(tmp_path) -> None:
    problem = _problem()
    partition = _partition()
    first_events = []
    second_events = []
    first_config = _config(
        tmp_path / "first",
        maximum_sweeps=3,
        block_order="shuffled",
        random_seed=73,
        global_projected_gradient_tolerance=None,
        relative_sweep_objective_tolerance=None,
    )
    second_config = _config(
        tmp_path / "second",
        maximum_sweeps=3,
        block_order="shuffled",
        random_seed=73,
        global_projected_gradient_tolerance=None,
        relative_sweep_objective_tolerance=None,
    )
    first = run_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=first_config,
        fingerprints=_fingerprints(first_config, partition),
        progress_callback=first_events.append,
    )
    second = run_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=second_config,
        fingerprints=_fingerprints(second_config, partition),
        progress_callback=second_events.append,
    )

    assert first.status == second.status == "stopped_by_sweep_budget"
    assert [event.block_or_batch for event in first_events] == [
        event.block_or_batch for event in second_events
    ]
    np.testing.assert_array_equal(first.latest_free_flow, second.latest_free_flow)
    assert first.state.random_state_json == second.state.random_state_json


def test_user_callback_interrupts_only_after_an_atomic_update(tmp_path) -> None:
    problem = _problem()
    partition = _partition()
    config = _config(tmp_path)
    events = []

    def stop(event) -> bool:
        events.append(event)
        return True

    result = run_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=config,
        fingerprints=_fingerprints(config, partition),
        stop_callback=stop,
    )
    assert result.status == "interrupted_with_approximate_solution"
    assert len(events) == 1
    assert result.state.accepted_updates + result.state.rejected_updates == 1
    assert result.state.schedule_position == 1


@pytest.mark.parametrize("stop_after", [1, 2])
def test_interrupted_shuffled_run_resumes_identically_to_uninterrupted(
    tmp_path, stop_after
) -> None:
    problem = _problem()
    partition = _partition()
    common = {
        "maximum_sweeps": 3,
        "maximum_block_updates": 5,
        "block_order": "shuffled",
        "random_seed": 19,
        "global_projected_gradient_tolerance": None,
        "relative_sweep_objective_tolerance": None,
    }
    uninterrupted_config = _config(tmp_path / "uninterrupted", **common)
    uninterrupted = run_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=uninterrupted_config,
        fingerprints=_fingerprints(uninterrupted_config, partition),
    )

    resumed_config = _config(tmp_path / "resumed", **common)
    identity = _fingerprints(resumed_config, partition)
    callbacks = 0

    def stop_at_requested_boundary(_event) -> bool:
        nonlocal callbacks
        callbacks += 1
        return callbacks == stop_after

    first_part = run_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=resumed_config,
        fingerprints=identity,
        stop_callback=stop_at_requested_boundary,
    )
    assert first_part.status == "interrupted_with_approximate_solution"
    resumed = resume_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=resumed_config,
        fingerprints=identity,
    )

    assert resumed.status == uninterrupted.status == "stopped_by_update_budget"
    np.testing.assert_array_equal(resumed.latest_free_flow, uninterrupted.latest_free_flow)
    np.testing.assert_array_equal(
        resumed.state.current_prediction, uninterrupted.state.current_prediction
    )
    assert resumed.state.current_objective == uninterrupted.state.current_objective
    assert resumed.state.best_objective == uninterrupted.state.best_objective
    assert resumed.state.accepted_updates == uninterrupted.state.accepted_updates
    assert resumed.state.rejected_updates == uninterrupted.state.rejected_updates
    recovered = BlockCheckpointStore(resumed_config.checkpoint_directory, identity).load()
    np.testing.assert_array_equal(recovered.current_free_flow, resumed.latest_free_flow)
    np.testing.assert_array_equal(
        recovered.current_prediction, resumed.state.current_prediction
    )


def test_time_budget_can_stop_at_the_initial_valid_solution(tmp_path) -> None:
    problem = _problem()
    partition = _partition()
    config = _config(tmp_path, maximum_elapsed_seconds=0.5)

    class Clock:
        value = -1.0

        def __call__(self) -> float:
            self.value += 1.0
            return self.value

    estimator = BlockCoordinateMAPEstimator(
        problem=problem,
        partition=partition,
        config=config,
        fingerprints=_fingerprints(config, partition),
        clock=Clock(),
    )
    result = estimator.run()
    assert result.status == "stopped_by_time_budget"
    assert result.state.accepted_updates == result.state.rejected_updates == 0
    np.testing.assert_array_equal(result.latest_free_flow, problem.prior_demand)


def test_keyboard_interrupt_during_block_loading_preserves_initial_state(tmp_path) -> None:
    problem = _problem()
    partition = _partition()
    config = _config(tmp_path)

    def interrupt(_block):
        raise KeyboardInterrupt

    result = run_block_coordinate_map(
        problem=problem,
        partition=partition,
        config=config,
        fingerprints=_fingerprints(config, partition),
        block_operator_factory=interrupt,
    )
    assert result.status == "interrupted_with_approximate_solution"
    assert result.state.accepted_updates == result.state.rejected_updates == 0
    np.testing.assert_array_equal(result.latest_free_flow, problem.prior_demand)


def test_interleaved_order_requires_exact_support_and_mismatched_fingerprints_are_rejected(
    tmp_path,
) -> None:
    problem = _problem()
    partition = _partition()
    interleaved = _config(tmp_path, block_order="interleaved")
    estimator = BlockCoordinateMAPEstimator(
            problem,
            partition,
            interleaved,
            _fingerprints(interleaved, partition),
        )
    with pytest.raises(ValueError, match="exact measurement_support"):
        estimator.run()

    config = _config(tmp_path)
    identity = _fingerprints(config, partition)
    wrong = BlockCoordinateFingerprints(
        scenario=identity.scenario,
        assignment_inputs=identity.assignment_inputs,
        od_layout=identity.od_layout,
        fixed_demand=identity.fixed_demand,
        measurements=identity.measurements,
        prior=identity.prior,
        routing=identity.routing,
        partition="wrong",
        solver_semantics=identity.solver_semantics,
    )
    with pytest.raises(ValueError, match="partition fingerprint"):
        BlockCoordinateMAPEstimator(problem, partition, config, wrong)
