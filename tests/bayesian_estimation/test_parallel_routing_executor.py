from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import sleep

import numpy as np
import pytest

from benchmarks.benchmark_sharded_gravity_operator import _inputs, _layout
from public_transportation.inference.parallel_partial_execution import (
    RoutingWorkUnit,
    build_balanced_microshard_plan,
    routing_group_work_units,
    plan_fixed_budget_routing_selection,
)
from public_transportation.inference.parallel_routing_executor import (
    ParallelRoutingExecutionInterrupted,
    ParallelRoutingExecutorConfig,
    PersistentParallelRoutingExecutor,
    plan_fixed_shape_routing_batches,
)
from public_transportation.inference.sharded_fixed_routing import (
    FixedRoutingPreparationConfig,
    prepare_fixed_routing_sharded,
)
from public_transportation.inference.sharded_matrix_free_operator import (
    ShardedMatrixFreeFixedRoutingMeasurementOperator,
)
from public_transportation.measurement.mapping import AggregationSpec


def _operator(root: Path, *, groups: int = 8):
    inputs = _inputs(nodes=32, degree=2, groups=groups, od_cells=32)
    prepared = prepare_fixed_routing_sharded(
        inputs=inputs,
        theta=1.0,
        config=FixedRoutingPreparationConfig(
            maximum_groups_per_shard=2,
            cache_directory=root / "routing",
            checkpoint_directory=root / "checkpoints",
            resident_shard_limit=2,
        ),
    )
    measurements = 16
    spec = AggregationSpec(
        num_measurements=measurements,
        measurement_index=np.arange(measurements, dtype=np.int32),
        link_index=np.arange(measurements, dtype=np.int32) % inputs.graph.num_links,
    )
    return ShardedMatrixFreeFixedRoutingMeasurementOperator(
        inputs=inputs,
        routing=prepared.routing,
        spec=spec,
        compact_layout=_layout(32),
        resident_shard_limit=2,
        operator_shards_per_batch=2,
    )


def _plan(operator, *, target=8):
    return build_balanced_microshard_plan(
        routing_group_work_units(operator),
        target_microshards=target,
        problem_fingerprint=operator.assignment_fingerprint,
    )


def test_fixed_shape_batch_planning_is_complete_deterministic_and_padded():
    with TemporaryDirectory() as temporary:
        operator = _operator(Path(temporary))
        plan = _plan(operator)
        first = plan_fixed_shape_routing_batches(
            plan, supported_group_batch_sizes=(1, 2, 4)
        )
        second = plan_fixed_shape_routing_batches(
            plan, supported_group_batch_sizes=(1, 2, 4)
        )
        assert first == second
        assert [item.padded_groups for item in first] == [4, 4]
        assert sorted(
            group for item in first for group in item.destination_group_indices
        ) == list(range(8))
        with pytest.raises(ValueError, match="unknown selected"):
            plan_fixed_shape_routing_batches(plan, selected_work_ids=("missing",))


def test_parallel_all_work_matches_existing_exact_forward_and_reverse():
    with TemporaryDirectory() as temporary:
        operator = _operator(Path(temporary))
        plan = _plan(operator)
        config = ParallelRoutingExecutorConfig(
            worker_count=4, supported_group_batch_sizes=(1, 2)
        )
        demand = np.linspace(0.1, 2.0, operator.num_free_od, dtype=np.float32)
        cotangent = np.linspace(-1.0, 1.0, operator.num_measurements, dtype=np.float32)
        expected_forward = operator.matvec(demand)
        expected_reverse = operator.rmatvec(cotangent)
        with PersistentParallelRoutingExecutor(
            operator=operator, microshard_plan=plan, config=config
        ) as executor:
            forward = executor.execute("matvec", demand)
            reverse = executor.execute("rmatvec", cotangent)
            np.testing.assert_allclose(
                forward.value, expected_forward, rtol=1.0e-5, atol=1.0e-5
            )
            np.testing.assert_allclose(
                reverse.value, expected_reverse, rtol=1.0e-5, atol=1.0e-5
            )
            assert len(operator._partial_compiled_forward) == 1
            assert len(operator._partial_compiled_reverse) == 1
            assert forward.worker_count == min(4, __import__("os").cpu_count() or 1)
            assert forward.dispatch_order == tuple(
                item.batch_id
                for item in sorted(
                    plan_fixed_shape_routing_batches(
                        plan, supported_group_batch_sizes=(1, 2)
                    ),
                    key=lambda item: (-item.predicted_cost, item.batch_id),
                )
            )


class _SlowAdditiveOperator:
    assignment_fingerprint = "fake-problem"

    def __init__(self):
        self.lock = Lock()
        self.active = 0
        self.peak_active = 0

    def _run(self, vector, destination_group_indices, **_):
        with self.lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        sleep(0.02)
        with self.lock:
            self.active -= 1
        return np.asarray(vector) * len(destination_group_indices)

    partial_matvec = _run
    partial_rmatvec = _run


class _FailingOperator(_SlowAdditiveOperator):
    def _run(self, vector, destination_group_indices, **kwargs):
        if 3 in destination_group_indices:
            raise RuntimeError("deliberate worker failure")
        return super()._run(vector, destination_group_indices, **kwargs)

    partial_matvec = _run
    partial_rmatvec = _run


def _fake_plan():
    units = tuple(
        RoutingWorkUnit(
            work_id=f"group-{index}",
            destination_group_indices=(index,),
            predicted_cost=float(8 - index),
            routing_bytes=1,
            active_od_cells=1,
            support_entries=1,
            measurement_support=1,
        )
        for index in range(8)
    )
    return build_balanced_microshard_plan(
        units, target_microshards=8, problem_fingerprint="fake-problem"
    )


def test_persistent_workers_run_in_parallel_reuse_pool_and_close_cleanly():
    operator = _SlowAdditiveOperator()
    executor = PersistentParallelRoutingExecutor(
        operator=operator,
        microshard_plan=_fake_plan(),
        config=ParallelRoutingExecutorConfig(
            worker_count=4, supported_group_batch_sizes=(1,)
        ),
    )
    first = executor.execute("matvec", np.ones(2))
    second = executor.execute("matvec", np.ones(2))
    assert operator.peak_active >= 2
    assert len(first.worker_thread_ids) >= 2
    assert set(second.worker_thread_ids).issubset(
        set(first.worker_thread_ids) | set(second.worker_thread_ids)
    )
    np.testing.assert_array_equal(first.value, np.full(2, 8.0))
    executor.close()
    assert executor.closed
    executor.close()
    with pytest.raises(RuntimeError, match="closed"):
        executor.execute("matvec", np.ones(2))


def test_cancellation_is_checked_at_dynamic_batch_boundaries():
    executor = PersistentParallelRoutingExecutor(
        operator=_SlowAdditiveOperator(),
        microshard_plan=_fake_plan(),
        config=ParallelRoutingExecutorConfig(
            worker_count=2, supported_group_batch_sizes=(1,)
        ),
    )
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        return calls > 2

    with pytest.raises(ParallelRoutingExecutionInterrupted, match="cancelled"):
        executor.execute("matvec", np.ones(2), cancellation_requested=cancel)
    executor.close()


def test_worker_failure_propagates_and_executor_can_close():
    executor = PersistentParallelRoutingExecutor(
        operator=_FailingOperator(),
        microshard_plan=_fake_plan(),
        config=ParallelRoutingExecutorConfig(
            worker_count=2, supported_group_batch_sizes=(1,)
        ),
    )
    with pytest.raises(RuntimeError, match="deliberate worker failure"):
        executor.execute("matvec", np.ones(2))
    executor.close()
    assert executor.closed


def test_weighted_partial_products_share_selection_and_satisfy_adjoint_identity():
    with TemporaryDirectory() as temporary:
        operator = _operator(Path(temporary))
        plan = _plan(operator)
        selection = plan_fixed_budget_routing_selection(
            plan, effort_percent=50, seed=3
        )
        demand = np.linspace(0.1, 2.0, operator.num_free_od, dtype=np.float32)
        cotangent = np.linspace(-1.0, 1.0, operator.num_measurements, dtype=np.float32)
        with PersistentParallelRoutingExecutor(
            operator=operator,
            microshard_plan=plan,
            config=ParallelRoutingExecutorConfig(
                worker_count=4, supported_group_batch_sizes=(1, 2)
            ),
        ) as executor:
            forward = executor.execute(
                "matvec",
                demand,
                selected_work_ids=selection.selected_work_ids,
                expansion_weights=selection.weight_by_work_id,
            )
            reverse = executor.execute(
                "rmatvec",
                cotangent,
                selected_work_ids=selection.selected_work_ids,
                expansion_weights=selection.weight_by_work_id,
            )
        np.testing.assert_allclose(
            np.dot(forward.value, cotangent),
            np.dot(demand, reverse.value),
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        assert forward.selected_work_ids == reverse.selected_work_ids


def test_forward_reverse_lifecycle_reuses_prepared_batches_and_releases_them():
    with TemporaryDirectory() as temporary:
        operator = _operator(Path(temporary))
        plan = _plan(operator)
        selection = plan_fixed_budget_routing_selection(
            plan, effort_percent=50, seed=3
        )
        config = ParallelRoutingExecutorConfig(
            worker_count=4,
            supported_group_batch_sizes=(1, 2),
            maximum_retained_batch_bytes=1024 * 1024,
        )
        with PersistentParallelRoutingExecutor(
            operator=operator, microshard_plan=plan, config=config
        ) as executor:
            forward = executor.forward_evaluation(
                "evaluation",
                np.ones(operator.num_free_od, dtype=np.float32),
                selected_work_ids=selection.selected_work_ids,
                expansion_weights=selection.weight_by_work_id,
            )
            assert forward.retained_batch_count > 0
            reverse = executor.reverse_evaluation(
                "evaluation",
                np.ones(operator.num_measurements, dtype=np.float32),
            )
            assert sum(item.prepared_cache_hit for item in reverse.observations) == (
                forward.retained_batch_count
            )
            with pytest.raises(ValueError, match="unknown or released"):
                executor.reverse_evaluation(
                    "evaluation", np.ones(operator.num_measurements, dtype=np.float32)
                )
