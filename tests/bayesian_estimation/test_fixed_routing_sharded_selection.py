from __future__ import annotations

import pytest

from public_transportation.inference.fixed_routing_sharded_builder import (
    ConstructionTask,
    ShardedConstructionConfig,
    ShardedConstructionPlan,
    pack_storage_shards,
)
from public_transportation.inference.sharded_sparse_operator import SparseShardIdentity
from public_transportation.inference.fixed_routing_sharded_selection import (
    ShardedSelectionConfig,
    select_sharded_fixed_routing_backend,
)


def _plan(*, safe=True) -> ShardedConstructionPlan:
    return ShardedConstructionPlan(
        num_measurements=1_000,
        num_free_od=2_000,
        num_active_od=2_000,
        num_groups=10,
        num_shards=20,
        candidate_entries=20_000,
        maximum_group_measurements=500,
        maximum_shard_measurements=256,
        estimated_kernel_bytes=10_000_000,
        worker_memory_budget_bytes=20_000_000 if safe else 1,
        safe=safe,
        reason="test",
        expected_shards=(),
    )


def test_complete_cache_is_preferred_as_sunk_cost():
    decision = select_sharded_fixed_routing_backend(
        plan=_plan(), cache_status="complete"
    )
    assert decision.selected_mode == "sharded"
    assert "sunk cost" in decision.reason
    assert decision.candidate_density_upper_bound == pytest.approx(0.01)


def test_one_use_cold_run_can_remain_matrix_free():
    decision = select_sharded_fixed_routing_backend(
        plan=_plan(),
        cache_status="none",
        config=ShardedSelectionConfig(
            expected_products=2,
            estimated_construction_seconds=100.0,
            matrix_free_product_seconds=1.0,
            sharded_product_seconds=0.001,
        ),
    )
    assert decision.selected_mode == "matrix_free"
    assert decision.estimated_break_even_products == pytest.approx(100.0 / 0.999)


def test_partial_cache_is_resumed_but_unsafe_plan_is_rejected():
    resumed = select_sharded_fixed_routing_backend(
        plan=_plan(), cache_status="partial"
    )
    assert resumed.selected_mode == "sharded"
    unsafe = select_sharded_fixed_routing_backend(
        plan=_plan(safe=False), cache_status="none"
    )
    assert unsafe.selected_mode == "matrix_free"
    with pytest.raises(MemoryError):
        select_sharded_fixed_routing_backend(
            plan=_plan(safe=False),
            cache_status="none",
            config=ShardedSelectionConfig(mode="sharded"),
        )


def test_pathological_tiny_tasks_are_packed_into_bounded_storage_shards():
    tasks = tuple(
        ConstructionTask(
            identity=SparseShardIdentity(
                group=index % 113,
                measurement_block=0,
                first_measurement_position=index % 3_690,
                measurement_count=1,
                support_pattern=index,
            ),
            group=index % 113,
            od_indices=(index,),
            measurements=(index % 3_690,),
            estimated_nonzeros=10,
        )
        for index in range(15_748)
    )
    config = ShardedConstructionConfig(
        target_nonzeros_per_storage_shard=2_000,
        maximum_nonzeros_per_storage_shard=2_500,
        maximum_patterns_per_storage_shard=256,
    )
    first = pack_storage_shards(tasks, config=config, itemsize=8)
    second = pack_storage_shards(tuple(reversed(tasks)), config=config, itemsize=8)
    assert 64 <= len(first) <= 256
    assert len(first) < len(tasks) // 100
    assert [item.task_keys for item in first] == [item.task_keys for item in second]
    assert sum(item.estimated_nonzeros for item in first) == 157_480
