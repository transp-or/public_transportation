from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.inference.assignment_adapter import AssignmentInputs
from public_transportation.inference.sharded_fixed_routing import (
    FixedRoutingPreparationConfig,
    FixedRoutingShard,
    FixedRoutingShardCacheError,
    FixedRoutingShardDescriptor,
    FixedRoutingShardProgress,
    _RoutingProgressHeartbeat,
    build_sharded_fixed_routing_inputs,
    load_fixed_routing_shard,
    materialize_sharded_fixed_routing_dense,
    plan_fixed_routing_shards,
    save_fixed_routing_shard,
)


def _inputs() -> AssignmentInputs:
    graph = SimpleNamespace(
        num_nodes=3,
        num_links=2,
        tail=np.array([0, 1], dtype=np.int32),
        head=np.array([1, 2], dtype=np.int32),
        topo_order=np.array([0, 1, 2], dtype=np.int32),
        out_links=np.array([[0], [1], [0]], dtype=np.int32),
        out_mask=np.array([[True], [True], [False]]),
    )
    return AssignmentInputs(
        graph=graph,
        base_link_cost=jnp.array([1.0, 2.0], dtype=jnp.float32),
        group_dest_node=jnp.array([1, 2, 2], dtype=jnp.int32),
        group_link_mask=jnp.array(
            [[True, False], [True, True], [False, True]], dtype=bool
        ),
        od_origin_node=jnp.array([0, 0, 1], dtype=jnp.int32),
        group_od_index_padded=jnp.array([[0], [1], [2]], dtype=jnp.int32),
        group_od_mask=jnp.ones((3, 1), dtype=bool),
    )


def _progress_event(*, completed_groups: int, phase: str = "shard_persisted"):
    return FixedRoutingShardProgress(
        phase=phase,
        status="completed",
        completed_groups=completed_groups,
        total_groups=16,
        completed_shards=completed_groups,
        total_shards=16,
        shard_index=None,
        cache_hits=0,
        cache_misses=16,
        elapsed_seconds=0.0,
        recent_shard_seconds=None,
        estimated_remaining_seconds=None,
        peak_rss_bytes=None,
        retained_cache_bytes=0,
        deadline_remaining_seconds=None,
    )


def test_progress_intervals_are_independent_and_serializable():
    seconds_only = FixedRoutingPreparationConfig(progress_interval_seconds=5.0)
    groups_only = FixedRoutingPreparationConfig(progress_interval_groups=3)
    both = FixedRoutingPreparationConfig(
        progress_interval_seconds=2.5,
        progress_interval_groups=6,
    )

    assert seconds_only.progress_interval_seconds == 5.0
    assert seconds_only.progress_interval_groups == 8
    assert groups_only.progress_interval_seconds == 1.0
    assert groups_only.progress_interval_groups == 3
    assert both.progress_configuration() == {
        "progress_interval_seconds": 2.5,
        "progress_interval_groups": 6,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("progress_interval_seconds", 0.0, "positive and finite"),
        ("progress_interval_seconds", float("nan"), "positive and finite"),
        ("progress_interval_seconds", float("inf"), "positive and finite"),
        ("progress_interval_groups", 0, "positive integer"),
        ("progress_interval_groups", -1, "positive integer"),
        ("progress_interval_groups", 1.5, "positive integer"),
        ("progress_interval_groups", True, "positive integer"),
    ],
)
def test_progress_interval_validation_names_field(field, value, message):
    with pytest.raises(ValueError, match=field):
        FixedRoutingPreparationConfig(**{field: value})
    with pytest.raises(ValueError, match=message):
        FixedRoutingPreparationConfig(**{field: value})


def test_group_progress_requires_group_boundary_and_wall_clock_interval():
    now = [0.0]
    events = []
    heartbeat = _RoutingProgressHeartbeat(
        events.append,
        _progress_event(completed_groups=0),
        interval_seconds=5.0,
        interval_groups=2,
        clock=lambda: now[0],
    )

    heartbeat.emit(_progress_event(completed_groups=1))
    assert events == []
    now[0] = 5.0
    heartbeat.emit(_progress_event(completed_groups=1))
    assert events == []
    heartbeat.emit(_progress_event(completed_groups=2))
    assert len(events) == 1
    assert events[0].completed_groups == 2

    now[0] = 6.0
    heartbeat.emit(_progress_event(completed_groups=4))
    assert len(events) == 1
    now[0] = 10.0
    heartbeat.emit(_progress_event(completed_groups=4))
    assert len(events) == 2
    assert events[-1].completed_groups == 4

    heartbeat.emit(_progress_event(completed_groups=4, phase="terminal"))
    assert len(events) == 3
    heartbeat.close()


def test_sharded_metadata_is_stable_and_contains_no_routing_payload(tmp_path):
    inputs = _inputs()
    partition = (
        FixedRoutingShardDescriptor(0, 0, 2),
        FixedRoutingShardDescriptor(1, 2, 3),
    )

    first = build_sharded_fixed_routing_inputs(
        inputs=inputs,
        theta=1.0,
        shard_partition=partition,
        cache_directory=tmp_path,
    )
    repeated = build_sharded_fixed_routing_inputs(
        inputs=inputs,
        theta=1.0,
        shard_partition=partition,
        cache_directory=tmp_path,
    )

    assert first.num_destination_groups == 3
    assert first.num_shards == 2
    assert first.destination_group_identifiers == (1, 2, 2)
    assert first.provenance.preparation_fingerprint == (
        repeated.provenance.preparation_fingerprint
    )
    assert not hasattr(first, "effective_group_link_mask")
    assert not hasattr(first, "group_link_probability")
    assert first.graph is inputs.graph


def test_shard_payload_is_bounded_read_only_and_canonically_indexed():
    descriptor = FixedRoutingShardDescriptor(1, 2, 3)
    shard = FixedRoutingShard(
        descriptor=descriptor,
        effective_group_link_mask=np.array([[True, False]]),
        group_link_probability=np.array([[0.75, 0.0]], dtype=np.float32),
    )

    assert shard.destination_group_indices == (2,)
    assert shard.retained_bytes == 10
    assert not shard.effective_group_link_mask.flags.writeable
    assert not shard.group_link_probability.flags.writeable
    with pytest.raises(ValueError):
        shard.group_link_probability[0, 0] = 1.0


def test_sharded_metadata_rejects_noncanonical_or_incomplete_partition(tmp_path):
    inputs = _inputs()

    with pytest.raises(ValueError, match="contiguous canonical partition"):
        build_sharded_fixed_routing_inputs(
            inputs=inputs,
            theta=1.0,
            shard_partition=(FixedRoutingShardDescriptor(0, 1, 3),),
            cache_directory=tmp_path,
        )

    with pytest.raises(ValueError, match="cover every destination group"):
        build_sharded_fixed_routing_inputs(
            inputs=inputs,
            theta=1.0,
            shard_partition=(FixedRoutingShardDescriptor(0, 0, 2),),
            cache_directory=tmp_path,
        )


def test_shard_payload_rejects_wrong_shape_or_invalid_probability():
    descriptor = FixedRoutingShardDescriptor(0, 0, 2)
    with pytest.raises(ValueError, match="rows"):
        FixedRoutingShard(
            descriptor=descriptor,
            effective_group_link_mask=np.ones((1, 2), dtype=bool),
            group_link_probability=np.ones((1, 2), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        FixedRoutingShard(
            descriptor=descriptor,
            effective_group_link_mask=np.ones((2, 2), dtype=bool),
            group_link_probability=np.array(
                [[1.0, 0.0], [-1.0, np.nan]], dtype=np.float32
            ),
        )


def test_planner_enforces_group_and_byte_ceilings_deterministically(tmp_path):
    inputs = _inputs()
    bytes_per_group = 2 * (1 + 4)
    config = FixedRoutingPreparationConfig(
        maximum_groups_per_shard=3,
        maximum_retained_bytes_per_shard=2 * bytes_per_group,
        maximum_temporary_bytes=4 * bytes_per_group,
        cache_directory=tmp_path,
    )

    first = plan_fixed_routing_shards(inputs=inputs, config=config)
    repeated = plan_fixed_routing_shards(inputs=inputs, config=config)

    assert first.groups_per_full_shard == 2
    assert first.descriptors == (
        FixedRoutingShardDescriptor(0, 0, 2),
        FixedRoutingShardDescriptor(1, 2, 3),
    )
    assert first.plan_fingerprint == repeated.plan_fingerprint


def test_planner_rejects_when_one_group_exceeds_either_budget(tmp_path):
    inputs = _inputs()
    with pytest.raises(ValueError, match="retained-byte"):
        plan_fixed_routing_shards(
            inputs=inputs,
            config=FixedRoutingPreparationConfig(
                maximum_retained_bytes_per_shard=9,
                cache_directory=tmp_path,
            ),
        )
    with pytest.raises(ValueError, match="temporary-byte"):
        plan_fixed_routing_shards(
            inputs=inputs,
            config=FixedRoutingPreparationConfig(
                maximum_temporary_bytes=19,
                cache_directory=tmp_path,
            ),
        )


def test_shard_persistence_is_atomic_validated_and_read_only(tmp_path):
    inputs = _inputs()
    partition = (FixedRoutingShardDescriptor(0, 0, 3),)
    routing = build_sharded_fixed_routing_inputs(
        inputs=inputs,
        theta=1.0,
        shard_partition=partition,
        cache_directory=tmp_path,
    )
    shard = FixedRoutingShard(
        descriptor=partition[0],
        effective_group_link_mask=np.ones((3, 2), dtype=bool),
        group_link_probability=np.full((3, 2), 0.5, dtype=np.float32),
    )

    path = save_fixed_routing_shard(routing=routing, shard=shard)
    loaded = load_fixed_routing_shard(routing=routing, descriptor=partition[0])

    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))
    np.testing.assert_array_equal(
        loaded.group_link_probability, shard.group_link_probability
    )
    assert not loaded.group_link_probability.flags.writeable

    path.write_bytes(b"not an npz")
    with pytest.raises(FixedRoutingShardCacheError, match="corrupt"):
        load_fixed_routing_shard(routing=routing, descriptor=partition[0])


def test_dense_materialization_requires_explicit_sufficient_memory(tmp_path):
    inputs = _inputs()
    partition = (FixedRoutingShardDescriptor(0, 0, 3),)
    routing = build_sharded_fixed_routing_inputs(
        inputs=inputs,
        theta=1.0,
        shard_partition=partition,
        cache_directory=tmp_path,
    )
    shard = FixedRoutingShard(
        descriptor=partition[0],
        effective_group_link_mask=np.ones((3, 2), dtype=bool),
        group_link_probability=np.full((3, 2), 0.5, dtype=np.float32),
    )
    save_fixed_routing_shard(routing=routing, shard=shard)

    with pytest.raises(MemoryError, match="above the explicit"):
        materialize_sharded_fixed_routing_dense(
            routing=routing,
            inputs=inputs,
            memory_limit_bytes=29,
        )
    dense = materialize_sharded_fixed_routing_dense(
        routing=routing,
        inputs=inputs,
        memory_limit_bytes=30,
    )
    np.testing.assert_array_equal(
        dense.effective_group_link_mask, shard.effective_group_link_mask
    )
    np.testing.assert_array_equal(
        dense.group_link_probability, shard.group_link_probability
    )
