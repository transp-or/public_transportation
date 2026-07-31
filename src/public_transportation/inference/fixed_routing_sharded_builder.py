"""Bounded, resumable construction of fixed-routing sparse operator shards."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse

from .assignment_adapter import (
    AssignmentInputs,
    FixedRoutingInputs,
    validate_fixed_routing_compatibility,
)
from .compact_od_assignment_layout import CompactODAssignmentLayout
from .fixed_routing_measurement_operator import (
    _mapping_slots,
    assignment_inputs_fingerprint,
    measurement_mapping_fingerprint,
)
from .fixed_routing_origin_support import (
    OriginSupportConfig,
    analyze_fixed_routing_origin_support,
)
from .sharded_sparse_operator import (
    ShardedOperatorManifest,
    SparseShardIdentity,
    SparseShardMetrics,
    load_sharded_operator_manifest,
    load_sparse_shard,
    manifest_path,
    save_sharded_operator_manifest,
    save_sparse_shard,
    shard_path,
)


@dataclass(frozen=True, slots=True)
class ShardedConstructionConfig:
    od_chunk_size: int = 128
    measurement_block_size: int = 512
    worker_memory_budget_bytes: int = 512 * 1024 * 1024
    zero_tolerance: float = 0.0
    compressed_shards: bool = False
    workers: int = 1
    origin_support_chunk_size: int = 64
    support_edge_block_size: int = 2048
    target_nonzeros_per_storage_shard: int = 2048
    maximum_nonzeros_per_storage_shard: int = 8192
    maximum_patterns_per_storage_shard: int = 256
    maximum_storage_shards: int = 256
    maximum_manifest_bytes: int = 4 * 1024 * 1024
    maximum_filesystem_operations: int = 4096
    maximum_sparse_calls_per_product: int = 256
    manifest_checkpoint_shards: int = 16
    maximum_construction_dispatches: int = 4096

    def __post_init__(self) -> None:
        if (
            self.od_chunk_size <= 0
            or self.measurement_block_size <= 0
            or self.origin_support_chunk_size <= 0
            or self.support_edge_block_size <= 0
        ):
            raise ValueError("OD chunk and measurement block sizes must be positive.")
        if self.worker_memory_budget_bytes <= 0:
            raise ValueError("worker memory budget must be positive.")
        if not math.isfinite(self.zero_tolerance) or self.zero_tolerance < 0:
            raise ValueError("zero_tolerance must be finite and non-negative.")
        if self.workers <= 0:
            raise ValueError("workers must be positive.")
        if not (
            0 < self.target_nonzeros_per_storage_shard
            <= self.maximum_nonzeros_per_storage_shard
        ):
            raise ValueError("storage-shard nonzero targets are invalid.")
        if min(
            self.maximum_patterns_per_storage_shard,
            self.maximum_storage_shards,
            self.maximum_manifest_bytes,
            self.maximum_filesystem_operations,
            self.maximum_sparse_calls_per_product,
            self.manifest_checkpoint_shards,
            self.maximum_construction_dispatches,
        ) <= 0:
            raise ValueError("operational shard limits must be positive.")


@dataclass(frozen=True, slots=True)
class SupportPattern:
    """Reusable origin-specific topology support within one destination group."""

    group: int
    pattern: int
    od_indices: tuple[int, ...]
    measurements: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ConstructionTask:
    """Bounded numerical work unit; several tasks may share one stored shard."""

    identity: SparseShardIdentity
    group: int
    od_indices: tuple[int, ...]
    measurements: tuple[int, ...]
    estimated_nonzeros: int


@dataclass(frozen=True, slots=True)
class StorageShardPlan:
    """Deterministic aggregate persistence and solver unit."""

    identity: SparseShardIdentity
    task_keys: tuple[str, ...]
    estimated_nonzeros: int
    estimated_uncompressed_bytes: int


@dataclass(frozen=True, slots=True)
class ShardedConstructionPlan:
    num_measurements: int
    num_free_od: int
    num_active_od: int
    num_groups: int
    num_shards: int
    candidate_entries: int
    maximum_group_measurements: int
    maximum_shard_measurements: int
    estimated_kernel_bytes: int
    worker_memory_budget_bytes: int
    safe: bool
    reason: str
    expected_shards: tuple[SparseShardIdentity, ...]
    support_patterns: int = 0
    group_level_candidate_entries: int = 0
    origin_support_seconds: float = 0.0
    maximum_support_edges: int = 0
    construction_tasks: int = 0
    storage_shards: tuple[StorageShardPlan, ...] = ()
    estimated_nonzeros_per_storage_shard: tuple[int, ...] = ()
    estimated_payload_min: int = 0
    estimated_payload_median: float = 0.0
    estimated_payload_p90: float = 0.0
    estimated_payload_p99: float = 0.0
    estimated_payload_max: int = 0
    estimated_filesystem_file_count: int = 0
    estimated_manifest_bytes: int = 0
    estimated_manifest_writes: int = 0
    estimated_cumulative_manifest_bytes: int = 0
    estimated_sparse_calls_per_product: int = 0
    estimated_eager_cache_opens: int = 0
    estimated_lru_cache_opens_per_product: int = 0
    estimated_construction_batches: int = 0
    estimated_construction_dispatches: int = 0
    estimated_batch_temporary_bytes: int = 0


def _make_forward_support_kernel(*, graph, chunk_size: int, edge_block_size: int):
    """Compile origin reachability once, then gather only supported link values."""
    topo = graph.topo_order
    out_links = graph.out_links
    out_mask = graph.out_mask
    head = graph.head
    tail = graph.tail
    num_nodes = graph.num_nodes

    def kernel(
        origin_nodes,
        valid_origins,
        link_probability,
        enabled_link_mask,
        selected_links,
        selected_link_mask,
    ):
        reach = jnp.zeros((chunk_size, num_nodes), dtype=link_probability.dtype)
        reach = reach.at[jnp.arange(chunk_size), origin_nodes].set(
            valid_origins.astype(link_probability.dtype)
        )

        def step(values, node):
            links = out_links[node]
            adjacency = out_mask[node]
            safe_links = jnp.where(adjacency, links, 0)
            enabled = adjacency & enabled_link_mask[safe_links]
            contribution = (
                values[:, node, None]
                * link_probability[safe_links][None, :]
                * enabled[None, :]
            )
            values = values.at[:, head[safe_links]].add(contribution)
            return values, None

        reach, _ = jax.lax.scan(step, reach, topo)
        safe_selected = jnp.where(selected_link_mask, selected_links, 0)
        return (
            reach[:, tail[safe_selected]]
            * link_probability[safe_selected][None, :]
            * enabled_link_mask[safe_selected][None, :]
            * selected_link_mask[None, :]
        )

    return jax.jit(kernel)


@dataclass(frozen=True, slots=True)
class ShardedConstructionResult:
    directory: Path
    manifest: ShardedOperatorManifest
    plan: ShardedConstructionPlan
    reused_shards: int
    rebuilt_shards: int
    rejected_shards: int
    support_discovery_seconds: float
    lowering_seconds: float
    compilation_seconds: float
    dispatch_seconds: float
    synchronization_seconds: float
    transfer_seconds: float
    zero_filtering_seconds: float
    shard_persistence_seconds: float
    manifest_seconds: float
    manifest_write_count: int
    cumulative_manifest_bytes: int
    recovery_scan_seconds: float
    finalization_seconds: float
    total_seconds: float
    construction_batches: int = 0
    dispatch_count: int = 0
    synchronization_count: int = 0
    support_edge_blocks: int = 0
    origins_per_dispatch: tuple[int, ...] = ()
    supported_edges_per_dispatch: tuple[int, ...] = ()
    output_values_per_dispatch: tuple[int, ...] = ()
    dispatch_time_quantiles: dict[str, float] | None = None
    synchronization_time_quantiles: dict[str, float] | None = None
    group_timing_seconds: dict[str, float] | None = None
    padded_buffer_allocations: int = 0
    routing_array_dispatch_uses: int = 0


@dataclass(frozen=True, slots=True)
class _Support:
    free_column: np.ndarray
    fixed_by_active: np.ndarray
    selected: np.ndarray
    group_od_indices: tuple[np.ndarray, ...]
    group_measurements: tuple[np.ndarray, ...]
    global_slots: np.ndarray
    global_slot_mask: np.ndarray
    shard_od_indices: dict[str, np.ndarray]
    shard_measurements: dict[str, np.ndarray]
    expected_shards: tuple[SparseShardIdentity, ...]
    group_level_candidate_entries: int
    origin_support_seconds: float
    support_patterns: int
    patterns: tuple[SupportPattern, ...]
    construction_tasks: tuple[ConstructionTask, ...]
    storage_task_keys: dict[str, tuple[str, ...]]


def _timing_quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("minimum", "median", "p90", "p99", "maximum")}
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(array.max()),
    }


def _discover_support(
    *,
    inputs: AssignmentInputs,
    routing: FixedRoutingInputs,
    spec,
    compact_layout: CompactODAssignmentLayout,
    config: ShardedConstructionConfig,
) -> _Support:
    num_active = int(inputs.od_origin_node.shape[0])
    free_column = np.full(num_active, -1, np.int64)
    free_indices = np.asarray(compact_layout.free_compact_indices, dtype=np.int64)
    free_column[free_indices] = np.arange(free_indices.size, dtype=np.int64)
    fixed_by_active = np.zeros(num_active, dtype=np.dtype(inputs.base_link_cost.dtype))
    fixed_by_active[np.asarray(compact_layout.fixed_compact_indices, dtype=np.int64)] = (
        np.asarray(
            compact_layout.fixed_compact_values,
            dtype=np.dtype(inputs.base_link_cost.dtype),
        )
    )
    selected = (free_column >= 0) | (fixed_by_active != 0.0)
    global_slots, global_slot_mask = _mapping_slots(spec, inputs.graph.num_links)
    effective = np.asarray(routing.effective_group_link_mask)
    group_indices = np.asarray(inputs.group_od_index_padded)
    group_masks = np.asarray(inputs.group_od_mask)
    od_groups: list[np.ndarray] = []
    measurements: list[np.ndarray] = []
    analyzed = analyze_fixed_routing_origin_support(
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact_layout,
        config=OriginSupportConfig(
            origin_chunk_size=config.origin_support_chunk_size,
            worker_memory_budget_bytes=config.worker_memory_budget_bytes,
            materialize=True,
        ),
    )
    assert analyzed.free_support is not None
    assert analyzed.positive_fixed_support is not None
    free_support = analyzed.free_support.tocsc()
    fixed_support = analyzed.positive_fixed_support.tocsc()
    support_by_active: dict[int, tuple[int, ...]] = {}
    for column, active_index in enumerate(free_indices):
        support_by_active[int(active_index)] = tuple(
            int(value)
            for value in free_support.indices[
                free_support.indptr[column] : free_support.indptr[column + 1]
            ]
        )
    positive_fixed = np.asarray(analyzed.positive_fixed_active_indices, dtype=np.int64)
    for column, active_index in enumerate(positive_fixed):
        support_by_active[int(active_index)] = tuple(
            int(value)
            for value in fixed_support.indices[
                fixed_support.indptr[column] : fixed_support.indptr[column + 1]
            ]
        )
    identities: list[SparseShardIdentity] = []
    discovered_patterns: list[SupportPattern] = []
    tasks: list[ConstructionTask] = []
    shard_od_indices: dict[str, np.ndarray] = {}
    shard_measurements: dict[str, np.ndarray] = {}
    pattern_count = 0
    for group in range(int(inputs.group_dest_node.shape[0])):
        relevant = group_indices[group][group_masks[group]]
        relevant = relevant[selected[relevant]].astype(np.int64, copy=False)
        od_groups.append(relevant)
        mapped = global_slot_mask & effective[group, :, None]
        measurements.append(
            np.unique(global_slots[mapped]).astype(np.int64, copy=False)
        )
        patterns: dict[tuple[int, ...], list[int]] = {}
        for active_index in relevant:
            pattern = support_by_active[int(active_index)]
            if pattern:
                patterns.setdefault(pattern, []).append(int(active_index))
        for pattern_id, pattern in enumerate(sorted(patterns)):
            pattern_count += 1
            pattern_measurements = np.asarray(pattern, dtype=np.int64)
            pattern_od = np.asarray(patterns[pattern], dtype=np.int64)
            discovered_patterns.append(
                SupportPattern(
                    group=group,
                    pattern=pattern_id,
                    od_indices=tuple(int(value) for value in pattern_od),
                    measurements=tuple(int(value) for value in pattern_measurements),
                )
            )
            for block, first in enumerate(
                range(0, pattern_measurements.size, config.measurement_block_size)
            ):
                count = min(
                    config.measurement_block_size,
                    pattern_measurements.size - first,
                )
                identity = SparseShardIdentity(
                    group=group,
                    measurement_block=block,
                    first_measurement_position=first,
                    measurement_count=count,
                    support_pattern=pattern_id,
                )
                identities.append(identity)
                shard_od_indices[identity.key] = pattern_od
                task_measurements = pattern_measurements[first : first + count]
                shard_measurements[identity.key] = task_measurements
                tasks.append(
                    ConstructionTask(
                        identity=identity,
                        group=group,
                        od_indices=tuple(int(value) for value in pattern_od),
                        measurements=tuple(int(value) for value in task_measurements),
                        estimated_nonzeros=int(pattern_od.size * task_measurements.size),
                    )
                )
    return _Support(
        free_column=free_column,
        fixed_by_active=fixed_by_active,
        selected=selected,
        group_od_indices=tuple(od_groups),
        group_measurements=tuple(measurements),
        global_slots=global_slots,
        global_slot_mask=global_slot_mask,
        shard_od_indices=shard_od_indices,
        shard_measurements=shard_measurements,
        expected_shards=tuple(identities),
        group_level_candidate_entries=(
            analyzed.metrics.group_level_candidate_entries
        ),
        origin_support_seconds=analyzed.metrics.support_discovery_seconds,
        support_patterns=pattern_count,
        patterns=tuple(discovered_patterns),
        construction_tasks=tuple(tasks),
        storage_task_keys={},
    )


def pack_storage_shards(
    tasks: tuple[ConstructionTask, ...],
    *,
    config: ShardedConstructionConfig,
    itemsize: int,
) -> tuple[StorageShardPlan, ...]:
    """Pack stable construction tasks into bounded aggregate solver units."""
    packed: list[StorageShardPlan] = []
    current: list[ConstructionTask] = []
    current_estimate = 0

    def flush() -> None:
        nonlocal current, current_estimate
        if not current:
            return
        rows = {row for task in current for row in task.measurements}
        identity = SparseShardIdentity(
            group=current[0].group,
            measurement_block=0,
            first_measurement_position=min(rows, default=0),
            measurement_count=len(rows),
            support_pattern=0,
            storage_shard=len(packed),
        )
        # COO payload plus conservative CSR row pointers and fixed-offset storage.
        estimated_bytes = int(current_estimate * (itemsize + 16) + (len(rows) + 1) * 8)
        packed.append(
            StorageShardPlan(
                identity=identity,
                task_keys=tuple(task.identity.key for task in current),
                estimated_nonzeros=current_estimate,
                estimated_uncompressed_bytes=estimated_bytes,
            )
        )
        current = []
        current_estimate = 0

    for task in sorted(
        tasks,
        key=lambda value: (
            value.group,
            value.identity.measurement_block,
            value.identity.support_pattern,
            value.identity.key,
        ),
    ):
        exceeds_hard = (
            current
            and current_estimate + task.estimated_nonzeros
            > config.maximum_nonzeros_per_storage_shard
        )
        reached_target = (
            current
            and current_estimate >= config.target_nonzeros_per_storage_shard
        )
        too_many_patterns = len(current) >= config.maximum_patterns_per_storage_shard
        if exceeds_hard or reached_target or too_many_patterns:
            flush()
        current.append(task)
        current_estimate += task.estimated_nonzeros
    flush()
    return tuple(packed)


def plan_sharded_fixed_routing_operator(
    *,
    inputs: AssignmentInputs,
    routing: FixedRoutingInputs,
    spec,
    compact_layout: CompactODAssignmentLayout,
    config: ShardedConstructionConfig | None = None,
) -> tuple[ShardedConstructionPlan, _Support]:
    """Discover structural support and reject unsafe kernels before lowering."""
    config = ShardedConstructionConfig() if config is None else config
    validate_fixed_routing_compatibility(inputs=inputs, routing=routing)
    if compact_layout.num_active != int(inputs.od_origin_node.shape[0]):
        raise ValueError("compact layout and assignment active dimensions differ.")
    support = _discover_support(
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact_layout,
        config=config,
    )
    task_identities = support.expected_shards
    candidates = sum(
        int(
            support.shard_od_indices[identity.key].size
            * support.shard_measurements[identity.key].size
        )
        for identity in task_identities
    )
    mapping_links = np.asarray(spec.link_index, dtype=np.int64)
    mapping_measurements = np.asarray(spec.measurement_index, dtype=np.int64)
    effective = np.asarray(routing.effective_group_link_mask, dtype=bool)
    maximum_support_edges = max(
        (
            int(
                np.count_nonzero(
                    effective[identity.group, mapping_links]
                    & np.isin(
                        mapping_measurements,
                        support.shard_measurements[identity.key],
                    )
                )
            )
            for identity in task_identities
        ),
        default=0,
    )
    itemsize = np.dtype(inputs.base_link_cost.dtype).itemsize
    storage_shards = pack_storage_shards(
        support.construction_tasks,
        config=config,
        itemsize=itemsize,
    )
    identities = tuple(shard.identity for shard in storage_shards)
    storage_task_keys = {
        shard.identity.key: shard.task_keys for shard in storage_shards
    }
    support = replace(support, storage_task_keys=storage_task_keys)
    # Forward reachability stores OD-chunk×node state and returns only one
    # bounded block of structurally supported mapping edges.
    estimated_kernel_bytes = int(
        config.od_chunk_size
        * (inputs.graph.num_nodes + config.support_edge_block_size)
        * itemsize
        * 2
    )
    payloads = np.asarray(
        [value.estimated_nonzeros for value in storage_shards], dtype=np.int64
    )
    manifest_writes = 1 + math.ceil(
        len(storage_shards) / config.manifest_checkpoint_shards
    )
    # JSON estimate is intentionally conservative and independent of private keys.
    manifest_bytes = 2048 + len(storage_shards) * 320
    filesystem_operations = len(storage_shards) * 2 + manifest_writes * 2
    task_by_key = {task.identity.key: task for task in support.construction_tasks}
    estimated_batches = 0
    estimated_dispatches = 0
    maximum_batch_rows = 0
    for storage_shard in storage_shards:
        tasks_by_group: dict[int, list[ConstructionTask]] = {}
        for key in storage_shard.task_keys:
            task = task_by_key[key]
            tasks_by_group.setdefault(task.group, []).append(task)
        for group, grouped_tasks in tasks_by_group.items():
            estimated_batches += 1
            batch_origins = {index for task in grouped_tasks for index in task.od_indices}
            batch_measurements = {
                row for task in grouped_tasks for row in task.measurements
            }
            maximum_batch_rows = max(maximum_batch_rows, len(batch_measurements))
            selected_edges = int(
                np.count_nonzero(
                    effective[group, mapping_links]
                    & np.isin(mapping_measurements, tuple(batch_measurements))
                )
            )
            estimated_dispatches += max(
                1, math.ceil(len(batch_origins) / config.od_chunk_size)
            ) * max(1, math.ceil(selected_edges / config.support_edge_block_size))
    operational_failures: list[str] = []
    if len(storage_shards) > config.maximum_storage_shards:
        operational_failures.append("storage-shard count exceeds configured maximum")
    if manifest_bytes > config.maximum_manifest_bytes:
        operational_failures.append("estimated manifest exceeds configured maximum")
    if filesystem_operations > config.maximum_filesystem_operations:
        operational_failures.append("estimated filesystem operations exceed configured maximum")
    if len(storage_shards) > config.maximum_sparse_calls_per_product:
        operational_failures.append("sparse calls per product exceed configured maximum")
    if estimated_dispatches > config.maximum_construction_dispatches:
        operational_failures.append("estimated construction dispatches exceed configured maximum")
    estimated_batch_temporary_bytes = int(
        config.od_chunk_size * maximum_batch_rows * itemsize
    )
    memory_safe = (
        estimated_kernel_bytes + estimated_batch_temporary_bytes
        <= config.worker_memory_budget_bytes
    )
    safe = memory_safe and not operational_failures
    if not memory_safe:
        reason = "node/OD block kernel estimate exceeds worker memory budget"
    elif operational_failures:
        reason = "; ".join(operational_failures)
    else:
        reason = "kernel memory and aggregate storage plan are operationally bounded"
    return (
        ShardedConstructionPlan(
            num_measurements=int(spec.num_measurements),
            num_free_od=compact_layout.num_free,
            num_active_od=compact_layout.num_active,
            num_groups=len(support.group_measurements),
            num_shards=len(identities),
            candidate_entries=candidates,
            maximum_group_measurements=max(
                (value.size for value in support.group_measurements), default=0
            ),
            maximum_shard_measurements=min(
                config.measurement_block_size,
                max((value.size for value in support.group_measurements), default=0),
            ),
            estimated_kernel_bytes=estimated_kernel_bytes,
            worker_memory_budget_bytes=config.worker_memory_budget_bytes,
            safe=safe,
            reason=reason,
            expected_shards=tuple(identities),
            support_patterns=support.support_patterns,
            group_level_candidate_entries=support.group_level_candidate_entries,
            origin_support_seconds=support.origin_support_seconds,
            maximum_support_edges=maximum_support_edges,
            construction_tasks=len(task_identities),
            storage_shards=storage_shards,
            estimated_nonzeros_per_storage_shard=tuple(int(value) for value in payloads),
            estimated_payload_min=int(payloads.min()) if payloads.size else 0,
            estimated_payload_median=float(np.median(payloads)) if payloads.size else 0.0,
            estimated_payload_p90=float(np.percentile(payloads, 90)) if payloads.size else 0.0,
            estimated_payload_p99=float(np.percentile(payloads, 99)) if payloads.size else 0.0,
            estimated_payload_max=int(payloads.max()) if payloads.size else 0,
            estimated_filesystem_file_count=len(storage_shards) + 1,
            estimated_manifest_bytes=manifest_bytes,
            estimated_manifest_writes=manifest_writes,
            estimated_cumulative_manifest_bytes=manifest_bytes * manifest_writes,
            estimated_sparse_calls_per_product=len(storage_shards),
            estimated_eager_cache_opens=len(storage_shards),
            estimated_lru_cache_opens_per_product=len(storage_shards),
            estimated_construction_batches=estimated_batches,
            estimated_construction_dispatches=estimated_dispatches,
            estimated_batch_temporary_bytes=estimated_batch_temporary_bytes,
        ),
        support,
    )


def _manifest(
    *,
    plan: ShardedConstructionPlan,
    config: ShardedConstructionConfig,
    provenance: dict[str, object],
    completed: set[str],
    aggregate_nonzeros: int,
) -> ShardedOperatorManifest:
    expected = {item.key for item in plan.expected_shards}
    plan_summary = {
        key: value
        for key, value in asdict(plan).items()
        if key not in {"expected_shards", "storage_shards"}
    }
    return ShardedOperatorManifest(
        num_measurements=plan.num_measurements,
        num_free_od=plan.num_free_od,
        dtype=str(provenance["dtype"]),
        provenance=provenance,
        expected_shards=plan.expected_shards,
        completed_shards=tuple(sorted(completed)),
        aggregate_nonzeros=aggregate_nonzeros,
        complete=completed == expected,
        measurement_block_size=config.measurement_block_size,
        od_chunk_size=config.od_chunk_size,
        plan_summary=plan_summary,
    )


def _plan_from_manifest(manifest: ShardedOperatorManifest) -> ShardedConstructionPlan:
    if manifest.plan_summary is None:
        raise ValueError("completed shard manifest has no construction plan summary.")
    summary = dict(manifest.plan_summary)
    summary["estimated_nonzeros_per_storage_shard"] = tuple(
        summary.get("estimated_nonzeros_per_storage_shard", ())
    )
    return ShardedConstructionPlan(
        expected_shards=manifest.expected_shards,
        storage_shards=(),
        **summary,
    )


def _construction_provenance(
    *,
    inputs: AssignmentInputs,
    spec,
    compact_layout: CompactODAssignmentLayout,
    assignment_fingerprint: str,
    od_layout_fingerprint: str,
    theta: float,
    config: ShardedConstructionConfig,
    assignment_inputs_fingerprint_value: str | None = None,
) -> dict[str, object]:
    return {
        "assignment_fingerprint": str(assignment_fingerprint),
        "assignment_inputs_fingerprint": (
            assignment_inputs_fingerprint(inputs)
            if assignment_inputs_fingerprint_value is None
            else str(assignment_inputs_fingerprint_value)
        ),
        "mapping_fingerprint": measurement_mapping_fingerprint(spec),
        "od_layout_fingerprint": str(od_layout_fingerprint),
        "compact_layout_fingerprint": compact_layout.fingerprint,
        "theta": float(theta),
        "dtype": str(np.dtype(inputs.base_link_cost.dtype)),
        "zero_tolerance": config.zero_tolerance,
        "measurement_block_size": config.measurement_block_size,
        "od_chunk_size": config.od_chunk_size,
        "origin_support_chunk_size": config.origin_support_chunk_size,
        "support_strategy": "positive_probability_group_batched_reachability_v2",
        "support_edge_block_size": config.support_edge_block_size,
        "target_nonzeros_per_storage_shard": config.target_nonzeros_per_storage_shard,
        "maximum_nonzeros_per_storage_shard": config.maximum_nonzeros_per_storage_shard,
        "maximum_patterns_per_storage_shard": config.maximum_patterns_per_storage_shard,
    }


def load_complete_sharded_fixed_routing_cache(
    *,
    directory: str | Path,
    inputs: AssignmentInputs,
    spec,
    compact_layout: CompactODAssignmentLayout,
    assignment_fingerprint: str,
    od_layout_fingerprint: str,
    theta: float,
    config: ShardedConstructionConfig | None = None,
    assignment_inputs_fingerprint_value: str | None = None,
) -> ShardedConstructionResult | None:
    """Return a compatible complete cache without routing or support planning."""
    config = ShardedConstructionConfig() if config is None else config
    directory = Path(directory)
    started = perf_counter()
    if not manifest_path(directory).exists():
        return None
    manifest = load_sharded_operator_manifest(directory)
    provenance = _construction_provenance(
        inputs=inputs,
        spec=spec,
        compact_layout=compact_layout,
        assignment_fingerprint=assignment_fingerprint,
        od_layout_fingerprint=od_layout_fingerprint,
        theta=theta,
        config=config,
        assignment_inputs_fingerprint_value=assignment_inputs_fingerprint_value,
    )
    if not manifest.complete or manifest.provenance != provenance:
        return None
    return ShardedConstructionResult(
        directory=directory,
        manifest=manifest,
        plan=_plan_from_manifest(manifest),
        reused_shards=len(manifest.expected_shards),
        rebuilt_shards=0,
        rejected_shards=0,
        support_discovery_seconds=0.0,
        lowering_seconds=0.0,
        compilation_seconds=0.0,
        dispatch_seconds=0.0,
        synchronization_seconds=0.0,
        transfer_seconds=0.0,
        zero_filtering_seconds=0.0,
        shard_persistence_seconds=0.0,
        manifest_seconds=0.0,
        manifest_write_count=0,
        cumulative_manifest_bytes=0,
        recovery_scan_seconds=0.0,
        finalization_seconds=0.0,
        total_seconds=perf_counter() - started,
    )
def prepare_sharded_fixed_routing_measurement_operator(
    *,
    directory: str | Path,
    inputs: AssignmentInputs,
    routing: FixedRoutingInputs,
    spec,
    compact_layout: CompactODAssignmentLayout,
    assignment_fingerprint: str,
    od_layout_fingerprint: str,
    config: ShardedConstructionConfig | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> ShardedConstructionResult:
    """Build missing/invalid shards and atomically advance a resumable manifest."""
    config = ShardedConstructionConfig() if config is None else config
    if config.workers != 1:
        raise NotImplementedError(
            "parallel shard construction is not enabled yet; use workers=1"
        )
    directory = Path(directory)
    total_start = perf_counter()
    provenance = _construction_provenance(
        inputs=inputs,
        spec=spec,
        compact_layout=compact_layout,
        assignment_fingerprint=assignment_fingerprint,
        od_layout_fingerprint=od_layout_fingerprint,
        theta=float(np.asarray(routing.theta)),
        config=config,
    )
    if manifest_path(directory).exists():
        existing = load_sharded_operator_manifest(directory)
        if existing.provenance != provenance:
            raise ValueError("existing shard manifest provenance is incompatible.")
        if existing.complete and existing.plan_summary is not None:
            valid = True
            aggregate_nonzeros = 0
            for identity in existing.expected_shards:
                try:
                    loaded = load_sparse_shard(
                        shard_path(directory, identity),
                        expected_provenance_hash=existing.provenance_hash,
                    )
                except (KeyError, OSError, ValueError):
                    valid = False
                    break
                aggregate_nonzeros += loaded.metadata.nonzero_entries
            if valid and aggregate_nonzeros == existing.aggregate_nonzeros:
                plan = _plan_from_manifest(existing)
                return ShardedConstructionResult(
                    directory=directory,
                    manifest=existing,
                    plan=plan,
                    reused_shards=len(existing.expected_shards),
                    rebuilt_shards=0,
                    rejected_shards=0,
                    support_discovery_seconds=0.0,
                    lowering_seconds=0.0,
                    compilation_seconds=0.0,
                    dispatch_seconds=0.0,
                    synchronization_seconds=0.0,
                    transfer_seconds=0.0,
                    zero_filtering_seconds=0.0,
                    shard_persistence_seconds=0.0,
                    manifest_seconds=0.0,
                    manifest_write_count=0,
                    cumulative_manifest_bytes=0,
                    recovery_scan_seconds=0.0,
                    finalization_seconds=0.0,
                    total_seconds=perf_counter() - total_start,
                )
    support_start = perf_counter()
    plan, support = plan_sharded_fixed_routing_operator(
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact_layout,
        config=config,
    )
    support_seconds = perf_counter() - support_start
    if not plan.safe:
        raise MemoryError(plan.reason)
    template = _manifest(
        plan=plan,
        config=config,
        provenance=provenance,
        completed=set(),
        aggregate_nonzeros=0,
    )
    if manifest_path(directory).exists():
        existing = load_sharded_operator_manifest(directory)
        if existing.provenance_hash != template.provenance_hash:
            raise ValueError("existing shard manifest provenance is incompatible.")

    recovery_start = perf_counter()
    completed: set[str] = set()
    aggregate_nonzeros = 0
    rejected = 0
    for identity in plan.expected_shards:
        path = shard_path(directory, identity)
        if not path.exists():
            continue
        try:
            loaded = load_sparse_shard(
                path, expected_provenance_hash=template.provenance_hash
            )
        except (KeyError, OSError, ValueError):
            rejected += 1
            continue
        completed.add(identity.key)
        aggregate_nonzeros += loaded.metadata.nonzero_entries
    recovery_scan_seconds = perf_counter() - recovery_start
    manifest_seconds = 0.0
    manifest_start = perf_counter()
    _, initial_manifest_bytes = save_sharded_operator_manifest(
        _manifest(
            plan=plan,
            config=config,
            provenance=provenance,
            completed=completed,
            aggregate_nonzeros=aggregate_nonzeros,
        ),
        directory,
    )
    manifest_seconds += perf_counter() - manifest_start

    if len(completed) == plan.num_shards:
        final = load_sharded_operator_manifest(directory)
        return ShardedConstructionResult(
            directory=directory,
            manifest=final,
            plan=plan,
            reused_shards=len(completed),
            rebuilt_shards=0,
            rejected_shards=rejected,
            support_discovery_seconds=support_seconds,
            lowering_seconds=0.0,
            compilation_seconds=0.0,
            dispatch_seconds=0.0,
            synchronization_seconds=0.0,
            transfer_seconds=0.0,
            zero_filtering_seconds=0.0,
            shard_persistence_seconds=0.0,
            manifest_seconds=manifest_seconds,
            manifest_write_count=1,
            cumulative_manifest_bytes=initial_manifest_bytes,
            recovery_scan_seconds=recovery_scan_seconds,
            finalization_seconds=0.0,
            total_seconds=perf_counter() - total_start,
        )

    kernel = _make_forward_support_kernel(
        graph=inputs.graph,
        chunk_size=config.od_chunk_size,
        edge_block_size=config.support_edge_block_size,
    )
    dummy_origins = jnp.zeros(config.od_chunk_size, jnp.int32)
    dummy_valid = jnp.zeros(config.od_chunk_size, bool)
    dummy_links = jnp.zeros(config.support_edge_block_size, jnp.int32)
    dummy_link_mask = jnp.zeros(config.support_edge_block_size, bool)
    lowering_start = perf_counter()
    lowered = kernel.lower(
        dummy_origins,
        dummy_valid,
        routing.group_link_probability[0],
        routing.effective_group_link_mask[0],
        dummy_links,
        dummy_link_mask,
    )
    lowering_seconds = perf_counter() - lowering_start
    compilation_start = perf_counter()
    compiled = lowered.compile()
    compilation_seconds = perf_counter() - compilation_start

    origins = np.asarray(inputs.od_origin_node, dtype=np.int32)
    effective = np.asarray(routing.effective_group_link_mask)
    mapping_links = np.asarray(spec.link_index, dtype=np.int32)
    mapping_measurements = np.asarray(spec.measurement_index, dtype=np.int64)
    dispatch = synchronization = transfer = filtering = persistence = 0.0
    rebuilt = 0
    manifest_write_count = 1
    cumulative_manifest_bytes = initial_manifest_bytes
    finalization_seconds = 0.0
    tasks_by_key = {task.identity.key: task for task in support.construction_tasks}
    dispatch_durations: list[float] = []
    synchronization_durations: list[float] = []
    origins_per_dispatch: list[int] = []
    supported_edges_per_dispatch: list[int] = []
    output_values_per_dispatch: list[int] = []
    group_timing: dict[str, float] = {}
    construction_batches = 0
    padded_buffer_allocations = 0
    routing_array_dispatch_uses = 0
    for identity in plan.expected_shards:
        if identity.key in completed:
            continue
        shard_tasks = [
            tasks_by_key[key] for key in support.storage_task_keys[identity.key]
        ]
        measurements = np.asarray(
            sorted({row for task in shard_tasks for row in task.measurements}),
            dtype=np.int64,
        )
        shard_row_lookup = {
            int(row): position for position, row in enumerate(measurements)
        }
        row_parts: list[np.ndarray] = []
        column_parts: list[np.ndarray] = []
        data_parts: list[np.ndarray] = []
        offset = np.zeros(measurements.size, dtype=np.dtype(inputs.base_link_cost.dtype))
        construction_start = perf_counter()
        tasks_by_group: dict[int, list[ConstructionTask]] = {}
        for task in shard_tasks:
            tasks_by_group.setdefault(task.group, []).append(task)
        for group, grouped_tasks in sorted(tasks_by_group.items()):
            batch_start = perf_counter()
            construction_batches += 1
            batch_measurements = np.asarray(
                sorted({row for task in grouped_tasks for row in task.measurements}),
                dtype=np.int64,
            )
            batch_lookup = np.full(plan.num_measurements, -1, dtype=np.int32)
            batch_lookup[batch_measurements] = np.arange(batch_measurements.size, dtype=np.int32)
            selected_mapping = effective[group, mapping_links] & (
                batch_lookup[mapping_measurements] >= 0
            )
            selected_links = mapping_links[selected_mapping]
            selected_local_rows = batch_lookup[mapping_measurements[selected_mapping]]
            od_support: dict[int, set[int]] = {}
            for task in grouped_tasks:
                for active_index in task.od_indices:
                    od_support.setdefault(active_index, set()).update(task.measurements)
            od_indices = np.asarray(sorted(od_support), dtype=np.int64)
            batch_to_shard = np.asarray(
                [shard_row_lookup[int(row)] for row in batch_measurements], dtype=np.int64
            )
            for first in range(0, od_indices.size, config.od_chunk_size):
                chunk = od_indices[first : first + config.od_chunk_size]
                padded = np.zeros(config.od_chunk_size, dtype=np.int32)
                valid = np.zeros(config.od_chunk_size, dtype=bool)
                padded[: chunk.size] = origins[chunk]
                valid[: chunk.size] = True
                values = np.zeros(
                    (chunk.size, batch_measurements.size),
                    dtype=np.dtype(inputs.base_link_cost.dtype),
                )
                padded_buffer_allocations += 3
                for edge_first in range(0, selected_links.size, config.support_edge_block_size):
                    edge_links = selected_links[edge_first : edge_first + config.support_edge_block_size]
                    edge_rows = selected_local_rows[edge_first : edge_first + config.support_edge_block_size]
                    padded_links = np.zeros(config.support_edge_block_size, dtype=np.int32)
                    edge_mask = np.zeros(config.support_edge_block_size, dtype=bool)
                    padded_links[: edge_links.size] = edge_links
                    edge_mask[: edge_links.size] = True
                    padded_buffer_allocations += 2
                    start = perf_counter()
                    device = compiled(
                        jnp.asarray(padded), jnp.asarray(valid),
                        routing.group_link_probability[group],
                        routing.effective_group_link_mask[group],
                        jnp.asarray(padded_links), jnp.asarray(edge_mask),
                    )
                    elapsed = perf_counter() - start
                    dispatch += elapsed
                    dispatch_durations.append(elapsed)
                    origins_per_dispatch.append(int(chunk.size))
                    supported_edges_per_dispatch.append(int(edge_links.size))
                    output_values_per_dispatch.append(int(chunk.size * edge_links.size))
                    routing_array_dispatch_uses += 2
                    start = perf_counter()
                    device.block_until_ready()
                    elapsed = perf_counter() - start
                    synchronization += elapsed
                    synchronization_durations.append(elapsed)
                    start = perf_counter()
                    edge_values = np.asarray(device)[: chunk.size, : edge_links.size]
                    transfer += perf_counter() - start
                    for edge_position, local_row in enumerate(edge_rows):
                        values[:, local_row] += edge_values[:, edge_position]
                start = perf_counter()
                for local_od, active_index in enumerate(chunk):
                    allowed = np.asarray(
                        sorted(
                            batch_lookup[np.asarray(tuple(od_support[int(active_index)]), dtype=np.int64)]
                        ),
                        dtype=np.int64,
                    )
                    supported_values = values[local_od, allowed]
                    column = support.free_column[active_index]
                    if column >= 0:
                        nonzero = np.flatnonzero(
                            np.abs(supported_values) > config.zero_tolerance
                        )
                        row_parts.append(batch_to_shard[allowed[nonzero]])
                        column_parts.append(np.full(nonzero.size, column, dtype=np.int64))
                        data_parts.append(supported_values[nonzero])
                    else:
                        offset[batch_to_shard[allowed]] += (
                            supported_values * support.fixed_by_active[active_index]
                        )
                filtering += perf_counter() - start
            group_key = str(group)
            group_timing[group_key] = group_timing.get(group_key, 0.0) + (
                perf_counter() - batch_start
            )
        construction_seconds = perf_counter() - construction_start
        rows = np.concatenate(row_parts) if row_parts else np.empty(0, np.int64)
        columns = (
            np.concatenate(column_parts) if column_parts else np.empty(0, np.int64)
        )
        data = (
            np.concatenate(data_parts)
            if data_parts
            else np.empty(0, dtype=np.dtype(inputs.base_link_cost.dtype))
        )
        matrix = sparse.coo_array(
            (data, (rows, columns)),
            shape=(measurements.size, plan.num_free_od),
        )
        start = perf_counter()
        metadata = save_sparse_shard(
            directory=directory,
            identity=identity,
            row_indices=measurements,
            matrix=matrix,
            fixed_offset=offset,
            num_measurements=plan.num_measurements,
            num_free_od=plan.num_free_od,
            dtype=inputs.base_link_cost.dtype,
            zero_tolerance=config.zero_tolerance,
            provenance_hash=template.provenance_hash,
            metrics=SparseShardMetrics(
                candidate_entries=sum(task.estimated_nonzeros for task in shard_tasks),
                realized_entries=int(data.size),
                discarded_entries=sum(task.estimated_nonzeros for task in shard_tasks) - int(data.size),
                construction_seconds=construction_seconds,
            ),
            compressed=config.compressed_shards,
        )
        persistence += perf_counter() - start
        completed.add(identity.key)
        aggregate_nonzeros += metadata.nonzero_entries
        rebuilt += 1
        should_checkpoint = (
            rebuilt % config.manifest_checkpoint_shards == 0
            or len(completed) == plan.num_shards
        )
        if should_checkpoint:
            manifest_start = perf_counter()
            current = _manifest(
                plan=plan,
                config=config,
                provenance=provenance,
                completed=completed,
                aggregate_nonzeros=aggregate_nonzeros,
            )
            _, checkpoint_bytes = save_sharded_operator_manifest(current, directory)
            checkpoint_seconds = perf_counter() - manifest_start
            manifest_seconds += checkpoint_seconds
            if len(completed) == plan.num_shards:
                finalization_seconds = checkpoint_seconds
            manifest_write_count += 1
            cumulative_manifest_bytes += checkpoint_bytes
        if progress is not None:
            progress(
                {
                    "completed_shards": len(completed),
                    "expected_shards": plan.num_shards,
                    "shard": identity.key,
                    "nonzero_entries": metadata.nonzero_entries,
                }
            )
    final = load_sharded_operator_manifest(directory)
    return ShardedConstructionResult(
        directory=directory,
        manifest=final,
        plan=plan,
        reused_shards=len(completed) - rebuilt,
        rebuilt_shards=rebuilt,
        rejected_shards=rejected,
        support_discovery_seconds=support_seconds,
        lowering_seconds=lowering_seconds,
        compilation_seconds=compilation_seconds,
        dispatch_seconds=dispatch,
        synchronization_seconds=synchronization,
        transfer_seconds=transfer,
        zero_filtering_seconds=filtering,
        shard_persistence_seconds=persistence,
        manifest_seconds=manifest_seconds,
        manifest_write_count=manifest_write_count,
        cumulative_manifest_bytes=cumulative_manifest_bytes,
        recovery_scan_seconds=recovery_scan_seconds,
        finalization_seconds=finalization_seconds,
        total_seconds=perf_counter() - total_start,
        construction_batches=construction_batches,
        dispatch_count=len(dispatch_durations),
        synchronization_count=len(synchronization_durations),
        support_edge_blocks=len(supported_edges_per_dispatch),
        origins_per_dispatch=tuple(origins_per_dispatch),
        supported_edges_per_dispatch=tuple(supported_edges_per_dispatch),
        output_values_per_dispatch=tuple(output_values_per_dispatch),
        dispatch_time_quantiles=_timing_quantiles(dispatch_durations),
        synchronization_time_quantiles=_timing_quantiles(
            synchronization_durations
        ),
        group_timing_seconds=group_timing,
        padded_buffer_allocations=padded_buffer_allocations,
        routing_array_dispatch_uses=routing_array_dispatch_uses,
    )
