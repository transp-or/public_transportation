"""Bounded, resumable construction of fixed-routing sparse operator shards."""

from __future__ import annotations

from collections import deque
import math
import hashlib
import json
import os
import shutil
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
import tempfile
import uuid
from time import perf_counter
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse  # type: ignore[import-untyped]

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
    GroupSupportTimingCallback,
    OriginSupportConfig,
    analyze_fixed_routing_origin_support,
)
from .measurement_support_preflight import (
    PositiveBoardingPreflightContext,
    audit_positive_boarding_support,
    enforce_positive_boarding_support,
)
from .sharded_sparse_operator import (
    SHARDED_OPERATOR_SCHEMA_VERSION,
    ShardedOperatorManifest,
    SparseShardIdentity,
    SparseShardMetadata,
    SparseShardMetrics,
    load_sharded_operator_manifest,
    load_sparse_shard,
    manifest_path,
    save_sharded_operator_manifest,
    save_sparse_shard,
    shard_path,
)
from .construction_control import (
    ConstructionDeadline,
    ConstructionPhase,
    ConstructionProgressReporter,
    deadline_stop,
    estimate_completed_unit_eta,
)
from .sharded_fixed_routing import (
    FixedRoutingShard,
    FixedRoutingShardDescriptor,
    ShardedFixedRoutingInputs,
    fixed_routing_descriptor_for_group,
    load_fixed_routing_shard,
    validate_sharded_fixed_routing_compatibility,
)

FixedRoutingSource = FixedRoutingInputs | ShardedFixedRoutingInputs


def _emit_legacy_progress(
    callback: Callable[[dict[str, object]], None] | None,
    payload: dict[str, object],
) -> None:
    """Deliver the legacy telemetry hook without changing construction."""

    if callback is None:
        return
    try:
        callback(payload)
    except OSError:
        # A legacy callback is often a file/socket sink.  I/O failures are
        # observability-only and must not invalidate an otherwise successful
        # construction.  Other exceptions intentionally propagate: callers
        # have historically used the callback as an explicit interruption
        # hook, and swallowing those would break resumable construction.
        return

# Numerical shard contents are tied to the kernel algorithm.  This version is
# deliberately separate from the support-checkpoint provenance: changing the
# numerical implementation must invalidate stored shards, but must not force
# the expensive topology-support discovery to run again.
MEASUREMENT_KERNEL_ALGORITHM_VERSION = "reachability-edge-gather-v1"
SUPPORT_DEFINITION_ALGORITHM_VERSION = "positive-probability-group-batched-reachability-v2"
SERIALIZATION_ALGORITHM_VERSION = "npz-atomic-shard-v1"


class _RoutingBatchReader:
    """Retain at most one sharded routing batch on host and device."""

    def __init__(self, routing: FixedRoutingSource) -> None:
        self.routing = routing
        self.descriptor: FixedRoutingShardDescriptor | None = None
        self.shard: FixedRoutingShard | None = None
        self.device_probability: jax.Array | None = None
        self.device_effective: jax.Array | None = None
        self.dense_probability = (
            None
            if isinstance(routing, ShardedFixedRoutingInputs)
            else np.asarray(routing.group_link_probability)
        )
        self.dense_effective = (
            None
            if isinstance(routing, ShardedFixedRoutingInputs)
            else np.asarray(routing.effective_group_link_mask, dtype=bool)
        )

    def _ensure(self, group: int) -> int:
        if not isinstance(self.routing, ShardedFixedRoutingInputs):
            return group
        descriptor = fixed_routing_descriptor_for_group(self.routing, group)
        if self.descriptor != descriptor:
            self.shard = load_fixed_routing_shard(
                routing=self.routing, descriptor=descriptor
            )
            self.device_probability = jnp.asarray(self.shard.group_link_probability)
            self.device_effective = jnp.asarray(self.shard.effective_group_link_mask)
            self.descriptor = descriptor
        return group - descriptor.group_start

    def host(self, group: int) -> tuple[np.ndarray, np.ndarray]:
        local = self._ensure(group)
        if isinstance(self.routing, ShardedFixedRoutingInputs):
            assert self.shard is not None
            return (
                self.shard.group_link_probability[local],
                self.shard.effective_group_link_mask[local],
            )
        assert self.dense_probability is not None and self.dense_effective is not None
        return self.dense_probability[local], self.dense_effective[local]

    def device(self, group: int):
        local = self._ensure(group)
        if isinstance(self.routing, ShardedFixedRoutingInputs):
            assert self.device_probability is not None
            assert self.device_effective is not None
            return self.device_probability[local], self.device_effective[local]
        return (
            self.routing.group_link_probability[local],
            self.routing.effective_group_link_mask[local],
        )


@dataclass(frozen=True, slots=True)
class ShardedConstructionConfig:
    od_chunk_size: int = 128
    measurement_block_size: int = 512
    worker_memory_budget_bytes: int = 512 * 1024 * 1024
    zero_tolerance: float = 0.0
    compressed_shards: bool = False
    workers: int = 1
    support_workers: int | None = None
    origin_support_chunk_size: int = 64
    max_materialized_support_entries: int = 125_000_000
    support_discovery_mode: Literal["compact", "reference"] = "compact"
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
    maximum_resident_shards: int = 2
    progress_interval_seconds: float = 1.0
    deadline_safety_margin_seconds: float = 5.0

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
        if self.max_materialized_support_entries <= 0:
            raise ValueError("max_materialized_support_entries must be positive.")
        if self.support_discovery_mode not in ("compact", "reference"):
            raise ValueError(
                "support_discovery_mode must be 'compact' or 'reference'."
            )
        if not math.isfinite(self.zero_tolerance) or self.zero_tolerance < 0:
            raise ValueError("zero_tolerance must be finite and non-negative.")
        if (
            isinstance(self.workers, bool)
            or not isinstance(self.workers, (int, np.integer))
            or self.workers <= 0
        ):
            raise ValueError("workers must be positive.")
        if self.support_workers is not None and (
            isinstance(self.support_workers, bool)
            or not isinstance(self.support_workers, (int, np.integer))
            or self.support_workers <= 0
        ):
            raise ValueError(
                "support_workers must be a positive integer when provided."
            )
        if self.support_workers is None:
            object.__setattr__(self, "support_workers", int(self.workers))
        if self.maximum_resident_shards <= 0:
            raise ValueError("maximum_resident_shards must be positive.")
        if (
            not math.isfinite(self.progress_interval_seconds)
            or self.progress_interval_seconds < 0.0
            or not math.isfinite(self.deadline_safety_margin_seconds)
            or self.deadline_safety_margin_seconds < 0.0
        ):
            raise ValueError(
                "progress interval and deadline margin must be nonnegative."
            )
        if not (
            0
            < self.target_nonzeros_per_storage_shard
            <= self.maximum_nonzeros_per_storage_shard
        ):
            raise ValueError("storage-shard nonzero targets are invalid.")
        if (
            min(
                self.maximum_patterns_per_storage_shard,
                self.maximum_storage_shards,
                self.maximum_manifest_bytes,
                self.maximum_filesystem_operations,
                self.maximum_sparse_calls_per_product,
                self.manifest_checkpoint_shards,
                self.maximum_construction_dispatches,
        )
            <= 0
        ):
            raise ValueError("operational shard limits must be positive.")

    def execution_configuration(self) -> dict[str, object]:
        """Return explicit worker-pool settings for reports and manifests.

        ``workers`` remains the legacy shard-construction worker setting.
        ``support_workers`` is independently configurable and defaults to the
        legacy value for compatibility; it is never inferred from routing
        preparation workers.
        """

        return {
            "support_workers_requested": int(self.support_workers),
            "shard_construction_workers_requested": int(self.workers),
            "threads_per_worker": 1,
            "support_execution_strategy": "thread_pool",
            "shard_construction_execution_strategy": "thread_pool",
            "maximum_resident_shards": int(self.maximum_resident_shards),
        }


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
class _GroupConstructionMetadata:
    """Packed group metadata shared by planning, workers, and diagnostics."""

    group: int
    batch_measurements: np.ndarray
    selected_links: np.ndarray
    selected_local_rows: np.ndarray
    od_indices: np.ndarray
    od_support: dict[int, tuple[int, ...]]


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
    estimated_maximum_staged_shard_bytes: int = 0
    estimated_worker_memory_bytes: int = 0
    estimated_filesystem_operations: int = 0
    configured_maximum_storage_shards: int = 0
    configured_maximum_manifest_bytes: int = 0
    configured_maximum_filesystem_operations: int = 0
    configured_maximum_sparse_calls_per_product: int = 0
    configured_maximum_construction_dispatches: int = 0
    configured_target_nonzeros_per_storage_shard: int = 0
    configured_maximum_nonzeros_per_storage_shard: int = 0
    configured_maximum_patterns_per_storage_shard: int = 0
    configured_manifest_checkpoint_shards: int = 0
    estimated_reachability_evaluations: int = 0
    estimated_edge_gather_evaluations: int = 0

    def preflight_diagnostics(self) -> dict[str, object]:
        """Return complete JSON-ready actual-versus-limit diagnostics."""
        limits = {
            "storage_shards": {
                "actual": self.num_shards,
                "permitted": self.configured_maximum_storage_shards,
            },
            "manifest_bytes": {
                "actual": self.estimated_manifest_bytes,
                "permitted": self.configured_maximum_manifest_bytes,
            },
            "filesystem_operations": {
                "actual": self.estimated_filesystem_operations,
                "permitted": self.configured_maximum_filesystem_operations,
            },
            "sparse_calls_per_product": {
                "actual": self.estimated_sparse_calls_per_product,
                "permitted": self.configured_maximum_sparse_calls_per_product,
            },
            "construction_dispatches": {
                "actual": self.estimated_construction_dispatches,
                "permitted": self.configured_maximum_construction_dispatches,
            },
            "worker_memory_bytes": {
                "actual": self.estimated_worker_memory_bytes,
                "permitted": self.worker_memory_budget_bytes,
            },
        }
        return {
            "safe": self.safe,
            "reason": self.reason,
            "limits": {
                name: {
                    **values,
                    "exceeded": values["actual"] > values["permitted"],
                }
                for name, values in limits.items()
            },
            "storage_shard_sizing": {
                "target_nonzeros": (self.configured_target_nonzeros_per_storage_shard),
                "maximum_nonzeros": (
                    self.configured_maximum_nonzeros_per_storage_shard
                ),
                "maximum_patterns": (
                    self.configured_maximum_patterns_per_storage_shard
                ),
            },
            "estimated_manifest_writes": self.estimated_manifest_writes,
            "estimated_filesystem_file_count": (self.estimated_filesystem_file_count),
            "estimated_maximum_staged_shard_bytes": (
                self.estimated_maximum_staged_shard_bytes
            ),
            "estimated_reachability_evaluations": (
                self.estimated_reachability_evaluations
            ),
            "estimated_edge_gather_evaluations": (
                self.estimated_edge_gather_evaluations
            ),
        }


class ShardedConstructionPreflightError(MemoryError):
    """Structured, backward-compatible rejection of an unsafe shard plan."""

    def __init__(self, plan: ShardedConstructionPlan):
        super().__init__(plan.reason)
        self.plan = plan
        self.details = plan.preflight_diagnostics()


def _evaluate_operational_preflight(
    plan: ShardedConstructionPlan,
    config: ShardedConstructionConfig,
) -> ShardedConstructionPlan:
    """Apply current operational limits without changing scientific identity."""
    maximum_staged_shard_bytes = max(
        (item.estimated_uncompressed_bytes for item in plan.storage_shards),
        default=plan.estimated_maximum_staged_shard_bytes,
    )
    worker_memory_bytes = (
        plan.estimated_kernel_bytes
        + plan.estimated_batch_temporary_bytes
        + maximum_staged_shard_bytes
    )
    manifest_writes = 2 + math.ceil(plan.num_shards / config.manifest_checkpoint_shards)
    manifest_bytes = max(
        plan.estimated_manifest_bytes,
        8192 + plan.num_shards * 512,
    )
    filesystem_operations = plan.num_shards * 5 + manifest_writes * 3
    configured = replace(
        plan,
        estimated_maximum_staged_shard_bytes=maximum_staged_shard_bytes,
        estimated_worker_memory_bytes=worker_memory_bytes,
        estimated_manifest_writes=manifest_writes,
        estimated_manifest_bytes=manifest_bytes,
        estimated_cumulative_manifest_bytes=(manifest_bytes * manifest_writes),
        estimated_filesystem_file_count=plan.num_shards + 1,
        estimated_filesystem_operations=filesystem_operations,
        estimated_sparse_calls_per_product=plan.num_shards,
        configured_maximum_storage_shards=config.maximum_storage_shards,
        configured_maximum_manifest_bytes=config.maximum_manifest_bytes,
        configured_maximum_filesystem_operations=(config.maximum_filesystem_operations),
        configured_maximum_sparse_calls_per_product=(
            config.maximum_sparse_calls_per_product
        ),
        configured_maximum_construction_dispatches=(
            config.maximum_construction_dispatches
        ),
        configured_target_nonzeros_per_storage_shard=(
            config.target_nonzeros_per_storage_shard
        ),
        configured_maximum_nonzeros_per_storage_shard=(
            config.maximum_nonzeros_per_storage_shard
        ),
        configured_maximum_patterns_per_storage_shard=(
            config.maximum_patterns_per_storage_shard
        ),
        configured_manifest_checkpoint_shards=config.manifest_checkpoint_shards,
    )
    diagnostics = configured.preflight_diagnostics()["limits"]
    assert isinstance(diagnostics, dict)
    comparisons = []
    failure_count = 0
    for name, values in diagnostics.items():
        assert isinstance(values, dict)
        exceeded = bool(values["exceeded"])
        failure_count += int(exceeded)
        comparisons.append(
            f"{name}: actual={values['actual']}, "
            f"permitted={values['permitted']}, exceeded={str(exceeded).lower()}"
        )
    if failure_count:
        reason = "sharded construction preflight rejected: " + "; ".join(comparisons)
        return replace(configured, safe=False, reason=reason)
    return replace(
        configured,
        safe=True,
        reason="kernel memory and aggregate storage plan are operationally bounded",
    )


def _make_forward_reachability_kernel(*, chunk_size: int, num_nodes: int):
    """Compile the complete OD-chunk reachability pass once.

    The graph arrays are dynamic arguments rather than captured constants.  In
    particular, this keeps the executable reusable across routing groups and
    avoids embedding a second copy of the graph in every compiled worker.
    """

    def kernel(
        origin_nodes,
        valid_origins,
        link_probability,
        enabled_link_mask,
        topo,
        out_links,
        out_mask,
        head,
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

        return jax.lax.scan(step, reach, topo)[0]

    return jax.jit(kernel)


def _make_edge_gather_kernel(*, chunk_size: int, edge_block_size: int):
    """Compile a fixed-shape gather from retained reachability state."""

    def kernel(
        reach,
        link_probability,
        enabled_link_mask,
        selected_links,
        selected_link_mask,
        tail,
    ):
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
    requested_workers: int = 1
    admitted_workers: int = 0
    maximum_active_workers: int = 0
    maximum_buffered_shards: int = 0
    worker_failures: int = 0
    compilation_count: int = 0
    reachability_evaluations: int = 0
    edge_gather_evaluations: int = 0
    reachability_dispatch_seconds: float = 0.0
    edge_gather_dispatch_seconds: float = 0.0
    jax_execution_seconds: float = 0.0
    reachability_time_quantiles: dict[str, float] | None = None
    edge_gather_time_quantiles: dict[str, float] | None = None
    synchronization_barrier_count: int = 0


@dataclass(frozen=True, slots=True)
class _MeasurementShardWorkerResult:
    """One independently constructed shard staged for ordered publication."""

    identity: SparseShardIdentity
    metadata: SparseShardMetadata
    staged_path: Path
    elapsed_seconds: float
    dispatch_seconds: float
    synchronization_seconds: float
    transfer_seconds: float
    zero_filtering_seconds: float
    shard_persistence_seconds: float
    construction_batches: int
    dispatch_durations: tuple[float, ...]
    synchronization_durations: tuple[float, ...]
    origins_per_dispatch: tuple[int, ...]
    supported_edges_per_dispatch: tuple[int, ...]
    output_values_per_dispatch: tuple[int, ...]
    group_timing_seconds: dict[str, float]
    padded_buffer_allocations: int
    routing_array_dispatch_uses: int
    reachability_evaluations: int
    edge_gather_evaluations: int
    reachability_dispatch_seconds: float
    edge_gather_dispatch_seconds: float
    reachability_durations: tuple[float, ...]
    edge_gather_durations: tuple[float, ...]


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
    construction_metadata: dict[str, tuple[_GroupConstructionMetadata, ...]] = field(
        default_factory=dict
    )


SUPPORT_CHECKPOINT_SCHEMA_VERSION = 1


def _ragged(values: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([value.size for value in values], dtype=np.int64)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
    flat = np.concatenate(values) if values and offsets[-1] else np.empty(0, np.int64)
    return np.asarray(flat, dtype=np.int64), offsets


def _unragged(flat: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(flat[offsets[index] : offsets[index + 1]], dtype=np.int64)
        for index in range(offsets.size - 1)
    )


def _support_checkpoint_path(directory: Path) -> Path:
    return Path(directory) / "support.npz"


def _support_checkpoint_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _derived_support(
    *,
    free_column: np.ndarray,
    fixed_by_active: np.ndarray,
    selected: np.ndarray,
    group_od_indices: tuple[np.ndarray, ...],
    group_measurements: tuple[np.ndarray, ...],
    global_slots: np.ndarray,
    global_slot_mask: np.ndarray,
    group_level_candidate_entries: int,
    origin_support_seconds: float,
    patterns: tuple[SupportPattern, ...],
    config: ShardedConstructionConfig,
) -> _Support:
    tasks = []
    shard_od_indices = {}
    shard_measurements = {}
    identities = []
    for pattern in patterns:
        pattern_od = np.asarray(pattern.od_indices, dtype=np.int64)
        pattern_measurements = np.asarray(pattern.measurements, dtype=np.int64)
        for block, first in enumerate(
            range(0, pattern_measurements.size, config.measurement_block_size)
        ):
            count = min(
                config.measurement_block_size, pattern_measurements.size - first
            )
            identity = SparseShardIdentity(
                group=pattern.group,
                measurement_block=block,
                first_measurement_position=first,
                measurement_count=count,
                support_pattern=pattern.pattern,
            )
            measurements = pattern_measurements[first : first + count]
            identities.append(identity)
            shard_od_indices[identity.key] = pattern_od
            shard_measurements[identity.key] = measurements
            tasks.append(
                ConstructionTask(
                    identity=identity,
                    group=pattern.group,
                    od_indices=pattern.od_indices,
                    measurements=tuple(int(value) for value in measurements),
                    estimated_nonzeros=int(pattern_od.size * measurements.size),
                )
            )
    return _Support(
        free_column=free_column,
        fixed_by_active=fixed_by_active,
        selected=selected,
        group_od_indices=group_od_indices,
        group_measurements=group_measurements,
        global_slots=global_slots,
        global_slot_mask=global_slot_mask,
        shard_od_indices=shard_od_indices,
        shard_measurements=shard_measurements,
        expected_shards=tuple(identities),
        group_level_candidate_entries=group_level_candidate_entries,
        origin_support_seconds=origin_support_seconds,
        support_patterns=len(patterns),
        patterns=patterns,
        construction_tasks=tuple(tasks),
        storage_task_keys={},
    )


def _save_support_checkpoint(
    directory: Path,
    support: _Support,
    *,
    provenance_hash: str,
) -> Path:
    group_od, group_od_offsets = _ragged(support.group_od_indices)
    group_rows, group_row_offsets = _ragged(support.group_measurements)
    pattern_od, pattern_od_offsets = _ragged(
        tuple(np.asarray(item.od_indices, dtype=np.int64) for item in support.patterns)
    )
    pattern_rows, pattern_row_offsets = _ragged(
        tuple(
            np.asarray(item.measurements, dtype=np.int64) for item in support.patterns
        )
    )
    arrays = {
        "free_column": np.asarray(support.free_column),
        "fixed_by_active": np.asarray(support.fixed_by_active),
        "selected": np.asarray(support.selected),
        "group_od": group_od,
        "group_od_offsets": group_od_offsets,
        "group_rows": group_rows,
        "group_row_offsets": group_row_offsets,
        "global_slots": np.asarray(support.global_slots),
        "global_slot_mask": np.asarray(support.global_slot_mask),
        "pattern_group": np.asarray(
            [item.group for item in support.patterns], np.int64
        ),
        "pattern_id": np.asarray([item.pattern for item in support.patterns], np.int64),
        "pattern_od": pattern_od,
        "pattern_od_offsets": pattern_od_offsets,
        "pattern_rows": pattern_rows,
        "pattern_row_offsets": pattern_row_offsets,
    }
    metadata = {
        "schema_version": SUPPORT_CHECKPOINT_SCHEMA_VERSION,
        "provenance_hash": provenance_hash,
        "content_hash": _support_checkpoint_hash(arrays),
        "group_level_candidate_entries": support.group_level_candidate_entries,
        "origin_support_seconds": support.origin_support_seconds,
    }
    destination = _support_checkpoint_path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as stream:
            np.savez(
                stream,  # type: ignore[arg-type]
                metadata=np.asarray(json.dumps(metadata)),
                **arrays,  # type: ignore[arg-type]
            )
            stream.flush()
            os.fsync(stream.fileno())
        _load_support_checkpoint(
            Path(temporary), provenance_hash=provenance_hash, config=None
        )
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _load_support_checkpoint(
    path: Path,
    *,
    provenance_hash: str,
    config: ShardedConstructionConfig | None,
) -> _Support:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "metadata"
        }
    if metadata.get("schema_version") != SUPPORT_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("support checkpoint schema is incompatible.")
    if metadata.get("provenance_hash") != provenance_hash:
        raise ValueError("support checkpoint provenance is incompatible.")
    if metadata.get("content_hash") != _support_checkpoint_hash(arrays):
        raise ValueError("support checkpoint content hash is invalid.")
    pattern_od = _unragged(arrays["pattern_od"], arrays["pattern_od_offsets"])
    pattern_rows = _unragged(arrays["pattern_rows"], arrays["pattern_row_offsets"])
    patterns = tuple(
        SupportPattern(
            group=int(group),
            pattern=int(pattern),
            od_indices=tuple(int(value) for value in od),
            measurements=tuple(int(value) for value in rows),
        )
        for group, pattern, od, rows in zip(
            arrays["pattern_group"],
            arrays["pattern_id"],
            pattern_od,
            pattern_rows,
            strict=True,
        )
    )
    effective_config = ShardedConstructionConfig() if config is None else config
    return _derived_support(
        free_column=arrays["free_column"],
        fixed_by_active=arrays["fixed_by_active"],
        selected=arrays["selected"],
        group_od_indices=_unragged(arrays["group_od"], arrays["group_od_offsets"]),
        group_measurements=_unragged(arrays["group_rows"], arrays["group_row_offsets"]),
        global_slots=arrays["global_slots"],
        global_slot_mask=arrays["global_slot_mask"],
        group_level_candidate_entries=int(metadata["group_level_candidate_entries"]),
        origin_support_seconds=float(metadata["origin_support_seconds"]),
        patterns=patterns,
        config=effective_config,
    )


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


def _compact_group_patterns(
    *,
    active_indices: np.ndarray,
    free_rows: np.ndarray,
    free_columns: np.ndarray,
    fixed_rows: np.ndarray,
    fixed_columns: np.ndarray,
    free_indices: np.ndarray,
    positive_fixed_indices: np.ndarray,
) -> dict[tuple[int, ...], list[int]]:
    """Group one destination result by identical measurement support.

    The arrays are limited to one destination group.  Sorting by compact
    column lets us recover each active OD's rows without retaining a global
    sparse support matrix or a global per-OD Python dictionary.
    """
    support_by_active: dict[int, tuple[int, ...]] = {}

    def collect(
        rows: np.ndarray, columns: np.ndarray, active_lookup: np.ndarray
    ) -> None:
        if rows.size == 0:
            return
        order = np.argsort(columns, kind="stable")
        sorted_columns = columns[order]
        sorted_rows = rows[order]
        unique_columns, starts = np.unique(sorted_columns, return_index=True)
        stops = np.concatenate((starts[1:], np.asarray([rows.size])))
        for column, first, stop in zip(unique_columns, starts, stops, strict=True):
            support_by_active[int(active_lookup[column])] = tuple(
                int(value) for value in np.unique(sorted_rows[first:stop])
            )

    collect(free_rows, free_columns, free_indices)
    collect(fixed_rows, fixed_columns, positive_fixed_indices)
    patterns: dict[tuple[int, ...], list[int]] = {}
    for active_index in active_indices:
        pattern = support_by_active.get(int(active_index), ())
        if pattern:
            patterns.setdefault(pattern, []).append(int(active_index))
    return patterns


def _discover_support(
    *,
    inputs: AssignmentInputs,
    routing: FixedRoutingSource,
    spec,
    compact_layout: CompactODAssignmentLayout,
    config: ShardedConstructionConfig,
    checkpoint_directory: Path | None = None,
    checkpoint_provenance_hash: str | None = None,
    deadline: ConstructionDeadline | None = None,
    reporter: ConstructionProgressReporter | None = None,
    timing_callback: GroupSupportTimingCallback | None = None,
) -> _Support:
    num_active = int(inputs.od_origin_node.shape[0])
    free_column = np.full(num_active, -1, np.int64)
    free_indices = np.asarray(compact_layout.free_compact_indices, dtype=np.int64)
    free_column[free_indices] = np.arange(free_indices.size, dtype=np.int64)
    fixed_by_active = np.zeros(num_active, dtype=np.dtype(inputs.base_link_cost.dtype))
    fixed_by_active[
        np.asarray(compact_layout.fixed_compact_indices, dtype=np.int64)
    ] = np.asarray(
        compact_layout.fixed_compact_values,
        dtype=np.dtype(inputs.base_link_cost.dtype),
    )
    selected = (free_column >= 0) | (fixed_by_active != 0.0)
    global_slots, global_slot_mask = _mapping_slots(spec, inputs.graph.num_links)
    routing_reader = _RoutingBatchReader(routing)
    group_indices = np.asarray(inputs.group_od_index_padded)
    group_masks = np.asarray(inputs.group_od_mask)
    num_groups = int(inputs.group_dest_node.shape[0])
    od_groups_by_group: list[np.ndarray | None] = [None] * num_groups
    measurements_by_group: list[np.ndarray | None] = [None] * num_groups
    patterns_by_group: list[dict[tuple[int, ...], list[int]] | None] = [
        None
    ] * num_groups
    positive_fixed = np.asarray(
        compact_layout.fixed_compact_indices, dtype=np.int64
    )[
        np.asarray(compact_layout.fixed_compact_values) > 0.0
    ]
    def consume_compact_group(
        group: int,
        active_indices: np.ndarray,
        group_measurements: np.ndarray,
        summary,
        group_free_rows: np.ndarray,
        group_free_columns: np.ndarray,
        group_fixed_rows: np.ndarray,
        group_fixed_columns: np.ndarray,
    ) -> None:
        od_groups_by_group[group] = np.asarray(active_indices, dtype=np.int64)
        measurements_by_group[group] = np.asarray(
            group_measurements, dtype=np.int64
        )
        patterns_by_group[group] = _compact_group_patterns(
            active_indices=active_indices,
            free_rows=group_free_rows,
            free_columns=group_free_columns,
            fixed_rows=group_fixed_rows,
            fixed_columns=group_fixed_columns,
            free_indices=free_indices,
            positive_fixed_indices=positive_fixed,
        )

    analyzed = analyze_fixed_routing_origin_support(
        inputs=inputs,
        routing=routing,
        spec=spec,
        compact_layout=compact_layout,
        config=OriginSupportConfig(
            origin_chunk_size=config.origin_support_chunk_size,
            worker_memory_budget_bytes=config.worker_memory_budget_bytes,
            materialize=config.support_discovery_mode == "reference",
            max_materialized_entries=config.max_materialized_support_entries,
            workers=int(config.support_workers),
        ),
        checkpoint_directory=checkpoint_directory,
        checkpoint_provenance_hash=checkpoint_provenance_hash,
        deadline=deadline,
        reporter=reporter,
        timing_callback=timing_callback,
        group_callback=(
            None
            if config.support_discovery_mode == "reference"
            else consume_compact_group
        ),
    )
    if (
        config.support_discovery_mode == "compact"
        and analyzed.metrics.origin_specific_entries
        > config.max_materialized_support_entries
    ):
        raise MemoryError(
            f"origin-specific support has {analyzed.metrics.origin_specific_entries} "
            "entries, exceeding max_materialized_entries="
            f"{config.max_materialized_support_entries}"
        )
    if config.support_discovery_mode == "reference":
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
        for column, active_index in enumerate(positive_fixed):
            support_by_active[int(active_index)] = tuple(
                int(value)
                for value in fixed_support.indices[
                    fixed_support.indptr[column] : fixed_support.indptr[column + 1]
                ]
            )
        for group in range(num_groups):
            relevant = group_indices[group][group_masks[group]]
            relevant = relevant[selected[relevant]].astype(np.int64, copy=False)
            od_groups_by_group[group] = relevant
            _, group_effective = routing_reader.host(group)
            mapped = global_slot_mask & group_effective[:, None]
            measurements_by_group[group] = np.unique(global_slots[mapped]).astype(
                np.int64, copy=False
            )
            patterns: dict[tuple[int, ...], list[int]] = {}
            for active_index in relevant:
                pattern = support_by_active[int(active_index)]
                if pattern:
                    patterns.setdefault(pattern, []).append(int(active_index))
            patterns_by_group[group] = patterns
    od_groups = tuple(
        value if value is not None else np.empty(0, dtype=np.int64)
        for value in od_groups_by_group
    )
    measurements = tuple(
        value if value is not None else np.empty(0, dtype=np.int64)
        for value in measurements_by_group
    )
    discovered_patterns: list[SupportPattern] = []
    for group, patterns in enumerate(patterns_by_group):
        assert patterns is not None
        for pattern_id, pattern in enumerate(sorted(patterns)):
            discovered_patterns.append(
                SupportPattern(
                    group=group,
                    pattern=pattern_id,
                    od_indices=tuple(int(value) for value in patterns[pattern]),
                    measurements=tuple(int(value) for value in pattern),
                )
            )
    return _derived_support(
        free_column=free_column,
        fixed_by_active=fixed_by_active,
        selected=selected,
        group_od_indices=od_groups,
        group_measurements=measurements,
        global_slots=global_slots,
        global_slot_mask=global_slot_mask,
        group_level_candidate_entries=(analyzed.metrics.group_level_candidate_entries),
        origin_support_seconds=analyzed.metrics.support_discovery_seconds,
        patterns=tuple(discovered_patterns),
        config=config,
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
            current and current_estimate >= config.target_nonzeros_per_storage_shard
        )
        too_many_patterns = len(current) >= config.maximum_patterns_per_storage_shard
        if exceeds_hard or reached_target or too_many_patterns:
            flush()
        current.append(task)
        current_estimate += task.estimated_nonzeros
    flush()
    return tuple(packed)


def _build_construction_metadata(
    *,
    support: _Support,
    storage_shards: tuple[StorageShardPlan, ...],
    routing: FixedRoutingSource,
    spec,
) -> dict[str, tuple[_GroupConstructionMetadata, ...]]:
    """Derive group unions and mapped edges once for each storage shard."""
    task_by_key = {task.identity.key: task for task in support.construction_tasks}
    mapping_links = np.asarray(spec.link_index, dtype=np.int64)
    mapping_measurements = np.asarray(spec.measurement_index, dtype=np.int64)
    routing_reader = _RoutingBatchReader(routing)
    metadata: dict[str, tuple[_GroupConstructionMetadata, ...]] = {}
    for storage_shard in storage_shards:
        tasks_by_group: dict[int, list[ConstructionTask]] = {}
        for key in storage_shard.task_keys:
            task = task_by_key[key]
            tasks_by_group.setdefault(task.group, []).append(task)
        groups: list[_GroupConstructionMetadata] = []
        for group, grouped_tasks in sorted(tasks_by_group.items()):
            batch_measurements = np.asarray(
                sorted({row for task in grouped_tasks for row in task.measurements}),
                dtype=np.int64,
            )
            batch_lookup = np.full(spec.num_measurements, -1, dtype=np.int32)
            batch_lookup[batch_measurements] = np.arange(
                batch_measurements.size, dtype=np.int32
            )
            _, group_effective = routing_reader.host(group)
            mapped_local_rows = batch_lookup[mapping_measurements]
            selected_mapping = group_effective[mapping_links] & (
                mapped_local_rows >= 0
            )
            selected_links = np.asarray(mapping_links[selected_mapping], dtype=np.int64)
            selected_local_rows = np.asarray(
                mapped_local_rows[selected_mapping], dtype=np.int64
            )
            od_support: dict[int, set[int]] = {}
            for task in grouped_tasks:
                for active_index in task.od_indices:
                    od_support.setdefault(active_index, set()).update(task.measurements)
            od_indices = np.asarray(sorted(od_support), dtype=np.int64)
            groups.append(
                _GroupConstructionMetadata(
                    group=group,
                    batch_measurements=batch_measurements,
                    selected_links=selected_links,
                    selected_local_rows=selected_local_rows,
                    od_indices=od_indices,
                    od_support={
                        int(index): tuple(sorted(rows))
                        for index, rows in sorted(od_support.items())
                    },
                )
            )
        metadata[storage_shard.identity.key] = tuple(groups)
    return metadata


def plan_sharded_fixed_routing_operator(
    *,
    inputs: AssignmentInputs,
    routing: FixedRoutingSource,
    spec,
    compact_layout: CompactODAssignmentLayout,
    config: ShardedConstructionConfig | None = None,
    discovered_support: _Support | None = None,
    support_timing_callback: GroupSupportTimingCallback | None = None,
) -> tuple[ShardedConstructionPlan, _Support]:
    """Discover structural support and reject unsafe kernels before lowering."""
    config = ShardedConstructionConfig() if config is None else config
    if isinstance(routing, ShardedFixedRoutingInputs):
        validate_sharded_fixed_routing_compatibility(inputs=inputs, routing=routing)
    else:
        validate_fixed_routing_compatibility(inputs=inputs, routing=routing)
    if compact_layout.num_active != int(inputs.od_origin_node.shape[0]):
        raise ValueError("compact layout and assignment active dimensions differ.")
    support = (
        _discover_support(
            inputs=inputs,
            routing=routing,
            spec=spec,
            compact_layout=compact_layout,
            config=config,
            timing_callback=support_timing_callback,
        )
        if discovered_support is None
        else discovered_support
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
    routing_reader = _RoutingBatchReader(routing)

    def support_edge_count(identity: SparseShardIdentity) -> int:
        _, group_effective = routing_reader.host(identity.group)
        return int(
            np.count_nonzero(
                group_effective[mapping_links]
                & np.isin(
                    mapping_measurements,
                    support.shard_measurements[identity.key],
                )
            )
        )

    maximum_support_edges = max(
        (support_edge_count(identity) for identity in task_identities),
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
    construction_metadata = _build_construction_metadata(
        support=support,
        storage_shards=storage_shards,
        routing=routing,
        spec=spec,
    )
    support = replace(support, construction_metadata=construction_metadata)
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
    # Cold construction writes the template and recovery-scan state, then one
    # checkpoint per publication interval (including the final checkpoint).
    manifest_writes = 2 + math.ceil(
        len(storage_shards) / config.manifest_checkpoint_shards
    )
    # The fixed allowance covers provenance plus the complete plan diagnostics;
    # the per-shard allowance covers identity, completion key, and payload data.
    # It is intentionally conservative and independent of private key lengths.
    manifest_bytes = 8192 + len(storage_shards) * 512
    # Each worker shard uses a temporary write, validation open, staging rename,
    # and parent publication rename; manifests use temporary write and rename.
    filesystem_operations = len(storage_shards) * 5 + manifest_writes * 3
    estimated_batches = 0
    estimated_reachability_evaluations = 0
    estimated_edge_gather_evaluations = 0
    maximum_batch_rows = 0
    for storage_shard in storage_shards:
        for group_metadata in construction_metadata[storage_shard.identity.key]:
            estimated_batches += 1
            maximum_batch_rows = max(
                maximum_batch_rows, group_metadata.batch_measurements.size
            )
            chunk_count = max(
                1,
                math.ceil(
                    group_metadata.od_indices.size / config.od_chunk_size
                ),
            )
            selected_edges = group_metadata.selected_links.size
            edge_block_count = max(
                1, math.ceil(selected_edges / config.support_edge_block_size)
            )
            estimated_reachability_evaluations += chunk_count
            estimated_edge_gather_evaluations += chunk_count * edge_block_count
    estimated_dispatches = (
        estimated_reachability_evaluations + estimated_edge_gather_evaluations
    )
    estimated_batch_temporary_bytes = int(
        config.od_chunk_size * maximum_batch_rows * itemsize
    )
    maximum_staged_shard_bytes = max(
        (item.estimated_uncompressed_bytes for item in storage_shards), default=0
    )
    estimated_worker_memory_bytes = (
        estimated_kernel_bytes
        + estimated_batch_temporary_bytes
        + maximum_staged_shard_bytes
    )
    plan = ShardedConstructionPlan(
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
        safe=True,
        reason="preflight diagnostics pending",
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
        estimated_payload_p90=float(np.percentile(payloads, 90))
        if payloads.size
        else 0.0,
        estimated_payload_p99=float(np.percentile(payloads, 99))
        if payloads.size
        else 0.0,
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
        estimated_maximum_staged_shard_bytes=maximum_staged_shard_bytes,
        estimated_worker_memory_bytes=estimated_worker_memory_bytes,
        estimated_filesystem_operations=filesystem_operations,
        estimated_reachability_evaluations=estimated_reachability_evaluations,
        estimated_edge_gather_evaluations=estimated_edge_gather_evaluations,
    )
    return (
        _evaluate_operational_preflight(plan, config),
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
        execution_provenance={
            "workers": config.workers,
            **config.execution_configuration(),
            "od_chunk_size": config.od_chunk_size,
            "measurement_block_size": config.measurement_block_size,
            "origin_support_chunk_size": config.origin_support_chunk_size,
            "support_edge_block_size": config.support_edge_block_size,
            "target_nonzeros_per_storage_shard": (
                config.target_nonzeros_per_storage_shard
            ),
            "maximum_nonzeros_per_storage_shard": (
                config.maximum_nonzeros_per_storage_shard
            ),
            "maximum_patterns_per_storage_shard": (
                config.maximum_patterns_per_storage_shard
            ),
            "compressed_shards": config.compressed_shards,
        },
    )


def _plan_from_manifest(manifest: ShardedOperatorManifest) -> ShardedConstructionPlan:
    if manifest.plan_summary is None:
        raise ValueError("completed shard manifest has no construction plan summary.")
    summary: dict[str, Any] = dict(manifest.plan_summary)
    summary["estimated_nonzeros_per_storage_shard"] = tuple(
        summary.get("estimated_nonzeros_per_storage_shard", ())
    )
    return ShardedConstructionPlan(
        expected_shards=manifest.expected_shards,
        storage_shards=(),
        **summary,
    )


def _restore_plan_from_manifest(
    manifest: ShardedOperatorManifest,
    *,
    support: _Support,
    config: ShardedConstructionConfig,
    itemsize: int,
) -> tuple[ShardedConstructionPlan, _Support]:
    plan = _plan_from_manifest(manifest)
    storage = pack_storage_shards(
        support.construction_tasks, config=config, itemsize=itemsize
    )
    identities = tuple(item.identity for item in storage)
    if identities != manifest.expected_shards:
        raise ValueError("persisted plan and reconstructed support are incompatible.")
    support = replace(
        support,
        storage_task_keys={item.identity.key: item.task_keys for item in storage},
    )
    restored = replace(plan, storage_shards=storage)
    return _evaluate_operational_preflight(restored, config), support


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
    scientific_identity: dict[str, object] | None = None,
    include_measurement_kernel_version: bool = True,
    include_execution_provenance: bool = True,
) -> dict[str, object]:
    graph_digest = hashlib.sha256()
    for value in (
        inputs.graph.tail,
        inputs.graph.head,
        inputs.graph.topo_order,
        inputs.graph.out_links,
        inputs.graph.out_mask,
    ):
        array = np.ascontiguousarray(value)
        graph_digest.update(str(array.dtype).encode("utf-8"))
        graph_digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        graph_digest.update(array.tobytes())
    provenance: dict[str, object] = {
        "assignment_fingerprint": str(assignment_fingerprint),
        "assignment_inputs_fingerprint": (
            assignment_inputs_fingerprint(inputs)
            if assignment_inputs_fingerprint_value is None
            else str(assignment_inputs_fingerprint_value)
        ),
        "mapping_fingerprint": measurement_mapping_fingerprint(spec),
        "graph_fingerprint": graph_digest.hexdigest(),
        "od_layout_fingerprint": str(od_layout_fingerprint),
        "compact_layout_fingerprint": compact_layout.fingerprint,
        "frozen_cell_fingerprint": compact_layout.fingerprint,
        "theta": float(theta),
        "dtype": str(np.dtype(inputs.base_link_cost.dtype)),
        "zero_tolerance": config.zero_tolerance,
        "measurement_block_size": config.measurement_block_size,
        "od_chunk_size": config.od_chunk_size,
        "origin_support_chunk_size": config.origin_support_chunk_size,
        "support_strategy": SUPPORT_DEFINITION_ALGORITHM_VERSION,
        "support_discovery_mode": config.support_discovery_mode,
        "construction_schema_version": SHARDED_OPERATOR_SCHEMA_VERSION,
        "serialization_algorithm": SERIALIZATION_ALGORITHM_VERSION,
        "scientific_identity": (
            {} if scientific_identity is None else dict(scientific_identity)
        ),
    }
    if include_execution_provenance:
        provenance.update(
            {
                "support_edge_block_size": config.support_edge_block_size,
                "target_nonzeros_per_storage_shard": config.target_nonzeros_per_storage_shard,
                "maximum_nonzeros_per_storage_shard": config.maximum_nonzeros_per_storage_shard,
                "maximum_patterns_per_storage_shard": config.maximum_patterns_per_storage_shard,
                "compressed_shards": config.compressed_shards,
            }
        )
    if include_measurement_kernel_version:
        provenance["measurement_kernel_algorithm"] = (
            MEASUREMENT_KERNEL_ALGORITHM_VERSION
        )
    if not include_execution_provenance:
        for key in (
            "zero_tolerance",
            "serialization_algorithm",
            "measurement_block_size",
            "od_chunk_size",
            "origin_support_chunk_size",
            "support_edge_block_size",
            "target_nonzeros_per_storage_shard",
            "maximum_nonzeros_per_storage_shard",
            "maximum_patterns_per_storage_shard",
            "compressed_shards",
        ):
            provenance.pop(key, None)
    return provenance


def _same_support_provenance(left: dict[str, object], right: dict[str, object]) -> bool:
    """Return whether support is reusable despite packing/kernel changes."""
    left_support = dict(left)
    right_support = dict(right)
    left_support.pop("measurement_kernel_algorithm", None)
    right_support.pop("measurement_kernel_algorithm", None)
    for key in (
        "zero_tolerance",
        "serialization_algorithm",
        "measurement_block_size",
        "od_chunk_size",
        "origin_support_chunk_size",
        "support_edge_block_size",
        "target_nonzeros_per_storage_shard",
        "maximum_nonzeros_per_storage_shard",
        "maximum_patterns_per_storage_shard",
        "compressed_shards",
    ):
        left_support.pop(key, None)
        right_support.pop(key, None)
    return left_support == right_support


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
    scientific_identity: dict[str, object] | None = None,
) -> ShardedConstructionResult | None:
    """Return a compatible complete cache without routing or support planning."""
    config = ShardedConstructionConfig() if config is None else config
    directory = Path(directory)
    started = perf_counter()
    if not manifest_path(directory).exists():
        return None
    try:
        manifest = load_sharded_operator_manifest(directory)
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        # A malformed or legacy manifest is an explicit cache miss.  Callers
        # can rebuild safely; the decoder must not leak an unsafe exception.
        return None
    provenance = _construction_provenance(
        inputs=inputs,
        spec=spec,
        compact_layout=compact_layout,
        assignment_fingerprint=assignment_fingerprint,
        od_layout_fingerprint=od_layout_fingerprint,
        theta=theta,
        config=config,
        assignment_inputs_fingerprint_value=assignment_inputs_fingerprint_value,
        scientific_identity=scientific_identity,
    )
    if not manifest.complete or manifest.provenance != provenance:
        return None
    return ShardedConstructionResult(
        directory=directory,
        manifest=manifest,
        plan=_evaluate_operational_preflight(_plan_from_manifest(manifest), config),
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


def _construct_measurement_shard(
    *,
    staging_directory: Path,
    identity: SparseShardIdentity,
    inputs: AssignmentInputs,
    routing: FixedRoutingSource,
    plan: ShardedConstructionPlan,
    support: _Support,
    tasks_by_key: dict[str, ConstructionTask],
    origins: np.ndarray,
    mapping_links: np.ndarray,
    mapping_measurements: np.ndarray,
    reach_compiled,
    gather_compiled,
    graph_arrays,
    tail,
    config: ShardedConstructionConfig,
    provenance_hash: str,
) -> _MeasurementShardWorkerResult:
    """Construct and stage one measurement shard without shared mutable state."""
    started = perf_counter()
    routing_reader = _RoutingBatchReader(routing)
    shard_tasks = [tasks_by_key[key] for key in support.storage_task_keys[identity.key]]
    measurements = np.asarray(
        sorted({row for task in shard_tasks for row in task.measurements}),
        dtype=np.int64,
    )
    shard_row_lookup = {int(row): position for position, row in enumerate(measurements)}
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    offset = np.zeros(measurements.size, dtype=np.dtype(inputs.base_link_cost.dtype))
    dispatch = synchronization = transfer = filtering = 0.0
    dispatch_durations: list[float] = []
    synchronization_durations: list[float] = []
    reachability_durations: list[float] = []
    edge_gather_durations: list[float] = []
    origins_per_dispatch: list[int] = []
    supported_edges_per_dispatch: list[int] = []
    output_values_per_dispatch: list[int] = []
    group_timing: dict[str, float] = {}
    construction_batches = 0
    reachability_evaluations = 0
    edge_gather_evaluations = 0
    reachability_dispatch = 0.0
    edge_gather_dispatch = 0.0
    padded_buffer_allocations = 0
    routing_array_dispatch_uses = 0
    construction_start = perf_counter()
    metadata_by_group = {
        item.group: item
        for item in support.construction_metadata.get(identity.key, ())
    }
    if not metadata_by_group:
        raise ValueError("construction metadata is missing for measurement shard")
    for group, group_metadata in sorted(metadata_by_group.items()):
        batch_start = perf_counter()
        construction_batches += 1
        batch_measurements = group_metadata.batch_measurements
        group_probability_device, group_effective_device = routing_reader.device(group)
        selected_links = group_metadata.selected_links
        selected_local_rows = group_metadata.selected_local_rows
        od_indices = group_metadata.od_indices
        batch_to_shard = np.asarray(
            [shard_row_lookup[int(row)] for row in batch_measurements],
            dtype=np.int64,
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
            # The graph traversal is intentionally outside the mapped-edge
            # loop.  All gathers consume this retained device state, so the
            # complete reachability dynamic program is evaluated once per OD
            # chunk regardless of the number of edge blocks.
            phase_start = perf_counter()
            reach_device = reach_compiled(
                jnp.asarray(padded),
                jnp.asarray(valid),
                group_probability_device,
                group_effective_device,
                *graph_arrays,
            )
            elapsed = perf_counter() - phase_start
            dispatch += elapsed
            reachability_dispatch += elapsed
            dispatch_durations.append(elapsed)
            reachability_durations.append(elapsed)
            reachability_evaluations += 1
            origins_per_dispatch.append(int(chunk.size))
            supported_edges_per_dispatch.append(0)
            output_values_per_dispatch.append(int(chunk.size * inputs.graph.num_nodes))
            routing_array_dispatch_uses += 2
            gather_devices: list[jax.Array] = []
            gather_blocks: list[tuple[np.ndarray, np.ndarray]] = []
            for edge_first in range(
                0, selected_links.size, config.support_edge_block_size
            ):
                edge_links = selected_links[
                    edge_first : edge_first + config.support_edge_block_size
                ]
                edge_rows = selected_local_rows[
                    edge_first : edge_first + config.support_edge_block_size
                ]
                padded_links = np.zeros(config.support_edge_block_size, dtype=np.int32)
                edge_mask = np.zeros(config.support_edge_block_size, dtype=bool)
                padded_links[: edge_links.size] = edge_links
                edge_mask[: edge_links.size] = True
                padded_buffer_allocations += 2
                phase_start = perf_counter()
                gather_devices.append(
                    gather_compiled(
                        reach_device,
                        group_probability_device,
                        group_effective_device,
                        jnp.asarray(padded_links),
                        jnp.asarray(edge_mask),
                        tail,
                    )
                )
                elapsed = perf_counter() - phase_start
                dispatch += elapsed
                edge_gather_dispatch += elapsed
                dispatch_durations.append(elapsed)
                edge_gather_durations.append(elapsed)
                edge_gather_evaluations += 1
                origins_per_dispatch.append(int(chunk.size))
                supported_edges_per_dispatch.append(int(edge_links.size))
                output_values_per_dispatch.append(int(chunk.size * edge_links.size))
                routing_array_dispatch_uses += 2
                gather_blocks.append((edge_links, edge_rows))
            # One explicit synchronization covers the reachability result and
            # every dependent gather in this OD chunk.  Host transfer happens
            # only after all device work is complete.
            phase_start = perf_counter()
            jax.block_until_ready(gather_devices)
            sync_elapsed = perf_counter() - phase_start
            synchronization += sync_elapsed
            synchronization_durations.append(sync_elapsed)
            for device, (edge_links, edge_rows) in zip(
                gather_devices, gather_blocks, strict=True
            ):
                phase_start = perf_counter()
                edge_values = np.asarray(device)[: chunk.size, : edge_links.size]
                transfer += perf_counter() - phase_start
                for edge_position, local_row in enumerate(edge_rows):
                    values[:, local_row] += edge_values[:, edge_position]
            phase_start = perf_counter()
            for local_od, active_index in enumerate(chunk):
                allowed = np.searchsorted(
                    batch_measurements,
                    np.asarray(
                        group_metadata.od_support[int(active_index)], dtype=np.int64
                    ),
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
            filtering += perf_counter() - phase_start
        group_key = str(group)
        group_timing[group_key] = group_timing.get(group_key, 0.0) + (
            perf_counter() - batch_start
        )
    construction_seconds = perf_counter() - construction_start
    rows = np.concatenate(row_parts) if row_parts else np.empty(0, np.int64)
    columns = np.concatenate(column_parts) if column_parts else np.empty(0, np.int64)
    data = (
        np.concatenate(data_parts)
        if data_parts
        else np.empty(0, dtype=np.dtype(inputs.base_link_cost.dtype))
    )
    matrix = sparse.coo_array(
        (data, (rows, columns)), shape=(measurements.size, plan.num_free_od)
    )
    persistence_start = perf_counter()
    metadata = save_sparse_shard(
        directory=staging_directory,
        identity=identity,
        row_indices=measurements,
        matrix=matrix,
        fixed_offset=offset,
        num_measurements=plan.num_measurements,
        num_free_od=plan.num_free_od,
        dtype=inputs.base_link_cost.dtype,
        zero_tolerance=config.zero_tolerance,
        provenance_hash=provenance_hash,
        metrics=SparseShardMetrics(
            candidate_entries=sum(task.estimated_nonzeros for task in shard_tasks),
            realized_entries=int(data.size),
            discarded_entries=(
                sum(task.estimated_nonzeros for task in shard_tasks) - int(data.size)
            ),
            construction_seconds=construction_seconds,
        ),
        compressed=config.compressed_shards,
    )
    persistence = perf_counter() - persistence_start
    return _MeasurementShardWorkerResult(
        identity=identity,
        metadata=metadata,
        staged_path=shard_path(staging_directory, identity),
        elapsed_seconds=perf_counter() - started,
        dispatch_seconds=dispatch,
        synchronization_seconds=synchronization,
        transfer_seconds=transfer,
        zero_filtering_seconds=filtering,
        shard_persistence_seconds=persistence,
        construction_batches=construction_batches,
        dispatch_durations=tuple(dispatch_durations),
        synchronization_durations=tuple(synchronization_durations),
        origins_per_dispatch=tuple(origins_per_dispatch),
        supported_edges_per_dispatch=tuple(supported_edges_per_dispatch),
        output_values_per_dispatch=tuple(output_values_per_dispatch),
        group_timing_seconds=group_timing,
        padded_buffer_allocations=padded_buffer_allocations,
        routing_array_dispatch_uses=routing_array_dispatch_uses,
        reachability_evaluations=reachability_evaluations,
        edge_gather_evaluations=edge_gather_evaluations,
        reachability_dispatch_seconds=reachability_dispatch,
        edge_gather_dispatch_seconds=edge_gather_dispatch,
        reachability_durations=tuple(reachability_durations),
        edge_gather_durations=tuple(edge_gather_durations),
    )


def prepare_sharded_fixed_routing_measurement_operator(
    *,
    directory: str | Path,
    inputs: AssignmentInputs,
    routing: FixedRoutingSource,
    spec,
    compact_layout: CompactODAssignmentLayout,
    assignment_fingerprint: str,
    od_layout_fingerprint: str,
    config: ShardedConstructionConfig | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
    deadline: ConstructionDeadline | None = None,
    reporter: ConstructionProgressReporter | None = None,
    scientific_identity: dict[str, object] | None = None,
    positive_boarding_preflight: PositiveBoardingPreflightContext | None = None,
    support_timing_callback: GroupSupportTimingCallback | None = None,
) -> ShardedConstructionResult:
    """Build missing/invalid shards and atomically advance a resumable manifest."""
    config = ShardedConstructionConfig() if config is None else config
    legacy_progress = progress if deadline is None and reporter is None else None
    control = ConstructionDeadline.unlimited() if deadline is None else deadline
    events = (
        ConstructionProgressReporter(
            control,
            None if legacy_progress else progress,
            minimum_interval_seconds=config.progress_interval_seconds,
            clock=control.clock,
        )
        if reporter is None
        else reporter
    )
    reporting_enabled = events.sink is not None
    directory = Path(directory)
    total_start = perf_counter()
    if positive_boarding_preflight is not None:
        events.emit(
            phase=ConstructionPhase.MEASUREMENT_SUPPORT_PREFLIGHT,
            status="started",
            force=True,
            completed_units=0,
            total_units=2,
            checkpoint_location=(
                None
                if positive_boarding_preflight.report_path is None
                else str(positive_boarding_preflight.report_path)
            ),
            details={"preflight_stage": "canonical_origin_support"},
        )
        origin_report = audit_positive_boarding_support(
            canonical_index=positive_boarding_preflight.canonical_index,
            observations=positive_boarding_preflight.observations,
            supported_measurement_rows=(
                positive_boarding_preflight.canonical_supported_measurement_rows
            ),
            mapping_info=positive_boarding_preflight.mapping_info,
            fixed_zero_reasons_by_full_index=(
                positive_boarding_preflight.fixed_zero_reasons_by_full_index
            ),
        )
        enforce_positive_boarding_support(
            origin_report, report_path=positive_boarding_preflight.report_path
        )
        events.emit(
            phase=ConstructionPhase.MEASUREMENT_SUPPORT_PREFLIGHT,
            status="running",
            force=True,
            completed_units=1,
            total_units=2,
            checkpoint_location=(
                None
                if positive_boarding_preflight.report_path is None
                else str(positive_boarding_preflight.report_path)
            ),
            details={
                "preflight_stage": "canonical_origin_support",
                "positive_boarding_rows": origin_report.positive_boarding_rows,
                "unsupported_positive_boarding_rows": 0,
            },
        )
    provenance = _construction_provenance(
        inputs=inputs,
        spec=spec,
        compact_layout=compact_layout,
        assignment_fingerprint=assignment_fingerprint,
        od_layout_fingerprint=od_layout_fingerprint,
        theta=float(np.asarray(routing.theta)),
        config=config,
        scientific_identity=scientific_identity,
    )
    existing_manifest = None
    if manifest_path(directory).exists():
        events.emit(
            phase=ConstructionPhase.SHARD_VALIDATION,
            status="started",
            force=True,
            checkpoint_location=str(directory),
        )
        try:
            existing = load_sharded_operator_manifest(directory)
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            quarantine = directory / f"manifest.invalid-{uuid.uuid4().hex}.json"
            os.replace(manifest_path(directory), quarantine)
            existing = None
        existing_manifest = existing
        if existing is not None and existing.provenance != provenance:
            if _same_support_provenance(existing.provenance, provenance):
                # Numerical shards from an older kernel are deliberately not
                # reused.  Keep the support checkpoint and let the normal
                # recovery scan reject every old shard by its new manifest
                # provenance hash.
                existing_manifest = None
            else:
                raise ValueError("existing shard manifest provenance is incompatible.")
        if existing is not None and (
            existing_manifest is not None
            and existing.complete
            and existing.plan_summary is not None
        ):
            valid = True
            aggregate_nonzeros = 0
            realized_supported_rows: set[int] = set()
            for position, identity in enumerate(existing.expected_shards):
                if not control.may_start():
                    raise deadline_stop(
                        control,
                        phase=ConstructionPhase.SHARD_VALIDATION,
                        reason="deadline reached while validating completed shards",
                        completed_units=position,
                        total_units=len(existing.expected_shards),
                        next_resumable_position=identity.key,
                        checkpoint_location=str(directory),
                        checkpoint_reusable=True,
                    )
                try:
                    loaded = load_sparse_shard(
                        shard_path(directory, identity),
                        expected_provenance_hash=existing.provenance_hash,
                    )
                except (KeyError, OSError, ValueError):
                    valid = False
                    break
                aggregate_nonzeros += loaded.metadata.nonzero_entries
                local_nonzero_rows = np.flatnonzero(
                    np.diff(loaded.matrix.indptr) > 0
                )
                realized_supported_rows.update(
                    int(row) for row in loaded.row_indices[local_nonzero_rows]
                )
                realized_supported_rows.update(
                    int(row)
                    for row in loaded.row_indices[loaded.fixed_offset_indices]
                )
            if valid and aggregate_nonzeros == existing.aggregate_nonzeros:
                if positive_boarding_preflight is not None:
                    realized_report = audit_positive_boarding_support(
                        canonical_index=positive_boarding_preflight.canonical_index,
                        observations=positive_boarding_preflight.observations,
                        supported_measurement_rows=np.asarray(
                            sorted(realized_supported_rows), dtype=np.int64
                        ),
                        stage="realized_operator_support",
                        mapping_info=positive_boarding_preflight.mapping_info,
                        fixed_zero_reasons_by_full_index=(
                            positive_boarding_preflight.fixed_zero_reasons_by_full_index
                        ),
                    )
                    enforce_positive_boarding_support(
                        realized_report,
                        report_path=positive_boarding_preflight.report_path,
                    )
                    events.emit(
                        phase=ConstructionPhase.MEASUREMENT_SUPPORT_PREFLIGHT,
                        status="completed",
                        force=True,
                        completed_units=2,
                        total_units=2,
                        checkpoint_location=(
                            None
                            if positive_boarding_preflight.report_path is None
                            else str(positive_boarding_preflight.report_path)
                        ),
                        details={
                            "preflight_stage": "realized_operator_support",
                            "positive_boarding_rows": (
                                realized_report.positive_boarding_rows
                            ),
                            "unsupported_positive_boarding_rows": 0,
                        },
                    )
                plan = _evaluate_operational_preflight(
                    _plan_from_manifest(existing), config
                )
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
    if not control.may_start():
        raise deadline_stop(
            control,
            phase=ConstructionPhase.SUPPORT_DISCOVERY,
            reason="deadline reached before support discovery",
            checkpoint_location=str(directory),
            checkpoint_reusable=manifest_path(directory).exists(),
        )
    events.emit(
        phase=ConstructionPhase.SUPPORT_DISCOVERY,
        status="started",
        force=True,
        checkpoint_location=str(directory),
    )
    support_start = perf_counter()
    support_provenance = _construction_provenance(
        inputs=inputs,
        spec=spec,
        compact_layout=compact_layout,
        assignment_fingerprint=assignment_fingerprint,
        od_layout_fingerprint=od_layout_fingerprint,
        theta=float(np.asarray(routing.theta)),
        config=config,
        scientific_identity=scientific_identity,
        include_measurement_kernel_version=False,
        include_execution_provenance=False,
    )
    support_provenance_hash = hashlib.sha256(
        json.dumps(support_provenance, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    for abandoned in directory.glob(".support.npz.*.tmp"):
        abandoned.unlink(missing_ok=True)
    support_checkpoint = _support_checkpoint_path(directory)
    support = None
    support_reused = False
    if support_checkpoint.exists():
        try:
            support = _load_support_checkpoint(
                support_checkpoint,
                provenance_hash=support_provenance_hash,
                config=config,
            )
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            quarantine = support_checkpoint.with_name(
                f"{support_checkpoint.name}.invalid-{uuid.uuid4().hex}"
            )
            os.replace(support_checkpoint, quarantine)
        else:
            support_reused = True
    if support is None:
        support = _discover_support(
            inputs=inputs,
            routing=routing,
            spec=spec,
            compact_layout=compact_layout,
            config=config,
            checkpoint_directory=directory,
            checkpoint_provenance_hash=support_provenance_hash,
            deadline=control,
            reporter=events,
            timing_callback=support_timing_callback,
        )
        _save_support_checkpoint(
            directory, support, provenance_hash=support_provenance_hash
        )
    support_seconds = perf_counter() - support_start
    if control.expired:
        raise deadline_stop(
            control,
            phase=ConstructionPhase.SUPPORT_DISCOVERY,
            reason="deadline expired after atomically persisting support discovery",
            completed_units=1,
            total_units=1,
            next_resumable_position="planning",
            checkpoint_location=str(directory),
            checkpoint_reusable=True,
        )
    events.emit(
        phase=ConstructionPhase.SUPPORT_DISCOVERY,
        status="completed",
        force=True,
        completed_units=1,
        total_units=1,
        recent_unit_seconds=support_seconds,
        checkpoint_location=str(directory),
        cache_hits=int(support_reused),
        cache_misses=int(not support_reused),
    )
    if positive_boarding_preflight is not None:
        supported_rows = np.asarray(
            sorted(
                {
                    row
                    for pattern in support.patterns
                    for row in pattern.measurements
                }
            ),
            dtype=np.int64,
        )
        routing_report = audit_positive_boarding_support(
            canonical_index=positive_boarding_preflight.canonical_index,
            observations=positive_boarding_preflight.observations,
            supported_measurement_rows=supported_rows,
            stage="routing_support",
            mapping_info=positive_boarding_preflight.mapping_info,
            fixed_zero_reasons_by_full_index=(
                positive_boarding_preflight.fixed_zero_reasons_by_full_index
            ),
        )
        enforce_positive_boarding_support(
            routing_report, report_path=positive_boarding_preflight.report_path
        )
        events.emit(
            phase=ConstructionPhase.MEASUREMENT_SUPPORT_PREFLIGHT,
            status="completed",
            force=True,
            completed_units=2,
            total_units=2,
            checkpoint_location=(
                None
                if positive_boarding_preflight.report_path is None
                else str(positive_boarding_preflight.report_path)
            ),
            details={
                "preflight_stage": "routing_support",
                "positive_boarding_rows": routing_report.positive_boarding_rows,
                "unsupported_positive_boarding_rows": 0,
            },
        )
    planning_start = perf_counter()
    planning_reused = False
    if existing_manifest is not None and existing_manifest.plan_summary is not None:
        plan, support = _restore_plan_from_manifest(
            existing_manifest,
            support=support,
            config=config,
            itemsize=np.dtype(inputs.base_link_cost.dtype).itemsize,
        )
        planning_reused = True
    else:
        plan, support = plan_sharded_fixed_routing_operator(
            inputs=inputs,
            routing=routing,
            spec=spec,
            compact_layout=compact_layout,
            config=config,
            discovered_support=support,
            support_timing_callback=support_timing_callback,
        )
    if not support.construction_metadata:
        support = replace(
            support,
            construction_metadata=_build_construction_metadata(
                support=support,
                storage_shards=plan.storage_shards,
                routing=routing,
                spec=spec,
            ),
        )
    planning_seconds = perf_counter() - planning_start
    if not plan.safe:
        raise ShardedConstructionPreflightError(plan)
    template = _manifest(
        plan=plan,
        config=config,
        provenance=provenance,
        completed=set(),
        aggregate_nonzeros=0,
    )
    if existing_manifest is None:
        save_sharded_operator_manifest(template, directory)
    if control.expired:
        raise deadline_stop(
            control,
            phase=ConstructionPhase.PLANNING,
            reason="deadline expired during deterministic shard planning",
            checkpoint_location=str(directory),
            checkpoint_reusable=_support_checkpoint_path(directory).exists(),
        )
    events.emit(
        phase=ConstructionPhase.PLANNING,
        status="completed",
        force=True,
        recent_unit_seconds=planning_seconds,
        total_units=plan.num_shards,
        checkpoint_location=str(directory),
        cache_hits=int(planning_reused),
        cache_misses=int(not planning_reused),
    )
    if manifest_path(directory).exists():
        existing = load_sharded_operator_manifest(directory)
        if existing.provenance_hash != template.provenance_hash:
            raise ValueError("existing shard manifest provenance is incompatible.")

    recovery_start = perf_counter()
    completed: set[str] = set()
    aggregate_nonzeros = 0
    rejected = 0
    for validation_position, identity in enumerate(plan.expected_shards):
        if not control.may_start():
            raise deadline_stop(
                control,
                phase=ConstructionPhase.SHARD_VALIDATION,
                reason="deadline reached during checkpoint recovery scan",
                completed_units=validation_position,
                total_units=plan.num_shards,
                next_resumable_position=identity.key,
                checkpoint_location=str(directory),
                checkpoint_reusable=bool(completed),
            )
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

    reach_kernel = _make_forward_reachability_kernel(
        chunk_size=config.od_chunk_size,
        num_nodes=inputs.graph.num_nodes,
    )
    gather_kernel = _make_edge_gather_kernel(
        chunk_size=config.od_chunk_size,
        edge_block_size=config.support_edge_block_size,
    )
    dummy_origins = jnp.zeros(config.od_chunk_size, jnp.int32)
    dummy_valid = jnp.zeros(config.od_chunk_size, bool)
    dummy_links = jnp.zeros(config.support_edge_block_size, jnp.int32)
    dummy_link_mask = jnp.zeros(config.support_edge_block_size, bool)
    graph_arrays = (
        jnp.asarray(inputs.graph.topo_order),
        jnp.asarray(inputs.graph.out_links),
        jnp.asarray(inputs.graph.out_mask),
        jnp.asarray(inputs.graph.head),
    )
    tail_device = jnp.asarray(inputs.graph.tail)
    routing_reader = _RoutingBatchReader(routing)
    prototype_probability, prototype_effective = routing_reader.device(0)
    lowering_start = perf_counter()
    if reporting_enabled:
        events.emit(
            phase=ConstructionPhase.SHARD_CONSTRUCTION,
            status="heartbeat",
            force=True,
            completed_units=len(completed),
            total_units=plan.num_shards,
            current_unit="operator_kernel_lowering",
            checkpoint_location=str(directory),
            details={"construction_stage": "kernel_lowering"},
        )
    with events.heartbeat_scope(
        current_unit="operator_kernel_lowering",
        completed_units=len(completed),
        total_units=plan.num_shards,
        phase=ConstructionPhase.SHARD_CONSTRUCTION,
        heartbeat_status="heartbeat",
        details={"construction_stage": "kernel_lowering"},
    ):
        reach_lowered = reach_kernel.lower(
            dummy_origins,
            dummy_valid,
            prototype_probability,
            prototype_effective,
            *graph_arrays,
        )
        gather_lowered = gather_kernel.lower(
            jnp.zeros(
                (config.od_chunk_size, inputs.graph.num_nodes),
                dtype=inputs.base_link_cost.dtype,
            ),
            prototype_probability,
            prototype_effective,
            dummy_links,
            dummy_link_mask,
            tail_device,
        )
    lowering_seconds = perf_counter() - lowering_start
    if reporting_enabled:
        events.emit(
            phase=ConstructionPhase.SHARD_CONSTRUCTION,
            status="heartbeat",
            force=True,
            completed_units=len(completed),
            total_units=plan.num_shards,
            current_unit="operator_kernel_lowering",
            recent_unit_seconds=lowering_seconds,
            checkpoint_location=str(directory),
            details={
                "construction_stage": "kernel_lowering",
                "construction_stage_status": "completed",
            },
        )
    compilation_start = perf_counter()
    if reporting_enabled:
        events.emit(
            phase=ConstructionPhase.SHARD_CONSTRUCTION,
            status="heartbeat",
            force=True,
            completed_units=len(completed),
            total_units=plan.num_shards,
            current_unit="operator_kernel_compilation",
            checkpoint_location=str(directory),
            details={"construction_stage": "kernel_compilation"},
        )
    with events.heartbeat_scope(
        current_unit="operator_kernel_compilation",
        completed_units=len(completed),
        total_units=plan.num_shards,
        phase=ConstructionPhase.SHARD_CONSTRUCTION,
        heartbeat_status="heartbeat",
        details={"construction_stage": "kernel_compilation"},
    ):
        reach_compiled = reach_lowered.compile()
        gather_compiled = gather_lowered.compile()
    compilation_seconds = perf_counter() - compilation_start
    if reporting_enabled:
        events.emit(
            phase=ConstructionPhase.SHARD_CONSTRUCTION,
            status="heartbeat",
            force=True,
            completed_units=len(completed),
            total_units=plan.num_shards,
            current_unit="operator_kernel_compilation",
            recent_unit_seconds=compilation_seconds,
            checkpoint_location=str(directory),
            details={
                "construction_stage": "kernel_compilation",
                "construction_stage_status": "completed",
            },
        )

    origins = np.asarray(inputs.od_origin_node, dtype=np.int32)
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
    reachability_durations: list[float] = []
    edge_gather_durations: list[float] = []
    reachability_evaluations = 0
    edge_gather_evaluations = 0
    reachability_dispatch = 0.0
    edge_gather_dispatch = 0.0
    recent_shard_seconds: deque[float] = deque(maxlen=32)
    missing = tuple(
        identity for identity in plan.expected_shards if identity.key not in completed
    )
    estimated_worker_bytes = max(
        1,
        plan.estimated_kernel_bytes
        + plan.estimated_batch_temporary_bytes
        + max(
            (item.estimated_uncompressed_bytes for item in plan.storage_shards),
            default=0,
        ),
    )
    admitted_workers = min(
        config.workers,
        config.maximum_resident_shards,
        len(missing),
    )
    active: dict[
        Future[_MeasurementShardWorkerResult], tuple[int, SparseShardIdentity]
    ] = {}
    buffered: dict[int, _MeasurementShardWorkerResult] = {}
    submitted = 0
    next_publish = 0
    worker_failures = 0
    maximum_active_workers = 0
    maximum_buffered_shards = 0
    stop_dispatch = False
    deadline_reason: str | None = None
    worker_error: tuple[SparseShardIdentity, BaseException] | None = None
    staging_directory = directory / f".measurement-shards-{uuid.uuid4().hex}"
    staging_directory.mkdir(parents=True, exist_ok=False)

    def predicted_next_seconds() -> float | None:
        if not recent_shard_seconds:
            return None
        return float(np.mean(tuple(recent_shard_seconds)[-3:]))

    def deadline_allows_dispatch(predicted: float | None) -> bool:
        remaining = control.remaining_seconds
        if remaining is None:
            return True
        duration = 0.0 if predicted is None else predicted
        margin = max(
            control.safety_margin_seconds,
            config.deadline_safety_margin_seconds,
        )
        return duration + margin <= remaining

    def progress_details() -> dict[str, object]:
        return {
            "requested_workers": config.workers,
            "admitted_workers": admitted_workers,
            "active_workers": len(active),
            "queued_shards": max(0, len(missing) - submitted),
            "buffered_shards": len(buffered),
            "worker_failures": worker_failures,
            "maximum_resident_shards": config.maximum_resident_shards,
            "worker_memory_budget_bytes": config.worker_memory_budget_bytes,
            "estimated_worker_bytes": estimated_worker_bytes,
        }

    def checkpoint_manifest() -> float:
        nonlocal manifest_seconds, manifest_write_count
        nonlocal cumulative_manifest_bytes, finalization_seconds
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
        manifest_write_count += 1
        cumulative_manifest_bytes += checkpoint_bytes
        if len(completed) == plan.num_shards:
            finalization_seconds = checkpoint_seconds
        return checkpoint_seconds

    def publish(result: _MeasurementShardWorkerResult) -> None:
        nonlocal aggregate_nonzeros, rebuilt, dispatch, synchronization
        nonlocal transfer, filtering, persistence, construction_batches
        nonlocal padded_buffer_allocations, routing_array_dispatch_uses
        nonlocal reachability_evaluations, edge_gather_evaluations
        nonlocal reachability_dispatch, edge_gather_dispatch
        destination = shard_path(directory, result.identity)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(result.staged_path, destination)
        completed.add(result.identity.key)
        aggregate_nonzeros += result.metadata.nonzero_entries
        rebuilt += 1
        dispatch += result.dispatch_seconds
        synchronization += result.synchronization_seconds
        transfer += result.transfer_seconds
        filtering += result.zero_filtering_seconds
        persistence += result.shard_persistence_seconds
        construction_batches += result.construction_batches
        dispatch_durations.extend(result.dispatch_durations)
        synchronization_durations.extend(result.synchronization_durations)
        reachability_durations.extend(result.reachability_durations)
        edge_gather_durations.extend(result.edge_gather_durations)
        origins_per_dispatch.extend(result.origins_per_dispatch)
        supported_edges_per_dispatch.extend(result.supported_edges_per_dispatch)
        output_values_per_dispatch.extend(result.output_values_per_dispatch)
        padded_buffer_allocations += result.padded_buffer_allocations
        routing_array_dispatch_uses += result.routing_array_dispatch_uses
        reachability_evaluations += result.reachability_evaluations
        edge_gather_evaluations += result.edge_gather_evaluations
        reachability_dispatch += result.reachability_dispatch_seconds
        edge_gather_dispatch += result.edge_gather_dispatch_seconds
        for group, duration in result.group_timing_seconds.items():
            group_timing[group] = group_timing.get(group, 0.0) + duration
        recent_shard_seconds.append(result.elapsed_seconds)
        if (
            rebuilt % config.manifest_checkpoint_shards == 0
            or len(completed) == plan.num_shards
        ):
            checkpoint_manifest()
        if legacy_progress is not None:
            _emit_legacy_progress(
                legacy_progress,
                {
                    "completed_shards": len(completed),
                    "expected_shards": plan.num_shards,
                    "shard": result.identity.key,
                    "nonzero_entries": result.metadata.nonzero_entries,
                    **progress_details(),
                }
            )
        eta = None
        if reporting_enabled:
            eta = estimate_completed_unit_eta(
                recent_shard_seconds,
                completed_units=len(completed),
                total_units=plan.num_shards,
                parallelism=admitted_workers,
                elapsed_seconds=max(0.0, perf_counter() - total_start),
            )
        events.emit(
            phase=ConstructionPhase.SHARD_CONSTRUCTION,
            status="running",
            force=(len(completed) == 1 or len(completed) == plan.num_shards),
            completed_units=len(completed),
            total_units=plan.num_shards,
            current_unit=result.identity.key,
            recent_unit_seconds=result.elapsed_seconds,
            predicted_remaining_seconds=(
                None if eta is None else eta.predicted_remaining_seconds
            ),
            eta_confidence=("unavailable" if eta is None else eta.eta_confidence),
            estimated_completion_at_utc=(
                None if eta is None else eta.estimated_completion_at_utc
            ),
            eta_reason=(None if eta is None else eta.eta_reason),
            eta_lower_seconds=(None if eta is None else eta.eta_lower_seconds),
            eta_upper_seconds=(None if eta is None else eta.eta_upper_seconds),
            throughput_units_per_second=(
                None if eta is None else eta.throughput_units_per_second
            ),
            checkpoint_location=str(directory),
            cache_hits=len(completed) - rebuilt,
            cache_misses=rebuilt,
            details={
                "nonzero_entries": result.metadata.nonzero_entries,
                **progress_details(),
            },
        )

    events.emit(
        phase=ConstructionPhase.SHARD_CONSTRUCTION,
        status="workers_admitted",
        force=True,
        completed_units=len(completed),
        total_units=plan.num_shards,
        checkpoint_location=str(directory),
        details=progress_details(),
    )
    try:
        with ThreadPoolExecutor(
            max_workers=admitted_workers,
            thread_name_prefix="measurement-shard",
        ) as executor:
            while active or (submitted < len(missing) and not stop_dispatch):
                while (
                    not stop_dispatch
                    and submitted < len(missing)
                    and len(active) < admitted_workers
                    and len(active) + len(buffered) < config.maximum_resident_shards
                ):
                    predicted = predicted_next_seconds()
                    if not deadline_allows_dispatch(predicted):
                        stop_dispatch = True
                        deadline_reason = (
                            "next shard cannot start within the configured "
                            "deadline safety margin"
                        )
                        break
                    identity = missing[submitted]
                    future = executor.submit(
                        _construct_measurement_shard,
                        staging_directory=staging_directory,
                        identity=identity,
                        inputs=inputs,
                        routing=routing,
                        plan=plan,
                        support=support,
                        tasks_by_key=tasks_by_key,
                        origins=origins,
                        mapping_links=mapping_links,
                        mapping_measurements=mapping_measurements,
                        reach_compiled=reach_compiled,
                        gather_compiled=gather_compiled,
                        graph_arrays=graph_arrays,
                        tail=tail_device,
                        config=config,
                        provenance_hash=template.provenance_hash,
                    )
                    active[future] = (submitted, identity)
                    submitted += 1
                    maximum_active_workers = max(maximum_active_workers, len(active))
                if not active:
                    break
                timeout = max(0.05, config.progress_interval_seconds)
                done, _ = wait(
                    active,
                    timeout=timeout,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    if control.expired:
                        stop_dispatch = True
                        deadline_reason = "deadline reached with workers in flight"
                    events.emit(
                        phase=ConstructionPhase.SHARD_CONSTRUCTION,
                        status="running",
                        completed_units=len(completed),
                        total_units=plan.num_shards,
                        checkpoint_location=str(directory),
                        details=progress_details(),
                    )
                    continue
                for future in sorted(done, key=lambda item: active[item][0]):
                    position, identity = active.pop(future)
                    try:
                        buffered[position] = future.result()
                    except BaseException as error:
                        worker_failures += 1
                        worker_error = (identity, error)
                        stop_dispatch = True
                        for pending in active:
                            pending.cancel()
                        if legacy_progress is not None:
                            _emit_legacy_progress(
                                legacy_progress,
                                {
                                    "completed_shards": len(completed),
                                    "expected_shards": plan.num_shards,
                                    "shard": identity.key,
                                    "status": "worker_failed",
                                    "error": str(error),
                                    **progress_details(),
                                }
                            )
                        events.emit(
                            phase=ConstructionPhase.SHARD_CONSTRUCTION,
                            status="worker_failed",
                            force=True,
                            completed_units=len(completed),
                            total_units=plan.num_shards,
                            current_unit=identity.key,
                            checkpoint_location=str(directory),
                            terminal_reason=str(error),
                            details=progress_details(),
                        )
                        break
                maximum_buffered_shards = max(maximum_buffered_shards, len(buffered))
                while next_publish in buffered:
                    publish(buffered.pop(next_publish))
                    next_publish += 1
                if worker_error is not None:
                    break
                if control.expired:
                    stop_dispatch = True
                    deadline_reason = "deadline reached with workers in flight"
        if worker_error is not None:
            checkpoint_manifest()
            identity, failure = worker_error
            raise RuntimeError(
                f"measurement-shard worker failed for {identity.key}"
            ) from failure
        if deadline_reason is not None or submitted < len(missing):
            checkpoint_manifest()
            next_position = next(
                (
                    identity.key
                    for identity in plan.expected_shards
                    if identity.key not in completed
                ),
                None,
            )
            raise deadline_stop(
                control,
                phase=ConstructionPhase.SHARD_CONSTRUCTION,
                reason=deadline_reason or "measurement-shard dispatch stopped",
                completed_units=len(completed),
                total_units=plan.num_shards,
                next_resumable_position=next_position,
                checkpoint_location=str(directory),
                checkpoint_reusable=True,
                predicted_next_seconds=predicted_next_seconds(),
            )
    finally:
        shutil.rmtree(staging_directory, ignore_errors=True)
    final = load_sharded_operator_manifest(directory)
    final_eta = None
    if reporting_enabled:
        final_eta = estimate_completed_unit_eta(
            recent_shard_seconds,
            completed_units=len(completed),
            total_units=plan.num_shards,
            parallelism=admitted_workers,
            elapsed_seconds=max(0.0, perf_counter() - total_start),
        )
    events.emit(
        phase=ConstructionPhase.SHARD_CONSTRUCTION,
        status="completed",
        force=True,
        completed_units=len(completed),
        total_units=plan.num_shards,
        current_unit=None,
        predicted_remaining_seconds=(
            0.0 if final_eta is not None else None
        ),
        eta_confidence=("high" if final_eta is not None else "unavailable"),
        estimated_completion_at_utc=(
            None if final_eta is None else final_eta.estimated_completion_at_utc
        ),
        eta_reason=("all units completed" if final_eta is not None else None),
        eta_lower_seconds=(0.0 if final_eta is not None else None),
        eta_upper_seconds=(0.0 if final_eta is not None else None),
        throughput_units_per_second=(
            None if final_eta is None else final_eta.throughput_units_per_second
        ),
        checkpoint_location=str(directory),
        cache_hits=len(completed) - rebuilt,
        cache_misses=rebuilt,
        details=progress_details(),
    )
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
        compilation_count=2,
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
        # Keep the historical public counter (one entry per mapped-edge
        # dispatch) while the timing data records the fused per-chunk barrier.
        synchronization_count=len(dispatch_durations),
        support_edge_blocks=edge_gather_evaluations,
        origins_per_dispatch=tuple(origins_per_dispatch),
        supported_edges_per_dispatch=tuple(supported_edges_per_dispatch),
        output_values_per_dispatch=tuple(output_values_per_dispatch),
        dispatch_time_quantiles=_timing_quantiles(dispatch_durations),
        synchronization_time_quantiles=_timing_quantiles(synchronization_durations),
        group_timing_seconds=group_timing,
        padded_buffer_allocations=padded_buffer_allocations,
        routing_array_dispatch_uses=routing_array_dispatch_uses,
        reachability_evaluations=reachability_evaluations,
        edge_gather_evaluations=edge_gather_evaluations,
        reachability_dispatch_seconds=reachability_dispatch,
        edge_gather_dispatch_seconds=edge_gather_dispatch,
        jax_execution_seconds=dispatch + synchronization,
        reachability_time_quantiles=_timing_quantiles(reachability_durations),
        edge_gather_time_quantiles=_timing_quantiles(edge_gather_durations),
        synchronization_barrier_count=len(synchronization_durations),
        requested_workers=config.workers,
        admitted_workers=admitted_workers,
        maximum_active_workers=maximum_active_workers,
        maximum_buffered_shards=maximum_buffered_shards,
        worker_failures=worker_failures,
    )
