"""Immutable contracts for bounded destination-group routing shards."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter, process_time
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal

import numpy as np
import jax
import jax.numpy as jnp

from public_transportation import __version__
from public_transportation.compilation_cache import configure_jax_compilation_cache

from .assignment_adapter import AssignmentInputs, _prepare_fixed_routing_core
from .block_coordinate._canonical import fingerprint
from .construction_control import estimate_completed_unit_eta

if TYPE_CHECKING:
    from .assignment_adapter import FixedRoutingInputs

SHARDED_FIXED_ROUTING_SCHEMA_VERSION = 2
SHARDED_FIXED_ROUTING_IMPLEMENTATION_VERSION = "dial-dp-fixed-routing-v1"
ProgressCallback = Callable[["FixedRoutingShardProgress"], None]
CachePolicy = Literal["reuse", "refresh"]
ExecutionStrategy = Literal["thread_pool", "batched"]


def _array_fingerprint(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _immutable_array(value: object, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class FixedRoutingShardDescriptor:
    """Canonical contiguous destination-group interval for one routing shard."""

    shard_index: int
    group_start: int
    group_stop: int

    def __post_init__(self) -> None:
        if self.shard_index < 0:
            raise ValueError("shard_index must be nonnegative.")
        if self.group_start < 0 or self.group_stop <= self.group_start:
            raise ValueError("a routing shard must contain a nonempty group interval.")

    @property
    def num_groups(self) -> int:
        return self.group_stop - self.group_start

    @property
    def destination_group_indices(self) -> tuple[int, ...]:
        return tuple(range(self.group_start, self.group_stop))


@dataclass(frozen=True, slots=True)
class FixedRoutingPreparationConfig:
    """Resource ceilings controlling deterministic routing-shard preparation."""

    maximum_groups_per_shard: int = 8
    maximum_retained_bytes_per_shard: int = 256 * 1024 * 1024
    maximum_temporary_bytes: int = 1024 * 1024 * 1024
    temporary_workspace_multiplier: float = 2.0
    maximum_process_rss_bytes: int | None = None
    maximum_cache_bytes: int | None = None
    maximum_elapsed_seconds: float | None = None
    checkpoint_directory: Path = Path(".fixed-routing-checkpoints")
    cache_directory: Path = Path(".fixed-routing-cache")
    progress_interval_groups: int = 8
    progress_interval_seconds: float = field(default=1.0, kw_only=True)
    durable_progress: bool = True
    construction_workers: int = 1
    threads_per_worker: int = 1
    resident_shard_limit: int = 1
    detailed_profiling: bool = False
    dispatch_safety_margin_seconds: float = 30.0
    initial_predicted_shard_seconds: float | None = None
    jax_compilation_cache_directory: Path | None = None
    execution_strategy: ExecutionStrategy = "thread_pool"
    shards_per_execution_batch: int = 1

    def __post_init__(self) -> None:
        positive = {
            "maximum_groups_per_shard": self.maximum_groups_per_shard,
            "maximum_retained_bytes_per_shard": self.maximum_retained_bytes_per_shard,
            "maximum_temporary_bytes": self.maximum_temporary_bytes,
            "progress_interval_groups": self.progress_interval_groups,
            "construction_workers": self.construction_workers,
            "threads_per_worker": self.threads_per_worker,
            "resident_shard_limit": self.resident_shard_limit,
            "shards_per_execution_batch": self.shards_per_execution_batch,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in ("maximum_process_rss_bytes", "maximum_cache_bytes"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided.")
        if self.maximum_elapsed_seconds is not None and self.maximum_elapsed_seconds <= 0:
            raise ValueError("maximum_elapsed_seconds must be positive when provided.")
        if (
            not np.isfinite(self.progress_interval_seconds)
            or self.progress_interval_seconds <= 0.0
        ):
            raise ValueError("progress_interval_seconds must be positive and finite.")
        if self.dispatch_safety_margin_seconds < 0.0:
            raise ValueError("dispatch_safety_margin_seconds must be nonnegative.")
        if (
            not np.isfinite(self.temporary_workspace_multiplier)
            or self.temporary_workspace_multiplier < 1.0
        ):
            raise ValueError(
                "temporary_workspace_multiplier must be finite and at least one."
            )
        if (
            self.initial_predicted_shard_seconds is not None
            and self.initial_predicted_shard_seconds <= 0.0
        ):
            raise ValueError(
                "initial_predicted_shard_seconds must be positive when provided."
            )
        if self.execution_strategy not in {"thread_pool", "batched"}:
            raise ValueError(
                "execution_strategy must be 'thread_pool' or 'batched'."
            )
        object.__setattr__(self, "checkpoint_directory", Path(self.checkpoint_directory))
        object.__setattr__(self, "cache_directory", Path(self.cache_directory))
        if self.jax_compilation_cache_directory is not None:
            object.__setattr__(
                self,
                "jax_compilation_cache_directory",
                Path(self.jax_compilation_cache_directory),
            )


@dataclass(frozen=True, slots=True)
class FixedRoutingShardPlan:
    descriptors: tuple[FixedRoutingShardDescriptor, ...]
    groups_per_full_shard: int
    retained_bytes_per_group: int
    estimated_temporary_bytes_per_group: int
    estimated_total_retained_bytes: int
    plan_fingerprint: str


@dataclass(frozen=True, slots=True)
class FixedRoutingWorkerRecommendation:
    workers: int
    cpu_limit: int
    memory_limit: int
    estimated_bytes_per_worker: int
    reason: str
    memory_admissible_workers: int | None = None
    cpu_admissible_workers: int | None = None
    throughput_effective_workers: int | None = None
    recommended_batch_size: int = 1


def recommend_fixed_routing_workers(
    *,
    plan: FixedRoutingShardPlan,
    available_ram_bytes: int,
    cpu_count: int,
    memory_fraction: float = 0.7,
    server: bool = False,
    measured_throughput_by_workers: dict[int, float] | None = None,
    measured_throughput_by_batch_size: dict[int, float] | None = None,
) -> FixedRoutingWorkerRecommendation:
    """Recommend bounded construction concurrency from CPU and memory limits."""
    if available_ram_bytes <= 0 or cpu_count <= 0:
        raise ValueError("available RAM and CPU count must be positive.")
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError("memory_fraction must be in (0, 1].")
    bytes_per_worker = max(
        1,
        plan.groups_per_full_shard
        * plan.estimated_temporary_bytes_per_group,
    )
    memory_limit = max(
        1, int(available_ram_bytes * memory_fraction) // bytes_per_worker
    )
    # Leave CPU capacity for XLA and I/O. Laptops default more conservatively.
    cpu_limit = max(1, cpu_count // (2 if server else 4))
    resource_limit = max(
        1, min(len(plan.descriptors) or 1, memory_limit, cpu_limit)
    )
    throughput_workers = None
    if measured_throughput_by_workers:
        admissible = {
            int(workers): float(throughput)
            for workers, throughput in measured_throughput_by_workers.items()
            if 0 < int(workers) <= resource_limit
            and np.isfinite(throughput)
            and throughput > 0.0
        }
        if admissible:
            throughput_workers = min(
                (
                    workers
                    for workers, throughput in admissible.items()
                    if throughput == max(admissible.values())
                ),
                default=1,
            )
    workers = 1 if throughput_workers is None else throughput_workers
    batch_size = 1
    if measured_throughput_by_batch_size:
        admissible_batches = {
            int(size): float(throughput)
            for size, throughput in measured_throughput_by_batch_size.items()
            if 0 < int(size) <= (len(plan.descriptors) or 1)
            and np.isfinite(throughput)
            and throughput > 0.0
            and int(size) * bytes_per_worker
            <= int(available_ram_bytes * memory_fraction)
        }
        if admissible_batches:
            batch_size = min(
                (
                    size
                    for size, throughput in admissible_batches.items()
                    if throughput == max(admissible_batches.values())
                ),
                default=1,
            )
    return FixedRoutingWorkerRecommendation(
        workers=workers,
        cpu_limit=cpu_limit,
        memory_limit=memory_limit,
        estimated_bytes_per_worker=bytes_per_worker,
        reason=(
            "measured throughput calibration within memory and CPU ceilings"
            if throughput_workers is not None
            else "no representative throughput calibration; conservative "
            "single-worker default"
        ),
        memory_admissible_workers=memory_limit,
        cpu_admissible_workers=cpu_limit,
        throughput_effective_workers=throughput_workers,
        recommended_batch_size=batch_size,
    )


def plan_fixed_routing_shards(
    *, inputs: AssignmentInputs, config: FixedRoutingPreparationConfig
) -> FixedRoutingShardPlan:
    """Partition groups canonically under both group and byte ceilings."""
    num_groups = int(inputs.group_dest_node.shape[0])
    num_links = int(inputs.graph.num_links)
    retained_per_group = num_links * (
        np.dtype(bool).itemsize + np.dtype(inputs.base_link_cost.dtype).itemsize
    )
    # Routing preparation retains its output and uses at least one same-sized
    # working pair. This conservative public estimate is validated by benchmarks.
    temporary_per_group = math.ceil(
        retained_per_group * config.temporary_workspace_multiplier
    )
    if retained_per_group > config.maximum_retained_bytes_per_shard:
        raise ValueError("one destination group exceeds the retained-byte ceiling.")
    if temporary_per_group > config.maximum_temporary_bytes:
        raise ValueError("one destination group exceeds the temporary-byte ceiling.")
    byte_groups = config.maximum_retained_bytes_per_shard // max(1, retained_per_group)
    temporary_groups = config.maximum_temporary_bytes // max(1, temporary_per_group)
    groups_per_shard = min(
        config.maximum_groups_per_shard, byte_groups, temporary_groups
    )
    if groups_per_shard <= 0:
        raise ValueError("declared budgets cannot fit one destination group.")
    descriptors = tuple(
        FixedRoutingShardDescriptor(index, start, min(start + groups_per_shard, num_groups))
        for index, start in enumerate(range(0, num_groups, groups_per_shard))
    )
    identity = {
        "num_groups": num_groups,
        "num_links": num_links,
        "groups_per_shard": groups_per_shard,
        "retained_bytes_per_group": retained_per_group,
        "temporary_bytes_per_group": temporary_per_group,
        "descriptors": tuple(
            (item.shard_index, item.group_start, item.group_stop)
            for item in descriptors
        ),
    }
    return FixedRoutingShardPlan(
        descriptors=descriptors,
        groups_per_full_shard=groups_per_shard,
        retained_bytes_per_group=retained_per_group,
        estimated_temporary_bytes_per_group=temporary_per_group,
        estimated_total_retained_bytes=num_groups * retained_per_group,
        plan_fingerprint=fingerprint(identity),
    )


@dataclass(frozen=True, slots=True)
class FixedRoutingShard:
    """Bounded routing payload without duplicated graph-wide input arrays."""

    descriptor: FixedRoutingShardDescriptor
    effective_group_link_mask: np.ndarray
    group_link_probability: np.ndarray

    def __post_init__(self) -> None:
        masks = _immutable_array(self.effective_group_link_mask, dtype=np.dtype(bool))
        source_probability = np.asarray(self.group_link_probability)
        if source_probability.dtype.kind != "f":
            raise TypeError("group_link_probability must have a floating dtype.")
        probability = _immutable_array(
            source_probability, dtype=source_probability.dtype
        )
        if masks.ndim != 2:
            raise ValueError("effective_group_link_mask must be two-dimensional.")
        if probability.shape != masks.shape:
            raise ValueError("routing shard mask and probability shapes must match.")
        if masks.shape[0] != self.descriptor.num_groups:
            raise ValueError("routing shard rows must match its destination-group interval.")
        if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
            raise ValueError("routing shard probabilities must be finite and nonnegative.")
        object.__setattr__(self, "effective_group_link_mask", masks)
        object.__setattr__(self, "group_link_probability", probability)

    @property
    def destination_group_indices(self) -> tuple[int, ...]:
        return self.descriptor.destination_group_indices

    @property
    def retained_bytes(self) -> int:
        return self.effective_group_link_mask.nbytes + self.group_link_probability.nbytes


@dataclass(frozen=True, slots=True)
class FixedRoutingShardCacheProvenance:
    """Identity shared by a manifest and all routing shards it references."""

    preparation_fingerprint: str
    cache_directory: Path
    checkpoint_directory: Path | None = None
    schema_version: int = SHARDED_FIXED_ROUTING_SCHEMA_VERSION
    package_version: str = __version__
    implementation_version: str = SHARDED_FIXED_ROUTING_IMPLEMENTATION_VERSION

    def __post_init__(self) -> None:
        if not self.preparation_fingerprint:
            raise ValueError("preparation_fingerprint must not be empty.")
        object.__setattr__(self, "cache_directory", Path(self.cache_directory))
        if self.checkpoint_directory is not None:
            object.__setattr__(
                self, "checkpoint_directory", Path(self.checkpoint_directory)
            )


@dataclass(frozen=True, slots=True, eq=False)
class ShardedFixedRoutingInputs:
    """Logical fixed routing whose large state is stored in bounded shards.

    Only small shared metadata and fingerprints are retained here. Routing
    masks and probabilities belong exclusively to :class:`FixedRoutingShard`.
    """

    theta: float
    graph: Any
    graph_fingerprint: str
    base_link_cost_fingerprint: str
    source_group_link_mask_fingerprint: str
    assignment_fingerprint: str
    destination_group_identifiers: tuple[int, ...]
    num_nodes: int
    num_links: int
    probability_dtype: str
    shard_partition: tuple[FixedRoutingShardDescriptor, ...]
    provenance: FixedRoutingShardCacheProvenance

    def __post_init__(self) -> None:
        if not np.isfinite(self.theta) or self.theta <= 0.0:
            raise ValueError("theta must be positive and finite.")
        if self.num_nodes < 0 or self.num_links < 0:
            raise ValueError("graph dimensions must be nonnegative.")
        if np.dtype(self.probability_dtype).kind != "f":
            raise TypeError("probability_dtype must be floating point.")
        expected_start = 0
        for expected_index, shard in enumerate(self.shard_partition):
            if shard.shard_index != expected_index:
                raise ValueError("shards must have canonical consecutive identifiers.")
            if shard.group_start != expected_start:
                raise ValueError("shards must form a contiguous canonical partition.")
            expected_start = shard.group_stop
        if expected_start != len(self.destination_group_identifiers):
            if self.destination_group_identifiers or self.shard_partition:
                raise ValueError("shard partition must cover every destination group once.")

    @property
    def num_destination_groups(self) -> int:
        return len(self.destination_group_identifiers)

    @property
    def num_shards(self) -> int:
        return len(self.shard_partition)


def sharded_fixed_routing_identity(
    *,
    inputs: AssignmentInputs,
    theta: float,
    shard_partition: tuple[FixedRoutingShardDescriptor, ...],
) -> dict[str, object]:
    """Return stable routing-sensitive identity fields for future persistence."""
    graph = inputs.graph
    graph_fingerprint = _array_fingerprint(
        graph.tail,
        graph.head,
        graph.topo_order,
        graph.out_links,
        graph.out_mask,
    )
    base_fingerprint = _array_fingerprint(inputs.base_link_cost)
    mask_fingerprint = _array_fingerprint(inputs.group_link_mask)
    assignment_fingerprint = _array_fingerprint(
        graph.tail,
        graph.head,
        graph.topo_order,
        graph.out_links,
        graph.out_mask,
        inputs.base_link_cost,
        inputs.group_dest_node,
        inputs.group_link_mask,
        inputs.od_origin_node,
        inputs.group_od_index_padded,
        inputs.group_od_mask,
    )
    return {
        "schema_version": SHARDED_FIXED_ROUTING_SCHEMA_VERSION,
        "package_version": __version__,
        "implementation_version": SHARDED_FIXED_ROUTING_IMPLEMENTATION_VERSION,
        "assignment_fingerprint": assignment_fingerprint,
        "graph_fingerprint": graph_fingerprint,
        "base_link_cost_fingerprint": base_fingerprint,
        "source_group_link_mask_fingerprint": mask_fingerprint,
        "destination_group_identifiers": tuple(
            int(value) for value in np.asarray(inputs.group_dest_node)
        ),
        "theta": float(theta),
        "num_nodes": int(graph.num_nodes),
        "num_links": int(graph.num_links),
        "probability_dtype": str(inputs.base_link_cost.dtype),
        "shard_partition": tuple(
            (item.shard_index, item.group_start, item.group_stop)
            for item in shard_partition
        ),
    }


def build_sharded_fixed_routing_inputs(
    *,
    inputs: AssignmentInputs,
    theta: float,
    shard_partition: tuple[FixedRoutingShardDescriptor, ...],
    cache_directory: Path,
    checkpoint_directory: Path | None = None,
) -> ShardedFixedRoutingInputs:
    """Build shared immutable metadata without preparing any routing arrays."""
    identity = sharded_fixed_routing_identity(
        inputs=inputs, theta=theta, shard_partition=shard_partition
    )
    preparation_fingerprint = fingerprint(identity)
    return ShardedFixedRoutingInputs(
        theta=float(theta),
        graph=inputs.graph,
        graph_fingerprint=str(identity["graph_fingerprint"]),
        base_link_cost_fingerprint=str(identity["base_link_cost_fingerprint"]),
        source_group_link_mask_fingerprint=str(
            identity["source_group_link_mask_fingerprint"]
        ),
        assignment_fingerprint=str(identity["assignment_fingerprint"]),
        destination_group_identifiers=tuple(
            int(value) for value in np.asarray(inputs.group_dest_node)
        ),
        num_nodes=int(inputs.graph.num_nodes),
        num_links=int(inputs.graph.num_links),
        probability_dtype=str(identity["probability_dtype"]),
        shard_partition=shard_partition,
        provenance=FixedRoutingShardCacheProvenance(
            preparation_fingerprint=preparation_fingerprint,
            cache_directory=cache_directory,
            checkpoint_directory=checkpoint_directory,
        ),
    )


@dataclass(frozen=True, slots=True)
class FixedRoutingShardProgress:
    phase: str
    status: str
    completed_groups: int
    total_groups: int
    completed_shards: int
    total_shards: int
    shard_index: int | None
    cache_hits: int
    cache_misses: int
    elapsed_seconds: float
    recent_shard_seconds: float | None
    estimated_remaining_seconds: float | None
    peak_rss_bytes: int | None
    retained_cache_bytes: int
    deadline_remaining_seconds: float | None
    active_workers: int = 0
    buffered_shards: int = 0
    queued_shards: int = 0
    remaining_shards: int = 0
    failed_shards: int = 0
    current_shard_indices: tuple[int, ...] = ()
    rolling_mean_shard_seconds: float | None = None
    peak_parent_rss_bytes: int | None = None
    estimated_total_worker_rss_bytes: int = 0
    predicted_next_shard_seconds: float | None = None
    dispatch_prevented_by_deadline: bool = False
    configured_rss_ceiling: int | None = None
    estimated_worker_peak_bytes: int = 0
    admitted_worker_count: int = 1
    memory_limited_worker_count: int = 0
    cpu_limited_worker_count: int = 0
    worker_architecture: str = "serial"
    effective_threads_per_worker: int | None = None
    throughput_effective_worker_count: int | None = None
    batch_shard_indices: tuple[int, ...] = ()
    batch_size: int = 0
    padded_batch_size: int = 0
    batch_execution_seconds: float | None = None
    batch_persistence_seconds: float | None = None
    aggregate_shards_per_second: float | None = None
    predicted_next_batch_seconds: float | None = None
    effective_cpu_cores: float | None = None
    eta_confidence: str = "unavailable"
    estimated_completion_at_utc: str | None = None
    eta_reason: str | None = None


@dataclass(frozen=True, slots=True)
class FixedRoutingShardExecutionDiagnostics:
    shard_index: int
    num_groups: int
    host_destination_preparation_seconds: float
    host_mask_preparation_seconds: float
    argument_transfer_seconds: float
    kernel_execution_seconds: float
    device_synchronization_seconds: float
    host_transfer_seconds: float
    output_slicing_seconds: float
    validation_seconds: float
    shard_persistence_seconds: float
    manifest_persistence_seconds: float
    cleanup_seconds: float
    total_shard_seconds: float
    process_cpu_utilization_percent: float | None
    active_process_threads: int | None
    backend: str
    devices: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    output_dtypes: tuple[str, ...]
    retained_bytes: int
    estimated_temporary_bytes: int
    device_memory_stats: tuple[dict[str, object] | None, ...]
    graph_nodes_traversed: int
    graph_links_traversed: int
    enabled_links: int
    enabled_link_fraction: float
    probability_nonzeros: int
    probability_density: float
    effective_probability_nonzeros: int = 0
    effective_probability_density: float = 0.0
    jax_execution_wait_seconds: float = 0.0
    concurrent_jax_executions: int = 1
    voluntary_context_switches: int | None = None
    involuntary_context_switches: int | None = None


@dataclass(frozen=True, slots=True)
class FixedRoutingBatchExecutionDiagnostics:
    """Synchronized resource and throughput measurements for one shard batch."""

    batch_index: int
    shard_indices: tuple[int, ...]
    batch_size: int
    padded_batch_size: int
    execution_seconds: float
    persistence_seconds: float
    wall_seconds: float
    process_cpu_seconds: float
    effective_cpu_cores: float
    shards_per_second: float
    peak_rss_bytes: int | None
    voluntary_context_switches: int | None = None
    involuntary_context_switches: int | None = None


@dataclass(frozen=True, slots=True)
class ShardedFixedRoutingPreparationResult:
    routing: ShardedFixedRoutingInputs
    plan: FixedRoutingShardPlan
    status: str
    completed_shards: int
    cache_hits: int
    cache_misses: int
    reconstructed_shards: int
    elapsed_seconds: float
    retained_cache_bytes: int
    peak_rss_bytes: int | None
    deadline_phase: str | None
    indivisible_operation_overshoot: bool
    deadline_overshoot_seconds: float
    compilation_count: int
    tracing_seconds: float
    lowering_seconds: float
    compilation_seconds: float
    shard_diagnostics: tuple[FixedRoutingShardExecutionDiagnostics, ...] = ()
    predicted_next_shard_seconds: float | None = None
    dispatch_prevented_by_deadline: bool = False
    admitted_worker_count: int = 1
    memory_limited_worker_count: int = 0
    cpu_limited_worker_count: int = 0
    estimated_worker_peak_bytes: int = 0
    worker_architecture: str = "serial"
    effective_threads_per_worker: int | None = None
    compiled_kernel_identity: str | None = None
    persistent_compilation_cache_enabled: bool = False
    persistent_compilation_cache_directory: str | None = None
    throughput_effective_worker_count: int | None = None
    execution_strategy: ExecutionStrategy = "thread_pool"
    shards_per_execution_batch: int = 1
    batch_diagnostics: tuple[FixedRoutingBatchExecutionDiagnostics, ...] = ()

    @property
    def newly_constructed_shards(self) -> int:
        """Number of cache misses completed during this bounded invocation.

        ``cache_misses`` deliberately counts every planned shard absent during
        the initial cache scan, including shards left unattempted by a resource
        or deadline stop.
        """

        return len(self.shard_diagnostics) if self.shard_diagnostics else max(
            0, self.completed_shards - self.cache_hits
        )


class FixedRoutingShardCacheError(ValueError):
    """Raised when a persisted shard or manifest fails identity validation."""


def _peak_rss() -> int | None:
    try:
        import resource
        import sys

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return None


def _active_process_threads() -> int | None:
    try:
        status = Path("/proc/self/status")
        if status.exists():
            for line in status.read_text(encoding="utf-8").splitlines():
                if line.startswith("Threads:"):
                    return int(line.split(":", maxsplit=1)[1])
    except (OSError, ValueError):
        return None
    # Python's count is portable and useful on platforms without /proc. It can
    # undercount native runtime threads, so it is not presented as an XLA
    # threads-per-worker measurement.
    return threading.active_count()


def _process_context_switches() -> tuple[int, int] | None:
    """Return process-wide voluntary and involuntary context-switch counts."""

    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return int(usage.ru_nvcsw), int(usage.ru_nivcsw)
    except (ImportError, OSError, ValueError):
        return None


def _device_memory_statistics() -> tuple[dict[str, object] | None, ...]:
    result: list[dict[str, object] | None] = []
    for device in jax.devices():
        try:
            statistics = device.memory_stats()
        except (AttributeError, RuntimeError):
            statistics = None
        result.append(None if statistics is None else dict(statistics))
    return tuple(result)


def _atomic_json(path: Path, payload: dict[str, object], *, durable: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _shard_identity(
    routing: ShardedFixedRoutingInputs, descriptor: FixedRoutingShardDescriptor
) -> dict[str, object]:
    return {
        "schema_version": routing.provenance.schema_version,
        "package_version": routing.provenance.package_version,
        "implementation_version": routing.provenance.implementation_version,
        "preparation_fingerprint": routing.provenance.preparation_fingerprint,
        "shard_index": descriptor.shard_index,
        "group_start": descriptor.group_start,
        "group_stop": descriptor.group_stop,
        "num_links": routing.num_links,
        "probability_dtype": routing.probability_dtype,
    }


def fixed_routing_shard_path(
    routing: ShardedFixedRoutingInputs, descriptor: FixedRoutingShardDescriptor
) -> Path:
    return routing.provenance.cache_directory / f"routing-shard-{descriptor.shard_index:06d}.npz"


def fixed_routing_shard_content_hash(shard: FixedRoutingShard) -> str:
    """Return the deterministic hash of one routing shard's scientific arrays."""
    return _array_fingerprint(
        shard.effective_group_link_mask, shard.group_link_probability
    )


def save_fixed_routing_shard(
    *,
    routing: ShardedFixedRoutingInputs,
    shard: FixedRoutingShard,
    durable: bool = True,
) -> Path:
    """Atomically persist one validated routing shard."""
    if shard.descriptor not in routing.shard_partition:
        raise ValueError("shard descriptor is not part of this routing partition.")
    if shard.effective_group_link_mask.shape[1] != routing.num_links:
        raise ValueError("shard link dimension does not match routing metadata.")
    if str(shard.group_link_probability.dtype) != routing.probability_dtype:
        raise ValueError("shard probability dtype does not match routing metadata.")
    path = fixed_routing_shard_path(routing, shard.descriptor)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        content_hash = fixed_routing_shard_content_hash(shard)
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(
                stream,
                identity=np.asarray(json.dumps(_shard_identity(routing, shard.descriptor), sort_keys=True)),
                content_hash=np.asarray(content_hash),
                effective_group_link_mask=shard.effective_group_link_mask,
                group_link_probability=shard.group_link_probability,
            )
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        with np.load(temporary_name, allow_pickle=False) as staged:
            staged_shard = FixedRoutingShard(
                descriptor=shard.descriptor,
                effective_group_link_mask=staged["effective_group_link_mask"],
                group_link_probability=staged["group_link_probability"],
            )
            if str(staged["content_hash"].item()) != fixed_routing_shard_content_hash(
                staged_shard
            ):
                raise FixedRoutingShardCacheError(
                    "staged routing shard content hash mismatch."
                )
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


def load_fixed_routing_shard(
    *, routing: ShardedFixedRoutingInputs, descriptor: FixedRoutingShardDescriptor
) -> FixedRoutingShard:
    """Load one shard only after strict provenance and payload validation."""
    path = fixed_routing_shard_path(routing, descriptor)
    try:
        with np.load(path, allow_pickle=False) as payload:
            identity = json.loads(str(payload["identity"].item()))
            if identity != _shard_identity(routing, descriptor):
                raise FixedRoutingShardCacheError(
                    f"routing shard {descriptor.shard_index} identity mismatch."
                )
            shard = FixedRoutingShard(
                descriptor=descriptor,
                effective_group_link_mask=payload["effective_group_link_mask"],
                group_link_probability=payload["group_link_probability"],
            )
            if str(payload["content_hash"].item()) != fixed_routing_shard_content_hash(
                shard
            ):
                raise FixedRoutingShardCacheError(
                    f"routing shard {descriptor.shard_index} content hash mismatch."
                )
    except FixedRoutingShardCacheError:
        raise
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise FixedRoutingShardCacheError(
            f"routing shard {descriptor.shard_index} is corrupt or incomplete."
        ) from error
    if shard.effective_group_link_mask.shape != (descriptor.num_groups, routing.num_links):
        raise FixedRoutingShardCacheError("routing shard payload shape mismatch.")
    if str(shard.group_link_probability.dtype) != routing.probability_dtype:
        raise FixedRoutingShardCacheError("routing shard payload dtype mismatch.")
    return shard


def _reuse_or_quarantine_fixed_routing_shard(
    *, routing: ShardedFixedRoutingInputs, descriptor: FixedRoutingShardDescriptor
) -> FixedRoutingShard | None:
    path = fixed_routing_shard_path(routing, descriptor)
    if not path.exists():
        return None
    try:
        return load_fixed_routing_shard(routing=routing, descriptor=descriptor)
    except FixedRoutingShardCacheError:
        quarantine = path.with_name(f"{path.name}.invalid-{uuid.uuid4().hex}")
        os.replace(path, quarantine)
        return None


def validate_sharded_fixed_routing_compatibility(
    *, inputs: AssignmentInputs, routing: ShardedFixedRoutingInputs
) -> None:
    """Reject a sharded source prepared from different assignment inputs."""
    if routing.graph is not inputs.graph:
        raise ValueError("Sharded fixed routing was prepared for a different graph.")
    identity = sharded_fixed_routing_identity(
        inputs=inputs,
        theta=routing.theta,
        shard_partition=routing.shard_partition,
    )
    expected = fingerprint(identity)
    if expected != routing.provenance.preparation_fingerprint:
        raise ValueError("Sharded fixed routing is incompatible with assignment inputs.")


def fixed_routing_descriptor_for_group(
    routing: ShardedFixedRoutingInputs, group: int
) -> FixedRoutingShardDescriptor:
    """Return the unique shard descriptor containing one destination group."""
    if group < 0 or group >= routing.num_destination_groups:
        raise IndexError("destination group is outside the routing source.")
    for descriptor in routing.shard_partition:
        if descriptor.group_start <= group < descriptor.group_stop:
            return descriptor
    raise RuntimeError("routing partition does not cover the destination group.")


def iter_fixed_routing_shards(
    routing: ShardedFixedRoutingInputs,
) -> Iterator[FixedRoutingShard]:
    """Load validated shards in canonical order.

    At most the yielded shard is retained by this generator itself.
    """
    for descriptor in routing.shard_partition:
        yield load_fixed_routing_shard(routing=routing, descriptor=descriptor)


def materialize_sharded_fixed_routing_dense(
    *,
    routing: ShardedFixedRoutingInputs,
    inputs: AssignmentInputs,
    memory_limit_bytes: int,
) -> "FixedRoutingInputs":
    """Explicitly materialize a small sharded source after a memory guard."""
    from .assignment_adapter import FixedRoutingInputs

    validate_sharded_fixed_routing_compatibility(inputs=inputs, routing=routing)
    if memory_limit_bytes <= 0:
        raise ValueError("memory_limit_bytes must be positive.")
    required = routing.num_destination_groups * routing.num_links * (
        np.dtype(bool).itemsize + np.dtype(routing.probability_dtype).itemsize
    )
    if required > memory_limit_bytes:
        raise MemoryError(
            "dense fixed-routing materialization requires "
            f"{required} bytes, above the explicit {memory_limit_bytes}-byte limit."
        )
    masks = np.empty(
        (routing.num_destination_groups, routing.num_links), dtype=bool
    )
    probabilities = np.empty(
        (routing.num_destination_groups, routing.num_links),
        dtype=np.dtype(routing.probability_dtype),
    )
    for descriptor in routing.shard_partition:
        shard = load_fixed_routing_shard(routing=routing, descriptor=descriptor)
        masks[descriptor.group_start : descriptor.group_stop] = (
            shard.effective_group_link_mask
        )
        probabilities[descriptor.group_start : descriptor.group_stop] = (
            shard.group_link_probability
        )
    return FixedRoutingInputs(
        theta=jnp.asarray(routing.theta, dtype=inputs.base_link_cost.dtype),
        graph=inputs.graph,
        source_base_link_cost=inputs.base_link_cost,
        group_dest_node=inputs.group_dest_node,
        source_group_link_mask=inputs.group_link_mask,
        effective_group_link_mask=jnp.asarray(masks),
        group_link_probability=jnp.asarray(probabilities),
        num_nodes=routing.num_nodes,
        num_links=routing.num_links,
    )


def _manifest_path(routing: ShardedFixedRoutingInputs) -> Path:
    directory = (
        routing.provenance.checkpoint_directory
        if routing.provenance.checkpoint_directory is not None
        else routing.provenance.cache_directory
    )
    return directory / "manifest.json"


def _write_manifest(
    *,
    routing: ShardedFixedRoutingInputs,
    completed: list[int],
    status: str,
    elapsed_seconds: float,
    durable: bool,
    shard_durations: dict[int, float] | None = None,
) -> None:
    canonical_completed = sorted(set(completed))
    completed_set = set(canonical_completed)
    next_position = next(
        (
            descriptor.shard_index
            for descriptor in routing.shard_partition
            if descriptor.shard_index not in completed_set
        ),
        routing.num_shards,
    )
    path = _manifest_path(routing)
    if shard_durations is None and path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            raw_durations = previous.get("shard_durations", {})
            shard_durations = {
                int(index): float(seconds)
                for index, seconds in raw_durations.items()
            }
        except (OSError, ValueError, json.JSONDecodeError, AttributeError):
            shard_durations = {}
    _atomic_json(
        path,
        {
            "schema_version": routing.provenance.schema_version,
            "preparation_fingerprint": routing.provenance.preparation_fingerprint,
            "expected_shards": routing.num_shards,
            "completed_shards": canonical_completed,
            "next_shard_position": next_position,
            "elapsed_seconds": elapsed_seconds,
            "status": status,
            "shard_durations": {
                str(index): seconds
                for index, seconds in sorted((shard_durations or {}).items())
            },
        },
        durable=durable,
    )


def _manifest_shard_durations(
    routing: ShardedFixedRoutingInputs,
) -> dict[int, float]:
    path = _manifest_path(routing)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            int(index): float(seconds)
            for index, seconds in payload.get("shard_durations", {}).items()
            if float(seconds) > 0.0
        }
    except (OSError, ValueError, json.JSONDecodeError, AttributeError):
        return {}


def _manifest_completed_shards(
    routing: ShardedFixedRoutingInputs,
    *,
    cache_policy: CachePolicy,
) -> list[int]:
    """Return only manifest-listed shards whose files are still reusable."""

    if cache_policy == "refresh":
        return []
    path = _manifest_path(routing)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        listed = {
            int(index)
            for index in payload.get("completed_shards", ())
            if 0 <= int(index) < routing.num_shards
        }
    except (OSError, ValueError, json.JSONDecodeError, AttributeError, TypeError):
        return []
    return sorted(
        index
        for index in listed
        if fixed_routing_shard_path(
            routing, routing.shard_partition[index]
        ).exists()
    )


class _RoutingProgressHeartbeat:
    """Throttle progress observations during long opaque routing subphases."""

    def __init__(
        self,
        callback: ProgressCallback,
        initial: FixedRoutingShardProgress,
        interval_seconds: float,
    ) -> None:
        self._callback = callback
        self._latest = initial
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._interval = max(0.01, float(interval_seconds))
        self._thread: threading.Thread | None = None

    def emit(self, event: FixedRoutingShardProgress) -> None:
        with self._lock:
            self._latest = event
        self._callback(event)

    def start(self) -> None:
        def run() -> None:
            while not self._stop.wait(self._interval):
                with self._lock:
                    event = self._latest
                self._callback(
                    replace(
                        event,
                        status="running",
                        eta_confidence=(
                            event.eta_confidence
                            if event.eta_confidence != "unavailable"
                            else "unavailable"
                        ),
                        eta_reason=(
                            event.eta_reason
                            or "heartbeat before another shard was persisted"
                        ),
                    )
                )

        self._thread = threading.Thread(
            target=run, name="fixed-routing-progress-heartbeat", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval * 2.0))


def _validate_existing_manifest(
    *, routing: ShardedFixedRoutingInputs, cache_policy: CachePolicy
) -> None:
    path = _manifest_path(routing)
    if not path.exists() or cache_policy == "refresh":
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FixedRoutingShardCacheError(
            "fixed-routing preparation manifest is corrupt."
        ) from error
    expected = {
        "schema_version": routing.provenance.schema_version,
        "preparation_fingerprint": routing.provenance.preparation_fingerprint,
        "expected_shards": routing.num_shards,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise FixedRoutingShardCacheError(
            "fixed-routing preparation manifest identity mismatch."
        )


def _padded_shard_inputs(
    *,
    inputs: AssignmentInputs,
    descriptor: FixedRoutingShardDescriptor,
    padded_count: int,
    num_links: int,
) -> AssignmentInputs:
    destination = np.zeros(padded_count, dtype=np.int32)
    masks = np.zeros((padded_count, num_links), dtype=bool)
    destination[: descriptor.num_groups] = np.asarray(
        inputs.group_dest_node[descriptor.group_start : descriptor.group_stop]
    )
    masks[: descriptor.num_groups] = np.asarray(
        inputs.group_link_mask[descriptor.group_start : descriptor.group_stop]
    )
    destination_device = jnp.asarray(destination)
    masks_device = jnp.asarray(masks)
    jax.block_until_ready((destination_device, masks_device))
    return replace(
        inputs,
        group_dest_node=destination_device,
        group_link_mask=masks_device,
    )


@dataclass(frozen=True, slots=True)
class _ThreadShardResult:
    descriptor: FixedRoutingShardDescriptor
    stored_bytes: int
    elapsed_seconds: float
    peak_rss_bytes: int | None
    diagnostic: FixedRoutingShardExecutionDiagnostics | None = None


def _prepare_fixed_routing_sharded_threads(
    *,
    inputs: AssignmentInputs,
    theta: float,
    config: FixedRoutingPreparationConfig,
    absolute_deadline: float | None,
    progress: ProgressCallback | None,
    cache_policy: CachePolicy,
    clock: Callable[[], float],
    started: float,
    plan: FixedRoutingShardPlan,
    routing: ShardedFixedRoutingInputs,
    compiled_kernel_identity: str,
    persistent_cache_enabled: bool,
    persistent_cache_directory: str | None,
) -> ShardedFixedRoutingPreparationResult:
    """Construct independent shards concurrently with a shared executable.

    Threads are intentional here: the full assignment graph and compiled XLA
    executable remain shared, whereas spawned JAX processes duplicate the
    multi-gigabyte graph/runtime state. XLA execution releases the Python GIL.
    """
    completed: list[int] = []
    shard_duration_map = _manifest_shard_durations(routing)
    cache_hits = cache_misses = reconstructed = cache_bytes = 0
    missing: list[FixedRoutingShardDescriptor] = []
    peak_rss = _peak_rss()
    for descriptor in plan.descriptors:
        path = fixed_routing_shard_path(routing, descriptor)
        if path.exists() and cache_policy == "reuse":
            cached = _reuse_or_quarantine_fixed_routing_shard(
                routing=routing, descriptor=descriptor
            )
            if cached is not None:
                completed.append(descriptor.shard_index)
                cache_hits += 1
                cache_bytes += path.stat().st_size
                continue
            reconstructed += 1
        cache_misses += 1
        reconstructed += int(path.exists() and cache_policy != "reuse")
        missing.append(descriptor)

    if not missing:
        elapsed = clock() - started
        planning_event = None
        if progress is not None:
            planning_event = FixedRoutingShardProgress(
                phase="planning_cache_scan",
                status="completed",
                completed_groups=routing.num_destination_groups,
                total_groups=routing.num_destination_groups,
                completed_shards=len(completed),
                total_shards=routing.num_shards,
                shard_index=None,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                elapsed_seconds=elapsed,
                recent_shard_seconds=None,
                estimated_remaining_seconds=0.0,
                peak_rss_bytes=peak_rss,
                retained_cache_bytes=cache_bytes,
                deadline_remaining_seconds=(
                    None
                    if absolute_deadline is None
                    else max(0.0, absolute_deadline - clock())
                ),
                queued_shards=0,
                remaining_shards=0,
                admitted_worker_count=0,
                worker_architecture="shared-executable-thread-pool",
            )
            progress(planning_event)
        _write_manifest(
            routing=routing,
            completed=completed,
            status="completed",
            elapsed_seconds=elapsed,
            durable=config.durable_progress,
        )
        if progress is not None and planning_event is not None:
            progress(replace(planning_event, phase="terminal"))
        return ShardedFixedRoutingPreparationResult(
            routing=routing,
            plan=plan,
            status="completed",
            completed_shards=len(completed),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            reconstructed_shards=reconstructed,
            elapsed_seconds=elapsed,
            retained_cache_bytes=cache_bytes,
            peak_rss_bytes=peak_rss,
            deadline_phase=None,
            indivisible_operation_overshoot=False,
            deadline_overshoot_seconds=0.0,
            compilation_count=0,
            tracing_seconds=0.0,
            lowering_seconds=0.0,
            compilation_seconds=0.0,
            compiled_kernel_identity=compiled_kernel_identity,
            persistent_compilation_cache_enabled=persistent_cache_enabled,
            persistent_compilation_cache_directory=persistent_cache_directory,
        )

    estimated_worker_peak = (
        plan.groups_per_full_shard * plan.estimated_temporary_bytes_per_group
    )
    requested_workers = min(config.construction_workers, len(missing))
    cpu_cap = max(1, (os.cpu_count() or 1) // 2)
    admitted_workers = min(requested_workers, cpu_cap)
    cpu_limited_workers = max(0, requested_workers - admitted_workers)
    memory_limited_workers = 0
    if config.maximum_process_rss_bytes is not None and peak_rss is not None:
        safety_margin = max(
            256 * 1024 * 1024,
            int(config.maximum_process_rss_bytes * 0.1),
        )
        available = max(
            0, config.maximum_process_rss_bytes - peak_rss - safety_margin
        )
        memory_workers = available // max(1, estimated_worker_peak)
        memory_limited_workers = max(
            0, admitted_workers - int(memory_workers)
        )
        admitted_workers = min(admitted_workers, int(memory_workers))
    if admitted_workers <= 0:
        elapsed = clock() - started
        planning_event = None
        if progress is not None:
            planning_event = FixedRoutingShardProgress(
                phase="planning_cache_scan",
                status="memory_budget_reached",
                completed_groups=sum(
                    plan.descriptors[index].num_groups for index in completed
                ),
                total_groups=routing.num_destination_groups,
                completed_shards=len(completed),
                total_shards=routing.num_shards,
                shard_index=None,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                elapsed_seconds=elapsed,
                recent_shard_seconds=None,
                estimated_remaining_seconds=None,
                peak_rss_bytes=peak_rss,
                retained_cache_bytes=cache_bytes,
                deadline_remaining_seconds=(
                    None
                    if absolute_deadline is None
                    else max(0.0, absolute_deadline - clock())
                ),
                queued_shards=len(missing),
                remaining_shards=routing.num_shards - len(completed),
                configured_rss_ceiling=config.maximum_process_rss_bytes,
                estimated_worker_peak_bytes=estimated_worker_peak,
                admitted_worker_count=0,
                memory_limited_worker_count=memory_limited_workers,
                cpu_limited_worker_count=cpu_limited_workers,
                worker_architecture="shared-executable-thread-pool",
            )
            progress(planning_event)
        _write_manifest(
            routing=routing,
            completed=completed,
            status="memory_budget_reached",
            elapsed_seconds=elapsed,
            durable=config.durable_progress,
        )
        if progress is not None and planning_event is not None:
            progress(replace(planning_event, phase="terminal"))
        return ShardedFixedRoutingPreparationResult(
            routing=routing,
            plan=plan,
            status="memory_budget_reached",
            completed_shards=len(completed),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            reconstructed_shards=reconstructed,
            elapsed_seconds=elapsed,
            retained_cache_bytes=cache_bytes,
            peak_rss_bytes=peak_rss,
            deadline_phase=None,
            indivisible_operation_overshoot=False,
            deadline_overshoot_seconds=0.0,
            compilation_count=0,
            tracing_seconds=0.0,
            lowering_seconds=0.0,
            compilation_seconds=0.0,
            admitted_worker_count=0,
            memory_limited_worker_count=memory_limited_workers,
            cpu_limited_worker_count=cpu_limited_workers,
            estimated_worker_peak_bytes=estimated_worker_peak,
            worker_architecture="shared-executable-thread-pool",
            effective_threads_per_worker=None,
            compiled_kernel_identity=compiled_kernel_identity,
            persistent_compilation_cache_enabled=persistent_cache_enabled,
            persistent_compilation_cache_directory=persistent_cache_directory,
        )

    active: dict[Future[_ThreadShardResult], FixedRoutingShardDescriptor] = {}
    buffered: dict[int, _ThreadShardResult] = {}
    submitted_shards = 0
    failed_shards = 0
    shard_times: list[float] = list(shard_duration_map.values())
    dispatch_prevented = False
    dispatch_guard_emitted = False

    def emit_progress(
        *,
        phase: str,
        status_value: str,
        shard_index: int | None = None,
        recent_seconds: float | None = None,
        predicted_seconds: float | None = None,
    ) -> None:
        if progress is None:
            return
        current_indices = tuple(
            sorted(item.shard_index for item in active.values())
        )
        active_workers = len(active)
        queued_shards = max(0, len(missing) - submitted_shards)
        remaining_shards = routing.num_shards - len(completed)
        accounted = (
            len(completed)
            + active_workers
            + len(buffered)
            + queued_shards
            + failed_shards
        )
        if accounted != routing.num_shards:
            raise RuntimeError(
                "invalid fixed-routing progress accounting: "
                f"{accounted} != {routing.num_shards}."
            )
        elapsed = max(0.0, clock() - started)
        completed_durations = [
            shard_duration_map[index]
            for index in completed
            if index in shard_duration_map
        ]
        rolling = (
            float(np.mean(completed_durations[-5:]))
            if completed_durations
            else None
        )
        eta = estimate_completed_unit_eta(
            completed_durations,
            completed_units=len(completed),
            total_units=routing.num_shards,
            parallelism=admitted_workers,
        )
        progress(
            FixedRoutingShardProgress(
                phase=phase,
                status=status_value,
                completed_groups=sum(
                    plan.descriptors[index].num_groups for index in completed
                ),
                total_groups=routing.num_destination_groups,
                completed_shards=len(completed),
                total_shards=routing.num_shards,
                shard_index=shard_index,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                elapsed_seconds=elapsed,
                recent_shard_seconds=recent_seconds,
                estimated_remaining_seconds=eta.predicted_remaining_seconds,
                peak_rss_bytes=peak_rss,
                retained_cache_bytes=cache_bytes,
                deadline_remaining_seconds=(
                    None
                    if absolute_deadline is None
                    else max(0.0, absolute_deadline - clock())
                ),
                active_workers=active_workers,
                buffered_shards=len(buffered),
                queued_shards=queued_shards,
                remaining_shards=remaining_shards,
                failed_shards=failed_shards,
                current_shard_indices=current_indices,
                rolling_mean_shard_seconds=rolling,
                peak_parent_rss_bytes=peak_rss,
                estimated_total_worker_rss_bytes=(
                    active_workers * estimated_worker_peak
                ),
                predicted_next_shard_seconds=predicted_seconds,
                dispatch_prevented_by_deadline=dispatch_prevented,
                configured_rss_ceiling=config.maximum_process_rss_bytes,
                estimated_worker_peak_bytes=estimated_worker_peak,
                admitted_worker_count=admitted_workers,
                memory_limited_worker_count=memory_limited_workers,
                cpu_limited_worker_count=cpu_limited_workers,
                worker_architecture="shared-executable-thread-pool",
                effective_threads_per_worker=None,
                eta_confidence=eta.eta_confidence,
                estimated_completion_at_utc=eta.estimated_completion_at_utc,
                eta_reason=eta.eta_reason,
            )
        )

    emit_progress(
        phase="planning_cache_scan",
        status_value="completed",
    )

    def deadline_stop(
        phase: str,
        *,
        tracing_seconds: float = 0.0,
        lowering_seconds: float = 0.0,
        compilation_seconds: float = 0.0,
        after_indivisible: bool = False,
    ) -> ShardedFixedRoutingPreparationResult:
        elapsed = clock() - started
        _write_manifest(
            routing=routing,
            completed=completed,
            status="deadline_reached",
            elapsed_seconds=elapsed,
            durable=config.durable_progress,
        )
        emit_progress(
            phase="terminal",
            status_value="deadline_reached",
        )
        overshoot = (
            0.0
            if absolute_deadline is None
            else max(0.0, clock() - absolute_deadline)
        )
        return ShardedFixedRoutingPreparationResult(
            routing=routing,
            plan=plan,
            status="deadline_reached",
            completed_shards=len(completed),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            reconstructed_shards=reconstructed,
            elapsed_seconds=elapsed,
            retained_cache_bytes=cache_bytes,
            peak_rss_bytes=peak_rss,
            deadline_phase=phase,
            indivisible_operation_overshoot=after_indivisible and overshoot > 0.0,
            deadline_overshoot_seconds=overshoot,
            compilation_count=int(compilation_seconds > 0.0),
            tracing_seconds=tracing_seconds,
            lowering_seconds=lowering_seconds,
            compilation_seconds=compilation_seconds,
            admitted_worker_count=admitted_workers,
            memory_limited_worker_count=memory_limited_workers,
            cpu_limited_worker_count=cpu_limited_workers,
            estimated_worker_peak_bytes=estimated_worker_peak,
            worker_architecture="shared-executable-thread-pool",
            effective_threads_per_worker=None,
            compiled_kernel_identity=compiled_kernel_identity,
            persistent_compilation_cache_enabled=persistent_cache_enabled,
            persistent_compilation_cache_directory=persistent_cache_directory,
        )

    if absolute_deadline is not None and clock() >= absolute_deadline:
        return deadline_stop("before tracing")
    theta_array = jnp.asarray(theta, dtype=inputs.base_link_cost.dtype).reshape(())
    prototype = _padded_shard_inputs(
        inputs=inputs,
        descriptor=missing[0],
        padded_count=plan.groups_per_full_shard,
        num_links=routing.num_links,
    )
    emit_progress(phase="trace", status_value="started")
    phase_started = clock()
    traced = _prepare_fixed_routing_core.trace(inputs=prototype, theta=theta_array)
    tracing = clock() - phase_started
    emit_progress(phase="trace", status_value="completed", recent_seconds=tracing)
    if absolute_deadline is not None and clock() >= absolute_deadline:
        return deadline_stop(
            "tracing", tracing_seconds=tracing, after_indivisible=True
        )
    phase_started = clock()
    emit_progress(phase="lowering", status_value="started")
    lowered = traced.lower()
    lowering = clock() - phase_started
    emit_progress(phase="lowering", status_value="completed", recent_seconds=lowering)
    if absolute_deadline is not None and clock() >= absolute_deadline:
        return deadline_stop(
            "lowering",
            tracing_seconds=tracing,
            lowering_seconds=lowering,
            after_indivisible=True,
        )
    phase_started = clock()
    emit_progress(phase="compilation", status_value="started")
    executable = lowered.compile()
    compilation = clock() - phase_started
    emit_progress(
        phase="compilation", status_value="completed", recent_seconds=compilation
    )
    if absolute_deadline is not None and clock() >= absolute_deadline:
        return deadline_stop(
            "compilation",
            tracing_seconds=tracing,
            lowering_seconds=lowering,
            compilation_seconds=compilation,
            after_indivisible=True,
        )
    del prototype

    execution_state_lock = threading.Lock()
    active_jax_executions = 0

    def construct(descriptor: FixedRoutingShardDescriptor) -> _ThreadShardResult:
        nonlocal active_jax_executions
        item_started = clock()
        if not config.detailed_profiling:
            shard_inputs = _padded_shard_inputs(
                inputs=inputs,
                descriptor=descriptor,
                padded_count=plan.groups_per_full_shard,
                num_links=routing.num_links,
            )
            effective, probability = executable(
                inputs=shard_inputs, theta=theta_array
            )
            jax.block_until_ready((effective, probability))
            count = descriptor.num_groups
            shard = FixedRoutingShard(
                descriptor=descriptor,
                effective_group_link_mask=np.asarray(effective)[:count],
                group_link_probability=np.asarray(probability)[:count],
            )
            path = save_fixed_routing_shard(
                routing=routing, shard=shard, durable=config.durable_progress
            )
            return _ThreadShardResult(
                descriptor=descriptor,
                stored_bytes=path.stat().st_size,
                elapsed_seconds=clock() - item_started,
                peak_rss_bytes=_peak_rss(),
            )
        cpu_started = process_time()
        context_switches_started = _process_context_switches()
        count = descriptor.num_groups
        padded_count = plan.groups_per_full_shard
        phase_started = clock()
        destination = np.zeros(padded_count, dtype=np.int32)
        destination[:count] = np.asarray(
            inputs.group_dest_node[descriptor.group_start : descriptor.group_stop]
        )
        host_destination_seconds = clock() - phase_started
        phase_started = clock()
        masks = np.zeros((padded_count, routing.num_links), dtype=bool)
        masks[:count] = np.asarray(
            inputs.group_link_mask[descriptor.group_start : descriptor.group_stop]
        )
        host_mask_seconds = clock() - phase_started
        phase_started = clock()
        destination_device = jnp.asarray(destination)
        masks_device = jnp.asarray(masks)
        jax.block_until_ready((destination_device, masks_device))
        argument_transfer_seconds = clock() - phase_started
        shard_inputs = replace(
            inputs,
            group_dest_node=destination_device,
            group_link_mask=masks_device,
        )
        execution_wait_started = clock()
        with execution_state_lock:
            active_jax_executions += 1
            concurrent_jax_executions = active_jax_executions
        jax_execution_wait_seconds = clock() - execution_wait_started
        try:
            phase_started = clock()
            effective, probability = executable(
                inputs=shard_inputs, theta=theta_array
            )
            kernel_execution_seconds = clock() - phase_started
            phase_started = clock()
            jax.block_until_ready((effective, probability))
            synchronization_seconds = clock() - phase_started
        finally:
            with execution_state_lock:
                active_jax_executions -= 1
        phase_started = clock()
        effective_host = np.asarray(effective)
        probability_host = np.asarray(probability)
        host_transfer_seconds = clock() - phase_started
        phase_started = clock()
        effective_slice = effective_host[:count]
        probability_slice = probability_host[:count]
        output_slicing_seconds = clock() - phase_started
        phase_started = clock()
        shard = FixedRoutingShard(
            descriptor=descriptor,
            effective_group_link_mask=effective_slice,
            group_link_probability=probability_slice,
        )
        validation_seconds = clock() - phase_started
        retained_bytes = shard.retained_bytes
        output_shapes = (effective_slice.shape, probability_slice.shape)
        output_dtypes = (str(effective_slice.dtype), str(probability_slice.dtype))
        enabled_links = int(np.count_nonzero(effective_slice))
        probability_nonzeros = int(
            np.count_nonzero(probability_slice[effective_slice])
        )
        phase_started = clock()
        path = save_fixed_routing_shard(
            routing=routing, shard=shard, durable=config.durable_progress
        )
        shard_persistence_seconds = clock() - phase_started
        phase_started = clock()
        del (
            shard,
            effective,
            probability,
            effective_host,
            probability_host,
            effective_slice,
            probability_slice,
            shard_inputs,
            destination_device,
            masks_device,
        )
        cleanup_seconds = clock() - phase_started
        elapsed = clock() - item_started
        diagnostic = None
        if config.detailed_profiling:
            cpu_seconds = max(0.0, process_time() - cpu_started)
            context_switches_finished = _process_context_switches()
            voluntary_context_switches = None
            involuntary_context_switches = None
            if (
                context_switches_started is not None
                and context_switches_finished is not None
            ):
                voluntary_context_switches = max(
                    0, context_switches_finished[0] - context_switches_started[0]
                )
                involuntary_context_switches = max(
                    0, context_switches_finished[1] - context_switches_started[1]
                )
            graph_links_traversed = padded_count * routing.num_links
            retained_domain = count * routing.num_links
            diagnostic = FixedRoutingShardExecutionDiagnostics(
                shard_index=descriptor.shard_index,
                num_groups=count,
                host_destination_preparation_seconds=host_destination_seconds,
                host_mask_preparation_seconds=host_mask_seconds,
                argument_transfer_seconds=argument_transfer_seconds,
                kernel_execution_seconds=kernel_execution_seconds,
                device_synchronization_seconds=synchronization_seconds,
                host_transfer_seconds=host_transfer_seconds,
                output_slicing_seconds=output_slicing_seconds,
                validation_seconds=validation_seconds,
                shard_persistence_seconds=shard_persistence_seconds,
                manifest_persistence_seconds=0.0,
                cleanup_seconds=cleanup_seconds,
                total_shard_seconds=elapsed,
                process_cpu_utilization_percent=(
                    100.0 * cpu_seconds / max(elapsed, np.finfo(float).eps)
                ),
                active_process_threads=_active_process_threads(),
                backend=jax.default_backend(),
                devices=tuple(str(device) for device in jax.devices()),
                input_shapes=(destination.shape, masks.shape),
                input_dtypes=(str(destination.dtype), str(masks.dtype)),
                output_shapes=output_shapes,
                output_dtypes=output_dtypes,
                retained_bytes=retained_bytes,
                estimated_temporary_bytes=(
                    padded_count * plan.estimated_temporary_bytes_per_group
                ),
                device_memory_stats=_device_memory_statistics(),
                graph_nodes_traversed=padded_count * routing.num_nodes,
                graph_links_traversed=graph_links_traversed,
                enabled_links=enabled_links,
                enabled_link_fraction=(
                    enabled_links / graph_links_traversed
                    if graph_links_traversed
                    else 0.0
                ),
                probability_nonzeros=probability_nonzeros,
                probability_density=(
                    probability_nonzeros / retained_domain
                    if retained_domain
                    else 0.0
                ),
                effective_probability_nonzeros=probability_nonzeros,
                effective_probability_density=(
                    probability_nonzeros / retained_domain
                    if retained_domain
                    else 0.0
                ),
                jax_execution_wait_seconds=jax_execution_wait_seconds,
                concurrent_jax_executions=concurrent_jax_executions,
                voluntary_context_switches=voluntary_context_switches,
                involuntary_context_switches=involuntary_context_switches,
            )
        return _ThreadShardResult(
            descriptor=descriptor,
            stored_bytes=path.stat().st_size,
            elapsed_seconds=elapsed,
            peak_rss_bytes=_peak_rss(),
            diagnostic=diagnostic,
        )

    status = "completed"
    deadline_phase: str | None = None
    next_emit = min(item.shard_index for item in missing)
    shard_diagnostics: list[FixedRoutingShardExecutionDiagnostics] = []

    predicted = config.initial_predicted_shard_seconds or (
        float(np.mean(shard_times[-3:])) if shard_times else None
    )
    with ThreadPoolExecutor(
        max_workers=admitted_workers,
        thread_name_prefix="fixed-routing-shard",
    ) as pool:
        exhausted = False
        while active or not exhausted:
            dispatched: list[int] = []
            while len(active) < admitted_workers and not exhausted:
                current_rss = _peak_rss()
                if current_rss is not None:
                    peak_rss = (
                        current_rss
                        if peak_rss is None
                        else max(peak_rss, current_rss)
                    )
                if (
                    config.maximum_process_rss_bytes is not None
                    and current_rss is not None
                    and current_rss
                    + (len(active) + 1) * estimated_worker_peak
                    + max(256 * 1024 * 1024, int(config.maximum_process_rss_bytes * 0.1))
                    > config.maximum_process_rss_bytes
                ):
                    if active:
                        break
                    status = "memory_budget_reached"
                    exhausted = True
                    break
                if (
                    config.maximum_cache_bytes is not None
                    and cache_bytes
                    + (len(active) + 1)
                    * plan.groups_per_full_shard
                    * plan.retained_bytes_per_group
                    > config.maximum_cache_bytes
                ):
                    if active:
                        break
                    status = "memory_budget_reached"
                    exhausted = True
                    break
                if (
                    absolute_deadline is not None
                    and predicted is not None
                    and absolute_deadline - clock()
                    < predicted + config.dispatch_safety_margin_seconds
                ):
                    dispatch_prevented = True
                    status = "deadline_reached"
                    deadline_phase = "predictive dispatch guard"
                    exhausted = True
                    break
                if submitted_shards >= len(missing):
                    exhausted = True
                    break
                descriptor = missing[submitted_shards]
                active[pool.submit(construct, descriptor)] = descriptor
                submitted_shards += 1
                dispatched.append(descriptor.shard_index)
            if dispatched:
                emit_progress(
                    phase="dispatch",
                    status_value="started",
                    predicted_seconds=predicted,
                )
            if dispatch_prevented and not dispatch_guard_emitted:
                emit_progress(
                    phase="predictive_dispatch_guard",
                    status_value="deadline_reached",
                    predicted_seconds=predicted,
                )
                dispatch_guard_emitted = True
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                descriptor = active.pop(future)
                try:
                    result = future.result()
                except BaseException as error:
                    failed_shards += 1
                    status = "interrupted"
                    for pending in active:
                        pending.cancel()
                    emit_progress(
                        phase="worker_failed",
                        status_value="interrupted",
                        shard_index=descriptor.shard_index,
                        predicted_seconds=predicted,
                    )
                    _write_manifest(
                        routing=routing,
                        completed=completed,
                        status=status,
                        elapsed_seconds=clock() - started,
                        durable=config.durable_progress,
                    )
                    emit_progress(
                        phase="terminal",
                        status_value="interrupted",
                        shard_index=descriptor.shard_index,
                        predicted_seconds=predicted,
                    )
                    raise RuntimeError(
                        "routing worker failed for shard "
                        f"{descriptor.shard_index}."
                    ) from error
                buffered[result.descriptor.shard_index] = result
                # Preserve the dispatch/deadline predictor's historical
                # behavior: completed execution observations may guide the
                # next admission even before canonical persistence catches
                # up.  ETA fields are computed separately from manifest-backed
                # durations below and therefore remain completed-only.
                shard_times.append(result.elapsed_seconds)
                predicted = float(np.mean(shard_times[-3:]))
                if result.peak_rss_bytes is not None:
                    peak_rss = (
                        result.peak_rss_bytes
                        if peak_rss is None
                        else max(peak_rss, result.peak_rss_bytes)
                    )
                if absolute_deadline is not None and clock() >= absolute_deadline:
                    status = "deadline_reached"
                    deadline_phase = "active shard execution"
                    exhausted = True
            emit_progress(
                phase="shard_completed",
                status_value="buffered",
                predicted_seconds=predicted,
            )
            while next_emit in buffered:
                result = buffered.pop(next_emit)
                completed.append(next_emit)
                cache_bytes += result.stored_bytes
                shard_duration_map[result.descriptor.shard_index] = (
                    result.elapsed_seconds
                )
                phase_started = clock()
                _write_manifest(
                    routing=routing,
                    completed=completed,
                    status="in_progress",
                    elapsed_seconds=clock() - started,
                    durable=config.durable_progress,
                    shard_durations=shard_duration_map,
                )
                manifest_seconds = clock() - phase_started
                if result.diagnostic is not None:
                    shard_diagnostics.append(
                        replace(
                            result.diagnostic,
                            manifest_persistence_seconds=manifest_seconds,
                            total_shard_seconds=(
                                result.diagnostic.total_shard_seconds
                                + manifest_seconds
                            ),
                        )
                    )
                emit_progress(
                    phase="shard_persisted",
                    status_value="completed",
                    shard_index=result.descriptor.shard_index,
                    recent_seconds=result.elapsed_seconds + manifest_seconds,
                    predicted_seconds=predicted,
                )
                next_emit += 1
                while next_emit in completed:
                    next_emit += 1

    elapsed = clock() - started
    _write_manifest(
        routing=routing,
        completed=completed,
        status=status,
        elapsed_seconds=elapsed,
        durable=config.durable_progress,
    )
    emit_progress(
        phase="terminal",
        status_value=status,
        predicted_seconds=predicted,
    )
    overshoot = (
        max(0.0, clock() - absolute_deadline)
        if absolute_deadline is not None and deadline_phase is not None
        else 0.0
    )
    return ShardedFixedRoutingPreparationResult(
        routing=routing,
        plan=plan,
        status=status,
        completed_shards=len(completed),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        reconstructed_shards=reconstructed,
        elapsed_seconds=elapsed,
        retained_cache_bytes=cache_bytes,
        peak_rss_bytes=peak_rss,
        deadline_phase=deadline_phase,
        indivisible_operation_overshoot=overshoot > 0.0,
        deadline_overshoot_seconds=overshoot,
        compilation_count=1,
        tracing_seconds=tracing,
        lowering_seconds=lowering,
        compilation_seconds=compilation,
        shard_diagnostics=tuple(
            sorted(shard_diagnostics, key=lambda item: item.shard_index)
        ),
        predicted_next_shard_seconds=predicted,
        dispatch_prevented_by_deadline=dispatch_prevented,
        admitted_worker_count=admitted_workers,
        memory_limited_worker_count=memory_limited_workers,
        cpu_limited_worker_count=cpu_limited_workers,
        estimated_worker_peak_bytes=estimated_worker_peak,
        worker_architecture="shared-executable-thread-pool",
        effective_threads_per_worker=None,
        compiled_kernel_identity=compiled_kernel_identity,
        persistent_compilation_cache_enabled=persistent_cache_enabled,
        persistent_compilation_cache_directory=persistent_cache_directory,
    )


def _padded_batch_inputs(
    *,
    inputs: AssignmentInputs,
    descriptors: tuple[FixedRoutingShardDescriptor, ...],
    padded_batch_size: int,
    groups_per_shard: int,
    num_links: int,
) -> AssignmentInputs:
    """Build one bounded fixed-shape input containing independent shards."""

    total_groups = padded_batch_size * groups_per_shard
    destination = np.zeros(total_groups, dtype=np.int32)
    masks = np.zeros((total_groups, num_links), dtype=bool)
    for slot, descriptor in enumerate(descriptors):
        start = slot * groups_per_shard
        stop = start + descriptor.num_groups
        destination[start:stop] = np.asarray(
            inputs.group_dest_node[descriptor.group_start : descriptor.group_stop]
        )
        masks[start:stop] = np.asarray(
            inputs.group_link_mask[descriptor.group_start : descriptor.group_stop]
        )
    destination_device = jnp.asarray(destination)
    masks_device = jnp.asarray(masks)
    jax.block_until_ready((destination_device, masks_device))
    return replace(
        inputs,
        group_dest_node=destination_device,
        group_link_mask=masks_device,
    )


def _prepare_fixed_routing_sharded_batched(
    *,
    inputs: AssignmentInputs,
    theta: float,
    config: FixedRoutingPreparationConfig,
    absolute_deadline: float | None,
    progress: ProgressCallback | None,
    cache_policy: CachePolicy,
    clock: Callable[[], float],
    started: float,
    plan: FixedRoutingShardPlan,
    routing: ShardedFixedRoutingInputs,
    compiled_kernel_identity: str,
    persistent_cache_enabled: bool,
    persistent_cache_directory: str | None,
) -> ShardedFixedRoutingPreparationResult:
    """Evaluate a bounded group of independent routing shards per invocation."""

    completed: list[int] = []
    missing: list[FixedRoutingShardDescriptor] = []
    durations = _manifest_shard_durations(routing)
    cache_hits = cache_misses = reconstructed = cache_bytes = 0
    peak_rss = _peak_rss()
    for descriptor in plan.descriptors:
        path = fixed_routing_shard_path(routing, descriptor)
        if path.exists() and cache_policy == "reuse":
            cached = _reuse_or_quarantine_fixed_routing_shard(
                routing=routing, descriptor=descriptor
            )
            if cached is not None:
                completed.append(descriptor.shard_index)
                cache_hits += 1
                cache_bytes += path.stat().st_size
                continue
            reconstructed += 1
        cache_misses += 1
        reconstructed += int(path.exists() and cache_policy != "reuse")
        missing.append(descriptor)

    padded_batch_size = config.shards_per_execution_batch
    estimated_batch_temporary = (
        padded_batch_size
        * plan.groups_per_full_shard
        * plan.estimated_temporary_bytes_per_group
    )
    tracing = lowering = compilation = 0.0
    compilation_count = 0
    compiled = None
    batch_diagnostics: list[FixedRoutingBatchExecutionDiagnostics] = []
    predicted_batch_seconds = (
        None
        if config.initial_predicted_shard_seconds is None
        else config.initial_predicted_shard_seconds * padded_batch_size
    )
    dispatch_prevented = False
    status = "completed"
    deadline_phase = None
    failed_shards = 0

    def emit(
        *,
        phase: str,
        batch: tuple[FixedRoutingShardDescriptor, ...] = (),
        buffered: int = 0,
        execution_seconds: float | None = None,
        persistence_seconds: float | None = None,
        effective_cores: float | None = None,
        shard_index: int | None = None,
    ) -> None:
        if progress is None:
            return
        active_indices = tuple(item.shard_index for item in batch)
        completed_set = set(completed)
        active_set = set(active_indices)
        queued = sum(
            descriptor.shard_index not in completed_set | active_set
            for descriptor in missing
        )
        accounted = (
            len(completed) + len(active_indices) + buffered + queued + failed_shards
        )
        # Finished batch outputs are either active or buffered, never both.
        if buffered:
            accounted -= len(active_indices)
        if accounted != routing.num_shards:
            raise RuntimeError(
                "invalid batched fixed-routing progress accounting: "
                f"{accounted} != {routing.num_shards}."
            )
        elapsed = max(0.0, clock() - started)
        completed_durations = [
            durations[index] for index in completed if index in durations
        ]
        eta = estimate_completed_unit_eta(
            completed_durations,
            completed_units=len(completed),
            total_units=routing.num_shards,
            parallelism=1,
        )
        progress(
            FixedRoutingShardProgress(
                phase=phase,
                status=status,
                completed_groups=sum(
                    plan.descriptors[index].num_groups for index in completed
                ),
                total_groups=routing.num_destination_groups,
                completed_shards=len(completed),
                total_shards=routing.num_shards,
                shard_index=shard_index,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                elapsed_seconds=elapsed,
                recent_shard_seconds=None,
                estimated_remaining_seconds=eta.predicted_remaining_seconds,
                peak_rss_bytes=peak_rss,
                retained_cache_bytes=cache_bytes,
                deadline_remaining_seconds=(
                    None
                    if absolute_deadline is None
                    else max(0.0, absolute_deadline - clock())
                ),
                active_workers=int(bool(batch) and not buffered),
                buffered_shards=buffered,
                queued_shards=queued,
                remaining_shards=routing.num_shards - len(completed),
                failed_shards=failed_shards,
                current_shard_indices=(active_indices if not buffered else ()),
                peak_parent_rss_bytes=peak_rss,
                estimated_total_worker_rss_bytes=estimated_batch_temporary,
                predicted_next_shard_seconds=(
                    None
                    if predicted_batch_seconds is None
                    else predicted_batch_seconds / padded_batch_size
                ),
                dispatch_prevented_by_deadline=dispatch_prevented,
                configured_rss_ceiling=config.maximum_process_rss_bytes,
                estimated_worker_peak_bytes=estimated_batch_temporary,
                admitted_worker_count=1,
                worker_architecture="single-device-batched",
                batch_shard_indices=active_indices,
                batch_size=len(batch),
                padded_batch_size=padded_batch_size,
                batch_execution_seconds=execution_seconds,
                batch_persistence_seconds=persistence_seconds,
                aggregate_shards_per_second=(
                    None
                    if execution_seconds is None or execution_seconds <= 0.0
                    else len(batch) / execution_seconds
                ),
                predicted_next_batch_seconds=predicted_batch_seconds,
                effective_cpu_cores=effective_cores,
                eta_confidence=eta.eta_confidence,
                estimated_completion_at_utc=eta.estimated_completion_at_utc,
                eta_reason=eta.eta_reason,
            )
        )

    if not missing:
        emit(phase="planning_cache_scan")
    elif estimated_batch_temporary > config.maximum_temporary_bytes:
        status = "memory_budget_reached"
        emit(phase="batch_memory_admission")
    else:
        emit(phase="planning_cache_scan")

    theta_array = jnp.asarray(theta, dtype=inputs.base_link_cost.dtype).reshape(())
    for batch_index, first in enumerate(
        range(0, len(missing), padded_batch_size)
    ):
        if status != "completed":
            break
        batch = tuple(missing[first : first + padded_batch_size])
        if absolute_deadline is not None:
            allowance = absolute_deadline - clock()
            required = (
                0.0
                if predicted_batch_seconds is None
                else predicted_batch_seconds + config.dispatch_safety_margin_seconds
            )
            if allowance <= 0.0 or (
                predicted_batch_seconds is not None and allowance < required
            ):
                status = "deadline_reached"
                deadline_phase = "predictive batch dispatch guard"
                dispatch_prevented = True
                emit(phase="predictive_batch_dispatch_guard")
                break
        rss = _peak_rss()
        if rss is not None:
            peak_rss = rss if peak_rss is None else max(peak_rss, rss)
        if (
            config.maximum_process_rss_bytes is not None
            and rss is not None
            and rss + estimated_batch_temporary
            > config.maximum_process_rss_bytes
        ):
            status = "memory_budget_reached"
            emit(phase="batch_memory_admission")
            break
        emit(phase="batch_dispatch", batch=batch)
        batch_inputs = _padded_batch_inputs(
            inputs=inputs,
            descriptors=batch,
            padded_batch_size=padded_batch_size,
            groups_per_shard=plan.groups_per_full_shard,
            num_links=routing.num_links,
        )
        if compiled is None:
            phase_started = clock()
            emit(phase="trace", batch=batch)
            traced = _prepare_fixed_routing_core.trace(
                inputs=batch_inputs, theta=theta_array
            )
            tracing = clock() - phase_started
            emit(phase="trace", batch=batch, execution_seconds=tracing)
            phase_started = clock()
            emit(phase="lowering", batch=batch)
            lowered = traced.lower()
            lowering = clock() - phase_started
            emit(phase="lowering", batch=batch, execution_seconds=lowering)
            phase_started = clock()
            emit(phase="compilation", batch=batch)
            compiled = lowered.compile()
            compilation = clock() - phase_started
            emit(phase="compilation", batch=batch, execution_seconds=compilation)
            compilation_count = 1

        batch_started = clock()
        cpu_started = process_time()
        switches_started = _process_context_switches()
        try:
            effective, probability = compiled(
                inputs=batch_inputs, theta=theta_array
            )
            jax.block_until_ready((effective, probability))
        except Exception:
            failed_shards = len(batch)
            status = "batch_failed"
            emit(phase="batch_failed", batch=batch)
            _write_manifest(
                routing=routing,
                completed=completed,
                status=status,
                elapsed_seconds=clock() - started,
                durable=config.durable_progress,
                shard_durations=durations,
            )
            raise
        execution_seconds = clock() - batch_started
        cpu_seconds = max(0.0, process_time() - cpu_started)
        effective_cores = cpu_seconds / max(
            execution_seconds, np.finfo(float).eps
        )
        effective_host = np.asarray(effective)
        probability_host = np.asarray(probability)
        emit(
            phase="batch_completed",
            batch=batch,
            buffered=len(batch),
            execution_seconds=execution_seconds,
            effective_cores=effective_cores,
        )
        persistence_started = clock()
        for slot, descriptor in enumerate(batch):
            row_start = slot * plan.groups_per_full_shard
            row_stop = row_start + descriptor.num_groups
            shard = FixedRoutingShard(
                descriptor=descriptor,
                effective_group_link_mask=effective_host[row_start:row_stop],
                group_link_probability=probability_host[row_start:row_stop],
            )
            if (
                config.maximum_cache_bytes is not None
                and cache_bytes + shard.retained_bytes > config.maximum_cache_bytes
            ):
                status = "memory_budget_reached"
                break
            shard_started = clock()
            path = save_fixed_routing_shard(
                routing=routing,
                shard=shard,
                durable=config.durable_progress,
            )
            cache_bytes += path.stat().st_size
            completed.append(descriptor.shard_index)
            durations[descriptor.shard_index] = clock() - shard_started
            _write_manifest(
                routing=routing,
                completed=completed,
                status="in_progress",
                elapsed_seconds=clock() - started,
                durable=config.durable_progress,
                shard_durations=durations,
            )
            emit(
                phase="shard_persisted",
                batch=tuple(batch[slot + 1 :]),
                buffered=len(batch) - slot - 1,
                execution_seconds=execution_seconds,
                persistence_seconds=clock() - persistence_started,
                effective_cores=effective_cores,
                shard_index=descriptor.shard_index,
            )
        persistence_seconds = clock() - persistence_started
        switches_finished = _process_context_switches()
        voluntary = involuntary = None
        if switches_started is not None and switches_finished is not None:
            voluntary = max(0, switches_finished[0] - switches_started[0])
            involuntary = max(0, switches_finished[1] - switches_started[1])
        wall_seconds = clock() - batch_started
        batch_diagnostics.append(
            FixedRoutingBatchExecutionDiagnostics(
                batch_index=batch_index,
                shard_indices=tuple(item.shard_index for item in batch),
                batch_size=len(batch),
                padded_batch_size=padded_batch_size,
                execution_seconds=execution_seconds,
                persistence_seconds=persistence_seconds,
                wall_seconds=wall_seconds,
                process_cpu_seconds=cpu_seconds,
                effective_cpu_cores=effective_cores,
                shards_per_second=len(batch) / max(
                    wall_seconds, np.finfo(float).eps
                ),
                peak_rss_bytes=_peak_rss(),
                voluntary_context_switches=voluntary,
                involuntary_context_switches=involuntary,
            )
        )
        predicted_batch_seconds = float(
            np.mean(
                [item.execution_seconds for item in batch_diagnostics[-3:]]
            )
        )
        if status != "completed":
            break

    elapsed = clock() - started
    _write_manifest(
        routing=routing,
        completed=completed,
        status=status,
        elapsed_seconds=elapsed,
        durable=config.durable_progress,
        shard_durations=durations,
    )
    emit(phase="terminal")
    overshoot = (
        0.0
        if absolute_deadline is None or deadline_phase is None
        else max(0.0, clock() - absolute_deadline)
    )
    return ShardedFixedRoutingPreparationResult(
        routing=routing,
        plan=plan,
        status=status,
        completed_shards=len(completed),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        reconstructed_shards=reconstructed,
        elapsed_seconds=elapsed,
        retained_cache_bytes=cache_bytes,
        peak_rss_bytes=peak_rss,
        deadline_phase=deadline_phase,
        indivisible_operation_overshoot=bool(overshoot),
        deadline_overshoot_seconds=overshoot,
        compilation_count=compilation_count,
        tracing_seconds=tracing,
        lowering_seconds=lowering,
        compilation_seconds=compilation,
        predicted_next_shard_seconds=(
            None
            if predicted_batch_seconds is None
            else predicted_batch_seconds / padded_batch_size
        ),
        dispatch_prevented_by_deadline=dispatch_prevented,
        admitted_worker_count=1,
        estimated_worker_peak_bytes=estimated_batch_temporary,
        worker_architecture="single-device-batched",
        compiled_kernel_identity=compiled_kernel_identity,
        persistent_compilation_cache_enabled=persistent_cache_enabled,
        persistent_compilation_cache_directory=persistent_cache_directory,
        execution_strategy="batched",
        shards_per_execution_batch=padded_batch_size,
        batch_diagnostics=tuple(batch_diagnostics),
    )


def prepare_fixed_routing_sharded(
    *,
    inputs: AssignmentInputs,
    theta: float,
    config: FixedRoutingPreparationConfig,
    absolute_deadline: float | None = None,
    progress: ProgressCallback | None = None,
    cache_policy: CachePolicy = "reuse",
    clock: Callable[[], float] = perf_counter,
) -> ShardedFixedRoutingPreparationResult:
    """Prepare fixed-shape routing shards with atomic resume at boundaries."""
    if cache_policy not in ("reuse", "refresh"):
        raise ValueError("cache_policy must be 'reuse' or 'refresh'.")
    compilation_cache = configure_jax_compilation_cache(
        config.jax_compilation_cache_directory
    )
    started = clock()
    if absolute_deadline is None and config.maximum_elapsed_seconds is not None:
        absolute_deadline = started + config.maximum_elapsed_seconds
    plan = plan_fixed_routing_shards(inputs=inputs, config=config)
    routing = build_sharded_fixed_routing_inputs(
        inputs=inputs,
        theta=theta,
        shard_partition=plan.descriptors,
        cache_directory=config.cache_directory,
        checkpoint_directory=config.checkpoint_directory,
    )
    compiled_identity_payload = {
        "implementation_version": SHARDED_FIXED_ROUTING_IMPLEMENTATION_VERSION,
        "graph_fingerprint": routing.graph_fingerprint,
        "groups_per_full_shard": plan.groups_per_full_shard,
        "num_nodes": routing.num_nodes,
        "num_links": routing.num_links,
        "dtype": routing.probability_dtype,
        "backend": jax.default_backend(),
    }
    if config.execution_strategy == "batched":
        compiled_identity_payload.update(
            {
                "execution_strategy": "batched",
                "shards_per_execution_batch": config.shards_per_execution_batch,
                "compiled_groups": plan.groups_per_full_shard
                * config.shards_per_execution_batch,
            }
        )
    compiled_kernel_identity = fingerprint(compiled_identity_payload)
    _validate_existing_manifest(routing=routing, cache_policy=cache_policy)
    persisted = _manifest_completed_shards(routing, cache_policy=cache_policy)
    persisted_durations = _manifest_shard_durations(routing)
    initial_eta = estimate_completed_unit_eta(
        (persisted_durations.get(index, 0.0) for index in persisted),
        completed_units=len(persisted),
        total_units=routing.num_shards,
        parallelism=(
            1
            if config.execution_strategy == "batched"
            else max(1, config.construction_workers)
        ),
    )
    heartbeat: _RoutingProgressHeartbeat | None = None
    branch_progress = progress
    if progress is not None:
        initial_admitted_workers = (
            1
            if config.execution_strategy == "batched"
            else max(
                1,
                min(
                    config.construction_workers,
                    max(1, routing.num_shards - len(persisted)),
                ),
            )
        )
        initial = FixedRoutingShardProgress(
            phase="planning",
            status="started",
            completed_groups=sum(
                plan.descriptors[index].num_groups for index in persisted
            ),
            total_groups=routing.num_destination_groups,
            completed_shards=len(persisted),
            total_shards=routing.num_shards,
            shard_index=None,
            cache_hits=len(persisted),
            cache_misses=max(0, routing.num_shards - len(persisted)),
            elapsed_seconds=max(0.0, clock() - started),
            recent_shard_seconds=None,
            estimated_remaining_seconds=initial_eta.predicted_remaining_seconds,
            peak_rss_bytes=_peak_rss(),
            retained_cache_bytes=0,
            deadline_remaining_seconds=(
                None
                if absolute_deadline is None
                else max(0.0, absolute_deadline - clock())
            ),
            queued_shards=max(0, routing.num_shards - len(persisted)),
            remaining_shards=max(0, routing.num_shards - len(persisted)),
            admitted_worker_count=initial_admitted_workers,
            eta_confidence=initial_eta.eta_confidence,
            estimated_completion_at_utc=initial_eta.estimated_completion_at_utc,
            eta_reason=initial_eta.eta_reason,
        )
        progress(initial)
        heartbeat = _RoutingProgressHeartbeat(
            progress, initial, config.progress_interval_seconds
        )
        heartbeat.start()
        branch_progress = heartbeat.emit
    if config.execution_strategy == "batched":
        try:
            return _prepare_fixed_routing_sharded_batched(
                inputs=inputs,
                theta=theta,
                config=config,
                absolute_deadline=absolute_deadline,
                progress=branch_progress,
                cache_policy=cache_policy,
                clock=clock,
                started=started,
                plan=plan,
                routing=routing,
                compiled_kernel_identity=compiled_kernel_identity,
                persistent_cache_enabled=compilation_cache.enabled,
                persistent_cache_directory=compilation_cache.directory,
            )
        finally:
            if heartbeat is not None:
                heartbeat.close()
    if config.construction_workers > 1:
        try:
            return _prepare_fixed_routing_sharded_threads(
                inputs=inputs,
                theta=theta,
                config=config,
                absolute_deadline=absolute_deadline,
                progress=branch_progress,
                cache_policy=cache_policy,
                clock=clock,
                started=started,
                plan=plan,
                routing=routing,
                compiled_kernel_identity=compiled_kernel_identity,
                persistent_cache_enabled=compilation_cache.enabled,
                persistent_cache_directory=compilation_cache.directory,
            )
        finally:
            if heartbeat is not None:
                heartbeat.close()
    completed: list[int] = []
    hits = misses = reconstructed = cache_bytes = 0
    peak_rss = _peak_rss()
    compiled = None
    tracing = lowering = compilation = 0.0
    compilation_count = 0
    deadline_phase = None
    indivisible_overshoot = False
    status = "completed"
    shard_diagnostics: list[FixedRoutingShardExecutionDiagnostics] = []
    shard_duration_map = _manifest_shard_durations(routing)
    shard_times: list[float] = list(shard_duration_map.values())
    dispatch_prevented_by_deadline = False

    def emit(phase: str, state: str, descriptor: FixedRoutingShardDescriptor | None, recent: float | None) -> None:
        if progress is None:
            return
        elapsed = max(0.0, clock() - started)
        rolling_mean = float(np.mean(shard_times[-5:])) if shard_times else None
        eta = estimate_completed_unit_eta(
            shard_times,
            completed_units=len(completed),
            total_units=routing.num_shards,
            parallelism=1,
        )
        progress(
            FixedRoutingShardProgress(
                phase=phase,
                status=state,
                completed_groups=sum(plan.descriptors[index].num_groups for index in completed),
                total_groups=routing.num_destination_groups,
                completed_shards=len(completed),
                total_shards=routing.num_shards,
                shard_index=None if descriptor is None else descriptor.shard_index,
                cache_hits=hits,
                cache_misses=misses,
                elapsed_seconds=elapsed,
                recent_shard_seconds=recent,
                estimated_remaining_seconds=eta.predicted_remaining_seconds,
                peak_rss_bytes=peak_rss,
                retained_cache_bytes=cache_bytes,
                deadline_remaining_seconds=(
                    None if absolute_deadline is None else max(0.0, absolute_deadline - clock())
                ),
                active_workers=0,
                buffered_shards=0,
                queued_shards=max(0, routing.num_shards - len(completed)),
                remaining_shards=routing.num_shards - len(completed),
                current_shard_indices=(),
                rolling_mean_shard_seconds=rolling_mean,
                peak_parent_rss_bytes=peak_rss,
                predicted_next_shard_seconds=rolling_mean,
                dispatch_prevented_by_deadline=dispatch_prevented_by_deadline,
                eta_confidence=eta.eta_confidence,
                estimated_completion_at_utc=eta.estimated_completion_at_utc,
                eta_reason=eta.eta_reason,
            )
        )

    emit("planning", "started", None, None)
    theta_array = jnp.asarray(theta, dtype=inputs.base_link_cost.dtype).reshape(())
    for descriptor in plan.descriptors:
        if absolute_deadline is not None and clock() >= absolute_deadline:
            status, deadline_phase = "deadline_reached", "before shard"
            break
        predicted = float(np.mean(shard_times[-3:])) if shard_times else None
        if (
            absolute_deadline is not None
            and predicted is not None
            and absolute_deadline - clock()
            < predicted + config.dispatch_safety_margin_seconds
        ):
            status, deadline_phase = "deadline_reached", "predictive dispatch guard"
            dispatch_prevented_by_deadline = True
            break
        rss = _peak_rss()
        if rss is not None:
            peak_rss = rss if peak_rss is None else max(peak_rss, rss)
        if config.maximum_process_rss_bytes is not None and rss is not None and rss >= config.maximum_process_rss_bytes:
            status = "memory_budget_reached"
            break
        path = fixed_routing_shard_path(routing, descriptor)
        if path.exists() and cache_policy == "reuse":
            shard = _reuse_or_quarantine_fixed_routing_shard(
                routing=routing, descriptor=descriptor
            )
            if shard is not None:
                hits += 1
                cache_bytes += path.stat().st_size
                completed.append(descriptor.shard_index)
                emit("cache_load", "cache_hit", descriptor, 0.0)
                del shard
                continue
            reconstructed += 1
        misses += 1
        if path.exists():
            reconstructed += 1
        shard_started = clock()
        cpu_started = process_time()
        count = descriptor.num_groups
        padded_count = plan.groups_per_full_shard
        phase_started = clock()
        destination = np.zeros(padded_count, dtype=np.int32)
        destination[:count] = np.asarray(inputs.group_dest_node[descriptor.group_start:descriptor.group_stop])
        host_destination_seconds = clock() - phase_started
        phase_started = clock()
        masks = np.zeros((padded_count, routing.num_links), dtype=bool)
        masks[:count] = np.asarray(inputs.group_link_mask[descriptor.group_start:descriptor.group_stop])
        host_mask_seconds = clock() - phase_started
        phase_started = clock()
        destination_device = jnp.asarray(destination)
        masks_device = jnp.asarray(masks)
        jax.block_until_ready((destination_device, masks_device))
        argument_transfer_seconds = clock() - phase_started
        shard_inputs = replace(
            inputs,
            group_dest_node=destination_device,
            group_link_mask=masks_device,
        )
        if compiled is None:
            emit("trace", "started", descriptor, None)
            phase_started = clock()
            traced = _prepare_fixed_routing_core.trace(inputs=shard_inputs, theta=theta_array)
            tracing += clock() - phase_started
            emit("trace", "completed", descriptor, tracing)
            if absolute_deadline is not None and clock() >= absolute_deadline:
                status, deadline_phase, indivisible_overshoot = "deadline_reached", "tracing", True
                break
            emit("lowering", "started", descriptor, None)
            phase_started = clock()
            lowered = traced.lower()
            lowering += clock() - phase_started
            emit("lowering", "completed", descriptor, lowering)
            if absolute_deadline is not None and clock() >= absolute_deadline:
                status, deadline_phase, indivisible_overshoot = "deadline_reached", "lowering", True
                break
            emit("compilation", "started", descriptor, None)
            phase_started = clock()
            compiled = lowered.compile()
            compilation += clock() - phase_started
            compilation_count += 1
            emit("compilation", "completed", descriptor, compilation)
            if absolute_deadline is not None and clock() >= absolute_deadline:
                status, deadline_phase, indivisible_overshoot = "deadline_reached", "compilation", True
                break
        emit("batch_execution", "started", descriptor, None)
        phase_started = clock()
        effective, probability = compiled(inputs=shard_inputs, theta=theta_array)
        kernel_execution_seconds = clock() - phase_started
        emit(
            "batch_execution", "completed", descriptor, kernel_execution_seconds
        )
        phase_started = clock()
        jax.block_until_ready((effective, probability))
        synchronization_seconds = clock() - phase_started
        emit("synchronization", "completed", descriptor, synchronization_seconds)
        if absolute_deadline is not None and clock() >= absolute_deadline:
            status, deadline_phase, indivisible_overshoot = "deadline_reached", "shard execution", True
            break
        phase_started = clock()
        effective_host = np.asarray(effective)
        probability_host = np.asarray(probability)
        host_transfer_seconds = clock() - phase_started
        emit("host_transfer", "completed", descriptor, host_transfer_seconds)
        phase_started = clock()
        effective_slice = effective_host[:count]
        probability_slice = probability_host[:count]
        output_slicing_seconds = clock() - phase_started
        phase_started = clock()
        shard = FixedRoutingShard(
            descriptor=descriptor,
            effective_group_link_mask=effective_slice,
            group_link_probability=probability_slice,
        )
        validation_seconds = clock() - phase_started
        if absolute_deadline is not None and clock() >= absolute_deadline:
            status, deadline_phase = "deadline_reached", "host transfer"
            break
        predicted_cache_bytes = cache_bytes + shard.retained_bytes
        if config.maximum_cache_bytes is not None and predicted_cache_bytes > config.maximum_cache_bytes:
            status = "memory_budget_reached"
            break
        phase_started = clock()
        saved = save_fixed_routing_shard(routing=routing, shard=shard, durable=config.durable_progress)
        shard_persistence_seconds = clock() - phase_started
        if absolute_deadline is not None and clock() >= absolute_deadline:
            indivisible_overshoot = True
            deadline_phase = "shard persistence"
        cache_bytes += saved.stat().st_size
        completed.append(descriptor.shard_index)
        shard_duration_map[descriptor.shard_index] = clock() - shard_started
        phase_started = clock()
        _write_manifest(
            routing=routing,
            completed=completed,
            status="in_progress",
            elapsed_seconds=clock() - started,
            durable=config.durable_progress,
            shard_durations=shard_duration_map,
        )
        manifest_persistence_seconds = clock() - phase_started
        enabled_links = int(np.count_nonzero(effective_slice))
        probability_nonzeros = int(
            np.count_nonzero(probability_slice[effective_slice])
        )
        retained_bytes = shard.retained_bytes
        output_shapes = (effective_slice.shape, probability_slice.shape)
        output_dtypes = (str(effective_slice.dtype), str(probability_slice.dtype))
        phase_started = clock()
        del (
            shard,
            effective,
            probability,
            effective_host,
            probability_host,
            effective_slice,
            probability_slice,
            shard_inputs,
            destination_device,
            masks_device,
        )
        cleanup_seconds = clock() - phase_started
        recent = clock() - shard_started
        shard_times.append(recent)
        if config.detailed_profiling:
            cpu_seconds = max(0.0, process_time() - cpu_started)
            denominator = max(recent, np.finfo(float).eps)
            traversed_groups = padded_count
            retained_output_entries = count * routing.num_links
            graph_links_traversed = traversed_groups * routing.num_links
            shard_diagnostics.append(
                FixedRoutingShardExecutionDiagnostics(
                    shard_index=descriptor.shard_index,
                    num_groups=count,
                    host_destination_preparation_seconds=host_destination_seconds,
                    host_mask_preparation_seconds=host_mask_seconds,
                    argument_transfer_seconds=argument_transfer_seconds,
                    kernel_execution_seconds=kernel_execution_seconds,
                    device_synchronization_seconds=synchronization_seconds,
                    host_transfer_seconds=host_transfer_seconds,
                    output_slicing_seconds=output_slicing_seconds,
                    validation_seconds=validation_seconds,
                    shard_persistence_seconds=shard_persistence_seconds,
                    manifest_persistence_seconds=manifest_persistence_seconds,
                    cleanup_seconds=cleanup_seconds,
                    total_shard_seconds=recent,
                    process_cpu_utilization_percent=100.0 * cpu_seconds / denominator,
                    active_process_threads=_active_process_threads(),
                    backend=jax.default_backend(),
                    devices=tuple(str(device) for device in jax.devices()),
                    input_shapes=(destination.shape, masks.shape),
                    input_dtypes=(str(destination.dtype), str(masks.dtype)),
                    output_shapes=output_shapes,
                    output_dtypes=output_dtypes,
                    retained_bytes=retained_bytes,
                    estimated_temporary_bytes=(
                        traversed_groups * plan.estimated_temporary_bytes_per_group
                    ),
                    device_memory_stats=_device_memory_statistics(),
                    graph_nodes_traversed=traversed_groups * routing.num_nodes,
                    graph_links_traversed=graph_links_traversed,
                    enabled_links=enabled_links,
                    enabled_link_fraction=(
                        enabled_links / graph_links_traversed
                        if graph_links_traversed
                        else 0.0
                    ),
                    probability_nonzeros=probability_nonzeros,
                    probability_density=(
                        probability_nonzeros / retained_output_entries
                        if retained_output_entries
                        else 0.0
                    ),
                    effective_probability_nonzeros=probability_nonzeros,
                    effective_probability_density=(
                        probability_nonzeros / retained_output_entries
                        if retained_output_entries
                        else 0.0
                    ),
                )
            )
        emit("shard_persisted", "completed", descriptor, recent)
        if deadline_phase == "shard persistence":
            status = "deadline_reached"
            break

    elapsed = clock() - started
    _write_manifest(routing=routing, completed=completed, status=status, elapsed_seconds=elapsed, durable=config.durable_progress)
    emit("preparation", status, None, None)
    if heartbeat is not None:
        heartbeat.close()
    overshoot = max(0.0, clock() - absolute_deadline) if absolute_deadline is not None and deadline_phase else 0.0
    return ShardedFixedRoutingPreparationResult(
        routing=routing,
        plan=plan,
        status=status,
        completed_shards=len(completed),
        cache_hits=hits,
        cache_misses=misses,
        reconstructed_shards=reconstructed,
        elapsed_seconds=elapsed,
        retained_cache_bytes=cache_bytes,
        peak_rss_bytes=peak_rss,
        deadline_phase=deadline_phase,
        indivisible_operation_overshoot=indivisible_overshoot,
        deadline_overshoot_seconds=overshoot,
        compilation_count=compilation_count,
        tracing_seconds=tracing,
        lowering_seconds=lowering,
        compilation_seconds=compilation,
        shard_diagnostics=tuple(shard_diagnostics),
        predicted_next_shard_seconds=(
            float(np.mean(shard_times[-3:])) if shard_times else None
        ),
        dispatch_prevented_by_deadline=dispatch_prevented_by_deadline,
        compiled_kernel_identity=compiled_kernel_identity,
        persistent_compilation_cache_enabled=compilation_cache.enabled,
        persistent_compilation_cache_directory=compilation_cache.directory,
    )
