"""On-demand matrix-free measurement products over persisted routing shards."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
import os
from threading import RLock
from time import perf_counter, process_time
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.measurement.mapping import AggregationSpec
from public_transportation.assignment.assign import (
    _assign_fixed_routing_core,
    _assign_fixed_routing_vectorized_core,
)

from .assignment_adapter import AssignmentInputs
from .compact_od_assignment_layout import CompactODAssignmentLayout
from .fixed_routing_measurement_operator import measurement_mapping_fingerprint
from .measurement_operator_protocol import GravityOperatorCapabilities
from .sharded_fixed_routing import (
    FixedRoutingShard,
    FixedRoutingShardDescriptor,
    ShardedFixedRoutingInputs,
    load_fixed_routing_shard,
)


Operation = Literal["matvec", "matmat", "rmatvec"]
GroupExecutionStrategy = Literal["scan", "vectorized"]
ShardExecutionStrategy = Literal["aggregate", "concurrent"]


def _peak_rss() -> int | None:
    try:
        import resource
        import sys

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return None


class ShardedOperatorProductInterrupted(RuntimeError):
    """Raised only at a safe batch boundary; no partial vector is returned."""

    def __init__(
        self,
        operation: Operation,
        completed_shards: int,
        *,
        reason: str = "dispatch_prevented",
        predicted_remaining_seconds: float | None = None,
        deadline_remaining_seconds: float | None = None,
        discarded_partial_seconds: float = 0.0,
        batch_size: int = 1,
        concurrency: int = 1,
    ) -> None:
        super().__init__(
            f"{operation} operator product stopped before a shard batch; "
            f"completed partial shards={completed_shards}, result discarded"
        )
        self.operation = operation
        self.completed_shards = completed_shards
        self.reason = reason
        self.predicted_remaining_seconds = predicted_remaining_seconds
        self.deadline_remaining_seconds = deadline_remaining_seconds
        self.discarded_partial_seconds = discarded_partial_seconds
        self.batch_size = batch_size
        self.concurrency = concurrency


@dataclass(frozen=True, slots=True)
class ShardedOperatorProgress:
    phase: str
    operation: Operation
    completed_shards: int
    total_shards: int
    current_shard_indices: tuple[int, ...]
    elapsed_seconds: float
    recent_batch_seconds: float | None
    predicted_remaining_seconds: float | None
    deadline_remaining_seconds: float | None
    peak_rss_bytes: int | None
    resident_shards: int
    cache_hits: int
    cache_misses: int
    effective_cpu_cores: float | None = None
    safety_margin_seconds: float = 0.0
    discarded_partial_seconds: float = 0.0
    batch_size: int = 1
    concurrency: int = 1


@dataclass(frozen=True, slots=True)
class ShardedMatrixFreeMetrics:
    stored_bytes: int = 0
    peak_construction_bytes: int = 0
    shard_load_seconds: float = 0.0
    archive_decompression_seconds: float = 0.0
    host_to_device_seconds: float = 0.0
    dispatch_seconds: float = 0.0
    compiled_execution_seconds: float = 0.0
    synchronization_seconds: float = 0.0
    device_to_host_seconds: float = 0.0
    measurement_aggregation_seconds: float = 0.0
    cache_eviction_seconds: float = 0.0
    process_cpu_seconds: float = 0.0
    effective_cpu_cores: float = 0.0
    peak_rss_bytes: int | None = None
    resident_routing_bytes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    product_count: int = 0
    compilation_count: int = 0
    compilation_seconds: float = 0.0
    tracing_seconds: float = 0.0
    lowering_seconds: float = 0.0
    input_assembly_seconds: float = 0.0
    accumulation_seconds: float = 0.0
    group_rows_processed: int = 0
    support_entries_processed: int = 0
    inflight_routing_bytes_peak: int = 0


@dataclass(slots=True)
class ShardedMatrixFreeFixedRoutingMeasurementOperator:
    """Apply routing shards without materializing full routing or operator arrays.

    Host callbacks provide JIT-compatible JAX values. The custom VJP calls the
    explicit reverse topological loading pass, keeping the adjoint bounded by
    one routing shard and node/link work vectors.
    """

    inputs: AssignmentInputs
    routing: ShardedFixedRoutingInputs
    spec: AggregationSpec
    compact_layout: CompactODAssignmentLayout
    resident_shard_limit: int = 1
    operator_shards_per_batch: int = 1
    group_execution_strategy: GroupExecutionStrategy = "scan"
    shard_execution_strategy: ShardExecutionStrategy = "aggregate"
    operator_concurrency: int | None = None
    maximum_concurrent_routing_bytes: int = 1024 * 1024 * 1024
    progress_callback: Callable[[ShardedOperatorProgress], None] | None = field(
        default=None, repr=False
    )
    absolute_deadline: float | None = field(default=None, repr=False)
    deadline_safety_margin_seconds: float = 0.0
    initial_predicted_batch_seconds: float | None = None
    cancellation_requested: Callable[[], bool] | None = field(
        default=None, repr=False
    )
    fixed_measurement_offset: np.ndarray = field(init=False, repr=False)
    _cache: OrderedDict[int, FixedRoutingShard] = field(
        init=False, default_factory=OrderedDict, repr=False
    )
    _lock: RLock = field(init=False, default_factory=RLock, repr=False)
    _jax_forward: object = field(init=False, repr=False)
    _jax_matmat_callback: object = field(init=False, repr=False)
    _compiled_forward: object = field(init=False, repr=False)
    _forward_kernel: object = field(init=False, repr=False)
    _compiled_reverse: object = field(init=False, repr=False)
    _fidelity_compiled_forward: object = field(init=False, repr=False)
    _fidelity_compiled_reverse: object = field(init=False, repr=False)
    _compiled_matmat: dict[int, object] = field(
        init=False, default_factory=dict, repr=False
    )
    _partial_compiled_forward: dict[int, object] = field(
        init=False, default_factory=dict, repr=False
    )
    _partial_compiled_reverse: dict[int, object] = field(
        init=False, default_factory=dict, repr=False
    )
    _metrics: ShardedMatrixFreeMetrics = field(
        init=False, default_factory=ShardedMatrixFreeMetrics, repr=False
    )
    _predicted_batch_seconds: dict[str, float] = field(
        init=False, default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        if self.resident_shard_limit <= 0:
            raise ValueError("resident_shard_limit must be positive.")
        if self.operator_shards_per_batch <= 0:
            raise ValueError("operator_shards_per_batch must be positive.")
        if self.group_execution_strategy not in {"scan", "vectorized"}:
            raise ValueError("group_execution_strategy must be 'scan' or 'vectorized'.")
        if self.shard_execution_strategy not in {"aggregate", "concurrent"}:
            raise ValueError(
                "shard_execution_strategy must be 'aggregate' or 'concurrent'."
            )
        if self.operator_concurrency is not None and self.operator_concurrency <= 0:
            raise ValueError("operator_concurrency must be positive when provided.")
        if self.maximum_concurrent_routing_bytes <= 0:
            raise ValueError("maximum_concurrent_routing_bytes must be positive.")
        if self.deadline_safety_margin_seconds < 0.0:
            raise ValueError("deadline_safety_margin_seconds must be nonnegative.")
        if (
            self.initial_predicted_batch_seconds is not None
            and (
                not np.isfinite(self.initial_predicted_batch_seconds)
                or self.initial_predicted_batch_seconds <= 0.0
            )
        ):
            raise ValueError(
                "initial_predicted_batch_seconds must be positive and finite."
            )
        if self.routing.graph is not self.inputs.graph:
            raise ValueError("sharded routing was prepared for a different graph.")
        if self.routing.num_destination_groups != int(
            self.inputs.group_dest_node.shape[0]
        ):
            raise ValueError("sharded routing destination-group count mismatch.")
        if self.compact_layout.num_active != int(self.inputs.od_origin_node.shape[0]):
            raise ValueError("compact layout active dimension mismatch.")
        link = np.asarray(self.spec.link_index)
        measurement = np.asarray(self.spec.measurement_index)
        if link.shape != measurement.shape:
            raise ValueError("measurement mapping arrays must have matching shapes.")

        measurement_index = jnp.asarray(measurement, dtype=jnp.int32)
        link_index = jnp.asarray(link, dtype=jnp.int32)
        num_measurements = self.num_measurements
        graph = self.inputs.graph
        origins = self.inputs.od_origin_node

        assignment_core = (
            _assign_fixed_routing_core
            if self.group_execution_strategy == "scan"
            else _assign_fixed_routing_vectorized_core
        )

        def compiled_forward(
            active, probabilities, masks, group_od_indices, group_od_masks
        ):
            link_flow = assignment_core(
                graph=graph,
                od_values=active,
                effective_group_link_mask=masks,
                group_link_probability=probabilities,
                od_origin_node=origins,
                group_od_index_padded=group_od_indices,
                group_od_mask=group_od_masks,
            )
            result = jnp.zeros((num_measurements,), dtype=active.dtype)
            return result.at[measurement_index].add(link_flow[link_index])

        self._forward_kernel = compiled_forward
        self._compiled_forward = jax.jit(compiled_forward)
        self._fidelity_compiled_forward = jax.jit(compiled_forward)

        def compiled_reverse(
            cotangent, probabilities, masks, group_od_indices, group_od_masks
        ):
            zero = jnp.zeros(
                (self.compact_layout.num_active,),
                dtype=self.inputs.base_link_cost.dtype,
            )
            _, pullback = jax.vjp(
                lambda active: compiled_forward(
                    active,
                    probabilities,
                    masks,
                    group_od_indices,
                    group_od_masks,
                ),
                zero,
            )
            return pullback(cotangent)[0]

        self._compiled_reverse = jax.jit(compiled_reverse)
        self._fidelity_compiled_reverse = jax.jit(compiled_reverse)

        output_spec = jax.ShapeDtypeStruct(
            (self.num_measurements,), self.inputs.base_link_cost.dtype
        )
        transpose_spec = jax.ShapeDtypeStruct(
            (self.num_free_od,), self.inputs.base_link_cost.dtype
        )

        @jax.custom_vjp
        def forward(value: jax.Array) -> jax.Array:
            return jax.pure_callback(
                self._host_matvec, output_spec, value, vmap_method="sequential"
            )

        def forward_rule(value: jax.Array):
            result = jax.pure_callback(
                self._host_matvec, output_spec, value, vmap_method="sequential"
            )
            return result, None

        def reverse_rule(_, cotangent: jax.Array):
            return (
                jax.pure_callback(
                    self._host_rmatvec,
                    transpose_spec,
                    cotangent,
                    vmap_method="sequential",
                ),
            )

        forward.defvjp(forward_rule, reverse_rule)
        self._jax_forward = forward

        def matmat_callback(value: jax.Array) -> jax.Array:
            columns = value.shape[1]
            output = jax.ShapeDtypeStruct(
                (self.num_measurements, columns),
                self.inputs.base_link_cost.dtype,
            )
            return jax.pure_callback(
                self._host_matmat,
                output,
                value,
                vmap_method="sequential",
            )

        self._jax_matmat_callback = matmat_callback

        if len(self.compact_layout.fixed_compact_indices) == 0:
            offset = np.zeros(self.num_measurements, dtype=self.dtype)
        else:
            active = np.zeros(self.compact_layout.num_active, dtype=self.dtype)
            active[np.asarray(self.compact_layout.fixed_compact_indices)] = np.asarray(
                self.compact_layout.fixed_compact_values, dtype=self.dtype
            )
            offset = self._host_active_matvec(active)
        offset.setflags(write=False)
        self.fixed_measurement_offset = offset

    @property
    def shape(self) -> tuple[int, int]:
        return self.num_measurements, self.num_free_od

    @property
    def num_measurements(self) -> int:
        return self.spec.num_measurements

    @property
    def num_free_od(self) -> int:
        return self.compact_layout.num_free

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.inputs.base_link_cost.dtype)

    @property
    def theta(self) -> float:
        return self.routing.theta

    @property
    def representation(self) -> str:
        return "matrix_free_sharded"

    @property
    def product_capabilities(self) -> GravityOperatorCapabilities:
        return GravityOperatorCapabilities(
            progress=True,
            absolute_deadline=True,
            cancellation=True,
            resident_cache_diagnostics=True,
            batched_shards=True,
            concurrent_shards=True,
            matmat=True,
        )

    @property
    def is_matrix_free(self) -> bool:
        return True

    @property
    def compact_layout_fingerprint(self) -> str:
        return self.compact_layout.fingerprint

    @property
    def assignment_fingerprint(self) -> str:
        return self.routing.assignment_fingerprint

    @property
    def graph_fingerprint(self) -> str:
        return self.routing.graph_fingerprint

    @property
    def mapping_fingerprint(self) -> str:
        return measurement_mapping_fingerprint(self.spec)

    @property
    def metrics(self) -> ShardedMatrixFreeMetrics:
        return self._metrics

    @property
    def resident_shards(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def effective_operator_concurrency(self) -> int:
        if self.shard_execution_strategy == "aggregate":
            return 1
        requested = self.operator_concurrency or min(4, os.cpu_count() or 1)
        bytes_per_shard = (
            self._groups_per_shard
            * self.routing.num_links
            * (self.dtype.itemsize + np.dtype(bool).itemsize)
        )
        memory_limit = max(1, self.maximum_concurrent_routing_bytes // bytes_per_shard)
        return max(
            1,
            min(
                requested,
                os.cpu_count() or 1,
                memory_limit,
                len(self.routing.shard_partition),
            ),
        )

    def validate_routing_cache(self) -> int:
        """Read and validate every persisted shard under the residency bound."""
        for descriptor in self.routing.shard_partition:
            self._load(descriptor)
        return len(self.routing.shard_partition)

    def _load(self, descriptor: FixedRoutingShardDescriptor) -> FixedRoutingShard:
        started = perf_counter()
        with self._lock:
            cached = self._cache.pop(descriptor.shard_index, None)
            if cached is None:
                cached = load_fixed_routing_shard(
                    routing=self.routing, descriptor=descriptor
                )
                self._metrics = replace(
                    self._metrics,
                    cache_misses=self._metrics.cache_misses + 1,
                )
            else:
                self._metrics = replace(
                    self._metrics,
                    cache_hits=self._metrics.cache_hits + 1,
                )
            self._cache[descriptor.shard_index] = cached
            eviction_started = perf_counter()
            while len(self._cache) > self.resident_shard_limit:
                self._cache.popitem(last=False)
            eviction_seconds = perf_counter() - eviction_started
            self._metrics = replace(
                self._metrics,
                shard_load_seconds=self._metrics.shard_load_seconds
                + perf_counter()
                - started,
                # NPZ loading/decompression is contained in shard loading. The
                # archive API does not expose a separate boundary.
                archive_decompression_seconds=self._metrics.archive_decompression_seconds,
                cache_eviction_seconds=self._metrics.cache_eviction_seconds
                + eviction_seconds,
                resident_routing_bytes=sum(
                    shard.retained_bytes for shard in self._cache.values()
                ),
                peak_rss_bytes=_peak_rss(),
            )
            return cached

    def _measurement_from_link_flow(self, link_flow: np.ndarray) -> np.ndarray:
        result = np.zeros(self.num_measurements, dtype=self.dtype)
        np.add.at(
            result,
            np.asarray(self.spec.measurement_index),
            link_flow[np.asarray(self.spec.link_index)],
        )
        return result

    def _link_cotangent(self, measurement_cotangent: np.ndarray) -> np.ndarray:
        result = np.zeros(self.routing.num_links, dtype=self.dtype)
        np.add.at(
            result,
            np.asarray(self.spec.link_index),
            measurement_cotangent[np.asarray(self.spec.measurement_index)],
        )
        return result

    def _group_initial_flow(self, active: np.ndarray, group_index: int) -> np.ndarray:
        indices = np.asarray(self.inputs.group_od_index_padded[group_index])
        valid = np.asarray(self.inputs.group_od_mask[group_index])
        safe = indices[valid]
        origins = np.asarray(self.inputs.od_origin_node)[safe]
        initial = np.zeros(self.routing.num_nodes, dtype=self.dtype)
        np.add.at(initial, origins, active[safe])
        return initial

    def _load_group_forward(
        self, probability: np.ndarray, enabled: np.ndarray, initial: np.ndarray
    ) -> np.ndarray:
        graph = self.inputs.graph
        node_flow = initial.copy()
        link_flow = np.zeros(self.routing.num_links, dtype=self.dtype)
        head = np.asarray(graph.head)
        out_links = np.asarray(graph.out_links)
        out_mask = np.asarray(graph.out_mask)
        for node in np.asarray(graph.topo_order):
            adjacency = out_mask[node]
            links = out_links[node][adjacency]
            links = links[enabled[links]]
            flows = node_flow[node] * probability[links]
            np.add.at(link_flow, links, flows)
            np.add.at(node_flow, head[links], flows)
        return link_flow

    def _load_group_reverse(
        self, probability: np.ndarray, enabled: np.ndarray, link_weight: np.ndarray
    ) -> np.ndarray:
        graph = self.inputs.graph
        node_cotangent = np.zeros(self.routing.num_nodes, dtype=self.dtype)
        head = np.asarray(graph.head)
        out_links = np.asarray(graph.out_links)
        out_mask = np.asarray(graph.out_mask)
        for node in np.asarray(graph.topo_order)[::-1]:
            adjacency = out_mask[node]
            links = out_links[node][adjacency]
            links = links[enabled[links]]
            node_cotangent[node] += np.sum(
                probability[links] * (link_weight[links] + node_cotangent[head[links]])
            )
        return node_cotangent

    def _reference_host_active_matvec(self, active: np.ndarray) -> np.ndarray:
        active = np.asarray(active, dtype=self.dtype)
        measurement = np.zeros(self.num_measurements, dtype=self.dtype)
        for descriptor in self.routing.shard_partition:
            shard = self._load(descriptor)
            link_flow = np.zeros(self.routing.num_links, dtype=self.dtype)
            for local, group_index in enumerate(descriptor.destination_group_indices):
                initial = self._group_initial_flow(active, group_index)
                link_flow += self._load_group_forward(
                    shard.group_link_probability[local],
                    shard.effective_group_link_mask[local],
                    initial,
                )
            measurement += self._measurement_from_link_flow(link_flow)
        return measurement

    def _reference_host_matvec(self, free: np.ndarray) -> np.ndarray:
        active = np.zeros(self.compact_layout.num_active, dtype=self.dtype)
        active[np.asarray(self.compact_layout.free_compact_indices)] = np.asarray(
            free, dtype=self.dtype
        )
        return self._reference_host_active_matvec(active)

    def _reference_host_rmatvec(self, measurement_cotangent: np.ndarray) -> np.ndarray:
        link_weight = self._link_cotangent(
            np.asarray(measurement_cotangent, dtype=self.dtype)
        )
        active = np.zeros(self.compact_layout.num_active, dtype=self.dtype)
        origins_all = np.asarray(self.inputs.od_origin_node)
        for descriptor in self.routing.shard_partition:
            shard = self._load(descriptor)
            for local, group_index in enumerate(descriptor.destination_group_indices):
                node_cotangent = self._load_group_reverse(
                    shard.group_link_probability[local],
                    shard.effective_group_link_mask[local],
                    link_weight,
                )
                indices = np.asarray(self.inputs.group_od_index_padded[group_index])
                valid = np.asarray(self.inputs.group_od_mask[group_index])
                safe = indices[valid]
                active[safe] += node_cotangent[origins_all[safe]]
        return active[np.asarray(self.compact_layout.free_compact_indices)]

    @property
    def _groups_per_shard(self) -> int:
        return max((item.num_groups for item in self.routing.shard_partition), default=1)

    def _batch_arrays(self, descriptors, *, padded_shards=None):
        padded = self.operator_shards_per_batch if padded_shards is None else padded_shards
        groups = padded * self._groups_per_shard
        links = self.routing.num_links
        width = int(self.inputs.group_od_index_padded.shape[1])
        probability = np.zeros((groups, links), dtype=self.dtype)
        enabled = np.zeros((groups, links), dtype=bool)
        indices = np.zeros((groups, width), dtype=np.int32)
        valid = np.zeros((groups, width), dtype=bool)
        for slot, descriptor in enumerate(descriptors):
            shard = self._load(descriptor)
            first = slot * self._groups_per_shard
            stop = first + descriptor.num_groups
            probability[first:stop] = shard.group_link_probability
            enabled[first:stop] = shard.effective_group_link_mask
            indices[first:stop] = np.asarray(
                self.inputs.group_od_index_padded[descriptor.group_start : descriptor.group_stop]
            )
            valid[first:stop] = np.asarray(
                self.inputs.group_od_mask[descriptor.group_start : descriptor.group_stop]
            )
        return probability, enabled, indices, valid

    def _destination_group_arrays(
        self,
        destination_group_indices: tuple[int, ...],
        *,
        padded_groups: int,
        group_weights: tuple[float, ...] | None = None,
    ):
        """Load arbitrary destination groups into one fixed-shape batch."""
        if not destination_group_indices:
            raise ValueError("a partial routing batch must contain destination groups.")
        if len(set(destination_group_indices)) != len(destination_group_indices):
            raise ValueError("a partial routing batch must not duplicate groups.")
        if padded_groups < len(destination_group_indices):
            raise ValueError("padded_groups cannot be smaller than the selected groups.")
        if group_weights is None:
            group_weights = (1.0,) * len(destination_group_indices)
        if len(group_weights) != len(destination_group_indices):
            raise ValueError("group_weights must align with destination groups.")
        if any(not np.isfinite(value) or value <= 0.0 for value in group_weights):
            raise ValueError("group_weights must be positive and finite.")
        if any(
            group < 0 or group >= self.routing.num_destination_groups
            for group in destination_group_indices
        ):
            raise ValueError("destination-group index is outside the routing layout.")
        links = self.routing.num_links
        width = int(self.inputs.group_od_index_padded.shape[1])
        probability = np.zeros((padded_groups, links), dtype=self.dtype)
        enabled = np.zeros((padded_groups, links), dtype=bool)
        indices = np.zeros((padded_groups, width), dtype=np.int32)
        valid = np.zeros((padded_groups, width), dtype=bool)
        descriptors = self.routing.shard_partition
        for row, group in enumerate(destination_group_indices):
            descriptor = next(
                item
                for item in descriptors
                if item.group_start <= group < item.group_stop
            )
            shard = self._load(descriptor)
            local = group - descriptor.group_start
            probability[row] = shard.group_link_probability[local]
            enabled[row] = shard.effective_group_link_mask[local]
            indices[row] = np.asarray(self.inputs.group_od_index_padded[group])
            valid[row] = np.asarray(self.inputs.group_od_mask[group])
        return probability, enabled, indices, valid

    def _partial_active_weights(
        self,
        destination_group_indices: tuple[int, ...],
        group_weights: tuple[float, ...] | None,
    ) -> np.ndarray:
        if group_weights is None:
            group_weights = (1.0,) * len(destination_group_indices)
        if len(group_weights) != len(destination_group_indices):
            raise ValueError("group_weights must align with destination groups.")
        weights = np.ones(self.compact_layout.num_active, dtype=self.dtype)
        seen: set[int] = set()
        for group, weight in zip(
            destination_group_indices, group_weights, strict=True
        ):
            indices = np.asarray(self.inputs.group_od_index_padded[group])
            valid = np.asarray(self.inputs.group_od_mask[group])
            active_indices = indices[valid]
            overlap = seen.intersection(int(item) for item in active_indices)
            if overlap:
                raise ValueError("destination groups overlap active OD cells.")
            seen.update(int(item) for item in active_indices)
            weights[active_indices] = weight
        return weights

    def prepare_partial_batch(
        self,
        *,
        destination_group_indices: tuple[int, ...],
        padded_groups: int,
        group_weights: tuple[float, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare reusable host arrays for matched partial forward and reverse."""
        return self._destination_group_arrays(
            destination_group_indices,
            padded_groups=padded_groups,
            group_weights=group_weights,
        )

    def _partial_executable(self, operation: str, padded_groups: int):
        cache = (
            self._partial_compiled_forward
            if operation == "matvec"
            else self._partial_compiled_reverse
        )
        with self._lock:
            executable = cache.get(padded_groups)
            if executable is not None:
                return executable
            if operation == "matvec":
                executable = jax.jit(self._forward_kernel)
            elif operation == "rmatvec":
                forward = self._forward_kernel

                def reverse(cotangent, probabilities, masks, indices, valid):
                    zero = jnp.zeros(
                        (self.compact_layout.num_active,),
                        dtype=self.inputs.base_link_cost.dtype,
                    )
                    _, pullback = jax.vjp(
                        lambda active: forward(
                            active, probabilities, masks, indices, valid
                        ),
                        zero,
                    )
                    return pullback(cotangent)[0]

                executable = jax.jit(reverse)
            else:
                raise ValueError(f"unsupported partial operation: {operation!r}.")
            cache[padded_groups] = executable
            return executable

    def partial_matvec(
        self,
        vector: object,
        *,
        destination_group_indices: tuple[int, ...],
        padded_groups: int,
        group_weights: tuple[float, ...] | None = None,
        prepared_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        | None = None,
    ) -> np.ndarray:
        """Apply one reusable fixed-shape destination-group forward batch."""
        value = np.asarray(vector, dtype=self.dtype)
        if value.shape != (self.num_free_od,):
            raise ValueError(f"forward vector must have shape ({self.num_free_od},).")
        active = np.zeros(self.compact_layout.num_active, dtype=self.dtype)
        active[np.asarray(self.compact_layout.free_compact_indices)] = value
        active *= self._partial_active_weights(
            destination_group_indices, group_weights
        )
        arrays = prepared_arrays or self.prepare_partial_batch(
            destination_group_indices=destination_group_indices,
            padded_groups=padded_groups,
            group_weights=group_weights,
        )
        executable = self._partial_executable("matvec", padded_groups)
        result = executable(
            jnp.asarray(active), *(jnp.asarray(item) for item in arrays)
        )
        jax.block_until_ready(result)
        return np.asarray(result)

    def partial_rmatvec(
        self,
        vector: object,
        *,
        destination_group_indices: tuple[int, ...],
        padded_groups: int,
        group_weights: tuple[float, ...] | None = None,
        prepared_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        | None = None,
    ) -> np.ndarray:
        """Apply the transpose of one fixed-shape destination-group batch."""
        value = np.asarray(vector, dtype=self.dtype)
        if value.shape != (self.num_measurements,):
            raise ValueError(
                f"transpose vector must have shape ({self.num_measurements},)."
            )
        arrays = prepared_arrays or self.prepare_partial_batch(
            destination_group_indices=destination_group_indices,
            padded_groups=padded_groups,
            group_weights=group_weights,
        )
        executable = self._partial_executable("rmatvec", padded_groups)
        active = executable(
            jnp.asarray(value), *(jnp.asarray(item) for item in arrays)
        )
        jax.block_until_ready(active)
        weighted_active = np.asarray(active) * self._partial_active_weights(
            destination_group_indices, group_weights
        )
        return weighted_active[np.asarray(self.compact_layout.free_compact_indices)]

    def _emit(
        self,
        phase,
        operation,
        completed,
        current,
        started,
        recent,
        cores=None,
        discarded=0.0,
    ):
        if self.progress_callback is None:
            return
        predicted = self._predicted_batch_seconds.get(
            operation, self.initial_predicted_batch_seconds
        )
        remaining = len(self.routing.shard_partition) - completed
        dispatch_width = (
            self.operator_shards_per_batch
            if self.shard_execution_strategy == "aggregate"
            else self.effective_operator_concurrency
        )
        self.progress_callback(
            ShardedOperatorProgress(
                phase=phase,
                operation=operation,
                completed_shards=completed,
                total_shards=len(self.routing.shard_partition),
                current_shard_indices=current,
                elapsed_seconds=perf_counter() - started,
                recent_batch_seconds=recent,
                predicted_remaining_seconds=(
                    None
                    if predicted is None
                    else np.ceil(remaining / dispatch_width) * predicted
                ),
                deadline_remaining_seconds=(
                    None
                    if self.absolute_deadline is None
                    else max(0.0, self.absolute_deadline - perf_counter())
                ),
                peak_rss_bytes=_peak_rss(),
                resident_shards=self.resident_shards,
                cache_hits=self._metrics.cache_hits,
                cache_misses=self._metrics.cache_misses,
                effective_cpu_cores=cores,
                safety_margin_seconds=self.deadline_safety_margin_seconds,
                discarded_partial_seconds=discarded,
                batch_size=(
                    self.operator_shards_per_batch
                    if self.shard_execution_strategy == "aggregate"
                    else 1
                ),
                concurrency=self.effective_operator_concurrency,
            )
        )

    def _check_dispatch(self, operation, completed, started):
        if self.cancellation_requested is not None and self.cancellation_requested():
            raise ShardedOperatorProductInterrupted(
                operation,
                completed,
                reason="cancelled",
                discarded_partial_seconds=perf_counter() - started,
                batch_size=self.operator_shards_per_batch,
            )
        if self.absolute_deadline is None:
            return
        predicted = self._predicted_batch_seconds.get(
            operation, self.initial_predicted_batch_seconds
        )
        allowance = self.absolute_deadline - perf_counter()
        remaining_shards = len(self.routing.shard_partition) - completed
        dispatch_width = (
            self.operator_shards_per_batch
            if self.shard_execution_strategy == "aggregate"
            else self.effective_operator_concurrency
        )
        remaining_batches = int(
            np.ceil(remaining_shards / dispatch_width)
        )
        predicted_remaining = (
            None if predicted is None else remaining_batches * predicted
        )
        complete_required = self.deadline_safety_margin_seconds + (
            predicted_remaining or 0.0
        )
        next_required = self.deadline_safety_margin_seconds + (predicted or 0.0)
        whole_product_infeasible = (
            predicted_remaining is not None and allowance < complete_required
        )
        next_dispatch_infeasible = allowance <= 0.0 or (
            predicted is not None and allowance < next_required
        )
        if whole_product_infeasible or next_dispatch_infeasible:
            phase = (
                "product_deadline_infeasible"
                if whole_product_infeasible
                else "deadline_prevented_dispatch"
            )
            self._emit(
                phase,
                operation,
                completed,
                (),
                started,
                predicted,
                discarded=perf_counter() - started,
            )
            raise ShardedOperatorProductInterrupted(
                operation,
                completed,
                reason=phase,
                predicted_remaining_seconds=predicted_remaining,
                deadline_remaining_seconds=max(0.0, allowance),
                discarded_partial_seconds=perf_counter() - started,
                batch_size=(
                    self.operator_shards_per_batch
                    if self.shard_execution_strategy == "aggregate"
                    else 1
                ),
                concurrency=self.effective_operator_concurrency,
            )

    def _compile(self, operation, arguments):
        executable = (
            self._compiled_forward if operation == "matvec" else self._compiled_reverse
        )
        if not hasattr(executable, "lower"):
            return executable
        tracing_started = perf_counter()
        traced = executable.trace(*arguments)
        tracing = perf_counter() - tracing_started
        lowering_started = perf_counter()
        lowered = traced.lower()
        lowering = perf_counter() - lowering_started
        compilation_started = perf_counter()
        executable = lowered.compile()
        compilation = perf_counter() - compilation_started
        if operation == "matvec":
            self._compiled_forward = executable
        else:
            self._compiled_reverse = executable
        self._metrics = replace(
            self._metrics,
            compilation_count=self._metrics.compilation_count + 1,
            compilation_seconds=self._metrics.compilation_seconds + compilation,
            tracing_seconds=self._metrics.tracing_seconds + tracing,
            lowering_seconds=self._metrics.lowering_seconds + lowering,
        )
        return executable

    def _host_active_matvec(self, active: np.ndarray) -> np.ndarray:
        active = np.asarray(active, dtype=self.dtype)
        if self.shard_execution_strategy == "concurrent":
            return self._host_concurrent_product("matvec", active)
        result = np.zeros(self.num_measurements, dtype=self.dtype)
        descriptors = self.routing.shard_partition
        started = perf_counter()
        cpu_started = process_time()
        completed = 0
        self._emit("product_started", "matvec", 0, (), started, None)
        for first in range(0, len(descriptors), self.operator_shards_per_batch):
            self._check_dispatch("matvec", completed, started)
            batch = tuple(descriptors[first : first + self.operator_shards_per_batch])
            current = tuple(item.shard_index for item in batch)
            assembly_started = perf_counter()
            arrays = self._batch_arrays(batch)
            assembly = perf_counter() - assembly_started
            support_entries = int(np.count_nonzero(arrays[1]))
            inflight_bytes = sum(item.nbytes for item in arrays)
            self._emit("batch_loaded", "matvec", completed, current, started, None)
            transfer_started = perf_counter()
            arguments = (jnp.asarray(active), *(jnp.asarray(item) for item in arrays))
            jax.block_until_ready(arguments)
            transfer = perf_counter() - transfer_started
            executable = self._compile("matvec", arguments)
            batch_started = perf_counter()
            dispatch_started = perf_counter()
            contribution = executable(*arguments)
            dispatch = perf_counter() - dispatch_started
            execution_started = perf_counter()
            jax.block_until_ready(contribution)
            execution = perf_counter() - execution_started
            sync_started = perf_counter()
            jax.block_until_ready(contribution)
            synchronization = perf_counter() - sync_started
            self._emit(
                "batch_executed",
                "matvec",
                completed,
                current,
                started,
                perf_counter() - batch_started,
            )
            host_started = perf_counter()
            host_contribution = np.asarray(contribution)
            host = perf_counter() - host_started
            accumulation_started = perf_counter()
            result += host_contribution
            accumulation = perf_counter() - accumulation_started
            recent = perf_counter() - batch_started
            self._predicted_batch_seconds["matvec"] = recent
            completed += len(batch)
            self._metrics = replace(
                self._metrics,
                host_to_device_seconds=self._metrics.host_to_device_seconds + transfer,
                dispatch_seconds=self._metrics.dispatch_seconds + dispatch,
                compiled_execution_seconds=self._metrics.compiled_execution_seconds
                + execution,
                synchronization_seconds=self._metrics.synchronization_seconds + synchronization,
                device_to_host_seconds=self._metrics.device_to_host_seconds + host,
                input_assembly_seconds=self._metrics.input_assembly_seconds + assembly,
                accumulation_seconds=self._metrics.accumulation_seconds + accumulation,
                group_rows_processed=self._metrics.group_rows_processed
                + arrays[0].shape[0],
                support_entries_processed=self._metrics.support_entries_processed
                + support_entries,
                inflight_routing_bytes_peak=max(
                    self._metrics.inflight_routing_bytes_peak, inflight_bytes
                ),
            )
            self._emit(
                "batch_accumulated",
                "matvec",
                completed,
                current,
                started,
                recent,
                (process_time() - cpu_started)
                / max(perf_counter() - started, np.finfo(float).eps),
            )
        elapsed = perf_counter() - started
        cpu = process_time() - cpu_started
        self._metrics = replace(
            self._metrics,
            process_cpu_seconds=self._metrics.process_cpu_seconds + cpu,
            effective_cpu_cores=cpu / max(elapsed, np.finfo(float).eps),
            peak_rss_bytes=_peak_rss(),
            product_count=self._metrics.product_count + 1,
        )
        self._emit("product_completed", "matvec", completed, (), started, elapsed)
        return result

    def _host_matvec(self, free: np.ndarray) -> np.ndarray:
        active = np.zeros(self.compact_layout.num_active, dtype=self.dtype)
        active[np.asarray(self.compact_layout.free_compact_indices)] = np.asarray(
            free, dtype=self.dtype
        )
        return self._host_active_matvec(active)

    def _host_rmatvec(self, measurement_cotangent: np.ndarray) -> np.ndarray:
        cotangent = np.asarray(measurement_cotangent, dtype=self.dtype)
        if self.shard_execution_strategy == "concurrent":
            active = self._host_concurrent_product("rmatvec", cotangent)
            return active[np.asarray(self.compact_layout.free_compact_indices)]
        active = np.zeros(self.compact_layout.num_active, dtype=self.dtype)
        descriptors = self.routing.shard_partition
        started = perf_counter()
        cpu_started = process_time()
        completed = 0
        self._emit("product_started", "rmatvec", 0, (), started, None)
        for first in range(0, len(descriptors), self.operator_shards_per_batch):
            self._check_dispatch("rmatvec", completed, started)
            batch = tuple(descriptors[first : first + self.operator_shards_per_batch])
            current = tuple(item.shard_index for item in batch)
            assembly_started = perf_counter()
            arrays = self._batch_arrays(batch)
            assembly = perf_counter() - assembly_started
            support_entries = int(np.count_nonzero(arrays[1]))
            inflight_bytes = sum(item.nbytes for item in arrays)
            self._emit("batch_loaded", "rmatvec", completed, current, started, None)
            transfer_started = perf_counter()
            arguments = (jnp.asarray(cotangent), *(jnp.asarray(item) for item in arrays))
            jax.block_until_ready(arguments)
            transfer = perf_counter() - transfer_started
            executable = self._compile("rmatvec", arguments)
            batch_started = perf_counter()
            dispatch_started = perf_counter()
            contribution = executable(*arguments)
            dispatch = perf_counter() - dispatch_started
            execution_started = perf_counter()
            jax.block_until_ready(contribution)
            execution = perf_counter() - execution_started
            sync_started = perf_counter()
            jax.block_until_ready(contribution)
            synchronization = perf_counter() - sync_started
            self._emit(
                "batch_executed",
                "rmatvec",
                completed,
                current,
                started,
                perf_counter() - batch_started,
            )
            host_started = perf_counter()
            host_contribution = np.asarray(contribution)
            host = perf_counter() - host_started
            accumulation_started = perf_counter()
            active += host_contribution
            accumulation = perf_counter() - accumulation_started
            recent = perf_counter() - batch_started
            self._predicted_batch_seconds["rmatvec"] = recent
            completed += len(batch)
            self._metrics = replace(
                self._metrics,
                host_to_device_seconds=self._metrics.host_to_device_seconds + transfer,
                dispatch_seconds=self._metrics.dispatch_seconds + dispatch,
                compiled_execution_seconds=self._metrics.compiled_execution_seconds
                + execution,
                synchronization_seconds=self._metrics.synchronization_seconds
                + synchronization,
                device_to_host_seconds=self._metrics.device_to_host_seconds + host,
                input_assembly_seconds=self._metrics.input_assembly_seconds + assembly,
                accumulation_seconds=self._metrics.accumulation_seconds + accumulation,
                group_rows_processed=self._metrics.group_rows_processed
                + arrays[0].shape[0],
                support_entries_processed=self._metrics.support_entries_processed
                + support_entries,
                inflight_routing_bytes_peak=max(
                    self._metrics.inflight_routing_bytes_peak, inflight_bytes
                ),
            )
            self._emit(
                "batch_accumulated",
                "rmatvec",
                completed,
                current,
                started,
                recent,
                (process_time() - cpu_started)
                / max(perf_counter() - started, np.finfo(float).eps),
            )
        elapsed = perf_counter() - started
        cpu = process_time() - cpu_started
        self._metrics = replace(
            self._metrics,
            process_cpu_seconds=self._metrics.process_cpu_seconds + cpu,
            effective_cpu_cores=cpu / max(elapsed, np.finfo(float).eps),
            peak_rss_bytes=_peak_rss(),
            product_count=self._metrics.product_count + 1,
        )
        self._emit("product_completed", "rmatvec", completed, (), started, elapsed)
        return active[np.asarray(self.compact_layout.free_compact_indices)]

    def _host_concurrent_product(
        self, operation: Operation, value: np.ndarray
    ) -> np.ndarray:
        """Dispatch independent single-shard executables with bounded concurrency."""
        descriptors = self.routing.shard_partition
        concurrency = self.effective_operator_concurrency
        if operation == "matvec":
            output_shape = (self.num_measurements,)
        elif operation == "rmatvec":
            output_shape = (self.compact_layout.num_active,)
        else:
            output_shape = (self.num_measurements, value.shape[1])
        result = np.zeros(output_shape, dtype=self.dtype)
        started = perf_counter()
        cpu_started = process_time()
        completed = 0
        self._emit("product_started", operation, 0, (), started, None)

        def execute(executable, arguments):
            dispatch_started = perf_counter()
            contribution = executable(*arguments)
            dispatch = perf_counter() - dispatch_started
            execution_started = perf_counter()
            jax.block_until_ready(contribution)
            execution = perf_counter() - execution_started
            synchronization_started = perf_counter()
            jax.block_until_ready(contribution)
            synchronization = perf_counter() - synchronization_started
            host_started = perf_counter()
            host_contribution = np.asarray(contribution)
            host = perf_counter() - host_started
            return host_contribution, dispatch, execution, synchronization, host

        with ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="routing-product",
        ) as executor:
            for first in range(0, len(descriptors), concurrency):
                self._check_dispatch(operation, completed, started)
                wave = tuple(descriptors[first : first + concurrency])
                current = tuple(item.shard_index for item in wave)
                assembly_started = perf_counter()
                arrays_by_shard = tuple(
                    self._batch_arrays((descriptor,), padded_shards=1)
                    for descriptor in wave
                )
                assembly = perf_counter() - assembly_started
                support_entries = sum(
                    int(np.count_nonzero(arrays[1])) for arrays in arrays_by_shard
                )
                inflight_bytes = sum(
                    item.nbytes
                    for arrays in arrays_by_shard
                    for item in arrays
                )
                self._emit("batch_loaded", operation, completed, current, started, None)
                transfer_started = perf_counter()
                arguments = tuple(
                    (jnp.asarray(value), *(jnp.asarray(item) for item in arrays))
                    for arrays in arrays_by_shard
                )
                jax.block_until_ready(arguments)
                transfer = perf_counter() - transfer_started
                if operation == "matmat":
                    columns = value.shape[1]
                    executable = self._compiled_matmat.get(columns)
                    if executable is None:
                        forward = self._forward_kernel
                        matmat = jax.jit(
                            lambda active_value, probability, enabled, indices, valid: jax.vmap(
                                lambda column: forward(
                                    column, probability, enabled, indices, valid
                                ),
                                in_axes=1,
                                out_axes=1,
                            )(active_value)
                        )
                        tracing_started = perf_counter()
                        traced = matmat.trace(*arguments[0])
                        tracing = perf_counter() - tracing_started
                        lowering_started = perf_counter()
                        lowered = traced.lower()
                        lowering = perf_counter() - lowering_started
                        compilation_started = perf_counter()
                        executable = lowered.compile()
                        compilation = perf_counter() - compilation_started
                        self._compiled_matmat[columns] = executable
                        self._metrics = replace(
                            self._metrics,
                            compilation_count=self._metrics.compilation_count + 1,
                            compilation_seconds=self._metrics.compilation_seconds
                            + compilation,
                            tracing_seconds=self._metrics.tracing_seconds + tracing,
                            lowering_seconds=self._metrics.lowering_seconds + lowering,
                        )
                else:
                    executable = self._compile(operation, arguments[0])
                wave_started = perf_counter()
                futures = tuple(
                    executor.submit(execute, executable, item) for item in arguments
                )
                contributions = tuple(future.result() for future in futures)
                recent = perf_counter() - wave_started
                self._emit(
                    "batch_executed",
                    operation,
                    completed,
                    current,
                    started,
                    recent,
                )
                accumulation_started = perf_counter()
                for contribution, *_ in contributions:
                    result += contribution
                accumulation = perf_counter() - accumulation_started
                self._predicted_batch_seconds[operation] = recent
                completed += len(wave)
                self._metrics = replace(
                    self._metrics,
                    host_to_device_seconds=self._metrics.host_to_device_seconds
                    + transfer,
                    dispatch_seconds=self._metrics.dispatch_seconds
                    + sum(item[1] for item in contributions),
                    compiled_execution_seconds=self._metrics.compiled_execution_seconds
                    + sum(item[2] for item in contributions),
                    synchronization_seconds=self._metrics.synchronization_seconds
                    + sum(item[3] for item in contributions),
                    device_to_host_seconds=self._metrics.device_to_host_seconds
                    + sum(item[4] for item in contributions),
                    input_assembly_seconds=self._metrics.input_assembly_seconds
                    + assembly,
                    accumulation_seconds=self._metrics.accumulation_seconds
                    + accumulation,
                    group_rows_processed=self._metrics.group_rows_processed
                    + len(wave) * self._groups_per_shard,
                    support_entries_processed=self._metrics.support_entries_processed
                    + support_entries,
                    inflight_routing_bytes_peak=max(
                        self._metrics.inflight_routing_bytes_peak, inflight_bytes
                    ),
                )
                self._emit(
                    "batch_accumulated",
                    operation,
                    completed,
                    current,
                    started,
                    recent,
                    (process_time() - cpu_started)
                    / max(perf_counter() - started, np.finfo(float).eps),
                )
        elapsed = perf_counter() - started
        cpu = process_time() - cpu_started
        self._metrics = replace(
            self._metrics,
            process_cpu_seconds=self._metrics.process_cpu_seconds + cpu,
            effective_cpu_cores=cpu / max(elapsed, np.finfo(float).eps),
            peak_rss_bytes=_peak_rss(),
            product_count=self._metrics.product_count + 1,
        )
        self._emit("product_completed", operation, completed, (), started, elapsed)
        return result

    def _host_matmat(self, matrix: np.ndarray) -> np.ndarray:
        value = np.asarray(matrix, dtype=self.dtype)
        columns = value.shape[1]
        active = np.zeros((self.compact_layout.num_active, columns), dtype=self.dtype)
        active[np.asarray(self.compact_layout.free_compact_indices)] = value
        if self.shard_execution_strategy == "concurrent":
            return self._host_concurrent_product("matmat", active)
        result = np.zeros((self.num_measurements, columns), dtype=self.dtype)
        descriptors = self.routing.shard_partition
        started = perf_counter()
        cpu_started = process_time()
        completed = 0
        self._emit("product_started", "matmat", 0, (), started, None)
        executable = self._compiled_matmat.get(columns)
        for first in range(0, len(descriptors), self.operator_shards_per_batch):
            self._check_dispatch("matmat", completed, started)
            batch = tuple(descriptors[first : first + self.operator_shards_per_batch])
            current = tuple(item.shard_index for item in batch)
            assembly_started = perf_counter()
            arrays = self._batch_arrays(batch)
            assembly = perf_counter() - assembly_started
            support_entries = int(np.count_nonzero(arrays[1]))
            inflight_bytes = sum(item.nbytes for item in arrays)
            self._emit("batch_loaded", "matmat", completed, current, started, None)
            transfer_started = perf_counter()
            arguments = (jnp.asarray(active), *(jnp.asarray(item) for item in arrays))
            jax.block_until_ready(arguments)
            transfer = perf_counter() - transfer_started
            if executable is None:
                forward = self._forward_kernel
                matmat = jax.jit(
                    lambda active_value, probability, enabled, indices, valid: jax.vmap(
                        lambda column: forward(
                            column, probability, enabled, indices, valid
                        ),
                        in_axes=1,
                        out_axes=1,
                    )(active_value)
                )
                compilation_started = perf_counter()
                executable = matmat.lower(*arguments).compile()
                self._compiled_matmat[columns] = executable
                self._metrics = replace(
                    self._metrics,
                    compilation_count=self._metrics.compilation_count + 1,
                    compilation_seconds=self._metrics.compilation_seconds
                    + perf_counter()
                    - compilation_started,
                )
            batch_started = perf_counter()
            dispatch_started = perf_counter()
            contribution = executable(*arguments)
            dispatch = perf_counter() - dispatch_started
            execution_started = perf_counter()
            jax.block_until_ready(contribution)
            execution = perf_counter() - execution_started
            sync_started = perf_counter()
            jax.block_until_ready(contribution)
            synchronization = perf_counter() - sync_started
            self._emit(
                "batch_executed",
                "matmat",
                completed,
                current,
                started,
                perf_counter() - batch_started,
            )
            host_started = perf_counter()
            host_contribution = np.asarray(contribution)
            host = perf_counter() - host_started
            accumulation_started = perf_counter()
            result += host_contribution
            accumulation = perf_counter() - accumulation_started
            recent = perf_counter() - batch_started
            self._predicted_batch_seconds["matmat"] = recent
            completed += len(batch)
            self._metrics = replace(
                self._metrics,
                host_to_device_seconds=self._metrics.host_to_device_seconds + transfer,
                dispatch_seconds=self._metrics.dispatch_seconds + dispatch,
                compiled_execution_seconds=self._metrics.compiled_execution_seconds
                + execution,
                synchronization_seconds=self._metrics.synchronization_seconds
                + synchronization,
                device_to_host_seconds=self._metrics.device_to_host_seconds + host,
                input_assembly_seconds=self._metrics.input_assembly_seconds + assembly,
                accumulation_seconds=self._metrics.accumulation_seconds + accumulation,
                group_rows_processed=self._metrics.group_rows_processed
                + arrays[0].shape[0],
                support_entries_processed=self._metrics.support_entries_processed
                + support_entries,
                inflight_routing_bytes_peak=max(
                    self._metrics.inflight_routing_bytes_peak, inflight_bytes
                ),
            )
            self._emit(
                "batch_accumulated",
                "matmat",
                completed,
                current,
                started,
                recent,
                (process_time() - cpu_started)
                / max(perf_counter() - started, np.finfo(float).eps),
            )
        elapsed = perf_counter() - started
        cpu = process_time() - cpu_started
        self._metrics = replace(
            self._metrics,
            process_cpu_seconds=self._metrics.process_cpu_seconds + cpu,
            effective_cpu_cores=cpu / max(elapsed, np.finfo(float).eps),
            peak_rss_bytes=_peak_rss(),
            product_count=self._metrics.product_count + 1,
        )
        self._emit("product_completed", "matmat", completed, (), started, elapsed)
        return result

    def matvec(self, vector: object) -> np.ndarray:
        value = np.asarray(vector, dtype=self.dtype)
        if value.shape != (self.num_free_od,):
            raise ValueError(f"forward vector must have shape ({self.num_free_od},).")
        return self._host_matvec(value)

    def rmatvec(self, vector: object) -> np.ndarray:
        value = np.asarray(vector, dtype=self.dtype)
        if value.shape != (self.num_measurements,):
            raise ValueError(
                f"transpose vector must have shape ({self.num_measurements},)."
            )
        return self._host_rmatvec(value)

    def fidelity_shard_statistics(self) -> tuple[dict[str, object], ...]:
        """Return load-free predicted-work metadata for progressive fidelity.

        Exact enabled-link counts live inside persisted shards and would require
        loading every shard merely to plan a low-effort evaluation. The public
        adapter therefore uses destination-group/link slots as the predicted
        work measure and the corresponding retained array bytes.
        """
        bytes_per_entry = self.dtype.itemsize + np.dtype(bool).itemsize
        return tuple(
            {
                "shard_id": f"routing-{descriptor.shard_index}",
                "shard_index": descriptor.shard_index,
                "support_entries": descriptor.num_groups * self.routing.num_links,
                "routing_bytes": descriptor.num_groups
                * self.routing.num_links
                * bytes_per_entry,
                "stratum": "destination-groups",
            }
            for descriptor in self.routing.shard_partition
        )

    def jax_matvec_fidelity_shard(
        self, shard_index: int, vector: jax.Array
    ) -> jax.Array:
        """Apply one persisted routing shard without loading any other shard."""
        value = np.asarray(vector, dtype=self.dtype)
        if value.shape != (self.num_free_od,):
            raise ValueError(f"forward vector must have shape ({self.num_free_od},).")
        descriptor = self.routing.shard_partition[int(shard_index)]
        active = np.zeros(self.compact_layout.num_active, dtype=self.dtype)
        active[np.asarray(self.compact_layout.free_compact_indices)] = value
        arrays = self._batch_arrays((descriptor,), padded_shards=1)
        contribution = self._fidelity_compiled_forward(
            jnp.asarray(active), *(jnp.asarray(item) for item in arrays)
        )
        return jnp.asarray(contribution)

    def jax_rmatvec_fidelity_shard(
        self, shard_index: int, vector: jax.Array
    ) -> jax.Array:
        """Apply the transpose of one persisted routing shard."""
        value = np.asarray(vector, dtype=self.dtype)
        if value.shape != (self.num_measurements,):
            raise ValueError(
                f"transpose vector must have shape ({self.num_measurements},)."
            )
        descriptor = self.routing.shard_partition[int(shard_index)]
        arrays = self._batch_arrays((descriptor,), padded_shards=1)
        active = self._fidelity_compiled_reverse(
            jnp.asarray(value), *(jnp.asarray(item) for item in arrays)
        )
        return jnp.asarray(active)[
            jnp.asarray(self.compact_layout.free_compact_indices)
        ]

    def jax_matvec(self, vector: jax.Array) -> jax.Array:
        value = jnp.asarray(vector, dtype=self.inputs.base_link_cost.dtype)
        if value.shape != (self.num_free_od,):
            raise ValueError(f"forward vector must have shape ({self.num_free_od},).")
        return self._jax_forward(value)  # type: ignore[operator]

    def jax_rmatvec(self, vector: jax.Array) -> jax.Array:
        value = jnp.asarray(vector, dtype=self.inputs.base_link_cost.dtype)
        if value.shape != (self.num_measurements,):
            raise ValueError(
                f"transpose vector must have shape ({self.num_measurements},)."
            )
        output = jax.ShapeDtypeStruct(
            (self.num_free_od,), self.inputs.base_link_cost.dtype
        )
        return jax.pure_callback(
            self._host_rmatvec, output, value, vmap_method="sequential"
        )

    def jax_matmat(self, matrix: jax.Array) -> jax.Array:
        value = jnp.asarray(matrix, dtype=self.inputs.base_link_cost.dtype)
        if value.ndim != 2 or value.shape[0] != self.num_free_od:
            raise ValueError(
                f"forward matrix must have shape ({self.num_free_od}, k)."
            )
        return self._jax_matmat_callback(value)  # type: ignore[operator]
