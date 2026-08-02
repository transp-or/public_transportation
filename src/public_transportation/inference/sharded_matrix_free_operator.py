"""On-demand matrix-free measurement products over persisted routing shards."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.measurement.mapping import AggregationSpec

from .assignment_adapter import AssignmentInputs
from .compact_od_assignment_layout import CompactODAssignmentLayout
from .fixed_routing_measurement_operator import measurement_mapping_fingerprint
from .sharded_fixed_routing import (
    FixedRoutingShard,
    FixedRoutingShardDescriptor,
    ShardedFixedRoutingInputs,
    load_fixed_routing_shard,
)


@dataclass(frozen=True, slots=True)
class ShardedMatrixFreeMetrics:
    stored_bytes: int = 0
    peak_construction_bytes: int = 0


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
    fixed_measurement_offset: np.ndarray = field(init=False, repr=False)
    _cache: OrderedDict[int, FixedRoutingShard] = field(
        init=False, default_factory=OrderedDict, repr=False
    )
    _lock: RLock = field(init=False, default_factory=RLock, repr=False)
    _jax_forward: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.resident_shard_limit <= 0:
            raise ValueError("resident_shard_limit must be positive.")
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
        return ShardedMatrixFreeMetrics()

    @property
    def resident_shards(self) -> int:
        with self._lock:
            return len(self._cache)

    def _load(self, descriptor: FixedRoutingShardDescriptor) -> FixedRoutingShard:
        with self._lock:
            cached = self._cache.pop(descriptor.shard_index, None)
            if cached is None:
                cached = load_fixed_routing_shard(
                    routing=self.routing, descriptor=descriptor
                )
            self._cache[descriptor.shard_index] = cached
            while len(self._cache) > self.resident_shard_limit:
                self._cache.popitem(last=False)
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

    def _host_active_matvec(self, active: np.ndarray) -> np.ndarray:
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

    def _host_matvec(self, free: np.ndarray) -> np.ndarray:
        active = np.zeros(self.compact_layout.num_active, dtype=self.dtype)
        active[np.asarray(self.compact_layout.free_compact_indices)] = np.asarray(
            free, dtype=self.dtype
        )
        return self._host_active_matvec(active)

    def _host_rmatvec(self, measurement_cotangent: np.ndarray) -> np.ndarray:
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
        return jax.vmap(self.jax_matvec, in_axes=1, out_axes=1)(value)
