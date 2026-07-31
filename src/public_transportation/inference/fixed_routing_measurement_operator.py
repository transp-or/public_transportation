"""Practical direct measurement operators for demand-independent routing.

The builder uses one fixed-shape compiled kernel.  That kernel propagates a
batch of unit OD injections and accumulates mapped measurements inside the DAG
scan; complete per-OD link-flow columns are never constructed or copied to the
host.  Operators may be persisted with strict provenance validation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import tracemalloc
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse as jsparse

from public_transportation import __version__
from public_transportation.measurement.mapping import AggregationSpec

from .assignment_adapter import (
    AssignmentInputs,
    FixedRoutingInputs,
    validate_fixed_routing_compatibility,
)
from .compact_od_assignment_layout import CompactODAssignmentLayout

Array = jnp.ndarray
OperatorRepresentation = Literal["dense", "bcoo"]
ActivationMode = Literal["off", "auto", "dense", "bcoo"]
_SCHEMA_VERSION = 3


def _hash_arrays(*arrays: object) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def assignment_inputs_fingerprint(inputs: AssignmentInputs) -> str:
    graph = inputs.graph
    return _hash_arrays(
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


def measurement_mapping_fingerprint(spec: AggregationSpec) -> str:
    return _hash_arrays(
        np.asarray([spec.num_measurements], dtype=np.int64),
        spec.measurement_index,
        spec.link_index,
    )


@dataclass(frozen=True, slots=True)
class MeasurementOperatorMetrics:
    construction_seconds: float
    dense_bytes: int
    stored_bytes: int
    peak_construction_bytes: int
    nonzero_entries: int
    total_entries: int
    density: float
    chunk_size: int
    compilation_count: int = 0
    num_chunks: int = 0
    chunk_shape: tuple[int, int] = (0, 0)
    compilation_seconds: float = 0.0
    routing_loading_seconds: float = 0.0
    device_synchronization_seconds: float = 0.0
    numpy_transfer_seconds: float = 0.0
    assembly_seconds: float = 0.0
    measurement_aggregation_seconds: float = 0.0
    device_peak_bytes: int = 0
    cache_load_seconds: float = 0.0
    cache_hit: bool = False


@dataclass(frozen=True, slots=True)
class FixedRoutingMeasurementOperator:
    matrix: Array | jsparse.BCOO
    fixed_measurement_offset: Array
    representation: OperatorRepresentation
    num_active_od: int
    num_free_od: int
    num_measurements: int
    od_layout_fingerprint: str | None
    compact_layout_fingerprint: str | None
    assignment_fingerprint: str
    graph_fingerprint: str
    mapping_fingerprint: str
    theta: float
    dtype: str
    metrics: MeasurementOperatorMetrics
    zero_tolerance: float = 0.0
    schema_version: int = _SCHEMA_VERSION
    package_version: str = __version__


def _mapping_slots(
    spec: AggregationSpec, num_links: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return padded measurement ids and masks for every link."""
    link = np.asarray(spec.link_index, dtype=np.int32)
    measurement = np.asarray(spec.measurement_index, dtype=np.int32)
    counts = np.bincount(link, minlength=num_links)
    width = max(1, int(counts.max(initial=0)))
    slots = np.zeros((num_links, width), dtype=np.int32)
    mask = np.zeros((num_links, width), dtype=bool)
    used = np.zeros(num_links, dtype=np.int32)
    for link_id, measurement_id in zip(link, measurement, strict=True):
        position = used[link_id]
        slots[link_id, position] = measurement_id
        mask[link_id, position] = True
        used[link_id] += 1
    return slots, mask


def _make_chunk_kernel(
    *,
    graph,
    chunk_size: int,
    num_measurements: int,
):
    """Build a fused reverse-DP measurement-contribution kernel."""
    topo, out_links, out_mask, head = (
        graph.topo_order,
        graph.out_links,
        graph.out_mask,
        graph.head,
    )
    num_nodes = graph.num_nodes

    def kernel(
        origin_nodes,
        valid_origins,
        link_probability,
        enabled_link_mask,
        link_measurement_runtime,
        link_measurement_mask_runtime,
    ):
        expected = jnp.zeros(
            (num_nodes, num_measurements), dtype=link_probability.dtype
        )

        def step(values, k):
            node = topo[num_nodes - 1 - k]
            links = out_links[node]
            adjacency_mask = out_mask[node]
            safe_links = jnp.where(adjacency_mask, links, 0)
            enabled = adjacency_mask & enabled_link_mask[safe_links]
            measurement_ids = link_measurement_runtime[safe_links]
            mapped = link_measurement_mask_runtime[safe_links] & enabled[:, None]
            reward = jnp.zeros(
                (safe_links.shape[0], num_measurements), dtype=link_probability.dtype
            )
            reward = reward.at[
                jnp.arange(safe_links.shape[0])[:, None], measurement_ids
            ].add(mapped.astype(link_probability.dtype))
            continuation = values[head[safe_links]] + reward
            node_value = jnp.sum(
                continuation * link_probability[safe_links, None] * enabled[:, None],
                axis=0,
            )
            return values.at[node].set(node_value), None

        result, _ = jax.lax.scan(step, expected, jnp.arange(num_nodes, dtype=jnp.int32))
        return result[origin_nodes] * valid_origins[:, None]

    return jax.jit(kernel)


def _storage_bytes(matrix: Array | jsparse.BCOO) -> int:
    return (
        int(matrix.data.nbytes + matrix.indices.nbytes)
        if isinstance(matrix, jsparse.BCOO)
        else int(matrix.nbytes)
    )


def _canonical_bcoo_arrays(
    data: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic row-major sparse arrays with duplicate entries summed."""
    if data.size == 0:
        return data, indices.reshape((0, 2))
    order = np.lexsort((indices[:, 1], indices[:, 0]))
    sorted_indices = indices[order]
    sorted_data = data[order]
    starts = np.empty(sorted_data.size, dtype=bool)
    starts[0] = True
    starts[1:] = np.any(sorted_indices[1:] != sorted_indices[:-1], axis=1)
    positions = np.flatnonzero(starts)
    return (
        np.add.reduceat(sorted_data, positions),
        sorted_indices[positions].astype(np.int32, copy=False),
    )


def _provenance(
    *,
    inputs,
    spec,
    assignment_fingerprint,
    compact_layout,
    od_layout_fingerprint,
    representation,
    dtype,
    routing=None,
    theta=None,
    zero_tolerance=0.0,
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "package_version": __version__,
        "assignment_fingerprint": str(assignment_fingerprint),
        "graph_fingerprint": assignment_inputs_fingerprint(inputs),
        "mapping_fingerprint": measurement_mapping_fingerprint(spec),
        "od_layout_fingerprint": od_layout_fingerprint,
        "compact_layout_fingerprint": None
        if compact_layout is None
        else compact_layout.fingerprint,
        "theta": float(np.asarray(routing.theta))
        if routing is not None
        else float(theta),
        "representation": representation,
        "dtype": str(np.dtype(dtype)),
        "zero_tolerance": float(zero_tolerance),
        "num_active_od": int(inputs.od_origin_node.shape[0]),
        "num_free_od": int(inputs.od_origin_node.shape[0])
        if compact_layout is None
        else compact_layout.num_free,
        "num_measurements": int(spec.num_measurements),
    }


def operator_cache_key(**provenance: object) -> str:
    payload = json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def prepare_fixed_routing_measurement_operator(
    *,
    inputs: AssignmentInputs,
    routing: FixedRoutingInputs,
    spec: AggregationSpec,
    assignment_fingerprint: str,
    compact_layout: CompactODAssignmentLayout | None = None,
    od_layout_fingerprint: str | None = None,
    representation: OperatorRepresentation = "dense",
    chunk_size: int = 128,
    zero_tolerance: float = 0.0,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> FixedRoutingMeasurementOperator:
    """Construct an operator with one compiled fused, padded chunk kernel."""
    validate_fixed_routing_compatibility(inputs=inputs, routing=routing)
    if representation not in ("dense", "bcoo"):
        raise ValueError("representation must be 'dense' or 'bcoo'.")
    if chunk_size <= 0 or zero_tolerance < 0:
        raise ValueError("chunk_size must be positive and zero_tolerance non-negative.")
    log = logging.getLogger(__name__)
    emit = progress or (lambda event: log.info("measurement-operator %s", event))
    num_active, num_measurements = (
        int(inputs.od_origin_node.shape[0]),
        int(spec.num_measurements),
    )
    if compact_layout is None:
        free_indices = np.arange(num_active, dtype=np.int32)
        fixed_indices, fixed_values, compact_fingerprint = (
            np.empty(0, np.int32),
            np.empty(0, np.float32),
            None,
        )
    else:
        if compact_layout.num_active != num_active:
            raise ValueError(
                "Compact layout active dimension does not match assignment inputs."
            )
        free_indices = np.asarray(compact_layout.free_compact_indices, np.int32)
        fixed_indices = np.asarray(compact_layout.fixed_compact_indices, np.int32)
        fixed_values = np.asarray(
            compact_layout.fixed_compact_values,
            dtype=np.dtype(inputs.base_link_cost.dtype),
        )
        compact_fingerprint = compact_layout.fingerprint
    dtype, num_free = np.dtype(inputs.base_link_cost.dtype), int(free_indices.size)
    if num_active == 0:
        dense_shape = (num_measurements, num_free)
        if representation == "dense":
            matrix: Array | jsparse.BCOO = jnp.zeros(
                dense_shape, dtype=inputs.base_link_cost.dtype
            )
        else:
            matrix = jsparse.BCOO(
                (
                    jnp.empty((0,), dtype=inputs.base_link_cost.dtype),
                    jnp.empty((0, 2), dtype=jnp.int32),
                ),
                shape=dense_shape,
            )
        offset = jnp.zeros((num_measurements,), dtype=inputs.base_link_cost.dtype)
        metrics = MeasurementOperatorMetrics(
            0.0,
            num_measurements * num_free * dtype.itemsize,
            _storage_bytes(matrix),
            0,
            0,
            num_measurements * num_free,
            0.0,
            chunk_size,
            chunk_shape=(chunk_size, num_measurements),
        )
        return FixedRoutingMeasurementOperator(
            matrix,
            offset,
            representation,
            0,
            num_free,
            num_measurements,
            od_layout_fingerprint,
            compact_fingerprint,
            str(assignment_fingerprint),
            assignment_inputs_fingerprint(inputs),
            measurement_mapping_fingerprint(spec),
            float(np.asarray(routing.theta)),
            str(dtype),
            metrics,
            zero_tolerance,
        )
    free_column = np.full(num_active, -1, np.int32)
    free_column[free_indices] = np.arange(num_free, dtype=np.int32)
    fixed_by_active = np.zeros(num_active, dtype=dtype)
    fixed_by_active[fixed_indices] = fixed_values
    selected = (free_column >= 0) | (fixed_by_active != 0)
    global_slots, global_slot_mask = _mapping_slots(spec, inputs.graph.num_links)
    effective_masks = np.asarray(routing.effective_group_link_mask)
    group_local_measurements: list[np.ndarray] = []
    group_local_slots: list[np.ndarray] = []
    group_local_masks: list[np.ndarray] = []
    for group in range(int(inputs.group_dest_node.shape[0])):
        mapped_enabled = global_slot_mask & effective_masks[group, :, None]
        local_measurements = np.unique(global_slots[mapped_enabled]).astype(np.int32)
        lookup = np.full(num_measurements, -1, dtype=np.int32)
        lookup[local_measurements] = np.arange(local_measurements.size, dtype=np.int32)
        local_slots = np.zeros_like(global_slots)
        local_slots[mapped_enabled] = lookup[global_slots[mapped_enabled]]
        group_local_measurements.append(local_measurements)
        group_local_slots.append(local_slots)
        group_local_masks.append(mapped_enabled)
    max_local_measurements = max(
        1, max((value.size for value in group_local_measurements), default=0)
    )
    # Pad every group's local measurement dimension to one static compiled shape.
    # The slot ids already lie below each group's local count and therefore below
    # this global maximum.
    kernel = _make_chunk_kernel(
        graph=inputs.graph,
        chunk_size=chunk_size,
        num_measurements=max_local_measurements,
    )
    dummy_origins = jnp.zeros(chunk_size, jnp.int32)
    dummy_valid = jnp.zeros(chunk_size, bool)
    compile_start = perf_counter()
    compiled = kernel.lower(
        dummy_origins,
        dummy_valid,
        routing.group_link_probability[0],
        routing.effective_group_link_mask[0],
        jnp.asarray(group_local_slots[0]),
        jnp.asarray(group_local_masks[0]),
    ).compile()
    compilation_seconds = perf_counter() - compile_start
    compilation_count = 1
    group_indices, group_masks = (
        np.asarray(inputs.group_od_index_padded),
        np.asarray(inputs.group_od_mask),
    )
    origins = np.asarray(inputs.od_origin_node, np.int32)
    jobs: list[tuple[int, np.ndarray]] = []
    for group in range(int(inputs.group_dest_node.shape[0])):
        relevant = group_indices[group][group_masks[group]]
        relevant = relevant[selected[relevant]]
        jobs.extend(
            (group, relevant[first : first + chunk_size])
            for first in range(0, relevant.size, chunk_size)
        )
    selected_columns = sum(len(job_indices) for _, job_indices in jobs)
    dense = (
        np.zeros((num_measurements, num_free), dtype=dtype)
        if representation == "dense"
        else None
    )
    sparse_rows: list[np.ndarray] = []
    sparse_columns: list[np.ndarray] = []
    sparse_data: list[np.ndarray] = []
    fixed_offset = np.zeros(num_measurements, dtype=dtype)
    routing_seconds = sync_seconds = transfer_seconds = 0.0
    tracing_before = tracemalloc.is_tracing()
    if not tracing_before:
        tracemalloc.start()
    _, peak_before = tracemalloc.get_traced_memory()
    total_start = perf_counter()
    for number, (group, od_indices) in enumerate(jobs, 1):
        padded_origins = np.zeros(chunk_size, np.int32)
        valid = np.zeros(chunk_size, bool)
        padded_origins[: od_indices.size] = origins[od_indices]
        valid[: od_indices.size] = True
        start = perf_counter()
        device_columns = compiled(
            jnp.asarray(padded_origins),
            jnp.asarray(valid),
            routing.group_link_probability[group],
            routing.effective_group_link_mask[group],
            jnp.asarray(group_local_slots[group]),
            jnp.asarray(group_local_masks[group]),
        )
        chunk_dispatch = perf_counter() - start
        routing_seconds += chunk_dispatch
        start = perf_counter()
        device_columns.block_until_ready()
        chunk_sync = perf_counter() - start
        sync_seconds += chunk_sync
        start = perf_counter()
        columns_local = np.asarray(device_columns)[: od_indices.size]
        chunk_transfer = perf_counter() - start
        transfer_seconds += chunk_transfer
        local_measurements = group_local_measurements[group]
        columns = columns_local[:, : local_measurements.size]
        for row, active_index in enumerate(od_indices):
            column = int(free_column[active_index])
            values = columns[row]
            if column >= 0:
                if dense is not None:
                    dense[local_measurements, column] = values
                else:
                    nz = np.flatnonzero(np.abs(values) > zero_tolerance)
                    sparse_rows.append(local_measurements[nz].astype(np.int32))
                    sparse_columns.append(np.full(nz.size, column, np.int32))
                    sparse_data.append(values[nz])
            else:
                fixed_offset[local_measurements] += (
                    values * fixed_by_active[active_index]
                )
        elapsed = perf_counter() - total_start
        emit(
            {
                "chunk": number,
                "chunks": len(jobs),
                "shape": (chunk_size, max_local_measurements),
                "columns_completed": min(number * chunk_size, selected_columns),
                "elapsed_seconds": elapsed,
                "chunk_seconds": chunk_dispatch + chunk_sync + chunk_transfer,
                "routing_loading_seconds": chunk_dispatch,
                "measurement_aggregation_seconds": 0.0,
                "device_synchronization_seconds": chunk_sync,
                "numpy_transfer_seconds": chunk_transfer,
                "host_peak_bytes": max(
                    0, tracemalloc.get_traced_memory()[1] - peak_before
                ),
            }
        )
    assembly_start = perf_counter()
    if dense is not None:
        if zero_tolerance:
            dense[np.abs(dense) <= zero_tolerance] = 0
        nonzero = int(np.count_nonzero(dense))
        matrix: Array | jsparse.BCOO = jnp.asarray(dense)
    else:
        rows = np.concatenate(sparse_rows) if sparse_rows else np.empty(0, np.int32)
        columns = (
            np.concatenate(sparse_columns) if sparse_columns else np.empty(0, np.int32)
        )
        data = (
            np.concatenate(sparse_data).astype(dtype, copy=False)
            if sparse_data
            else np.empty(0, dtype)
        )
        indices = np.column_stack((rows, columns)).astype(np.int32, copy=False)
        data, indices = _canonical_bcoo_arrays(data, indices)
        matrix = jsparse.BCOO(
            (jnp.asarray(data), jnp.asarray(indices)),
            shape=(num_measurements, num_free),
            indices_sorted=True,
            unique_indices=True,
        )
        nonzero = int(data.size)
    matrix.block_until_ready()
    offset = jnp.asarray(fixed_offset)
    offset.block_until_ready()
    assembly_seconds = perf_counter() - assembly_start
    elapsed = perf_counter() - total_start
    _, peak_after = tracemalloc.get_traced_memory()
    if not tracing_before:
        tracemalloc.stop()
    total = num_measurements * num_free
    metrics = MeasurementOperatorMetrics(
        elapsed,
        total * dtype.itemsize,
        _storage_bytes(matrix),
        max(0, int(peak_after - peak_before)),
        nonzero,
        total,
        0.0 if total == 0 else nonzero / total,
        chunk_size,
        compilation_count,
        len(jobs),
        (chunk_size, max_local_measurements),
        compilation_seconds,
        routing_seconds,
        sync_seconds,
        transfer_seconds,
        assembly_seconds,
        0.0,
        int((jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0)),
    )
    return FixedRoutingMeasurementOperator(
        matrix,
        offset,
        representation,
        num_active,
        num_free,
        num_measurements,
        od_layout_fingerprint,
        compact_fingerprint,
        str(assignment_fingerprint),
        assignment_inputs_fingerprint(inputs),
        measurement_mapping_fingerprint(spec),
        float(np.asarray(routing.theta)),
        str(dtype),
        metrics,
        zero_tolerance,
    )


def validate_fixed_routing_measurement_operator(
    *,
    operator,
    inputs,
    routing,
    spec,
    assignment_fingerprint,
    compact_layout,
    od_layout_fingerprint,
    zero_tolerance=0.0,
) -> None:
    validate_fixed_routing_compatibility(inputs=inputs, routing=routing)
    expected = _provenance(
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint=assignment_fingerprint,
        compact_layout=compact_layout,
        od_layout_fingerprint=od_layout_fingerprint,
        representation=operator.representation,
        dtype=inputs.base_link_cost.dtype,
        zero_tolerance=zero_tolerance,
    )
    actual = {key: getattr(operator, key) for key in expected}
    for key, value in expected.items():
        if actual[key] != value:
            raise ValueError(
                f"Fixed-routing measurement operator {key.replace('_', ' ')} mismatch."
            )
    if operator.matrix.shape != (
        operator.num_measurements,
        operator.num_free_od,
    ) or operator.fixed_measurement_offset.shape != (operator.num_measurements,):
        raise ValueError(
            "Fixed-routing measurement operator stored array shape mismatch."
        )


def save_fixed_routing_measurement_operator(
    operator: FixedRoutingMeasurementOperator, path: Path
) -> None:
    """Atomically persist an operator and its complete provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        key: getattr(operator, key)
        for key in (
            "representation",
            "num_active_od",
            "num_free_od",
            "num_measurements",
            "od_layout_fingerprint",
            "compact_layout_fingerprint",
            "assignment_fingerprint",
            "graph_fingerprint",
            "mapping_fingerprint",
            "theta",
            "dtype",
            "zero_tolerance",
            "schema_version",
            "package_version",
        )
    }
    metadata["metrics"] = asdict(operator.metrics)
    if isinstance(operator.matrix, jsparse.BCOO):
        arrays = {
            "data": np.asarray(operator.matrix.data),
            "indices": np.asarray(operator.matrix.indices),
            "offset": np.asarray(operator.fixed_measurement_offset),
        }
    else:
        arrays = {
            "matrix": np.asarray(operator.matrix),
            "offset": np.asarray(operator.fixed_measurement_offset),
        }
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(
                stream, metadata=np.asarray(json.dumps(metadata)), **arrays
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_fixed_routing_measurement_operator(
    path: Path,
) -> FixedRoutingMeasurementOperator:
    start = perf_counter()
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(
                "Unsupported fixed-routing measurement operator schema version."
            )
        if metadata.get("representation") not in {"dense", "bcoo"}:
            raise ValueError("Stored operator representation must be dense or bcoo.")
        zero_tolerance = metadata.get("zero_tolerance")
        if (
            not isinstance(zero_tolerance, (int, float))
            or not np.isfinite(zero_tolerance)
            or zero_tolerance < 0.0
        ):
            raise ValueError("Stored operator zero tolerance is invalid.")
        rows = metadata.get("num_measurements")
        columns = metadata.get("num_free_od")
        if (
            not isinstance(rows, int)
            or rows < 0
            or not isinstance(columns, int)
            or columns < 0
        ):
            raise ValueError("Stored operator dimensions are invalid.")
        metrics_payload = metadata.pop("metrics")
        metrics_payload["chunk_shape"] = tuple(metrics_payload["chunk_shape"])
        metrics = replace(
            MeasurementOperatorMetrics(**metrics_payload),
            cache_hit=True,
            cache_load_seconds=perf_counter() - start,
        )
        if metadata["representation"] == "bcoo":
            raw_data = np.asarray(archive["data"])
            raw_indices = np.asarray(archive["indices"])
            if raw_data.ndim != 1 or raw_indices.shape != (raw_data.size, 2):
                raise ValueError("Stored BCOO data and indices have invalid shapes.")
            if raw_indices.dtype.kind not in "iu":
                raise ValueError("Stored BCOO indices must be integers.")
            if not np.all(np.isfinite(raw_data)) or np.any(raw_data < 0.0):
                raise ValueError("Stored BCOO data must be finite and non-negative.")
            if raw_indices.size and (
                np.any(raw_indices[:, 0] < 0)
                or np.any(raw_indices[:, 0] >= rows)
                or np.any(raw_indices[:, 1] < 0)
                or np.any(raw_indices[:, 1] >= columns)
            ):
                raise ValueError("Stored BCOO indices are out of bounds.")
            data, indices = _canonical_bcoo_arrays(raw_data, raw_indices)
            matrix = jsparse.BCOO(
                (jnp.asarray(data), jnp.asarray(indices)),
                shape=(metadata["num_measurements"], metadata["num_free_od"]),
                indices_sorted=True,
                unique_indices=True,
            )
        else:
            dense = np.asarray(archive["matrix"])
            if dense.shape != (rows, columns):
                raise ValueError("Stored dense operator has an invalid shape.")
            if not np.all(np.isfinite(dense)) or np.any(dense < 0.0):
                raise ValueError(
                    "Stored dense operator must be finite and non-negative."
                )
            matrix = jnp.asarray(dense)
        stored_offset = np.asarray(archive["offset"])
        if stored_offset.shape != (rows,):
            raise ValueError("Stored fixed measurement offset has an invalid shape.")
        if not np.all(np.isfinite(stored_offset)) or np.any(stored_offset < 0.0):
            raise ValueError(
                "Stored fixed measurement offset must be finite and non-negative."
            )
        offset = jnp.asarray(stored_offset)
    return FixedRoutingMeasurementOperator(
        matrix=matrix, fixed_measurement_offset=offset, metrics=metrics, **metadata
    )


def cached_operator_path(cache_directory: Path, **provenance: object) -> Path:
    return (
        Path(cache_directory)
        / f"fixed-routing-measurement-{operator_cache_key(**provenance)}.npz"
    )


def fixed_routing_measurement_operator_cache_path(
    *,
    cache_directory: Path,
    inputs,
    spec,
    assignment_fingerprint,
    compact_layout=None,
    od_layout_fingerprint=None,
    representation="bcoo",
    routing=None,
    theta=None,
    zero_tolerance=0.0,
) -> Path:
    """Resolve the exact provenance-keyed cache path without accessing it."""
    provenance = _provenance(
        inputs=inputs,
        routing=routing,
        theta=theta,
        spec=spec,
        assignment_fingerprint=assignment_fingerprint,
        compact_layout=compact_layout,
        od_layout_fingerprint=od_layout_fingerprint,
        representation=representation,
        dtype=inputs.base_link_cost.dtype,
        zero_tolerance=zero_tolerance,
    )
    return cached_operator_path(cache_directory, **provenance)


def load_valid_cached_fixed_routing_measurement_operator(
    *,
    cache_directory: Path,
    inputs,
    theta: float,
    spec,
    assignment_fingerprint,
    compact_layout=None,
    od_layout_fingerprint=None,
    representation="bcoo",
    zero_tolerance=0.0,
) -> FixedRoutingMeasurementOperator | None:
    """Load and validate a cache hit without computing routing probabilities."""
    provenance = _provenance(
        inputs=inputs,
        theta=theta,
        spec=spec,
        assignment_fingerprint=assignment_fingerprint,
        compact_layout=compact_layout,
        od_layout_fingerprint=od_layout_fingerprint,
        representation=representation,
        dtype=inputs.base_link_cost.dtype,
        zero_tolerance=zero_tolerance,
    )
    path = cached_operator_path(cache_directory, **provenance)
    if not path.exists():
        return None
    try:
        operator = load_fixed_routing_measurement_operator(path)
        actual = {key: getattr(operator, key) for key in provenance}
        for key, expected in provenance.items():
            if actual[key] != expected:
                raise ValueError(f"Cached operator {key} mismatch.")
        if operator.matrix.shape != (operator.num_measurements, operator.num_free_od):
            raise ValueError("Cached operator matrix shape mismatch.")
        if operator.fixed_measurement_offset.shape != (operator.num_measurements,):
            raise ValueError("Cached operator offset shape mismatch.")
        return operator
    except (ValueError, KeyError, OSError, json.JSONDecodeError):
        logging.getLogger(__name__).warning("Rejecting invalid operator cache %s", path)
        return None


def load_or_prepare_fixed_routing_measurement_operator(
    *,
    cache_directory: Path,
    inputs,
    routing,
    spec,
    assignment_fingerprint,
    compact_layout=None,
    od_layout_fingerprint=None,
    representation="dense",
    chunk_size=128,
    zero_tolerance=0.0,
    progress=None,
) -> FixedRoutingMeasurementOperator:
    provenance = _provenance(
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint=assignment_fingerprint,
        compact_layout=compact_layout,
        od_layout_fingerprint=od_layout_fingerprint,
        representation=representation,
        dtype=inputs.base_link_cost.dtype,
        zero_tolerance=zero_tolerance,
    )
    path = cached_operator_path(cache_directory, **provenance)
    if path.exists():
        try:
            operator = load_fixed_routing_measurement_operator(path)
            validate_fixed_routing_measurement_operator(
                operator=operator,
                inputs=inputs,
                routing=routing,
                spec=spec,
                assignment_fingerprint=assignment_fingerprint,
                compact_layout=compact_layout,
                od_layout_fingerprint=od_layout_fingerprint,
                zero_tolerance=zero_tolerance,
            )
            return operator
        except (ValueError, KeyError, OSError, json.JSONDecodeError):
            logging.getLogger(__name__).warning(
                "Rejecting invalid operator cache %s", path
            )
    operator = prepare_fixed_routing_measurement_operator(
        inputs=inputs,
        routing=routing,
        spec=spec,
        assignment_fingerprint=assignment_fingerprint,
        compact_layout=compact_layout,
        od_layout_fingerprint=od_layout_fingerprint,
        representation=representation,
        chunk_size=chunk_size,
        zero_tolerance=zero_tolerance,
        progress=progress,
    )
    save_fixed_routing_measurement_operator(operator, path)
    return operator


def choose_fixed_measurement_operator(
    *,
    mode: ActivationMode,
    cached: bool,
    expected_evaluations: int,
    construction_seconds: float | None,
    reference_evaluation_seconds: float = 1.94,
    operator_evaluation_seconds: float = 0.0,
) -> OperatorRepresentation | None:
    """Apply the activation policy while preserving explicit overrides."""
    if mode == "off":
        return None
    if mode in ("dense", "bcoo"):
        return mode
    if mode != "auto":
        raise ValueError("mode must be 'off', 'auto', 'dense', or 'bcoo'.")
    if cached:
        return "bcoo"
    if construction_seconds is None:
        return None
    saving = reference_evaluation_seconds - operator_evaluation_seconds
    return (
        "bcoo"
        if saving > 0 and expected_evaluations * saving > construction_seconds
        else None
    )


def predict_measurements_fixed_operator(
    *, operator: FixedRoutingMeasurementOperator, free_demand: Array, rho: Array
) -> Array:
    demand = jnp.asarray(free_demand)
    if demand.ndim != 1 or demand.shape != (operator.num_free_od,):
        raise ValueError(
            f"free_demand must have shape ({operator.num_free_od},), got {demand.shape}."
        )
    if isinstance(operator.matrix, jsparse.BCOO):
        indices = operator.matrix.indices
        contributions = operator.matrix.data * demand[indices[:, 1]]
        unscaled = jnp.zeros_like(operator.fixed_measurement_offset)
        unscaled = unscaled.at[indices[:, 0]].add(contributions)
        unscaled = unscaled + operator.fixed_measurement_offset
    else:
        unscaled = operator.matrix @ demand + operator.fixed_measurement_offset
    return jnp.asarray(rho).reshape(()) * unscaled
