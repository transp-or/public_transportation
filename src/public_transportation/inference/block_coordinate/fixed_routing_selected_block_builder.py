"""Bounded production construction of selected fixed-routing OD blocks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import tempfile
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse  # type: ignore[import-untyped]

from public_transportation import __version__
from public_transportation.assignment.dial_dp import prepare_destination_routing
from public_transportation.inference.assignment_adapter import (
    AssignmentInputs,
    _routing_inputs_for_destination,
)
from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)
from public_transportation.measurement.mapping import AggregationSpec

from ._canonical import fingerprint
from .blocks import ODBlock
from .operator import SupportedRowsSparseBlockLinearOperator
from .partition import ODBlockPartition
from .selected_blocks import BlockConstructionResourceError
from .support_preflight import (
    SupportPreflightFingerprints,
    _reachability,
)

SELECTED_BLOCK_SUPPORT_SCHEMA_VERSION = 1
SELECTED_BLOCK_CACHE_SCHEMA_VERSION = 2
SELECTED_BLOCK_KERNEL_SCHEMA_VERSION = 1
SELECTED_BLOCK_PROGRESS_SCHEMA_VERSION = 1
_MAXIMUM_COMPILED_KERNELS = 8


def _make_forward_reach_kernel(*, chunk_size: int, num_nodes: int):
    """Create a fixed-shape pass with graph arrays supplied dynamically."""

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
            return values.at[:, head[safe_links]].add(contribution), None

        return jax.lax.scan(step, reach, topo)[0]

    return kernel


def _make_edge_gather_kernel(*, edge_block_size: int):
    """Gather a fixed-size mapped-edge block from retained device reach state."""

    def kernel(
        reach,
        link_probability,
        enabled_link_mask,
        selected_links,
        selected_link_mask,
        tail,
    ):
        safe = jnp.where(selected_link_mask, selected_links, 0)
        return (
            reach[:, tail[safe]]
            * link_probability[safe][None, :]
            * enabled_link_mask[safe][None, :]
            * selected_link_mask[None, :]
        )

    return kernel


def _peak_rss_bytes() -> int:
    """Return process peak RSS in bytes on macOS and Linux."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if os.uname().sysname == "Darwin" else value * 1024


@dataclass(frozen=True, slots=True)
class SelectedBlockBuilderConfig:
    cache_directory: Path
    support_directory: Path | None = None
    od_chunk_size: int = 32
    od_batch_size: int | None = None
    measurement_chunk_size: int = 512
    mapped_edge_chunk_size: int = 2048
    maximum_variables: int = 512
    maximum_support_rows: int = 100_000
    maximum_nonzeros: int = 10_000_000
    maximum_temporary_bytes: int = 512 * 1024**2
    maximum_retained_block_bytes: int = 512 * 1024**2
    per_worker_memory_ceiling_bytes: int = 512 * 1024**2
    zero_tolerance: float = 0.0
    storage_dtype: str = "float64"
    maximum_retained_blocks: int = 1

    def __post_init__(self) -> None:
        cache = Path(self.cache_directory).expanduser()
        support = (
            cache / "support"
            if self.support_directory is None
            else Path(self.support_directory).expanduser()
        )
        for name in (
            "od_chunk_size",
            "measurement_chunk_size",
            "mapped_edge_chunk_size",
            "maximum_variables",
            "maximum_support_rows",
            "maximum_nonzeros",
            "maximum_temporary_bytes",
            "maximum_retained_block_bytes",
            "per_worker_memory_ceiling_bytes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.od_batch_size is not None and self.od_batch_size <= 0:
            raise ValueError("od_batch_size must be positive or None for automatic.")
        if self.maximum_retained_blocks < 0:
            raise ValueError("maximum_retained_blocks must be non-negative.")
        if not np.isfinite(self.zero_tolerance) or self.zero_tolerance < 0:
            raise ValueError("zero_tolerance must be finite and non-negative.")
        dtype = np.dtype(self.storage_dtype)
        if dtype.kind != "f":
            raise TypeError("storage_dtype must be floating point.")
        object.__setattr__(self, "cache_directory", cache)
        object.__setattr__(self, "support_directory", support)
        object.__setattr__(self, "storage_dtype", str(dtype))


@dataclass(frozen=True, slots=True)
class SelectedBlockBuilderProvenance:
    fingerprints: SupportPreflightFingerprints
    semantic_preflight_fingerprint: str
    theta: float

    def __post_init__(self) -> None:
        if not self.semantic_preflight_fingerprint:
            raise ValueError("semantic_preflight_fingerprint must be nonempty.")
        if not np.isfinite(self.theta) or self.theta <= 0:
            raise ValueError("theta must be finite and positive.")


@dataclass(frozen=True, slots=True)
class SelectedBlockSupportArtifact:
    block_id: str
    block_fingerprint: str
    destination_group: int
    free_column_indices: tuple[int, ...]
    active_od_indices: tuple[int, ...]
    support_rows: tuple[int, ...]
    column_indptr: tuple[int, ...]
    column_support_rows: tuple[int, ...]
    exact_nonzeros: int
    fingerprint: str
    path: Path
    disk_bytes: int

    def rows_for_local_column(self, local_column: int) -> np.ndarray:
        first = self.column_indptr[local_column]
        last = self.column_indptr[local_column + 1]
        return np.asarray(self.column_support_rows[first:last], dtype=np.int64)


@dataclass(frozen=True, slots=True)
class SelectedBlockResourceEstimate:
    support_index_bytes: int
    coo_assembly_bytes: int
    csr_csc_bytes: int
    measurement_chunk_bytes: int
    od_chunk_bytes: int
    routing_temporary_bytes: int
    solver_working_bytes: int
    retained_cache_bytes: int
    peak_worker_bytes: int
    requested_od_batch_size: int | None
    effective_od_batch_size: int
    effective_od_columns: int


@dataclass(frozen=True, slots=True)
class SelectedBlockConstructionDiagnostics:
    measurement_index_preparation_seconds: float
    od_chunk_preparation_seconds: float
    routing_evaluation_seconds: float
    measurement_support_filtering_seconds: float
    sparse_triplet_generation_seconds: float
    duplicate_reduction_seconds: float
    csr_csc_assembly_seconds: float
    od_batches: int
    routing_evaluations: int
    measurement_mapping_filtering_passes: int
    candidate_contributions_examined: int
    accepted_nonzeros: int
    requested_od_batch_size: int | None
    effective_od_batch_size: int
    effective_od_columns: int
    jax_argument_transfer_seconds: float = 0.0
    jax_tracing_seconds: float = 0.0
    jax_lowering_seconds: float = 0.0
    jax_compilation_seconds: float = 0.0
    jax_execution_seconds: float = 0.0
    jax_host_transfer_seconds: float = 0.0
    compiled_kernel_cache_hits: int = 0
    compiled_kernel_cache_misses: int = 0
    captured_constant_bytes: int = 0
    rss_before_bytes: int = 0
    rss_after_bytes: int = 0
    jax_backend: str = ""
    jax_devices: tuple[str, ...] = ()
    reach_input_shapes: tuple[tuple[int, ...], ...] = ()
    reach_input_dtypes: tuple[str, ...] = ()
    compiled_kernel_identity: str = ""


@dataclass(frozen=True, slots=True)
class SelectedBlockDeadlineDiagnostics:
    block_id: str
    phase: str
    elapsed_construction_seconds: float
    absolute_deadline: float
    deadline_overshoot_seconds: float
    indivisible_operation_overshoot: bool
    completed_od_batches: int
    total_od_batches: int
    completed_mapping_passes: int
    total_mapping_passes: int
    candidate_contributions_examined: int
    accepted_nonzeros_accumulated: int
    support_cache_hit: bool
    numerical_cache_persistence_completed: bool
    valid_warm_cache_exists: bool
    partial_work_discarded: bool
    current_temporary_memory_estimate: int


class SelectedBlockConstructionDeadlineError(RuntimeError):
    """A selected block stopped safely between bounded construction phases."""

    def __init__(self, diagnostics: SelectedBlockDeadlineDiagnostics) -> None:
        super().__init__(
            f"selected block {diagnostics.block_id!r} reached its deadline "
            f"during {diagnostics.phase}."
        )
        self.diagnostics = diagnostics


@dataclass(frozen=True, slots=True)
class SelectedBlockConstructionProgress:
    block_id: str
    completed_od_chunks: int
    total_od_chunks: int
    completed_measurement_chunks: int
    candidate_entries: int


@dataclass(frozen=True, slots=True)
class SelectedBlockPhaseProgress:
    schema_version: int
    monotonic_timestamp: float
    wall_clock_timestamp: str
    block_id: str
    phase: str
    state: str
    event: str
    elapsed_construction_seconds: float
    absolute_deadline: float | None
    remaining_seconds: float | None
    effective_od_columns: int
    od_batch_index: int
    od_batch_count: int
    mapped_edge_plan_index: int
    mapped_edge_plan_count: int
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    process_rss_bytes: int
    active_thread: str
    active_thread_count: int
    backend: str
    devices: tuple[str, ...]
    compiled_kernel_identity: str
    compiled_kernel_cache_hits: int
    compiled_kernel_cache_misses: int
    completed_mapping_passes: int
    candidate_contributions_examined: int
    accepted_nonzeros: int


class SelectedBlockJSONLProgressSink:
    """Append complete, flushed phase records independently of numerical cache."""

    def __init__(self, path: Path, *, durable: bool = False) -> None:
        self.path = Path(path).expanduser()
        self.durable = durable
        self._lock = threading.Lock()

    def __call__(self, event: SelectedBlockPhaseProgress) -> None:
        payload = json.dumps(asdict(event), sort_keys=True, separators=(",", ":"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(payload + "\n")
            stream.flush()
            if self.durable:
                os.fsync(stream.fileno())


class SelectedBlockDiagnosticStop(RuntimeError):
    """Intentional stop after a requested diagnostic JAX phase."""

    def __init__(self, event: SelectedBlockPhaseProgress) -> None:
        super().__init__(f"selected-block probe stopped after {event.phase}.")
        self.event = event


@dataclass(frozen=True, slots=True)
class _MappedEdgeChunk:
    padded_links: np.ndarray
    edge_mask: np.ndarray
    local_rows: np.ndarray
    valid_edges: int


@dataclass(frozen=True, slots=True)
class _MeasurementChunkPlan:
    rows: np.ndarray
    edge_chunks: tuple[_MappedEdgeChunk, ...]


@dataclass(frozen=True, slots=True)
class SelectedBlockConstructionResult:
    block_id: str
    operator: SupportedRowsSparseBlockLinearOperator
    support_artifact: SelectedBlockSupportArtifact
    estimate: SelectedBlockResourceEstimate
    cache_hit: bool
    support_artifact_load_seconds: float
    support_discovery_seconds: float
    routing_preparation_seconds: float
    numerical_construction_seconds: float
    sparse_assembly_seconds: float
    persistence_seconds: float
    cache_load_seconds: float
    exact_nonzeros: int
    support_rows: int
    disk_bytes: int
    resident_bytes: int
    peak_temporary_bytes: int
    construction_count: int
    reuse_count: int
    estimate_observed_memory_ratio: float
    diagnostics: SelectedBlockConstructionDiagnostics


def _cached_diagnostics(
    estimate: SelectedBlockResourceEstimate, *, accepted_nonzeros: int
) -> SelectedBlockConstructionDiagnostics:
    return SelectedBlockConstructionDiagnostics(
        measurement_index_preparation_seconds=0.0,
        od_chunk_preparation_seconds=0.0,
        routing_evaluation_seconds=0.0,
        measurement_support_filtering_seconds=0.0,
        sparse_triplet_generation_seconds=0.0,
        duplicate_reduction_seconds=0.0,
        csr_csc_assembly_seconds=0.0,
        od_batches=0,
        routing_evaluations=0,
        measurement_mapping_filtering_passes=0,
        candidate_contributions_examined=0,
        accepted_nonzeros=accepted_nonzeros,
        requested_od_batch_size=estimate.requested_od_batch_size,
        effective_od_batch_size=estimate.effective_od_batch_size,
        effective_od_columns=estimate.effective_od_columns,
    )


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_npz(path: Path, **arrays: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path.stat().st_size


class FixedRoutingSelectedBlockBuilder:
    """Construct selected blocks from exact support without a complete operator."""

    def __init__(
        self,
        *,
        inputs: AssignmentInputs,
        spec: AggregationSpec,
        compact_layout: CompactODAssignmentLayout,
        partition: ODBlockPartition,
        provenance: SelectedBlockBuilderProvenance,
        config: SelectedBlockBuilderConfig,
        progress: Callable[[SelectedBlockConstructionProgress], None] | None = None,
        phase_progress: Callable[[SelectedBlockPhaseProgress], None] | None = None,
        progress_file: Path | None = None,
        durable_progress: bool = False,
        diagnostic_stop_after: str | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.inputs = inputs
        self.spec = spec
        self.compact_layout = compact_layout
        self.partition = partition
        self.provenance = provenance
        self.config = config
        self.progress = progress
        if phase_progress is not None and progress_file is not None:
            raise ValueError("provide phase_progress or progress_file, not both.")
        self.phase_progress = (
            SelectedBlockJSONLProgressSink(progress_file, durable=durable_progress)
            if progress_file is not None
            else phase_progress
        )
        allowed_stops = {None, "tracing", "lowering", "compilation", "execution"}
        if diagnostic_stop_after not in allowed_stops:
            raise ValueError(
                "diagnostic_stop_after must be tracing, lowering, compilation, "
                "execution, or None."
            )
        self.diagnostic_stop_after = diagnostic_stop_after
        self.clock = clock
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self._blocks = {block.block_id: block for block in partition.blocks}
        self._retained: OrderedDict[str, SupportedRowsSparseBlockLinearOperator] = (
            OrderedDict()
        )
        self._last_result: SelectedBlockConstructionResult | None = None
        self._construction_count = 0
        self._reuse_count = 0
        self._mapping_links = np.asarray(getattr(spec, "link_index"), dtype=np.int32)
        self._mapping_rows = np.asarray(
            getattr(spec, "measurement_index"), dtype=np.int64
        )
        self._num_measurements = int(getattr(spec, "num_measurements"))
        if self._mapping_links.shape != self._mapping_rows.shape:
            raise ValueError("measurement mapping arrays have inconsistent shapes.")
        if compact_layout.num_active != int(inputs.od_origin_node.shape[0]):
            raise ValueError("compact layout is incompatible with assignment inputs.")
        self._mapping_order = np.argsort(self._mapping_rows, kind="stable")
        self._sorted_mapping_rows = self._mapping_rows[self._mapping_order]
        self._reach_kernels: OrderedDict[str, Callable[..., Any]] = OrderedDict()
        self._edge_gather_kernels: OrderedDict[str, Callable[..., Any]] = OrderedDict()
        self._compiled_kernel_lock = threading.RLock()
        self._routing_group: int | None = None
        self._routing_value: Any = None
        self._deadline_started = 0.0
        self._deadline_block: ODBlock | None = None
        self._absolute_deadline: float | None = None
        self._deadline_phase = "not_started"
        self._completed_od_batches = 0
        self._total_od_batches = 0
        self._completed_mapping_passes = 0
        self._total_mapping_passes = 0
        self._candidate_contributions = 0
        self._accepted_nonzeros = 0
        self._support_cache_hit = False
        self._persistence_completed = False
        self._current_temporary_estimate = 0
        self._deadline_cache_path: Path | None = None
        self._phase_effective_columns = 0
        self._phase_od_batch_index = 0
        self._phase_edge_plan_index = 0
        self._phase_edge_plan_count = 0
        self._phase_kernel_identity = ""
        self._phase_kernel_hits = 0
        self._phase_kernel_misses = 0

    def _emit_phase(
        self,
        phase: str,
        state: str,
        arguments: tuple[Any, ...] = (),
    ) -> None:
        callback = self.phase_progress
        block = self._deadline_block
        if callback is None or block is None:
            return
        now = self.clock()
        deadline = self._absolute_deadline
        event = SelectedBlockPhaseProgress(
            schema_version=SELECTED_BLOCK_PROGRESS_SCHEMA_VERSION,
            monotonic_timestamp=now,
            wall_clock_timestamp=datetime.now(UTC).isoformat(),
            block_id=block.block_id,
            phase=phase,
            state=state,
            event=f"{phase}_{state}",
            elapsed_construction_seconds=max(0.0, now - self._deadline_started),
            absolute_deadline=deadline,
            remaining_seconds=(None if deadline is None else max(0.0, deadline - now)),
            effective_od_columns=self._phase_effective_columns,
            od_batch_index=self._phase_od_batch_index,
            od_batch_count=self._total_od_batches,
            mapped_edge_plan_index=self._phase_edge_plan_index,
            mapped_edge_plan_count=self._phase_edge_plan_count,
            input_shapes=tuple(
                tuple(int(size) for size in value.shape)
                for value in arguments
                if hasattr(value, "shape")
            ),
            input_dtypes=tuple(
                str(value.dtype) for value in arguments if hasattr(value, "dtype")
            ),
            process_rss_bytes=_peak_rss_bytes(),
            active_thread=threading.current_thread().name,
            active_thread_count=threading.active_count(),
            backend=jax.default_backend(),
            devices=tuple(str(device) for device in jax.devices()),
            compiled_kernel_identity=self._phase_kernel_identity,
            compiled_kernel_cache_hits=self._phase_kernel_hits,
            compiled_kernel_cache_misses=self._phase_kernel_misses,
            completed_mapping_passes=self._completed_mapping_passes,
            candidate_contributions_examined=self._candidate_contributions,
            accepted_nonzeros=self._accepted_nonzeros,
        )
        callback(event)
        if (
            state == "complete"
            and self.diagnostic_stop_after is not None
            and phase == f"jax_{self.diagnostic_stop_after}"
        ):
            raise SelectedBlockDiagnosticStop(event)

    supports_absolute_deadline = True

    def _compiled_kernel_identity(self, effective_columns: int) -> str:
        graph = self.inputs.graph
        return fingerprint(
            {
                "schema": SELECTED_BLOCK_KERNEL_SCHEMA_VERSION,
                "assignment": self.provenance.fingerprints.assignment_inputs,
                "backend": jax.default_backend(),
                "routing_dtype": str(self.inputs.base_link_cost.dtype),
                "effective_od_columns": effective_columns,
                "mapped_edge_chunk_size": self.config.mapped_edge_chunk_size,
                "num_nodes": int(graph.num_nodes),
                "num_links": int(graph.num_links),
                "maximum_out_degree": int(graph.out_links.shape[1]),
            }
        )

    def _remember_compiled(
        self,
        cache: OrderedDict[str, Callable[..., Any]],
        key: str,
        executable: Callable[..., Any],
    ) -> None:
        with self._compiled_kernel_lock:
            cache[key] = executable
            cache.move_to_end(key)
            while len(cache) > _MAXIMUM_COMPILED_KERNELS:
                cache.popitem(last=False)

    def _check_deadline(self, phase: str, *, indivisible: bool = False) -> None:
        self._deadline_phase = phase
        deadline = self._absolute_deadline
        if deadline is None:
            return
        now = self.clock()
        if now < deadline:
            return
        block = self._deadline_block
        assert block is not None
        cache_exists = bool(
            self._deadline_cache_path is not None
            and self._deadline_cache_path.is_file()
        )
        self._routing_group = None
        self._routing_value = None
        raise SelectedBlockConstructionDeadlineError(
            SelectedBlockDeadlineDiagnostics(
                block_id=block.block_id,
                phase=phase,
                elapsed_construction_seconds=max(0.0, now - self._deadline_started),
                absolute_deadline=deadline,
                deadline_overshoot_seconds=max(0.0, now - deadline),
                indivisible_operation_overshoot=indivisible,
                completed_od_batches=self._completed_od_batches,
                total_od_batches=self._total_od_batches,
                completed_mapping_passes=self._completed_mapping_passes,
                total_mapping_passes=self._total_mapping_passes,
                candidate_contributions_examined=self._candidate_contributions,
                accepted_nonzeros_accumulated=self._accepted_nonzeros,
                support_cache_hit=self._support_cache_hit,
                numerical_cache_persistence_completed=self._persistence_completed,
                valid_warm_cache_exists=cache_exists,
                partial_work_discarded=not self._persistence_completed,
                current_temporary_memory_estimate=self._current_temporary_estimate,
            )
        )

    def _guard_support_discovery(self, block: ODBlock) -> None:
        """Reject an unsafe block before routing or reachability allocation."""
        cfg = self.config
        if block.num_free_variables > cfg.maximum_variables:
            raise BlockConstructionResourceError("block exceeds the variable limit.")
        value_bytes = np.dtype(cfg.storage_dtype).itemsize
        node_bytes = cfg.od_chunk_size * int(self.inputs.graph.num_nodes)
        mapped_bytes = cfg.od_chunk_size * int(self._mapping_links.size)
        routing_bytes = int(self.inputs.graph.num_links) * (value_bytes + 1)
        discovery_bytes = node_bytes + mapped_bytes + routing_bytes
        if discovery_bytes > cfg.maximum_temporary_bytes:
            raise BlockConstructionResourceError(
                "estimated support-discovery temporary exceeds budget."
            )
        if discovery_bytes > cfg.per_worker_memory_ceiling_bytes:
            raise BlockConstructionResourceError(
                "estimated support-discovery worker peak exceeds budget."
            )

    @property
    def last_result(self) -> SelectedBlockConstructionResult | None:
        return self._last_result

    @property
    def retained_bytes(self) -> int:
        return sum(operator.retained_bytes for operator in self._retained.values())

    def _support_key(self, block: ODBlock) -> str:
        return fingerprint(
            {
                "schema": SELECTED_BLOCK_SUPPORT_SCHEMA_VERSION,
                "package": __version__,
                "provenance": self.provenance,
                "partition": self.partition.fingerprint,
                "block": block.fingerprint,
                "probability_support": "strictly-positive-routing-v1",
            }
        )

    def _support_path(self, block: ODBlock) -> Path:
        assert self.config.support_directory is not None
        return self.config.support_directory / f"support-{self._support_key(block)}.npz"

    def _cache_key(self, block: ODBlock, artifact: SelectedBlockSupportArtifact) -> str:
        return fingerprint(
            {
                "schema": SELECTED_BLOCK_CACHE_SCHEMA_VERSION,
                "package": __version__,
                "provenance": self.provenance,
                "partition": self.partition.fingerprint,
                "block": block.fingerprint,
                "support": artifact.fingerprint,
                "dtype": self.config.storage_dtype,
                "zero_tolerance": self.config.zero_tolerance,
                "accumulation_order": "canonical-mapped-edge-order-v1",
            }
        )

    def _cache_path(
        self, block: ODBlock, artifact: SelectedBlockSupportArtifact
    ) -> Path:
        return (
            self.config.cache_directory
            / f"block-{self._cache_key(block, artifact)}.npz"
        )

    def _routing(self, group: int):
        if self._routing_group == group and self._routing_value is not None:
            return self._routing_value
        self._check_deadline("routing_preparation")
        enabled, cost = _routing_inputs_for_destination(
            graph=self.inputs.graph,
            base_link_cost=self.inputs.base_link_cost,
            group_link_mask=self.inputs.group_link_mask[group],
            dest_node=self.inputs.group_dest_node[group],
        )
        routing = prepare_destination_routing(
            graph=self.inputs.graph,
            link_cost=cost,
            enabled_link_mask=enabled,
            dest_node=self.inputs.group_dest_node[group],
            theta=self.provenance.theta,
        )
        if hasattr(routing.link_prob, "block_until_ready"):
            routing.link_prob.block_until_ready()
        self._check_deadline("routing_preparation", indivisible=True)
        self._routing_group = group
        self._routing_value = routing
        return routing

    def _discover_support(self, block: ODBlock) -> SelectedBlockSupportArtifact:
        self._check_deadline("support_cache_lookup")
        path = self._support_path(block)
        if path.exists():
            self._support_cache_hit = True
            self._check_deadline("support_artifact_loading")
            artifact = self._load_support(block, path)
            self._check_deadline("support_artifact_loading")
            return artifact
        if len(block.destination_group_indices) != 1:
            raise ValueError(
                "selected blocks must belong to exactly one destination group."
            )
        group = block.destination_group_indices[0]
        routing = self._routing(group)
        enabled = np.asarray(routing.enabled_link_mask)
        probability = np.asarray(routing.link_prob)
        eligible = enabled[self._mapping_links] & (
            probability[self._mapping_links] > 0.0
        )
        graph = self.inputs.graph
        origins = np.asarray(self.inputs.od_origin_node, dtype=np.int64)
        column_rows: list[np.ndarray] = []
        union_rows: set[int] = set()
        exact_nonzeros = 0
        for first in range(0, block.num_free_variables, self.config.od_chunk_size):
            self._check_deadline("support_discovery")
            active = np.asarray(
                block.active_od_indices[first : first + self.config.od_chunk_size],
                dtype=np.int64,
            )
            reachable = _reachability(
                origins[active],
                enabled=enabled,
                topo=np.asarray(graph.topo_order, dtype=np.int64),
                out_links=np.asarray(graph.out_links, dtype=np.int64),
                out_mask=np.asarray(graph.out_mask, dtype=bool),
                head=np.asarray(graph.head, dtype=np.int64),
                num_nodes=int(graph.num_nodes),
            )
            mapped = (
                reachable[:, np.asarray(graph.tail)[self._mapping_links]] & eligible
            )
            for local in range(active.size):
                rows = np.unique(self._mapping_rows[mapped[local]])
                exact_nonzeros += int(rows.size)
                if exact_nonzeros > self.config.maximum_nonzeros:
                    raise BlockConstructionResourceError(
                        "block exceeds the nonzero limit during support discovery."
                    )
                union_rows.update(int(row) for row in rows)
                if len(union_rows) > self.config.maximum_support_rows:
                    raise BlockConstructionResourceError(
                        "block exceeds the support-row limit during discovery."
                    )
                column_rows.append(rows)
            self._check_deadline("support_discovery")
        indptr = np.zeros(block.num_free_variables + 1, dtype=np.int64)
        indptr[1:] = np.cumsum([rows.size for rows in column_rows])
        flattened = (
            np.concatenate(column_rows).astype(np.int64, copy=False)
            if column_rows
            else np.empty(0, dtype=np.int64)
        )
        union = np.asarray(sorted(union_rows), dtype=np.int64)
        metadata = {
            "schema_version": SELECTED_BLOCK_SUPPORT_SCHEMA_VERSION,
            "support_key": self._support_key(block),
            "block_id": block.block_id,
            "block_fingerprint": block.fingerprint,
            "destination_group": group,
            "free_column_indices": list(block.free_column_indices),
            "active_od_indices": list(block.active_od_indices),
            "provenance": asdict(self.provenance),
            "content_hash": _array_hash(union, indptr, flattened),
        }
        self._check_deadline("support_cache_persistence")
        disk = _atomic_npz(
            path,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            support_rows=union,
            column_indptr=indptr,
            column_support_rows=flattened,
        )
        self._check_deadline("support_cache_persistence", indivisible=True)
        return SelectedBlockSupportArtifact(
            block.block_id,
            block.fingerprint,
            group,
            block.free_column_indices,
            block.active_od_indices,
            tuple(union.tolist()),
            tuple(indptr.tolist()),
            tuple(flattened.tolist()),
            int(flattened.size),
            str(metadata["content_hash"]),
            path,
            disk,
        )

    def _load_support(self, block: ODBlock, path: Path) -> SelectedBlockSupportArtifact:
        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
                rows = np.asarray(archive["support_rows"], dtype=np.int64)
                indptr = np.asarray(archive["column_indptr"], dtype=np.int64)
                flattened = np.asarray(archive["column_support_rows"], dtype=np.int64)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid selected-block support artifact {path}."
            ) from error
        if metadata.get("schema_version") != SELECTED_BLOCK_SUPPORT_SCHEMA_VERSION:
            raise ValueError("unsupported selected-block support schema.")
        if metadata.get("support_key") != self._support_key(block):
            raise ValueError("selected-block support fingerprint mismatch.")
        if metadata.get("block_id") != block.block_id:
            raise ValueError("selected-block support block ID mismatch.")
        if metadata.get("block_fingerprint") != block.fingerprint:
            raise ValueError("selected-block support block fingerprint mismatch.")
        if metadata.get("free_column_indices") != list(block.free_column_indices):
            raise ValueError("selected-block support free-column mismatch.")
        if metadata.get("active_od_indices") != list(block.active_od_indices):
            raise ValueError("selected-block support active-OD mismatch.")
        if metadata.get("content_hash") != _array_hash(rows, indptr, flattened):
            raise ValueError("selected-block support content hash mismatch.")
        if (
            indptr.shape != (block.num_free_variables + 1,)
            or indptr[-1] != flattened.size
        ):
            raise ValueError("selected-block per-column support is inconsistent.")
        return SelectedBlockSupportArtifact(
            block.block_id,
            block.fingerprint,
            int(metadata["destination_group"]),
            block.free_column_indices,
            block.active_od_indices,
            tuple(rows.tolist()),
            tuple(indptr.tolist()),
            tuple(flattened.tolist()),
            int(flattened.size),
            metadata["content_hash"],
            path,
            path.stat().st_size,
        )

    def prepare_support(
        self, block: ODBlock, *, absolute_deadline: float | None = None
    ) -> SelectedBlockSupportArtifact:
        self._absolute_deadline = absolute_deadline
        self._deadline_block = block
        self._deadline_started = self.clock()
        self._check_deadline("builder_entry")
        if self._blocks.get(block.block_id) != block:
            raise ValueError("block does not match the authoritative partition.")
        self._guard_support_discovery(block)
        return self._discover_support(block)

    def estimate_resources(
        self, block: ODBlock, artifact: SelectedBlockSupportArtifact
    ) -> SelectedBlockResourceEstimate:
        cfg = self.config
        if block.num_free_variables > cfg.maximum_variables:
            raise BlockConstructionResourceError("block exceeds the variable limit.")
        if len(artifact.support_rows) > cfg.maximum_support_rows:
            raise BlockConstructionResourceError("block exceeds the support-row limit.")
        if artifact.exact_nonzeros > cfg.maximum_nonzeros:
            raise BlockConstructionResourceError("block exceeds the nonzero limit.")
        value = np.dtype(cfg.storage_dtype).itemsize
        index = np.dtype(np.int64).itemsize
        support_index = (
            len(artifact.column_support_rows) + len(artifact.column_indptr)
        ) * index
        coo = artifact.exact_nonzeros * (value + 2 * index)
        csr_csc = 2 * artifact.exact_nonzeros * (value + index)
        csr_csc += (len(artifact.support_rows) + block.num_free_variables + 2) * index
        routing = int(self.inputs.graph.num_links) * (value + 1)
        solver = (block.num_free_variables * 3 + self._num_measurements) * value
        retained = csr_csc + len(artifact.support_rows) * index
        if retained > cfg.maximum_retained_block_bytes:
            raise BlockConstructionResourceError(
                "estimated retained block exceeds budget."
            )
        maximum_batches = max(
            1, int(np.ceil(block.num_free_variables / cfg.od_chunk_size))
        )
        requested_batches = (
            maximum_batches
            if cfg.od_batch_size is None
            else min(cfg.od_batch_size, maximum_batches)
        )
        static_temporary = support_index + coo + csr_csc + routing
        for effective_batches in range(requested_batches, 0, -1):
            effective_columns = min(
                block.num_free_variables, cfg.od_chunk_size * effective_batches
            )
            measurement_chunk = effective_columns * cfg.measurement_chunk_size * value
            measurement_chunk += effective_columns * cfg.mapped_edge_chunk_size * value
            measurement_chunk += int(self._mapping_links.size) * (
                np.dtype(np.int32).itemsize
                + np.dtype(np.int64).itemsize
                + np.dtype(bool).itemsize
            )
            od_chunk = effective_columns * int(self.inputs.graph.num_nodes) * value
            temporary = static_temporary + measurement_chunk + od_chunk
            peak = temporary + retained + solver
            if (
                temporary <= cfg.maximum_temporary_bytes
                and peak <= cfg.per_worker_memory_ceiling_bytes
            ):
                return SelectedBlockResourceEstimate(
                    support_index,
                    coo,
                    csr_csc,
                    measurement_chunk,
                    od_chunk,
                    routing,
                    solver,
                    retained,
                    peak,
                    cfg.od_batch_size,
                    effective_batches,
                    effective_columns,
                )
        raise BlockConstructionResourceError(
            "even one OD batch exceeds the construction or worker-memory budget."
        )

    def _load_operator(
        self, block: ODBlock, artifact: SelectedBlockSupportArtifact, path: Path
    ) -> SupportedRowsSparseBlockLinearOperator:
        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
                data = np.asarray(archive["data"])
                indices = np.asarray(archive["indices"])
                indptr = np.asarray(archive["indptr"])
                rows = np.asarray(archive["support_rows"], dtype=np.int64)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid selected-block cache {path}.") from error
        if metadata.get("cache_key") != self._cache_key(block, artifact):
            raise ValueError("selected-block cache fingerprint mismatch.")
        if metadata.get("schema_version") != SELECTED_BLOCK_CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported selected-block cache schema.")
        if metadata.get("block_fingerprint") != block.fingerprint:
            raise ValueError("selected-block cache block fingerprint mismatch.")
        if metadata.get("support_fingerprint") != artifact.fingerprint:
            raise ValueError("selected-block cache support fingerprint mismatch.")
        if metadata.get("content_hash") != _array_hash(data, indices, indptr, rows):
            raise ValueError("selected-block cache content hash mismatch.")
        if rows.shape != (len(artifact.support_rows),) or not np.array_equal(
            rows, np.asarray(artifact.support_rows, dtype=np.int64)
        ):
            raise ValueError("selected-block cache support rows mismatch.")
        if data.dtype != np.dtype(self.config.storage_dtype):
            raise ValueError("selected-block cache dtype mismatch.")
        if indices.ndim != 1 or indptr.shape != (rows.size + 1,):
            raise ValueError("selected-block cache sparse arrays are inconsistent.")
        matrix = sparse.csr_array(
            (data, indices, indptr), shape=(rows.size, block.num_free_variables)
        )
        return SupportedRowsSparseBlockLinearOperator(
            matrix, self._num_measurements, tuple(rows.tolist())
        )

    def _retain(
        self, key: str, operator: SupportedRowsSparseBlockLinearOperator
    ) -> None:
        if self.config.maximum_retained_blocks == 0:
            return
        self._retained[key] = operator
        self._retained.move_to_end(key)
        while len(self._retained) > self.config.maximum_retained_blocks:
            self._retained.popitem(last=False)

    def build_result(
        self, block: ODBlock, *, absolute_deadline: float | None = None
    ) -> SelectedBlockConstructionResult:
        if absolute_deadline is not None and not math.isfinite(absolute_deadline):
            raise ValueError("absolute_deadline must be finite when provided.")
        self._absolute_deadline = absolute_deadline
        self._deadline_block = block
        self._deadline_started = self.clock()
        self._deadline_phase = "builder_entry"
        self._completed_od_batches = 0
        self._total_od_batches = 0
        self._completed_mapping_passes = 0
        self._total_mapping_passes = 0
        self._candidate_contributions = 0
        self._accepted_nonzeros = 0
        self._support_cache_hit = False
        self._persistence_completed = False
        self._current_temporary_estimate = 0
        self._deadline_cache_path = None
        self._phase_effective_columns = 0
        self._phase_od_batch_index = 0
        self._phase_edge_plan_index = 0
        self._phase_edge_plan_count = 0
        self._phase_kernel_identity = ""
        self._phase_kernel_hits = 0
        self._phase_kernel_misses = 0
        self._emit_phase("builder_entry", "start")
        self._check_deadline("builder_entry")
        self._emit_phase("builder_entry", "complete")
        support_was_cached = self._support_path(block).exists()
        self._emit_phase("support_cache_lookup", "start")
        self._emit_phase("support_cache_lookup", "complete")
        self._emit_phase(
            "support_artifact_load" if support_was_cached else "support_discovery",
            "start",
        )
        support_start = self.clock()
        artifact = self.prepare_support(block, absolute_deadline=absolute_deadline)
        support_seconds = self.clock() - support_start
        self._emit_phase(
            "support_artifact_load" if support_was_cached else "support_discovery",
            "complete",
        )
        self._check_deadline("resource_estimation")
        estimate = self.estimate_resources(block, artifact)
        self._current_temporary_estimate = estimate.peak_worker_bytes
        self._deadline_cache_path = self._cache_path(block, artifact)
        self._check_deadline("numerical_cache_lookup")
        key = self._cache_key(block, artifact)
        retained = self._retained.get(key)
        if retained is not None:
            self._reuse_count += 1
            result = SelectedBlockConstructionResult(
                block.block_id,
                retained,
                artifact,
                estimate,
                True,
                support_seconds if support_was_cached else 0.0,
                0.0 if support_was_cached else support_seconds,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                int(cast(Any, retained.compact_matrix).nnz),
                len(retained.measurement_support_indices),
                self._cache_path(block, artifact).stat().st_size,
                retained.retained_bytes,
                estimate.peak_worker_bytes,
                self._construction_count,
                self._reuse_count,
                estimate.peak_worker_bytes / max(1, retained.retained_bytes),
                _cached_diagnostics(
                    estimate,
                    accepted_nonzeros=int(cast(Any, retained.compact_matrix).nnz),
                ),
            )
            self._last_result = result
            self._check_deadline("retained_cache_reuse")
            self._emit_phase("construction", "complete")
            return result
        path = self._cache_path(block, artifact)
        if path.exists():
            started = self.clock()
            try:
                operator = self._load_operator(block, artifact, path)
            except ValueError:
                path.unlink(missing_ok=True)
            else:
                load_seconds = self.clock() - started
                self._check_deadline("numerical_cache_validation")
                self._reuse_count += 1
                self._retain(key, operator)
                result = SelectedBlockConstructionResult(
                    block.block_id,
                    operator,
                    artifact,
                    estimate,
                    True,
                    support_seconds if support_was_cached else 0.0,
                    0.0 if support_was_cached else support_seconds,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    load_seconds,
                    int(cast(Any, operator.compact_matrix).nnz),
                    len(operator.measurement_support_indices),
                    path.stat().st_size,
                    operator.retained_bytes,
                    estimate.peak_worker_bytes,
                    self._construction_count,
                    self._reuse_count,
                    estimate.peak_worker_bytes / max(1, operator.retained_bytes),
                    _cached_diagnostics(
                        estimate,
                        accepted_nonzeros=int(cast(Any, operator.compact_matrix).nnz),
                    ),
                )
                self._last_result = result
                self._check_deadline("numerical_cache_reuse")
                self._emit_phase("construction", "complete")
                return result

        self._emit_phase("routing_preparation", "start")
        routing_started = self.clock()
        routing = self._routing(artifact.destination_group)
        routing_seconds = self.clock() - routing_started
        self._emit_phase("routing_preparation", "complete")
        effective_columns = estimate.effective_od_columns
        self._phase_effective_columns = effective_columns
        compiled_identity = self._compiled_kernel_identity(effective_columns)
        self._phase_kernel_identity = compiled_identity
        with self._compiled_kernel_lock:
            reach_kernel = self._reach_kernels.get(compiled_identity)
            gather_kernel = self._edge_gather_kernels.get(compiled_identity)
        compiled_hits = int(reach_kernel is not None) + int(gather_kernel is not None)
        compiled_misses = 0
        self._phase_kernel_hits = compiled_hits
        argument_transfer_seconds = 0.0
        tracing_seconds = 0.0
        lowering_seconds = 0.0
        compilation_seconds = 0.0
        execution_seconds = 0.0
        host_transfer_seconds = 0.0
        rss_before = _peak_rss_bytes()
        transfer_started = self.clock()
        graph_arguments = (
            jnp.asarray(self.inputs.graph.topo_order),
            jnp.asarray(self.inputs.graph.out_links),
            jnp.asarray(self.inputs.graph.out_mask),
            jnp.asarray(self.inputs.graph.head),
            jnp.asarray(self.inputs.graph.tail),
        )
        for value in graph_arguments:
            value.block_until_ready()
        argument_transfer_seconds += self.clock() - transfer_started
        origins = np.asarray(self.inputs.od_origin_node, dtype=np.int32)
        support_rows = np.asarray(artifact.support_rows, dtype=np.int64)
        support_lookup = {int(row): i for i, row in enumerate(support_rows)}
        row_parts: list[np.ndarray] = []
        column_parts: list[np.ndarray] = []
        data_parts: list[np.ndarray] = []
        numerical_started = self.clock()
        self._check_deadline("measurement_index_preparation")
        self._emit_phase("measurement_index_preparation", "start")
        mapping_started = self.clock()
        enabled_links = np.asarray(routing.enabled_link_mask)
        measurement_plans: list[_MeasurementChunkPlan] = []
        for row_first in range(
            0, support_rows.size, self.config.measurement_chunk_size
        ):
            self._check_deadline("measurement_index_preparation")
            row_chunk = support_rows[
                row_first : row_first + self.config.measurement_chunk_size
            ]
            left = np.searchsorted(self._sorted_mapping_rows, row_chunk, side="left")
            right = np.searchsorted(self._sorted_mapping_rows, row_chunk, side="right")
            mapping_positions = np.concatenate(
                [self._mapping_order[a:b] for a, b in zip(left, right, strict=True)]
            )
            links = self._mapping_links[mapping_positions]
            mapped_rows = self._mapping_rows[mapping_positions]
            enabled = enabled_links[links]
            links = links[enabled]
            local_rows = np.searchsorted(row_chunk, mapped_rows[enabled])
            edge_chunks: list[_MappedEdgeChunk] = []
            for edge_first in range(0, links.size, self.config.mapped_edge_chunk_size):
                self._check_deadline("measurement_index_preparation")
                edge_links = links[
                    edge_first : edge_first + self.config.mapped_edge_chunk_size
                ]
                edge_rows = local_rows[
                    edge_first : edge_first + self.config.mapped_edge_chunk_size
                ]
                padded_links = np.zeros(
                    self.config.mapped_edge_chunk_size, dtype=np.int32
                )
                edge_mask = np.zeros(self.config.mapped_edge_chunk_size, dtype=bool)
                padded_links[: edge_links.size] = edge_links
                edge_mask[: edge_links.size] = True
                edge_chunks.append(
                    _MappedEdgeChunk(
                        padded_links=padded_links,
                        edge_mask=edge_mask,
                        local_rows=edge_rows,
                        valid_edges=int(edge_links.size),
                    )
                )
            measurement_plans.append(
                _MeasurementChunkPlan(rows=row_chunk, edge_chunks=tuple(edge_chunks))
            )
        measurement_index_preparation_seconds = self.clock() - mapping_started
        self._emit_phase("measurement_index_preparation", "complete")
        self._check_deadline("measurement_index_preparation")
        measurement_chunks = 0
        completed_chunks = 0
        od_batches = int(np.ceil(block.num_free_variables / effective_columns))
        self._total_od_batches = od_batches
        self._total_mapping_passes = od_batches * len(measurement_plans)
        self._phase_edge_plan_count = sum(
            len(plan.edge_chunks) for plan in measurement_plans
        )
        total_chunks = od_batches * len(measurement_plans)
        od_preparation_seconds = 0.0
        routing_evaluation_seconds = 0.0
        filtering_seconds = 0.0
        triplet_seconds = 0.0
        candidate_contributions = 0
        accepted_nonzeros = 0
        for od_first in range(0, block.num_free_variables, effective_columns):
            self._phase_od_batch_index = self._completed_od_batches
            self._check_deadline("od_batch_preparation")
            od_started = self.clock()
            active = np.asarray(
                block.active_od_indices[od_first : od_first + effective_columns],
                dtype=np.int64,
            )
            padded_origins = np.zeros(effective_columns, dtype=np.int32)
            valid_origins = np.zeros(effective_columns, dtype=bool)
            padded_origins[: active.size] = origins[active]
            valid_origins[: active.size] = True
            od_preparation_seconds += self.clock() - od_started
            self._check_deadline("routing_evaluation")
            transfer_started = self.clock()
            reach_arguments = (
                jnp.asarray(padded_origins),
                jnp.asarray(valid_origins),
                routing.link_prob,
                routing.enabled_link_mask,
                *graph_arguments[:4],
            )
            for value in reach_arguments[:2]:
                value.block_until_ready()
            argument_transfer_seconds += self.clock() - transfer_started
            if reach_kernel is None:
                compiled_misses += 1
                self._phase_kernel_misses = compiled_misses
                raw_reach = _make_forward_reach_kernel(
                    chunk_size=effective_columns,
                    num_nodes=int(self.inputs.graph.num_nodes),
                )
                self._emit_phase("jax_tracing", "start", reach_arguments)
                self._check_deadline("jax_reach_tracing")
                started = self.clock()
                traced = jax.jit(raw_reach).trace(*reach_arguments)
                tracing_seconds += self.clock() - started
                self._emit_phase("jax_tracing", "complete", reach_arguments)
                self._check_deadline("jax_reach_tracing", indivisible=True)
                self._emit_phase("jax_lowering", "start", reach_arguments)
                self._check_deadline("jax_reach_lowering")
                started = self.clock()
                lowered = traced.lower()
                lowering_seconds += self.clock() - started
                self._emit_phase("jax_lowering", "complete", reach_arguments)
                self._check_deadline("jax_reach_lowering", indivisible=True)
                self._emit_phase("jax_compilation", "start", reach_arguments)
                self._check_deadline("jax_reach_compilation")
                started = self.clock()
                reach_kernel = lowered.compile()
                compilation_seconds += self.clock() - started
                self._emit_phase("jax_compilation", "complete", reach_arguments)
                self._check_deadline("jax_reach_compilation", indivisible=True)
                self._remember_compiled(
                    self._reach_kernels, compiled_identity, reach_kernel
                )
            self._emit_phase("jax_execution", "start", reach_arguments)
            self._check_deadline("jax_reach_execution")
            evaluation_started = self.clock()
            reach = reach_kernel(*reach_arguments)
            reach.block_until_ready()
            elapsed = self.clock() - evaluation_started
            routing_evaluation_seconds += elapsed
            execution_seconds += elapsed
            self._emit_phase("jax_execution", "complete", reach_arguments)
            self._check_deadline("jax_reach_execution", indivisible=True)
            for plan in measurement_plans:
                self._emit_phase("support_filtering", "start")
                self._check_deadline("measurement_support_filtering")
                filtering_started = self.clock()
                values = np.zeros((active.size, plan.rows.size), dtype=np.float64)
                for edge_chunk in plan.edge_chunks:
                    self._phase_edge_plan_index += 1
                    self._check_deadline("mapped_edge_chunk")
                    transfer_started = self.clock()
                    gather_arguments = (
                        reach,
                        routing.link_prob,
                        routing.enabled_link_mask,
                        jnp.asarray(edge_chunk.padded_links),
                        jnp.asarray(edge_chunk.edge_mask),
                        graph_arguments[4],
                    )
                    for value in gather_arguments[3:5]:
                        value.block_until_ready()
                    argument_transfer_seconds += self.clock() - transfer_started
                    if gather_kernel is None:
                        compiled_misses += 1
                        self._phase_kernel_misses = compiled_misses
                        raw_gather = _make_edge_gather_kernel(
                            edge_block_size=self.config.mapped_edge_chunk_size
                        )
                        self._emit_phase("jax_tracing", "start", gather_arguments)
                        self._check_deadline("jax_gather_tracing")
                        started = self.clock()
                        traced = jax.jit(raw_gather).trace(*gather_arguments)
                        tracing_seconds += self.clock() - started
                        self._emit_phase("jax_tracing", "complete", gather_arguments)
                        self._check_deadline("jax_gather_tracing", indivisible=True)
                        self._emit_phase("jax_lowering", "start", gather_arguments)
                        self._check_deadline("jax_gather_lowering")
                        started = self.clock()
                        lowered = traced.lower()
                        lowering_seconds += self.clock() - started
                        self._emit_phase("jax_lowering", "complete", gather_arguments)
                        self._check_deadline("jax_gather_lowering", indivisible=True)
                        self._emit_phase("jax_compilation", "start", gather_arguments)
                        self._check_deadline("jax_gather_compilation")
                        started = self.clock()
                        gather_kernel = lowered.compile()
                        compilation_seconds += self.clock() - started
                        self._emit_phase(
                            "jax_compilation", "complete", gather_arguments
                        )
                        self._check_deadline("jax_gather_compilation", indivisible=True)
                        self._remember_compiled(
                            self._edge_gather_kernels, compiled_identity, gather_kernel
                        )
                    self._emit_phase("jax_execution", "start", gather_arguments)
                    self._check_deadline("jax_gather_execution")
                    started = self.clock()
                    gathered = gather_kernel(*gather_arguments)
                    gathered.block_until_ready()
                    elapsed = self.clock() - started
                    execution_seconds += elapsed
                    self._emit_phase("jax_execution", "complete", gather_arguments)
                    self._emit_phase("host_transfer", "start", (gathered,))
                    self._check_deadline("host_transfer")
                    transfer_started = self.clock()
                    edge_values = np.asarray(gathered)[
                        : active.size, : edge_chunk.valid_edges
                    ]
                    host_transfer_seconds += self.clock() - transfer_started
                    self._emit_phase("host_transfer", "complete", (gathered,))
                    self._check_deadline("mapped_edge_chunk", indivisible=True)
                    candidate_contributions += active.size * edge_chunk.valid_edges
                    self._candidate_contributions = candidate_contributions
                    for edge_position, local_row in enumerate(edge_chunk.local_rows):
                        values[:, local_row] += edge_values[:, edge_position]
                filtering_seconds += self.clock() - filtering_started
                self._emit_phase("support_filtering", "complete")
                measurement_chunks += 1
                self._emit_phase("triplet_generation", "start")
                self._check_deadline("sparse_triplet_generation")
                triplet_started = self.clock()
                for local, column in enumerate(range(od_first, od_first + active.size)):
                    allowed_global = artifact.rows_for_local_column(column)
                    in_chunk = allowed_global[
                        (allowed_global >= plan.rows[0])
                        & (allowed_global <= plan.rows[-1])
                    ]
                    if not in_chunk.size:
                        continue
                    positions = np.searchsorted(plan.rows, in_chunk)
                    selected = values[local, positions]
                    nonzero = np.flatnonzero(
                        np.abs(selected) > self.config.zero_tolerance
                    )
                    accepted_nonzeros += int(nonzero.size)
                    self._accepted_nonzeros = accepted_nonzeros
                    row_parts.append(
                        np.asarray(
                            [support_lookup[int(row)] for row in in_chunk[nonzero]],
                            dtype=np.int64,
                        )
                    )
                    column_parts.append(np.full(nonzero.size, column, dtype=np.int64))
                    data_parts.append(selected[nonzero])
                triplet_seconds += self.clock() - triplet_started
                self._emit_phase("triplet_generation", "complete")
                completed_chunks += 1
                self._completed_mapping_passes += 1
                self._check_deadline("sparse_triplet_generation")
                if self.progress is not None:
                    self.progress(
                        SelectedBlockConstructionProgress(
                            block.block_id,
                            completed_chunks,
                            total_chunks,
                            measurement_chunks,
                            accepted_nonzeros,
                        )
                    )
            self._completed_od_batches += 1
        numerical_seconds = self.clock() - numerical_started
        self._routing_group = None
        self._routing_value = None
        self._emit_phase("sparse_assembly", "start")
        self._check_deadline("sparse_assembly")
        assembly_started = self.clock()
        rows = np.concatenate(row_parts) if row_parts else np.empty(0, np.int64)
        columns = (
            np.concatenate(column_parts) if column_parts else np.empty(0, np.int64)
        )
        data = (
            np.concatenate(data_parts).astype(self.config.storage_dtype, copy=False)
            if data_parts
            else np.empty(0, dtype=self.config.storage_dtype)
        )
        coo = sparse.coo_array(
            (data, (rows, columns)),
            shape=(support_rows.size, block.num_free_variables),
        )
        self._emit_phase("duplicate_reduction", "start")
        self._check_deadline("duplicate_reduction")
        reduction_started = self.clock()
        matrix = coo.tocsr()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        duplicate_reduction_seconds = self.clock() - reduction_started
        self._emit_phase("duplicate_reduction", "complete")
        self._check_deadline("csr_csc_assembly")
        csr_csc_started = self.clock()
        operator = SupportedRowsSparseBlockLinearOperator(
            matrix, self._num_measurements, artifact.support_rows
        )
        csr_csc_seconds = self.clock() - csr_csc_started
        assembly_seconds = self.clock() - assembly_started
        self._emit_phase("sparse_assembly", "complete")
        self._emit_phase("validation", "start")
        self._check_deadline("numerical_validation")
        if not set(operator.measurement_support_indices).issubset(
            artifact.support_rows
        ):
            raise ValueError("numerical operator contains rows outside exact support.")
        self._emit_phase("validation", "complete")
        compact_matrix = cast(Any, operator.compact_matrix)
        metadata = {
            "schema_version": SELECTED_BLOCK_CACHE_SCHEMA_VERSION,
            "cache_key": key,
            "block_id": block.block_id,
            "block_fingerprint": block.fingerprint,
            "support_fingerprint": artifact.fingerprint,
            "content_hash": _array_hash(
                compact_matrix.data,
                compact_matrix.indices,
                compact_matrix.indptr,
                support_rows,
            ),
        }
        self._emit_phase("cache_persistence", "start")
        self._check_deadline("numerical_cache_persistence")
        persistence_started = self.clock()
        disk = _atomic_npz(
            path,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            data=compact_matrix.data,
            indices=compact_matrix.indices,
            indptr=compact_matrix.indptr,
            support_rows=support_rows,
        )
        persistence_seconds = self.clock() - persistence_started
        self._persistence_completed = True
        self._emit_phase("cache_persistence", "complete")
        self._check_deadline("numerical_cache_persistence", indivisible=True)
        self._check_deadline("final_cache_validation")
        self._load_operator(block, artifact, path)
        self._check_deadline("final_cache_validation")
        self._construction_count += 1
        self._retain(key, operator)
        result = SelectedBlockConstructionResult(
            block.block_id,
            operator,
            artifact,
            estimate,
            False,
            support_seconds if support_was_cached else 0.0,
            0.0 if support_was_cached else support_seconds,
            routing_seconds,
            numerical_seconds,
            assembly_seconds,
            persistence_seconds,
            0.0,
            int(matrix.nnz),
            len(operator.measurement_support_indices),
            disk,
            operator.retained_bytes,
            estimate.peak_worker_bytes,
            self._construction_count,
            self._reuse_count,
            estimate.peak_worker_bytes / max(1, operator.retained_bytes),
            SelectedBlockConstructionDiagnostics(
                measurement_index_preparation_seconds=(
                    measurement_index_preparation_seconds
                ),
                od_chunk_preparation_seconds=od_preparation_seconds,
                routing_evaluation_seconds=routing_evaluation_seconds,
                measurement_support_filtering_seconds=filtering_seconds,
                sparse_triplet_generation_seconds=triplet_seconds,
                duplicate_reduction_seconds=duplicate_reduction_seconds,
                csr_csc_assembly_seconds=csr_csc_seconds,
                od_batches=od_batches,
                routing_evaluations=od_batches,
                measurement_mapping_filtering_passes=len(measurement_plans),
                candidate_contributions_examined=candidate_contributions,
                accepted_nonzeros=int(matrix.nnz),
                requested_od_batch_size=estimate.requested_od_batch_size,
                effective_od_batch_size=estimate.effective_od_batch_size,
                effective_od_columns=estimate.effective_od_columns,
                jax_argument_transfer_seconds=argument_transfer_seconds,
                jax_tracing_seconds=tracing_seconds,
                jax_lowering_seconds=lowering_seconds,
                jax_compilation_seconds=compilation_seconds,
                jax_execution_seconds=execution_seconds,
                jax_host_transfer_seconds=host_transfer_seconds,
                compiled_kernel_cache_hits=compiled_hits,
                compiled_kernel_cache_misses=compiled_misses,
                captured_constant_bytes=0,
                rss_before_bytes=rss_before,
                rss_after_bytes=_peak_rss_bytes(),
                jax_backend=jax.default_backend(),
                jax_devices=tuple(str(device) for device in jax.devices()),
                reach_input_shapes=tuple(
                    tuple(int(size) for size in value.shape)
                    for value in reach_arguments
                ),
                reach_input_dtypes=tuple(str(value.dtype) for value in reach_arguments),
                compiled_kernel_identity=compiled_identity,
            ),
        )
        self._last_result = result
        self._emit_phase("construction", "complete")
        return result

    def build(
        self, block: ODBlock, *, absolute_deadline: float | None = None
    ) -> SupportedRowsSparseBlockLinearOperator:
        return self.build_result(block, absolute_deadline=absolute_deadline).operator

    def __call__(self, block: ODBlock) -> SupportedRowsSparseBlockLinearOperator:
        return self.build(block)

    def release(self, block: ODBlock) -> bool:
        artifact = self.prepare_support(block)
        return self._retained.pop(self._cache_key(block, artifact), None) is not None

    def release_all(self) -> None:
        self._retained.clear()
        self._routing_group = None
        self._routing_value = None
