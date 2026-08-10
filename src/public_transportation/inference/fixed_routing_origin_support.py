"""Read-only origin-specific support analysis for fixed-routing measurements."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
from jax.experimental import sparse as jsparse
from scipy import sparse  # type: ignore[import-untyped]

from .assignment_adapter import (
    AssignmentInputs,
    FixedRoutingInputs,
    validate_fixed_routing_compatibility,
)
from .construction_control import (
    ConstructionDeadline,
    ConstructionPhase,
    ConstructionProgressReporter,
    deadline_stop,
)
from .compact_od_assignment_layout import CompactODAssignmentLayout
from .fixed_routing_measurement_operator import FixedRoutingMeasurementOperator
from .sharded_fixed_routing import (
    ShardedFixedRoutingInputs,
    fixed_routing_descriptor_for_group,
    load_fixed_routing_shard,
    validate_sharded_fixed_routing_compatibility,
)

ORIGIN_SUPPORT_GROUP_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class OriginSupportConfig:
    """Memory and numerical controls for structural support discovery."""

    origin_chunk_size: int = 64
    worker_memory_budget_bytes: int = 512 * 1024 * 1024
    probability_tolerance: float = 0.0
    materialize: bool = True
    max_materialized_entries: int = 100_000_000

    def __post_init__(self) -> None:
        if self.origin_chunk_size <= 0:
            raise ValueError("origin_chunk_size must be positive.")
        if self.worker_memory_budget_bytes <= 0:
            raise ValueError("worker_memory_budget_bytes must be positive.")
        if (
            not math.isfinite(self.probability_tolerance)
            or self.probability_tolerance < 0.0
        ):
            raise ValueError("probability_tolerance must be finite and non-negative.")
        if self.max_materialized_entries <= 0:
            raise ValueError("max_materialized_entries must be positive.")


@dataclass(frozen=True, slots=True)
class GroupOriginSupportSummary:
    group: int
    selected_od_cells: int
    free_od_cells: int
    positive_fixed_od_cells: int
    group_measurements: int
    group_level_candidate_entries: int
    origin_specific_entries: int

    @property
    def reduction_fraction(self) -> float:
        if self.group_level_candidate_entries == 0:
            return 0.0
        return 1.0 - self.origin_specific_entries / self.group_level_candidate_entries


@dataclass(frozen=True, slots=True)
class OriginSupportMetrics:
    support_discovery_seconds: float
    reachability_seconds: float
    measurement_projection_seconds: float
    canonicalization_seconds: float
    estimated_peak_working_bytes: int
    group_level_candidate_entries: int
    origin_specific_entries: int
    reduction_fraction: float
    free_support_entries: int
    positive_fixed_support_entries: int


@dataclass(frozen=True, slots=True)
class OriginSpecificMeasurementSupport:
    """Structural support in canonical measurement/free-OD coordinates."""

    num_measurements: int
    num_free_od: int
    free_support: sparse.csr_array | None
    positive_fixed_support: sparse.csr_array | None
    positive_fixed_active_indices: np.ndarray
    summaries: tuple[GroupOriginSupportSummary, ...]
    metrics: OriginSupportMetrics
    fingerprint: str

    @property
    def materialized(self) -> bool:
        return self.free_support is not None


@dataclass(frozen=True, slots=True)
class OriginSupportValidation:
    realized_free_entries: int
    realized_fixed_offset_entries: int
    missing_free_entries: int
    missing_fixed_offset_entries: int
    free_false_positive_entries: int
    fixed_false_positive_entries: int

    @property
    def complete(self) -> bool:
        return self.missing_free_entries == 0 and self.missing_fixed_offset_entries == 0


def _support_fingerprint(
    free_support: sparse.csr_array | None,
    fixed_support: sparse.csr_array | None,
    summaries: tuple[GroupOriginSupportSummary, ...],
) -> str:
    digest = hashlib.sha256()
    for matrix in (free_support, fixed_support):
        if matrix is None:
            digest.update(b"not-materialized")
            continue
        for array in (matrix.data, matrix.indices, matrix.indptr):
            value = np.ascontiguousarray(array)
            digest.update(str(value.dtype).encode())
            digest.update(value.tobytes())
    for item in summaries:
        digest.update(repr(item).encode())
    return digest.hexdigest()


def _group_content_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _group_checkpoint_path(directory: Path, group: int) -> Path:
    return Path(directory) / "support_groups" / f"group-{group:06d}.npz"


def _save_group_checkpoint(
    *,
    directory: Path,
    group: int,
    provenance_hash: str,
    summary: GroupOriginSupportSummary,
    free_rows: np.ndarray,
    free_columns: np.ndarray,
    fixed_rows: np.ndarray,
    fixed_columns: np.ndarray,
) -> Path:
    arrays = {
        "free_rows": np.asarray(free_rows, dtype=np.int64),
        "free_columns": np.asarray(free_columns, dtype=np.int64),
        "fixed_rows": np.asarray(fixed_rows, dtype=np.int64),
        "fixed_columns": np.asarray(fixed_columns, dtype=np.int64),
    }
    metadata = {
        "schema_version": ORIGIN_SUPPORT_GROUP_SCHEMA_VERSION,
        "provenance_hash": provenance_hash,
        "group": group,
        "summary": {
            name: getattr(summary, name) for name in summary.__dataclass_fields__
        },
        "content_hash": _group_content_hash(arrays),
    }
    destination = _group_checkpoint_path(directory, group)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as stream:
            np.savez(
                stream,
                metadata=np.asarray(json.dumps(metadata)),
                **arrays,  # type: ignore[arg-type]
            )
            stream.flush()
            os.fsync(stream.fileno())
        _load_group_checkpoint(
            Path(temporary), group=group, provenance_hash=provenance_hash
        )
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _load_group_checkpoint(
    path: Path, *, group: int, provenance_hash: str
) -> tuple[
    GroupOriginSupportSummary, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "metadata"
        }
    if metadata.get("schema_version") != ORIGIN_SUPPORT_GROUP_SCHEMA_VERSION:
        raise ValueError("origin-support group schema is incompatible.")
    if metadata.get("provenance_hash") != provenance_hash:
        raise ValueError("origin-support group provenance is incompatible.")
    if metadata.get("group") != group:
        raise ValueError("origin-support group identity is incompatible.")
    if metadata.get("content_hash") != _group_content_hash(arrays):
        raise ValueError("origin-support group content hash is invalid.")
    try:
        summary = GroupOriginSupportSummary(**metadata["summary"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("origin-support group summary is invalid.") from error
    return (
        summary,
        arrays["free_rows"],
        arrays["free_columns"],
        arrays["fixed_rows"],
        arrays["fixed_columns"],
    )


def _chunk_reachability(
    *,
    origins: np.ndarray,
    enabled_links: np.ndarray,
    topo_order: np.ndarray,
    out_links: np.ndarray,
    out_mask: np.ndarray,
    head: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    """Propagate boolean reachability through the assignment DAG."""
    reachable = np.zeros((origins.size, num_nodes), dtype=bool)
    reachable[np.arange(origins.size), origins] = True
    for node in topo_order:
        active_rows = reachable[:, node]
        if not np.any(active_rows):
            continue
        links = out_links[node][out_mask[node]]
        links = links[enabled_links[links]]
        if links.size:
            reachable[np.ix_(active_rows, np.unique(head[links]))] = True
    return reachable


def analyze_fixed_routing_origin_support(
    *,
    inputs: AssignmentInputs,
    routing: FixedRoutingInputs | ShardedFixedRoutingInputs,
    spec,
    compact_layout: CompactODAssignmentLayout,
    config: OriginSupportConfig | None = None,
    checkpoint_directory: str | Path | None = None,
    checkpoint_provenance_hash: str | None = None,
    deadline: ConstructionDeadline | None = None,
    reporter: ConstructionProgressReporter | None = None,
) -> OriginSpecificMeasurementSupport:
    """Discover support without evaluating passenger-flow values."""
    config = OriginSupportConfig() if config is None else config
    if (checkpoint_directory is None) != (checkpoint_provenance_hash is None):
        raise ValueError(
            "checkpoint_directory and checkpoint_provenance_hash must be provided together."
        )
    checkpoint_root = (
        None if checkpoint_directory is None else Path(checkpoint_directory)
    )
    if checkpoint_root is not None:
        group_directory = checkpoint_root / "support_groups"
        if group_directory.exists():
            for abandoned in group_directory.glob(".*.tmp"):
                abandoned.unlink(missing_ok=True)
    if isinstance(routing, ShardedFixedRoutingInputs):
        validate_sharded_fixed_routing_compatibility(inputs=inputs, routing=routing)
    else:
        validate_fixed_routing_compatibility(inputs=inputs, routing=routing)
    num_active = int(inputs.od_origin_node.shape[0])
    if compact_layout.num_active != num_active:
        raise ValueError("compact layout and assignment active dimensions differ.")
    num_measurements = int(spec.num_measurements)
    num_free = compact_layout.num_free
    num_nodes = int(inputs.graph.num_nodes)
    mapping_links = np.asarray(spec.link_index, dtype=np.int64)
    mapping_measurements = np.asarray(spec.measurement_index, dtype=np.int64)
    if mapping_links.shape != mapping_measurements.shape:
        raise ValueError("measurement mapping arrays have inconsistent shapes.")
    estimated_peak = int(
        config.origin_chunk_size
        * (num_nodes + mapping_links.size)
        * np.dtype(bool).itemsize
    )
    if estimated_peak > config.worker_memory_budget_bytes:
        raise MemoryError(
            "origin-support reachability estimate exceeds worker memory budget"
        )

    free_column = np.full(num_active, -1, dtype=np.int64)
    free_active = np.asarray(compact_layout.free_compact_indices, dtype=np.int64)
    free_column[free_active] = np.arange(num_free, dtype=np.int64)
    fixed_active = np.asarray(compact_layout.fixed_compact_indices, dtype=np.int64)
    fixed_values = np.asarray(compact_layout.fixed_compact_values)
    positive_fixed = fixed_active[fixed_values > 0.0]
    fixed_column = np.full(num_active, -1, dtype=np.int64)
    fixed_column[positive_fixed] = np.arange(positive_fixed.size, dtype=np.int64)
    selected = (free_column >= 0) | (fixed_column >= 0)

    group_indices = np.asarray(inputs.group_od_index_padded)
    group_masks = np.asarray(inputs.group_od_mask)
    origins_by_active = np.asarray(inputs.od_origin_node, dtype=np.int64)
    topo = np.asarray(inputs.graph.topo_order, dtype=np.int64)
    out_links = np.asarray(inputs.graph.out_links, dtype=np.int64)
    out_mask = np.asarray(inputs.graph.out_mask, dtype=bool)
    head = np.asarray(inputs.graph.head, dtype=np.int64)
    tail = np.asarray(inputs.graph.tail, dtype=np.int64)
    probabilities = (
        None
        if isinstance(routing, ShardedFixedRoutingInputs)
        else np.asarray(routing.group_link_probability)
    )
    effective = (
        None
        if isinstance(routing, ShardedFixedRoutingInputs)
        else np.asarray(routing.effective_group_link_mask, dtype=bool)
    )
    resident_descriptor = None
    resident_shard = None

    def routing_for_group(group: int) -> tuple[np.ndarray, np.ndarray]:
        nonlocal resident_descriptor, resident_shard
        if not isinstance(routing, ShardedFixedRoutingInputs):
            assert probabilities is not None and effective is not None
            return probabilities[group], effective[group]
        descriptor = fixed_routing_descriptor_for_group(routing, group)
        if resident_descriptor != descriptor:
            resident_shard = load_fixed_routing_shard(
                routing=routing, descriptor=descriptor
            )
            resident_descriptor = descriptor
        assert resident_shard is not None
        local = group - descriptor.group_start
        return (
            resident_shard.group_link_probability[local],
            resident_shard.effective_group_link_mask[local],
        )

    free_rows: list[np.ndarray] = []
    free_columns: list[np.ndarray] = []
    fixed_rows: list[np.ndarray] = []
    fixed_columns: list[np.ndarray] = []
    summaries: list[GroupOriginSupportSummary] = []
    reachability_seconds = projection_seconds = 0.0
    total_start = perf_counter()
    total_origin_specific = total_group_bound = 0
    recent_group_seconds: list[float] = []
    for group in range(int(inputs.group_dest_node.shape[0])):
        predicted_group = (
            float(np.mean(recent_group_seconds[-3:]))
            if recent_group_seconds
            else None
        )
        if deadline is not None and not deadline.may_start(predicted_group):
            raise deadline_stop(
                deadline,
                phase=ConstructionPhase.SUPPORT_DISCOVERY,
                reason="next destination support group cannot start safely",
                completed_units=group,
                total_units=int(inputs.group_dest_node.shape[0]),
                next_resumable_position=f"group-{group:06d}",
                checkpoint_location=(
                    None if checkpoint_root is None else str(checkpoint_root)
                ),
                checkpoint_reusable=group > 0,
                predicted_next_seconds=predicted_group,
            )
        group_started = deadline.clock() if deadline is not None else perf_counter()
        cached = None
        group_path = (
            None
            if checkpoint_root is None
            else _group_checkpoint_path(checkpoint_root, group)
        )
        if group_path is not None and group_path.exists():
            try:
                cached = _load_group_checkpoint(
                    group_path,
                    group=group,
                    provenance_hash=str(checkpoint_provenance_hash),
                )
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                quarantine = group_path.with_name(
                    f"{group_path.name}.invalid-{os.getpid()}"
                )
                os.replace(group_path, quarantine)
        if cached is not None:
            (
                summary,
                group_free_rows,
                group_free_columns,
                group_fixed_rows,
                group_fixed_columns,
            ) = cached
            free_rows.append(group_free_rows)
            free_columns.append(group_free_columns)
            fixed_rows.append(group_fixed_rows)
            fixed_columns.append(group_fixed_columns)
            summaries.append(summary)
            total_origin_specific += summary.origin_specific_entries
            total_group_bound += summary.group_level_candidate_entries
            recent_group_seconds.append(0.0)
            if reporter is not None:
                reporter.emit(
                    phase=ConstructionPhase.SUPPORT_DISCOVERY,
                    status="running",
                    force=True,
                    completed_units=group + 1,
                    total_units=int(inputs.group_dest_node.shape[0]),
                    current_unit=f"group-{group:06d}",
                    recent_unit_seconds=0.0,
                    checkpoint_location=str(checkpoint_root),
                    cache_hits=1,
                    cache_misses=0,
                )
            continue
        active_indices = group_indices[group][group_masks[group]]
        active_indices = active_indices[selected[active_indices]].astype(
            np.int64, copy=False
        )
        group_probability, group_effective = routing_for_group(group)
        enabled = group_effective & (
            group_probability > config.probability_tolerance
        )
        eligible_mapping = enabled[mapping_links]
        group_measurements = np.unique(mapping_measurements[eligible_mapping])
        group_bound = int(active_indices.size * group_measurements.size)
        group_entries = 0
        group_free = int(np.count_nonzero(free_column[active_indices] >= 0))
        group_fixed = int(np.count_nonzero(fixed_column[active_indices] >= 0))
        group_free_rows_parts: list[np.ndarray] = []
        group_free_column_parts: list[np.ndarray] = []
        group_fixed_rows_parts: list[np.ndarray] = []
        group_fixed_column_parts: list[np.ndarray] = []
        for first in range(0, active_indices.size, config.origin_chunk_size):
            if deadline is not None and deadline.expired:
                raise deadline_stop(
                    deadline,
                    phase=ConstructionPhase.SUPPORT_DISCOVERY,
                    reason="deadline reached inside a destination support group",
                    completed_units=group,
                    total_units=int(inputs.group_dest_node.shape[0]),
                    next_resumable_position=f"group-{group:06d}",
                    checkpoint_location=(
                        None if checkpoint_root is None else str(checkpoint_root)
                    ),
                    checkpoint_reusable=group > 0,
                )
            chunk = active_indices[first : first + config.origin_chunk_size]
            start = perf_counter()
            reachable = _chunk_reachability(
                origins=origins_by_active[chunk],
                enabled_links=enabled,
                topo_order=topo,
                out_links=out_links,
                out_mask=out_mask,
                head=head,
                num_nodes=num_nodes,
            )
            reachability_seconds += perf_counter() - start
            start = perf_counter()
            mapped_support = (
                reachable[:, tail[mapping_links]] & eligible_mapping[None, :]
            )
            for local_row, active_index in enumerate(chunk):
                rows = np.unique(mapping_measurements[mapped_support[local_row]])
                group_entries += int(rows.size)
                if not config.materialize:
                    continue
                free = free_column[active_index]
                if free >= 0:
                    group_free_rows_parts.append(rows)
                    group_free_column_parts.append(
                        np.full(rows.size, free, dtype=np.int64)
                    )
                else:
                    fixed = fixed_column[active_index]
                    group_fixed_rows_parts.append(rows)
                    group_fixed_column_parts.append(
                        np.full(rows.size, fixed, dtype=np.int64)
                    )
            projection_seconds += perf_counter() - start
        total_origin_specific += group_entries
        total_group_bound += group_bound
        summary = GroupOriginSupportSummary(
            group=group,
            selected_od_cells=int(active_indices.size),
            free_od_cells=group_free,
            positive_fixed_od_cells=group_fixed,
            group_measurements=int(group_measurements.size),
            group_level_candidate_entries=group_bound,
            origin_specific_entries=group_entries,
        )
        group_free_rows = (
            np.concatenate(group_free_rows_parts)
            if group_free_rows_parts
            else np.empty(0, dtype=np.int64)
        )
        group_free_columns = (
            np.concatenate(group_free_column_parts)
            if group_free_column_parts
            else np.empty(0, dtype=np.int64)
        )
        group_fixed_rows = (
            np.concatenate(group_fixed_rows_parts)
            if group_fixed_rows_parts
            else np.empty(0, dtype=np.int64)
        )
        group_fixed_columns = (
            np.concatenate(group_fixed_column_parts)
            if group_fixed_column_parts
            else np.empty(0, dtype=np.int64)
        )
        if checkpoint_root is not None:
            _save_group_checkpoint(
                directory=checkpoint_root,
                group=group,
                provenance_hash=str(checkpoint_provenance_hash),
                summary=summary,
                free_rows=group_free_rows,
                free_columns=group_free_columns,
                fixed_rows=group_fixed_rows,
                fixed_columns=group_fixed_columns,
            )
        free_rows.append(group_free_rows)
        free_columns.append(group_free_columns)
        fixed_rows.append(group_fixed_rows)
        fixed_columns.append(group_fixed_columns)
        summaries.append(summary)
        now = deadline.clock() if deadline is not None else perf_counter()
        recent_group_seconds.append(max(0.0, now - group_started))
        if reporter is not None:
            reporter.emit(
                phase=ConstructionPhase.SUPPORT_DISCOVERY,
                status="running",
                force=True,
                completed_units=group + 1,
                total_units=int(inputs.group_dest_node.shape[0]),
                current_unit=f"group-{group:06d}",
                recent_unit_seconds=recent_group_seconds[-1],
                predicted_remaining_seconds=(
                    recent_group_seconds[-1]
                    * (int(inputs.group_dest_node.shape[0]) - group - 1)
                ),
                checkpoint_location=str(checkpoint_root),
                cache_hits=0,
                cache_misses=1,
            )
    if config.materialize and total_origin_specific > config.max_materialized_entries:
        raise MemoryError(
            f"origin-specific support has {total_origin_specific} entries, exceeding "
            "max_materialized_entries="
            f"{config.max_materialized_entries}"
        )
    canonical_start = perf_counter()

    def build_matrix(
        rows: list[np.ndarray], columns: list[np.ndarray], shape: tuple[int, int]
    ) -> sparse.csr_array:
        row = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
        column = (
            np.concatenate(columns) if columns else np.empty(0, dtype=np.int64)
        )
        matrix = sparse.csr_array(
            (np.ones(row.size, dtype=bool), (row, column)), shape=shape
        )
        matrix.sum_duplicates()
        matrix.data[:] = True
        matrix.sort_indices()
        return matrix

    free_support = (
        build_matrix(free_rows, free_columns, (num_measurements, num_free))
        if config.materialize
        else None
    )
    fixed_support = (
        build_matrix(
            fixed_rows,
            fixed_columns,
            (num_measurements, positive_fixed.size),
        )
        if config.materialize
        else None
    )
    canonical_seconds = perf_counter() - canonical_start
    summary_tuple = tuple(summaries)
    reduction = (
        0.0
        if total_group_bound == 0
        else 1.0 - total_origin_specific / total_group_bound
    )
    metrics = OriginSupportMetrics(
        support_discovery_seconds=perf_counter() - total_start,
        reachability_seconds=reachability_seconds,
        measurement_projection_seconds=projection_seconds,
        canonicalization_seconds=canonical_seconds,
        estimated_peak_working_bytes=estimated_peak,
        group_level_candidate_entries=total_group_bound,
        origin_specific_entries=total_origin_specific,
        reduction_fraction=reduction,
        free_support_entries=(0 if free_support is None else int(free_support.nnz)),
        positive_fixed_support_entries=(
            0 if fixed_support is None else int(fixed_support.nnz)
        ),
    )
    positive_fixed = np.array(positive_fixed, copy=True)
    positive_fixed.setflags(write=False)
    return OriginSpecificMeasurementSupport(
        num_measurements=num_measurements,
        num_free_od=num_free,
        free_support=free_support,
        positive_fixed_support=fixed_support,
        positive_fixed_active_indices=positive_fixed,
        summaries=summary_tuple,
        metrics=metrics,
        fingerprint=_support_fingerprint(free_support, fixed_support, summary_tuple),
    )


def validate_origin_support_against_operator(
    *,
    support: OriginSpecificMeasurementSupport,
    operator: FixedRoutingMeasurementOperator,
    zero_tolerance: float = 0.0,
) -> OriginSupportValidation:
    """Prove that discovered support contains every realized operator entry."""
    if not support.materialized:
        raise ValueError("support must be materialized for numerical validation.")
    if operator.matrix.shape != (support.num_measurements, support.num_free_od):
        raise ValueError("support and operator dimensions differ.")
    if isinstance(operator.matrix, jsparse.BCOO):
        data = np.asarray(operator.matrix.data)
        indices = np.asarray(operator.matrix.indices)
        retained = np.abs(data) > zero_tolerance
        realized_rows = indices[retained, 0]
        realized_columns = indices[retained, 1]
    else:
        realized_rows, realized_columns = np.nonzero(
            np.abs(np.asarray(operator.matrix)) > zero_tolerance
        )
    free_support = support.free_support
    assert free_support is not None
    covered = np.asarray(
        free_support[realized_rows, realized_columns]
    ).reshape(-1).astype(bool)
    missing_free = int(np.count_nonzero(~covered))
    realized_free = int(realized_rows.size)

    offset_rows = np.flatnonzero(
        np.abs(np.asarray(operator.fixed_measurement_offset)) > zero_tolerance
    )
    fixed_support = support.positive_fixed_support
    assert fixed_support is not None
    fixed_union = np.diff(fixed_support.indptr) > 0
    missing_fixed = int(np.count_nonzero(~fixed_union[offset_rows]))
    free_false_positive = max(0, int(free_support.nnz) - realized_free)
    fixed_false_positive = max(0, int(np.count_nonzero(fixed_union)) - offset_rows.size)
    return OriginSupportValidation(
        realized_free_entries=realized_free,
        realized_fixed_offset_entries=int(offset_rows.size),
        missing_free_entries=missing_free,
        missing_fixed_offset_entries=missing_fixed,
        free_false_positive_entries=free_false_positive,
        fixed_false_positive_entries=fixed_false_positive,
    )
