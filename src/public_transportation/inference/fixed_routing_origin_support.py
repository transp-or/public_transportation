"""Read-only origin-specific support analysis for fixed-routing measurements."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from jax.experimental import sparse as jsparse
from scipy import sparse

from .assignment_adapter import (
    AssignmentInputs,
    FixedRoutingInputs,
    validate_fixed_routing_compatibility,
)
from .compact_od_assignment_layout import CompactODAssignmentLayout
from .fixed_routing_measurement_operator import FixedRoutingMeasurementOperator


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
    routing: FixedRoutingInputs,
    spec,
    compact_layout: CompactODAssignmentLayout,
    config: OriginSupportConfig | None = None,
) -> OriginSpecificMeasurementSupport:
    """Discover support without evaluating passenger-flow values."""
    config = OriginSupportConfig() if config is None else config
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
    probabilities = np.asarray(routing.group_link_probability)
    effective = np.asarray(routing.effective_group_link_mask, dtype=bool)

    free_rows: list[np.ndarray] = []
    free_columns: list[np.ndarray] = []
    fixed_rows: list[np.ndarray] = []
    fixed_columns: list[np.ndarray] = []
    summaries: list[GroupOriginSupportSummary] = []
    reachability_seconds = projection_seconds = 0.0
    total_start = perf_counter()
    total_origin_specific = total_group_bound = 0
    for group in range(int(inputs.group_dest_node.shape[0])):
        active_indices = group_indices[group][group_masks[group]]
        active_indices = active_indices[selected[active_indices]].astype(
            np.int64, copy=False
        )
        enabled = effective[group] & (
            probabilities[group] > config.probability_tolerance
        )
        eligible_mapping = enabled[mapping_links]
        group_measurements = np.unique(mapping_measurements[eligible_mapping])
        group_bound = int(active_indices.size * group_measurements.size)
        group_entries = 0
        group_free = int(np.count_nonzero(free_column[active_indices] >= 0))
        group_fixed = int(np.count_nonzero(fixed_column[active_indices] >= 0))
        for first in range(0, active_indices.size, config.origin_chunk_size):
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
                    free_rows.append(rows)
                    free_columns.append(np.full(rows.size, free, dtype=np.int64))
                else:
                    fixed = fixed_column[active_index]
                    fixed_rows.append(rows)
                    fixed_columns.append(np.full(rows.size, fixed, dtype=np.int64))
            projection_seconds += perf_counter() - start
        total_origin_specific += group_entries
        total_group_bound += group_bound
        summaries.append(
            GroupOriginSupportSummary(
                group=group,
                selected_od_cells=int(active_indices.size),
                free_od_cells=group_free,
                positive_fixed_od_cells=group_fixed,
                group_measurements=int(group_measurements.size),
                group_level_candidate_entries=group_bound,
                origin_specific_entries=group_entries,
            )
        )
    if config.materialize and total_origin_specific > config.max_materialized_entries:
        raise MemoryError(
            "origin-specific support exceeds max_materialized_entries; rerun with "
            "materialize=False for summary-only analysis"
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
