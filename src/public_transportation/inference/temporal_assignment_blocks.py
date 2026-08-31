"""Exact sparse temporal assignment blocks and bounded support profiling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .assignment_contract import (
    AssignmentArtifactIdentity,
    AssignmentCompatibilityError,
    CanonicalAssignmentIndex,
)
from .scheduled_reference_operator import ScheduledTimeExpandedReferenceOperator
from .construction_control import estimate_completed_unit_eta


@dataclass(frozen=True, slots=True, order=True)
class TemporalBlockKey:
    """Measurement interval and departure interval identifying one block."""

    measurement_interval_id: str
    departure_interval_id: str


@dataclass(frozen=True, slots=True)
class TemporalSupportProfileConfig:
    """Conservative resource assumptions for temporal-block preflight."""

    maximum_journey_duration_seconds: int
    value_bytes: int = 4
    index_bytes: int = 4

    def __post_init__(self) -> None:
        if self.maximum_journey_duration_seconds <= 0:
            raise ValueError("maximum journey duration must be positive.")
        if self.value_bytes <= 0 or self.index_bytes <= 0:
            raise ValueError("value and index byte sizes must be positive.")


@dataclass(frozen=True, slots=True)
class TemporalBlockSupportEstimate:
    key: TemporalBlockKey
    demand_columns: int
    measurement_rows: int
    candidate_entries: int
    projected_storage_bytes: int


@dataclass(frozen=True, slots=True)
class TemporalBlockSupportProfile:
    blocks: tuple[TemporalBlockSupportEstimate, ...]
    total_candidate_entries: int
    projected_storage_bytes: int
    dense_candidate_entries: int
    excluded_by_temporal_structure: int


def profile_temporal_block_support(
    *,
    canonical_index: CanonicalAssignmentIndex,
    config: TemporalSupportProfileConfig,
) -> TemporalBlockSupportProfile:
    """Return a safe temporal upper bound without running assignment."""
    intervals = {item.interval_id: item for item in canonical_index.time_intervals}
    demand_counts = {key: 0 for key in intervals}
    measurement_counts = {key: 0 for key in intervals}
    for cell in canonical_index.demand_cells:
        if cell.operator_column is not None:
            demand_counts[cell.departure_interval_id] += 1
    for measurement in canonical_index.measurements:
        measurement_counts[measurement.interval_id] += 1

    estimates = []
    for departure_id, departure_count in demand_counts.items():
        departure = intervals[departure_id]
        latest_effect = departure.end_seconds + config.maximum_journey_duration_seconds
        for measurement_id, measurement_count in measurement_counts.items():
            measurement = intervals[measurement_id]
            temporally_possible = (
                measurement.end_seconds > departure.start_seconds
                and measurement.start_seconds < latest_effect
            )
            if not temporally_possible or not departure_count or not measurement_count:
                continue
            entries = departure_count * measurement_count
            estimates.append(
                TemporalBlockSupportEstimate(
                    key=TemporalBlockKey(measurement_id, departure_id),
                    demand_columns=departure_count,
                    measurement_rows=measurement_count,
                    candidate_entries=entries,
                    projected_storage_bytes=entries
                    * (config.value_bytes + 2 * config.index_bytes),
                )
            )
    estimates.sort(key=lambda item: item.key)
    dense = (
        canonical_index.number_of_demand_cells
        * canonical_index.number_of_measurements
    )
    total = sum(item.candidate_entries for item in estimates)
    return TemporalBlockSupportProfile(
        blocks=tuple(estimates),
        total_candidate_entries=total,
        projected_storage_bytes=sum(
            item.projected_storage_bytes for item in estimates
        ),
        dense_candidate_entries=dense,
        excluded_by_temporal_structure=dense - total,
    )


def _canonical_triplets(
    rows: object, columns: object, values: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row = np.asarray(rows, dtype=np.int32)
    column = np.asarray(columns, dtype=np.int32)
    value = np.asarray(values)
    if row.ndim != 1 or column.shape != row.shape or value.shape != row.shape:
        raise ValueError("sparse block triplets must be aligned one-dimensional arrays.")
    if not row.size:
        return row, column, value
    order = np.lexsort((column, row))
    row, column, value = row[order], column[order], value[order]
    starts = np.r_[True, (row[1:] != row[:-1]) | (column[1:] != column[:-1])]
    positions = np.flatnonzero(starts)
    value = np.add.reduceat(value, positions)
    row, column = row[positions], column[positions]
    nonzero = value != 0
    row, column, value = row[nonzero], column[nonzero], value[nonzero]
    row.setflags(write=False)
    column.setflags(write=False)
    value.setflags(write=False)
    return row, column, value


@dataclass(frozen=True, slots=True)
class TemporalSparseBlock:
    key: TemporalBlockKey
    row_indices: np.ndarray
    column_indices: np.ndarray
    values: np.ndarray
    number_of_measurements: int
    number_of_demand_cells: int

    def __post_init__(self) -> None:
        rows, columns, values = _canonical_triplets(
            self.row_indices, self.column_indices, self.values
        )
        if rows.size and (
            np.any(rows < 0)
            or np.any(rows >= self.number_of_measurements)
            or np.any(columns < 0)
            or np.any(columns >= self.number_of_demand_cells)
        ):
            raise ValueError("temporal block indices are out of bounds.")
        if not np.all(np.isfinite(values)):
            raise ValueError("temporal block values must be finite.")
        object.__setattr__(self, "row_indices", rows)
        object.__setattr__(self, "column_indices", columns)
        object.__setattr__(self, "values", values)

    @property
    def nonzero_entries(self) -> int:
        return int(self.values.size)


@dataclass(frozen=True, slots=True)
class TemporalBlockConstructionProgress:
    completed_columns: int
    total_columns: int
    elapsed_seconds: float
    predicted_remaining_seconds: float | None
    nonzero_entries: int
    schema_version: int = 1
    phase: str = "temporal_block_assembly"
    status: str = "running"
    recent_unit_seconds: float | None = None
    eta_confidence: str = "unavailable"
    eta_reason: str | None = None
    estimated_completion_at_utc: str | None = None
    eta_lower_seconds: float | None = None
    eta_upper_seconds: float | None = None
    throughput_units_per_second: float | None = None


TemporalBlockProgressCallback = Callable[[TemporalBlockConstructionProgress], None]


def _emit_temporal_progress(
    callback: TemporalBlockProgressCallback | None,
    event: TemporalBlockConstructionProgress,
) -> None:
    """Deliver legacy progress while shielding file/socket sink failures.

    A callback may still deliberately raise a non-I/O exception to stop a
    construction and resume from its checkpoint.  I/O failures are telemetry
    failures and must not alter the numerical result.
    """

    if callback is None:
        return
    try:
        callback(event)
    except OSError:
        return


@dataclass(frozen=True, slots=True)
class TemporalBlockConstructionDiagnostics:
    construction_seconds: float
    nonzero_entries: int
    retained_l1_mass: float
    removed_l1_mass: float
    zero_tolerance: float
    columns_processed: int
    compilation_count: int = 0
    compilation_seconds: float = 0.0
    execution_seconds: float = 0.0
    device_transfer_seconds: float = 0.0
    num_chunks: int = 0
    chunk_shape: tuple[int, int] = (0, 0)


@dataclass(frozen=True, slots=True)
class TemporalBlockAssignmentOperator:
    """Exact in-memory sparse temporal-block assignment operator."""

    canonical_index: CanonicalAssignmentIndex
    identity: AssignmentArtifactIdentity
    blocks: tuple[TemporalSparseBlock, ...]
    fixed_measurement_offset: np.ndarray
    diagnostics: TemporalBlockConstructionDiagnostics

    def __post_init__(self) -> None:
        if (
            self.identity.canonical_index_fingerprint
            != self.canonical_index.artifact_fingerprint
        ):
            raise AssignmentCompatibilityError(
                "temporal blocks and canonical index have different identities."
            )
        # One logical temporal key may be partitioned across independently
        # checkpointed construction fragments. Products sum those partitions.
        for block in self.blocks:
            if block.number_of_measurements != self.number_of_measurements:
                raise ValueError("temporal block measurement dimension differs.")
            if block.number_of_demand_cells != self.number_of_demand_cells:
                raise ValueError("temporal block demand dimension differs.")
        offset = np.array(self.fixed_measurement_offset, copy=True)
        if offset.shape != (self.number_of_measurements,) or not np.all(
            np.isfinite(offset)
        ):
            raise ValueError("fixed measurement offset is invalid.")
        offset.setflags(write=False)
        object.__setattr__(self, "fixed_measurement_offset", offset)

    @property
    def number_of_demand_cells(self) -> int:
        return self.canonical_index.number_of_demand_cells

    @property
    def number_of_measurements(self) -> int:
        return self.canonical_index.number_of_measurements

    @property
    def canonical_index_fingerprint(self) -> str:
        return self.canonical_index.artifact_fingerprint

    @property
    def artifact_fingerprint(self) -> str:
        return self.identity.fingerprint

    @property
    def representation(self) -> str:
        return "sparse_temporal_blocks"

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.identity.numeric_dtype)

    def matvec(self, demand: object) -> jax.Array:
        value = jnp.asarray(demand, dtype=self.dtype)
        if value.shape != (self.number_of_demand_cells,):
            raise ValueError("demand has an incompatible shape.")
        result = jnp.zeros((self.number_of_measurements,), dtype=self.dtype)
        for block in self.blocks:
            result = result.at[jnp.asarray(block.row_indices)].add(
                jnp.asarray(block.values, dtype=self.dtype)
                * value[jnp.asarray(block.column_indices)]
            )
        return result

    def rmatvec(self, residual: object) -> jax.Array:
        value = jnp.asarray(residual, dtype=self.dtype)
        if value.shape != (self.number_of_measurements,):
            raise ValueError("residual has an incompatible shape.")
        result = jnp.zeros((self.number_of_demand_cells,), dtype=self.dtype)
        for block in self.blocks:
            result = result.at[jnp.asarray(block.column_indices)].add(
                jnp.asarray(block.values, dtype=self.dtype)
                * value[jnp.asarray(block.row_indices)]
            )
        return result

    def jax_matvec(self, demand: object) -> jax.Array:
        return self.matvec(demand)

    def jax_rmatvec(self, residual: object) -> jax.Array:
        return self.rmatvec(residual)


def build_exact_temporal_block_operator(
    *,
    reference: ScheduledTimeExpandedReferenceOperator,
    zero_tolerance: float = 0.0,
    progress: TemporalBlockProgressCallback | None = None,
    absolute_deadline: float | None = None,
    clock: Callable[[], float] = perf_counter,
) -> TemporalBlockAssignmentOperator:
    """Construct exact sparse blocks directly from scheduled column responses."""
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and nonnegative.")
    started = clock()
    total = reference.number_of_demand_cells
    measurement_intervals = tuple(
        item.interval_id for item in reference.canonical_index.measurements
    )
    departure_by_column = {
        int(cell.operator_column): cell.departure_interval_id
        for cell in reference.canonical_index.demand_cells
        if cell.operator_column is not None
    }
    triplets: dict[TemporalBlockKey, tuple[list[int], list[int], list[float]]] = {}
    retained_l1 = 0.0
    removed_l1 = 0.0
    nonzeros = 0
    recent_durations: deque[float] = deque(maxlen=32)
    for column in range(total):
        if absolute_deadline is not None and clock() >= absolute_deadline:
            raise TimeoutError(
                f"temporal block construction stopped after {column}/{total} columns."
            )
        unit_started = clock() if progress is not None else 0.0
        basis = jnp.zeros((total,), dtype=reference.dtype).at[column].set(1.0)
        response = np.asarray(jax.block_until_ready(reference.matvec(basis)))
        for row in np.flatnonzero(response):
            coefficient = float(response[row])
            if abs(coefficient) <= zero_tolerance:
                removed_l1 += abs(coefficient)
                continue
            key = TemporalBlockKey(
                measurement_intervals[row], departure_by_column[column]
            )
            rows, columns, values = triplets.setdefault(key, ([], [], []))
            rows.append(int(row))
            columns.append(column)
            values.append(coefficient)
            retained_l1 += abs(coefficient)
            nonzeros += 1
        if progress is not None:
            elapsed = max(0.0, clock() - started)
            completed = column + 1
            unit_seconds = max(0.0, clock() - unit_started)
            if unit_seconds > 0.0:
                recent_durations.append(unit_seconds)
            eta = estimate_completed_unit_eta(
                recent_durations,
                completed_units=completed,
                total_units=total,
                parallelism=1,
                elapsed_seconds=elapsed,
            )
            _emit_temporal_progress(
                progress,
                TemporalBlockConstructionProgress(
                    completed_columns=completed,
                    total_columns=total,
                    elapsed_seconds=elapsed,
                    predicted_remaining_seconds=eta.predicted_remaining_seconds,
                    nonzero_entries=nonzeros,
                    status="completed" if completed == total else "running",
                    recent_unit_seconds=unit_seconds,
                    eta_confidence=eta.eta_confidence,
                    eta_reason=eta.eta_reason,
                    estimated_completion_at_utc=eta.estimated_completion_at_utc,
                    eta_lower_seconds=eta.eta_lower_seconds,
                    eta_upper_seconds=eta.eta_upper_seconds,
                    throughput_units_per_second=eta.throughput_units_per_second,
                ),
            )
    blocks = tuple(
        TemporalSparseBlock(
            key=key,
            row_indices=np.asarray(values[0], dtype=np.int32),
            column_indices=np.asarray(values[1], dtype=np.int32),
            values=np.asarray(values[2], dtype=reference.dtype),
            number_of_measurements=reference.number_of_measurements,
            number_of_demand_cells=reference.number_of_demand_cells,
        )
        for key, values in sorted(triplets.items())
    )
    return TemporalBlockAssignmentOperator(
        canonical_index=reference.canonical_index,
        identity=reference.identity,
        blocks=blocks,
        fixed_measurement_offset=np.asarray(reference.fixed_measurement_offset),
        diagnostics=TemporalBlockConstructionDiagnostics(
            construction_seconds=max(0.0, clock() - started),
            nonzero_entries=sum(block.nonzero_entries for block in blocks),
            retained_l1_mass=retained_l1,
            removed_l1_mass=removed_l1,
            zero_tolerance=zero_tolerance,
            columns_processed=total,
        ),
    )


def build_chunked_temporal_block_operator(
    *,
    reference: ScheduledTimeExpandedReferenceOperator,
    chunk_size: int,
    zero_tolerance: float = 0.0,
    progress: TemporalBlockProgressCallback | None = None,
    absolute_deadline: float | None = None,
    clock: Callable[[], float] = perf_counter,
) -> TemporalBlockAssignmentOperator:
    """Build sparse blocks with one compiled fixed-shape measurement kernel.

    The kernel performs scheduled assignment and measurement aggregation on the
    device.  Only bounded measurement-by-chunk responses are transferred to the
    host, and the final chunk is padded to preserve the compiled shape.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not np.isfinite(zero_tolerance) or zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and nonnegative.")
    started = clock()
    total = reference.number_of_demand_cells
    measurement_intervals = tuple(
        item.interval_id for item in reference.canonical_index.measurements
    )
    departure_by_column = {
        int(cell.operator_column): cell.departure_interval_id
        for cell in reference.canonical_index.demand_cells
        if cell.operator_column is not None
    }

    def measurement_chunk(batch: jax.Array) -> jax.Array:
        return jax.vmap(reference.matvec)(batch)

    compilation_started = clock()
    compiled = jax.jit(measurement_chunk).lower(
        jnp.zeros((chunk_size, total), dtype=reference.dtype)
    ).compile()
    compilation_seconds = max(0.0, clock() - compilation_started)
    triplets: dict[TemporalBlockKey, tuple[list[int], list[int], list[float]]] = {}
    retained_l1 = 0.0
    removed_l1 = 0.0
    nonzeros = 0
    execution_seconds = 0.0
    transfer_seconds = 0.0
    completed = 0
    num_chunks = 0
    recent_durations: deque[float] = deque(maxlen=32)
    for start in range(0, total, chunk_size):
        if absolute_deadline is not None and clock() >= absolute_deadline:
            raise TimeoutError(
                f"temporal block construction stopped after {completed}/{total} columns."
            )
        unit_started = clock() if progress is not None else 0.0
        stop = min(total, start + chunk_size)
        width = stop - start
        basis = np.zeros((chunk_size, total), dtype=reference.dtype)
        basis[np.arange(width), np.arange(start, stop)] = 1.0
        execution_started = clock()
        response_device = compiled(jnp.asarray(basis))
        jax.block_until_ready(response_device)
        execution_seconds += max(0.0, clock() - execution_started)
        transfer_started = clock()
        response = np.asarray(response_device)[:width]
        transfer_seconds += max(0.0, clock() - transfer_started)
        for local_column in range(width):
            column = start + local_column
            column_response = response[local_column]
            for row in np.flatnonzero(column_response):
                coefficient = float(column_response[row])
                if abs(coefficient) <= zero_tolerance:
                    removed_l1 += abs(coefficient)
                    continue
                key = TemporalBlockKey(
                    measurement_intervals[row], departure_by_column[column]
                )
                rows, columns, values = triplets.setdefault(key, ([], [], []))
                rows.append(int(row))
                columns.append(column)
                values.append(coefficient)
                retained_l1 += abs(coefficient)
                nonzeros += 1
        completed = stop
        num_chunks += 1
        if progress is not None:
            elapsed = max(0.0, clock() - started)
            unit_seconds = max(0.0, clock() - unit_started)
            if unit_seconds > 0.0:
                recent_durations.append(unit_seconds)
            eta = estimate_completed_unit_eta(
                recent_durations,
                completed_units=completed,
                total_units=total,
                parallelism=1,
                elapsed_seconds=elapsed,
            )
            _emit_temporal_progress(
                progress,
                TemporalBlockConstructionProgress(
                    completed_columns=completed,
                    total_columns=total,
                    elapsed_seconds=elapsed,
                    predicted_remaining_seconds=eta.predicted_remaining_seconds,
                    nonzero_entries=nonzeros,
                    status="completed" if completed == total else "running",
                    recent_unit_seconds=unit_seconds,
                    eta_confidence=eta.eta_confidence,
                    eta_reason=eta.eta_reason,
                    estimated_completion_at_utc=eta.estimated_completion_at_utc,
                    eta_lower_seconds=eta.eta_lower_seconds,
                    eta_upper_seconds=eta.eta_upper_seconds,
                    throughput_units_per_second=eta.throughput_units_per_second,
                ),
            )
    blocks = tuple(
        TemporalSparseBlock(
            key=key,
            row_indices=np.asarray(values[0], dtype=np.int32),
            column_indices=np.asarray(values[1], dtype=np.int32),
            values=np.asarray(values[2], dtype=reference.dtype),
            number_of_measurements=reference.number_of_measurements,
            number_of_demand_cells=reference.number_of_demand_cells,
        )
        for key, values in sorted(triplets.items())
    )
    return TemporalBlockAssignmentOperator(
        canonical_index=reference.canonical_index,
        identity=reference.identity,
        blocks=blocks,
        fixed_measurement_offset=np.asarray(reference.fixed_measurement_offset),
        diagnostics=TemporalBlockConstructionDiagnostics(
            construction_seconds=max(0.0, clock() - started),
            nonzero_entries=sum(block.nonzero_entries for block in blocks),
            retained_l1_mass=retained_l1,
            removed_l1_mass=removed_l1,
            zero_tolerance=zero_tolerance,
            columns_processed=total,
            compilation_count=1,
            compilation_seconds=compilation_seconds,
            execution_seconds=execution_seconds,
            device_transfer_seconds=transfer_seconds,
            num_chunks=num_chunks,
            chunk_shape=(chunk_size, reference.number_of_measurements),
        ),
    )
