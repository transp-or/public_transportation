"""Budgeted selection and construction of representative OD-block operators."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .blocks import ODBlock
from .operator import BlockLinearOperatorProtocol
from .partition import ODBlockPartition
from .support_preflight import (
    BlockSupportSummary,
    SupportPreflightBudget,
    SupportPreflightResult,
)


@dataclass(frozen=True, slots=True)
class SelectedBlockResourceEstimate:
    """Conservative storage and workspace estimate checked before construction."""

    block_id: str
    sparse_storage_bytes: int
    transpose_storage_bytes: int
    construction_temporary_bytes: int
    solver_working_bytes: int
    retained_cache_bytes: int
    worker_high_water_bytes: int


@dataclass(frozen=True, slots=True)
class SelectedBlockConstructionMeasurement:
    """Observed construction and product behavior for one selected block."""

    block_id: str
    estimate: SelectedBlockResourceEstimate
    construction_seconds: float
    forward_seconds: float
    transpose_seconds: float
    exact_nonzeros: int
    exact_support_rows: int
    resident_bytes: int
    disk_bytes: int
    cache_hit: bool
    forward_checksum: float
    transpose_checksum: float


class BlockConstructionResourceError(RuntimeError):
    """Raised before a selected block would exceed its configured budget."""


def select_representative_block_ids(
    result: SupportPreflightResult,
    *,
    explicit_block_ids: Iterable[str] = (),
) -> tuple[str, ...]:
    """Select small, median, p95, and largest observed blocks deterministically."""
    summaries = sorted(
        result.block_summaries,
        key=lambda item: (item.estimated_operator_bytes, item.block_id),
    )
    if not summaries:
        raise ValueError("preflight contains no exact block summaries.")
    known = {item.block_id for item in summaries}
    explicit = tuple(dict.fromkeys(str(value) for value in explicit_block_ids))
    missing = [value for value in explicit if value not in known]
    if missing:
        raise ValueError(f"unknown selected block IDs: {missing}.")
    positions = (
        0,
        round((len(summaries) - 1) * 0.5),
        round((len(summaries) - 1) * 0.95),
        len(summaries) - 1,
    )
    return tuple(
        dict.fromkeys(
            [summaries[position].block_id for position in positions] + list(explicit)
        )
    )


def estimate_selected_block_resources(
    summary: BlockSupportSummary,
    *,
    dtype: np.dtype | str = np.float64,
) -> SelectedBlockResourceEstimate:
    """Estimate CSR, CSC, construction, cache, and solver storage."""
    value_bytes = np.dtype(dtype).itemsize
    index_bytes = np.dtype(np.int64).itemsize
    csr = summary.exact_nonzeros * (value_bytes + index_bytes)
    csr += (summary.measurement_support_rows + 1) * index_bytes
    csc = summary.exact_nonzeros * (value_bytes + index_bytes)
    csc += (summary.free_columns + 1) * index_bytes
    construction = csr + csc + summary.exact_nonzeros * (value_bytes + 2 * index_bytes)
    solver = (3 * summary.free_columns + summary.measurement_support_rows) * value_bytes
    retained = csr + csc
    return SelectedBlockResourceEstimate(
        block_id=summary.block_id,
        sparse_storage_bytes=csr,
        transpose_storage_bytes=csc,
        construction_temporary_bytes=construction,
        solver_working_bytes=solver,
        retained_cache_bytes=retained,
        worker_high_water_bytes=construction + retained + solver,
    )


def construct_selected_block_operators(
    *,
    result: SupportPreflightResult,
    partition: ODBlockPartition,
    builder: Callable[[ODBlock], BlockLinearOperatorProtocol],
    budget: SupportPreflightBudget,
    block_ids: Iterable[str] = (),
) -> tuple[SelectedBlockConstructionMeasurement, ...]:
    """Construct only selected blocks after conservative pre-allocation checks."""
    selected = select_representative_block_ids(result, explicit_block_ids=block_ids)
    summaries = {item.block_id: item for item in result.block_summaries}
    blocks = {item.block_id: item for item in partition.blocks}
    measurements: list[SelectedBlockConstructionMeasurement] = []
    for block_id in selected:
        summary = summaries[block_id]
        block = blocks.get(block_id)
        if block is None:
            raise ValueError(f"selected block {block_id!r} is absent from partition.")
        estimate = estimate_selected_block_resources(summary)
        if summary.measurement_support_rows > budget.maximum_support_rows_per_block:
            raise BlockConstructionResourceError(
                f"block {block_id!r} exceeds the support-row budget."
            )
        if summary.exact_nonzeros > budget.maximum_nonzeros_per_block:
            raise BlockConstructionResourceError(
                f"block {block_id!r} exceeds the nonzero budget."
            )
        if (
            estimate.retained_cache_bytes > budget.maximum_block_operator_bytes
            or estimate.worker_high_water_bytes > budget.maximum_temporary_bytes
        ):
            raise BlockConstructionResourceError(
                f"block {block_id!r} exceeds the block or worker-memory budget."
            )
        started = perf_counter()
        operator = builder(block)
        construction_seconds = perf_counter() - started
        expected_shape = (operator.num_measurements, block.num_free_variables)
        if operator.shape != expected_shape:
            raise ValueError(
                f"constructed block {block_id!r} has inconsistent shape {operator.shape}."
            )
        vector = np.linspace(0.25, 1.25, block.num_free_variables)
        measurement_vector = np.linspace(-0.5, 0.5, operator.num_measurements)
        started = perf_counter()
        forward = operator.matvec(vector)
        forward_seconds = perf_counter() - started
        started = perf_counter()
        transpose = operator.rmatvec(measurement_vector)
        transpose_seconds = perf_counter() - started
        preparation = getattr(operator, "preparation_metrics", None)
        builder_result = getattr(builder, "last_result", None)
        if (
            preparation is None
            and builder_result is not None
            and getattr(builder_result, "block_id", None) == block_id
        ):
            preparation = builder_result
        matrix = getattr(operator, "matrix", None)
        compact_matrix = getattr(operator, "compact_matrix", None)
        observed_nonzeros = int(
            getattr(
                preparation,
                "exact_nonzeros",
                getattr(
                    preparation,
                    "nonzero_entries",
                    getattr(
                        compact_matrix,
                        "nnz",
                        getattr(matrix, "nnz", summary.exact_nonzeros),
                    ),
                ),
            )
        )
        observed_resident = int(
            getattr(
                preparation,
                "resident_bytes",
                getattr(
                    preparation,
                    "retained_bytes",
                    getattr(operator, "retained_bytes", estimate.retained_cache_bytes),
                ),
            )
        )
        if preparation is None and matrix is not None:
            transpose_matrix = getattr(operator, "transpose_matrix", None)
            observed_resident = sum(
                array.nbytes
                for storage in (matrix, transpose_matrix)
                if storage is not None
                for array in (storage.data, storage.indices, storage.indptr)
            )
        measurements.append(
            SelectedBlockConstructionMeasurement(
                block_id=block_id,
                estimate=estimate,
                construction_seconds=construction_seconds,
                forward_seconds=forward_seconds,
                transpose_seconds=transpose_seconds,
                exact_nonzeros=observed_nonzeros,
                exact_support_rows=len(operator.measurement_support_indices),
                resident_bytes=observed_resident,
                disk_bytes=int(getattr(preparation, "disk_bytes", 0)),
                cache_hit=bool(getattr(preparation, "cache_hit", False)),
                forward_checksum=float(np.sum(forward)),
                transpose_checksum=float(np.sum(transpose)),
            )
        )
    return tuple(measurements)
