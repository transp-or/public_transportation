"""Preparation and selection of fixed-routing linear measurement backends."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Literal

import numpy as np
from scipy import sparse

from .fixed_routing_matrix_free_operator import (
    MatrixFreeFixedRoutingMeasurementOperator,
)
from .fixed_routing_measurement_operator import (
    FixedRoutingMeasurementOperator,
    fixed_routing_measurement_operator_cache_path,
    load_valid_cached_fixed_routing_measurement_operator,
    prepare_fixed_routing_measurement_operator,
    save_fixed_routing_measurement_operator,
)
from .linear_operator import LinearOperatorProtocol, SparseLinearOperator

LinearOperatorMode = Literal["matrix_free", "sparse", "auto"]


def _available_memory_bytes() -> int | None:
    """Return an OS estimate without adding an optional runtime dependency."""
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    value = pages * page_size
    return value if value > 0 else None


@dataclass(frozen=True, slots=True)
class SparseOperatorSelectionConfig:
    """Memory and reuse assumptions used before explicit allocation."""

    mode: LinearOperatorMode = "auto"
    memory_budget_bytes: int | None = None
    memory_fraction: float = 0.25
    estimated_density: float = 0.1
    expected_matvec_calls: int = 20
    expected_rmatvec_calls: int = 20
    minimum_reuse_products: int = 4
    estimated_construction_seconds: float | None = None
    matrix_free_product_seconds: float | None = None
    sparse_product_seconds: float = 0.0
    zero_tolerance: float = 0.0
    chunk_size: int = 128

    def __post_init__(self) -> None:
        if self.mode not in {"matrix_free", "sparse", "auto"}:
            raise ValueError("mode must be 'matrix_free', 'sparse', or 'auto'.")
        if self.memory_budget_bytes is not None and self.memory_budget_bytes <= 0:
            raise ValueError("memory_budget_bytes must be positive when provided.")
        if not math.isfinite(self.memory_fraction) or not 0 < self.memory_fraction <= 1:
            raise ValueError("memory_fraction must lie in (0, 1].")
        if not math.isfinite(self.estimated_density) or not 0 <= self.estimated_density <= 1:
            raise ValueError("estimated_density must lie in [0, 1].")
        if self.expected_matvec_calls < 0 or self.expected_rmatvec_calls < 0:
            raise ValueError("expected product counts must be non-negative.")
        if self.minimum_reuse_products < 0:
            raise ValueError("minimum_reuse_products must be non-negative.")
        for name in (
            "estimated_construction_seconds",
            "matrix_free_product_seconds",
            "sparse_product_seconds",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative.")
        if not math.isfinite(self.zero_tolerance) or self.zero_tolerance < 0:
            raise ValueError("zero_tolerance must be finite and non-negative.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")


@dataclass(frozen=True, slots=True)
class SparseOperatorSelection:
    requested_mode: LinearOperatorMode
    selected_mode: Literal["matrix_free", "sparse"]
    reason: str
    cache_available: bool
    rows: int
    columns: int
    logical_entries: int
    dense_bytes: int
    estimated_nonzero_entries: int
    estimated_sparse_solver_bytes: int
    available_memory_bytes: int | None
    memory_budget_bytes: int
    expected_product_calls: int
    estimated_construction_seconds: float | None
    estimated_product_savings_seconds: float | None
    estimated_break_even_products: float | None


@dataclass(frozen=True, slots=True)
class LinearMeasurementBackendMetrics:
    cache_lookup_seconds: float
    cache_load_seconds: float
    operator_construction_seconds: float
    cache_persistence_seconds: float
    device_transfer_and_sparse_conversion_seconds: float
    total_preparation_seconds: float
    fixed_routing_preparation_seconds: float
    cache_hit: bool
    representation: str
    nonzero_entries: int | None
    solver_storage_bytes: int | None


@dataclass(frozen=True, slots=True)
class PreparedLinearMeasurementBackend:
    operator: LinearOperatorProtocol
    fixed_measurement_offset: np.ndarray
    selection: SparseOperatorSelection
    metrics: LinearMeasurementBackendMetrics
    persisted_operator: FixedRoutingMeasurementOperator | None = None


def select_fixed_routing_linear_operator(
    *,
    rows: int,
    columns: int,
    dtype: object,
    cache_available: bool,
    config: SparseOperatorSelectionConfig,
    available_memory_bytes: int | None = None,
) -> SparseOperatorSelection:
    """Select a backend without allocating the explicit matrix."""
    if rows <= 0 or columns <= 0:
        raise ValueError("operator dimensions must be strictly positive.")
    dtype = np.dtype(dtype)
    logical_entries = int(rows * columns)
    dense_bytes = logical_entries * dtype.itemsize
    estimated_nnz = min(
        logical_entries, int(math.ceil(logical_entries * config.estimated_density))
    )
    # Persistent CSR and CSC: values + int32 indices in each, plus pointers.
    estimated_sparse_bytes = int(
        2 * estimated_nnz * (dtype.itemsize + np.dtype(np.int32).itemsize)
        + (rows + columns + 2) * np.dtype(np.int32).itemsize
    )
    available = (
        _available_memory_bytes()
        if available_memory_bytes is None
        else available_memory_bytes
    )
    if config.memory_budget_bytes is not None:
        budget = config.memory_budget_bytes
    elif available is not None:
        budget = max(1, int(available * config.memory_fraction))
    else:
        budget = 512 * 1024 * 1024
    expected = config.expected_matvec_calls + config.expected_rmatvec_calls
    safe = estimated_sparse_bytes <= budget
    saving_per_product = (
        None
        if config.matrix_free_product_seconds is None
        else config.matrix_free_product_seconds - config.sparse_product_seconds
    )
    estimated_savings = (
        None if saving_per_product is None else expected * saving_per_product
    )
    break_even = (
        None
        if config.estimated_construction_seconds is None
        or saving_per_product is None
        or saving_per_product <= 0
        else config.estimated_construction_seconds / saving_per_product
    )

    if config.mode == "matrix_free":
        selected, reason = "matrix_free", "explicit matrix-free override"
    elif config.mode == "sparse":
        if not safe:
            raise MemoryError(
                "estimated persistent CSR/CSC storage "
                f"({estimated_sparse_bytes} bytes) exceeds the configured memory "
                f"budget ({budget} bytes)"
            )
        selected, reason = "sparse", (
            "explicit sparse override using an existing cache"
            if cache_available
            else "explicit sparse override within memory budget"
        )
    elif cache_available and safe:
        selected, reason = "sparse", "compatible sparse cache is available"
    elif not safe:
        selected, reason = "matrix_free", "estimated sparse storage exceeds memory budget"
    elif (
        estimated_savings is not None
        and config.estimated_construction_seconds is not None
        and estimated_savings <= config.estimated_construction_seconds
    ):
        selected, reason = (
            "matrix_free",
            "estimated product savings do not amortize sparse construction",
        )
    elif expected < config.minimum_reuse_products:
        selected, reason = "matrix_free", "too few expected products to amortize construction"
    else:
        selected, reason = "sparse", "expected repeated products amortize sparse construction"

    return SparseOperatorSelection(
        requested_mode=config.mode,
        selected_mode=selected,
        reason=reason,
        cache_available=cache_available,
        rows=rows,
        columns=columns,
        logical_entries=logical_entries,
        dense_bytes=dense_bytes,
        estimated_nonzero_entries=estimated_nnz,
        estimated_sparse_solver_bytes=estimated_sparse_bytes,
        available_memory_bytes=available,
        memory_budget_bytes=budget,
        expected_product_calls=expected,
        estimated_construction_seconds=config.estimated_construction_seconds,
        estimated_product_savings_seconds=estimated_savings,
        estimated_break_even_products=break_even,
    )


def scipy_sparse_operator_from_fixed_routing(
    operator: FixedRoutingMeasurementOperator,
) -> SparseLinearOperator:
    """Transfer a persisted dense/BCOO artifact once into CPU CSR/CSC storage."""
    shape = (operator.num_measurements, operator.num_free_od)
    if operator.matrix.shape != shape:
        raise ValueError("operator matrix shape disagrees with its metadata.")
    if operator.representation == "bcoo":
        data = np.asarray(operator.matrix.data)
        indices = np.asarray(operator.matrix.indices)
        if indices.shape != (data.size, 2):
            raise ValueError("BCOO data and indices have incompatible shapes.")
        matrix = sparse.coo_array(
            (data, (indices[:, 0], indices[:, 1])), shape=shape
        )
    elif operator.representation == "dense":
        matrix = sparse.csr_array(np.asarray(operator.matrix))
    else:
        raise ValueError("operator representation must be dense or bcoo.")
    return SparseLinearOperator(matrix)


def prepare_fixed_routing_linear_measurement_backend(
    *,
    inputs,
    routing=None,
    theta: float | None = None,
    routing_factory: Callable[[], object] | None = None,
    spec,
    compact_layout,
    assignment_fingerprint: str,
    od_layout_fingerprint: str,
    cache_directory: str | Path,
    config: SparseOperatorSelectionConfig | None = None,
) -> PreparedLinearMeasurementBackend:
    """Prepare one reusable matrix-free or persistent sparse solver backend."""
    config = SparseOperatorSelectionConfig() if config is None else config
    cache_directory = Path(cache_directory)
    rows, columns = int(spec.num_measurements), int(compact_layout.num_free)
    if theta is None:
        if routing is None:
            raise ValueError("theta is required when routing is not supplied.")
        theta = float(np.asarray(routing.theta))
    if not math.isfinite(theta) or theta <= 0.0:
        raise ValueError("theta must be finite and strictly positive.")
    cache_path = fixed_routing_measurement_operator_cache_path(
        cache_directory=cache_directory,
        inputs=inputs,
        theta=theta,
        spec=spec,
        assignment_fingerprint=assignment_fingerprint,
        compact_layout=compact_layout,
        od_layout_fingerprint=od_layout_fingerprint,
        representation="bcoo",
        zero_tolerance=config.zero_tolerance,
    )
    selection = select_fixed_routing_linear_operator(
        rows=rows,
        columns=columns,
        dtype=inputs.base_link_cost.dtype,
        cache_available=cache_path.exists(),
        config=config,
    )
    total_start = perf_counter()
    routing_seconds = 0.0

    def require_routing():
        nonlocal routing, routing_seconds
        if routing is None:
            if routing_factory is None:
                raise ValueError(
                    "routing or routing_factory is required to build the selected backend"
                )
            start = perf_counter()
            routing = routing_factory()
            routing_seconds = perf_counter() - start
        return routing

    if selection.selected_mode == "matrix_free":
        prepared_routing = require_routing()
        start = perf_counter()
        matrix_free = MatrixFreeFixedRoutingMeasurementOperator(
            inputs=inputs,
            routing=prepared_routing,
            spec=spec,
            compact_layout=compact_layout,
        )
        construction = perf_counter() - start
        return PreparedLinearMeasurementBackend(
            operator=matrix_free,
            fixed_measurement_offset=np.asarray(matrix_free.fixed_measurement_offset),
            selection=selection,
            metrics=LinearMeasurementBackendMetrics(
                cache_lookup_seconds=0.0,
                cache_load_seconds=0.0,
                operator_construction_seconds=construction,
                cache_persistence_seconds=0.0,
                device_transfer_and_sparse_conversion_seconds=0.0,
                total_preparation_seconds=perf_counter() - total_start,
                fixed_routing_preparation_seconds=routing_seconds,
                cache_hit=False,
                representation="matrix_free",
                nonzero_entries=None,
                solver_storage_bytes=None,
            ),
        )

    lookup_start = perf_counter()
    persisted = load_valid_cached_fixed_routing_measurement_operator(
        cache_directory=cache_directory,
        inputs=inputs,
        theta=theta,
        spec=spec,
        assignment_fingerprint=assignment_fingerprint,
        compact_layout=compact_layout,
        od_layout_fingerprint=od_layout_fingerprint,
        representation="bcoo",
        zero_tolerance=config.zero_tolerance,
    )
    lookup_elapsed = perf_counter() - lookup_start
    cache_hit = persisted is not None
    construction = 0.0
    persistence = 0.0
    if persisted is None:
        prepared_routing = require_routing()
        start = perf_counter()
        persisted = prepare_fixed_routing_measurement_operator(
            inputs=inputs,
            routing=prepared_routing,
            spec=spec,
            assignment_fingerprint=assignment_fingerprint,
            compact_layout=compact_layout,
            od_layout_fingerprint=od_layout_fingerprint,
            representation="bcoo",
            chunk_size=config.chunk_size,
            zero_tolerance=config.zero_tolerance,
        )
        construction = perf_counter() - start
        start = perf_counter()
        save_fixed_routing_measurement_operator(persisted, cache_path)
        persistence = perf_counter() - start
    conversion_start = perf_counter()
    cpu_operator = scipy_sparse_operator_from_fixed_routing(persisted)
    conversion = perf_counter() - conversion_start
    return PreparedLinearMeasurementBackend(
        operator=cpu_operator,
        fixed_measurement_offset=np.asarray(persisted.fixed_measurement_offset),
        selection=selection,
        metrics=LinearMeasurementBackendMetrics(
            cache_lookup_seconds=lookup_elapsed,
            cache_load_seconds=(lookup_elapsed if cache_hit else 0.0),
            operator_construction_seconds=construction,
            cache_persistence_seconds=persistence,
            device_transfer_and_sparse_conversion_seconds=conversion,
            total_preparation_seconds=perf_counter() - total_start,
            fixed_routing_preparation_seconds=routing_seconds,
            cache_hit=cache_hit,
            representation="csr_csc",
            nonzero_entries=cpu_operator.nonzero_entries,
            solver_storage_bytes=cpu_operator.solver_storage_bytes,
        ),
        persisted_operator=persisted,
    )
