from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse as jsparse

from public_transportation.inference.fixed_routing_linear_backend import (
    SparseOperatorSelectionConfig,
    scipy_sparse_operator_from_fixed_routing,
    select_fixed_routing_linear_operator,
)
from public_transportation.inference.fixed_routing_measurement_operator import (
    FixedRoutingMeasurementOperator,
    MeasurementOperatorMetrics,
)


def test_auto_selects_sparse_for_repeated_products_with_safe_memory():
    decision = select_fixed_routing_linear_operator(
        rows=100,
        columns=200,
        dtype=np.float32,
        cache_available=False,
        config=SparseOperatorSelectionConfig(
            mode="auto", estimated_density=0.05, expected_matvec_calls=20
        ),
        available_memory_bytes=1_000_000_000,
    )
    assert decision.selected_mode == "sparse"
    assert decision.estimated_sparse_solver_bytes < decision.memory_budget_bytes
    assert "amortize" in decision.reason


def test_auto_rejects_sparse_before_unsafe_allocation():
    decision = select_fixed_routing_linear_operator(
        rows=10_000,
        columns=20_000,
        dtype=np.float64,
        cache_available=True,
        config=SparseOperatorSelectionConfig(
            mode="auto", memory_budget_bytes=1_000_000, estimated_density=0.2
        ),
    )
    assert decision.selected_mode == "matrix_free"
    assert "memory budget" in decision.reason

    with pytest.raises(MemoryError, match="exceeds"):
        select_fixed_routing_linear_operator(
            rows=10_000,
            columns=20_000,
            dtype=np.float64,
            cache_available=False,
            config=SparseOperatorSelectionConfig(
                mode="sparse", memory_budget_bytes=1_000_000
            ),
        )


def test_cached_operator_is_preferred_when_safe():
    decision = select_fixed_routing_linear_operator(
        rows=10,
        columns=20,
        dtype=np.float64,
        cache_available=True,
        config=SparseOperatorSelectionConfig(
            mode="auto", expected_matvec_calls=0, expected_rmatvec_calls=0
        ),
        available_memory_bytes=1_000_000,
    )
    assert decision.selected_mode == "sparse"
    assert decision.cache_available


def test_auto_uses_explicit_break_even_estimates():
    config = SparseOperatorSelectionConfig(
        mode="auto",
        expected_matvec_calls=2,
        expected_rmatvec_calls=2,
        estimated_construction_seconds=10.0,
        matrix_free_product_seconds=1.0,
        sparse_product_seconds=0.01,
    )
    decision = select_fixed_routing_linear_operator(
        rows=10,
        columns=20,
        dtype=np.float64,
        cache_available=False,
        config=config,
        available_memory_bytes=1_000_000,
    )
    assert decision.selected_mode == "matrix_free"
    assert decision.estimated_break_even_products == pytest.approx(10.0 / 0.99)
    assert "do not amortize" in decision.reason


def test_bcoo_conversion_builds_persistent_cpu_csr_and_csc():
    matrix = jsparse.BCOO.fromdense(
        jnp.asarray([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    )
    metrics = MeasurementOperatorMetrics(
        construction_seconds=0.0,
        dense_bytes=48,
        stored_bytes=36,
        peak_construction_bytes=0,
        nonzero_entries=3,
        total_entries=6,
        density=0.5,
        chunk_size=1,
    )
    persisted = FixedRoutingMeasurementOperator(
        matrix=matrix,
        fixed_measurement_offset=jnp.asarray([4.0, 5.0]),
        representation="bcoo",
        num_active_od=3,
        num_free_od=3,
        num_measurements=2,
        od_layout_fingerprint="od",
        compact_layout_fingerprint="compact",
        assignment_fingerprint="assignment",
        graph_fingerprint="graph",
        mapping_fingerprint="mapping",
        theta=1.0,
        dtype="float32",
        metrics=metrics,
    )
    operator = scipy_sparse_operator_from_fixed_routing(persisted)
    x = np.asarray([2.0, 4.0, 6.0])
    y = np.asarray([7.0, 8.0])
    np.testing.assert_allclose(operator.matvec(x), [14.0, 12.0])
    np.testing.assert_allclose(operator.rmatvec(y), [7.0, 24.0, 14.0])
    assert operator.nonzero_entries == 3
    assert operator.solver_storage_bytes >= operator.stored_bytes
