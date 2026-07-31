"""Tests for fixed-routing block construction, persistence, and retention."""

from __future__ import annotations

import os

import numpy as np
import pytest
from scipy import sparse

from public_transportation.inference.block_coordinate import (
    BlockCoordinateFingerprints,
    BlockCoordinateMAPConfig,
    BlockOperatorCacheConfig,
    BlockOperatorCacheProvenance,
    FixedRoutingBlockOperatorFactory,
    ODBlock,
    run_block_coordinate_map,
    validate_block_partition,
)
from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
)
from public_transportation.inference.linear_operator import (
    DenseLinearOperator,
    SparseLinearOperator,
)


def _provenance(**overrides) -> BlockOperatorCacheProvenance:
    values = {
        "assignment_inputs": "assignment",
        "od_layout": "layout",
        "fixed_demand_layout": "fixed",
        "measurement_mapping": "mapping",
        "routing_parameters": "theta=1",
    }
    values.update(overrides)
    return BlockOperatorCacheProvenance(**values)


def _block(identifier: str, columns: tuple[int, ...], support=None) -> ODBlock:
    return ODBlock(
        block_id=identifier,
        free_column_indices=columns,
        active_od_indices=columns,
        destination_group_indices=(0,),
        time_bin_ids=("t0",),
        measurement_support_indices=support,
    )


@pytest.mark.parametrize(
    "complete",
    [
        DenseLinearOperator(
            np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]])
        ),
        SparseLinearOperator(
            sparse.csr_array(
                [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]]
            )
        ),
    ],
)
def test_cold_build_and_cache_hit_match_selected_columns(tmp_path, complete) -> None:
    block = _block("selected", (0, 2), support=(0, 2))
    config = BlockOperatorCacheConfig(tmp_path, maximum_retained_blocks=1)
    factory = FixedRoutingBlockOperatorFactory(
        complete_operator=complete, provenance=_provenance(), config=config
    )
    cold = factory.get(block)
    expected = np.array([[1.0, 2.0], [0.0, 0.0], [4.0, 5.0]])

    assert not cold.preparation_metrics.cache_hit
    assert factory.metrics.cold_builds == 1
    np.testing.assert_allclose(cold.matvec([2.0, 3.0]), expected @ [2.0, 3.0])
    np.testing.assert_allclose(cold.rmatvec([1.0, 2.0, 3.0]), expected.T @ [1.0, 2.0, 3.0])
    assert cold.product_metrics.matvec_count == 1
    assert cold.product_metrics.rmatvec_count == 1
    assert cold.product_metrics.matvec_seconds >= 0.0
    assert factory.cache_path(block).is_file()

    loaded_factory = FixedRoutingBlockOperatorFactory(
        complete_operator=complete, provenance=_provenance(), config=config
    )
    loaded = loaded_factory.get(block)
    assert loaded.preparation_metrics.cache_hit
    assert loaded_factory.metrics.disk_cache_hits == 1
    assert loaded.preparation_metrics.construction_seconds == 0.0
    np.testing.assert_allclose(loaded.matvec([2.0, 3.0]), expected @ [2.0, 3.0])


def test_matrix_free_source_constructs_only_requested_columns(tmp_path) -> None:
    matrix = np.arange(24, dtype=float).reshape(6, 4)

    class CountingOperator:
        shape = matrix.shape
        dtype = matrix.dtype

        def __init__(self):
            self.matvec_count = 0

        def matvec(self, vector):
            self.matvec_count += 1
            return matrix @ np.asarray(vector)

        def rmatvec(self, vector):
            return matrix.T @ np.asarray(vector)

    complete = CountingOperator()
    block = _block("matrix-free", (1, 3), support=(0, 1, 2, 3, 4, 5))
    factory = FixedRoutingBlockOperatorFactory(
        complete_operator=complete,
        provenance=_provenance(),
        config=BlockOperatorCacheConfig(tmp_path),
    )
    operator = factory.get(block)
    assert complete.matvec_count == block.num_free_variables
    np.testing.assert_allclose(operator.matvec([0.5, 2.0]), matrix[:, (1, 3)] @ [0.5, 2.0])
    assert complete.matvec_count == block.num_free_variables


def test_lru_eviction_and_explicit_release_bound_retained_storage(tmp_path) -> None:
    complete = DenseLinearOperator(np.eye(4))
    factory = FixedRoutingBlockOperatorFactory(
        complete_operator=complete,
        provenance=_provenance(),
        config=BlockOperatorCacheConfig(tmp_path, maximum_retained_blocks=1),
    )
    first_block = _block("first", (0, 1), support=(0, 1))
    second_block = _block("second", (2, 3), support=(2, 3))
    first = factory.get(first_block)
    second = factory.get(second_block)

    assert first.released
    assert not second.released
    assert factory.retained_block_count == 1
    assert factory.metrics.evictions == 1
    with pytest.raises(RuntimeError, match="released"):
        first.matvec([1.0, 1.0])
    assert factory.release(second_block)
    assert factory.metrics.explicit_releases == 1
    assert second.released
    assert factory.retained_block_count == 0
    assert not factory.release(second_block)


def test_fingerprints_change_with_every_authoritative_input(tmp_path) -> None:
    complete = DenseLinearOperator(np.eye(3))
    block = _block("block", (0, 1), support=(0, 1))
    base = FixedRoutingBlockOperatorFactory(
        complete_operator=complete,
        provenance=_provenance(),
        config=BlockOperatorCacheConfig(tmp_path),
    )
    changed_provenance = FixedRoutingBlockOperatorFactory(
        complete_operator=complete,
        provenance=_provenance(measurement_mapping="changed"),
        config=BlockOperatorCacheConfig(tmp_path),
    )
    changed_config = FixedRoutingBlockOperatorFactory(
        complete_operator=complete,
        provenance=_provenance(),
        config=BlockOperatorCacheConfig(tmp_path, zero_tolerance=1.0e-8),
    )
    changed_block = _block("block", (1, 2), support=(1, 2))

    keys = {
        base.cache_key(block),
        changed_provenance.cache_key(block),
        changed_config.cache_key(block),
        base.cache_key(changed_block),
    }
    assert len(keys) == 4


def test_incomplete_publication_is_not_treated_as_cache_hit(tmp_path, monkeypatch) -> None:
    complete = DenseLinearOperator(np.eye(2))
    block = _block("block", (0,), support=(0,))
    factory = FixedRoutingBlockOperatorFactory(
        complete_operator=complete,
        provenance=_provenance(),
        config=BlockOperatorCacheConfig(tmp_path),
    )

    def fail_replace(_source, _target):
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="publication failure"):
        factory.get(block)
    assert not factory.cache_path(block).exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_corrupt_cache_and_incorrect_declared_support_are_rejected(tmp_path) -> None:
    complete = DenseLinearOperator(np.eye(3))
    wrong_support = _block("wrong", (0,), support=(1,))
    factory = FixedRoutingBlockOperatorFactory(
        complete_operator=complete,
        provenance=_provenance(),
        config=BlockOperatorCacheConfig(tmp_path),
    )
    with pytest.raises(ValueError, match="declared measurement support"):
        factory.get(wrong_support)

    valid = _block("valid", (0,), support=(0,))
    operator = factory.get(valid)
    operator.release()
    path = factory.cache_path(valid)
    path.write_bytes(b"not an npz")
    fresh = FixedRoutingBlockOperatorFactory(
        complete_operator=complete,
        provenance=_provenance(),
        config=BlockOperatorCacheConfig(tmp_path),
    )
    with pytest.raises(ValueError, match="invalid block-operator cache"):
        fresh.get(valid)


def test_estimator_with_cached_blocks_is_identical_on_cold_and_hit_runs(tmp_path) -> None:
    matrix = np.array(
        [[1.0, 0.2, 0.0], [0.0, 1.0, 0.4], [0.5, 0.0, 1.0], [0.2, 0.3, 0.1]]
    )
    problem = FixedRoutingLinearProblem(
        measurement_operator=matrix,
        fixed_measurement_offset=np.zeros(4),
        observations=np.array([3.0, 2.0, 4.0, 1.5]),
        observation_weights=np.ones(4),
        prior_demand=np.ones(3),
        lower_bounds=np.zeros(3),
        upper_bounds=np.full(3, 8.0),
        provenance=FixedRoutingLinearProvenance("od", "assignment", "mapping", 1.0),
        regularization_selection="none",
    )
    blocks = (
        _block("first", (0, 1), support=(0, 1, 2, 3)),
        _block("second", (2,), support=(1, 2, 3)),
    )
    partition = validate_block_partition(blocks, free_to_active_indices=(0, 1, 2))
    cache_config = BlockOperatorCacheConfig(
        tmp_path / "operator-cache", maximum_retained_blocks=1
    )

    def run(checkpoint_name: str):
        config = BlockCoordinateMAPConfig(
            maximum_block_updates=2,
            global_projected_gradient_tolerance=None,
            relative_sweep_objective_tolerance=None,
            checkpoint_directory=tmp_path / checkpoint_name,
        )
        identity = BlockCoordinateFingerprints(
            scenario="scenario",
            assignment_inputs="assignment",
            od_layout="layout",
            fixed_demand="fixed",
            measurements="measurements",
            prior="prior",
            routing="routing",
            partition=partition.fingerprint,
            solver_semantics=config.fingerprint,
        )
        factory = FixedRoutingBlockOperatorFactory(
            complete_operator=problem.measurement_operator,
            provenance=_provenance(),
            config=cache_config,
        )
        result = run_block_coordinate_map(
            problem=problem,
            partition=partition,
            config=config,
            fingerprints=identity,
            block_operator_factory=factory,
        )
        return result, factory.metrics

    cold, cold_metrics = run("cold-checkpoint")
    cached, cached_metrics = run("hit-checkpoint")
    assert cold_metrics.cold_builds == 2
    assert cached_metrics.disk_cache_hits == 2
    np.testing.assert_array_equal(cached.latest_free_flow, cold.latest_free_flow)
    np.testing.assert_array_equal(
        cached.state.current_prediction, cold.state.current_prediction
    )
    assert cached.state.current_objective == cold.state.current_objective
