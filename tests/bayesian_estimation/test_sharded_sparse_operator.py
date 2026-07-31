from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse

from public_transportation.inference.sharded_sparse_operator import (
    ShardedOperatorManifest,
    ShardedSparseLinearOperator,
    SparseShardIdentity,
    SparseShardMetrics,
    load_sparse_shard,
    save_sharded_operator_manifest,
    save_sparse_shard,
    shard_path,
)
from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
)
from public_transportation.inference.fixed_routing_linear_trf_solver import (
    TRFLSMRConfig,
    solve_trf_lsmr,
)
from public_transportation.inference.linear_operator import SparseLinearOperator


def _cache(tmp_path, *, complete=True):
    provenance = {"assignment": "a", "mapping": "m", "theta": 1.0}
    identities = (
        SparseShardIdentity(0, 0, 0, 2),
        SparseShardIdentity(1, 0, 0, 2),
    )
    template = ShardedOperatorManifest(
        num_measurements=3,
        num_free_od=4,
        dtype="float64",
        provenance=provenance,
        expected_shards=identities,
        completed_shards=tuple(item.key for item in identities) if complete else (),
        aggregate_nonzeros=5 if complete else 0,
        complete=complete,
        measurement_block_size=2,
        od_chunk_size=2,
    )
    first = save_sparse_shard(
        directory=tmp_path,
        identity=identities[0],
        row_indices=[0, 1],
        matrix=sparse.csr_array([[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 0.0]]),
        fixed_offset=[4.0, 0.0],
        num_measurements=3,
        num_free_od=4,
        dtype=np.float64,
        zero_tolerance=0.0,
        provenance_hash=template.provenance_hash,
        metrics=SparseShardMetrics(8, 3, 5),
    )
    second = save_sparse_shard(
        directory=tmp_path,
        identity=identities[1],
        row_indices=[1, 2],
        matrix=sparse.csr_array([[0.0, 0.0, 0.0, 5.0], [7.0, 0.0, 0.0, 0.0]]),
        fixed_offset=[2.0, 1.0],
        num_measurements=3,
        num_free_od=4,
        dtype=np.float64,
        zero_tolerance=0.0,
        provenance_hash=template.provenance_hash,
        metrics=SparseShardMetrics(8, 2, 6),
    )
    save_sharded_operator_manifest(template, tmp_path)
    return template, first, second


def test_sharded_products_offsets_and_adjoint_match_monolithic(tmp_path):
    _cache(tmp_path)
    operator = ShardedSparseLinearOperator(tmp_path, max_cached_shards=1)
    matrix = np.asarray(
        [[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 5.0], [7.0, 0.0, 0.0, 0.0]]
    )
    x = np.asarray([2.0, 3.0, 4.0, 5.0])
    y = np.asarray([0.5, 1.5, -2.0])
    np.testing.assert_allclose(operator.matvec(x), matrix @ x)
    np.testing.assert_allclose(operator.rmatvec(y), matrix.T @ y)
    np.testing.assert_allclose(operator.fixed_measurement_offset, [4.0, 2.0, 1.0])
    assert np.vdot(y, operator.matvec(x)) == pytest.approx(
        np.vdot(x, operator.rmatvec(y))
    )
    assert operator.matvec_count == 2
    assert operator.rmatvec_count == 2
    assert operator.shard_load_count > 2


def test_eager_shards_are_loaded_once_and_reused(tmp_path):
    _cache(tmp_path)
    operator = ShardedSparseLinearOperator(tmp_path)
    initial_loads = operator.shard_load_count
    initial_opens = operator.file_open_count
    operator.matvec(np.ones(4))
    operator.rmatvec(np.ones(3))
    assert initial_loads == 2
    assert operator.shard_load_count == initial_loads
    assert operator.shard_cache_hit_count >= 4
    assert operator.file_open_count == initial_opens
    assert operator.sparse_matrix_calls == 2
    assert operator.shard_eviction_count == 0
    assert operator.uses_merged_operator
    assert operator.merged_storage_bytes > 0


def test_incomplete_manifest_is_not_solver_ready(tmp_path):
    _cache(tmp_path, complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        ShardedSparseLinearOperator(tmp_path)


def test_memory_budget_selects_bounded_lru_loading(tmp_path):
    _cache(tmp_path)
    operator = ShardedSparseLinearOperator(tmp_path, memory_budget_bytes=1)
    assert operator.loading_policy == "lru"
    assert operator.max_cached_shards == 1
    initial_opens = operator.file_open_count
    operator.matvec(np.ones(4))
    assert operator.file_open_count > initial_opens
    assert operator.shard_eviction_count > 0


def test_corrupted_shard_content_is_rejected(tmp_path):
    manifest, _, _ = _cache(tmp_path)
    path = shard_path(tmp_path, manifest.expected_shards[0])
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["data"] = np.array(payload["data"], copy=True)
    payload["data"][0] += 1.0
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="content hash"):
        load_sparse_shard(path, expected_provenance_hash=manifest.provenance_hash)


def test_manifest_rejects_tampered_provenance(tmp_path):
    _cache(tmp_path)
    path = tmp_path / "manifest.json"
    payload = json.loads(path.read_text())
    payload["provenance"]["theta"] = 2.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="provenance hash"):
        ShardedSparseLinearOperator(tmp_path)


def test_sharded_solver_matches_monolithic_sparse_solver(tmp_path):
    _cache(tmp_path)
    sharded = ShardedSparseLinearOperator(tmp_path)
    matrix = sparse.csr_array(
        [[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 5.0], [7.0, 0.0, 0.0, 0.0]]
    )
    common = {
        "fixed_measurement_offset": np.asarray([4.0, 2.0, 1.0]),
        "observations": np.asarray([8.0, 13.0, 15.0]),
        "observation_weights": np.ones(3),
        "prior_demand": np.ones(4),
        "lower_bounds": np.zeros(4),
        "upper_bounds": np.full(4, np.inf),
        "provenance": FixedRoutingLinearProvenance("od", "assignment", "mapping", 1.0),
        "regularization_selection": "none",
    }
    sharded_problem = FixedRoutingLinearProblem(
        measurement_operator=sharded, **common
    )
    monolithic_problem = FixedRoutingLinearProblem(
        measurement_operator=SparseLinearOperator(matrix), **common
    )
    config = TRFLSMRConfig(tolerance=1e-10, lsmr_tolerance=1e-12)
    sharded_result = solve_trf_lsmr(sharded_problem, config=config)
    monolithic_result = solve_trf_lsmr(monolithic_problem, config=config)
    np.testing.assert_allclose(
        sharded_result.evaluation.data_fit.prediction,
        monolithic_result.evaluation.data_fit.prediction,
        rtol=1e-8,
        atol=1e-8,
    )
    assert sharded_result.evaluation.objective == pytest.approx(
        monolithic_result.evaluation.objective, rel=1e-10, abs=1e-10
    )


@pytest.mark.parametrize("compressed", [False, True])
def test_shard_persistence_formats_are_compatible(tmp_path, compressed):
    identity = SparseShardIdentity(0, 0, 0, 2)
    provenance_hash = "a" * 64
    metadata = save_sparse_shard(
        directory=tmp_path,
        identity=identity,
        row_indices=[0, 2],
        matrix=sparse.csr_array([[1.0, 0.0], [0.0, 2.0]]),
        fixed_offset=[0.0, 3.0],
        num_measurements=3,
        num_free_od=2,
        dtype=np.float64,
        zero_tolerance=0.0,
        provenance_hash=provenance_hash,
        metrics=SparseShardMetrics(4, 2, 2),
        compressed=compressed,
    )
    loaded = load_sparse_shard(
        shard_path(tmp_path, identity), expected_provenance_hash=provenance_hash
    )
    np.testing.assert_array_equal(loaded.matrix.toarray(), [[1.0, 0.0], [0.0, 2.0]])
    assert metadata.metrics.serialization_seconds >= 0.0
    assert metadata.metrics.disk_bytes > 0
