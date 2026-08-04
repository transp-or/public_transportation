from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from public_transportation.inference.gravity import (
    GravityFidelityRequest,
    build_gravity_fidelity_context,
    gravity_value_and_gradient_progressive,
)
from tests.gravity.test_phase2_objective import problem

from benchmarks.benchmark_progressive_gravity_sharded import run as run_sharded_benchmark


class ShardAwareOperator:
    def __init__(self, base):
        self.base = base
        for name in (
            "num_free_od",
            "num_measurements",
            "compact_layout_fingerprint",
            "fixed_measurement_offset",
            "representation",
            "is_matrix_free",
            "assignment_fingerprint",
            "graph_fingerprint",
            "mapping_fingerprint",
            "theta",
            "dtype",
            "metrics",
        ):
            setattr(self, name, getattr(base, name))
        matrix = np.asarray(base.matrix)
        self.matrices = []
        for index, columns in enumerate(np.array_split(np.arange(matrix.shape[1]), 4)):
            shard = np.zeros_like(matrix)
            shard[:, columns] = matrix[:, columns]
            self.matrices.append(jax.numpy.asarray(shard))
        self.forward_shards = []
        self.reverse_shards = []

    @property
    def product_capabilities(self):
        return self.base.product_capabilities

    def jax_matvec(self, vector):
        return self.base.jax_matvec(vector)

    def jax_rmatvec(self, vector):
        return self.base.jax_rmatvec(vector)

    def jax_matmat(self, matrix):
        return self.base.jax_matmat(matrix)

    def fidelity_shard_statistics(self):
        return tuple(
            {
                "shard_id": f"routing-{index}",
                "shard_index": index,
                "support_entries": max(1, int(np.count_nonzero(matrix))),
                "routing_bytes": matrix.nbytes,
                "stratum": "destinations",
            }
            for index, matrix in enumerate(self.matrices)
        )

    def jax_matvec_fidelity_shard(self, index, vector):
        self.forward_shards.append(index)
        return self.matrices[index] @ vector

    def jax_rmatvec_fidelity_shard(self, index, vector):
        self.reverse_shards.append(index)
        return self.matrices[index].T @ vector


def test_dense_operator_builds_context_automatically():
    with jax.enable_x64():
        item = problem()
        result = gravity_value_and_gradient_progressive(
            np.zeros(3),
            problem=item,
            fidelity=GravityFidelityRequest(effort_percent=25, seed=4),
        )
        assert not result.fidelity.exact
        assert 1 <= result.fidelity.selected_shard_count <= item.operator.num_free_od


def test_shard_aware_adapter_plans_without_products_and_loads_only_selection():
    with jax.enable_x64():
        base_problem = problem()
        operator = ShardAwareOperator(base_problem.operator)
        item = replace(base_problem, operator=operator)
        context = build_gravity_fidelity_context(item)
        assert operator.forward_shards == []
        assert operator.reverse_shards == []
        result = gravity_value_and_gradient_progressive(
            np.zeros(3),
            problem=item,
            fidelity=GravityFidelityRequest(effort_percent=1, seed=2),
            context=context,
        )
        assert len(operator.forward_shards) == result.fidelity.selected_shard_count
        assert len(operator.reverse_shards) == result.fidelity.selected_shard_count
        assert set(operator.forward_shards) == set(operator.reverse_shards)
        assert len(operator.forward_shards) < len(operator.matrices)


def test_committed_public_benchmark_covers_requested_effort_grid():
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/progressive_gravity_public.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["public_synthetic_problem"]["cells"] == 200
    assert [row["requested_effort_percent"] for row in report["results"]] == [
        1,
        2,
        5,
        10,
        25,
        50,
        75,
        100,
    ]
    exact = report["results"][-1]
    assert exact["effective_effort_percent"] == 100.0
    assert exact["objective_absolute_error"] == 0.0
    assert exact["gradient_relative_norm_error"] == 0.0
    assert exact["quality_score"] == 1.0


def test_real_persisted_sharded_operator_supports_progressive_products():
    report = run_sharded_benchmark(
        SimpleNamespace(
            nodes=32,
            maximum_out_degree=2,
            destination_groups=4,
            groups_per_shard=2,
            od_cells=32,
            measurements=16,
            resident_shard_limit=1,
            operator_shards_per_batch=2,
            group_execution_strategy="scan",
            shard_execution_strategy="aggregate",
            operator_concurrency=1,
            repetitions=1,
            seed=20260804,
            efforts=(50.0,),
        )
    )
    assert report["problem"]["routing_shards"] == 2
    assert report["results"][0]["selected_shards"] == 1
    assert np.isfinite(report["results"][0]["median_wall_seconds"])
