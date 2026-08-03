from __future__ import annotations

from benchmarks.benchmark_sharded_gravity_operator import run_benchmark


def test_sharded_gravity_benchmark_is_bounded_and_reports_all_products():
    report = run_benchmark(
        nodes=24,
        maximum_out_degree=2,
        destination_groups=5,
        groups_per_shard=2,
        od_cells=10,
        measurements=7,
        resident_shard_limit=1,
        operator_batch_sizes=(1, 2, 4),
    )
    assert report["routing_shards"] == 3
    assert report["dense_measurement_od_constructed"] is False
    assert report["complete_routing_array_materialized"] is False
    assert report["best_operator_shards_per_batch"] in (1, 2, 4)
    assert report["best_shard_execution_strategy"] in {"aggregate", "concurrent"}
    assert isinstance(report["material_forward_throughput_improvement"], bool)
    for case in report["cases"]:
        assert set(case["products"]) == {"matvec", "rmatvec", "matmat"}
        for product in case["products"].values():
            assert product["total_elapsed_seconds"] >= 0.0
            assert product["peak_rss_bytes"] is not None
            assert product["resident_routing_bytes"] >= 0
        objective = case["objective_gradient"]
        assert objective["gradient_max_abs_difference"] < 1e-3
        assert objective["recommendation"]["gradient_strategy"] in {
            "adjoint",
            "batched_forward",
        }
