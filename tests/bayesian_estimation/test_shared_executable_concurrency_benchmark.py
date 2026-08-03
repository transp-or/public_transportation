from __future__ import annotations

import json
from pathlib import Path


def test_shared_executable_benchmark_separates_latency_and_throughput():
    root = Path(__file__).resolve().parents[2]
    report = json.loads(
        (root / "benchmarks/shared_executable_concurrency.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["warmup_completed"] is True
    assert report["threads_per_worker_controls_xla_threads"] is False
    assert [item["workers"] for item in report["concurrency"]] == [1, 2, 4]
    for item in report["concurrency"]:
        assert item["batch_wall_seconds"] > 0.0
        assert item["shards_per_second"] > 0.0
        assert item["per_shard_seconds"]["minimum"] > 0.0
        assert item["per_shard_seconds"]["median"] > 0.0
        assert item["per_shard_seconds"]["maximum"] > 0.0
        assert item["compilation_count_during_batch"] == 0
        assert item["shared_executable"] is True
        assert item["maximum_numerical_difference"] == 0.0
        assert item["effective_average_cpu_cores"] > 0.0
    assert report["system"]["allocated_or_visible_cpu_count"] > 0
    assert report["system"]["jax_version"]
    assert report["system"]["jaxlib_version"]
    assert report["system"]["logical_devices"]
    assert [item["shards_per_execution_batch"] for item in report["batched"]] == [
        1,
        2,
        4,
    ]
    assert all(item["effective_average_cpu_cores"] > 0 for item in report["batched"])
    assert (
        report["experimental_four_shard_batch"][
            "production_routing_fingerprint_changed"
        ]
        is False
    )
