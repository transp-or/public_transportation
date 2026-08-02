from __future__ import annotations

import json
from pathlib import Path


def test_sharded_fixed_routing_benchmark_records_structural_invariants():
    root = Path(__file__).resolve().parents[2]
    report = json.loads(
        (root / "benchmarks/sharded_fixed_routing.json").read_text(encoding="utf-8")
    )

    assert report["destination_groups"] > 1
    assert report["shards"] > 1
    assert report["groups_per_shard"] == 2
    assert report["compilation_count"] == 1
    assert report["cache_hits_on_reload"] == report["shards"]
    assert report["maximum_mask_difference"] == 0
    assert report["maximum_probability_difference"] <= 1.0e-7
    assert report["global_measurement_matrix_constructed"] is False
