from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "benchmarks/geneva_fixed_routing_performance.json"


def test_committed_geneva_fixed_routing_benchmark_is_equivalent():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["example"] == "geneva_gtfs"
    assert report["num_od_total"] == 15_128
    assert report["num_free_od"] == 96
    assert {case["layout"] for case in report["cases"]} == {"full", "compact"}
    for case in report["cases"]:
        assert case["theta"] == 5.0
        assert case["cache_bytes"] > 0
        assert case["max_link_flow_abs_error"] < 1.0e-3
        assert case["max_gradient_abs_error"] < 1.0e-3
        assert case["warm_forward_speedup"] > 1.0
        assert case["warm_value_gradient_speedup"] > 1.0
