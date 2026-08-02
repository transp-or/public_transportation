from __future__ import annotations

import json
from pathlib import Path


def test_fixed_routing_scaling_benchmark_records_structural_evidence():
    root = Path(__file__).resolve().parents[2]
    report = json.loads(
        (root / "benchmarks/fixed_routing_scaling.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["full_graph_traversal_confirmed_by_code"] is True
    assert report["kernel_complexity"].startswith("O(groups *")
    densities = {item["enabled_density"] for item in report["cases"]}
    assert densities >= {0.1, 0.5, 1.0}
    workers = {item["workers"] for item in report["parallel_cases"]}
    assert workers == {1, 2, 4}
    assert all(
        item["compilation_count"] == 1 for item in report["parallel_cases"]
    )
    assert all(
        item["maximum_probability_difference"] <= 1.0e-7
        for item in report["parallel_cases"]
    )
