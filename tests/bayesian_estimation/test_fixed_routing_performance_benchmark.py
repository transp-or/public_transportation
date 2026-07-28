from __future__ import annotations

import pytest

from benchmarks.benchmark_fixed_routing_performance import run_benchmark


@pytest.fixture(scope="module")
def report():
    return run_benchmark(repeats=1, theta_values=(1.0,))


def test_performance_benchmark_covers_full_and_compact_layouts(report):
    assert report["example"] == "simple_example_02"
    assert report["repeats"] == 1
    assert {case["layout"] for case in report["cases"]} == {"full", "compact"}


def test_performance_benchmark_checks_equivalence_and_reports_memory(report):
    for case in report["cases"]:
        assert case["cache_bytes"] > 0
        assert case["routing_first_preparation_s"] > 0.0
        assert case["routing_warm_preparation_s"] > 0.0
        assert case["max_link_flow_abs_error"] <= 2.0e-6
        assert case["max_gradient_abs_error"] <= 3.0e-5
        assert case["dynamic_warm_forward_s"] > 0.0
        assert case["cached_warm_forward_s"] > 0.0
        assert case["dynamic_warm_value_gradient_s"] > 0.0
        assert case["cached_warm_value_gradient_s"] > 0.0
