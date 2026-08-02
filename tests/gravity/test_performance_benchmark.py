from __future__ import annotations

import pytest

from benchmarks.benchmark_gravity_performance import run_benchmark


@pytest.fixture(scope="module")
def report():
    return run_benchmark(
        parameter_counts=(3, 7),
        representations=("dense", "bcoo"),
        od_cells=64,
        measurements=24,
        repeats=1,
    )


def test_benchmark_spans_parameter_counts_and_operator_backends(report):
    assert report["parameter_counts"] == (3, 7)
    assert report["representations"] == ("dense", "bcoo")
    assert {
        (case["parameter_count"], case["representation"])
        for case in report["cases"]
    } == {(3, "dense"), (7, "dense"), (3, "bcoo"), (7, "bcoo")}


def test_benchmark_reports_required_phases_memory_and_cache_reuse(report):
    for case in report["cases"]:
        assert case["od_cell_count"] == 64
        assert case["measurement_count"] == 24
        assert case["operator_stored_bytes"] > 0
        assert case["forward_routing_first_seconds"] >= 0
        assert case["forward_routing_warm_seconds"] >= 0
        assert case["transpose_routing_first_seconds"] >= 0
        assert case["transpose_routing_warm_seconds"] >= 0
        assert case["maximum_strategy_gradient_difference"] < 3e-4
        assert case["fastest_warm_strategy"] in {"batched_forward", "adjoint"}
        assert {item["strategy"] for item in case["strategies"]} == {
            "batched_forward",
            "adjoint",
        }
        for strategy in case["strategies"]:
            for name in (
                "tracing_seconds",
                "lowering_seconds",
                "compilation_seconds",
                "first_execution_seconds",
                "warm_execution_seconds",
                "changed_parameter_execution_seconds",
            ):
                assert strategy[name] >= 0
            assert strategy["lowered_text_bytes"] > 0
            assert strategy["peak_host_rss_bytes"] > 0
            assert strategy["compiled_kernel_cache_misses"] == 1
            assert strategy["compiled_kernel_cache_hits"] == 2
            assert strategy["parameter_value_reuse_verified"]
            assert (
                strategy["persistent_cache_hit_status"]
                == "requires_fresh_process_protocol"
            )
