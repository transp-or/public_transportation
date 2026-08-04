from __future__ import annotations

import json
from pathlib import Path

from public_transportation.inference.parallel_partial_gate import (
    assess_parallel_partial_gate,
)


def _rows(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))["results"]


def test_committed_public_partial_benchmark_passes_all_effort_gates():
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/parallel_partial_gate_public.json"
    )
    report = assess_parallel_partial_gate(_rows(path))
    assert report.passed
    assert all(item.passed for item in report.effort_results)
    assert report.to_dict()["schema_version"] == 1


def test_gate_rejects_slow_or_inaccurate_effort():
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/parallel_partial_gate_public.json"
    )
    rows = _rows(path)
    rows[1]["gradient_relative_norm_error"] = 0.5
    report = assess_parallel_partial_gate(rows)
    assert not report.passed
    assert "gradient error" in report.effort_results[1].reasons[0]
