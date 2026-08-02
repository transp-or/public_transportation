from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "benchmarks/matrix_free_gravity.json"


def test_committed_matrix_free_benchmark_has_bounded_structural_evidence():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    logical = report["logical_operator"]
    assert report["schema_version"] == 1
    assert logical["measurements"] == 270
    assert logical["free_od"] == 70
    assert logical["entries"] == 18_900
    assert logical["avoided_dense_float32_bytes"] == 75_600
    assert not logical["global_matrix_constructed"]
    preparation = report["preparation"]
    assert preparation["forward_compilation_count"] == 1
    assert preparation["transpose_compilation_count"] == 1
    assert preparation["forward_execution_count"] == 2
    assert preparation["transpose_execution_count"] == 2
    assert preparation["forward_input_shape"] == [70]
    assert preparation["transpose_input_shape"] == [270]
    assert preparation["backend"] == "cpu"
    assert report["objective"]["lowered_text_bytes"] > 0
    assert report["checkpoint"]["bytes"] > 0
