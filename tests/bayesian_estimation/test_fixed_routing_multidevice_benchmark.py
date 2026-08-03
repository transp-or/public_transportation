from __future__ import annotations

import json
from pathlib import Path


def test_multidevice_benchmark_uses_explicit_placement_and_is_equivalent():
    root = Path(__file__).resolve().parents[2]
    report = json.loads(
        (root / "benchmarks/fixed_routing_multidevice_cpu.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["explicit_pmap"] is True
    assert report["requested_devices"] == 2
    assert len(report["placed_devices"]) == 2
    assert len(set(report["placed_devices"])) == 2
    assert report["effective_average_cpu_cores"] > 0.0
    assert report["shards_per_second"] > 0.0
    assert report["maximum_mask_difference"] == 0
    assert report["maximum_probability_difference"] <= 1.0e-7
