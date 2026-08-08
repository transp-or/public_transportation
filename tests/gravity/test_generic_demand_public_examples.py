from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from public_transportation.inference.reduced_od import (
    DemandModelDimensions,
    build_demand_parameter_layout,
    progressive_model_ladder,
    warm_start_demand_parameters,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "docs/source/examples"


def _example_dimensions(name: str) -> DemandModelDimensions:
    if name == "geneva_gtfs":
        report = json.loads(
            (
                EXAMPLES / name / "estimation/results/gravity_validation_summary.json"
            ).read_text()
        )
        assert report["num_free_od"] > 0
        return DemandModelDimensions(periods=2, origin_groups=4, destination_groups=4)
    report = json.loads(
        (
            EXAMPLES / name / "estimation/results/gravity_estimation_summary.json"
        ).read_text()
    )
    assert report["num_free_od"] > 0
    return DemandModelDimensions(
        periods=2 if name.endswith("02") else 1, origin_groups=2, destination_groups=2
    )


def test_generic_ladder_resolves_and_warm_starts_on_every_gravity_example() -> None:
    ladder = progressive_model_ladder()
    for example in ("simple_example_01", "simple_example_02", "geneva_gtfs"):
        dimensions = _example_dimensions(example)
        layouts = [
            build_demand_parameter_layout(ladder[name], dimensions)
            for name in ("M0", "M1", "M2", "M3", "M4", "M5")
        ]
        raw = np.zeros(layouts[0].size)
        for parent, child in zip(layouts[:-1], layouts[1:], strict=True):
            raw, report = warm_start_demand_parameters(parent, child, raw)
            assert raw.shape == (child.size,)
            assert not report.dropped
        assert layouts[-1].size > layouts[0].size
