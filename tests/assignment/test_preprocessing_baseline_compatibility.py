from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.build_time_expanded import build_jax_graph
from public_transportation.domain import Scenario
from public_transportation.preprocessing import (
    build_canonical_timetable_index,
    generate_candidate_od_pairs,
)


ROOT = Path(__file__).resolve().parents[2]
SCENARIO = ROOT / "docs/source/examples/geneva_gtfs/data"
FIXTURE = ROOT / "benchmarks/preprocessing_baseline_geneva.json"
OPTIMIZED_FIXTURE = ROOT / "benchmarks/preprocessing_optimized_geneva.json"


def _array_hash(value: object) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def test_canonical_index_preserves_geneva_graph_and_universe_identity() -> None:
    baseline = json.loads(FIXTURE.read_text(encoding="utf-8"))
    scenario = Scenario.from_folder(
        SCENARIO, strict=True, demand_file=SCENARIO / "prior_demand.csv"
    )
    timetable_index = build_canonical_timetable_index(scenario)
    universe = generate_candidate_od_pairs(
        scenario, timetable_index=timetable_index
    )
    graph = build_jax_graph(
        scenario=scenario,
        config=AssignmentConfig(),
        timetable_index=timetable_index,
    )

    assert timetable_index.fingerprint == baseline["scientific_provenance"][
        "canonical_timetable_fingerprint"
    ]
    assert universe.fingerprint == baseline["scientific_provenance"][
        "od_universe_fingerprint"
    ]
    for name, expected in baseline["scientific_provenance"]["graph_arrays"].items():
        assert _array_hash(getattr(graph, name)) == expected


def test_optimized_benchmark_preserves_baseline_scientific_outputs() -> None:
    baseline = json.loads(FIXTURE.read_text(encoding="utf-8"))
    optimized = json.loads(OPTIMIZED_FIXTURE.read_text(encoding="utf-8"))
    assert optimized["benchmark"].endswith("_optimized")
    assert optimized["scientific_provenance"] == baseline["scientific_provenance"]
    assert optimized["dimensions"] == baseline["dimensions"]
    assert optimized["od_time_expansion_seconds"] < baseline["od_time_expansion_seconds"]
