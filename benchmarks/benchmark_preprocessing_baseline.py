"""Bounded baseline diagnostics for the direct-scheduled preprocessing path.

This benchmark intentionally measures only functionality retained after the
timetable-journey backend removal.  It records unavailable retired stages as
explicitly not applicable instead of inventing a replacement implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.assignment.build_time_expanded import build_jax_graph
from public_transportation.domain import Scenario
from public_transportation.inference.assignment_adapter import (
    build_assignment_inputs,
    prepare_fixed_routing,
)
from public_transportation.preprocessing import (
    build_canonical_timetable_index,
    TimetableFeasibilityIndex,
    build_structural_zero_topology,
    compute_od_path_metrics,
    expand_candidate_od_time_cells,
    generate_candidate_od_pairs,
)
from public_transportation.preprocessing.structural_zeros.config import (
    StructuralZeroAssignmentConfig,
)
from public_transportation.preprocessing.structural_zeros.scenario_fingerprint import (
    scenario_fingerprint_payload_json,
)
from public_transportation.preprocessing.structural_zeros.types import ODTimeKey


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = ROOT / "docs/source/examples/geneva_gtfs/data"
DEFAULT_OUTPUT = ROOT / "benchmarks/preprocessing_baseline_geneva.json"


def _peak_rss_bytes() -> int | None:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    return value if platform.system() == "Darwin" else value * 1024


def _sha256_array(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _timed(
    diagnostics: dict[str, Any],
    name: str,
    operation: Callable[[], Any],
) -> Any:
    started = perf_counter()
    result = operation()
    diagnostics[name] = perf_counter() - started
    return result


def run_baseline(
    *, scenario_directory: Path, label: str = "baseline"
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    started = perf_counter()
    scenario = _timed(
        diagnostics,
        "scenario_loading_and_validation_seconds",
        lambda: Scenario.from_folder(
            scenario_directory,
            strict=True,
            demand_file=scenario_directory / "prior_demand.csv",
        ),
    )

    assignment_config = AssignmentConfig()
    structural_config = StructuralZeroAssignmentConfig()
    timetable_index = _timed(
        diagnostics,
        "canonical_timetable_index_seconds",
        lambda: build_canonical_timetable_index(scenario),
    )
    universe = _timed(
        diagnostics,
        "candidate_pair_generation_seconds",
        lambda: generate_candidate_od_pairs(
            scenario, timetable_index=timetable_index
        ),
    )
    feasibility_index = _timed(
        diagnostics,
        "timetable_feasibility_index_seconds",
        lambda: TimetableFeasibilityIndex.from_scenario(
            scenario,
            physical_stop_mapping=universe.physical_stop_mapping,
            timetable_index=timetable_index,
        ),
    )
    expansion = _timed(
        diagnostics,
        "od_time_expansion_seconds",
        lambda: expand_candidate_od_time_cells(
            universe,
            scenario.time_bins,
            scenario=scenario,
            feasibility_index=feasibility_index,
            timetable_index=timetable_index,
        ),
    )

    graph_profile: dict[str, float] = {}
    graph = _timed(
        diagnostics,
        "time_expanded_graph_construction_seconds",
        lambda: build_jax_graph(
            scenario=scenario,
            config=assignment_config,
            profile=graph_profile,
            timetable_index=timetable_index,
        ),
    )
    diagnostics["time_expanded_graph_profile_seconds"] = graph_profile

    topology = _timed(
        diagnostics,
        "structural_zero_topology_construction_seconds",
        lambda: build_structural_zero_topology(
            scenario,
            structural_config,
            timetable_index=timetable_index,
        ),
    )
    metric_keys = tuple(
        ODTimeKey(*cell.tuple) for cell in expansion.cells
    )
    metrics = _timed(
        diagnostics,
        "structural_zero_path_metrics_seconds",
        lambda: compute_od_path_metrics(topology, keys=metric_keys),
    )

    artifacts = _timed(
        diagnostics,
        "assignment_artifact_preparation_seconds",
        lambda: prepare_assignment(
            scenario=scenario,
            config=assignment_config,
            timetable_index=timetable_index,
        ),
    )
    inputs = _timed(
        diagnostics,
        "assignment_input_adapter_seconds",
        lambda: build_assignment_inputs(artifacts=artifacts),
    )
    routing = _timed(
        diagnostics,
        "fixed_routing_shard_preparation_seconds",
        lambda: prepare_fixed_routing(inputs=inputs, theta=assignment_config.theta_default),
    )

    diagnostics.update(
        {
            "schema_version": 1,
            "benchmark": f"direct_scheduled_preprocessing_{label}",
            "scenario_directory": str(scenario_directory),
            "scientific_provenance": {
                "scenario_fingerprint": hashlib.sha256(
                    scenario_fingerprint_payload_json(scenario).encode("utf-8")
                ).hexdigest(),
                "canonical_timetable_fingerprint": timetable_index.fingerprint,
                "graph_arrays": {
                    name: _sha256_array(getattr(graph, name))
                    for name in ("tail", "head", "link_type", "topo_order")
                },
                "structural_topology_fingerprint": topology.fingerprint,
                "od_universe_fingerprint": universe.fingerprint,
                "od_time_expansion_fingerprint": expansion.fingerprint,
                "fixed_theta": float(assignment_config.theta_default),
            },
            "dimensions": {
                "stops": len(scenario.stops),
                "trips": len(scenario.timetable.trips),
                "stop_times": len(scenario.timetable.stop_times),
                "candidate_pairs": universe.pair_count,
                "candidate_pair_exclusions": len(universe.exclusions),
                "od_time_cells": expansion.cell_count,
                "retained_od_time_cells": expansion.cell_count,
                "graph_nodes": graph.num_nodes,
                "graph_links": graph.num_links,
                "structural_metric_records": len(metrics),
                "routing_groups": int(routing.group_dest_node.shape[0]),
            },
            "execution_provenance": {
                "worker_count": 1,
                "assignment_config": assignment_config.__dict__
                if hasattr(assignment_config, "__dict__")
                else {
                    name: getattr(assignment_config, name)
                    for name in assignment_config.__dataclass_fields__
                },
            },
            "unavailable_retired_stages": {
                "support_discovery_seconds": None,
                "measurement_shard_seconds": None,
                "reason": "timetable-journey response backend is retired",
            },
            "wall_time_seconds": perf_counter() - started,
            "peak_rss_bytes": _peak_rss_bytes(),
        }
    )
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()
    report = run_baseline(
        scenario_directory=args.scenario.resolve(), label=str(args.label)
    )
    baseline_path = args.output.parent / "preprocessing_baseline_geneva.json"
    if str(args.label) == "optimized" and baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        report["comparison_to_baseline"] = {
            "od_time_expansion_speedup": (
                baseline["od_time_expansion_seconds"]
                / report["od_time_expansion_seconds"]
            ),
            "wall_time_speedup": (
                baseline["wall_time_seconds"] / report["wall_time_seconds"]
            ),
            "same_scientific_provenance": (
                baseline["scientific_provenance"] == report["scientific_provenance"]
            ),
            "same_dimensions": baseline["dimensions"] == report["dimensions"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
