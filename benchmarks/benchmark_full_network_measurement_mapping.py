"""Profile event-indexed strict measurement mapping on the TPG full network."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = ROOT.parent / "public_transport_TPG/processed_data_for_models/full_network"


def _compile_and_time(kernel, argument):
    import jax
    import numpy as np

    started = perf_counter()
    lowered = jax.jit(kernel).lower(argument)
    lowering = perf_counter() - started
    started = perf_counter()
    compiled = lowered.compile()
    compilation = perf_counter() - started
    started = perf_counter()
    first = compiled(argument)
    jax.block_until_ready(first)
    first_execution = perf_counter() - started
    samples = []
    for _ in range(5):
        started = perf_counter()
        value = compiled(argument)
        jax.block_until_ready(value)
        samples.append(perf_counter() - started)
    return {
        "lowering_seconds": lowering,
        "compilation_seconds": compilation,
        "first_execution_seconds": first_execution,
        "warm_median_seconds": float(np.median(samples)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-folder", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from public_transportation.assignment import AssignmentConfig
    from public_transportation.assignment.assign import prepare_assignment
    from public_transportation.assignment.id_manager import AssignmentIDManager
    from public_transportation.domain import Scenario
    from public_transportation.measurement import (
        build_event_aligned_aggregation_spec,
        predict_measurements_event_aligned,
        profile_mapping_spec_strict,
        read_measurements_csv,
    )
    from public_transportation.measurement.likelihood_jax import (
        predict_measurements_from_link_flow,
    )

    total_started = perf_counter()
    phases = {}
    started = perf_counter()
    scenario = Scenario.from_folder(args.scenario_folder.resolve(), strict=True)
    phases["scenario_loading_seconds"] = perf_counter() - started
    started = perf_counter()
    artifacts = prepare_assignment(
        scenario=scenario, config=AssignmentConfig(), cache_policy="off"
    )
    phases["assignment_preparation_seconds"] = perf_counter() - started
    started = perf_counter()
    id_manager = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)
    phases["id_manager_seconds"] = perf_counter() - started
    started = perf_counter()
    table = read_measurements_csv(
        args.scenario_folder.resolve() / "measurements_boarding_alighting.csv"
    )
    phases["measurement_csv_seconds"] = perf_counter() - started
    mapping, profile = profile_mapping_spec_strict(
        id_manager=id_manager,
        table=table,
        include_link_lists_for_report=False,
    )
    measurement = np.asarray(mapping.spec.measurement_index, dtype=np.int64)
    link = np.asarray(mapping.spec.link_index, dtype=np.int64)
    multiplicity = np.bincount(
        measurement, minlength=mapping.spec.num_measurements
    )
    measurement_types = np.asarray(
        [entry.measurement_type for entry in mapping.info.entries]
    )
    direct = build_event_aligned_aggregation_spec(mapping.spec)
    link_flow = jnp.linspace(0.0, 1.0, id_manager.num_links, dtype=jnp.float32)
    generic_measurement = jnp.asarray(mapping.spec.measurement_index)
    generic_link = jnp.asarray(mapping.spec.link_index)
    primary = jnp.asarray(direct.primary_link_index)
    secondary_measurement = jnp.asarray(direct.secondary_measurement_index)
    secondary_link = jnp.asarray(direct.secondary_link_index)

    def generic_kernel(flow):
        return predict_measurements_from_link_flow(
            flow,
            spec_num_measurements=mapping.spec.num_measurements,
            spec_measurement_index=generic_measurement,
            spec_link_index=generic_link,
        )

    def direct_kernel(flow):
        return predict_measurements_event_aligned(
            flow, primary, secondary_measurement, secondary_link
        )

    generic_prediction = generic_kernel(link_flow)
    direct_prediction = direct_kernel(link_flow)
    jax.block_until_ready((generic_prediction, direct_prediction))
    maximum_prediction_difference = float(
        np.max(np.abs(np.asarray(generic_prediction - direct_prediction)))
    )
    aggregation_benchmark = {
        "generic": _compile_and_time(generic_kernel, link_flow),
        "event_aligned": _compile_and_time(direct_kernel, link_flow),
        "maximum_prediction_difference": maximum_prediction_difference,
    }

    def type_multiplicity(label: str) -> dict[str, int | float]:
        selected = multiplicity[measurement_types == label]
        return {
            "measurements": int(selected.size),
            "minimum": int(selected.min()),
            "maximum": int(selected.max()),
            "mean": float(selected.mean()),
            "one_link": int(np.count_nonzero(selected == 1)),
            "multiple_links": int(np.count_nonzero(selected > 1)),
        }
    report = {
        "schema_version": 1,
        "mode": "event_indexed_strict_mapping",
        "platform": {
            "system": platform.platform(),
            "architecture": platform.machine(),
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
        },
        "dimensions": {
            "nodes": id_manager.num_nodes,
            "links": id_manager.num_links,
            "measurements": mapping.spec.num_measurements,
            "mapping_contributions": int(link.size),
            "unique_contributing_links": int(np.unique(link).size),
        },
        "profile": asdict(profile),
        "aggregation_benchmark": aggregation_benchmark,
        "multiplicity": {
            "minimum": int(multiplicity.min()),
            "maximum": int(multiplicity.max()),
            "mean": float(multiplicity.mean()),
            "one_link_measurements": int(np.count_nonzero(multiplicity == 1)),
            "multiple_link_measurements": int(np.count_nonzero(multiplicity > 1)),
            "histogram": {
                str(value): int(count)
                for value, count in zip(
                    *np.unique(multiplicity, return_counts=True), strict=True
                )
            },
            "boarding": type_multiplicity("boarding"),
            "alighting": type_multiplicity("alighting"),
        },
        "phases": phases,
        "storage_bytes": {
            "measurement_index": int(mapping.spec.measurement_index.nbytes),
            "link_index": int(mapping.spec.link_index.nbytes),
            "observations": int(np.asarray(mapping.y_obs).nbytes),
            "event_aligned_primary": int(direct.primary_link_index.nbytes),
            "event_aligned_secondary_measurement": int(
                direct.secondary_measurement_index.nbytes
            ),
            "event_aligned_secondary_link": int(direct.secondary_link_index.nbytes),
        },
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "total_process_seconds": perf_counter() - total_started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
