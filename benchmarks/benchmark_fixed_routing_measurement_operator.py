"""Benchmark direct fixed-routing measurement operators on a package example.

This intentionally uses a short synthetic objective and never runs a complete
estimation. The external TPG repository can reuse the same reported metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.domain import Scenario
from public_transportation.inference.assignment_adapter import (
    assign_link_flow_fixed_routing,
    build_assignment_inputs,
    prepare_fixed_routing,
)
from public_transportation.inference.fixed_routing_measurement_operator import (
    predict_measurements_fixed_operator,
    prepare_fixed_routing_measurement_operator,
)
from public_transportation.measurement.likelihood_jax import (
    negbinom_loglikelihood,
    predict_measurements_from_link_flow,
)
from public_transportation.measurement.mapping import AggregationSpec

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/source/examples/simple_example_02"
NETWORK_FILES = (
    "metadata.json",
    "stops.csv",
    "lines.csv",
    "trips.csv",
    "stop_times.csv",
    "time_bins.csv",
)


def _synchronize(value):
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def _time_first_and_warm(function, *, repeats: int) -> tuple[float, float]:
    start = perf_counter()
    _synchronize(function())
    first = perf_counter() - start
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        _synchronize(function())
        samples.append(perf_counter() - start)
    return first, float(np.median(samples))


def _prepare_example():
    directory = Path(tempfile.mkdtemp(prefix="measurement-operator-"))
    for name in NETWORK_FILES:
        shutil.copy2(EXAMPLE / "data" / name, directory / name)
    shutil.copy2(
        EXAMPLE / "pre_processing/results/demand.csv", directory / "demand.csv"
    )
    scenario = Scenario.from_folder(directory, strict=True)
    artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
    return directory, build_assignment_inputs(artifacts=artifacts)


def benchmark(*, repeats: int, chunk_size: int) -> dict[str, object]:
    directory, inputs = _prepare_example()
    try:
        routing_start = perf_counter()
        routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
        routing.group_link_probability.block_until_ready()
        routing_seconds = perf_counter() - routing_start

        num_links = int(inputs.graph.num_links)
        links = np.arange(min(num_links, 24), dtype=np.int32)
        spec = AggregationSpec(
            num_measurements=8,
            measurement_index=np.arange(links.size, dtype=np.int32) % 8,
            link_index=links,
        )
        demand = jnp.linspace(0.5, 2.0, inputs.od_origin_node.shape[0])
        y_obs = jnp.arange(1, 9, dtype=demand.dtype)
        rho = jnp.asarray(0.8, dtype=demand.dtype)
        dispersion = jnp.asarray(20.0, dtype=demand.dtype)

        def reference_measurements(value):
            link_flow = assign_link_flow_fixed_routing(
                inputs=inputs,
                routing=routing,
                f=value,
            )
            return predict_measurements_from_link_flow(
                link_flow,
                spec_num_measurements=spec.num_measurements,
                spec_measurement_index=jnp.asarray(spec.measurement_index),
                spec_link_index=jnp.asarray(spec.link_index),
            )

        def likelihood(mean):
            return negbinom_loglikelihood(
                y_obs=y_obs,
                mu=jnp.maximum(rho * mean, 1e-9),
                r=dispersion,
            )

        reference_value_and_grad = jax.jit(
            jax.value_and_grad(lambda value: -likelihood(reference_measurements(value)))
        )
        reference_forward = jax.jit(reference_measurements)
        reference_first, reference_warm = _time_first_and_warm(
            lambda: reference_forward(demand), repeats=repeats
        )
        reference_vg_first, reference_vg_warm = _time_first_and_warm(
            lambda: reference_value_and_grad(demand), repeats=repeats
        )

        representations = {}
        for representation in ("dense", "bcoo"):
            operator = prepare_fixed_routing_measurement_operator(
                inputs=inputs,
                routing=routing,
                spec=spec,
                assignment_fingerprint="simple-example-02",
                representation=representation,
                chunk_size=chunk_size,
            )

            def direct_measurements(value):
                return predict_measurements_fixed_operator(
                    operator=operator,
                    free_demand=value,
                    rho=jnp.asarray(1.0, dtype=value.dtype),
                )

            direct_forward = jax.jit(direct_measurements)
            direct_value_and_grad = jax.jit(
                jax.value_and_grad(
                    lambda value: -likelihood(direct_measurements(value))
                )
            )
            first, warm = _time_first_and_warm(
                lambda: direct_forward(demand), repeats=repeats
            )
            vg_first, vg_warm = _time_first_and_warm(
                lambda: direct_value_and_grad(demand), repeats=repeats
            )
            difference = np.max(
                np.abs(
                    np.asarray(direct_forward(demand))
                    - np.asarray(reference_forward(demand))
                )
            )
            saved_per_evaluation = reference_vg_warm - vg_warm
            break_even = (
                None
                if saved_per_evaluation <= 0.0
                else operator.metrics.construction_seconds / saved_per_evaluation
            )
            representations[representation] = {
                "construction_seconds": operator.metrics.construction_seconds,
                "first_forward_seconds": first,
                "warm_forward_seconds": warm,
                "first_value_and_gradient_seconds": vg_first,
                "warm_value_and_gradient_seconds": vg_warm,
                "dense_bytes": operator.metrics.dense_bytes,
                "stored_bytes": operator.metrics.stored_bytes,
                "peak_construction_bytes": operator.metrics.peak_construction_bytes,
                "nonzero_entries": operator.metrics.nonzero_entries,
                "total_entries": operator.metrics.total_entries,
                "density": operator.metrics.density,
                "max_measurement_difference": float(difference),
                "break_even_evaluations": break_even,
            }

        return {
            "example": "simple_example_02",
            "num_active_od": int(inputs.od_origin_node.shape[0]),
            "num_measurements": spec.num_measurements,
            "num_links": num_links,
            "routing_preparation_seconds": routing_seconds,
            "repeats": repeats,
            "chunk_size": chunk_size,
            "reference": {
                "first_forward_seconds": reference_first,
                "warm_forward_seconds": reference_warm,
                "first_value_and_gradient_seconds": reference_vg_first,
                "warm_value_and_gradient_seconds": reference_vg_warm,
            },
            "representations": representations,
            "recommendation": (
                "Enable manually when the expected objective-evaluation count exceeds "
                "the measured break-even point and the dense/BCOO storage fits the memory budget."
            ),
        }
    finally:
        shutil.rmtree(directory)


def _markdown(report: dict[str, object]) -> str:
    reference = report["reference"]
    lines = [
        "# Fixed-routing measurement operator benchmark",
        "",
        f"Example: `{report['example']}`",
        "",
        f"- Active OD cells: {report['num_active_od']}",
        f"- Measurements: {report['num_measurements']}",
        f"- Links: {report['num_links']}",
        f"- Reference warm value-and-gradient: {reference['warm_value_and_gradient_seconds']:.6f} s",
        "",
        "| Representation | Construction (s) | Warm forward (s) | Warm value+grad (s) | Stored (MiB) | Density | Break-even evals |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in report["representations"].items():
        break_even = metrics["break_even_evaluations"]
        break_even_text = "n/a" if break_even is None else f"{break_even:.1f}"
        lines.append(
            f"| {name} | {metrics['construction_seconds']:.6f} | "
            f"{metrics['warm_forward_seconds']:.6f} | "
            f"{metrics['warm_value_and_gradient_seconds']:.6f} | "
            f"{metrics['stored_bytes'] / 2**20:.3f} | "
            f"{metrics['density']:.4f} | {break_even_text} |"
        )
    lines.extend(("", str(report["recommendation"]), ""))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "benchmarks/fixed_routing_measurement_operator",
    )
    args = parser.parse_args()
    report = benchmark(repeats=args.repeats, chunk_size=args.chunk_size)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    args.output_prefix.with_suffix(".md").write_text(_markdown(report))


if __name__ == "__main__":
    main()
