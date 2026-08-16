"""Bounded matrix-free gravity benchmark on packaged Simple Example 02."""

from __future__ import annotations

import argparse
import json
import resource
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from docs.source.examples.direct_scheduled_gravity_validation import _features
from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.domain import Scenario, read_fixed_demand_csv
from public_transportation.inference.assignment_adapter import (
    build_assignment_inputs,
    prepare_fixed_routing,
)
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.fixed_routing_matrix_free_operator import (
    MatrixFreeFixedRoutingMeasurementOperator,
)
from public_transportation.inference.gravity import (
    GravityEstimatorConfig,
    GravityExecutionPolicy,
    GravityLikelihood,
    GravityModelSpecification,
    GravityObjectiveProblem,
    GravityParameterLayout,
    estimate_gravity_model,
    gravity_value_and_gradient_adjoint,
)
from public_transportation.inference.od_parameter_layout import (
    build_od_parameter_layout,
)
from public_transportation.measurement import (
    build_mapping_spec_strict,
    read_measurements_csv,
)
from public_transportation.assignment.id_manager import AssignmentIDManager

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/source/examples/simple_example_02"
DEFAULT_OUTPUT = ROOT / "benchmarks/matrix_free_gravity.json"


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if value > 10_000_000 else value * 1024


def run_benchmark(*, checkpoint: Path) -> dict[str, object]:
    data = EXAMPLE / "data"
    scenario = Scenario.from_folder(
        data, strict=True, demand_file=data / "prior_demand.csv"
    )
    od_layout = build_od_parameter_layout(
        scenario=scenario,
        fixed_demand=read_fixed_demand_csv(
            data / "fixed_demand.csv", scenario=scenario
        ),
    )
    compact = build_compact_od_assignment_layout(parameter_layout=od_layout)
    features = _features(scenario, od_layout, compact)
    artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
    inputs = build_assignment_inputs(artifacts=artifacts, compact_layout=compact)
    routing = prepare_fixed_routing(inputs=inputs, theta=1.0)
    id_manager = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)
    mapped = build_mapping_spec_strict(
        id_manager=id_manager,
        table=read_measurements_csv(
            EXAMPLE / "pre_processing/results/measurements_boarding_alighting.csv"
        ),
        include_link_lists_for_report=False,
    )
    rss_before = _peak_rss_bytes()
    operator = MatrixFreeFixedRoutingMeasurementOperator(
        inputs=inputs,
        routing=routing,
        spec=mapped.spec,
        compact_layout=compact,
    )
    preparation = operator.prepare_device_products(products="forward_and_transpose")
    layout = GravityParameterLayout(GravityModelSpecification())
    problem = GravityObjectiveProblem(
        features=features,
        parameter_layout=layout,
        operator=operator,
        observations=np.asarray(mapped.y_obs),
        likelihood=GravityLikelihood.NEGATIVE_BINOMIAL,
    )
    raw = jnp.asarray(
        layout.raw_from_physical((0.5, 1.0, 10.0)), dtype=features.dtype
    )
    kernel = jax.jit(
        lambda value: gravity_value_and_gradient_adjoint(value, problem=problem)
    )
    started = perf_counter()
    traced = kernel.trace(raw)
    tracing = perf_counter() - started
    started = perf_counter()
    lowered = traced.lower()
    lowering = perf_counter() - started
    started = perf_counter()
    compiled = lowered.compile()
    compilation = perf_counter() - started
    started = perf_counter()
    first = compiled(raw)
    jax.block_until_ready(first)
    first_seconds = perf_counter() - started
    started = perf_counter()
    warm = compiled(raw + 0.01)
    jax.block_until_ready(warm)
    warm_seconds = perf_counter() - started
    checkpoint.unlink(missing_ok=True)
    result = estimate_gravity_model(
        problem=problem,
        compact_layout=compact,
        initial_raw_parameters=np.asarray(raw),
        config=GravityEstimatorConfig(maximum_iterations=1),
        execution=GravityExecutionPolicy(
            gradient_strategy="adjoint", checkpoint_path=checkpoint
        ),
    )
    report = {
        "schema_version": 1,
        "example": "simple_example_02",
        "logical_operator": {
            "measurements": operator.num_measurements,
            "free_od": operator.num_free_od,
            "entries": operator.num_measurements * operator.num_free_od,
            "avoided_dense_float32_bytes": (
                operator.num_measurements * operator.num_free_od * 4
            ),
            "global_matrix_constructed": False,
        },
        "preparation": asdict(preparation),
        "objective": {
            "tracing_seconds": tracing,
            "lowering_seconds": lowering,
            "compilation_seconds": compilation,
            "first_execution_seconds": first_seconds,
            "warm_execution_seconds": warm_seconds,
            "lowered_text_bytes": len(lowered.as_text().encode("utf-8")),
        },
        "memory": {
            "rss_before_bytes": rss_before,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
        "checkpoint": {
            "bytes": checkpoint.stat().st_size,
            "status": result.status,
            "iterations": result.iterations,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=Path("/tmp/matrix-free-gravity-checkpoint.json"))
    arguments = parser.parse_args()
    report = run_benchmark(checkpoint=arguments.checkpoint)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
