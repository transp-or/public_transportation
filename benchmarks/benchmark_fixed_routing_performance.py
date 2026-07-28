"""Benchmark dynamic versus precomputed fixed routing on ``simple_example_02``.

The script checks numerical equivalence before reporting timings. Timing values
are machine-dependent and intentionally remain outside pytest assertions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from statistics import median
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.domain import Scenario, read_fixed_demand_csv
from public_transportation.inference.assignment_adapter import (
    assign_link_flow,
    assign_link_flow_fixed_routing,
    build_assignment_inputs,
    prepare_fixed_routing,
)
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.likelihood import (
    loglikelihood_from_link_flow,
    prepare_likelihood_inputs,
)
from public_transportation.inference.od_parameter_layout import build_od_parameter_layout
from public_transportation.inference.priors import build_f0_from_scenario_demand
from public_transportation.measurement import (
    build_mapping_spec_strict,
    read_measurements_csv,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs/source/examples"
NETWORK_FILES = (
    "metadata.json",
    "stops.csv",
    "lines.csv",
    "trips.csv",
    "stop_times.csv",
    "time_bins.csv",
)


def _load_scenario(directory: Path, example: Path) -> Scenario:
    for name in NETWORK_FILES:
        shutil.copy2(example / "data" / name, directory / name)
    shutil.copy2(example / "pre_processing/results/demand.csv", directory / "demand.csv")
    return Scenario.from_folder(directory, strict=True)


def _timed(call):
    started = perf_counter()
    value = call()
    jax.block_until_ready(value)
    return perf_counter() - started, value


def _warm_median(call, repeats: int) -> float:
    return median(_timed(call)[0] for _ in range(repeats))


def _cache_bytes(routing) -> int:
    """Incremental array storage owned by the cache, excluding source aliases."""
    arrays = (
        routing.theta,
        routing.effective_group_link_mask,
        routing.group_link_probability,
    )
    return int(sum(np.asarray(value).nbytes for value in arrays))


def _measure_path(
    *,
    label: str,
    inputs,
    demand: jax.Array,
    prepared_likelihood,
    theta: float,
    repeats: int,
) -> dict:
    first_preparation_s, routing = _timed(
        lambda: prepare_fixed_routing(inputs=inputs, theta=theta)
    )
    warm_preparation_s = _warm_median(
        lambda: prepare_fixed_routing(inputs=inputs, theta=theta),
        repeats,
    )

    def dynamic_forward(value):
        return assign_link_flow(
            inputs=inputs,
            f=value,
            theta=jnp.asarray(theta, dtype=value.dtype),
        )

    def cached_forward(value):
        return assign_link_flow_fixed_routing(inputs=inputs, routing=routing, f=value)

    def objective(forward, value):
        flow = forward(value)
        return -loglikelihood_from_link_flow(
            link_flow=flow,
            prepared=prepared_likelihood,
            theta=jnp.asarray(theta, dtype=value.dtype),
            rho=jnp.asarray(1.0, dtype=value.dtype),
            r=jnp.asarray(50.0, dtype=value.dtype),
        )

    dynamic_forward_jit = jax.jit(dynamic_forward)
    cached_forward_jit = jax.jit(cached_forward)
    dynamic_value_grad = jax.jit(
        jax.value_and_grad(lambda value: objective(dynamic_forward, value))
    )
    cached_value_grad = jax.jit(
        jax.value_and_grad(lambda value: objective(cached_forward, value))
    )

    dynamic_first_forward_s, dynamic_flow = _timed(lambda: dynamic_forward_jit(demand))
    dynamic_warm_forward_s = _warm_median(
        lambda: dynamic_forward_jit(demand), repeats
    )
    cached_first_forward_s, cached_flow = _timed(lambda: cached_forward_jit(demand))
    cached_warm_forward_s = _warm_median(lambda: cached_forward_jit(demand), repeats)

    dynamic_first_vg_s, (dynamic_objective, dynamic_gradient) = _timed(
        lambda: dynamic_value_grad(demand)
    )
    dynamic_warm_vg_s = _warm_median(lambda: dynamic_value_grad(demand), repeats)
    cached_first_vg_s, (cached_objective, cached_gradient) = _timed(
        lambda: cached_value_grad(demand)
    )
    cached_warm_vg_s = _warm_median(lambda: cached_value_grad(demand), repeats)

    np.testing.assert_allclose(cached_flow, dynamic_flow, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        cached_objective,
        dynamic_objective,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        cached_gradient,
        dynamic_gradient,
        rtol=3e-5,
        atol=3e-5,
    )

    return {
        "layout": label,
        "theta": theta,
        "num_assignment_od": int(demand.shape[0]),
        "num_destination_groups": int(inputs.group_dest_node.shape[0]),
        "routing_first_preparation_s": first_preparation_s,
        "routing_warm_preparation_s": warm_preparation_s,
        "cache_bytes": _cache_bytes(routing),
        "cache_mib": _cache_bytes(routing) / (1024.0**2),
        "dynamic_first_forward_s": dynamic_first_forward_s,
        "dynamic_warm_forward_s": dynamic_warm_forward_s,
        "cached_first_forward_s": cached_first_forward_s,
        "cached_warm_forward_s": cached_warm_forward_s,
        "warm_forward_speedup": dynamic_warm_forward_s / cached_warm_forward_s,
        "dynamic_first_value_gradient_s": dynamic_first_vg_s,
        "dynamic_warm_value_gradient_s": dynamic_warm_vg_s,
        "cached_first_value_gradient_s": cached_first_vg_s,
        "cached_warm_value_gradient_s": cached_warm_vg_s,
        "warm_value_gradient_speedup": dynamic_warm_vg_s / cached_warm_vg_s,
        "objective": float(np.asarray(dynamic_objective)),
        "gradient_norm": float(np.linalg.norm(np.asarray(dynamic_gradient))),
        "max_link_flow_abs_error": float(
            np.max(np.abs(np.asarray(cached_flow) - np.asarray(dynamic_flow)))
        ),
        "max_gradient_abs_error": float(
            np.max(np.abs(np.asarray(cached_gradient) - np.asarray(dynamic_gradient)))
        ),
    }


def run_benchmark(
    *,
    repeats: int,
    theta_values: tuple[float, ...],
    example_name: str = "simple_example_02",
) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    if not theta_values or any(not np.isfinite(value) or value <= 0 for value in theta_values):
        raise ValueError("theta_values must contain positive finite values.")

    if example_name not in {"simple_example_02", "geneva_gtfs"}:
        raise ValueError(f"Unsupported benchmark example: {example_name!r}.")
    example = EXAMPLES / example_name
    with tempfile.TemporaryDirectory(prefix="pt-fixed-routing-performance-") as temporary:
        scenario = _load_scenario(Path(temporary), example)
        fixed = read_fixed_demand_csv(example / "data/fixed_demand.csv", scenario=scenario)
        layout = build_od_parameter_layout(scenario=scenario, fixed_demand=fixed)
        compact = build_compact_od_assignment_layout(parameter_layout=layout)
        artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
        full_inputs = build_assignment_inputs(artifacts=artifacts)
        compact_inputs = build_assignment_inputs(
            artifacts=artifacts,
            compact_layout=compact,
        )
        id_manager = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)
        mapped = build_mapping_spec_strict(
            id_manager=id_manager,
            table=read_measurements_csv(
                example / "pre_processing/results/measurements_boarding_alighting.csv"
            ),
            include_link_lists_for_report=False,
        )
        prepared_likelihood = prepare_likelihood_inputs(
            y_obs=mapped.y_obs,
            spec=mapped.spec,
        )
        f0 = jnp.asarray(
            build_f0_from_scenario_demand(
                scenario=scenario,
                id_manager=id_manager,
                dtype=jnp.float32,
            )
        )
        z = jnp.linspace(-0.35, 0.40, layout.num_free, dtype=jnp.float32)
        full_demand = layout.reconstruct_jax(z)
        compact_demand = compact.assemble_compact_jax(z)
        cases = []
        for theta in theta_values:
            cases.append(
                _measure_path(
                    label="full",
                    inputs=full_inputs,
                    demand=full_demand,
                    prepared_likelihood=prepared_likelihood,
                    theta=theta,
                    repeats=repeats,
                )
            )
            cases.append(
                _measure_path(
                    label="compact",
                    inputs=compact_inputs,
                    demand=compact_demand,
                    prepared_likelihood=prepared_likelihood,
                    theta=theta,
                    repeats=repeats,
                )
            )

    return {
        "example": example_name,
        "jax_backend": jax.default_backend(),
        "arithmetic": "float32",
        "repeats": repeats,
        "num_od_total": int(f0.shape[0]),
        "num_free_od": layout.num_free,
        "num_fixed_od": layout.num_fixed,
        "num_links": int(artifacts.graph.num_links),
        "num_measurements": int(mapped.spec.num_measurements),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--theta", type=float, nargs="+", default=[1.0, 5.0])
    parser.add_argument(
        "--example",
        choices=("simple_example_02", "geneva_gtfs"),
        default="simple_example_02",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(
        repeats=args.repeats,
        theta_values=tuple(args.theta),
        example_name=args.example,
    )
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
