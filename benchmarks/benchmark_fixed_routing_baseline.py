"""Capture the fixed-theta assignment baseline for ``simple_example_02``.

This benchmark is the correctness and timing baseline for the planned
fixed-routing optimization.  It deliberately uses the current dynamic routing
implementation.  Timings are reported, but never asserted: they depend on the
machine and JAX backend.

Run from the repository root, for example::

    uv run python benchmarks/benchmark_fixed_routing_baseline.py \
        --repeats 5 --output-json /tmp/fixed-routing-baseline.json \
        --output-npz /tmp/fixed-routing-baseline.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.domain import Scenario, read_fixed_demand_csv
from public_transportation.inference.assignment_adapter import (
    assign_link_flow_fixed_routing,
    assign_link_flow,
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
from public_transportation.inference.parameterization import smooth_bound
from public_transportation.inference.priors import build_f0_from_scenario_demand
from public_transportation.measurement import (
    build_mapping_spec_strict,
    read_measurements_csv,
)
from public_transportation.measurement.likelihood_jax import (
    predict_measurements_from_link_flow,
)


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
FIXED_THETA = 1.0
RHO = 1.0
NB_DISPERSION = 50.0
Z_BOUND = 6.0


@dataclass(frozen=True)
class BaselineCase:
    name: str
    parameter: jax.Array


@dataclass(frozen=True)
class BaselineSetup:
    cases: tuple[BaselineCase, ...]
    forward: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]]
    objective: Callable[[jax.Array], jax.Array]
    cached_forward: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array]]
    cached_objective: Callable[[jax.Array], jax.Array]
    metadata: dict[str, int | float | str]


def _load_scenario(directory: Path) -> Scenario:
    for name in NETWORK_FILES:
        shutil.copy2(EXAMPLE / "data" / name, directory / name)
    shutil.copy2(EXAMPLE / "pre_processing/results/demand.csv", directory / "demand.csv")
    return Scenario.from_folder(directory, strict=True)


def build_baseline_setup(directory: Path) -> BaselineSetup:
    """Build the reference dynamic-routing computation and deterministic probes."""
    scenario = _load_scenario(directory)
    fixed = read_fixed_demand_csv(EXAMPLE / "data/fixed_demand.csv", scenario=scenario)
    layout = build_od_parameter_layout(scenario=scenario, fixed_demand=fixed)
    compact = build_compact_od_assignment_layout(parameter_layout=layout)
    artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
    assignment_inputs = build_assignment_inputs(artifacts=artifacts, compact_layout=compact)
    fixed_routing = prepare_fixed_routing(inputs=assignment_inputs, theta=FIXED_THETA)
    id_manager = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)
    measurements = read_measurements_csv(
        EXAMPLE / "pre_processing/results/measurements_boarding_alighting.csv"
    )
    mapped = build_mapping_spec_strict(
        id_manager=id_manager,
        table=measurements,
        include_link_lists_for_report=False,
    )
    prepared = prepare_likelihood_inputs(y_obs=mapped.y_obs, spec=mapped.spec)
    f0 = build_f0_from_scenario_demand(
        scenario=scenario,
        id_manager=id_manager,
        dtype=jnp.float32,
    )
    if int(f0.shape[0]) != layout.num_od_total:
        raise ValueError("Scenario demand and parameter layout are not aligned.")

    def forward(parameter: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        bounded = smooth_bound(parameter, Z_BOUND)
        demand = compact.assemble_compact_jax(bounded)
        link_flow = assign_link_flow(
            inputs=assignment_inputs,
            f=demand,
            theta=jnp.asarray(FIXED_THETA, dtype=demand.dtype),
        )
        prediction = predict_measurements_from_link_flow(
            link_flow=link_flow,
            spec_num_measurements=mapped.spec.num_measurements,
            spec_measurement_index=prepared.spec_measurement_index,
            spec_link_index=prepared.spec_link_index,
        )
        loglik = loglikelihood_from_link_flow(
            link_flow=link_flow,
            prepared=prepared,
            theta=jnp.asarray(FIXED_THETA, dtype=demand.dtype),
            rho=jnp.asarray(RHO, dtype=demand.dtype),
            r=jnp.asarray(NB_DISPERSION, dtype=demand.dtype),
        )
        return link_flow, prediction, loglik

    def objective(parameter: jax.Array) -> jax.Array:
        _, _, loglik = forward(parameter)
        # Match simple_example_02/run_ml_fixed_theta.py, which uses pure ML
        # (prior_weight=0).
        return -loglik

    def cached_forward(parameter: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        bounded = smooth_bound(parameter, Z_BOUND)
        demand = compact.assemble_compact_jax(bounded)
        link_flow = assign_link_flow_fixed_routing(
            inputs=assignment_inputs,
            routing=fixed_routing,
            f=demand,
        )
        prediction = predict_measurements_from_link_flow(
            link_flow=link_flow,
            spec_num_measurements=mapped.spec.num_measurements,
            spec_measurement_index=prepared.spec_measurement_index,
            spec_link_index=prepared.spec_link_index,
        )
        loglik = loglikelihood_from_link_flow(
            link_flow=link_flow,
            prepared=prepared,
            theta=jnp.asarray(FIXED_THETA, dtype=demand.dtype),
            rho=jnp.asarray(RHO, dtype=demand.dtype),
            r=jnp.asarray(NB_DISPERSION, dtype=demand.dtype),
        )
        return link_flow, prediction, loglik

    def cached_objective(parameter: jax.Array) -> jax.Array:
        _, _, loglik = cached_forward(parameter)
        return -loglik

    num_free = layout.num_free
    cases = (
        BaselineCase("baseline", jnp.zeros((num_free,), dtype=jnp.float32)),
        BaselineCase(
            "perturbed",
            jnp.linspace(-0.35, 0.40, num_free, dtype=jnp.float32),
        ),
        BaselineCase(
            "alternating",
            jnp.where(
                jnp.arange(num_free) % 2 == 0,
                jnp.asarray(-0.6, dtype=jnp.float32),
                jnp.asarray(0.25, dtype=jnp.float32),
            ),
        ),
    )
    metadata: dict[str, int | float | str] = {
        "example": "simple_example_02",
        "implementation": "dynamic_routing_reference",
        "fixed_theta": FIXED_THETA,
        "num_od_total": layout.num_od_total,
        "num_free_od": layout.num_free,
        "num_fixed_od": layout.num_fixed,
        "num_fixed_zero_od": layout.num_fixed_zero,
        "num_destination_groups": int(assignment_inputs.group_dest_node.shape[0]),
        "num_links": int(artifacts.graph.num_links),
        "num_measurements": int(mapped.spec.num_measurements),
        "od_layout_fingerprint": layout.fingerprint,
        "compact_layout_fingerprint": compact.fingerprint,
    }
    return BaselineSetup(
        cases=cases,
        forward=forward,
        objective=objective,
        cached_forward=cached_forward,
        cached_objective=cached_objective,
        metadata=metadata,
    )


def _timed(call: Callable[[], object]) -> tuple[float, object]:
    started = perf_counter()
    value = call()
    jax.block_until_ready(value)
    return perf_counter() - started, value


def _array_digest(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def run_benchmark(*, repeats: int) -> tuple[dict, dict[str, np.ndarray]]:
    """Evaluate all probes, returning JSON metadata and full reference arrays."""
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    with tempfile.TemporaryDirectory(prefix="pt-fixed-routing-baseline-") as temporary:
        setup = build_baseline_setup(Path(temporary))
        forward = jax.jit(setup.forward)
        value_and_grad = jax.jit(jax.value_and_grad(setup.objective))
        cached_forward = jax.jit(setup.cached_forward)
        cached_value_and_grad = jax.jit(jax.value_and_grad(setup.cached_objective))
        report = dict(setup.metadata)
        report.update(
            jax_backend=jax.default_backend(),
            arithmetic="float32",
            repeats=repeats,
            cases=[],
        )
        arrays: dict[str, np.ndarray] = {}
        for case in setup.cases:
            compile_forward_s, forward_value = _timed(lambda: forward(case.parameter))
            warm_forward = [_timed(lambda: forward(case.parameter))[0] for _ in range(repeats)]
            compile_value_and_grad_s, objective_and_gradient = _timed(
                lambda: value_and_grad(case.parameter)
            )
            warm_value_and_grad = [
                _timed(lambda: value_and_grad(case.parameter))[0] for _ in range(repeats)
            ]
            first_cached_forward_s, cached_forward_value = _timed(
                lambda: cached_forward(case.parameter)
            )
            warm_cached_forward = [
                _timed(lambda: cached_forward(case.parameter))[0] for _ in range(repeats)
            ]
            first_cached_value_and_grad_s, cached_objective_and_gradient = _timed(
                lambda: cached_value_and_grad(case.parameter)
            )
            warm_cached_value_and_grad = [
                _timed(lambda: cached_value_and_grad(case.parameter))[0]
                for _ in range(repeats)
            ]
            link_flow, prediction, loglik = forward_value
            objective, gradient = objective_and_gradient
            cached_link_flow, cached_prediction, cached_loglik = cached_forward_value
            cached_objective, cached_gradient = cached_objective_and_gradient
            np.testing.assert_allclose(cached_link_flow, link_flow, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(cached_prediction, prediction, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(cached_loglik, loglik, rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(cached_objective, objective, rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(cached_gradient, gradient, rtol=3e-5, atol=3e-5)
            warm_forward_median = median(warm_forward)
            warm_value_and_grad_median = median(warm_value_and_grad)
            warm_cached_forward_median = median(warm_cached_forward)
            warm_cached_value_and_grad_median = median(warm_cached_value_and_grad)
            case_arrays = {
                "parameter": np.asarray(case.parameter),
                "link_flow": np.asarray(link_flow),
                "prediction": np.asarray(prediction),
                "gradient": np.asarray(gradient),
            }
            for key, value in case_arrays.items():
                arrays[f"{case.name}_{key}"] = value
            report["cases"].append(
                {
                    "name": case.name,
                    "first_forward_call_s": compile_forward_s,
                    "warm_forward_median_s": warm_forward_median,
                    "first_value_and_gradient_call_s": compile_value_and_grad_s,
                    "warm_value_and_gradient_median_s": warm_value_and_grad_median,
                    "first_cached_forward_call_s": first_cached_forward_s,
                    "warm_cached_forward_median_s": warm_cached_forward_median,
                    "warm_cached_forward_speedup": (
                        warm_forward_median / warm_cached_forward_median
                    ),
                    "first_cached_value_and_gradient_call_s": (
                        first_cached_value_and_grad_s
                    ),
                    "warm_cached_value_and_gradient_median_s": (
                        warm_cached_value_and_grad_median
                    ),
                    "warm_cached_value_and_gradient_speedup": (
                        warm_value_and_grad_median / warm_cached_value_and_grad_median
                    ),
                    "loglikelihood": float(np.asarray(loglik)),
                    "objective": float(np.asarray(objective)),
                    "gradient_norm": float(np.linalg.norm(np.asarray(gradient))),
                    "link_flow_sum": float(np.asarray(link_flow).sum()),
                    "prediction_sum": float(np.asarray(prediction).sum()),
                    "digests": {
                        key: _array_digest(value) for key, value in case_arrays.items()
                    },
                }
            )
    return report, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-npz", type=Path)
    args = parser.parse_args()
    report, arrays = run_benchmark(repeats=args.repeats)
    rendered = json.dumps(report, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    if args.output_npz is not None:
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_npz, **arrays)
    print(rendered)


if __name__ == "__main__":
    main()
