"""Compare sparse prediction kernels on the cached TPG MAP problem."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
TPG_MODEL = ROOT.parent / "public_transport_TPG/models/two_lines_morning_time"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warm-evaluations", type=int, default=10)
    args = parser.parse_args()
    os.environ["PUBLIC_TRANSPORTATION_OPERATOR_CACHE_DIR"] = str(args.operator_cache)
    sys.path.insert(0, str(TPG_MODEL))

    from map_profile import MapSettings, prepare_map_problem

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental.sparse import BCOO
    from public_transportation.inference.maximum_likelihood_pipeline import (
        _fixed_operator_logprior,
    )
    from public_transportation.inference.parameterization import smooth_bound
    from public_transportation.measurement.likelihood_jax import (
        negbinom_loglikelihood,
    )

    prepared_map = prepare_map_problem(
        settings=MapSettings(
            fixed_measurement_operator="bcoo",
            fixed_measurement_operator_chunk_size=1024,
        )
    )
    problem = prepared_map.problem
    operator = problem.fixed_measurement_operator
    assert operator is not None
    matrix = operator.matrix
    assert isinstance(matrix, BCOO)
    data = problem.data

    def finish(prediction, dynamic_data):
        mu = dynamic_data["rho"] * (prediction + dynamic_data["fixed_offset"])
        mu = jnp.maximum(mu, jnp.asarray(1e-9, dtype=mu.dtype))
        return negbinom_loglikelihood(
            y_obs=dynamic_data["y_obs"], mu=mu, r=dynamic_data["r_nb"]
        )

    def demand(parameter, dynamic_data):
        z = smooth_bound(parameter, dynamic_data["z_bound"])
        return dynamic_data["free_f0"] * jnp.exp(z)

    def bcoo_loglik(parameter, dynamic_data):
        sparse = BCOO(
            (dynamic_data["operator_values"], dynamic_data["operator_indices"]),
            shape=matrix.shape,
            indices_sorted=True,
            unique_indices=True,
        )
        return finish(sparse @ demand(parameter, dynamic_data), dynamic_data)

    def scatter_loglik(parameter, dynamic_data):
        free_demand = demand(parameter, dynamic_data)
        values = dynamic_data["operator_values"] * free_demand[
            dynamic_data["operator_columns"]
        ]
        prediction = jnp.zeros_like(dynamic_data["fixed_offset"])
        return finish(
            prediction.at[dynamic_data["operator_rows"]].add(values), dynamic_data
        )

    def segment_loglik(parameter, dynamic_data):
        free_demand = demand(parameter, dynamic_data)
        values = dynamic_data["operator_values"] * free_demand[
            dynamic_data["operator_columns"]
        ]
        prediction = jax.ops.segment_sum(
            values,
            dynamic_data["operator_rows"],
            num_segments=dynamic_data["fixed_offset"].shape[0],
            indices_are_sorted=True,
        )
        return finish(prediction, dynamic_data)

    benchmark_data = dict(data)
    benchmark_data["operator_indices"] = matrix.indices
    def prior(parameter):
        return _fixed_operator_logprior(parameter, sigma_z=1.0)

    def measure(name, loglik):
        def objective(parameter, dynamic_data):
            ll = loglik(parameter, dynamic_data)
            return -ll - prior(parameter)

        kernel = jax.jit(jax.value_and_grad(objective))
        started = perf_counter()
        traced = kernel.trace(problem.theta0, benchmark_data)
        trace_seconds = perf_counter() - started
        started = perf_counter()
        lowered = traced.lower()
        lower_seconds = perf_counter() - started
        started = perf_counter()
        compiled = lowered.compile()
        compile_seconds = perf_counter() - started
        started = perf_counter()
        first = compiled(problem.theta0, benchmark_data)
        jax.block_until_ready(first)
        first_seconds = perf_counter() - started
        warm = []
        for _ in range(args.warm_evaluations):
            started = perf_counter()
            value = compiled(problem.theta0, benchmark_data)
            jax.block_until_ready(value)
            warm.append(perf_counter() - started)
        value, gradient = first
        return {
            "name": name,
            "trace_seconds": trace_seconds,
            "lower_seconds": lower_seconds,
            "compile_seconds": compile_seconds,
            "first_execution_seconds": first_seconds,
            "warm_median_seconds": float(np.median(warm)),
            "objective": float(np.asarray(value)),
            "gradient_norm": float(np.linalg.norm(np.asarray(gradient))),
            "gradient": np.asarray(gradient),
        }

    measured = [
        measure("bcoo_matvec", bcoo_loglik),
        measure("direct_scatter", scatter_loglik),
        measure("sorted_segment_sum", segment_loglik),
    ]
    baseline = measured[0]
    baseline_gradient = baseline["gradient"]
    for item in measured:
        gradient = item.pop("gradient")
        item["objective_absolute_difference"] = abs(
            item["objective"] - baseline["objective"]
        )
        item["gradient_max_absolute_difference"] = float(
            np.max(np.abs(gradient - baseline_gradient))
        )
    report = {
        "operator": {
            "shape": list(matrix.shape),
            "nonzero_count": int(matrix.nse),
            "indices_sorted": bool(matrix.indices_sorted),
            "unique_indices": bool(matrix.unique_indices),
        },
        "variants": measured,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
