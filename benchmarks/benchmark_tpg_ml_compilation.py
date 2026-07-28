"""Fresh-process compilation benchmark for the cached TPG BCOO MAP problem."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import resource
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
TPG_MODEL = ROOT.parent / "public_transport_TPG/models/two_lines_morning_time"


def _files(directory: Path | None) -> dict[str, int]:
    if directory is None or not directory.exists():
        return {}
    return {
        str(path.relative_to(directory)): path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-cache", type=Path, required=True)
    parser.add_argument("--compilation-cache", type=Path)
    parser.add_argument("--assignment-cache", type=Path)
    parser.add_argument(
        "--assignment-cache-policy",
        choices=("off", "auto", "refresh", "readonly"),
        default="off",
    )
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--warm-evaluations", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    process_start = perf_counter()
    os.environ["PUBLIC_TRANSPORTATION_OPERATOR_CACHE_DIR"] = str(args.operator_cache)
    if args.assignment_cache is not None:
        os.environ["PUBLIC_TRANSPORTATION_ASSIGNMENT_CACHE_DIR"] = str(
            args.assignment_cache
        )
        os.environ["PUBLIC_TRANSPORTATION_ASSIGNMENT_CACHE_POLICY"] = (
            args.assignment_cache_policy
        )
    if args.compilation_cache is not None:
        os.environ["PUBLIC_TRANSPORTATION_JAX_COMPILATION_CACHE_DIR"] = str(
            args.compilation_cache
        )
        os.environ["PUBLIC_TRANSPORTATION_JAX_CACHE_MIN_COMPILE_SECONDS"] = "0"
        os.environ["PUBLIC_TRANSPORTATION_JAX_CACHE_MIN_ENTRY_BYTES"] = "0"
    cache_before = _files(args.compilation_cache)

    sys.path.insert(0, str(TPG_MODEL))
    from map_profile import MapSettings, prepare_map_problem

    import jax
    import jaxlib
    import numpy as np
    from public_transportation.estimation.maximum_likelihood import (
        MLConfig,
        compile_ml_objective,
        prepare_ml_objective,
        run_ml,
    )

    preparation_start = perf_counter()
    prepared_map = prepare_map_problem(
        settings=MapSettings(
            fixed_measurement_operator="bcoo",
            fixed_measurement_operator_chunk_size=1024,
        )
    )
    preparation_wall = perf_counter() - preparation_start
    problem = prepared_map.problem
    operator = problem.fixed_measurement_operator
    assert operator is not None
    prepared_objective = prepare_ml_objective(
        theta_example=problem.theta0,
        data=problem.data,
        loglik=problem.loglik,
        logprior=problem.logprior,
        prior_weight=1.0,
    )
    compiled = compile_ml_objective(prepared_objective)

    warm = []
    for _ in range(args.warm_evaluations):
        started = perf_counter()
        value = compiled.callable(problem.theta0, problem.data)
        jax.block_until_ready(value)
        warm.append(perf_counter() - started)

    first_run_ml_execution = None

    def instrumented(parameter, data):
        nonlocal first_run_ml_execution
        started = perf_counter()
        value = compiled.callable(parameter, data)
        jax.block_until_ready(value)
        elapsed = perf_counter() - started
        if first_run_ml_execution is None:
            first_run_ml_execution = elapsed
        return value

    optimization_start = perf_counter()
    result = run_ml(
        dim=problem.dim,
        data=problem.data,
        loglik=problem.loglik,
        logprior=problem.logprior,
        theta0=problem.theta0,
        config=MLConfig(
            method="L-BFGS-B",
            maxiter=args.max_iterations,
            gtol=1e-5,
            prior_weight=1.0,
            compute_hessian=False,
            log_every=1,
        ),
        compiled_objective=instrumented,
    )
    optimization_seconds = perf_counter() - optimization_start
    cache_after = _files(args.compilation_cache)
    matrix = operator.matrix
    report = {
        "versions": {"jax": jax.__version__, "jaxlib": jaxlib.__version__},
        "platform": {
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "system": platform.platform(),
            "architecture": platform.machine(),
            "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        },
        "problem": {
            "parameter_shape": list(problem.theta0.shape),
            "parameter_dtype": str(problem.theta0.dtype),
            "operator_shape": list(matrix.shape),
            "operator_dtype": str(matrix.dtype),
            "bcoo_index_shape": list(matrix.indices.shape),
            "nonzero_count": int(matrix.nse),
        },
        "preparation": {
            "wall_seconds": preparation_wall,
            "stages": prepared_map.preparation_seconds,
            "operator_cache_load_seconds": operator.metrics.cache_load_seconds,
            "operator_cache_hit": operator.metrics.cache_hit,
            "assignment_cache": dataclasses.asdict(
                problem.assignment_cache_metrics
            ),
        },
        "compilation": dataclasses.asdict(compiled.metrics),
        "warm_execution_seconds": warm,
        "warm_execution_median_seconds": float(np.median(warm)),
        "run_ml": {
            "first_evaluation_seconds": first_run_ml_execution,
            "optimization_seconds": optimization_seconds,
            "iterations": result.num_iterations,
            "objective_evaluations": result.num_compiled_evaluations,
            "function_evaluations": result.num_function_evaluations,
            "gradient_evaluations": result.num_gradient_evaluations,
            "objective_final": result.objective_value,
            "gradient_norm_final": result.gradient_norm,
        },
        "persistent_compilation_cache": {
            "enabled": args.compilation_cache is not None,
            "directory": None
            if args.compilation_cache is None
            else str(args.compilation_cache.resolve()),
            "files_before": cache_before,
            "files_after": cache_after,
            "new_files": sorted(set(cache_after) - set(cache_before)),
            "inferred_hit": bool(cache_before) and not bool(set(cache_after) - set(cache_before)),
        },
        "total_process_seconds": perf_counter() - process_start,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
