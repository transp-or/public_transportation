"""Bounded compile and warm-evaluation benchmark for generic demand kernels."""

from __future__ import annotations

import resource
import sys
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .demand_model import DemandModelProblem, evaluate_demand_model


@dataclass(frozen=True, slots=True)
class DemandModelBenchmark:
    parameter_count: int
    compile_seconds: float
    warm_value_gradient_seconds: float
    objective: float
    gradient_norm: float
    peak_rss_bytes: int


def benchmark_demand_model(
    *, problem: DemandModelProblem, raw_parameters: object, warm_evaluations: int = 3
) -> DemandModelBenchmark:
    if warm_evaluations <= 0:
        raise ValueError("warm_evaluations must be positive.")
    raw = jnp.asarray(raw_parameters)
    compiled = jax.jit(
        jax.value_and_grad(
            lambda value: evaluate_demand_model(value, problem=problem).objective
        )
    )
    started = time.perf_counter()
    first = compiled(raw)
    jax.block_until_ready(first)
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    result = first
    for _ in range(warm_evaluations):
        result = compiled(raw)
        jax.block_until_ready(result)
    warm = (time.perf_counter() - started) / warm_evaluations
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        rss *= 1024
    return DemandModelBenchmark(
        raw.size,
        compile_seconds,
        warm,
        float(result[0]),
        float(np.linalg.norm(np.asarray(result[1]))),
        rss,
    )
