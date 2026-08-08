"""Admission timing for one compact reduced-OD objective and gradient."""

from __future__ import annotations

import math
import resource
import time
from dataclasses import asdict, dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .objective import MinimalGravityProblem, evaluate_minimal_gravity_objective


@dataclass(frozen=True, slots=True)
class ReducedODObjectiveBenchmark:
    trace_seconds: float
    lowering_seconds: float
    compilation_seconds: float
    first_execution_seconds: float
    warm_seconds: tuple[float, ...]
    objective: float
    gradient_norm: float
    finite: bool
    cache_entries_before_value_change: int | None
    cache_entries_after_value_change: int | None
    recompiled_after_value_change: bool | None
    rss_before_bytes: int
    rss_after_bytes: int
    peak_rss_bytes: int

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable admission diagnostics."""
        return asdict(self)


def _rss_bytes() -> int:
    # macOS reports bytes and Linux reports KiB.
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if value > 10_000_000 else value * 1024


def benchmark_minimal_gravity_objective(
    *,
    problem: MinimalGravityProblem,
    raw_parameters: object,
    warm_evaluations: int = 5,
    clock: Callable[[], float] = time.perf_counter,
) -> ReducedODObjectiveBenchmark:
    """Trace, compile and time J0 without running an optimizer.

    Changing parameter *values* while retaining shape and dtype is explicitly
    checked against JAX's executable cache when that counter is available.
    """
    if warm_evaluations <= 0:
        raise ValueError("warm_evaluations must be positive.")
    raw = jnp.asarray(raw_parameters)
    if raw.shape != (problem.parameter_layout.size,):
        raise ValueError("raw_parameters has an incompatible dimension.")

    def objective(value: jax.Array) -> jax.Array:
        return evaluate_minimal_gravity_objective(value, problem=problem).objective

    value_and_gradient = jax.value_and_grad(objective)
    rss_before = _rss_bytes()
    started = clock()
    jax.make_jaxpr(value_and_gradient)(raw)
    trace_seconds = clock() - started

    jitted = jax.jit(value_and_gradient)
    started = clock()
    lowered = jitted.lower(raw)
    lowering_seconds = clock() - started
    started = clock()
    executable = lowered.compile()
    compilation_seconds = clock() - started

    started = clock()
    value, gradient = executable(raw)
    jax.block_until_ready((value, gradient))
    first_execution_seconds = clock() - started

    warm: list[float] = []
    for index in range(warm_evaluations):
        candidate = raw + jnp.asarray((index + 1) * 1.0e-6, dtype=raw.dtype)
        started = clock()
        value, gradient = jitted(candidate)
        jax.block_until_ready((value, gradient))
        warm.append(clock() - started)

    cache_size = getattr(jitted, "_cache_size", None)
    before = int(cache_size()) if cache_size is not None else None
    changed = raw + jnp.asarray(0.125, dtype=raw.dtype)
    value, gradient = jitted(changed)
    jax.block_until_ready((value, gradient))
    after = int(cache_size()) if cache_size is not None else None
    objective_value = float(value)
    gradient_norm = float(np.linalg.norm(np.asarray(gradient, dtype=np.float64)))
    finite = math.isfinite(objective_value) and math.isfinite(gradient_norm)
    rss_after = _rss_bytes()
    return ReducedODObjectiveBenchmark(
        trace_seconds=trace_seconds,
        lowering_seconds=lowering_seconds,
        compilation_seconds=compilation_seconds,
        first_execution_seconds=first_execution_seconds,
        warm_seconds=tuple(warm),
        objective=objective_value,
        gradient_norm=gradient_norm,
        finite=finite,
        cache_entries_before_value_change=before,
        cache_entries_after_value_change=after,
        recompiled_after_value_change=(None if before is None else before != after),
        rss_before_bytes=rss_before,
        rss_after_bytes=rss_after,
        peak_rss_bytes=max(rss_before, rss_after),
    )
