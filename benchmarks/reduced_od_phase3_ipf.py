"""Route-length, iteration, time, and RSS benchmark for route-level IPF."""

from __future__ import annotations

import json
import resource
from time import perf_counter

import numpy as np

from public_transportation.inference.reduced_od import (
    RouteLevelCounts,
    RouteLevelIPFConfig,
    estimate_route_level_ipf,
)


def _problem(length: int) -> RouteLevelCounts:
    generator = np.random.default_rng(71 + length)
    truth = np.triu(generator.uniform(0.1, 2.0, size=(length, length)), k=1)
    return RouteLevelCounts(
        route_pattern_id=f"length_{length}",
        service_period_id="benchmark",
        stop_ids=tuple(f"S{index:04d}" for index in range(length)),
        boarding_counts=truth.sum(axis=1),
        alighting_counts=truth.sum(axis=0),
        boarding_observed=np.ones(length, dtype=bool),
        alighting_observed=np.ones(length, dtype=bool),
    )


def main() -> None:
    rows: list[dict[str, float | int]] = []
    for length in (10, 25, 50, 100):
        problem = _problem(length)
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        started = perf_counter()
        result = estimate_route_level_ipf(
            problem,
            config=RouteLevelIPFConfig(tolerance=1e-8),
        )
        elapsed = perf_counter() - started
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rows.append(
            {
                "iterations": result.diagnostics.iterations,
                "length": length,
                "matrix_bytes": result.leg_od_matrix.nbytes,
                "ru_maxrss_after": rss_after,
                "ru_maxrss_before": rss_before,
                "seconds": elapsed,
            }
        )
    print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__":
    main()
