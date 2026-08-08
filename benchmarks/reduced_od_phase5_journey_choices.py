"""CPU benchmark for bounded Phase-5 journey-choice construction."""

from __future__ import annotations

import argparse
import json
import resource
import time

from public_transportation.preprocessing.reduced_od import (
    JourneyChoicePolicy,
    RaptorQuery,
    build_journey_choices,
    prepare_reduced_od_timetable,
    run_raptor_query,
)
from reduced_od_phase4_raptor import _scenario


def _run(
    number_of_stops: int,
    departures_per_pattern: int,
    repeats: int,
) -> dict[str, int | float]:
    timetable = prepare_reduced_od_timetable(
        _scenario(number_of_stops, departures_per_pattern),
        configuration_fingerprint="phase-5-benchmark",
    )
    raptor = run_raptor_query(
        timetable,
        RaptorQuery(timetable.physical_stop_ids[0], 6 * 3600, 3),
    )
    policy = JourneyChoicePolicy(maximum_alternatives_per_cell=4)
    build_journey_choices(timetable, raptor, policy=policy)
    started = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = build_journey_choices(timetable, raptor, policy=policy)
    elapsed = (time.perf_counter() - started) / repeats
    assert result is not None
    diagnostics = result.diagnostics
    return {
        "stops": number_of_stops,
        "feasible_destinations": diagnostics.feasible_destinations,
        "choice_cells": diagnostics.choice_cells,
        "candidate_alternatives": diagnostics.candidate_alternatives,
        "retained_alternatives": diagnostics.retained_alternatives,
        "pruned_alternatives": diagnostics.pruned_alternatives,
        "alternatives_per_cell": (
            diagnostics.retained_alternatives / diagnostics.choice_cells
            if diagnostics.choice_cells
            else 0.0
        ),
        "build_seconds": elapsed,
        "cells_per_second": diagnostics.choice_cells / elapsed,
        "estimated_payload_bytes": diagnostics.estimated_payload_bytes,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stops", type=int, nargs="+", default=[25, 50, 100, 200])
    parser.add_argument("--departures-per-pattern", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    arguments = parser.parse_args()
    if any(value < 2 for value in arguments.stops):
        parser.error("all --stops values must be at least 2")
    if arguments.departures_per_pattern < 1 or arguments.repeats < 1:
        parser.error("departures and repeats must be positive")
    print(
        json.dumps(
            [
                _run(value, arguments.departures_per_pattern, arguments.repeats)
                for value in arguments.stops
            ],
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
