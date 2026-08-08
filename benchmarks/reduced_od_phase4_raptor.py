"""CPU scaling benchmark for Phase-4 bounded timetable queries."""

from __future__ import annotations

import argparse
import json
import resource
import time

from public_transportation.domain import (
    Metadata,
    ODDemand,
    Scenario,
    Stop,
    StopTime,
    TimeBin,
    TimeOfDay,
    Timetable,
    Trip,
)
from public_transportation.domain.line import Line
from public_transportation.preprocessing.reduced_od import (
    RaptorQuery,
    prepare_reduced_od_timetable,
    run_raptor_query,
)


def _scenario(number_of_stops: int, departures_per_pattern: int) -> Scenario:
    stop_ids = tuple(f"S{index:04d}" for index in range(number_of_stops))
    stops = [Stop(stop_id, stop_id, 46.0 + index * 1e-4, 6.0) for index, stop_id in enumerate(stop_ids)]
    lines: list[Line] = []
    trips: list[Trip] = []
    stop_times: list[StopTime] = []
    pattern_length = min(20, number_of_stops)
    stride = max(1, pattern_length - 2)
    starts = list(range(0, max(1, number_of_stops - 1), stride))
    if starts[-1] + 1 >= number_of_stops:
        starts.pop()
    for pattern, start in enumerate(starts):
        end = min(number_of_stops, start + pattern_length)
        line_id = f"L{pattern:04d}"
        lines.append(Line(line_id))
        for departure_index in range(departures_per_pattern):
            trip_id = f"T{pattern:04d}_{departure_index:03d}"
            trips.append(Trip(trip_id, line_id, service_id="day", direction_id=0))
            first_departure = 6 * 3600 + departure_index * 600 + pattern * 120
            for sequence, stop_id in enumerate(stop_ids[start:end], start=1):
                seconds = first_departure + (sequence - 1) * 90
                stop_times.append(
                    StopTime(trip_id, stop_id, sequence, seconds, seconds)
                )
    return Scenario(
        metadata=Metadata(title="Phase 4 benchmark", created_at="2026-01-01T00:00:00"),
        stops=stops,
        lines=lines,
        time_bins=[TimeBin("day", TimeOfDay(0), TimeOfDay(30 * 3600))],
        demand=ODDemand(records=[]),
        timetable=Timetable(trips=trips, stop_times=stop_times),
    )


def _run(number_of_stops: int, departures_per_pattern: int, repeats: int) -> dict[str, int | float]:
    scenario = _scenario(number_of_stops, departures_per_pattern)
    build_start = time.perf_counter()
    timetable = prepare_reduced_od_timetable(
        scenario, configuration_fingerprint="phase-4-benchmark"
    )
    build_seconds = time.perf_counter() - build_start
    query = RaptorQuery(timetable.physical_stop_ids[0], 6 * 3600, 3)
    run_raptor_query(timetable, query)
    query_start = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = run_raptor_query(timetable, query)
    query_seconds = (time.perf_counter() - query_start) / repeats
    assert result is not None
    return {
        "stops": number_of_stops,
        "patterns": len(timetable.route_patterns.patterns),
        "trips": len(timetable.trip_ids),
        "stop_times": int(timetable.array("arrival_seconds").size),
        "build_seconds": build_seconds,
        "query_seconds": query_seconds,
        "queries_per_second": 1.0 / query_seconds,
        "timetable_bytes": timetable.retained_bytes,
        "candidate_labels": result.diagnostics.candidate_labels,
        "retained_labels": result.diagnostics.retained_labels,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stops", type=int, nargs="+", default=[25, 50, 100, 200])
    parser.add_argument("--departures-per-pattern", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
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
