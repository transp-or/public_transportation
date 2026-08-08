"""Build-time and retained-memory benchmark for the Phase-2 timetable index."""

from __future__ import annotations

import argparse
import json
import resource
from time import perf_counter

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
    prepare_reduced_od_timetable,
)


def _scenario(num_trips: int, stops_per_trip: int) -> Scenario:
    stops = [
        Stop(f"S{index:03d}", f"Stop {index}", 46.0 + index * 1e-4, 6.0)
        for index in range(stops_per_trip)
    ]
    trips = [
        Trip(
            f"T{index:06d}",
            "L1",
            service_id="weekday",
            direction_id=index % 2,
        )
        for index in range(num_trips)
    ]
    stop_times: list[StopTime] = []
    for trip_index, trip in enumerate(trips):
        order = (
            range(stops_per_trip)
            if trip.direction_id == 0
            else range(stops_per_trip - 1, -1, -1)
        )
        start = 5 * 3600 + trip_index * 30
        for sequence, stop_index in enumerate(order, start=1):
            arrival = start + (sequence - 1) * 90
            stop_times.append(
                StopTime(
                    trip.trip_id,
                    f"S{stop_index:03d}",
                    sequence,
                    arrival,
                    arrival + 10,
                )
            )
    return Scenario(
        metadata=Metadata(title="Phase 2 benchmark"),
        stops=stops,
        lines=[Line("L1")],
        time_bins=[
            TimeBin("all", TimeOfDay(0), TimeOfDay(48 * 3600))
        ],
        demand=ODDemand(records=[]),
        timetable=Timetable(trips=trips, stop_times=stop_times),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trips", type=int, default=5_000)
    parser.add_argument("--stops-per-trip", type=int, default=15)
    args = parser.parse_args()
    scenario = _scenario(args.trips, args.stops_per_trip)

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = perf_counter()
    index = prepare_reduced_od_timetable(
        scenario, configuration_fingerprint="benchmark", mapping_policy=None
    )
    elapsed = perf_counter() - started
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "build_seconds": elapsed,
                "num_patterns": len(index.route_patterns.patterns),
                "num_stop_times": index.array("arrival_seconds").size,
                "num_trips": len(index.trip_ids),
                "retained_array_bytes": index.retained_bytes,
                "ru_maxrss_after": rss_after,
                "ru_maxrss_before": rss_before,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
