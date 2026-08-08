"""CPU and cache benchmark for Phase-6 direct measurement responses."""

from __future__ import annotations

import argparse
import json
import resource
import tempfile
import time
from pathlib import Path

from public_transportation.measurement import (
    MeasurementRecord,
    MeasurementTable,
    MeasurementType,
)
from public_transportation.preprocessing.reduced_od import (
    JourneyChoicePolicy,
    RaptorQuery,
    build_journey_choices,
    build_measurement_response,
    prepare_reduced_od_timetable,
    run_raptor_query,
    save_measurement_response_cache,
)
from reduced_od_phase4_raptor import _scenario


def _measurements(scenario) -> MeasurementTable:
    assert scenario.timetable is not None
    records: list[MeasurementRecord] = []
    for stop_time in sorted(
        scenario.timetable.stop_times,
        key=lambda item: (str(item.trip_id), int(item.sequence)),
    ):
        for measurement_type, event_time in (
            (MeasurementType.ALIGHTING, stop_time.arrival),
            (MeasurementType.BOARDING, stop_time.departure),
        ):
            records.append(
                MeasurementRecord(
                    method_id="synthetic_apc",
                    measurement_type=measurement_type,
                    stop_id=str(stop_time.stop_id),
                    time=event_time,
                    value=1.0,
                    trip_id=str(stop_time.trip_id),
                )
            )
    return MeasurementTable.from_records(records)


def _run(number_of_stops: int, departures_per_pattern: int, repeats: int) -> dict[str, int | float]:
    scenario = _scenario(number_of_stops, departures_per_pattern)
    timetable = prepare_reduced_od_timetable(
        scenario, configuration_fingerprint="phase-6-benchmark"
    )
    raptor = run_raptor_query(
        timetable,
        RaptorQuery(timetable.physical_stop_ids[0], 6 * 3600, 3),
    )
    choices = build_journey_choices(
        timetable,
        raptor,
        policy=JourneyChoicePolicy(maximum_alternatives_per_cell=4),
    )
    measurements = _measurements(scenario)
    build_measurement_response(
        timetable=timetable,
        journey_choices=choices,
        measurements=measurements,
        configuration_fingerprint="phase-6-benchmark",
    )
    started = time.perf_counter()
    artifact = None
    for _ in range(repeats):
        artifact = build_measurement_response(
            timetable=timetable,
            journey_choices=choices,
            measurements=measurements,
            configuration_fingerprint="phase-6-benchmark",
        )
    build_seconds = (time.perf_counter() - started) / repeats
    assert artifact is not None
    with tempfile.TemporaryDirectory(prefix="reduced-od-phase6-") as directory:
        cache_path = Path(directory) / "response.npz"
        save_measurement_response_cache(cache_path, artifact)
        cache_bytes = cache_path.stat().st_size
    return {
        "stops": number_of_stops,
        "measurements": artifact.number_of_measurements,
        "free_cells": artifact.number_of_free_cells,
        "nnz": artifact.nnz,
        "build_seconds": build_seconds,
        "measurements_per_second": artifact.number_of_measurements / build_seconds,
        "retained_array_bytes": artifact.retained_bytes,
        "cache_bytes": cache_bytes,
        "equivalence_classes": artifact.equivalence.number_of_classes,
        "compression_ratio": artifact.equivalence.compression_ratio,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stops", type=int, nargs="+", default=[25, 50, 100, 200])
    parser.add_argument("--departures-per-pattern", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
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
