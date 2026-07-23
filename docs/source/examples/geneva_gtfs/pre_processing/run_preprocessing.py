"""Assign the synthetic true Geneva OD matrix and generate complete stop counts."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from public_transportation.assignment.assign import assign, prepare_assignment
from public_transportation.assignment.config import AssignmentConfig
from public_transportation.assignment.graph_sentinels import LINK_TYPE_ACCESS, LINK_TYPE_EGRESS
from public_transportation.domain import Scenario, TimeOfDay
from public_transportation.measurement.io import write_measurements_csv
from public_transportation.measurement.schema import MeasurementRecord, MeasurementTable, MeasurementType


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = Path(__file__).resolve().parent / "results"
NETWORK_FILES = ("metadata.json", "stops.csv", "lines.csv", "trips.csv", "stop_times.csv", "time_bins.csv")
TRUE_THETA = 5.0


def _scenario_with_demand(demand_path: Path) -> tempfile.TemporaryDirectory[str]:
    temporary = tempfile.TemporaryDirectory()
    target = Path(temporary.name)
    for name in NETWORK_FILES:
        shutil.copy2(DATA / name, target / name)
    shutil.copy2(demand_path, target / "demand.csv")
    return temporary


def _measurement_records(*, scenario: Scenario, artifacts, link_flow: np.ndarray) -> list[MeasurementRecord]:
    """Return boarding and alighting counts at every retained stop event.

    Zero observations are retained deliberately: “all stop counts” means the
    complete observation design, not only locations carrying positive demand.
    """
    graph = artifacts.graph
    link_type = np.asarray(graph.link_type)
    tail = np.asarray(graph.tail)
    head = np.asarray(graph.head)
    node_stop_index = np.asarray(graph.node_stop_index)
    node_trip_index = np.asarray(graph.node_trip_index)
    node_time_s = np.asarray(graph.node_time_s)
    stop_ids = list(graph.node_stop_id)
    trip_ids = list(graph.trip_id)

    values: dict[tuple[str, str, str, int], float] = {}

    for link_id in np.flatnonzero(link_type == LINK_TYPE_ACCESS):
        node = int(head[link_id])
        trip_index = int(node_trip_index[node])
        stop_index = int(node_stop_index[node])
        if trip_index >= 0 and stop_index >= 0:
            key = ("boarding", str(stop_ids[stop_index]), str(trip_ids[trip_index]), int(node_time_s[node]))
            values[key] = values.get(key, 0.0) + float(link_flow[link_id])

    for link_id in np.flatnonzero(link_type == LINK_TYPE_EGRESS):
        node = int(tail[link_id])
        trip_index = int(node_trip_index[node])
        stop_index = int(node_stop_index[node])
        if trip_index >= 0 and stop_index >= 0:
            key = ("alighting", str(stop_ids[stop_index]), str(trip_ids[trip_index]), int(node_time_s[node]))
            values[key] = values.get(key, 0.0) + float(link_flow[link_id])

    return [
        MeasurementRecord(
            method_id="synthetic_all_stop_counts",
            measurement_type=MeasurementType(measurement_type),
            stop_id=stop_id,
            time=TimeOfDay(seconds_from_midnight=time_s),
            # The negative-binomial observation model has count support.  Use
            # deterministic nearest-integer synthetic APC counts so repeated
            # preprocessing remains reproducible.
            value=float(round(value)),
            trip_id=trip_id,
            line_id=None,
        )
        for (measurement_type, stop_id, trip_id, time_s), value in sorted(values.items())
    ]


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with _scenario_with_demand(DATA / "true_demand.csv") as scenario_dir:
        scenario = Scenario.from_folder(scenario_dir, strict=True)
        od_values = jnp.asarray([record.flow for record in scenario.demand.records], dtype=jnp.float32)
        config = AssignmentConfig()
        artifacts = prepare_assignment(scenario=scenario, config=config)
        assignment = assign(od_values=od_values, artifacts=artifacts, theta=TRUE_THETA)
        link_flow = np.asarray(assignment.link_flow, dtype=float)

        records = _measurement_records(scenario=scenario, artifacts=artifacts, link_flow=link_flow)
        write_measurements_csv(
            MeasurementTable.from_records(records),
            RESULTS / "measurements_boarding_alighting.csv",
        )
        shutil.copy2(DATA / "prior_demand.csv", RESULTS / "demand.csv")
        np.savez_compressed(
            RESULTS / "true_assignment.npz",
            theta=TRUE_THETA,
            od_values=np.asarray(od_values, dtype=float),
            link_flow=link_flow,
            link_cost=np.asarray(assignment.link_cost, dtype=float),
        )

        positive = sum(record.value > 1.0e-8 for record in records)
        summary = {
            "num_stops": len(scenario.stops),
            "num_trips": len(scenario.timetable.trips),
            "num_stop_times": len(scenario.timetable.stop_times),
            "num_od_cells": len(scenario.demand.records),
            "num_measurements": len(records),
            "num_positive_measurements": positive,
            "num_zero_measurements": len(records) - positive,
            "true_theta": TRUE_THETA,
            "true_total_demand": float(np.asarray(od_values).sum()),
        }
        (RESULTS / "preprocessing_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
