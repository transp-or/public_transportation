# tests/helpers/build_scenarios.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# NOTE:
# We intentionally build domain objects directly rather than writing files.
# These builders are used for unit tests where we want tiny scenarios with
# fully known expected behavior.

# The exact class names in your domain package may evolve. These builders
# reflect the current design we have been using (Scenario, Stop, TimeBins,
# Demand, Timetable, Trip, StopTime, Metadata). If you rename/move classes,
# update imports here once, and all tests stay consistent.


@dataclass(frozen=True)
class TinyScenarioSpec:
    """
    High-level specification for tiny synthetic scenarios.

    :param title: Scenario title.
    :param description: Scenario description.
    """
    title: str = "Tiny scenario"
    description: str = "Programmatically generated tiny scenario for tests."


def build_tiny_line_scenario(*, spec: TinyScenarioSpec | None = None):
    """
    Build a minimal but non-trivial Scenario:

    - 3 stops: A, B, C
    - One line (two-direction possible in later builders), one trip A->B->C
    - Time bins (two bins)
    - Demand: A->C for bin 0

    This scenario is suitable for unit tests of:
    - time bin handling,
    - timetable consistency,
    - graph building (time-expanded),
    - cost computation (ride + access),
    - Dial DP behavior on a simple DAG.

    :param spec: Optional scenario metadata spec.
    :return: Scenario instance.
    """
    if spec is None:
        spec = TinyScenarioSpec(title="Tiny line scenario", description="A->B->C single trip")

    from public_transportation.domain import (
        Scenario,
        ScenarioMetadata,
        Stop,
        TimeBins,
        TimeBin,
        Demand,
        DemandRecord,
        Timetable,
        Trip,
        StopTime,
    )

    # Stops (geocoded, arbitrary coords)
    stops = {
        "A": Stop(stop_id="A", name="Stop A", lat=46.0, lon=6.0),
        "B": Stop(stop_id="B", name="Stop B", lat=46.001, lon=6.001),
        "C": Stop(stop_id="C", name="Stop C", lat=46.002, lon=6.002),
    }

    # Time bins in seconds-from-midnight: [08:00-08:10), [08:10-08:20)
    # (If your TimeBin expects minutes, adjust accordingly.)
    tb0 = TimeBin(time_bin_id="t0", start_s=8 * 3600, end_s=8 * 3600 + 10 * 60)
    tb1 = TimeBin(time_bin_id="t1", start_s=8 * 3600 + 10 * 60, end_s=8 * 3600 + 20 * 60)
    time_bins = TimeBins(bins=[tb0, tb1])

    # One trip A->B->C with increasing event times
    trip = Trip(
        trip_id="T1",
        route_id="R1",
        direction_id=0,
        headsign="C",
        capacity=50,
    )
    stop_times = [
        StopTime(trip_id="T1", stop_id="A", stop_sequence=1, dep_s=8 * 3600 + 2 * 60, arr_s=8 * 3600 + 2 * 60),
        StopTime(trip_id="T1", stop_id="B", stop_sequence=2, dep_s=8 * 3600 + 7 * 60, arr_s=8 * 3600 + 7 * 60),
        StopTime(trip_id="T1", stop_id="C", stop_sequence=3, dep_s=8 * 3600 + 12 * 60, arr_s=8 * 3600 + 12 * 60),
    ]
    timetable = Timetable(trips=[trip], stop_times=stop_times)

    # Demand: 10 pax from A to C in bin t0
    demand = Demand(
        records=[
            DemandRecord(
                origin_stop_id="A",
                destination_stop_id="C",
                time_bin_id="t0",
                flow=10.0,
            )
        ]
    )

    metadata = ScenarioMetadata(title=spec.title, description=spec.description)

    scenario = Scenario(
        metadata=metadata,
        stops=stops,
        time_bins=time_bins,
        demand=demand,
        timetable=timetable,
    )
    return scenario


def build_tiny_transfer_scenario(*, spec: TinyScenarioSpec | None = None):
    """
    Build a tiny scenario with a transfer opportunity:

    - Stops: A, X, B (transfer at X)
    - Two lines:
        L1: A -> X
        L2: X -> B
    - Timetable arranged so a transfer is possible
    - Demand: A -> B

    This is useful to test:
    - transfer arcs generation,
    - transfer cost coefficient effect,
    - Dial DP splits across alternatives when multiple departures exist.

    :param spec: Optional scenario metadata spec.
    :return: Scenario instance.
    """
    if spec is None:
        spec = TinyScenarioSpec(title="Tiny transfer scenario", description="A->X transfer ->B")

    from public_transportation.domain import (
        Scenario,
        ScenarioMetadata,
        Stop,
        TimeBins,
        TimeBin,
        Demand,
        DemandRecord,
        Timetable,
        Trip,
        StopTime,
    )

    stops = {
        "A": Stop(stop_id="A", name="Stop A", lat=46.0, lon=6.0),
        "X": Stop(stop_id="X", name="Stop X", lat=46.001, lon=6.001),
        "B": Stop(stop_id="B", name="Stop B", lat=46.002, lon=6.002),
    }

    tb0 = TimeBin(time_bin_id="t0", start_s=8 * 3600, end_s=8 * 3600 + 15 * 60)
    time_bins = TimeBins(bins=[tb0])

    # L1 trip A->X
    trip1 = Trip(trip_id="T1", route_id="L1", direction_id=0, headsign="X", capacity=40)
    st1 = [
        StopTime(trip_id="T1", stop_id="A", stop_sequence=1, dep_s=8 * 3600 + 1 * 60, arr_s=8 * 3600 + 1 * 60),
        StopTime(trip_id="T1", stop_id="X", stop_sequence=2, dep_s=8 * 3600 + 6 * 60, arr_s=8 * 3600 + 6 * 60),
    ]

    # L2 trip X->B departing after a feasible transfer
    trip2 = Trip(trip_id="T2", route_id="L2", direction_id=0, headsign="B", capacity=40)
    st2 = [
        StopTime(trip_id="T2", stop_id="X", stop_sequence=1, dep_s=8 * 3600 + 8 * 60, arr_s=8 * 3600 + 8 * 60),
        StopTime(trip_id="T2", stop_id="B", stop_sequence=2, dep_s=8 * 3600 + 13 * 60, arr_s=8 * 3600 + 13 * 60),
    ]

    timetable = Timetable(trips=[trip1, trip2], stop_times=[*st1, *st2])

    demand = Demand(
        records=[
            DemandRecord(
                origin_stop_id="A",
                destination_stop_id="B",
                time_bin_id="t0",
                flow=12.0,
            )
        ]
    )

    metadata = ScenarioMetadata(title=spec.title, description=spec.description)

    scenario = Scenario(
        metadata=metadata,
        stops=stops,
        time_bins=time_bins,
        demand=demand,
        timetable=timetable,
    )
    return scenario