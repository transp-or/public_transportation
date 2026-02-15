from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Literal

from public_transportation.domain import Scenario, Stop, StopTime, Trip


# Okabe–Ito color-blind safe palette (8 colors).
# https://jfly.uni-koeln.de/color/ (common reference)
OKABE_ITO = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]


@dataclass(frozen=True)
class TripPolyline:
    """
    Preprocessed trip geometry and times for plotting.

    :param trip: Trip object.
    :param stop_ids: Ordered stop sequence for this trip.
    :param lats: Latitudes aligned with stop_ids.
    :param lons: Longitudes aligned with stop_ids.
    :param arrival_s: Arrival times in seconds from midnight.
    :param departure_s: Departure times in seconds from midnight.
    """
    trip: Trip
    stop_ids: list[str]
    lats: list[float]
    lons: list[float]
    arrival_s: list[int]
    departure_s: list[int]


def format_hhmm(seconds_from_midnight: int) -> str:
    """
    Format seconds from midnight as HH:MM.

    :param seconds_from_midnight: Seconds from midnight.
    :return: Formatted string "HH:MM".
    """
    s = int(seconds_from_midnight)
    if s < 0:
        s = 0
    hh = (s // 3600) % 24
    mm = (s % 3600) // 60
    return f"{hh:02d}:{mm:02d}"


def infer_hub_stop_id(scenario: Scenario, preferred: str | None = None) -> str:
    """
    Choose a stop_id for displaying timetable panel by default.

    Rule:
    - if preferred provided and exists -> use it
    - else if 'C' exists -> use it
    - else first stop in scenario.stops

    :param scenario: Scenario.
    :param preferred: Optional preferred hub stop_id.
    :return: Selected stop_id.
    """
    stop_ids = [s.stop_id for s in scenario.stops]
    if preferred is not None and preferred in stop_ids:
        return preferred
    if "C" in stop_ids:
        return "C"
    return stop_ids[0]


def stops_by_id(scenario: Scenario) -> dict[str, Stop]:
    """
    Build a dictionary stop_id -> Stop.

    :param scenario: Scenario.
    :return: Dictionary mapping stop_id to Stop.
    """
    return {s.stop_id: s for s in scenario.stops}


def trips_by_id(scenario: Scenario) -> dict[str, Trip]:
    """
    Build a dictionary trip_id -> Trip.

    :param scenario: Scenario.
    :return: Dictionary mapping trip_id to Trip.
    """
    if scenario.timetable is None:
        return {}
    return {t.trip_id: t for t in scenario.timetable.trips}


def stop_times_by_trip(scenario: Scenario) -> dict[str, list[StopTime]]:
    """
    Group stop times by trip_id and sort each list by sequence.

    :param scenario: Scenario.
    :return: Dictionary trip_id -> sorted list of StopTime.
    """
    if scenario.timetable is None:
        return {}
    by_trip: dict[str, list[StopTime]] = {}
    for st in scenario.timetable.stop_times:
        by_trip.setdefault(st.trip_id, []).append(st)
    for tid in list(by_trip.keys()):
        by_trip[tid] = sorted(by_trip[tid], key=lambda x: x.sequence)
    return by_trip


def build_trip_polylines(scenario: Scenario) -> dict[str, TripPolyline]:
    """
    Build polylines for all trips based on stop lat/lon and stop_times.

    Any stop_times referencing unknown stop_id are skipped for plotting
    (validation should have already caught these).

    :param scenario: Scenario.
    :return: Mapping trip_id -> TripPolyline.
    """
    if scenario.timetable is None:
        return {}

    sb = stops_by_id(scenario)
    tb = trips_by_id(scenario)
    st_by_trip = stop_times_by_trip(scenario)

    polylines: dict[str, TripPolyline] = {}
    for tid, sts in st_by_trip.items():
        trip = tb.get(tid)
        if trip is None:
            continue

        stop_ids: list[str] = []
        lats: list[float] = []
        lons: list[float] = []
        arr: list[int] = []
        dep: list[int] = []

        for st in sts:
            s = sb.get(st.stop_id)
            if s is None:
                continue
            stop_ids.append(st.stop_id)
            lats.append(float(s.lat))
            lons.append(float(s.lon))
            arr.append(int(st.arrival.seconds_from_midnight))
            dep.append(int(st.departure.seconds_from_midnight))

        if len(stop_ids) >= 2:
            polylines[tid] = TripPolyline(
                trip=trip,
                stop_ids=stop_ids,
                lats=lats,
                lons=lons,
                arrival_s=arr,
                departure_s=dep,
            )

    return polylines


def select_representative_trips(
    scenario: Scenario,
    *,
    rule: Literal["earliest", "latest"] = "earliest",
) -> dict[tuple[str | None, int | None], str]:
    """
    Select one representative trip_id per (line_id, direction_id).

    The selection is based on the first departure time of the trip.

    :param scenario: Scenario with timetable.
    :param rule: Selection rule ("earliest" or "latest").
    :return: Mapping from (line_id, direction_id) to trip_id.
    """
    if scenario.timetable is None:
        return {}

    st_by_trip = stop_times_by_trip(scenario)
    trip_by_id = trips_by_id(scenario)

    best: dict[tuple[str | None, int | None], tuple[int, str]] = {}

    for tid, sts in st_by_trip.items():
        trip = trip_by_id.get(tid)
        if trip is None or not sts:
            continue
        first_dep = int(sts[0].departure.seconds_from_midnight)

        key = (trip.line_id, trip.direction_id)
        if key not in best:
            best[key] = (first_dep, tid)
        else:
            cur_time, cur_tid = best[key]
            if rule == "earliest":
                if first_dep < cur_time:
                    best[key] = (first_dep, tid)
            else:
                if first_dep > cur_time:
                    best[key] = (first_dep, tid)

    return {k: v[1] for k, v in best.items()}


def stable_line_styles() -> list[tuple[str, str]]:
    """
    Provide a set of (linestyle, marker) pairs to add redundancy beyond colors.

    :return: List of (linestyle, marker).
    """
    return [
        ("-", "o"),
        ("--", "s"),
        ("-.", "D"),
        (":", "^"),
    ]


def line_color_map(line_ids: Iterable[str | None]) -> dict[str | None, str]:
    """
    Assign a color-blind safe color to each line_id.

    :param line_ids: Iterable of line ids (may include None).
    :return: Mapping line_id -> hex color.
    """
    unique = []
    for lid in line_ids:
        if lid not in unique:
            unique.append(lid)

    cmap: dict[str | None, str] = {}
    # Skip black for lines by default (keep for text/axes), start at orange
    palette = OKABE_ITO[1:]
    for i, lid in enumerate(unique):
        cmap[lid] = palette[i % len(palette)]
    return cmap


def demand_width(flow: float, *, min_w: float = 0.8, max_w: float = 4.0) -> float:
    """
    Convert demand flow to a reasonable line width for plotting.

    Uses a sqrt scaling to avoid domination by large flows.

    :param flow: Demand flow.
    :param min_w: Minimum width.
    :param max_w: Maximum width.
    :return: Line width.
    """
    f = max(0.0, float(flow))
    w = min_w + 0.15 * sqrt(f)
    return max(min_w, min(max_w, w))