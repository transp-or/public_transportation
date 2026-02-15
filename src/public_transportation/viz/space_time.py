from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from public_transportation.domain import Scenario

from .utils import format_hhmm, stop_times_by_trip, trips_by_id, line_color_map, stable_line_styles


def plot_space_time_diagrams(
    scenario: Scenario,
    *,
    figsize: tuple[float, float],
    combine: bool = False,
) -> dict[str, Figure]:
    """
    Generate space–time diagrams.

    Output:
    - by default, one figure per (line_id, direction_id) with all trips on that line-direction.

    Space–time definition:
    - x-axis: stop sequence index (with stop labels)
    - y-axis: time of day (arrival times)
    - each trip is a polyline

    :param scenario: Scenario with timetable.
    :param figsize: Figure size.
    :param combine: If True, returns a single figure under key "space_time".
    :return: Mapping from keys like 'space_time_L1_dir0' to figures.
    """
    if scenario.timetable is None:
        return {}

    trip_map = trips_by_id(scenario)
    st_by_trip = stop_times_by_trip(scenario)

    # Group trip_ids by (line_id, direction_id)
    groups: dict[tuple[str | None, int | None], list[str]] = defaultdict(list)
    for tid in st_by_trip.keys():
        trip = trip_map.get(tid)
        if trip is None:
            continue
        groups[(trip.line_id, trip.direction_id)].append(tid)

    colors = line_color_map([t.line_id for t in scenario.timetable.trips])
    styles = stable_line_styles()

    figures: dict[str, Figure] = {}

    # Helper to make one figure for a group
    def make_group_fig(line_id: str | None, direction_id: int | None, tids: list[str]) -> Figure:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        # Choose a reference stop label sequence from the earliest trip in this group
        tids_sorted = sorted(
            tids,
            key=lambda tid_: int(st_by_trip[tid_][0].departure.seconds_from_midnight) if st_by_trip[tid_] else 10**9,
        )
        ref_tid = tids_sorted[0]
        ref_stops = [st.stop_id for st in st_by_trip[ref_tid]]
        x = list(range(len(ref_stops)))
        ax.set_xticks(x)
        ax.set_xticklabels(ref_stops, rotation=0)

        color = colors.get(line_id, "#0072B2")
        ls, mk = styles[0 if direction_id is None else int(direction_id) % len(styles)]

        # Plot each trip
        for tid_ in tids_sorted:
            sts = st_by_trip[tid_]
            stop_ids = [st.stop_id for st in sts]
            times = [int(st.arrival.seconds_from_midnight) for st in sts]

            # Align to reference by index only; (future: align by stop_id)
            xx = list(range(len(stop_ids)))
            ax.plot(xx, times, linestyle=ls, marker=None, linewidth=1.3, alpha=0.35, color=color)

        # Emphasize the first (earliest) trip
        sts0 = st_by_trip[ref_tid]
        ax.plot(
            list(range(len(sts0))),
            [int(st.arrival.seconds_from_midnight) for st in sts0],
            linestyle=ls,
            marker=mk,
            linewidth=2.8,
            alpha=0.9,
            color=color,
        )

        # Y-axis formatting: show HH:MM on ticks (use a formatter to avoid warnings)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: format_hhmm(int(v))))

        lid = line_id or "Line?"
        did = direction_id if direction_id is not None else "?"
        ax.set_title(f"Space–time diagram: {lid} dir {did}")
        ax.set_xlabel("Stop sequence")
        ax.set_ylabel("Time (HH:MM)")

        fig.tight_layout()
        return fig

    if combine:
        # Simple combined approach: stack all groups sequentially in one figure later.
        # For now, return separate figures only (combine will be implemented when needed).
        pass

    for (lid, did), tids in sorted(groups.items(), key=lambda kv: ("" if kv[0][0] is None else kv[0][0], -1 if kv[0][1] is None else int(kv[0][1]))):
        fig = make_group_fig(lid, did, tids)
        key = f"space_time_{(lid or 'Line')}_dir{did if did is not None else 'X'}"
        figures[key] = fig

    return figures