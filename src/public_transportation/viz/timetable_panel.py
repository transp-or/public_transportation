from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from public_transportation.domain import Scenario

from .utils import format_hhmm, infer_hub_stop_id, stop_times_by_trip, trips_by_id


def plot_timetable_panel(
    scenario: Scenario,
    *,
    stop_id: str | None,
    figsize: tuple[float, float],
) -> Figure:
    """
    Plot a timetable panel for a selected stop.

    The panel is text-based (readable, robust, test-friendly):
    - grouped by line_id and direction/headsign
    - lists departure times at the selected stop

    :param scenario: Scenario with timetable.
    :param stop_id: Stop to display. If None, use 'C' if present else first stop.
    :param figsize: Figure size.
    :return: Matplotlib Figure.
    """
    sid = infer_hub_stop_id(scenario, stop_id)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")

    if scenario.timetable is None:
        ax.text(0.02, 0.98, "No timetable in scenario.", va="top", fontsize=12)
        fig.tight_layout()
        return fig

    trip_map = trips_by_id(scenario)
    st_by_trip = stop_times_by_trip(scenario)

    # Collect departures at the stop, grouped by (line_id, direction_id, headsign)
    groups: dict[tuple[str | None, int | None, str | None], list[int]] = defaultdict(list)

    for tid, sts in st_by_trip.items():
        trip = trip_map.get(tid)
        if trip is None:
            continue
        for st in sts:
            if st.stop_id == sid:
                groups[(trip.line_id, trip.direction_id, trip.headsign)].append(int(st.departure.seconds_from_midnight))
                break

    if not groups:
        ax.text(0.02, 0.98, f"No departures found at stop {sid}.", va="top", fontsize=12)
        fig.tight_layout()
        return fig

    # Sort groups for stable display
    def key_sort(k: tuple[str | None, int | None, str | None]) -> tuple[str, int, str]:
        lid, did, hs = k
        return ("" if lid is None else lid, -1 if did is None else int(did), "" if hs is None else hs)

    lines = []
    lines.append(f"Timetable at stop {sid}")
    lines.append("-" * 60)

    for gk in sorted(groups.keys(), key=key_sort):
        lid, did, hs = gk
        times = sorted(groups[gk])
        times_str = ", ".join(format_hhmm(t) for t in times)

        dir_str = f"dir {did}" if did is not None else "dir ?"
        hs_str = f"→ {hs}" if hs else ""
        header = f"{lid or 'Line ?'} ({dir_str}) {hs_str}".strip()
        lines.append(header)
        lines.append(f"  {times_str}")
        lines.append("")

    text = "\n".join(lines)

    ax.text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
    )

    fig.tight_layout()
    return fig