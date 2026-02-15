from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from public_transportation.domain import Scenario

from .utils import (
    build_trip_polylines,
    line_color_map,
    select_representative_trips,
    stable_line_styles,
)


def plot_network_map(
    scenario: Scenario,
    *,
    show_all_trips: bool = False,
    representative: Literal["earliest", "latest"] = "earliest",
    figsize: tuple[float, float] = (12.0, 7.0),
    show_stop_labels: bool = True,
    stop_marker_size: float = 70.0,
    stop_halo_width: float = 2.5,
) -> Figure:
    """Plot a clean network map: stops + trips only (no demand overlay).

    This figure is meant to remain readable and is intended to be paired with
    a separate demand map (two-layer visualization strategy).

    Readability and accessibility rules
    -----------------------------------
    - Stops are always drawn *on top* of lines.
    - Stop markers include a white halo (edgecolor) so they remain visible.
    - Colors use a color-blind safe palette (Okabe–Ito via utils.line_color_map).
    - Direction is additionally encoded using line styles (solid/dashed/dot-dash).

    Trip drawing logic
    ------------------
    - If `show_all_trips=False` (default), draw ONE representative trip per
      (line_id, direction_id).
    - If `show_all_trips=True`, draw all trips lightly, then draw the
      representatives on top.

    :param scenario: Scenario containing stops and timetable.
    :param show_all_trips: Whether to overlay all trips (faint) in addition to
        representatives (bold).
    :param representative: Rule to select representative trips per line-direction.
    :param figsize: Matplotlib figure size.
    :param show_stop_labels: Whether to label stops with their stop_id.
    :param stop_marker_size: Marker size for stops.
    :param stop_halo_width: Marker edge width (halo thickness).
    :return: Matplotlib Figure.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    # Always plot stops, even if no timetable
    stops = list(scenario.stops)
    if not stops:
        ax.set_title("Empty scenario (no stops)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.tight_layout()
        return fig

    # Precompute polylines for trips
    polylines = build_trip_polylines(scenario)

    # Prepare color + style encoding
    if scenario.timetable is not None:
        line_ids = [t.line_id for t in scenario.timetable.trips]
    else:
        line_ids = []
    colors = line_color_map(line_ids)
    styles = stable_line_styles()

    def style_for(direction_id: int | None) -> tuple[str, str]:
        idx = 0 if direction_id is None else int(direction_id) % len(styles)
        return styles[idx]  # (linestyle, marker)

    # Choose representative trips per (line_id, direction_id)
    rep_map = select_representative_trips(scenario, rule=representative)
    rep_trip_ids = set(rep_map.values())

    # ---------------------------------------------------------------------
    # 1) Trips (background)
    # ---------------------------------------------------------------------
    # If show_all_trips is enabled: draw all trips faintly first.
    if show_all_trips:
        for tid, poly in polylines.items():
            trip = poly.trip
            color = colors.get(trip.line_id, "#0072B2")
            ls, _mk = style_for(trip.direction_id)

            is_rep = tid in rep_trip_ids
            ax.plot(
                poly.lons,
                poly.lats,
                linestyle=ls,
                marker=None,
                linewidth=1.0 if not is_rep else 1.4,
                alpha=0.16 if not is_rep else 0.26,
                color=color,
                zorder=1,
            )

    # Draw representative trips on top (clear and readable).
    legend_handles: dict[tuple[str | None, int | None], object] = {}
    for (line_id, direction_id), tid in rep_map.items():
        poly = polylines.get(tid)
        if poly is None:
            continue
        trip = poly.trip
        color = colors.get(trip.line_id, "#0072B2")
        ls, _mk = style_for(trip.direction_id)

        (h,) = ax.plot(
            poly.lons,
            poly.lats,
            linestyle=ls,
            marker=None,
            linewidth=2.6,
            alpha=0.92,
            color=color,
            zorder=2,
        )
        legend_handles[(trip.line_id, trip.direction_id)] = h

    # ---------------------------------------------------------------------
    # 2) Stops (foreground)
    # ---------------------------------------------------------------------
    _draw_stops(
        ax,
        scenario,
        show_stop_labels=show_stop_labels,
        stop_marker_size=stop_marker_size,
        stop_halo_width=stop_halo_width,
    )

    # ---------------------------------------------------------------------
    # 3) Cosmetics
    # ---------------------------------------------------------------------
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        "Public transportation network map (trips only)\n"
        f"Trips shown: {'ALL (faint) + representative (bold)' if show_all_trips else 'representative only'}"
    )
    ax.margins(0.10)

    # Legend: one entry per (line, direction). Keep it compact.
    if legend_handles:
        handles = []
        labels = []
        for (lid, did), h in sorted(
            legend_handles.items(),
            key=lambda kv: ("" if kv[0][0] is None else kv[0][0], -1 if kv[0][1] is None else int(kv[0][1])),
        ):
            lid_str = lid if lid is not None else "Line ?"
            did_str = f"dir {did}" if did is not None else "dir ?"
            labels.append(f"{lid_str} ({did_str})")
            handles.append(h)
        ax.legend(handles, labels, loc="best", framealpha=0.85, fontsize=9)

    fig.tight_layout()
    return fig


def _draw_stops(
    ax,
    scenario: Scenario,
    *,
    show_stop_labels: bool,
    stop_marker_size: float,
    stop_halo_width: float,
) -> None:
    """Draw stops with a white halo so they remain visible over lines.

    :param ax: Matplotlib axes.
    :param scenario: Scenario with stops.
    :param show_stop_labels: Whether to label each stop by stop_id.
    :param stop_marker_size: Marker size.
    :param stop_halo_width: Edge width (halo thickness).
    :return: None.
    """
    lons = [float(s.lon) for s in scenario.stops]
    lats = [float(s.lat) for s in scenario.stops]

    # Stops with halo: facecolor is automatic, halo ensures visibility.
    ax.scatter(
        lons,
        lats,
        s=stop_marker_size,
        marker="o",
        edgecolors="white",
        linewidths=stop_halo_width,
        zorder=3,
    )

    # Optional labels with a subtle white background
    if show_stop_labels:
        for s in scenario.stops:
            ax.text(
                float(s.lon),
                float(s.lat),
                f" {s.stop_id}",
                fontsize=9,
                va="center",
                ha="left",
                zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.65, linewidth=0.0),
            )