from __future__ import annotations

from dataclasses import dataclass
from math import log1p
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


@dataclass(frozen=True)
class DemandMapSummary:
    """Lightweight summary of what was drawn on a demand map.

    :param num_records_input: Number of demand records available in the scenario.
    :param num_records_used: Number of demand records actually drawn (after filtering/aggregation/top-k).
    :param mode: Demand drawing mode.
    :param time_bin_id: Time bin used, if any.
    :param top_k: Top-k filter applied, if any.
    """
    num_records_input: int
    num_records_used: int
    mode: Literal["od_lines", "od_arrows"]
    time_bin_id: str | None
    top_k: int | None


def plot_demand_map(
    scenario: Scenario,
    *,
    mode: Literal["od_lines", "od_arrows"] = "od_lines",
    time_bin_id: str | None = None,
    aggregate_over_bins: bool = True,
    aggregate_by_od: bool = True,
    top_k: int | None = 15,
    figsize: tuple[float, float] = (12.0, 7.0),
    show_stop_labels: bool = True,
    stop_marker_size: float = 70.0,
    stop_halo_width: float = 2.5,
    show_network_context: bool = False,
    representative: Literal["earliest", "latest"] = "earliest",
    network_alpha: float = 0.12,
    network_linewidth: float = 1.0,
    demand_alpha: float = 0.35,
    width_scale: Literal["sqrt", "log"] = "log",
    min_width: float = 0.6,
    max_width: float = 2.0,
) -> tuple[Figure, DemandMapSummary]:
    """Plot a demand-focused map: demand overlay + stops (and optional faint network context).

    This is intended to be the second layer in the two-map visualization strategy:
    - Network map (trips only): readable network geometry.
    - Demand map (this function): readable demand patterns without hiding stops.

    Demand encoding (accessibility)
    -------------------------------
    - Demand is drawn in a neutral color (black) and semi-transparent.
    - Magnitude is encoded primarily by line width (not color).

    Filtering / aggregation
    -----------------------
    - If `time_bin_id` is provided, demand is filtered to that bin.
    - Otherwise, if `aggregate_over_bins=True`, demand is aggregated over all bins.
    - If `aggregate_by_od=True`, demand is summed by (origin, destination).
    - If `top_k` is set, keep only the top-k OD pairs by flow.

    :param scenario: Scenario containing stops and demand.
    :param mode: "od_lines" draws simple OD segments; "od_arrows" draws arrows (more clutter).
    :param time_bin_id: If provided, show demand for this specific time bin only.
    :param aggregate_over_bins: Aggregate demand over all time bins if `time_bin_id` is None.
    :param aggregate_by_od: Sum flows by (origin_stop_id, dest_stop_id).
    :param top_k: Keep only the top-k OD pairs by flow (None = no filtering).
    :param figsize: Matplotlib figure size.
    :param show_stop_labels: Whether to label stops by stop_id.
    :param stop_marker_size: Stop marker size.
    :param stop_halo_width: Stop marker edge width (halo thickness).
    :param show_network_context: If True, draw a faint network context behind demand.
    :param representative: Representative selection rule for the faint context.
    :param network_alpha: Alpha for network context lines.
    :param network_linewidth: Line width for network context.
    :param demand_alpha: Alpha for demand lines.
    :param width_scale: Width scaling for flows: "log" recommended for readability.
    :param min_width: Minimum demand line width.
    :param max_width: Maximum demand line width.
    :return: (Figure, DemandMapSummary)
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    # Stops dict for coordinates
    stop_map = {s.stop_id: s for s in scenario.stops}

    # Optional faint network context (representative trips only)
    if show_network_context:
        _draw_network_context(
            ax,
            scenario,
            representative=representative,
            alpha=network_alpha,
            linewidth=network_linewidth,
        )

    # Collect demand records
    records_all = list(scenario.demand.records)
    num_input = len(records_all)

    # Filter by time bin (if requested)
    if time_bin_id is not None:
        records = [r for r in records_all if r.time_bin_id == time_bin_id]
        used_bin = time_bin_id
    else:
        records = records_all
        used_bin = None

    # If no specific time bin, we either aggregate over bins or not.
    # In this simple OD overlay, "not aggregating over bins" means: keep records as-is.
    # (Still typically aggregated by OD if aggregate_by_od=True.)
    if time_bin_id is None and aggregate_over_bins:
        used_bin = None

    # Aggregate by OD
    if aggregate_by_od:
        agg: dict[tuple[str, str], float] = {}
        for r in records:
            key = (r.origin_stop_id, r.dest_stop_id)
            agg[key] = agg.get(key, 0.0) + float(r.flow)
        od_items = [(o, d, f) for (o, d), f in agg.items()]
    else:
        od_items = [(r.origin_stop_id, r.dest_stop_id, float(r.flow)) for r in records]

    # Remove self-loops, missing stops, and non-positive flows
    od_items = [
        (o, d, f)
        for (o, d, f) in od_items
        if o != d and o in stop_map and d in stop_map and f > 0.0
    ]

    # Top-k filtering
    if top_k is not None:
        od_items = sorted(od_items, key=lambda t: t[2], reverse=True)[: int(top_k)]

    # Width scaling
    flows = [f for _o, _d, f in od_items]
    fmax = max(flows) if flows else 0.0

    def flow_to_width(flow: float) -> float:
        if fmax <= 0.0:
            return min_width
        x = max(0.0, float(flow))
        if width_scale == "log":
            # Map log1p(x) into [min_width, max_width]
            a = log1p(x)
            b = log1p(fmax)
            t = 0.0 if b <= 0.0 else a / b
        else:
            # sqrt scaling
            t = (x ** 0.5) / (fmax ** 0.5) if fmax > 0.0 else 0.0
        return min_width + (max_width - min_width) * t

    # Draw demand
    if not od_items:
        ax.set_title("Demand map (no demand to draw)")
    else:
        for o, d, f in od_items:
            so = stop_map[o]
            sd = stop_map[d]
            x0, y0 = float(so.lon), float(so.lat)
            x1, y1 = float(sd.lon), float(sd.lat)
            lw = flow_to_width(f)

            if mode == "od_arrows":
                # Arrowheads add clutter; keep them small.
                ax.arrow(
                    x0,
                    y0,
                    x1 - x0,
                    y1 - y0,
                    length_includes_head=True,
                    head_width=0.0009,
                    head_length=0.0009,
                    linewidth=lw,
                    alpha=demand_alpha,
                    color="#000000",
                    zorder=2,
                )
            else:
                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    linewidth=lw,
                    alpha=demand_alpha,
                    color="#000000",
                    zorder=2,
                )

        title = "Demand map (OD overlay)"
        if top_k is not None:
            title += f" — top {int(top_k)} OD"
        if time_bin_id is not None:
            title += f", time bin={time_bin_id}"
        elif aggregate_over_bins:
            title += ", aggregated over bins"
        ax.set_title(title)

    # Draw stops on top
    _draw_stops(
        ax,
        scenario,
        show_stop_labels=show_stop_labels,
        stop_marker_size=stop_marker_size,
        stop_halo_width=stop_halo_width,
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.margins(0.10)
    fig.tight_layout()

    summary = DemandMapSummary(
        num_records_input=num_input,
        num_records_used=len(od_items),
        mode=mode,
        time_bin_id=used_bin,
        top_k=top_k,
    )
    return fig, summary


def _draw_stops(
    ax,
    scenario: Scenario,
    *,
    show_stop_labels: bool,
    stop_marker_size: float,
    stop_halo_width: float,
) -> None:
    """Draw stops with a white halo so they remain visible over demand lines.

    :param ax: Matplotlib axes.
    :param scenario: Scenario with stops.
    :param show_stop_labels: Whether to label each stop by stop_id.
    :param stop_marker_size: Marker size.
    :param stop_halo_width: Edge width (halo thickness).
    :return: None.
    """
    lons = [float(s.lon) for s in scenario.stops]
    lats = [float(s.lat) for s in scenario.stops]

    ax.scatter(
        lons,
        lats,
        s=stop_marker_size,
        marker="o",
        edgecolors="white",
        linewidths=stop_halo_width,
        zorder=3,
    )

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


def _draw_network_context(
    ax,
    scenario: Scenario,
    *,
    representative: Literal["earliest", "latest"],
    alpha: float,
    linewidth: float,
) -> None:
    """Draw a faint network context behind demand.

    Uses the same representative-trip logic as the network map, but draws
    everything in faint grey to avoid visual competition with demand.

    :param ax: Matplotlib axes.
    :param scenario: Scenario.
    :param representative: Representative selection rule.
    :param alpha: Line transparency.
    :param linewidth: Line width.
    :return: None.
    """
    polylines = build_trip_polylines(scenario)
    if not polylines:
        return

    # Style encoding retained (linestyle per direction) for robustness
    if scenario.timetable is not None:
        line_ids = [t.line_id for t in scenario.timetable.trips]
    else:
        line_ids = []
    _ = line_color_map(line_ids)  # kept for future; context is grey
    styles = stable_line_styles()

    def style_for(direction_id: int | None) -> str:
        idx = 0 if direction_id is None else int(direction_id) % len(styles)
        return styles[idx][0]

    rep_map = select_representative_trips(scenario, rule=representative)

    for (_lid, did), tid in rep_map.items():
        poly = polylines.get(tid)
        if poly is None:
            continue
        ls = style_for(did)
        ax.plot(
            poly.lons,
            poly.lats,
            linestyle=ls,
            marker=None,
            linewidth=linewidth,
            alpha=alpha,
            color="#444444",
            zorder=1,
        )