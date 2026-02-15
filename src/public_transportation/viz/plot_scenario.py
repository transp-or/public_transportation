from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from public_transportation.domain import Scenario

from .demand_map import plot_demand_map
from .map_trips import plot_network_map
from .space_time import plot_space_time_diagrams
from .timetable_panel import plot_timetable_panel
from .utils import infer_hub_stop_id


@dataclass(slots=True)
class RenderOutputs:
    """Which outputs to generate.

    :param network_map: Network map (stops + trips only).
    :param demand_map: Demand map (OD overlay + stops; optional faint network context).
    :param timetable_panel: Timetable panel for a selected stop (default: hub).
    :param space_time: Space–time diagrams per line-direction.
    """

    network_map: bool = True
    demand_map: bool = True
    timetable_panel: bool = True
    space_time: bool = True


@dataclass(slots=True)
class BackgroundMapOptions:
    """Options for adding a background map (future extension).

    This is a placeholder so that the API stays stable when we add OSM tiles later.

    :param provider: Background provider. Only "none" is implemented now.
        Planned: "osm" (OpenStreetMap tiles), potentially others.
    :param zoom: Tile zoom level (if provider supports it).
    :param alpha: Transparency of background layer.
    """

    provider: Literal["none", "osm"] = "none"
    zoom: int = 14
    alpha: float = 0.8


@dataclass(slots=True)
class RenderOptions:
    """Rendering options shared across plots.

    Network map options
    -------------------
    :param show_all_trips: If False, plot one representative trip per line-direction.
        If True, plot all trips (plus representatives highlighted).
    :param representative: How to choose the representative trip when show_all_trips=False.

    Timetable panel options
    -----------------------
    :param hub_stop_id: Stop for the timetable panel. If None, pick 'C' if present, else first stop.

    Common output options
    ---------------------
    :param figsize: Matplotlib figure size in inches.
    :param dpi: DPI for PNG output when saving.
    :param save_format: Output format for saved figures (currently "png" recommended).
    :param basename: Base name for saved files (without extension).

    Demand map options
    ------------------
    :param demand_mode: "od_lines" (default) or "od_arrows" (more clutter).
    :param demand_time_bin_id: If provided, show demand for this specific time bin only.
    :param aggregate_demand_over_bins: If True, aggregate demand over all bins (ignored if demand_time_bin_id provided).
    :param aggregate_demand_by_od: If True, sum flows by (origin, destination).
    :param demand_top_k: If provided, draw only the top-k OD pairs.
    :param demand_show_network_context: If True, show a faint network context behind demand.

    :param background: Background map options (placeholder; tiles not implemented yet).
    """

    # Network map
    show_all_trips: bool = False
    representative: Literal["earliest", "latest"] = "earliest"

    # Timetable panel
    hub_stop_id: str | None = None

    # Output
    figsize: tuple[float, float] = (12.0, 7.0)
    dpi: int = 160
    save_format: Literal["png"] = "png"
    basename: str = "scenario"

    # Demand map
    demand_mode: Literal["od_lines", "od_arrows"] = "od_lines"
    demand_time_bin_id: str | None = None
    aggregate_demand_over_bins: bool = True
    aggregate_demand_by_od: bool = True
    demand_top_k: int | None = 15
    demand_show_network_context: bool = False

    background: BackgroundMapOptions = field(default_factory=BackgroundMapOptions)


def render_scenario(
    scenario: Scenario,
    *,
    outputs: RenderOutputs | None = None,
    options: RenderOptions | None = None,
    display: bool = True,
    out_dir: str | Path | None = None,
) -> dict[str, Figure]:
    """Render standard visualizations for a Scenario.

    The function returns Matplotlib Figure objects, and can optionally display
    them on screen and/or save PNGs.

    Keys returned are stable and descriptive, for example:
    - "network_map"
    - "demand_map"
    - "timetable_<STOP_ID>"
    - "space_time_<LINE>_dir<DIR>"

    :param scenario: Domain scenario.
    :param outputs: Which plots to generate. If None, generate all.
    :param options: Rendering options. If None, use defaults.
    :param display: If True, display figures (blocks until closed).
    :param out_dir: If provided, save figures to this directory.
    :return: Dictionary mapping plot keys to Matplotlib Figure objects.
    """

    outs = outputs if outputs is not None else RenderOutputs()
    opt = options if options is not None else RenderOptions()

    figs: dict[str, Figure] = {}

    # 1) Network map (stops + trips only)
    if outs.network_map:
        fig = plot_network_map(
            scenario,
            show_all_trips=opt.show_all_trips,
            representative=opt.representative,
            figsize=opt.figsize,
        )
        figs["network_map"] = fig

    # 2) Demand map (OD overlay + stops)
    if outs.demand_map:
        fig, _summary = plot_demand_map(
            scenario,
            mode=opt.demand_mode,
            time_bin_id=opt.demand_time_bin_id,
            aggregate_over_bins=opt.aggregate_demand_over_bins,
            aggregate_by_od=opt.aggregate_demand_by_od,
            top_k=opt.demand_top_k,
            figsize=opt.figsize,
            show_network_context=opt.demand_show_network_context,
            representative=opt.representative,
        )
        figs["demand_map"] = fig

    # 3) Timetable panel (at hub or user-selected stop)
    if outs.timetable_panel:
        sid = infer_hub_stop_id(scenario, opt.hub_stop_id)
        fig = plot_timetable_panel(scenario, stop_id=sid, figsize=opt.figsize)
        figs[f"timetable_{sid}"] = fig

    # 4) Space–time diagrams
    if outs.space_time:
        st_figs = plot_space_time_diagrams(scenario, figsize=opt.figsize, combine=False)
        figs.update(st_figs)

    # Save if requested
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for key, fig in figs.items():
            filename = f"{opt.basename}_{key}.{opt.save_format}"
            fig.savefig(out_path / filename, dpi=opt.dpi)

    # Display if requested
    if display:
        # Let Matplotlib manage displaying all created figures.
        # This is the most reliable approach when running from a CLI script.
        plt.show()

    return figs