"""Visualization utilities for public_transportation.

This subpackage provides plotting functions for the domain-layer Scenario:
- network map (stops + trips only),
- demand map (OD overlay),
- timetable panel at a stop,
- space–time diagrams.

The domain layer remains dependency-light and does not include plotting code.
"""

from .plot_scenario import BackgroundMapOptions, RenderOptions, RenderOutputs, render_scenario
from .map_trips import plot_network_map
from .demand_map import DemandMapSummary, plot_demand_map

__all__ = [
    "BackgroundMapOptions",
    "RenderOptions",
    "RenderOutputs",
    "render_scenario",
    "plot_network_map",
    "plot_demand_map",
    "DemandMapSummary",
]