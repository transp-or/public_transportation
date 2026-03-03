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
from .time_expanded_report import write_time_expanded_report_html, write_time_expanded_report_from_assignment
from .inference_comparison_report import write_od_theta_comparison_report_html