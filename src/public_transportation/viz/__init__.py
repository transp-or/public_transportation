"""Visualization utilities for public_transportation.

This subpackage provides plotting functions for the domain-layer Scenario:
- network map (stops + trips only),
- demand map (OD overlay),
- timetable panel at a stop,
- space–time diagrams.

The domain layer remains dependency-light and does not include plotting code.
"""

from .plot_scenario import (
    BackgroundMapOptions as BackgroundMapOptions,
    RenderOptions as RenderOptions,
    RenderOutputs as RenderOutputs,
    render_scenario as render_scenario,
)
from .map_trips import plot_network_map as plot_network_map
from .demand_map import (
    DemandMapSummary as DemandMapSummary,
    plot_demand_map as plot_demand_map,
)
from .time_expanded_report import (
    write_time_expanded_report_from_assignment as write_time_expanded_report_from_assignment,
    write_time_expanded_report_html as write_time_expanded_report_html,
)
from .inference_comparison_report import (
    write_od_theta_comparison_report_html as write_od_theta_comparison_report_html,
)
