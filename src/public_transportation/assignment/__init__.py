"""
Assignment subpackage (JAX-ready time-expanded graph + differentiable loading).

This subpackage contains:
- configuration objects (fixed coefficients such as early/late penalties, transfer weight),
- builders that convert a validated domain `Scenario` into a static, JAX-compatible graph,
- differentiable Dial-style assignment routines producing link flows from OD demand (and optionally theta).

Public API (stable entry points)
--------------------------------
Typical usage:

    from public_transportation.domain import Scenario
    from public_transportation.assignment import AssignmentConfig, build_jax_graph, assign_flows

    scenario = Scenario.from_folder(...)
    config = AssignmentConfig(...)
    graph = build_jax_graph(scenario, config=config)
    x_link = assign_flows(graph, od_params, theta=..., config=config)

Notes
-----
The domain layer is intentionally free of JAX. The conversion happens here.
"""

from __future__ import annotations

# Keep imports tolerant while the subpackage is built file-by-file.
# Once all modules exist, these imports will resolve normally.
try:
    from .config import AssignmentConfig
except Exception:  # pragma: no cover
    AssignmentConfig = None  # type: ignore[misc,assignment]

try:
    from .jax_graph_types import JaxTimeExpandedGraph, ODGrouping, GraphBuildReport
except Exception:  # pragma: no cover
    JaxTimeExpandedGraph = None  # type: ignore[misc,assignment]
    ODGrouping = None  # type: ignore[misc,assignment]
    GraphBuildReport = None  # type: ignore[misc,assignment]

try:
    from .build_time_expanded import build_jax_graph
except Exception:  # pragma: no cover
    build_jax_graph = None  # type: ignore[misc,assignment]

try:
    from .assign import assign_flows
except Exception:  # pragma: no cover
    assign_flows = None  # type: ignore[misc,assignment]

from .cache import (
    ASSIGNMENT_CACHE_SCHEMA_VERSION,
    AssignmentCacheMetrics,
    assignment_cache_path,
    assignment_cache_provenance,
    load_or_prepare_assignment,
)

__all__ = [
    "AssignmentConfig",
    "JaxTimeExpandedGraph",
    "ODGrouping",
    "GraphBuildReport",
    "build_jax_graph",
    "assign_flows",
    "ASSIGNMENT_CACHE_SCHEMA_VERSION",
    "AssignmentCacheMetrics",
    "assignment_cache_path",
    "assignment_cache_provenance",
    "load_or_prepare_assignment",
]
