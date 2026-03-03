

"""Human-readable HTML report for a time-expanded public transport graph.

This module is intentionally *pure Python* (no JAX tracing). It is meant for
inspection/debugging, not for performance.

This report does NOT compute or reconstruct model quantities (costs/flows): it only displays
the graph structure and any per-link flows/costs explicitly provided. In particular:
- The report does NOT compute assignment results or generalized link costs.
- If you want to display per-link generalized costs, you MUST pass in the exact vector
  used by your assignment (e.g., via the `link_cost` argument). Otherwise, generalized
  costs will be omitted or shown as `—`.

It produces:
- a compact summary (counts, degrees, time span),
- node listings by kind (centroid-in, event-arr, event-dep, centroid-out),
- link listings by type (ride, transfer, access, egress, dwell/continue),
- an optional simple SVG visualization when the graph is small enough.

The visualization uses:
- time on the vertical axis,
- stops on the horizontal axis,
- centroid nodes as squares, ARR event nodes drawn left and DEP event nodes drawn right (both as circles),
- ride/transfer/dwell links as solid lines,
- access/egress links as dotted lines,
- event nodes annotated with trip_id and line_ref (when available).

Node kinds:
- 0 = centroid-in
- 1 = event-arr
- 2 = event-dep
- 3 = centroid-out

Link types:
- 0 = ride (event_dep -> event_arr)
- 1 = transfer (event_arr -> event_dep, inter-line)
- 2 = access (centroid_in -> event_dep)
- 3 = egress (event_arr -> centroid_out)
- 4 = dwell/continue (event_arr -> event_dep, same trip)

The report works best when the JaxGraph includes optional metadata:
- node_stop_id: tuple[str, ...]
- trip_id: tuple[str, ...]
- trip_line_ref: tuple[str | None, ...]
- node_trip_index: Array[int] shape (num_nodes,)  (event nodes only; others -1)

If some of these fields are missing, the report degrades gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import html

from public_transportation.viz.html_utils import esc, wrap_html
import numpy as np

# --------------------------------------------------------------------------------------
# Visualization constants (SVG)
# --------------------------------------------------------------------------------------
# NOTE: These values control only the *report visualization* (not model semantics).
# Keeping them centralized makes the rendering easier to tune.

# Node kind codes (kept consistent with the builder / JaxGraph conventions)
NODE_KIND_CENTROID_IN = 0
NODE_KIND_EVENT_ARR = 1
NODE_KIND_EVENT_DEP = 2
NODE_KIND_CENTROID_OUT = 3

# Link type codes (kept consistent with the builder / JaxGraph conventions)
LINK_TYPE_RIDE = 0
LINK_TYPE_TRANSFER = 1
LINK_TYPE_ACCESS = 2
LINK_TYPE_EGRESS = 3
LINK_TYPE_DWELL = 4

# SVG layout (base units before (sx, sy) scaling)
SVG_X_STEP = 140.0
SVG_LEFT_MARGIN = 220.0
SVG_TIME_LABEL_PAD = 40.0  # gap between time labels and the network (in px, before scaling)
SVG_TIME_LABEL_MIN_LEFT = 60.0  # minimum left gutter before the time-label anchor (in px, before scaling)

# Time-label sizing heuristics (pure-Python, conservative; used only for SVG layout)
SVG_TIME_LABEL_FONT_PX = 11.0          # nominal font size for left time labels (before ui scaling)
SVG_TIME_LABEL_CHAR_EM = 0.78          # estimated character width in em-units (conservative)
SVG_TIME_LABEL_EXTRA_PAD_MULT = 2.5    # extra padding factor to keep labels well separated from the network
SVG_TOP_MARGIN = 40.0
SVG_ROW_STEP = 26.0
SVG_PX_PER_MIN = 30.0
SVG_BOTTOM_LABEL_PAD = 50.0

# Node rendering offsets / sizes (base units before (sx, sy) scaling)
SVG_EVENT_LABEL_DX = 10.0
SVG_ARR_X_OFFSET = +30.0
SVG_DEP_X_OFFSET = -30.0
SVG_EVENT_RADIUS_ARR = 5.5
SVG_EVENT_RADIUS_DEP = 4.8
SVG_CENTROID_HALF_SIZE = 6.0

# Collision handling (base units before (sx, sy) scaling)
SVG_JITTER_STEP = 16.0

# Styling
SVG_GRID_STROKE = "#888"
SVG_LINK_STROKE = "#444"
SVG_NODE_STROKE = "#111"
SVG_TEXT_FILL = "#111"
SVG_GRID_DASH = "2,4"
SVG_LINK_DASH_ACCESS_EGRESS = "4,4"


# We deliberately avoid importing JAX here.
# The graph contains jax.numpy arrays, but they are array-like; we convert to numpy.


@dataclass(frozen=True, slots=True)
class ReportDrawLimits:
    """Thresholds controlling whether an SVG visualization is produced."""

    max_nodes: int = 300
    max_links: int = 800


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


# Convenience wrapper: generate report from scenario, assignment, config (no cost computation).
def write_time_expanded_report_from_assignment(
    *,
    scenario: Any,
    assignment: Any,
    config: Any,
    output_path: str | Path,
    title: str = "Time-expanded graph report",
    y_mode: str = "equal",
    svg_scale_x: float = 1.0,
    svg_scale_y: float = 1.0,
    draw_limits: ReportDrawLimits = ReportDrawLimits(),
) -> Path:
    """Generate an HTML report for a Scenario using artifacts produced by the assignment.

    This is a convenience wrapper so example scripts do not need to:
    - call `prepare_assignment`,
    - compute link costs,
    - pass domain-specific time-bin interval logic.

    The report will display exactly the per-link generalized costs used by the assignment
    if the assignment result provides them.

    :param scenario: Domain Scenario used to build the time-expanded graph and enrich labels.
    :param assignment: Assignment result object. Must provide `link_flow`.
        If it provides `link_cost` (per-link generalized costs, minutes) it will be displayed.
    :param config: AssignmentConfig used to build the time-expanded graph artifacts.
    :param output_path: Where to write the HTML file.
    :param title: HTML title.
    :param y_mode: SVG vertical layout mode.
    :param svg_scale_x: Horizontal scale factor for the SVG visualization.
    :param svg_scale_y: Vertical scale factor for the SVG visualization.
    :param draw_limits: Thresholds controlling SVG generation.
    :return: Path to the generated HTML file.
    """
    # Import locally to avoid import-time cycles between viz and assignment.
    from public_transportation.assignment.assign import prepare_assignment  # local import

    arts = prepare_assignment(scenario, config)
    graph = getattr(arts, "graph")

    try:
        link_flow_obj = assignment.link_flow
    except AttributeError as e:
        raise AttributeError(
            "Assignment result must provide `link_flow` for reporting."
        ) from e

    link_flow = np.asarray(link_flow_obj, dtype=float).reshape(-1)

    # link_cost is optional: if present, we display exactly what assignment used
    link_cost_obj = getattr(assignment, "link_cost", None)
    link_cost = None
    if link_cost_obj is not None:
        link_cost = np.asarray(link_cost_obj, dtype=float).reshape(-1)

    return write_time_expanded_report_html(
        graph=graph,
        output_path=output_path,
        scenario=scenario,
        title=title,
        y_mode=y_mode,
        svg_scale_x=svg_scale_x,
        svg_scale_y=svg_scale_y,
        link_flow=link_flow,
        link_cost=link_cost,
        draw_limits=draw_limits,
    )


def write_time_expanded_report_html(
    *,
    graph: Any,
    output_path: str | Path,
    scenario: Any | None = None,
    title: str = "Time-expanded graph report",
    y_mode: str = "equal",
    svg_scale_x: float = 1.0,
    svg_scale_y: float = 1.0,
    link_flow: Sequence[float] | None = None,
    link_cost: Sequence[float] | None = None,
    draw_limits: ReportDrawLimits = ReportDrawLimits(),
) -> Path:
    """Generate an HTML report describing a time-expanded graph.

    :param graph: A `JaxGraph`-like object produced by `build_time_expanded.build_jax_graph`.
        Required attributes: num_nodes, num_links, tail, head, link_type, node_kind, node_time,
        node_time_s, node_stop_index, travel_time (or legacy travel_time_min).
        Optional: node_stop_id, trip_id, trip_line_ref, node_trip_index, node_stop_name.    :param output_path: Where to write the HTML file.
    :param scenario: Optional domain Scenario. Used only to enrich labels (when available).
    :param title: HTML title.
    :param svg_scale_x: Horizontal scale factor for the SVG visualization (stops spacing, horizontal margins, label offsets).
    :param svg_scale_y: Vertical scale factor for the SVG visualization (row spacing / time scaling, vertical margins).
    :param link_flow: Optional per-link flows (used for display).
    :param link_cost: Optional per-link *generalized* costs in minutes; **must be the exact vector used by the assignment**.
        This function will NOT attempt to reconstruct or compute generalized costs; if not provided,
        the report will show `—` for generalized cost.
    :param draw_limits: Thresholds controlling SVG generation.
    :return: Path to the generated HTML file.
    """

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    g = _np_view(graph)

    flow = None
    if link_flow is not None:
        flow_arr = np.asarray(link_flow, dtype=float).reshape(-1)
        if flow_arr.shape[0] != g.num_links:
            raise ValueError(
                "link_flow has inconsistent length: expected num_links="
                f"{g.num_links}, got {flow_arr.shape[0]}."
            )
        flow = flow_arr

    cost = None
    if link_cost is not None:
        cost_arr = np.asarray(link_cost, dtype=float).reshape(-1)
        if cost_arr.shape[0] != g.num_links:
            raise ValueError(
                "link_cost has inconsistent length: expected num_links="
                f"{g.num_links}, got {cost_arr.shape[0]}."
            )
        cost = cost_arr

    # Basic derived stats
    deg_out = _out_degrees(g.num_nodes, g.tail)
    deg_in = _in_degrees(g.num_nodes, g.head)

    # Metadata helpers
    stop_ids = _stop_ids(graph)
    trip_ids = _trip_ids(graph)
    trip_line_ref = _trip_line_refs(graph)
    node_trip_index = _node_trip_index(graph, g.num_nodes)

    # Time span over event nodes (kind==1 or kind==2)
    event_mask = (g.node_kind == 1) | (g.node_kind == 2)
    if event_mask.any():
        tmin = float(np.min(g.node_time[event_mask]))
        tmax = float(np.max(g.node_time[event_mask]))
    else:
        tmin, tmax = 0.0, 0.0

    # Stop name enrichment
    stop_names = _stop_names(scenario)
    node_stop_names = _node_stop_names(graph, g.num_nodes)
    # Time bin enrichment
    time_bins = _time_bins_info(scenario)

    # Build sections
    summary_html = _render_summary(
        title=title,
        num_nodes=g.num_nodes,
        num_links=g.num_links,
        deg_in=deg_in,
        deg_out=deg_out,
        event_time_min=tmin,
        event_time_max=tmax,
        has_lines=trip_line_ref is not None,
    )

    nodes_html = _render_nodes_table(
        g=g,
        stop_ids=stop_ids,
        trip_ids=trip_ids,
        trip_line_ref=trip_line_ref,
        node_trip_index=node_trip_index,
        stop_names=stop_names,
        node_stop_names=node_stop_names,
        time_bins=time_bins,
    )

    links_html = _render_links_table(
        g=g,
        stop_ids=stop_ids,
        trip_ids=trip_ids,
        trip_line_ref=trip_line_ref,
        node_trip_index=node_trip_index,
        link_flow=flow,
        link_cost=cost,
        stop_names=stop_names,
        node_stop_names=node_stop_names,
        time_bins=time_bins,
    )

    paths_html = ""
    if flow is not None:
        paths_html = _render_paths_table(
            g=g,
            stop_ids=stop_ids,
            trip_ids=trip_ids,
            trip_line_ref=trip_line_ref,
            node_trip_index=node_trip_index,
            link_flow=flow,
            link_cost=cost,
            stop_names=stop_names,
            node_stop_names=node_stop_names,
            time_bins=time_bins,
        )

    svg_html = ""
    if g.num_nodes <= draw_limits.max_nodes and g.num_links <= draw_limits.max_links:
        svg_html = _render_svg(
            g=g,
            stop_ids=stop_ids,
            trip_ids=trip_ids,
            trip_line_ref=trip_line_ref,
            node_trip_index=node_trip_index,
            link_flow=flow,
            link_cost=cost,
            svg_scale_x=svg_scale_x,
            svg_scale_y=svg_scale_y,
            y_mode=y_mode,
            time_bins=time_bins,
            title="Graph view (time vertical, stops horizontal)",
            stop_names=stop_names,
            node_stop_names=node_stop_names,
        )
    else:
        svg_html = (
            f"<p><em>Visualization skipped</em>: graph too large to draw "
            f"(nodes={g.num_nodes}, links={g.num_links}; limits: "
            f"nodes≤{draw_limits.max_nodes}, links≤{draw_limits.max_links}).</p>"
        )

    page = _wrap_html(title, summary_html + svg_html + nodes_html + paths_html + links_html)
    out.write_text(page, encoding="utf-8")
    return out


# --------------------------------------------------------------------------------------
# Numpy view + metadata access
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _GraphNP:
    num_nodes: int
    num_links: int
    tail: np.ndarray
    head: np.ndarray
    link_type: np.ndarray
    travel_time: np.ndarray
    node_kind: np.ndarray
    node_time: np.ndarray
    node_time_s: np.ndarray
    node_stop_index: np.ndarray


def _np_array(x: Any, *, dtype: Any | None = None) -> np.ndarray:
    a = np.asarray(x)
    if dtype is None:
        return a
    return a.astype(dtype, copy=False)


def _np_view(graph: Any) -> _GraphNP:
    """Convert a JaxGraph-like object to a small numpy-backed view.

    This is intentionally fail-fast: if required graph attributes are missing or
    inconsistent, we raise instead of silently filling placeholders.
    """

    # Required scalar sizes
    num_nodes = int(graph.num_nodes)
    num_links = int(graph.num_links)

    # Required topology
    tail = _np_array(graph.tail, dtype=int)
    head = _np_array(graph.head, dtype=int)
    link_type = _np_array(graph.link_type, dtype=int)

    # Required node fields
    node_kind = _np_array(graph.node_kind, dtype=int)
    node_time = _np_array(graph.node_time, dtype=float)
    node_time_s = _np_array(graph.node_time_s, dtype=int)
    node_stop_index = _np_array(graph.node_stop_index, dtype=int)

    # Required per-link travel time (support legacy attribute name)
    if hasattr(graph, "travel_time"):
        travel_time_obj = graph.travel_time
    elif hasattr(graph, "travel_time_min"):
        travel_time_obj = graph.travel_time_min
    else:
        raise AttributeError(
            "Graph is missing required per-link travel time attribute: expected `travel_time` "
            "(or legacy `travel_time_min`)."
        )
    travel_time = _np_array(travel_time_obj, dtype=float)

    # Shape checks (catch API mismatches early)
    if tail.shape[0] != num_links or head.shape[0] != num_links or link_type.shape[0] != num_links:
        raise ValueError(
            "Graph link arrays have inconsistent length: expected num_links="
            f"{num_links}, got tail={tail.shape}, head={head.shape}, link_type={link_type.shape}."
        )
    if travel_time.shape[0] != num_links:
        raise ValueError(
            "Graph travel_time has inconsistent length: expected num_links="
            f"{num_links}, got travel_time={travel_time.shape}."
        )
    if (
        node_kind.shape[0] != num_nodes
        or node_time.shape[0] != num_nodes
        or node_time_s.shape[0] != num_nodes
        or node_stop_index.shape[0] != num_nodes
    ):
        raise ValueError(
            "Graph node arrays have inconsistent length: expected num_nodes="
            f"{num_nodes}, got node_kind={node_kind.shape}, node_time={node_time.shape}, "
            f"node_time_s={node_time_s.shape}, node_stop_index={node_stop_index.shape}."
        )

    return _GraphNP(
        num_nodes=num_nodes,
        num_links=num_links,
        tail=tail,
        head=head,
        link_type=link_type,
        travel_time=travel_time,
        node_kind=node_kind,
        node_time=node_time,
        node_time_s=node_time_s,
        node_stop_index=node_stop_index,
    )


def _stop_ids(graph: Any) -> tuple[str, ...] | None:
    v = getattr(graph, "node_stop_id", None)
    if v is None or len(v) == 0:
        return None
    return tuple(str(s) for s in v)


def _trip_ids(graph: Any) -> tuple[str, ...] | None:
    v = getattr(graph, "trip_id", None)
    if v is None or len(v) == 0:
        return None
    return tuple(str(s) for s in v)


def _trip_line_refs(graph: Any) -> tuple[str | None, ...] | None:
    v = getattr(graph, "trip_line_ref", None)
    if v is None:
        return None
    if isinstance(v, tuple):
        return tuple((None if x is None else str(x)) for x in v)
    try:
        return tuple((None if x is None else str(x)) for x in list(v))
    except Exception:
        return None


def _node_trip_index(graph: Any, num_nodes: int) -> np.ndarray:
    v = getattr(graph, "node_trip_index", None)
    if v is None:
        return -np.ones((num_nodes,), dtype=int)
    return _np_array(v, dtype=int)

# --------------------------------------------------------------------------------------
# Time-bin enrichment (optional)
# --------------------------------------------------------------------------------------

def _time_bins_info(scenario: Any | None) -> list[dict[str, Any]] | None:
    """Return time-bin interval information for reporting/layout only.

    This MUST NOT be used to compute model quantities (costs/flows). It is only
    used to label and place duplicated centroid-in nodes (one per time bin).

    :param scenario: Optional domain Scenario.
    :return: List of dicts with keys: idx, bin_id, start_s, end_s, label; or None.
    """
    if scenario is None:
        return None
    tbs = getattr(scenario, "time_bins", None)
    if not tbs:
        return None

    out: list[dict[str, Any]] = []
    for idx, tb in enumerate(list(tbs)):
        bin_id = getattr(tb, "bin_id", None)
        start = getattr(tb, "start", None)
        end = getattr(tb, "end", None)
        try:
            start_s = int(getattr(start, "seconds_from_midnight"))
            end_s = int(getattr(end, "seconds_from_midnight"))
        except Exception:
            return None

        label = f"[{_fmt_hms(start_s)} – {_fmt_hms(end_s)}]"
        if bin_id is not None and str(bin_id):
            label = f"{bin_id} {label}"

        out.append(
            {
                "idx": int(idx),
                "bin_id": None if bin_id is None else str(bin_id),
                "start_s": start_s,
                "end_s": end_s,
                "label": label,
            }
        )
    return out


def _centroid_in_time_bin_index(*, node_idx: int, node_kind: int, stop_index: int, num_time_bins: int) -> int | None:
    """Infer the time-bin index for a centroid-in node under the builder ordering.

    Current builder ordering for centroid-in nodes:
      node = stop_pos * num_time_bins + bin_idx

    :param node_idx: Node index.
    :param node_kind: Node kind code.
    :param stop_index: stop_pos.
    :param num_time_bins: number of time bins.
    :return: bin index or None.
    """
    if node_kind != NODE_KIND_CENTROID_IN:
        return None
    if num_time_bins <= 0:
        return None
    start = int(stop_index) * int(num_time_bins)
    end = start + int(num_time_bins)
    if start <= int(node_idx) < end:
        return int(node_idx) - start
    return None

# --------------------------------------------------------------------------------------
# Node stop name enrichment
# --------------------------------------------------------------------------------------

def _node_stop_names(graph: Any, num_nodes: int) -> np.ndarray | None:
    """Return per-node stop names if the graph provides them, else None.

    Expected optional attribute: `node_stop_name` aligned with nodes.
    """
    v = getattr(graph, "node_stop_name", None)
    if v is None:
        return None
    try:
        if len(v) != num_nodes:
            return None
        # Keep as unicode array for cheap indexing
        return np.asarray(["" if x is None else str(x) for x in v], dtype=object)
    except Exception:
        return None

# --------------------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------------------

_NODE_KIND = {0: "centroid-in", 1: "event-arr", 2: "event-dep", 3: "centroid-out"}
_LINK_TYPE = {0: "ride", 1: "transfer", 2: "access", 3: "egress", 4: "dwell"}


def _fmt_hms(seconds: int) -> str:
    if seconds < 0:
        return ""
    s = int(seconds) % 86400
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _fmt_min(minutes: float) -> str:
    if minutes > 1e11:
        return "+∞"
    if minutes < -1e11:
        return "-∞"
    return f"{minutes:.2f}"


def _out_degrees(num_nodes: int, tail: np.ndarray) -> np.ndarray:
    deg = np.zeros((num_nodes,), dtype=int)
    np.add.at(deg, tail, 1)
    return deg


def _in_degrees(num_nodes: int, head: np.ndarray) -> np.ndarray:
    deg = np.zeros((num_nodes,), dtype=int)
    np.add.at(deg, head, 1)
    return deg


def _wrap_html(title: str, body: str) -> str:
    """Wrap a full HTML page.

    We reuse the shared HTML utilities to keep styling consistent across report generators.
    This module adds only SVG-specific CSS.
    """

    svg_css = """
    .svgbox { border:1px solid #ddd; border-radius: 10px; padding: 10px; overflow: auto; }
    """
    return wrap_html(title=title, body=body, css=svg_css)


def _render_summary(
    *,
    title: str,
    num_nodes: int,
    num_links: int,
    deg_in: np.ndarray,
    deg_out: np.ndarray,
    event_time_min: float,
    event_time_max: float,
    has_lines: bool,
) -> str:
    return (
        f"<h1>{html.escape(title)}</h1>"
        "<div class='kpi'>"
        f"<div><div class='muted'>Nodes</div><div><b>{num_nodes}</b></div></div>"
        f"<div><div class='muted'>Links</div><div><b>{num_links}</b></div></div>"
        f"<div><div class='muted'>Max out-degree</div><div><b>{int(deg_out.max())}</b></div></div>"
        f"<div><div class='muted'>Max in-degree</div><div><b>{int(deg_in.max())}</b></div></div>"
        f"<div><div class='muted'>Event time span (min)</div><div><b>{_fmt_min(event_time_min)}</b> → <b>{_fmt_min(event_time_max)}</b></div></div>"
        f"<div><div class='muted'>Line annotations</div><div><b>{'yes' if has_lines else 'no'}</b></div></div>"
        "</div>"
    )


def _node_label(
    *,
    node_idx: int,
    kind: int,
    stop_ids: tuple[str, ...] | None,
    stop_index: int,
    time_s: int,
    node_trip_index: np.ndarray,
    trip_ids: tuple[str, ...] | None,
    trip_line_ref: tuple[str | None, ...] | None,
    time_bins: list[dict[str, Any]] | None = None,
) -> str:
    stop = stop_ids[stop_index] if (stop_ids is not None and 0 <= stop_index < len(stop_ids)) else str(stop_index)
    base = f"{node_idx} · {stop}"
    # For centroid-in nodes, append interval label if available
    if kind == NODE_KIND_CENTROID_IN and time_bins is not None and len(time_bins) > 0:
        tb_idx = _centroid_in_time_bin_index(
            node_idx=node_idx,
            node_kind=kind,
            stop_index=stop_index,
            num_time_bins=len(time_bins),
        )
        if tb_idx is not None and 0 <= tb_idx < len(time_bins):
            interval_label = str(time_bins[tb_idx].get("label", ""))
            if interval_label:
                base = base + " · " + interval_label
    if kind in (1, 2):
        ti = int(node_trip_index[node_idx])
        trip = trip_ids[ti] if (trip_ids is not None and 0 <= ti < len(trip_ids)) else (str(ti) if ti >= 0 else "")
        line = trip_line_ref[ti] if (trip_line_ref is not None and 0 <= ti < len(trip_line_ref)) else None
        extra = []
        if time_s >= 0:
            extra.append(_fmt_hms(time_s))
        if trip:
            extra.append(f"trip={trip}")
        if line:
            extra.append(f"line={line}")
        if extra:
            return base + " · " + ", ".join(extra)
    return base


def _render_nodes_table(
    *,
    g: _GraphNP,
    stop_ids: tuple[str, ...] | None,
    trip_ids: tuple[str, ...] | None,
    trip_line_ref: tuple[str | None, ...] | None,
    node_trip_index: np.ndarray,
    stop_names: Mapping[str, str] | None = None,
    node_stop_names: np.ndarray | None = None,
    time_bins: list[dict[str, Any]] | None = None,
) -> str:
    rows = []
    for i in range(g.num_nodes):
        kind = int(g.node_kind[i])
        stop_idx = int(g.node_stop_index[i])
        stop_id = (
            stop_ids[stop_idx]
            if (stop_ids is not None and 0 <= stop_idx < len(stop_ids))
            else str(stop_idx)
        )
        stop_name = ""
        if node_stop_names is not None:
            stop_name = str(node_stop_names[i] or "")
        elif stop_names is not None:
            stop_name = stop_names.get(str(stop_id), "")
        # Compute interval_label for centroid-in nodes
        if kind == NODE_KIND_CENTROID_IN and time_bins is not None and len(time_bins) > 0:
            tb_idx = _centroid_in_time_bin_index(
                node_idx=i,
                node_kind=kind,
                stop_index=stop_idx,
                num_time_bins=len(time_bins),
            )
            if tb_idx is not None and 0 <= tb_idx < len(time_bins):
                interval_label = str(time_bins[tb_idx].get("label", ""))
            else:
                interval_label = ""
        else:
            interval_label = ""
        hms = _fmt_hms(int(g.node_time_s[i]))
        tmin = _fmt_min(float(g.node_time[i]))
        trip = ""
        line = ""
        if kind in (1, 2):
            ti = int(node_trip_index[i])
            trip = (
                trip_ids[ti]
                if (trip_ids is not None and 0 <= ti < len(trip_ids))
                else (str(ti) if ti >= 0 else "")
            )
            line = (
                trip_line_ref[ti]
                if (trip_line_ref is not None and 0 <= ti < len(trip_line_ref))
                else ""
            )
        label = _node_label(
            node_idx=i,
            kind=kind,
            stop_ids=stop_ids,
            stop_index=stop_idx,
            time_s=int(g.node_time_s[i]),
            node_trip_index=node_trip_index,
            trip_ids=trip_ids,
            trip_line_ref=trip_line_ref,
            time_bins=time_bins,
        )
        rows.append(
            (
                kind,
                i,
                _NODE_KIND.get(kind, str(kind)),
                stop_id,
                stop_name,
                interval_label,
                hms,
                tmin,
                trip if kind in (1, 2) else "",
                line if kind in (1, 2) else "",
                label,
            )
        )

    # Sort by kind, stop_id, time, node index
    rows.sort(key=lambda r: (r[0], r[3], r[6], r[1]))

    body = ["<h2>Nodes</h2>"]
    body.append("<table><thead><tr>" + "".join(
        f"<th>{h}</th>" for h in [
            "kind", "node", "stop_id", "stop_name", "departure interval", "time (HH:MM:SS)", "time (min)", "trip", "line", "label"
        ]
    ) + "</tr></thead><tbody>")

    for kind, idx, kind_name, stop_id, stop_name, interval_label, hms, tmin, trip, line, label in rows:
        body.append(
            "<tr>"
            f"<td><span class='pill'>{html.escape(kind_name)}</span></td>"
            f"<td><code>{idx}</code></td>"
            f"<td>{html.escape(str(stop_id))}</td>"
            f"<td>{html.escape(stop_name)}</td>"
            f"<td>{html.escape(interval_label)}</td>"
            f"<td>{html.escape(hms)}</td>"
            f"<td>{html.escape(tmin)}</td>"
            f"<td>{html.escape(trip)}</td>"
            f"<td>{html.escape(line)}</td>"
            f"<td>{html.escape(label)}</td>"
            "</tr>"
        )

    body.append("</tbody></table>")
    return "\n".join(body)


def _render_links_table(
    *,
    g: _GraphNP,
    stop_ids: tuple[str, ...] | None,
    trip_ids: tuple[str, ...] | None,
    trip_line_ref: tuple[str | None, ...] | None,
    node_trip_index: np.ndarray,
    link_flow: np.ndarray | None = None,
    link_cost: np.ndarray | None = None,
    stop_names: Mapping[str, str] | None = None,
    node_stop_names: np.ndarray | None = None,
    time_bins: list[dict[str, Any]] | None = None,
) -> str:
    rows = []
    has_cost = link_cost is not None
    for e in range(g.num_links):
        t = int(g.tail[e])
        h = int(g.head[e])
        lt = int(g.link_type[e])
        lt_name = _LINK_TYPE.get(lt, str(lt))

        # For endpoint label formatting
        t_stop_index = int(g.node_stop_index[t])
        h_stop_index = int(g.node_stop_index[h])
        t_stop_id = (
            stop_ids[t_stop_index]
            if (stop_ids is not None and 0 <= t_stop_index < len(stop_ids))
            else str(t_stop_index)
        )
        h_stop_id = (
            stop_ids[h_stop_index]
            if (stop_ids is not None and 0 <= h_stop_index < len(stop_ids))
            else str(h_stop_index)
        )
        if node_stop_names is not None:
            t_stop_name = str(node_stop_names[t] or "")
            h_stop_name = str(node_stop_names[h] or "")
        else:
            t_stop_name = stop_names.get(str(t_stop_id), "") if stop_names is not None else ""
            h_stop_name = stop_names.get(str(h_stop_id), "") if stop_names is not None else ""
        t_time_s = int(g.node_time_s[t])
        h_time_s = int(g.node_time_s[h])
        tail_label = _endpoint_label(
            node_idx=t,
            stop_ids=stop_ids,
            stop_index=t_stop_index,
            time_s=t_time_s,
            stop_name=t_stop_name,
            node_kind=int(g.node_kind[t]),
            time_bins=time_bins,
        )
        head_label = _endpoint_label(
            node_idx=h,
            stop_ids=stop_ids,
            stop_index=h_stop_index,
            time_s=h_time_s,
            stop_name=h_stop_name,
            node_kind=int(g.node_kind[h]),
            time_bins=time_bins,
        )

        # Notes for link types
        extra = ""
        if lt == 1:
            # transfer: line_from → line_to (if available)
            ti1 = int(node_trip_index[t])
            ti2 = int(node_trip_index[h])
            l1 = trip_line_ref[ti1] if (trip_line_ref is not None and 0 <= ti1 < len(trip_line_ref)) else None
            l2 = trip_line_ref[ti2] if (trip_line_ref is not None and 0 <= ti2 < len(trip_line_ref)) else None
            if l1 or l2:
                extra = f"{l1 or '?'} → {l2 or '?'}"
        elif lt == 4:
            # dwell: continue (same trip) if trip indices match and trip id is available
            ti1 = int(node_trip_index[t])
            ti2 = int(node_trip_index[h])
            if ti1 == ti2 and trip_ids is not None and 0 <= ti1 < len(trip_ids):
                extra = f"continue (same trip {trip_ids[ti1]})"
            else:
                extra = "continue"
        elif lt == 2:
            # access → dep, append interval label if available for centroid-in
            interval_label = ""
            tail_kind = int(g.node_kind[t])
            if tail_kind == NODE_KIND_CENTROID_IN and time_bins is not None and len(time_bins) > 0:
                tb_idx = _centroid_in_time_bin_index(
                    node_idx=t,
                    node_kind=tail_kind,
                    stop_index=t_stop_index,
                    num_time_bins=len(time_bins),
                )
                if tb_idx is not None and 0 <= tb_idx < len(time_bins):
                    interval_label = str(time_bins[tb_idx].get("label", ""))
            if interval_label:
                extra = f"access → dep ({interval_label})"
            else:
                extra = "access → dep"
        elif lt == 3:
            extra = "arr → egress"

        fval = float(link_flow[e]) if link_flow is not None else float("nan")
        if link_cost is not None:
            gen_cost_cell = _fmt_min(float(link_cost[e]))
        else:
            gen_cost_cell = "—"
        travel_cell = _fmt_min(float(g.travel_time[e]))

        rows.append(
            (
                lt,
                e,
                lt_name,
                tail_label,
                head_label,
                fval,
                gen_cost_cell,
                travel_cell,
                extra,
            )
        )

    rows.sort(key=lambda r: (r[0], r[1]))

    body = ["<h2>Links</h2>"]
    if link_cost is None:
        body.append("<p class='muted'><b>Note:</b> Generalized link costs were not provided to the report, so the column ‘generalized travel time (min)’ is shown as ‘—’. Pass <code>link_cost</code> equal to the exact per-link cost vector used in the assignment to display it.</p>")
    body.append("<table><thead><tr>" + "".join(
        f"<th>{h}</th>" for h in [
            "type", "link", "tail", "head", "flow", "generalized travel time (min)", "travel time (min)", "notes"
        ]
    ) + "</tr></thead><tbody>")

    for lt, e, name, t_label, h_label, f, gen_cost_cell, travel_cell, extra in rows:
        if link_flow is None:
            flow_cell = ""
        else:
            flow_cell = f"{f:.6g}"
        body.append(
            "<tr>"
            f"<td><span class='pill'>{html.escape(name)}</span></td>"
            f"<td><code>{e}</code></td>"
            f"<td>{html.escape(t_label)}</td>"
            f"<td>{html.escape(h_label)}</td>"
            f"<td>{html.escape(flow_cell)}</td>"
            f"<td>{html.escape(gen_cost_cell)}</td>"
            f"<td>{html.escape(travel_cell)}</td>"
            f"<td>{html.escape(extra)}</td>"
            "</tr>"
        )

    body.append("</tbody></table>")
    return "\n".join(body)


# Helper for endpoint label in links table
def _endpoint_label(
    *,
    node_idx: int,
    stop_ids: tuple[str, ...] | None,
    stop_index: int,
    time_s: int,
    stop_name: str | None = None,
    node_kind: int | None = None,
    time_bins: list[dict[str, Any]] | None = None,
) -> str:
    stop = stop_ids[stop_index] if (stop_ids is not None and 0 <= stop_index < len(stop_ids)) else str(stop_index)
    # For centroid-in nodes, append interval label if available
    interval_label = ""
    if node_kind == NODE_KIND_CENTROID_IN and time_bins is not None and len(time_bins) > 0:
        tb_idx = _centroid_in_time_bin_index(
            node_idx=node_idx,
            node_kind=node_kind,
            stop_index=stop_index,
            num_time_bins=len(time_bins),
        )
        if tb_idx is not None and 0 <= tb_idx < len(time_bins):
            interval_label = str(time_bins[tb_idx].get("label", ""))
    if stop_name:
        if time_s >= 0:
            if interval_label:
                return f"{node_idx} ({stop} \"{stop_name}\", {interval_label}, {_fmt_hms(time_s)})"
            else:
                return f"{node_idx} ({stop} \"{stop_name}\", {_fmt_hms(time_s)})"
        else:
            if interval_label:
                return f"{node_idx} ({stop} \"{stop_name}\", {interval_label})"
            else:
                return f"{node_idx} ({stop} \"{stop_name}\")"
    else:
        if time_s >= 0:
            if interval_label:
                return f"{node_idx} ({stop}, {interval_label}, {_fmt_hms(time_s)})"
            else:
                return f"{node_idx} ({stop}, {_fmt_hms(time_s)})"
        else:
            if interval_label:
                return f"{node_idx} ({stop}, {interval_label})"
            else:
                return f"{node_idx} ({stop})"


def _render_svg(
    *,
    g: _GraphNP,
    stop_ids: tuple[str, ...] | None,
    trip_ids: tuple[str, ...] | None,
    trip_line_ref: tuple[str | None, ...] | None,
    node_trip_index: np.ndarray,
    link_flow: np.ndarray | None = None,
    link_cost: np.ndarray | None = None,
    svg_scale_x: float = 1.0,
    svg_scale_y: float = 1.0,
    y_mode: str = "equal",
    time_bins: list[dict[str, Any]] | None = None,
    title: str,
    stop_names: Mapping[str, str] | None = None,
    node_stop_names: np.ndarray | None = None,
) -> str:
    """Render a compact SVG view of the time-expanded graph.

    Layout:
      - x axis: stops (one column per stop)
      - y axis: time rows (unique event times) + centroid rows

    Labels:
      - stop labels are shown once at the bottom
      - time tags are shown once at the left
      - event nodes show only line_ref (fallback trip_id) next to the node
      - full details remain available via SVG tooltip (<title>)

    Spacing:
      - y_mode='equal' (default): equal vertical spacing between consecutive time rows
      - y_mode='scaled': vertical spacing proportional to time differences

    Collisions:
      - multiple event nodes sharing the same (stop, time, kind) are spread via horizontal jitter; ordering is line-aware.
    """

    # -----------------------------
    # Validate / basic geometry
    # -----------------------------
    if y_mode not in {"equal", "scaled"}:
        y_mode = "equal"

    # Scaling factors (geometry vs UI)
    sx = float(svg_scale_x) if svg_scale_x is not None else 1.0
    sy = float(svg_scale_y) if svg_scale_y is not None else 1.0
    if sx <= 0.0:
        sx = 1.0
    if sy <= 0.0:
        sy = 1.0
    ui = min(sx, sy)

    stop_count = int(np.max(g.node_stop_index)) + 1 if g.node_stop_index.size else 0

    # Geometry constants
    x_step = SVG_X_STEP * sx
    top = SVG_TOP_MARGIN * sy
    row_step = SVG_ROW_STEP * sy  # for equal mode
    px_per_min = SVG_PX_PER_MIN * sy  # for scaled mode

    # Space for bottom stop labels
    bottom_label_pad = SVG_BOTTOM_LABEL_PAD * sy

    # Left margin is computed *after* we know the time-label strings, so labels never overlap the network.
    # We use a simple character-based width estimate to remain pure-Python and deterministic.
    base_left = SVG_LEFT_MARGIN * sx
    time_label_pad = SVG_TIME_LABEL_PAD * sx
    min_left_gutter = SVG_TIME_LABEL_MIN_LEFT * sx

    def stop_name(s: int) -> str:
        sid = stop_ids[s] if (stop_ids is not None and 0 <= s < len(stop_ids)) else f"stop{s}"
        if stop_names is not None:
            n = stop_names.get(str(sid), "")
            if n:
                return f"{sid} \"{n}\""
        return f"{sid}"

    # -----------------------------
    # Build time rows (unique event times) + centroid rows
    # -----------------------------
    event_mask = (g.node_kind == NODE_KIND_EVENT_ARR) | (g.node_kind == NODE_KIND_EVENT_DEP)
    event_times_s = sorted({int(t) for t in g.node_time_s[event_mask] if int(t) >= 0})

    CENTROID_OUT = "CENTROID_OUT"

    # With duplicated centroid-in per time bin, draw one centroid-in row per bin when available.
    if time_bins is not None and len(time_bins) > 0:
        tb_sorted = sorted(time_bins, key=lambda d: int(d.get("start_s", 0)))
        centroid_in_rows: list[object] = [
            ("CENTROID_IN", int(d["idx"])) for d in tb_sorted
        ]
        centroid_in_label: dict[int, str] = {
            int(d["idx"]): str(d["label"]) for d in tb_sorted
        }
        rows: list[object] = centroid_in_rows + event_times_s + [CENTROID_OUT]
    else:
        centroid_in_rows = ["CENTROID_IN"]
        centroid_in_label = {0: "centroid-in"}
        rows = ["CENTROID_IN"] + event_times_s + [CENTROID_OUT]

    # Precompute y positions
    if event_times_s:
        t0 = event_times_s[0]
        t_last = event_times_s[-1]
    else:
        t0 = 0
        t_last = 0

    def y_of_row(r: object) -> float:
        # With time increasing top → bottom:
        # - centroid-in (origins / departures) at the top
        # - centroid-out (destinations / arrivals) at the bottom
        # Multiple centroid-in rows: r may be "CENTROID_IN" or ("CENTROID_IN", bin_idx)
        if r == "CENTROID_IN" or (
            isinstance(r, tuple) and len(r) == 2 and r[0] == "CENTROID_IN"
        ):
            if r == "CENTROID_IN":
                k = 0
            else:
                k = int(rows.index(r))
            return float(top + k * row_step)
        if r == CENTROID_OUT:
            if y_mode == "equal":
                return float(top + (len(rows) - 1) * row_step)
            # scaled
            base = top + row_step
            return float(base + ((t_last - t0) / 60.0) * px_per_min + row_step)

        # event time row (seconds)
        t_s = int(r)
        if y_mode == "equal":
            # row index: 1..len(event_times_s)
            # (events start at row 1 because centroid-in occupies row 0)
            k = 1 + int(np.searchsorted(event_times_s, t_s))
            return float(top + k * row_step)

        # scaled
        base = top + row_step
        return float(base + ((t_s - t0) / 60.0) * px_per_min)

    row_y = {r: y_of_row(r) for r in rows}

    # Precompute the time-label strings so we can size the left margin.
    time_label_strings: list[str] = []
    for r in rows:
        if r == "CENTROID_IN":
            lab = "centroid-in"
        elif isinstance(r, tuple) and len(r) == 2 and r[0] == "CENTROID_IN":
            bidx = int(r[1])
            lab = f"centroid-in {centroid_in_label.get(bidx, str(bidx))}"
        elif r == CENTROID_OUT:
            lab = "centroid-out"
        else:
            lab = _fmt_hms(int(r))
        time_label_strings.append(lab)

    # Estimate maximum label width (conservative) so time labels never overlap the network.
    # This remains pure-Python and deterministic (no font measurement).
    time_font_px = SVG_TIME_LABEL_FONT_PX * ui
    max_label_chars = max((len(s) for s in time_label_strings), default=0)
    est_label_width = SVG_TIME_LABEL_CHAR_EM * time_font_px * max_label_chars

    # --- Compute jitter for ARR/DEP event node collisions before determining left margin ---
    # Group event nodes by (stop_idx, time_s, kind) for jitter.
    # Jitter is applied on top of the ARR/DEP horizontal offsets, so multiple ARR (or DEP)
    # nodes at the same (stop, time) remain readable.
    groups: dict[tuple[int, int, int], list[int]] = {}
    for i in range(g.num_nodes):
        kind = int(g.node_kind[i])
        if kind not in (NODE_KIND_EVENT_ARR, NODE_KIND_EVENT_DEP):
            continue
        s = int(g.node_stop_index[i])
        t = int(g.node_time_s[i])
        groups.setdefault((s, t, kind), []).append(i)

    jitter: np.ndarray = np.zeros((g.num_nodes,), dtype=float)
    for (s, t, kind), idxs in groups.items():
        n = len(idxs)
        if n <= 1:
            continue

        def _line_key(node_i: int) -> str:
            ti = int(node_trip_index[node_i])
            if trip_line_ref is not None and 0 <= ti < len(trip_line_ref):
                lr = trip_line_ref[ti]
                if lr is not None and str(lr):
                    return str(lr)
            if trip_ids is not None and 0 <= ti < len(trip_ids):
                return str(trip_ids[ti])
            return str(ti)

        # Spread symmetrically around 0.
        # Sort by line (fallback trip) so different lines are separated clearly.
        ordered = [node_i for _, node_i in sorted(((_line_key(i), i) for i in idxs), key=lambda z: z[0])]
        step = SVG_JITTER_STEP * sx
        for k, node_i in enumerate(ordered):
            jitter[node_i] = (k - (n - 1) / 2.0) * step

    # --- Compute left margin based on the leftmost network extent (including DEP offset) and jitter ---
    # We must account for:
    # - centroid squares extending left by half-size
    # - event circles shifted by ARR/DEP offsets plus possible jitter and their radii
    max_jitter_abs = float(np.max(np.abs(jitter))) if jitter.size else 0.0

    min_dx_centroid = -SVG_CENTROID_HALF_SIZE * ui

    # Event-node left extent: DEP is typically the left-shifted one (negative offset).
    min_dx_event_arr = (SVG_ARR_X_OFFSET * sx) - max_jitter_abs - (SVG_EVENT_RADIUS_ARR * ui)
    min_dx_event_dep = (SVG_DEP_X_OFFSET * sx) - max_jitter_abs - (SVG_EVENT_RADIUS_DEP * ui)
    min_dx_event = float(min(min_dx_event_arr, min_dx_event_dep))

    # Overall leftmost extent relative to the stop column center.
    min_dx = float(min(0.0, min_dx_centroid, min_dx_event))

    # Place the time-label anchor in a left gutter so labels never overlap the network.
    # (Labels are right-aligned with text-anchor='end', so they extend left of time_label_x.)
    extra_pad = SVG_TIME_LABEL_EXTRA_PAD_MULT * time_label_pad
    left = max(base_left, min_left_gutter + est_label_width + extra_pad - min_dx)
    time_label_x = left - time_label_pad

    # -----------------------------
    # Compute per-node (x,y), including jitter for event collisions
    # -----------------------------
    def x_of_stop(stop_idx: int) -> float:
        return float(left + stop_idx * x_step)

    nx = np.zeros((g.num_nodes,), dtype=float)
    ny = np.zeros((g.num_nodes,), dtype=float)

    for i in range(g.num_nodes):
        s = int(g.node_stop_index[i])
        kind = int(g.node_kind[i])
        # centroid-out is kind==3, centroid-in is kind==0
        if kind == NODE_KIND_CENTROID_OUT:
            rkey: object = CENTROID_OUT
        elif kind == NODE_KIND_CENTROID_IN:
            if time_bins is not None and len(time_bins) > 0:
                tb_idx = _centroid_in_time_bin_index(
                    node_idx=i,
                    node_kind=kind,
                    stop_index=s,
                    num_time_bins=len(time_bins),
                )
                rkey = ("CENTROID_IN", int(tb_idx) if tb_idx is not None else 0)
            else:
                rkey = "CENTROID_IN"
        else:
            # event node
            t = int(g.node_time_s[i])
            rkey = t
        # Event nodes should not be vertically aligned with centroids.
        # We draw ARR nodes to the *left* of the stop's centroid column and DEP nodes to the *right*.
        # This is a pure visualization choice and does not affect the graph semantics.
        offset = 0.0
        if kind == NODE_KIND_EVENT_ARR:  # ARR
            offset = SVG_ARR_X_OFFSET * sx
        elif kind == NODE_KIND_EVENT_DEP:  # DEP
            offset = SVG_DEP_X_OFFSET * sx
        nx[i] = x_of_stop(s) + (jitter[i] if kind in (NODE_KIND_EVENT_ARR, NODE_KIND_EVENT_DEP) else 0.0) + offset
        ny[i] = row_y.get(rkey, float(top))

    # -----------------------------
    # SVG dimensions
    # -----------------------------
    x_right = x_of_stop(max(0, stop_count - 1))
    y_bottom = row_y[rows[-1]]

    width = int(x_right + 90 * sx)
    height = int(y_bottom + bottom_label_pad + 30)

    y_stop_labels = float(height - 20)

    # -----------------------------
    # Grid lines + axis labels
    # -----------------------------
    # Vertical dotted lines for each stop
    grid_elems: list[str] = []
    for s in range(stop_count):
        x = x_of_stop(s)
        grid_elems.append(
            f"<line x1='{x:.1f}' y1='{row_y[CENTROID_OUT]:.1f}' x2='{x:.1f}' y2='{y_bottom:.1f}' "
            f"stroke='{SVG_GRID_STROKE}' stroke-width='1' stroke-dasharray='{SVG_GRID_DASH}' />"
        )

    # Horizontal dotted lines for each row (including centroids)
    for r in rows:
        y = row_y[r]
        grid_elems.append(
            f"<line x1='{x_of_stop(0):.1f}' y1='{y:.1f}' x2='{x_right:.1f}' y2='{y:.1f}' "
            f"stroke='{SVG_GRID_STROKE}' stroke-width='1' stroke-dasharray='{SVG_GRID_DASH}' />"
        )

    # Time labels at left (precomputed; guaranteed to stay in the left margin)
    time_labels: list[str] = []
    for r, lab in zip(rows, time_label_strings):
        y = row_y[r]
        time_labels.append(
            f"<text x='{time_label_x:.1f}' y='{y:.1f}' text-anchor='end' dominant-baseline='middle' "
            f"font-size='{SVG_TIME_LABEL_FONT_PX * ui:.3g}' fill='{SVG_TEXT_FILL}'>{html.escape(lab)}</text>"
        )

    # Stop labels at bottom
    stop_labels: list[str] = []
    for s in range(stop_count):
        x = x_of_stop(s)
        stop_labels.append(
            f"<text x='{x:.1f}' y='{y_stop_labels:.1f}' text-anchor='middle' dominant-baseline='middle' "
            f"font-size='{SVG_TIME_LABEL_FONT_PX * ui:.3g}' fill='{SVG_TEXT_FILL}'>{html.escape(stop_name(s))}</text>"
        )

    # -----------------------------
    # Draw links (use precomputed node positions)
    # -----------------------------
    link_elems: list[str] = []
    for e in range(g.num_links):
        t = int(g.tail[e])
        h = int(g.head[e])
        lt = int(g.link_type[e])
        dotted = (lt == LINK_TYPE_ACCESS or lt == LINK_TYPE_EGRESS)  # access/egress
        dash = f"stroke-dasharray='{SVG_LINK_DASH_ACCESS_EGRESS}'" if dotted else ""
        mx = (nx[t] + nx[h]) / 2
        my = (ny[t] + ny[h]) / 2
        group = "<g>"
        group += (
            f"<line x1='{nx[t]:.1f}' y1='{ny[t]:.1f}' x2='{nx[h]:.1f}' y2='{ny[h]:.1f}' "
            f"stroke='{SVG_LINK_STROKE}' stroke-width='{1.2 * ui:.3g}' {dash} />"
        )
        if link_flow is not None:
            fval = float(link_flow[e])
            flab = f"{fval:.6g}"
            dx = nx[h] - nx[t]
            dy = ny[h] - ny[t]
            L = max(1e-6, (dx * dx + dy * dy) ** 0.5)
            ox = -dy / L * 8
            oy = dx / L * 8
            tx = mx + ox
            ty = my + oy
            # Compose hover title
            if link_cost is not None:
                ctext = _fmt_min(float(link_cost[e]))
            else:
                ctext = "—"
            group += (
                f"<title>link={e}, type={_LINK_TYPE.get(lt, lt)}, flow={flab}, cost_min={ctext}</title>"
            )
            group += (
                f"<text x='{tx:.1f}' y='{ty:.1f}' font-size='{9 * ui:.3g}' text-anchor='middle' dominant-baseline='middle' "
                f"fill='{SVG_TEXT_FILL}' style='paint-order:stroke;stroke:#fff;stroke-width:{3 * ui:.3g}px;stroke-linejoin:round'>{flab}</text>"
            )
        group += "</g>"
        link_elems.append(group)

    # -----------------------------
    # Draw nodes
    # -----------------------------
    node_elems: list[str] = []

    for i in range(g.num_nodes):
        kind = int(g.node_kind[i])
        x = nx[i]
        y = ny[i]
        stop_idx = int(g.node_stop_index[i])
        stop_id = stop_ids[stop_idx] if (stop_ids is not None and 0 <= stop_idx < len(stop_ids)) else str(stop_idx)
        stop_name_val = ""
        if node_stop_names is not None:
            stop_name_val = str(node_stop_names[i] or "")
        elif stop_names is not None:
            stop_name_val = stop_names.get(str(stop_id), "")

        if kind in (NODE_KIND_EVENT_ARR, NODE_KIND_EVENT_DEP):
            ti = int(node_trip_index[i])
            trip = (
                trip_ids[ti]
                if (trip_ids is not None and 0 <= ti < len(trip_ids))
                else (str(ti) if ti >= 0 else "")
            )
            line = trip_line_ref[ti] if (trip_line_ref is not None and 0 <= ti < len(trip_line_ref)) else None

            # Visible label: line_ref if available else trip_id
            visible = line or trip

            # Tooltip: kind, stop_id, stop_name, time, trip, line
            t_s = int(g.node_time_s[i])
            parts: list[str] = []
            parts.append(_NODE_KIND.get(kind, str(kind)))
            parts.append(f"stop_id={stop_id}")
            if stop_name_val:
                parts.append(f"stop_name={stop_name_val}")
            if t_s >= 0:
                parts.append(f"time={_fmt_hms(t_s)}")
            if trip:
                parts.append(f"trip={trip}")
            if line:
                parts.append(f"line={line}")
            tooltip = " · ".join(parts)

            # Differentiate arr/dep by radius
            radius = (SVG_EVENT_RADIUS_ARR if kind == NODE_KIND_EVENT_ARR else SVG_EVENT_RADIUS_DEP) * ui
            group = (
                f"<g>"
                f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{radius:.3g}' fill='#fff' stroke='{SVG_NODE_STROKE}' stroke-width='{1.2 * ui:.3g}' />"
            )
            if tooltip:
                group += f"<title>{html.escape(tooltip)}</title>"
            if visible:
                # label to the right of the node, with white stroke for readability
                group += (
                    f"<text x='{x + SVG_EVENT_LABEL_DX * sx:.1f}' y='{y:.1f}' font-size='{10 * ui:.3g}' fill='{SVG_TEXT_FILL}' dominant-baseline='middle' "
                    f"style='paint-order:stroke;stroke:#fff;stroke-width:{3 * ui:.3g}px;stroke-linejoin:round'>"
                    f"{html.escape(visible)}</text>"
                )
            group += "</g>"
            node_elems.append(group)
        else:
            # centroid in/out as squares
            tooltip = _NODE_KIND.get(kind, str(kind))
            tooltip += f" · stop_id={stop_id}"
            if stop_name_val:
                tooltip += f" · stop_name={stop_name_val}"
            # For centroid-in nodes, add interval label if available
            if kind == NODE_KIND_CENTROID_IN and time_bins is not None and len(time_bins) > 0:
                tb_idx = _centroid_in_time_bin_index(
                    node_idx=i,
                    node_kind=kind,
                    stop_index=stop_idx,
                    num_time_bins=len(time_bins),
                )
                if tb_idx is not None and 0 <= tb_idx < len(time_bins):
                    interval_label = str(time_bins[tb_idx].get("label", ""))
                    if interval_label:
                        tooltip += f" · {interval_label}"
            node_elems.append(
                f"<rect x='{x - SVG_CENTROID_HALF_SIZE * ui:.1f}' y='{y - SVG_CENTROID_HALF_SIZE * ui:.1f}' width='{2 * SVG_CENTROID_HALF_SIZE * ui:.3g}' height='{2 * SVG_CENTROID_HALF_SIZE * ui:.3g}' fill='#fff' stroke='{SVG_NODE_STROKE}' stroke-width='{1.2 * ui:.3g}' >"
                + (f"<title>{html.escape(tooltip)}</title>" if tooltip else "")
                + "</rect>"
            )

    svg = (
        f"<h2>{html.escape(title)}</h2>"
        "<div class='svgbox'>"
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>"
        + "".join(grid_elems)
        + "".join(time_labels)
        + "".join(stop_labels)
        + "".join(link_elems)
        + "".join(node_elems)
        + "</svg></div>"
    )
    return svg

# --------------------------------------------------------------------------------------
# Stop name enrichment
# --------------------------------------------------------------------------------------

def _stop_names(scenario: Any | None) -> Mapping[str, str] | None:
    """Return mapping stop_id -> stop_name if scenario provides it, else None."""
    if scenario is None:
        return None
    stops = getattr(scenario, "stops", None)
    if stops is None:
        return None
    m: dict[str, str] = {}
    if isinstance(stops, dict):
        it = stops.values()
    else:
        it = stops
    for s in it:
        sid = getattr(s, "stop_id", None)
        name = getattr(s, "name", None)
        if sid is None:
            continue
        m[str(sid)] = "" if name is None else str(name)
    return m
# --------------------------------------------------------------------------------------
# Stop name enrichment
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Path flow decomposition (post-processing)
# --------------------------------------------------------------------------------------


def _render_paths_table(
    *,
    g: _GraphNP,
    stop_ids: tuple[str, ...] | None,
    trip_ids: tuple[str, ...] | None,
    trip_line_ref: tuple[str | None, ...] | None,
    node_trip_index: np.ndarray,
    link_flow: np.ndarray,
    link_cost: np.ndarray | None = None,
    stop_names: Mapping[str, str] | None = None,
    node_stop_names: np.ndarray | None = None,
    time_bins: list[dict[str, Any]] | None = None,
    eps: float = 1e-9,
    max_paths: int = 2000,
) -> str:
    """Render a table of source→sink paths carrying positive flow.

    This section is computed by *decomposing the realized link flows* on the DAG
    into a (non-unique) set of paths with associated path flows.

    Notes
    -----
    - The decomposition is not unique.
    - For large graphs / highly dispersed flows, the number of paths can be large;
      `max_paths` acts as a safety cap for the report.
    """

    paths = _decompose_link_flows_into_paths(
        g=g,
        link_flow=link_flow,
        eps=eps,
        max_paths=max_paths,
    )

    body: list[str] = ["<h2>Paths (flow decomposition)</h2>"]
    body.append(
        "<p class='muted'>This section is a <b>post-processing</b> of the link flows: "
        "it decomposes the realized DAG link flows into a set of source→sink paths. "
        "The decomposition is <b>not unique</b>. Only paths with positive extracted flow are shown.</p>"
    )

    if not paths:
        body.append("<p class='muted'>No positive-flow paths could be extracted (flow is all zero or below tolerance).</p>")
        return "\n".join(body)

    has_cost = link_cost is not None

    # Build rows
    # We do NOT aggregate costs here: the report must display exactly the values used by the model.
    # When link_cost is provided (already computed by the model), we display the per-link costs along the path.
    rows: list[tuple[float, str, str, str]] = []  # (flow, nodes_str, links_str, link_costs_str)
    for path_links, f in paths:
        if f <= eps:
            continue

        node_seq = _path_nodes_from_links(g, path_links)
        nodes_str = _format_path_nodes(
            g=g,
            node_seq=node_seq,
            stop_ids=stop_ids,
            trip_ids=trip_ids,
            trip_line_ref=trip_line_ref,
            node_trip_index=node_trip_index,
            stop_names=stop_names,
            node_stop_names=node_stop_names,
            time_bins=time_bins,
        )
        links_str = _format_path_links(g=g, link_seq=path_links)

        if has_cost:
            # Report the exact per-link costs used by the model (no summation here).
            costs = [float(link_cost[int(e)]) for e in path_links]
            link_costs_str = html.escape(" → ".join(_fmt_min(c) for c in costs))
        else:
            link_costs_str = "—"

        rows.append((float(f), nodes_str, links_str, link_costs_str))

    # Sort: highest flow first (deterministic ordering)
    rows.sort(key=lambda r: (-r[0],))

    if len(rows) >= max_paths:
        body.append(
            f"<p class='muted'><b>Note:</b> showing at most {max_paths} paths (safety cap). "
            "If you need more, increase <code>max_paths</code> in the report generator.</p>"
        )

    body.append(
        "<table><thead><tr>"
        "<th>#</th><th>path flow</th><th>per-link generalized costs (min)</th><th>path (nodes)</th><th>path (links)</th>"
        "</tr></thead><tbody>"
    )

    for k, (f, nodes_str, links_str, link_costs_str) in enumerate(rows, start=1):
        body.append(
            "<tr>"
            f"<td><code>{k}</code></td>"
            f"<td>{html.escape(f'{f:.6g}')}</td>"
            f"<td><code>{link_costs_str}</code></td>"
            f"<td>{nodes_str}</td>"
            f"<td>{links_str}</td>"
            "</tr>"
        )

    body.append("</tbody></table>")
    return "\n".join(body)


def _decompose_link_flows_into_paths(
    *,
    g: _GraphNP,
    link_flow: np.ndarray,
    eps: float,
    max_paths: int,
) -> list[tuple[list[int], float]]:
    """Decompose positive link flows into a list of (path_links, path_flow).

    Algorithm: greedy residual flow decomposition on a DAG.

    - Residual r_e starts as link_flow[e].
    - Repeatedly pick a source node (centroid-in) with residual outflow > eps.
    - Follow deterministic positive-residual outgoing links until reaching a sink (centroid-out).
    - Extract the bottleneck flow on that path and subtract it.

    Returns a finite list; decomposition is not unique.
    """
    r = np.asarray(link_flow, dtype=float).copy().reshape(-1)
    if r.shape[0] != g.num_links:
        return []

    # Build outgoing adjacency lists (pure python lists for simplicity)
    out_links: list[list[int]] = [[] for _ in range(g.num_nodes)]
    for e in range(g.num_links):
        out_links[int(g.tail[e])].append(int(e))

    # Deterministic order: increasing link index
    for lst in out_links:
        lst.sort()

    sources = [i for i in range(g.num_nodes) if int(g.node_kind[i]) == NODE_KIND_CENTROID_IN]
    sinks_set = {i for i in range(g.num_nodes) if int(g.node_kind[i]) == NODE_KIND_CENTROID_OUT}

    def residual_out(i: int) -> float:
        s = 0.0
        for e in out_links[i]:
            if r[e] > eps:
                s += r[e]
        return s

    paths: list[tuple[list[int], float]] = []

    # Safety: avoid infinite loops even if the input is inconsistent
    it_guard = 0
    max_iters = max_paths * max(10, g.num_nodes)

    while it_guard < max_iters and len(paths) < max_paths:
        it_guard += 1

        # Pick next source with residual outflow
        src = None
        for i in sources:
            if residual_out(i) > eps:
                src = i
                break
        if src is None:
            break

        # Walk from src to a sink following positive residual links
        cur = int(src)
        path_links: list[int] = []
        visited_nodes: set[int] = set()

        while cur not in sinks_set:
            if cur in visited_nodes:
                # Should not happen in a DAG with monotone times, but guard anyway
                break
            visited_nodes.add(cur)

            nxt_e = None
            for e in out_links[cur]:
                if r[e] > eps:
                    nxt_e = int(e)
                    break
            if nxt_e is None:
                # Dead end: cannot reach a sink; stop this attempt.
                path_links = []
                break

            path_links.append(nxt_e)
            cur = int(g.head[nxt_e])

        if not path_links:
            # Cannot extract a path from this source with the current residuals.
            # Zero-out tiny outgoing residuals to avoid repeated dead-ends.
            for e in out_links[int(src)]:
                if 0.0 < r[e] <= eps:
                    r[e] = 0.0
            continue

        # Extract bottleneck residual on the path
        f = float(min(r[e] for e in path_links))
        if f <= eps:
            # Nothing meaningful
            for e in path_links:
                if 0.0 < r[e] <= eps:
                    r[e] = 0.0
            continue

        # Subtract
        for e in path_links:
            r[e] -= f
            if r[e] < eps:
                r[e] = 0.0

        paths.append((path_links, f))

    return paths


def _path_nodes_from_links(g: _GraphNP, link_seq: Sequence[int]) -> list[int]:
    """Return node sequence implied by an ordered link sequence."""
    if not link_seq:
        return []
    nodes = [int(g.tail[int(link_seq[0])])]
    for e in link_seq:
        nodes.append(int(g.head[int(e)]))
    return nodes


def _format_path_links(*, g: _GraphNP, link_seq: Sequence[int]) -> str:
    parts: list[str] = []
    for e in link_seq:
        lt = int(g.link_type[int(e)])
        name = _LINK_TYPE.get(lt, str(lt))
        parts.append(f"{int(e)}:{name}")
    return html.escape(" → ".join(parts))


def _format_path_nodes(
    *,
    g: _GraphNP,
    node_seq: Sequence[int],
    stop_ids: tuple[str, ...] | None,
    trip_ids: tuple[str, ...] | None,
    trip_line_ref: tuple[str | None, ...] | None,
    node_trip_index: np.ndarray,
    stop_names: Mapping[str, str] | None = None,
    node_stop_names: np.ndarray | None = None,
    time_bins: list[dict[str, Any]] | None = None,
) -> str:
    """Readable node path string with stop/time/kind and line/trip when available."""
    chunks: list[str] = []
    for i in node_seq:
        kind = int(g.node_kind[int(i)])
        stop_idx = int(g.node_stop_index[int(i)])
        sid = stop_ids[stop_idx] if (stop_ids is not None and 0 <= stop_idx < len(stop_ids)) else str(stop_idx)
        sname = ""
        if node_stop_names is not None:
            sname = str(node_stop_names[int(i)] or "")
        elif stop_names is not None:
            sname = stop_names.get(str(sid), "")

        t_s = int(g.node_time_s[int(i)])
        t_lab = _fmt_hms(t_s) if t_s >= 0 else "CENTROID"
        k_lab = _NODE_KIND.get(kind, str(kind))

        extra = ""
        if kind in (NODE_KIND_EVENT_ARR, NODE_KIND_EVENT_DEP):
            ti = int(node_trip_index[int(i)])
            trip = trip_ids[ti] if (trip_ids is not None and 0 <= ti < len(trip_ids)) else ""
            line = trip_line_ref[ti] if (trip_line_ref is not None and 0 <= ti < len(trip_line_ref)) else None
            if line:
                extra = f"[{line}]"
            elif trip:
                extra = f"[{trip}]"
        elif kind == NODE_KIND_CENTROID_IN and time_bins is not None and len(time_bins) > 0:
            tb_idx = _centroid_in_time_bin_index(
                node_idx=int(i),
                node_kind=kind,
                stop_index=stop_idx,
                num_time_bins=len(time_bins),
            )
            if tb_idx is not None and 0 <= tb_idx < len(time_bins):
                t_lab = f"{_NODE_KIND.get(kind, str(kind))} {time_bins[tb_idx].get('label','')}"
            else:
                t_lab = _NODE_KIND.get(kind, str(kind))
        elif kind == NODE_KIND_CENTROID_OUT:
            t_lab = _NODE_KIND.get(kind, str(kind))

        if sname:
            chunks.append(f"{int(i)}:{k_lab}@{sid}({sname})/{t_lab}{extra}")
        else:
            chunks.append(f"{int(i)}:{k_lab}@{sid}/{t_lab}{extra}")

    # Use <code> for readability in HTML tables
    return "<code>" + html.escape(" → ".join(chunks)) + "</code>"