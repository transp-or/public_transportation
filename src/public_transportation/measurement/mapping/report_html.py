from __future__ import annotations

from pathlib import Path
import html
import numpy as np

from public_transportation.assignment.id_manager import AssignmentIDManager
from .spec import MappingInfo


def write_mapping_report_html(
    *,
    info: MappingInfo,
    id_manager: AssignmentIDManager,
    assignment_link_flow: np.ndarray,
    output_path: str | Path,
) -> None:
    """Write an HTML report describing how measurements were mapped to assignment objects."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    stop_id = id_manager.stop_id
    trip_id = id_manager.trip_id

    def node_label(n: int) -> str:
        si = int(id_manager.node_stop_index[n])
        sid = stop_id[si] if 0 <= si < len(stop_id) else f"stop_index={si}"

        t = int(id_manager.node_time_s[n])
        hh = t // 3600
        mm = (t % 3600) // 60
        ss = t % 60

        ti = int(id_manager.node_trip_index[n])
        tid = trip_id[ti] if 0 <= ti < len(trip_id) else f"trip_index={ti}"
        return f"{sid} / {tid} @ {hh:02d}:{mm:02d}:{ss:02d} (node={n})"

    def link_label(link: int) -> str:
        u = int(id_manager.link_tail[link])
        v = int(id_manager.link_head[link])
        lt = int(id_manager.link_type[link])
        return f"link {link}: tail={u}, head={v}, type={lt}, flow={assignment_link_flow[link]:.6g}"

    css = (
        "body{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;}"
        "table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #ddd;padding:6px;vertical-align:top;}"
        "th{background:#f6f6f6;}"
        "code{background:#f3f3f3;padding:1px 3px;border-radius:4px;}"
    )

    rows: list[str] = []
    rows.append("<!doctype html>")
    rows.append("<html><head><meta charset='utf-8'>")
    rows.append("<title>Measurement ↔ Assignment Mapping Report</title>")
    rows.append(f"<style>{css}</style>")
    rows.append("</head><body>")

    rows.append("<h1>Measurement ↔ Assignment Mapping Report</h1>")
    rows.append(
        f"<p><b>Assignment fingerprint:</b> <code>{html.escape(info.fingerprint)}</code></p>"
    )
    rows.append(f"<p><b>Number of mapped measurements:</b> {len(info.entries)}</p>")

    rows.append("<table>")
    rows.append(
        "<tr>"
        "<th>#</th>"
        "<th>measurement</th>"
        "<th>observed</th>"
        "<th>predicted</th>"
        "<th>matched event node</th>"
        "<th>contributing links</th>"
        "</tr>"
    )

    for e in info.entries:
        meas = (
            f"type=<code>{html.escape(e.measurement_type)}</code>, "
            f"stop=<code>{html.escape(e.stop_id)}</code>, "
            f"time=<code>{html.escape(e.time_hms)}</code>, "
            f"trip_id=<code>{html.escape(str(e.trip_id))}</code>, "
            f"line_id=<code>{html.escape(str(e.line_id))}</code>, "
            f"method=<code>{html.escape(e.method_id)}</code>"
        )

        if e.matched_link_indices is None:
            links_html = "<i>(link list not stored; enable include_link_lists_for_report)</i>"
        else:
            links_html = "<br/>".join(html.escape(link_label(link)) for link in e.matched_link_indices)

        rows.append("<tr>")
        rows.append(f"<td>{e.row_index}</td>")
        rows.append(f"<td>{meas}</td>")
        rows.append(f"<td>{e.observed_value:.6g}</td>")
        rows.append(f"<td>{e.predicted_value:.6g}</td>")
        rows.append(f"<td>{html.escape(node_label(e.matched_event_node))}</td>")
        rows.append(f"<td>{links_html}</td>")
        rows.append("</tr>")

    rows.append("</table>")
    rows.append("</body></html>")

    p.write_text("\n".join(rows), encoding="utf-8")