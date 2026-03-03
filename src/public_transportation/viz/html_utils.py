"""
public_transportation.viz.html_utils

Small HTML helpers shared by report generators.

Design goals
------------
- Pure Python, no external deps.
- Keep report modules focused on *content*, not HTML boilerplate.
- Deterministic output (stable ordering, explicit escaping).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import html


def esc(x: object) -> str:
    """HTML-escape a value."""
    return html.escape("" if x is None else str(x))


def wrap_html(*, title: str, body: str, css: str | None = None) -> str:
    """Wrap a full HTML page with minimal styling."""
    base_css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px; }
    h1,h2,h3 { margin-top: 1.2em; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }
    th { background: #f5f5f5; text-align: left; }
    code { background: #f7f7f7; padding: 1px 4px; border-radius: 4px; }
    .muted { color: #666; }
    .pill { display:inline-block; padding: 2px 8px; border-radius: 999px; background: #eee; font-size: 12px; }
    .kpi { display:flex; gap: 14px; flex-wrap: wrap; }
    .kpi div { background:#fafafa; border:1px solid #eee; padding:10px 12px; border-radius:10px; }
    .warn { background:#fff3cd; border:1px solid #ffeeba; padding:10px 12px; border-radius:10px; }
    .ok { background:#e7f7ee; border:1px solid #c7efd7; padding:10px 12px; border-radius:10px; }
    """
    css_final = base_css + ("\n" + css if css else "")
    return (
        "<!doctype html>\n"
        "<html><head><meta charset='utf-8'>"
        f"<title>{esc(title)}</title>"
        f"<style>{css_final}</style></head><body>"
        f"{body}</body></html>"
    )


def h1(text: str) -> str:
    return f"<h1>{esc(text)}</h1>"


def h2(text: str) -> str:
    return f"<h2>{esc(text)}</h2>"


def p(text: str, *, muted: bool = False) -> str:
    cls = " class='muted'" if muted else ""
    return f"<p{cls}>{esc(text)}</p>"


def raw_p(html_fragment: str, *, muted: bool = False) -> str:
    cls = " class='muted'" if muted else ""
    return f"<p{cls}>{html_fragment}</p>"


def code(text: str) -> str:
    return f"<code>{esc(text)}</code>"


def ul(items: Iterable[str]) -> str:
    lis = "".join(f"<li>{esc(it)}</li>" for it in items)
    return f"<ul>{lis}</ul>"


def link(href: str, label: str) -> str:
    return f"<a href='{esc(href)}'>{esc(label)}</a>"


@dataclass(frozen=True, slots=True)
class KPI:
    label: str
    value: str


def kpi_row(items: Sequence[KPI]) -> str:
    blocks = []
    for it in items:
        blocks.append(
            "<div>"
            f"<div class='muted'>{esc(it.label)}</div>"
            f"<div><b>{esc(it.value)}</b></div>"
            "</div>"
        )
    return "<div class='kpi'>" + "".join(blocks) + "</div>"


def table(
    *,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    caption: str | None = None,
) -> str:
    cap = f"<p class='muted'><b>{esc(caption)}</b></p>" if caption else ""
    thead = "<thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in headers) + "</tr></thead>"
    tbody_rows = []
    for r in rows:
        tbody_rows.append("<tr>" + "".join(f"<td>{esc(x)}</td>" for x in r) + "</tr>")
    tbody = "<tbody>" + "".join(tbody_rows) + "</tbody>"
    return cap + "<table>" + thead + tbody + "</table>"


def table_html_cells(
    *,
    headers: Sequence[str],
    rows_html: Sequence[Sequence[str]],
    caption: str | None = None,
) -> str:
    """Like table(), but assumes each cell is already HTML (not escaped)."""
    cap = f"<p class='muted'><b>{esc(caption)}</b></p>" if caption else ""
    thead = "<thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in headers) + "</tr></thead>"
    tbody_rows = []
    for r in rows_html:
        tbody_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in r) + "</tr>")
    tbody = "<tbody>" + "".join(tbody_rows) + "</tbody>"
    return cap + "<table>" + thead + tbody + "</table>"