from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def generate_ml_report_html(
    report_data: dict[str, Any],
    output_file: str | Path,
) -> None:
    """
    Generate an HTML report from structured report data.

    Parameters
    ----------
    report_data:
        Dictionary produced by `build_ml_report_data`.
    output_file:
        Path to the HTML file to generate.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_text = _build_html_document(report_data)

    output_path.write_text(html_text, encoding="utf-8")


def _build_html_document(report_data: dict[str, Any]) -> str:
    """Build the full HTML document."""
    title = _escape(report_data.get("title", "ML Diagnostic Report"))
    subtitle = _escape(report_data.get("subtitle", ""))

    metadata_html = _render_metadata(report_data.get("metadata", []))
    executive_summary_html = _render_bullet_list(
        report_data.get("executive_summary", [])
    )
    recommendations_html = _render_recommendations(
        report_data.get("recommendations", [])
    )
    sections_html = "\n".join(
        _render_section(section) for section in report_data.get("sections", [])
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.5;
      margin: 2rem auto;
      max-width: 1100px;
      padding: 0 1.2rem;
      color: #222;
      background: #fff;
    }}

    h1, h2, h3 {{
      color: #111;
      margin-top: 1.5rem;
      margin-bottom: 0.6rem;
    }}

    h1 {{
      font-size: 2rem;
      border-bottom: 2px solid #ddd;
      padding-bottom: 0.4rem;
    }}

    h2 {{
      font-size: 1.4rem;
      border-bottom: 1px solid #e5e5e5;
      padding-bottom: 0.2rem;
    }}

    h3 {{
      font-size: 1.1rem;
    }}

    p {{
      margin: 0.5rem 0 0.8rem 0;
    }}

    ul {{
      margin-top: 0.4rem;
      margin-bottom: 1rem;
    }}

    .subtitle {{
      color: #555;
      margin-top: -0.4rem;
      margin-bottom: 1.4rem;
    }}

    .section {{
      margin-top: 2rem;
      padding-top: 0.4rem;
    }}

    .summary-box {{
      background: #f7f7f9;
      border: 1px solid #e2e2e8;
      border-radius: 8px;
      padding: 1rem 1.1rem;
      margin: 1rem 0 1.2rem 0;
    }}

    .metadata-table,
    .report-table {{
      border-collapse: collapse;
      width: 100%;
      margin: 0.8rem 0 1.2rem 0;
      font-size: 0.95rem;
    }}

    .metadata-table th,
    .metadata-table td,
    .report-table th,
    .report-table td {{
      border: 1px solid #ddd;
      padding: 0.5rem 0.6rem;
      text-align: left;
      vertical-align: top;
    }}

    .metadata-table th,
    .report-table th {{
      background: #f2f2f2;
      font-weight: 600;
    }}

    .interpretation-list li {{
      margin-bottom: 0.35rem;
    }}

    .recommendations {{
      display: grid;
      gap: 0.8rem;
      margin: 1rem 0 1.2rem 0;
    }}

    .recommendation {{
      border-radius: 8px;
      padding: 0.9rem 1rem;
      border: 1px solid #ddd;
    }}

    .recommendation h3 {{
      margin-top: 0;
      margin-bottom: 0.35rem;
      font-size: 1rem;
    }}

    .recommendation.info {{
      background: #f7fbff;
      border-color: #bcdcff;
    }}

    .recommendation.warning {{
      background: #fff8e8;
      border-color: #f0cf7a;
    }}

    .recommendation.critical {{
      background: #fff1f0;
      border-color: #e3a29b;
    }}

    .recommendation-severity {{
      display: inline-block;
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 0.4rem;
      color: #555;
    }}

    .figure-box {{
      border: 1px dashed #c8c8c8;
      border-radius: 8px;
      padding: 0.9rem 1rem;
      margin: 0.7rem 0 1rem 0;
      background: #fafafa;
    }}

    .figure-title {{
      font-weight: 600;
      margin-bottom: 0.25rem;
    }}

    .muted {{
      color: #666;
    }}

    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #f5f5f5;
      padding: 0.08rem 0.25rem;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="subtitle">{subtitle}</div>

  <div class="section">
    <h2>Run metadata</h2>
    {metadata_html}
  </div>

  <div class="section">
    <h2>Executive summary</h2>
    <div class="summary-box">
      {executive_summary_html}
    </div>
  </div>

  <div class="section">
    <h2>Recommendations</h2>
    {recommendations_html}
  </div>

  {sections_html}
</body>
</html>
"""


def _render_metadata(metadata: list[tuple[Any, Any]]) -> str:
    """Render metadata key-value pairs as a table."""
    rows = []
    for key, value in metadata:
        rows.append(
            "<tr>"
            f"<th>{_escape(key)}</th>"
            f"<td>{_escape(value)}</td>"
            "</tr>"
        )

    return (
        '<table class="metadata-table">'
        "<tbody>"
        + "".join(rows)
        + "</tbody>"
        "</table>"
    )


def _render_bullet_list(items: list[Any], css_class: str | None = None) -> str:
    """Render a list of items as an HTML unordered list."""
    if not items:
        return '<p class="muted">No information available.</p>'

    class_attr = f' class="{_escape(css_class)}"' if css_class else ""
    lis = "".join(f"<li>{_escape(item)}</li>" for item in items)
    return f"<ul{class_attr}>{lis}</ul>"


def _render_recommendations(recommendations: list[dict[str, Any]]) -> str:
    """Render recommendation cards."""
    if not recommendations:
        return '<p class="muted">No recommendations available.</p>'

    blocks = []
    for item in recommendations:
        severity = str(item.get("severity", "info"))
        title = _escape(item.get("title", "Recommendation"))
        message = _escape(item.get("message", ""))

        blocks.append(
            f'''
            <div class="recommendation {severity}">
              <div class="recommendation-severity">{_escape(severity)}</div>
              <h3>{title}</h3>
              <p>{message}</p>
            </div>
            '''
        )

    return '<div class="recommendations">' + "".join(blocks) + "</div>"


def _render_section(section: dict[str, Any]) -> str:
    """Render one report section."""
    section_id = _escape(section.get("id", "section"))
    title = _escape(section.get("title", "Section"))
    summary = section.get("summary", "")
    interpretation = section.get("interpretation", [])
    tables = section.get("tables", [])
    figures = section.get("figures", [])

    summary_html = ""
    if summary:
        summary_html = f'<div class="summary-box"><p>{_escape(summary)}</p></div>'

    interpretation_html = ""
    if interpretation:
        interpretation_html = (
            "<h3>Interpretation</h3>"
            + _render_bullet_list(interpretation, css_class="interpretation-list")
        )

    tables_html = "".join(_render_table(table) for table in tables)
    figures_html = "".join(_render_figure_placeholder(figure) for figure in figures)

    return f"""
    <div class="section" id="{section_id}">
      <h2>{title}</h2>
      {summary_html}
      {interpretation_html}
      {tables_html}
      {figures_html}
    </div>
    """


def _render_table(table: dict[str, Any]) -> str:
    """Render a generic report table."""
    title = _escape(table.get("title", "Table"))
    columns = table.get("columns", [])
    rows = table.get("rows", [])

    header_html = "".join(f"<th>{_escape(col)}</th>" for col in columns)

    body_rows = []
    for row in rows:
        body_cells = "".join(f"<td>{_escape(cell)}</td>" for cell in row)
        body_rows.append(f"<tr>{body_cells}</tr>")

    body_html = "".join(body_rows)

    return f"""
    <h3>{title}</h3>
    <table class="report-table">
      <thead>
        <tr>{header_html}</tr>
      </thead>
      <tbody>
        {body_html}
      </tbody>
    </table>
    """


def _render_figure_placeholder(figure: dict[str, Any]) -> str:
    """Render either an embedded figure or a placeholder."""
    title = _escape(figure.get("title", "Figure"))
    description = _escape(figure.get("description", ""))
    kind = _escape(figure.get("kind", ""))
    file_path = figure.get("file")

    kind_html = f'<p class="muted">Figure type: <code>{kind}</code></p>' if kind else ""

    image_html = ""
    if file_path:
        image_html = (
            f'<div><img src="{_escape(file_path)}" alt="{title}" '
            'style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px;" /></div>'
        )

    fallback_html = ""
    if not file_path:
        fallback_html = (
            '<p class="muted">Figure rendering can be added here when plot files become available.</p>'
        )

    return f"""
    <div class="figure-box">
      <div class="figure-title">{title}</div>
      <p>{description}</p>
      {kind_html}
      {image_html}
      {fallback_html}
    </div>
    """

def _escape(value: Any) -> str:
    """Escape arbitrary content for safe HTML rendering."""
    if value is None:
        return ""
    return html.escape(str(value))