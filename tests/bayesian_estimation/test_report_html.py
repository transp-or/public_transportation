# tests/variational/test_report_html.py
from __future__ import annotations

from pathlib import Path

import pytest

from public_transportation.estimation.bayesian.report_html import (
    _build_html_document,
    _escape,
    _render_bullet_list,
    _render_figure_placeholder,
    _render_metadata,
    _render_recommendations,
    _render_section,
    _render_table,
    generate_vi_report_html,
)


def _minimal_report_data() -> dict:
    return {
        "title": "VI Diagnostic Report",
        "subtitle": "Synthetic test run",
        "metadata": [("Model", "Test model"), ("Iterations", 100)],
        "executive_summary": ["Converged.", "No major numerical issue."],
        "recommendations": [
            {
                "severity": "info",
                "title": "Continue",
                "message": "The run appears stable.",
            }
        ],
        "sections": [
            {
                "id": "elbo",
                "title": "ELBO diagnostics",
                "summary": "The ELBO increases.",
                "interpretation": ["Trend is stable."],
                "tables": [
                    {
                        "title": "ELBO table",
                        "columns": ["Metric", "Value"],
                        "rows": [["Final ELBO", "-123.4"]],
                    }
                ],
                "figures": [
                    {
                        "title": "ELBO trace",
                        "description": "Trace of the evidence lower bound.",
                        "kind": "line",
                        "file": "figures/elbo.png",
                    }
                ],
            }
        ],
    }


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def test_generate_vi_report_html_writes_file_and_creates_parent_directory(tmp_path: Path):
    output_file = tmp_path / "nested" / "report.html"

    generate_vi_report_html(_minimal_report_data(), output_file)

    assert output_file.exists()
    html = output_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert "<title>VI Diagnostic Report</title>" in html
    assert "Synthetic test run" in html
    assert "ELBO diagnostics" in html


def test_generate_vi_report_html_accepts_string_path(tmp_path: Path):
    output_file = tmp_path / "report.html"

    generate_vi_report_html(_minimal_report_data(), str(output_file))

    assert output_file.exists()
    assert "VI Diagnostic Report" in output_file.read_text(encoding="utf-8")


# ---------------------------------------------------------------------
# Full document
# ---------------------------------------------------------------------


def test_build_html_document_contains_main_sections():
    html_text = _build_html_document(_minimal_report_data())

    assert html_text.startswith("<!DOCTYPE html>")
    assert '<html lang="en">' in html_text
    assert "<h1>VI Diagnostic Report</h1>" in html_text
    assert "<h2>Run metadata</h2>" in html_text
    assert "<h2>Executive summary</h2>" in html_text
    assert "<h2>Recommendations</h2>" in html_text
    assert "<h2>ELBO diagnostics</h2>" in html_text


def test_build_html_document_uses_defaults_for_missing_top_level_fields():
    html_text = _build_html_document({})

    assert "<title>VI Diagnostic Report</title>" in html_text
    assert "<h1>VI Diagnostic Report</h1>" in html_text
    assert "No information available." in html_text
    assert "No recommendations available." in html_text


def test_build_html_document_escapes_title_and_subtitle():
    html_text = _build_html_document(
        {
            "title": '<script>alert("x")</script>',
            "subtitle": "A < B & C > D",
        }
    )

    assert "<script>" not in html_text
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in html_text
    assert "A &lt; B &amp; C &gt; D" in html_text


# ---------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ""),
        ("plain", "plain"),
        ("A < B", "A &lt; B"),
        ("A > B", "A &gt; B"),
        ("A & B", "A &amp; B"),
        ('A "quote"', "A &quot;quote&quot;"),
        (123, "123"),
        (3.14, "3.14"),
    ],
)
def test_escape(value, expected):
    assert _escape(value) == expected


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------


def test_render_metadata_generates_table_rows():
    html_text = _render_metadata([("Model", "M1"), ("Seed", 123)])

    assert '<table class="metadata-table">' in html_text
    assert "<th>Model</th>" in html_text
    assert "<td>M1</td>" in html_text
    assert "<th>Seed</th>" in html_text
    assert "<td>123</td>" in html_text


def test_render_metadata_escapes_keys_and_values():
    html_text = _render_metadata([("<key>", "<value>")])

    assert "<key>" not in html_text
    assert "<value>" not in html_text
    assert "&lt;key&gt;" in html_text
    assert "&lt;value&gt;" in html_text


def test_render_metadata_empty_list_returns_empty_table():
    html_text = _render_metadata([])

    assert '<table class="metadata-table">' in html_text
    assert "<tbody></tbody>" in html_text


# ---------------------------------------------------------------------
# Bullet lists
# ---------------------------------------------------------------------


def test_render_bullet_list_with_items():
    html_text = _render_bullet_list(["First", "Second"])

    assert html_text.startswith("<ul")
    assert "<li>First</li>" in html_text
    assert "<li>Second</li>" in html_text
    assert html_text.endswith("</ul>")


def test_render_bullet_list_with_css_class():
    html_text = _render_bullet_list(["First"], css_class="interpretation-list")

    assert '<ul class="interpretation-list">' in html_text


def test_render_bullet_list_escapes_items_and_css_class():
    html_text = _render_bullet_list(["A < B"], css_class='x" onclick="bad')

    assert "A &lt; B" in html_text
    assert 'onclick="bad' not in html_text
    assert "x&quot; onclick=&quot;bad" in html_text


def test_render_bullet_list_empty_returns_muted_placeholder():
    html_text = _render_bullet_list([])

    assert html_text == '<p class="muted">No information available.</p>'


# ---------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------


def test_render_recommendations_empty_returns_placeholder():
    html_text = _render_recommendations([])

    assert html_text == '<p class="muted">No recommendations available.</p>'


def test_render_recommendations_generates_cards():
    html_text = _render_recommendations(
        [
            {
                "severity": "warning",
                "title": "Check convergence",
                "message": "ELBO is noisy.",
            },
            {
                "severity": "critical",
                "title": "Rerun",
                "message": "Divergence detected.",
            },
        ]
    )

    assert '<div class="recommendations">' in html_text
    assert 'class="recommendation warning"' in html_text
    assert 'class="recommendation critical"' in html_text
    assert "Check convergence" in html_text
    assert "ELBO is noisy." in html_text
    assert "Rerun" in html_text


def test_render_recommendations_uses_defaults_for_missing_fields():
    html_text = _render_recommendations([{}])

    assert 'class="recommendation info"' in html_text
    assert "Recommendation" in html_text


def test_render_recommendations_escapes_content():
    html_text = _render_recommendations(
        [
            {
                "severity": 'warning" onclick="bad',
                "title": "<title>",
                "message": "<message>",
            }
        ]
    )

    assert "<title>" not in html_text
    assert "<message>" not in html_text
    assert "&lt;title&gt;" in html_text
    assert "&lt;message&gt;" in html_text
    assert "onclick=&quot;bad" in html_text


# ---------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------


def test_render_section_with_all_content():
    section = {
        "id": "posterior",
        "title": "Posterior diagnostics",
        "summary": "Posterior summary.",
        "interpretation": ["Looks reasonable."],
        "tables": [
            {
                "title": "Parameter table",
                "columns": ["Name", "Mean"],
                "rows": [["beta", "1.2"]],
            }
        ],
        "figures": [
            {
                "title": "Trace",
                "description": "Trace plot.",
                "kind": "line",
                "file": "trace.png",
            }
        ],
    }

    html_text = _render_section(section)

    assert 'id="posterior"' in html_text
    assert "<h2>Posterior diagnostics</h2>" in html_text
    assert "Posterior summary." in html_text
    assert "<h3>Interpretation</h3>" in html_text
    assert "Looks reasonable." in html_text
    assert "Parameter table" in html_text
    assert "Trace plot." in html_text
    assert 'src="trace.png"' in html_text


def test_render_section_uses_defaults_and_omits_optional_blocks():
    html_text = _render_section({})

    assert 'id="section"' in html_text
    assert "<h2>Section</h2>" in html_text
    assert "Interpretation" not in html_text
    assert "summary-box" not in html_text


def test_render_section_escapes_id_title_and_summary():
    html_text = _render_section(
        {
            "id": 'x" onclick="bad',
            "title": "<Title>",
            "summary": "<Summary>",
        }
    )

    assert 'onclick="bad' not in html_text
    assert "x&quot; onclick=&quot;bad" in html_text
    assert "&lt;Title&gt;" in html_text
    assert "&lt;Summary&gt;" in html_text


# ---------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------


def test_render_table_with_columns_and_rows():
    table = {
        "title": "Metrics",
        "columns": ["Metric", "Value"],
        "rows": [["ELBO", "-10.0"], ["Rhat", "1.01"]],
    }

    html_text = _render_table(table)

    assert "<h3>Metrics</h3>" in html_text
    assert '<table class="report-table">' in html_text
    assert "<th>Metric</th>" in html_text
    assert "<th>Value</th>" in html_text
    assert "<td>ELBO</td>" in html_text
    assert "<td>-10.0</td>" in html_text
    assert "<td>Rhat</td>" in html_text


def test_render_table_empty_columns_and_rows():
    html_text = _render_table({})

    assert "<h3>Table</h3>" in html_text
    assert "<thead>" in html_text
    assert "<tbody>" in html_text


def test_render_table_escapes_title_columns_and_cells():
    html_text = _render_table(
        {
            "title": "<Table>",
            "columns": ["<Column>"],
            "rows": [["<Cell>"]],
        }
    )

    assert "<Table>" not in html_text
    assert "<Column>" not in html_text
    assert "<Cell>" not in html_text
    assert "&lt;Table&gt;" in html_text
    assert "&lt;Column&gt;" in html_text
    assert "&lt;Cell&gt;" in html_text


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------


def test_render_figure_placeholder_with_file():
    figure = {
        "title": "Trace plot",
        "description": "A trace plot.",
        "kind": "line",
        "file": "figures/trace.png",
    }

    html_text = _render_figure_placeholder(figure)

    assert '<div class="figure-box">' in html_text
    assert "Trace plot" in html_text
    assert "A trace plot." in html_text
    assert "<code>line</code>" in html_text
    assert '<img src="figures/trace.png"' in html_text
    assert "Figure rendering can be added here" not in html_text


def test_render_figure_placeholder_without_file():
    html_text = _render_figure_placeholder(
        {
            "title": "Missing plot",
            "description": "No image file yet.",
            "kind": "scatter",
        }
    )

    assert "Missing plot" in html_text
    assert "No image file yet." in html_text
    assert "<code>scatter</code>" in html_text
    assert "<img" not in html_text
    assert "Figure rendering can be added here" in html_text


def test_render_figure_placeholder_uses_defaults():
    html_text = _render_figure_placeholder({})

    assert "Figure" in html_text
    assert '<div class="figure-box">' in html_text


def test_render_figure_placeholder_escapes_content_and_file_path():
    html_text = _render_figure_placeholder(
        {
            "title": '<Title "bad">',
            "description": "<Description>",
            "kind": "<kind>",
            "file": 'figures/x" onerror="bad.png',
        }
    )

    assert "<Description>" not in html_text
    assert "&lt;Description&gt;" in html_text
    assert "&lt;kind&gt;" in html_text
    assert 'onerror="bad' not in html_text
    assert "onerror=&quot;bad.png" in html_text