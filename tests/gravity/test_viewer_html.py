from __future__ import annotations

from pathlib import Path


VIEWER = (
    Path(__file__).resolve().parents[2] / "tools" / "gravity_viewer" / "index.html"
)


def test_viewer_exposes_stop_and_od_pair_interaction_controls() -> None:
    source = VIEWER.read_text(encoding="utf-8")
    for text in (
        'id="stops-mode"',
        'id="od-mode"',
        'id="stop-search"',
        'id="od-origin"',
        'id="od-destination"',
        'id="od-list"',
        'id="reset-selection"',
        "od_map.csv",
        "stop_summary.csv",
        "chooseOdStop",
        "selectOd",
    ):
        assert text in source


def test_viewer_keeps_network_map_and_tables_when_derived_files_are_missing() -> None:
    source = VIEWER.read_text(encoding="utf-8")
    assert "Interactive summaries unavailable" in source
    assert "The validated tables and network map remain available" in source
    assert "bundle tables remain available without map tiles" in source
