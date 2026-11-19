from __future__ import annotations

from pathlib import Path


VIEWER = Path(__file__).resolve().parents[2] / "tools" / "gravity_viewer" / "index.html"


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
        'id="measurement-od"',
        'id="measurement-od-direction"',
        'id="map-measurement-mode"',
        'id="measurement-row-select"',
        'id="measurement-row-next"',
        'id="measurement-contribution-next"',
        'id="measurement-bin"',
        'id="measurement-od-load"',
    ):
        assert text in source


def test_viewer_keeps_network_map_and_tables_when_derived_files_are_missing() -> None:
    source = VIEWER.read_text(encoding="utf-8")
    assert "Interactive summaries unavailable" in source
    assert "The validated tables and network map remain available" in source
    assert "bundle tables remain available without map tiles" in source


def test_viewer_starts_from_manifest_and_streams_selected_partitions() -> None:
    source = VIEWER.read_text(encoding="utf-8")
    viewer_source = VIEWER.with_name("viewer.js").read_text(encoding="utf-8")
    assert 'src="viewer.js"' in source
    assert "viewer_summary.json" in viewer_source
    assert "new Worker" in viewer_source
    assert "indexedDB" in viewer_source
    assert "validatePartition" in viewer_source
    assert "<h2>Demand explorer</h2>" in source
    assert 'id="flow-bin"' in source
    assert 'id="flow-origin"' in source
    assert 'id="flow-destination"' in source
    assert 'id="flow-more"' in source
    assert 'id="flow-all"' in source
    assert 'id="flow-page-next"' in source
    assert "route-assignment" not in source.lower()
    assert "route_flows" not in viewer_source
    assert "route_sections" not in viewer_source
    assert "section_loads" not in viewer_source
    assert "measurement_od_contributions" in viewer_source
    assert "refreshMeasurementOd" in viewer_source
    assert "rankBy" in viewer_source
    assert "measurementLocationIndex" in viewer_source
    assert "query_measurement_od_contributions" in source or "measurementContributionPage" in viewer_source
    assert "No measurements at this location" in viewer_source
    assert "observation_time" in viewer_source and "departure_time_bin" in viewer_source
    assert "streamMatches: true" in viewer_source
    assert "measurementContributions" in viewer_source
    assert 'id="measurement-row-index"' not in source


def test_viewer_uses_the_local_service_for_selected_measurement_pages() -> None:
    source = VIEWER.read_text(encoding="utf-8")
    viewer_source = VIEWER.with_name("viewer.js").read_text(encoding="utf-8")
    assert 'id="measurement-query-service"' in source
    assert 'value="http://127.0.0.1:8765"' in source
    assert "queryLocalMeasurementPage" in viewer_source
    assert "measurement_index" in viewer_source
    assert "/api/measurement-od-contributions" in viewer_source
    assert "Local query service fit or mapping identity differs" in viewer_source
    assert "Still processing the selected contribution data." in viewer_source
    assert 'id="measurement-od-status" class="status" role="status"' in source
    assert "status.classList.add('is-loading')" in viewer_source
    assert "summary.available && summary.query_mode === 'local_service'" in viewer_source
    assert "summary.reverse_query_available === false" in viewer_source
    assert "Reverse OD-to-measurement contributions are unavailable" in viewer_source


def test_quick_guide_is_collapsible_linked_and_matches_visible_controls() -> None:
    source = VIEWER.read_text(encoding="utf-8")
    guide = source.split('<details class="panel quick-guide">', 1)[1].split("</details>", 1)[0]
    assert "<summary>Quick guide</summary>" in guide
    for target in ("summary", "network-map", "flows", "measurement-od"):
        assert f'href="#{target}"' in guide
        assert f'id="{target}"' in source
    for label in (
        "Select a viewer-bundle directory",
        "bundle_manifest.json",
        "Departure time interval",
        "Load selected partition",
        "Load 50 more",
        "Load all matching rows",
        "Measurement row",
        "Previous rows",
        "Next rows",
        "Observed count",
        "Modeled mean",
        "What do you want to look up?",
        "Measured row → contributing OD pairs",
        "Load OD contributors",
        "Rows per page",
        "Previous contributors",
        "Next contributors",
        "OD pair → affected measured rows",
        "OD departure interval",
        "Load affected measurements",
    ):
        assert label in guide
    assert "query service" not in guide.lower()
    assert "terminal" not in guide.lower()
    assert "server" not in guide.lower()
    assert "By Michel Bierlaire · Implemented 8 September 2026" in source
    assert source.index("<h1>Gravity Viewer Bundle</h1>") < source.index("By Michel Bierlaire")
    assert source.index("By Michel Bierlaire") < source.index("<main>")
