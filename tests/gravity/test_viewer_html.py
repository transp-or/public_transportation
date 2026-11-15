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
        'id="route-sections"',
        'id="section-lines"',
        'id="section-bin"',
        'id="section-id"',
        'id="section-page-next"',
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
    assert "flow-kind" in source
    assert "route-assignment flow" in viewer_source
    assert "query_section_od" in viewer_source or "route_sections" in viewer_source
    assert "section_loads" in viewer_source
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
    assert "Querying only the selected measurement interval" in viewer_source
    assert "summary.available && summary.query_mode === 'local_service'" in viewer_source
