from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


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
        "selectOd",
        'id="measurement-od"',
        'id="measurement-od-direction"',
        'id="map-measurement-mode"',
        'id="measurement-row-select"',
        'id="measurement-row-next"',
        'id="measurement-contribution-next"',
        'id="map-reset-view"',
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
    assert 'id="route-path-browser"' in source
    assert 'id="route-path-threshold"' in source
    assert 'id="route-path-load"' in source
    assert 'id="route-path-service"' in source
    assert 'value="1"' in source
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
    assert "initialMapBounds = L.latLngBounds(points).pad(0.08)" in viewer_source
    assert "viewer.map.fitBounds(viewer.initialMapBounds)" in viewer_source
    assert "Reset map view" in source
    assert "observation_time" in viewer_source and "departure_time_bin" in viewer_source
    assert "streamMatches: true" in viewer_source
    assert "measurementContributions" in viewer_source
    assert "updateRoutePathBrowser" in viewer_source
    assert "browser.classList.toggle('hidden', !selected)" in viewer_source
    assert "new URL('/api/route-paths'" in viewer_source
    assert "minimum_probability" in viewer_source
    assert "route_paths_available" in viewer_source
    assert "route-path-load" in viewer_source
    assert 'id="measurement-row-index"' not in source


def test_network_map_od_selection_requires_an_explicit_endpoint_action() -> None:
    source = VIEWER.read_text(encoding="utf-8")
    viewer_source = VIEWER.with_name("viewer.js").read_text(encoding="utf-8")
    popup_start = viewer_source.index("  function makeOdEndpointPopup(stop) {")
    popup_end = viewer_source.index("\n  function ", popup_start + 1)
    popup = viewer_source[popup_start:popup_end]
    assign_start = viewer_source.index("  function assignOdEndpoint(endpoint, stopId) {")
    assign_end = viewer_source.index("\n  function ", assign_start + 1)
    assign = viewer_source[assign_start:assign_end]
    network_start = viewer_source.index("  async function renderNetwork() {")
    network_end = viewer_source.index("\n  async function renderLineGeometry", network_start)
    network = viewer_source[network_start:network_end]
    mode_start = viewer_source.index("  function setMapClickMode(mode) {")
    mode_end = viewer_source.index("\n  function formatSelectedStop", mode_start)
    mode_switch = viewer_source[mode_start:mode_end]

    assert "Set as origin" in popup
    assert "Set as destination" in popup
    assert "aria-label" in popup and "stop ID ${stop.stop_id}" in popup
    assert "role', 'group'" in popup and "Choose an endpoint for stop ${stop.stop_id}" in popup
    assert "event.stopPropagation()" in popup
    assert "L.DomEvent.disableClickPropagation(content)" in popup
    assert "assignOdEndpoint(endpoint, stop.stop_id)" in popup
    assert "selector.value = String(stopId)" in assign
    assert not re.search(r"\$\('flow-origin'\)\.value\s*=(?!=)", assign)
    assert not re.search(r"\$\('flow-destination'\)\.value\s*=(?!=)", assign)
    assert "this stop is not an available" in assign
    assert "may be excluded by the OD universe" in assign

    click_handler = network.split(".on('click', (event) => {", 1)[1].split("\n        })", 1)[0]
    assert "if (viewer.mapClickMode === 'measurement')" in click_handler
    assert "void selectMeasurementStop(stop.stop_id);" in click_handler
    assert "return;" in click_handler
    assert click_handler.index("selectMeasurementStop(stop.stop_id)") < click_handler.index("stopPropagation")
    assert "makeOdEndpointPopup(stop)" in click_handler
    assert "assignOdEndpoint(" not in click_handler
    assert "chooseOdStop" not in viewer_source
    assert "if (mode === 'measurement') viewer.map?.closePopup();" in mode_switch

    assert "Current endpoints — Origin: ${origin}; Destination: ${destination}." in viewer_source
    assert "originSelected ? '#166534'" in viewer_source
    assert "destinationSelected ? '#c2410c'" in viewer_source
    assert "originSelected && destinationSelected" in viewer_source
    assert 'role="status" aria-live="polite"' in source
    assert "neither endpoint changes until you choose an action" in source


def test_viewer_guide_documents_explicit_od_endpoint_actions() -> None:
    source = VIEWER.read_text(encoding="utf-8")
    guide = source.split('<details class="panel quick-guide">', 1)[1].split("</details>", 1)[0]
    docs = VIEWER.parents[2] / "docs" / "source" / "gravity_viewer_guide.md"
    documentation = docs.read_text(encoding="utf-8")

    for content in (guide, documentation):
        assert "OD selection" in content
        assert "click a stop" in content.lower()
        assert "Set as origin" in content
        assert "Set as destination" in content
        assert "only the chosen endpoint changes" in content or "leaves the" in content
    assert "Measurement" in guide
    assert 'go to <a href="#network-map">Network map</a>, choose <strong>Measurement</strong> mode' in guide
    assert "Measurement** mode" in documentation
    assert "Reset map view" in guide and "Reset map view" in documentation


def test_viewer_guide_documents_on_demand_modeled_path_browser() -> None:
    source = VIEWER.read_text(encoding="utf-8")
    guide = source.split('<details class="panel quick-guide">', 1)[1].split("</details>", 1)[0]
    docs = VIEWER.parents[2] / "docs" / "source" / "gravity_viewer_guide.md"
    documentation = docs.read_text(encoding="utf-8")
    for content in (guide, documentation):
        assert "modeled path" in content.lower()
        assert "on demand" in content.lower()
        assert "1%" in content
        assert "route-choice source" in content.lower()


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
    assert 'class="title-credit"' in source
    assert source.index("<h1>Gravity Viewer Bundle</h1>") < source.index("By Michel Bierlaire")
    assert source.index("By Michel Bierlaire") < source.index("<main>")
    check = subprocess.run(
        [sys.executable, str(VIEWER.with_name("build.py")), "--check"],
        cwd=VIEWER.parents[2],
        capture_output=True,
        check=False,
        text=True,
    )
    assert check.returncode == 0, check.stdout + check.stderr
