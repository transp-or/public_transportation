# Gravity-result viewer: launch and file selection

This guide explains how to open the read-only browser viewer included with the
public package and how to select a result bundle. The viewer is a small,
standalone browser application; it does not run routing, rebuild an operator,
evaluate an objective, or modify any input or result file.

The viewer is useful for inspecting a completed
`gravity_viewer_bundle_v1`. It is not a replacement for the case-study driver
or for the validation and fit reports.

## What is needed

Obtain a completed viewer-bundle directory produced by
`write_gravity_viewer_bundle` or `write_persisted_gravity_viewer_bundle`.
The bundle can be copied from the case-study results root to the computer on
which the browser will run. A JED scratch path is not directly visible to a
browser running on a laptop, so copy or archive the bundle first when needed.

At its root, a complete bundle normally looks like this:

```text
viewer_bundle/
  bundle_manifest.json
  model_specification.json
  report.json
  executive_summary.md             optional
  report.md                        optional
  full_od.csv
  predicted_measurements.csv
  residuals.csv
  grouped_residuals.csv            optional
  parameters.csv                   optional
  od_map.csv                       optional display summary
  stop_summary.csv                 optional display summary
  network/                          optional network snapshot
    stops.csv
    lines.csv
    trips.csv
    stop_times.csv
```

The manifest and the six core files are required by the viewer:
`bundle_manifest.json`, `model_specification.json`, `report.json`,
`full_od.csv`, `predicted_measurements.csv`, and `residuals.csv`. Optional
display files enable richer interaction but are not required for ordinary
table inspection. In particular, `od_map.csv` enables the **OD pairs** mode
and `stop_summary.csv` enables the **Stops** mode. Older valid bundles without
these files remain readable, with the corresponding interaction disabled.

Keep private bundles outside Git and do not upload them to a public issue or
repository. The bundle contains results and provenance, not the routing
engine itself, so the original case directory is not needed for viewing.

## Launch the viewer on macOS

The viewer source is in the public repository at:

```text
tools/gravity_viewer/index.html
```

There is no build step and no JavaScript package installation. From the public
repository root, the simplest option is to open the file in the default
browser:

```bash
cd /Users/bierlair/MyFiles/github/public_transportation
open tools/gravity_viewer/index.html
```

If the browser blocks local-file resources, or if the page does not behave as
expected when opened directly, serve the static directory locally instead:

```bash
cd /Users/bierlair/MyFiles/github/public_transportation
uv run --no-project python -m http.server 8765 --directory tools/gravity_viewer
```

Then open <http://127.0.0.1:8765/> (or
<http://localhost:8765/>) in the browser. The server only serves the viewer
files; the bundle is selected separately through the browser's folder picker.
Leave this terminal running while the page is open and press `Ctrl-C` when the
inspection is finished.

The viewer uses Leaflet and OpenStreetMap tiles for the optional map. Internet
access is therefore needed for the map background, but the bundle tables and
diagnostic panels remain useful if the map library or tile service is
unavailable.

## Select the result files

1. In the viewer, click **Select a viewer-bundle directory**.
2. Choose the bundle directory itself, for example `viewer_bundle/` or
   `results/viewer_bundle/`.
3. Do not select an individual CSV file, the `network/` subdirectory, or an
   unrelated parent directory containing several runs. The selected directory
   should be the one containing `bundle_manifest.json`.
4. Confirm the selection in the system folder picker. The browser may display
   only the folder name; it will read the files below that folder recursively.

After selection, the page reads the files in memory for display and validates
the manifest's file sizes and SHA-256 checksums. A successful load is confirmed
by the green message:

```text
Bundle loaded and all manifest checks passed.
```

The viewer never writes back to the selected directory. If the message reports
missing files, size mismatches, or checksum mismatches, select the exact bundle
root again and verify that the copy completed. Do not mix files from different
runs to make the warning disappear.

## Inspect the result

The summary panel shows the bundle schema, row counts, model-specification
fingerprint, likelihood, provenance, and available diagnostics. The model
specification is read from `model_specification.json`; it is the authoritative
description of the fitted model in the bundle.

The data preview displays bounded samples of the OD, prediction, and residual
tables. It does not load a second copy of the full case results from the
original repository.

When the optional display files and a network snapshot are present, the
**Network map** panel provides two modes:

- **Stops**: search by stop name or ID, choose the marker size and colour
  metric, and click a stop to inspect observed, modelled, residual, incoming,
  and outgoing totals plus the largest available OD pairs.
- **OD pairs**: choose an origin, destination, and departure-time bin; filter
  by minimum demand or residual; select a bounded number of pairs; and click a
  row or line for its details.

OD lines are a display aid, not a newly computed assignment. Their colour uses
the report convention (blue for positive observed-minus-modelled residual,
red for negative residual, and grey near zero), and the viewer draws only the
selected or filtered top-N rows.

## Troubleshooting

**“Missing `bundle_manifest.json`.”** The wrong folder was selected. Reopen the
picker and choose the directory whose top level contains the manifest.

**Checksum or size mismatch.** The bundle may be incomplete or may have been
changed during copying. Re-copy the complete bundle from its durable results
location and select the new copy. Do not edit the CSV or manifest manually.

**“Interactive summaries unavailable.”** The bundle is valid but does not
contain `od_map.csv` and/or `stop_summary.csv`. Use the ordinary tables, or
regenerate a viewer bundle with the optional display summaries when those
inputs are available. This does not change the fitted model.

**Blank map or a Leaflet error.** Check internet access and reload the page.
Without the map tiles, the validated tables remain usable. Without a bundled
`network/` snapshot, the viewer cannot draw stop markers or route context.

**The folder picker is unavailable or the page cannot read a local directory.**
Use the local HTTP-server command above and select the bundle from the page.
Browser security settings may also require granting the page permission to
read the selected folder.

**The bundle is very large.** Select only the bundle directory, keep the
browser tab dedicated to this inspection, and close it when finished. Viewing
does not require the original operator, routing cache, checkpoints, or case
configuration.

## Short checklist

Before interpreting a result, confirm:

- the bundle was exported only after the intended fit and validation stages;
- the selected folder contains `bundle_manifest.json` at its root;
- the green manifest-validation message is displayed;
- the model-specification fingerprint and provenance correspond to the run;
- optional `od_map.csv`, `stop_summary.csv`, and `network/` files are present
  when the corresponding map interaction is required;
- the original bundle remains unchanged and is retained with the case-study
  results.

For the export and data-contract details, see
[`gravity_estimation.md`](gravity_estimation.md). For the complete case-study
workflow and its `export-viewer` stage, see
[`new_case_study_walkthrough.md`](new_case_study_walkthrough.md), Section 23.
