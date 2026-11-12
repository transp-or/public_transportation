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
which the browser will run. A cluster scratch path is not directly visible to a
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
  viewer_summary.json              partition/index metadata
  partitions/od/                   one CSV per departure-time bin
  partitions/route_flows/          optional sparse post-fit route flows
  partitions/route_sections/       optional OD-to-directed-section contributors
  partitions/section_loads/        optional section-load summaries by OD interval
  line_geometry/sections.csv       optional directed section catalog
  line_geometry/lines.geojson      optional compact line geometry
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
cd /path/to/public_transportation
open tools/gravity_viewer/index.html
```

If the browser blocks local-file resources, or if the page does not behave as
expected when opened directly, serve the static directory locally instead:

```bash
cd /path/to/public_transportation
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

After selection, the page reads the manifest, model specification, report, and
small partition summary first. It does not parse `full_od.csv`,
`predicted_measurements.csv`, or `residuals.csv` at startup. Selecting a time
slot streams only that partition through a Web Worker, verifies its size and
checksum, and stores the bounded result in IndexedDB. A successful load is
confirmed by the green message:

```text
Bundle manifest and summary loaded. Partition checksums are verified when selected.
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

The data preview and **Demand and route-assignment explorer** display bounded
samples. They never load a second copy of the full case results into browser
memory. Use **Clear viewer cache** after replacing a bundle or when browser
storage is under pressure; cache entries are keyed by the bundle fingerprints,
partition checksum, and cache-format version.

The explorer distinguishes estimated OD demand from post-fit assignment data:

- choose a departure-time slot, origin, and destination for origin-first or
  destination-first OD exploration;
- select **Route-assignment flow** and filter by line, trip, origin,
  destination, or realized transfer count when `partitions/route_flows/` is
  present;
- service totals are expected assigned demand or passenger-leg usage as stated
  by the bundle, not automatically unique passengers;
- the transfer selector uses realized route-assignment transfers only. A
  minimum-feasible-transfer feature used during demand estimation is never
  substituted for an observed route-choice outcome.

Existing bundles without `viewer_summary.json` or route-flow partitions remain
readable. They expose the validated legacy tables and clearly report that
lazy time-slot and route-level exploration is unavailable until a refreshed
export is produced.

## Directed line sections

Bundles produced with an explicit post-fit route export may also contain a
`route_sections` entry in `viewer_summary.json`. The catalog contains one
directed section for each consecutive stop pair in each unique line pattern.
When no validated source shape is available, the geometry is labelled
`stop_to_stop_approximation`; it is a display connector and must not be read as
road-following vehicle geometry. Branches and directions retain separate
pattern and section identities.

The section explorer first loads only the small line geometry and the selected
OD departure-time interval's section-load summary. Clicking a section then
loads only that interval's contributor partition. Each contributor preserves
the OD, stable route-path and leg identity, path share, assigned contribution,
line/pattern/trip, and realized transfer count. Assigned section flow is the
sum of the saved path contributions once per traversed section. It is passenger
section usage (a transfer journey can contribute on more than one line), not a
count of unique passengers. The interval is the OD departure interval, not the
clock interval at which a vehicle passes the selected section.

The Python API mirrors the viewer:

```python
bundle.query_section_load("s-…", departure_time_bin="t3")
bundle.query_section_od("s-…", departure_time_bin="t3", page=0, page_size=100)
```

### Expected OD use of scheduled sections

For a completed fixed-routing fit, the public producer
`export_gravity_od_section_marginals` replays the persisted destination-group
probabilities one OD cell at a time. It emits only scheduled ride links, with
`flow_semantics="service_marginal"` and the OD's departure-time slot. Each row
means fitted OD demand multiplied by the cached probability of using that
adjacent-stop ride link. It is therefore expected OD use of a section, not an
enumerated passenger route and not an exact transfer-count distribution.

The producer validates the assignment identity, every routing-shard descriptor
and content hash, and the canonical/compact OD ordering before consuming the
fit. It writes bounded durable checkpoints within each destination group and publishes
the final CSV atomically. A representative dry-run/pilot should be recorded
before scheduling the full export; no routing preparation or model estimation
is triggered by this API.

The planning API accepts `checkpoint_directory`, `resume=True`,
`checkpoint_batch_size`, `checkpoint_identity`, and a progress callback. A
checkpoint is stored as `plan_checkpoint.json` plus atomic per-group records;
resume validates the schema, algorithm version, scientific-input identity, and
pilot settings before skipping verified groups. The completed checkpoint is
distinct from the private case runner's final `status: awaiting_review` plan
file, so interrupted pilots cannot be approved accidentally.

The export itself accepts `od_chunk_size` and
`progress_interval_seconds`. Each chunk is fsynced and then recorded in the
checkpoint manifest with its OD range, row count, elapsed time, and checksum.
Progress callbacks distinguish non-durable OD traversal heartbeats from
`chunk_checkpointed` events and include inner-cell counts, rolling ETA, and ETA
confidence. Existing verified group-level checkpoints are migrated as legacy
completed units; uncommitted groups are recomputed safely.

`query_section_od` streams the selected interval and section only, and pages
the returned rows without truncating the underlying contributor data. Older
schema-1 bundles explicitly report section data as unavailable and continue to
support the destination/OD view.

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
contain the optional display summaries or partition index. Use the ordinary
tables, or regenerate a viewer bundle. This does not change the fitted model.

**“Route flows unavailable.”** The fit and destination demand are present, but
no post-fit route export was supplied. The viewer does not invent route paths
from OD demand. Run the separate export step below only after checking the
saved routing cache and its expected cost.

**Blank map or a Leaflet error.** Check internet access and reload the page.
Without the map tiles, the validated tables remain usable. Without a bundled
`network/` snapshot, the viewer cannot draw stop markers or route context.

**The folder picker is unavailable or the page cannot read a local directory.**
Use the local HTTP-server command above and select the bundle from the page.
Browser security settings may also require granting the page permission to
read the selected folder.

**The bundle is very large.** Select only the bundle directory. Startup reads
only small metadata, and active memory is bounded by the selected partition and
display sample. Safari should be tested with the complete Model 6 bundle; a
reload indicates that the chosen partition or network snapshot is still too
large and should be split further rather than loading the legacy CSV.

## Optional post-fit route-flow export

The core gravity fit supplies estimated destination demand, not OD-to-line/trip
paths. If persisted assignment artifacts expose the correct fixed-routing
operator, prepare a CSV with the canonical fields below and pass it as
`route_flow_rows` to `write_gravity_viewer_bundle` (or
`write_persisted_gravity_viewer_bundle`):

```text
origin_stop_id,destination_stop_id,departure_time_bin,line_id,trip_id,
boarding_stop_id,alighting_stop_id,route_leg_index,route_leg_count,
route_legs,assigned_demand,path_share,transfer_count,flow_semantics,
route_path_id,pattern_id,direction,assignment_fingerprint
```

`flow_semantics` is `path_probability` when complete path shares are available,
`service_marginal` for link/trip marginals, or `passenger_leg_usage` for
leg-level usage. Before any expensive computation, inspect the saved routing
cache identity and report whether it can provide these rows, the expected
runtime/storage, and the checkpoint plan. If the cache cannot recover route
contributions, use the separate post-fit export with the saved fit and routing
implementation; do not rerun estimation and do not label service marginals as
complete path probabilities.

The public helper makes that boundary explicit:

```python
from public_transportation.inference.gravity import (
    export_gravity_route_flows,
    plan_gravity_route_flow_export,
)

plan = plan_gravity_route_flow_export(
    routing_cache="/path/to/saved-routing-cache",
    expected_assignment_fingerprint="<fit-assignment-fingerprint>",
    expected_rows=<estimated-row-count>,
)
if not plan.available:
    raise RuntimeError(plan.reason)
export_gravity_route_flows(
    route_flow_rows=rows_from_the_validated_routing_implementation,
    output_path="route_flows.csv",
    checkpoint_path="route_flows.checkpoint.json",
    progress=lambda rows: print(f"exported {rows:,} rows"),
)
```

The resulting CSV can be supplied as `route_flow_rows` to the bundle writer;
the writer partitions it by departure-time bin and records checksums and row
counts in the manifest.

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
