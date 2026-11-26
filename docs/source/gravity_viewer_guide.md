# Gravity viewer: quick user guide

This guide covers common tasks in the read-only browser viewer for completed
gravity result bundles. The viewer also includes a collapsible **Quick guide**
panel near the page title.

## Before you start

Choose a completed viewer-bundle directory. It contains a `bundle_manifest.json`
file at its top level. The bundle can be copied to the computer where you will
open the viewer.

## Open the viewer

Open `tools/gravity_viewer/index.html` in a browser. Click
**Select a viewer-bundle directory** and choose the folder containing
`bundle_manifest.json`. The loaded overview is shown under **Bundle summary**.

The viewer is read-only. If it reports missing files or a checksum problem,
select the bundle’s root folder again and verify that the complete bundle was
copied.

## Explore estimated OD demand

Go to **Demand explorer**. Select a **Departure time interval**, and optionally
an **Origin** and **Destination**, then click **Load selected partition**. A
partition is the saved slice of demand for that selection; the viewer finds it
for you, so there is no need to locate or open a partition file.

Use **Load 50 more** to extend the results in 50-row increments, or **Load all
matching rows** to display all rows matching the selection. Click a column
heading to sort the rows currently loaded; rows not yet loaded are not included
in that sorting.

### Browse modeled paths between two stops

When both **Origin** and **Destination** are selected, the **Modeled paths
between selected stops** browser appears. Choose one **Departure time
interval**, set the minimum complete-path probability (1% by default), and
click **Show modeled paths**. The threshold applies to each complete path
individually, not to the sum of all paths. Listed probabilities are products of
the saved route-choice transition probabilities; they are conditional on the
selected endpoints and departure interval. The results are ranked from highest
to lowest probability and bounded to keep the query responsive.

Path lists are calculated on demand and are not saved in the viewer bundle. The
local query service must be started with a prepared route-choice source that
matches the selected fit and mapping. If the service reports that no compatible
route-choice source is available, its other viewer queries remain usable, but
the path browser cannot calculate paths.

For deployments that already retain prepared dense or sharded fixed-routing
inputs, the local service accepts an adapter factory with
`--route-choice-query-factory module:callable`. The callable receives the
validated `GravityViewerQueryService` and should return
`RouteChoicePathQuery.from_fixed_routing(...)`, using the service's mapping
fingerprint, fit fingerprint, and canonical departure intervals. This is an
integration hook for loading existing routing state; it does not rebuild
routing or persist paths. Without the adapter, the path browser stays disabled.

For example, add the option to the normal local-service command:

```sh
python -m public_transportation.inference.gravity.viewer_query_service \
  --bundle /path/to/viewer-bundle \
  --fit-manifest /path/to/fit-manifest.json \
  --mapping-artifact /path/to/temporal-mapping \
  --route-choice-query-factory my_project.viewer_routes:build_query
```

The adapter's `build_query(service)` function loads the already-prepared routing
source and returns a `RouteChoicePathQuery`; it must not rebuild or alter that
source.

## Choose endpoints from the network map

Go to **Network map** and choose **OD selection**. Click a stop to open its
popup, then choose **Set as origin** or **Set as destination**. Clicking a stop
alone does not change either endpoint, and choosing one endpoint leaves the
other unchanged. The map status shows both selections; the selected origin and
destination have distinct marker outlines. Use the **Origin** and
**Destination** selectors in **Demand explorer** to replace or clear either
selection. Click **Reset map view** in the map controls to restore the full
network extent shown when the bundle was loaded; the button does not change
the selected endpoints or map layers.

## Find measurements at a stop

Go to **Network map**, choose **Measurement** mode, then click a stop with
measurements. If several rows are available, choose one with the **Measurement
row** selector or browse with **Previous rows** and **Next rows**. The selected
row shows its **Observed count** and **Modeled mean**.

Map-based measurement lookup depends on stop and measurement lookup data being
included in the bundle. Stops without matching rows show a clear message.

## See the OD contributors to a measured row

When contribution data are available in the bundle, go to **Measured flows by OD
pair**. Under **What do you want to look up?**, choose **Measured row →
contributing OD pairs**, select the measured row, and click **Load OD
contributors**.

Use **Rows per page**, **Previous contributors**, and **Next contributors** to
browse. Contributors are ranked by the magnitude of their modeled-mean
contribution and decompose the **Modeled mean**, not the observed count; fixed,
additive, and floor components shown separately are not assigned to individual
OD pairs. Column sorting applies to the displayed page.

## Find measurements affected by an OD pair

In **Measured flows by OD pair**, choose **OD pair → affected measured rows**
under **What do you want to look up?** Select the **OD departure interval**,
**Origin**, and **Destination**, then click **Load affected measurements**.
The OD departure interval is when the trip leaves; it is distinct from a
measurement’s observation time. Availability depends on the bundle contents.

## While a contribution query is running

Follow the status and progress area while the viewer processes the query. Wait
for the results, or change your selection to request a different query.

## Viewer bundle and query-service revisions

The bundle's `runtime_package_revision` records which package wrote the bundle;
it is separate from the fit and mapping source revisions. The query service
accepts bundles from its current runtime when they use the supported bundle
type/schema and measurement-query descriptor. It also explicitly supports the
verified producer revision
`387758616ee8684592501de53fcef07a62f39ae6` for
`gravity_viewer_bundle` schema 1 and measurement-query descriptor 2. The
producer value remains unchanged in the bundle. Fit/mapping source revisions,
artifact identities, and all bundle and mapping checksums are still validated
independently; producer compatibility does not relax those checks.

Before adding an older producer revision to the compatibility policy,
maintainers must inspect that revision's bundle writer and query descriptor,
then exercise a bundle produced by it with the current query service. Verify the
bundle checksum scan, a bounded forward and reverse measurement query, exact
fit/mapping and source-revision checks, and that opening/querying leaves every
bundle file unchanged. Add only the tested producer/format/descriptor
combination and keep a regression fixture for it. Do not approve revision
ranges or infer query compatibility from bundle schema alone.

## Network map

When the bundle includes the relevant network and display data, the map also
offers **Stops** and **OD pairs** views. Static transit-line geometry, when
present, is map context and does not represent passenger assignment flows.

## Troubleshooting

**The bundle cannot be opened.** Choose the directory whose top level contains
`bundle_manifest.json`, not an individual CSV or a parent directory containing
several bundles.

**Measurement selection is unavailable.** Map-based lookup depends on network
stop and measurement lookup data in the bundle.

**OD contributions are unavailable.** The measured-row and reverse OD queries
depend on compatible contribution data being included in the bundle.

**The map background is blank.** Map tiles require internet access. The tables
and other viewer panels remain available without the map background.
