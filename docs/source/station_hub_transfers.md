# Station hubs and inter-platform transfers

Scheduled events remain attached to platform stop IDs. A station or hub is a
separate transfer-equivalence concept; it does not replace platform IDs in
timetables, measurements, or OD endpoints.

## Scenario inputs

Add both tables to a scenario folder when transfer geometry is available:

`platform_stations.csv`

| Column | Meaning |
| --- | --- |
| `platform_id` | Existing platform/stop ID used by scheduled events |
| `station_id` | Physical station or hub containing the platform |

`hub_platform_distances.csv`

| Column | Meaning |
| --- | --- |
| `station_id` | Hub containing the platform |
| `platform_id` | Platform ID |
| `distance_m` | Non-negative distance from hub reference point to platform, metres |

Every platform used by an event must have exactly one station membership. Each
platform may have one hub-distance row for its mapped station. IDs must exist
in the scenario; distance keys must be unique; distances must be finite and
non-negative. Missing distance values are errors when a cross-platform
transfer needs them. Both tables must be present, or neither.

If neither table is present, the package uses the legacy identity mapping
`station_id = platform_id`. This preserves the existing same-platform transfer
behavior. The public package never parses platform IDs to infer a station.

OD origins and destinations also remain explicit. A caller with station-level
OD endpoints must provide an explicit platform/station endpoint mapping;
transfer geometry does not silently collapse platform-level OD cells.

## Transfer eligibility and cost

A transfer is a **change of line**, not a change of trip ID. A scheduled
connection is eligible when the arrival and departure platforms map to the
same station, the departing line differs from the arriving line, the departure
follows the arrival, and the existing maximum transfer-wait window is
satisfied. Same-line reboarding is excluded, including across platforms.
Different stations are not connected.

For different platforms at one station, the proxy walking time is

\[
t_{\mathrm{walk}} = \frac{d_{h,p}+d_{h,q}}{60v},
\]

where `d_{h,p}` and `d_{h,q}` are hub-to-platform distances in metres and `v`
is the configured walking speed in metres per second. For a same-platform
transfer, `t_walk = 0`.

If `Δ` is scheduled elapsed time in minutes, waiting and walking are charged
separately:

\[
t_{\mathrm{wait}} = \max(0,\Delta-t_{\mathrm{walk}}), \qquad
C_{\mathrm{transfer}} = \beta_T t_{\mathrm{wait}} + \beta_W t_{\mathrm{walk}}.
\]

The fixed `AssignmentConfig` defaults are `walking_speed_m_per_s = 1.2` and
`beta_walk = 1.0`. The speed is an assumption; `beta_walk` is a fixed
generalized-cost weight, not estimated here. No detour factor is applied.
Walking time changes cost but **does not impose a minimum-time feasibility
test**: a short timetable gap remains eligible, with zero waiting cost and the
walking cost retained.

The station hub is represented with platform connectors and retained incoming
line context, rather than a complete arrival-platform × departure-platform
link set. Hub connectors are not transfers and do not increment the line-change
count.

## OD feasibility versus assignment paths

The configured maximum-transfer rule applies to scheduled OD feasibility and
support generation. It counts line changes, including a cross-platform change
at one station, exactly once. It does not add a transfer-count dimension or a
two-transfer path cap to the assignment graph. Therefore, the feasibility
search restricts which OD cells are admitted, while paths in the assignment
graph may contain more line changes.

## Case-specific geometry example

A case-specific preparation process may derive station membership from its own
source data and compute each hub coordinate as the centroid of its platforms
in a suitable projected coordinate reference system. Straight-line distance
from that centroid is only a proxy, not a pedestrian-network route. The
private TPG case uses LV03 coordinates for this centroid and distance
calculation; its stop-ID convention and any coordinate-audit rules remain in
the private preparation layer, not in this package.

## Cache identity and migration

Assignment graph and operator identities include the effective
platform-to-station mapping, hub distances, walking speed, `beta_walk`, the
other assignment and route-choice settings, and the graph-semantics version.
Changing any of these selects a different assignment artifact. A missing or
mismatched artifact is rebuilt only through the normal preparation/activation
workflow; it must not silently load an older incompatible operator.

Feasibility support depends on station membership and line-change semantics,
but not on hub distances or walking-cost settings. Thus, changing only a
distance or walking coefficient invalidates assignment/operator artifacts but
leaves compatible OD-feasibility checkpoints reusable. The base scenario
fingerprint and unrelated artifacts remain reusable when their own inputs are
unchanged. Changing station membership invalidates feasibility support as well
as assignment artifacts.

Existing scenarios need no migration if they should keep same-platform-only
transfers: omit both sidecars and retain the identity fallback. To enable
inter-platform transfers, add both validated sidecars and prepare a new
assignment operator. Existing operator caches are not compatible with the new
graph semantics and must not be adopted as if they were.
