# Structural-zero preprocessing

Structural-zero preprocessing removes OD/time cells that cannot plausibly carry
demand before ML, MAP, or Bayesian estimation starts. The preprocessor will read
one versioned TOML file, analyze the scheduled network, reject conflicts with any
existing fixed-demand file, and write a fixed-demand file plus an audit report.
The estimation pipeline can then omit those cells from its parameter vector.

The configuration, result contracts, scheduled-topology adapter, path-metric
engine, rule classification, and fixed-demand reconciliation documented here
are available, together with deterministic output persistence.

## Workflow

1. Prepare a scenario folder containing the network and time-bin inputs.
2. Copy and edit
   [`structural_zeros.toml`](examples/geneva_gtfs/structural_zeros.toml).
3. Call `run_structural_zero_preprocessing(path)`. It first loads the TOML
   strictly: unknown or missing parameters, invalid types, invalid ranges, and
   missing input paths are errors.
4. Build the scheduled topology using the same assignment feasibility settings.
5. Classify every OD/time cell and retain its path metrics and triggered rules.
6. If an existing fixed-demand file assigns a nonzero value to a newly detected
   structural zero, stop with an error. This is the only supported conflict
   policy and therefore is not configurable.
7. Write the merged fixed-demand file, audit table, summary, resolved TOML, and
   fingerprints to the output folder.
8. Pass the fixed-demand file to ML, MAP, or Bayesian estimation. Structural-zero
   cells are absent from the estimated parameter vector.

Relative paths are resolved relative to the TOML file, not the current working
directory. The output folder need not exist when configuration is loaded, but it
must differ from the scenario folder. The loader does not create or modify it.

## Complete parameter reference

### Top-level parameters

| Parameter | Required | Valid values | Meaning |
|---|---:|---|---|
| `version` | yes | integer `1` | Configuration schema version. Other versions are rejected. |
| `[scenario]` | yes | table | Scenario input location. |
| `[output]` | yes | table | Output location and reporting choices. |
| `[rules]` | yes | table | Rule selection and rule-specific parameters. |
| `[assignment]` | yes | table | Feasibility settings shared with assignment. It may be empty to use all defaults. |
| `[existing_fixed_demand]` | no | table | Existing fixed-demand input to merge and validate. |

### Scenario and output

| Parameter | Required | Valid values / default | Meaning |
|---|---:|---|---|
| `scenario.folder` | yes | nonempty path to an existing directory | Scenario root. |
| `scenario.demand_file` | no | path to an existing CSV, Parquet, or JSON table | Explicit demand candidate table. When omitted, `demand.*` is loaded from `scenario.folder`. |
| `output.folder` | yes | nonempty path different from `scenario.folder` | Destination for generated artifacts. |
| `output.include_retained_cells_in_report` | no | Boolean; default `true` | Include non-structural-zero cells in the detailed audit table. |

### Rule selection

`[rules.enabled]` is required and must contain every key below. Each value is a
TOML Boolean (`true` or `false`). When a rule is enabled, its corresponding
`[rules.<name>]` table is required. A disabled rule's table may be omitted; if
present, it is still validated.

| Key | Structural-zero condition |
|---|---|
| `same_stop` | Origin and destination are the same stop. |
| `no_feasible_path` | No path satisfies the scheduled-network feasibility model. |
| `maximum_transfers` | The minimum feasible transfer count is greater than the configured maximum. |
| `maximum_initial_wait` | The minimum feasible initial wait is greater than the configured maximum. |
| `maximum_journey_time` | The minimum feasible journey time is greater than the configured maximum. |
| `minimum_feasible_departures` | The number of feasible departures is below the configured minimum. |

The empty `[rules.same_stop]` and `[rules.no_feasible_path]` tables have no
parameters. The parameterized tables are:

| Parameter | Required when table exists | Valid values | Semantics |
|---|---:|---|---|
| `rules.maximum_transfers.max_transfers` | yes | integer `>= 0` | Largest accepted transfer count. With `2`, paths requiring 3 or more transfers are structural zeros. |
| `rules.maximum_initial_wait.max_initial_wait_minutes` | yes | finite number `>= 0` | Largest accepted minimum initial wait, in minutes. |
| `rules.maximum_journey_time.max_journey_time_minutes` | yes | finite number `> 0` | Largest accepted minimum journey time, in minutes. |
| `rules.minimum_feasible_departures.min_feasible_departures` | yes | integer `>= 1` | Smallest accepted count of feasible departures in the OD/time cell. |

Threshold comparisons are strict in the rejection direction: a value equal to a
configured maximum is retained, while a count equal to the configured minimum is
retained.

### Assignment feasibility

| Parameter | Required | Valid values / default | Meaning |
|---|---:|---|---|
| `assignment.max_access_deviation_minutes` | no | finite number `>= 0`; default `15.0` | Maximum deviation allowed when accessing the scheduled service. |
| `assignment.max_transfer_wait_minutes` | no | finite number `>= 0`; default `30.0` | Maximum waiting time accepted for a transfer. |
| `assignment.minimum_dwell_seconds` | no | integer `>= 1`; default `1` | Minimum positive dwell used by the feasibility model. |

These values must match the assignment assumptions used later. Otherwise the
preprocessor and estimator would use different definitions of a feasible path.

The topology adapter enforces this correspondence directly: it translates these
three parameters to `AssignmentConfig`, invokes the production time-expanded
graph builder, validates its DAG and centroid invariants, and adds immutable
forward and reverse adjacency indexes. Its fingerprint covers the effective
feasibility settings, graph arrays, stop and time-bin identifiers, trips, and
lines. Demand values are not part of this representation.

## Path metrics

The metric engine can evaluate the complete Cartesian product of origin stops,
destination stops, and time bins. The end-to-end service deliberately evaluates
only the unique keys in `scenario.demand.records`: cells absent from that input
are already absent from the estimation parameter vector and must not be added to
the generated fixed-demand file. It uses a backward dynamic program once per
destination instead of a separate graph search for every cell. Its complexity
is linear in the graph size per destination, plus the number of candidate
OD/time cells and their access links.

| Metric | Definition |
|---|---|
| `feasible` | At least one access link leads to the requested destination under destination-absorption semantics. |
| `minimum_transfers` | Smallest number of inter-line transfer links over feasible paths. Boarding the first vehicle is not a transfer. |
| `minimum_initial_wait_minutes` | Smallest nonnegative difference between the time-bin start and a feasible boarding departure. Departures before the bin start, permitted by the access-deviation window, have zero wait. |
| `minimum_journey_time_minutes` | Smallest difference between boarding departure and final destination arrival. |
| `feasible_departure_count` | Number of access links—scheduled boarding opportunities—that can reach the destination. |
| `earliest_arrival_seconds` | Earliest final destination arrival, in seconds from midnight. |

Metric minima need not describe one common path. For example, the earliest
boarding service can minimize initial wait while a later express service
minimizes journey time. At a destination stop, an arrival is absorbing: the
engine allows only its egress link and cannot manufacture feasibility by riding
past the destination and returning. Same-stop cells do not receive an invented
zero-length transit path; they are handled explicitly by the `same_stop` rule.

## Rule classification

Only enabled rules participate. Maximum thresholds trigger when a metric is
strictly greater than the configured value; the minimum-departures threshold
triggers when the count is strictly smaller. Equality therefore retains a cell.
For an infeasible path, transfer, wait, and journey-time metrics are undefined
and cannot trigger, while a positive minimum-departures requirement does trigger
because the feasible-departure count is zero.

Every triggered reason is retained in the audit record. The primary reason is
selected using this fixed precedence, independent of TOML table order:

1. `same_stop`
2. `no_feasible_path`
3. `maximum_transfers_exceeded`
4. `maximum_initial_wait_exceeded`
5. `maximum_journey_time_exceeded`
6. `insufficient_feasible_departures`

Summary reason counts use the primary reason, so each structural-zero cell is
counted exactly once. The high-level analysis entry point also verifies that the
topology was built with the same assignment-feasibility settings as the loaded
configuration before it classifies any cells.

### Existing fixed demand

| Parameter | Required | Valid values | Meaning |
|---|---:|---|---|
| `existing_fixed_demand.file` | yes when the table exists | nonempty path to an existing file | Previously defined fixed OD/time values. |

There is deliberately no `conflict_policy` option. A nonzero existing fixed value
for a detected structural-zero cell is always an error; silently overwriting it
could conceal inconsistent inputs.

Reconciliation validates every existing key against the analyzed OD/time
universe. Compatible existing entries are preserved exactly, including positive
fixed values on retained cells and explicit zeroes. Newly detected structural
zeroes are added with `fixed_flow = 0`. Duplicate keys, keys outside the analyzed
universe, non-finite or negative values, and any nonzero overlap with a detected
structural zero are errors. All conflicts are collected and reported together;
there is no partial merged result when an error occurs.

## Output artifacts

Persistence validates fingerprint and reconciliation consistency before creating
the output folder or replacing any file. It renders every payload in memory and
then writes each destination through a temporary file on the same filesystem,
flushes and synchronizes it, and performs an atomic replacement. Repeated writes
of identical inputs produce identical bytes.

| File | Contents |
|---|---|
| `fixed_demand.csv` | Canonically sorted merged fixed-demand input accepted by the estimation pipeline. |
| `structural_zero_audit.csv` | Per-cell decision, all reasons, and all path metrics. Retained cells are included according to `output.include_retained_cells_in_report`. |
| `structural_zero_summary.json` | Cell counts, mutually exclusive primary-reason counts, and reconciliation counts. |
| `resolved_config.toml` | Effective configuration with absolute paths and explicit defaults. |
| `fingerprints.json` | Scenario, graph, and configuration fingerprints, the canonical configuration payload, and SHA-256 hashes of the other four artifacts. |

The audit uses empty fields for metrics that are undefined on infeasible cells.
Triggered reasons are written in primary-precedence order and separated with a
semicolon. The summary always covers the complete analyzed universe, even when
retained cells are omitted from the detailed audit.

## Python entry point

The complete workflow has one substantive input—the TOML path:

```python
from public_transportation.preprocessing import run_structural_zero_preprocessing

result = run_structural_zero_preprocessing("structural_zeros.toml")
print(result.analysis.num_structural_zero)
print(result.outputs.fixed_demand)
```

Long-running applications can observe the complete workflow through the
dependency-free callback API. Each immutable `StructuralZeroProgress` event
contains `phase`, `completed`, `total`, `elapsed_seconds`, and an optional
`message`. Stable workflow phases are `load_scenario`, `build_topology`,
`destination_profiles`, `classify_cells`, `reconcile_fixed_demand`,
`render_outputs`, `write_outputs`, and `complete`; small auxiliary rendering
phases identify fixed-demand and summary preparation.

```python
from public_transportation.preprocessing import (
    run_structural_zero_preprocessing,
    structural_zero_tqdm_progress,
)

with structural_zero_tqdm_progress() as progress:
    result = run_structural_zero_preprocessing(
        "structural_zeros.toml",
        progress=progress,
    )
```

The numerical functions never print. `tqdm` is imported only when the adapter
is enabled, and its bar is written to stderr so JSON on stdout remains clean.
The documented CLI accepts `--progress` and `--no-progress`, defaulting to
progress only on an interactive terminal.

Destination, classification, and audit loops emit every item for at most 100
items. Larger loops emit about every 1 percent, never more frequently than every
25 items, while also emitting when ten seconds have passed. Every loop emits
zero initially and its exact total finally. Callback exceptions propagate.
Artifact payloads are completed before atomic per-file replacement, so an
exception during rendering cannot replace an existing file with partial data.

The service loads the scenario with `strict=True`, fingerprints its canonical
domain content, rejects duplicate demand keys, builds the production assignment
topology, computes and classifies metrics for the demand-key universe,
reconciles existing fixed demand, and persists the complete artifact set. The
scenario fingerprint is independent of collection order and excludes
`metadata.created_at`, whose domain default is clock-based and does not affect
the model. Demand values are included, so changing the a-priori OD input changes
the scenario fingerprint even when its key universe is unchanged.

## Reproducibility contracts

The loaded configuration is immutable and exposes a deterministic resolved-TOML
serialization and SHA-256 fingerprint. Paths in the resolved form are absolute,
and omitted optional values are expanded to their defaults. Analysis results are
also immutable: records must be unique and sorted by `(origin_stop_id,
dest_stop_id, time_bin_id)`, path metrics must be internally consistent, and each
structural zero must identify a primary reason and all triggered rules.
