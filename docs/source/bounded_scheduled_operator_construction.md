# Bounded scheduled-operator construction

Direct scheduled construction accepts one monotonic deadline for the complete
lifecycle. A deadline controls construction effort only; it never changes the
mathematical definition of a completed operator.

Use `ConstructionDeadline.from_budget(...)` for a relative laptop or batch-job
budget, or `ConstructionDeadline.from_absolute(...)` when a scheduler supplies
an absolute monotonic cutoff. Leave the deadline unset for backward-compatible
unlimited execution. The safety margin prevents a new indivisible unit from
starting unless its predicted duration fits inside the remaining safe time.

`activate_direct_scheduled_temporal_operator` distinguishes three normal
outcomes:

- an activation-policy decline returns no operator and does no routing work;
- a deadline stop returns a `ConstructionTermination` and retains validated
  checkpoints;
- a valid completed artifact is loaded before routing or deadline enforcement,
  including when the new budget is almost exhausted.

Actual failures still raise their original errors. A deadline stop never
publishes a partial final artifact.

## Progress events

Pass a callback with `progress=...`. Events are plain dictionaries and include
schema version, phase, status, elapsed and remaining seconds, safety margin,
completed and total work units, current unit, recent duration, predicted
remaining duration, cache counters, checkpoint location, memory when available,
and terminal reason. The numerical core does not depend on a display library;
callers may write JSONL, logs, `tqdm`, or UI updates.

Phases include cache validation, routing preparation, support discovery,
planning, shard validation and construction, temporal-block assembly,
validation, persistence, and completion.

### Hierarchical progress and ETA semantics

The flat fields above remain the compatibility contract.  Newer reporters add
the following observational fields without changing the schema version:

- `job_elapsed_seconds` and `phase_elapsed_seconds` distinguish the complete
  job clock from the clock for the current phase; `phase` changes reset only
  the latter;
- `work_stack` is an outer-to-inner list of named units.  Each entry may carry
  `completed_units`, `total_units`, `current_unit`, and optional weighted
  counts.  Nested counts are not added together: use the deepest active unit
  for local progress and the explicit weighted fields for a job-level view;
- `active_units`, `queued_units`, `active_workers`, and `requested_workers`
  describe scheduling state without claiming that queued work is complete;
- `current_unit_elapsed_seconds` and
  `current_unit_predicted_remaining_seconds` describe an in-flight unit and
  remain separate from completed-unit ETA;
- `completed_weight`, `total_weight`, `weighted_fraction`, and throughput
  fields are available when units have materially different costs;
- `eta_lower_seconds`/`eta_upper_seconds`, `eta_confidence`, and
  `eta_reason` expose uncertainty.  Estimates use only completed, persisted
  units; the first estimate is provisional and may remain unavailable for an
  opaque operation;
- `predicted_job_remaining_seconds` and
  `estimated_job_completion_at_utc` are emitted only when the current phase
  estimate and all declared future-phase durations are known;
- `checkpoint_reusable`, `next_resumable_position`, `reused_units`, and
  `rebuilt_units` make restart state explicit.  `deadline_margin_seconds` and
  `will_finish_before_deadline` are conservative and remain null when either
  ETA or deadline is unknown.

An indivisible planning, compilation, or synchronization call may therefore
emit a heartbeat with a current work name but no completed-unit ETA.  This is
intentional: the reporter must never invent precision for a callback that does
not expose progress.  All of these fields are reporting metadata only and are
excluded from numerical identities, checkpoints, and artifact fingerprints.

Telemetry delivery is isolated from construction.  Worker-origin callbacks
queue their already-built event in memory and return without waiting for JSONL
file I/O; the main phase-boundary emission drains the queue so durable records
remain ordered.  If no callback is supplied, reporting is a no-op.  Sink
exceptions are recorded as reporting diagnostics and never change a successful
scientific result.

Callers that know a phase schedule may provide reporting hints without changing
the scientific run:

```python
reporter.set_phase_plan({
    "planning": 30.0,
    "routing_preparation": 900.0,
    "temporal_block_assembly": 120.0,
})
with reporter.work_scope("routing_shards", total_units=num_shards):
    ...
```

An omitted or opaque phase keeps the job-level ETA unavailable; the plan is
never treated as a scientific assumption.

Routing preparation accepts two independent progress controls:

```python
routing_config = FixedRoutingPreparationConfig(
    progress_interval_seconds=5.0,
    progress_interval_groups=8,
)
```

`progress_interval_seconds` is the minimum wall-clock interval between
heartbeat/progress observations, including long indivisible planning, cache
scan, compilation, synchronization, batch-execution, checkpoint, and
finalization subphases. `progress_interval_groups` samples progress at
completed destination-group (routing-unit) boundaries. The settings are not
interchangeable: seconds are not converted to groups and groups are not
converted to seconds. When both are supplied, a repeated work-progress event
is emitted only when both its wall-clock and completed-group constraints are
met; lifecycle transitions and terminal records remain visible immediately.
When only one is supplied, the other keeps its default (`1.0` second or `8`
groups). A reliable ETA is still unavailable until enough completed units have
been observed. These controls affect reporting only and never routing values,
shard contents, checkpoints, scheduling, memory layout, or artifact
fingerprints.

Older callers that provide only `progress_interval_seconds` remain supported;
new callers may additionally provide `progress_interval_groups`.

### Profiling support discovery before changing the worker plan

Support discovery is a different workload from routing preparation and from
the later numerical measurement-shard construction.  The three worker plans
must therefore be measured and configured independently.  In
`ShardedConstructionConfig`, `support_workers` controls the destination-group
support pool and `workers` controls numerical shard construction.  If
`support_workers` is omitted it inherits `workers` for compatibility; it is
never inferred from `FixedRoutingPreparationConfig.construction_workers`.
Changing either value is execution metadata only and does not enter the
scientific support or artifact fingerprint.

For a case-owned benchmark, enable the optional profile recorder and write it
below the same isolated results root as the run:

```toml
# config/model.toml
support_workers = 2
shard_construction_workers = 1
support_progress_interval_seconds = 5.0
profile_support_discovery = true
```

The generic direct-scheduled template writes
`results/audit/support_discovery_profile.json`.  Each completed destination
group records selected/free OD cells, measurement count, origin-chunk count,
reachability time, projection time, checkpoint time, total time, cache reuse,
worker identity, and peak RSS when available.  Aggregate fields include
completed groups, cache hits/misses, weighted cell throughput, component
timings, worker IDs, and elapsed wall time.  Profiling is observational: it
does not materialize a second support set, alter scheduling, or change any
numerical artifact.

The profile also reports deterministic representative small/median/large group
IDs.  A pilot benchmark may use those IDs only with an explicitly isolated
pilot input or case-owned adapter (the public helper is
`run_support_discovery_pilot`); the normal production call must still
evaluate every group.  Compare profiles from the same scientific identity
with `compare_support_profiles` and record CPU allocation, memory, JAX
backend, input/config fingerprints, and cache state alongside the JSON.  Do
not compare a cold run with a resumed run as if they were the same workload.

Support progress is hierarchical.  During a long group, throttled heartbeat
events expose `destination_groups` as the outer unit and `origin_chunks` as
the inner unit, together with active/queued group IDs, worker ID, checkpoint
location, weighted throughput, and ETA bounds.  The first group may still have
an unavailable ETA because no completed group duration exists; after completed
groups accumulate, the estimate is based on persisted group timings.  These
records remain durable reporting metadata and are excluded from all numerical
identities.

For a controlled worker matrix, use `benchmark_support_discovery` with a
case-owned operation factory. The factory receives the worker count, an
isolated artifact root, an isolated checkpoint root, and the timing callback:

```python
benchmark = benchmark_support_discovery(
    operation_factory,
    worker_counts=(1, 2, 4, 8),
    output_root=case_results / "benchmarks/support-workers",
    metadata={
        "scenario_fingerprint": scenario_fingerprint,
        "configuration_fingerprint": configuration_fingerprint,
        "cpu_allocation": 8,
        "memory_bytes": 32 * 1024**3,
        "jax_backend": "cpu",
    },
)
```

The summary contains wall time, groups/second, weighted OD-cell throughput,
CPU seconds/utilization (when `cpu_allocation` is provided), peak RSS,
effective parallelism, and speedup against the one-worker record.  The
per-worker profile files remain under separate
`workers-XX/artifacts` and `workers-XX/checkpoints` directories.  A pilot
profile can be projected with `project_support_duration`, but its interval is
explicitly low/medium confidence for heterogeneous group costs and must not
be presented as a guaranteed full-run ETA.

## Resume and checkpoint identity

Prepared routing batches, each completed destination group's origin-specific
support, the aggregate materialized support, the deterministic construction
plan, sparse numerical shards, and temporal assembly fragments are all
persisted beneath the identity-addressed checkpoint directory. Files are
object-free NPZ or JSON payloads with schema, provenance, and content hashes.

Passing `routing_preparation_config=FixedRoutingPreparationConfig(...)` to
direct activation selects persistent destination-batched routing. Its routing
directory contains a layout-identity subdirectory, `manifest.json`, and one
`routing-shard-NNNNNN.npz` per contiguous destination-group batch. Every batch
stores only its effective mask and probabilities, plus schema, scientific
identity, shape, dtype, and content hash. Staged files are reopened and
validated before atomic publication. Corrupt batches are quarantined and
recomputed. A different resource plan uses a separate layout identity; it does
not alter route-choice probabilities.

Sparse shards and manifests are written atomically. A shard becomes completed
only after its payload validates and is renamed into place. Resume scans and
hash-validates existing shards, rejects incompatible provenance, and begins at
the first missing unit. The provenance covers assignment and graph inputs,
measurement mapping, OD and compact layouts, canonical temporal indexing,
fixed route-choice parameters, departure and feasibility assumptions,
coefficient policy, dtype, and construction schema.

Temporal assembly writes one validated fragment per numerical source shard.
Consequently, a stop during assembly resumes at the first missing fragment
instead of rebuilding the global triplet accumulator. Logical temporal keys may
have several independently persisted partitions; forward and adjoint products
sum those exact partitions.

The finalized temporal artifact is also staged and published atomically. If a
deadline is reached while staging, the staging directory is removed and the
validated construction shards remain reusable.

## Preprocessing reuse and provenance

Direct-scheduled preprocessing now builds one immutable canonical timetable
index. The same normalized stop-time ordering is consumed by candidate-pair
generation, timetable feasibility, the time-expanded assignment graph, and
structural-zero topology construction. Directed reachability is computed once
per OD universe and reused for filtering, fingerprints, and expansion. For
each origin and approved time bin, temporal feasibility is evaluated once and
then applied to all destinations in that slice; this preserves the historical
pairwise result while avoiding repeated timetable searches.

Artifact identity is split into scientific and execution provenance. Scientific
identity includes the assignment and graph fingerprints, measurement mapping,
OD and frozen-cell layouts, fixed theta, approved time bins, support
definition, and algorithm/schema versions. Execution provenance records worker
counts, chunk and block sizes, packing limits, compression, and checkpoint
layout. Changing only execution or packing settings therefore preserves the
origin-support checkpoint, while a graph, mapping, OD-layout, theta, or support
definition change invalidates the smallest affected downstream stage.

Malformed, corrupt, or ambiguous manifests are treated as incompatible cache
entries and rebuilt or quarantined; they are never silently reused. Numerical
measurement-shard algorithm changes rebuild measurement shards while retaining
compatible routing and support artifacts. The retired timetable-journey/
reduced-OD response backend is not part of this workflow.

On the packaged Geneva snapshot (62 stops, 173 trips, 15,128 OD/time cells),
the bounded preprocessing benchmark reduced the OD-time expansion from about
38.3 s to about 12.4 s, with identical scientific fingerprints and dimensions.
The comparison is recorded in
`benchmarks/preprocessing_baseline_geneva.json` and
`benchmarks/preprocessing_optimized_geneva.json`; it is a bounded fixture
benchmark, not a full-network run.

## Laptop and cluster usage

The conservative default remains `workers=1`.  Measurement-shard construction
also supports explicit bounded thread parallelism: one compiled JAX executable
is shared, while every worker owns an independent mutable routing-batch reader.
`worker_memory_budget_bytes` preflights each worker's kernel, temporary batch,
and staged-shard estimate; `maximum_resident_shards` bounds active plus buffered
results.  Workers stage validated atomic shard files, but only the parent
publishes them, updates the manifest, and reports completion, always in
canonical plan order.  Consequently worker count is not scientific provenance
and serial and parallel runs share the same cache.

`ShardedConstructionConfig.max_materialized_support_entries` separately bounds
the number of origin-specific support coordinates materialized by production
construction.  Its default is 125,000,000.  The limit is an operational memory
guard, must be positive, and is deliberately excluded from scientific artifact
identity: raising it preserves compatible per-group support checkpoints and
numerical shard caches.  Summary analysis may use `materialize=False`, whereas
the production sharded builder always requires materialized support.

The numerical-shard preflight records both the actual estimate and configured
ceiling for storage shards, manifest bytes, filesystem operations, sparse calls
per product, construction dispatches, and per-worker memory.  Rejection remains
a `MemoryError`, with a structured plan and JSON-ready diagnostics attached.
Storage-shard sizing controls participate in scientific construction provenance
because they change persisted shard identities.  The five operational ceilings
do not change numerical content: they are serialized in the plan and are
re-evaluated on every call, allowing a caller to raise an evidence-based ceiling
and reuse compatible support checkpoints safely.  Defaults remain conservative;
production callers must opt into larger limits explicitly.

Select shard and OD chunk sizes explicitly, reserve a safety margin for final
checkpointing, and resume with the same cache roots and scientific identity.
The parent stops admitting work when the predicted shard duration plus
`deadline_safety_margin_seconds` no longer fits.  Already active workers finish
their indivisible calls; their canonically publishable results are checkpointed
before the bounded stop is returned.  Worker failures cancel pending admission,
remove unpublished staging files, and leave the last valid manifest reusable.
A cluster job should derive its monotonic budget from the scheduler wall time.
Increasing concurrency is explicit; it is never inferred automatically.

A conservative routing profile is explicit as well:

```python
routing_config = FixedRoutingPreparationConfig(
    maximum_groups_per_shard=1,
    maximum_retained_bytes_per_shard=256 * 1024**2,
    maximum_temporary_bytes=2 * 1024**3,
    temporary_workspace_multiplier=4.0,
    construction_workers=1,
    resident_shard_limit=1,
    dispatch_safety_margin_seconds=120.0,
    progress_interval_seconds=5.0,
    progress_interval_groups=8,
)
```

The temporary multiplier covers estimated workspace conservatively; XLA's
actual backend workspace is not exactly predictable. The library never sizes a
batch from all detected machine memory. Cluster callers may raise budgets or
explicit concurrency after measurement; platform names do not select policy.

The public example supports a bounded call:

```bash
uv run python docs/source/examples/direct_scheduled_gravity_validation.py \
  --cache-directory /tmp/direct-scheduled-bounded \
  --time-budget-seconds 0.2 \
  --safety-margin-seconds 0.05 \
  --allow-deadline-stop
```

Inspect the `SUMMARY` termination record, then repeat with a larger budget and
the same directory. Once completed, a third process with
`--require-cache-reuse` validates the persistent cache path.

## Current indivisible phases

One fixed-shape routing batch is now the JAX execution and synchronization
boundary. Trace/lower/compile and each dispatched batch remain internally
indivisible, but their shape is bounded independently of the total destination
count. The executable is compiled once and reused. Each completed batch is
transferred, validated, persisted, and made resumable before the next dispatch;
the full dense `(destination groups, links)` arrays are never reconstructed by
the direct path. The legacy dense API remains available for small examples, and
explicit dense materialization refuses requests above a caller-supplied memory
limit.
Origin-specific support checks the deadline between bounded origin chunks and
atomically commits each completed destination group. Resume validates those
group fragments and continues with the first missing group; only an unfinished
origin chunk or group is repeated. Planning and temporal assembly are likewise
persistently resumable. An individual CPU reachability chunk and a dispatched
JAX backend operation remain the smallest internally indivisible units, so
their sizes and the safety margin still bound practical deadline overshoot.
