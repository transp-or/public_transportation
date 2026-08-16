# Sharded fixed-routing measurement operators

The monolithic sparse builder is effective when its complete BCOO artifact and
the solver's CSR/CSC copies fit comfortably in memory. It is not the scalable
construction path: its reverse dynamic-programming kernel returns
`OD chunk × maximum local measurements`, including entries later discarded as
zero. For a large destination group this static output and the corresponding
node state are excessive even when the realized operator is very sparse.

## Positive-boarding support is checked first

Direct scheduled activation requires the observation vector. Before routing,
the preflight verifies that every positive boarding row has a mapped access
link whose tail is an active free or fixed-positive assignment origin. It
reports rows whose origin/time domain is absent, whose corresponding OD cells
are all fixed at zero, or whose mapped access links have no active origin.

After origin-specific routing support is available, a second exact check runs
before planning, lowering, compilation, or numerical shard generation. It
rejects any remaining positive boarding row that no retained free or
fixed-positive route can reach. Valid cached operators are checked from their
realized nonzero rows and fixed offset before reuse.

Failures raise `UnsupportedPositiveBoardingError`. The full row-level report is
written atomically to
`positive_boarding_support_preflight.json` in the identity-specific checkpoint
directory. Optional fixed-zero reason provenance lets callers report the exact
preprocessing rules responsible for excluded origin cells.

The sharded builder instead partitions each destination group's structurally
eligible measurements into fixed-width blocks. Its largest construction state
is proportional to

```text
OD chunk × (graph nodes + supported mapping-edge block)
```

and never to `OD chunk × all local measurements` or to the full logical
operator. Preflight estimates this state from the actual graph, dtype, OD chunk
and measurement block, and rejects it before JAX lowering when it exceeds the
per-worker budget.

## Support and construction

Support discovery combines the compact OD layout, destination-group
membership, effective routing links and link-to-measurement mapping. Frozen
zero cells are absent from the selected OD indices. Positive fixed cells are
evaluated but enter only the shard's sparse fixed-offset contribution. Every
group's sorted measurement support is split into configurable blocks. OD cells
with identical origin-specific support are grouped into deterministic support
patterns. Construction propagates origin-to-node reach probabilities and then
gathers only supported mapped links. It does not carry a measurement dimension
through every graph node. Kernel state is bounded by

```text
OD chunk × (graph nodes + supported mapping-edge block)
```

A support pattern is not a file or a solver matrix. The implementation keeps
four explicit layers:

1. `SupportPattern` describes reusable topology.
2. `ConstructionTask` describes bounded pattern/measurement computation.
3. `StorageShardPlan` deterministically packs many tasks to a payload target.
4. `SparseShardIdentity` identifies the aggregate NPZ and solver matrix.

The sharded builder defaults to `ShardedConstructionConfig.support_discovery_mode="compact"`.
It consumes each destination-group support result as soon as its checkpoint is
written, forms deterministic support patterns, and releases the group arrays
before the next group. It therefore avoids constructing the global
origin-by-measurement CSR matrices used by the `"reference"` mode. The
reference mode remains available for small-fixture validation; the two modes
have separate provenance and cannot silently reuse one another's checkpoints
or numerical shards.

When `ShardedConstructionConfig.workers` is greater than one, exact support
groups are computed in a bounded thread pool. Each worker owns its routing
shard reader; only the parent writes group checkpoints, emits progress, and
publishes groups in canonical order. Worker count is deliberately absent from
scientific provenance, so serial and parallel runs share compatible support
checkpoints and produce the same patterns.

The optimized worker path uses a fixed-shape JAX boolean reachability scan,
compiled once before dispatch. It retains the NumPy implementation for the
serial/reference path and validates the two paths against the same support
matrices and sparse operator products on the public small example.

Packing is stable under construction completion order. Contributions are
canonicalized after aggregation, so duplicate row/column contributions and
fixed offsets are accumulated exactly once. Defaults target 2,048 candidate
nonzeros per storage shard, with hard limits on payload, patterns, shard count,
filesystem work, manifest bytes, and sparse calls per product. A
kernel-memory-safe but operationally pathological plan is unsafe before JAX
lowering.

Construction no longer evaluates each support pattern independently. Within
each storage shard, compatible tasks are grouped by destination group and
routing state. One batch traverses the graph for the union of its bounded
origins and supported mapping edges, then filters every OD column back to its
exact origin-specific support. This avoids `OD chunk × all local measurements`
while greatly reducing graph traversals and synchronization calls. The group
measurement unions, mapped-edge selections, and origin-support metadata are
computed once and reused by preflight, planning, workers, and diagnostics.
Preflight reports batch temporary memory and estimated dispatch count.

Runtime metrics report batches, dispatches, synchronizations, support-edge
blocks, origins and edges per dispatch, output values, timing quantiles, group
timings, padded-buffer allocations, and routing-array reuse.

Every returned OD–measurement cell is structurally supported. Metrics
distinguish supported candidates, realized numerical entries and entries
discarded by the configured numerical tolerance.

The block sizes to benchmark first are 128, 256, 512, 1,024 and 2,048. The
appropriate value is hardware and graph dependent. `plan_sharded_fixed_routing_operator`
must be called before construction when exploring a new scale.

## Cache layout and resumability

A cache directory contains:

```text
manifest.json
shards/
  storage-000000.npz
  ...
```

The manifest records global dimensions, aggregate storage-shard identities,
completed storage shards, aggregate nonzeros, block configuration, schema and full
provenance. Provenance includes assignment inputs, mapping, OD and compact
layouts, routing parameter, dtype and zero tolerance.

The persisted identity separates scientific inputs from execution and packing
settings. Scientific provenance also records graph and frozen-cell
fingerprints, support-definition and numerical-kernel versions, and the
serialization version. The manifest's separate execution provenance records
workers, chunk/block sizes, storage limits, and compression. Changing packing
therefore rebuilds measurement shards without discarding a compatible
origin-support checkpoint; changing the graph, OD layout, frozen cells, theta,
mapping, or support definition invalidates the affected downstream artifacts.
Malformed manifests and checkpoints are treated as incompatible and quarantined
rather than reused.

Each shard stores sorted global measurement rows, canonical local CSR arrays,
sparse fixed-offset entries, construction metrics, a numerical content hash
and the manifest provenance hash. Writes use a temporary sibling followed by
atomic replacement. Each aggregate NPZ is written once. The manifest is
checkpointed after a configurable number of completed storage shards and at
finalization; it is not rewritten after each construction task. Startup scans
and validates existing shards, including valid files newer than the last
checkpoint. Missing and
invalid shards are rebuilt independently; valid shards survive interruption
and are reused. Completion order does not change the manifest ordering or
solver accumulation order.

Uncompressed NPZ is the default for shards because cache construction and load
latency matter more than maximum compression. Compressed NPZ remains an
explicit compatible option and is covered by format tests. Separate memory-
mapped NPY arrays are not yet the default; their operational benefit should be
measured on genuinely large shards before a versioned format addition.

## Solver operator

`ShardedSparseLinearOperator` presents all shards through the standard
`matvec`/`rmatvec` protocol without constructing a global sparse matrix.

- Forward products apply local CSR and deterministically scatter-add into the
  global measurement vector.
- Transpose products gather global rows, apply the persistent local transpose,
  and accumulate into the free-OD gradient.
- Eager mode retains every shard for smaller problems.
- LRU mode bounds the number of resident shards for large problems.
- Supplying a solver memory budget selects eager loading when estimated
  resident storage fits and otherwise derives a bounded LRU shard count.
- Eager mode merges loaded shards once into one process-local CSR and transpose.
  Persistence remains sharded, but repeated products require one SciPy call.
  The merged representation is never written to disk.
- Products never evaluate routing or convert JAX arrays.
- Product, sparse-call, file-open, bytes-read, shard-load, cache-hit and eviction
  counters are exposed. Warm eager products perform no file I/O.

The fixed offset is assembled once by streaming shard offsets. The global
offset and one input/output vector are unavoidable global vectors, but no
global sparse matrix is allocated.

## Automatic selection

`select_sharded_fixed_routing_backend` distinguishes absent, partial and
complete caches. It uses the candidate-density upper bound, estimated disk
storage, kernel-memory and operational-plan preflight, expected products and cache reuse, and
measured construction/product timing when supplied. A complete compatible
cache is treated as a sunk cost. A partial compatible cache is resumable. A
cold one-use run may remain matrix-free when construction does not amortize.

The benchmark probes a complete manifest before preparing fixed routing or
running support planning. A compatible hit performs no routing, support
discovery, JAX lowering/compilation, or manifest write. Shard numerical hashes
are still validated as the eager or LRU operator loads them.

## Stopping and preconditioning

Diagonal scaling remains optional. `TRFLSMRConfig.success_policy` may be
`scipy`, `kkt`, or `both`. The default retains SciPy compatibility. Strict
production validation should use `both` and a declared `kkt_tolerance`; this
prevents relative-cost termination from being reported as adequate when the
physical projected-gradient norm remains large. Results report the SciPy
status, selected policy and whether the KKT requirement was satisfied.

## Current boundary

The implemented builder is deterministic and resumable. Measurement-shard
construction can use the bounded shared-executable worker pool; the parent
publishes completed shards in canonical order and owns the manifest. The
default remains one worker until full-network memory and throughput benchmarks
justify a larger value.
`analyze_fixed_routing_origin_support` provides the origin-specific boolean
reachability pass used by construction. For every selected OD cell it propagates
boolean reachability through positive-probability links in the fixed-routing
DAG and projects reachable mapped links into canonical measurement rows. Work
is bounded by configurable origin chunks. Summary-only mode reports counts and
reductions without materializing a potentially large global support matrix;
validated public examples may materialize canonical CSR support for a
no-false-negative comparison with the existing operator.

On the public Geneva benchmark, origin-specific analysis reduced 411,720
group-level candidates to exactly 4,788 supported entries, equal to the
realized monolithic operator. This was a 98.84% reduction, took approximately
0.714 seconds, and used an estimated 1.37 MB working state. After integration
with the forward-support kernel, cold shard preparation fell from 32.66 seconds
for the original group-blocked design to approximately 4.78 seconds. Kernel
synchronization fell from 32.30 seconds to 3.69 seconds. The completed 96-shard
cache validates and resumes in approximately 0.035 seconds; forward and
transpose products remain about 0.4 and 1.0 milliseconds, respectively.

After aggregate packing and group-compatible batching, the same public case
uses 96 construction tasks, 20 graph-traversal batches/dispatches, and three
persisted shards. Cold operator preparation is approximately 2.17 seconds,
including 1.11 seconds of support discovery and 0.83 seconds of synchronization.
A compatible-cache run performs zero routing or JAX work and completes the
whole benchmark process in approximately 1.17 seconds. The eager merged
operator gives warm forward and transpose medians of approximately 0.017 and
0.022 milliseconds and a bounded solver time of approximately 0.0015 seconds.
Numerical results remain within the documented tolerances, and strict `both`
policy correctly rejects the bounded non-KKT-converged solve.

The former one-pattern-per-file design was rejected after validation exposed
15,748 patterns for only about 158,219 candidate nonzeros. Aggregate packing
now targets tens to low hundreds of solver matrices for that shape and rejects
plans beyond the configured operational limits. These aggregate private
dimensions guide validation only; no private rows, identifiers, or numerical
operator values are stored in this repository.

Multi-process construction must still be addressed before claiming
full-network production readiness.

No conclusion about full-network readiness should be drawn solely from a
two-line benchmark. The full-network validation must first produce a safe
shard plan, including maximum block state, candidate-density bound, disk bound
and resumable shard count, without constructing the complete operator.
