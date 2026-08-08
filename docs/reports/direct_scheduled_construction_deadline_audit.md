# Direct scheduled construction deadline audit

This audit follows the production call graph beginning at
`activate_direct_scheduled_temporal_operator`. It records the state before the
shared construction-control contract was introduced.

| Phase | Potentially long work | Deadline before hardening | Progress | Resumable output | Safe pre-start guard |
|---|---|---:|---:|---:|---:|
| Cache validation | Read and hash every temporal block | No | No | Completed artifact only | No |
| Routing preparation | JAX trace, lower, compile, execute and synchronize | Partial; only effective with profiling callback | Diagnostics only | No | No duration guard |
| Support discovery | Origin-support kernels, CSC conversion, OD/group loops | No | No | No | No |
| Planning | Pattern construction, shard packing, dispatch/memory estimates | No | No | Persisted only as part of the later manifest | No |
| Shard validation | Open and hash all referenced shards | No | No | Valid shards reused | No |
| Shard construction | Group, OD-chunk and edge-block loops; device dispatch and transfer | No | One event after each shard | Atomic shard plus periodic manifest | No |
| Temporal assembly | Read all shards and build one in-memory triplet dictionary | No | No | No | No |
| Final validation | Canonicalize all blocks and validate aggregate metadata | No | No | No | No |
| Persistence | Serialize every block, offset and manifest to staging | No | No | Atomic final artifact only | No |

The shard payload and manifest writers were already atomic. Completed shards
could survive interruption, but routing, support discovery, planning and
temporal assembly were recomputed. The final artifact could not be mistaken for
complete because publication used an atomic directory rename. The largest
remaining risks were therefore unbounded indivisible routing compilation,
opaque support/planning, and non-resumable temporal assembly.

The hardening design uses one monotonic `ConstructionDeadline`, structured
`ConstructionTermination` records, and versioned progress events throughout the
call graph. Indivisible operations must be guarded using a conservative
prediction and report measured overshoot. Incremental phases commit work before
emitting completion and stop only at valid resumable boundaries.

## Implemented checkpoint boundaries

The hardened workflow now persists and validates destination-batched routing,
materialized support, the deterministic shard plan, numerical shards, and one
temporal fragment per numerical shard. Each fixed-shape routing batch is
synchronized, transferred, content-hashed and atomically published before the
next batch. Resume loads validated batches without constructing global dense
routing arrays, bypasses support discovery and planning
when their checkpoints are compatible, reuses every valid numerical shard, and
continues temporal assembly at the first missing fragment. Abandoned temporary
files are removed and incompatible support, routing, shard, fragment, and final
artifact payloads are rejected or quarantined.

Origin-support discovery now checks the shared deadline between bounded origin
chunks and atomically commits one validated, provenance-bound fragment per
completed destination group. A resumed process reuses those fragments and
continues at the first missing group; corrupt or incompatible fragments are
quarantined and recomputed. The aggregate support checkpoint and plan are still
written after discovery completes. Consequently, only the currently unfinished
CPU chunk or destination group can be repeated. JAX compiler calls and one
bounded routing-batch execution remain indivisible backend operations, with
predicted-start guards, per-batch persistent output, and measured overshoot.
