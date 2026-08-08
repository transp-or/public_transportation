# Destination-batched routing integration

The direct scheduled construction path can now consume the existing persistent
`ShardedFixedRoutingInputs` representation. Routing sharding changes only the
computational representation: destination-specific route-choice masks and
probabilities are numerically identical to the legacy dense preparation. It is
independent of the gravity or other demand model.

The stable source interface exposes destination and link counts, dtype, theta,
assignment identity, deterministic contiguous batch descriptors, validated
batch loading and iteration, compatibility validation, and an overall
fingerprint. Downstream origin-support and numerical construction retain at
most one routing batch, reuse it for contiguous groups, and release it before
loading the next. Global destination-by-link arrays are not assembled. The
explicit `materialize_sharded_fixed_routing_dense` utility is restricted to
small compatibility tests and refuses an insufficient memory limit.

## Public benchmark

Command:

```text
UV_CACHE_DIR=/tmp/public-transportation-uv-cache \
  uv run --frozen python benchmarks/benchmark_sharded_fixed_routing.py \
  --output /tmp/sharded-routing-integration.json
```

On the packaged `simple_example_02` CPU case (7 destination groups, 5,846
links), the legacy dense routing preparation took 0.170 seconds. Persistent
two-group routing batches took 0.239 seconds cold, including one compilation,
and 0.0029 seconds to validate/reuse all four batches. Compilation took 0.117
seconds; peak process RSS was 459 MB and the routing files occupied 215 kB.
Effective masks agreed exactly and the largest probability difference was
zero. The benchmark also exercised lazy forward and adjoint products without a
global measurement matrix. These small-example measurements validate behavior;
they do not predict private full-network runtime or backend workspace.

## Identity and deadline policy

Scientific identity covers the graph arrays, base costs, group destinations and
masks, OD/group assignment inputs, theta, dtype, package and routing
implementation schema. Resource settings determine a deterministic batch plan.
The plan fingerprint selects a separate checkpoint subdirectory, favoring
simple reliable invalidation over cross-layout reuse. Different batch layouts
must produce the same probabilities but do not share batch files.

The direct activation order remains final-artifact reuse, economic activation
policy, routing-plan validation/resume, missing routing batches, support,
numerical shards, and temporal assembly. A routing deadline stop reports the
completed and total batch counts, next batch, reusable checkpoint, predicted
next duration, overshoot, and the trace/lower/compile/execute/synchronize/
transfer/persistence subphase. A subsequent call resumes at the first missing
batch.

One fixed-shape JAX compilation and one batch execution remain internally
indivisible. Batch size, retained/temporary byte ceilings, workspace
multiplier, workers, resident-batch limit and safety margin are explicit. No
platform-name heuristic or all-available-memory policy is used. A future
scheduled or frequency-based routing backend can implement the same routing
source boundary without changing the demand model or temporal operator
contract.
