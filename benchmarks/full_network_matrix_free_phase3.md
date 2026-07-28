# Full-network matrix-free feasibility: Phase 3

## Outcome

This phase adds a replayable, memory-budgeted two-pass destination provider and
benchmarks the existing fixed-routing loader across all six representative TPG
full-network destination sizes. No full routing cache, measurement operator,
Jacobian, or MAP optimization was constructed.

## Replayable provider

`replayable_streamed_measurement_value_and_grad` calls a destination provider
twice. During the first call it accumulates the complete measurement prediction
and retains only compact free-parameter index signatures. During the second it
requires the same groups and ordering, constructs each local VJP independently,
and scatters its result into the global gradient. It does not retain predictor
closures or routing arrays from the first pass.

Every yielded group declares its persistent byte requirement. A configured
ceiling rejects the group before prediction. Negative byte counts, duplicate or
overlapping parameters, changed replay order, missing groups, extra groups,
invalid indices, wrong prediction shapes, and dtype mismatches are rejected.
The byte declaration covers destination-local persistent state; a production
driver must add process baseline and compiler headroom when choosing its limit.

## Representative full-network samples

The opt-in orchestrator runs each OD shape in a fresh subprocess. This prevents
compiled executables for six different shapes from accumulating in one process
and makes the 24 GiB ceiling meaningful. Each child repeats global preparation,
so its total process time is not an estimate of a shared-preparation production
driver. The complete record is
`benchmarks/full_network_matrix_free_samples.json`; individual records are in
`benchmarks/full_network_matrix_free_samples/`.

| Free cells | Routing preparation (s) | Warm forward (s) | Warm value--gradient (s) | Peak RSS (GiB) |
|---:|---:|---:|---:|---:|
| 9 | 38.11 | 5.987 | 12.221 | 16.26 |
| 137 | 39.40 | 5.977 | 12.105 | 17.02 |
| 276 | 38.08 | 5.988 | 12.379 | 18.16 |
| 465 | 41.08 | 6.075 | 12.093 | 17.85 |
| 770 | 38.52 | 6.057 | 12.155 | 16.02 |
| 1,181 | 37.46 | 5.994 | 12.068 | 17.42 |

The median warm forward time is 5.991 s and median ordinary value-and-gradient
time is 12.130 s. Median routing preparation is 38.316 s. All samples remained
under 24 GiB. Runtime is effectively independent of local OD count over a
131-fold range, demonstrating that graph traversal dominates demand injection.
The variation in peak RSS is not monotone in OD count and is primarily compiler
and allocator noise.

## Full-network estimates

Using the sample median for all 1,898 destinations gives:

- one ordinary value-and-gradient sweep: 23,023 s, or 6.40 hours;
- exact two-pass streaming (forward accumulation plus recomputed VJP):
  34,394 s, or 9.55 hours per objective-and-gradient evaluation;
- serial routing preparation without persistence: about 20.2 hours.

The sampled warm range gives a narrow ordinary-sweep range of approximately
6.36--6.53 hours and a two-pass range close to 9.5--9.7 hours. These are local
12-core CPU estimates and exclude optimizer overhead. Since optimizers normally
request multiple objective evaluations per iteration, even one iteration would
be impractical sequentially.

## Decision

The replayable provider solves destination-state retention and exact nonlinear
likelihood semantics, but the existing loader is rejected as a sequential
full-network MAP engine. Further work must reduce graph traversal cost or
parallelize it. The next useful phase is measurement-structure analysis and
direct boarding/alighting accumulation, followed by an explicit adjoint or
rematerialized loader benchmark and a per-worker server memory model.
