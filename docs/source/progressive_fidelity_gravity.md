# Progressive-fidelity gravity objective and gradient

Progressive fidelity reduces the routing work used by one gravity-model
objective and gradient evaluation. It is intended for exploratory steps of a
future trust-region optimizer. It does not change the gravity specification or
the final statistical objective.

## Basic use

```python
from public_transportation.inference.gravity import (
    GravityFidelityRequest,
    gravity_value_and_gradient_progressive,
)

result = gravity_value_and_gradient_progressive(
    raw_parameters,
    problem=problem,
    fidelity=GravityFidelityRequest(
        effort_percent=10,
        seed=1234,
    ),
)

print(result.evaluation.objective)
print(result.gradient)
print(result.quality.quality_score)
print(result.quality.gradient_relative_error_estimate)
```

`effort_percent` must be finite and lie in `[1, 100]`. It controls the
requested fraction of expensive routing work, not the number of optimizer
iterations. Diagnostics report the realized support fraction because shard
sizes and mandatory stratum representation can make it differ from the
request.

Only effort 100 is called **exact**. It delegates to the established complete
adjoint calculation, selects every shard once with weight one, reports quality
one, and reports zero approximation error.

## Approximate nonlinear objective

Routing-shard contributions are additive:

```text
mu = fixed offset + sum of shard contributions.
```

The likelihood is nonlinear in `mu`. The implementation therefore never
calculates separate shard likelihoods and scales their sum. Below effort 100 it
instead:

1. selects routing shards;
2. expands their additive contributions to estimate the complete count vector;
3. adds the fixed offset exactly once;
4. evaluates the likelihood at that approximate complete vector;
5. evaluates the likelihood cotangent there; and
6. applies the same selected and weighted transpose products.

The returned gradient is consequently the analytical gradient of the reported
approximate objective. Fixed offsets, gravity transformations, regularization,
negative-binomial dispersion terms, and other direct parameter derivatives are
calculated exactly.

An unbiased additive count estimator does **not** make the nonlinear likelihood
or its gradient unbiased. All sub-100 error diagnostics are estimates, not
guarantees.

## Deterministic nested selection

Each routing shard receives a stable hash-derived uniform value based on the
problem identity, seed, stratum, and immutable shard identifier. A shard is
selected when this value falls below the requested effort fraction. If a
stratum would be empty, its minimum-hash shard is retained.

For fixed problem identity and seed, selections are nested as effort increases.
The implementation does not use filesystem order, Python object identity, or
process-randomized hashes.

For a stratum containing `n` shards and sampling threshold `p`, the inclusion
probability is

```text
pi = p + (1 - p)**n / n.
```

Selected additive contributions receive the Horvitz--Thompson expansion weight
`1 / pi`. The planner records requested and effective effort, selected and total
support entries, routing bytes, probabilities, weights, seed, and selection
digest.

Persisted sharded operators provide load-free predicted-work metadata. Planning
a low-effort request therefore does not load every routing cache shard. Dense
reference operators are partitioned by OD columns for public tests and small
examples. Other sparse representations require an explicit additive fidelity
context; they are never silently densified.

## Quality estimates

Selected shards are assigned deterministically to replicate groups. The normal
forward and reverse calculations simultaneously accumulate group estimates, so
quality calculation neither evaluates the exact operator nor repeats routing
products.

The report includes estimated:

- objective standard and relative errors;
- predicted-count relative error;
- gradient error norm and relative error;
- a conservative gradient-cosine diagnostic;
- calibration-measurement coverage; and
- effective replicate-group sample size.

The gradient estimate is linearized around the reported approximate count
vector. Replicate groups can share systematic sampling error, particularly at
very low effort. The quality score therefore includes a realized-support factor
in addition to coverage, objective uncertainty, gradient uncertainty, and
direction agreement. It remains a summary diagnostic, not an error bound.

Fewer than two nonempty quality groups produces `insufficient_sample`, score
zero, unavailable error estimates, and an explicit warning. Missing calibration
measurements also lowers the score and produces a warning.

## Control-variate anchors

An already completed exact or documented-fidelity result can be reused:

```python
from public_transportation.inference.gravity import build_gravity_fidelity_anchor

anchor = build_gravity_fidelity_anchor(
    anchor_parameters,
    problem=problem,
    result=completed_result,
)

nearby = gravity_value_and_gradient_progressive(
    new_parameters,
    problem=problem,
    fidelity=GravityFidelityRequest(
        effort_percent=20,
        seed=1234,
        anchor=anchor,
    ),
)
```

The estimator then samples routing differences relative to the anchor. At the
anchor parameters it reproduces the stored prediction. Nearby parameters often
have much lower sampling variance. Anchors record parameter, prediction,
operator, routing-cache, schema, package, fidelity, coverage, and creation-time
identities. Incompatible anchors are rejected.

## Progress, checkpoints, and interruption

`GravityFidelityExecution` optionally supplies a progress callback, absolute
deadline, cancellation callback, checkpoint path, and resume request.
Checkpoints are written atomically after completed shard products and contain
the exact subset, weights, parameter and anchor digests, forward and reverse
accumulators, measurement coverage, and quality-group state.

Cancellation or deadline expiration raises
`GravityFidelityEvaluationInterrupted`. No partial objective or gradient is
published as complete. Resume validates all identities and continues with the
same subset and weights.

## Initial trust-region policy

The operator reports evidence; a future optimizer must make acceptance
decisions. A conservative initial policy is:

- permit exploratory trial steps only when coverage is 1, quality score is at
  least 0.70, the estimated gradient relative error is at most 0.10, and the
  cosine estimate is at least 0.90;
- otherwise increase fidelity and recompute;
- refresh an anchor when parameter displacement exceeds the optimizer's current
  trust radius or anchored quality deteriorates;
- compare predicted and higher-fidelity objective reduction before enlarging a
  trust region;
- require effort 100 before declaring convergence or publishing final
  estimates.

These thresholds are starting policies, not universal constants. They must be
recalibrated on representative public and private applications.

## Public benchmark

Run:

```bash
uv run python benchmarks/benchmark_progressive_gravity.py \
  --output benchmarks/progressive_gravity_public.json
```

The committed public synthetic benchmark uses 200 OD cells, 60 measurements,
32 dense column shards, and efforts `1, 2, 5, 10, 25, 50, 75, 100`.

The benchmark demonstrates improving accuracy but **no runtime speedup** on this
small dense problem. Median times are approximately 26--38 ms across all
efforts, and the exact dense product takes about 27 ms. Dispatch, Python loops,
and quality diagnostics dominate the inexpensive dense matrix multiplication.

At about 76% realized support, the observed gradient relative error is about
3.1%, gradient cosine similarity is 0.99994, predicted-count relative error is
about 15%, and quality score is about 0.77. At very low effort, the gradient can
point in the wrong direction and the quality score is correspondingly low.

These measurements do not establish a speedup for a large persisted routing
cache. The intended benefit must be benchmarked where each omitted routing
shard represents genuinely expensive assignment work. Fixed per-evaluation
overhead will limit scaling at the lowest effort levels.

### Larger persisted-shard benchmark

The reproducible persisted-operator benchmark can be run with:

```bash
uv run python benchmarks/benchmark_progressive_gravity_sharded.py \
  --output benchmarks/progressive_gravity_sharded_large.json
```

The committed CPU result uses 512 graph nodes, 1,021 links, 64 destination
groups, 2,048 OD cells, 256 measurements, and 32 routing shards. Preparation
and compilation warm-up are excluded, and each steady-state measurement is
repeated three times with an identical deterministic shard selection. Exact
evaluation takes 60.8 ms.

The realized 3.125% and 9.375% selections are respectively 1.80 and 1.27 times
faster than exact evaluation, but their gradient relative errors are 63% and
31%; they are unsuitable for optimization decisions. At 31.25% realized work,
gradient error falls to 6.7%, but execution is 23% slower than exact. At 75%
realized work, gradient error is 0.59% and predicted-count error is 2.16%, but
execution is 2.12 times slower than exact.

This result confirms that the current sequential one-shard adapter cannot
provide a useful steady-state speedup, even when routing is persisted and the
problem is substantially larger than the dense smoke benchmark. The next
performance step is therefore a batched or concurrent selected-shard executor;
increasing the effort schedule alone will not solve the overhead.

## Current limitations

- Progressive shard products currently execute through the public one-shard
  adapter. The existing bounded shard residency is preserved, but a dedicated
  selected-shard concurrent executor is still needed to recover all available
  server parallelism.
- Current automatic stratification uses metadata already exposed by the
  operator. Richer time, origin, destination, line, demand-mass, and measurement
  metadata would improve representativeness.
- Measurement coverage is evaluated at the requested parameter vector; a zero
  numerical contribution is not a structural support certificate.
- Replicate error estimates can miss shared systematic bias.
- Effort 100 remains mandatory for final validation and convergence.

## Parallel partial-execution foundation

The first three phases of the parallel redesign provide contracts and planning
without replacing the active numerical backend. The public
`parallel_partial_execution` module defines immutable routing work units,
microshard and selected-execution plans, fixed execution batches, observations,
and a versioned JSON representation. Invalid costs, duplicate identities,
overlapping destination groups, and incomplete selected-work batching are
rejected at their contract boundaries.

`ShardedWorkInstrumentation` can be installed as the existing sharded
operator's progress callback. It records one observation per executed batch,
including routing bytes, destination groups, load-and-assembly time,
transfer-and-execution time, accumulation time, cache activity, and execution
lane. It observes the current exact path and does not alter its arithmetic.

`routing_group_work_units` derives scheduling metadata without loading
persisted routing shards. Its documented provisional cost model combines group
count, structural link support, active OD cells, retained routing bytes, and
measurement support. `build_balanced_microshard_plan` then applies a
deterministic longest-processing-time partition. The target microshard count
is configurable; future public scaling measurements will calibrate both the
cost weights and the appropriate count before a server default is selected.

These facilities are preparatory. Persistent workers, fixed-shape compiled
batches, and dynamic scheduling are introduced by subsequent phases. The
current executor therefore remains the production exact implementation.

### Persistent parallel batch executor

The next three implementation phases add a public comparison backend while
leaving the established exact backend available. `PersistentParallelRoutingExecutor`
owns a configurable pool for its full lifetime, rejects overlapping products
on the same executor, propagates worker failures, and checks cancellation and
deadlines at safe batch boundaries. The requested worker count and the declared
threads per worker are explicit report fields; the implementation does not
silently modify global JAX threading settings.

`plan_fixed_shape_routing_batches` packs complete microshards into a small
configured family of destination-group shapes. The final batch is zero padded,
and the sharded operator caches one compiled forward and reverse executable per
used shape. This prevents arbitrary shape-dependent recompilation. Executing
all microshards through these products reproduces the existing exact forward
and reverse calculations within the documented floating-point tolerance.

Scheduling is dynamic: the coordinator orders batches by decreasing predicted
cost, initially fills every worker lane, and dispatches the next batch whenever
a lane becomes available. Contributions are nevertheless accumulated in
canonical batch order, so worker completion order does not define the numerical
result. Per-batch observations expose queue wait, execution time, worker lane,
predicted cost, selected group count, and padded shape. This backend is not yet
the production gravity path; the next exact-equivalence and performance phase
will determine whether it should replace the current 100% executor.

The committed public scaling probe uses 2,048 nodes, 4,093 links, 256
destination groups and microshards, 8,192 OD cells, and 512 measurements. All
worker configurations reproduce the existing exact products (maximum forward
absolute difference (1.84\times10^{-4}), exact reverse agreement). The
one-worker comparison backend is slower, while eight worker lanes take 182.5
ms versus 183.9 ms for the established exact backend, a statistically modest
1.008 speedup. Thus the architecture uses all requested lanes and removes the
large regression seen on the small dispatch-dominated case, but this public
probe does not yet demonstrate material parallel scaling. Phase 7 must retain
the existing backend unless larger public exact-equivalence benchmarks show a
clear advantage.

### Phase 7 exact promotion gate

The parallel comparison backend is assessed by
`assess_parallel_exact_gate`. Promotion requires all of the following:

- forward and reverse products within declared absolute tolerances;
- complete gravity objective within the declared absolute tolerance;
- gravity gradient within the declared relative tolerance;
- every requested worker lane performing routing work; and
- at least a 1.10 warm steady-state speedup over the established exact
  objective-and-gradient calculation.

The five-repetition public gate used the same 2,048-node problem and expanded
the compiled group shapes through 64. With eight workers, the complete gravity
objective and gradient took 215.9 ms versus 218.2 ms for the existing exact
backend, a 1.010 speedup. Numerical equivalence passed: objective error was
zero, gradient relative error was (1.13\times10^{-7}), forward maximum
absolute error was (1.83\times10^{-4}), and the reverse error was zero. All
eight lanes performed work. Performance nevertheless failed the 1.10 gate, so
the recorded recommendation is `retain_existing` and the production exact path
has not changed.

### Fixed-budget parallel partial routing

The experimental sub-100% backend now selects a fixed number of approximately
cost-balanced microshards within each stratum. A stable seeded order makes
selections nested as effort rises. Selecting (k) of (n) units in a stratum
gives every unit first-order inclusion probability (k/n) and expansion weight
(n/k). Requested and realized predicted work are both reported. Effort 100 is
rejected by the approximate adapter and continues to use the established exact
backend.

Expansion weights scale each selected destination group's OD contribution once.
They do not scale individual link probabilities, which would incorrectly
compound a weight along multi-link paths. The same selected groups and weights
are used by the forward product and its reverse adjoint. The complete nonlinear
likelihood is evaluated once at the resulting approximate measurement mean.

Forward execution retains prepared selected-batch arrays under a configurable
byte ceiling. The reverse pass consumes those exact arrays and then releases
the evaluation. Unknown, duplicate, or already released evaluation identities
are rejected. If the ceiling cannot hold every selected batch, missing batches
are prepared again without changing the numerical result.

The public five-repetition benchmark uses the same 2,048-node, 256-microshard
problem and eight workers. Its measured time--accuracy trade-off is:

| Requested effort | Realized effort | Speedup | Gradient relative error | Count relative error |
|---:|---:|---:|---:|---:|
| 10% | 10.16% | 4.35 | 8.03% | 14.89% |
| 25% | 25% | 3.12 | 8.31% | 8.74% |
| 50% | 50% | 1.67 | 3.31% | 4.55% |
| 75% | 75% | 1.28 | 1.15% | 1.68% |

All four levels pass the committed public partial-performance gate. Prepared
batch retention improves forward-plus-reverse time by approximately 18% at 25%
effort, 32% at 50%, and 34% at 75%. These results justify retaining the feature
as an experimental partial backend. They do not authorize approximate final
results: convergence and reporting still require the established exact 100%
calculation.

### Exact control-variate anchors

`ParallelGravityAnchor` stores an exact parameter vector, demand vector, routed
measurement contribution, complete measurement mean, gradient, objective, and
strict problem/operator identities. Its JSON representation is versioned and
contains no routing-cache payload. The identity includes observations,
calibration mask, likelihood, scale and floor settings, parameter layout,
assignment, compact OD layout, and measurement mapping. Mismatches are rejected
before evaluation.

For a nearby parameter vector, the partial backend evaluates

\[
A f(\beta_0) + \widehat{A\,[f(\beta)-f(\beta_0)]}.
\]

At the anchor itself, the cached exact objective and gradient are returned.
Away from it, the selected forward and reverse products remain a matched
adjoint of the anchored approximate objective. Creating an anchor performs one
exact forward and reverse calculation; it does not repeat the exact forward
merely to construct metadata.

On the public benchmark with parameter displacement
\((0.03,-0.02,0.01)\), the anchor substantially improves accuracy:

| Effort | Gradient error, unanchored | Gradient error, anchored | Count error, anchored | Anchored speedup |
|---:|---:|---:|---:|---:|
| 25% | 8.31% | 0.60% | 0.013% | 2.26 |
| 50% | 3.31% | 0.24% | 0.006% | 1.41 |
| 75% | 1.15% | 0.094% | 0.002% | 1.11 |

Anchoring introduces some fixed evaluation overhead, so its speedup is smaller
than the unanchored calculation. The accuracy gain is large enough to make
anchored low and medium effort the preferred experimental path for nearby
optimizer iterations. Exact evaluation remains required to create or refresh
the anchor and to verify convergence.
