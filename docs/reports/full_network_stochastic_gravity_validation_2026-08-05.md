# Full-network stochastic gravity validation — 2026-08-05

## Purpose and assessment

This validation tested whether deterministic nested uniform sampling of
persisted routing shards could combine the memory safety of sequential
streaming with a useful runtime–accuracy tradeoff for gravity optimization.

The memory-bounded design succeeded. The sampled evaluations retained only one
routing shard at a time and reduced internal peak RSS from 98.67 GiB for the
exact displaced-point reference to about 8.5 GiB.

The sampling strategy was unsuccessful for its intended optimization use. At
10% effort it was faster but retained a 19.2% relative gradient error. At 25%
effort it was slower than exact while gradient error remained 17.2%. Uniform
shard-count reduction is therefore an experimental diagnostic capability, not
a validated drop-in replacement for exact optimization gradients.

## Provenance and configuration

The jobs used public revision
`55ed81601d0d786f80b333f4e2b0474505c6a472` and reused all 234 persisted
routing shards, prepared full-network inputs, a compatible exact
control-variate anchor, and an existing exact displaced-point reference. The
network, assignment, compact OD layout, measurement mapping, routing,
likelihood, parameter, and numerical-precision settings were identical.

Both sampled jobs used seed `20260722`, deterministic nested selection,
anchoring, direct persisted-shard streaming, sequential forward and reverse
passes, routing concurrency one, resident-shard limit one, no prepared-batch
retention, no exact recomputation, and no optimization. Each ran in a separate
exclusive 128 GiB Slurm allocation with a 110 GiB RSS ceiling and an 8 GiB
safety margin. They were submitted concurrently on different nodes and
overlapped for 8 minutes 5 seconds.

The exact displaced reference was not rerun: objective 13,361,880, runtime
3,176.25 seconds, and peak RSS 98.67 GiB.

## Results

| Quantity | Exact reference | 10% request | 25% request |
|---|---:|---:|---:|
| Slurm job | existing reference | 65964974 | 65964975 |
| Node | — | `jst195` | `jst189` |
| Selected shards | 234 | 24 | 59 |
| Realized effort | 100% | 10.2564% | 25.2137% |
| Objective | 13,361,880 | 13,361,850 | 13,361,677 |
| Objective absolute error | — | 30 | 203 |
| Objective relative error | — | 0.0002245% | 0.001519% |
| Measurement relative error | — | 0.4461% | 0.2438% |
| Gradient relative norm error | — | 19.2019% | 17.2077% |
| Gradient cosine | — | 0.998700 | 0.999234 |
| Maximum gradient absolute error | — | 44,459.78 | 42,586.39 |
| Evaluation runtime | 3,176.25 s | 1,427.48 s | 3,475.87 s |
| Speedup versus exact | 1.000 | 2.225 | 0.9138 |
| Internal peak RSS | 98.67 GiB | about 8.53 GiB | about 8.58 GiB |
| Slurm-observed peak RSS | 98.67 GiB | about 19.72 GiB | about 19.76 GiB |
| Measurement coverage | — | 0.33586 | 0.40893 |
| Maximum shard influence | — | 0.10760 | 0.04675 |
| Measurement SE indicator | — | 9.23394 | 3.41801 |
| Gradient SE indicator | — | 9.23649 | 3.42986 |
| Quality status | exact | `poor` | `poor` |

Selection fingerprints were
`a3bce0f044977b31022ede3f8adf66583a68ecdfe3ef23c2f43540a22cd1a99a`
at 10% and
`0ef4df0104d42105610a69c94b33fdc7c0278818d2775d42b4911b1736e0ec06`
at 25%. The 24-shard selection was a strict prefix and subset of the 59-shard
selection.

The 10% setup, forward, and reverse times were respectively 135.20, 900.70,
and 523.67 seconds; total evaluation time was 1,427.48 seconds and Slurm elapsed
time was 1,601 seconds. At 25% they were 133.47, 2,202.34, and 1,270.42
seconds; total evaluation time was 3,475.87 seconds and Slurm elapsed time was
3,649 seconds.

## Interpretation

Peak memory stayed approximately constant as effort rose and far below the
exact evaluator's peak. Sequential residency and RSS admission therefore met
their memory-safety objective.

Runtime, however, scaled almost directly with sampled work. Moving from 24 to
59 shards multiplied shard count by 2.46 and evaluation time by about 2.44,
while gradient relative error improved only from 19.2% to 17.2%. The 25%
evaluation was approximately 9.4% slower than exact. Merely increasing the
uniform sampling percentage is not a reasonable route to an accurate, faster
optimizer at this full-network point.

Objective agreement and gradient direction are insufficient acceptance
criteria. Both objective values were close and both gradient cosines exceeded
0.998, yet gradient magnitude errors remained material. The 25% objective
estimate was also less accurate than the nested 10% estimate, illustrating
that a larger deterministic sample need not improve every realized statistic.

The internal dispersion indicators and `quality.status` are diagnostics, not
confidence intervals or certified error bounds. They were pessimistic relative
to the measured reference errors, but their `poor` classification correctly
did not certify either result as optimization-quality.

## Timing and operational limitations

The setup, forward, reverse, evaluation, and Slurm times come from separate
instrumentation layers. Their boundaries need not include identical scheduler,
allocator, synchronization, initialization, and teardown costs. Only total
evaluation times are used for direct speedup comparisons. The jobs overlapped
in wall-clock time but ran on different exclusive nodes.

The supplied final validation record does not contain identifiers or failure
modes for earlier failed submissions. They are therefore not reconstructed
here. This report records only jobs 65964974 and 65964975 and the existing exact
reference.

## Recommendation

Do not use deterministic nested uniform persisted-shard sampling as a
transparent replacement for exact full-network gradients in a
precision-oriented deterministic optimizer. Do not respond by merely changing
the uniform effort percentage. Callers using the capability experimentally
must independently validate gradient accuracy. Effort 100 continues to
delegate to the established exact backend and remains required for exact
evaluation.

This conclusion is specific to uniform persisted-shard sampling with the
sequential two-pass evaluator. It does not establish that every stochastic
routing method must fail. Materially different future approaches could include
stratified or influence-aware sampling, variance reduction across iterations,
gradient calibration, or an optimizer explicitly designed for noisy gradients.
