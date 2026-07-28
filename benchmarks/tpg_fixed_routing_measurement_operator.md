# TPG fixed-routing measurement operator benchmark

Date: 2026-07-28. JAX float32 on the local CPU backend. The existing TPG
`two_lines_morning_time/profile_map.py` harness was used unchanged. Construction
uses the fused reverse-DP BCOO builder; the 512 result was measured without an
objective timing pass, and all other rows include three warm objective/gradient
measurements.

## Structure and construction

The operator shape is 3,690 × 15,772 (58,198,680 logical entries). It has
158,219 stored nonzeros, density 0.271860%, logical dense size 222.01 MiB, and
BCOO stored size 1.811 MiB. BCOO is therefore the appropriate automatic
representation.

| OD chunk | Calls | Compile (s) | Construction (s) | Device sync (s) | NumPy transfer (s) | Host peak (MiB) | Warm value+grad (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 184 | 0.133 | 107.168 | 105.587 | 0.0100 | 10.43 | 0.000823 |
| 256 | 124 | 0.136 | 73.982 | 72.504 | 0.0070 | 10.39 | 0.000804 |
| 512 | 113 | 0.132 | 66.657 | 65.199 | 0.0061 | 10.39 | — |
| 1024 | 113 | 0.139 | 66.574 | 65.113 | 0.0062 | 10.39 | 0.000808 |
| 2048 (largest tested safe) | 113 | 0.135 | 66.894 | 65.419 | 0.0061 | 10.40 | 0.000806 |

Chunk sizes of 512 or greater cover every destination group in one call, so
1024 is a conservative default and larger values do not improve construction.
The explicit compilation count is one for every build. Construction break-even
at the measured 1.94 s reference evaluation is about 34.3 evaluations. A valid
persistent cache removes construction, reducing map-problem construction from
110.65 s in the cache-seeding process to 0.70–0.74 s in fresh processes.

## Numerical validation

Across zero and two nonuniform parameter vectors, the maximum objective
difference was 0.00390625, likelihood difference 0.001953125, and absolute
gradient-coordinate difference 2.861e-6. Initial objective was exactly
57,079.0234375 in both paths. Initial gradient norms were 98.0923352 (loader)
and 98.0923345 (operator). These differences are within float32 tolerances.
Focused tests additionally cover measurement predictions, likelihood,
objective, gradients, frozen-zero cells, positive-frozen offsets, cache
round-trips, corrupted-cache rebuilds, and provenance mismatch rejection.

## Fresh-process unchanged-harness comparison

The operator runs used the persistent BCOO cache. Each fresh sparse process had
a one-time objective/gradient compilation of 28.5–29.2 s; the loader required
30.7–31.3 s. Warm timings and optimizer timings exclude that separately reported
compile pass, matching the existing harness schema.

| Iterations | Evaluations | Loader optimizer (s) | Cached BCOO optimizer (s) | Speedup | Loader final objective | BCOO final objective |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 8.543 | 0.477 | 17.9× | 55,243.4453 | 55,243.4414 |
| 5 | 8 | 18.551 | 2.948 | 6.3× | 49,635.4922 | 49,635.4922 |
| 20 | 24 | 52.648 | 5.679 | 9.3× | 49,140.4219 | 49,140.4258 |

Warm value-and-gradient was 1.868–1.954 s for the loader and
0.000784–0.000841 s for cached BCOO. The automatic policy always reuses a valid
cache, declines an uncached build when expected savings do not exceed measured
construction cost, and preserves explicit `off`, `dense`, and `bcoo` overrides.
