# Integral adaptive departure benchmark

## Purpose

This public deterministic benchmark models a six-hour timetable in which the
selected vehicle changes every 30 minutes and every vehicle maps to a distinct
measurement row. It therefore exercises the failure mode observed privately:
nearby desired-departure times can have disjoint exact measurement support.

## Result

| Method | Routing evaluations | Supported rows | Relative L1 vs 5 min | Unresolved mass |
|---|---:|---:|---:|---:|
| Three fixed midpoints | 3 | 3/12 | 1.50 | 0 |
| Five-minute fixed step | 72 | 12/12 | 0 | 0 |
| Old pointwise adaptive | 127 | 12/12 | $6.9\,10^{-17}$ | 0.0391 |
| Integral response | 36 | 12/12 | $1.7\,10^{-16}$ | 0 |
| One-minute fixed step | 360 | 12/12 | $1.7\,10^{-16}$ | 0 |

The integral method reproduces the five-minute operator to floating-point
precision with half its routing evaluations. The pointwise method also happens
to reproduce this regularly aligned example, but consumes almost its complete
128-evaluation budget and still declares 3.91% unresolved. Three midpoint
samples miss nine measurement rows.

The embedded estimator reports zero error here because timetable boundary
seeding makes every half-hour response interval explicit. Separate public
step, smooth, support-change, infeasibility, and budget-exhaustion tests verify
that its reported error responds directionally when coarse and refined
integrals disagree.

## Interpretation

Timetable departure times affect the numerical partition but not behavioral
weights. Each interval still receives exactly its elapsed-time probability
mass. Exact trip/event coordinates remain distinct in the accumulated sparse
operator; they are not aggregated or declared behaviorally equivalent.

This synthetic result admits another deterministic two-group private
comparison. It does not yet justify a 40-group run. The private comparison must
materially improve both empirical operator error and unresolved numerical
error before scaling up.

Machine-readable results are in
`docs/reports/adaptive_departure_quadrature_benchmark.json`.
