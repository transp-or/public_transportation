# Integral-level adaptive desired-departure quadrature

## Audit

The pointwise implementation already provides useful infrastructure: an exact
timestamp cache, deterministic priority queue, whole-period initial coverage,
sparse responses, probability-mass canonicalization, progress callbacks,
serialization, and separate requested/effective comparison modes. These parts
are retained.

Its convergence criterion is replaced for `integral_response`. The former
criterion compares responses at interval endpoints and midpoint. With
trip-specific measurements, an ordinary timetable departure changes sparse
support and therefore makes almost every interval appear unstable even when
the elapsed-time integral is already adequate.

## Embedded rule

For interval \(I=[a,b]\), the coarse estimate is its probability weight times
the response at \((a+b)/2\). The refined estimate is the sum of two half-weight
responses at the child midpoints. Their sparse weighted difference supplies an
absolute L1 estimate. The global stopping rule is

\[
 \sum_I \|Q_I^{\mathrm{refined}}-Q_I^{\mathrm{coarse}}\|_1
 \leq \epsilon_a + \epsilon_r
       \left\|\sum_I Q_I^{\mathrm{refined}}\right\|_1.
\]

The queue is ordered by absolute error contribution, then start time, depth,
and insertion order. Splitting a leaf replaces its refined contribution and
error with those of its two children. No dense measurement vector is formed.

Infeasible evaluations are sparse zero but keep their elapsed-time weight.
Feasible and infeasible mass, numerical error, and unresolved interval mass
remain distinct diagnostics.

## Boundary safeguard

Initial partition edges may include caller-supplied timetable departure change
points. This informs numerical partitioning only: interval probability remains
exactly proportional to elapsed duration. Each interval still receives its
embedded midpoint evaluations, so a listed service boundary cannot be skipped.

The pointwise modes remain available for compatibility and diagnostics. They
do not share cache identity with `integral_response`.
