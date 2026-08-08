# Adaptive service-aware departure quadrature

## Architecture

Desired departure time remains uniformly distributed within each OD period.
The timetable controls only where numerical response evaluations are refined;
elapsed interval duration controls every quadrature weight. Infeasible time is
an explicit zero response and retains its mass.

The public implementation has three strategies:

- `fixed_count` (and compatibility alias `uniform_midpoint`);
- `fixed_time_step`, with exact final partial-bin weights;
- `adaptive_service_aware`, using cached sparse response comparisons and
  localized bisection.

The adaptive callback can expose complete sparse assignment responses. The
recommended `integral_response` path compares probability-weighted parent and
child interval integrals, not point responses. The persisted pipeline resolves
active observation events once and accumulates sparse expected measurement
responses separated by destination. Exact service identity, pointwise
measurement support, and route-pattern signatures remain diagnostic modes.
All destinations for an origin-period group reuse each routing evaluation.
The accepted representative choices are then processed by the ordinary sparse
measurement-response and response-equivalence stages.

Conditional route shares remain normalized. Under `preserve_mass`, the new
`JourneyChoiceSet.served_time_fraction` carries feasible elapsed-time mass and
scales response atoms. This avoids both timetable-dependent demand weights and
renormalization after infeasible samples.

The private smoke test exposed `raw_feasible_time_fraction =
1.0000000000000004`. The domain invariant correctly rejected that value. The
fix is not a weaker invariant: a reusable boundary helper now accepts only
finite values within `weight_tolerance` of `[0,1]`, projects only the tiny
excursion, and records raw value, canonical value, application flag, and delta.
Material violations still fail.

The first adaptive implementation refined depth-first. One unstable coarse
interval could consume nearly the complete group budget. Version 2 evaluates
all shared coarse edges and midpoints first, then allocates remaining pairs of
evaluations through a deterministic priority queue using interval mass times
estimated error. Requested and effective comparison modes are separate. The
integrated pipeline uses `integral_response` only when requested explicitly;
it is never silently aliased to the pointwise `aggregate_response`. Requested
and effective names are reported separately.

## Integral error control

For each interval, the coarse midpoint contribution is compared with the sum
of its two child-midpoint contributions. The sparse absolute L1 discrepancy is
the local estimator. Global refinement selects the largest contributor and
stops at an absolute-plus-relative target. Infeasible evaluations are sparse
zero while retaining their elapsed-time mass. Numerical error, unresolved
interval mass, and infeasible time mass are separate diagnostics.

A bounded set of timetable departure and maximum-wait edges seeds the initial
partition. This is a miss-detection safeguard, not timetable-dependent demand
weighting. Schema 4 identifies the embedded rule and integration schema 6
invalidates the earlier pointwise artifacts.

## Demand period and event period

Period semantics version 2 distinguishes the latent desired-departure period
from the realized first-boarding period. The former identifies the OD demand
cell and is supplied explicitly for every sampled routing query. Alternatives
that first board in different periods are ranked, pruned, and normalized
jointly. Event periods remain unchanged, so a `t0` demand column may affect
actual boarding and alighting observations in `t1`. Sampling integration
schema 5 invalidates old sampled and downstream artifacts.

## Quality contract

Diagnostics distinguish heuristic response error from a rigorous bound. The
primary warning is `unresolved_interval_weight`: probability mass stopped by a
sample cap, minimum resolution, or depth limit while still unstable. A result
is `quadrature_converged` only when that mass is negligible. Configuration,
quadrature schema, responses, diagnostics, routing ancestry, group support, and
fixed-cell status participate in persisted identity.

Progress is callback-based and JSON serializable. Phase persistence prevents
recomputing completed preprocessing phases. Mid-group checkpoint/resume is not
yet implemented.

## Public microbenchmark

The deterministic one-hour benchmark compares three midpoint samples,
five-minute and one-minute fixed steps, and adaptive integration. Stable and
single-boundary cases used 6 adaptive evaluations versus 12 for five-minute
and 60 for one-minute sampling, with no deviation from the five-minute result.
The multiple-change case required 46 evaluations, differed by 0.668% from the
five-minute response, and reported 3.125% unresolved mass at the one-minute
minimum. In contrast, three midpoint samples differed by 45.0% and supported
only half of the positive observed mass in that case.

The 11.25-hour observation-response case compared budgets 128, 256, and 512.
All stopped after 107 evaluations, retained 99.72% stable mass, left 0.28%
unresolved at the one-minute limit, supported all observed rows, and agreed
with the five-minute response to floating-point precision. The smallest budget
is therefore recommended for the next private smoke test.

These are numerical microcases, not evidence that the public examples reproduce
TPG service density. The next admission step is a private 20--50
origin-period-group pilot. Recommended starting values are 900-second coarse
intervals, 60-second minimum resolution, response tolerance `1e-3`, sample cap
128, `integral_response`, absolute tolerance `1e-3`, relative tolerance
`2e-2`, and `preserve_mass`.

An adversarial 11.25-hour case placed rapid changes only in its first
15-minute interval. Under a 128-evaluation budget, all 45 initial intervals
received 91 baseline evaluations. Priority refinement used 18 more and left
0.83% unresolved without reaching the cap. The depth-first reference consumed
all 128 evaluations and left 8.89% unresolved.
