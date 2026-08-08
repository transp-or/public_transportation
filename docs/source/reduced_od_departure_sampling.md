# Desired-departure sampling for reduced OD

Long demand periods must not be represented by one convenient scheduled
departure when observations distinguish individual vehicle journeys. The
public reduced-OD preprocessing layer models desired passenger departure time
independently of the timetable:

\[
T^*_{odt}\sim\operatorname{Uniform}(a_t,b_t).
\]

For `S` equal-width midpoint samples, sample `s` is

\[
\tau_{ts}=a_t+(s+1/2)(b_t-a_t)/S,\qquad w_{ts}=1/S.
\]

The timetable is queried only after these times and weights have been fixed.
Changing service frequency therefore cannot change the desired-time samples.
Uniform desired departures are a modeling assumption, not an empirical
finding and not a supply-derived demand profile.

## Configuration and fidelity

```python
from public_transportation.preprocessing.reduced_od import (
    DepartureTimeSamplingConfig,
)

sampling = DepartureTimeSamplingConfig(
    strategy="adaptive_service_aware",
    initial_interval_seconds=900,
    minimum_interval_seconds=60,
    response_tolerance=1.0e-3,
    comparison_mode="integral_response",
    absolute_response_tolerance=1.0e-3,
    relative_response_tolerance=2.0e-2,
    maximum_samples_per_cell=128,
    minimum_feasible_fraction=0.5,
    warning_feasible_fraction=0.9,
    infeasible_policy="preserve_mass",
    convergence_levels=(3, 6, 12),
)
```

`uniform_midpoint` is a compatibility alias for `fixed_count`. Counts may be
one positive integer or a mapping from period identifier to
positive count. One sample is retained only for low-fidelity backward
comparison. Three, six, and twelve samples are useful minimal, moderate, and
default starting levels; these labels do not guarantee accuracy. Counts of 24
or more provide user-controlled refinement.

`fixed_time_step` partitions the complete interval into bins of at most
`time_step_seconds`, evaluates each bin midpoint, and weights it by its exact
duration. Thus a 13-minute interval at five-minute resolution uses bins of
5, 5, and 3 minutes, midpoints at 2.5, 7.5, and 11.5 minutes, and weights
5/13, 5/13, and 3/13. A five-minute step is the practical reference; a
one-minute step is useful for cheap convergence checks.

`adaptive_service_aware` supports the compatible pointwise modes and the
recommended `integral_response` mode. Integral adaptation compares a
probability-weighted parent midpoint integral with the sum of two child
midpoint integrals. Sparse support additions contribute naturally to the L1
difference; support equality is not required. A deterministic global queue
refines the largest absolute integral-error contributor until

\[
 \sum_I e_I \leq \epsilon_a+epsilon_r\|Q\|_1.
\]

The initial embedded baseline uses three evaluations per interval and reserves
part of the group budget for refinement. Timetable departures and maximum-wait
window edges may seed a bounded number of partition edges, preventing narrow
service windows from being silently skipped. These edges affect numerical
partitioning only; probability remains proportional to elapsed time. Evaluated
timestamps are cached and sparse responses are never densified.

`absolute_response_tolerance` and `relative_response_tolerance` define the
mixed global target. When the latter is omitted, the legacy
`response_tolerance` value is used as its backward-compatible alias. The scale
floor affects only relative normalization near sparse zero.

For compatibility, the budget field remains named
`maximum_samples_per_cell`; in integrated preparation its effective scope is
one origin-period group because one routing evaluation is reused across all of
that group's destinations. Preparation rejects a budget below the required
$2N+1$ baseline rather than leaving coarse intervals unevaluated.

The standalone sparse callback honors `assignment_response`. Integrated
preparation resolves active measurements once. With explicit
`integral_response`, convergence compares sparse interval integrals separated
by destination. The older pointwise `aggregate_response` remains available;
exact trip identity remains available as
`exact_service_identity` (and the compatibility spelling `service_signature`).
`measurement_support` ignores coefficient changes, while
`route_pattern_signature` ignores exact trip identities. `two_stage` currently
resolves to the aggregate pointwise comparison. Requested and effective modes are both
recorded.

Pass the configuration in `ReducedODPreparationInputs.departure_time_sampling`.
Omitting it selects the explicitly labeled legacy representative-departure
path. When sampling is enabled, queries are generated for the exact sparse
`departure_sampling_origin_period_groups`, or deterministically for
`production_inputs.keys()` when that field is omitted. Positive fixed-demand
groups are added because their response is required for the fixed offset.
The origin-list API remains only as an explicit Cartesian convenience.

## Canonical cell status

The caller's canonical contract classifies every candidate as free,
fixed-zero, or fixed-positive. Timetable feasibility is recorded separately
and never changes that status. Only retained free cells become response,
equivalence, gravity-feature, and estimated-demand columns. A timetable-feasible
fixed-zero cell remains diagnostic-only and contributes no offset. A feasible
fixed-positive cell contributes its sampled averaged response times its fixed
flow to the offset; preparation fails if that positive flow cannot be assigned.
Production coverage is checked only against origin-period groups represented by
retained free cells, so fixed-only groups do not require production inputs.

## Completed-journey conditioning and preserved mass

For feasible sample set \(F_c\), preprocessing retains both

\[
p_c^{\mathrm{feasible}}=\sum_{s\in F_c}w_s
\]

and the conditional served response

\[
\bar A_c^{\mathrm{served}}=
\frac{\sum_{s\in F_c}w_s A_{c,s}}{p_c^{\mathrm{feasible}}}.
\]

The legacy default estimates completed public-transport journeys. It does not estimate
latent unserved demand. Conditioning does not move passengers to earlier or
later services, another day, another sample, or a supply-weighted time. Original
feasible and infeasible weights remain in persisted diagnostics.

Default classifications are:

- zero feasible weight: `frozen_no_feasible_sample`;
- feasible weight below 0.5: `excluded_low_feasibility`;
- feasible weight from 0.5 up to 0.9: `warning`;
- feasible weight at least 0.9: `normal`.

Thresholds are configurable operational safeguards, not universal behavioral
constants. With `preserve_mass` (or its compatibility spelling
`retain_unserved_mass`), route shares remain conditional on service and sum to
one, while each choice set stores a separate `served_time_fraction`. Response
construction multiplies conditional shares by that fraction. Infeasible time
therefore contributes an explicit zero response and is never reassigned to a
feasible service. The two policies have different statistical targets and
different artifact fingerprints.

## Weighted path and response merging

The merger preserves both probability layers:

\[
P(s,r)=w_sP(r\mid s).
\]

Identical scheduled paths seen at several sample times receive the sum of their
joint weights. They are never deduplicated and subsequently assigned equal
shares. The desired-departure period remains the OD cell period and is explicit
on every sampled alternative. The realized first-boarding period is separate
and may differ across alternatives in one jointly normalized choice set.
Boarding and alighting events retain their actual timetable periods, so `t0`
demand may affect `t1` measurement rows. Legacy callers that omit the demand
period retain first-boarding grouping and are reported as legacy semantics.
The final operator still has one column per retained demand-period OD cell.

Sample-specific journey objects are retained for only one origin-period batch.
They are merged into deterministic sparse choices and then released. Exact
sample definitions, cell classifications, and the averaged-journey fingerprint
are persisted atomically.

## Infeasibility

`DepartureSampleInfeasibilityReason` includes end of service, timetable
horizon, excessive initial wait, excessive journey duration, transfer limit,
missing transfer connection, directed platform unreachability, absence of a
feasible alternative, and `unknown`. Callers should report the most specific
safely supported reason and may preserve diagnostic candidates when one cause
cannot be established uniquely. Physical disconnection, directed-platform
reachability, timetable infeasibility, behavioral rejection, and missing input
are not interchangeable.

## Convergence before estimation

`compare_departure_sampling_levels` accepts a preprocessing evaluator, not an
optimizer:

```python
report = compare_departure_sampling_levels(
    evaluator=build_level_without_fitting,
    levels=(3, 6, 12),
    observations=observed_counts,
    relative_change_tolerance=0.05,
    progress=handle_progress,
)
```

It reports relative L1 and L2 prediction changes, maximum absolute and relative
row changes, predicted-total and zero-observation-mass changes, classification
changes, and sparse-support additions and removals. Each level also carries
query counts, phase time, memory, artifact size, response nonzeros/classes,
feasible weights, classification maps, and operator identity.

`preflight_departure_sampling` projects actual sparse queries, the hypothetical
Cartesian group/query counts, avoided work and wall time, temporary memory,
retained sparse support, and disk use. Its linear support projection is
intentionally conservative.

## Diagnostics and recommendations

`build_departure_sampling_diagnostics` reports configuration and exact samples,
network totals, period/origin/destination summaries, cell feasibility,
conditional concentration, expected journey attributes, path/service
diversity, and optional pre-estimation observation alignment. Alignment includes
boarding/alighting totals, zero-row prediction mass, MAE, RMSE, largest
residuals, maximum row prediction, and vehicle-journey concentration.

`recommend_departure_sampling_actions` produces deterministic advisory records.
It can recommend higher resolution, period splitting, a longer timetable
horizon, review of waiting/duration/transfer constraints, caller-provided
footpaths, freezing or excluding low-feasibility cells, observation aggregation,
a caller-provided nonuniform profile, or investigation of concentrated vehicle
responses. It never modifies the configuration. Invalid accounting, nonfinite
predictions, inconsistent observation ordering, widespread low feasibility, or
gross convergence instability should block estimation.

## Progress, persistence, and invalidation

Structured callbacks emit JSON-compatible `started`, `in_progress`,
`completed`, and `failed` events with work counters, elapsed time, ETA, current
sampling level, recent query rate, and peak RSS where applicable. Count and
wall-time throttles are configurable. Callbacks do not cause additional
journey searches.

The same machine-readable ETA contract covers the upstream timetable index,
bounded RAPTOR rounds and destination scans, journey-choice cells, sparse
measurement-response construction, structural-zero destination profiles and
OD-metric materialization, and array-by-array artifact publication. Terminal
events report zero remaining time only after the corresponding output is valid;
the initial event reports an unavailable ETA until measured work exists.

Adaptive integral events additionally report routing evaluations, cache hits,
accepted and refined intervals, current infeasible fraction, unresolved mass,
sample-cap state, reserved baseline and remaining refinement budgets, stable
weight, coarse and refined norms, absolute and relative error estimates, global
target and status, support additions/removals, recent evaluation time, elapsed
time, an ETA confidence label, and peak RSS. Batch events report completed cells, throughput, mean and maximum
evaluations per group, aggregate unresolved weight, mean and maximum unresolved
fraction, fully unresolved group count, and mean stable fraction. Existing
preprocessing persistence resumes at completed phase boundaries. It does not
yet checkpoint halfway through one origin-period group; applications should
choose group sizes and deadlines accordingly.

Sampling fingerprints contain period-semantics version 2, strategy, counts,
periods, exact ordered sparse
group support, canonical fixed/free status and fixed values, construction,
weights, thresholds, infeasibility policy, timetable ancestry, stop/footpath
configuration, and journey-policy ancestry. A sampling-only change reuses
physical stops, service patterns, and the timetable index while rebuilding
samples, sampled choices, responses, operators, and model manifests. Partial
or incompatible artifacts fail closed.

See `docs/source/examples/reduced_od_departure_sampling.py` for a complete
public hand-calculable example.

One representative departure is generally unsuitable for long periods with
vehicle-journey-specific observations. Before a private full-network rebuild,
run an adaptive 20--50 origin-period-group pilot against five-minute and, where
cheap, one-minute fixed-step references. Inspect supported positive mass,
relative response error, unresolved interval weight, and sample-cap frequency.
The timetable-discrete public benchmark supports starting with
`comparison_mode="integral_response"`, absolute tolerance `1e-3`, relative
tolerance `2e-2`, and budget 128. It matched the five-minute reference with 36
routing evaluations instead of 72; the old pointwise rule used 127.
