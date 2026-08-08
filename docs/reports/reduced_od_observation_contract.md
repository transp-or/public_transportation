# Reduced-dimensional OD estimation: observation and journey contract

Status: Phase 0 scientific and data contract. No reduced-dimensional production
code is implemented by this document. Later phases must implement this contract
or explicitly revise it, its tests, and `traffic_assignment.tex` together.

## 1. Scope and invariants

The estimated quantity is **passenger journey demand**, not vehicle-leg demand.
A journey begins when a passenger first boards the public transport system and
ends when that passenger finally alights at the destination. A transfer is an
internal part of the same journey.

The following invariants are mandatory:

1. A transfer boarding is never classified as a new journey origin.
2. A transfer alighting is never classified as a final journey destination.
3. A missing measurement is unobserved; it is not a measured zero.
4. A recorded numerical zero is an observed zero and participates in the
   likelihood.
5. Frozen and structural-zero OD cells are not statistical parameters.
6. The detailed time-expanded assignment is not called from the reduced
   estimator's objective or gradient. It remains an explicit validation
   backend.
7. Every aggregation from platform events to another spatial or temporal level
   is explicit, versioned, and fingerprinted.

## 2. Canonical terminology

**Passenger journey**
: One passenger's complete movement from the first boarding stop to the final
  alighting stop, possibly using multiple vehicles.

**Vehicle leg**
: The part of a journey made on one vehicle trip, from one boarding event to
  the corresponding alighting event.

**Transfer**
: The connection between two consecutive vehicle legs. A journey with `L`
  vehicle legs has `L - 1` transfers. Boarding the first vehicle is not a
  transfer.

**Timetable event**
: A scheduled arrival or departure identified by stop, event time, and trip;
  a line identifier may be used only when those fields still identify one
  unique event.

**Journey alternative**
: An ordered sequence of one or more vehicle legs, including any walking or
  waiting connections, for one origin, destination, and desired-departure bin.

**Leg OD flow**
: The passenger flow between boarding and alighting events on one vehicle leg.
  It is useful for route-level reconciliation but is not the final journey OD
  estimate.

**Physical stop place**
: A passenger-facing interchange that may contain several platforms or
  scenario stop identifiers. Platform-to-place aggregation is an input policy,
  not an inference performed silently by the estimator.

## 3. Journey-demand key and time semantics

The external logical key remains compatible with the current package:

```text
(origin_stop_id, dest_stop_id, time_bin_id)
```

It identifies journey demand whose first boarding occurs at the origin and
whose final alighting occurs at the destination. External results use scenario
stop identifiers in the first implementation. A physical-stop mapping may be
used internally for routing and transfer construction, but results can be
aggregated to physical stop places only through an explicit output policy.

Adjacent desired-departure bins use half-open membership `[start, end)`, with
an optional inclusive end only for the final bin of a declared analysis
period. This avoids assigning a boundary departure to two bins. The existing
assignment's admissible access window remains a separate closed eligibility
interval expanded around the desired-departure bin.

Timetable event times use integer seconds from the service-day origin and may
exceed 24 hours for after-midnight service. The time-bin identifier—not an
implicit clock-time lookup—is retained in every OD key and artifact.

## 4. Observation unit

The atomic input observation is the existing `MeasurementRecord` identity:

```text
(method_id, measurement_type, stop_id, time, trip_id, line_id)
```

The first reduced backend accepts `boarding` and `alighting` observations. Each
record must resolve to exactly one timetable departure or arrival event. At
least one of `trip_id` or `line_id` is required; a line-only record is valid
only when stop, time, direction implied by the timetable, and line identify a
unique event. Ambiguous and unmatched records are errors.

The public schema currently also declares `load`. Load observations remain
outside the first reduced backend and must be rejected there with a diagnostic
until their response semantics and tests are implemented. This restriction
does not remove load from the existing public schema or other backends.

Raw records are never silently grouped. Route-pattern-period, trip-stop,
platform-period, physical-stop-period, or other aggregates are derived views
with their own mapping, units, masks, and fingerprint.

## 5. Boarding, alighting, and transfer accounting

For a journey alternative with `L` vehicle legs, one unit of journey demand
generates:

- one boarding event and one alighting event on every vehicle leg;
- one initial boarding: the boarding of leg 1;
- one final alighting: the alighting of leg `L`;
- `L - 1` transfer alightings and `L - 1` transfer boardings.

Thus, a complete APC table generally satisfies

```text
total observed boardings = initial boardings + transfer boardings,
total observed alightings = final alightings + transfer alightings.
```

Neither total is a direct journey-production or journey-attraction marginal in
a transfer-rich network. The response operator predicts all leg events from
journey demand. It does not create an independent latent demand variable at a
transfer.

Route-level IPF may estimate leg OD or alighting probabilities as a diagnostic
baseline and initializer. Its output must remain labeled `leg_level`; it cannot
be presented as network journey OD until an explicit, validated stitching
operation exists.

## 6. Missing, zero, duplicate, and invalid observations

| Input condition | Contract |
|---|---|
| Record absent | Unobserved; no likelihood term and no implicit zero |
| Record present with value `0` | Observed zero; include in likelihood |
| Record present with positive finite value | Include in likelihood |
| Negative, NaN, or infinite value | Reject before preprocessing |
| Duplicate atomic identity | Reject; never sum silently |
| No matching timetable event | Reject with record identity and reason |
| Multiple matching timetable events | Reject as ambiguous |
| Known sensor outage or invalid quality flag | Exclude through an explicit mask before constructing the immutable observation table; record exclusion counts and provenance |

The first implementation assumes the supplied observation table has already
been APC-cleaned. Cleaning, imputation, and reconciliation must not occur
inside the likelihood. Different collection methods may later receive
method-specific detection or dispersion parameters, but Phase 0 does not make
those parameters identifiable from the counts alone.

## 7. Productions and identifiability

Let `Q[o,t]` be the total passenger journeys starting at origin `o` in desired
departure bin `t`. It is not equal to all boardings observed at `(o,t)` when
the stop also receives transferring passengers.

The reduced model admits two explicit production modes:

1. `provided`: externally justified journey-entry totals are fixed inputs;
2. `estimated_basis`: productions are estimated through a declared
   low-dimensional, nonnegative spatial/temporal basis with regularization.

There is no `raw_boardings` production mode. The initial numerical model may
start with `provided` on synthetic examples. Real-data use without external
entry totals must use `estimated_basis` and report sensitivity to its basis and
prior. Free production per stop and fine time bin is not the default because
APC boarding/alighting counts alone may not identify it separately from
destination and transfer choice.

Destination attractiveness, route shares, production levels, detection rates,
and dispersion cannot all be unconstrained simultaneously without additional
information. Every implemented model must declare which quantities are fixed,
estimated, normalized, or regularized.

## 8. Fixed demand and structural zeros

Frozen-demand keys use the same journey OD/time key. A frozen positive value is
included in predicted measurement counts through a fixed response offset. A
frozen zero contributes neither a parameter nor an offset. Structural zeros
are frozen zeros with an auditable topology/rule reason.

The first reduced backend retains the existing conflict policy: if structural-
zero preprocessing identifies a key with existing positive fixed demand, it
raises an error. It never overwrites that demand.

## 9. Physical stops and transfers

A platform-to-physical-stop mapping is required before transfer construction
on a network with multiple platforms. The preferred source is an authoritative
operator or GTFS parent-station mapping. A distance-derived fallback may be
proposed, but it must be saved, reviewed, and fingerprinted; it is never
silently accepted.

Footpaths are directional records with nonnegative duration and provenance.
Same-place membership does not imply zero transfer time. Transfer feasibility
uses the declared footpath and minimum-change-time policy.

## 10. Phase 0 configuration outline

This is the reviewed semantic outline for the later strict TOML schema. Phase 1
will define exact Python types, defaults, and validation; unknown keys will be
errors.

```toml
schema_version = 1

[observations]
unit = "timetable_event"             # only valid Phase-0 value
accepted_types = ["boarding", "alighting"]
missing_policy = "exclude"           # only valid value
duplicate_policy = "error"           # only valid value
ambiguous_event_policy = "error"     # only valid value
cleaning_stage = "external"          # only valid value

[journeys]
origin_semantics = "first_boarding"  # only valid value
destination_semantics = "final_alighting" # only valid value
time_bin_membership = "half_open"    # only valid value
maximum_transfers = 2                # integer >= 0
route_shares = "fixed_within_fit"     # first implementation

[productions]
mode = "provided"                    # provided | estimated_basis
# input_path = "journey_productions.csv"  # required for provided
# basis = "origin_period"             # required for estimated_basis initially

[stops]
mapping_policy = "authoritative"     # authoritative | reviewed_generated
# physical_stop_mapping_path = "physical_stops.csv"
# footpaths_path = "footpaths.csv"

[outputs]
spatial_level = "scenario_stop"      # scenario_stop | physical_stop
reconstruct_full_od = false

[validation]
detailed_assignment = "explicit_only" # only valid value
```

## 11. Acceptance examples

1. A direct one-leg journey produces one boarding and one alighting.
2. A two-leg journey produces two boardings and two alightings but only one
   journey origin and one journey destination.
3. Removing an observation removes its likelihood term; it does not create a
   zero target.
4. Supplying an explicit zero retains a likelihood term with target zero.
5. Two adjacent departure bins never claim the same boundary departure.
6. A frozen-zero journey cell creates no estimator coordinate.
7. A line-only record matching two trips is rejected.
8. A platform transfer cannot be created without an accepted physical-stop and
   footpath policy.

## 12. Decisions still required before private full-network use

The public implementation can proceed with explicit fixtures and both
production modes, but private TPG validation still requires:

- the authoritative APC cleaning/exclusion definition;
- the authoritative platform-to-physical-stop mapping or approval of a
  generated mapping;
- the selected production mode and, if `estimated_basis`, its spatial and
  temporal resolution;
- confirmation of the required published output geography;
- the service day, analysis period, and after-midnight convention;
- confirmation of sensor coverage and known outage metadata.

These decisions affect artifact fingerprints. They cannot be inferred from
the observed counts during estimation.
