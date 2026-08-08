# Reduced-dimensional OD estimation: development workflow

Status: design proposal only. No implementation is authorized by this document.

## 1. Executive decision

The detailed time-expanded assignment remains the authoritative validation
backend, but it must not be used inside the new estimation loop. The new design
has two public packages:

- `public_transportation.preprocessing.reduced_od` produces reusable,
  versioned timetable and journey-response summaries without constructing a
  global time-expanded graph;
- `public_transportation.inference.reduced_od` fits route-level, entropy, and
  structured journey-OD models against those summaries.

`reduced_od` is preferable to `destination_choice` because the scope includes
route-level count reconciliation, latent transfers, entropy balancing,
reduced operators, reconstruction, and validation in addition to destination
choice.

The first implementation should proceed in two steps:

1. a route-pattern/vehicle-leg IPF baseline that audits and reconciles the
   boarding and alighting measurements without claiming to recover complete
   journeys;
2. a minimal journey-level conditional gravity model whose precomputed journey
   responses translate each complete journey into all of its leg boardings and
   alightings.

The first step is fast, transparent, and directly tied to the observations. It
also provides an initialization and data-quality diagnostic. It is not the
final journey-OD estimator: independently stitching leg matrices is generally
underidentified. The second step represents transfers as latent internal
events constrained by complete journey alternatives, thereby avoiding the
incorrect assumption that every boarding is a new journey origin.

## 2. Repository inspection findings

The repository already separates domain, assignment, measurement,
preprocessing, and inference concerns, but several reusable statistical
components are coupled to the current fixed-routing operator.

### Existing strengths

- `domain.scenario`, `domain.timetable`, `domain.trip`, and
  `domain.stop_time` provide validated schedule inputs and folder I/O.
- `measurement.mapping.strict` provides authoritative boarding/alighting event
  alignment; `measurement.likelihood_jax` provides stable Poisson and
  negative-binomial kernels.
- `preprocessing.structural_zeros` provides strict TOML configuration,
  structural-zero rules, deterministic fingerprints, progress, persistence,
  and path-metric contracts.
- `inference.od_parameter_layout` and
  `inference.compact_od_assignment_layout` provide canonical OD keys,
  fixed/free partitions, structural-zero removal, and full-table
  reconstruction.
- `inference.gravity` provides immutable feature and model specifications,
  stable parameter transformations, JAX demand generation, Poisson and
  negative-binomial objectives, L-BFGS estimation, warm starts, model lineage,
  adequacy diagnostics, relaxation recommendations, structured holdouts,
  manifests, JSONL progress, deadlines, and checkpoints.
- The simple and Geneva examples provide synthetic networks, transfers,
  boarding/alighting observations, and end-to-end comparison infrastructure.

### Gaps relevant to the new backend

- `Stop` has no physical-station/platform hierarchy. Such mappings must be an
  explicit preprocessing input or a backwards-compatible domain extension.
- There is no route-pattern equivalence-class object. `Line` and `Trip` are too
  coarse and too fine, respectively.
- Structural-zero path metrics currently call `build_jax_graph` and therefore
  still construct the time-expanded graph.
- Current gravity features contain one row per free OD cell and the operator
  remains the expensive fixed-routing mapping.
- The existing strict measurement map targets time-expanded links, whereas the
  new backend needs a direct map from journey legs to observed boarding and
  alighting events.
- Current origin-time production totals are externally supplied. Boarding
  counts cannot automatically supply them when transfers are present.

## 3. Scientific formulation

Let `g=(o,t)` denote a journey-origin and departure-period group, and let
`D_g` be its feasible destinations. The minimal model is

```text
x_odt = q_ot * p(d | o,t)
p(d | o,t) = softmax_d(V_odt beta), d in D_(o,t)
```

with utility terms drawn from precomputed timetable summaries:

```text
V_odt beta = log(attraction_dt)
             - beta_time * scaled_travel_time_odt
             - beta_transfer * transfers_odt
             - beta_wait * scaled_wait_odt
             - beta_walk * scaled_walk_odt
             + selected centered effects.
```

The initial specification should use only travel time and transfers. Waiting,
walking, and spatial distance enter only after their definitions and data
sources are validated. `q_ot` may be fixed from genuine journey-entry data or
estimated as a smaller nonnegative origin-time state. It must never be set
equal to raw boardings by default.

For each feasible OD-time cell `c`, preprocessing produces a small set of
journey alternatives `j` with fixed initial shares `s_j|c`. Each alternative
has an event-incidence vector `b_j` containing every leg boarding and
alighting. The expected response is

```text
b_c = sum_j s_j|c * b_j
y_hat = fixed_offset + sum_c x_c * b_c.
```

The first estimator keeps route shares fixed. An occasional external update
may recompute them from the detailed assignment, but route shares do not change
inside each statistical optimization.

### Reduced linear bases

For a genuinely linear representation `x = Phi z`, preprocessing constructs

```text
H = B Phi
```

directly from journey response atoms without constructing the full OD-to-link
operator `A`. The loop uses `y_hat = H z + offset`.

For conditional softmax gravity, `x(beta)` is nonlinear, so a globally fixed
`H=A Phi` is not exact. The compact loop instead evaluates segmented softmax
probabilities and applies sparse journey-response atoms `B`. Equivalent cells
with identical origin-time denominator membership, feature vectors, and
measurement responses may be collapsed exactly. A local Jacobian or basis
response may be cached for diagnostics, but must not be described as a global
linear operator.

## 4. Boarding, alighting, and transfers

The observation contract must distinguish:

- **journey origin/destination:** first boarding and final alighting;
- **vehicle-leg boarding/alighting:** every entry to and exit from a vehicle;
- **transfer event:** an internal alighting followed by a boarding, possibly at
  different platforms of one physical stop.

The new journey response counts every leg event. Passenger conservation is
then automatic within a complete journey alternative: every internal transfer
contributes one alighting and one subsequent boarding. Aggregated observations
may still differ because of missing sensors, time-bin boundaries, platform
mapping, and measurement error.

Alternatives considered:

| Approach | Role | Decision |
|---|---|---|
| Route/vehicle-leg OD then independent stitching | Fast and directly identified on each pattern, but stitching is generally nonunique | Implement first as baseline and initialization, not as the final journey estimate |
| Direct journey OD with compact transfer choice | Scientifically closest to the target and conserves complete journeys | Implement as the first network-level model after journey-response preprocessing |
| Free latent transfer flows with conservation constraints | Flexible but adds many weakly identified states | Defer; use only targeted corrections after the compact model is diagnosed |
| Route-level initialization followed by journey model | Combines stable marginal recovery with network consistency | Recommended operational sequence |

Li and Cassidy's route-level method motivates stable alighting probabilities
within a route, but transfer-rich network inference needs the additional
journey layer. RAPTOR supplies timetable-feasible journey alternatives without
a global event graph. Frequency-based hyperpaths remain a later alternative
for high-frequency services; they require explicit common-line and transfer
strategy adaptations.

## 5. Baseline methods and ordering

Let `C` be feasible OD-time cells, `J` journey alternatives, `M`
measurements, `K` statistical parameters, and `nnz(B)` the event-response
nonzeros.

| Method | Estimated state | Precomputation and complexity | Transfers/time | Laptop feasibility and JAX | Main scientific risk |
|---|---|---|---|---|---|
| Route-level IPF | Leg OD or alighting probabilities per route-pattern/service-period | Ordered route stops; roughly `O(iterations * feasible_leg_cells)` and linear memory | No journey stitching; separate service periods | Excellent; NumPy first, JAX optional | Leg flows are not journey OD and transfer boardings inflate origins |
| Entropy/Sinkhorn | Transport plan matching supplied origin and destination marginals | Sparse generalized-cost support; `O(iterations * C)` | Timetable costs can vary by period; transfers enter cost only | Good with sparse segmented scaling; JAX compatible | Journey marginals are not directly observed in transfer-rich APC data; regularization controls the answer |
| Unbalanced entropy transport | Transport plan with penalized marginal mismatch | Same support plus marginal penalties | Handles incomplete/noisy counts, not transfer semantics by itself | Good | Penalty parameters and marginal interpretation may dominate results |
| Minimal conditional gravity | A few cost coefficients, dispersion, and optionally origin-time productions | Feasible sets, features, and `B`; each evaluation `O(C*K + nnz(B))`, less after equivalence compression | Complete journey alternatives create all transfer events | Primary network model; JAX friendly | Identification of productions, attraction, and route shares |
| Low-rank spatial-temporal | Gravity baseline plus rank-`r` centered factors | Coarse zone/time indices; evaluation roughly linear in `C*r` | Journey response unchanged | Feasible for small rank | Rotational/scale nonidentification and overfitting |
| Reduced-state temporal | Day-specific low-dimensional state | Multiple aligned days and state-transition statistics | Period/day dynamics explicit | Feasible after compact observation model exists | Not identified from one day; state assumptions may be wrong |
| Frequency/hybrid assignment | Hyperpath shares for frequent services, timetable choices elsewhere | Route frequencies, common-line sets, and low-frequency schedule summaries | Transfers require compatible strategy composition | Potentially feasible; separate routing engine | Loss of schedule detail and discontinuities at hybrid boundaries |
| Sparse correction | Sparse deviations from gravity/low-rank baseline | Compact correction dictionary and response columns | Corrections inherit journey responses | Proximal NumPy/JAX feasible if dictionary is small | APC data may not identify arbitrary local corrections |

Compressed sensing should therefore be restricted to corrections around a
structured baseline. Assuming that the entire full-network OD table is sparse
is not the default scientific position.

### Adaptation of the cited methods

The cited methods are useful design inputs, but none should be transferred into
this package without accounting for the semantics of a time-dependent public
transport network.

| Source | What should be reused | Required adaptation or limitation |
|---|---|---|
| [Li and Cassidy (2007)](https://doi.org/10.1016/j.trb.2006.04.001) | Iterative proportional fitting of route-level OD or alighting probabilities from boarding and alighting totals; a valuable transparent baseline and initializer. | The original route-level setting does not by itself construct network journeys across lines. Transfers must be represented as internal legs of a passenger journey rather than as new exogenous demand. |
| [Cuturi (2013)](https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html) | Entropic transport and Sinkhorn scaling as a fast, parallelizable balancing method. | Balanced Sinkhorn is appropriate only when both journey-origin and journey-destination marginals are genuinely observed or otherwise justified. Unbalanced variants can absorb noisy/incomplete marginals, but they do not solve the semantic problem created by treating transfer boardings as new origins. |
| [Delling, Pajor, and Werneck (2015)](https://pubsonline.informs.org/doi/pdf/10.1287/trsc.2014.0534) | RAPTOR's route-based rounds, transfer count control, and parallel preprocessing are the preferred foundation for enumerating feasible schedule-based journey alternatives. | RAPTOR is a journey generator, not the estimation operator. It must be adapted to range departures/time bins, deterministic alternative pruning, response-vector construction, and persistent compact caches. |
| [Spiess and Florian (1989)](https://doi.org/10.1016/0191-2615(89)90034-9) | Optimal-strategy/hyperpath ideas for high-frequency common-line services and compact frequency-based choice sets. | The full network may mix frequent and schedule-sensitive services. A hybrid backend therefore needs an explicit regime rule, transfer treatment, and validation against timetable-based journeys before a frequency representation can replace scheduled alternatives. |
| [Kumar et al. (2019)](https://journals.sagepub.com/doi/10.1177/0361198119845896) | L1 regularization as an optional sparse residual/correction around a structured baseline. | Sparsity should not be imposed on the complete OD matrix by default: diffuse demand may be real. The first implementation should regularize only an explicitly enabled correction layer. |
| [Chen, Cheng, and Sun (2025)](https://doi.org/10.1016/j.trb.2025.103278) | Conditional destination probabilities, time-varying structure, low-dimensional latent representations, and Bayesian uncertainty motivate the proposed model hierarchy. | The initial backend should establish deterministic likelihood/MAP behavior and measurement semantics before adding low-rank temporal or Bayesian layers. The paper's assumptions must be checked against network-wide journeys, transfers, and the observations actually available in this package. |

The implementation order below deliberately begins with the simplest auditable
interpretation of Li--Cassidy/IPF, uses RAPTOR-style preprocessing to obtain
network-wide alternatives, and introduces gravity, temporal, low-rank, and
uncertainty components only after their additional assumptions can be tested.

## 6. Package architecture

Proposed preprocessing package:

```text
preprocessing/reduced_od/
    config.py                 strict versioned TOML
    physical_stops.py         platform-to-place mapping
    route_patterns.py         ordered stop-pattern equivalence classes
    service_periods.py        timetable-equivalent periods
    timetable_index.py        array-based route/trip index
    raptor.py                 bounded-round timetable profiles
    journey_choices.py        compact Pareto/choice alternatives
    response_atoms.py         boarding/alighting event incidence
    equivalence.py            exact cell-response compression
    artifacts.py              immutable schemas and fingerprints
    persistence.py            atomic cache I/O and validation
    progress.py               phase and resource events
    service.py                TOML-driven orchestration
```

Proposed inference package:

```text
inference/reduced_od/
    contracts.py              canonical keys and problem contracts
    route_level.py            IPF/alighting-probability baseline
    entropy.py                balanced and unbalanced Sinkhorn
    features.py               compact destination-choice features
    specification.py          explicit model complexity
    parameters.py             transforms and warm-start mappings
    response_operator.py      B, H, matvec/rmatvec and diagnostics
    objective.py              Poisson/NB/MAP objectives
    estimator.py              L-BFGS and optional proximal solvers
    reconstruction.py         full OD table only on request
    lineage.py                parent/child models and warm starts
    validation.py             adequacy and predictive validation
    diagnostics.py            residual-driven relaxations
    operations.py             manifests, progress, deadlines, checkpoints
```

The package must not import the sharded matrix-free, microshard, or
block-coordinate implementations. A separate validation adapter may call the
detailed assignment after estimation.

## 7. Reuse, adapt, or replace

| Existing component | Current location | Reuse unchanged | Adapt | Replace | Reason | Target component |
|---|---|---:|---:|---:|---|---|
| Scenario/timetable/trip/stop-time I/O | `domain/` | Yes | Minor | No | Stable public schedule contract | preprocessing inputs |
| Stop object | `domain/stop.py` | No | Yes | No | Missing platform/physical-place identity | `physical_stops.py` plus optional compatible metadata extension |
| Time-expanded graph builder | `assignment/build_time_expanded.py` | Validation only | No | Yes | Prohibited from new inner pipeline | `timetable_index.py`, `raptor.py` |
| Strict boarding/alighting parsing | `measurement/mapping/strict.py`, `measurement/schema.py` | Parsing/schema | Yes | No | Link mapping is assignment-specific | direct event response builder |
| Event-aligned aggregation | `measurement/event_aligned.py` | Concepts/kernels | Yes | No | Useful compact gather/scatter pattern | `response_operator.py` |
| Poisson/NB likelihoods | `measurement/likelihood_jax.py` | Yes | No | No | Stable tested kernels | reduced objective |
| Structural-zero configuration/service | `preprocessing/structural_zeros/` | Contracts and persistence | Yes | Routing core | Current topology constructs time-expanded graph | reduced preprocessing service |
| Path metric record types | `preprocessing/structural_zeros/types.py` | Mostly | Yes | No | Metrics remain relevant; need walk/wait and alternative summaries | journey feature artifacts |
| OD parameter layout | `inference/od_parameter_layout.py` | Key/fixed-demand semantics | Yes | No | Full cell parameterization is not the new optimizer state | reconstruction boundary |
| Compact assignment layout | `inference/compact_od_assignment_layout.py` | Validation adapter | Yes | No | Still useful when sending reconstructed demand to detailed assignment | reconstruction/validation adapter |
| Gravity features/specification | `inference/gravity/features.py`, `specification.py` | Semantics | Yes | No | Current arrays are cell-level and operator-coupled | reduced features/specification |
| Gravity transformations/demand | `inference/gravity/parameters.py`, `demand.py` | Stable transforms/segmented softmax | Yes | No | Need optional production state and compressed cells | reduced parameters/destination choice |
| Gravity objective/estimator | `inference/gravity/objective.py`, `estimator.py` | Optimizer and likelihood patterns | Yes | Operator binding | Current operator is detailed assignment | reduced objective/estimator |
| Warm starts and lineage | `inference/gravity/relaxations.py`, `lineage.py` | Yes | Extend scopes | No | Already explicit and tested | reduced lineage |
| Adequacy/recommendations | `inference/gravity/validation.py`, `diagnostics.py` | Metrics | Yes | No | Group labels need route-pattern and transfer semantics | reduced validation/diagnostics |
| Holdouts | `inference/gravity/holdout.py` | Yes | Extend units | No | Vehicle-journey grouping is already supported | reduced predictive validation |
| Manifests/progress | `inference/gravity/operations.py` | Patterns | Yes | No | Fingerprints must identify new caches and response atoms | reduced operations |
| Fixed/sharded routing operators | `inference/fixed_routing_*`, `sharded_*` | Validation only | No | Yes in inner loop | Too large and too tightly coupled | compact response operator |
| Block-coordinate estimator | `inference/block_coordinate/` | No | No | Yes | Solves the wrong high-dimensional state | low-dimensional L-BFGS/proximal estimator |

## 8. Preprocessing artifacts and schemas

Artifacts must be split into timetable-invariant and measurement-specific
caches so that changing observations does not rerun timetable routing.

### Timetable cache

1. `physical_stops`: platform ID, physical-place ID, coordinates, minimum
   transfer time, and provenance.
2. `route_patterns`: pattern ID, line/mode/direction, ordered platforms,
   equivalence members, and calendar/service IDs.
3. `service_periods`: time-period ID, representative schedule rule, member
   departures, and validity interval.
4. `timetable_index`: flattened route stops, trip stop times, stop-to-route
   incidence, footpaths, and transfer closure required by RAPTOR.
5. `feasible_destinations`: CSR by origin-time group with structural-zero
   reasons for excluded cells.
6. `journey_features`: travel, initial wait, transfer wait, walk, transfers,
   distance, generalized cost, and feature provenance.
7. `journey_choices`: CSR offsets plus leg sequences, event times, pattern IDs,
   and initial shares for a small nondominated choice set.

### Measurement-response cache

1. canonical measurement identity and labels;
2. sparse alternative-to-measurement response atoms;
3. expected OD-cell response under fixed route shares;
4. fixed-demand response offset;
5. exact equivalence classes for cells with identical denominator group,
   feature vector, and response;
6. optional basis-response matrix `H` and basis manifest.

Every manifest records schema and implementation versions, package and source
revision, scenario/timetable fingerprint, physical-stop mapping, time bins,
route-pattern policy, RAPTOR criteria and transfer limit, measurement identity,
feature scaling, structural-zero policy, dtype, array shapes, checksums,
creation time, peak RSS, and elapsed phases. Cache publication is atomic.

Parquet is suitable for user-inspectable tables; NPZ or a simple chunked array
format is suitable for numeric CSR artifacts. A new storage dependency should
not be added until NPZ scalability is measured.

## 9. Route-based preprocessing design

Build an array-based timetable index from `Trip` and `StopTime`:

- group trips only when they share an ordered platform sequence and compatible
  pickup/drop-off semantics;
- sort trips within a pattern by departure time;
- index patterns serving each platform;
- represent walking transfers as a transitively closed bounded footpath graph;
- run bounded-round RAPTOR profiles by origin and departure/service period;
- retain nondominated labels for arrival time and transfers, optionally walk;
- reconstruct only a small deterministic journey choice set per feasible cell.

RAPTOR avoids a conventional global event graph and scans each route at most
once per round. It must be adapted here to produce reusable OD features and leg
sequences rather than interactive point-to-point answers. Schedule-dependent
feasibility requires multiple departure samples or range RAPTOR within each
time bin; a single representative departure is insufficient unless explicitly
declared as an approximation.

## 10. Complexity hierarchy

Parameter counts below exclude a negative-binomial dispersion parameter when
Poisson is used. Let `P` be broad periods, `D` coarse destination zones, `O`
coarse origin zones, `S` selected transfer places, and `r` a rank.

| Level | Added state | Approximate parameter count | Assumption relaxed | Evidence required | Warm start |
|---|---|---:|---|---|---|
| R0 | Route-level IPF | No global statistical parameters | Measurement marginals only | Marginal consistency and route coverage | Supplies leg probabilities and scale initialization |
| J0 | Global time and transfer sensitivity; fixed attraction and productions | 2 (+ dispersion) | Uniform conditional destination response to cost | Finite gradients, identified curvature, acceptable global residuals | Physical defaults or prior fit |
| J0q | Optional latent origin-time productions with smooth/offset parameterization | `G` or a much smaller production basis | External production totals | No valid journey-entry totals; conservation and regularization checks | Route-level origin scale after transfer adjustment |
| J1 | Broad-period centered cost effects | `P-1` per selected coefficient | Constant behavior through day | Structured time-period residuals and adequate support | Replicate parent coefficient |
| J2 | Coarse destination-zone centered attractions | `D-1` | Fully exogenous attraction | Destination-zone residual pattern and identification | Zero centered deviations |
| J3 | Coarse origin-zone production effects | `O-1` | Uniform production calibration | Origin-zone residual pattern not explained by transfer counts | Zero centered deviations |
| J4 | Selected transfer-place corrections | `S` with shrinkage/centering | Uniform transfer penalty | Repeated paired alight/board discrepancies at named hubs | Zero corrections |
| J5 | Low-rank zone-by-period interaction | roughly `r(D+P-r)` after constraints | Additive spatial/temporal effects | Multi-period residual structure, rank stability, held-out gain | Rank expansion with zero new factors |
| J6 | Sparse response corrections | selected dictionary size | Smooth structured baseline | Local residuals persist and support is adequate | Zero correction |

The hierarchy is not advanced automatically. Recommendations remain advisory,
and every child is compared with its parent in adequacy first and predictive
validation second.

## 11. Estimation algorithms

- **IPF:** safeguarded alternating scaling on triangular route-pattern support,
  explicit handling of zero marginals, convergence and infeasibility reports.
- **Balanced Sinkhorn:** log-domain or stabilized scaling over sparse feasible
  support when meaningful journey origin and destination marginals exist.
- **Unbalanced Sinkhorn:** KL marginal penalties for noisy/incomplete journey
  marginals; penalty values must be user-visible and sensitivity-tested.
- **Gravity ML/MAP:** JAX value/gradient over segmented softmax and sparse
  response atoms, optimized by L-BFGS. MAP reuses prior/regularization
  contracts; full Bayesian inference is optional only after the compact model
  is demonstrably small and identified.
- **Low rank:** constrained centered factors with explicit identifiability;
  alternate or joint gradient optimization only after synthetic recovery tests.
- **Sparse correction:** proximal gradient or coordinate updates on a small
  correction dictionary, never on all OD cells.

The estimator state contains parameters and compact sufficient statistics, not
the reconstructed full OD vector. Full reconstruction is an explicit output
operation.

## 12. Validation workflow

### Model adequacy

Fit all available counts, reconstruct journey OD, map it through the compact
response, and then run the detailed time-expanded assignment once. Compare:

- compact-response versus detailed-assignment counts;
- both predictions versus observed boarding/alighting counts;
- totals and residuals by measurement type, route pattern, trip, platform,
  physical stop, period, mode, and transfer place;
- journey-origin/final-destination flows separately from leg events;
- transfer conservation and route-share drift.

This selects an adequate model class; it is not predictive evidence.

### Predictive validation

After selecting the class, refit with structured groups held out. Hold out
whole trips, platform time series, route-pattern periods, or physical-stop
blocks rather than random individual rows. Score calibration and holdout sets
separately and verify that held-out values never affect estimation.

Residual diagnostics may recommend one relaxation at a time, but never modify
the model automatically.

## 13. Public test plan

No private TPG files enter the public repository. Reuse the current simple and
Geneva examples, then add compact synthetic fixtures for:

- one directed line with exact marginals;
- two intersecting lines and a mandatory transfer;
- multiple feasible direct/transfer journeys;
- disconnected and schedule-dependent OD pairs;
- physical-stop platforms connected by a footpath;
- inconsistent boarding/alighting totals;
- noisy marginals and missing observations;
- known gravity and low-rank demand;
- temporal effects, zone aggregation, and zone splitting.

Required tests cover deterministic route-pattern and service-period identities,
RAPTOR feasibility against a small enumerated timetable, structural zeros,
journey-leg accounting, IPF recovery and infeasibility, balanced and
unbalanced Sinkhorn, gravity recovery, JAX finite differences, reduced response
equivalence to a small explicit operator, `H=B Phi`, fixed-demand offsets,
equivalence compression, full-table reconstruction, warm starts, lineage,
adequacy, holdouts, serialization, corrupted-cache rejection, deadlines, and
memory scaling independent of the detailed assignment size.

## 14. Benchmark plan and targets

Targets are gates to measure, not promises.

| Scale | Purpose | Measurements |
|---|---|---|
| Tiny enumerated | Exact correctness | journey sets, response equality, gradients |
| Existing two-line examples | Transfer semantics and recovery | fit error, parameter recovery, detailed-assignment agreement |
| Geneva public snapshot | Real timetable structure | preprocessing time/RSS/cache size, warm fit time/RSS |
| Synthetic scaling series | Asymptotic behavior | vary stops, patterns, periods, `C`, `nnz(B)`, `K` |
| Private TPG, only after gates pass | Operational readiness | full preprocessing and repeated model fits |

Provisional public acceptance targets:

- preprocessing is restartable and bounded by route/journey summaries rather
  than time-expanded events;
- compact cache size scales with `nnz(B)` and remains at least an order of
  magnitude below the detailed routing cache on a comparable benchmark;
- a warm J0 objective/gradient takes seconds, not minutes, on Geneva-scale
  data and uses comfortably less than 16 GiB;
- repeated specifications reuse the timetable and response cache;
- optimizer iterations never reconstruct the full OD table;
- detailed assignment is called only by an explicit validation command.

The private TPG target is minimal-model fitting in seconds or a few minutes on
a laptop, but it is not accepted until measured. A several-hour one-time
preprocessing run is acceptable if restartable and reusable.

### Pre-implementation runtime feasibility assessment

The proposed backend is computationally promising, but only if its compact
contracts are enforced. This assessment distinguishes three costs that must
never be combined in one headline timing:

1. **cold timetable and journey preprocessing**, performed once per compatible
   timetable, physical-stop mapping, period policy, and choice-set policy;
2. **cold response construction and JAX compilation**, performed once per
   compatible journey/measurement specification and compiled shape;
3. **warm estimation**, repeated for alternative statistical specifications,
   priors, initial values, or observations that preserve the cache contract.

The previous full-network difficulty came from graph traversal and retained
assignment state, not from the number of statistical coefficients. The
existing detailed fixed-routing measurements show approximately 12 seconds for
an ordinary value--gradient evaluation and approximately 6.7 seconds with its
explicit adjoint for one representative destination calculation, with roughly
14--16 GiB resident memory. Repeating such traversal in an optimizer is not an
acceptable design for the new backend.

By contrast, the already implemented small TPG direct operator has 158,219
nonzeros and a warm value--gradient time near 1 millisecond. The proposed
journey-response operator should retain this computational character: after
preprocessing, each gravity evaluation consists mainly of segmented softmax,
one sparse response product, a measurement likelihood, and one transpose
product. Its leading work is

\[
 O(CK + \operatorname{nnz}(B) + M),
\]

not the number of time-expanded nodes or destination-specific graph scans.
Here, `C` is the number of feasible OD--time cells, `K` the small feature or
parameter dimension, `B` the compact journey-event response, and `M` the
number of measurements.

For the known full-network scale of approximately 628,000 free OD cells and
426,000 measurements, a deliberately conservative planning envelope is:

| Quantity | Plausible first-model envelope | Main control |
|---|---:|---|
| Journey alternatives per feasible cell | 1--3 retained alternatives | Pareto pruning and deterministic choice cap |
| Mean vehicle legs per journey | 1--3 | maximum two transfers |
| Response nonzeros | roughly 2--10 million | event atoms and exact equivalence compression |
| Compact numerical artifacts | hundreds of MiB; target below 4 GiB total RSS | integer widths, CSR/COO layout, bounded alternatives |
| Warm J0 objective plus gradient | working estimate 0.05--0.5 s on CPU | fused segmented operations and sparse products |
| Typical 50--200 objective evaluations | working estimate 3--100 s of kernel time | compiled objective and L-BFGS evaluation count |
| Repeated end-to-end J0 fit from valid caches | target below 5 min | cache reuse and no full OD reconstruction |

These are engineering forecasts, not measured promises. The lower part of the
range follows a near-linear extrapolation from the existing sparse direct
operator; the upper part allows for several alternatives, segmented softmax,
larger measurement arrays, sparse-index traffic, optimizer overhead, and CPU
memory bandwidth. JAX can parallelize the array kernels across CPU threads,
but Python loops over OD cells, alternatives, measurements, or destination
groups would invalidate the forecast and are prohibited from the warm path.

The main residual runtime risk is preprocessing rather than estimation.
Naively running a separate journey search for every OD--time cell could again
take hours. RAPTOR preprocessing must therefore work by origin/departure range,
reuse route scans across destinations, bound transfer rounds, prune labels
during—not after—search, run independent origins or periods in parallel, and
write restartable partitions. A long one-time build can be tolerated only if
its artifact is reusable; it must never be repeated by an optimizer iteration.

### Runtime stop/go gates

The following gates protect the project from discovering the same scalability
problem only on the private network:

| Gate | Evidence required | Proceed | Stop and redesign |
|---|---|---|---|
| Phase 4, route search | synthetic scaling by origins, periods, patterns, and transfer rounds | close to linear in route scans; bounded memory; projected full preprocessing is operationally acceptable and restartable | superlinear label growth, unbounded alternatives, or projected preprocessing that cannot be partitioned/resumed |
| Phase 6, response cache | measured alternatives, legs, `nnz(B)`, bytes and construction throughput | projected full artifact below 4 GiB RSS target and no dense `M x C` object | projection above 8 GiB or any allocation proportional to dense `M x C` |
| Phase 7, operator | warm matvec/rmatvec scaling through and beyond Geneva size | near-linear in `nnz(B)` and `M`; projected full product comfortably below 0.5 s | projected product above 2 s or evidence of graph traversal in a product |
| Phase 9, objective | cold compile, first execution, warm value and gradient over a synthetic full-scale shape | preferred warm value--gradient below 0.5 s; hard admission ceiling 2 s | warm value--gradient above 2 s, recompilation between parameter values, or full OD reconstruction |
| Phase 10, estimator | 50--200 evaluation runs with checkpoints and diagnostics | preferred cached fit below 5 min; hard admission ceiling 15 min | repeated cached fit above 15 min or runtime dominated by Python/checkpoint overhead |
| Before Phase 15 | measured scaling model with uncertainty from public cases | upper confidence projection passes memory and time ceilings | no private full-network run; revise representation or model first |

The 2-second objective ceiling is intentionally much stricter than merely
being faster than the current exact assignment. At 200 evaluations it already
implies about 6.7 minutes of kernel time before optimizer and I/O overhead. It
therefore marks the point at which implementation work must focus on the
operator representation rather than proceed to larger data.

Runtime regression tests should use ratios or scaling slopes rather than
fragile absolute CI timings. Absolute admission measurements must be performed
in a controlled benchmark process and record hardware, thread counts, dtype,
JAX/JAXLIB versions, cold/warm state, compilation time, cache identity, RSS,
`C`, `M`, alternatives, legs, and `nnz(B)`.

## 15. Documentation plan

Documentation is part of every implementation phase, not a final cleanup
activity. The reference report
`docs/reports/traffic_assignment.tex` is the normative scientific description
of the implemented estimation methods and must remain consistent with the
production code at every accepted phase boundary.

For every phase:

1. classify the report impact before editing code: `none`, `clarification`, or
   `method change`;
2. update `traffic_assignment.tex` in the same phase whenever contracts,
   equations, assumptions, algorithms, configuration, outputs, validation, or
   limitations change;
3. add or update the relevant cross-reference to the implementation and public
   example;
4. compile with `latexmk -pdf traffic_assignment.tex` from `docs/reports` and
   inspect the log for errors, undefined references, and material layout
   regressions;
5. record an explicit `no report change required` justification when the impact
   is `none`.

The report must progressively include:

1. a conceptual chapter distinguishing journey OD, leg OD, and observed
   boarding/alighting events;
2. the observation contract, transfer semantics, structural zeros, and fixed
   demand;
3. route-level IPF and entropy baselines, including when their marginals are
   scientifically valid;
4. the reduced response operator and conditional gravity formulation;
5. ML/MAP estimation, regularization, identifiability, diagnostics, and model
   progression;
6. the role of detailed assignment as an explicit validation backend rather
   than part of the reduced estimator's inner loop;
7. benchmark tables separating cold preprocessing, compilation, warm
   estimation, reconstruction, and final detailed validation;
8. limitations: fixed route shares, no capacity feedback in the first model,
   nonidentification of transfers from APC alone, and diagnostic rather than
   causal relaxation recommendations.

The TOML schema, valid values, defaults, fingerprints, cache compatibility,
tutorials, and operational commands belong in the user documentation. The
LaTeX report should explain the scientific model and its actual implementation,
without duplicating the complete API reference.

## 16. Migration and compatibility

- No existing inference API is removed or silently redirected.
- Existing detailed, sharded, stochastic, and block-coordinate backends remain
  available under their current names.
- New result files carry a backend identifier and cannot be loaded as current
  gravity/fixed-routing results without an explicit adapter.
- Canonical OD keys and fixed-demand semantics are shared, enabling explicit
  reconstruction into `CompactODAssignmentLayout` for final assignment.
- The new backend may compare against an existing detailed routing cache, but
  never requires it to fit.
- Cache schema changes require version bumps and fail-closed validation; no
  best-effort reuse of mismatched journey choices is allowed.

## 17. Scientific and data questions

### Phase 0 decisions for public implementation

The detailed contract is recorded in
`docs/reports/reduced_od_observation_contract.md`. The public implementation
uses exact timetable-event boarding/alighting observations after external APC
cleaning; absent records are unobserved rather than zero. Journey origins and
destinations are first boarding and final alighting. Desired-departure bins are
half-open. Outputs default to scenario-stop identifiers. Productions are either
provided journey-entry totals or an explicitly regularized estimated basis;
raw boarding totals are never a production mode.

Physical-stop mappings and footpaths are explicit fingerprinted inputs.
Authoritative mappings are preferred, and generated mappings require review.
Public tests may use declared fixtures, so these data choices do not block
package construction.

### Blocking before private full-network validation

1. Which APC cleaning, outage, and exclusion rules produced the private counts?
2. Is an authoritative platform-to-physical-stop and walking-transfer mapping
   available, or who approves a generated mapping?
3. Which production mode and, for an estimated basis, which spatial and
   temporal resolution should be used?
4. Which output geography is required for the private deliverable?
5. Which service day, analysis period, and after-midnight convention is
   authoritative?
6. Which events lack operating sensors? Missing coverage must be explicit.

### Defaults that can safely be proposed and reviewed

- maximum two transfers in the first journey choice set, matching the current
  policy that three or more transfers are structural zeros;
- earliest-arrival and transfer-count Pareto criteria;
- fixed route shares during one fit;
- Poisson for deterministic synthetic recovery and negative binomial for noisy
  counts;
- coarse broad periods before low-rank temporal factors;
- no capacity feedback in the first reduced model, with a documented warning;
- one parent-to-child relaxation at a time.

### Nonblocking future questions

- availability of multiple comparable days for state-space or Gaussian-process
  temporal models;
- whether frequent services justify frequency-based hyperpaths;
- whether route shares should be updated by an outer validation loop;
- whether walking time and destination attraction require external GIS or land
  use inputs.

## 18. Precise implementation phases

No phase starts until this workflow and the blocking data contract are approved.

### Sequential launch and acceptance protocol

The phases below are independent launch units and are executed strictly in
numeric order. A request such as **“Proceed with Phase 3”** authorizes Phase 3
only; it does not authorize Phase 4. Multiple phases are not combined unless
the user explicitly requests a range.

Each phase follows the same gated sequence:

1. **Preflight:** record `git status --short` and `git rev-parse HEAD`; inspect
   overlapping user changes; restate the phase scope and report impact.
2. **Implementation:** make only the production, test, example, and
   documentation changes listed for that phase.
3. **Focused validation:** run the new/modified unit and integration tests and
   the phase-specific performance measurement.
4. **Regression validation:** run the smallest relevant existing suite that
   proves backward compatibility. Broader campaigns occur only at designated
   gates.
5. **Report synchronization:** update `traffic_assignment.tex` as required,
   compile it with `latexmk -pdf`, and check its log. This step is mandatory
   even when its recorded outcome is `no report change required`.
6. **Acceptance report:** show changed files, test and benchmark results,
   report changes, unresolved risks, and final `git status --short`. Do not
   start the next phase.

A phase is accepted only when its acceptance criteria, tests, performance
measurement, and report-synchronization gate all pass. If a phase fails, work
stops within that phase until the defect is fixed or the user explicitly
accepts a documented limitation. A conditional phase, such as an entropy or
low-rank extension, is still launched in sequence; it may close as
`not applicable` only with evidence and a corresponding explanation in the
report. No phase commits or pushes changes unless explicitly requested.

The phase hand-off record is therefore:

```text
phase number and status
HEAD and preserved pre-existing working-tree state
implemented contracts and files
focused and regression test results
performance measurements against the previous accepted phase
traffic_assignment.tex sections changed (or justified no-change decision)
latexmk result and relevant warnings
known limitations and decision needed before the next phase
```

### Phase 0 — Observation and journey contract

- **Inputs:** answers to blocking questions; representative public schemas.
- **Outputs:** journey/leg/event definitions, canonical keys, approved TOML
  outline, identifiability note. The normative Phase 0 design record is
  `docs/reports/reduced_od_observation_contract.md`.
- **Files/modules affected:** design docs and the journey/leg/measurement
  terminology in `docs/reports/traffic_assignment.tex`; later `contracts.py`.
- **Tests:** schema examples and invalid-case tables, no computation.
- **Performance measurements:** none.
- **Acceptance criteria:** no boarding is implicitly classified as a journey
  origin; transfer and missing-count semantics are explicit; the LaTeX report
  compiles and uses the approved terminology consistently.
- **Dependencies:** workflow approval.

### Phase 1 — Package skeleton and immutable contracts

- **Inputs:** Phase 0 contract.
- **Outputs:** package namespaces, typed configs, artifact dataclasses, schema
  versions, canonical serialization and fingerprints.
- **Files/modules affected:** new `preprocessing/reduced_od/{config,artifacts}.py`,
  `inference/reduced_od/contracts.py`, package exports.
- **Tests:** TOML valid/invalid values, deterministic fingerprints, immutable
  arrays, round trips.
- **Performance measurements:** serialization size/time on synthetic metadata.
- **Acceptance criteria:** no assignment/sharded imports; unknown config keys
  fail; equivalent inputs have identical fingerprints.
- **Dependencies:** Phase 0.

### Phase 2 — Physical stops, route patterns, and service periods

- **Inputs:** `Scenario`, approved platform mapping and period policy.
- **Outputs:** normalized stop places, route-pattern classes, service-period
  classes, array timetable index.
- **Files/modules affected:** `physical_stops.py`, `route_patterns.py`,
  `service_periods.py`, `timetable_index.py`.
- **Tests:** repeated-stop patterns, opposite directions, express/local trips,
  platforms, after-midnight trips, deterministic grouping.
- **Performance measurements:** build time, peak RSS, retained bytes.
- **Acceptance criteria:** every trip/stop-time maps exactly once; grouping does
  not merge different ordered stop sequences; no time-expanded graph exists.
- **Dependencies:** Phase 1 and platform decision.

### Phase 3 — Route-level IPF baseline

- **Inputs:** route patterns, event counts and masks.
- **Outputs:** route-leg OD/alighting probabilities, reconciled marginals,
  infeasibility and data-quality report.
- **Files/modules affected:** `inference/reduced_od/route_level.py`.
- **Tests:** exact/noisy/inconsistent marginals, zeros, triangular support,
  convergence, transfer boarding labels.
- **Performance measurements:** iterations, wall time, RSS versus route length.
- **Acceptance criteria:** exact marginals are recovered to tolerance; impossible
  cases fail diagnostically; outputs are explicitly labeled leg-level.
- **Dependencies:** Phases 1–2.

### Phase 4 — RAPTOR feasibility and feature summaries

- **Inputs:** timetable index, footpaths, origin/time queries, transfer bound.
- **Outputs:** feasible destinations, Pareto labels, travel/wait/walk/transfer
  features and structural-zero reasons.
- **Files/modules affected:** `raptor.py`, feature artifact extensions.
- **Tests:** enumerated one/two-line cases, mandatory transfer, multiple routes,
  disconnected stops, departure-time feasibility, range queries.
- **Performance measurements:** queries/second, preprocessing time/RSS, cache
  bytes by stops, patterns, and periods.
- **Acceptance criteria:** exact agreement with enumerated journeys; bounded
  transfer rounds; no global time-expanded graph; deterministic results.
- **Dependencies:** Phase 2.

### Phase 5 — Compact journey choices and transfer accounting

- **Inputs:** Pareto labels, route-level initialization, choice-set policy.
- **Outputs:** deterministic journey alternatives, leg sequences, fixed initial
  shares, first/final and internal-transfer event labels.
- **Files/modules affected:** `journey_choices.py`.
- **Tests:** event conservation, platform transfers, route alternatives,
  deterministic pruning, journeys crossing periods.
- **Performance measurements:** alternatives and bytes per OD cell, build RSS.
- **Acceptance criteria:** every internal transfer has paired leg events; no
  feasible cell lacks a choice; choice caps and pruning are reported.
- **Dependencies:** Phases 3–4.

### Phase 6 — Direct measurement responses and cache

- **Inputs:** journey choices, strict measurement table, fixed demand.
- **Outputs:** sparse response atoms `B`, fixed offset, measurement-specific
  manifest, exact equivalence classes.
- **Files/modules affected:** `response_atoms.py`, `equivalence.py`,
  `persistence.py`, adapted measurement mapping entry point.
- **Tests:** explicit tiny operator equality, missing sensors, duplicate
  mappings, fixed offsets, compression equivalence, corrupted caches.
- **Performance measurements:** `nnz(B)`, cache size, build time/RSS, compression
  ratio.
- **Acceptance criteria:** direct predictions match enumerated leg events;
  response construction never materializes `A`; cache identity is fail-closed.
- **Dependencies:** Phases 1 and 5 plus observation contract.

### Phase 7 — Reduced linear operator and basis response

- **Inputs:** `B`, optional sparse/separable basis `Phi`.
- **Outputs:** bounded matvec/rmatvec, direct `H=B Phi`, operator diagnostics.
- **Files/modules affected:** `inference/reduced_od/response_operator.py`.
- **Tests:** adjoint identities, dense reference equality, sparse/dense `H`,
  empty/fixed cells, JAX gradients.
- **Performance measurements:** product time/RSS versus `C`, `nnz(B)`, and `K`.
- **Acceptance criteria:** no full OD-to-link matrix; memory tracks compact
  response/basis size; `H` equals a tiny explicit reference.
- **Dependencies:** Phase 6.

### Phase 8 — Balanced and unbalanced entropy baselines

- **Inputs:** feasible support, generalized costs, approved journey marginals.
- **Outputs:** transport plans, convergence/marginal diagnostics, warm-start
  demand for gravity.
- **Files/modules affected:** `inference/reduced_od/entropy.py`.
- **Tests:** balanced exact marginals, unbalanced noisy totals, disconnected
  support, numerical stabilization, sensitivity to regularization.
- **Performance measurements:** iteration time/RSS and convergence by support.
- **Acceptance criteria:** balanced marginals meet tolerance; unbalanced
  deviations match the declared penalty; transfer APC is never passed as a
  journey marginal without explicit conversion.
- **Dependencies:** Phases 4 and 7; may be deferred if no valid marginals exist.

### Phase 9 — Minimal conditional gravity model

- **Inputs:** compressed features, productions or production-state decision,
  response operator, observations.
- **Outputs:** J0 specification/layout, demand generator, Poisson/NB objective,
  exact JAX gradients.
- **Files/modules affected:** `features.py`, `specification.py`, `parameters.py`,
  `objective.py`.
- **Tests:** normalization by origin-time group, structural zeros, extreme
  utilities, parameter recovery, finite differences, fixed offset.
- **Performance measurements:** cold compile and warm value/gradient time/RSS.
- **Acceptance criteria:** parameter dimension follows the declared model, not
  `C`; no full OD reconstruction during evaluation; synthetic truth recovered.
- **Dependencies:** Phases 6–7 and production decision.

### Phase 10 — Estimator and operations

- **Inputs:** J0 objective and initial parameters from IPF/entropy/defaults.
- **Outputs:** ML/MAP estimator, checkpoints, manifests, progress, deadlines,
  restart and result schema.
- **Files/modules affected:** `estimator.py`, `operations.py`.
- **Tests:** warm/cold execution, checkpoint resume, incompatible fingerprint,
  deadline boundary, ML/MAP equivalence when prior is flat.
- **Performance measurements:** iterations, compile time, warm iteration time,
  RSS, checkpoint overhead.
- **Acceptance criteria:** repeated fits reuse caches; laptop-scale synthetic
  fits complete in seconds/minutes; interruption never publishes completion.
- **Dependencies:** Phase 9.

### Phase 11 — Reconstruction and detailed validation adapter

- **Inputs:** fitted reduced model, OD key/layout, detailed assignment config.
- **Outputs:** optional full OD table, compact prediction, one-shot detailed
  assignment comparison and transfer audit.
- **Files/modules affected:** `reconstruction.py`, validation adapter outside
  the estimator hot path.
- **Tests:** canonical reconstruction, fixed/structural-zero cells, compact vs
  detailed counts on public small networks.
- **Performance measurements:** reconstruction and final-assignment time/RSS,
  reported separately from fitting.
- **Acceptance criteria:** optimizer never calls detailed assignment; full table
  is reconstructed only on explicit request; discrepancies are quantified.
- **Dependencies:** Phase 10 and existing assignment backend.

### Phase 12 — Adequacy, holdout, and advisory relaxation

- **Inputs:** fitted results, measurement metadata and lineage.
- **Outputs:** adequacy report, structured holdout validation, one-step
  relaxation recommendations.
- **Files/modules affected:** `validation.py`, `diagnostics.py`, `lineage.py`.
- **Tests:** no holdout leakage, route/transfer group summaries, weak
  identification warnings, warm starts J0–J4.
- **Performance measurements:** diagnostic and child-score overhead.
- **Acceptance criteria:** adequacy and prediction remain distinct; advice is
  not automatic; every child has an auditable parent and warm start.
- **Dependencies:** Phases 10–11.

### Phase 13 — Low-rank and sparse corrections

- **Inputs:** accepted structured baseline and residual evidence.
- **Outputs:** constrained low-rank factors and/or small sparse correction
  dictionary.
- **Files/modules affected:** specification, parameters, objective, proximal
  solver if required.
- **Tests:** identifiability constraints, known-rank recovery, rank expansion,
  shrinkage, held-out comparison.
- **Performance measurements:** scaling by rank/dictionary size.
- **Acceptance criteria:** child improves structured validation without unstable
  factors; parameter growth remains explicit and laptop-feasible.
- **Dependencies:** Phase 12; not required for first TPG pilot.

### Phase 14 — Public examples, benchmarks, and documentation

- **Inputs:** accepted phases and public examples.
- **Outputs:** tutorials, benchmark reports, TOML reference, migration guide,
  and a consolidated audit of the LaTeX methodology accumulated in Phases
  0--13.
- **Files/modules affected:** `docs/source/examples`, `docs/reports`, benchmark
  scripts and committed small results.
- **Tests:** documentation commands and examples run in CI-sized configurations.
- **Performance measurements:** tiny, two-line, Geneva, and synthetic scaling
  tables with cold/warm separation.
- **Acceptance criteria:** public evidence meets Phase 14 targets; no private
  data; limitations and failed approaches are documented; every implemented
  backend, equation, assumption, and validation boundary is accurately
  represented in `traffic_assignment.tex`; `latexmk -pdf` succeeds.
- **Dependencies:** Phases 3–12 as applicable.

### Phase 15 — Private full-network pilot

- **Inputs:** immutable public release, private TPG adapter and data.
- **Outputs:** preprocessing and fit reports only in the private repository.
- **Files/modules affected:** none in public during the run.
- **Tests:** private schema/fingerprint preflight; no private data copied back.
- **Performance measurements:** cache size, phase times/RSS, warm J0 fitting,
  adequacy assignment, exact detailed benchmark comparison.
- **Acceptance criteria:** fitting excludes detailed assignment, fits laptop
  resource targets or reports why not, and preserves journey/leg semantics.
- **Dependencies:** approved public acceptance campaign and blocking TPG data
  questions resolved.

## 19. Risks and explicit non-goals

- APC counts alone may not identify journey origins, destinations, and transfer
  stitching; regularization does not create information.
- Fixed route shares can bias OD estimates when alternatives differ materially.
- Equivalence compression is exact only when denominator membership, features,
  and response are identical.
- Frequency approximations may erase schedule-dependent feasibility.
- Capacity and crowding feedback are excluded initially; the detailed
  assignment must reveal whether this omission is material.
- A low-dimensional parameter vector does not guarantee identification;
  curvature, profile likelihood, and holdout behavior remain necessary.
- The project does not initially implement real-time estimation, Gaussian
  processes, or a general multimodal trip planner.

The immediate milestone is not a full TPG estimate. It is a public two-line
demonstration showing correct transfer accounting and a compact response whose
runtime and memory no longer scale with the detailed time-expanded assignment
state.
