# Reduced-OD public integration gap analysis

Audit date: 2026-08-05. Scope: the uncommitted reduced-OD implementation based
on public synthetic inputs only. Benchmark scripts are not treated as APIs.

| Capability | Existing public symbol | Module | Status before this integration pass | Tests | Required correction / outcome |
|---|---|---|---|---|---|
| Configuration parsing and validation | `load_reduced_od_config` | `preprocessing.reduced_od.config` | Partial | `test_reduced_od_config.py` | Schema 2 now requires service day/window, after-midnight, APC/coverage/outage identifiers, wait/journey/alternative limits, footpath policy, likelihood, and production semantics. |
| Physical-stop mapping | `build_physical_stop_index` | `physical_stops` | Complete algorithm, absent independent persistence | `test_reduced_od_timetable_index.py` | Persisted independently and linked fail-closed. |
| Service periods | `build_service_period_index` | `service_periods` | Complete algorithm, absent independent persistence | timetable tests | Persisted with route patterns. |
| Route patterns | `build_route_pattern_index` | `route_patterns` | Complete algorithm, absent independent persistence | route-level/timetable tests | Persisted with service periods. |
| Timetable indexing | `prepare_reduced_od_timetable` | `timetable_index` | Complete numerical index, partial persistence | timetable tests | Stable orchestration persists and reloads it. |
| RAPTOR accessibility | `run_raptor_query`, `run_raptor_range_query` | `raptor` | Partial | `test_reduced_od_raptor.py` | Added explicit maximum wait and journey duration to query identity and filtering. |
| Journey-choice generation | `build_journey_choices` | `journey_choices` | Complete per query, absent persistence/orchestration | journey tests | Orchestrator builds bounded public choices and persists them. |
| Response atoms | `build_measurement_response` | `response_atoms` | Complete | response tests | Used directly; never builds dense measurement-by-OD. |
| Response-equivalence compression | `build_response_equivalence` | `equivalence` | Complete | response/operator tests | Persisted independently and revalidated by the operator builder. |
| Persistence / fail-closed loading | response-cache functions only | `persistence` | Partial | response cache tests | Added general atomic phase store with schema, content/config/upstream fingerprints, shapes, dtypes, version and semantics. |
| Production totals | `build_conditional_gravity_features` mapping input | `features` | Partial | minimal-gravity tests | Typed semantics now distinguish external journey totals, transfer-adjusted totals, estimated basis, and labelled route-leg baseline. Persisted separately. |
| Production basis | `MinimalGravityProblem.production_basis` | `objective` | Complete numerical support, absent persistence-level contract | minimal-gravity tests | Builder validates specification/configuration and basis dimensions. |
| Destination attractiveness | mapping input to feature builder | `features` | Partial | minimal-gravity tests | Persisted separately before compact feature construction. |
| J0 parameter layout | `MinimalGravityParameterLayout` | `parameters` | Complete | minimal-gravity tests | High-level builder exposes names and raw/transformed dimensions. |
| Reduced response operator | `build_reduced_response_operator` | `response_operator` | Complete | operator tests | Persisted independently; diagnostics exposed by preflight. |
| Conditional-gravity objective | `evaluate_minimal_gravity_objective` | `objective` | Complete | minimal-gravity tests | Builder binds compact artifacts only. |
| Poisson / negative-binomial likelihood | `MinimalGravitySpecification` and objective | `specification`, `objective` | Complete | minimal-gravity tests | Likelihood is now required in the preprocessing configuration fingerprint. |
| ML / MAP | `estimate_minimal_gravity` | `estimator` | Partial operational behavior | estimator tests | Removed duplicate initial/final objective-gradient work; deadline checked around evaluations; accepted-point checkpoints preserved. |
| Checkpoints / restart | checkpoint save/load functions | `operations` | Partial | estimator tests | Atomic and fail-closed already; resume counting and accepted boundaries retained. |
| Flat prior | `GaussianRawParameterPrior` | `operations` | Complete | explicit ML/MAP equivalence test | Infinite scales remain exact zero objective/gradient contributions. |
| Adequacy | `diagnose_reduced_od_adequacy` | `diagnostics` | Complete | validation tests | Explicitly labelled in-sample. |
| Grouped holdout / refit | split and validation functions | `validation` | Complete | validation tests | Complete vehicle-journey grouping, complementary masks, deterministic fingerprint and refit already public. |
| J1–J4 advice | `recommend_reduced_od_relaxations` | `diagnostics` | Complete | validation tests | Advisory-only contract retained. |
| Model lineage | lineage constructors | `lineage` | Complete | validation tests | Stable public imports retained. |
| Canonical OD reconstruction | `reconstruct_full_od` | `reconstruction` | Complete and explicit | reconstruction tests | Remains outside fitting, preflight and diagnostics. |
| Manifests / fingerprints | artifact/problem/model contracts | `artifacts`, `contracts` | Partial | contracts tests | Added full phase dependency graph and problem manifest. |
| Performance/memory diagnostics | benchmark scripts | `benchmarks/` | Absent stable API | new integration test | Added `benchmark_minimal_gravity_objective` and phase timing/peak-RSS reports. |

## Remaining scientific decisions

The public library cannot decide whether a private APC data set identifies
passenger-journey productions. The adapter must explicitly select and document
external journey totals, a transfer adjustment, an estimated production basis,
or a route-leg baseline. It must also define physical-stop and footpath policy,
departure sampling, sensor exclusions, destination attractiveness and structural
fixed cells. Old time-expanded assignment caches are not compatible inputs.
