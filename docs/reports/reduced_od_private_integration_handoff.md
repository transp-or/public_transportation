# Reduced-OD J0 private integration handoff

## Revision and import verification

Required base revision: `55ed81601d0d786f80b333f4e2b0474505c6a472`
plus the uncommitted reduced-OD integration files described by `git status`.
No commit identifier exists yet because this work was intentionally not
committed. Verify the checkout explicitly:

```bash
python -c "import public_transportation; print(public_transportation.__file__, public_transportation.__version__)"
```

The path must resolve under the intended public checkout. Do not point the
adapter at old detailed-assignment or time-expanded routing caches.

## Stable public imports

```python
from public_transportation.preprocessing.reduced_od import load_reduced_od_config
from public_transportation.inference.reduced_od import (
    MinimalGravitySpecification,
    ReducedODPreparationInputs,
    benchmark_minimal_gravity_objective,
    build_minimal_gravity_problem,
    load_reduced_od_artifacts,
    preflight_reduced_od_j0,
    prepare_reduced_od_artifacts,
    estimate_minimal_gravity,
    diagnose_reduced_od_adequacy,
    build_reduced_od_holdout_split,
    validate_reduced_od_holdout,
    recommend_reduced_od_relaxations,
    reconstruct_full_od,
)
```

## Dependency graph and filenames

```text
configuration
  -> physical_stops
  -> service_periods_route_patterns
  -> timetable_index
  -> journey_choices
       -> measurement_response -> response_equivalence
       -> production_inputs
       -> destination_attractiveness
  -> conditional_gravity_features
  -> reduced_response_operator
  -> problem_manifest
```

Each name is a directory containing `manifest.json` and zero or more immutable
`array_*.npy` files. Every manifest records schema and library versions,
content/configuration/upstream fingerprints, semantic conventions, dimensions,
and array dtypes/shapes/digests. The directories are independently loadable by
the phase store, while `load_reduced_od_artifacts` validates the graph in order.

## Configuration schema

Schema version 2 requires explicit service day, analysis start/end seconds,
extended-service-day after-midnight convention, APC cleaning identifier, sensor
coverage and outage policies, half-open time bins, maximum transfers/wait/journey,
alternative cap, footpath policy, physical-stop mapping policy, production mode
and semantics, output geography, likelihood, and explicit-only detailed
assignment. See
`tests/bayesian_estimation/test_reduced_od_integration.py::_configuration` for a
complete public synthetic file.

Production semantics are one of:

- `external_journey_productions`;
- `transfer_adjusted_journey_productions`;
- `estimated_production_basis`;
- `route_leg_baseline` (explicitly a baseline, not passenger-journey totals).

## Expensive preprocessing and repeated estimation

```python
prepared = prepare_reduced_od_artifacts(
    scenario=scenario,
    measurements=measurements,
    configuration=configuration,
    inputs=ReducedODPreparationInputs(...),
    output_directory=artifact_directory,
    cache_policy="reuse_or_build",
    progress=progress,
)
```

Use `cache_policy="rebuild"` to force preprocessing and `"read_only"` to forbid
it. The read-only admission path is:

```python
report = preflight_reduced_od_j0(
    configuration=configuration,
    artifact_directory=artifact_directory,
)
assert report["compatible"], report
artifacts = load_reduced_od_artifacts(
    configuration=configuration,
    artifact_directory=artifact_directory,
)
specification = MinimalGravitySpecification(
    likelihood=configuration.model.likelihood,
    production_mode=configuration.productions.mode,
)
built = build_minimal_gravity_problem(
    artifacts=artifacts,
    specification=specification,
)
```

For `estimated_basis`, provide the two-dimensional production basis both when
building and during preflight, and set `production_basis_columns` accordingly.

## Objective admission benchmark

```python
timing = benchmark_minimal_gravity_objective(
    problem=built.problem,
    raw_parameters=initial_raw,
    warm_evaluations=5,
)
assert timing.finite
assert timing.recompiled_after_value_change is not True
```

The report separates trace, lowering, compilation, first execution and warm
times and records process RSS. Threshold policy belongs in the private adapter.

## ML, MAP and restart

```python
ml = estimate_minimal_gravity(
    problem=built.problem,
    initial_raw_parameters=initial_raw,
    model_fingerprint=built.model_fingerprint,
    checkpoint_path="j0-ml.checkpoint.json",
)
resumed = estimate_minimal_gravity(
    problem=built.problem,
    initial_raw_parameters=initial_raw,
    model_fingerprint=built.model_fingerprint,
    checkpoint_path="j0-ml.checkpoint.json",
    resume=True,
)
```

For MAP, use `ReducedODFitConfig(method="map")` and
`GaussianRawParameterPrior`. Infinite scales are exact flat priors and reproduce
ML to deterministic optimizer tolerance. Checkpoint identity includes model,
method and parameter dimension and rejects incompatibility.

## Adequacy, holdout, advice and reconstruction

`diagnose_reduced_od_adequacy` is explicitly in-sample. Build a deterministic
`vehicle_journey` grouped split with `build_reduced_od_holdout_split`, then call
`validate_reduced_od_holdout`; its calibration/holdout metrics include totals,
log likelihood, Poisson and NB deviances, MAE, RMSE and variance-weighted RMSE.
Pass adequacy and metadata to `recommend_reduced_od_relaxations` for advisory
J1–J4 children and use lineage helpers for warm starts.

Full OD materialization is deliberately separate: construct the canonical
`ReducedODProblemContract` and call `reconstruct_full_od` only after estimation.
Detailed assignment validation remains another explicit adapter-owned action.

## Failure and recovery

Loading stops at the first missing, corrupt or incompatible phase. Preflight
returns `missing_or_incompatible_phase`, the exact error and the public rebuild
call. Never copy a downstream directory across configurations. Re-run
`prepare_reduced_od_artifacts(..., cache_policy="reuse_or_build")`; use
`"rebuild"` after policy or source-data changes.

## Responsibility boundary and limitations

Public code owns deterministic preprocessing algorithms, compact likelihoods,
optimization, diagnostics, grouped validation, fingerprints and persistence.
The private adapter owns every TPG-specific path, cleaning/coverage rule,
physical-stop map, footpaths, departure sampling, production interpretation,
attractiveness, structural fixed cells and acceptance thresholds.

Current limitations: departure instants are explicitly supplied rather than
generated from a private service policy; multi-query alternatives are merged
deterministically with equal initial shares; artifact reuse is whole-graph (a
single incompatible phase causes an atomic full public rebuild); and no public
code can decide whether APC boardings identify journey productions.
