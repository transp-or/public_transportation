# Numerically robust reduced-OD estimation

Large count likelihoods must be fitted with JAX float64. At an objective near
`1.26e7`, float32 spacing is of order one, so an L-BFGS-B relative function
tolerance such as `1e-10` cannot be resolved. SciPy may then terminate through
relative function reduction while the gradient is still large.

Enable x64 before importing and constructing the problem:

```bash
JAX_ENABLE_X64=true uv run python my_reduced_od_fit.py
```

or, before creating JAX arrays or compiled functions:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

`ReducedODFitConfig` defaults to `float64_required`. It records requested and
actual input, objective, and gradient dtypes. It also compares
`eps * max(1, abs(objective))` with the absolute resolution implied by `ftol`.
An unresolved tolerance fails by default; exploratory float32 must be enabled
explicitly with `ReducedODNumericalConfig` and can request a structured warning
instead.

## Three different decisions

The fit result distinguishes:

1. `optimizer_success`: SciPy's status and message, preserved verbatim;
2. `convergence.numerically_converged`: finite values, resolvable tolerance,
   and a sufficiently small projected gradient; and
3. `convergence.scientifically_admissible`: deliberately left `None` for the
   caller to decide from domain-specific evidence.

Optimizer termination alone never establishes numerical convergence.
Transformed-parameter plausibility is also separate: a stationary solution can
still have dispersion at its positivity floor, extreme production multipliers,
implausible production totals, or singular curvature.

Post-fit diagnostics contain transformed time and transfer sensitivities,
negative-binomial dispersion when present, production coefficients, compact
origin-period production summaries, prior contributions, active raw bounds,
the symmetric Hessian, eigenvalues, rank, condition number, and named weak
direction loadings. No diagnostic reconstructs the full OD vector.

## Diagnostic model comparison

Use the same compact features and response operator for Poisson and negative
binomial fits:

```python
comparison = compare_reduced_od_likelihoods(
    problem=estimated_production_problem,
    initial_raw_parameters={
        "poisson": poisson_initial,
        "negative_binomial": negative_binomial_initial,
    },
    artifact_fingerprint=upstream_artifact_fingerprint,
    fit_configs={
        "poisson": poisson_fit_config,
        "negative_binomial": negative_binomial_fit_config,
    },
    progress=progress_callback,
)
```

Likelihood-specific configurations are necessary when raw bounds are aligned
vectors: Poisson has no dispersion coordinate, whereas negative binomial does.
A single vector therefore cannot be dimensionally correct for both layouts.
The legacy singular `fit_config` remains available only for unbounded,
dimension-independent settings; supplying both forms is rejected.

Named bounds avoid positional ambiguity:

```python
common_bounds = {
    "beta_time": (-10.0, 10.0),
    "beta_transfer": (-10.0, 10.0),
    "global_log_scale": (-3.0, 3.0),
}
poisson_fit_config = ReducedODFitConfig(
    named_raw_parameter_bounds=ReducedODNamedRawParameterBounds(common_bounds)
)
negative_binomial_fit_config = ReducedODFitConfig(
    named_raw_parameter_bounds=ReducedODNamedRawParameterBounds(
        {**common_bounds, "dispersion": (-5.0, 60.0)}
    )
)
```

Canonical raw ordering is `beta_time`, `beta_transfer`, optional `dispersion`,
then the declared production-basis labels (or generated
`production_coefficient_N` names). Unknown names and incomplete strict maps
fail before compilation. Each comparison entry records the ordering, initial
point, resolved bound vectors, shared artifact fingerprint, distinct model
fingerprint, and active solution bounds.

Long operations accept a structured `progress` callback. Events are
JSON-serializable and report phase, status, likelihood or prior scenario,
elapsed time, completed and total work, optimizer iteration, objective, memory,
and estimated remaining time when available. Compilation, checkpoint, deadline,
and completion boundaries are emitted immediately; iteration events are
throttled by `progress_interval_iterations` and
`progress_interval_seconds`. Callbacks never trigger an extra objective or
gradient evaluation. Configure distinct `checkpoint_path` values for the two
fits; a deadline produces a resumable `deadline` result, not a completed one.

Poisson is a diagnostic comparison, not an automatic recommendation. It helps
determine whether a pathological solution is driven by the free dispersion
parameter.

Bounds and MAP alternatives are explicit:

```python
bounded = ReducedODFitConfig(
    raw_parameter_bounds=ReducedODRawParameterBounds(lower, upper)
)

sensitivity = run_reduced_od_prior_sensitivity(
    problem=estimated_production_problem,
    initial_raw_parameters=initial,
    model_fingerprint=model_fingerprint,
    scenarios={"weak": weak_prior, "moderate": moderate_prior},
    fit_config=ReducedODFitConfig(),
)
```

The sensitivity report includes prior fingerprints, ML distances, warm-start
lineage, transformed and production diagnostics, likelihood/prior
contributions, and whether the MAP objective is prior dominated. The library
does not provide undocumented informative prior defaults.

## Artifact reuse

Artifact configuration identities are phase-specific. Service interval,
physical-stop mapping, footpaths, journey limits, departure sampling,
alternative caps, route-share policy, and applicable measurement identity
invalidate their affected preprocessing phases. Likelihood, production mode,
production-basis values, priors, optimizer tolerances, bounds, and holdout
configuration do not invalidate timetable or journey-choice artifacts.

Changing production or attraction inputs republishes only those inputs,
conditional gravity features, and model identity. Changing journey limits
reuses the physical-stop, route-pattern, and timetable phases and rebuilds
journey choices and downstream artifacts. Every preparation result reports a
status and fingerprint for each phase.

Journey-choice preparation emits throttled structured progress with completed
and total query counts, current origin/departure, elapsed and recent query
times, predicted remaining time, and peak RSS. Presentation layers may render
these events with `tqdm`; the core does not depend on it.

The public preprocessing primitives use the same rule. Physical-stop,
route-pattern, service-period and timetable indexing; RAPTOR rounds and
destination classification; journey-choice grouping; measurement-response
aggregation; and phase-artifact persistence accept an optional `progress`
callback. Loop events contain `completed_units`, `total_units`, elapsed time,
observed throughput, `estimated_remaining_seconds`, `eta_confidence`, and the
current unit. They are emitted at startup, completion, approximately every one
percent, or at least at the next unit boundary after ten seconds. A start event
has an unavailable ETA until one unit has calibrated the rate. A single
third-party or backend call cannot report internal progress; it is identified
as one indivisible unit rather than assigned a misleading ETA.

The complete public workflow remains in
`docs/source/examples/reduced_od_j0_integration.py`: prepare artifacts, build a
compact problem, fit ML and bounded/MAP alternatives, inspect convergence and
production diagnostics, compare likelihoods, and run prior sensitivity without
reconstructing preprocessing artifacts.
