# Reduced-dimensional gravity estimation

Progressive-fidelity objective and gradient evaluation, including deterministic
shard sampling, quality diagnostics, control-variate anchors, and resumable
execution, is documented in
[`progressive_fidelity_gravity.md`](progressive_fidelity_gravity.md).
Full-network validation found that its streaming implementation bounds memory
successfully, but uniform persisted-shard sampling is not a validated faster
replacement for exact optimization gradients. See the
[2026-08-05 validation report](../reports/full_network_stochastic_gravity_validation_2026-08-05.md)
before using sub-100% effort.

## Boundary-flow extension

The opt-in boundary-flow and separate boarding/alighting observation-process
contracts are described in [`gravity_boundary_flows.md`](gravity_boundary_flows.md).
They add named linear measurement blocks for initial onboard and terminal
outflow cohorts while preserving the existing OD-only path and assignment
artifact. The case adapter supplies the matrices, row-order identity, and
scientific interpretation; the public package validates their dimensions,
fingerprints, joint parameter layout, and contribution reporting.

Residual diagnostics for persisted fit artifacts are available through the
generic [residual-audit tool](residual_audit.md). It reports ordinary residuals,
near-zero-support failures, arbitrary metadata groupings, contribution
accounting, and optional row-by-row comparison of two runs without rerunning
the model.

## Phase 1: demand contracts

The gravity package represents a complete dynamic OD table as a sparse
one-dimensional list of canonical OD-time cells. It does not create a dense
origin-by-destination-by-time tensor. `GravityFeatures` stores immutable cell
indices, journey times, transfer counts, structural feasibility, externally
prepared origin-time production totals, and externally prepared positive
destination-attractiveness offsets. Applications decide how those totals and
offsets are prepared; the public library performs no geographic downloads or
smoothing.

Each feature set carries the authoritative OD-layout fingerprint and canonical
full-OD indices. `validate_compact_layout` verifies both the fingerprint and
exact free-cell order against `CompactODAssignmentLayout`. Thus fixed-routing
assignment and gravity demand cannot silently disagree about column identity.
Positive frozen cells remain external fixed demand and contribute through the
operator's fixed measurement offset. Prepared `Q_ot` values supplied to the
gravity kernel must describe the production allocated across its free cells
after any externally fixed production has been accounted for.

The minimal specification estimates three positive quantities:

- `beta_time`, multiplying journey time divided by the declared time scale;
- `beta_transfer`, multiplying the transfer count;
- `dispersion`, reserved for the Phase-2 negative-binomial likelihood.

The production total correction is disabled. Destination attractiveness is a
fixed input, not an estimated effect. Waiting-time, spatial, temporal, and
residual-demand effects are absent. The specification contains explicit scope
slots for extensions, but the minimal specification keeps them disabled rather
than silently ignoring them. An optional time-regime production deviation is
documented below; it is a separate, explicitly declared specification and is
not enabled by the minimal parent.

`GravityParameterLayout` maps a three-element unconstrained optimizer vector to
strictly positive physical parameters using stable softplus transformations.
Its names and slices are explicit and deterministic. Optimizer bounds are not
required to enforce behavioral signs.

## Demand model

For feasible canonical cell `(o,d,t)`, demand is

```text
q_odt = Q_ot * p(d | o,t)

p(d | o,t) ∝ A_dt * exp(
    -beta_time * journey_time_odt / journey_time_scale
    -beta_transfer * transfers_odt
)
```

The JAX implementation uses stable masked segmented maxima and sums over the
origin-time group indices. Infeasible cells receive an exact numeric zero,
feasible probabilities sum to one, and demand sums to the externally supplied
`Q_ot`. Extremely unattractive feasible alternatives remain finite. A pure
NumPy implementation exists only as a testing reference.

The principal Phase-1 API is:

```python
from public_transportation.inference.gravity import (
    GravityFeatures,
    GravityModelSpecification,
    GravityParameterLayout,
    generate_gravity_demand,
)

specification = GravityModelSpecification()
layout = GravityParameterLayout(specification)
result = generate_gravity_demand(
    raw_parameters,
    features=features,
    parameter_layout=layout,
)
```

`result.demand` is in canonical free-cell order. `result.probabilities`,
masked utilities, and origin-time sums support diagnostics. Both float32 and
float64 inputs are supported; JAX must have x64 enabled when float64 execution
is requested.

## Interpretation

A detailed OD table generated by a low-parameter gravity model is not equivalent
to independently estimating every OD cell. Its detail is induced by the gravity
structure, structural feasibility, production totals, and fixed attractiveness
offsets. Consequently, cell-level detail must not be interpreted as the same
number of independently identified demand parameters.

## Phase 2: measurement objective and gradients

`GravityObjectiveProblem` combines the Phase-1 demand inputs with an existing
`FixedRoutingMeasurementOperator`. Dense and JAX BCOO representations use the
same contract, including its validated compact-layout fingerprint and fixed
positive-demand measurement offset. The gravity layer neither reconstructs
routing nor creates selected OD blocks. Prediction is

```text
mu = rho * (operator.matrix @ gravity_demand + fixed_measurement_offset)
```

`rho` is fixed and defaults to one; Phase 2 never silently estimates it. A
strictly positive numerical floor protects likelihood evaluation at zero mean.
The objective supports Poisson and negative-binomial observations. The latter
reuses the shared mean/dispersion implementation with
`Var(Y) = mu + mu^2 / dispersion`. Objective evaluations separately expose the
data log likelihood, zero Phase-2 regularization contribution, total negative
log objective, demand, measurement mean, and calibration/excluded measurement
counts.

Two explicit gradient functions are available:

- `batched_forward` uses a small demand Jacobian and applies the routing matrix
  to all parameter directions together;
- `adjoint` applies one routing transpose product to the likelihood cotangent
  and propagates it through the gravity kernel with a JAX VJP.

Neither strategy constructs a dense measurement-by-parameter or routing
Jacobian. Phase-2 tests compare both strategies with direct JAX differentiation
and central finite differences. Automatic strategy selection belongs to the
estimator phase and is intentionally not implemented yet.

Phase 2 does not include optimization, checkpointing, zone hierarchies, model
relaxation, or automatic derivative selection. Those are separate reviewed
phases.

## Phase 3: minimal estimator

`estimate_gravity_model` fits the three-parameter minimal model with SciPy
L-BFGS-B by default, using an explicitly traced, lowered, compiled, and
synchronized JAX objective-and-gradient kernel. Smooth softplus
transformations enforce positive time, transfer, and negative-binomial
dispersion parameters; optimizer bounds are unnecessary. The optional
`optimizer = "biogeme_tr_bfgs"` selector uses Biogeme's trust-region BFGS
implementation against that same compiled `raw_parameters -> (objective,
gradient)` callback. It does not rebuild the objective, demand model, routing
operator, likelihood, or derivatives. Biogeme is imported lazily and remains
outside the default dependency set; selecting it without the optional
environment produces an actionable import error.

The optional path must be installed in a separate environment from the public
package's default dependencies. For a case study, pin the revised Biogeme
source (and `biogeme_optimization` separately when it is a separate
distribution) to immutable Git revisions in the case `pyproject.toml` and
`uv.lock`. Do not rely on a moving branch or on the older pilot versions. Verify
the resolved imports and pandas 3 before selecting the optional optimizer:

```bash
uv run --frozen python -c '
import pandas, biogeme
print("pandas:", pandas.__version__)
print("biogeme:", biogeme.__file__)
try:
    import biogeme_optimization
except ImportError:
    print("biogeme_optimization: not a separate installed distribution")
else:
    print("biogeme_optimization:", biogeme_optimization.__file__)
'
```

Do not add Biogeme to the public package's default environment or relax its
dependency bounds without repeating the compatibility and full-test checks.
The adapter implements Biogeme's `FunctionToMinimize` protocol and uses the
same relative-gradient acceptance criterion as the public estimator. It sets
Biogeme's relative-gradient epsilon and Dennis--Schnabel `typx`/`typf`
explicitly; callers must not rely on the optional wrapper's native convergence
flag alone.

`GravityEstimatorConfig` contains statistical stopping controls: maximum total
iterations, gradient tolerance, and objective tolerance. It also records the
Dennis--Schnabel scaled-gradient audit. For parameter vector $x$, gradient
$g$, typical parameter scales $\operatorname{typx}$, and typical objective scale
$\operatorname{typf}$, the reported infinity norm is

```text
max_i |g_i| max(|x_i|, typx_i) / max(|objective|, typf).
```

`gradient_tolerance`, `typical_objective_scale`, and the optional scalar or
per-parameter `typical_parameter_scales` are validated as finite positive
values. The single convergence and acceptance criterion is the relative
gradient inequality `scaled_gradient_inf_norm <= gradient_tolerance`. The
optimizer-native termination message remains diagnostic; it does not
introduce a second acceptance gate.
The library defaults (`typical_objective_scale = 1.0` and unit `typx`) are
generic fallbacks for library callers, not evidence that those scales are
appropriate for a case study. A case-study driver should require explicit
case-owned values. `typf` is a lower bound in the denominator: when the
objective is approximately $1.8\times10^6$, `typf = 1.0` is inactive. Measure
the objective at the initial raw parameter vector and choose a fixed,
documented value, preferably
`max(abs(initial_objective), objective_floor)`; never derive it from the final
objective. Choose `typx` from natural parameter units or prior scales and
provide exactly one value per raw parameter unless one scalar genuinely applies
to all parameters.
`GravityEstimatorConfig.optimizer` accepts only `scipy` and `biogeme_tr_bfgs`;
omission selects SciPy and preserves the historical L-BFGS-B behavior. The
optimizer choice is execution metadata, not part of the scientific model
fingerprint. `GravityExecutionPolicy` separately controls the derivative strategy, bounded
automatic-strategy threshold, wall-time allowance, progress interval,
checkpoint path, and optional persistent JAX compilation-cache directory. The
saved model specification is therefore independent of laptop or cluster
resource choices.

Explicit `batched_forward` and `adjoint` policies compile only the requested
kernel. `auto` benchmarks one first and one warm execution of both candidates.
For parameter counts no larger than the configured forward limit it selects the
lowest measured warm time; above that limit it restricts selection to adjoint.
The result records the request, selection reason, and separate tracing,
lowering, compilation, first-execution, warm-execution, and lowered-module-size
diagnostics for every benchmarked candidate. The persistent-cache configuration
is also recorded. Automatic preflight is deliberately bounded to these two
fixed candidates and two executions apiece.

Checkpoints are atomic, flushed JSON records written at the initial state and
after every completed SciPy L-BFGS-B iteration. They contain the raw
parameters, total completed iterations, accumulated elapsed time, schema, the
selected optimizer, and a model fingerprint
covering gravity inputs, specification, parameter layout, compact OD layout,
routing/mapping provenance, observations, calibration mask, likelihood, and
fixed numerical semantics. Resume rejects any mismatch. Invocation limits may
change without changing model identity. SciPy's internal L-BFGS history is not
portable across processes, so resume restarts the quasi-Newton history from the
saved iterate; converged predictions are deterministic within the documented
floating-point tolerance rather than bitwise identical. A checkpoint tagged for
one optimizer cannot be resumed by the other; comparison runs must use distinct
checkpoint and result paths.

The monotonic wall-time check occurs before optimization and after completed
iterations. An initial checkpoint therefore remains valid even if compilation
uses the entire allowance, while every completed iteration is resumable. An
external scheduler limit is still required to interrupt an active JAX
compilation or objective execution.

The immutable result reports raw and physical parameters, free-cell demand,
active compact demand including frozen-positive cells, the complete full-layout
OD vector with structural zeros, measurement predictions, objective and
likelihood, gradient, iterations, runtime, strategy diagnostics, checkpoint,
and model fingerprint. It additionally records raw and scaled gradient norms,
all scaling settings, objective and gradient dtypes, objective spacing, the
reduction between the final accepted iterates, and whether the requested
objective tolerance is below the available floating-point resolution. It also
records the initial objective, the resolved scale vector, and provenance/rule
metadata describing how `typf` and `typx` were selected. The same convergence
diagnostics are included in the result-aware
`convergence_diagnostics` section of `build_gravity_run_manifest`.

Both production algorithms use the same convergence certificate: the
Dennis--Schnabel scaled-gradient infinity norm must be no larger than
`gradient_tolerance`. A solver's native convergence flag, raw gradient norm, or
relative-objective message alone is not the acceptance criterion. Results and
manifests include `optimizer`, `optimizer_message`,
`optimizer_iterations`, `optimizer_evaluations`, and `optimizer_options`
alongside the common scaled-gradient and precision diagnostics.

### Exact model specification and provenance fingerprints

Every fitted result, adequacy/validation result, detailed report, and viewer
bundle carries the exact `GravityModelSpecification` used for the run. The
canonical representation is produced by `GravityModelSpecification.to_dict()`
and its immutable identity by `GravityModelSpecification.fingerprint`; the
package does not reconstruct a specification from a human-readable model name.
The fit, validation, report, and bundle must all carry the same
`specification_fingerprint` and serialized `model_specification`. Missing,
non-canonical, or conflicting specifications are rejected before a derived
artifact is written.

Three related fingerprints have distinct meanings:

```text
specification_fingerprint
    Identity of the model structure and declared parameters.

model_fingerprint
    Identity of the complete fitted scientific problem, including the model,
    features, observations, OD layout, measurement operator, and auxiliary data.

report_fingerprint
    Identity of the generated report contents.
```

The broader `model_fingerprint` remains unchanged when the exact specification
fields are added. Execution choices such as optimizer, worker count, and
checkpoint location are not silently folded into the scientific specification.
When inspecting a persisted result or manifest, the authoritative values can
be checked directly:

```python
specification = problem.parameter_layout.specification

print(specification.to_dict())
print(specification.fingerprint)
print(result.specification_fingerprint)
print(result.model_fingerprint)
```

This check is useful before generating reports or exporting a bundle: the
specification fingerprint should agree across every stage, while the model
fingerprint continues to certify the complete data/operator identity.

For a controlled pilot, `compare_gravity_optimizers` runs SciPy and Biogeme
independently from the same initial raw vector, callback, parameter names,
unconstrained bounds, tolerances, and maximum iteration count. It never
warm-starts one method from the other. Supply separate checkpoint/result paths
and compare final objective, parameter distance, raw/scaled gradient norms,
termination reasons, and dtypes. Report elapsed optimizer time separately
from any common model construction or JAX compilation time; a comparison that
includes compilation should say so. If a common operator activation is timed,
pass it once as `operator_activation_seconds`; both summaries then report it
separately and include it equally in `elapsed_total_seconds`. The optional Biogeme pilot reports the same
scaled-gradient and precision diagnostics and does not alter the default
production path.

When reviewing a result written by an older version that used a second
post-fit threshold, `reclassify_gravity_result` can migrate it without
rerunning the operator or optimizer. Supply the case's single
`gradient_tolerance` plus matching model and operator fingerprints. It
recomputes the relative-gradient diagnostic from the stored final
parameters/objective/gradient and changes only status and acceptance metadata.
Supplying an output path writes a new record and refuses to overwrite an
existing result; the original result remains unchanged. It must not be used
after any scientific input, operator, data, or scaling change.

Phase 3 does not implement spatial effects, model adequacy reporting, holdout
validation, or relaxation recommendations.

## Phase 4: full-data model adequacy

`validate_full_data_gravity_adequacy` evaluates a completed model against every
measurement used for calibration. It rejects calibration masks containing an
excluded row and validates the complete fitted-model fingerprint before
recomputing demand and measurement predictions. This is a model-adequacy or
model-invalidation analysis, not independent predictive validation.

The immutable report contains observed and modeled totals, negative-binomial
and reference Poisson deviances, MAE, RMSE, inverse-NB-variance weighted RMSE,
raw residuals, standardized NB residuals, configurable exceedance counts and
fractions, and observed-versus-modeled quantiles. Its fingerprint covers the
fitted model, predictions, metadata, dispersion, and diagnostic policy.

Applications may provide generic measurement-aligned labels through
`GravityValidationMetadata`. Available groupings are measurement type, line,
direction, stop, time period, origin zone, and destination zone. Each group
reports counts, observed and modeled totals, mean residual, MAE, RMSE, weighted
RMSE, mean standardized residual, and maximum absolute standardized residual.
No geographic data or application-specific labels are embedded in the library.

Ordered vehicle-journey labels enable adjacent standardized-residual
correlations for journeys with enough nonconstant observations. Configurable,
cautious findings distinguish grouped gravity-demand patterns, possible routing
or timing discrepancies, isolated suspect observations, and systematic
overdispersion. These flags do not establish causality and never alter the
model. Holdout validation and relaxation recommendations remain future phases.

## Portable persisted-data reporting

Detailed gravity reports can be generated on a separate reporting machine from
persisted fit and validation data. The portable entry point
`write_persisted_gravity_detailed_report` accepts a serialized fit manifest, a
serialized validation manifest, the observed measurement vector, an
`ODParameterLayout`, and optional `GravityValidationMetadata`. It restores the
typed gravity result and calls the same report writer used by the ordinary
in-memory workflow, so the CSV, JSON, and Markdown outputs retain the same
meaning.

For example, after loading the JSON manifests and persisted arrays on the
reporting machine:

```python
from public_transportation.inference.gravity import (
    write_persisted_gravity_detailed_report,
)

write_persisted_gravity_detailed_report(
    fit_manifest=fit_manifest,
    validation_manifest=validation_manifest,
    observations=observations,
    od_layout=od_layout,
    metadata=metadata,
    likelihood="poisson",
    output_directory=report_directory,
)
```

This path is deliberately independent of a case context. It does not load or
activate routing, read a temporal cache, rebuild an assignment operator, rerun
an objective, or invoke an optimizer. Reporting can therefore be performed
where the original model data or execution environment is unavailable. The fit
and validation manifests must both be completed and must contain matching
predictions and all canonical provenance fields: artifact, assignment,
binding, canonical-index, compact-layout, gravity-feature, mapping,
OD-layout, and package-revision fingerprints. The supplied layout fingerprint
must agree with the persisted value. A mismatch, missing field, non-completed
manifest, vector-shape error, or prediction disagreement is a hard failure and
is checked before the output directory is created.

The portable report's `report.json` records the canonical provenance values
exactly and labels its origin as `persisted_fit_validation`. This makes it
possible to audit which fit, validation predictions, OD layout, and package
revision produced the report without rerunning scientific computation. The
layout and validation metadata contracts provide deterministic `to_dict` and
`from_dict` methods; restored arrays are immutable, while layout fingerprints
and vector dimensions are verified. Persisted reports are diagnostic artifacts only:
they do not alter model identity, checkpoints, routing artifacts, or optimizer
behavior.

## Phase 5: atomic centered relaxations

### Time-regime production deviations

The package also supports a low-dimensional correction to the externally
prepared origin--time production totals.  A case supplies a mapping
`time_period_index` from detailed departure bins to `R` case-defined regimes
and declares a `GravityDeviationSpecification` attached to the global
production component:

```python
production = GravityComponentSpecification(
    name="production",
    scope=GravityEffectScope.GLOBAL,
    parameterization=GravityParameterization.LOG_MULTIPLIER,
    source="origin_time_totals",
    deviation=GravityDeviationSpecification(
        scope=GravityEffectScope.TIME_PERIOD,
        grouping="time_period_index",
        group_count=number_of_regimes,
        constraint=GravityConstraint.SUM_ZERO,
        regularization=GravityRegularization(
            GravityRegularizationType.RIDGE,
            ridge_strength,
        ),
    ),
)
```

The production multiplier is

```text
log multiplier(o, t) = alpha + delta[regime(t)]
```

with `sum(delta) = 0`.  The layout therefore stores `R - 1` deviation
coordinates in addition to the unpenalized global scale, using stable names
`production_scale` and `production_time_deviation[...]`.  The ridge penalty
applies to the deviation block only and is expressed in native objective
units.  Zero deviations reproduce the minimal model exactly; nonzero values
change origin--time totals, not the destination softmax within each group.

The adapter must validate that regime labels are contiguous and constant
within each origin--time group, and must record the mapping and ridge strength
in the model specification and fit manifest.  The package does not infer
regimes or attach semantic names to them.  Changing this gravity specification
requires a new gravity fit checkpoint and model fingerprint, but does not
invalidate routing or fixed measurement-operator artifacts.

Three deliberately small child models can now be added one at a time to the
minimal parent. Destination-zone deviations add to log attractiveness; broad
time-period deviations multiply the positive journey-time coefficient; and
origin-zone deviations multiply externally prepared production totals. A plain
period intercept is not used because it would cancel inside every origin-time
softmax.

Applications supply the compact zone and period indices. Active mappings must
use every integer from zero through `K - 1`; origin-zone and time-period labels
must be constant within each origin-time group. The library does not infer or
download zones. Each block uses `K - 1` optimizer coordinates and reconstructs
the final effect as the negative sum, enforcing an exact zero-centered vector.
The child therefore adds exactly `K - 1` parameters without an additive
invariance.

`add_gravity_relaxation` returns both the child specification and a user-facing
description of its parameter and execution cost. `warm_start_gravity_parameters`
copies all parent coordinates and initializes new deviations to exact zero, so
the child initially reproduces the parent. A configurable ridge penalty is
applied to the complete centered effect vector and is exactly zero at that warm
start. `remove_gravity_relaxation` restores the atomic parent contract.

The execution impact is linear indexing over cells or origin-time groups; no
dense OD-time tensor or routing Jacobian is introduced. Relaxed objectives use
the same dense/BCOO measurement operators, adjoint or batched-forward gradients,
checkpoint fingerprinting, and estimator as the minimal model.

## Phase 6: advisory relaxation diagnostics

`recommend_gravity_relaxations` constructs a catalog without fitting or applying
any child model. At the fitted parent, every applicable Phase-5 child is initialized
with the exact zero-deviation warm start. If `g_c` and `H_c` are the objective
gradient and regularized Hessian restricted to its new coordinates, the reported
local gain is

```text
gain_c = 0.5 * g_c.T @ stabilized_inverse(H_c) @ g_c
```

The Hessian already includes the configured ridge penalty. Non-positive or nearly
singular eigenvalues are floored only for the diagnostic approximation and produce
an explicit weak-identification warning. The report also warns about poor
observation support, ill-conditioned curvature, and a parent iterate whose gradient
is too large for an optimum-based score interpretation.

Each immutable catalog entry reports applicability, discrepancy addressed, exact
added parameter count when known, score statistic, gain, observation and grouping
support, approximate calibration-deviance improvement, execution cost, identification
warnings, recommendation strength, and a plain-language explanation. Grouped
destination-zone, time-period, and origin-zone standardized-residual patterns from
the Phase-4 adequacy report supplement the local scores.

Journey-time and transfer-sensitivity correlations are also reported as diagnostic
catalog entries. They may motivate designing a future centered impedance or
transfer-penalty child, but those entries are explicitly unavailable and have no
invented parameter count or gain. Coherent residuals along vehicle journeys warn
about possible routing or timing error; isolated extremes warn about possible data
problems. Recommendations are always advisory: the function returns no modified
specification, parameters, or fitted child.

## Phase 7: immutable model lineage

`GravityModelNode` freezes every completed model as a reproducible lineage node.
Its deterministic identifier covers the parent and applied relaxation, complete
specification and parameter layout, raw and physical estimates, feature and compact
layout fingerprints, assignment/graph/measurement-mapping provenance, optimizer
state, assigned-count adequacy diagnostics, runtime and stored-memory diagnostics,
checkpoint path, and explicit calibration and validation measurement identities.
The complete immutable estimation result is retained as well. Root nodes have no
parent or relaxation; every child must remove its declared atomic relaxation and
recover its recorded parent specification exactly.

`GravityModelLineage` is append-only: adding a child returns a new lineage while
retaining all parent nodes and results. Parents must precede children and identifiers
must be unique. `gravity_measurement_identity` fingerprints an explicit unique
measurement-index vector and semantic label, preventing two different calibration
or validation sets from being compared silently.

The interactive progression is exposed as inspectable stages:

1. load a completed parent node;
2. list currently applicable atomic relaxations;
3. obtain the advisory Phase-6 recommendation catalog;
4. pass exactly one explicit `GravityEffectScope` selection;
5. construct a child layout and verify its zero-deviation warm start reproduces
   the parent measurements within tolerance;
6. estimate and validate the child with the existing checkpointed estimator;
7. compare objectives, likelihoods, deviances, RMSE, parameter count, and runtime;
8. return a new lineage preserving both completed results.

`progress_gravity_model_lineage` orchestrates those same public stages but does
not choose a recommendation. It rejects changed measurement identities before
performing child diagnostics or optimization. Phase 7 remains a full-calibration
development workflow; grouped holdout validation is a separate subsequent phase.

## Phase 8: grouped holdout re-estimation

Predictive validation begins only after selecting a model structure through the
full-data adequacy and lineage workflow. `build_gravity_holdout_split` partitions
whole groups rather than independent measurement rows. Supported units are complete
vehicle journeys, complete stop-time series, lines, directions, broad time blocks,
and arbitrary application-provided group labels. For `stop_time_series`, the supplied
`metadata.stop` label must identify the complete series that must remain together;
composite application groupings can always use `explicit_group`.

Splits are deterministic from a non-negative seed and can be stratified by any
combination of measurement type, line, direction, stop, time period, origin zone,
and destination zone. A group crossing several stratum labels remains indivisible;
its complete label signature defines its stratum. Singleton strata remain in
calibration unless a pooled singleton must be selected to obtain a nonempty holdout.
Both masks are immutable, complementary, and nonempty.

`GravityHoldoutSplit` stores the measurement identity, metadata fingerprint,
configuration, selected group labels, masks, and a content fingerprint. Its
`to_dict`/`from_dict` representation validates that fingerprint on load. The split
therefore remains reusable across competing model structures while preserving the
same measurement rows and grouping decision.

`estimate_and_validate_gravity_holdout` requires an initially full-data problem,
replaces its calibration mask with the split calibration groups, and re-estimates
all parameters of the selected model structure. It then assigns the resulting
complete OD vector to the complete measurement operator and scores the untouched
holdout groups. Calibration and holdout reports separately expose observation and
prediction totals, negative-binomial and reference Poisson deviance, likelihood,
MAE, RMSE, and inverse-NB-variance weighted RMSE. The immutable report also retains
the full predictions, free and complete OD demand, fitted result, split fingerprint,
and selected specification fingerprint.

The intended model-selection workflow is:

1. diagnose and develop structure with full-data adequacy;
2. select a satisfactory structure explicitly;
3. create one reusable grouped split and re-estimate every compared structure;
4. compare honest holdout predictive metrics separately from calibration metrics;
5. revise structure only when holdout evidence warrants it;
6. optionally refit the accepted structure on all observations;
7. retain the grouped holdout report as the predictive assessment.

Changing only excluded observations cannot affect the calibration estimate. Tests
verify this no-leakage property directly. Random row holdout is intentionally absent.

## Performance and execution evidence

### Explicit and matrix-free measurement operators

Gravity objectives depend only on three device-native products exposed by the
public `GravityMeasurementOperator` protocol: `jax_matvec`, `jax_rmatvec`, and
`jax_matmat`. Explicit dense and BCOO operators implement those products while
retaining their existing `.matrix` compatibility API. The matrix-free
fixed-routing operator implements the same contract by loading demand through
the prepared routing and aggregating measurements directly; it never constructs
a complete measurement-by-OD matrix and never transfers intermediate link flow
to the host.

Use an explicit operator when its stored representation is comfortably bounded
and repeated linear-algebra access is useful. Use
`MatrixFreeFixedRoutingMeasurementOperator` when the logical matrix is too large
to construct. The recommended large-network configuration is matrix-free routing
with the adjoint gradient:

```python
operator = MatrixFreeFixedRoutingMeasurementOperator(
    inputs=assignment_inputs,
    routing=fixed_routing,
    spec=measurement_mapping,
    compact_layout=compact_layout,
)
problem = GravityObjectiveProblem(
    features=features,
    parameter_layout=parameter_layout,
    operator=operator,
    observations=observations,
)
execution = GravityExecutionPolicy(
    gradient_strategy="adjoint",
    checkpoint_path=checkpoint_path,
    jax_compilation_cache_directory=compilation_cache,
)
```

The matrix-free forward, transpose, and bounded multiple-right-hand-side
products accept and return JAX arrays without NumPy conversion or internal
synchronization. Host-oriented `matvec` and `rmatvec` remain available for
diagnostics. One adjoint value-and-gradient evaluation performs exactly one
forward routing product and one transpose product; the evaluation record is
assembled from that routed mean rather than routing demand a second time.

For an explicit operator, `auto` may benchmark both derivative strategies on a
small parameter problem. For a matrix-free operator, `auto` selects adjoint
directly and records that it avoided benchmarking two expensive cold network
kernels. An explicit `batched_forward` request remains available for a small
number of parameter directions and uses `jax_matmat` rather than constructing a
measurement-by-OD-by-parameter tensor. Compilation remains an indivisible JAX
operation; checkpoints contain only stable model and operator provenance, not
the runtime operator object.

`prepare_device_products(products="forward")` or
`prepare_device_products(products="forward_and_transpose")` makes cold work
explicit. Its diagnostics separate tracing, lowering, compilation, first and
warm execution, product counts, deadline phase and overshoot, shapes, dtype,
backend, and devices. Captured-constant bytes are reported when authoritative;
otherwise the field is `None` rather than an inferred value.

The bounded `benchmark_matrix_free_gravity.py` uses packaged Simple Example 02,
constructs no global operator, prepares both products, times a complete adjoint
value-and-gradient kernel, records RSS, and writes a one-iteration checkpoint.
Its timings are descriptive; tests assert structural properties rather than
machine-dependent speed thresholds.

## Bounded sharded fixed routing

Complete fixed routing stores two destination-group-by-link arrays. This is
convenient for small networks but can dominate laptop memory before a gravity
operator is constructed. `prepare_fixed_routing_sharded` instead partitions
destination groups canonically, pads every shard to one configured shape, and
persists each completed routing payload atomically. The final partial shard
therefore reuses the same compiled kernel.

`FixedRoutingPreparationConfig` bounds groups and retained bytes per shard,
estimated temporary bytes, process RSS, total cache bytes, elapsed time, and
resident shards. A laptop should begin with one worker and eight groups per
shard. Larger worker counts are admitted only when measured parent RSS plus
active-shard estimates and a safety margin fit the ceiling. Explicit
configuration remains authoritative.

Parallel preparation uses a bounded thread pool so workers share the large
assignment graph and one compiled executable; spawned JAX processes would
duplicate both. XLA execution releases the Python GIL. Progress and manifests
are published in canonical shard order even when execution finishes out of
order, and worker failures leave already published artifacts reusable.

With `detailed_profiling=True`, synchronized diagnostics separate destination
and mask preparation, argument transfer, kernel dispatch, device
synchronization, host transfer, slicing, validation, shard and manifest
persistence, and cleanup. They also report CPU utilization, shapes, dtypes,
memory information, full-graph work, enabled-link density, and effective
probability density. Thread workers return immutable, worker-local diagnostics;
the coordinator attaches manifest-persistence time and orders the records by
shard index. Cache hits do not receive execution diagnostics. Profiling remains
opt-in.

The cache identity includes assignment, graph, costs, source masks,
destinations, theta, dtype, numerical implementation and package/schema
versions, and the canonical partition. Shards are written through a flushed
temporary file followed by atomic replacement. A separate atomic manifest
records completed shards and the next canonical position. A valid restart loads
completed shards without recomputation; corrupt or mismatching artifacts raise
`FixedRoutingShardCacheError` unless the caller explicitly requests refresh.

Structured `FixedRoutingShardProgress` events report groups, shards, cache
hits/misses, elapsed and recent time, estimated remaining time, RSS, cache bytes,
and deadline allowance. Parallel runs emit one planning/cache-scan event,
bounded dispatch events when worker batches start, buffered `shard_completed`
events, canonical `shard_persisted` events only after manifest commit,
predictive-guard events, and one terminal event; they do not poll workers.
Every snapshot uses the coordinator's complete state. `active_workers` and
`current_shard_indices` describe all executing futures, `buffered_shards`
counts finished results awaiting canonical manifest commit, and `queued_shards`
counts only missing shards never submitted. Consequently
`completed_shards + active_workers + buffered_shards + queued_shards +
failed_shards == total_shards` at every event, while `remaining_shards` is
`total_shards - completed_shards`.
Applications may render those events with `tqdm` on stderr while reserving
stdout for JSON. `cache_misses` means every planned shard absent at the initial
scan, including work left unattempted by a bounded invocation;
`newly_constructed_shards` counts the misses actually committed in that
invocation. Deadline and memory stops occur at shard boundaries and leave
completed artifacts reusable.
After warm timings exist, predictive dispatch declines new work when the
remaining allowance is shorter than the rolling shard estimate plus a safety
margin. Active indivisible work may still overshoot and is reported explicitly.

Worker admission distinguishes memory and CPU-count ceilings from demonstrated
throughput. The portable default remains one worker, and explicit concurrency
is admitted within those resource ceilings. Because throughput depends on the
graph and CPU memory hierarchy, `throughput_effective_worker_count` remains
unknown unless a representative target benchmark establishes it. In public
warm-executable tests, four callers improved aggregate throughput despite
higher individual latency; the private large-network evidence did not. The
configuration's `threads_per_worker` value is scheduling metadata and does not
control XLA's internal CPU pool.

An opt-in `execution_strategy="batched"` preparation path evaluates
`shards_per_execution_batch` bounded shards in one fixed-shape compiled call.
The final batch retains the same padded shape, results are sliced and committed
as canonical individual shard artifacts, and a manifest update follows every
individual commit. Thus either strategy can reuse the other strategy's routing
cache, and a restart after partial batch persistence resumes at the first
missing shard. The batch's temporary-storage estimate must pass admission
before dispatch, and the predictive deadline guard treats the complete batch
as indivisible. Batch width affects compiled-kernel identity and diagnostics,
but not the numerical preparation fingerprint. Public CPU measurements show no
throughput benefit from batching, so `thread_pool` remains the default.

Worker recommendations report separate memory-admissible,
CPU-count-admissible, and empirically throughput-effective concurrency. Without
representative measured throughput, the recommendation is deliberately one
worker. Optional calibration maps may select a worker count and batch size only
within the resource ceilings.

The present dynamic program visits every graph node and all links for every
destination group. Controlled experiments from 10% through 100% enabled density
have nearly unchanged runtime, establishing full-graph rather than
enabled-support complexity. Compact support remains conditional: the packaged
example is approximately 91% dense, so its indexing overhead would outweigh the
reduction. Measure a target network with detailed diagnostics before selecting
a sparse-support cache schema.

#### Persistent temporal-operator cache loading

Direct-scheduled temporal artifacts have two distinct persistence layers. The
source artifact contains one validated ``block-*.npz`` file per temporal block;
the identity-bound operator cache contains packed ``.npy`` arrays, offsets,
keys, and the fixed measurement offset. Source-block validation is retained as
the fail-closed fallback and still checks every source block when no validated
packed cache is available. Once the packed cache has been validated, normal
``fit``, ``preflight``, and optimizer-comparison activation memory-map the
packed arrays and do not reopen the individual source block files or recreate
one Python object per block. CSR and CSC execution structures are built
directly from those packed arrays; duplicate summation, zero elimination,
sorting, dtypes, dimensions, and products retain the source-operator contract.

The first use of an existing artifact may therefore still spend time on the
one-time packed-cache consolidation (or on CSR/CSC construction). That work is
reported separately from source-block validation. Subsequent processes reuse
the packed cache and, when present, the identity-bound
``positive_boarding_support_cache`` without scanning all temporal blocks.
Progress events distinguish ``packed_array_validation``,
``packed_cache_loading``, ``csr_csc_construction``,
``realized_operator_support``, and ``positive_boarding_support_cache``. Finite
loops report completed/total units and a bounded ETA; a CSR/CSC build is an
indivisible operation and emits regular heartbeats while its ETA is otherwise
unavailable. A packed-cache hit is represented as one logical completed unit,
not as the number of source blocks that were not opened.

Packed-cache acceptance remains strict: completion, schema and validator
versions, artifact and canonical-index identities, binding and manifest
fingerprints, packed-array content hashes, shapes, offsets, dtypes, and the
fixed-offset hash must all match. A modified or incomplete packed cache is
rejected and the existing source-artifact validation path is used. Progress
settings and cache representation are execution metadata; they are excluded
from scientific identities and do not change objective or gradient kernels.

`ShardedMatrixFreeFixedRoutingMeasurementOperator` loads at most the configured
number of routing shards. Production products no longer traverse graph nodes in
Python. Fixed-shape compiled JAX forward and reverse kernels process every
padded group in a shard using the persisted effective masks and probabilities;
they do not recompute paths or routing probabilities. Measurement aggregation
is fused into the forward kernel, and the reverse is its VJP. Set
`operator_shards_per_batch` to process a bounded number of shards per compiled
execution; the final partial batch is padded. The operator constructs neither
complete routing arrays, a link-flow-by-shard stack, nor a measurement-by-OD
matrix.

Aggregate batching still evaluates destination groups through the assignment
kernel's sequential `lax.scan`; private full-network measurements and public
synthetic benchmarks show that enlarging this scan does not materially improve
CPU utilization. For multicore execution, use
`shard_execution_strategy="concurrent"` with
`group_execution_strategy="scan"`. This dispatches already-compiled
single-shard kernels through a bounded worker pool and accumulates completed
results in canonical shard order. `operator_concurrency` is limited by the
available CPU count, shard count, and `maximum_concurrent_routing_bytes`; when
unspecified, its conservative requested ceiling is four. LRU residency and
live in-flight routing bytes are diagnosed separately.

A vectorized group strategy is available for controlled experiments. Although
it exposed more CPU parallelism, it was substantially slower for forward
loading on the larger public benchmark because it materializes a group-by-link
intermediate. It is therefore not the recommended full-network strategy.

The common `GravityMeasurementOperator` protocol documents numerical products,
fingerprints, fixed offsets and operational capabilities. The sharded operator
supports structured product progress, absolute deadlines, cancellation and LRU
residency diagnostics. A deadline is checked before each indivisible shard
batch using its recent predicted duration plus the configured safety margin.
Interruption raises `ShardedOperatorProductInterrupted`; no partial product is
returned as complete. Gravity estimation propagates its wall-time deadline into
the operator and checkpoints only completed optimizer iterates.

After one representative batch or concurrent wave, the deadline guard projects
the time for every remaining wave, not merely the next dispatch. If the complete
product cannot finish with its safety margin, it emits
`product_deadline_infeasible` and discards the partial sum before launching more
work. The diagnostic includes completed/total shards, measured wave duration,
predicted remaining time, deadline allowance, discarded work, batch size, and
concurrency.

For the first product of a new process, set
`initial_predicted_batch_seconds` from the corresponding preflight measurement.
Until a batch completes, this conservative value participates in the same
deadline test as the rolling empirical prediction. The measured duration then
replaces it. This operational value does not change routing or model
fingerprints.

Use `run_gravity_preflight` before a large estimation. It validates all routing
shards and fingerprints, then can stop after validation, forward, reverse,
objective-gradient, projection, or recommendation. A complete preflight times
warm adjoint and batched-forward gradients and recommends a strategy, current
operator batch and residency settings, expected evaluation time and memory,
wall-time budget, checkpoint interval, and laptop versus server execution.

Run the public structural benchmark with:

```bash
uv run python benchmarks/benchmark_sharded_fixed_routing.py \
  --output benchmarks/sharded_fixed_routing.json
```

To demonstrate interruption and resume, give the first call a short absolute
deadline, retain its cache and checkpoint directories, then call it again with
the identical inputs and configuration. The second call reports cache hits for
every valid completed shard. Delete both directories, or use explicit refresh,
when intentionally changing routing semantics.

The public `benchmark_gravity_performance.py` benchmark covers several parameter
counts on dense and BCOO measurement operators. It reports JAX tracing, lowering,
compilation, first execution, warm execution, changed-parameter reuse, forward and
transpose routing, both derivative strategies, objective-and-gradient execution,
operator storage, process peak RSS where measurable, dimensions, and explicit
in-process compiled-handle hits and misses. Batched-forward and adjoint gradients
must agree before timings are accepted.

On the documented 128-cell CPU case, the fastest strategy varied by representation
and parameter count, with differences often near timer noise. The automatic
forward limit therefore remains eight as a conservative large-network memory guard;
the small benchmark does not justify materializing wider demand Jacobians on large
OD systems. JAX exposes no authoritative public in-process persistent-cache hit
counter, so persistent reuse is reported only through an explicit fresh-process
populate/reuse protocol and is never inferred from elapsed time.

For persisted routing shards, run:

```bash
uv run python benchmarks/benchmark_sharded_gravity_operator.py \
  --operator-batch-sizes 1 2 4 8 --output sharded-gravity.json
```

This scalable synthetic benchmark reports synchronized matvec, rmatvec and
matmat timings; shard loading, dispatch, compiled execution, synchronization,
transfers, aggregation and eviction counters; CPU and RSS diagnostics; cache
hits and misses; both minimal three-parameter gradient strategies; and projected
optimizer times. Batch selection uses aggregate warm product time. The report
also records explicitly that neither a dense measurement-by-OD matrix nor a
complete routing array was materialized.

The representative 8,192-node public comparison is recorded in
`benchmarks/sharded_gravity_concurrency.md`. Four concurrent scan workers
improved warm forward time from 28.1 ms to 14.7 ms and eight workers to 13.9 ms,
while effective CPU use increased from approximately one to 2.28 and 2.74 cores
respectively. These measurements satisfy the public throughput acceptance
criterion but do not replace a short private-cache preflight.

Long runs should publish an atomic manifest and use one durable JSONL sink for
both operator and optimizer progress:

```python
sink = GravityJSONLProgressSink(run_directory / "progress.jsonl", durable=True)
operator.progress_callback = sink
manifest = build_gravity_run_manifest(
    problem=problem,
    compact_layout=compact_layout,
    estimator_config=estimator_config,
    execution=execution,
    repository_revision=repository_revision,
    preflight=preflight,
)
write_gravity_run_manifest(run_directory / "manifest.json", manifest)
result = estimate_gravity_model(
    problem=problem,
    compact_layout=compact_layout,
    initial_raw_parameters=initial_raw,
    config=estimator_config,
    execution=execution,
    progress=sink,
)
```

The manifest records revisions, numerical fingerprints, dimensions, dtype,
theta, batching, residency, estimator policy, JAX devices and relevant thread
environment. Progress records are append-only, timestamped, flushed and
optionally `fsync`-committed. Resume with the same manifest and configuration.

Estimator progress retains the flat iteration/objective fields and adds the
common hierarchical contract: `job_elapsed_seconds`, `phase_elapsed_seconds`,
`work_stack`, active/queued units and workers, weighted progress, ETA bounds and
confidence, plus checkpoint reuse and next-resumable-iteration metadata. ETA is
based only on completed optimizer iterations; it is intentionally unavailable
while an objective/gradient evaluation is opaque or while too few iterations
have completed. These reporting fields do not enter the model fingerprint.

The committed public Geneva integration exercises the complete direct-scheduled
gravity workflow on 96 free cells, 15,128 full-layout cells, and 8,967
boarding/alighting measurements. Scheduled path metrics and synthetic prior demand
prepare the gravity inputs. An exact scalar-column fixed-routing construction builds
BCOO indices and data directly, avoiding the general node-by-measurement
intermediate. The bounded run covers minimal estimation, adequacy, recommendations,
an exact child warm start, lineage, and grouped journey holdout. Its iteration-limited
metrics validate integration and are not converged scientific estimates.

## OD-cell information and identifiability diagnostics

After a fit, `compute_gravity_od_identifiability` provides an optional local
information diagnostic for every canonical full-OD cell. It is deliberately a
post-estimation operation: it evaluates derivatives at the supplied result and
does not refit the model, alter the objective, or change the routing operator.
The calculation is chunked over measurement and OD cells, so it does not
materialize a complete measurement-by-OD Jacobian.

For the calibration rows, the count contribution is the Gauss--Newton
curvature

```text
H_counts = J_mu^T W J_mu,
```

with `W = 1 / max(mu, mean_floor)` for Poisson observations and
`W = r / (max(mu, mean_floor) * (r + max(mu, mean_floor)))` for a
negative-binomial likelihood with dispersion `r`. The existing parameter
regularization is differentiated separately as `H_reg`. If optional observed
data channels (for example, a trip-attribute histogram) are enabled, their
curvature is retained as a third, separate contribution rather than being
folded into the count share.

The symmetric total Hessian is eigendecomposed. Eigenvalues below
`max(eigenvalue_absolute_tolerance,
eigenvalue_relative_tolerance * max(abs(eigenvalues)))` are discarded and the
remaining eigenvalues define a pseudoinverse. For a free OD cell `j`, let
`g_j` be the derivative of its demand with respect to the raw fitted
parameters. The local variance proxy is `g_j^T H^+ g_j`; each information
share is the corresponding component quadratic form divided by this proxy.
Shares are therefore local linearized diagnostics of curvature allocation, not
probabilities, posterior probabilities, or standard errors. All free OD cells
remain coupled through the shared parameters and the origin-time normalization.

The result classifies free cells as `count_dominated`, `assumption_dominated`,
`auxiliary_data_dominated`, `mixed_information`, or
`not_locally_identifiable`. Cells fixed by the input or structural-zero policy
have null shares and variance and are classified as `fixed_by_policy` with
reason `fixed_input_or_structural_zero_policy`. Small floating-point excursions
outside `[0, 1]` are clipped; larger violations are retained as a diagnostic
reason rather than hidden.

```python
from public_transportation.inference.gravity import (
    GravityIdentifiabilityConfig,
    compute_gravity_od_identifiability,
    write_gravity_od_identifiability,
)

diagnostic = compute_gravity_od_identifiability(
    problem=problem,
    result=fit_result,
    config=GravityIdentifiabilityConfig(
        measurement_chunk_size=4096,
        od_chunk_size=4096,
    ),
)
write_gravity_od_identifiability(diagnostic, results / "identifiability")
```

Persistence consists of `identifiability.json` metadata and a compressed
`identifiability.npz` numerical payload. The metadata records schema version,
artifact/model/layout provenance, effective Hessian rank and dimension,
discarded eigenvalues, classification counts, configuration, array names, and
array checksums. Readers verify dimensions, checksums, and any supplied
provenance before returning the immutable arrays.

The detailed report can receive the diagnostic through its optional
`identifiability=` argument. It adds the six diagnostic columns to
`full_od.csv` while preserving the existing `inference_score` and
`inference_basis` fields. `report.json` contains a compact identifiability
summary and classification counts. If no diagnostic is supplied, reports state
exactly:

```text
OD identifiability diagnostics were not available; inference_score remains only a structural fixed/free indicator.
```

Set `require_identifiability=True` when a report is not acceptable without the
diagnostic. A supplied diagnostic whose model, operator/layout provenance,
parameter dimension, array lengths, or fitted-vector fingerprints disagree
with the report inputs is rejected before the output directory is created.

## Viewer-ready gravity-result bundles

The public package can export a completed fit, validation report, OD table,
measurement diagnostics, network snapshot, and exact model specification as a
portable `gravity_viewer_bundle_v1`. The bundle is a data contract for an
independent graphical application; the public package does not install or
implement a GUI and bundle reading never activates routing, evaluates the
objective, or calls an optimizer.

Use the report and typed fit/validation results that have already been
produced:

```python
from public_transportation.inference.gravity import write_gravity_viewer_bundle

bundle = write_gravity_viewer_bundle(
    output_directory=results / "viewer_bundle",
    fit_result=fit_result,
    validation_result=validation_report.adequacy,
    report=detailed_report,
    identifiability=identifiability,       # optional
    network_files={                         # optional, explicit paths
        "stops.csv": network_dir / "stops.csv",
        "lines.csv": network_dir / "lines.csv",
        "trips.csv": network_dir / "trips.csv",
        "stop_times.csv": network_dir / "stop_times.csv",
    },
    metadata={
        "time_zone": "Europe/Zurich",
        "coordinate_system": "latitude_longitude",
        "x_column": "lon",
        "y_column": "lat",
    },
)
```

The fit result is the authoritative source for the exact model specification.
The writer validates it through `GravityModelSpecification`, writes it to
`model_specification.json`, and records its fingerprint in both that file and
`bundle_manifest.json`. If a fit result does not carry a specification, the
writer stops instead of creating a viewer bundle that cannot explain the model.
An explicit `model_specification=` argument may be supplied when restoring a
result from an older persistence format. It must have the same fingerprint as
the fitted result when that fingerprint is present.

The resulting directory contains:

```text
gravity_viewer_bundle_v1/
  bundle_manifest.json
  model_specification.json
  report.json
  executive_summary.md
  report.md                         # when present in the source report
  full_od.csv
  predicted_measurements.csv
  residuals.csv
  grouped_residuals.csv
  parameters.csv
  od_map.csv                         # optional display-oriented OD summary
  stop_summary.csv                   # optional display-oriented stop summary
  identifiability/                  # only when a diagnostic is supplied
    identifiability.json
    identifiability.npz
  network/                          # only when an explicit snapshot is supplied
    stops.csv
    lines.csv
    trips.csv
    stop_times.csv
```

When the source report contains sufficient OD and measurement information, the
writer also adds two display-oriented aggregations:

```text
  od_map.csv                         # one display row per OD/time pair
  stop_summary.csv                   # observed/modelled totals by stop
```

`od_map.csv` contains `origin_stop_id`, `destination_stop_id`,
`departure_time_bin`, `observed_demand`, `modeled_demand`, `residual`,
`fixed_free_status`, and `identifiability_class`. The core fit normally does
not observe demand at OD-cell resolution; in that situation
`observed_demand` and `residual` are left empty rather than being filled with
modeled or prior demand. A case-specific report may provide an aligned
observed-OD vector when one is scientifically justified. `stop_summary.csv`
contains observed and modeled stop totals, residuals, and observed incoming
(`alighting`) and outgoing (`boarding`) totals. These files are intended to
make map interaction fast; they are not independent diagnostics and never
change the fit, report, or scientific fingerprints.

Both files are optional. Older bundles without them remain valid and readable;
an independent viewer should retain the ordinary tables and network map while
indicating that interactive summaries are unavailable. When present, their
checksums, sizes, and row counts are included in `bundle_manifest.json` and
are checked during bundle validation.

### Exporting directly from persisted manifests

For a completed case whose fit and validation outputs are already persisted,
use `write_persisted_gravity_viewer_bundle`. It restores the typed fit result,
creates the detailed report through the persisted-data contract, and exports
the bundle in one call. It does not rebuild routing, load a case context,
evaluate the objective, or fit the model. The temporary report directory is
removed automatically unless `report_output_directory` is supplied.

The case must supply the observed measurement vector and the canonical
`ODParameterLayout` used by the fit; these are case-owned data contracts and
cannot be inferred safely from a generic manifest. Network files and viewer
metadata are explicit inputs.

```python
from public_transportation.inference.gravity import (
    write_persisted_gravity_viewer_bundle,
)

bundle = write_persisted_gravity_viewer_bundle(
    output_directory=results / "viewer_bundle",
    fit_manifest=fit_manifest,
    validation_manifest=validation_manifest,
    observations=observations,
    od_layout=od_layout,
    metadata=validation_metadata,
    likelihood="poisson",
    network_files={
        "stops.csv": network_dir / "stops.csv",
        "lines.csv": network_dir / "lines.csv",
        "trips.csv": network_dir / "trips.csv",
        "stop_times.csv": network_dir / "stop_times.csv",
    },
    bundle_metadata={
        "time_zone": "Europe/Zurich",
        "coordinate_system": "latitude_longitude",
        "x_column": "lon",
        "y_column": "lat",
    },
)
```

The helper validates the same provenance, prediction, model-specification,
report, network, and checksum contracts as the lower-level writer. A case
driver may expose this call as an `export-viewer` stage after checking the
existing manifests, without rerunning preparation or fit. It refuses to
overwrite an existing bundle directory.

`full_od.csv` retains the complete canonical OD schema, including structural
fixed/free status and the local identifiability columns. The latter are
linearized information diagnostics, not probabilities, standard errors, or
causal attribution. If no diagnostic is supplied, the bundle records
`identifiability.available = false` and preserves the message that
`inference_score` is only a structural fixed/free indicator. Pass
`require_identifiability=True` to reject such a bundle.

The manifest records SHA-256 checksums, file sizes, row counts, fit,
validation, report, and identifiability provenance, all canonical layout and
operator fingerprints, package revision, model-specification fingerprint,
time-zone convention, and coordinate convention. The default coordinate
convention is latitude/longitude with `lon` as x and `lat` as y. A projected
coordinate system must be supplied explicitly; the package never infers or
claims one. Network snapshots require `stops.csv` with
`stop_id,name,lat,lon`; report stop IDs and available line/trip references are
checked against that snapshot. When route geometries are unavailable, a GUI
may connect consecutive stops to form an approximate display.

Read and validate a bundle on another computer without the original case
directory:

```python
from public_transportation.inference.gravity import read_gravity_viewer_bundle

bundle = read_gravity_viewer_bundle("gravity_viewer_bundle_v1")
bundle.validate()  # safe to repeat before a viewer session
specification = bundle.model_specification
od_for_peak = bundle.query_od(
    departure_time_bin="am",
    minimum_flow=1.0,
    top_n=100,
)
for chunk in bundle.iter_table("full_od.csv", chunksize=100_000):
    consume(chunk)
```

The reader verifies every checksum, required file, row count, model and
report provenance, model-specification fingerprint, identifiability payload,
and network joins. Mismatches identify the field, conflicting values, and
affected file; the reader never repairs or overwrites data. `iter_table` and
`query_od` use bounded pandas chunks for large OD tables, and support filtering
by origin, destination, time bin, identifiability class, minimum flow, and
top-N demand. `aggregate_measurements` provides a chunked observed/modelled
summary by line, stop, or time-period columns. The optional
`od_measurement_links.csv` influence artifact is not generated by the core
writer; if a viewer needs it, it must be supplied and labelled as a local
influence diagnostic rather than a unique explanation of an OD cell.

An independent browser viewer can use `stop_summary.csv` for a **Stops** mode
and `od_map.csv` for an **OD pairs** mode. Stops mode displays bundled stop
markers, scales marker size by a selected observed/modelled/residual metric,
and opens incoming/outgoing OD details when a marker is selected. OD-pair mode
uses stop-name/ID selectors, a departure-bin and demand/residual filter, and a
bounded top-N list; residual direction and minimum magnitude can be selected,
and only the selected or filtered pairs are drawn over the network. Line width
represents demand and line colour represents signed residual (blue for
underprediction, red for overprediction, gray near zero). The viewer never
reruns routing, evaluates the objective, or modifies the bundle. OpenStreetMap
background tiles require internet access; the local bundle tables remain
usable without tiles.

### Local browser viewer

The repository includes a browser-based inspection viewer at
`tools/gravity_viewer/index.html`. It is intentionally read-only: the browser
loads a selected bundle directory, checks the manifest file sizes and SHA-256
checksums, and displays the model specification, provenance, diagnostics,
network map, and bounded previews of the OD, prediction, and residual tables.
The report summary is rendered as Markdown. The map uses the bundled stop and
stop-time files and OpenStreetMap tiles; it requires internet access for the
background tiles. It does not run Python or recompute the model.

Start it from the repository root with any local static server:

```bash
uv run --no-project python -m http.server 8765 --directory tools/gravity_viewer
```

Open <http://localhost:8765>, choose the case's
`results/viewer_bundle/` directory, and confirm that the page reports
“Bundle loaded and all manifest checks passed.” The viewer uses the browser's
directory picker, so the bundle remains in its original location and is never
copied or changed. Stop the local server after the inspection.
