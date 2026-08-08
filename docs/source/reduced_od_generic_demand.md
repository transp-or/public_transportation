# Composable reduced-dimensional demand models

`DemandModelSpecification` selects blocks before compilation. The default
minimal gravity API remains supported; the generic API is used when period,
group, interaction, or zero-inflation components are required.

```python
from public_transportation.inference.reduced_od import (
    DemandModelDimensions,
    DemandModelSpecification,
    ImpedanceSpecification,
    ObservationSpecification,
    ProductionSpecification,
    build_demand_parameter_layout,
)

specification = DemandModelSpecification(
    production=ProductionSpecification(intercept=True, period_effects=True),
    impedance=ImpedanceSpecification(travel_time="period", transfers="global"),
    observation=ObservationSpecification(family="negative_binomial"),
)
layout = build_demand_parameter_layout(
    specification,
    DemandModelDimensions(periods=2, origin_groups=1, destination_groups=1),
)
```

All categorical effects use an explicit sum-to-zero convention represented by
one fewer free coordinate. Origin-group-by-period and
destination-group-by-period interactions are double centered. Low-rank origin
and destination factors are column centered and ridge regularized; their
individual columns are not identified under rotations and must not be given a
behavioral interpretation.

The negative binomial uses mean `mu`, dispersion `r`, and variance
`mu + mu**2 / r`. ZIP and ZINB use a Bernoulli structural-zero probability for
retained measurement records. This is distinct from timetable-infeasible OD
cells removed during preprocessing. Compare NB before interpreting excess
zeros as evidence for zero inflation.

`warm_start_demand_parameters` maps named blocks between resolved layouts and
returns an audit report. `build_grouping_hierarchy` builds deterministic nested
regions from caller-prepared supply, topology, and observability signatures;
external group memberships may instead be supplied through the problem index
vectors.

## Migration from `MinimalGravitySpecification`

Existing minimal problems and estimators are unchanged and numerically
compatible. New applications should reproduce that baseline first (M0), then
construct `DemandModelProblem` over the same `ConditionalGravityFeatures` and
`ReducedResponseOperator`. Checkpoint owners must include the generic
specification, layout, grouping, feature, operator, data, and software
fingerprints; minimal checkpoints are not silently relabeled as generic ones.

Existing `GravityFeatures` and device-native fixed-routing operators can be
reused without constructing a dense response matrix through
`build_generic_demand_problem_from_gravity`. The generic fitter supports ML,
Gaussian-prior MAP, raw or canonical named bounds, atomic deadline
checkpoints, and identity-validated resume. The public synthetic and Geneva
workflows execute a bounded M0 fit and report compilation, optimization, and
warm value-and-gradient timings.

During optimization, every accepted iteration reports its duration, the
rolling mean duration over at most ten iterations, remaining iterations,
estimated remaining seconds, and the expected UTC completion time. Compilation
is timed separately and is excluded from the iteration estimate. When a
checkpoint is configured, progress also reports its path and age. Pressing
Ctrl-C returns an `interrupted` result after atomically saving the latest
accepted iterate, which can subsequently be resumed with the same fit identity.
