# Boundary flows and separate boarding/alighting observations

This page documents the opt-in boundary-flow extension of the gravity
objective. The public package supplies generic infrastructure; a case adapter
defines the scientific meaning of a trip start, observation window, initial
onboard cohort, and terminal outflow.

## Measurement equation

The ordinary OD model predicts a demand vector `x` and applies the existing
fixed-routing operator `A`. A boundary-enabled model adds named latent-flow
blocks:

```text
x              = gravity_demand(od_parameters)
z_initial      = initial_flow_model(initial_parameters)
z_terminal     = terminal_flow_model(terminal_parameters)

latent = A x + B_initial z_initial + B_terminal z_terminal + fixed_offset
mean_m = rho * observation_scale[type(m)] * latent_m
```

`B_initial` represents passengers already onboard before the modeled segment;
it can contribute to alighting rows at any later stop. `B_terminal` represents
passengers boarding at the end of the modeled segment and leaving afterwards;
it contributes to boarding rows. The operators are not assumed to act only on
the first or last row.

Boundary blocks are part of the primary measurement mean. They are therefore
different from `GravityObservationBundle`, whose channels add an auxiliary
likelihood without changing the primary count prediction, and from a fixed
offset, which is known rather than estimated.

## API building blocks

`GravityLinearMeasurementOperator` is a deliberately small protocol requiring
only `num_rows`, `num_columns`, and JAX `matvec`, `rmatvec`, and `matmat`
products. It does not require OD, routing, or assignment metadata. Small
matrices can use `GravityDenseLinearMeasurementOperator`; a case adapter may
provide a matrix-free implementation with the same products.

`GravityAdditiveFlowBlock` combines a stable name, a linear operator, a latent
flow model, and an optional quadratic regularization strength. The supplied
flow models are:

- `DirectNonnegativeFlowModel`, for one positive grouped flow per parameter;
- `LinearNonnegativeFlowModel`, for a lower-dimensional fixed basis whose
  coefficients generate a full non-negative flow vector.

`GravityJointParameterLayout` appends additive parameters after the existing
OD/gravity parameters. It supplies deterministic names and slices,
serialization, fingerprints, transformations, and regularization. With no
additive blocks, the legacy `GravityParameterLayout` and objective path remain
unchanged.

## Boarding and alighting scales

`GravityObservationModel` can attach fixed row-aligned scales to the primary
mean. Set `measurement_types` to a sequence containing `"boarding"` and
`"alighting"`, and provide `boarding_scale` and `alighting_scale`. Estimated
scales are intentionally not enabled yet. When they are introduced, an
identifiability constraint will be required because an unconstrained pair of
event scales is redundant with the existing production/global scale (for
example, fixing `rho_boarding = 1`).

The calibration mask remains explicit: rows included in the likelihood are
distinct from rows supported by the OD operator, rows supported by each
boundary operator, excluded rows, and rows that reach the numerical mean
floor. A positive observation with no active support must be rejected or
excluded by an explicit case-owned calibration policy; it is never silently
treated as an OD zero.

## Grouped initial-onboard parameters

Unrestricted trip-by-stop boundary parameters are usually underidentified:
many combinations of initial stocks and alighting hazards produce the same
stop counts. Prefer grouped cohorts (for example, route, direction, and
departure band) or a low-dimensional survival/hazard basis. Select groups from
the case’s observation window and regularize the latent coefficients. The
adapter owns the mapping from its case data to the rows and columns of each
boundary matrix.

## Persistence and reuse

Boundary operators are persisted separately from the OD routing artifact. The
portable `write_gravity_boundary_artifact` and
`load_gravity_boundary_artifact` helpers store a matrix, shape and dtype,
content hash, row-order fingerprint, boundary-group identifiers,
parameter-layout and model-specification fingerprints, construction settings,
and a completion marker. Corrupt content, changed row order, or incompatible
model fingerprints are rejected.

An unchanged OD assignment artifact remains reusable when its routing,
mapping, canonical-index, and dtype identity are unchanged. Adding or changing
a boundary block changes the joint model fingerprint and therefore requires a
new gravity fit/checkpoint, but does not by itself require rebuilding the OD
operator.

Fit manifests record the OD operator identity, each additive operator
fingerprint, the observation-model fingerprint, the calibration-mask identity,
the complete model specification, and row-support counts. Detailed reports can
include `measurement_contributions.csv`, separating the OD contribution,
each named additive contribution, fixed offset, latent total, and scaled mean
for every measurement row, with summaries by boarding/alighting type.

## Scope of the current implementation

The current extension supports fixed boarding/alighting scales and generic
additive blocks. Estimated event scales, type-specific dispersions, and richer
route-wide hazard models are intentionally deferred. Those additions must
preserve the same joint-layout and provenance contracts rather than introduce
case-specific assumptions into the public package.
