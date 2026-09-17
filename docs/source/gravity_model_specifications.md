# Gravity-model specifications

The gravity interface can estimate production and destination attractiveness
without an observed OD matrix.  A canonical cell `(o,d,t)` is generated as

```text
q_odt = P_ot * A_dt exp(-beta_T T_odt - beta_X X_odt)
             / sum_d' A_d't exp(-beta_T T_od't - beta_X X_od't)
```

`P_ot` is either the supplied one-dimensional `origin_time_totals` (the
historical default) or an estimated production level from `unit_exposure` (one
neutral exposure per origin/time group).  An a priori OD matrix is deliberately
not part of this interface: matrix-valued inputs and old matrix source names
are rejected rather than interpreted for compatibility.
Destination terms affect relative shares inside the
origin/time softmax; a destination-only time intercept is rejected because it
would cancel exactly.

## Named presets

Use `gravity_model_specification_from_preset(name, features=features)` or put
`preset: ...` in the YAML specification.  Feature-dependent group counts and
combined zone/time mappings are resolved deterministically from `GravityFeatures`.

| Preset | Production | Destination attractiveness | Source |
| --- | --- | --- | --- |
| `minimal_fixed` | none | fixed feature cache | `origin_time_totals` |
| `time_regime` | legacy global correction | fixed feature cache | `origin_time_totals` |
| `zone` | global intercept + origin-zone effect | destination-zone effect | `unit_exposure` |
| `zone_time` | global intercept + origin-zone + time + centered zone/time interaction | destination-zone + centered zone/time interaction | `unit_exposure` |
| `origin_time_full` | global intercept + origin-time effect | destination-zone effect | `unit_exposure` |

The first two presets retain the numerical behavior of the original minimal
and time-regime models.  The remaining presets are intentionally explicit and
their expanded term list is part of the model fingerprint.

## Explicit terms and identification

For advanced models, supply `production.terms` and/or
`destination_attractiveness.terms`.  Each term has a stable `name`, `scope`,
`grouping`, `group_count`, `constraint`, and optional `regularization`.
Categorical terms use `sum_zero` (the final effect is reconstructed as the
negative sum) or `reference` with `reference_category`.  A zone/time
interaction may use `two_way_centered`; it uses a deterministic contrast basis
with zero row and column sums.  Unrestricted grouped effects are rejected when
they would be redundant with another term.  Positive ridge regularization is
recommended for main effects and required for high-dimensional production
interactions.

Example YAML:

```yaml
schema_version: 1
model_name: zone_time_example
production:
  baseline: unit_exposure
  terms:
    - name: production.origin_zone
      scope: origin_zone
      constraint: sum_zero
      regularization: {type: ridge, strength: 1.0}
    - name: production.time_period
      scope: time_period
      constraint: sum_zero
      regularization: {type: ridge, strength: 1.0}
destination_attractiveness:
  source: feature_cache
  terms:
    - name: destination.zone
      scope: destination_zone
      constraint: sum_zero
      regularization: {type: ridge, strength: 1.0}
```

The parser resolves omitted group counts from the feature mappings.  Built-in
mapping names include `origin_zone_index`, `destination_zone_index`,
`time_period_index`, `origin_zone_time_index`, and
`destination_zone_time_index`; custom integer mappings are accepted through
`GravityFeatures.custom_group_indices`.

Every result and run manifest records the preset, production source, and the
fact that no external OD matrix is used, together with the expanded term
specification, parameter names,
constraints, regularization, mapping fingerprints, and layout fingerprint.
This prevents a checkpoint produced by a different specification from being
silently reused. Existing callers that construct
`GravityModelSpecification()` continue to use the original three-parameter
model, but they must provide origin-time totals rather than an OD matrix.

## Obsolete OD-matrix input

The gravity model no longer accepts an a priori origin-by-destination matrix.
`origin_time_totals` must be a one-dimensional vector aligned with the
origin-time groups. Passing a two-dimensional array, an obsolete matrix field
such as `od_matrix` or `a_priori_od_matrix`, or an obsolete matrix production
source raises an explicit validation error. Use `unit_exposure` when observed
production totals are not part of the model.
