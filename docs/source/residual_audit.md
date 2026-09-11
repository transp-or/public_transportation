# Residual audit of persisted model reports

The residual-audit tool analyzes persisted prediction, residual, contribution,
and metadata tables. It does not rerun preparation, routing, assignment, or
fitting. It is therefore suitable for investigating systematic model--data
mismatches after a fit has completed.

The public package owns the numerical diagnostics, grouping, contribution
accounting, provenance, and report schema. A private case adapter owns the
meaning of its line, stop, trip, time-regime, journey-position, and
boundary-role labels. The tool contains no network- or case-specific rules.

## Input contract

The prediction table must contain `observation_id`, `observed_value`, and
`predicted_value`. The persisted gravity-report aliases are accepted:

| Persisted gravity column | Canonical audit column |
| --- | --- |
| `row_index` | `observation_id` (and retained as `source_row_index`) |
| `observed` | `observed_value` |
| `modeled` or `predicted` | `predicted_value` |

Thus an existing `predicted_measurements.csv` can be audited directly. A
`residuals.csv` table may add `variance` and previously persisted residual
fields.

An optional contribution table is joined by `observation_id`. After alias
normalization, the gravity-report fields are:

```text
observation_id, measurement_type, observation_scale,
od_contribution, fixed_offset, latent_total, scaled_mean,
initial_onboard_contribution, terminal_outflow_contribution,
active_components, source_row_index
```

Only the explicit component names `od_contribution`, `fixed_offset`,
`initial_onboard_contribution`, and `terminal_outflow_contribution`, together
with documented future columns ending in `_contribution` or `_flow`, are
treated as additive components. Identifiers, `source_row_index`, observed
values, `measurement_type`, `observation_scale`, `latent_total`, predicted
means, and `active_components` are never inferred as contributions merely
because they are numeric. An optional metadata table is joined by
`observation_id` and can contain arbitrary case-defined columns such as
`line`, `stop`, `time_bin`, `time_regime`, `journey_position`, or
`boundary_role`.

Observation identifiers must be unique in every table. The audit refuses a
join when identifiers differ, and it never removes observations automatically.

## Diagnostics

For each row the tool computes raw, relative, Pearson, and signed Poisson
deviance residuals. Relative residuals are left unavailable when the observed
denominator is below the configured threshold. Pearson variance defaults to
the predicted mean and is protected by a variance floor.

Rows with

```text
observed_value > positive_observation_threshold
predicted_value <= predicted_mean_floor
```

are marked with `positive_observation`, `near_zero_prediction`, and
`support_failure`. They are reported separately because a positive observation
with no predicted support is a support/data-contract diagnostic, not an
ordinary large residual.

Any supplied grouping columns can be combined. The grouped report includes
totals, MAE, RMSE, variance-weighted RMSE, Pearson diagnostics, and support
failure counts/fractions. Weighted metrics always include both
`weighted_rmse_all` and `weighted_rmse_excluding_support_failures`; the legacy
`weighted_rmse` field follows the explicit
`include_support_failures_in_weighted_metrics` policy. The raw and deviance
residual summaries always retain all observations. A metric with no valid
denominator is written as an empty value, never as a misleading zero.

## Contribution accounting

Contribution checks are performed at two levels. When `latent_total` is
available, the audit checks:

```text
od_contribution + fixed_offset + boundary contributions ≈ latent_total
```

It separately checks the scaled prediction. With an observation scale and a
persisted modeled mean, this is:

```text
latent_total × observation_scale ≈ modeled_mean
```

If a latent total or modeled mean is absent, the available component sum and
`predicted_value` are used instead. An unscaled component sum is never compared
directly with a scaled modeled mean when `observation_scale` is not one. The
machine-readable manifest reports `latent_component_check` and
`scaled_prediction_check`, including rows checked, rows not matching, and the
maximum absolute difference.

## Command-line use

From a package environment, audit an existing report directory with:

```bash
uv run python -m public_transportation.inference.residual_audit \
  --predicted-measurements path/to/predicted_measurements.csv \
  --residuals path/to/residuals.csv \
  --contributions path/to/measurement_contributions.csv \
  --metadata path/to/observation_metadata.csv \
  --group-by measurement_type \
  --group-by line \
  --group-by journey_position \
  --output path/to/residual_audit
```

For a Poisson-only audit, require an explicit likelihood declaration in the
model report:

```bash
uv run python -m public_transportation.inference.residual_audit \
  --predicted-measurements path/to/predicted_measurements.csv \
  --model-manifest path/to/report.json \
  --expected-likelihood-family poisson \
  --output path/to/residual_audit
```

The declaration may be at `model_specification.likelihood.family` or
`likelihood.family`. The command fails when the declaration is missing or is
not `poisson`; it never silently reinterprets a negative-binomial or
Dirichlet--multinomial model.

For two persisted runs, the loader accepts either a flat run directory:

```text
run/
├── predicted_measurements.csv
├── residuals.csv
└── report.json
```

or the gravity-report layout:

```text
run/
└── report/
    ├── predicted_measurements.csv
    ├── residuals.csv
    ├── measurement_contributions.csv
    └── report.json
```

Files directly below the supplied directory take precedence. Otherwise the
`report/` subdirectory is inspected. If neither location contains the required
prediction table, the error lists both paths. The resolved paths are retained
in `audit_manifest.json`.

```bash
uv run python -m public_transportation.inference.residual_audit \
  --run-a path/to/run_a \
  --run-b path/to/run_b \
  --metadata path/to/observation_metadata.csv \
  --group-by measurement_type \
  --output path/to/comparison_audit
```

Comparison requires identical observation identifiers and observed values.
Use `--allow-subset-comparison` only when a deliberately documented subset
comparison is scientifically intended; otherwise the command fails closed.
Thresholds can be supplied as CLI options or in a small JSON/TOML file passed
with `--config`, for example:

```toml
positive_observation_threshold = 0.5
predicted_mean_floor = 1e-8
pearson_threshold = 3.0
variance_floor = 1e-8
relative_observation_threshold = 1e-8
include_support_failures_in_weighted_metrics = false
expected_likelihood_family = "poisson"
```

## Generic synthetic example

The API can be exercised without a network or private repository:

```python
import pandas as pd

from public_transportation.inference.residual_audit import (
    ResidualAuditRun,
    write_residual_audit,
)

observations = pd.DataFrame(
    {
        "observation_id": ["b1", "a1"],
        "observed_value": [12.0, 5.0],
        "predicted_value": [11.0, 1.0e-12],
        "measurement_type": ["boarding", "alighting"],
        "line": ["line-A", "line-B"],
        "journey_position": ["first", "last"],
    }
)
contributions = pd.DataFrame(
    {
        "observation_id": ["b1", "a1"],
        "od_contribution": [11.0, 0.0],
        "initial_onboard_contribution": [0.0, 1.0],
        "fixed_offset": [0.0, 0.0],
        "scaled_mean": [11.0, 1.0],
    }
)

write_residual_audit(
    ResidualAuditRun(observations, contributions=contributions),
    output_directory="residual_audit",
    group_by=["measurement_type", "line"],
)
```

The second row is intentionally a positive observation with near-zero model
support, and is written to `support_failures.csv` rather than silently
discarded.

An existing gravity contribution file can be audited without converting its
schema first:

```csv
row_index,measurement_type,observed,observation_scale,od_flow,fixed_offset,latent_total,modeled_mean,initial_onboard_flow,terminal_outflow_flow,active_components
0,alighting,0,1,0,0,2.5,2.5,2.5,0,initial_onboard
```

Here `od_flow`, `initial_onboard_flow`, and `terminal_outflow_flow` are
normalized to contribution names. `observed` and the generated
`source_row_index` remain metadata, and the latent and scaled checks both pass.

## Output and interpretation

The output directory contains:

```text
audit_manifest.json
summary.md
row_diagnostics.csv
grouped_metrics.csv
support_failures.csv
contribution_summary.csv
run_comparison.csv
comparison_grouped_metrics.csv
```

`audit_manifest.json` records schema version, tool version, resolved input
paths and SHA-256 fingerprints, model/specification fingerprints when
available, likelihood-family verification, thresholds, support-failure policy,
weighted-metric policy, aggregate support-failure/weighted metrics, groupings,
comparison identity, and the holdout flag.
The summary contains dedicated sections for likelihood family, support
failures, weighted residual metrics, contribution consistency, and artifact
provenance. Fit-observation analysis is not independent validation merely
because the report was generated after fitting; set the explicit holdout flag
only for a genuinely independent evaluation.

Boundary contributions are additive measurement components. The report may
show that a boundary component dominates a row or that OD contribution is zero
while the total prediction is positive, but it does not interpret those facts
as flow-conservation violations or as OD demand.
