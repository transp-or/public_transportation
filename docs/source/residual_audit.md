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
`predicted_value`. The persisted gravity-report aliases `row_index`,
`observed`, and `modeled` are accepted, so an existing
`predicted_measurements.csv` can be audited directly. A `residuals.csv` table
may add `variance` and previously persisted residual fields.

An optional contribution table is joined by `observation_id`. Known component
names include `od_contribution`, `initial_onboard_contribution`,
`terminal_outflow_contribution`, `fixed_offset`, and `scaled_mean`; additional
numeric component columns are retained. An optional metadata table is also
joined by `observation_id` and can contain arbitrary case-defined columns such
as `line`, `stop`, `time_bin`, `time_regime`, `journey_position`, or
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
failure counts/fractions. A metric with no valid denominator is written as an
empty value, never as a misleading zero.

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

For two persisted runs, each directory must contain
`predicted_measurements.csv`; `residuals.csv`,
`measurement_contributions.csv`, and `report.json` are discovered when
present:

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

`audit_manifest.json` records schema version, tool version, input paths and
SHA-256 fingerprints, model/specification fingerprints when available,
thresholds, groupings, comparison identity, and the holdout flag. The summary
states observed/predicted totals, residual metrics, support failures, largest
grouped mismatches, contribution checks, and whether the run is an independent
holdout. Fit-observation analysis is not independent validation merely because
the report was generated after fitting.

Boundary contributions are additive measurement components. The report may
show that a boundary component dominates a row or that OD contribution is zero
while the total prediction is positive, but it does not interpret those facts
as flow-conservation violations or as OD demand.
