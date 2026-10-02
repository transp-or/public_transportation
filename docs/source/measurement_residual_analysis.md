# Measurement residual analysis

`analyze_measurement_residuals` is a reusable, provenance-aware diagnostic
layer for a completed fit. It consumes observed values, fitted means, the
variance used by the likelihood, and optional measurement metadata. It does
not rebuild routing, refit parameters, alter observations, delete rows, or
create a calibration mask.

The analysis is explicitly in-sample unless the caller supplies an independent
holdout table. Its candidate patterns and suggestions are advisory evidence
for a data audit or a future model change; they are not an automatic model
selector and are not evidence of out-of-sample predictive performance.

## API and residual definitions

```python
import pandas as pd

from public_transportation.inference.residual_audit import (
    ResidualBootstrapConfig,
    analyze_measurement_residuals,
    write_measurement_residual_analysis,
)

table = pd.DataFrame(
    {
        "observed_value": [12.0, 10.0],
        "predicted_value": [11.0, 8.0],
        "variance": [11.0, 8.0],
        "measurement_type": ["boarding", "alighting"],
        "line": ["L1", "L1"],
        "direction": ["out", "out"],
        "stop": ["s1", "s2"],
        "time_period": ["am", "am"],
        "trip_id": ["journey-1", "journey-1"],
        "journey_position": [1, 2],
    }
)
analysis = analyze_measurement_residuals(
    table,
    table["predicted_value"],
    variance=table["variance"],
    bootstrap_config=ResidualBootstrapConfig(replicates=500, seed=2026),
    provenance={
        "likelihood_family": "poisson",
        "model_fingerprint": "fit-fingerprint",
        "artifact_fingerprint": "operator-fingerprint",
        "package_revision": "revision",
    },
)
write_measurement_residual_analysis(analysis, "report")
```

The default convention is `observed_minus_modeled`:

```text
raw residual       = observed - modeled
absolute residual  = |raw residual|
relative residual  = raw residual / observed
Pearson residual   = raw residual / sqrt(variance)
deviance residual  = signed sqrt(likelihood deviance)
```

Relative residuals are unavailable (written as empty/`null`) when the
observed denominator is zero or numerically negligible. The supplied variance
is always used for Pearson residuals. For a negative-binomial fit, callers
should supply the fitted variance or a positive `dispersion` in provenance;
otherwise the result records an explicit Poisson fallback warning. Poisson and
negative-binomial deviance residuals are distinguished in provenance.

The provenance manifest records the residual and variance conventions,
likelihood family, model and artifact fingerprints, package revision, analysis
configuration, input fingerprints, and an analysis fingerprint. Missing model
or artifact fingerprints are never silently presented as scientific identity;
the input-derived artifact fingerprint is used only to make the diagnostic
record reproducible.

## Grouped diagnostics

When the corresponding metadata columns exist, the rich grouped table contains
rows for:

- overall;
- method identifier;
- measurement type;
- line and direction (including `line_direction`);
- stop;
- time period (including `line_time_period`);
- stop × measurement type;
- origin and destination zone;
- vehicle journey (`trip_id` or `vehicle_journey`).

Each row reports counts, independent cluster counts, observed and modeled
totals, signed and relative bias, mean residual, MAE, RMSE, weighted RMSE,
mean Pearson and deviance residuals, positive/negative fractions, the maximum
absolute standardized residual, fitted-value range, and zero-observation
count. This table is complementary to the historical `grouped_residuals.csv`;
the latter remains available for existing consumers.

## Cluster-aware uncertainty

If `trip_id` (or `vehicle_journey`) exists, the default uncertainty method is a
deterministic cluster bootstrap. Complete journeys are resampled, preserving
all their observations. The seed, replicate count, confidence level, and
confidence intervals for mean and relative bias are recorded in the manifest.

Without a cluster identifier, uncertainty is unavailable by default. A caller
must explicitly set `allow_observation_level=True` to request a row bootstrap;
the result is marked less reliable and contains a warning. The tool never
reports row-wise standard errors as if rows were independent journeys.

## Journey-sequence diagnostics

With a journey identifier and an ordering column (`journey_position`,
`stop_sequence`, `sequence`, or `observation_time`), the report includes
adjacent residual correlation, sign-run length, cumulative residual, cumulative
boarding/alighting imbalance, first/last residuals, and mean Pearson residual.
If either identifier or ordering is unavailable, the report explicitly states
that sequence diagnostics could not be computed.

## Candidate patterns and suggestions

A candidate requires practical effect size and sufficient observations. When
cluster uncertainty is available, its confidence interval must exclude zero;
when uncertainty is unavailable, the candidate is explicitly downgraded. The
labels include `candidate_systematic_bias`, `candidate_overprediction`,
`candidate_underprediction`, `candidate_time_regime_pattern`,
`candidate_line_direction_pattern`, `candidate_stop_pattern`,
`candidate_boarding_alighting_imbalance`, `candidate_journey_sequence_pattern`,
and `candidate_destination_pattern` as applicable.

Suggestions distinguish data audits from model reviews. Examples include
checking stop/platform mapping, direction and timetable coding, time-bin
boundaries, boarding/alighting definitions, initial-onboard and terminal-outflow
corrections, destination representation, and variance/offset assumptions.
Large standardized residuals alone never trigger an automatic recommendation
to use a negative-binomial likelihood, and no suggestion is grounds for
deleting a row. Any future calibration mask must be explicit, reproducible,
separately fingerprinted, and justified in writing.

## Report integration and outputs

Every completed gravity detailed report now runs this analysis after the fit
and prediction vectors have been validated. The report retains the historical
files and adds:

```text
report/residual_analysis.json
report/grouped_residual_analysis.csv
report/journey_residual_analysis.csv
report/residual_suggestions.json
report/residual_suggestions.md
report/residual_analysis_manifest.json
```

`report/report.json` contains a compact `residual_analysis` section with
schema version, scope (`in_sample`), status, cluster-bootstrap method,
candidate patterns, suggestions, warnings, output paths, and the analysis
fingerprint. The detailed manifest is the authoritative record of thresholds,
provenance, uncertainty configuration, and limitations.

The Markdown suggestions are a convenience for human review. Machine-readable
JSON and CSV files should be used for downstream automation. A failed or
unavailable analysis must be represented by a manifest with its reason and
recommended corrective action rather than being silently omitted.

## Worked report review

After a fit report has been written, inspect the largest candidate groups
without rerunning the model:

```python
import json
import pandas as pd

groups = pd.read_csv("report/grouped_residual_analysis.csv")
largest = groups.loc[
    groups["grouping"].ne("overall")
].sort_values("maximum_absolute_standardized_residual", ascending=False).head(10)
print(largest[["grouping", "group_key", "mean_residual", "relative_total_bias"]])

with open("report/residual_suggestions.json", encoding="utf-8") as stream:
    suggestions = json.load(stream)["suggestions"]
for item in suggestions:
    print(item["category"], item["evidence_group"], item["suggested_action"])
```

For a boarding/alighting imbalance, first inspect
`candidate_boarding_alighting_imbalance` and then compare the corresponding
rows in `journey_residual_analysis.csv`. A persistent imbalance is a reason to
audit measurement definitions, initial-onboard and terminal-outflow boundary
rules, and stop/journey mapping. It is not a reason to force the two processes
to balance, delete rows, or change the likelihood before the mean model has
been checked.

If a stop or line-direction pattern has a confidence interval excluding zero,
review the indicated data contract first. If the interval is unavailable, the
manifest marks the candidate as downgraded; treat it as a hypothesis requiring
independent evidence.
