# Geneva gravity integration validation

The public Geneva snapshot now exercises the reduced-dimensional gravity workflow
end to end. The OD truth and prior are synthetic; no private TPG demand data is
included.

Scheduled path preprocessing supplies minimum journey time and transfer count.
Positive prior free demand supplies origin-period production totals and
destination-period attractiveness offsets. The four half-hour bins are mapped into
two explicit broad periods. All 15,032 frozen-zero cells remain outside the 96-cell
compact assignment, and the existing boarding/alighting table supplies 8,967
observations.

Geneva's measurement dimension makes the general fused operator constructor
unsuitable on the laptop because its reverse dynamic program carries a
node-by-measurement state. The runner instead compiles the validated fixed-routing
forward assignment once, evaluates its 96 basis columns sequentially, and assembles
BCOO data and indices directly. It never creates a dense measurement matrix. The
result has 0.556% density and occupies 95,760 bytes.

The committed two-iteration bounded run completed minimal negative-binomial
estimation and adequacy, advisory diagnostics, an exact zero-difference warm start
for a selected broad-period child, immutable two-node lineage, and grouped
vehicle-journey holdout re-estimation with 8,103 calibration and 864 held-out
measurements.

The minimal model reached NB deviance 1,236.65 and RMSE 1.837 at the imposed
iteration limit. The broad-period child was exercised as an integration test, not
accepted as a scientific improvement: its local approximate gain was only 0.089.
The grouped holdout RMSE was 1.166 versus calibration RMSE 1.795. These bounded
values validate software integration and must not be treated as converged estimates
or a model-selection conclusion.

```bash
uv run python docs/source/examples/geneva_gtfs/estimation/run_gravity_validation.py \
  --maximum-iterations 2 --holdout-iterations 2
```

The machine-readable report is `results/gravity_validation_summary.json`. Pytest
validates it by default. Set `RUN_GENEVA_GRAVITY_ACCEPTANCE=1` to rerun the live
one-iteration acceptance test. Longer estimation belongs in staged fresh processes
or on the server so accumulated JAX compilations do not determine laptop memory.
