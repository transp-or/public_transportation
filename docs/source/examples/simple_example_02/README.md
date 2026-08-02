# Simple Example 02

Synthetic route-choice bus network. The reference data are in `data/`.

`data/fixed_demand.csv` is the sparse list of OD/time-bin cells excluded from
estimation. A blank or omitted `fixed_flow` means zero; an explicit value fixes
the cell at that nonnegative flow. All estimation scripts read this file and
use the same reduced layout for Bayesian VI, maximum likelihood, and MAP.

To compare unregularized, ridge-to-prior, and scaled-ridge fixed-routing linear
least squares, run:

```bash
python estimation/run_linear_fixed_routing.py
```

## Reduced-dimensional gravity estimation

From the repository root, run:

```bash
uv run python docs/source/examples/simple_example_02/estimation/run_gravity_estimation.py
```

The script illustrates the complete model-development sequence:

1. prepare path features, prior production totals, and attractiveness;
2. build or reuse a fingerprint-validated BCOO measurement operator;
3. estimate the minimal negative-binomial model and inspect adequacy;
4. select a broad time-period relaxation and verify its exact parent warm start;
5. re-estimate on calibration journeys and score untouched journeys.

It writes `estimation/results/gravity_estimation_summary.json`. The committed
reference uses 70 free cells, two frozen-positive cells, and 270 measurements.
It demonstrates API and validation semantics, not a scientific model choice.
