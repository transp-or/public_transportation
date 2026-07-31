# Simple Example 01

This example illustrates the complete workflow for generating synthetic public-transportation observations, estimating OD demand, and post-processing the estimation results.

The workflow is organized in three stages:

1. **Pre-processing**: generate synthetic observations.
2. **Estimation**: estimate OD demand using Bayesian inference, maximum likelihood, or both.
3. **Post-processing**: analyze and report the estimation results.

---

## Directory structure

```text
simple_example_01/
    data/
        README.md
        metadata.json
        stops.csv
        lines.csv
        trips.csv
        stop_times.csv
        time_bins.csv
        true_demand.csv
        prior_demand.csv

    pre_processing/
        run_preprocessing.py
        results/

    estimation/
        run_bayesian.py
        run_maximum_likelihood.py
        run_both.py
        results/

    post_processing/
        run_postprocessing.py
        results/
```

The `data/` directory contains the reference input data distributed with the example.  
Each stage writes its generated outputs to its own `results/` directory.

---

## Step 1: pre-processing

Run:

```bash
cd pre_processing
python run_preprocessing.py
```

This script reads the reference files from:

```text
../data/
```

and generates synthetic estimation inputs in:

```text
pre_processing/results/
```

Expected generated files include:

```text
pre_processing/results/demand.csv
pre_processing/results/measurements_boarding_alighting.csv
```

These files are used by the estimation scripts.

---

## Step 2: estimation

Run one of the following scripts:

```bash
cd ../estimation
python run_bayesian.py
```

or:

```bash
python run_maximum_likelihood.py
```

or:

```bash
python run_both.py
```

For fixed-routing linear least squares with explicit regularization, run:

```bash
python run_linear_fixed_routing.py
```

The estimation scripts read:

```text
../data/
../pre_processing/results/
```

and write estimation results to:

```text
estimation/results/
```

Typical result files are:

```text
estimation/results/vi_od_theta_results.npz
estimation/results/ml_od_theta_results.npz
estimation/results/compare_vi_ml_od_theta_results.npz
```

---

## Step 3: post-processing

Run:

```bash
cd ../post_processing
python run_postprocessing.py
```

The post-processing script reads:

```text
../data/
../pre_processing/results/
../estimation/results/
```

and writes reports and derived outputs to:

```text
post_processing/results/
```

Typical outputs are organized by estimation method:

```text
post_processing/results/bayesian/
post_processing/results/ml/
```

---

## Reproducibility convention

The example follows this convention:

```text
data/       stable reference input data
results/    generated outputs of a workflow stage
```

The `data/` directory should not be modified by the scripts.  
All generated files should be written to the corresponding `results/` directory.

This avoids duplicated input files and keeps the workflow reproducible.

## Frozen OD cells

`data/fixed_demand.csv` is the sparse list of OD/time-bin cells excluded from
estimation. A blank or omitted `fixed_flow` means zero; an explicit value fixes
the cell at that nonnegative flow. The same reduced layout is used by Bayesian
VI, maximum likelihood, and MAP estimation.

---

## Recommended execution order

From the `simple_example_01/` directory:

```bash
cd pre_processing
python run_preprocessing.py

cd ../estimation
python run_both.py

cd ../post_processing
python run_postprocessing.py
```
