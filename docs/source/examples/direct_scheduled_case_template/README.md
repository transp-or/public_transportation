# Direct-scheduled case-study template

This directory is a current-version, TPG-agnostic starting point for a
private case-study driver. Copy it into a private case repository and replace
the example paths and scientific choices in `config/case.toml`,
`config/structural_zeros.toml`, and `config/model.toml`. The template does not
make a case scientifically valid merely by being copied.

The public template intentionally does not include `pyproject.toml` or
`uv.lock`: the case owner must create them, pin the selected immutable public
package commit, and generate the lockfile before using the `uv run --frozen`
commands below.

The driver owns orchestration. The public package is imported normally from
the frozen environment; no checkout path is added to `sys.path`.

## Layout and input contract

```text
case_study/
├── adapter.py
├── run_case.py
├── config/
│   ├── case.toml
│   ├── structural_zeros.toml
│   └── model.toml
├── inputs/
│   ├── scenario/
│   ├── measurements.csv
│   └── fixed_demand.csv
└── results/
```

The scenario directory must contain `metadata.json`, `stops.csv`,
`lines.csv`, `time_bins.csv`, and a demand file with columns
`origin_stop_id,dest_stop_id,time_bin_id,flow`. Scheduled scenarios also
contain `trips.csv` and `stop_times.csv`. The measurements CSV has the exact
columns `method_id` (string), `measurement_type` (`boarding` or `alighting`),
`stop_id` (string), `time` (`HH:MM:SS`), `value` (finite non-negative float),
and optional `trip_id` and `line_id` strings. The fixed-demand CSV has
`origin_stop_id,dest_stop_id,time_bin_id,fixed_flow`; blank `fixed_flow` means
zero. Every fixed key must exist in the scenario demand table.
The template expects `fixed_demand.csv` to exist; when no cells are frozen,
provide the header row and no data rows.

`config/case.toml` selects the input and persistent output roots:

```toml
scenario = "inputs/scenario"
scenario_demand = "inputs/scenario/prior_demand.csv"
measurements = "inputs/measurements.csv"
fixed_demand = "inputs/fixed_demand.csv"
results = "results"
package_revision = "REPLACE_WITH_PUBLIC_TRANSPORTATION_COMMIT"
theta = 1.0
rho = 1.0
expected_evaluations = 100
construction_time_budget_seconds = 0.0
safety_margin_seconds = 0.0
```

Zero construction budget means no deadline. `expected_evaluations` is used
to choose and record the construction policy; it is not a demand value.
Replace the package-revision placeholder with the exact commit installed in
the case environment before running `check`.

`config/model.toml` is intentionally explicit. The initial `*_` values are
raw optimizer coordinates (before positive transformations), while
`gradient_strategy`, likelihood, ridge strength, tolerances, wall-time, JAX
cache, and shard/memory limits are execution/model choices that the case owner
must review.

## Commands

Run all commands from the copied case-study root:

```bash
uv run --frozen python run_case.py check
uv run --frozen python run_case.py structural-zeros
uv run --frozen python run_case.py prepare
uv run --frozen python run_case.py preflight
uv run --frozen python run_case.py benchmark
uv run --frozen python run_case.py fit
uv run --frozen python run_case.py validate
```

To continue a clean time-budget stop, wait for the previous fit process to
exit, then run exactly one writer with:

```bash
uv run --frozen python run_case.py fit --resume
```

Every stage writes a JSON summary under `results/manifests/`, and long stages
append JSONL progress to `results/logs/<stage>.jsonl`. A stage is complete only
when its process exits with status zero and its summary contains
`"status": "completed"`. A deadline stop is resumable only when the summary
contains `"status": "deadline_stopped"` and the persistent checkpoint or
construction manifest is present.

The template deliberately performs strict validation before constructing an
operator. It refuses to use a stale artifact when the assignment, mapping,
OD-layout, or package identity differs. Do not mix result roots between cases.
After `structural-zeros`, the generated
`results/structural_zeros/fixed_demand.csv` becomes the authoritative fixed
demand for later stages; review its audit and fingerprint before continuing.

## Scheduler scripts

The `scripts/*.sbatch` files are examples only. Review the requested memory,
CPU, wall time, account, and partition with the local Jed policy. They write
stdout and stderr below `results/logs/` and call the same stage commands as the
interactive workflow. Submit the dependency chain with:

```bash
bash scripts/submit_chain.sh
```

The chain is `check → structural-zeros → prepare → preflight → benchmark → fit
→ validate`. Never start two writers against one checkpoint or artifact root;
resume a time-budget stop as a new dependent job after the predecessor has
finished cleanly.
