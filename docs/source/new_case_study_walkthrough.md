# Walkthrough for a new OD-estimation case study

This is the authoritative operational runbook for conducting a new case study
with the current direct-scheduled `public_transportation` workflow. It is
intended for the person who owns the case data and runs the computation. The
case-study repository owns the data, scientific assumptions, configuration,
orchestration, and scheduler jobs; the public package supplies validated
assignment, structural-zero, fixed-routing, gravity, MAP, diagnostics, and
persistence APIs.

This document describes only the current workflow.

## 1. Current package contract

The current workflow uses the direct-scheduled assignment and estimation APIs.
There is no generic `case_study.runner` dispatcher or case-specific template
inside the public package. The private case-study repository must provide a
small driver that assembles the public objects and writes case-specific
summaries.

Record the exact package revision and runtime before any run:

```bash
git rev-parse HEAD
git status --short
python --version
uv run --frozen python -c \
'import public_transportation; print(public_transportation.__file__)'
```

The case manifest should record the package commit and version, Python/JAX
versions, platform, input checksums, configuration files, model
specification, and persistent checkpoint/artifact roots. The current
`pyproject.toml` declares Python `>=3.14,<3.15`; verify that the case
environment satisfies this constraint.

The public package must be installed as a normal Python dependency. The case
driver must not import it from an absolute checkout path.

## 2. Non-negotiable scientific and operational rules

1. Do not run a long fit before input validation, artifact validation, a
   bounded preflight, and a warm objective-and-gradient benchmark have passed.
2. Missing observations are unobserved; they are not zero counts. A numerical
   zero in the observation table is an observed zero.
3. Every boarding or alighting record must resolve to exactly one canonical
   timetable event. Never resolve ambiguity by row order.
4. A positive observed count with no supported modeled response is an
   admission failure. Stop and diagnose the mapping, event time, stop level,
   or OD support; never discard the row silently.
5. Frozen-zero OD cells are not estimator parameters. Positive frozen cells are
   constants whose measured contribution is included through the fixed offset.
6. Reuse is allowed only after identity and fingerprint validation. A matching
   filename is not evidence of compatibility.
7. Keep input preparation, operator construction, preflight, estimation,
   reconstruction, and detailed validation as separate stages.
8. Use the case owner's declared numerical precision consistently. For
   float64 runs, set `JAX_ENABLE_X64=true` before starting the process.
9. Never guess time-bin edges, service-day or after-midnight conventions,
   measurement timestamp semantics, transfer limits, demand units, missing-data
   policy, package revision, or data paths. Stop and record the missing
   decision.

## 3. Current case-study layout

The private case-study repository should own a small driver that assembles
public objects and writes case-specific summaries. A recommended layout is:

```text
case_studies/<case_name>/
├── README.md
├── pyproject.toml
├── uv.lock
├── config/
│   ├── structural_zeros.toml       # if topology rules are enabled
│   └── model.toml                  # optional gravity/MAP settings
├── inputs/
│   ├── scenario/                   # metadata, stops, lines, trips, stop_times, bins
│   ├── measurements_boarding_alighting.csv
│   ├── prior_demand.csv
│   └── fixed_demand.csv
├── run_case.py                     # case-owned orchestration driver
├── scripts/
│   ├── 00_check.sbatch
│   ├── 10_prepare.sbatch
│   ├── 20_preflight.sbatch
│   ├── 30_benchmark.sbatch
│   ├── 40_fit_01.sbatch
│   ├── 41_fit_02.sbatch
│   └── submit_chain.sh
└── results/
    ├── audit/
    ├── checkpoints/
    ├── artifacts/
    ├── logs/
    ├── preflight/
    ├── fits/
    └── validation/
```

The case owner is responsible for `inputs/`, `config/*.toml`, `pyproject.toml`,
`uv.lock`, and the orchestration driver. The public package must be
installed as a normal Python dependency; the driver must not import the
package from an absolute checkout path.

The case-owned `run_case.py` should expose independent stages with these
names (the driver may use a different CLI implementation, but the stage
boundaries must remain explicit):

```bash
uv run --frozen python run_case.py check
uv run --frozen python run_case.py structural-zeros
uv run --frozen python run_case.py prepare
uv run --frozen python run_case.py preflight
uv run --frozen python run_case.py benchmark
uv run --frozen python run_case.py fit
uv run --frozen python run_case.py validate
```

These are commands supplied by the private case driver, not commands provided
by the public package. Each stage must write a durable, identity-bearing
summary and must fail before starting the next expensive stage when its
inputs, fingerprints, or prerequisites are invalid.

## 4. Current workflow at a glance

The current workflow is:

```text
strict scenario and measurement audit
  → canonical timetable and OD/measurement indexes
  → structural-zero analysis and fixed-demand reconciliation
  → compact OD layout and fixed-routing inputs
  → direct-scheduled temporal operator construction
  → gravity/support preflight
  → warm benchmark
  → checkpointed MAP/gravity fit
  → diagnostics, reconstruction, and detailed assignment validation
```

The direct-scheduled operator is the current scalable representation. It uses
content-addressed, persistent artifacts and exposes forward/adjoint products
without requiring a global dense measurement-by-OD matrix.

## 5. Stage 0 — freeze provenance and inputs

Before running any expensive stage, record:

```bash
git rev-parse HEAD
git status --short
python --version
uv run --frozen python -c \
'import public_transportation; print(public_transportation.__file__)'
```

The case manifest should record:

- public-package commit and package version;
- Python, JAX, and platform information;
- checksums of scenario, measurement, prior, and fixed-demand inputs;
- all configuration files;
- the scientific model specification;
- the intended cache and result roots.

All generated artifacts must remain beneath the current case's persistent
result and cache roots, never in a temporary directory.

## 6. Stage 1 — strict input and mapping audit

The case driver must load the scenario with strict validation:

```python
from public_transportation.domain import Scenario

scenario = Scenario.from_folder(
    "inputs/scenario",
    strict=True,
    demand_file="inputs/prior_demand.csv",
)
```

Then build the canonical timetable and assignment identifiers:

```python
from public_transportation.preprocessing import build_canonical_timetable_index
from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.assignment.id_manager import AssignmentIDManager

timetable_index = build_canonical_timetable_index(scenario)
assignment = prepare_assignment(
    scenario=scenario,
    config=AssignmentConfig(),
    timetable_index=timetable_index,
)
id_manager = AssignmentIDManager.build(
    scenario=scenario,
    graph=assignment.graph,
)
```

The audit must stop on duplicate IDs, malformed times, unknown stops or trips,
unresolved measurement events, non-finite values, duplicate OD keys, or
inconsistent service-day conventions. Save the audit and all input checksums
under `results/audit/`.

## 7. Stage 2 — structural zeros and fixed demand

Topology-driven structural-zero preprocessing is TOML-driven. A
structural-zero TOML file can be passed to the public service:

```python
from public_transportation.preprocessing import run_structural_zero_preprocessing

result = run_structural_zero_preprocessing(
    "config/structural_zeros.toml",
    progress=progress_callback,
)
```

The service writes:

```text
fixed_demand.csv
structural_zero_audit.csv
structural_zero_summary.json
fingerprints.json
resolved_config.toml
```

The structural-zero summary reports the number of candidate cells,
structural-zero cells, retained cells, reason counts, and reconciliation counts.
The fingerprints file records scenario, graph, configuration, algorithm, and
serialization provenance. A completed call without an exception and with these
files present is the success condition.

Before proceeding, verify:

- every disconnected or rule-excluded cell is present as fixed zero;
- a pre-existing positive fixed demand never conflicts with a structural zero;
- every positive observation has at least one active free or fixed-positive
  origin/time support;
- the fixed-demand file uses the same canonical OD/time key format as the
  scenario.

## 8. Stage 3 — compact OD layout and direct-scheduled preparation

The current estimator uses a compact layout. Free cells are parameters; frozen
zero cells are absent from the assignment representation; frozen positive cells
are retained only as constants.

The case driver should construct:

```python
from public_transportation.domain import read_fixed_demand_csv
from public_transportation.inference.od_parameter_layout import (
    build_od_parameter_layout,
)
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)

fixed = read_fixed_demand_csv(
    "inputs/fixed_demand.csv",
    scenario=scenario,
)
od_layout = build_od_parameter_layout(
    scenario=scenario,
    fixed_demand=fixed,
)
compact_layout = build_compact_od_assignment_layout(
    parameter_layout=od_layout,
)
```

For direct-scheduled gravity/MAP, the driver then builds the canonical
measurement index and calls
`activate_direct_scheduled_temporal_operator`. The call must provide:

- assignment inputs and fixed routing factory;
- compact OD layout;
- canonical time intervals, demand cells, and measurements;
- observation values and strict measurement mapping;
- a complete content-addressed artifact identity;
- persistent checkpoint and artifact roots;
- explicit construction/resource limits.

The public example is:

```bash
uv run --frozen python docs/source/examples/direct_scheduled_gravity_validation.py \
  --example docs/source/examples/simple_example_02 \
  --cache-directory /path/to/persistent/direct-scheduled-cache
```

This is a reference activation/equivalence run, not a generic medium-network
case runner. A private driver must perform the same assembly for its own data.

A completed direct-scheduled artifact contains:

```text
<artifact-root>/<identity-fingerprint>/
├── manifest.json
├── blocks/block-000000.npz
├── ...
└── fixed_measurement_offset.npy
```

The manifest must contain `"complete": true`, a matching identity
fingerprint, dimensions, block hashes, and fixed-offset hash. Routing
checkpoints have their own manifest with `"status": "completed"` and all
expected routing shards. Incomplete checkpoints are resumable; a final artifact
is never published as complete until all blocks validate.

## 9. Stage 4 — bounded preflight

Do not begin a full fit before preflight. For a gravity problem, construct a
`GravityObjectiveProblem` and call:

```python
from public_transportation.inference.gravity import run_gravity_preflight

preflight = run_gravity_preflight(
    problem=problem,
    raw_parameters=initial_raw,
)
```

A complete result has `completed_phase == GravityPreflightPhase.RECOMMENDATION`.
It reports routing-cache validation, forward/reverse timings, warm
objective-gradient timings, gradient agreement, peak RSS, expected evaluation
time, derivative-strategy recommendation, and a suggested fit wall-time budget.

For block-coordinate MAP, use the bounded support preflight before constructing
all blocks. The public benchmark command is:

```bash
uv run --frozen --extra dev python benchmarks/benchmark_support_preflight.py \
  --mode sampled-exact-support \
  --check \
  --checkpoint-directory /path/to/support-preflight \
  --output /path/to/support-preflight.json
```

That benchmark is wired to a packaged public fixture. For a private network,
the case driver must call `run_support_preflight` with its own partition,
fingerprints, budgets, and persistent checkpoint directory. A complete result
has `status == "completed"` and `result.complete is True`.

## 10. Stage 5 — warm benchmark

The warm benchmark must measure the exact operator and model that will be fit.
At minimum, record:

- first and warm forward products;
- first and warm adjoint products;
- objective and gradient timings;
- gradient agreement between adjoint and batched-forward strategies;
- peak RSS and resident routing bytes;
- operator dimensions and nonzero counts;
- cache hits, misses, and compilation counts.

Useful public reference benchmarks are:

```bash
uv run --frozen python benchmarks/benchmark_sharded_fixed_routing.py \
  --output results/preflight/sharded_fixed_routing.json

uv run --frozen python benchmarks/benchmark_sharded_gravity_operator.py \
  --operator-batch-sizes 1 2 4 8 \
  --output results/preflight/sharded_gravity.json
```

These scripts exercise public fixtures and are not private-case drivers. The
private driver should write its own benchmark JSON with scientific and
execution fingerprints. Reject the fit if the benchmark is non-finite, if
gradient agreement fails, or if the projected runtime/memory is incompatible
with the selected machine or Jed allocation.

## 11. Progress reporting and durable logs

The current package does not support the former generic `--json-progress`
command-line flag. Progress is enabled by passing callbacks. For gravity
estimation, use the durable JSONL sink:

```python
from pathlib import Path
from public_transportation.inference.gravity import GravityJSONLProgressSink

sink = GravityJSONLProgressSink(
    Path("results/logs/gravity.progress.jsonl"),
    durable=True,
)

result = estimate_gravity_model(
    problem=problem,
    compact_layout=compact_layout,
    initial_raw_parameters=initial_raw,
    config=estimator_config,
    execution=execution_policy,
    progress=sink,
)
```

Direct-scheduled construction accepts the same kind of callback through
`progress=`. Keep stdout for human summaries and store progress logs outside
temporary directories. A run is complete only after the process exits
successfully and its complete artifact/result manifest exists.

The JSONL records include schema, timestamp, event type, model context, and
serialized event data. Construction events report phase, status, completed and
total units, elapsed and remaining time, current unit, cache counters, and
checkpoint location when available. Estimator events report iteration,
objective, gradient norm, elapsed time, and checkpoint state. An early ETA is
provisional; it becomes more reliable after completed units accumulate.

## 12. Stage 6 — initial MAP/gravity fit

The recommended production fit is the smallest declared gravity model with
MAP regularization or prior structure, after a short Poisson diagnostic fit.
The case driver must create a `GravityEstimatorConfig` and
`GravityExecutionPolicy` with:

- an explicit iteration limit;
- an explicit derivative strategy from preflight;
- a persistent checkpoint path;
- a wall-time budget below the scheduler limit;
- a progress sink;
- a persistent JAX compilation-cache directory when appropriate.

The estimator returns `GravityEstimationResult`. Interpret the status as:

| Status | Meaning |
|---|---|
| `converged` | optimizer convergence reported; also require `success == True` |
| `iteration_limit` | valid result at the configured limit, not evidence of convergence |
| `stopped_by_time_budget` | clean resumable stop; do not call it complete |
| numerical/other failure | diagnose before changing the model |

Checkpoints are identity-bound. Resume only with the same scientific model,
OD layout, routing, mapping, observations, and checkpoint identity. A changed
resource budget may be allowed; a changed model or artifact identity is not.

The current public package has no generic `run.py fit` command. The private
driver owns the fit command and should write at least:

```text
results/fits/<fit-id>/manifest.json
results/fits/<fit-id>/result.json or result.npz
results/fits/<fit-id>/checkpoint.json
results/logs/<fit-id>.progress.jsonl
```

## 13. Stage 7 — diagnostics, reconstruction, and validation

Only after accepting a fit:

1. save the compact/free demand and the full OD vector;
2. assign the reconstructed demand with the same routing identity;
3. compare modeled and observed boarding/alighting counts;
4. report raw residuals, deviance/RMSE, grouped residuals, and bound activity;
5. compare prior versus estimate and record prior sensitivity;
6. preserve the exact model, operator, and data fingerprints.

Never reconstruct a full OD vector before fit acceptance merely for convenience:
the estimator should operate only on free/compact cells.

## 14. Observations that do not match a modeled path

For each observation, distinguish:

- unknown stop/trip/event: input or mapping error;
- known event with no access/egress support: structural or timetable mismatch;
- known event with only frozen-zero origins: fixed-demand/support conflict;
- observed zero with valid support: legitimate data;
- positive observation with no valid support: hard failure.

The current direct-scheduled builder performs a positive-boarding support preflight
before construction. Resolve the source data, timestamp convention, physical-stop
mapping, or OD candidate universe before proceeding. Do not drop the observation,
invent a route, or replace it by a nearby event without an explicit documented
policy.

## 15. Jed scheduling and restartable long runs

Use separate jobs for preparation, preflight, benchmark, and fit. Keep each job
below the scheduler wall-time limit and preserve checkpoint/artifact roots on
persistent storage.

A typical dependency chain is:

```bash
PREPARE_JOB=$(sbatch --parsable scripts/10_prepare.sbatch)
PREFLIGHT_JOB=$(sbatch --parsable \
  --dependency=afterok:${PREPARE_JOB} scripts/20_preflight.sbatch)
BENCHMARK_JOB=$(sbatch --parsable \
  --dependency=afterok:${PREFLIGHT_JOB} scripts/30_benchmark.sbatch)
FIT_1_JOB=$(sbatch --parsable \
  --dependency=afterok:${BENCHMARK_JOB} scripts/40_fit_01.sbatch)
FIT_2_JOB=$(sbatch --parsable \
  --dependency=afterok:${FIT_1_JOB} scripts/41_fit_02.sbatch)
printf 'prepare=%s preflight=%s benchmark=%s fit=%s,%s\n' \
  "$PREPARE_JOB" "$PREFLIGHT_JOB" "$BENCHMARK_JOB" "$FIT_1_JOB" "$FIT_2_JOB"
```

Each fit segment must use the same model fingerprint and checkpoint path. The
first segment starts a fresh fit; later segments resume it. Never launch a
second writer against the same checkpoint. Use separate fit IDs for sensitivity
runs.

Slurm wrappers should write stdout and stderr to persistent files:

```text
results/logs/<stage>-<job-id>.out
results/logs/<stage>-<job-id>.err
```

Monitor with:

```bash
tail -f results/logs/<stage>-<job-id>.out
tail -f results/logs/<stage>-<job-id>.err
```

A clean scheduler stop is acceptable only when the durable checkpoint or
partial artifact is explicitly marked resumable. A zero exit code alone is not
sufficient if the final manifest is absent.

## 16. Final acceptance checklist

Before calling the current case complete, verify:

- [ ] package revision, Python version, and environment are recorded;
- [ ] scenario, observations, prior, fixed demand, and configurations are checksummed;
- [ ] canonical timetable and measurement identities are stable;
- [ ] all positive observations have supported modeled responses;
- [ ] structural-zero reconciliation has no positive fixed-demand conflict;
- [ ] compact layout excludes frozen-zero cells from estimation and assignment;
- [ ] direct-scheduled artifact manifest is complete and fingerprint-compatible;
- [ ] preflight reaches its declared completion phase;
- [ ] warm objective and gradient are finite and agree across strategies;
- [ ] fit checkpoint and result carry the expected model and artifact fingerprints;
- [ ] convergence, time-budget stop, and iteration-limit statuses are distinguished;
- [ ] progress and stderr logs are stored on persistent storage;
- [ ] accepted fits are reconstructed and validated against observations;
- [ ] all case-specific scientific assumptions are documented in the private README.
