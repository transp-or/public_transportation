# Configuration-driven case-study template

This directory is a public, runnable template for a canonical OD-estimation
case. It uses the committed small multi-line synthetic dataset in
`inputs/scenario/` and the canonical measurement table in `inputs/`;
no private repository or generated artifact is required.

For a real case, copy this directory into the case-study repository, replace
the paths and explicit policy values in `config/case.toml`, and pin the public
package commit in `pyproject.toml`. The generic adapter and runner remain
unchanged for canonical inputs. A custom hook is required only when a source
format cannot be represented by the declarative mappings.

## Setup

From the repository root, enter the case-study root (the directory containing
`adapter.py`, `run.py`, `config/`, and `pyproject.toml`) before running these
commands:

```bash
CASE_ROOT=case_studies/<case_name>
cd "$CASE_ROOT"

test -f adapter.py &&
test -f run.py &&
test -f config/case.toml &&
test -f config/time_discretization.toml &&
test -f config/reduced_od.toml &&
test -f config/structural_zeros.toml &&
test -f config/model.toml &&
test -f pyproject.toml

uv lock
uv sync --frozen
JAX_ENABLE_X64=true uv run --frozen python run.py check
```

The template intentionally does not include a lockfile: the case owner must
replace `<commit>` with an immutable public-repository commit and generate the
case-specific `uv.lock`. A sibling editable checkout is not reproducible.
The TOML files are illustrative contracts and must be reviewed and edited by
the case owner; copying the template does not make the case scientifically
ready. A missing path in the check is a case-setup/documentation failure and
must be fixed before diagnosing the public package. Replace `<commit>` and
review the contracts before running `uv lock`; the copied values are not a
scientific approval. If the structural check fails, report the exact missing
path as `CASE-SETUP/DOCUMENTATION FAILURE` before continuing.

## Independent OD-universe stages

Run exactly one stage per invocation:

```bash
JAX_ENABLE_X64=true uv run --frozen python run.py od-universe
JAX_ENABLE_X64=true uv run --frozen python run.py time-discretization
JAX_ENABLE_X64=true uv run --frozen python run.py materialize-bins \
  --candidate recommendation --reviewer "case owner"
JAX_ENABLE_X64=true uv run --frozen python run.py expand-od
JAX_ENABLE_X64=true uv run --frozen python run.py structural-zeros
JAX_ENABLE_X64=true uv run --frozen python run.py prepare
JAX_ENABLE_X64=true uv run --frozen python run.py preflight
JAX_ENABLE_X64=true uv run --frozen python run.py benchmark
JAX_ENABLE_X64=true uv run --frozen python run.py fit --method ml --likelihood poisson
JAX_ENABLE_X64=true uv run --frozen python run.py diagnose --fit results/fits/ml_poisson.json
JAX_ENABLE_X64=true uv run --frozen python run.py reconstruct --fit results/fits/ml_poisson.json
JAX_ENABLE_X64=true uv run --frozen python run.py validate-detailed --od results/validation/reconstructed_od.json
```

The `check` command plus the following stages through `benchmark` are the
admission path. A failed stage stops the
workflow; it never falls through to a later stage. Long stages emit JSON
progress events with `--json-progress`. The Slurm wrappers under `scripts/`
derive the case-study root from their own location, so they remain correct
even when `sbatch` is submitted from the parent repository.

`check`, `od-universe`, `time-discretization`, and `materialize-bins` are
lightweight and must not construct timetable-feasibility searches. `expand-od`
is the first expensive stage. It persists the approved pair-by-bin expansion,
priors, and provenance; `structural-zeros` and `prepare` consume those files
and fail on missing or incompatible fingerprints instead of recomputing them.
Review the reported complexity estimate before allowing `expand-od` to run.

`config/case.toml` contains paths, source-column mappings, explicit time
discretization limits, and references to the scientific contracts. The
referenced files are `config/time_discretization.toml`,
`config/reduced_od.toml`, `config/structural_zeros.toml`, and
`config/model.toml`; they remain user-owned. The runner writes only beneath
`results/`.
The template keeps that directory ignored; commit reports and checkpoints from
the case repository only when the case owner explicitly decides to archive
them.

`materialize-bins` writes a reviewed candidate under
`results/generated_inputs/`; it never overwrites the scenario's source
`time_bins.csv`. For a real case, re-bin the demand and explicitly promote or
repoint the reviewed file before `expand-od`, `structural-zeros` and `prepare`.
The bundled
template already uses the same one-bin contract in its committed scenario, so
the smoke run needs no manual promotion.

New cases use an independent candidate OD universe. Supply the pair-only
`inputs/od_pairs.csv` (`origin_stop_id,destination_stop_id`) or declare
`source = "network_ordered_pairs"` in `[od_universe]`. The pair universe is
validated before count-based time-bin selection; `expand-od` forms the Cartesian
product with the reviewer-approved bins and records timetable structural-zero
reasons. A time-dependent `demand.csv` remains supported only as a legacy path
and is labelled `input_semantics = "legacy_time_dependent_demand"`.

The optional `inputs/prior_demand.csv` is pair-level and contains
`origin_stop_id,destination_stop_id,prior_value`. Alternatively
`[prior_demand] source = "all_ones"` creates a neutral value for each retained
OD-time cell. This prior is not observed demand, a forecast, production, or
destination attractiveness. Those components must be declared separately in
`config/model.toml` as `provided`, `fixed`, or `estimated`, with explicit
transformation, correction scope, identifiability constraint, and
regularization.

## Ownership

| Item | Owner |
|---|---|
| `adapter.py` | copied from the public template; modified only for a documented custom hook |
| `run.py` | copied from the public template; normally unchanged |
| `config/*.toml` | case-study owner |
| `pyproject.toml` | case-study owner |
| `uv.lock` | case-study owner |
| `inputs/` | case-study owner or documented external source |
| `results/` | generated by the run |

The public template deliberately omits `uv.lock`; the case owner pins the
selected immutable public-package commit and generates the lockfile locally.

The template `pyproject.toml` explicitly declares `packages = []`: the case
repository contains configuration and data, not an installable Python package,
so `uv sync` must not attempt automatic discovery under `config/` or
`results/`. After replacing the placeholder package revision and generating
the lockfile, verify the setup with `uv lock`, `uv sync --frozen`, and
`JAX_ENABLE_X64=true uv run --frozen python run.py check` before changing any
scientific inputs.

## Configuration contract

Canonical cases use `adapter.py` unchanged. A genuinely noncanonical case may
provide a `GenericCaseHook` and pass it explicitly to `GenericCaseAdapter`; the
hook receives the validated configuration and `GenericCaseData`, must return a
new validated `GenericCaseData`, and must be recorded in the case manifest.

`config/case.toml` uses `schema_version = 1`, which identifies this generic
case-study wrapper format (the referenced reduced-OD contract has its own
schema version). All paths are resolved relative to the case root. Unknown
keys and missing required values are errors. No command in this template
expects a root-level configuration file; all contracts are under `config/`.

| Section | Required values | Valid values and meaning |
|---|---|---|
| `[case]` | `name`, ISO `service_day`, IANA `timezone`, `after_midnight_convention` | `after_midnight_convention` is `seconds_from_service_day`; the service day and timezone are explicit provenance, not inferred. |
| `[paths]` | `scenario_directory`, `measurements`, `results_directory`; optional `od_pairs`, `prior_demand`, `fixed_demand`, component inputs | New cases do not require a time-binned demand file. Files/directories must exist when their stage runs; all outputs stay below `results_directory`. |
| `[od_universe]` | `source`, `level`, `include_same_stop`, `active_service_only`, `connectivity_policy`; optional `pair_file` | `source` is `file`, `network_ordered_pairs`, or the explicitly legacy `legacy_time_dependent_demand`; `level` is `stop` or `physical_stop`; connectivity is `none` or `directed_reachable`. |
| `[prior_demand]` | `source`, `semantics`, `expansion`; `value` for `all_ones` | `source` is `all_ones`, `external_file`, or legacy; pair-level external files are independent of time bins. |
| `[observations]` | four source-column names plus `timestamp_semantics`, `missing_value_policy`, `ambiguous_event_policy` | `timestamp_semantics = "event_time"`; both policies are currently `"error"`. At least one of `trip_id_column` and `line_id_column` is required. `method_id_column` is optional when the canonical header is already `method_id`. |
| `[time_discretization]` | explicit resolution limits, positive `max_od_cells`, `horizon_start`, `horizon_end`, `candidate` | `num_od_pairs` is optional legacy metadata; for new cases it is audited from `od-universe`. `max_od_cells` is an approved complexity budget, not a default. |
| `[sampling]` | `strategy`, `samples_per_period`, `time_step_seconds` | Strategy is `uniform_midpoint`, `fixed_count`, `fixed_time_step`, or `adaptive_service_aware`; counts may be one integer or a period-keyed table. |
| `[structural_zeros]` | `configuration_file` | Points to the strict structural-zero TOML; existing positive fixed demand conflicts are errors. |
| `[model]` | `configuration_file`, optional `settings_file` | The model settings declare likelihood and explicit production/destination-attractiveness component blocks. Legacy `production_mode` remains accepted for compatibility. |

`config/time_discretization.toml` repeats the resolution, horizon, OD budget, and
candidate fields with its own `schema_version = 1`; the loader rejects a
contradiction between it and `[time_discretization]` in `config/case.toml`.
`config/reduced_od.toml` and `config/structural_zeros.toml` retain their
documented public schemas. `config/model.toml` requires `schema_version = 1`, a supported likelihood,
and explicit component declarations for new cases. Each component has
`mode = "provided" | "fixed" | "estimated"`, a baseline, correction scope,
transformation, identifiability constraint, regularization, and positive prior
scale. `provided` requires a component CSV using approved time-bin identifiers;
`fixed` records a declared baseline without adding parameters; `estimated`
adds a constrained correction block and reports its dimension. Global
production and global destination-attractiveness corrections are rejected as
confounded. Optional optimizer tolerances, iterations, dispersion, and MAP
prior scale must be finite and positive.
