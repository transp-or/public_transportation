# Walkthrough for a new OD-estimation case study

This document is an operational runbook for preparing, fitting, diagnosing,
and validating a new public-transport OD-estimation case study with
`public_transportation`. It is intended to be handed to the person conducting
the run. The case-specific repository owns the data and scientific decisions;
the public repository supplies the generic canonical-file adapter, stage
runner, routing, response, estimation, persistence, and diagnostic algorithms.

The recommended production path is the reduced-OD, fixed-routing gravity model
fitted by maximum a posteriori (MAP) estimation. Start with the smallest model
and a Poisson diagnostic fit. Introduce the negative-binomial observation model
and richer demand blocks only after the basic response has passed its checks.
The detailed time-expanded assignment is a validation backend, not something
to call inside every optimization iteration.

## 1. Non-negotiable rules

1. Never run a long fit before input audit, artifact preflight, and one warm
   objective-and-gradient benchmark have passed.
2. Missing observations are unobserved; they are not zeroes. Numerical zeroes
   present in the observation table are observed zeroes.
3. Every boarding or alighting record must resolve to exactly one timetable
   event. Never resolve an ambiguous record by arbitrary row order.
4. A positive count with no modeled response is an admission failure. Do not
   silently discard it and do not invent a path.
5. Structural-zero and otherwise frozen OD cells are not estimation
   parameters. A nonzero pre-existing fixed value that conflicts with a newly
   detected structural zero is an error.
6. Never reuse an artifact or checkpoint merely because its filename matches.
   Reuse is allowed only after fingerprint validation.
7. Keep preparation, estimation, OD reconstruction, and detailed-assignment
   validation as separate commands and separate result stages.
8. Use JAX float64 for real count data:

   ```bash
   export JAX_ENABLE_X64=true
   ```

9. Apply a strict no-guessing rule. Never silently choose time-bin edges,
   service-day or after-midnight conventions, measurement timestamp semantics,
   OD-cell budgets, transfer limits, missing-data policies, production
   semantics, package revisions, or raw/processed-data paths. If one of these
   is not specified by the case owner or source-data contract, stop and report
   the missing decision.

## 2. Case-study setup, files, and scripts

For canonical inputs, do not start by writing orchestration code. Copy
`docs/source/examples/case_study_template/` from the public repository into
the case-study repository, then edit the TOML files for the case. The intended
workflow is:

```text
user supplies input files and declarative configuration
→ generic adapter validates and fingerprints the inputs
→ generic runner executes exactly one stage per invocation
→ custom hook only if a source format cannot be represented declaratively
```

The user supplies input paths, explicit column mappings, policy values,
approved time-discretization settings, model settings, package commit, and
runtime dependencies. A custom adapter hook is justified only after the
generic adapter has shown that the source data cannot be represented by the
canonical formats; it must be isolated, named, versioned, and included in the
provenance manifest. `pyproject.toml`, `uv.lock`, and scientific configuration
remain owned by the case-study repository. A missing TOML or a missing copied
template is a case setup/documentation problem, not evidence of a package
failure.

Create the case in the private or case-specific repository. A practical layout
is:

```text
case_studies/<case_name>/
├── README.md
├── config/
│   ├── case.toml
│   ├── time_discretization.toml
│   ├── reduced_od.toml
│   ├── structural_zeros.toml
│   └── model.toml
├── inputs/
│   ├── od_pairs.csv                 # optional pair-only candidate universe
│   └── prior_demand.csv             # optional pair-level external prior
├── adapter.py
├── run.py
├── scripts/
│   ├── probe.sbatch
│   ├── 00_check.sbatch
│   ├── 05_od_universe.sbatch
│   ├── 10_time_discretization.sbatch
│   ├── 15_expand_od.sbatch
│   ├── 20_structural_zeros.sbatch
│   ├── 30_prepare.sbatch
│   ├── 40_preflight.sbatch
│   ├── 50_fit.sbatch
│   └── submit_chain.sh
└── results/                         # generated, normally not committed
    ├── audit/
    ├── artifacts/
    ├── preflight/
    ├── checkpoints/
    ├── fits/
    ├── diagnostics/
    └── validation/
```

The case-specific `run.py` should expose the following stable actions. It may
use subcommands or flags, but the actions must remain independent:

```bash
cd case_studies/<case_name>
python run.py check
python run.py od-universe
python run.py time-discretization
python run.py materialize-bins --candidate recommendation --reviewer "name"
python run.py expand-od
python run.py structural-zeros
python run.py prepare
python run.py preflight
python run.py benchmark
python run.py fit --method map --likelihood poisson
python run.py fit --method map --likelihood negative_binomial
python run.py diagnose --fit <fit-result>
python run.py reconstruct --fit <accepted-fit>
python run.py validate-detailed --od <reconstructed-od>
```

The commands above are run from the case-study root, after the template has
been copied and reviewed.

If an existing custom hook uses names such as `--check`, `--prepare`,
`--smoke-objective`, or `--diagnose-production`, map them to the corresponding
stage below. Do not combine the stages just to shorten the command list.

The complete public API sequence is illustrated in
[`examples/reduced_od_j0_integration.py`](examples/reduced_od_j0_integration.py).
The adapter should use the stable imports from
`public_transportation.preprocessing.reduced_od` and
`public_transportation.inference.reduced_od`, rather than importing private
module internals.

### 2.1 Configuration-to-stage contract

Each stage has one declared source of control. This table is part of the case
contract and should be copied into the case README:

| Stage | Configuration that controls it | Output class |
|---|---|---|
| `check` | `config/case.toml` paths, observation mappings, service-day and timestamp policies | audit/provenance only |
| `od-universe` | `config/case.toml [od_universe]` and optional `inputs/od_pairs.csv` | pair universe and exclusion audit |
| `time-discretization` | `config/case.toml [time_discretization]` and `config/time_discretization.toml` | recommendation JSON only |
| `materialize-bins` | reviewed candidate plus reviewer identity | generated input (`time_bins.csv`) |
| `expand-od` | approved bins plus the `od-universe` output | candidate OD-time cells and exclusion audit |
| `structural-zeros` | `config/structural_zeros.toml` plus expanded OD-time cells | preprocessing artifacts/audit |
| `prepare` | `config/reduced_od.toml`, sampling and input mappings | persistent reduced-OD artifacts |
| `preflight` / `benchmark` | `config/model.toml` plus prepared-artifact fingerprints | read-only diagnostics |
| `fit` | `config/model.toml` and explicit method/likelihood flags | fit result/checkpoint |
| `diagnose` / `reconstruct` / `validate-detailed` | accepted fit and validation inputs | diagnostics/validation |

The generic runner writes only beneath the configured `results_directory`,
separates audit, generated-input, artifact, fit, and validation outputs, and
records configuration and package fingerprints in each stage manifest. It
supports `--dry-run` to print a planned stage and `--json-progress` for long
stages.

### 2.2 What must exist before the first command

The following files are supplied by the case-specific repository; they are not
generated by the public package:

| Path | Owner | Commit? | Purpose |
|---|---|---:|---|
| `README.md` | case owner | yes | data provenance, assumptions, and run entry point |
| `pyproject.toml` | case owner | yes | pinned package and runtime dependencies |
| `uv.lock` | case owner | yes | resolved dependency graph and immutable public-package source |
| `config/*.toml` | case owner | yes | paths, mappings, time discretization, structural-zero, and model contracts |
| `inputs/` | case owner | source-dependent | raw or canonicalized inputs, with checksums/provenance |
| `adapter.py` | copied from public template | yes | generic canonical adapter; modify only for a documented noncanonical hook |
| `run.py` | copied from public template | yes | generic independent stage dispatcher and exit-status contract |
| `scripts/*.sbatch` | case owner | yes | optional Jed wrappers and dependent submission script |
| `results/` | run | no, normally ignored | audits, artifacts, checkpoints, fits, and validation outputs |

Copy the template and perform the structural check before editing or running a
case. Run the copy command from the public-repository root, then run the checks
from the case-study root:

```bash
CASE_ROOT=case_studies/<case_name>

cp -R docs/source/examples/case_study_template "$CASE_ROOT"
cd "$CASE_ROOT"

test -f adapter.py &&
test -f run.py &&
test -f config/case.toml &&
test -f config/time_discretization.toml &&
test -f config/reduced_od.toml &&
test -f config/structural_zeros.toml &&
test -f config/model.toml &&
test -f pyproject.toml
```

The copied TOML files are examples, not scientific defaults. The case owner
must review and edit every value, confirm the input semantics, and document
the resulting assumptions; copying the template does not make a case
scientifically ready. The template intentionally does not provide a
case-specific `uv.lock`: the case owner must pin the selected immutable public
package commit in `pyproject.toml`, run `uv lock`, and commit the generated
lockfile. A missing path in the check is a case-setup/documentation failure and
must be fixed before any package or estimation diagnosis.

The public repository supplies the generic adapter and runner as a complete
reference template. If any of the TOML files, `adapter.py`, or `run.py` is
absent from a copied case, classify the situation as an incomplete case-study
setup or documentation gap and stop. It is not evidence of a public-package
defect. A first-time user with canonical inputs should only need to edit the
configuration and supply the files; no new orchestration code is expected.

For a new case, start by copying the structure and adapting the complete
public examples:

- [Geneva example README](examples/geneva_gtfs/README.md),
  [preprocessing adapter](examples/geneva_gtfs/pre_processing/run_preprocessing.py),
  [structural-zero runner](examples/geneva_gtfs/pre_processing/run_structural_zeros.py),
  and [estimation runner](examples/geneva_gtfs/estimation/run_estimation.py);
- [simple example 01 README](examples/simple_example_01/README.md) for the
  smaller file-oriented workflow;
- [public reduced-OD integration example](examples/reduced_od_j0_integration.py)
  for artifact preparation, preflight, objective benchmarking, estimation, and
  reconstruction.

These examples are templates, not files to import blindly: replace their
paths, service-day assumptions, production semantics, and measurement policy
with decisions documented for the new case.

### 2.3 Minimal case-project dependency file

The case project must have its own `pyproject.toml`. The following is the
minimal public-package dependency declaration; add case-specific dependencies
only when they are actually used:

```toml
[project]
name = "case-study-<case-name>"
version = "0.1.0"
requires-python = ">=3.14,<3.15"
dependencies = [
  "public_transportation @ git+https://github.com/transp-or/public_transportation.git@<commit>",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

The `<commit>` placeholder must be replaced with a real immutable commit hash
selected by the case owner. Run `uv lock`, inspect the resulting `uv.lock`, and
commit both files. A sibling editable checkout is not an acceptable substitute
for this dependency in a reproducible run.

### 2.4 Minimum generic-adapter and dispatcher responsibilities

The copied generic `adapter.py` reads the canonical scenario and measurement
formats and is the only place that should know the case’s source-table columns,
platform mapping, service calendar, and external production inputs. For a
canonical case it is used unchanged. At minimum it must:

1. locate the declared `inputs/` files and fail if a required file is missing;
2. validate CSV headers, types, units, keys, duplicates, and checksums;
3. construct `Scenario.from_folder(..., strict=True)`;
4. load a `MeasurementTable` with `read_measurements_csv` or explicitly map
   case columns to the canonical fields `method_id`, `measurement_type`,
   `stop_id`, `time`, `value`, and `trip_id`/`line_id`;
5. resolve every retained measurement to exactly one timetable event;
6. construct `ReducedODPreparationInputs` and the declared time periods;
7. return audit data and fingerprints without silently modifying source files.

A minimal loading function has this shape:

```python
from pathlib import Path

from public_transportation.domain import Scenario
from public_transportation.measurement import read_measurements_csv
from public_transportation.preprocessing.reduced_od import load_reduced_od_config


def load_case(case_root: Path):
    inputs = case_root / "inputs"
    scenario = Scenario.from_folder(inputs, strict=True)
    measurements = read_measurements_csv(
        case_root / "results/audit/measurements_boarding_alighting.csv"
    )
    report = scenario.validate()
    errors = [item for item in report.issues if item.severity.name == "ERROR"]
    if errors:
        raise ValueError(f"scenario validation failed: {errors!r}")
    return scenario, measurements


def load_configuration(case_root: Path):
    from public_transportation.case_study import load_case_study_config

    return load_case_study_config(case_root / "config/case.toml", case_root=case_root)
```

The generic implementation already performs source-column mapping, event
identity resolution, service-day checks, checksums, audit-file writing, and
stage dispatch. A custom hook may add a genuinely noncanonical transformation
such as platform-to-stop mapping, but it must fail explicitly when its input
contract is not met and must be recorded in the manifest. `run.py` must load
the TOML, call the adapter, dispatch exactly one independent stage, write
JSON/CSV outputs, and return a nonzero process status on any validation
failure. It must not catch an exception and continue into estimation.

The public package also provides the early period validator
`preflight_reduced_od_time_periods`. The case adapter should call it during
`check`, before constructing a timetable index or running RAPTOR:

```python
import json

from public_transportation.preprocessing.reduced_od import (
    DepartureTimeSamplingConfig,
    preflight_reduced_od_time_periods,
)

period_report = preflight_reduced_od_time_periods(
    time_periods,
    relevant_event_seconds=relevant_journey_event_seconds,
    sampling_config=DepartureTimeSamplingConfig(
        strategy="fixed_count",
        samples_per_period={"morning": 12, "peak": 24},
    ),
)
(results_directory / "audit/time_period_preflight.json").write_text(
    json.dumps(period_report.to_dict(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if not period_report.valid:
    raise RuntimeError("time-period preflight failed")
```

`relevant_journey_event_seconds` must be supplied by the case adapter from
the event universe that the reduced response is expected to represent. The
validator is deliberately agnostic about whether production inputs are rates
or interval totals; the adapter must record and review that scientific
exposure convention separately. The preflight is an admission check, not a
repair step: periods must be sorted and non-overlapping, and every relevant
event time must fall in exactly one named period. An event in a gap is an
explicit failure. Sampling is also period-specific: twelve samples in a
30-minute period do not have the same temporal resolution as twelve samples in
a three-hour period, so use period-specific counts, a fixed time step, or the
documented adaptive strategy. Finally, demand and production values are
interpreted per named interval; the package does not rescale them merely
because interval lengths differ. Record the intended rate-versus-interval-total
meaning in the case configuration and audit.

### 2.5 Independent OD universe, expansion, and prior contract

New case studies must not use a time-binned demand table as the definition of
the candidate universe. Choose exactly one pair-universe source:

1. `inputs/od_pairs.csv`, with only
   `origin_stop_id,destination_stop_id`; or
2. `source = "network_ordered_pairs"` in `[od_universe]`, with its spatial
   level, active-service rule, same-node rule, and directed-connectivity policy
   declared explicitly.

The pair file is independent of time bins and must be duplicate-free. The
declared level must say whether identifiers are scenario stops, platforms, or
physical stops. Same platform, same physical stop, same station, and identical
origin/destination identifiers are not interchangeable concepts. Network
generation distinguishes all stops, stops with departures, stops with arrivals,
static directed reachability, and timetable-feasible cells; static graph
reachability never implies timetable feasibility.

The user must decide and record:

- stop, platform, or physical-stop level;
- whether same-node pairs are included;
- whether inactive stops are included;
- whether static directed connectivity is required;
- maximum transfers, initial wait, and journey time;
- whether feasibility is evaluated independently for every approved time bin;
- whether the prior is neutral or externally informed;
- whether production is provided, fixed, or estimated;
- whether destination attractiveness is provided, fixed, or estimated;
- correction scopes, transformations, constraints, and regularization.

The stages are deliberately separate:

```text
od-universe
→ time-discretization
→ review recommendation
→ materialize-bins
→ expand-od (pair universe × approved bins)
→ structural-zeros (timetable feasibility)
→ prepare
```

`od-universe` writes `results/audit/od_universe.json`,
`results/audit/od_pairs.csv`, and `results/audit/od_universe_exclusions.csv`.
`expand-od` writes the candidate cells and
`results/audit/od_time_expansion.json` plus its exclusion audit. A physically
reachable pair can therefore produce a timetable structural zero in one bin
and a retained cell in another. Every exclusion reason is preserved.

An optional pair-level `inputs/prior_demand.csv` has columns
`origin_stop_id,destination_stop_id,prior_value`. Alternatively, declare
`[prior_demand] source = "all_ones"`, which generates one value per retained
OD-time cell only after expansion. A prior or `all_ones` seed defines candidate
support or a numerical baseline. It is not automatically an observed demand
matrix, production total, destination-attractiveness measure, or passenger
forecast.

Production totals and destination attractiveness are estimated only when the
model specification declares corresponding parameter blocks. They must never
be inferred implicitly from the candidate OD universe or from an `all_ones`
prior. `mode = "provided"` requires a matching input file; `mode = "fixed"`
requires declared, fingerprinted values; and `mode = "estimated"` adds named
correction parameters with explicit scopes, constraints, regularization, and
identifiability diagnostics. Empty groups, destinations with no retained
cells, unconstrained blocks, weak measurement support, and singular directions
are validation failures or explicit warnings—not evidence that the parameters
are estimable.

The generated provenance files are:

```text
results/generated_inputs/prior_demand.csv
results/audit/prior_generation.json
results/audit/production_attractiveness_provenance.json
```

They record source checksums, package and configuration fingerprints, spatial
and connectivity policies, approved-bin fingerprint, prior-generator identity,
and production/attractiveness modes. Source inputs are never overwritten.

## 3. Before the first run

### 3.1 Install and pin the public package

Treat `public_transportation` like any other Python dependency. A sibling
checkout is not required. Until a suitable release is available from PyPI, pin
an immutable GitHub commit in the case-study project:

```bash
uv add "public_transportation @ git+https://github.com/transp-or/public_transportation.git@<commit>"
uv lock
uv sync --frozen
```

Once the package is published on PyPI, pin the selected release instead:

```bash
uv add "public_transportation==<version>"
uv lock
uv sync --frozen
```

Commit `pyproject.toml` and `uv.lock` in the case-study repository. The lockfile
is the deployment contract: it records the resolved source and, for a VCS
dependency, the exact commit. An editable local checkout is useful during joint
library development, but it is an explicit development mode and must not be
the default assumed by this runbook.

### 3.2 Freeze provenance

Record the following in `results/audit/run_manifest.json`:

- case name and creation timestamp;
- private/case-study repository revision and `git status --short`;
- adapter and stage-dispatcher revision (commit hash, or file checksums when
  they are generated outside version control);
- installed `public_transportation` distribution version;
- resolved package source: PyPI release or GitHub URL and commit, as recorded in
  `uv.lock`;
- checksum of `uv.lock`;
- Python, NumPy, SciPy, JAX, and JAXlib versions;
- JAX backend, devices, and x64 status;
- host name, logical CPU count, and available memory;
- checksums or immutable source identifiers for timetable, observations,
  fixed demand, stop mapping, and external covariates;
- service day, analysis interval, and after-midnight convention.

Run this import check in the same environment that will run the case:

```bash
uv run --frozen python - <<'PY'
from importlib.metadata import distribution
import platform

import jax
import public_transportation

installed = distribution("public_transportation")
print("distribution:", installed.metadata["Name"])
print("distribution_version:", installed.version)
print("package_version:", public_transportation.__version__)
print("python_version:", platform.python_version())
print("backend:", jax.default_backend())
print("devices:", jax.devices())
print("x64:", jax.config.x64_enabled)
PY

# The VCS source and immutable revision must be visible in the lockfile.
rg -n -A40 -B2 'name = "public-transportation"' uv.lock
```

Stop if the installed version differs from the version resolved in `uv.lock`,
if the lockfile does not identify the intended release/commit, or if the
`public-transportation` lock entry has `source = { editable = "." }` rather
than a PyPI source or a Git source with a full immutable `rev`. Inspecting
`public_transportation.__file__` can be useful when troubleshooting an
accidental editable install, but its absolute installation path is not part of
the scientific identity and need not be recorded.

### 3.3 Validate the TOML configuration

The generic case-study configuration is `config/case.toml`; it contains paths,
source-column mappings, and references to the other contracts. Its first
top-level entry must be:

```toml
schema_version = 1
```

The reduced-OD contract is a separate file, `config/reduced_od.toml`. Its first
top-level entry must be:

```toml
schema_version = 2
```

Here, `schema` means the documented structure of the configuration file: the
section names, parameter names, data types, required fields, and allowed
values. Despite its TOML syntax, `schema_version` is not a user-selectable
parameter or a modeling option. It is a mandatory configuration-format marker.
The user copies the supported value into the file and does not tune it or
increment it for a new case or run.

The number `2` tells the installed package which configuration format it is
reading. It is not the version of the case study, the routing algorithm, the
gravity model, or the input data. The package developers change this number
only when the configuration format changes incompatibly.

For the currently implemented reduced-OD configuration format, write exactly
`schema_version = 2`; it is presently the only accepted value. The marker is
still useful because configuration files and case studies may outlive the
current package release. If a later package introduces format 3, an archived
file marked as format 2 can be interpreted by version-2 rules, migrated
explicitly, or rejected with a clear compatibility error. Without the marker,
the parser would have to guess which meanings apply. For the same reason, a
file declaring an unsupported version is rejected rather than silently
reinterpreted.

The rest of the file is divided into named TOML sections. For example:

```toml
schema_version = 2

[observations]
service_day = "2026-01-15"
# ...the remaining observation settings...

[journeys]
maximum_transfers = 2
# ...the remaining journey settings...

[model]
likelihood = "negative_binomial"
```

This abbreviated fragment only illustrates the syntax; it is not a complete
configuration. The complete TOML must explicitly state:

- service day, analysis start/end seconds, and extended-service-day policy;
- accepted observation types and missing/duplicate/ambiguity policies;
- APC cleaning, coverage, and outage policy identifiers;
- journey origin and destination semantics;
- half-open OD-time bins;
- maximum transfers, waiting time, journey time, and alternatives per cell;
- footpath and physical-stop mapping policies;
- route-share policy (`fixed_within_fit` for fixed routing);
- production mode and its scientific meaning;
- likelihood and output spatial level;
- detailed assignment as `explicit_only`.

Unknown or missing keys must fail. Do not weaken the parser to accept an
incomplete file. Use the complete synthetic configuration in
[`examples/reduced_od_j0_integration.py`](examples/reduced_od_j0_integration.py)
as a template, then replace every synthetic policy identifier with the actual
case policy. Validate the completed file by loading it before any expensive
work. Run this snippet from the case-study root (`cd case_studies/<case_name>`):

```bash
cd case_studies/<case_name>
uv run --frozen python - <<'PY'
from public_transportation.preprocessing.reduced_od import load_reduced_od_config

configuration = load_reduced_od_config("config/reduced_od.toml")
print("schema_version:", configuration.schema_version)
print("configuration_fingerprint:", configuration.fingerprint)
PY
```

Successful loading proves that the reduced-OD file follows schema version 2.
Load the complete generic case configuration as well:

```bash
cd case_studies/<case_name>
uv run --frozen python - <<'PY'
from public_transportation.case_study import load_case_study_config

configuration = load_case_study_config(
    "config/case.toml",
    case_root=".",
)
print("configuration_fingerprint:", configuration.fingerprint)
PY
```

Successful loading proves that the files follow their documented schemas. It does not
prove that the chosen assumptions are scientifically appropriate; those still
require the reviews described below.

### 3.4 Prepare, but do not silently clean, the source tables

The adapter must produce immutable in-memory `Scenario` and
`MeasurementTable` objects. Perform cleaning before constructing the immutable
measurement table and report every exclusion by reason.

Check at least:

- unique stop, line, trip, time-bin, and stop-time keys;
- strictly ordered stop sequences and valid arrival/departure times;
- active service on the selected service day;
- correct handling of times greater than 24 hours;
- complete platform-to-physical-stop mapping;
- explicit and plausible transfer footpaths;
- finite nonnegative measurement values;
- uniqueness of the atomic measurement identity;
- explicit distinction between an absent record and a record with value zero;
- sensor coverage and outage exclusions made before likelihood construction;
- units (passenger counts, not rates or cumulative counters);
- consistency between count timestamps and timetable event timestamps;
- complete OD candidate universe and unique fixed-demand keys.

The audit output should include totals and histograms by observation type,
line, stop, period, and vehicle journey; the number of zero observations; the
number excluded by each documented rule; timetable and demand dimensions; and
the selected production semantics.

### 3.5 Mapping an existing medium case

An existing private repository may have a layout such as:

```text
data/data_medium/
processed_data_for_models/medium_network/
data_process_codes/medium_network/
```

Do not treat these directories as if they were already the generic case-study
project. Map them explicitly:

| Existing location | New case-study role |
|---|---|
| `data/data_medium/` | configured raw-input directory (`paths.scenario_directory`, `paths.measurements`, and demand paths) |
| `processed_data_for_models/medium_network/` | legacy comparison directory only; never an implicit active artifact source |
| `data_process_codes/medium_network/` | no longer required for canonical inputs; source material for a named custom hook only if needed |
| copied `docs/source/examples/case_study_template/` | `case_studies/medium_network/adapter.py`, `run.py`, `config/`, and Jed wrappers |
| `case_studies/medium_network/README.md` | provenance, mappings, assumptions, reviewer decisions, and exact commands |

The preferred workflow is to create a new `case_studies/medium_network/`
directory by copying the public template, edit its declarative configuration,
and reference raw data only through documented paths with checksums. Existing
processed artifacts may be retained under a clearly named legacy directory for
comparison, but they must not be fed into the new workflow as if they were
newly generated artifacts. Existing private scripts are consulted only when a
canonical mapping cannot express a source transformation; such a hook is a
data-adapter limitation, not a prerequisite for ordinary cases.

For the medium network, the canonical sequence is therefore: generate or
validate a pair-only OD universe from the network, run the count-based
time-discretization diagnostic, obtain reviewer approval, expand the pairs over
the approved bins, and apply timetable structural-zero rules. Use `all_ones`
only when it is explicitly declared as the neutral prior. The historical
coarse demand bins are not copied into newly selected bins; if they are used at
all, the manifest must label the run
`input_semantics = "legacy_time_dependent_demand"` and record the scientific
re-binning rule. Production and destination-attractiveness corrections belong
in the model component tables, never in an inferred sum of the prior.

The public hook contract is `GenericCaseHook`: it receives the validated
`CaseStudyConfig` and `GenericCaseData`, returns a new `GenericCaseData`, and is
passed explicitly to `GenericCaseAdapter`/`GenericCaseRunner`. Do not hide a
hook behind an import-time side effect.

When this walkthrough requires regeneration from source data, discard the
legacy time bins, fixed-demand files, structural-zero outputs, routing caches,
response matrices, and model checkpoints from the active output directory.
Regenerate them with the selected public-package revision and new configuration
fingerprints. Reuse is allowed only when the public loader proves exact
fingerprint compatibility; a matching filename is not evidence of compatibility.

If the private repository cannot identify which raw file generated an existing
processed artifact, stop and report the missing provenance. Do not guess, and
do not silently mix raw data from one case revision with processed data from
another.

### 3.6 Choose a data-driven time discretization (mandatory when timestamps exist)

If the source measurements retain event timestamps, running the standalone time
discretization diagnostic is mandatory. It must happen after the raw-table
audit and before structural-zero or reduced-journey preparation. An existing
`time_bins.csv` is only a provisional input until this diagnostic has been run
and a reviewer has recorded the adopted candidate. It must not automatically
be accepted as final merely because it is present in the repository.

The required workflow is:

```text
raw count audit
→ validate/generate OD universe
→ time-discretization diagnostic (without changing the OD universe)
→ review candidate bins
→ materialize time_bins.csv
→ expand OD pairs across approved bins
→ structural-zero preprocessing
→ reduced-journey and response preparation
→ estimation
```

If timestamps have already been irreversibly aggregated into wide intervals,
the diagnostic cannot recover the lost temporal information. Stop and request
clarification of the intended time bins, event-time convention, and complexity
budget; do not silently treat the existing bins as validated.

Before running the diagnostic, obtain the following values from the audited
inputs and record them in the run manifest:

1. `num_od_pairs`: the number of distinct ordered pairs in the independent
   `od_pairs.csv` file or the network-derived universe, after applying the
   declared spatial level and pair-level rules. Do not count time bins here;
   this value must not change when the temporal resolution changes;
2. `max_od_cells`: an approved upper bound on
   `num_od_pairs × number_of_time_bins`, chosen by the case owner from the
   memory and estimation budget. There is no scientifically safe package
   default. If the owner has not approved this budget, stop;
3. the analysis horizon, expressed in service-day seconds, from the source
   service-day contract and validated against the minimum/maximum accepted
   count timestamps. Do not guess `05:00–25:00`;
4. the measurement column mapping. The canonical CSV requires
   `method_id`, `measurement_type`, `stop_id`, `time`, `value`, and at least one
   of `trip_id` or `line_id`;
5. the event-time convention: timezone, service-day origin, after-midnight
   representation, and whether the timestamp is a boarding departure,
   alighting arrival, or another event;
6. the candidate-bin selection rule and reviewer. The diagnostic recommends a
   candidate; it does not make the scientific decision automatically.

For a pair universe stored as `od_pairs.csv`, this command computes
`num_od_pairs` without reading any time-bin membership:

```bash
uv run --frozen python - <<'PY'
import csv
from pathlib import Path

path = Path("case_studies/<case_name>/inputs/od_pairs.csv")
with path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
pairs = {(row["origin_stop_id"], row["destination_stop_id"]) for row in rows}
print("num_od_pairs:", len(pairs))
PY
```

If the universe is network-derived, `run.py od-universe` is the authoritative
count and audit. Once the owner has selected an approved maximum number of
bins, record `max_od_cells = num_od_pairs * approved_max_bins`; do not use a
larger limit merely to make the diagnostic pass. Never replicate a legacy
demand table across newly selected bins without an explicit scientific rule.

For canonical timestamped measurements, this command reports the observed
event-time range and the column-level measurement types. It does not infer the
service-day convention:

```bash
uv run --frozen python - <<'PY'
from public_transportation.measurement import read_measurements_csv

table = read_measurements_csv(
    "case_studies/<case_name>/results/audit/measurements_boarding_alighting.csv"
)
seconds = [record.time.seconds_from_midnight for record in table.records]
print("measurement_rows:", len(table.records))
print("event_time_min_seconds:", min(seconds))
print("event_time_max_seconds:", max(seconds))
print("measurement_types:", sorted({record.measurement_type.value for record in table.records}))
PY

shasum -a 256 \
  docs/source/examples/geneva_gtfs/pre_processing/results/measurements_boarding_alighting.csv
```

If any required value is unavailable or contradictory, stop and request a
case-owner decision. The diagnostic must not choose a horizon, OD count,
measurement mapping, or event convention silently.

Run it with a small base resolution and explicit complexity limits. Adapt the
measurement path and horizon to the case; the horizon is especially important
when the source file contains only a subset of the service day:

```bash
uv run --frozen python -m public_transportation.preprocessing.time_discretization \
  --measurements case_studies/<case_name>/results/audit/measurements_boarding_alighting.csv \
  --base-resolution-minutes 5 \
  --min-bin-minutes 10 \
  --max-bin-minutes 60 \
  --max-bins 24 \
  --num-od-pairs <audited_od_pair_count> \
  --max-od-cells <approved_od_cell_budget> \
  --horizon-start 05:00 \
  --horizon-end 25:00 \
  --output-json case_studies/<case_name>/results/audit/time_discretization_recommendation.json
```

The command writes JSON only. Review the JSON before choosing a production
discretization, in particular:

- `profile` and `peak_intervals`, to confirm that the detected peaks are
  plausible and are not caused by an outage, duplicated records, or a partial
  observation window;
- every entry in `candidates`, including `within_bin_deviance`, bin count,
  estimated OD-cell count, validity, and warnings;
- `recommendation.time_bins`, its `estimated_od_cells`, and the warnings
  attached to the recommended candidate.

The OD-cell limit is an operational guard, not part of the scientific identity
of the data. If it rejects all candidates, increase the limit only after
checking the available memory and routing budget; do not hide the warning by
dropping observations. If the measurements have no raw event timestamp, this
diagnostic cannot recover finer peaks and the existing timetable/time-bin
definition must be reviewed manually instead. When the source contains several
days, compare recommendations across representative or held-out days before
fixing a common production horizon.

Adopt a recommendation deliberately: check that the returned bins are
half-open, cover the intended horizon, have strictly increasing edges, and do
not create empty or unsupported periods unless that is an explicit modelling
choice. Materialize the reviewed candidate explicitly; the materializer
validates the report schema, candidate validity, unique IDs, contiguous
half-open edges, and writes the CSV atomically:

```bash
uv run --frozen python -m public_transportation.preprocessing.materialize_time_bins \
  --recommendation-json case_studies/<case_name>/results/audit/time_discretization_recommendation.json \
  --output case_studies/<case_name>/inputs/time_bins.csv
```

This is the only step that turns the diagnostic JSON into a model input. The
resulting CSV is what the case adapter and scenario loader consume; the JSON
remains a review and provenance artifact.

Use `--candidate peak_adaptive` (or another candidate name in the report) when
you have deliberately selected a candidate other than the report's default
recommendation. An existing `time_bins.csv` is never replaced unless
`--overwrite` is supplied explicitly after review. Record the JSON report path
and its fingerprint in the run manifest, then re-run Stage 1 after this
change. Do not feed the diagnostic JSON directly to estimation and do not let
the preparation script overwrite an existing `time_bins.csv` automatically.

The handoff to the rest of the pipeline is therefore an ordinary scenario
input, not a second report format. The adopted file must contain the columns
expected by the scenario loader:

```text
bin_id,start_s,end_s
T00,18000,21600
T01,21600,22200
```

Here `start_s` and `end_s` are seconds from midnight (the loader also accepts
`HH:MM` or `HH:MM:SS`). In the new workflow this file defines only the approved
time intervals: the pair-only OD universe is already fixed and is not edited
when bin edges change. `expand-od` forms pair × bin cells, fingerprints that
expansion, and reruns timetable feasibility for each cell. Existing
time-dependent demand files are accepted only through the explicitly labelled
legacy compatibility path; a new case must not replicate their rows into new
bins implicitly. Once expansion and structural-zero stages complete, the
estimation and validation commands consume the resulting `Scenario` and
prepared artifacts as usual. The diagnostic JSON is retained only as
provenance and review evidence.

See [`time_discretization.md`](time_discretization.md) for the report schema,
candidate scoring details, and the limitations of interpreting event-time
peaks as departure-time demand bins. The diagnostic JSON's own report-schema
version is separate from the reduced-OD TOML `schema_version = 2`; neither
value is a tunable modelling parameter.

## First-run decision tree

Do not skip ahead because a later file happens to exist. On a new case, run
these stages in order. Except where a repository root is stated explicitly,
steps 2–11 are run from the case-study root (`cd "$CASE_ROOT"`).
Run step 1 from the case-study repository root, then set `CASE_ROOT` and enter
the copied case directory before step 2.

| Order | Command | Location | May modify files? | Expected result / stop condition |
|---:|---|---|---|---|
| 1 | `git rev-parse HEAD`; `uv lock --check`; package provenance check | local | no | Correct case revision and locked public package; stop on mismatch. |
| 2 | `test -f adapter.py &&`<br>`test -f run.py &&`<br>`test -f config/case.toml &&`<br>`test -f config/time_discretization.toml &&`<br>`test -f config/reduced_od.toml &&`<br>`test -f config/structural_zeros.toml &&`<br>`test -f config/model.toml` | local, from case-study root | no | Complete case setup; a missing path is a case-setup/documentation failure and identifies the missing path. |
| 3 | `uv run --frozen python run.py check` | local | audit JSON and rejected-row files only | All source schemas and identities valid; stop on unexplained rows. |
| 4 | `uv run --frozen python run.py od-universe` | local | pair universe and exclusion audit | Pair source, spatial level, and pair-level exclusions are reviewed. |
| 5 | `uv run --frozen python run.py time-discretization` | local | recommendation JSON only | Candidate bins, horizon, and budget are explicit; stop if any input is unknown. |
| 6 | `uv run --frozen python run.py materialize-bins --candidate recommendation --reviewer "<name>"` | local | reviewed `time_bins.csv` | Reviewer adopts one candidate; stop if the bin contract is not approved. |
| 7 | `uv run --frozen python run.py expand-od` | local | OD-time cells and exclusion audit | Pair universe is expanded over approved bins; every exclusion has a reason. |
| 8 | `uv run --frozen python run.py structural-zeros` | local first, Jed if large | structural-zero artifacts | Timetable feasibility and reason counts reviewed; stop on positive fixed conflict. |
| 9 | `uv run --frozen python run.py prepare` | local first, Jed if large | persistent phase artifacts | All phases complete or compatibly reused; stop on missing/invalid phase. |
| 10 | `uv run --frozen python run.py preflight` | local or Jed | preflight JSON only | Read-only artifacts, dimensions, dtype, and fingerprints agree. |
| 11 | `uv run --frozen python run.py benchmark` | local or Jed | benchmark JSON only | Finite warm value/gradient and no value-change recompilation. |
| 12 | `sbatch scripts/submit_chain.sh` | Jed only | checkpoints, fits, logs | Submit estimation only after stages 1–11 pass. |

If step 2 returns a nonzero status, identify the missing path explicitly before
doing anything else:

```bash
for required_path in \
  adapter.py run.py config/case.toml config/time_discretization.toml \
  config/reduced_od.toml config/structural_zeros.toml config/model.toml
do
  if [[ ! -f "$required_path" ]]; then
    printf 'CASE-SETUP/DOCUMENTATION FAILURE: missing %s\n' "$required_path" >&2
    exit 1
  fi
done
```

Stages 1–11 are admission checks, not optional progress bars. A stage that
fails must be diagnosed and recorded before any downstream stage is attempted.
The repository, setup, pair-universe, and time-discretization checks should be
completed locally; preparation, benchmarking, and estimation can move to Jed
once the local smoke run fits the resource budget.

## 4. Stage 1 — Input check

Run locally first:

```bash
cd case_studies/<case_name>
JAX_ENABLE_X64=true uv run --frozen python \
  run.py check
```

This stage must not construct expensive routing artifacts. It should:

1. verify the environment, case revision, and locked dependency identity;
2. load and validate the TOML;
3. stream or load all source tables;
4. audit service-day and time conventions;
5. verify stop mapping and observation identities;
6. report candidate OD-time, fixed, structural-zero candidate, measurement,
   trip, stop-time, and physical-stop counts;
7. run `preflight_reduced_od_time_periods` and write its JSON report;
8. estimate memory/disk needs where possible;
9. write machine-readable audit JSON and any rejected-record CSV.

Accept the stage only if all counts are explainable, every rejection has a
documented policy reason, the package identity and case revision are correct,
and there are
no unresolved event identities. Treat unexpected zero rows, empty periods,
missing lines, or orders-of-magnitude changes from the source statistics as
errors until investigated.

## 5. Stage 2 — Candidate OD universe

Run this stage before choosing time bins:

```bash
cd case_studies/<case_name>
JAX_ENABLE_X64=true uv run --frozen python run.py od-universe
```

The stage validates `inputs/od_pairs.csv` when the case declares
`source = "file"`, or deterministically generates the declared network-derived
universe otherwise. The pair file contains only ordered pairs—never a time-bin
column or a numerical flow. The audit records the declared stop/platform or
physical-stop level, same-node policy, active-service policy, directed
connectivity policy, every excluded pair, and the pair-universe fingerprint.

`od-universe` does not inspect count timestamps and does not change when the
approved time-bin edges change. A static directed path is only a cheap
pair-level filter; it is not evidence that a scheduled journey exists in every
time interval.

## 6. Stage 3 — Time-bin approval and OD-time expansion

Run `time-discretization`, review its recommendation, and materialize the
chosen bins. Then expand the immutable pair universe over those approved bins:

```bash
cd case_studies/<case_name>
JAX_ENABLE_X64=true uv run --frozen python run.py time-discretization
JAX_ENABLE_X64=true uv run --frozen python run.py materialize-bins \
  --candidate recommendation --reviewer "<name>"
JAX_ENABLE_X64=true uv run --frozen python run.py expand-od
```

`expand-od` applies the declared active-origin, active-destination, directed
connectivity, initial-wait, journey-time, and maximum-transfer rules once for
each pair/bin combination. It writes
`results/audit/od_time_expansion.json` and
`results/audit/od_time_exclusions.csv`; a physically reachable pair may still
be a timetable structural zero in one bin and retained in another. The
generated `prior_demand.csv` is created only after this expansion and contains
one value per retained OD-time cell.

## 7. Stage 4 — Structural zeroes and feasibility

Structural-zero preprocessing and reduced-journey preparation solve related
but distinct problems:

- structural-zero preprocessing decides which OD-time cells must not be
  parameters;
- reduced-journey preparation constructs timetable-feasible journey
  alternatives and their count responses for the retained cells.

For the separate TOML-driven structural-zero tool, follow
[`structural_zero_preprocessing.md`](structural_zero_preprocessing.md). Enable
the same-stop, no-feasible-path, and maximum-transfer rules at minimum. The
assignment feasibility settings must agree with the later routing assumptions.

Review these generated files before estimation:

- `fixed_demand.csv`;
- `structural_zero_audit.csv`;
- `structural_zero_summary.json`;
- `resolved_config.toml`;
- `fingerprints.json`.

Verify that reason counts add up, all fixed values are finite and nonnegative,
and a frozen-zero cell has been removed from the parameter vector. A
topologically feasible cell may still be fixed to zero for externally justified
case-study reasons; label that decision separately from a structural zero.

## 8. Stage 5 — Prepare reduced artifacts

Run:

```bash
cd case_studies/<case_name>
JAX_ENABLE_X64=true uv run --frozen python \
  run.py prepare
```

The adapter should call `prepare_reduced_od_artifacts` with an explicit
`ReducedODPreparationInputs` and a persistent output directory. The first run
uses `cache_policy="reuse_or_build"`. Use `"read_only"` in estimation jobs.
Use `"rebuild"` only after a deliberate input or policy change.

The artifact dependency chain is:

```text
configuration
  -> physical_stops
  -> service_periods_route_patterns
  -> timetable_index
  -> journey_choices
       -> measurement_response -> response_equivalence
       -> production_inputs
       -> destination_attractiveness
  -> conditional_gravity_features
  -> reduced_response_operator
  -> problem_manifest
```

Every phase directory contains a `manifest.json` and, where needed, immutable
NumPy arrays. Inspect the preparation report and verify:

- every phase is `built` or compatibly `reused`;
- all upstream, configuration, and content fingerprints are present;
- dimensions, dtypes, and array checksums pass loading;
- the number of free cells excludes all frozen cells;
- all configured origins, periods, and departure samples appear;
- route searches completed and no unexplained origin/period has zero journeys;
- journey alternatives obey transfer, waiting, duration, footpath, and service
  constraints;
- measurement rows and response rows use the same immutable identity;
- the response operator is finite and has the expected shape;
- production and attraction inputs have the declared semantics.

Do not infer that one departure sample represents a broad period. If counts are
vehicle-journey specific, assess departure-sampling convergence before trusting
the response. Increase samples or use the documented adaptive integration
method until the retained measurement response is stable enough for the case.

## 9. Observations that do not correspond to a path

Use the following decision tree. Save every affected atomic record and its
reason to a CSV; summary counts alone are insufficient.

### 7.1 The observation does not match a timetable event

Examples are an unknown trip, platform, or stop; an inactive service-day trip;
the wrong after-midnight convention; a boarding recorded at an arrival rather
than departure time; or a line-only identity matching several trips.

This is an input-resolution error. Check, in order:

1. service day and calendar exceptions;
2. timezone and seconds-from-service-day conversion;
3. times after midnight (`25:10`, for example);
4. trip and line identifier translation;
5. platform-to-physical-stop mapping;
6. arrival-versus-departure event semantics;
7. allowed, explicitly documented timestamp tolerance;
8. duplicate or ambiguous timetable records.

Correct the adapter or source policy and rebuild affected artifacts. Do not
attach the count to the nearest event merely because it is nearby. Exclusion is
allowed only when an independent sensor-quality, coverage, or outage policy
justifies it; record that exclusion before model construction.

### 7.2 The event exists, but no retained journey contributes to it

The observation identity is valid, but its measurement-response row is empty.
For a positive count, the model then predicts zero for every parameter value,
so estimation cannot repair the problem.

Investigate:

1. whether the event lies inside the analysis and OD departure windows;
2. whether all relevant origins and destinations are in the candidate universe;
3. departure sampling within each period;
4. maximum initial wait and journey duration;
5. maximum transfers;
6. missing or incorrect transfer footpaths;
7. physical-stop aggregation;
8. maximum alternatives and journey-pruning rules;
9. whether fixed route shares assigned zero mass to every contributing journey;
10. whether the count is actually a load or another unsupported observation
    type mislabeled as a boarding/alighting.

Change one justified assumption at a time, create a new configuration
fingerprint, rebuild the affected phases, and compare response coverage. Never
inject an artificial journey. A present zero count with an empty response does
not contradict the model, but it is also uninformative; report it separately
rather than claiming it validates routing.

### 7.3 An OD-time pair has no feasible journey

This is an OD support decision, not an observation-resolution decision. The
RAPTOR result distinguishes:

- no topological path;
- minimum boardings exceeding the transfer limit;
- a topological path but no timetable-feasible journey at the query time.

If the classification is correct, freeze the cell to zero and remove it from
the parameter layout. If external fixed demand is positive for that cell, stop:
the only supported conflict policy is `error`. Investigate stop mapping,
footpaths, time conventions, and the external value; do not overwrite it.

### 7.4 A path exists but the fit is systematically poor

This is neither an unmatched record nor a structural zero. Examine residuals
by line, stop, period, direction, and vehicle journey. Likely causes include
fixed route shares, inadequate departure sampling, crowding/capacity effects
not present in the reduced model, incomplete production structure, inconsistent
count coverage, or overdispersion. Address these through model diagnostics and
explicit child models, not by deleting inconvenient observations.

## Troubleshooting: classify the failure before changing anything

Do not respond to every failed command by changing a model option or deleting
rows. First classify the failure and save the evidence below. The classification
belongs in the run manifest and in the final report.

| Category | Typical symptom | Evidence to collect before a fix | Stop condition |
|---|---|---|---|
| Configuration or case-setup failure | A TOML file, copied template file, mapping, approved budget, package pin, or reviewer decision is missing or contradictory | case-study tree, exact command, current case revision/status, missing file or unresolved decision, and relevant walkthrough paragraph | Stop and complete the setup or documentation. This is not a public-package defect. |
| Public-package failure | A complete, pinned canonical case reaches a minimal reproducible public call and fails consistently with a traceback or failing public test | public-package commit, case commit, Python/dependency versions, exact command, smallest input reproducer, traceback, and relevant test output | Stop after the minimal reproducer is recorded; do not work around it by changing case assumptions. |
| Source-data or format failure | Schema/type/key errors, duplicate or ambiguous events, unknown stops/trips, event-time gaps, positive fixed demand in a structural-zero cell, or unexplained empty response rows | offending rows and checksums, canonical schema, stop/line/trip mapping, service-day convention, units, rejection reason, and audit JSON | Stop data preparation. Correct the source or document an explicit policy, then rebuild affected fingerprints. |
| Environment or scheduler failure | Lock/sync failure, import/backend mismatch, JAX compilation failure, out-of-memory, wall-time limit, cancelled job, or missing Slurm output | `uv` command and lockfile, package/backend/device report, full log, job ID, `scontrol show job`, `sacct` state/exit code/MaxRSS/elapsed time, and host information | Stop the affected run and diagnose the environment or resource request separately from the scientific model. |

Never hide a failure by relaxing an unrelated rule, accepting an ambiguous
event, changing a time convention, or reusing an old artifact. Report a missing
custom hook as a data-adapter limitation only after the generic adapter has
validated that the source cannot be represented by the declarative contract. A corrected
input or configuration creates a new fingerprint and requires the documented
downstream rebuild.

## Fully worked dry-runs available in the public checkout

The public repository contains two complementary, runnable smoke cases and a
single generic template. The synthetic reduced-OD example is the complete
artifact/preflight/benchmark path, the Geneva example is the complete
real-timetable count and structural-zero path, and
`examples/case_study_template/` wires the canonical workflow end to end. A
new case copies that template and changes only its declared inputs and policy
files unless a documented custom hook is required.

### A. Synthetic reduced-OD J0 smoke (clean checkout, no private files)

From the public repository root, with the committed lockfile and package
environment installed:

```bash
uv sync --frozen
JAX_ENABLE_X64=true MPLCONFIGDIR=/tmp/public-transportation-mpl-cache \
  uv run --frozen python docs/source/examples/reduced_od_j0_integration.py
```

The script creates a temporary directory and therefore does not rely on
pre-existing generated artifacts. It constructs a four-stop timetable and
timestamped boarding/alighting table in memory, writes a fully specified
`schema_version = 2` configuration, then executes, in order:

1. reduced-OD artifact preparation (timetable index, journey/response data,
   fixed/free layout and manifests);
2. read-only artifact loading and `preflight_reduced_od_j0`;
3. a warm objective-and-gradient benchmark;
4. ML, flat-MAP, informative-MAP, likelihood, prior-sensitivity, adequacy,
   holdout, and full-OD reconstruction checks.

The final JSON summary is printed to stdout. A successful run must report a
valid preflight, finite warm value/gradient timings, successful short fits, and
consistent reconstruction. Any traceback is first classified using the table
above. This smoke does not invoke the TOML structural-zero CLI because its
synthetic demand and fixed-cell policy are constructed directly in Python.
It validates the public API from the public checkout; it does not replace the
separate case-project VCS pin and provenance check in Section 3.1–3.2.

### B. Geneva count-to-structural-zero smoke (real timetable, regenerated outputs)

This case uses the committed derived Geneva tables and their provenance file;
it does not require the 155 MB source GTFS ZIP. From the public repository root:

```bash
uv run --frozen python docs/source/examples/geneva_gtfs/pre_processing/run_preprocessing.py
```

This regenerates `docs/source/examples/geneva_gtfs/pre_processing/results/`.
Before accepting the committed four half-hour bins, audit the generated count
file and candidate demand universe. The committed Geneva source has 3,782
distinct stop-level OD pairs and 15,128 candidate OD-time rows (four bins), so
the worked example uses `max_od_cells = 15128`; for a different case these
numbers and the horizon must be derived and approved, never copied:

```bash
uv run --frozen python - <<'PY'
import csv
from pathlib import Path

path = Path("docs/source/examples/geneva_gtfs/data/prior_demand.csv")
with path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
pairs = {(row["origin_stop_id"], row["destination_stop_id"]) for row in rows}
print("num_od_pairs:", len(pairs))
print("candidate_od_time_rows:", len(rows))
PY

uv run --frozen python - <<'PY'
from public_transportation.measurement import read_measurements_csv

table = read_measurements_csv(
    "docs/source/examples/geneva_gtfs/pre_processing/results/measurements_boarding_alighting.csv"
)
seconds = [record.time.seconds_from_midnight for record in table.records]
print("measurement_rows:", len(table.records))
print("event_time_min_seconds:", min(seconds))
print("event_time_max_seconds:", max(seconds))
print("measurement_types:", sorted({record.measurement_type.value for record in table.records}))
PY
```

The committed Geneva `prior_demand.csv` is a historical, four-bin fixture for
the worked example. It is therefore a legacy time-dependent demand input, not
the canonical independent-universe format. A new Geneva case should first
materialize `inputs/od_pairs.csv` (or use the documented network-derived
policy), then run `od-universe` and `expand-od`; do not replicate these four
coarse rows across newly selected bins without an explicit scientific rule.

Run the mandatory count-based diagnostic with the audited Geneva horizon and
budget:

```bash
uv run --frozen python -m public_transportation.preprocessing.time_discretization \
  --measurements docs/source/examples/geneva_gtfs/pre_processing/results/measurements_boarding_alighting.csv \
  --base-resolution-minutes 5 --min-bin-minutes 10 --max-bin-minutes 60 \
  --max-bins 24 --num-od-pairs 3782 --max-od-cells 15128 \
  --horizon-start 07:00 --horizon-end 09:00 \
  --output-json docs/source/examples/geneva_gtfs/pre_processing/results/time_discretization_recommendation.json
```

Review the JSON candidate, warnings, coverage, and estimated cell count. The
committed `data/time_bins.csv` contains four half-hour bins from 07:00 to
09:00. With the command above, the current snapshot recommends a valid
four-bin `uniform_35m` candidate, whose edges differ from the committed
half-hour bins. That difference is an intentional review decision, not a
reason to silently overwrite the input. Materialize the recommendation to a
separate reviewed file first:

```bash
uv run --frozen python -m public_transportation.preprocessing.materialize_time_bins \
  --recommendation-json docs/source/examples/geneva_gtfs/pre_processing/results/time_discretization_recommendation.json \
  --output docs/source/examples/geneva_gtfs/pre_processing/results/time_bins.reviewed.csv

diff -u docs/source/examples/geneva_gtfs/data/time_bins.csv \
  docs/source/examples/geneva_gtfs/pre_processing/results/time_bins.reviewed.csv
shasum -a 256 \
  docs/source/examples/geneva_gtfs/pre_processing/results/time_bins.reviewed.csv
```

If the reviewed candidate differs, the reviewer must either (a) adopt it,
re-bin `prior_demand.csv`, and rerun structural-zero and preparation stages, or
(b) explicitly retain the committed half-hour bins because they are the
declared synthetic-demand convention. Record that decision and rationale in
the manifest; in either branch, do not claim that the existing file was
validated automatically. After the decision and any required re-binning, run
the structural-zero preprocessor:

```bash
uv run --frozen python \
  docs/source/examples/geneva_gtfs/pre_processing/run_structural_zeros.py \
  --no-progress
```

The command writes the TOML-declared outputs under
`docs/source/examples/geneva_gtfs/pre_processing/results/structural_zeros/`
and prints reason counts, retained/free counts, and the output folder. The
expected committed snapshot has 15,128 candidate cells, 15,032 fixed zeroes,
and 96 free cells; a positive fixed value conflicting with a structural zero
must stop the run.

For this Geneva example, the public `estimation/run_estimation.py` and
`estimation/run_gravity_validation.py` scripts then demonstrate the detailed
fixed-routing and gravity estimators. They are useful validation examples, but
they are not a generic reduced-OD J0 adapter: the artifact-preparation,
read-only-preflight, and warm-benchmark part of that API is demonstrated by
Dry-run A above. A private case that needs the complete combined flow must
implement and commit its adapter/dispatcher before it is submitted to Jed.

## 10. Stage 6 — Read-only artifact preflight

Run this before every new fit environment, especially on Jed:

```bash
cd case_studies/<case_name>
JAX_ENABLE_X64=true uv run --frozen python \
  run.py preflight
```

The stage should call `preflight_reduced_od_j0` (for the minimal model) or the
corresponding generic-demand admission code, then load artifacts read-only and
build the intended problem. It must report:

- compatibility and the first missing/incompatible phase, if any;
- artifact, model, data, specification, and parameter-layout fingerprints;
- likelihood, production mode, model blocks, parameter names, and dimension;
- measurement, free-cell, fixed-cell, origin-period, and response dimensions;
- input, objective, and gradient dtypes;
- bounds and prior fingerprints;
- process RSS before and after problem construction.

If incompatible, rerun preparation with `reuse_or_build`; do not copy a
downstream artifact directory from another configuration.

## 11. Stage 7 — Objective and gradient admission benchmark

Before optimization, benchmark the exact initial problem:

```python
timing = benchmark_minimal_gravity_objective(
    problem=built.problem,
    raw_parameters=initial_raw,
    warm_evaluations=5,
)
```

Require:

- finite objective and gradient;
- successful trace, lowering, compilation, first execution, and warm runs;
- no recompilation when only parameter values change;
- float64 objective and gradient;
- stable warm evaluation times after compilation;
- memory comfortably below the allocation;
- a projected full-fit runtime compatible with the checkpoint plan.

Compilation time and first execution are not representative iteration times.
Use the warm value-and-gradient time and observed accepted-iteration time for
runtime planning. If a warm evaluation is already too slow, stop before the
optimizer and simplify the model/response or inspect accidental dense arrays.

## 12. Stage 8 — Fit the model ladder

Use one immutable response and grouped split while comparing models.

### 10.1 M0 Poisson diagnostic

Fit the smallest production and impedance model first. Poisson has no free
dispersion and is useful for localizing whether an unstable negative-binomial
fit is driven by dispersion. It is not automatically the final scientific
model.

Verify finite values, projected-gradient convergence, sensible production
totals, non-pathological time/transfer coefficients, inactive accidental
bounds, and residual structure.

### 10.2 M0 negative binomial

Fit the same response and demand structure with the negative-binomial
observation model. Compare objective, deviance, residuals, parameter stability,
and fitted dispersion. Dispersion at its numerical floor or an extremely broad
variance is a warning, not evidence of a successful model.

### 10.3 MAP and prior sensitivity

MAP is the recommended primary estimator for a large underdetermined case.
Specify the Gaussian prior explicitly; the library deliberately supplies no
undocumented informative prior. Run at least a broad/weak and a moderate prior
scenario. Report likelihood and prior contributions and whether the result is
prior dominated.

An infinite-scale Gaussian prior is exactly flat and should reproduce ML to
deterministic optimizer tolerance. Use that as a software check, not as the
final regularization strategy.

### 10.4 Add richer demand blocks only when motivated

If M0 has structured residuals or inadequate production variation, add the
next predeclared model block—for example period effects, origin/destination
groups, or low-rank effects. Warm-start by parameter name and preserve lineage.
Compare each child against its parent. Do not activate several new mechanisms
at once: an improved objective would then be difficult to attribute.

## 13. Checkpoints, interruption, and resume

Each fit must have a unique checkpoint path. Include in its identity:

- artifact and model fingerprints;
- likelihood and full model specification;
- parameter layout and dimension;
- bounds;
- prior;
- data identity;
- software/schema version.

Set an application deadline 10–15 minutes before the Slurm wall-time. The
estimator should save the latest accepted iterate atomically and return a
structured `deadline` result with exit status zero. A subsequent identical run
uses `resume=True`. A changed fingerprint, dimension, model, prior, or bound
must reject the checkpoint.

Pressing Ctrl-C should likewise save the latest accepted iterate and return an
`interrupted` result. After any interruption, inspect the checkpoint manifest
before resuming. Never edit checkpoint JSON by hand.

## 14. Running on Jed with Slurm

### 12.1 One-time server check

From the case-study repository on Jed:

```bash
mkdir -p case_studies/<case_name>/results
git rev-parse HEAD
git status --short
uv sync --frozen
uv run --frozen python -c \
  'import public_transportation; print(public_transportation.__version__)'
sbatch case_studies/<case_name>/scripts/probe.sbatch
```

Inspect the probe output for CPU topology, memory, temporary storage, Python,
JAX backend, and devices. Confirm the account, partition, QoS, time, CPU, and
memory limits with the current Jed policy; do not assume the example values
below are universally available. Always submit from the case-study repository
root so that `SLURM_SUBMIT_DIR` is unambiguous. The Slurm output directory must
exist before submission because Slurm opens the log before the job script runs.

### 12.2 Common Slurm wrapper

Each `.sbatch` file should use one process and explicitly control numerical
threads. Adapt `<ACCOUNT>`, `<PARTITION>`, resources, and paths:

```bash
#!/bin/bash
#SBATCH --job-name=od-<stage>
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=case_studies/<case_name>/results/slurm-%x-%j.out

set -euo pipefail

PRIVATE_ROOT="${PRIVATE_ROOT:-$SLURM_SUBMIT_DIR}"
UV_EXECUTABLE="${UV_EXECUTABLE:-$HOME/.local/bin/uv}"

cd "$PRIVATE_ROOT"
test -f case_studies/<case_name>/run.py
unset VIRTUAL_ENV

export JAX_ENABLE_X64=true
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg-cache"
export UV_CACHE_DIR="${TMPDIR:-/tmp}/uv-cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$UV_CACHE_DIR"

/usr/bin/time -v "$UV_EXECUTABLE" run --frozen python \
  case_studies/<case_name>/run.py <stage-and-options>

sstat --jobs="${SLURM_JOB_ID}.batch" \
  --format=JobID,AveCPU,MaxRSS,MaxVMSize 2>/dev/null || true
```

Store reusable artifacts and checkpoints on persistent project storage, not
`$TMPDIR`. Use `$TMPDIR` only for disposable caches. Every wrapper should print
the case-study revision, installed package version and locked source, resolved
config fingerprint, job ID, host, allocation, and output directory before
substantive work starts. A suitable hardware-probe template is available in
the public repository as `jedtests/jed_probe.run`; copy it into the case-study
repository and adapt its account and partition when creating `probe.sbatch`.

### 12.3 Submit a dependent chain

Use `afterok` so that an invalid prerequisite blocks downstream work. A
submission script can be:

```bash
#!/bin/bash
set -euo pipefail

CHECK_JOB="$(sbatch --parsable \
  case_studies/<case_name>/scripts/00_check.sbatch)"
UNIVERSE_JOB="$(sbatch --parsable --dependency="afterok:${CHECK_JOB}" \
  case_studies/<case_name>/scripts/05_od_universe.sbatch)"
BINS_JOB="$(sbatch --parsable --dependency="afterok:${UNIVERSE_JOB}" \
  case_studies/<case_name>/scripts/10_time_discretization.sbatch)"
EXPAND_JOB="$(sbatch --parsable --dependency="afterok:${BINS_JOB}" \
  case_studies/<case_name>/scripts/15_expand_od.sbatch)"
ZERO_JOB="$(sbatch --parsable --dependency="afterok:${EXPAND_JOB}" \
  case_studies/<case_name>/scripts/20_structural_zeros.sbatch)"
PREPARE_JOB="$(sbatch --parsable --dependency="afterok:${ZERO_JOB}" \
  case_studies/<case_name>/scripts/30_prepare.sbatch)"
PREFLIGHT_JOB="$(sbatch --parsable --dependency="afterok:${PREPARE_JOB}" \
  case_studies/<case_name>/scripts/40_preflight.sbatch)"
FIT_1_JOB="$(sbatch --parsable --dependency="afterok:${PREFLIGHT_JOB}" \
  --export=ALL,RESUME=0 \
  case_studies/<case_name>/scripts/50_fit.sbatch)"
FIT_2_JOB="$(sbatch --parsable --dependency="afterok:${FIT_1_JOB}" \
  --export=ALL,RESUME=1 \
  case_studies/<case_name>/scripts/50_fit.sbatch)"
FIT_3_JOB="$(sbatch --parsable --dependency="afterok:${FIT_2_JOB}" \
  --export=ALL,RESUME=1 \
  case_studies/<case_name>/scripts/50_fit.sbatch)"

echo "check=${CHECK_JOB}"
echo "universe=${UNIVERSE_JOB}"
echo "bins=${BINS_JOB}"
echo "expand=${EXPAND_JOB}"
echo "zeros=${ZERO_JOB}"
echo "prepare=${PREPARE_JOB}"
echo "preflight=${PREFLIGHT_JOB}"
echo "fit_segments=${FIT_1_JOB},${FIT_2_JOB},${FIT_3_JOB}"
echo "squeue -j ${CHECK_JOB},${UNIVERSE_JOB},${BINS_JOB},${EXPAND_JOB},${ZERO_JOB},${PREPARE_JOB},${PREFLIGHT_JOB},${FIT_1_JOB},${FIT_2_JOB},${FIT_3_JOB} -o '%.18i %.28j %.2t %.12M %.12l %R'"
```

The fit wrapper should translate `RESUME=0/1` into the estimator's `resume`
setting. All segments use the same checkpoint and result identity. If the fit
is already complete, a later resume segment should validate the completion
manifest and exit successfully without optimizing again. This makes it safe to
schedule enough segments in advance for a very long run.

Prefer this sequence to a single request for a very long wall time. It gives
regular atomic checkpoints, makes queue placement easier, and limits work lost
to an unexpected node failure. Do not use `afterany` in the normal chain: an
actual validation or model failure must not automatically launch estimation.
If Slurm kills a segment before its application deadline, inspect the log and
checkpoint manually, then submit a resume job explicitly.

The example chain is for several wall-time segments of one immutable fit. Run
the scientific model ladder separately: review the completed Poisson diagnostic
before submitting the negative-binomial fit, and review M0 before activating a
richer child model. Independent sensitivity fits may run concurrently only
after their common parent and artifact identities have been accepted, and they
must use different checkpoint and result paths.

### 12.4 Monitoring and resource interpretation

Useful commands are:

```bash
squeue -u "$USER" -o '%.18i %.28j %.2t %.12M %.12l %.6D %R'
scontrol show job <job-id>
sacct -j <job-id> --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
tail -f case_studies/<case_name>/results/slurm-<job>-<id>.out
```

Interpret them carefully:

- high allocation with low CPU use can mean a single-threaded bottleneck,
  compilation, I/O, or excessive Python preprocessing;
- `MaxRSS` near the allocation means the next run needs either more memory or a
  representation change, not merely more CPUs;
- JAX compilation can use CPU without optimizer progress; progress events must
  distinguish compilation from iterations;
- a stale checkpoint age while iterations continue indicates checkpoint wiring
  is not functioning;
- rapidly changing objective with no accepted iterations may indicate line
  search or scaling problems;
- a scheduler timeout is not numerical convergence.

## 15. Stage 9 — Interpret the fit result

Read the machine-readable result before any plot. Distinguish:

1. `optimizer_success`: what SciPy reported;
2. numerical convergence: finite/resolvable objective and sufficiently small
   projected gradient;
3. scientific admissibility: a separate case-study decision.

An optimizer success message alone is insufficient. Review:

- final objective and gradient norm;
- termination status and accepted iterations;
- objective resolution relative to float64 precision and configured tolerance;
- transformed time and transfer sensitivities;
- production coefficients and totals by origin and period;
- fitted destination-attractiveness values and their declared correction block;
- negative-binomial dispersion, if present;
- active raw bounds;
- Hessian rank, eigenvalues, condition number, and weak directions;
- likelihood and prior contributions;
- predicted and observed totals;
- MAE, RMSE, deviance, and variance-weighted RMSE;
- residuals by line, stop, period, direction, and vehicle journey;
- response coverage and positive zero-response observations;
- sensitivity to initialization, likelihood, bounds, and prior.

Large OD error cannot be diagnosed from count fit alone when the system is
underdetermined. A good count fit means the estimate reproduces observed
measurements under the assumed response; it does not prove that every OD cell
is identified.

## 16. Stage 10 — Grouped validation and model advice

Run full-data adequacy diagnostics, but label them in-sample. Then create a
deterministic grouped holdout, preferably by `vehicle_journey`, using
`build_reduced_od_holdout_split`, and refit on calibration rows before calling
`validate_reduced_od_holdout`. Never split individual boarding/alighting rows
from the same vehicle journey independently.

Compare calibration and holdout totals, log likelihood, Poisson or NB deviance,
MAE, RMSE, and variance-weighted RMSE. Examine grouped residuals, not only a
global score. Pass adequacy and metadata to
`recommend_reduced_od_relaxations`; its proposed child models are advisory and
must be scientifically reviewed.

Use the gravity model's weak directions and measurement-response structure to
identify poorly informed demand combinations. These are candidates for
additional data collection, but the recommendation should be expressed at the
level supported by the response (stops, periods, routes, or OD combinations),
not as unjustified certainty about individual cells.

## 17. Stage 11 — Reconstruct and validate only an accepted fit

The optimizer works only with free cells. After accepting a fit, construct a
`ReducedODProblemContract` and call `reconstruct_full_od` to insert frozen cells
and restore canonical OD-time ordering. Verify:

- every canonical cell appears exactly once;
- frozen values are byte-for-byte or numerically unchanged;
- structural zeroes remain exactly zero;
- reconstructed free values match the compact fit;
- totals by origin, destination, and period are consistent with the fit report.

Next, perform one explicit detailed assignment of the reconstructed estimate.
Compare its boarding, alighting, and link flows with observations and with the
reduced response. This is the place to detect disagreement caused by bounded
journey sets, fixed shares, capacity/crowding, or response aggregation. Do not
put the detailed assignment back inside the reduced optimizer.

## 18. Documentation consistency check

Before committing a walkthrough or template update, run the search from the
public-repository root:

```bash
rg -n 'config\.toml|config/case\.toml|config/' \
  docs/source/new_case_study_walkthrough.md \
  docs/source/examples/case_study_template/README.md \
  docs/source/examples/case_study_template/scripts
```

Review every result manually. `config/case.toml` is the generic case-study
configuration; the other contracts must remain under `config/` as
`config/time_discretization.toml`, `config/reduced_od.toml`,
`config/structural_zeros.toml`, and `config/model.toml`. The generated `resolved_config.toml` mentioned in the structural-
zero outputs is an explicitly named audit artifact, not a root-level
configuration contract. No command may assume a root-level legacy configuration
file.
Confirm that every command states whether it is run from the public repository
root, the case-study root, or the case-study repository root on Jed. Finish
with:

```bash
git diff --check
```

## 19. Final acceptance checklist

A case is ready to report only when all applicable boxes can be checked.

### Reproducibility

- [ ] Case-study revision and dirty state recorded.
- [ ] The generic adapter and runner revision are recorded.
- [ ] Any custom adapter hook is named, versioned, and its revision is recorded;
      otherwise the report states that no hook was used.
- [ ] Installed public-package version, locked source/commit, and lockfile
      checksum recorded.
- [ ] The public-package GitHub commit (or immutable PyPI release) was checked
      directly in `uv.lock`, not inferred from an installation path.
- [ ] Runtime versions recorded.
- [ ] Source checksums/provenance recorded.
- [ ] The exact source count file and checksum are recorded.
- [ ] Configuration file checksums and the deterministic configuration
      fingerprint are recorded.
- [ ] Resolved TOML, every generated artifact fingerprint, and every
      parameter-layout/response fingerprint are archived.
- [ ] The complete command transcript (including failed and diagnostic
      commands) is archived.
- [ ] Commands, Slurm scripts, job IDs, logs, and resource use archived.

### Data and routing

- [ ] Observation missing/zero semantics verified.
- [ ] If used, the time-discretization JSON recommendation and its review are
      archived.
- [ ] The adopted candidate and the review decision (including the reviewer)
      are stated explicitly.
- [ ] The reviewer-selected candidate and rationale are archived.
- [ ] The approved candidate was materialized through
      `materialize_time_bins`, and the resulting `time_bins.csv` is recorded.
- [ ] The materialized `time_bins.csv` checksum is recorded.
- [ ] Demand `time_bin_id` values match the adopted bin IDs and the event-time
      to departure-time assignment policy is documented.
- [ ] The demand re-binning policy and any discarded or reassigned rows are
      documented.
- [ ] All re-binning actions are listed in the manifest.
- [ ] Structural-zero, OD-candidate, reduced-journey, and response/operator
      artifacts were rebuilt after any time-bin change.
- [ ] Every retained observation resolves to one timetable event.
- [ ] All unmatched, ambiguous, excluded, and zero-response rows audited.
- [ ] No unexplained positive observation has an empty response.
- [ ] Physical-stop mapping and footpaths reviewed.
- [ ] Structural-zero conflicts resolved without overwriting positive demand.
- [ ] Departure sampling and journey limits are adequate for the periods.

### Numerical estimation

- [ ] Artifacts load read-only with compatible fingerprints.
- [ ] Objective and gradient are finite float64 values.
- [ ] Warm benchmark shows no parameter-value recompilation.
- [ ] Checkpoint interruption and resume were tested on a short run.
- [ ] Every generated artifact and checkpoint fingerprint is archived.
- [ ] Projected gradient and tolerance-resolution checks pass.
- [ ] Active bounds, weak Hessian directions, and production totals reviewed.
- [ ] Poisson/NB and weak/moderate-prior sensitivities reviewed.

### Scientific validation

- [ ] Production semantics are defensible and clearly labeled.
- [ ] Full-data diagnostics are labeled in-sample.
- [ ] Grouped holdout refits and metrics are available.
- [ ] Residuals have been reviewed by operational group.
- [ ] Full OD was reconstructed only after fit acceptance.
- [ ] At least one detailed-assignment comparison was performed.
- [ ] Limitations and rejected model attempts are reported.
- [ ] Any stage stopped because the documentation, case adapter, or public
      package was incomplete is reported explicitly, with the evidence and
      follow-up action.
- [ ] Every failed stage is recorded with one of the four failure categories
      above and its supporting evidence.

## 20. What to deliver to a reviewer

Deliver a compact human-readable report plus the machine-readable directory.
The report should contain:

1. the scientific question and analysis period;
2. input provenance and observation coverage;
3. stop mapping, feasibility, departure sampling, and route-share assumptions;
4. dimensions before and after structural-zero removal;
5. response coverage and all unmatched/unsupported observation decisions;
6. model ladder, priors, bounds, and likelihoods tried;
7. runtime, peak memory, checkpoint history, and Jed job IDs;
8. numerical convergence and identification diagnostics;
9. grouped validation and detailed-assignment comparisons;
10. the accepted model, rejected attempts, limitations, and next action.

Never report only the final parameter vector. The provenance, response
coverage, numerical checks, and unsuccessful attempts are part of the result.

## 21. Related documentation

- [`reduced_od_estimation.md`](reduced_od_estimation.md): numerical robustness,
  likelihood comparison, bounds, progress, and artifact invalidation.
- [`reduced_od_generic_demand.md`](reduced_od_generic_demand.md): composable
  demand models, parameter blocks, warm starts, and restart behavior.
- [`structural_zero_preprocessing.md`](structural_zero_preprocessing.md): TOML
  schema, feasibility rules, conflicts, and generated artifacts.
- [`../reports/reduced_od_observation_contract.md`](../reports/reduced_od_observation_contract.md):
  observation and passenger-journey semantics.
- [`../reports/traffic_assignment.pdf`](../reports/traffic_assignment.pdf):
  mathematical assumptions, RAPTOR response construction, MAP estimation,
  validation, and limitations.
- [`examples/geneva_gtfs/README.md`](examples/geneva_gtfs/README.md): a complete
  real-timetable validation example with synthetic demand and counts.
- [`examples/case_study_template/README.md`](examples/case_study_template/README.md):
  the generic configuration-driven adapter and independent stage runner for
  canonical case-study inputs.
- [`../reports/reduced_od_private_integration_handoff.md`](../reports/reduced_od_private_integration_handoff.md):
  public/private API boundary and artifact graph.
