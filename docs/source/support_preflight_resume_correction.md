# Support-preflight time-budget resume correction

## Defect and correction

Earlier checkpoints stored one elapsed value and used it both as cumulative
work and as the next invocation's allowance clock. A checkpoint that had
already exhausted 120 seconds therefore stopped immediately when resumed.

Schema 3 separates:

- semantic configuration identity;
- invocation-policy identity and serialized provenance;
- cumulative, previous-invocation, and current-invocation elapsed time.

Only current-invocation time is compared with
`maximum_elapsed_seconds`. Cumulative time remains monotonic across resumes and
continues to feed reporting and projections. `elapsed_seconds` remains a
backward-compatible alias for cumulative time.

## Compatibility rules

Semantic changes are incompatible: mode, selected-group or sampling definition,
seed, support chunking, probability tolerance, persisted representation, schema,
or any scenario/assignment/OD/fixed-demand/measurement/routing/partition
fingerprint. Operational changes receive a separate policy fingerprint.
Elapsed, RSS, temporary, retained, row, nonzero, and block-size ceilings may be
increased. Tightened retained or block ceilings are rejected before processing
when checkpoint state already violates them; tighter RSS and temporary limits
are checked before the next group allocation. Worker-count changes are accepted
because the current implementation performs deterministic serial support
discovery regardless of the requested downstream construction count.

Schema-2 checkpoints are migrated only when first resumed with their exact old
configuration, because their single fingerprint does not permit safe separation
of semantic and policy changes. The resulting checkpoint is schema 3. Schema 1
is explicitly unsupported.

## Public demonstration

`support_preflight_simple_example_02_time_stop.json` records invocation 1
stopping before its first group. Invocation 2 changes its allowance from one
microsecond to 30 seconds without changing the semantic fingerprint and
completes all four deterministically selected groups. The resumed report is
`support_preflight_simple_example_02_time_resumed.json`; it records invocation
count 2 and exact previous-plus-current cumulative accounting.

Clock-controlled tests repeat short invocations until completion and compare
the resulting ordered summaries and block statistics with an uninterrupted
run. Separate tests stop inside a chunk, verify that the incomplete group is not
committed, and record its discarded time.

## External-case migration

Migration of checkpoints belonging to a private or external case must be
validated in that case's own repository. This public document deliberately
does not record private paths, fingerprints, checkpoints, or resource
measurements. Apply the compatibility rules above and retain the resulting
manifest with the case owner's durable results.

## Verification

The focused support/checkpoint command passed 30 tests. The prescribed complete
regression passed 177 tests with one existing stop-time regularization warning.
Ruff, `git diff --check`, and the 25-page LaTeX build all passed. No estimation,
materialized full-network planning, commit, or push was performed.

```text
MPLCONFIGDIR=/tmp/public-transportation-mpl-cache UV_CACHE_DIR=/tmp/public-transportation-uv-cache uv run --frozen --extra dev pytest -q tests/block_coordinate/test_support_preflight.py tests/block_coordinate/test_checkpoint_store.py --tb=short

MPLCONFIGDIR=/tmp/public-transportation-mpl-cache UV_CACHE_DIR=/tmp/public-transportation-uv-cache uv run --frozen --extra dev pytest -q tests/block_coordinate tests/bayesian_estimation/test_fixed_routing_measurement_operator.py tests/bayesian_estimation/test_fixed_routing_inputs.py tests/bayesian_estimation/test_simple_example_01_all_estimators.py

UV_CACHE_DIR=/tmp/public-transportation-uv-cache uv run --frozen --extra dev ruff check src/public_transportation tests/block_coordinate benchmarks/benchmark_support_preflight.py

git diff --check

.venv/bin/python benchmarks/benchmark_support_preflight.py --mode sampled-exact-support --sample-count 4 --maximum-elapsed-seconds 0.000001 --check --output benchmarks/support_preflight_simple_example_02_time_stop.json --checkpoint-directory /tmp/support-preflight-public-resume-v3b

.venv/bin/python benchmarks/benchmark_support_preflight.py --mode sampled-exact-support --sample-count 4 --maximum-elapsed-seconds 30 --resume --check --output benchmarks/support_preflight_simple_example_02_time_resumed.json --checkpoint-directory /tmp/support-preflight-public-resume-v3b
```

Case-specific commands, checkpoint locations, and resource limits are not part
of this public fixture demonstration. Keep them in the owning case repository
and use the same fingerprint-safe resume procedure.
