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

## Private read-only migration validation

The existing TPG schema-2 checkpoint was copied to `/tmp`; the private artifact
was not modified. Its original configuration and fingerprints validated, and a
fresh 120-second invocation began work on pending group 128 rather than stopping
immediately. It stopped inside a bounded chunk after 120.398 seconds, an
explicit 0.398-second overshoot. No partial group summary was committed.

The migrated schema-3 result reports 266.452 cumulative seconds: 146.053 from
the legacy invocation plus 120.398 from the new invocation. It records 120.008
seconds of discarded group-128 work from the new invocation. Schema 2 could not
distinguish the corresponding categories within its earlier 146.053 seconds,
so that legacy time remains cumulative but unattributed. Completed groups are
still 40 and 60; pending groups remain 128 and 737. Peak RSS remained about
11.98 GiB and retained checkpoint summaries occupy 1,203 bytes.

The private result is `benchmarks/support_preflight_tpg_schema2_resumed.json`.
It does not authorize representative block construction or the 100-update
pilot because exact support for the pending representative groups is not yet
complete.

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

The private command used the same benchmark with `--resume`, theta 5,
sample count 2, seed 20260730, origin chunk 16, block ceiling 512, a 120-second
invocation allowance, 14 GiB RSS, 512 MiB temporary and block ceilings, 64 MiB
retained state, and the copied checkpoint directory
`/tmp/support-preflight-tpg-schema2-migration`. Its output is the private-resume
JSON named above.
