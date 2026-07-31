# Bounded streaming exact-support preflight

## Scope and unsafe legacy path

The current materializing call path is
`benchmark_fixed_routing_linear_pilot.py` →
`plan_sharded_fixed_routing_operator` → `_discover_support` →
`analyze_fixed_routing_origin_support(materialize=True)`. It creates complete
free and positive-fixed support matrices and then retains a support tuple per
active OD cell, global support-pattern tables, all construction tasks, and all
future storage-shard plans. That path is not a lightweight full-network
preflight.

The replacement in `block_coordinate/support_preflight.py` routes and analyzes
one destination group at a time. Only group and block summaries survive the
group boundary. It supports deterministic sampling, effective resource guards,
atomic partial results, exact fingerprint-safe resume, representative-block
selection, pre-allocation rejection, and pilot authorization without launching
estimation.

## Public validation

Validation used `simple_example_02`, which contains measurements and positive
fixed demand. Complete streaming analysis processed all 7 destination groups,
70 free OD cells, and 42 structural blocks. It found 1,414 exact
OD-to-measurement support entries. Two positive fixed cells were routed for the
fixed offset and were absent from every free block. The retained final summary
was 11,130 bytes; the conservative per-group temporary estimate was 78,766
bytes. Peak process RSS was approximately 451 MB, dominated by assignment and
JAX runtime state rather than retained support.

Streaming totals exactly match the existing materialized origin-support result
on the focused public fixture. Interrupt plus resume produces the same ordered
deterministic summaries as an uninterrupted run. Scenario/routing mismatches
and corrupt checkpoints are rejected. Tests also demonstrate valid typed stops
for elapsed-time, RSS, temporary-memory, and retained-state limits.

Representative construction tests select deterministic small, median, p95,
and largest blocks. Forward products match the corresponding columns of a
complete reference operator. A one-byte worker limit rejects construction
before the injected builder is called.

## Production selected-block construction

`FixedRoutingSelectedBlockBuilder` now turns an explicitly requested block's
exact support into a numerical operator without a complete global operator.
Support artifacts store canonical union rows and per-column rows, authoritative
block/OD coordinates, fingerprints, and a content hash. Writes are atomic and
only requested blocks are retained. Row, nonzero, variable, temporary,
retained-operator, and worker ceilings are enforced before numerical work and
incrementally during support discovery.

For each fixed-size OD chunk, one fixed-shape JIT kernel performs the forward
fixed-routing dynamic program once. A second fixed-shape kernel gathers bounded
mapped-edge chunks from the retained device reach state; the host aggregates at
most `od_chunk_size × measurement_chunk_size` values. No array has shape
`num_measurements × block_variables`. Canonical sparse assembly produces a
compact supported-row CSR and CSC. Forward products scatter into full
measurement coordinates, and transpose products gather the same rows.

The public `simple_example_02` validation selected small, median, p95, and
largest blocks. All cold builds had exact zero outside structural support,
adjoint errors between zero and (1.4\times10^{-17}), and fresh-builder warm
loads were bit-identical cache hits. Warm disk loads were below 0.5 ms in this
run. Corrupt numerical caches were safely rebuilt; corrupt support artifacts
were rejected. The focused support-preflight suite has 25 passing tests.

The reproducible result is
`benchmarks/support_preflight_simple_example_02.json`.

## Commands executed

```text
.venv/bin/pytest tests/block_coordinate/test_support_preflight.py -q
.venv/bin/ruff check src/public_transportation/inference/block_coordinate/support_preflight.py src/public_transportation/inference/block_coordinate/selected_blocks.py tests/block_coordinate/test_support_preflight.py benchmarks/benchmark_support_preflight.py
.venv/bin/python benchmarks/benchmark_support_preflight.py --mode streaming-exact-support --check --output benchmarks/support_preflight_simple_example_02.json --checkpoint-directory /tmp/support-preflight-public-validation
git diff --check
```

Focused result: 10 tests passed. The prescribed regression set passed 164 tests
with one existing stop-time regularization warning. Full Ruff and
`git diff --check` passed. `traffic_assignment.tex` compiled successfully to a
24-page PDF with TeX Live.

## Private read-only validation and authorization

Structural mode exactly confirmed 1,408,084 candidate OD cells, 627,123 free
cells, 780,961 frozen-zero cells, no frozen-positive cells, 426,436
measurements, 433,462 nodes, 1,456,458 links, 1,871 destination groups, and
3,778 blocks at a 512-variable ceiling. Peak RSS was 11.69 GiB and retained
structural summaries were approximately 0.41 MiB. Structural support took
61.72 seconds after preparation. No exact planner or estimator was invoked.

Deterministic sampled exact support selected groups 40, 60, 128, and 737. The
first process completed three groups and stopped at the 180-second boundary
with group 737 pending. A fresh process validated every fingerprint, skipped
the three completed groups, and processed only group 737. The combined result
contains 2,323 free cells, 743,435 exact support entries, and nine observed
blocks. Retained summaries occupy 3,150 bytes; the conservative group
temporary estimate is 34,972,242 bytes; observed peak RSS is 11.99 GiB.

The largest observed block has 512 variables, 221,169 exact nonzeros, 39,293
support rows, and an estimated 5,622,408-byte CSR/transpose representation.
The deliberately conservative sampled projection is 4.52--18.09 GB of cache,
2,104--8,420 storage shards, and 0.078--5.62 MB per observed block. Its
support-time range is broad (approximately 17.2--71.2 hours) because routing a
selected destination group took 33--137 seconds and no linear-scaling claim is
made.

The evidence authorizes neither complete-network representative construction
nor a 100-update TPG pilot: four sampled groups cannot bound every unobserved
block, and the cache/time projection remains too uncertain. The next safe step
is a wider deterministic support sample on the target server, in separate
fresh processes with the same fingerprints and a 512 MiB worker ceiling.
`exact-materialized-plan` must remain off and no estimation should start.

The production builder was subsequently validated read-only against the
completed sampled report. It constructed the distinct smallest, median, and
largest/p95 representatives without a complete operator. The largest,
`block-000255`, retained all 221,169 exact entries on 39,293 rows. Numerical
construction took 200.97 seconds; the conservative peak estimate was 150.34 MB,
resident CSR/CSC state was 7.71 MB, disk storage was 4.17 MB, and a fresh warm
load took 10.42 ms. Forward and transpose products took 1.01 and 1.17 ms, the
adjoint error was (2.84\times10^{-14}), and values were exactly zero outside
support. The two other representatives were also exact and bit-identical after
warm loading. All new state was written below `/tmp`; the private repository
was not changed. Results are recorded in
`benchmarks/selected_fixed_routing_blocks_tpg_20260730.json`.

Representative numerical construction is therefore validated. Complete
support discovery across every nonempty destination group remains the blocker,
so the sampled evidence still does not authorize the 100-update pilot.

Checkpoint schema 3 subsequently corrected time-budget resume semantics. The
existing private schema-2 checkpoint was migrated from a read-only copy and a
fresh invocation performed another 120.398 seconds of bounded work on pending
group 128. Cumulative support time advanced from 146.053 to 266.452 seconds;
the incomplete group remained uncommitted and group 737 was not skipped. See
`support_preflight_resume_correction.md` for the fingerprint and policy rules.
