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

## External-case validation

Validation on private or otherwise external scenarios is performed in the
case-study repository that owns the data. Those runs are intentionally not
included in this public report: their input paths, checkpoints, fingerprints,
and case-specific resource measurements must remain with the case owner. The
public fixture results above are the reproducible evidence for this package;
case owners should produce an equivalent report with their own results root and
record only the resulting summary and fingerprints in their private workflow.
