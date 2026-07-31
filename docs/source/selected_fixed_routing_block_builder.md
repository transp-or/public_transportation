# Selected fixed-routing block builder

## Purpose and safety boundary

The production selected-block builder constructs only explicitly requested OD
blocks. It does not accept or synthesize a complete fixed-routing measurement
operator, does not call the materialized global support planner, and does not
allocate a dense full-measurement-by-block matrix.

## Construction algorithm

1. Validate the block against the authoritative partition and apply a
   conservative support-discovery guard before routing.
2. Load a fingerprinted exact-support artifact, or discover support for the
   block's single destination group in bounded OD chunks. Persist canonical
   union rows and per-column rows atomically.
3. Estimate support indices, COO assembly, CSR and CSC storage, fixed-shape
   numerical buffers, routing state, solver vectors, retained bytes, and peak
   worker bytes. Reject any configured ceiling before compiling the numerical
   kernel.
4. Prepare the measurement-row lookup and enabled mapped-edge chunks once for
   the block. Combine one or more base OD chunks into a separately configured
   OD batch, run one compiled forward fixed-routing dynamic program for that
   batch, and reuse its device reach state for all fixed-size mapped-edge
   gathers. Aggregate at most
   `effective_od_columns × measurement_chunk_size` host values.
5. Intersect numerical values with exact per-column support, filter configured
   numerical zeros, assemble canonical compact CSR/CSC storage, validate its
   support, and release routing temporaries.
6. Persist the block atomically with full provenance and a content hash. A fresh
   process validates and loads this cache without routing or reconstruction.

The returned operator has logical shape
`(total_measurements, block_variables)`, but stores only supported rows.
`matvec` scatters a compact product into a full measurement vector; `rmatvec`
gathers the same rows from a full cotangent before applying the CSC transpose.

## OD batching and memory model

`od_chunk_size` remains the base support-discovery unit. `od_batch_size` is the
number of those base chunks combined for numerical routing. A positive integer
requests an explicit factor; `None` selects the largest safe factor
automatically. Before compiling or allocating the reach buffer, the builder
estimates:

- `effective_od_columns × num_nodes × value_bytes` for reach state;
- bounded measurement and mapped-edge buffers;
- support indices and COO triplets;
- canonical CSR and CSC storage;
- routing state, solver vectors, and retained cache bytes.

The requested factor is reduced deterministically until both temporary and
worker ceilings pass. If one base batch cannot pass, construction raises
`BlockConstructionResourceError` before numerical allocation. The effective
factor and column width are available in both the resource estimate and
construction diagnostics.

Measurement lookup, enabled-link filtering, global-to-local row translation,
and padded mapped-edge plans are computed once per block. Diagnostics report
their preparation time separately from OD preparation, graph evaluations,
support filtering, triplet generation, duplicate reduction, CSR/CSC assembly,
and persistence. They also expose graph evaluation and mapping-pass counts,
candidate contributions, and accepted nonzeros.

The numerical kernels no longer close over graph arrays. Topological order,
padded adjacency, link endpoints, routing probabilities, and masks are dynamic
JAX arguments; only fixed dimensions remain in the Python closure. Construction
stages each kernel explicitly through tracing, lowering, compilation, execution,
device synchronization, and host transfer. Structured diagnostics report those
timings, input shapes and dtypes, backend and devices, peak RSS, and captured
constant bytes.

Compiled reach and mapped-edge executables are retained in a bounded,
lock-protected per-builder LRU. Its deterministic identity includes the kernel
schema, assignment fingerprint, backend, routing dtype, effective OD width,
mapped-edge width, and graph dimensions. It excludes the deadline, clock, block
ID, progress callback, and cache directory. Compatible blocks therefore reuse
executables; incompatible assignments, dtypes, shapes, or backends cannot.

## Shared construction deadline

`build_result(block, absolute_deadline=deadline)` and
`build(block, absolute_deadline=deadline)` implement the estimator's explicit
deadline-aware factory capability. The same absolute monotonic deadline can be
passed through matrix-free preparation and block-coordinate estimation; no
phase receives a fresh duration.

The builder checks the deadline around support-cache lookup and loading,
bounded support-discovery chunks, routing preparation, measurement-plan and
mapped-edge chunks, every OD batch, support filtering, sparse triplet creation,
duplicate reduction, CSR/CSC assembly, numerical validation, atomic
persistence, and final cache validation. Checks also bracket tracing, lowering,
compilation, and synchronized execution; deadline state never enters a JAX
argument or closure. JAX compilation or dispatched work is
indivisible in-process: the builder synchronizes, records an overshoot, raises
`SelectedBlockConstructionDeadlineError`, and starts no later expensive phase.
An external process or scheduler timeout remains necessary for a hard limit.

The exception carries the block and phase, elapsed and overshoot time, completed
OD batches and mapping passes, candidate and accepted-entry counts, support
cache status, persistence and warm-cache status, partial-work disposition, and
the current memory estimate. Numerical construction is not claimed to be
resumable. Work stopped before validated atomic publication is discarded, a
previous valid cache is preserved, and a cache fully published before the
deadline remains reusable.

OD batching does not change per-column accumulation order. Consequently,
batch size and other pure scheduling chunk sizes are not part of cache identity;
operators constructed with different safe batch factors share the same logical
cache. Cache schema 2 distinguishes this canonical accumulation contract from
older selected-block caches.

## Public validation

The public small example validates numerical columns against the established
complete fixed-routing operator, forward and transpose products, the adjoint
identity, exact zero outside support, deterministic cold/warm storage, cache
corruption recovery, support corruption rejection, and rejection before
support discovery under a one-byte budget.

The operational benchmark command is:

```text
uv run --frozen python benchmarks/benchmark_support_preflight.py \
  --mode streaming-exact-support \
  --construct-representative-blocks \
  --persist-selected-block-support \
  --block-cache-directory PATH \
  --selected-support-directory PATH \
  --storage-dtype float64 \
  --measurement-chunk-size 512 \
  --od-chunk-size 32 \
  --od-batch-size auto \
  --maximum-block-operator-bytes 536870912 \
  --maximum-temporary-bytes 536870912 \
  --check
```

The command performs no estimation and constructs only representative or
explicitly named blocks present in the bounded preflight result.

The routine public batching benchmark is enabled with
`--benchmark-od-batching --synthetic-od-columns 512`. It derives a synthetic
512-column block only by repeating origins from one destination group in the
public example. The recorded run reduced graph evaluations from 512 to 1 and
numerical construction from 0.361 s to 0.101 s (3.58×); all batch sizes were
exactly equal and warm loads remained below 0.7 ms. Machine-readable results
are in `benchmarks/selected_block_od_batching_simple_example_02.json`.
