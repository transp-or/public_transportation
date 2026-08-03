# Release notes

## Unreleased

- Formalized a common gravity measurement-operator protocol and replaced the
  sharded operator's production Python graph traversal with reusable
  fixed-shape compiled forward, reverse, and multi-right-hand-side kernels over
  persisted routing probabilities. Added bounded shard execution batches,
  structured product progress, predictive deadlines and cancellation, detailed
  timing/residency metrics, estimator deadline propagation, a phase-bounded
  public gravity preflight, and a scalable synthetic operator benchmark. Added
  a conservative first-batch deadline estimate, an atomic run-manifest writer,
  and a shared durable JSONL sink for operator and optimizer progress.
- Added bounded concurrent dispatch of independent compiled routing-shard
  products. CPU, shard-count, and in-flight routing-byte ceilings constrain the
  worker count, while canonical ordered accumulation preserves deterministic
  forward, reverse, and matmat results. Whole-product deadline projection now
  stops after a measured wave when all remaining work cannot finish. Public
  8,192-node measurements improved warm forward throughput by 2.01 times with
  eight workers; aggregate scan batching and vectorized-group loading remain
  benchmarked alternatives.

### Structural-zero preprocessing progress

- Added immutable structured progress events and optional callbacks across
  scenario loading, topology construction, destination profiles, cell
  classification, fixed-demand reconciliation, artifact rendering, atomic
  writes, and completion. Large loops use bounded count/time throttling.
- Added an optional context-managed `tqdm` adapter that renders one phase at a
  time on stderr, plus `--progress`/`--no-progress` support in the documented
  structural-zero example CLI. Library behavior remains silent by default.

### Reduced-dimensional gravity demand

- Added immutable, fingerprinted sparse cell-level gravity features with exact
  compact-layout validation, prepared production totals, fixed attractiveness
  offsets, and eager shape/index/finite-value validation.
- Added the minimal declarative gravity specification and explicit three-value
  parameter layout with stable positive transformations.
- Added a JAX masked grouped-softmax demand generator that preserves structural
  zeros and origin-time totals without a dense OD-time tensor, plus a NumPy
  testing reference.
- Connected gravity demand to existing dense and BCOO fixed-routing measurement
  operators, including fixed positive-demand offsets and compact-layout checks.
- Added Poisson and shared mean/dispersion negative-binomial objectives with
  calibration-mask decomposition, plus batched-forward and adjoint JAX gradient
  strategies verified against direct differentiation and finite differences.
- Added the minimal checkpointed L-BFGS gravity estimator with immutable full-OD
  results, exact model fingerprints, clean iteration-boundary deadlines, and
  compatible resume.
- Added bounded automatic batched-forward/adjoint preflight with separate JAX
  tracing, lowering, compilation, first-execution, and warm-execution metrics,
  plus optional persistent compilation-cache configuration.
- Added full-calibration gravity model-adequacy reports with Poisson/NB deviance,
  residual thresholds, generic grouped summaries, optional within-journey
  correlation, immutable fingerprints, and cautious non-causal findings.
- Added atomic destination-zone attractiveness, broad time-period journey-time,
  and origin-zone production relaxations with exact sum-zero parameterizations,
  parent-preserving warm starts, ridge regularization, applicability checks, and
  explicit parameter/execution-impact descriptions.
- Added immutable advisory relaxation catalogs that score applicable centered
  children from new-coordinate gradients and regularized Hessian blocks, combine
  those scores with grouped residual evidence, disclose support and identification
  warnings, and flag possible routing, timing, or observation problems without
  changing or fitting a child model.
- Added immutable gravity-model lineage nodes containing complete specifications,
  estimates, routing/input provenance, optimizer and adequacy state, runtime/memory
  diagnostics, checkpoint locations, and explicit measurement identities. Added
  an explicit-selection progression API that verifies the child warm start,
  estimates and compares one child, and returns a new lineage preserving both fits.
- Added deterministic, serializable, fingerprinted grouped holdout splits for
  journeys, stop-time series, lines, directions, time blocks, and explicit groups,
  with optional operational and geographic stratification. Added calibration-only
  parameter re-estimation followed by separate full-assignment calibration and
  untouched-holdout predictive metrics, including explicit no-leakage tests.
- Added a public multi-parameter gravity performance benchmark covering dense and
  BCOO routing, forward and transpose products, tracing/lowering/compilation,
  batched-forward and adjoint gradients, warm parameter reuse, memory, dimensions,
  and honest in-process compilation-cache counts. Added relaxed-model BCOO and
  checkpoint/resume hardening tests.
- Added a bounded end-to-end gravity validation on the public Geneva snapshot:
  scheduled feature preparation, direct sparse fixed-routing construction, minimal
  NB estimation, adequacy, recommendations, a selected period child, immutable
  lineage, and grouped journey holdout. Corrected adequacy recomputation checks to
  use explicit float32 tolerances while retaining stricter float64 validation.
- Added worked gravity-estimation scripts and reference summaries for both simple
  synthetic examples. Example 01 demonstrates exact frozen-positive offsets;
  Example 02 demonstrates fitting, explicit period relaxation, an exact child
  warm start, lineage, and grouped journey holdout.
- Added a typed device-native gravity measurement-operator protocol. Existing
  dense and BCOO operators and the matrix-free fixed-routing operator now expose
  forward, transpose, and bounded multiple-right-hand-side JAX products without
  requiring gravity code to access `.matrix`.
- Refactored batched-forward and adjoint gravity gradients to reuse their routed
  prediction instead of performing a duplicate forward product. Matrix-free
  automatic strategy selection now chooses adjoint directly and avoids compiling
  both expensive cold full-network strategies.
- Added forward-mode transformation support to the linear matrix-free product,
  deadline-governed forward/transpose preparation with separated cold and warm
  diagnostics, and a bounded packaged-example benchmark proving that no global
  operator is constructed.

### Bounded large-network MAP pilots

- Added bounded destination-group fixed-routing shards with deterministic byte-
  limited planning, one padded compiled shape, atomic fingerprinted persistence,
  resumable manifests, structured progress, deadline/RSS/cache controls, and
  explicit cache refresh semantics.
- Added an on-demand sharded matrix-free measurement operator with bounded LRU
  residency and an explicit reverse-topological custom VJP. Public numerical
  tests and a packaged benchmark verify complete-routing equivalence without a
  global measurement matrix.
- Added synchronized warm-shard phase diagnostics and controlled scaling
  benchmarks, confirming that the current dynamic program traverses the full
  graph independently of enabled-link density.
- Enabled deterministic bounded thread-pool construction with one shared XLA
  executable, per-dispatch RSS admission, canonical progress and manifest
  commits, worker-failure recovery, and rolling deadline prediction. Serial,
  two-worker, and four-worker public outputs are numerically identical.
- Completed opt-in profiling for that threaded path: workers return immutable
  synchronized phase and density diagnostics, the coordinator attaches the
  canonical manifest-commit time, and planning/cache-scan plus bounded dispatch
  progress events make long warm-shard runs observable.
- Corrected parallel progress to derive active, buffered, queued, failed, and
  completed counts from the complete coordinator state, with an exact lifecycle
  invariant and full sorted active-shard tuples. Added warm shared-executable
  concurrency measurements and context-switch diagnostics without changing
  production kernel or cache identity.
- Added an experimental fixed-shape batched routing-shard execution strategy,
  synchronized batch CPU diagnostics, partial-batch padding, per-shard atomic
  persistence/resume, predictive batch deadlines, and temporary-memory
  admission. Existing thread-produced shards remain reusable because execution
  strategy changes compiled shape identity but not routing-data identity.
- Extended worker recommendations with optional measured-throughput calibration
  while retaining a conservative one-worker default when target measurements
  are unavailable. Public benchmarks now cover default and controlled external
  thread environments plus explicit two-device CPU `pmap` placement.

- Fixed matrix-free initialization for layouts without positive fixed demand:
  the exact fixed offset is now a read-only zero vector and triggers no routing
  assignment or JAX compilation.
- Made complete forward and transpose compilation lazy and independent, added a
  shareable absolute preparation/estimation deadline, and exposed structured
  offset, compilation, execution, and deadline diagnostics.
- Propagated the same absolute deadline into the production selected-block
  builder. Bounded internal checks now stop safely with structured diagnostics,
  preserve atomic cache validity, and leave the pending block and pre-update
  checkpoint unchanged for exact retry on resume.

- Added an explicit, fingerprinted global-product policy for initial, periodic,
  resume, and final diagnostics. Periodic exact products can now be disabled
  with `None`.
- Added validated supplied initial predictions, deferred resume validation,
  restricted pilot schedules, structured global/block/checkpoint work metrics,
  and safe update-budget extension across resume.
- Added selected-block subset authorization and a separate exact final
  validation API. Deferred modes postpone—not replace—global consistency and
  optimality checks.
- Extended the monotonic deadline across initialization and indivisible phases;
  JAX compilation cannot be interrupted in-process, and deadline overshoot is
  reported explicitly.

### Selected fixed-routing block construction

- Removed closure capture of complete graph arrays from selected-block JAX
  kernels; graph and routing arrays are explicit dynamic arguments.
- Added separate tracing, lowering, compilation, synchronized execution, host
  transfer, RSS, shape/dtype, backend/device, and captured-constant diagnostics.
- Added a bounded, lock-protected compiled-kernel cache keyed by assignment
  provenance, backend, dtype, fixed shapes, and kernel schema. Deadline and
  callback state do not affect compiled identity.
- Added immediate structured phase-start/completion callbacks and an optional
  flushed, fsync-capable JSONL sink so scheduler cancellation preserves the
  identity of the active tracing, lowering, compilation, execution, or transfer
  phase independently of numerical-cache publication.
- Added diagnostic stop boundaries after tracing, lowering, compilation, and
  execution to the public support-preflight benchmark.

- Added independently memory-guarded OD batching to
  `FixedRoutingSelectedBlockBuilder`. Automatic mode chooses the largest safe
  batch and explicit batch factors fall back deterministically when necessary.
- Hoisted measurement indexing, enabled-link filtering, local-row translation,
  and padded edge planning out of the OD loop.
- Added programmatic phase timings and work counters for mapping preparation,
  OD preparation, graph evaluation, support filtering, sparse triplets,
  duplicate reduction, CSR/CSC assembly, persistence, candidate contributions,
  and accepted nonzeros.
- Updated selected-block cache schema to version 2. Pure scheduling parameters,
  including OD batch size, no longer fragment the cache because canonical
  per-column accumulation is unchanged. Existing schema-1 numerical caches are
  rejected and rebuilt; selected-support artifacts remain compatible.
- Added a public 512-column synthetic benchmark. Automatic batching reduced
  graph evaluations from 512 to 1 and numerical construction time from 0.361 s
  to 0.101 s (3.58×) while producing exactly identical sparse operators.
