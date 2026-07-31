# Release notes

## Unreleased

### Bounded large-network MAP pilots

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
