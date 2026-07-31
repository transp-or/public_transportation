# Bounded block-coordinate pilots

Large fixed-routing problems should use the complete measurement operator only
for explicitly requested global diagnostics. Block updates use
`FixedRoutingSelectedBlockBuilder`, exact-support artifacts, automatic OD
batching, and persistent numerical block caches.

Create one external absolute monotonic deadline and pass it both to
`MatrixFreeFixedRoutingMeasurementOperator(preparation_deadline=deadline)` and
to `run_block_coordinate_map(..., absolute_deadline=deadline)`. This covers
matrix-free offset preparation, the initial prediction, selected-block
construction, solving, and checkpoint persistence. Report matrix-free
preparation time and estimator elapsed time separately, as well as their
combined external-wall-clock total.

The estimator passes this deadline only to factories declaring the explicit
deadline-aware capability. `FixedRoutingSelectedBlockBuilder` implements it;
plain callables remain compatible but only receive entry/exit checks, so their
internal work cannot be bounded by the library.

If the compact layout has no positive fixed cells, matrix-free construction
uses an exact read-only NumPy zero offset and performs no assignment or JIT
compilation. Forward compilation remains lazy until the initial global
prediction, and transpose compilation remains lazy until an explicitly enabled
exact gradient requests it.

## Global-product policy

`GlobalProductPolicy` independently controls the initial prediction, initial
exact projected gradient, periodic exact gradients, resume validation, final
prediction validation, and final exact gradient. Defaults preserve the former
exact behavior. Set `exact_global_diagnostic_every_sweeps=None` to disable
periodic products; a large sentinel integer is not supported.

For a scalable bounded pilot, supply a previously validated prediction and its
complete `BlockCoordinateFingerprints.fingerprint`:

```python
policy = GlobalProductPolicy(
    initial_prediction_mode="provided",
    initial_prediction_validation="deferred",
    initial_exact_gradient=False,
    resume_prediction_validation="deferred",
    final_prediction_validation="deferred",
    final_exact_gradient=False,
)
config = BlockCoordinateMAPConfig(
    maximum_block_updates=1,
    maximum_elapsed_seconds=600.0,
    exact_global_diagnostic_every_sweeps=None,
    global_product_policy=policy,
    pilot_block_schedule=("small", "median", "maximum"),
    construction_workers=1,
    solver_workers=1,
    threads_per_worker=1,
    checkpoint_directory=checkpoint,
)
identity = BlockCoordinateFingerprints(..., solver_semantics=config.fingerprint)
result = run_block_coordinate_map(
    problem=problem,
    partition=partition,
    config=config,
    fingerprints=identity,
    initial_free_flow=initial_flow,
    initial_prediction=initial_prediction,
    fixed_measurement_offset=problem.fixed_measurement_offset,
    initial_prediction_fingerprint=identity.fingerprint,
    block_operator_factory=selected_block_builder,
    absolute_deadline=deadline,
)
```

The supplied arrays are shape-, finiteness-, bounds-, offset-, and fingerprint-
validated and copied into immutable state. Select
`initial_prediction_validation="exact"` on small problems to recompute it.
Deferred validation never claims numerical equality: its diagnostic kind and
result status are explicitly `deferred`.

Use `authorize_selected_block_pilot` before constructing a restricted pilot.
It authorizes only the requested deterministic schedule and requires every
block to belong to a completed support group, have a compatible persisted
exact-support artifact, and satisfy the supplied row, nonzero, operator, and
temporary-memory ceilings. This authorization never implies full-sweep
authorization.

Update, elapsed-time, and sweep limits are invocation policy and may be
extended on resume. The block schedule, solver behavior, global-product policy,
and all numerical semantics remain fingerprinted. Thus extending a five-update
checkpoint to six updates is safe, while changing validation or solver
semantics is rejected.

`validate_block_coordinate_result` performs optional exact prediction,
objective, projected-gradient, bounds, and checkpoint-state checks separately,
including in a fresh process. Deferring exact products makes the pilot scalable
but postpones global optimality and complete prediction-consistency claims.

The elapsed-time deadline is checked before and after global products, selected
block construction, solving, and checkpointing. JAX compilation and dispatched
device work are indivisible in-process and can exceed a deadline. The result's
`deadline_exceeded_by_indivisible_operation` field records this; the estimator
does not start another expensive phase afterward. Use an external scheduler
wall-time margin for hard process termination.

Selected-block deadline checks stay in Python and bracket tracing, lowering,
compilation, synchronized execution, and filtering. The deadline and clock do
not participate in compiled-kernel identity. No-deadline and far-future runs
therefore use the same numerical executable; an overshoot after an indivisible
stage prevents the next stage from starting.

Structured progress and result diagnostics report complete forward/transpose
counts and time, selected-block construction time, block solve time, checkpoint
time, validation status, prediction source, termination reason, and deadline
overshoot status. Library code does not print unconditionally.

If selected-block construction reaches the deadline, the estimator returns
`stopped_by_time_budget`, performs no solve, leaves accepted/rejected counts and
schedule position unchanged, and preserves the checkpoint from before the
pending block. A fresh-process resume retries that same block. If construction
finished and published its validated cache before the solve boundary expired,
resume loads the warm cache and still attempts the pending update exactly once.
Construction attempts, completed and deadline-stopped constructions, pending
unsolved block, solver-started state, cache hits/misses, deadline phase, and
overshoot are exposed in work diagnostics.
