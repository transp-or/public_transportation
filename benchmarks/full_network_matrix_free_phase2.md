# Full-network matrix-free feasibility: Phase 2

## Outcome

This phase implements and validates the exact two-pass decomposition needed to
stream fixed-routing destination groups without constructing an OD-to-
measurement operator, a Jacobian, or one global reverse-mode tape.

For destination groups \(g\), prediction functions \(p_g(z_g)\), and a
nonlinear measurement objective \(L\), the calculation is:

1. \(p = \sum_g p_g(z_g)\)
2. \(c = \partial L(p) / \partial p\)
3. \(\partial L / \partial z_g = J_{p_g}(z_g)^T c\)

The likelihood is evaluated exactly once after all destination contributions
have been summed. Evaluating a likelihood for each partial prediction and then
summing those likelihoods is not equivalent and is deliberately unsupported.

## Implementation

`streamed_measurement_value_and_grad` performs one forward accumulation pass,
calculates the measurement-space cotangent, and then recomputes one destination
at a time under `jax.vjp`. It returns the complete prediction, scalar
measurement objective, measurement cotangent, and global free-parameter
gradient. It does not call `jacrev`, `jacfwd`, or materialize a measurement by
OD array.

The single-group extraction helper now retains positive-frozen OD cells and
their fixed values while removing frozen-zero cells. Its demand assembler
places free demand and fixed offsets into their correct local coordinates.

This is intentionally a bounded primitive. The current API accepts a sequence
of predictors, which is appropriate for validation and small destination
blocks. A production full-network driver must supply group state through a
replayable provider or cache with an explicit memory budget; retaining all
1,898 naive routing arrays would still require at least 12.9 GiB in addition to
the 3.56 GiB assignment and compiler state.

## Numerical validation

Two independent comparisons were added:

- A nonlinear synthetic measurement objective split across two groups,
  including a positive-frozen measurement offset. The streamed value,
  prediction, and gradient match monolithic JAX autodiff within `2e-6`.
- The package's real fixed-routing loader on the simple example, split into its
  actual destination groups. The streamed value and complete OD gradient match
  the monolithic fixed-routing calculation within float32 tolerances (`2e-5`
  for the value and `3e-5` for the gradient).

The focused fixed-routing, frozen-cell, and streaming suite has 21 passing
tests. Duplicate or overlapping global parameter coordinates, invalid indices,
wrong prediction shapes, and dtype mismatches are rejected.

## Feasibility implications

Phase 1 measured 6.06 s for a warm forward load and 11.86 s for ordinary warm
value-and-gradient on the representative 465-cell full-network group. A
two-pass streamed derivative necessarily recomputes each destination after the
complete prediction is available. A first central estimate is therefore about
17.9 s per group (one forward pass plus one forward/reverse pass), excluding
routing preparation and Python orchestration.

At 1,898 groups this gives roughly 9.4 hours per full-network objective and
gradient evaluation on the measured local CPU. This extrapolation is crude—the
destination groups vary in OD count, while the dynamic-programming graph size
is broadly similar—but it is sufficient to reject a purely sequential local
implementation as an estimation engine. Fixed-routing preparation extrapolated
from 36.53 s for the representative group would add roughly 19.3 hours if done
serially without persistence or parallelism.

The two-pass method solves the correctness and global-autodiff-memory problem;
it does not solve runtime. The next measured phase should therefore focus on:

1. a replayable, memory-bounded destination provider that prepares or loads one
   routing block at a time;
2. representative timings across the six identified group sizes;
3. direct boarding/alighting link-to-measurement accumulation to remove the
   121 s global strict-mapping setup where possible;
4. rematerialized or explicit-adjoint loading, followed by process-level
   destination parallelism estimates.

No full routing cache, full operator, full Jacobian, or MAP optimization was
attempted in this phase.
