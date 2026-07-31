# Fixed-routing linear estimation specification

This document defines the mathematical and data contracts for the
fixed-routing linear estimation mode. It covers dense reference calculations,
matrix-free products, and persistent explicit sparse operators for scalable
CPU solvers.

## Estimation modes

The package distinguishes two estimation modes.

1. **Estimated routing.** The routing parameter is estimated jointly with OD
   demand. Assignment is nonlinear in the estimated parameters, and the
   existing likelihood-based implementation is used. This mode is intended for
   small problems.
2. **Fixed-routing linear estimation.** Routing is prepared once from a
   user-supplied routing parameter. The free OD entries are then estimated from
   a linear measurement model using explicitly selected weights, bounds, and
   regularization. This mode does not use the negative-binomial likelihood or
   estimate the routing parameter.

The modes may share scenario preparation, OD ordering, fixed-demand handling,
routing preparation, and measurement mapping. They do not share an
optimization objective.

## Linear measurement model

Let

- \(n\) be the number of free OD/time cells;
- \(m\) be the number of observations;
- \(x\in\mathbb{R}^n\) be free OD demand;
- \(A\in\mathbb{R}^{m\times n}\) be the fixed-routing measurement operator;
- \(c\in\mathbb{R}^m\) be the measurement contribution of positive fixed
  demand; and
- \(y\in\mathbb{R}^m\) be the observed counts.

The predicted measurements are

\[
\widehat y(x)=Ax+c.
\]

Columns of \(A\) use the canonical free-OD ordering. Rows use the canonical
observation ordering. Both orderings and their provenance must be retained with
the operator.

### Units

- An entry of \(x\) or fixed demand is a passenger count for one OD/time cell.
- An entry of \(y\), \(\widehat y\), or \(c\) is a passenger count in one
  measurement record.
- An entry \(A_{ji}\) is the expected contribution to measurement \(j\) per
  passenger in free OD/time cell \(i\); it is dimensionless.
- Observation weights and regularization scaling determine the units of their
  residuals and must be recorded explicitly.

### Fixed demand

Free OD/time cells appear as variables in \(x\). Fixed cells never appear in
\(x\) or as columns of \(A\).

- A fixed-zero cell, including a structural zero, contributes nothing.
- A positive fixed cell is assigned under the same fixed routing as the free
  demand. Its complete measurement contribution is included in \(c\).

Consequently,

\[
c=\widehat y(0),
\]

where zero denotes zero demand in every free cell while positive fixed demand
is retained.

## Weighted least-squares objective

Let \(w\in\mathbb{R}^m\) contain user-supplied observation precision weights,
with \(w_j>0\), and let \(W=\operatorname{diag}(w)\). The package accepts the
precision weights \(w_j\), not their square roots. It constructs

\[
r_D(x)=W^{1/2}(Ax+c-y)
\]

and

\[
f_D(x)=\frac{1}{2}\lVert r_D(x)\rVert_2^2.
\]

The factor \(1/2\) is part of the objective convention. Unit weights mean
ordinary unweighted least squares. The implementation requires finite,
strictly positive weights; omitting observations or accepting zero weights may
be added later as an explicit data-processing policy.

Weights have a statistical inverse-variance interpretation only when the user
constructs them from an observation-error model. Otherwise they are numerical
confidence weights, and output diagnostics must not be labeled confidence or
posterior intervals.

The package reports both raw residuals

\[
r_{\mathrm{raw}}(x)=Ax+c-y
\]

and weighted residuals \(r_D(x)\).

## Bounds

Physical OD demand satisfies componentwise bounds

\[
\ell\le x\le u.
\]

Bounds are expressed in passenger counts. The recommended basic choice is
\(\ell=0\) and no finite upper bound, but the library does not silently insert
this choice. Bounds are part of the problem specification, and heterogeneous
bounds are supported.

## Explicit regularization

Regularization is an ordered collection of named linear residual blocks. Block
\(k\) contains a linear operator
\(L_k\in\mathbb{R}^{p_k\times n}\), a target
\(r_k\in\mathbb{R}^{p_k}\), and an explicitly supplied strength
\(\lambda_k\ge 0\). Its residual and objective contribution are

\[
r_{R,k}(x)=\sqrt{\lambda_k}(L_kx-r_k),
\qquad
f_{R,k}(x)=\frac{1}{2}\lVert r_{R,k}(x)\rVert_2^2.
\]

The complete problem is

\[
\min_{\ell\le x\le u}
\left[
\frac{1}{2}\lVert W^{1/2}(Ax+c-y)\rVert_2^2
+\frac{1}{2}\sum_k\lambda_k\lVert L_kx-r_k\rVert_2^2
\right].
\]

Equivalently, it is

\[
\min_{\ell\le x\le u}
\frac{1}{2}\left\lVert
\begin{bmatrix}
W^{1/2}A\\
\sqrt{\lambda_1}L_1\\
\vdots
\end{bmatrix}x-
\begin{bmatrix}
W^{1/2}(y-c)\\
\sqrt{\lambda_1}r_1\\
\vdots
\end{bmatrix}
\right\rVert_2^2.
\]

No regularization block or strength is implicit. The implementation supports
these explicit selections:

- `none`;
- `ridge_to_prior`, with \(L=I\) and \(r=x_0\); and
- `scaled_ridge_to_prior`, with \(L=S^{-1}\) and
  \(r=S^{-1}x_0\).

If the user supplies no selection, the package may analyze the unregularized
operator and recommend choices, but it must not apply a recommendation without
an explicit selection.

## Prior demand and internal solver variables

Let \(x_0\in\mathbb{R}^n\) be prior free OD demand in the same physical units
and ordering as \(x\). The public problem, results, bounds, and diagnostics are
defined in terms of \(x\).

A solver may internally use a scaled deviation \(d\):

\[
x=x_0+Sd,
\]

where \(S\) is a positive diagonal scaling matrix. The unscaled deviation uses
\(S=I\). Solver-variable bounds are

\[
S^{-1}(\ell-x_0)\le d\le S^{-1}(u-x_0).
\]

This affine change of variables does not change the physical problem or cure
rank deficiency. It provides an initial point \(d=0\) at the prior and can
improve variable scaling. Every result must be transformed back to physical OD
demand before it is exposed to the user.

## Objective components and derivatives

The reported objective is

\[
f(x)=f_D(x)+\sum_k f_{R,k}(x).
\]

Its physical-demand gradient is

\[
\nabla f(x)=
A^\mathsf{T}W(Ax+c-y)
+\sum_k\lambda_kL_k^\mathsf{T}(L_kx-r_k).
\]

Implementations must expose separate data and regularization contributions in
their results. They must not form \(A^\mathsf{T}A\) as the primary solution
method; dense reference solutions use QR or SVD, and iterative solvers use
forward and transpose operator products.

## Quality and identifiability contract

Rank deficiency is a property to diagnose, not to hide with an implicit
penalty. The small-example implementation will report at least:

- numerical rank and rank tolerance;
- singular values or an equivalent condition diagnostic;
- objective decomposition;
- raw and weighted residual norms;
- bound activity and an independently evaluated KKT residual;
- deviation from prior demand; and
- data-versus-regularization resolution on variables not fixed by active
  bounds.

If weights are merely numerical and regularization is not a probabilistic
prior, these outputs are labeled resolution, sensitivity, and prior reliance.
Statistical uncertainty terminology is reserved for a separately declared
probabilistic interpretation.

## Operator backends and persistence

`SparseOperatorSelectionConfig` provides three explicit modes:

- `matrix_free` evaluates assignment for every forward and transpose product;
- `sparse` constructs or loads the explicit measurement operator and converts
  it once to persistent SciPy CSR and CSC storage; and
- `auto` selects sparse storage only when its estimated memory use is safe and
  the expected product count justifies construction. Otherwise it retains the
  matrix-free backend.

The preflight decision reports logical and dense sizes, estimated nonzeros,
estimated CSR/CSC bytes, the memory budget, expected product count, selected
mode, and a human-readable reason. An explicit `sparse` request is rejected
before allocation when it exceeds the configured budget. The policy does not
select a dense matrix merely from its logical dimensions.

The persisted artifact remains the canonical BCOO fixed-routing measurement
operator. Its cache key covers assignment, graph, measurement mapping, compact
and OD layouts, routing parameter, representation, dtype, zero tolerance,
schema, and package version. Writes are atomic and loads validate provenance,
dimensions, indices, values, and offsets. A valid cache hit is loaded without
preparing fixed routing. BCOO data are transferred to the host and converted
to canonical CSR and CSC exactly once per prepared backend. Repeated solver
products therefore perform only SciPy sparse products; they do not call JAX,
retrace, convert formats, or evaluate routing.

Preparation diagnostics distinguish cache lookup/load, fixed-routing
preparation, operator construction, host sparse conversion, and total time.
They also report the realized nonzero count and persistent CSR/CSC storage.

## Diagonal solver scaling

TRF/LSMR optionally uses an exact diagonal column scaling computed from

\[
\operatorname{diag}(A^\mathsf{T}WA)
+\sum_k\lambda_k\operatorname{diag}(L_k^\mathsf{T}L_k).
\]

This does not form a normal matrix or change the physical objective. It only
changes internal solver coordinates and combines with the configured variable
scales. Structurally zero or near-zero columns retain their original scale.
The implementation currently requires explicit dense or sparse measurement
and regularization operators; it is not silently approximated for a
matrix-free operator. Preconditioner construction time is reported separately.

## Product accounting

The TRF/LSMR result separates products performed while constructing solver
coordinates, products requested by SciPy's combined TRF/LSMR phase, and final
objective/KKT products. SciPy's public `lsq_linear` interface does not expose a
stable boundary between individual Krylov, trust-region, and bound-optimality
calls, so those internal calls are reported as one solver phase rather than
assigned speculative labels. LSMR inherently needs paired forward and
transpose Krylov products; TRF additionally evaluates residuals and
optimality. This explains why one outer iteration can invoke many more than
one pair of products and why an explicit sparse backend is valuable.

Automatic regularization-strength selection remains outside the current scope.

## Package workflow for small problems

The supported high-level entry point is
`run_fixed_routing_linear_estimation`. It receives an unconfigured
`FixedRoutingLinearProblem` and a `FixedRoutingLinearEstimationConfig`. The
configuration requires one explicit regularization choice: `none`,
`ridge_to_prior`, or `scaled_ridge_to_prior`. A strength is required for either
ridge choice and is rejected for `none`.

For the current small-problem implementation, the workflow performs the
following operations as one validated unit:

1. computes a non-binding regularization recommendation;
2. applies the user's explicit regularization choice;
3. solves a dense reference problem;
4. solves the operator problem with TRF/LSMR;
5. verifies objective values and predictions between the two solvers;
6. computes exact small-problem estimate-quality diagnostics; and
7. optionally writes the versioned `FixedRoutingLinearResult` archive.

Scenario loading and backend preparation remain outside this workflow. Use
`prepare_fixed_routing_linear_measurement_backend` to perform memory-safe
selection, persistent cache reuse, and one-time CPU sparse conversion before
building the problem. This boundary allows dense, sparse, matrix-free, and
sharded sparse operators to use the same estimation API.

## Native sparse construction

The fixed-routing measurement builder supports a native BCOO representation.
For each routed OD column, it transfers only entries whose absolute value is
larger than the explicitly configured zero tolerance. It therefore does not
allocate the full measurement-by-OD matrix. At the linear-problem boundary,
the BCOO data and row/column coordinates are transferred directly to canonical
CSR storage without dense materialization.

The dense construction remains available as a small-problem reference. Tests
on both packaged examples require the native sparse and dense representations
to agree for forward products, transpose products, and the positive
fixed-demand measurement offset.

## Sparse operator persistence and reuse

Native sparse operators may be stored in a versioned, compressed archive and
reused through `load_or_prepare_fixed_routing_measurement_operator`. Cache file
names are hashes of the complete numerical provenance, including:

- schema and package versions;
- assignment and graph fingerprints;
- measurement-mapping fingerprint;
- OD and compact-layout fingerprints;
- fixed routing parameter;
- numeric type and matrix representation;
- operator dimensions; and
- sparse zero tolerance.

The sparse zero tolerance is part of the identity because changing it can
change both the stored coefficients and subsequent estimates. Chunk size is
not part of the identity because it changes construction scheduling but not
the mathematical operator.

Archives are loaded without pickle support. Metadata, dimensions, sparse index
bounds, numeric finiteness, non-negativity, offsets, and expected provenance
are validated before reuse. A missing, incompatible, or corrupt archive is
rejected and rebuilt atomically. Both packaged examples use this cache and
report whether their operator was constructed or loaded.

## Solver interface and small-problem comparison

`solve_fixed_routing_linear` is the solver-neutral entry point. Every backend
returns the same physical-demand solution, objective decomposition, gradient,
bound KKT diagnostics, status, iteration count, elapsed time, and available
operator-product counts. A backend reports unavailable metrics as missing; it
does not report a misleading zero. Backend-specific results remain available
as an explicit escape hatch.

The initially registered backends are:

- `trf_lsmr`, the sparse/operator implementation intended as the present
  scalable baseline; and
- `dense_reference`, the small-problem BVLS or SVD implementation used for
  independent verification.

`benchmark_fixed_routing_linear_solvers` runs distinct registered backends on
the identical configured problem. Its records compare objective value, raw and
weighted residual norms, KKT and feasibility residuals, iterations, elapsed
time, and forward/transpose product counts. Tests apply this comparison to both
packaged examples.

The 1991 paper by Bierlaire, Toint, and Tuyttens,
[“On iterative algorithms for linear least squares problems with bound
constraints”](https://doi.org/10.1016/0024-3795(91)90009-L), describes iterative
algorithm variants rather than defining one backend solely through its
citation. A package implementation therefore requires an explicit choice of
variant, initialization, active-set rules, linear subproblem method, and
stopping tests. Once those decisions are recorded, it can be added behind the
same solver contract and compared without changing the estimation workflow.

## Matrix-free measurement products

`MatrixFreeFixedRoutingMeasurementOperator` implements the same forward and
transpose protocol without storing dense, BCOO, or CSR coefficients. Its
forward product performs fixed-routing demand loading followed by measurement
aggregation. Free OD values are scattered into compact assignment coordinates;
positive fixed demand is excluded from the linear product and evaluated once
to obtain the separate fixed measurement offset.

Construction performs contract validation but does not compile either complete
product. When `fixed_compact_indices` is empty, the fixed offset is created as
an owned, read-only NumPy zero vector with the measurement dimension and
operator dtype; no active-demand vector, assignment, or JIT operation is used
for the offset. A nonempty positive-fixed layout retains the routed offset
calculation.

Forward and transpose compilation are lazy and independent. The first
`matvec` compiles only the forward product and subsequent calls reuse it; the
transpose is not compiled until the first `rmatvec`. An optional absolute
monotonic `preparation_deadline` is checked before and after offset preparation,
compilation, and execution. XLA compilation cannot be interrupted in-process,
so an indivisible operation may overshoot; a
`MatrixFreePreparationDeadlineError` then exposes diagnostics and prevents the
next expensive phase.

The `diagnostics` property reports the zero-offset path, positive-fixed count,
validation, routing, offset compilation and execution times, independent
forward/transpose compilation counts and times, execution counts and cumulative
times, and deadline/overshoot status. It performs no printing.

The transpose product applies reverse-mode differentiation to the linear
forward product. The assignment portion uses the package's explicit
fixed-routing custom adjoint, whose working state is node- and link-sized. The
measurement aggregation adjoint and free-coordinate selection are composed
with it automatically. Tests on both packaged examples require:

- agreement with dense and native sparse forward products;
- agreement with their transpose products;
- linearity for signed combinations of OD vectors;
- the adjoint identity
  \(\langle Ax,z\rangle=\langle x,A^\mathsf{T}z\rangle\);
- agreement of the positive fixed-demand offset; and
- exact zero-offset construction without assignment compilation when no
  positive fixed-demand cell exists;
- agreement of TRF/LSMR solutions, objectives, and predictions.

This operator removes storage of the measurement-by-OD matrix, but each
product still performs an assignment loading pass. The current high-level
small-problem workflow also performs dense reference and exact quality
analysis, both of which materialize operator information. Large applications
must therefore call the iterative solver and scalable quality diagnostics
without those small-problem reference operations.

## Scalable approximate quality diagnostics

`analyze_linear_estimate_quality_scalable` avoids operator materialization. It
reports sampled quantities with semantics that deliberately differ from the
exact small-problem diagnostics:

- the largest and a configurable number of smallest singular values are
  estimated with iterative operator methods;
- sampled near-zero singular values establish a **lower bound** on nullity and
  therefore an **upper bound** on rank;
- absence of a sampled near-zero value is not presented as a proof of full
  rank;
- the condition estimate and spectral convergence message are reported
  explicitly; and
- data-resolution diagonals and effective data degrees of freedom are
  estimated with reproducible Rademacher probes and iterative combined-Hessian
  solves.

The output records the random seed, requested and converged sample counts,
per-OD Monte Carlo standard errors, and a trace standard error. Data-informed
and regularization-dominated classifications require their uncertainty interval
to clear the corresponding threshold; otherwise the estimate is classified as
mixed. If a sampled null space is found without positive regularization, the
resolution solve is skipped and those cells are reported as weakly identified
rather than incorrectly assigning null-space reliance to the prior.

`run_fixed_routing_linear_estimation_scalable` combines explicit
regularization, TRF/LSMR, and these diagnostics using operator products only.
It does not run the dense reference solver or the exact recommendation
analysis. It currently returns an in-memory record; persistence will be added
only with a versioned approximate-diagnostic result schema.

## Interruptible block-coordinate MAP estimation

`BlockCoordinateMAPEstimator` provides the large-problem anytime solver for the
fixed-routing linear objective. An `ODBlock` contains only free OD columns; the
partition never cuts the time-expanded network. The default deterministic
partition follows destination group, time bin, and canonical free-column order,
then subdivides groups to satisfy hard variable and nonzero ceilings. Custom
partitions must cover every free column exactly once without overlap and cannot
contain frozen columns.

For current prediction (z=Ax+c), block (B) is evaluated using

\[
z_{\mathrm{trial}}=z+A_B(x_B^{\mathrm{trial}}-x_B).
\]

All other variables remain fixed. A candidate is accepted only when it is
finite, feasible, and non-increasing for the complete objective within the
configured numerical tolerance. Damping and backtracking are available;
rejection leaves the global state unchanged. Incremental prediction is compared
with a full product at configurable validation boundaries. The implemented
conditional objective supports separable quadratic priors. Required
cross-block prior or constraint terms are never silently dropped: unsupported
coupling is rejected.

The initial state and every accepted block or parallel batch are complete valid
approximate solutions. Durable journal entries contain replayable flow and
prediction deltas and are published using a temporary file, filesystem flush,
atomic replacement, and commit marker. Periodic compact checkpoints contain the
full current and best flow, prediction, objectives, diagnostics, schedule,
random state, and next position. Resume validates scenario, assignment, OD
layout, fixed demand, measurements, prior, routing, partition, solver, and
schema fingerprints.

Progress events distinguish exact, sampled, stale, and unavailable diagnostics.
They report objective improvement, projected-gradient measures, flow changes,
block and variable coverage, sweeps, elapsed time, checkpoint status, and
estimated remaining sweep time. Results distinguish convergence, configured
budgets, graceful interruption with an approximate solution, resource guards,
and numerical failure.

Block operators are independently constructible, persistable, releasable, and
retained under a bounded LRU policy. Independent construction can run in
parallel. For parallel solves, a deterministic conflict graph joins blocks that
share measurement support or declared prior/constraint coupling. Only blocks
in a conflict-free color batch are solved concurrently; their accepted deltas
are merged atomically in deterministic order. Overlapping proposals are not
treated as independent Gauss--Seidel updates. Explicit worker and native-thread
limits prevent oversubscription.

This solver is restricted to fixed routing and the supported exact conditional
objective. It is not a changing-routing or nonlinear assignment algorithm.

## Bounded streaming support preflight

Full-network block construction must not begin with
`plan_sharded_fixed_routing_operator`: that compatibility planner materializes
origin support and retains global per-column patterns and future construction
tasks. `run_support_preflight` is the bounded alternative. It first consumes a
structural `ODBlockPartition`, then processes one destination group at a time.
For each selected group it prepares fixed routing, discovers exact reachable
measurement rows in bounded origin chunks, reduces them immediately to group
and block summaries, checkpoints atomically, and releases group-local arrays.
It never retains an OD-by-measurement support matrix.

The public modes are `structural`, `sampled_exact_support`,
`streaming_exact_support`, and `exact_materialized_plan`. The last mode requires
explicit authorization and a conservative logical-support estimate below the
retained-state budget. It exists only for reviewed small compatibility tests;
large runs must use streaming mode. Sample selection is deterministic and
includes small, median, p95, and maximum structural groups. Sampled results
carry observed-range projections and assumptions, and are not sufficient by
themselves to authorize a full-network pilot.

`SupportPreflightBudget` enforces elapsed time, process RSS, group temporary
storage, retained summaries, support rows, nonzeros, and block-operator bytes.
A limit or `KeyboardInterrupt` returns a typed partial result and atomically
updates the checkpoint. Resume validates scenario, assignment, OD layout,
fixed demand, measurement mapping, routing, partition, configuration, and
schema fingerprints before skipping completed groups. Corrupt or incompatible
checkpoints are rejected.

Elapsed time has two distinct meanings. `cumulative_elapsed_seconds` retains
all committed and discarded support work across invocations, while
`current_invocation_elapsed_seconds` starts at zero for each call and is the
only value compared with that invocation's elapsed-time allowance. The legacy
`elapsed_seconds` field aliases cumulative elapsed time. Results also record
previous-invocation time, invocation count, allowance, bounded overshoot, stop
location, stop group, and time discarded when an incomplete group must restart.
Consequently, exhausting one invocation does not make the next invocation stop
immediately.

The schema-3 semantic fingerprint covers mode, deterministic selection and
sampling, support chunking and tolerance, persisted-support representation,
and schema. Authoritative scenario, assignment, OD, fixed-demand, measurement,
routing, and partition fingerprints remain separate and mandatory. Operational
policy has its own fingerprint and serialized provenance: elapsed/RSS/storage
ceilings, checkpoint and progress cadence, and deterministic worker/thread
counts may change on resume. A tighter policy is accepted only when retained
summaries and accepted blocks already satisfy it; otherwise resume fails before
new support work. Schema-2 checkpoints can be migrated only once using their
exact original configuration, after which policy changes use schema 3. Schema 1
is rejected with an explicit restart message.

After exact support discovery, `select_representative_block_ids` chooses the
smallest, median, p95, and largest observed blocks plus any explicit IDs.
`construct_selected_block_operators` estimates CSR, transpose, construction,
cache, and solver storage and rejects an unsafe block before invoking its
builder. It constructs no unselected block. `authorize_block_coordinate_pilot`
then returns either explicit rejection reasons or a conservative configuration;
it never starts estimation.

The operational command is:

```bash
uv run --frozen --extra dev python benchmarks/benchmark_support_preflight.py \
  --mode sampled-exact-support --check
```

It prints group progress and writes both partial and final JSON atomically.
Use `--checkpoint-directory` and `--resume` for restartable runs. Resource
limits and explicit destination-group IDs are command-line options; the
command has no estimation path.

## Resource-adaptive block-coordinate preflight

Large block-coordinate MAP runs should create a structural partition, run the
streaming support preflight, and then measure only authorized representative
blocks. `detect_machine_resources` measures available memory, logical and
physical CPUs, and cache storage. Applications with scheduler or container
limits may instead supply a `MachineResourceSnapshot`; this is also the
preferred way to obtain reproducible deployment tests.

`measure_representative_blocks` deterministically selects at most the requested
number of small, median, and large candidate blocks. It constructs only those
block operators, times forward and transpose products, records operator,
solver, checkpoint, and cache sizes, and releases each operator after
measurement. It does not call the global materialized-support planner.

Pass the measurements to `recommend_block_resources` with a
`ResourcePreflightConfig`. The profiles `laptop`, `workstation`, and `server`
use respectively 55%, 68%, and 78% of available memory as conservative total
job ceilings. `auto` selects one of these policies from measured memory. The
recommendation limits workers by both CPUs and the safety-adjusted measured
peak memory per worker, prevents thread oversubscription, checks estimated
cache storage, and reports block ceilings, worker allocation, expected memory
and cache use, sweep times, and uncertainty.

Automatic sizing is deliberately a proposal, not an implicit configuration.
The caller must construct an `AcceptedBlockResourceProposal` containing the
exact recommendation fingerprint. `apply_accepted_resource_recommendation`
then produces a hard-ceiling `BlockSizingConfig` and a
`BlockCoordinateMAPConfig` with the accepted worker and thread allocation.
Stale or modified recommendations are rejected. Users who do not accept the
proposal must supply explicit block ceilings and runtime allocation instead.

### Adaptive refinement before execution

After accepting the resource proposal, construct a `BlockResourceCostModel`
from the bounded preflight samples. `split_partition_for_resource_limits`
compares every proposed block with an `AdaptiveBlockSplitConfig` containing a
worker-memory limit, a block-runtime limit, or both. Unsafe blocks are bisected
deterministically until every child satisfies the limits. Splitting changes
only the free-column exposure of each operator: every route still uses the
complete assignment network. Child blocks conservatively retain the parent's
measurement support, and their union preserves every free OD column exactly
once.

The refinement must happen before estimation, so an unsafe parent operator is
never constructed. If even the minimum permitted block is unsafe,
`BlockResourceGuardError` stops the workflow before allocation. The procedure
never enlarges a block.

An adapted partition has a new authoritative fingerprint. Call
`fingerprints_for_adapted_partition` before constructing the estimator and use
the returned fingerprints for the run and all resumes. Estimator
initialization then checkpoints the complete revised schedule before the first
update. Checkpoints created for either the obsolete parent partition or a
different refinement are rejected. Thus a resumed run cannot silently revert
to an unsafe schedule.
