# TPG cached-operator ML/MAP compilation benchmark

Environment: JAX/JAXLIB 0.8.1, CPU backend, Apple arm64, float32, cached
3,690 x 15,772 BCOO operator with 158,219 canonical sorted unique entries.

The previously attributed 29.484 s compilation delay was not XLA compilation.
The fixed-routing preparation had launched asynchronous device work even though a
valid operator cache entry was subsequently used; the first objective result
synchronized that unused work. Cache validation now occurs before routing
preparation. The isolated objective traces in about 0.026 s, lowers in about
0.011 s, compiles cold in 0.120--0.129 s, and first executes in about 0.0014 s.

## Sparse objective variants

| Variant | Trace (s) | Compile (s) | Warm value+gradient (s) | Objective difference | Max gradient difference |
|---|---:|---:|---:|---:|---:|
| BCOO matvec | 0.0229 | 0.1186 | 0.001073 | 0 | 0 |
| Direct indexed scatter | 0.0063 | 0.1155 | 0.000944 | 0 | 0 |
| Sorted segment sum | 0.0046 | 0.1116 | 0.000973 | 0 | 0 |

BCOO remains the compact persistent representation. The stable package-level
objective receives values, rows, columns, observations, baseline demand, and
offsets as dynamic arrays and performs direct indexed aggregation. This avoids
reconstructing a sparse object inside the objective and gives the best measured
warm time. Changing parameter values, observations, or optimizer iteration count
does not change the compiled signature when shapes, dtypes, backend, JAX/JAXLIB,
and compilation-relevant configuration remain unchanged.

## Fresh-process optimization

The explicit `prepare_ml_objective` and `compile_ml_objective` API was used once
per process, and the resulting executable was passed to `run_ml`. Times below
include normal TPG scenario and assignment preparation in the total.

| Iterations | Cache state | Compile/load (s) | Warm value+gradient (s) | Optimizer (s) | Process (s) | Final objective |
|---:|---|---:|---:|---:|---:|---:|
| 1 | cold | 0.1204 | 0.000975 | 0.303 | 5.800 | 55243.4375 |
| 1 | populate | 0.1285 | 0.000996 | 0.292 | 5.845 | 55243.4375 |
| 1 | hit | 0.0070 | 0.000983 | 0.297 | 5.167 | 55243.4375 |
| 5 | cold | 0.1231 | 0.001001 | 2.878 | 8.377 | 49635.4922 |
| 5 | populate | 0.1194 | 0.001021 | 2.604 | 8.167 | 49635.4922 |
| 5 | hit | 0.0075 | 0.000958 | 2.539 | 7.622 | 49635.4922 |
| 20 | cold | 0.1260 | 0.000991 | 5.805 | 11.623 | 49140.4258 |
| 20 | populate | 0.1260 | 0.000972 | 5.777 | 11.401 | 49140.4258 |
| 20 | hit | 0.0070 | 0.000996 | 5.820 | 11.047 | 49140.4258 |

The persistent JAX compilation cache is enabled before JAX work through
`PUBLIC_TRANSPORTATION_JAX_COMPILATION_CACHE_DIR`, or explicitly through
`configure_jax_compilation_cache`. Both eligibility thresholds default to zero
and can be controlled with `PUBLIC_TRANSPORTATION_JAX_CACHE_MIN_COMPILE_SECONDS`
and `PUBLIC_TRANSPORTATION_JAX_CACHE_MIN_ENTRY_BYTES`.

On this JAX 0.8.1 Apple CPU runtime, loading cached AOT entries emits a target
machine feature warning even though the reported compile and host feature sets
are both empty. All hit-process executions completed and reproduced cold-process
objectives and gradients exactly. This appears to be a runtime diagnostic rather
than a package cache-key mismatch, but should be rechecked after JAX upgrades.

Raw machine-readable results are in `tpg_ml_compilation*.json` and
`tpg_sparse_objective_variants.json`.
