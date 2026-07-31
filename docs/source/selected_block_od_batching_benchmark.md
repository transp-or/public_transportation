# Selected-block OD batching benchmark

The benchmark uses `simple_example_02` and creates a generic 512-column
synthetic block by repeating public-example origins within one destination
group. It constructs no global operator and runs no estimation.

The 2026-07-31 regression investigation added explicit JAX phase diagnostics
and made graph arrays dynamic arguments instead of closure constants. On the
public example, a fresh-process one-variable block completed in 0.093--0.194 s;
automatic mode took 0.093 s, including 0.068 s compilation and 0.0010 s
execution. Captured array constants and sparse differences were both zero, and
warm loading was below 0.7 ms. A 64-variable repeated-origin block completed in
0.141--0.215 s with execution below 0.004 s. Machine-readable reports are in
`benchmarks/selected_block_jax_phases_one_variable.json` and
`benchmarks/selected_block_jax_phases.json`.

The separated timings identify staging and compilation, rather than device
execution, as the dominant public cold cost. Timings are benchmark evidence,
not unit-test thresholds.

| Requested batch | Effective columns | Graph evaluations | Cold numerical time | Peak estimate | Maximum difference |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 512 | 0.361 s | 1.25 MB | 0 |
| 2 | 2 | 256 | 0.232 s | 1.27 MB | 0 |
| 4 | 4 | 128 | 0.169 s | 1.29 MB | 0 |
| 8 | 8 | 64 | 0.127 s | 1.34 MB | 0 |
| automatic | 512 | 1 | 0.101 s | 7.31 MB | 0 |

Automatic batching is 3.58 times faster than the batch-1 reference for the
numerical phase. Measurement mapping preparation remains two passes for every
configuration, candidate and accepted-entry counts agree, and every sparse
value is exactly equal. Warm disk loads range from 0.51 to 0.54 ms.

Machine-readable configuration, phase timings, counters, memory estimates,
correctness results, and environment information are stored in
`benchmarks/selected_block_od_batching_simple_example_02.json`.

The exact command was:

```text
MPLCONFIGDIR=/tmp/public-transportation-mpl-cache \
UV_CACHE_DIR=/tmp/public-transportation-uv-cache \
uv run --frozen python benchmarks/benchmark_support_preflight.py \
  --mode streaming-exact-support \
  --benchmark-od-batching \
  --synthetic-od-columns 512 \
  --output benchmarks/selected_block_od_batching_simple_example_02.json \
  --checkpoint-directory /tmp/public-od-batch-checkpoint-20260730e \
  --block-cache-directory /tmp/public-od-batch-cache-20260730e \
  --selected-support-directory /tmp/public-od-batch-support-20260730e \
  --maximum-variables-per-block 16 \
  --od-chunk-size 1 \
  --measurement-chunk-size 64 \
  --mapped-edge-chunk-size 256 \
  --check
```
