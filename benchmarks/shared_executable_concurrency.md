# Shared-executable CPU concurrency

This benchmark uses a deterministic public DAG with 32,768 nodes, 131,062
links, two destination groups per shard, and 16 identical warm tasks. It
compiles one executable, warms it, and excludes tracing, lowering, and
compilation from batch timings.

```bash
MPLCONFIGDIR=/tmp/public-transportation-mpl-cache \
XDG_CACHE_HOME=/tmp/public-transportation-xdg-cache \
UV_CACHE_DIR=/tmp/public-transportation-uv-cache \
uv run --frozen python benchmarks/benchmark_shared_executable_concurrency.py \
  --tasks 16 --nodes 32768 \
  --output benchmarks/shared_executable_concurrency.json
```

| Workers | Batch wall (s) | Median shard (s) | Shards/s | Process CPU | Peak RSS |
|---:|---:|---:|---:|---:|---:|
| 1 | 13.400 | 0.835 | 1.194 | 2.46 cores | 425 MB |
| 2 | 8.930 | 1.117 | 1.792 | 4.68 cores | 431 MB |
| 4 | 4.609 | 1.151 | 3.471 | 8.05 cores | 437 MB |

All modes used the same compiled executable, compiled zero times during the
timed batch, and produced a maximum numerical difference of zero. Dispatch and
host transfer stayed below 0.1 ms; virtually all latency was synchronized
device execution. Therefore this CPU backend does not serialize concurrent
calls and no material Python/JAX entry lock was observed. CPU use above 100%
for one caller proves that XLA already uses internal parallelism;
`threads_per_worker=1` does not control it.

Four callers increased median latency by 38% but improved aggregate throughput
by 2.91×. This differs from the private full-network observation, where four
callers roughly doubled latency without improving throughput. The combined
evidence points to workload- or machine-specific contention—most plausibly
memory bandwidth, cache/NUMA effects, or competing XLA pools—not a universal
backend serialization or the Python GIL. The public benchmark cannot identify
which hardware resource saturates on Jed.

The fixed-shape batched experiment produced 1.226, 1.200, and 1.203 equivalent
shards/s for batch sizes one, two, and four. CPU use stayed close to 2.5 cores
and runtime grew almost linearly with the batch width. On this machine, making
the existing destination-group dimension larger does not expose additional
parallelism; four concurrent calls are substantially faster. The batched
production path consequently remains explicit and experimental.

## Controlled external threads and recognized XLA controls

JAX and jaxlib were both 0.11.0. Querying this exact backend with
`XLA_FLAGS=--help` confirmed these relevant controls:

- `--xla_cpu_multi_thread_eigen`;
- `--xla_cpu_parallel_codegen_split_count` (compilation only);
- `--xla_force_host_platform_device_count`;
- `--xla_cpu_use_onednn`, `--xla_cpu_use_xnnpack`, and experimental fusion
  controls that change generated implementation and are not generic thread
  limits.

No recognized flag provides a reliable per-executable CPU-thread count. In
particular, obsolete `intra_op_parallelism_threads` recipes were not used.
With `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` and the recognized
`--xla_cpu_multi_thread_eigen=true`, one/two/four callers achieved
1.174/1.735/3.226 shards/s and used 2.45/4.53/7.61 effective cores. These remain
qualitatively consistent with the default results; modest run-to-run variation
does not change the ranking. The controlled external variables do not
govern this XLA dynamic program.

## Explicit logical CPU devices

A separate fresh process exposed two logical CPU devices and explicitly used
`pmap`. The output shards were placed on `cpu:0` and `cpu:1`, matched the
single-device result exactly, used 4.77 effective cores, and achieved 1.852
shards/s. That is essentially the ordinary two-caller throughput and is far
below four callers on the public workload. The XLA help text also states that
all forced host devices share one thread pool and may add context-switching
overhead. Multi-device CPU routing is therefore not a default.

The production default remains the conservative single worker. Explicit worker
counts remain authoritative and memory/CPU admission remains enforced. No
portable target-independent throughput-effective count can be inferred from
these conflicting public and private results, so the reported
`throughput_effective_worker_count` remains null rather than inventing a value.
