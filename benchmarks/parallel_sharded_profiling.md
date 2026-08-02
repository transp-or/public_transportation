# Parallel sharded fixed-routing profiling

Run with:

```bash
MPLCONFIGDIR=/tmp/public-transportation-mpl-cache \
XDG_CACHE_HOME=/tmp/public-transportation-xdg-cache \
UV_CACHE_DIR=/tmp/public-transportation-uv-cache \
uv run --frozen python benchmarks/benchmark_parallel_sharded_profiling.py
```

On the public `simple_example_02` scenario (7 destination groups, 5,846
links, four shards), a representative laptop run produced:

| Mode | Elapsed (s) | Diagnostics |
|---|---:|---:|
| serial, profiling disabled | 0.227 | 0 |
| two workers, profiling disabled | 0.00990 | 0 |
| two workers, profiling enabled | 0.00958 | 4 |

The two parallel results had identical masks and zero maximum probability
difference from the serial result. Diagnostics were ordered `[0, 1, 2, 3]`.
The profiled run was 0.000321 seconds faster (-3.24%) in this run, so profiling
overhead was below the timing noise of this very short example rather than a
measurable penalty.
The comparison is descriptive: in-process JAX warm-up makes the serial absolute
time unsuitable as a scaling claim, while the adjacent parallel modes isolate
the opt-in profiling overhead more usefully.

The profiled run separated host preparation, argument transfer, asynchronous
kernel dispatch, device synchronization, host transfer, slicing, validation,
shard persistence, coordinator manifest persistence, and cleanup. Enabled-link
fractions ranged from 0.453 for the padded final shard to 0.890; retained-domain
effective probability densities ranged from 0.847 to 0.907. Peak process RSS
was approximately 444 MB. The extra work is principally nonzero counting,
device/process metadata collection, and phase timestamping.
