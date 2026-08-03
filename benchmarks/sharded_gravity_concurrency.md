# Sharded gravity operator concurrency benchmark

This public synthetic CPU benchmark reproduces the workload structure reported
by the private full-network pilot without using private data. It uses an
8,192-node DAG, maximum out-degree four, 32 destination groups, eight persisted
routing shards, 256 free OD cells, and 256 measurements. Times are synchronized
warm complete products on the development machine and are not full-network
runtime projections.

| Execution | Width | Matvec (s) | Rmatvec (s) | Matmat, 3 columns (s) | Matvec effective cores |
|---|---:|---:|---:|---:|---:|
| Aggregate scan | 1 shard/batch | 0.0338 | 0.0397 | 0.1015 | 0.98 |
| Aggregate scan | 2 shards/batch | 0.0313 | 0.0336 | 0.1000 | 0.98 |
| Aggregate scan | 4 shards/batch | 0.0345 | 0.0301 | 0.0976 | 0.93 |
| Aggregate scan | 8 shards/batch | 0.0281 | 0.0285 | 0.0952 | 0.98 |
| Concurrent scan | 1 worker | 0.0370 | 0.0402 | 0.1014 | 0.97 |
| Concurrent scan | 2 workers | 0.0205 | 0.0225 | 0.0667 | 1.55 |
| Concurrent scan | 4 workers | 0.0147 | 0.0149 | 0.0314 | 2.28 |
| Concurrent scan | 8 workers | 0.0139 | 0.0121 | 0.0224 | 2.74 |

Eight concurrent workers improved warm forward throughput by 2.01 times over
the best aggregate scan. Four workers improved it by 1.91 times. In contrast,
the vectorized-group experiment exposed 3.6--5.1 effective cores but made the
forward product three to four times slower on the larger synthetic case because
it materialized per-group link-flow intermediates. It remains an explicit
experimental strategy rather than the default.

The production recommendation is therefore `group_execution_strategy="scan"`
with `shard_execution_strategy="concurrent"`. Concurrency is bounded by the
explicit user limit, detected CPU count, routing-shard count, and
`maximum_concurrent_routing_bytes`. When no explicit concurrency is supplied,
the conservative requested ceiling is four. Accumulation remains in canonical
shard order, so concurrent completion does not alter deterministic results.

Run the benchmark with:

```bash
uv run python benchmarks/benchmark_sharded_gravity_operator.py \
  --nodes 8192 --maximum-out-degree 4 \
  --destination-groups 32 --groups-per-shard 4 \
  --od-cells 256 --measurements 256 \
  --operator-batch-sizes 1 2 4 8 \
  --group-execution-strategies scan vectorized \
  --shard-execution-strategies aggregate concurrent \
  --operator-concurrencies 1 2 4 8
```

The JSON report includes a clear regression warning unless concurrent scan
improves warm forward throughput by at least ten percent.
