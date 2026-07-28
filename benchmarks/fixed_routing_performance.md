# Fixed-routing performance benchmark

Run the systematic benchmark from the repository root:

```bash
uv run python benchmarks/benchmark_fixed_routing_performance.py \
  --repeats 7 --theta 1 5 \
  --output benchmarks/fixed_routing_performance.json
```

It compares dynamic and cached routing for full and compact OD layouts. Each case
reports first and warm routing preparation, incremental cache storage, and
compilation/first-call and warm timings for forward assignment and the complete
likelihood gradient.

Numerical equivalence of link flows, objectives, and gradients is checked before
timings are reported. Timing values are intentionally not asserted by pytest.

Run the larger package-native Geneva validation with:

```bash
uv run python benchmarks/benchmark_fixed_routing_performance.py \
  --example geneva_gtfs --repeats 3 --theta 5 \
  --output benchmarks/geneva_fixed_routing_performance.json
```
