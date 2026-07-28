# Fixed-routing optimization baseline

This baseline captures the current dynamic-routing implementation on
`simple_example_02` before routing precomputation is introduced.

Regenerate the machine-readable results from the repository root:

```bash
uv run python benchmarks/benchmark_fixed_routing_baseline.py \
  --repeats 5 \
  --output-json benchmarks/fixed_routing_baseline.json \
  --output-npz benchmarks/fixed_routing_baseline.npz
```

The JSON file records dimensions, numerical summaries, array digests, and
machine-dependent timings for both dynamic and cached routing. The NPZ file stores
the complete dynamic-routing parameter, link-flow, measurement-prediction, and
gradient arrays for three deterministic probes.

Pytest compares future implementations with the NPZ reference using floating-point
tolerances. The benchmark also checks cached-routing equivalence before reporting
its speedup. It deliberately does not assert timings.
