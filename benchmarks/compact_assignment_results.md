# Compact assignment benchmark

Measured on 2026-07-23 with the JAX CPU backend. Each warmed value is the
median of nine evaluations after compilation. JAX caches were cleared before
compiling each full and compact path. The benchmark verifies link-flow and
gradient equivalence before accepting a timing result.

Run the benchmark with:

```bash
.venv/bin/python benchmarks/benchmark_compact_assignment.py --example all --repeats 9
```

## Warmed execution results

| Example | Pattern | Active OD | Destination groups | Forward full / compact | Forward speedup | Gradient full / compact | Gradient speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| 01 | all free | 6 / 6 | 2 / 2 | 0.280 / 0.266 ms | 1.05x | 0.540 / 0.563 ms | 0.96x |
| 01 | 90% distributed zero | 1 / 6 | 1 / 2 | 0.260 / 0.148 ms | 1.75x | 0.551 / 0.344 ms | 1.60x |
| 01 | destination-concentrated zero | 3 / 6 | 1 / 2 | 0.279 / 0.146 ms | 1.91x | 0.553 / 0.337 ms | 1.64x |
| 02 | all free | 72 / 72 | 7 / 7 | 11.407 / 11.467 ms | 0.99x | 15.469 / 15.562 ms | 0.99x |
| 02 | 90% distributed zero | 7 / 72 | 5 / 7 | 11.528 / 8.227 ms | 1.40x | 15.573 / 11.476 ms | 1.36x |
| 02 | destination-concentrated zero | 18 / 72 | 3 / 7 | 11.388 / 4.885 ms | 2.33x | 15.654 / 7.034 ms | 2.23x |

`Active OD` and `Destination groups` are shown as compact / full counts.

## Interpretation

The all-free cases have effectively identical warmed performance, showing that
the compact layer itself adds no material overhead. Removing OD coordinates
without removing all their destination groups gives a moderate improvement.
The largest benefit occurs when frozen-zero cells eliminate complete
destination groups, because the assignment performs one graph traversal per
surviving destination. Compilation-time differences were small and varied in
direction; the optimization should therefore be justified by repeated forward
and gradient evaluations rather than compilation time.

These values are machine-dependent and should be treated as representative,
not as performance requirements.
