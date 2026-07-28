# Geneva fixed-routing validation

The package-native `geneva_gtfs` snapshot was measured on the JAX CPU backend
using `float32`, fixed dispersion 5, and three warm repetitions. Complete
machine-readable results are in `geneva_fixed_routing_performance.json`.

| Layout | Active OD | Destination groups | Cache | Forward speedup | Likelihood-gradient speedup |
|---|---:|---:|---:|---:|---:|
| Full | 15,128 | 62 | 11.83 MiB | 4.06x | 3.03x |
| Compact | 96 | 18 | 3.43 MiB | 4.12x | 2.98x |

For the production compact path, warm forward time fell from about 0.449 s to
0.109 s and warm likelihood value-and-gradient time fell from about 0.537 s to
0.180 s. Warm routing preparation cost about 0.348 s, so it is recovered during
the second likelihood-gradient evaluation.

Dynamic and cached routing agreed within `3.06e-5` for link flows and `8.95e-7`
for the compact-layout gradient. A one-iteration MAP integration run completed
in about 1.59 s and stopped at the requested iteration limit.

These timings characterize this machine and are not pytest acceptance
thresholds. Pytest checks the committed report's dimensions, equivalence bounds,
and that both measured cached paths are faster than their dynamic counterparts.
