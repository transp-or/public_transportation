# Gravity objective-and-gradient performance

This public synthetic benchmark exercises 128 OD cells and 96 measurements in
float32 on the CPU. It compares dense and JAX BCOO measurement operators at 3,
7, 15, and 31 fitted parameters. The source is
`benchmark_gravity_performance.py`; complete phase timings and memory fields are
stored in `gravity_performance.json`.

The benchmark validates batched-forward and adjoint gradients numerically before
reporting timings. Values are machine-dependent and are not test thresholds.

| Representation | Parameters | Warm forward routing (ms) | Warm transpose routing (ms) | Warm batched forward (ms) | Warm adjoint (ms) | Fastest observed |
|---|---:|---:|---:|---:|---:|---|
| dense | 3 | 0.006 | 0.005 | 0.066 | 0.051 | adjoint |
| dense | 7 | 0.034 | 0.015 | 0.412 | 0.075 | adjoint |
| dense | 15 | 0.009 | 0.006 | 0.110 | 0.066 | adjoint |
| dense | 31 | 0.006 | 0.006 | 0.095 | 0.168 | batched forward |
| BCOO | 3 | 0.040 | 0.010 | 0.069 | 0.083 | batched forward |
| BCOO | 7 | 0.013 | 0.005 | 0.083 | 0.110 | batched forward |
| BCOO | 15 | 0.013 | 0.005 | 0.101 | 0.076 | adjoint |
| BCOO | 31 | 0.032 | 0.005 | 0.110 | 0.087 | adjoint |

Compilation took approximately 0.20--0.33 seconds per strategy after separate
tracing and lowering. First compiled executions took 0.5--2.7 ms. The maximum
batched-forward/adjoint gradient difference was below the configured float32
tolerance in every case. Dense operator storage was 49,152 bytes; BCOO storage
was 12,672 bytes.

The process peak-RSS metric is cumulative and therefore useful as an upper bound,
not as an allocation attribution for an individual case. The JSON report includes
both peak RSS and observed peak growth where the operating system exposed it.

## Automatic-strategy conclusion

The winner is not determined by parameter count alone at this size, and several
differences are close to timer noise. These results do not justify increasing the
current eight-parameter limit above which automatic selection avoids materializing
the batched demand Jacobian. That limit remains a conservative large-network
memory guard. For counts through eight, automatic mode benchmarks both compiled
candidates and selects the lower measured warm time. Larger-network evidence is
required before raising the limit.

## Compilation-cache interpretation

Each compiled strategy reports one in-process compiled-kernel miss and one hit per
warm or changed-parameter reuse. The changed-parameter execution verifies that
parameter values are dynamic and do not trigger a new compilation. JAX does not
expose authoritative persistent-cache hit counters through its public in-process
API. Consequently the benchmark labels persistent hits as requiring a fresh-process
populate/reuse protocol instead of inferring a hit from elapsed time.
