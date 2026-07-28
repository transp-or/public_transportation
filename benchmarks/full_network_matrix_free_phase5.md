# Full-network matrix-free feasibility: Phase 5

## Outcome

This phase implements an exact fixed-routing adjoint, compares it with ordinary
JAX reverse mode and whole-loader rematerialization, and derives a conservative
process-parallel server model. The private TPG scenario was read-only and no MAP
optimization was run.

## Explicit adjoint

For fixed link probabilities, forward loading is linear. If `c[l]` is a
link-flow cotangent, the node adjoint in reverse topological order is

```text
a[i] = sum over outgoing links l of P[l] * (c[l] + a[head[l]])
```

The derivative with respect to initial node injection is `a`; the free-demand
gradient is obtained through the existing injection scatter. The custom VJP
returns no probability derivative by design and is exposed only through an
explicitly named fixed-routing API. Its reverse state is one node vector plus
the link cotangent, not the retained state of the complete forward scan.

Tests compare the custom VJP with ordinary autodiff on the real package loader
using nonuniform demand and link cotangents. Values agree exactly and gradients
agree within `2e-5` float32 tolerances.

## Representative full-network comparison

The same 465-free-cell destination group was measured in isolated processes.

| Loader derivative | Warm forward (s) | Warm value--gradient (s) | RSS after V\&G (GiB) | Process peak (GiB) |
|---|---:|---:|---:|---:|
| Ordinary reverse mode | 6.075 | 12.093 | 16.21 | 17.85 |
| Whole-loader rematerialization | 6.018 | 8.173 | 17.54 | 17.54 |
| Explicit adjoint | 5.961 | 6.690 | 13.91 | 15.21 |

All three produced objective 29,840,308, gradient norm 10,712.1973,
36,732 nonzero predictions, and prediction sum 930.000122. The explicit adjoint
is 1.81 times faster than ordinary reverse mode and reduces the value-gradient
checkpoint by 2.30 GiB. It is only 0.73 s slower than the forward pass alone.

Whole-loader rematerialization is rejected: it is slower than the explicit
adjoint and did not reduce observed process memory on this backend.

## Revised full-network estimate

The exact two-pass streamed calculation requires one forward accumulation pass
and one recomputed value/adjoint pass per destination. Combining the six-sample
median forward time (5.991 s) with the representative explicit-adjoint time
(6.690 s) gives 12.681 s per group, or 6.69 hours for 1,898 groups sequentially.
This improves the previous ordinary-autodiff estimate of 9.55 hours, but remains
far too slow for local MAP estimation.

## Conservative server model

The measured custom-adjoint process peak is 15.21 GiB. Treating all JAX state as
replicated and adding 20% headroom requires about 18.25 GiB per worker. Idealized
two-pass times are:

| Workers | Minimum memory with headroom | Idealized evaluation time |
|---:|---:|---:|
| 4 | 73 GiB | 1.67 h |
| 8 | 146 GiB | 50 min |
| 12 | 219 GiB | 33 min |
| 16 | 292 GiB | 25 min |
| 24 | 438 GiB | 17 min |
| 32 | 584 GiB | 13 min |

These are optimistic lower bounds. The local JAX process used 12 logical CPUs;
running many such workers without limiting per-process threads would
oversubscribe the server. Host assignment arrays might be shared with fork or
memory mapping, but JAX device buffers are not credited as shareable. The target
server must measure thread counts and memory bandwidth before selecting a worker
count. The machine-readable assumptions are in
`benchmarks/full_network_custom_adjoint_server_model.json`.

## Decision

The explicit adjoint is adopted as the viable bounded derivative primitive, but
not automatically selected in estimation pipelines yet. Full-network execution
still requires process-level destination distribution and target-server thread
scaling. The next phase should build the worker protocol and benchmark controlled
thread counts on the actual server; local extrapolation alone cannot establish
parallel efficiency.
