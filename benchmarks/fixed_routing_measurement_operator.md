# Fixed-routing measurement operator benchmark

Example: `simple_example_02`

- Active OD cells: 72
- Measurements: 8
- Links: 5846
- Reference warm value-and-gradient: 0.006145 s

| Representation | Construction (s) | Warm forward (s) | Warm value+grad (s) | Stored (MiB) | Density | Break-even evals |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0.509825 | 0.000007 | 0.000010 | 0.002 | 0.1042 | 83.1 |
| bcoo | 1.049421 | 0.000005 | 0.000007 | 0.001 | 0.1042 | 171.0 |

Enable manually when the expected objective-evaluation count exceeds the measured break-even point and the dense/BCOO storage fits the memory budget.
