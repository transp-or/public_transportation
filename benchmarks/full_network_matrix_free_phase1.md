# Full-network matrix-free feasibility: Phase 1

## Scope

This phase measures the existing fixed-routing loader on one representative
destination group from the TPG full-network scenario. It deliberately does not
construct global fixed routing, a measurement operator, a Jacobian, or start a
MAP optimizer. The TPG repository and its prepared data are read-only inputs.

The benchmark is reproducible with:

```bash
MPLCONFIGDIR=/tmp/public-transportation-mpl \
XDG_CACHE_HOME=/tmp/public-transportation-xdg \
UV_CACHE_DIR=/tmp/public-transportation-uv-cache \
uv run python benchmarks/benchmark_full_network_matrix_free_group.py \
  --group-index 140 \
  --memory-ceiling-gib 24 \
  --warm-evaluations 3 \
  --output benchmarks/full_network_matrix_free_group_140.json
```

The run used JAX/JAXlib 0.8.1 on the CPU backend of a 12-core Apple arm64
machine. The detailed machine-readable record is
`benchmarks/full_network_matrix_free_group_140.json`.

## Selected group

Group 140 was selected as the representative medium-sized group identified by
the preliminary sampling pass. It contains 465 free OD cells, has 1,242,992
enabled links, and touches 213,362 of the 426,436 measurements. The complete
scenario contains 628,164 free cells in 1,898 destination groups.

The helper constructs an `AssignmentInputs` view containing only these 465 free
cells and the selected destination mask. Frozen-zero cells are removed. The
Phase-1 helper rejects frozen-positive cells rather than silently omitting their
constant flow contribution.

## Results

| Operation | Compile (s) | First execution (s) | Warm median (s) |
|---|---:|---:|---:|
| Flow loading | 5.640 | 6.005 | 6.055 |
| Measurement aggregation | 0.054 | 0.00061 | 0.00039 |
| Negative-binomial likelihood | 0.134 | 0.00078 | 0.00058 |
| Objective value | 5.908 | 5.892 | 6.091 |
| Gradient only | 5.693 | 12.499 | 12.126 |
| Value and gradient | 5.791 | 11.816 | 11.855 |

Scenario validation, OD-layout construction, and assignment preparation took
22.58 s, 3.94 s, and 5.42 s. One-group extraction took 0.70 s and fixed-routing
preparation took 36.53 s. Strict construction of the full measurement mapping
took 121.22 s and is the largest setup cost in this bounded run. Total process
time was 382.93 s.

Peak resident memory was 21,096,103,936 bytes (19.65 GiB), below the configured
24 GiB ceiling. The first value-and-gradient checkpoint reached 17,608,409,088
bytes (16.40 GiB), up from 9,807,937,536 bytes (9.13 GiB) immediately before
kernel compilation. Compiling the extra diagnostic kernels increased peak RSS
to 19.65 GiB. Rendering the compiler IR is intentionally disabled because the
textual HLO for this graph can itself consume gigabytes and would contaminate
the measurement.

Persistent array storage was 3.83 GB for the full assignment artifacts, 7.29 MB
for the one-group inputs, 14.56 MB for its fixed routing, and 5.16 MB for the
measurement mapping. The baseline prediction had 36,732 nonzero measurements
and total predicted flow 930.0001. The bounded scalar objective was 29,840,308
with gradient norm 10,712.20; these values are recorded as reproducibility
checks, not as a full-network statistical objective.

## Interpretation

The measurement aggregation and likelihood are negligible once link flows are
available. The existing dynamic-programming loader dominates both value and
gradient time: a warm value-and-gradient evaluation for only 465 free cells is
11.86 s and requires roughly 16.4 GiB at its first checkpoint. This rules out a
straightforward full-network objective that differentiates all destination
groups together on this machine.

The benchmark also shows that strict measurement mapping is unnecessarily
global and expensive for a one-group study. A practical next phase should
partition or pre-index the mapping once, stream destination-group
contributions, and design a custom derivative or accumulation strategy that
does not retain all dynamic-programming states. Summing independently computed
per-group likelihoods would be mathematically wrong because the
negative-binomial likelihood must be evaluated after contributions from all
groups have been summed.

## Safety checks

The JSON record confirms that exactly one destination group was selected and
that global routing, measurement-operator construction, Jacobian construction,
and MAP optimization were not started. The routing preflight and observed peak
both remained below the configured ceiling. Compiler-IR rendering is excluded
from memory profiling.
