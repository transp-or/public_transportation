# public_transportation

Python tools for modeling and analyzing public transportation systems.

## OD and route-choice estimation

The estimation layer supports variational Bayesian inference, maximum
likelihood, and maximum a posteriori estimation of OD demand and, optionally,
the assignment dispersion parameter. All methods share the same differentiable
assignment and measurement likelihood.

Selected OD/time-bin cells can be frozen through a sparse CSV file with columns
`origin_stop_id`, `dest_stop_id`, `time_bin_id`, and optional `fixed_flow`
(default zero). Frozen cells are not estimator parameters. Frozen-zero cells are
also removed from assignment demand and padded destination groups; positive
frozen cells remain as physical demand constants.

Each estimation result exposes a runtime profile reporting free, frozen-zero,
and frozen-positive counts, compact assignment size, and surviving destination
groups. The same metadata and compact-layout fingerprints are persisted with
result artifacts.

See [traffic_assignment.tex](docs/reports/traffic_assignment.tex) for the model
definition and comparison of VI, ML, and MAP. Reproducible compact-assignment
performance measurements are documented in
[compact_assignment_results.md](benchmarks/compact_assignment_results.md).
