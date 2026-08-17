# public_transportation

Python tools for modeling and analyzing public transportation systems.

## OD and route-choice estimation

The estimation layer supports variational Bayesian inference, maximum
likelihood, and maximum a posteriori estimation of OD demand and, optionally,
the assignment dispersion parameter. These likelihood-based methods share the
same differentiable assignment and negative-binomial measurement model. A
separate fixed-routing mode provides explicitly weighted and regularized linear
estimation when routing is held constant.

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

The mathematical and data contracts for the fixed-routing linear estimation
mode are specified in
[fixed_routing_linear_estimation.md](docs/source/fixed_routing_linear_estimation.md).
That specification also documents the interruptible block-coordinate MAP
estimator, conflict-free parallel batches, resource-adaptive preflight, and
deterministic adaptive block refinement. Sharded persistent operator construction
is documented in
[sharded_fixed_routing_operator.md](docs/source/sharded_fixed_routing_operator.md).
Bounded OD batching, cache compatibility, and selected-block construction
diagnostics are documented in
[selected_fixed_routing_block_builder.md](docs/source/selected_fixed_routing_block_builder.md).
Scalable global-product policies, restricted update schedules, and deferred
validation are documented in
[bounded_block_coordinate_pilots.md](docs/source/bounded_block_coordinate_pilots.md).

The operational runbook for a new case study—including current
direct-scheduled preparation, preflight, checkpointed fitting, Jed scheduling,
and progress logs—is
[new_case_study_walkthrough.md](docs/source/new_case_study_walkthrough.md).
The matching TPG-agnostic driver skeleton and Slurm wrappers are in
[direct_scheduled_case_template](docs/source/examples/direct_scheduled_case_template/).
Reuse and migration rules for results produced by the former workflow are
documented separately in
[legacy_case_study_migration.md](docs/source/legacy_case_study_migration.md).

Topology-driven structural-zero detection is configured entirely through TOML.
Its rules, valid parameter values, conflict policy, audit artifacts, and Python
workflow are documented in
[structural_zero_preprocessing.md](docs/source/structural_zero_preprocessing.md).
