# Traffic-assignment report reorganization ledger

Status: Phases 1--12 complete. The inventory below records the original
monolithic source, and the preserved fragments and relocation manifest provide
byte-level traceability. Parts I--VI now contain the reorganized scientific
narrative; mixed implementation, phase-history, and benchmark blocks have been
relocated intact to the implementation appendices. The final consistency audit
against the current public APIs, focused tests, terminology, preservation
manifest, and compiled PDF has been completed.

Source snapshot:

- source: `docs/reports/traffic_assignment.tex`;
- revision: `55ed81601d0d786f80b333f4e2b0474505c6a472`;
- inventoried length: 3,064 lines;
- inventory date: 2026-08-05.

This ledger is the preservation contract for reorganizing the report. Every
source line belongs to one and only one primary relocation block below. A block
may later be split into scientific text and an implementation appendix, but no
text may be deleted during the mechanical relocation phase. Editorial
deduplication requires a later explicit review and traceability note.

## Target source structure

```text
docs/reports/traffic_assignment/
    traffic_assignment.tex
    frontmatter.tex
    part1_problem_definition/
        introduction.tex
        system_and_observations.tex
        underdetermination.tex
        key_modeling_questions.tex
    part2_mathematical_forward_models/
        detailed_assignment.tex
        reduced_journey_response.tex
        raptor_example.tex
        additive_operator_decomposition.tex
        measurement_model.tex
    part3_demand_models/
        cell_level_deviations.tex
        conditional_gravity.tex
        route_level_ipf.tex
        entropy_models.tex
        model_assumptions_comparison.tex
    part4_map_and_alternatives/
        map_primary_method.tex
        maximum_likelihood_reference.tex
        variational_bayesian_inference.tex
        numerical_optimization.tex
        method_comparison.tex
    part5_validation/
        adequacy_and_identification.tex
        grouped_holdout.tex
        detailed_assignment_validation.tex
        advisory_relaxations.tex
        recommended_workflow.tex
    part6_discussion/
        limitations.tex
        conclusions.tex
    appendices/
        software_architecture.tex
        indexing_contracts_and_provenance.tex
        persistence_restart_and_reporting.tex
        computational_backends.tex
        scalable_linear_map.tex
        stochastic_and_progressive_fidelity.tex
        benchmarks_and_validation_record.tex
        future_extensions.tex
    bibliography.tex
```

The directory names express editorial ownership; final displayed titles can be
shorter. The top-level file will retain packages, theorem definitions, title,
and ordered `\input` statements. Existing labels will be preserved during
movement.

## Editorial rules

1. Part I defines the inference problem and its intrinsic underdetermination.
2. Part II contains mathematics, assumptions, and pedagogical examples only.
   Python names, files, caches, fingerprints, JAX mechanics, timings, and
   hardware results go to appendices.
3. Additive sharding may appear in Part II only as the mathematical identity
   \(B=\sum_s B_s\), with assumptions needed for exact forward and adjoint
   decomposition. Execution and storage details go to the computational
   appendices.
4. Part III presents every demand model under a common template: formulation,
   required information, identifying assumptions, parameter dimension,
   recoverable quantities, limitations, and appropriate use.
5. Part IV presents MAP first and as the recommended operational estimator.
   ML is the flat-prior diagnostic reference; VI is the uncertainty-aware but
   approximate and more costly alternative.
6. Part V keeps calibration adequacy, predictive holdout, detailed-forward
   validation, and numerical convergence distinct.
7. Current behavior and future extensions must never share an unlabeled
   description.
8. Benchmarks support a method after its definition; they do not interrupt the
   mathematical exposition.

## Complete relocation map

Line ranges refer to the inventoried source snapshot. “Split” means the block
is preserved as one mechanical unit initially, then divided only during the
later methodology/implementation separation phase.

| Original lines | Existing heading or content | Primary destination | Treatment |
|---:|---|---|---|
| 1–25 | Preamble, title, theorem declarations | top-level file and `frontmatter.tex` | Preserve exactly; later update title/date only by review. |
| 26–42 | Introduction | Part I, `introduction.tex` | Preserve; rewrite roadmap after relocation. |
| 43–46 | Modeling Elements introduction | Part I, `system_and_observations.tex` | Preserve as transition material. |
| 47–57 | Spatial Representation | Part I, `system_and_observations.tex` | Scientific definition. |
| 58–91 | Temporal Representation | Part I, `system_and_observations.tex` | Scientific definition and time-index assumptions. |
| 92–151 | Passenger journeys, vehicle legs, and observed events | Part I, `system_and_observations.tex` | Core terminology and transfer semantics. |
| 152–202 | Immutable reduced-OD configuration and identity contracts | Appendix, `indexing_contracts_and_provenance.tex` | Implementation/provenance; Part I receives only a short canonical-index assumption. |
| 203–256 | Physical stops, route patterns, and compact timetable index | Split: Part I system definitions; Appendix software architecture | Preserve whole block first; move array/index/build details to appendix. |
| 257–335 | Route-level iterative proportional fitting baseline | Part III, `route_level_ipf.tex` | Move benchmark and implementation-specific diagnostics later to benchmark appendix. |
| 336–411 | Bounded timetable feasibility and feature summaries | Split: Part II, `reduced_journey_response.tex`; Appendix computational backends | RAPTOR mathematics and assumptions remain in Part II; current implementation deviations and timings move to appendices. |
| 412–485 | Compact journey choices and transfer-event accounting | Split: Part II reduced response; Appendix software architecture | Mathematical choice/event rules in Part II; fingerprints, payloads, and benchmark figures in appendices. |
| 486–557 | Direct measurement responses and fail-closed cache | Split: Part II reduced response; Appendix provenance/persistence | Response equation in Part II; cache construction and validation mechanics in appendices. |
| 558–619 | Exact reduced operator and basis response | Split: Part II additive decomposition; Appendix computational backends | Preserve operator equations in Part II and representation/benchmark material in appendices. |
| 620–666 | Balanced and unbalanced entropy baselines | Part III, `entropy_models.tex` | Recast around assumptions and limitations; retain benchmark in appendix. |
| 667–725 | Minimal conditional gravity model | Part III, `conditional_gravity.tex` | Core model; move JAX and runtime material to appendices. |
| 726–776 | Reduced-model ML/MAP estimation and restart operations | Split: Part IV MAP; Appendix persistence/restart | MAP equations in Part IV; SciPy/JIT/checkpoint/timing details in appendices. |
| 777–805 | Explicit OD reconstruction and detailed validation | Split: Part V detailed validation; Appendix software architecture | Validation concept in Part V; callback/API and benchmark details in appendices. |
| 806–875 | Adequacy, grouped holdout, and advisory relaxation | Part V, three validation files | Preserve distinction between adequacy, prediction, and advice; move implementation timings to benchmark appendix. |
| 876–883 | Baseline demand and Bayesian deviation parameterization introduction | Part III, `cell_level_deviations.tex` | Core cell-level demand model. |
| 884–915 | Smoothly bounded log-deviation | Part III, `cell_level_deviations.tex` | Mathematical formulation and assumption. |
| 916–922 | Zero baseline cells | Part III, `cell_level_deviations.tex` | Limitation/identification assumption. |
| 923–939 | Prior scale | Part IV, `map_primary_method.tex` | Prior interpretation; cross-reference cell model. |
| 940–953 | Separation of responsibilities | Appendix, `software_architecture.tex` | Software boundary. |
| 954–960 | Desired departure interval and boarding window | Part II, `detailed_assignment.tex` | Mathematical timing assumption. |
| 961–972 | Admissible access links | Part II, `detailed_assignment.tex` | Mathematical feasibility assumption. |
| 973–994 | Schedule-deviation penalty | Part II, `detailed_assignment.tex` | Mathematical generalized-cost component. |
| 995–1003 | Implementation note on departure intervals | Appendix, `software_architecture.tex` | Implementation only. |
| 1004–1069 | Network Representation | Part II, `detailed_assignment.tex` | Mathematical graph definition. |
| 1070–1161 | Remark following network representation | Split: Part II detailed assignment; Appendix computational backends | Preserve; classify each remark paragraph during Phase 6. |
| 1162–1176 | Generalized Cost introduction | Part II, `detailed_assignment.tex` | Mathematical definition and assumptions. |
| 1177–1181 | Link-type-specific cost formulation | Part II, `detailed_assignment.tex` | Mathematical organization. |
| 1182–1192 | Ride links | Part II, `detailed_assignment.tex` | Mathematical cost. |
| 1193–1202 | Dwell/continue links | Part II, `detailed_assignment.tex` | Mathematical cost. |
| 1203–1214 | Transfer links | Part II, `detailed_assignment.tex` | Mathematical cost and bounded-wait assumption. |
| 1215–1231 | Access links | Part II, `detailed_assignment.tex` | Mathematical cost and flat-region assumption. |
| 1232–1234 | Egress links | Part II, `detailed_assignment.tex` | Mathematical cost. |
| 1235–1241 | Interpretation of cost coefficients | Part II, `detailed_assignment.tex` | Assumption interpretation. |
| 1242–1243 | Assignment Model introduction | Part II, `detailed_assignment.tex` | Preserve as chapter transition. |
| 1244–1247 | Assignment input | Part II, `detailed_assignment.tex` | Mathematical inputs only after editing. |
| 1248–1253 | Assignment procedure | Part II, `detailed_assignment.tex` | Mathematical overview. |
| 1254–1258 | Differentiable Dial-style assignment | Part II, `detailed_assignment.tex` | Mathematical method. |
| 1259–1267 | Capacity constraints, current status | Part II assumptions and Part VI limitations | Primary location Part II assumption box; cross-reference limitations. |
| 1268–1299 | Meaning of fixed routing | Part II, `detailed_assignment.tex` | Central assumption; explicitly answer time variation and crowding questions. |
| 1300–1311 | Logit routing model | Part II, `detailed_assignment.tex` | Mathematical route-choice model. |
| 1312–1316 | Computational and behavioral motivation | Split: Part II behavioral motivation; Appendix computational backends | Preserve whole paragraph before split. |
| 1317–1347 | Differentiable dynamic programming | Part II, `detailed_assignment.tex` | Mathematical recursion. |
| 1348–1369 | Flow propagation | Part II, `detailed_assignment.tex` | Mathematical recursion. |
| 1370–1373 | Destination gating | Part II, `detailed_assignment.tex` | Mathematical boundary condition. |
| 1374–1377 | Batched evaluation | Appendix, `computational_backends.tex` | Implementation/performance. |
| 1378–1401 | Separation of routing preparation and demand loading | Appendix, `software_architecture.tex` | Implementation architecture; Part II retains fixed-routing assumption only. |
| 1402–1418 | Precomputation for fixed dispersion | Appendix, `computational_backends.tex` | Numerical implementation. |
| 1419–1427 | Cache consistency | Appendix, `indexing_contracts_and_provenance.tex` | Provenance. |
| 1428–1458 | Computational consequences | Appendix, `computational_backends.tex` | Complexity and implementation. |
| 1459–1508 | Direct fixed-routing measurement operator | Split: Part II operator equation; Appendix computational backends | Mathematical linear map in Part II; construction details in appendix. |
| 1509–1518 | Construction profiling | Appendix, `benchmarks_and_validation_record.tex` | Benchmark. |
| 1519–1528 | Validity and fallback | Part II assumptions plus Part VI limitations | Primary Part II assumption statement; operational fallback in appendix. |
| 1529–1541 | Persistent cache and provenance | Appendix, `indexing_contracts_and_provenance.tex` | Persistence/provenance. |
| 1542–1550 | Activation policy | Appendix, `software_architecture.tex` | Operational policy. |
| 1551–1611 | TPG structural measurement | Appendix, `benchmarks_and_validation_record.tex` | Empirical implementation evidence. |
| 1612–1635 | Scalability limits for very large OD layouts | Part VI, `limitations.tex` | Scientific/computational limitation; detailed numbers cross-referenced to appendix. |
| 1636–1657 | Event-aligned measurement resolution | Split: Part I observations; Appendix software architecture | Measurement semantics in Part I, resolver details in appendix. |
| 1658–1681 | Explicit fixed-routing adjoint | Split: Part II additive operator/adjoint; Appendix computational backends | Mathematical adjoint in Part II; kernel details in appendix. |
| 1682–1742 | Stable objective compilation and reuse | Appendix, `computational_backends.tex` | JAX/compilation evidence. |
| 1743–1808 | Persistent assignment artifacts | Appendix, `persistence_restart_and_reporting.tex` | Persistence and measured reuse. |
| 1809–1844 | Fixed-routing linear MAP estimation at scale | Split: Part IV MAP; Appendix scalable linear MAP | Scientific objective in Part IV; solver engineering in appendix. |
| 1845–1863 | Operator representations | Appendix, `scalable_linear_map.tex` | Implementation representations. |
| 1864–1890 | OD blocks without network cuts | Split: Part II additive decomposition; Appendix scalable linear MAP | Block mathematics in Part II; solver mechanics in appendix. |
| 1891–1959 | Anytime state and recovery | Appendix, `persistence_restart_and_reporting.tex` | Operational recovery. |
| 1960–1970 | Safe parallelism | Appendix, `computational_backends.tex` | Execution policy. |
| 1971–2094 | Resource preflight and adaptive refinement | Appendix, `scalable_linear_map.tex` | Solver/resource engineering and benchmarks. |
| 2095–2099 | Opt-out alternative, future extension | Appendix, `future_extensions.tex` | Clearly future. |
| 2100–2144 | Reduced-dimensional gravity demand estimation | Part III, `conditional_gravity.tex` | Mathematical demand model; implementation details later separated. |
| 2145–2541 | Progressive-fidelity objective and gradient | Split: Part II additive decomposition; Appendix stochastic/progressive fidelity; benchmark appendix | Preserve complete block first. Mathematical sampling/decomposition in Part II; all APIs, execution, gates, and validation results in appendices. |
| 2542–2554 | Implementation notes for fast JAX evaluation | Appendix, `computational_backends.tex` | Implementation only. |
| 2555–2560 | Deterministic indexing requirement | Appendix, `indexing_contracts_and_provenance.tex` | Implementation/provenance. |
| 2561–2566 | Measurement Model and Likelihood introduction | Part I observations and Part II measurement model | Primary Part II equation chapter; Part I defines data meaning. |
| 2567–2578 | Predicted measurement | Part II, `measurement_model.tex` | Mathematical observation equation. |
| 2579–2585 | Asymmetric counting device | Part II, `measurement_model.tex` | Measurement assumption. |
| 2586–2609 | Negative-binomial likelihood | Part II, `measurement_model.tex` | Statistical mathematics. |
| 2610–2615 | Fixed measurement parameters | Part II assumptions | Explicit fixed-parameter assumption. |
| 2616–2641 | Likelihood and normalization | Part II, `measurement_model.tex` | Mathematical likelihood. |
| 2642–2645 | Alternative observation models, future | Appendix, `future_extensions.tex` | Clearly future. |
| 2646–2696 | Free and Frozen OD Demand Cells | Part I, `system_and_observations.tex`; Part III cell model | Primary Part I definition; estimation consequence cross-reference in Part III. |
| 2697–2758 | Topology-driven structural-zero preprocessing | Split: Part I support definitions; Part II RAPTOR/feasibility; Appendix software architecture | Explicitly distinguish physical-stop, platform, transfer-limit, and timetable cases. |
| 2759–2795 | Complexity boundary | Appendix, `computational_backends.tex` | Complexity guarantee. |
| 2796–2813 | ML and MAP introduction | Part IV, `map_primary_method.tex` | Reorder so MAP is introduced first. |
| 2814–2851 | Maximum likelihood estimation | Part IV, `maximum_likelihood_reference.tex` | Present as flat-prior diagnostic reference. |
| 2852–2879 | Maximum a posteriori estimation | Part IV, `map_primary_method.tex` | Primary operational method. |
| 2880–2894 | Numerical optimization and local uncertainty | Part IV, `numerical_optimization.tex` | Mathematical/numerical interpretation. |
| 2895–2913 | Bayesian inference via VI introduction | Part IV, `variational_bayesian_inference.tex` | Alternative method. |
| 2914–2929 | Variational approximation | Part IV, `variational_bayesian_inference.tex` | Mathematical approximation. |
| 2930–2942 | Guide family | Part IV, `variational_bayesian_inference.tex` | Approximation assumption. |
| 2943–2974 | Prior on theta | Part IV MAP and VI files | Primary VI location; MAP cross-reference ensures identical prior semantics. |
| 2975–2992 | Posterior transformation and reporting | Part IV, `variational_bayesian_inference.tex` | Reporting interpretation. |
| 2993–2996 | Alternative inference engines | Appendix, `future_extensions.tex` | Future. |
| 2997–3002 | Comparison of estimation methods introduction | Part IV, `method_comparison.tex` | Preserve. |
| 3003–3010 | ML versus MAP | Part IV, `method_comparison.tex` | Reframe around MAP default. |
| 3011–3018 | MAP versus Bayesian inference | Part IV, `method_comparison.tex` | Preserve scientific distinction. |
| 3019–3031 | Variational approximation comparison | Part IV, `method_comparison.tex` | Preserve limitation. |
| 3032–3037 | When estimates should be close | Part IV, `method_comparison.tex` | Preserve asymptotic qualification. |
| 3038–3046 | Recommended comparisons | Part IV, `method_comparison.tex` | Preserve validation advice. |
| 3047–3058 | Discussion | Part VI, `conclusions.tex` | Expand after reorganization; retain original text. |
| 3059–3064 | Bibliography and document ending | `bibliography.tex` and top-level file | Preserve exactly. |

The ranges cover lines 1–3,064 without overlap or omission.

## New material to add after mechanical relocation

New text is additive and has no source range in the table above.

### Part I: intrinsic underdetermination

- A dimension/rank explanation of why boarding and alighting counts generally
  cannot identify all OD-time cells.
- A small example with two distinct OD tables that produce the same observed
  counts.
- Explicit separation of structural non-identification, weak practical
  identification, and numerical optimization difficulty.
- Explanation that priors, gravity restrictions, fixed cells, and structural
  zeros add assumptions or reduce dimension; they do not create observations.

### Part I: direct answers to modeling questions

- Connectivity is defined for passenger-demand support at the physical-stop
  level after platform normalization. Platform constraints still enter
  timetable and transfer feasibility.
- `no_topological_path`, `exceeds_transfer_limit`, and
  `no_timetable_feasible_journey` are distinct outcomes. A highly connected
  network may empirically have few or no first-category pairs.
- Fixed routing means fixed conditional shares per OD-time cell during a fit,
  not one share vector for the whole day.
- Current dynamic assignment is time dependent but has no endogenous crowding,
  denied-boarding, or capacity feedback.
- Current diagnostics expose weak curvature and low-information residual
  groups, but optimal sensor placement is not implemented.

### Part II: pedagogical RAPTOR example

Use two lines, `A–B–C` and `B–D–E`, with scheduled departure/arrival times.
Show access labels, route rounds, the transfer at B, arrival labels, and a
time-versus-transfer Pareto comparison. State precisely whether rounds count
vehicle boardings or transfers and how the configured transfer limit maps to
rounds.

### Part II: mathematical sharding

Introduce

```text
B = sum_s B_s,
mu = b_fixed + sum_s B_s x,
B^T v = sum_s B_s^T v.
```

State additivity, partition/overlap, weighting, and matched-adjoint assumptions.
Keep storage, worker, cache, and timing discussions in appendices.

### Part III: common model template

Apply the same seven headings to cell-level deviations, gravity, route-level
IPF, balanced entropy, and unbalanced entropy. Add a comparison table covering
identifying assumptions, dimension, required inputs, and principal limitation.

### Part IV: MAP-first recommendation

Explain why MAP is the operational focus for the underdetermined large-scale
problem. Retain ML as a flat-prior and implementation diagnostic, and VI as an
approximate posterior alternative when dimension and computational cost permit.

## Traceability checks required in later phases

1. Record a hash of every extracted source block before and after movement.
2. Verify concatenated extracted blocks reproduce the original body modulo
   `\input` boundaries and intentionally added transitions.
3. Preserve all existing `\label`, `\ref`, citation, equation, table, and figure
   content during the mechanical phase.
4. Compile after each part is introduced.
5. Maintain a temporary duplicate-content register; do not delete duplication
   during relocation.
6. Before final acceptance, map every ledger row to its actual destination file
   and line and mark it `moved`, `split with trace`, or `unchanged`.
