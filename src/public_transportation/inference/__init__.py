# src/public_transportation/inference/__init__.py
"""
Shared statistical-inference subpackage.

Public API:
- Prior construction (see `priors.py`)
- Model wiring (see `model.py`)
- Likelihood wiring (see `likelihood.py`), which bridges:
    mapper outputs (AggregationSpec, y_obs) -> JAX log-likelihood
"""

from __future__ import annotations

# Re-export core lightweight types/helpers
from .types import *  # noqa: F401,F403

# Re-export priors / model entry points (keep these stable)
from .priors import *  # noqa: F401,F403
from .model import *  # noqa: F401,F403
from .od_parameter_layout import (  # noqa: F401
    ODParameterLayout,
    assert_od_layout_fingerprint_matches,
    build_od_parameter_layout,
)
from .compact_od_assignment_layout import (  # noqa: F401
    CompactODAssignmentLayout,
    build_compact_od_assignment_layout,
)
from .compact_od_groups import compact_od_groups  # noqa: F401
from .assignment_adapter import FixedRoutingPreparationDiagnostics  # noqa: F401
from .sharded_fixed_routing import (  # noqa: F401
    SHARDED_FIXED_ROUTING_IMPLEMENTATION_VERSION,
    SHARDED_FIXED_ROUTING_SCHEMA_VERSION,
    FixedRoutingShard,
    FixedRoutingPreparationConfig,
    FixedRoutingShardCacheError,
    FixedRoutingShardCacheProvenance,
    FixedRoutingShardDescriptor,
    FixedRoutingShardExecutionDiagnostics,
    FixedRoutingBatchExecutionDiagnostics,
    FixedRoutingShardPlan,
    FixedRoutingShardProgress,
    FixedRoutingWorkerRecommendation,
    ShardedFixedRoutingPreparationResult,
    ShardedFixedRoutingInputs,
    build_sharded_fixed_routing_inputs,
    fixed_routing_shard_path,
    load_fixed_routing_shard,
    plan_fixed_routing_shards,
    prepare_fixed_routing_sharded,
    recommend_fixed_routing_workers,
    save_fixed_routing_shard,
    sharded_fixed_routing_identity,
)
from .gravity import (  # noqa: F401
    GravityAdequacyConfig,
    GravityAdequacyFindings,
    GravityAdequacyReport,
    GravityAssignedCountDiagnostics,
    GravityChildWarmStart,
    GravityDemandResult,
    GravityEstimationResult,
    GravityEstimatorConfig,
    GravityEstimatorProgress,
    GravityEffectScope,
    GravityFeatures,
    GravityGradientStrategy,
    GravityGroupedResidualSummary,
    GravityHoldoutSplit,
    GravityHoldoutSplitConfig,
    GravityHoldoutUnit,
    GravityHoldoutValidationReport,
    GravityJSONLProgressSink,
    GravityJourneyCorrelationSummary,
    GravityLineageProgressionResult,
    GravityLikelihood,
    GravityModelSpecification,
    GravityModelComparison,
    GravityModelLineage,
    GravityModelNode,
    GravityObjectiveEvaluation,
    GravityObjectiveProblem,
    GravityParameterLayout,
    GravityPreflightPhase,
    GravityPreflightRecommendation,
    GravityPreflightResult,
    GravityRecommendationConfig,
    GravityRelaxationInfo,
    GravityRelaxationRecommendation,
    GravityRelaxationRecommendationReport,
    GravityOptimizerState,
    GravityPredictiveMetrics,
    GravityRuntimeMemoryDiagnostics,
    GravityExecutionPolicy,
    GravityCompilationDiagnostics,
    GravityStrategySelection,
    GravityValidationMetadata,
    GRAVITY_PROGRESS_SCHEMA_VERSION,
    GRAVITY_RUN_MANIFEST_SCHEMA_VERSION,
    MinimalGravityParameters,
    add_gravity_relaxation,
    build_gravity_holdout_split,
    build_gravity_run_manifest,
    compare_gravity_model_nodes,
    construct_gravity_child_warm_start,
    create_gravity_model_node,
    estimate_gravity_model,
    estimate_and_validate_gravity_holdout,
    evaluate_gravity_objective,
    generate_gravity_demand,
    gravity_value_and_gradient,
    gravity_value_and_gradient_adjoint,
    gravity_value_and_gradient_batched_forward,
    gravity_demand_kernel,
    gravity_demand_numpy_reference,
    gravity_measurement_identity,
    list_applicable_gravity_relaxations,
    predict_gravity_measurements,
    progress_gravity_model_lineage,
    recommend_gravity_relaxations,
    remove_gravity_relaxation,
    run_gravity_preflight,
    validate_gravity_relaxation_features,
    validate_full_data_gravity_adequacy,
    warm_start_gravity_parameters,
    write_gravity_run_manifest,
)
from .block_coordinate import *  # noqa: F401,F403
from .fixed_routing_measurement_operator import (  # noqa: F401
    FixedRoutingMeasurementOperator,
    MeasurementOperatorMetrics,
    choose_fixed_measurement_operator,
    fixed_routing_measurement_operator_cache_path,
    load_fixed_routing_measurement_operator,
    load_or_prepare_fixed_routing_measurement_operator,
    predict_measurements_fixed_operator,
    prepare_fixed_routing_measurement_operator,
    save_fixed_routing_measurement_operator,
    validate_fixed_routing_measurement_operator,
)
from .fixed_routing_matrix_free_operator import (  # noqa: F401
    MatrixFreePreparationDeadlineError,
    MatrixFreePreparationDiagnostics,
    MatrixFreeFixedRoutingMeasurementOperator,
)
from .sharded_matrix_free_operator import (  # noqa: F401
    ShardedMatrixFreeFixedRoutingMeasurementOperator,
    ShardedMatrixFreeMetrics,
    ShardedOperatorProductInterrupted,
    ShardedOperatorProgress,
)
from .parallel_partial_execution import (  # noqa: F401
    FixedBudgetRoutingSelection,
    PARALLEL_PARTIAL_EXECUTION_SCHEMA_VERSION,
    PartialExecutionBatch,
    PartialExecutionPlan,
    RoutingCostModel,
    RoutingMicroshardPlan,
    RoutingWorkObservation,
    RoutingWorkUnit,
    ShardedWorkInstrumentation,
    build_balanced_microshard_plan,
    plan_fixed_budget_routing_selection,
    routing_group_work_units,
)
from .parallel_routing_executor import (  # noqa: F401
    ParallelApproximateRoutingOperator,
    ParallelExactRoutingOperator,
    ParallelRoutingExecutionInterrupted,
    ParallelRoutingExecutionResult,
    ParallelRoutingExecutorConfig,
    PersistentParallelRoutingExecutor,
    RoutingBatchExecutionObservation,
    RoutingExecutionBatch,
    plan_fixed_shape_routing_batches,
)
from .parallel_exact_gate import (  # noqa: F401
    ParallelExactGateConfig,
    ParallelExactGateReport,
    assess_parallel_exact_gate,
)
from .parallel_partial_gate import (  # noqa: F401
    DEFAULT_PARTIAL_EFFORT_REQUIREMENTS,
    ParallelPartialGateReport,
    PartialEffortGateResult,
    PartialEffortRequirement,
    assess_parallel_partial_gate,
)
from .parallel_gravity_anchor import (  # noqa: F401
    ParallelGravityAnchor,
    create_parallel_gravity_anchor,
    parallel_anchored_value_and_gradient,
)
from .stochastic_gravity import (  # noqa: F401
    StochasticGravityConfig,
    StochasticGravityResult,
    StochasticQualityDiagnostics,
    StochasticShardProgress,
    StochasticShardSelection,
    select_stochastic_routing_shards,
    stochastic_gravity_value_and_gradient,
)
from .measurement_operator_protocol import (  # noqa: F401
    GravityMeasurementOperator,
    GravityOperatorCapabilities,
    GravityOperatorMetrics,
)
from .fixed_routing_linear_backend import (  # noqa: F401
    LinearMeasurementBackendMetrics,
    LinearOperatorMode,
    PreparedLinearMeasurementBackend,
    SparseOperatorSelection,
    SparseOperatorSelectionConfig,
    prepare_fixed_routing_linear_measurement_backend,
    scipy_sparse_operator_from_fixed_routing,
    select_fixed_routing_linear_operator,
)
from .fixed_routing_sharded_builder import (  # noqa: F401
    ConstructionTask,
    ShardedConstructionConfig,
    ShardedConstructionPlan,
    ShardedConstructionResult,
    StorageShardPlan,
    SupportPattern,
    pack_storage_shards,
    load_complete_sharded_fixed_routing_cache,
    plan_sharded_fixed_routing_operator,
    prepare_sharded_fixed_routing_measurement_operator,
)
from .fixed_routing_sharded_selection import (  # noqa: F401
    ShardedCacheStatus,
    ShardedSelectionConfig,
    ShardedSelectionDecision,
    select_sharded_fixed_routing_backend,
)
from .fixed_routing_origin_support import (  # noqa: F401
    GroupOriginSupportSummary,
    OriginSpecificMeasurementSupport,
    OriginSupportConfig,
    OriginSupportMetrics,
    OriginSupportValidation,
    analyze_fixed_routing_origin_support,
    validate_origin_support_against_operator,
)
from .sharded_sparse_operator import (  # noqa: F401
    SHARDED_OPERATOR_SCHEMA_VERSION,
    ShardedOperatorManifest,
    ShardedSparseLinearOperator,
    SparseShardIdentity,
    SparseShardMetadata,
    SparseShardMetrics,
    load_sharded_operator_manifest,
    load_sparse_shard,
    save_sharded_operator_manifest,
    save_sparse_shard,
)
from .maximum_likelihood_pipeline import (  # noqa: F401
    ODThetaMLProblem,
    build_od_theta_ml_problem,
)
from .complexity import ODParameterComplexity, build_od_parameter_complexity  # noqa: F401
from .runtime_profile import (  # noqa: F401
    ODAssignmentRuntimeProfile,
    build_od_assignment_runtime_profile,
)

# Re-export likelihood utilities used by model construction
from .likelihood import (  # noqa: F401
    PreparedLikelihoodInputs,
    prepare_likelihood_inputs,
    predict_y,
    predict_mu,
    loglikelihood_from_link_flow,
    loglikelihood_from_measurement_mean,
)
from .fixed_routing_linear_problem import (  # noqa: F401
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
    LinearRegularizationBlock,
    build_fixed_routing_linear_problem_from_dense_operator,
    build_fixed_routing_linear_problem_from_backend,
    build_fixed_routing_linear_problem_from_operator,
)
from .fixed_routing_linear_objective import (  # noqa: F401
    LinearDataFitEvaluation,
    evaluate_linear_data_fit,
    linear_data_gradient,
    linear_data_objective,
    predict_linear_measurements,
    raw_linear_residual,
    weighted_linear_residual,
)
from .fixed_routing_linear_regularization import (  # noqa: F401
    AugmentedLinearLeastSquaresOperator,
    AugmentedLinearLeastSquaresSystem,
    LinearLeastSquaresEvaluation,
    RegularizationBlockEvaluation,
    build_augmented_linear_least_squares_system,
    evaluate_linear_least_squares,
    evaluate_regularization_block,
    ridge_to_prior,
    scaled_ridge_to_prior,
)
from .fixed_routing_linear_transform import (  # noqa: F401
    ColumnScaledLinearOperator,
    PhysicalDemandTransform,
    SolverVariableLeastSquaresSystem,
    build_solver_variable_least_squares_system,
)
from .fixed_routing_linear_dense_solver import (  # noqa: F401
    BoundKKTDiagnostics,
    DenseReferenceResult,
    evaluate_bound_kkt,
    solve_dense_reference,
)
from .fixed_routing_linear_trf_solver import (  # noqa: F401
    TRFLSMRConfig,
    TRFLSMRResult,
    solve_trf_lsmr,
)
from .fixed_routing_linear_quality import (  # noqa: F401
    LinearEstimateQuality,
    analyze_linear_estimate_quality,
)
from .fixed_routing_linear_scalable_quality import (  # noqa: F401
    ScalableLinearEstimateQuality,
    ScalableQualityConfig,
    analyze_linear_estimate_quality_scalable,
)
from .fixed_routing_linear_recommendation import (  # noqa: F401
    RegularizationOptionRecommendation,
    RegularizationRecommendation,
    RegularizationSelectionRequiredError,
    recommend_linear_regularization,
    require_explicit_regularization_selection,
)
from .fixed_routing_linear_results import (  # noqa: F401
    FIXED_ROUTING_LINEAR_RESULT_SCHEMA_VERSION,
    FixedRoutingLinearResult,
    build_fixed_routing_linear_result,
    load_fixed_routing_linear_result,
    save_fixed_routing_linear_result,
)
from .fixed_routing_linear_workflow import (  # noqa: F401
    FixedRoutingLinearEstimationConfig,
    FixedRoutingLinearEstimationRun,
    RegularizationChoice,
    ScalableFixedRoutingLinearEstimationRun,
    configure_fixed_routing_linear_regularization,
    run_fixed_routing_linear_estimation,
    run_fixed_routing_linear_estimation_scalable,
)
from .fixed_routing_linear_solver import (  # noqa: F401
    FixedRoutingLinearSolverConfig,
    FixedRoutingLinearSolverResult,
    LinearSolverBackend,
    LinearSolverBenchmarkRecord,
    REGISTERED_LINEAR_SOLVER_BACKENDS,
    benchmark_fixed_routing_linear_solvers,
    solve_fixed_routing_linear,
)
from .fixed_routing_linear_validation import (  # noqa: F401
    ForwardEquivalenceCase,
    ForwardEquivalenceValidation,
    NoiseFreeRecoveryValidation,
    validate_fixed_routing_forward_equivalence,
    validate_noise_free_linear_recovery,
)
from .linear_operator import (  # noqa: F401
    DenseLinearOperator,
    LinearOperatorProtocol,
    SparseLinearOperator,
    as_linear_operator,
    as_sparse_linear_operator,
    materialize_linear_operator,
)
