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
