"""Reduced-dimensional gravity demand contracts."""

# Re-exports define the focused public facade; keep them explicit without __all__.
# ruff: noqa: F401

from .demand import (
    GravityDemandResult,
    generate_gravity_demand,
    gravity_demand_kernel,
    gravity_demand_numpy_reference,
)
from .configuration import (
    GravitySpecificationValidation,
    gravity_model_specification_from_mapping,
    load_gravity_model_specification,
    validate_gravity_model_specification,
)
from .comparison import (
    GravityModelFitSummary,
    rank_gravity_model_summaries,
    summarize_gravity_model_fit,
)
from .diagnostics import (
    GravityRecommendationConfig,
    GravityRelaxationRecommendation,
    GravityRelaxationRecommendationReport,
    recommend_gravity_relaxations,
)
from .features import GravityFeatures
from .fidelity import (
    GravityApproximationQuality,
    GravityFidelityAnchor,
    GravityFidelityContext,
    GravityFidelityDiagnostics,
    GravityFidelityEvaluationInterrupted,
    GravityFidelityExecution,
    GravityFidelityPlan,
    GravityFidelityProgress,
    GravityFidelityRequest,
    GravityFidelityShard,
    GravityFidelityShardProduct,
    GravityFidelityStrategy,
    GravityObjectiveGradientResult,
    build_gravity_fidelity_anchor,
    build_gravity_fidelity_context,
    gravity_fidelity_problem_identity,
    gravity_value_and_gradient_progressive,
    plan_gravity_fidelity,
)
from .lineage import (
    GravityAssignedCountDiagnostics,
    GravityChildWarmStart,
    GravityLineageProgressionResult,
    GravityModelComparison,
    GravityModelLineage,
    GravityModelNode,
    GravityOptimizerState,
    GravityRuntimeMemoryDiagnostics,
    compare_gravity_model_nodes,
    construct_gravity_child_warm_start,
    create_gravity_model_node,
    gravity_measurement_identity,
    list_applicable_gravity_relaxations,
    progress_gravity_model_lineage,
)
from .holdout import (
    GravityHoldoutSplit,
    GravityHoldoutSplitConfig,
    GravityHoldoutUnit,
    GravityHoldoutValidationReport,
    GravityPredictiveMetrics,
    build_gravity_holdout_split,
    estimate_and_validate_gravity_holdout,
)
from .estimator import (
    GRAVITY_CHECKPOINT_SCHEMA_VERSION,
    GRAVITY_RESULT_SCHEMA_VERSION,
    GravityCompilationDiagnostics,
    GravityEstimationResult,
    GravityEstimatorConfig,
    GravityEstimatorProgress,
    GravityExecutionPolicy,
    GravityStrategySelection,
    estimate_gravity_model,
    gravity_model_fingerprint,
    reclassify_gravity_result,
    scaled_gradient_inf_norm,
)
from .biogeme_pilot import (
    GravityBiogemePilotResult,
    GravityOptimizerComparisonResult,
    GravityOptimizerRunSummary,
    compare_gravity_optimizers,
    run_biogeme_tr_bfgs_pilot,
)
from .parameters import (
    GravityParameterBlock,
    GravityParameterLayout,
    MinimalGravityParameters,
    validate_gravity_relaxation_features,
    warm_start_gravity_parameters,
)
from .objective import (
    GravityGradientStrategy,
    GravityLikelihood,
    GravityObjectiveEvaluation,
    GravityObjectiveProblem,
    evaluate_gravity_objective,
    gravity_value_and_gradient,
    gravity_value_and_gradient_adjoint,
    gravity_value_and_gradient_batched_forward,
    predict_gravity_measurements,
)
from .operator import GravityMeasurementOperator
from .aggregate import (
    GRAVITY_AGGREGATE_SCHEMA_VERSION,
    SUPPORTED_AGGREGATE_LIKELIHOODS,
    SUPPORTED_AGGREGATE_UNITS,
    GravityAggregateBin,
    GravityAggregateHistogram,
    GravityAggregateObservation,
    GravityAggregateStratum,
    GravityAggregateUncertainty,
    load_gravity_aggregate,
)
from .attribute_operator import (
    GravityAttributeResponseOperator,
    GravityAttributeResponseProvenance,
    GravityAttributeSupportError,
    GravityRouteShare,
    validate_aggregate_support,
)
from .likelihoods import (
    GravityAggregateLikelihoodEvaluation,
    aggregate_histogram_log_likelihood,
    evaluate_gravity_aggregate_likelihood,
    gravity_aggregate_log_likelihood,
    normalize_aggregate_masses,
)
from .aggregate_channel import (
    GravityAggregateObservationChannel,
    build_gravity_aggregate_observation_bundle,
)
from .observations import GravityObservationBundle, GravityObservationChannel
from .operations import (
    GRAVITY_PROGRESS_SCHEMA_VERSION,
    GRAVITY_RUN_MANIFEST_SCHEMA_VERSION,
    GravityJSONLProgressSink,
    build_gravity_run_manifest,
    write_gravity_run_manifest,
)
from .preflight import (
    GravityPreflightPhase,
    GravityPreflightRecommendation,
    GravityPreflightResult,
    run_gravity_preflight,
)
from .specification import (
    GravityComponentSpecification,
    GravityConstraint,
    GravityEffectScope,
    GravityLikelihoodSpecification,
    GravityModelSpecification,
    GravityParameterization,
    GravityRegularization,
    GravityRegularizationType,
    GravityTimeSpecification,
)
from .relaxations import (
    GravityRelaxationInfo,
    add_gravity_relaxation,
    remove_gravity_relaxation,
)
from .validation import (
    GravityAdequacyConfig,
    GravityAdequacyFindings,
    GravityAdequacyReport,
    GravityGroupedResidualSummary,
    GravityJourneyCorrelationSummary,
    GravityValidationMetadata,
    build_gravity_adequacy_report,
    validate_full_data_gravity_adequacy,
)
from .reporting import (
    GRAVITY_DETAILED_REPORT_SCHEMA_VERSION,
    GravityDetailedReport,
    write_gravity_detailed_report,
)
