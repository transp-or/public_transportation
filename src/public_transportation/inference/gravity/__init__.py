"""Reduced-dimensional gravity demand contracts."""

# Re-exports define the focused public facade; keep them explicit without __all__.
# ruff: noqa: F401

from .demand import (
    GravityDemandResult,
    generate_gravity_demand,
    gravity_demand_kernel,
    gravity_demand_numpy_reference,
)
from .diagnostics import (
    GravityRecommendationConfig,
    GravityRelaxationRecommendation,
    GravityRelaxationRecommendationReport,
    recommend_gravity_relaxations,
)
from .features import GravityFeatures
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
)
from .parameters import (
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
from .specification import GravityEffectScope, GravityModelSpecification
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
    validate_full_data_gravity_adequacy,
)
