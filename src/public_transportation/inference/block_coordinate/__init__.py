"""Public contracts for interruptible block-coordinate MAP estimation."""

# Re-exports define this package facade; keep it explicit without ``__all__``.
# ruff: noqa: F401

from .adaptive import (
    AdaptiveBlockSplitConfig,
    AdaptiveBlockSplitRecord,
    AdaptiveBlockSplitResult,
    BlockResourceCostModel,
    BlockResourceGuardError,
    fingerprints_for_adapted_partition,
    split_partition_for_resource_limits,
)
from .blocks import ODBlock
from .checkpoint import (
    BLOCK_COORDINATE_CHECKPOINT_SCHEMA_VERSION,
    BlockCheckpointMetadata,
    BlockCoordinateFingerprints,
)
from .config import BlockCoordinateMAPConfig, BlockSizingConfig, GlobalProductPolicy
from .estimator import (
    BlockCoordinateMAPEstimator,
    resume_block_coordinate_map,
    run_block_coordinate_map,
)
from .checkpoint_store import BlockCheckpointStore
from .incremental import (
    BlockUpdateProposal,
    IncrementalLinearState,
    IncrementalPredictionValidation,
    apply_incremental_update,
    block_data_gradient,
    initialize_incremental_state,
    propose_incremental_update,
    validate_incremental_prediction,
)
from .fixed_routing_selected_block_builder import (
    FixedRoutingSelectedBlockBuilder,
    SelectedBlockBuilderConfig,
    SelectedBlockBuilderProvenance,
    SelectedBlockConstructionDiagnostics,
    SelectedBlockConstructionDeadlineError,
    SelectedBlockDeadlineDiagnostics,
    SelectedBlockConstructionProgress,
    SelectedBlockConstructionResult,
    SelectedBlockDiagnosticStop,
    SelectedBlockJSONLProgressSink,
    SelectedBlockPhaseProgress,
    SelectedBlockSupportArtifact,
)
from .operator import (
    BlockLinearOperatorProtocol,
    ColumnSelectedLinearOperator,
    DenseBlockLinearOperator,
    SparseBlockLinearOperator,
    SupportedRowsSparseBlockLinearOperator,
)
from .operator_cache import (
    BLOCK_OPERATOR_CACHE_SCHEMA_VERSION,
    BlockOperatorCacheConfig,
    BlockOperatorCacheProvenance,
    BlockOperatorFactoryMetrics,
    BlockOperatorPreparationMetrics,
    BlockOperatorProductMetrics,
    CachedBlockLinearOperator,
    FixedRoutingBlockOperatorFactory,
)
from .objective import (
    BlockObjectiveEvaluation,
    ConditionalBlockObjective,
    SeparableQuadraticPrior,
    UnsupportedConditionalPriorError,
    build_conditional_block_objective,
    prepare_separable_quadratic_prior,
    projected_gradient,
)
from .progress import BlockProgressEvent, DiagnosticValue
from .partition import (
    ODBlockPartition,
    partition_assignment_od_blocks,
    partition_od_blocks,
    require_measurements_for_block_estimation,
    validate_block_partition,
)
from .resources import (
    AcceptedBlockResourceProposal,
    BlockPreflightSample,
    BlockResourceRecommendation,
    MachineResourceSnapshot,
    ResourcePreflightConfig,
    apply_accepted_resource_recommendation,
    detect_machine_resources,
    measure_representative_blocks,
    recommend_block_resources,
    select_representative_blocks,
    validate_resource_acceptance,
)
from .results import (
    VALID_BLOCK_COORDINATE_STATUSES,
    BlockConvergenceDiagnostics,
    BlockCoordinateMAPResult,
    BlockCoordinateState,
    BlockCoordinateWorkDiagnostics,
    BlockObjectiveComponents,
)
from .solver import (
    BlockSolverConfig,
    BlockSolverResult,
    BlockUpdateDecision,
    BlockUpdatePolicy,
    decide_block_update,
    solve_and_decide_block_update,
    solve_conditional_block,
)
from .scheduling import (
    BlockConflictGraph,
    ConflictFreeBatchDecision,
    ConflictFreeBlockSchedule,
    ParallelBlockExecutionConfig,
    build_block_conflict_graph,
    color_block_conflict_graph,
    conflict_free_batch_id,
    construct_block_operators,
    solve_conflict_free_batch,
)
from .selected_blocks import (
    BlockConstructionResourceError,
    SelectedBlockConstructionMeasurement,
    SelectedBlockResourceEstimate,
    construct_selected_block_operators,
    estimate_selected_block_resources,
    select_representative_block_ids,
)
from .support_preflight import (
    SUPPORT_PREFLIGHT_SCHEMA_VERSION,
    BlockSupportSummary,
    DestinationSupportSummary,
    PilotAuthorization,
    SelectedBlockPilotAuthorization,
    SampledSupportExtrapolation,
    SupportPreflightBudget,
    SupportPreflightConfig,
    SupportPreflightFingerprints,
    SupportPreflightInvocationPolicy,
    SupportPreflightMode,
    SupportPreflightProgress,
    SupportPreflightResult,
    SupportPreflightStatus,
    SupportPreflightStopLocation,
    authorize_block_coordinate_pilot,
    authorize_selected_block_pilot,
    load_support_preflight_checkpoint,
    run_support_preflight,
)
from .validation import (
    BlockCoordinateValidationResult,
    validate_block_coordinate_result,
)
