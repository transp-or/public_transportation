"""Immutable model lineage and explicit parent-to-child progression."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from public_transportation.inference.block_coordinate._canonical import fingerprint
from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)

from .diagnostics import (
    GravityRecommendationConfig,
    GravityRelaxationRecommendationReport,
    recommend_gravity_relaxations,
)
from .estimator import (
    GravityEstimationResult,
    GravityEstimatorConfig,
    GravityExecutionPolicy,
    estimate_gravity_model,
    gravity_model_fingerprint,
)
from .objective import GravityObjectiveProblem, predict_gravity_measurements
from .parameters import GravityParameterLayout, warm_start_gravity_parameters
from .relaxations import (
    GravityRelaxationInfo,
    add_gravity_relaxation,
    remove_gravity_relaxation,
)
from .specification import GravityEffectScope, GravityModelSpecification
from .validation import (
    GravityAdequacyConfig,
    GravityAdequacyReport,
    GravityValidationMetadata,
    validate_full_data_gravity_adequacy,
)


@dataclass(frozen=True, slots=True)
class GravityOptimizerState:
    status: str
    success: bool
    message: str
    iterations: int
    objective: float
    data_log_likelihood: float
    gradient_infinity_norm: float


@dataclass(frozen=True, slots=True)
class GravityAssignedCountDiagnostics:
    report_fingerprint: str
    measurements: int
    observed_total: float
    modeled_total: float
    negative_binomial_deviance: float
    poisson_deviance: float
    mae: float
    rmse: float
    weighted_rmse: float


@dataclass(frozen=True, slots=True)
class GravityRuntimeMemoryDiagnostics:
    estimation_seconds: float
    result_array_bytes: int
    operator_stored_bytes: int
    operator_peak_construction_bytes: int
    selected_gradient_strategy: str
    compilation_seconds: float
    warm_evaluation_seconds: float


@dataclass(frozen=True, slots=True)
class GravityModelNode:
    schema_version: int
    model_identifier: str
    parent_identifier: str | None
    relaxation_applied: GravityEffectScope | None
    specification: GravityModelSpecification
    parameter_layout: GravityParameterLayout
    raw_parameter_estimates: np.ndarray
    physical_parameter_estimates: np.ndarray
    feature_fingerprint: str
    compact_layout_fingerprint: str
    assignment_fingerprint: str
    graph_fingerprint: str
    measurement_mapping_fingerprint: str
    optimizer_state: GravityOptimizerState
    assigned_count_diagnostics: GravityAssignedCountDiagnostics
    runtime_memory_diagnostics: GravityRuntimeMemoryDiagnostics
    checkpoint_path: Path | None
    calibration_measurement_identity: str
    validation_measurement_identity: str
    estimation_result: GravityEstimationResult

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported gravity model-node schema version.")
        if not self.model_identifier:
            raise ValueError("model_identifier must be nonempty.")
        for name in ("raw_parameter_estimates", "physical_parameter_estimates"):
            array = np.array(getattr(self, name), copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if self.raw_parameter_estimates.shape != (self.parameter_layout.size,):
            raise ValueError("raw parameter estimates do not match the parameter layout.")
        if not self.calibration_measurement_identity:
            raise ValueError("calibration_measurement_identity must be nonempty.")
        if not self.validation_measurement_identity:
            raise ValueError("validation_measurement_identity must be nonempty.")


@dataclass(frozen=True, slots=True)
class GravityModelLineage:
    nodes: tuple[GravityModelNode, ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for node in self.nodes:
            if node.model_identifier in seen:
                raise ValueError("lineage contains a duplicate model identifier.")
            if node.parent_identifier is not None and node.parent_identifier not in seen:
                raise ValueError("each lineage parent must precede its child.")
            seen.add(node.model_identifier)

    def load_completed_parent(self, model_identifier: str) -> GravityModelNode:
        for node in self.nodes:
            if node.model_identifier == model_identifier:
                if not node.estimation_result.status:
                    raise ValueError("the selected lineage node is not completed.")
                return node
        raise KeyError(f"unknown gravity model identifier {model_identifier!r}.")

    def append(self, node: GravityModelNode) -> GravityModelLineage:
        return GravityModelLineage((*self.nodes, node))


@dataclass(frozen=True, slots=True)
class GravityChildWarmStart:
    parent_identifier: str
    relaxation: GravityEffectScope
    relaxation_info: GravityRelaxationInfo
    child_specification: GravityModelSpecification
    child_parameter_layout: GravityParameterLayout
    raw_parameters: np.ndarray
    maximum_parent_prediction_difference: float
    verified: bool

    def __post_init__(self) -> None:
        raw = np.array(self.raw_parameters, copy=True)
        raw.setflags(write=False)
        object.__setattr__(self, "raw_parameters", raw)
        if not self.verified:
            raise ValueError("an unverified child warm start cannot be used.")


@dataclass(frozen=True, slots=True)
class GravityModelComparison:
    parent_identifier: str
    child_identifier: str
    added_parameters: int
    objective_change: float
    data_log_likelihood_change: float
    negative_binomial_deviance_change: float
    poisson_deviance_change: float
    rmse_change: float
    runtime_change_seconds: float


@dataclass(frozen=True, slots=True)
class GravityLineageProgressionResult:
    lineage: GravityModelLineage
    recommendations: GravityRelaxationRecommendationReport
    warm_start: GravityChildWarmStart
    parent: GravityModelNode
    child: GravityModelNode
    comparison: GravityModelComparison


def gravity_measurement_identity(
    *, measurement_indices: object, label: str
) -> str:
    indices = np.asarray(measurement_indices)
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or np.any(indices < 0):
        raise ValueError("measurement_indices must be a non-negative integer vector.")
    if np.unique(indices).size != indices.size:
        raise ValueError("measurement_indices must be unique.")
    if not label:
        raise ValueError("measurement identity label must be nonempty.")
    return fingerprint({"schema_version": 1, "label": label, "indices": indices})


def _runtime_memory(
    result: GravityEstimationResult, problem: GravityObjectiveProblem
) -> GravityRuntimeMemoryDiagnostics:
    array_bytes = sum(
        int(getattr(result, name).nbytes)
        for name in (
            "raw_parameters",
            "physical_parameters",
            "free_od_demand",
            "active_od_demand",
            "full_od_demand",
            "predicted_measurements",
            "gradient",
        )
    )
    selected = result.strategy_selection.selected
    diagnostic = next(
        item
        for item in result.strategy_selection.candidates
        if item.strategy == selected
    )
    return GravityRuntimeMemoryDiagnostics(
        result.elapsed_seconds,
        array_bytes,
        problem.operator.metrics.stored_bytes,
        problem.operator.metrics.peak_construction_bytes,
        selected,
        diagnostic.compilation_seconds,
        diagnostic.warm_execution_seconds,
    )


def create_gravity_model_node(
    *,
    result: GravityEstimationResult,
    problem: GravityObjectiveProblem,
    compact_layout: CompactODAssignmentLayout,
    adequacy_report: GravityAdequacyReport,
    calibration_measurement_identity: str,
    validation_measurement_identity: str,
    parent: GravityModelNode | None = None,
    relaxation_applied: GravityEffectScope | None = None,
) -> GravityModelNode:
    """Freeze a completed fit and all required provenance into one lineage node."""
    expected = gravity_model_fingerprint(problem, compact_layout)
    if result.model_fingerprint != expected:
        raise ValueError("gravity result and model-node problem fingerprints differ.")
    if adequacy_report.model_fingerprint != expected:
        raise ValueError("adequacy report and model-node problem fingerprints differ.")
    if (parent is None) != (relaxation_applied is None):
        raise ValueError("a child node requires both parent and applied relaxation.")
    if parent is not None:
        if parent.specification == problem.parameter_layout.specification:
            raise ValueError("a child node must change the parent specification.")
        if parent.calibration_measurement_identity != calibration_measurement_identity:
            raise ValueError("parent and child calibration-measurement identities differ.")
        if parent.validation_measurement_identity != validation_measurement_identity:
            raise ValueError("parent and child validation-measurement identities differ.")
        assert relaxation_applied is not None
        if (
            remove_gravity_relaxation(
                problem.parameter_layout.specification, relaxation_applied
            )
            != parent.specification
        ):
            raise ValueError(
                "the declared child relaxation does not reconstruct the parent specification."
            )
    optimizer = GravityOptimizerState(
        result.status,
        result.success,
        result.message,
        result.iterations,
        result.objective,
        result.data_log_likelihood,
        float(np.max(np.abs(result.gradient), initial=0.0)),
    )
    assigned = GravityAssignedCountDiagnostics(
        adequacy_report.report_fingerprint,
        adequacy_report.measurements,
        adequacy_report.observed_total,
        adequacy_report.modeled_total,
        adequacy_report.negative_binomial_deviance,
        adequacy_report.poisson_deviance,
        adequacy_report.mae,
        adequacy_report.rmse,
        adequacy_report.weighted_rmse,
    )
    runtime = _runtime_memory(result, problem)
    operator = problem.operator
    payload = {
        "schema_version": 1,
        "parent_identifier": None if parent is None else parent.model_identifier,
        "relaxation_applied": None if relaxation_applied is None else relaxation_applied.value,
        "specification": problem.parameter_layout.specification.to_dict(),
        "parameter_layout": problem.parameter_layout.fingerprint,
        "raw_parameters": result.raw_parameters,
        "physical_parameters": result.physical_parameters,
        "feature_fingerprint": problem.features.fingerprint,
        "compact_layout_fingerprint": compact_layout.fingerprint,
        "assignment_fingerprint": operator.assignment_fingerprint,
        "graph_fingerprint": operator.graph_fingerprint,
        "measurement_mapping_fingerprint": operator.mapping_fingerprint,
        "optimizer_state": asdict(optimizer),
        "assigned_count_diagnostics": asdict(assigned),
        "runtime_memory_diagnostics": asdict(runtime),
        "checkpoint_path": None if result.checkpoint_path is None else str(result.checkpoint_path),
        "calibration_measurement_identity": calibration_measurement_identity,
        "validation_measurement_identity": validation_measurement_identity,
    }
    identifier = fingerprint(payload)
    return GravityModelNode(
        1,
        identifier,
        None if parent is None else parent.model_identifier,
        relaxation_applied,
        problem.parameter_layout.specification,
        problem.parameter_layout,
        result.raw_parameters,
        result.physical_parameters,
        problem.features.fingerprint,
        compact_layout.fingerprint,
        operator.assignment_fingerprint,
        operator.graph_fingerprint,
        operator.mapping_fingerprint,
        optimizer,
        assigned,
        runtime,
        result.checkpoint_path,
        calibration_measurement_identity,
        validation_measurement_identity,
        result,
    )


def list_applicable_gravity_relaxations(
    *, parent: GravityModelNode, problem: GravityObjectiveProblem
) -> tuple[GravityRelaxationInfo, ...]:
    if parent.specification != problem.parameter_layout.specification:
        raise ValueError("parent node and progression problem specifications differ.")
    result = []
    for scope in (
        GravityEffectScope.DESTINATION_ZONE,
        GravityEffectScope.TIME_PERIOD,
        GravityEffectScope.ORIGIN_ZONE,
    ):
        try:
            _, info = add_gravity_relaxation(
                parent.specification, features=problem.features, scope=scope
            )
        except ValueError:
            continue
        result.append(info)
    return tuple(result)


def construct_gravity_child_warm_start(
    *,
    parent: GravityModelNode,
    problem: GravityObjectiveProblem,
    selected_relaxation: GravityEffectScope,
    ridge: float = 1.0,
    tolerance: float = 1.0e-10,
) -> GravityChildWarmStart:
    """Accept one explicit selection and verify exact parent prediction reproduction."""
    if parent.specification != problem.parameter_layout.specification:
        raise ValueError("parent node and child-planning problem specifications differ.")
    specification, info = add_gravity_relaxation(
        parent.specification,
        features=problem.features,
        scope=selected_relaxation,
        ridge=ridge,
    )
    layout = GravityParameterLayout(specification, parent.parameter_layout.positivity_floor)
    raw = warm_start_gravity_parameters(
        parent.parameter_layout, layout, parent.raw_parameter_estimates
    )
    child_problem = replace(problem, parameter_layout=layout)
    parent_mean = np.asarray(
        predict_gravity_measurements(parent.raw_parameter_estimates, problem=problem)[0]
    )
    child_mean = np.asarray(predict_gravity_measurements(raw, problem=child_problem)[0])
    difference = float(np.max(np.abs(parent_mean - child_mean), initial=0.0))
    if difference > tolerance:
        raise ValueError(
            f"child warm start does not reproduce parent predictions: {difference:.6g}."
        )
    return GravityChildWarmStart(
        parent.model_identifier,
        selected_relaxation,
        info,
        specification,
        layout,
        raw,
        difference,
        True,
    )


def compare_gravity_model_nodes(
    parent: GravityModelNode, child: GravityModelNode
) -> GravityModelComparison:
    if child.parent_identifier != parent.model_identifier:
        raise ValueError("the compared child does not descend from the parent.")
    return GravityModelComparison(
        parent.model_identifier,
        child.model_identifier,
        child.parameter_layout.size - parent.parameter_layout.size,
        child.optimizer_state.objective - parent.optimizer_state.objective,
        child.optimizer_state.data_log_likelihood - parent.optimizer_state.data_log_likelihood,
        child.assigned_count_diagnostics.negative_binomial_deviance
        - parent.assigned_count_diagnostics.negative_binomial_deviance,
        child.assigned_count_diagnostics.poisson_deviance
        - parent.assigned_count_diagnostics.poisson_deviance,
        child.assigned_count_diagnostics.rmse - parent.assigned_count_diagnostics.rmse,
        child.runtime_memory_diagnostics.estimation_seconds
        - parent.runtime_memory_diagnostics.estimation_seconds,
    )


def progress_gravity_model_lineage(
    *,
    lineage: GravityModelLineage,
    parent_identifier: str,
    selected_relaxation: GravityEffectScope,
    problem: GravityObjectiveProblem,
    compact_layout: CompactODAssignmentLayout,
    parent_adequacy_report: GravityAdequacyReport,
    calibration_measurement_identity: str,
    validation_measurement_identity: str,
    estimator_config: GravityEstimatorConfig = GravityEstimatorConfig(),
    execution_policy: GravityExecutionPolicy = GravityExecutionPolicy(),
    recommendation_config: GravityRecommendationConfig = GravityRecommendationConfig(),
    validation_metadata: GravityValidationMetadata | None = None,
    adequacy_config: GravityAdequacyConfig = GravityAdequacyConfig(),
    ridge: float = 1.0,
) -> GravityLineageProgressionResult:
    """Run one explicitly selected, verified, immutable parent-to-child step."""
    parent = lineage.load_completed_parent(parent_identifier)
    if parent.calibration_measurement_identity != calibration_measurement_identity:
        raise ValueError("parent and requested calibration-measurement identities differ.")
    if parent.validation_measurement_identity != validation_measurement_identity:
        raise ValueError("parent and requested validation-measurement identities differ.")
    recommendations = recommend_gravity_relaxations(
        result=parent.estimation_result,
        problem=problem,
        compact_layout=compact_layout,
        adequacy_report=parent_adequacy_report,
        metadata=validation_metadata,
        config=recommendation_config,
    )
    warm = construct_gravity_child_warm_start(
        parent=parent,
        problem=problem,
        selected_relaxation=selected_relaxation,
        ridge=ridge,
    )
    child_problem = replace(problem, parameter_layout=warm.child_parameter_layout)
    child_result = estimate_gravity_model(
        problem=child_problem,
        compact_layout=compact_layout,
        initial_raw_parameters=warm.raw_parameters,
        config=estimator_config,
        execution=execution_policy,
    )
    child_adequacy = validate_full_data_gravity_adequacy(
        result=child_result,
        problem=child_problem,
        compact_layout=compact_layout,
        metadata=validation_metadata,
        config=adequacy_config,
    )
    child = create_gravity_model_node(
        result=child_result,
        problem=child_problem,
        compact_layout=compact_layout,
        adequacy_report=child_adequacy,
        calibration_measurement_identity=calibration_measurement_identity,
        validation_measurement_identity=validation_measurement_identity,
        parent=parent,
        relaxation_applied=selected_relaxation,
    )
    updated = lineage.append(child)
    return GravityLineageProgressionResult(
        updated,
        recommendations,
        warm,
        parent,
        child,
        compare_gravity_model_nodes(parent, child),
    )
