from __future__ import annotations

import jax
import numpy as np
import pytest

from public_transportation.inference.gravity import (
    GravityEffectScope,
    GravityEstimatorConfig,
    GravityExecutionPolicy,
    GravityModelLineage,
    compare_gravity_model_nodes,
    construct_gravity_child_warm_start,
    create_gravity_model_node,
    gravity_measurement_identity,
    list_applicable_gravity_relaxations,
    progress_gravity_model_lineage,
    validate_full_data_gravity_adequacy,
)
from tests.gravity.test_phase6_recommendations import metadata, recommendation_case


def parent_node(*, destination_effect: float = 0.35):
    problem, compact, result = recommendation_case(
        destination_effect=destination_effect
    )
    report = validate_full_data_gravity_adequacy(
        result=result,
        problem=problem,
        compact_layout=compact,
        metadata=metadata(),
    )
    calibration = gravity_measurement_identity(
        measurement_indices=np.arange(problem.observations.size),
        label="full calibration counts",
    )
    validation = gravity_measurement_identity(
        measurement_indices=np.arange(problem.observations.size),
        label="full-data adequacy counts",
    )
    node = create_gravity_model_node(
        result=result,
        problem=problem,
        compact_layout=compact,
        adequacy_report=report,
        calibration_measurement_identity=calibration,
        validation_measurement_identity=validation,
    )
    return problem, compact, report, node


def test_root_node_stores_complete_immutable_provenance_and_is_deterministic():
    with jax.enable_x64():
        problem, compact, report, node = parent_node()
        duplicate = create_gravity_model_node(
            result=node.estimation_result,
            problem=problem,
            compact_layout=compact,
            adequacy_report=report,
            calibration_measurement_identity=node.calibration_measurement_identity,
            validation_measurement_identity=node.validation_measurement_identity,
        )
        assert duplicate.model_identifier == node.model_identifier
        assert node.parent_identifier is None
        assert node.relaxation_applied is None
        assert node.specification == problem.parameter_layout.specification
        assert node.parameter_layout == problem.parameter_layout
        assert node.feature_fingerprint == problem.features.fingerprint
        assert node.compact_layout_fingerprint == compact.fingerprint
        assert node.assignment_fingerprint == problem.operator.assignment_fingerprint
        assert node.graph_fingerprint == problem.operator.graph_fingerprint
        assert node.measurement_mapping_fingerprint == problem.operator.mapping_fingerprint
        assert node.optimizer_state.objective == node.estimation_result.objective
        assert node.assigned_count_diagnostics.report_fingerprint == report.report_fingerprint
        assert node.runtime_memory_diagnostics.result_array_bytes > 0
        assert node.runtime_memory_diagnostics.operator_stored_bytes > 0
        assert not node.raw_parameter_estimates.flags.writeable
        with pytest.raises(ValueError):
            node.raw_parameter_estimates[0] = 9


def test_lineage_loads_parent_lists_relaxations_and_verifies_selected_warm_start():
    with jax.enable_x64():
        problem, _, _, node = parent_node()
        lineage = GravityModelLineage((node,))
        assert lineage.load_completed_parent(node.model_identifier) is node
        assert {item.scope for item in list_applicable_gravity_relaxations(
            parent=node, problem=problem
        )} == {
            GravityEffectScope.DESTINATION_ZONE,
            GravityEffectScope.TIME_PERIOD,
            GravityEffectScope.ORIGIN_ZONE,
        }
        warm = construct_gravity_child_warm_start(
            parent=node,
            problem=problem,
            selected_relaxation=GravityEffectScope.DESTINATION_ZONE,
        )
        assert warm.verified
        assert warm.maximum_parent_prediction_difference == pytest.approx(0.0)
        assert warm.raw_parameters.size == node.raw_parameter_estimates.size + 1
        assert warm.raw_parameters[-1] == 0
        with pytest.raises(ValueError, match="atomic"):
            construct_gravity_child_warm_start(
                parent=node,
                problem=problem,
                selected_relaxation=GravityEffectScope.GLOBAL,
            )


def test_explicit_progression_estimates_child_compares_and_preserves_both_nodes():
    with jax.enable_x64():
        problem, compact, report, node = parent_node(destination_effect=0.5)
        lineage = GravityModelLineage((node,))
        progressed = progress_gravity_model_lineage(
            lineage=lineage,
            parent_identifier=node.model_identifier,
            selected_relaxation=GravityEffectScope.DESTINATION_ZONE,
            problem=problem,
            compact_layout=compact,
            parent_adequacy_report=report,
            calibration_measurement_identity=node.calibration_measurement_identity,
            validation_measurement_identity=node.validation_measurement_identity,
            estimator_config=GravityEstimatorConfig(maximum_iterations=3),
            execution_policy=GravityExecutionPolicy(gradient_strategy="adjoint"),
            validation_metadata=metadata(),
        )
        assert len(progressed.lineage.nodes) == 2
        assert progressed.lineage.nodes[0] is node
        assert progressed.child.parent_identifier == node.model_identifier
        assert progressed.child.relaxation_applied is GravityEffectScope.DESTINATION_ZONE
        assert progressed.child.model_identifier != node.model_identifier
        assert progressed.comparison.added_parameters == 1
        assert progressed.comparison == compare_gravity_model_nodes(
            node, progressed.child
        )
        assert progressed.recommendations.advisory_only
        assert progressed.warm_start.verified
        assert progressed.lineage.load_completed_parent(
            progressed.child.model_identifier
        ) is progressed.child


def test_lineage_rejects_identity_and_parent_mismatches():
    with jax.enable_x64():
        problem, compact, report, node = parent_node()
        with pytest.raises(ValueError, match="calibration"):
            progress_gravity_model_lineage(
                lineage=GravityModelLineage((node,)),
                parent_identifier=node.model_identifier,
                selected_relaxation=GravityEffectScope.TIME_PERIOD,
                problem=problem,
                compact_layout=compact,
                parent_adequacy_report=report,
                calibration_measurement_identity="different",
                validation_measurement_identity=node.validation_measurement_identity,
            )
        with pytest.raises(ValueError, match="unique"):
            gravity_measurement_identity(
                measurement_indices=np.asarray((0, 0)), label="duplicate"
            )
        with pytest.raises(KeyError, match="unknown"):
            GravityModelLineage((node,)).load_completed_parent("missing")
