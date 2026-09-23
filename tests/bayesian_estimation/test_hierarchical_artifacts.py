"""Tests for the parent-aware scheduled assignment artifact hierarchy."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse

from public_transportation.inference.hierarchical_artifacts import (
    ARTIFACT_LAYERS,
    DAG_PARENT_LAYERS,
    EstimationAssignmentMapping,
    HierarchicalArtifactStore,
    HierarchicalArtifactUnavailableError,
    HierarchicalProgressReporter,
    ObsoleteArtifactFormatError,
    derive_layer_fingerprints,
    prepare_hierarchical_assignment_mapping,
)


def _specifications(*, theta: float = 1.0, structural_zero: str = "zero"):
    return {
        layer: {
            "scientific_parameters": (
                {"theta": theta}
                if layer == "route_choice_materialization"
                else {"structural_zero": structural_zero}
                if layer == "estimation_assignment_mapping"
                else {}
            ),
            "input_fingerprints": {},
        }
        for layer in ARTIFACT_LAYERS
    }


def _mapping(specifications):
    expected = derive_layer_fingerprints(specifications)
    parents = {
        name: expected[name]
        for name in DAG_PARENT_LAYERS["estimation_assignment_mapping"]
    }
    return EstimationAssignmentMapping(
        A_free=sparse.csr_array([[1.0, 0.0], [0.5, 2.0]]),
        fixed_offset=np.asarray([3.0, 4.0]),
        free_column_index=np.asarray([0, 1]),
        measurement_row_index=np.asarray([0, 1]),
        full_to_compact=np.asarray([0, 1, -1]),
        compact_to_full=np.asarray([0, 1]),
        full_od_fingerprint="full-od",
        active_od_fingerprint="active-od",
        parent_fingerprints=parents,
        fingerprint=expected["estimation_assignment_mapping"],
    )


def test_hierarchy_build_then_fresh_process_reuse(tmp_path):
    specifications = _specifications()
    progress_path = tmp_path / "progress.jsonl"
    reporter = HierarchicalProgressReporter(progress_path, interval_seconds=0.0)
    first = prepare_hierarchical_assignment_mapping(
        root=tmp_path,
        specifications=specifications,
        mapping_builder=lambda: _mapping(specifications),
        activation_policy="build_or_reuse",
        progress=reporter,
    )
    assert first.rebuilt_layers == ARTIFACT_LAYERS
    assert first.mapping.matvec(np.asarray([2.0, 3.0])).tolist() == [2.0, 7.0]

    second = prepare_hierarchical_assignment_mapping(
        root=tmp_path,
        specifications=specifications,
        activation_policy="reuse_only",
    )
    assert second.reused_layers == ARTIFACT_LAYERS
    assert second.mapping.rmatvec(np.asarray([1.0, 1.0])).tolist() == [1.5, 2.0]
    events = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert {event["artifact_layer"] for event in events} == set(ARTIFACT_LAYERS)
    assert all(event["schema_version"] == 1 for event in events)


def test_l7_payload_roundtrip_preserves_explicit_fixed_positive_mapping(tmp_path):
    specifications = _specifications()
    expected = derive_layer_fingerprints(specifications)
    parents = {
        name: expected[name]
        for name in DAG_PARENT_LAYERS["estimation_assignment_mapping"]
    }
    mapping = EstimationAssignmentMapping(
        A_free=sparse.csr_array([[1.0, 0.0], [0.5, 2.0]]),
        fixed_offset=np.asarray([3.0, 4.0]),
        free_column_index=np.asarray([0, 1]),
        measurement_row_index=np.asarray([0, 1]),
        full_to_compact=np.asarray([0, 1, -1]),
        compact_to_full=np.asarray([0, 1]),
        fixed_positive_full_index=np.asarray([2]),
        fixed_positive_values=np.asarray([7.0]),
        full_od_fingerprint="full-od",
        active_od_fingerprint="active-od",
        parent_fingerprints=parents,
        fingerprint=expected["estimation_assignment_mapping"],
    )
    directory = tmp_path / "payload"
    mapping.to_payload(directory)
    restored = EstimationAssignmentMapping.from_payload(directory)
    np.testing.assert_array_equal(restored.fixed_positive_full_index, [2])
    np.testing.assert_array_equal(restored.fixed_positive_values, [7.0])
    np.testing.assert_allclose(
        restored.fixed_offset + restored.matvec(np.asarray([2.0, 3.0])),
        [5.0, 11.0],
    )


def test_execution_controls_do_not_change_layer_identity(tmp_path):
    specifications = _specifications()
    first = prepare_hierarchical_assignment_mapping(
        root=tmp_path,
        specifications=specifications,
        mapping_builder=lambda: _mapping(specifications),
        activation_policy="build_or_reuse",
        execution_parameters={"workers": 1, "batch_size": 8},
    )
    second = prepare_hierarchical_assignment_mapping(
        root=tmp_path,
        specifications=specifications,
        activation_policy="reuse_only",
        execution_parameters={"workers": 32, "batch_size": 1024},
    )
    assert first.manifests["estimation_assignment_mapping"].fingerprint == (
        second.manifests["estimation_assignment_mapping"].fingerprint
    )


def test_manifests_use_explicit_dag_parent_sets(tmp_path):
    specifications = _specifications()
    prepared = prepare_hierarchical_assignment_mapping(
        root=tmp_path,
        specifications=specifications,
        mapping_builder=lambda: _mapping(specifications),
        activation_policy="build_or_reuse",
    )
    assert prepared.manifests["route_choice_materialization"].parent_fingerprints == {
        "route_choice_basis": prepared.manifests["route_choice_basis"].fingerprint
    }
    assert prepared.manifests["estimation_assignment_mapping"].parent_fingerprints == {
        "full_od_assignment_mapping": prepared.manifests[
            "full_od_assignment_mapping"
        ].fingerprint,
        "observation_projection": prepared.manifests[
            "observation_projection"
        ].fingerprint,
    }


def test_theta_change_invalidates_materialization_and_descendants(tmp_path):
    specifications = _specifications(theta=1.0)
    prepare_hierarchical_assignment_mapping(
        root=tmp_path,
        specifications=specifications,
        mapping_builder=lambda: _mapping(specifications),
        activation_policy="build_or_reuse",
    )
    changed = _specifications(theta=2.0)
    rebuilt = prepare_hierarchical_assignment_mapping(
        root=tmp_path,
        specifications=changed,
        mapping_builder=lambda: _mapping(changed),
        activation_policy="build_or_reuse",
    )
    assert rebuilt.reused_layers == (
        "scenario_base",
        "canonical_od_time_universe",
        "feasibility_support",
        "route_choice_basis",
    )
    assert rebuilt.rebuilt_layers == (
        "route_choice_materialization",
        "full_od_assignment_mapping",
        "observation_projection",
        "estimation_assignment_mapping",
    )


def test_strict_reuse_reports_missing_l7(tmp_path):
    specifications = _specifications()
    with pytest.raises(HierarchicalArtifactUnavailableError) as caught:
        prepare_hierarchical_assignment_mapping(
            root=tmp_path,
            specifications=specifications,
            activation_policy="reuse_only",
        )
    assert caught.value.artifact_layer == "estimation_assignment_mapping"
    assert caught.value.reason_code == "artifact_missing"


def test_fixed_offset_change_rebuilds_l7_only(tmp_path):
    specifications = _specifications()
    first_mapping = _mapping(specifications)
    prepare_hierarchical_assignment_mapping(
        root=tmp_path,
        specifications=specifications,
        mapping_builder=lambda: first_mapping,
        activation_policy="build_or_reuse",
        expected_fixed_offset_fingerprint=first_mapping.fixed_offset_fingerprint,
    )
    changed = EstimationAssignmentMapping(
        A_free=first_mapping.A_free,
        fixed_offset=np.asarray([8.0, 9.0]),
        free_column_index=first_mapping.free_column_index,
        measurement_row_index=first_mapping.measurement_row_index,
        full_to_compact=first_mapping.full_to_compact,
        compact_to_full=first_mapping.compact_to_full,
        full_od_fingerprint=first_mapping.full_od_fingerprint,
        active_od_fingerprint=first_mapping.active_od_fingerprint,
        parent_fingerprints=first_mapping.parent_fingerprints,
        fingerprint=first_mapping.fingerprint,
    )
    rebuilt = prepare_hierarchical_assignment_mapping(
        root=tmp_path,
        specifications=specifications,
        mapping_builder=lambda: changed,
        activation_policy="build_or_reuse",
        expected_fixed_offset_fingerprint=changed.fixed_offset_fingerprint,
    )
    assert rebuilt.reused_layers == ARTIFACT_LAYERS[:-1]
    assert rebuilt.rebuilt_layers == ("estimation_assignment_mapping",)
    np.testing.assert_array_equal(rebuilt.mapping.fixed_offset, [8.0, 9.0])


def test_old_manifest_is_not_reinterpreted(tmp_path):
    store = HierarchicalArtifactStore(tmp_path)
    path = store.artifact_path("scenario_base", "a" * 64)
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "complete": True}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ObsoleteArtifactFormatError):
        store.load("scenario_base", "a" * 64)


def test_strict_preparation_propagates_obsolete_parent_layer(tmp_path):
    specifications = _specifications()
    expected = derive_layer_fingerprints(specifications)
    path = HierarchicalArtifactStore(tmp_path).artifact_path(
        "scenario_base", expected["scenario_base"]
    )
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "complete": True}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ObsoleteArtifactFormatError):
        prepare_hierarchical_assignment_mapping(
            root=tmp_path,
            specifications=specifications,
            activation_policy="reuse_only",
        )


def test_incompatible_manifest_reports_parent_diagnostics(tmp_path):
    specifications = _specifications()
    prepare_hierarchical_assignment_mapping(
        root=tmp_path,
        specifications=specifications,
        mapping_builder=lambda: _mapping(specifications),
        activation_policy="build_or_reuse",
    )
    expected = derive_layer_fingerprints(specifications)
    manifest_path = HierarchicalArtifactStore(tmp_path).manifest_path(
        "estimation_assignment_mapping", expected["estimation_assignment_mapping"]
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["parent_fingerprints"]["observation_projection"] = "wrong-parent"
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    store = HierarchicalArtifactStore(tmp_path)
    with pytest.raises(HierarchicalArtifactUnavailableError) as caught:
        store.require(
            "estimation_assignment_mapping",
            expected["estimation_assignment_mapping"],
            expected_parents={
                name: expected[name]
                for name in DAG_PARENT_LAYERS["estimation_assignment_mapping"]
            },
        )
    details = caught.value.details
    assert details["expected_parent_fingerprints"]["observation_projection"] == (
        expected["observation_projection"]
    )
    assert details["actual_parent_fingerprints"]["observation_projection"] == (
        "wrong-parent"
    )
    assert details["first_incompatible_field"] == (
        "parent_fingerprints.observation_projection"
    )


def test_invalidation_rules_keep_unchanged_ancestors(tmp_path):
    specifications = _specifications()
    prepare_hierarchical_assignment_mapping(
        root=tmp_path,
        specifications=specifications,
        mapping_builder=lambda: _mapping(specifications),
        activation_policy="build_or_reuse",
    )

    measurement_changed = _specifications()
    measurement_changed["observation_projection"]["scientific_parameters"] = {
        "measurement_mapping": "changed"
    }
    changed = prepare_hierarchical_assignment_mapping(
        root=tmp_path,
        specifications=measurement_changed,
        mapping_builder=lambda: _mapping(measurement_changed),
        activation_policy="build_or_reuse",
    )
    assert changed.reused_layers == (
        "scenario_base",
        "canonical_od_time_universe",
        "feasibility_support",
        "route_choice_basis",
        "route_choice_materialization",
        "full_od_assignment_mapping",
    )
    assert changed.rebuilt_layers == (
        "observation_projection",
        "estimation_assignment_mapping",
    )

    feasibility_changed = _specifications()
    feasibility_changed["feasibility_support"]["scientific_parameters"] = {
        "maximum_transfers": 2
    }
    rebuilt = prepare_hierarchical_assignment_mapping(
        root=tmp_path,
        specifications=feasibility_changed,
        mapping_builder=lambda: _mapping(feasibility_changed),
        activation_policy="build_or_reuse",
    )
    assert rebuilt.reused_layers == ("scenario_base", "canonical_od_time_universe")
    assert rebuilt.rebuilt_layers == (
        "feasibility_support",
        "route_choice_basis",
        "route_choice_materialization",
        "full_od_assignment_mapping",
        "observation_projection",
        "estimation_assignment_mapping",
    )
