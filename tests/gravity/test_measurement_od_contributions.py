from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from public_transportation.inference.assignment_contract import (
    AssignmentArtifactIdentity,
    CanonicalAssignmentIndex,
    CanonicalMeasurement,
    CanonicalODTimeCell,
    CanonicalTimeInterval,
)
from public_transportation.inference.gravity import (
    GRAVITY_REPORT_PROVENANCE_FIELDS,
    GravityValidationMetadata,
    read_gravity_viewer_bundle,
    write_gravity_detailed_report,
    write_gravity_viewer_bundle,
)
from public_transportation.inference.temporal_assignment_blocks import (
    TemporalBlockAssignmentOperator,
    TemporalBlockConstructionDiagnostics,
    TemporalBlockKey,
    TemporalSparseBlock,
)
from public_transportation.inference.temporal_assignment_persistence import (
    save_temporal_block_operator,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout

from tests.gravity.test_viewer_bundle import _inputs


def _mapping_bundle_inputs(tmp_path, *, changed_network: bool = False, save=True):
    base_result, _, _ = _inputs(tmp_path)
    layout = ODParameterLayout(
        num_od_total=3,
        od_keys=(
            ("o1", "d1", "am"),
            ("o1", "d2", "am"),
            ("o1", "d3", "am"),
        ),
        free_od_indices=(0, 1),
        fixed_od_indices=(2,),
        fixed_od_values=(0.5,),
        free_baseline_values=(1.0, 2.0),
        fixed_zero_indices=(),
        fixed_positive_indices=(2,),
    )
    canonical_index = CanonicalAssignmentIndex(
        time_intervals=(CanonicalTimeInterval("am", 0, 3600),),
        demand_cells=(
            CanonicalODTimeCell(0, "o1", "d1", "am", "free", 0),
            CanonicalODTimeCell(1, "o1", "d2", "am", "free", 1),
            CanonicalODTimeCell(2, "o1", "d3", "am", "fixed_positive", None, 0.5),
        ),
        measurements=(
            CanonicalMeasurement(0, "m0", "boarding", "o1", "am"),
            CanonicalMeasurement(1, "m1", "boarding", "d1", "am"),
            CanonicalMeasurement(2, "m2", "alighting", "d2", "am"),
        ),
        source_od_layout_fingerprint=layout.fingerprint,
        source_compact_layout_fingerprint="compact-layout-fingerprint",
    )
    identity = AssignmentArtifactIdentity(
        canonical_index_fingerprint=canonical_index.artifact_fingerprint,
        network_fingerprint="network-other" if changed_network else "network",
        timetable_fingerprint="assignment-fingerprint",
        temporal_discretization_fingerprint="temporal-discretization",
        route_choice_fingerprint="route-choice",
        departure_choice_fingerprint="departure-choice",
        feasibility_fingerprint="feasibility",
        measurement_mapping_fingerprint="mapping-fingerprint",
        coefficient_policy_fingerprint="coefficient-policy",
        numeric_dtype="float64",
    )
    result = replace(
        base_result,
        active_od_demand=np.asarray((1.0, 2.0, 0.5)),
        full_od_demand=np.asarray((1.0, 2.0, 0.5)),
        predicted_measurements=np.asarray((8.0, 2.625, 1e-8)),
        direct_operator_artifact_fingerprint=identity.fingerprint,
        latent_measurements=np.asarray((4.0, 5.25, 0.0)),
        od_measurement_contribution=np.asarray((3.0, 2.0, 0.0)),
        additive_measurement_contributions=(np.asarray((0.5, 0.25, 0.0)),),
        observation_scales=np.asarray((2.0, 0.5, 3.0)),
    )
    provenance = {
        field: f"{field}-value" for field in GRAVITY_REPORT_PROVENANCE_FIELDS
    }
    provenance.update(
        {
            "artifact_identity_fingerprint": identity.fingerprint,
            "assignment_fingerprint": identity.timetable_fingerprint,
            "binding_fingerprint": canonical_index.binding_fingerprint,
            "canonical_index_fingerprint": canonical_index.artifact_fingerprint,
            "compact_layout_fingerprint": "compact-layout-fingerprint",
            "mapping_fingerprint": identity.measurement_mapping_fingerprint,
            "od_layout_fingerprint": layout.fingerprint,
        }
    )
    metadata = GravityValidationMetadata(
        3,
        measurement_type=np.asarray(("boarding", "boarding", "alighting")),
        line=np.asarray(("L1", "L1", "L1")),
        stop=np.asarray(("o1", "d1", "d2")),
        time_period=np.asarray(("am", "am", "am")),
    )
    report = write_gravity_detailed_report(
        result=result,
        observations=np.asarray((3.0, 5.0, 7.0)),
        predicted_measurements=result.predicted_measurements,
        od_layout=layout,
        metadata=metadata,
        likelihood="poisson",
        output_directory=tmp_path / "report-with-mapping",
        provenance=provenance,
    )
    operator = TemporalBlockAssignmentOperator(
        canonical_index=canonical_index,
        identity=identity,
        blocks=(
            TemporalSparseBlock(
                key=TemporalBlockKey("am", "am"),
                row_indices=np.asarray((0, 0, 1), dtype=np.int32),
                column_indices=np.asarray((0, 1, 0), dtype=np.int32),
                values=np.asarray((1.0, 1.0, 2.0)),
                number_of_measurements=3,
                number_of_demand_cells=2,
            ),
        ),
        fixed_measurement_offset=np.asarray((0.5, 3.0, 0.0)),
        diagnostics=TemporalBlockConstructionDiagnostics(
            construction_seconds=0.0,
            nonzero_entries=3,
            retained_l1_mass=4.0,
            removed_l1_mass=0.0,
            zero_tolerance=0.0,
            columns_processed=2,
        ),
    )
    artifact_path = tmp_path / "linear-mapping"
    if save:
        save_temporal_block_operator(artifact_path, operator)
    return result, report, provenance, artifact_path


def test_sparse_mapping_decomposes_measurements_by_od_and_closes_components(tmp_path):
    result, report, provenance, artifact = _mapping_bundle_inputs(tmp_path)
    bundle = write_gravity_viewer_bundle(
        output_directory=tmp_path / "bundle",
        fit_result=result,
        validation_result=report.adequacy,
        report=report,
        metadata={"provenance": provenance},
        linear_mapping_artifact=artifact,
    )

    assert bundle.measurement_od_contributions_available
    assert bundle.viewer_summary["measurement_od_contributions"][
        "observation_scale_source"
    ] == "fit_result.observation_scales"
    assert bundle.viewer_summary["measurement_od_contributions"][
        "fixed_positive_od_attribution_available"
    ] is False
    contributors = bundle.query_measurement_od_contributions(0)
    assert contributors["od_index"].tolist() == [1, 0]
    np.testing.assert_allclose(
        contributors["linear_flow_contribution"], (2.0, 1.0)
    )
    np.testing.assert_allclose(
        contributors["modeled_mean_contribution"], (4.0, 2.0)
    )
    assert contributors["origin_stop_id"].tolist() == ["o1", "o1"]
    assert contributors["measurement_type"].tolist() == ["boarding", "boarding"]

    affected = bundle.query_od_measurement_contributions(0)
    assert affected["measurement_index"].tolist() == [0, 1]
    np.testing.assert_allclose(
        affected["modeled_mean_contribution"], (2.0, 1.0)
    )
    with pytest.raises(ValueError, match="bounded contribution-query window"):
        bundle.query_measurement_od_contributions(0, page=50, page_size=100)
    components = bundle.query_measurement_contribution_components(0)
    assert components is not None
    assert components["fixed_offset_contribution"] == pytest.approx(1.0)
    assert components["fixed_positive_od_attribution_available"] is False
    assert "aggregated in the mapping's fixed offset" in components[
        "fixed_positive_od_attribution_message"
    ]
    assert components["additive_contribution"] == pytest.approx(1.0)
    assert components["mean_floor_adjustment"] == pytest.approx(0.0)
    assert sum(contributors["modeled_mean_contribution"]) + components[
        "fixed_offset_contribution"
    ] + components["additive_contribution"] == pytest.approx(
        components["modeled_mean"]
    )
    floor_components = bundle.query_measurement_contribution_components(2)
    assert floor_components is not None
    assert floor_components["mean_floor_adjustment"] == pytest.approx(1e-8)
    for measurement_index in range(3):
        row_components = bundle.query_measurement_contribution_components(
            measurement_index
        )
        assert row_components is not None
        row_contributors = bundle.query_measurement_od_contributions(
            measurement_index
        )
        assert row_contributors["modeled_mean_contribution"].sum() + sum(
            float(row_components[name])
            for name in (
                "fixed_offset_contribution",
                "additive_contribution",
                "mean_floor_adjustment",
            )
        ) == pytest.approx(float(row_components["modeled_mean"]))

    reopened = read_gravity_viewer_bundle(bundle.path)
    assert reopened.query_measurement_od_contributions(1)["od_index"].tolist() == [0]


def test_mapping_identity_mismatch_and_corrupt_payload_fail_closed(tmp_path):
    result, report, provenance, mismatched = _mapping_bundle_inputs(
        tmp_path / "mismatch", changed_network=True
    )
    result = replace(result, direct_operator_artifact_fingerprint="different-fit-artifact")
    with pytest.raises(ValueError, match="identity does not match"):
        write_gravity_viewer_bundle(
            output_directory=tmp_path / "mismatch-bundle",
            fit_result=result,
            validation_result=report.adequacy,
            report=report,
            metadata={"provenance": provenance},
            linear_mapping_artifact=mismatched,
        )

    result, report, provenance, artifact = _mapping_bundle_inputs(tmp_path / "corrupt")
    block_path = artifact / "blocks" / "block-000000.npz"
    block_path.write_bytes(block_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="output size is invalid"):
        write_gravity_viewer_bundle(
            output_directory=tmp_path / "corrupt-bundle",
            fit_result=result,
            validation_result=report.adequacy,
            report=report,
            metadata={"provenance": provenance},
            linear_mapping_artifact=artifact,
        )


def test_missing_exact_mean_scale_keeps_bundle_readable_but_marks_breakdown_unavailable(
    tmp_path,
):
    result, report, provenance, artifact = _mapping_bundle_inputs(tmp_path)
    result = replace(result, observation_scales=None)
    report = write_gravity_detailed_report(
        result=result,
        observations=np.asarray((3.0, 5.0, 7.0)),
        predicted_measurements=result.predicted_measurements,
        od_layout=ODParameterLayout(
            num_od_total=3,
            od_keys=(("o1", "d1", "am"), ("o1", "d2", "am"), ("o1", "d3", "am")),
            free_od_indices=(0, 1),
            fixed_od_indices=(2,),
            fixed_od_values=(0.5,),
            free_baseline_values=(1.0, 2.0),
            fixed_zero_indices=(),
            fixed_positive_indices=(2,),
        ),
        metadata=GravityValidationMetadata(
            3,
            measurement_type=np.asarray(("boarding", "boarding", "alighting")),
            line=np.asarray(("L1", "L1", "L1")),
            stop=np.asarray(("o1", "d1", "d2")),
            time_period=np.asarray(("am", "am", "am")),
        ),
        likelihood="poisson",
        output_directory=tmp_path / "report-without-scale",
        provenance=provenance,
    )
    bundle = write_gravity_viewer_bundle(
        output_directory=tmp_path / "bundle-no-scale",
        fit_result=result,
        validation_result=report.adequacy,
        report=report,
        metadata={"provenance": provenance},
        linear_mapping_artifact=artifact,
    )
    assert not bundle.measurement_od_contributions_available
    assert "pass rho" in bundle.viewer_summary["measurement_od_contributions"]["message"]
    with pytest.raises(RuntimeError, match="pass rho"):
        bundle.query_measurement_od_contributions(0)
