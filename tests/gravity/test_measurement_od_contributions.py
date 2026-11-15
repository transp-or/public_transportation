from __future__ import annotations

import json
from dataclasses import asdict, replace
from enum import Enum
from pathlib import Path

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
    MeasurementODContributionQuery,
    GravityValidationMetadata,
    read_gravity_viewer_bundle,
    write_gravity_detailed_report,
    write_gravity_viewer_bundle,
)
from public_transportation.inference.gravity.viewer_query_service import (
    GravityViewerQueryService,
)
from public_transportation.inference.temporal_assignment_blocks import (
    TemporalBlockAssignmentOperator,
    TemporalBlockConstructionDiagnostics,
    TemporalBlockKey,
    TemporalSparseBlock,
)
from public_transportation.inference.temporal_assignment_persistence import (
    open_temporal_block_artifact,
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


def _split_interval_query_inputs(tmp_path):
    base_result, _, _ = _inputs(tmp_path)
    canonical_index = CanonicalAssignmentIndex(
        time_intervals=(
            CanonicalTimeInterval("am", 0, 3600),
            CanonicalTimeInterval("pm", 3600, 7200),
        ),
        demand_cells=(
            CanonicalODTimeCell(0, "o1", "d1", "am", "free", 0),
            CanonicalODTimeCell(1, "o1", "d2", "pm", "free", 1),
            CanonicalODTimeCell(2, "o1", "d3", "am", "free", 2),
            CanonicalODTimeCell(3, "o1", "d4", "am", "fixed_positive", None, 0.75),
        ),
        measurements=(
            CanonicalMeasurement(0, "m0", "boarding", "d1", "am"),
            CanonicalMeasurement(1, "m1", "alighting", "d2", "pm"),
        ),
    )
    identity = AssignmentArtifactIdentity(
        canonical_index_fingerprint=canonical_index.artifact_fingerprint,
        network_fingerprint="query-network",
        timetable_fingerprint="query-assignment",
        temporal_discretization_fingerprint="query-time",
        route_choice_fingerprint="query-route-choice",
        departure_choice_fingerprint="query-departure-choice",
        feasibility_fingerprint="query-feasibility",
        measurement_mapping_fingerprint="query-measurement-mapping",
        coefficient_policy_fingerprint="query-coefficient-policy",
        numeric_dtype="float64",
    )
    operator = TemporalBlockAssignmentOperator(
        canonical_index=canonical_index,
        identity=identity,
        blocks=(
            TemporalSparseBlock(
                key=TemporalBlockKey("am", "am"),
                row_indices=np.asarray((0, 0), dtype=np.int32),
                column_indices=np.asarray((0, 2), dtype=np.int32),
                values=np.asarray((1.0, 3.0)),
                number_of_measurements=2,
                number_of_demand_cells=3,
            ),
            # Same logical block key, as when construction fragments are merged.
            TemporalSparseBlock(
                key=TemporalBlockKey("am", "am"),
                row_indices=np.asarray((0,), dtype=np.int32),
                column_indices=np.asarray((0,), dtype=np.int32),
                values=np.asarray((0.5,)),
                number_of_measurements=2,
                number_of_demand_cells=3,
            ),
            TemporalSparseBlock(
                key=TemporalBlockKey("am", "pm"),
                row_indices=np.asarray((0,), dtype=np.int32),
                column_indices=np.asarray((1,), dtype=np.int32),
                values=np.asarray((3.0,)),
                number_of_measurements=2,
                number_of_demand_cells=3,
            ),
            TemporalSparseBlock(
                key=TemporalBlockKey("pm", "pm"),
                row_indices=np.asarray((1,), dtype=np.int32),
                column_indices=np.asarray((1,), dtype=np.int32),
                values=np.asarray((0.25,)),
                number_of_measurements=2,
                number_of_demand_cells=3,
            ),
        ),
        fixed_measurement_offset=np.asarray((4.0, 1.0)),
        diagnostics=TemporalBlockConstructionDiagnostics(
            construction_seconds=0.0,
            nonzero_entries=5,
            retained_l1_mass=7.75,
            removed_l1_mass=0.0,
            zero_tolerance=0.0,
            columns_processed=3,
        ),
    )
    result = replace(
        base_result,
        free_od_demand=np.asarray((1.0, 2.0, 0.5)),
        active_od_demand=np.asarray((1.0, 2.0, 0.5, 0.75)),
        full_od_demand=np.asarray((1.0, 2.0, 0.5, 0.75)),
        predicted_measurements=np.asarray((27.5, 6.0)),
        direct_operator_artifact_fingerprint=identity.fingerprint,
        latent_measurements=np.asarray((13.25, 2.0)),
        od_measurement_contribution=np.asarray((9.0, 0.5)),
        additive_measurement_contributions=(np.asarray((0.25, 0.5)),),
        observation_scales=np.asarray((2.0, 3.0)),
    )
    artifact_path = tmp_path / "split-interval-mapping"
    save_temporal_block_operator(artifact_path, operator)
    return result, open_temporal_block_artifact(artifact_path), artifact_path


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
    summary = bundle.viewer_summary["measurement_od_contributions"]
    assert summary["query_mode"] == "local_service"
    assert (
        bundle.viewer_summary["measurement_od_contributions"]["observation_scale_source"]
        == "fit_result.observation_scales"
    )
    assert bundle.viewer_summary["measurement_od_contributions"]["fixed_positive_od_attribution_available"] is False
    query = MeasurementODContributionQuery(artifact=open_temporal_block_artifact(artifact), fit_result=result)
    page = query.query(0, page=0, page_size=10)
    contributors = page["rows"]
    assert [row["od_index"] for row in contributors] == [1, 0]
    assert [row["linear_flow_contribution"] for row in contributors] == pytest.approx((2.0, 1.0))
    assert [row["modeled_mean_contribution"] for row in contributors] == pytest.approx((4.0, 2.0))
    assert [row["origin_stop_id"] for row in contributors] == ["o1", "o1"]
    components = page["components"]
    assert components["fixed_offset_contribution"] == pytest.approx(1.0)
    assert components["additive_contribution"] == pytest.approx(1.0)
    assert components["mean_floor_adjustment"] == pytest.approx(0.0)
    assert sum(row["modeled_mean_contribution"] for row in contributors) + sum(
        components[name]
        for name in (
            "fixed_offset_contribution",
            "additive_contribution",
            "mean_floor_adjustment",
        )
    ) == pytest.approx(components["modeled_mean"])
    floor_components = query.query(2)["components"]
    assert floor_components["mean_floor_adjustment"] == pytest.approx(1e-8)

    with pytest.raises(RuntimeError, match="local gravity viewer query service"):
        bundle.query_measurement_od_contributions(0)

    reopened = read_gravity_viewer_bundle(bundle.path)
    assert reopened.measurement_od_contributions_available


def test_on_demand_query_selects_measurement_intervals_aggregates_duplicates_and_pages(
    tmp_path,
):
    result, reader, _artifact_path = _split_interval_query_inputs(tmp_path)
    requested_intervals = []

    class RecordingReader:
        def __getattr__(self, name):
            return getattr(reader, name)

        def iter_blocks_for_measurement_interval(self, interval_id):
            requested_intervals.append(interval_id)
            yield from reader.iter_blocks_for_measurement_interval(interval_id)

    query = MeasurementODContributionQuery(
        artifact=RecordingReader(),
        fit_result=result,
        max_cached_pages=2,
    )
    first = query.query(0, page=0, page_size=1)
    assert requested_intervals == ["am"]
    assert first["total"] == 3
    assert first["rows"][0]["od_index"] == 1
    assert first["rows"][0]["linear_flow_contribution"] == pytest.approx(6.0)
    assert first["rows"][0]["modeled_mean_contribution"] == pytest.approx(12.0)
    assert query.query(0, page=0, page_size=1) == first
    assert query.cache_hits == 1

    second = query.query(0, page=1, page_size=1)
    third = query.query(0, page=2, page_size=1)
    assert [second["rows"][0]["od_index"], third["rows"][0]["od_index"]] == [0, 2]
    assert second["rows"][0]["operator_coefficient"] == pytest.approx(1.5)
    assert second["rows"][0]["linear_flow_contribution"] == pytest.approx(1.5)
    assert first["components"] == {
        "fixed_offset_contribution": pytest.approx(8.0),
        "additive_contribution": pytest.approx(0.5),
        "od_contribution": pytest.approx(18.0),
        "mean_floor_adjustment": pytest.approx(1.0),
        "modeled_mean": pytest.approx(27.5),
    }
    assert [
        row["modeled_mean_contribution"] for row in (first["rows"] + second["rows"] + third["rows"])
    ] == pytest.approx((12.0, 3.0, 3.0))
    assert len(query._cache) <= 2

    affected_page = query.query_od(1, page=0, page_size=1)
    affected_next = query.query_od(1, page=1, page_size=1)
    assert affected_page["total"] == 2
    assert [
        affected_page["rows"][0]["measurement_index"],
        affected_next["rows"][0]["measurement_index"],
    ] == [0, 1]
    assert affected_page["rows"][0]["modeled_mean_contribution"] == pytest.approx(12.0)
    assert affected_next["rows"][0]["modeled_mean_contribution"] == pytest.approx(1.5)

    other_fit = replace(result, model_fingerprint="different-fit-fingerprint")
    other_query = MeasurementODContributionQuery(artifact=reader, fit_result=other_fit)
    assert other_query.cache_namespace != query.cache_namespace
    assert other_query.query(1)["rows"][0]["od_index"] == 1
    assert other_query.cache_misses == 1
    with pytest.raises(ValueError, match="bounded contribution-query window"):
        query.query(0, page=50, page_size=100)


def test_on_demand_query_verifies_only_selected_interval_payloads(tmp_path):
    result, reader, artifact_path = _split_interval_query_inputs(tmp_path)
    query = MeasurementODContributionQuery(artifact=reader, fit_result=result)
    unrelated = artifact_path / "blocks" / "block-000003.npz"
    with np.load(unrelated, allow_pickle=False) as saved:
        rows = np.asarray(saved["row_indices"])
        columns = np.asarray(saved["column_indices"])
        values = np.asarray(saved["values"])
    values[0] += 0.01
    np.savez_compressed(
        unrelated,
        row_indices=rows,
        column_indices=columns,
        values=values,
    )

    # Measurement 0 reads only the am interval; the damaged pm payload is not
    # loaded or needed to answer this bounded query.
    assert query.query(0, page_size=1)["total"] == 3
    # Querying measurement 1 selects pm and must fail its existing checksum /
    # payload validation contract before returning any contribution.
    with pytest.raises(ValueError, match="payload is missing or corrupt"):
        query.query(1)


def test_bundle_mapping_uses_query_descriptor_without_full_matrix_writer(tmp_path, monkeypatch):
    result, report, provenance, artifact = _mapping_bundle_inputs(tmp_path)

    def fail_if_called(**_kwargs):
        pytest.fail("bundle creation attempted to materialize the complete contribution matrix")

    monkeypatch.setattr(
        "public_transportation.inference.gravity.measurement_od_contributions.write_measurement_od_contributions_dataset",
        fail_if_called,
    )
    bundle = write_gravity_viewer_bundle(
        output_directory=tmp_path / "on-demand-bundle",
        fit_result=result,
        validation_result=report.adequacy,
        report=report,
        metadata={"provenance": provenance},
        linear_mapping_artifact=artifact,
    )
    summary = bundle.viewer_summary["measurement_od_contributions"]
    assert summary["query_mode"] == "local_service"
    assert summary["query_path"] == "/api/measurement-od-contributions"
    assert summary["mapping_artifact_fingerprint"] == provenance["artifact_identity_fingerprint"]
    assert not (bundle.path / "partitions" / "measurement_od_contributions").exists()
    assert not list(bundle.path.rglob("contributions.sqlite"))


def test_local_query_service_binds_fit_bundle_and_mapping_before_query(tmp_path):
    result, report, provenance, artifact = _mapping_bundle_inputs(tmp_path)
    bundle = write_gravity_viewer_bundle(
        output_directory=tmp_path / "service-bundle",
        fit_result=result,
        validation_result=report.adequacy,
        report=report,
        metadata={"provenance": provenance},
        linear_mapping_artifact=artifact,
    )
    fit_payload = {
        **provenance,
        "model_fingerprint": result.model_fingerprint,
        "result": asdict(result),
    }

    def json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Enum):
            return value.value
        raise TypeError(type(value).__name__)

    fit_path = tmp_path / "fit-manifest.json"
    fit_path.write_text(json.dumps(fit_payload, default=json_default), encoding="utf-8")
    service = GravityViewerQueryService(
        bundle=read_gravity_viewer_bundle(bundle.path),
        fit_manifest_path=fit_path,
        mapping_artifact_path=artifact,
    )
    response = service.query_measurement(0, page=0, page_size=1)
    assert response["fit_fingerprint"] == result.model_fingerprint
    assert response["mapping_artifact_fingerprint"] == provenance["artifact_identity_fingerprint"]
    assert response["rows"][0]["od_index"] == 1
    assert response["components"]["modeled_mean"] == pytest.approx(8.0)

    fit_payload["artifact_identity_fingerprint"] = "stale-fit-artifact"
    fit_path.write_text(json.dumps(fit_payload, default=json_default), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_identity_fingerprint differs"):
        GravityViewerQueryService(
            bundle=read_gravity_viewer_bundle(bundle.path),
            fit_manifest_path=fit_path,
            mapping_artifact_path=artifact,
        )


def test_mapping_identity_uses_persisted_operator_provenance_when_fit_has_wrapper_id(
    tmp_path,
):
    result, report, provenance, artifact = _mapping_bundle_inputs(tmp_path)
    result = replace(
        result,
        direct_operator_artifact_fingerprint="higher-level-mapping-wrapper",
    )
    bundle = write_gravity_viewer_bundle(
        output_directory=tmp_path / "bundle-wrapper-identity",
        fit_result=result,
        validation_result=report.adequacy,
        report=report,
        metadata={"provenance": provenance},
        linear_mapping_artifact=artifact,
    )
    assert bundle.manifest["artifact_identity_fingerprint"] == provenance["artifact_identity_fingerprint"]
    assert (
        bundle.viewer_summary["measurement_od_contributions"]["mapping_artifact_fingerprint"]
        == provenance["artifact_identity_fingerprint"]
    )


def test_mapping_identity_mismatch_and_corrupt_payload_fail_closed(tmp_path):
    result, report, provenance, artifact = _mapping_bundle_inputs(tmp_path / "mismatch")
    report_payload = json.loads(report.files["report.json"].read_text(encoding="utf-8"))
    report_payload["provenance"]["artifact_identity_fingerprint"] = "different-fit-artifact"
    report.files["report.json"].write_text(json.dumps(report_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity does not match"):
        write_gravity_viewer_bundle(
            output_directory=tmp_path / "mismatch-bundle",
            fit_result=result,
            validation_result=report.adequacy,
            report=report,
            metadata={"provenance": provenance},
            linear_mapping_artifact=artifact,
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
    assert not bundle.viewer_summary["measurement_od_contributions"]["reverse_query_available"]
    with pytest.raises(RuntimeError, match="pass rho"):
        bundle.query_measurement_od_contributions(0)
