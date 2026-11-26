from __future__ import annotations

import hashlib
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
    GravityJourneyLatentClassModel,
    GravityObservationModel,
    MeasurementODContributionQuery,
    RouteChoicePathQuery,
    GravityValidationMetadata,
    read_gravity_viewer_bundle,
    write_gravity_detailed_report,
    write_gravity_viewer_bundle,
)
from public_transportation.inference.gravity.viewer_bundle import (
    _validate_linear_mapping_for_bundle,
)
from public_transportation.inference.gravity.viewer_query_service import (
    GravityViewerQueryService,
)
from public_transportation.inference.gravity import viewer_bundle as viewer_bundle_module
from public_transportation.inference.temporal_assignment_blocks import (
    TemporalBlockAssignmentOperator,
    TemporalBlockConstructionDiagnostics,
    TemporalBlockKey,
    TemporalSparseBlock,
)
from public_transportation.inference.temporal_assignment_persistence import (
    TemporalBlockArtifactReader,
    open_temporal_block_artifact,
    open_temporal_block_artifact_for_viewer,
    save_temporal_block_operator,
)
from public_transportation.inference.hierarchical_artifacts import package_revision
from public_transportation.inference.od_parameter_layout import ODParameterLayout
from public_transportation.inference.gravity.measurement_od_contributions import (
    measurement_od_query_descriptor,
    validate_raw_od_component_consistency,
)

from tests.gravity.test_viewer_bundle import _inputs
from tests.gravity.test_route_path_query import _graph as _route_query_test_graph


def _mapping_bundle_inputs(
    tmp_path,
    *,
    changed_network: bool = False,
    save=True,
    base_result=None,
    fit_source_revision: str | None = None,
    mapping_source_revision: str | None = None,
    fit_assignment_fingerprint: str = "fit-assignment-fingerprint",
    fit_mapping_fingerprint: str = "fit-mapping-fingerprint",
    artifact_timetable_fingerprint: str = "artifact-timetable-fingerprint",
    artifact_mapping_fingerprint: str = "artifact-measurement-mapping-fingerprint",
    compact_layout_fingerprint: str = "compact-layout-fingerprint",
    result_overrides: dict[str, object] | None = None,
):
    if base_result is None:
        base_result, _, _ = _inputs(tmp_path)
    fit_source_revision = fit_source_revision or package_revision()
    mapping_source_revision = mapping_source_revision or fit_source_revision
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
        source_compact_layout_fingerprint=compact_layout_fingerprint,
    )
    identity = AssignmentArtifactIdentity(
        canonical_index_fingerprint=canonical_index.artifact_fingerprint,
        network_fingerprint="network-other" if changed_network else "network",
        timetable_fingerprint=artifact_timetable_fingerprint,
        temporal_discretization_fingerprint="temporal-discretization",
        route_choice_fingerprint="route-choice",
        departure_choice_fingerprint="departure-choice",
        feasibility_fingerprint="feasibility",
        measurement_mapping_fingerprint=artifact_mapping_fingerprint,
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
    if result_overrides:
        result = replace(result, **result_overrides)
    provenance = {
        field: f"{field}-value" for field in GRAVITY_REPORT_PROVENANCE_FIELDS
    }
    provenance.update(
        {
            "artifact_identity_fingerprint": identity.fingerprint,
            "assignment_fingerprint": fit_assignment_fingerprint,
            "binding_fingerprint": canonical_index.binding_fingerprint,
            "canonical_index_fingerprint": canonical_index.artifact_fingerprint,
            "compact_layout_fingerprint": compact_layout_fingerprint,
            "mapping_fingerprint": fit_mapping_fingerprint,
            "od_layout_fingerprint": layout.fingerprint,
            "package_revision": fit_source_revision,
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
        if mapping_source_revision != package_revision():
            _rewrite_mapping_source(artifact_path, mapping_source_revision)
    return result, report, provenance, artifact_path


def _rewrite_mapping_source(
    artifact_path: Path, source_revision: str, *, schema_version: int | None = None
) -> None:
    manifest_path = artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["package_revision"] = source_revision
    if schema_version is not None:
        manifest["schema_version"] = schema_version
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    marker_path = artifact_path / "COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    marker_path.write_text(
        json.dumps(marker, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _write_query_service_fit_manifest(
    path: Path, result: object, provenance: dict[str, object]
) -> None:
    payload = {
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

    path.write_text(json.dumps(payload, default=json_default), encoding="utf-8")


def _rewrite_viewer_summary(
    bundle_path: Path,
    updates: dict[str, object],
    *,
    contribution_updates: dict[str, object] | None = None,
) -> None:
    summary_path = bundle_path / "viewer_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(updates)
    if contribution_updates is not None:
        summary["measurement_od_contributions"].update(contribution_updates)
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    manifest_path = bundle_path / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["viewer_summary.json"].update(
        {
            "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "size_bytes": summary_path.stat().st_size,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _make_query_compatibility_bundle(
    tmp_path: Path,
    *,
    producer_revision: str,
    monkeypatch,
):
    inputs_path = tmp_path / "source-inputs"
    inputs_path.mkdir(parents=True, exist_ok=True)
    result, report, provenance, artifact = _mapping_bundle_inputs(inputs_path)
    monkeypatch.setattr(
        viewer_bundle_module, "package_revision", lambda: producer_revision
    )
    bundle = write_gravity_viewer_bundle(
        output_directory=tmp_path / "compatibility-bundle",
        fit_result=result,
        validation_result=report.adequacy,
        report=report,
        metadata={"provenance": provenance},
        linear_mapping_artifact=artifact,
    )
    fit_path = tmp_path / "fit-manifest.json"
    _write_query_service_fit_manifest(fit_path, result, provenance)
    return result, provenance, artifact, bundle, fit_path


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
    assert summary["observation_scale_source"] == "fit_result.observation_scales"
    assert summary["fixed_positive_od_attribution_available"] is False
    query = MeasurementODContributionQuery(
        artifact=open_temporal_block_artifact(artifact), fit_result=result
    )
    page = query.query(0, page=0, page_size=10)
    contributors = page["rows"]
    assert [row["od_index"] for row in contributors] == [1, 0]
    assert [row["linear_flow_contribution"] for row in contributors] == pytest.approx(
        (2.0, 1.0)
    )
    assert [row["modeled_mean_contribution"] for row in contributors] == pytest.approx(
        (4.0, 2.0)
    )
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


def test_raw_od_precision_bound_accounts_for_float32_accumulation_and_dtype(tmp_path):
    fit_result, _, _, _ = _mapping_bundle_inputs(tmp_path)
    reconstructed = 2.306265441526364
    saved = 2.3062613010406494
    common = {
        "reconstructed": reconstructed,
        "saved": saved,
        "absolute_term_sum": reconstructed,
        "term_count": 1_000,
    }

    float32_fit = replace(
        fit_result,
        objective_dtype="float32",
        gradient_dtype="float32",
    )
    validate_raw_od_component_consistency(fit_result=float32_fit, **common)

    float64_fit = replace(
        fit_result,
        objective_dtype="float64",
        gradient_dtype="float64",
    )
    with pytest.raises(ValueError, match="computation dtype=float64.*allowed tolerance="):
        validate_raw_od_component_consistency(fit_result=float64_fit, **common)

    with pytest.raises(
        ValueError,
        match="computation dtype=float32.*reconstructed=.*saved=.*absolute difference=.*relative difference=.*allowed tolerance=",
    ):
        validate_raw_od_component_consistency(
            fit_result=float32_fit,
            **{**common, "saved": 2.2},
        )


def test_float32_accumulation_difference_is_accepted_by_selected_row_query(tmp_path):
    result, _, _, artifact_path = _mapping_bundle_inputs(
        tmp_path,
        result_overrides={
            "objective_dtype": "float32",
            "gradient_dtype": "float32",
        },
    )
    reader = open_temporal_block_artifact(artifact_path)
    reconstructed = 2.306265441526364
    saved = 2.3062613010406494
    saved_od = np.asarray(result.od_measurement_contribution, dtype=np.float64).copy()
    saved_od[0] = saved
    latent = np.asarray(result.latent_measurements, dtype=np.float64).copy()
    latent[0] = 0.5 + saved + 0.5  # preserve the fit's fixed and additive components
    modeled = np.asarray(result.predicted_measurements, dtype=np.float64).copy()
    modeled[0] = 2.0 * (0.5 + 0.5 + saved)
    result = replace(
        result,
        free_od_demand=np.ones(1_000, dtype=np.float64),
        od_measurement_contribution=saved_od,
        latent_measurements=latent,
        predicted_measurements=modeled,
    )
    synthetic_index = CanonicalAssignmentIndex(
        time_intervals=(CanonicalTimeInterval("am", 0, 3_600),),
        demand_cells=tuple(
            CanonicalODTimeCell(
                index,
                f"origin-{index}",
                "destination",
                "am",
                "free",
                index,
            )
            for index in range(1_000)
        ),
        measurements=reader.canonical_index.measurements,
    )

    class RepeatedContributionReader:
        def __getattr__(self, name):
            return getattr(reader, name)

        @property
        def canonical_index(self):
            return synthetic_index

        @property
        def number_of_demand_cells(self):
            return 1_000

        def measurement_interval_nonzero_entries(self, interval_id):
            return 1_000

        def iter_blocks_for_measurement_interval(self, interval_id):
            block = next(reader.iter_blocks_for_measurement_interval(interval_id))
            count = 1_000
            yield TemporalSparseBlock(
                key=block.key,
                row_indices=np.zeros(count, dtype=np.int32),
                column_indices=np.arange(count, dtype=np.int32),
                values=np.full(count, reconstructed / count, dtype=np.float64),
                number_of_measurements=block.number_of_measurements,
                number_of_demand_cells=count,
            )

    query = MeasurementODContributionQuery(
        artifact=RepeatedContributionReader(),
        fit_result=result,
    )
    page = query.query(0, page=0, page_size=10)
    assert page["total"] == 1_000
    assert page["components"]["od_contribution"] == pytest.approx(2.0 * saved)


def test_raw_od_precision_policy_is_shared_by_query_and_bundle_validation(tmp_path):
    result, report, provenance, artifact = _mapping_bundle_inputs(
        tmp_path,
        result_overrides={
            "objective_dtype": "float32",
            "gradient_dtype": "float32",
        },
    )
    float32_saved_value = float(np.nextafter(np.float32(3.0), np.float32(np.inf)))
    json_restored_od = np.asarray(json.loads(json.dumps([float32_saved_value, 2.0, 0.0])))
    assert json_restored_od.dtype == np.float64
    result = replace(result, od_measurement_contribution=json_restored_od)

    bundle = write_gravity_viewer_bundle(
        output_directory=tmp_path / "precision-aware-bundle",
        fit_result=result,
        validation_result=report.adequacy,
        report=report,
        metadata={"provenance": provenance},
        linear_mapping_artifact=artifact,
    )
    query = MeasurementODContributionQuery(
        artifact=open_temporal_block_artifact(artifact),
        fit_result=result,
    )
    page = query.query(0, page=0, page_size=10)
    assert page["total"] == 2
    # The raw OD component (3.0) is validated before the outer observation
    # scale (2.0) is applied to this reported modeled-mean component.
    assert page["components"]["od_contribution"] == pytest.approx(6.0)
    report_payload = json.loads(report.files["report.json"].read_text(encoding="utf-8"))
    _validate_linear_mapping_for_bundle(
        artifact=open_temporal_block_artifact(artifact),
        fit_result=result,
        report_payload=report_payload,
        metadata={"provenance": provenance},
        full_od_path=bundle.path / "full_od.csv",
        validate_block_payloads=True,
    )

    float64_fit = replace(
        result,
        objective_dtype="float64",
        gradient_dtype="float64",
    )
    query64 = MeasurementODContributionQuery(
        artifact=open_temporal_block_artifact(artifact),
        fit_result=float64_fit,
    )
    with pytest.raises(ValueError, match="computation dtype=float64.*reconstructed=.*saved=.*allowed tolerance="):
        query64.query(0, page=0, page_size=10)
    with pytest.raises(ValueError, match="computation dtype=float64.*reconstructed=.*saved=.*allowed tolerance="):
        _validate_linear_mapping_for_bundle(
            artifact=open_temporal_block_artifact(artifact),
            fit_result=float64_fit,
            report_payload=report_payload,
            metadata={"provenance": provenance},
            full_od_path=bundle.path / "full_od.csv",
            validate_block_payloads=True,
        )

    inconsistent_fit = replace(
        result,
        od_measurement_contribution=np.asarray((3.01, 2.0, 0.0), dtype=np.float64),
        # Keep the independently validated fixed-offset decomposition intact
        # so the test reaches the raw mapping-versus-fit consistency check.
        latent_measurements=np.asarray((4.01, 5.25, 0.0), dtype=np.float64),
        predicted_measurements=np.asarray((8.02, 2.625, 1e-8), dtype=np.float64),
    )
    query_bad = MeasurementODContributionQuery(
        artifact=open_temporal_block_artifact(artifact),
        fit_result=inconsistent_fit,
    )
    with pytest.raises(ValueError, match="computation dtype=float32.*absolute difference=.*allowed tolerance="):
        query_bad.query(0, page=0, page_size=10)
    with pytest.raises(ValueError, match="computation dtype=float32.*absolute difference=.*allowed tolerance="):
        _validate_linear_mapping_for_bundle(
            artifact=open_temporal_block_artifact(artifact),
            fit_result=inconsistent_fit,
            report_payload=report_payload,
            metadata={"provenance": provenance},
            full_od_path=bundle.path / "full_od.csv",
            validate_block_payloads=True,
        )


@pytest.mark.parametrize(
    ("scale_contract", "result_overrides", "expected_scale", "expected_means", "expected_modeled"),
    [
        (
            "journey_latent_class",
            {
                "measurement_model": GravityJourneyLatentClassModel(
                    class_probabilities=(0.5, 0.5),
                    class_effects=(1.0, 1.4),
                ).to_dict(),
                "latent_measurements": np.asarray((4.7, 5.7, 0.0)),
                "predicted_measurements": np.asarray((9.4, 2.85, 1e-8)),
            },
            1.2,
            (4.8, 2.4),
            9.4,
        ),
        (
            "rowwise_observation_scale",
            {
                "measurement_model": GravityObservationModel(
                    measurement_types=("boarding", "boarding", "alighting"),
                    boarding_scale=2.0,
                    alighting_scale=0.5,
                ).to_dict(),
                "observation_scales": np.asarray((2.0, 2.0, 0.5)),
                "predicted_measurements": np.asarray((8.0, 10.5, 1e-8)),
            },
            1.0,
            (4.0, 2.0),
            8.0,
        ),
    ],
)
def test_query_scale_contracts_create_descriptors_and_close_modeled_rows(
    tmp_path,
    scale_contract,
    result_overrides,
    expected_scale,
    expected_means,
    expected_modeled,
):
    result, report, provenance, artifact = _mapping_bundle_inputs(
        tmp_path,
        result_overrides=result_overrides,
    )
    bundle = write_gravity_viewer_bundle(
        output_directory=tmp_path / f"{scale_contract}-bundle",
        fit_result=result,
        validation_result=report.adequacy,
        report=report,
        metadata={"provenance": provenance},
        linear_mapping_artifact=artifact,
    )
    descriptor = bundle.viewer_summary["measurement_od_contributions"]
    assert descriptor["available"] is True
    assert descriptor["measurement_scale_contract"] == scale_contract
    assert descriptor["aggregate_flow_scale"] == pytest.approx(expected_scale)
    if scale_contract == "rowwise_observation_scale":
        assert "journey_aggregate_scale" not in descriptor
    else:
        assert descriptor["journey_aggregate_scale"] == pytest.approx(expected_scale)

    reader = open_temporal_block_artifact_for_viewer(
        artifact,
        fit_source_package_revision=provenance["package_revision"],
    )
    query = MeasurementODContributionQuery(artifact=reader, fit_result=result)
    page = query.query(0, page=0, page_size=10)
    components = page["components"]
    assert [row["modeled_mean_contribution"] for row in page["rows"]] == pytest.approx(
        expected_means
    )
    assert components["fixed_offset_contribution"] == pytest.approx(1.0)
    assert components["additive_contribution"] == pytest.approx(
        1.2 if scale_contract == "journey_latent_class" else 1.0
    )
    assert components["mean_floor_adjustment"] == pytest.approx(0.0)
    assert (
        sum(row["modeled_mean_contribution"] for row in page["rows"])
        + components["fixed_offset_contribution"]
        + components["additive_contribution"]
        + components["mean_floor_adjustment"]
    ) == pytest.approx(expected_modeled)
    assert components["modeled_mean"] == pytest.approx(expected_modeled)


def _routed_od_only_payload(**overrides):
    return {
        "type": "opaque_external_scale",
        "scale_scope": "routed_od_only",
        "scales_routed_od_contribution": True,
        "scales_assignment_and_boundary_flow_contributions": False,
        "scales_additive_boundary_flow_contributions": False,
        "scales_fixed_operator_offsets": False,
        **overrides,
    }


def _routed_od_only_result(tmp_path, *, overrides=None):
    result, report, provenance, artifact = _mapping_bundle_inputs(
        tmp_path,
        result_overrides={
            "measurement_model": _routed_od_only_payload(),
            "latent_measurements": np.asarray((7.0, 7.25, 0.0)),
            "predicted_measurements": np.asarray((15.0, 3.625, 1e-8)),
        },
    )
    if overrides:
        result = replace(result, **overrides)
    return result, report, provenance, artifact


def test_routed_od_only_contract_decomposes_forward_and_reverse_queries(tmp_path, monkeypatch):
    result, _, provenance, artifact = _routed_od_only_result(tmp_path)

    def fail_deserialize(*_args, **_kwargs):
        pytest.fail("opaque scale payload was passed to a built-in model deserializer")

    monkeypatch.setattr(GravityJourneyLatentClassModel, "from_dict", fail_deserialize)
    monkeypatch.setattr(GravityObservationModel, "from_dict", fail_deserialize)
    descriptor = measurement_od_query_descriptor(
        artifact=open_temporal_block_artifact(artifact),
        fit_result=result,
        fit_provenance=provenance,
        runtime_package_revision=package_revision(),
    )
    assert descriptor["measurement_scale_contract"] == "routed_od_only"
    assert descriptor["scale_component_scope"] == "routed_od_only"
    assert descriptor["aggregate_flow_scale"] is None
    assert descriptor["reverse_query_available"] is True

    query = MeasurementODContributionQuery(
        artifact=open_temporal_block_artifact(artifact),
        fit_result=result,
    )
    page = query.query(0, page=0, page_size=10)
    assert [row["linear_flow_contribution"] for row in page["rows"]] == pytest.approx((2.0, 1.0))
    assert [row["modeled_mean_contribution"] for row in page["rows"]] == pytest.approx((8.0, 4.0))
    assert page["components"] == {
        "fixed_offset_contribution": pytest.approx(1.0),
        "additive_contribution": pytest.approx(1.0),
        "od_contribution": pytest.approx(12.0),
        "mean_floor_adjustment": pytest.approx(1.0),
        "modeled_mean": pytest.approx(15.0),
    }
    assert page["modeled_mean"] == pytest.approx(15.0)
    assert sum(row["modeled_mean_contribution"] for row in page["rows"]) + sum(
        page["components"][name]
        for name in (
            "fixed_offset_contribution",
            "additive_contribution",
            "mean_floor_adjustment",
        )
    ) == pytest.approx(15.0)

    mismatched_fit_component = replace(
        result,
        od_measurement_contribution=np.asarray((3.01, 2.0, 0.0)),
    )
    mismatched_query = MeasurementODContributionQuery(
        artifact=open_temporal_block_artifact(artifact),
        fit_result=mismatched_fit_component,
    )
    with pytest.raises(ValueError, match="mapping OD contribution differs"):
        mismatched_query.query(0, page=0, page_size=10)

    affected = query.query_od(0, page=0, page_size=10)
    assert [row["measurement_index"] for row in affected["rows"]] == [0, 1]
    assert [row["modeled_mean_contribution"] for row in affected["rows"]] == pytest.approx((4.0, 2.0))
    zero_row = query.query(2, page=0, page_size=10)
    assert zero_row["rows"] == []
    assert zero_row["components"]["mean_floor_adjustment"] == pytest.approx(1e-8)


def test_routed_od_only_serialization_sized_negative_floor_is_clamped_and_reconciled(tmp_path):
    result, _, provenance, artifact = _routed_od_only_result(
        tmp_path,
        overrides={"predicted_measurements": np.asarray((14.0 - 2e-5, 3.625, 1e-8))},
    )
    descriptor = measurement_od_query_descriptor(
        artifact=open_temporal_block_artifact(artifact),
        fit_result=result,
        fit_provenance=provenance,
        runtime_package_revision=package_revision(),
    )
    assert descriptor["available"] is True

    query = MeasurementODContributionQuery(
        artifact=open_temporal_block_artifact(artifact),
        fit_result=result,
    )
    page = query.query(0, page=0, page_size=10)
    assert page["components"]["mean_floor_adjustment"] == 0.0
    component_sum = (
        sum(row["modeled_mean_contribution"] for row in page["rows"])
        + page["components"]["fixed_offset_contribution"]
        + page["components"]["additive_contribution"]
        + page["components"]["mean_floor_adjustment"]
    )
    assert component_sum == pytest.approx(14.0)
    assert page["modeled_mean"] == pytest.approx(14.0 - 2e-5)


def test_routed_od_only_rejects_negative_floor_discrepancy_beyond_shared_tolerance(tmp_path):
    result, _, provenance, artifact = _routed_od_only_result(
        tmp_path,
        overrides={"predicted_measurements": np.asarray((14.0 - 2e-4, 3.625, 1e-8))},
    )
    with pytest.raises(ValueError, match="components exceed the saved modeled mean"):
        measurement_od_query_descriptor(
            artifact=open_temporal_block_artifact(artifact),
            fit_result=result,
            fit_provenance=provenance,
            runtime_package_revision=package_revision(),
        )

    valid_result, _, _, valid_artifact = _routed_od_only_result(tmp_path / "row-query")
    query = MeasurementODContributionQuery(
        artifact=open_temporal_block_artifact(valid_artifact),
        fit_result=valid_result,
    )
    changed_predictions = np.array(query._scales["modeled"], copy=True)
    changed_predictions[0] = 14.0 - 2e-4
    query._scales["modeled"] = changed_predictions
    with pytest.raises(ValueError, match="components exceed the saved modeled mean"):
        query.query(0, page=0, page_size=10)


def test_routed_od_only_scope_is_fail_closed_for_invalid_metadata_and_zero_flow(tmp_path):
    result, _, provenance, artifact = _routed_od_only_result(tmp_path)
    for payload in (
        _routed_od_only_payload(scales_fixed_operator_offsets=True),
        _routed_od_only_payload(scale_scope="unsupported_scope"),
        {
            "type": "opaque_external_scale",
            "scale_scope": "routed_od_only",
            "scales_routed_od_contribution": True,
        },
        {
            "type": "opaque_external_scale",
            "scales_routed_od_contribution": True,
            "scales_assignment_and_boundary_flow_contributions": False,
            "scales_additive_boundary_flow_contributions": False,
            "scales_fixed_operator_offsets": False,
        },
    ):
        with pytest.raises(ValueError, match="scale|scope|component flags"):
            measurement_od_query_descriptor(
                artifact=open_temporal_block_artifact(artifact),
                fit_result=replace(result, measurement_model=payload),
                fit_provenance=provenance,
                runtime_package_revision=package_revision(),
            )

    inconsistent = replace(
        result,
        latent_measurements=np.asarray((7.0, 7.25, 0.1)),
    )
    with pytest.raises(ValueError, match="zero raw OD contribution"):
        measurement_od_query_descriptor(
            artifact=open_temporal_block_artifact(artifact),
            fit_result=inconsistent,
            fit_provenance=provenance,
            runtime_package_revision=package_revision(),
        )


def test_routed_od_only_reverse_query_is_unavailable_without_saved_raw_od_component(tmp_path):
    result, _, provenance, artifact = _routed_od_only_result(
        tmp_path,
        overrides={"od_measurement_contribution": None},
    )
    descriptor = measurement_od_query_descriptor(
        artifact=open_temporal_block_artifact(artifact),
        fit_result=result,
        fit_provenance=provenance,
        runtime_package_revision=package_revision(),
    )
    assert descriptor["reverse_query_available"] is False
    assert "persisted raw OD measurement component" in descriptor["reverse_query_message"]
    query = MeasurementODContributionQuery(
        artifact=open_temporal_block_artifact(artifact),
        fit_result=result,
    )
    assert query.query(0)["components"]["od_contribution"] == pytest.approx(12.0)
    with pytest.raises(ValueError, match="reverse OD-to-measurement contributions are unavailable"):
        query.query_od(0)


def test_unknown_and_incomplete_measurement_scale_contracts_fail_clearly(tmp_path):
    result, _, provenance, artifact = _mapping_bundle_inputs(tmp_path)
    with pytest.raises(ValueError, match="unsupported fitted measurement-scale contract"):
        measurement_od_query_descriptor(
            artifact=open_temporal_block_artifact(artifact),
            fit_result=replace(result, measurement_model={"type": "unsupported_scale"}),
            fit_provenance=provenance,
            runtime_package_revision=package_revision(),
        )

    with pytest.raises(ValueError, match="missing its scale specification"):
        measurement_od_query_descriptor(
            artifact=open_temporal_block_artifact(artifact),
            fit_result=replace(
                result,
                measurement_model={"type": "rowwise_measurement_scale"},
            ),
            fit_provenance=provenance,
            runtime_package_revision=package_revision(),
        )

    rowwise = GravityObservationModel(
        measurement_types=("boarding", "boarding", "alighting"),
        boarding_scale=2.0,
        alighting_scale=0.5,
    ).to_dict()
    with pytest.raises(ValueError, match="require persisted observation_scales"):
        measurement_od_query_descriptor(
            artifact=open_temporal_block_artifact(artifact),
            fit_result=replace(
                result,
                measurement_model=rowwise,
                observation_scales=None,
            ),
            fit_provenance=provenance,
            runtime_package_revision=package_revision(),
            rho=1.0,
        )

    with pytest.raises(ValueError, match="components exceed the saved modeled mean"):
        measurement_od_query_descriptor(
            artifact=open_temporal_block_artifact(artifact),
            fit_result=replace(
                result,
                predicted_measurements=np.asarray((7.9, 2.625, 1e-8)),
            ),
            fit_provenance=provenance,
            runtime_package_revision=package_revision(),
        )


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
    assert requested_intervals == ["am"]
    assert query.measurement_cache_misses == 1
    assert query.measurement_cache_hits == 2
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

    resized_page = query.query(0, page=0, page_size=2)
    assert [row["od_index"] for row in resized_page["rows"]] == [1, 0]
    assert requested_intervals == ["am"]
    assert query.measurement_cache_hits == 3

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


def test_on_demand_query_evicts_old_completed_measurement_cache_rows(tmp_path):
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
        max_cached_measurements=1,
    )
    query.query(0, page=0, page_size=1)
    query.query(1, page=0, page_size=1)
    assert len(query._measurement_cache) == 1
    query.query(0, page=1, page_size=1)
    assert requested_intervals == ["am", "pm", "am"]
    assert query.measurement_cache_misses == 3


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
    result, report, provenance, artifact = _routed_od_only_result(tmp_path)

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
    _validate_linear_mapping_for_bundle(
        artifact=open_temporal_block_artifact(artifact),
        fit_result=result,
        report_payload=json.loads(report.files["report.json"].read_text(encoding="utf-8")),
        metadata={"provenance": provenance},
        full_od_path=bundle.path / "full_od.csv",
        validate_block_payloads=True,
    )
    summary = bundle.viewer_summary["measurement_od_contributions"]
    assert summary["query_mode"] == "local_service"
    assert summary["query_path"] == "/api/measurement-od-contributions"
    assert summary["mapping_artifact_fingerprint"] == provenance["artifact_identity_fingerprint"]
    assert summary["measurement_scale_contract"] == "routed_od_only"
    assert summary["scale_component_scope"] == "routed_od_only"
    assert summary["reverse_query_available"] is True
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
    assert response["observed"] == pytest.approx(3.0)
    assert response["modeled_mean"] == pytest.approx(8.0)

    fit_payload["artifact_identity_fingerprint"] = "stale-fit-artifact"
    fit_path.write_text(json.dumps(fit_payload, default=json_default), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_identity_fingerprint differs"):
        GravityViewerQueryService(
            bundle=read_gravity_viewer_bundle(bundle.path),
            fit_manifest_path=fit_path,
            mapping_artifact_path=artifact,
        )


def test_routed_od_only_bundle_query_service_returns_observed_and_saved_modeled_mean(tmp_path):
    result, report, provenance, artifact = _routed_od_only_result(tmp_path)
    bundle = write_gravity_viewer_bundle(
        output_directory=tmp_path / "routed-service-bundle",
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

    fit_path = tmp_path / "routed-fit-manifest.json"
    fit_path.write_text(json.dumps(fit_payload, default=json_default), encoding="utf-8")
    service = GravityViewerQueryService(
        bundle=read_gravity_viewer_bundle(bundle.path),
        fit_manifest_path=fit_path,
        mapping_artifact_path=artifact,
    )
    response = service.query_measurement(0, page=0, page_size=1)
    assert response["observed"] == pytest.approx(3.0)
    assert response["modeled_mean"] == pytest.approx(15.0)
    assert response["components"]["modeled_mean"] == pytest.approx(15.0)
    assert response["rows"][0]["modeled_mean_contribution"] == pytest.approx(8.0)


def test_viewer_supports_matching_legacy_schema_revision_with_distinct_provenance(tmp_path, monkeypatch):
    source_revision = "579a70af1e15a88d34ad8907b19740db11caff01"
    supplied = {
        "artifact_identity": "8b6a5675a63d7797278a74fde03f78940829ab55d61e36185f4c6d590eccfed5",
        "fit_assignment": "44a6cf1ba0fe177cb4d218146056aa1bae33c2ce961a3e6da9a2a762d71e25bf",
        "fit_mapping": "44a6cf1ba0fe177cb4d218146056aa1bae33c2ce961a3e6da9a2a762d71e25bf",
        "artifact_timetable": "7a711efcbbf896254840ff2a1b7ebaeb51c985137bb619d2f8a1919dcf4aa9d0",
        "artifact_mapping": "c26fc89b28cb34d605c429ea4f8693e6d2d2edfdedf4ab5874bc245d572d1db1",
        "canonical_index": "70d41ba56745a20442c666f02fc4bbb493e03f9d191dc2dc73d5c04e70fbd693",
        "binding": "e5941fc0f404c6fbbe79f7d5a7563b97e1df724becdc7cd1264fd1b0e6b39124",
        "od_layout": "544a8d00b05806828a16a192b36e85d7b8e62d9d8bdd84e5d57488813d228945",
        "compact_layout": "d494daf2ff89865bcf48366e00c53f88ea802ebf797ac08aa5d5fb115069b63a",
    }
    base_result, _, _ = _inputs(tmp_path)
    with monkeypatch.context() as patcher:
        patcher.setattr(
            AssignmentArtifactIdentity,
            "fingerprint",
            property(lambda _self: supplied["artifact_identity"]),
        )
        patcher.setattr(
            CanonicalAssignmentIndex,
            "artifact_fingerprint",
            property(lambda _self: supplied["canonical_index"]),
        )
        patcher.setattr(
            CanonicalAssignmentIndex,
            "binding_fingerprint",
            property(lambda _self: supplied["binding"]),
        )
        patcher.setattr(
            ODParameterLayout,
            "fingerprint",
            property(lambda _self: supplied["od_layout"]),
        )
        result, report, provenance, artifact = _mapping_bundle_inputs(
            tmp_path,
            base_result=base_result,
            fit_source_revision=source_revision,
            fit_assignment_fingerprint=supplied["fit_assignment"],
            fit_mapping_fingerprint=supplied["fit_mapping"],
            artifact_timetable_fingerprint=supplied["artifact_timetable"],
            artifact_mapping_fingerprint=supplied["artifact_mapping"],
            compact_layout_fingerprint=supplied["compact_layout"],
        )
        compatible_reader = open_temporal_block_artifact_for_viewer(
            artifact,
            fit_source_package_revision=source_revision,
        )
        assert compatible_reader.source_package_revision == source_revision
        assert compatible_reader.artifact_schema_version == 2
        bundle = write_gravity_viewer_bundle(
            output_directory=tmp_path / "legacy-source-bundle",
            fit_result=result,
            validation_result=report.adequacy,
            report=report,
            metadata={"provenance": provenance},
            linear_mapping_artifact=artifact,
        )
        summary = bundle.viewer_summary["measurement_od_contributions"]
        assert summary["assignment_fingerprint"] == supplied["fit_assignment"]
        assert summary["mapping_fingerprint"] == supplied["fit_mapping"]
        assert summary["artifact_timetable_fingerprint"] == supplied["artifact_timetable"]
        assert summary["artifact_measurement_mapping_fingerprint"] == supplied["artifact_mapping"]
        assert summary["mapping_artifact_fingerprint"] == supplied["artifact_identity"]
        assert summary["canonical_index_fingerprint"] == supplied["canonical_index"]
        assert summary["binding_fingerprint"] == supplied["binding"]
        assert summary["od_layout_fingerprint"] == supplied["od_layout"]
        assert summary["mapping_artifact_schema_version"] == 2
        assert summary["runtime_package_revision"] == package_revision()
        assert summary["fit_source_package_revision"] == source_revision
        assert summary["mapping_source_package_revision"] == source_revision
        assert bundle.manifest["runtime_package_revision"] == package_revision()
        assert bundle.manifest["package_revision"] == source_revision
        assert not (bundle.path / "partitions" / "measurement_od_contributions").exists()

        with pytest.raises(ValueError, match="package revision is incompatible"):
            open_temporal_block_artifact(artifact)

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

        fit_path = tmp_path / "legacy-source-fit.json"
        fit_path.write_text(json.dumps(fit_payload, default=json_default), encoding="utf-8")
        service = GravityViewerQueryService(
            bundle=read_gravity_viewer_bundle(bundle.path),
            fit_manifest_path=fit_path,
            mapping_artifact_path=artifact,
        )
        page = service.query_measurement(0, page=0, page_size=1)
        assert page["total"] == 2
        assert len(page["rows"]) == 1
        assert page["rows"][0]["measurement_index"] == 0


def test_viewer_query_service_accepts_current_bundle_producer(tmp_path, monkeypatch):
    current_revision = "1b626de8152e44aed7eca84ddd06d41fa45132ad"
    result, _provenance, artifact, bundle, fit_path = _make_query_compatibility_bundle(
        tmp_path,
        producer_revision=current_revision,
        monkeypatch=monkeypatch,
    )
    monkeypatch.setattr(
        "public_transportation.inference.gravity.viewer_query_service.package_revision",
        lambda: current_revision,
    )
    service = GravityViewerQueryService(
        bundle=read_gravity_viewer_bundle(bundle.path),
        fit_manifest_path=fit_path,
        mapping_artifact_path=artifact,
    )
    assert service.query_measurement(0, page=0, page_size=1)["total"] == 2
    assert service.route_path_query is None


def test_verified_older_bundle_producer_is_read_only_and_keeps_queries_available(
    tmp_path, monkeypatch
):
    current_revision = "1b626de8152e44aed7eca84ddd06d41fa45132ad"
    older_revision = "387758616ee8684592501de53fcef07a62f39ae6"
    result, _provenance, artifact, bundle, fit_path = _make_query_compatibility_bundle(
        tmp_path,
        producer_revision=older_revision,
        monkeypatch=monkeypatch,
    )
    original_manifest = json.loads(
        (bundle.path / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert original_manifest["runtime_package_revision"] == older_revision
    original_hashes = {
        path.relative_to(bundle.path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in bundle.path.rglob("*")
        if path.is_file()
    }

    monkeypatch.setattr(
        "public_transportation.inference.gravity.viewer_query_service.package_revision",
        lambda: current_revision,
    )
    service = GravityViewerQueryService(
        bundle=read_gravity_viewer_bundle(bundle.path),
        fit_manifest_path=fit_path,
        mapping_artifact_path=artifact,
    )
    measurement_page = service.query_measurement(0, page=0, page_size=1)
    od_page = service.query_od(0, page=0, page_size=1)
    assert measurement_page["total"] == 2
    assert len(measurement_page["rows"]) == 1
    assert od_page["total"] >= 1
    route_probabilities = np.asarray((0.6, 0.4, 1.0, 1.0, 1.0, 1.0))
    route_source = RouteChoicePathQuery(
        graph=_route_query_test_graph(),
        group_dest_node=np.asarray((5,), dtype=np.int64),
        time_intervals=service.artifact.canonical_index.time_intervals,
        mapping_artifact_fingerprint=service.artifact.identity.fingerprint,
        fit_fingerprint=result.model_fingerprint,
        route_choice_fingerprint=service.artifact.identity.route_choice_fingerprint,
        probability_for_group=lambda _group: (
            route_probabilities,
            np.ones(route_probabilities.shape, dtype=bool),
        ),
        number_of_groups=1,
    )
    service.set_route_path_query(route_source)
    paths = service.query_route_paths(
        origin_stop_id="A",
        destination_stop_id="B",
        departure_interval_id="am",
        minimum_probability=0.3,
    )
    assert len(paths["paths"]) == 2

    final_hashes = {
        path.relative_to(bundle.path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in bundle.path.rglob("*")
        if path.is_file()
    }
    assert final_hashes == original_hashes
    assert result.model_fingerprint == measurement_page["fit_fingerprint"]


def test_viewer_query_service_rejects_unknown_bundle_producer_actionably(
    tmp_path, monkeypatch
):
    current_revision = "1b626de8152e44aed7eca84ddd06d41fa45132ad"
    unknown_revision = "unverified-producer-revision"
    _result, _provenance, artifact, bundle, fit_path = _make_query_compatibility_bundle(
        tmp_path,
        producer_revision=unknown_revision,
        monkeypatch=monkeypatch,
    )
    monkeypatch.setattr(
        "public_transportation.inference.gravity.viewer_query_service.package_revision",
        lambda: current_revision,
    )
    with pytest.raises(
        ValueError,
        match=(
            "producer revision is not verified.*"
            "producer revision='unverified-producer-revision'.*"
            "schema version=1.*query descriptor version=2.*Supported options"
        ),
    ):
        GravityViewerQueryService(
            bundle=read_gravity_viewer_bundle(bundle.path),
            fit_manifest_path=fit_path,
            mapping_artifact_path=artifact,
        )


def test_unsupported_bundle_schema_and_query_descriptor_errors_are_actionable(
    tmp_path, monkeypatch
):
    producer_revision = "387758616ee8684592501de53fcef07a62f39ae6"
    _result, _provenance, _artifact, bundle, _fit_path = _make_query_compatibility_bundle(
        tmp_path / "schema",
        producer_revision=producer_revision,
        monkeypatch=monkeypatch,
    )
    manifest_path = bundle.path / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="producer revision='387758.*schema version=99.*supported bundle type",
    ):
        read_gravity_viewer_bundle(bundle.path)

    _result, _provenance, _artifact, bundle, _fit_path = _make_query_compatibility_bundle(
        tmp_path / "descriptor",
        producer_revision=producer_revision,
        monkeypatch=monkeypatch,
    )
    _rewrite_viewer_summary(
        bundle.path,
        {},
        contribution_updates={"query_descriptor_version": 99},
    )
    with pytest.raises(
        ValueError,
        match=(
            "producer revision='387758.*schema version=1, descriptor version=99.*"
            "supported descriptor version=2"
        ),
    ):
        read_gravity_viewer_bundle(bundle.path)


def test_verified_producer_compatibility_does_not_bypass_fit_or_mapping_checks(
    tmp_path, monkeypatch
):
    current_revision = "1b626de8152e44aed7eca84ddd06d41fa45132ad"
    older_revision = "387758616ee8684592501de53fcef07a62f39ae6"
    _result, _provenance, artifact, bundle, fit_path = _make_query_compatibility_bundle(
        tmp_path,
        producer_revision=older_revision,
        monkeypatch=monkeypatch,
    )
    monkeypatch.setattr(
        "public_transportation.inference.gravity.viewer_query_service.package_revision",
        lambda: current_revision,
    )
    fit_payload = json.loads(fit_path.read_text(encoding="utf-8"))
    fit_payload["assignment_fingerprint"] = "mismatched-assignment"
    fit_path.write_text(json.dumps(fit_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="assignment_fingerprint differs"):
        GravityViewerQueryService(
            bundle=read_gravity_viewer_bundle(bundle.path),
            fit_manifest_path=fit_path,
            mapping_artifact_path=artifact,
        )

    _result, _provenance, artifact, bundle, fit_path = _make_query_compatibility_bundle(
        tmp_path / "mapping-checksum",
        producer_revision=older_revision,
        monkeypatch=monkeypatch,
    )
    mapping_manifest_path = artifact / "manifest.json"
    mapping_manifest = json.loads(mapping_manifest_path.read_text(encoding="utf-8"))
    mapping_manifest["compatibility_test_annotation"] = "changed manifest"
    mapping_manifest_path.write_text(
        json.dumps(mapping_manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    marker_path = artifact / "COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["manifest_sha256"] = hashlib.sha256(
        mapping_manifest_path.read_bytes()
    ).hexdigest()
    marker_path.write_text(
        json.dumps(marker, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="mapping manifest checksum differs"):
        GravityViewerQueryService(
            bundle=read_gravity_viewer_bundle(bundle.path),
            fit_manifest_path=fit_path,
            mapping_artifact_path=artifact,
        )


def test_viewer_rejects_fit_mapping_source_revision_mismatch_and_unknown_schema(tmp_path):
    result, report, provenance, artifact = _mapping_bundle_inputs(
        tmp_path / "revision-mismatch",
        fit_source_revision=package_revision(),
        mapping_source_revision="579a70af1e15a88d34ad8907b19740db11caff01",
    )
    with pytest.raises(ValueError, match="fit and mapping source package revisions differ"):
        write_gravity_viewer_bundle(
            output_directory=tmp_path / "revision-mismatch-bundle",
            fit_result=result,
            validation_result=report.adequacy,
            report=report,
            metadata={"provenance": provenance},
            linear_mapping_artifact=artifact,
        )

    result, report, provenance, artifact = _mapping_bundle_inputs(
        tmp_path / "unsupported-schema",
        fit_source_revision="579a70af1e15a88d34ad8907b19740db11caff01",
    )
    _rewrite_mapping_source(
        artifact,
        "579a70af1e15a88d34ad8907b19740db11caff01",
        schema_version=99,
    )
    with pytest.raises(ValueError, match="schema is incompatible"):
        write_gravity_viewer_bundle(
            output_directory=tmp_path / "unsupported-schema-bundle",
            fit_result=result,
            validation_result=report.adequacy,
            report=report,
            metadata={"provenance": provenance},
            linear_mapping_artifact=artifact,
        )


def test_reader_reuses_validated_dimensions_for_each_block(tmp_path, monkeypatch):
    _, _, provenance, artifact = _mapping_bundle_inputs(tmp_path)
    original = CanonicalAssignmentIndex.number_of_demand_cells.fget
    calls = 0

    def counted(index):
        nonlocal calls
        calls += 1
        return original(index)

    monkeypatch.setattr(
        CanonicalAssignmentIndex,
        "number_of_demand_cells",
        property(counted),
    )
    reader = open_temporal_block_artifact_for_viewer(
        artifact,
        fit_source_package_revision=provenance["package_revision"],
    )
    assert calls == 1

    list(reader.iter_blocks())
    assert calls == 1


def test_bundle_creation_skips_sparse_payload_scan_and_queries_verify_selected_blocks(
    tmp_path, monkeypatch
):
    result, report, provenance, artifact = _mapping_bundle_inputs(tmp_path)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["blocks"][0]["content_hash"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    marker_path = artifact / "COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    marker_path.write_text(
        json.dumps(marker, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    def fail_if_scanned(_reader):
        pytest.fail("on-demand bundle creation scanned every sparse mapping block")

    with monkeypatch.context() as patcher:
        patcher.setattr(TemporalBlockArtifactReader, "iter_blocks", fail_if_scanned)
        bundle = write_gravity_viewer_bundle(
            output_directory=tmp_path / "corrupt-block-bundle",
            fit_result=result,
            validation_result=report.adequacy,
            report=report,
            metadata={"provenance": provenance},
            linear_mapping_artifact=artifact,
        )
    assert bundle.viewer_summary["measurement_od_contributions"]["available"] is True

    reader = open_temporal_block_artifact_for_viewer(
        artifact,
        fit_source_package_revision=provenance["package_revision"],
    )
    interval_id = manifest["blocks"][0]["key"]["measurement_interval_id"]
    with pytest.raises(ValueError, match="payload is missing or corrupt"):
        list(reader.iter_blocks_for_measurement_interval(interval_id))


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
