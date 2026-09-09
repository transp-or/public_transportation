from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from public_transportation.inference.gravity import (
    GRAVITY_REPORT_PROVENANCE_FIELDS,
    GRAVITY_RESULT_SCHEMA_VERSION,
    GravityEstimationResult,
    GravityIdentifiabilityConfig,
    GravityModelSpecification,
    GravityODIdentifiability,
    GravityStrategySelection,
    GravityTimeSpecification,
    GravityValidationMetadata,
    read_gravity_viewer_bundle,
    write_gravity_detailed_report,
    write_gravity_viewer_bundle,
    write_persisted_gravity_viewer_bundle,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout


def _inputs(tmp_path, specification=None):
    layout = ODParameterLayout(
        num_od_total=2,
        od_keys=(("o1", "d1", "am"), ("o1", "d2", "am")),
        free_od_indices=(0, 1),
        fixed_od_indices=(),
        fixed_od_values=(),
        free_baseline_values=(1.0, 2.0),
        fixed_zero_indices=(),
        fixed_positive_indices=(),
    )
    specification = specification or GravityModelSpecification()
    result = GravityEstimationResult(
        schema_version=GRAVITY_RESULT_SCHEMA_VERSION,
        status="converged",
        success=True,
        message="converged",
        raw_parameters=np.asarray((0.25,)),
        physical_parameters=np.asarray((1.25,)),
        free_od_demand=np.asarray((1.0, 2.0)),
        active_od_demand=np.asarray((1.0, 2.0)),
        full_od_demand=np.asarray((1.0, 2.0)),
        predicted_measurements=np.asarray((4.0, 5.0, 6.0)),
        objective=10.0,
        data_log_likelihood=-10.0,
        gradient=np.asarray((0.0,)),
        iterations=3,
        elapsed_seconds=0.5,
        model_fingerprint="model-fingerprint",
        strategy_selection=GravityStrategySelection(
            requested="adjoint",
            selected="adjoint",
            reason="test",
            candidates=(),
            persistent_compilation_cache_enabled=False,
            persistent_compilation_cache_directory=None,
        ),
        resumed=False,
        checkpoint_path=None,
        specification_fingerprint=specification.fingerprint,
        model_specification=specification.to_dict(),
        parameter_names=("beta_time",),
        direct_operator_artifact_fingerprint="artifact-value",
    )
    provenance = {field: f"{field}-value" for field in GRAVITY_REPORT_PROVENANCE_FIELDS}
    provenance["artifact_identity_fingerprint"] = "artifact-value"
    provenance["od_layout_fingerprint"] = layout.fingerprint
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
        output_directory=tmp_path / "report",
        provenance=provenance,
    )
    return result, report, provenance


def _network_files(tmp_path):
    stops = tmp_path / "stops.csv"
    with stops.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("stop_id", "name", "lat", "lon"))
        writer.writeheader()
        writer.writerows(
            (
                {"stop_id": "o1", "name": "Origin", "lat": 46.2, "lon": 6.1},
                {"stop_id": "d1", "name": "Destination 1", "lat": 46.21, "lon": 6.11},
                {"stop_id": "d2", "name": "Destination 2", "lat": 46.22, "lon": 6.12},
            )
        )
    lines = tmp_path / "lines.csv"
    lines.write_text("line_id,name\nL1,Line 1\n", encoding="utf-8")
    return {"stops.csv": stops, "lines.csv": lines}


def _identifiability(result):
    return GravityODIdentifiability(
        count_information_share=np.asarray((0.9, 0.7)),
        regularization_information_share=np.asarray((0.1, 0.3)),
        auxiliary_information_share=None,
        local_information_variance=np.asarray((1.0, 2.0)),
        classification=np.asarray(("count_dominated", "mixed_information")),
        diagnostic_reason=np.asarray(
            ("count_information_dominates_local_curvature", "counts_and_assumptions_both_contribute")
        ),
        effective_hessian_rank=1,
        total_hessian_dimension=result.raw_parameters.size,
        config=GravityIdentifiabilityConfig(),
        provenance={"model_fingerprint": result.model_fingerprint},
        free_cells=2,
    )


def test_viewer_bundle_contains_exact_model_specification_and_round_trips(tmp_path):
    result, report, provenance = _inputs(tmp_path)
    bundle = write_gravity_viewer_bundle(
        output_directory=tmp_path / "bundle",
        fit_result=result,
        validation_result=report.adequacy,
        report=report,
        network_files=_network_files(tmp_path),
        metadata={"time_zone": "Europe/Zurich"},
    )
    assert bundle.manifest["model_specification"]["fingerprint"] == result.specification_fingerprint
    assert bundle.manifest["specification_fingerprint"] == result.specification_fingerprint
    assert bundle.manifest["model_fingerprint"] == result.model_fingerprint
    assert bundle.manifest["fit_provenance"]["specification_fingerprint"] == result.specification_fingerprint
    assert bundle.manifest["validation_provenance"]["specification_fingerprint"] == result.specification_fingerprint
    assert bundle.manifest["report_provenance"]["specification_fingerprint"] == result.specification_fingerprint
    assert bundle.manifest["files"]["full_od.csv"]["row_count"] == 2
    assert bundle.manifest["files"]["od_map.csv"]["row_count"] == 2
    assert bundle.manifest["files"]["stop_summary.csv"]["row_count"] == 3
    assert set(
        bundle.read_table("od_map.csv").columns
    ) == {
        "origin_stop_id",
        "destination_stop_id",
        "departure_time_bin",
        "observed_demand",
        "modeled_demand",
        "residual",
        "fixed_free_status",
        "identifiability_class",
    }
    assert bundle.model_specification == result.model_specification
    assert bundle.manifest["time_zone"] == "Europe/Zurich"
    loaded = read_gravity_viewer_bundle(tmp_path / "bundle")
    assert loaded.model_specification == result.model_specification
    queried = loaded.query_od(origin="o1", minimum_flow=1.5)
    assert queried["od_index"].tolist() == [1]
    grouped = loaded.aggregate_measurements("line")
    assert grouped["observed"].tolist() == [15.0]
    assert loaded.manifest["provenance"]["od_layout_fingerprint"] == provenance[
        "od_layout_fingerprint"
    ]


def test_display_derived_files_aggregate_observations_and_preserve_status(tmp_path):
    result, report, _ = _inputs(tmp_path)
    full_od_path = report.files["full_od.csv"]
    source = full_od_path.read_text(encoding="utf-8").splitlines()
    source[0] = source[0] + ",observed_demand"
    source[1] = source[1] + ",2.5"
    source[2] = source[2] + ",4.0"
    full_od_path.write_text("\n".join(source) + "\n", encoding="utf-8")
    bundle = write_gravity_viewer_bundle(
        output_directory=tmp_path / "bundle",
        fit_result=result,
        validation_result=report.adequacy,
        report=report,
        network_files=_network_files(tmp_path),
    )
    od_rows = list(csv.DictReader((bundle.path / "od_map.csv").open(encoding="utf-8")))
    assert [row["observed_demand"] for row in od_rows] == ["2.5", "4"]
    assert [row["modeled_demand"] for row in od_rows] == ["1", "2"]
    assert [row["residual"] for row in od_rows] == ["1.5", "2"]
    assert [row["fixed_free_status"] for row in od_rows] == ["free", "free"]

    stop_rows = {
        row["stop_id"]: row
        for row in csv.DictReader((bundle.path / "stop_summary.csv").open(encoding="utf-8"))
    }
    assert stop_rows["o1"]["observed_total"] == "3"
    assert stop_rows["o1"]["modeled_total"] == "4"
    assert stop_rows["o1"]["residual"] == "-1"
    assert stop_rows["o1"]["outgoing_total"] == "3"
    assert stop_rows["d2"]["incoming_total"] == "7"
    assert bundle.manifest["row_counts"]["od_map.csv"] == 2
    assert bundle.manifest["row_counts"]["stop_summary.csv"] == 3
    for name in ("od_map.csv", "stop_summary.csv"):
        path = bundle.path / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert bundle.manifest["files"][name]["sha256"] == digest
        assert bundle.manifest["files"][name]["size_bytes"] == path.stat().st_size
    bundle.validate()


def test_older_bundle_without_derived_files_remains_readable(tmp_path):
    result, report, _ = _inputs(tmp_path)
    bundle = write_gravity_viewer_bundle(
        output_directory=tmp_path / "bundle",
        fit_result=result,
        validation_result=report.adequacy,
        report=report,
    )
    manifest_path = bundle.path / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("od_map.csv", "stop_summary.csv"):
        (bundle.path / name).unlink()
        manifest["files"].pop(name, None)
        manifest["row_counts"].pop(name, None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    restored = read_gravity_viewer_bundle(bundle.path)
    assert "od_map.csv" not in restored.table_names
    assert "stop_summary.csv" not in restored.table_names
    assert restored.read_table("full_od.csv").shape[0] == 2


def test_bundle_round_trips_identifiability_payload(tmp_path):
    result, report, _ = _inputs(tmp_path)
    diagnostic = _identifiability(result)
    bundle = write_gravity_viewer_bundle(
        output_directory=tmp_path / "bundle",
        fit_result=result,
        validation_result=report.adequacy,
        report=report,
        identifiability=diagnostic,
    )
    assert bundle.manifest["identifiability"]["available"] is True
    loaded = read_gravity_viewer_bundle(bundle.path)
    restored = read_gravity_viewer_bundle(bundle.path)
    persisted = restored.path / "identifiability"
    from public_transportation.inference.gravity import read_gravity_od_identifiability

    round_trip = read_gravity_od_identifiability(persisted)
    np.testing.assert_array_equal(
        round_trip.count_information_share, diagnostic.count_information_share
    )
    assert loaded.manifest["identifiability"]["provenance"]["model_fingerprint"] == result.model_fingerprint


def test_persisted_bundle_export_restores_report_and_cleans_staging(tmp_path):
    result, _report, provenance = _inputs(tmp_path)
    layout = ODParameterLayout(
        num_od_total=2,
        od_keys=(("o1", "d1", "am"), ("o1", "d2", "am")),
        free_od_indices=(0, 1),
        fixed_od_indices=(),
        fixed_od_values=(),
        free_baseline_values=(1.0, 2.0),
        fixed_zero_indices=(),
        fixed_positive_indices=(),
    )
    metadata = GravityValidationMetadata(
        3,
        measurement_type=np.asarray(("boarding", "boarding", "alighting")),
        line=np.asarray(("L1", "L1", "L1")),
        stop=np.asarray(("o1", "d1", "d2")),
        time_period=np.asarray(("am", "am", "am")),
    )
    validation_manifest = {
        "stage": "validate",
        "status": "completed",
        "fit_status": "converged",
        "predicted_measurements": result.predicted_measurements.tolist(),
        "model_fingerprint": result.model_fingerprint,
        "specification_fingerprint": result.specification_fingerprint,
        "model_specification": result.model_specification,
        **provenance,
    }
    result_payload = asdict(result)
    for name in (
        "raw_parameters",
        "physical_parameters",
        "free_od_demand",
        "active_od_demand",
        "full_od_demand",
        "predicted_measurements",
        "gradient",
    ):
        result_payload[name] = result_payload[name].tolist()
    fit_manifest = {
        "stage": "fit",
        "status": "completed",
        "model_fingerprint": result.model_fingerprint,
        "specification_fingerprint": result.specification_fingerprint,
        "model_specification": result.model_specification,
        "result": result_payload,
        **provenance,
    }

    bundle = write_persisted_gravity_viewer_bundle(
        output_directory=tmp_path / "bundle",
        fit_manifest=fit_manifest,
        validation_manifest=validation_manifest,
        observations=np.asarray((3.0, 5.0, 7.0)),
        od_layout=layout,
        metadata=metadata,
        likelihood="poisson",
        network_files=_network_files(tmp_path),
        bundle_metadata={"time_zone": "Europe/Zurich"},
    )

    assert bundle.path.is_dir()
    assert not list(tmp_path.glob(".bundle.report-*"))
    assert (
        read_gravity_viewer_bundle(bundle.path).manifest["bundle_type"]
        == "gravity_viewer_bundle"
    )


def test_persisted_bundle_export_accepts_json_normalized_tuple_specification(tmp_path):
    specification = GravityModelSpecification(
        time=GravityTimeSpecification(bin_labels=("morning", "evening"))
    )
    result, _report, provenance = _inputs(tmp_path, specification)
    layout = ODParameterLayout(
        num_od_total=2,
        od_keys=(("o1", "d1", "am"), ("o1", "d2", "am")),
        free_od_indices=(0, 1),
        fixed_od_indices=(),
        fixed_od_values=(),
        free_baseline_values=(1.0, 2.0),
        fixed_zero_indices=(),
        fixed_positive_indices=(),
    )
    metadata = GravityValidationMetadata(
        3,
        measurement_type=np.asarray(("boarding", "boarding", "alighting")),
        line=np.asarray(("L1", "L1", "L1")),
        stop=np.asarray(("o1", "d1", "d2")),
        time_period=np.asarray(("am", "am", "am")),
    )
    validation_manifest = {
        "stage": "validate",
        "status": "completed",
        "fit_status": "converged",
        "predicted_measurements": result.predicted_measurements.tolist(),
        "model_fingerprint": result.model_fingerprint,
        "specification_fingerprint": result.specification_fingerprint,
        "model_specification": result.model_specification,
        **provenance,
    }
    result_payload = asdict(result)
    for name in (
        "raw_parameters",
        "physical_parameters",
        "free_od_demand",
        "active_od_demand",
        "full_od_demand",
        "predicted_measurements",
        "gradient",
    ):
        result_payload[name] = result_payload[name].tolist()
    fit_manifest = {
        "stage": "fit",
        "status": "completed",
        "model_fingerprint": result.model_fingerprint,
        "specification_fingerprint": result.specification_fingerprint,
        "model_specification": result.model_specification,
        "result": result_payload,
        **provenance,
    }
    # A manifest has crossed JSON, so tuple-valued time-bin labels are lists.
    fit_manifest = json.loads(json.dumps(fit_manifest))
    validation_manifest = json.loads(json.dumps(validation_manifest))

    bundle = write_persisted_gravity_viewer_bundle(
        output_directory=tmp_path / "json-normalized-bundle",
        fit_manifest=fit_manifest,
        validation_manifest=validation_manifest,
        observations=np.asarray((3.0, 5.0, 7.0)),
        od_layout=layout,
        metadata=metadata,
        likelihood="poisson",
        network_files=_network_files(tmp_path),
    )

    assert bundle.path.is_dir()
    assert bundle.manifest["specification_fingerprint"] == specification.fingerprint


def test_bundle_reader_rejects_checksum_tampering(tmp_path):
    result, report, _ = _inputs(tmp_path)
    write_gravity_viewer_bundle(
        output_directory=tmp_path / "bundle",
        fit_result=result,
        validation_result=report.adequacy,
        report=report,
    )
    path = tmp_path / "bundle" / "full_od.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="(checksum|size) mismatch"):
        read_gravity_viewer_bundle(tmp_path / "bundle")


def test_bundle_reader_rejects_manifest_provenance_mismatch(tmp_path):
    result, report, _ = _inputs(tmp_path)
    write_gravity_viewer_bundle(
        output_directory=tmp_path / "bundle",
        fit_result=result,
        validation_result=report.adequacy,
        report=report,
    )
    manifest_path = tmp_path / "bundle" / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fit_provenance"]["model_fingerprint"] = "wrong-model"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fit/model provenance"):
        read_gravity_viewer_bundle(tmp_path / "bundle")


def test_bundle_reader_rejects_validation_specification_mismatch(tmp_path):
    result, report, _ = _inputs(tmp_path)
    write_gravity_viewer_bundle(
        output_directory=tmp_path / "bundle",
        fit_result=result,
        validation_result=report.adequacy,
        report=report,
    )
    manifest_path = tmp_path / "bundle" / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["validation_provenance"]["specification_fingerprint"] = "wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="validation/specification"):
        read_gravity_viewer_bundle(tmp_path / "bundle")


def test_bundle_can_omit_identifiability_or_require_it(tmp_path):
    result, report, _ = _inputs(tmp_path)
    bundle = write_gravity_viewer_bundle(
        output_directory=tmp_path / "bundle",
        fit_result=result,
        validation_result=report.adequacy,
        report=report,
    )
    assert bundle.manifest["identifiability"]["available"] is False
    with pytest.raises(ValueError, match="identifiability"):
        write_gravity_viewer_bundle(
            output_directory=tmp_path / "required",
            fit_result=result,
            validation_result=report.adequacy,
            report=report,
            require_identifiability=True,
        )


def test_bundle_requires_exact_model_specification(tmp_path):
    result, report, _ = _inputs(tmp_path)
    without_spec = replace(result, model_specification=None, specification_fingerprint="")
    with pytest.raises(ValueError, match="exact model specification"):
        write_gravity_viewer_bundle(
            output_directory=tmp_path / "bundle",
            fit_result=without_spec,
            validation_result=report.adequacy,
            report=report,
        )


def test_bundle_rejects_validation_model_mismatch(tmp_path):
    result, report, _ = _inputs(tmp_path)
    mismatched = replace(report.adequacy, model_fingerprint="other-model")
    with pytest.raises(ValueError, match="validation provenance"):
        write_gravity_viewer_bundle(
            output_directory=tmp_path / "bundle",
            fit_result=result,
            validation_result=mismatched,
            report=report,
        )
