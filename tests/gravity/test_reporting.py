from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np
import pytest

from public_transportation.inference.gravity import (
    GRAVITY_REPORT_PROVENANCE_FIELDS,
    GRAVITY_RESULT_SCHEMA_VERSION,
    GravityEstimationResult,
    GravityLikelihood,
    GravityStrategySelection,
    GravityValidationMetadata,
    PersistedGravityReportInputs,
    validate_gravity_report_provenance,
    write_persisted_gravity_detailed_report,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _portable_inputs() -> tuple[dict[str, object], dict[str, object], ODParameterLayout]:
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
        parameter_names=("beta_time",),
    )
    provenance = {
        field: f"{field}-value"
        for field in GRAVITY_REPORT_PROVENANCE_FIELDS
    }
    provenance["od_layout_fingerprint"] = layout.fingerprint
    result_payload = _jsonable(asdict(result))
    assert isinstance(result_payload, dict)
    fit_manifest = {
        "status": "completed",
        "result": result_payload,
        **provenance,
    }
    validation_manifest = {
        "status": "completed",
        "predicted_measurements": [4.0, 5.0, 6.0],
        **provenance,
    }
    return fit_manifest, validation_manifest, layout


def test_persisted_report_is_portable_and_records_canonical_provenance(tmp_path):
    fit, validation, layout = _portable_inputs()
    metadata = GravityValidationMetadata(
        3,
        measurement_type=np.asarray(("boarding", "boarding", "alighting")),
        line=np.asarray(("L1", "L1", "L1")),
    )
    inputs = PersistedGravityReportInputs(
        fit_manifest=fit,
        validation_manifest=validation,
        observations=np.asarray((3.0, 5.0, 7.0)),
        od_layout=layout,
        metadata=metadata,
        likelihood=GravityLikelihood.POISSON,
    )
    assert not inputs.observations.flags.writeable
    np.testing.assert_array_equal(inputs.predicted_measurements, (4.0, 5.0, 6.0))
    output = tmp_path / "report"
    report = write_persisted_gravity_detailed_report(
        fit_manifest=fit,
        validation_manifest=validation,
        observations=np.asarray((3.0, 5.0, 7.0)),
        od_layout=layout,
        metadata=metadata,
        likelihood=GravityLikelihood.POISSON,
        output_directory=output,
    )
    assert report.output_directory == output.resolve()
    payload = json.loads((output / "report.json").read_text())
    assert payload["provenance"]["source"] == "persisted_fit_validation"
    assert {
        key: payload["provenance"][key]
        for key in GRAVITY_REPORT_PROVENANCE_FIELDS
    } == {
        key: fit[key] for key in GRAVITY_REPORT_PROVENANCE_FIELDS
    }


def test_portable_report_does_not_require_case_context(tmp_path):
    fit, validation, layout = _portable_inputs()
    report = write_persisted_gravity_detailed_report(
        fit_manifest=fit,
        validation_manifest=validation,
        observations=[3.0, 5.0, 7.0],
        od_layout=layout,
        likelihood="poisson",
        output_directory=tmp_path / "portable",
    )
    assert report.files["report.json"].is_file()


@pytest.mark.parametrize(
    "field",
    ("artifact_identity_fingerprint", "package_revision"),
)
def test_provenance_mismatch_fails_before_output_creation(tmp_path, field):
    fit, validation, layout = _portable_inputs()
    validation[field] = "different"
    output = tmp_path / "should-not-exist"
    with pytest.raises(ValueError, match=field):
        write_persisted_gravity_detailed_report(
            fit_manifest=fit,
            validation_manifest=validation,
            observations=[3.0, 5.0, 7.0],
            od_layout=layout,
            likelihood="poisson",
            output_directory=output,
        )
    assert not output.exists()


def test_supplied_and_layout_provenance_mismatches_are_rejected(tmp_path):
    fit, validation, layout = _portable_inputs()
    supplied = {field: fit[field] for field in GRAVITY_REPORT_PROVENANCE_FIELDS}
    supplied["package_revision"] = "wrong-package"
    with pytest.raises(ValueError, match="package_revision"):
        validate_gravity_report_provenance(
            fit_manifest=fit,
            validation_manifest=validation,
            od_layout=layout,
            supplied_provenance=supplied,
        )

    changed_layout = ODParameterLayout(
        num_od_total=2,
        od_keys=(("o1", "d1", "am"), ("o1", "d2", "pm")),
        free_od_indices=(0, 1),
        fixed_od_indices=(),
        fixed_od_values=(),
        free_baseline_values=(1.0, 2.0),
        fixed_zero_indices=(),
        fixed_positive_indices=(),
    )
    output = tmp_path / "layout-mismatch"
    with pytest.raises(ValueError, match="od_layout_fingerprint"):
        write_persisted_gravity_detailed_report(
            fit_manifest=fit,
            validation_manifest=validation,
            observations=[3.0, 5.0, 7.0],
            od_layout=changed_layout,
            likelihood="poisson",
            output_directory=output,
        )
    assert not output.exists()


def test_missing_provenance_and_predictions_are_rejected(tmp_path):
    fit, validation, layout = _portable_inputs()
    del validation["binding_fingerprint"]
    with pytest.raises(ValueError, match="binding_fingerprint"):
        validate_gravity_report_provenance(
            fit_manifest=fit,
            validation_manifest=validation,
            od_layout=layout,
        )

    fit, validation, layout = _portable_inputs()
    del validation["predicted_measurements"]
    with pytest.raises(ValueError, match="predicted_measurements"):
        write_persisted_gravity_detailed_report(
            fit_manifest=fit,
            validation_manifest=validation,
            observations=[3.0, 5.0, 7.0],
            od_layout=layout,
            likelihood="poisson",
            output_directory=tmp_path / "missing-predictions",
        )


def test_predictions_must_agree_with_validation(tmp_path):
    fit, validation, layout = _portable_inputs()
    validation["predicted_measurements"] = [4.0, 5.0, 99.0]
    with pytest.raises(ValueError, match="predictions differ"):
        write_persisted_gravity_detailed_report(
            fit_manifest=fit,
            validation_manifest=validation,
            observations=[3.0, 5.0, 7.0],
            od_layout=layout,
            likelihood="poisson",
            output_directory=tmp_path / "prediction-mismatch",
        )


def test_layout_and_validation_metadata_round_trip_as_immutable_arrays():
    _, _, layout = _portable_inputs()
    restored_layout = ODParameterLayout.from_dict(layout.to_dict())
    assert restored_layout == layout
    assert restored_layout.fingerprint == layout.fingerprint

    metadata = GravityValidationMetadata(
        2,
        line=np.asarray(("L1", "L2")),
        observation_time=np.asarray(("08:00", "09:00")),
    )
    restored_metadata = GravityValidationMetadata.from_dict(metadata.to_dict())
    np.testing.assert_array_equal(restored_metadata.line, metadata.line)
    assert restored_metadata.line is not None
    assert not restored_metadata.line.flags.writeable
    assert restored_metadata.to_dict() == metadata.to_dict()

    broken_layout = layout.to_dict()
    broken_layout["fingerprint"] = "not-the-layout-fingerprint"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ODParameterLayout.from_dict(broken_layout)
    broken_metadata = metadata.to_dict()
    broken_metadata["num_measurements"] = 3
    with pytest.raises(ValueError, match="shape"):
        GravityValidationMetadata.from_dict(broken_metadata)


def test_non_completed_manifests_are_rejected_before_output(tmp_path):
    fit, validation, layout = _portable_inputs()
    fit["status"] = "running"
    output = tmp_path / "not-completed"
    with pytest.raises(ValueError, match="not completed"):
        write_persisted_gravity_detailed_report(
            fit_manifest=fit,
            validation_manifest=validation,
            observations=[3.0, 5.0, 7.0],
            od_layout=layout,
            likelihood="poisson",
            output_directory=output,
        )
    assert not output.exists()
