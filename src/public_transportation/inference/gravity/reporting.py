"""Persistent detailed and executive reports for gravity-model estimates."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from public_transportation.inference.block_coordinate._canonical import canonical_json
from public_transportation.inference.od_parameter_layout import ODParameterLayout

from .estimator import GravityEstimationResult
from .identifiability import GravityODIdentifiability
from .objective import GravityLikelihood
from .specification import GravityModelSpecification
from .validation import (
    GravityAdequacyConfig,
    GravityAdequacyReport,
    GravityValidationMetadata,
    build_gravity_adequacy_report,
)


GRAVITY_DETAILED_REPORT_SCHEMA_VERSION = 1
GRAVITY_REPORT_PROVENANCE_FIELDS = (
    "artifact_identity_fingerprint",
    "assignment_fingerprint",
    "binding_fingerprint",
    "canonical_index_fingerprint",
    "compact_layout_fingerprint",
    "gravity_features_fingerprint",
    "mapping_fingerprint",
    "od_layout_fingerprint",
    "package_revision",
)
_REPORT_FILES = (
    "parameters.csv",
    "full_od.csv",
    "predicted_measurements.csv",
    "residuals.csv",
    "grouped_residuals.csv",
    "report.json",
    "report.md",
    "executive_summary.md",
)
_METADATA_FIELDS = (
    "method_id",
    "measurement_type",
    "line",
    "direction",
    "stop",
    "observation_time",
    "time_period",
    "origin_zone",
    "destination_zone",
    "vehicle_journey",
)
_PARAMETER_FIELDS = (
    "parameter_index",
    "parameter_name",
    "raw_value",
    "physical_value",
)
_OD_FIELDS = (
    "od_index",
    "origin_stop_id",
    "destination_stop_id",
    "departure_time_bin",
    "estimated_demand",
    "cell_status",
    "inference_basis",
    "inference_score",
    "fixed_value",
    "prior_demand",
    "estimated_to_prior_ratio",
    "count_information_share",
    "regularization_information_share",
    "auxiliary_information_share",
    "local_information_variance",
    "identifiability_class",
    "identifiability_reason",
)
_MISSING = object()
_PREDICTION_RTOL = 5.0e-6
_PREDICTION_ATOL = 5.0e-6


def _validated_result_specification(
    result: GravityEstimationResult,
) -> tuple[dict[str, object], str]:
    """Return the exact specification carried by a fit result.

    Reports must never infer a model from a human-readable name.  The result
    must contain the canonical serialization produced by
    :class:`GravityModelSpecification` and its matching fingerprint.
    """
    raw = result.model_specification
    if raw is None or not isinstance(raw, Mapping):
        raise ValueError(
            "gravity result is missing model_specification; regenerate the fit "
            "before generating a detailed report."
        )
    try:
        specification = GravityModelSpecification.from_dict(dict(raw))
    except (TypeError, ValueError) as error:
        raise ValueError("gravity result model_specification is invalid.") from error
    serialized = specification.to_dict()
    if canonical_json(dict(raw)) != canonical_json(serialized):
        raise ValueError(
            "gravity result model_specification is not the canonical serialized specification."
        )
    if result.specification_fingerprint != specification.fingerprint:
        raise ValueError(
            "gravity result specification_fingerprint does not match its "
            "serialized model_specification."
        )
    return serialized, specification.fingerprint


def _immutable_vector(value: object) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError("persisted vectors must be one-dimensional.")
    array.setflags(write=False)
    return array


def _validate_prediction_agreement(
    fit_predictions: object, validation_predictions: object
) -> None:
    fit = np.asarray(fit_predictions, dtype=np.float64)
    validation = np.asarray(validation_predictions, dtype=np.float64)
    if fit.ndim != 1 or validation.ndim != 1:
        raise ValueError("fit and validation predictions must be one-dimensional.")
    if fit.shape != validation.shape:
        raise ValueError(
            "persisted fit and validation predictions have different lengths: "
            f"{fit.size} != {validation.size}."
        )
    if not np.all(np.isfinite(fit)) or not np.all(np.isfinite(validation)):
        raise ValueError("persisted fit and validation predictions must be finite.")
    if not np.allclose(
        fit,
        validation,
        rtol=_PREDICTION_RTOL,
        atol=_PREDICTION_ATOL,
    ):
        raise ValueError(
            "persisted fit and validation predictions differ beyond the allowed "
            f"tolerance (rtol={_PREDICTION_RTOL:g}, atol={_PREDICTION_ATOL:g})."
        )


def _validation_predictions(manifest: Mapping[str, object]) -> object:
    """Return persisted validation predictions from supported manifest shapes."""
    direct = manifest.get("predicted_measurements", _MISSING)
    if direct is not _MISSING:
        return direct
    nested = manifest.get("result")
    if isinstance(nested, Mapping):
        nested_value = nested.get("predicted_measurements", _MISSING)
        if nested_value is not _MISSING:
            return nested_value
    return _MISSING


def _manifest_value(
    manifest: Mapping[str, object], field: str
) -> object:
    """Find a canonical field in supported persisted manifest locations."""
    direct = manifest.get(field, _MISSING)
    if direct is not _MISSING and direct is not None:
        return direct

    nested_names = ("fingerprints", "provenance", "result")
    for nested_name in nested_names:
        nested = manifest.get(nested_name)
        if not isinstance(nested, Mapping):
            continue
        value = nested.get(field, _MISSING)
        if value is not _MISSING and value is not None:
            return value
        nested_fingerprints = nested.get("fingerprints")
        if isinstance(nested_fingerprints, Mapping):
            value = nested_fingerprints.get(field, _MISSING)
            if value is not _MISSING and value is not None:
                return value

    # Gravity run manifests historically called this field
    # ``repository_revision``.  It has the same identity meaning here.
    if field == "package_revision":
        legacy = manifest.get("repository_revision", _MISSING)
        if legacy is not _MISSING and legacy is not None:
            return legacy
    return _MISSING


def _manifest_specification(
    manifest: Mapping[str, object], name: str
) -> tuple[dict[str, object], str]:
    """Recover and validate an exact specification from a stage manifest."""
    roots: list[Mapping[str, object]] = [manifest]
    for key in ("result", "adequacy", "validation_result", "provenance"):
        value = manifest.get(key)
        if isinstance(value, Mapping):
            roots.append(value)

    serialized_candidates: list[dict[str, object]] = []
    fingerprint_candidates: list[str] = []
    for root in roots:
        raw = root.get("model_specification", _MISSING)
        if isinstance(raw, Mapping) and isinstance(raw.get("specification"), Mapping):
            raw = raw["specification"]
        if raw is not _MISSING and raw is not None:
            if not isinstance(raw, Mapping):
                raise ValueError(f"{name} manifest model_specification must be a mapping.")
            try:
                parsed = GravityModelSpecification.from_dict(dict(raw))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} manifest model_specification is invalid.") from error
            canonical = parsed.to_dict()
            if canonical_json(dict(raw)) != canonical_json(canonical):
                raise ValueError(
                    f"{name} manifest model_specification is not canonical."
                )
            serialized_candidates.append(canonical)
            fingerprint_candidates.append(parsed.fingerprint)
        raw_fingerprint = root.get("specification_fingerprint", _MISSING)
        if raw_fingerprint is not _MISSING and raw_fingerprint is not None:
            fingerprint_candidates.append(str(raw_fingerprint))

    if not serialized_candidates:
        raise ValueError(
            f"{name} manifest is missing model_specification; regenerate the stage "
            "with a package version that persists the exact specification."
        )
    specification = serialized_candidates[0]
    if any(candidate != specification for candidate in serialized_candidates[1:]):
        raise ValueError(f"{name} manifest contains conflicting model specifications.")
    computed = GravityModelSpecification.from_dict(specification).fingerprint
    if any(candidate != computed for candidate in fingerprint_candidates):
        raise ValueError(
            f"{name} manifest specification_fingerprint does not match its "
            "serialized model_specification."
        )
    return specification, computed


def _manifest_status(manifest: Mapping[str, object], name: str) -> None:
    status = manifest.get("status", _MISSING)
    if status != "completed":
        raise ValueError(
            f"{name} manifest is not completed: status={status!r}; "
            "portable report generation requires completed fit and validation manifests."
        )


def validate_gravity_report_provenance(
    *,
    fit_manifest: Mapping[str, object],
    validation_manifest: Mapping[str, object],
    od_layout: ODParameterLayout,
    supplied_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate persisted report identity before any output is created."""
    if not isinstance(fit_manifest, Mapping):
        raise TypeError("fit_manifest must be a mapping.")
    if not isinstance(validation_manifest, Mapping):
        raise TypeError("validation_manifest must be a mapping.")
    if not isinstance(od_layout, ODParameterLayout):
        raise TypeError("od_layout must be an ODParameterLayout.")
    result = fit_manifest.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("fit_manifest['result'] must be a mapping.")
    validation_predictions = _validation_predictions(validation_manifest)
    if validation_predictions is _MISSING:
        raise ValueError(
            "validation_manifest is missing predicted_measurements."
        )
    _manifest_status(fit_manifest, "fit")
    _manifest_status(validation_manifest, "validation")
    fit_predictions = result.get("predicted_measurements", _MISSING)
    if fit_predictions is _MISSING:
        raise ValueError(
            "fit_manifest['result'] is missing predicted_measurements."
        )
    _validate_prediction_agreement(fit_predictions, validation_predictions)

    fit_specification, fit_specification_fingerprint = _manifest_specification(
        fit_manifest, "fit"
    )
    validation_specification, validation_specification_fingerprint = (
        _manifest_specification(validation_manifest, "validation")
    )
    if fit_specification_fingerprint != validation_specification_fingerprint:
        raise ValueError(
            "specification_fingerprint differs between fit and validation: "
            f"fit={fit_specification_fingerprint!r}, "
            f"validation={validation_specification_fingerprint!r}."
        )
    if fit_specification != validation_specification:
        raise ValueError(
            "model_specification differs between fit and validation manifests."
        )

    fit_model_fingerprint = _manifest_value(fit_manifest, "model_fingerprint")
    validation_model_fingerprint = _manifest_value(
        validation_manifest, "model_fingerprint"
    )
    if fit_model_fingerprint is _MISSING or validation_model_fingerprint is _MISSING:
        raise ValueError(
            "model_fingerprint is required in both fit and validation manifests."
        )
    if fit_model_fingerprint != validation_model_fingerprint:
        raise ValueError(
            "model_fingerprint differs between fit and validation: "
            f"fit={fit_model_fingerprint!r}, "
            f"validation={validation_model_fingerprint!r}."
        )

    persisted: dict[str, object] = {}
    for field in GRAVITY_REPORT_PROVENANCE_FIELDS:
        fit_value = _manifest_value(fit_manifest, field)
        validation_value = _manifest_value(validation_manifest, field)
        if fit_value is _MISSING or validation_value is _MISSING:
            raise ValueError(
                f"required provenance field {field!r} is missing: "
                f"fit={None if fit_value is _MISSING else fit_value!r}, "
                f"validation={None if validation_value is _MISSING else validation_value!r}."
            )
        if fit_value != validation_value:
            raise ValueError(
                f"provenance field {field!r} differs between fit and validation: "
                f"fit={fit_value!r}, validation={validation_value!r}."
            )
        persisted[field] = fit_value

    persisted["model_fingerprint"] = fit_model_fingerprint
    persisted["specification_fingerprint"] = fit_specification_fingerprint
    persisted["model_specification"] = fit_specification

    if persisted["od_layout_fingerprint"] != od_layout.fingerprint:
        raise ValueError(
            "provenance field 'od_layout_fingerprint' differs from the supplied "
            f"OD layout: persisted={persisted['od_layout_fingerprint']!r}, "
            f"layout={od_layout.fingerprint!r}."
        )

    if supplied_provenance is not None:
        if not isinstance(supplied_provenance, Mapping):
            raise TypeError("supplied_provenance must be a mapping or None.")
        for field in GRAVITY_REPORT_PROVENANCE_FIELDS:
            supplied = supplied_provenance.get(field, _MISSING)
            if supplied is _MISSING or supplied != persisted[field]:
                raise ValueError(
                    f"supplied provenance field {field!r} differs from persisted "
                    f"value: supplied={None if supplied is _MISSING else supplied!r}, "
                    f"persisted={persisted[field]!r}."
                )
        for field in (
            "model_fingerprint",
            "specification_fingerprint",
            "model_specification",
        ):
            supplied = supplied_provenance.get(field, _MISSING)
            if supplied is _MISSING:
                continue
            if supplied != persisted[field]:
                raise ValueError(
                    f"supplied provenance field {field!r} differs from persisted "
                    f"value: supplied={supplied!r}, persisted={persisted[field]!r}."
                )
    return persisted


def _validate_expected_provenance(
    provenance: Mapping[str, object] | None,
    expected: Mapping[str, object],
) -> None:
    if not isinstance(expected, Mapping):
        raise TypeError("expected_provenance must be a mapping or None.")
    if not isinstance(provenance, Mapping):
        raise ValueError(
            "expected persisted provenance was supplied, but report provenance is missing."
        )
    for field in (
        *GRAVITY_REPORT_PROVENANCE_FIELDS,
        "model_fingerprint",
        "specification_fingerprint",
        "model_specification",
    ):
        expected_value = expected.get(field, _MISSING)
        actual_value = provenance.get(field, _MISSING)
        if expected_value is _MISSING or actual_value is _MISSING:
            raise ValueError(
                f"required provenance field {field!r} is missing from the report "
                f"provenance: expected={None if expected_value is _MISSING else expected_value!r}, "
                f"actual={None if actual_value is _MISSING else actual_value!r}."
            )
        if expected_value != actual_value:
            raise ValueError(
                f"report provenance field {field!r} differs from the persisted "
                f"value: expected={expected_value!r}, actual={actual_value!r}."
            )


def _validate_restored_result(
    result: GravityEstimationResult, od_layout: ODParameterLayout
) -> None:
    """Validate typed result vectors before portable report output is created."""
    _validated_result_specification(result)
    vectors: dict[str, np.ndarray] = {}
    for name in (
        "raw_parameters",
        "physical_parameters",
        "free_od_demand",
        "active_od_demand",
        "full_od_demand",
        "predicted_measurements",
        "gradient",
    ):
        array = np.asarray(getattr(result, name), dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"persisted result field {name!r} must be one-dimensional.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"persisted result field {name!r} must be finite.")
        vectors[name] = array
    if vectors["raw_parameters"].shape != vectors["physical_parameters"].shape:
        raise ValueError(
            "persisted raw_parameters and physical_parameters have different lengths."
        )
    if vectors["gradient"].shape != vectors["raw_parameters"].shape:
        raise ValueError("persisted gradient length does not match raw_parameters.")
    if vectors["full_od_demand"].size != od_layout.num_od_total:
        raise ValueError(
            "persisted full_od_demand length does not match the supplied OD layout: "
            f"{vectors['full_od_demand'].size} != {od_layout.num_od_total}."
        )
    if vectors["free_od_demand"].size != od_layout.num_free:
        raise ValueError(
            "persisted free_od_demand length does not match the supplied OD layout: "
            f"{vectors['free_od_demand'].size} != {od_layout.num_free}."
        )


@dataclass(frozen=True, slots=True)
class GravityDetailedReport:
    """Summary and paths produced by :func:`write_gravity_detailed_report`."""

    schema_version: int
    report_fingerprint: str
    output_directory: Path
    files: Mapping[str, Path]
    adequacy: GravityAdequacyReport
    estimated_od_cells: int
    fixed_od_cells: int
    structural_zero_cells: int
    executive_messages: tuple[str, ...]
    identifiability: GravityODIdentifiability | None = None


@dataclass(frozen=True, slots=True)
class PersistedGravityReportInputs:
    """Validated fit, validation, and metadata inputs for portable reports."""

    fit_manifest: Mapping[str, object]
    validation_manifest: Mapping[str, object]
    observations: np.ndarray
    od_layout: ODParameterLayout
    metadata: GravityValidationMetadata | None
    likelihood: GravityLikelihood | str

    def __post_init__(self) -> None:
        if not isinstance(self.fit_manifest, Mapping):
            raise TypeError("fit_manifest must be a mapping.")
        if not isinstance(self.validation_manifest, Mapping):
            raise TypeError("validation_manifest must be a mapping.")
        result = self.fit_manifest.get("result")
        if not isinstance(result, Mapping):
            raise ValueError("fit_manifest['result'] must be a mapping.")
        validation_predictions = _validation_predictions(self.validation_manifest)
        if validation_predictions is _MISSING:
            raise ValueError(
                "validation_manifest must contain predicted_measurements."
            )

        observed = np.asarray(self.observations, dtype=np.float64)
        if observed.ndim != 1:
            raise ValueError("observations must be one-dimensional.")
        if not np.all(np.isfinite(observed)):
            raise ValueError("observations must contain finite values.")
        object.__setattr__(self, "observations", _immutable_vector(observed))

        predicted = np.asarray(
            validation_predictions, dtype=np.float64
        )
        if predicted.ndim != 1:
            raise ValueError("persisted predictions must be one-dimensional.")
        if not np.all(np.isfinite(predicted)):
            raise ValueError("persisted predictions must contain finite values.")
        if observed.shape != predicted.shape:
            raise ValueError(
                "observations and persisted predictions have different lengths: "
                f"{observed.size} != {predicted.size}."
            )
        persisted_fit_predictions = result.get("predicted_measurements")
        if persisted_fit_predictions is None:
            raise ValueError(
                "fit_manifest['result'] must contain predicted_measurements."
            )
        fit_predicted = np.asarray(persisted_fit_predictions, dtype=np.float64)
        _validate_prediction_agreement(fit_predicted, predicted)

        if self.metadata is not None:
            if not isinstance(self.metadata, GravityValidationMetadata):
                raise TypeError(
                    "metadata must be GravityValidationMetadata or None."
                )
            if self.metadata.num_measurements != observed.size:
                raise ValueError(
                    "metadata length does not match observations: "
                    f"{self.metadata.num_measurements} != {observed.size}."
                )
        validate_gravity_report_provenance(
            fit_manifest=self.fit_manifest,
            validation_manifest=self.validation_manifest,
            od_layout=self.od_layout,
        )

    @property
    def predicted_measurements(self) -> np.ndarray:
        """Return the validated persisted validation predictions."""
        return np.asarray(
            _validation_predictions(self.validation_manifest), dtype=np.float64
        )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _metadata_rows(
    observations: np.ndarray, metadata: GravityValidationMetadata
) -> list[dict[str, object]]:
    rows = []
    for index, observed in enumerate(observations):
        row: dict[str, object] = {"row_index": index, "observed": float(observed)}
        for field in _METADATA_FIELDS:
            labels = getattr(metadata, field)
            row[field] = "" if labels is None else str(labels[index])
        rows.append(row)
    return rows


def _parameter_rows(result: GravityEstimationResult) -> list[dict[str, object]]:
    raw = np.asarray(result.raw_parameters, dtype=np.float64)
    physical = np.asarray(result.physical_parameters, dtype=np.float64)
    names = tuple(result.parameter_names)
    if raw.shape != physical.shape:
        raise ValueError("raw and physical parameter vectors have different shapes.")
    if names and len(names) != raw.size:
        raise ValueError("parameter_names does not match the parameter vector.")
    labels = names or tuple(f"parameter_{index}" for index in range(raw.size))
    if np.any(~np.isfinite(raw)) or np.any(~np.isfinite(physical)):
        raise ValueError("gravity parameter vectors must be finite.")
    return [
        {
            "parameter_index": index,
            "parameter_name": str(name),
            "raw_value": float(raw[index]),
            "physical_value": float(physical[index]),
        }
        for index, name in enumerate(labels)
    ]


def _validate_identifiability_for_report(
    diagnostic: GravityODIdentifiability,
    *,
    result: GravityEstimationResult,
    od_layout: ODParameterLayout,
    expected_provenance: Mapping[str, object] | None = None,
) -> None:
    """Validate a diagnostic before report output is created."""
    if not isinstance(diagnostic, GravityODIdentifiability):
        raise TypeError("identifiability must be GravityODIdentifiability or None.")
    if diagnostic.num_cells != od_layout.num_od_total:
        raise ValueError(
            "identifiability length does not match the OD layout: "
            f"{diagnostic.num_cells} != {od_layout.num_od_total}."
        )
    expected_fixed = np.ones(od_layout.num_od_total, dtype=bool)
    expected_fixed[np.asarray(od_layout.free_od_indices, dtype=np.int64)] = False
    actual_fixed = np.asarray(diagnostic.classification) == "fixed_by_policy"
    if not np.array_equal(actual_fixed, expected_fixed):
        raise ValueError(
            "identifiability fixed/free classification does not match the OD layout."
        )
    if diagnostic.total_hessian_dimension != np.asarray(result.raw_parameters).size:
        raise ValueError(
            "identifiability parameter dimension does not match the fitted result."
        )
    provenance = diagnostic.provenance
    if not isinstance(provenance, Mapping):
        raise ValueError("identifiability provenance is missing or invalid.")
    diagnostic_layout = provenance.get("od_layout_fingerprint")
    allowed_layouts = {od_layout.fingerprint}
    compact_layout = provenance.get("compact_layout_fingerprint")
    if compact_layout is not None:
        allowed_layouts.add(compact_layout)
    if diagnostic_layout is not None and diagnostic_layout not in allowed_layouts:
        raise ValueError("identifiability OD-layout fingerprint does not match the report layout.")
    if provenance.get("model_fingerprint") != result.model_fingerprint:
        raise ValueError("identifiability model_fingerprint does not match the fit result.")
    raw_fingerprint = provenance.get("raw_parameters_fingerprint")
    if raw_fingerprint is not None and raw_fingerprint != _vector_fingerprint(
        result.raw_parameters
    ):
        raise ValueError("identifiability raw-parameter fingerprint does not match the fit result.")
    prediction_fingerprint = provenance.get("predicted_measurements_fingerprint")
    if prediction_fingerprint is not None and prediction_fingerprint != _vector_fingerprint(
        result.predicted_measurements
    ):
        raise ValueError(
            "identifiability prediction fingerprint does not match the fit result."
        )
    if expected_provenance is not None:
        if not isinstance(expected_provenance, Mapping):
            raise TypeError("expected_provenance must be a mapping or None.")
        for field in (*GRAVITY_REPORT_PROVENANCE_FIELDS, "model_fingerprint"):
            expected = expected_provenance.get(field, _MISSING)
            if expected is _MISSING:
                continue
            actual = provenance.get(field, _MISSING)
            if actual is _MISSING or actual != expected:
                raise ValueError(
                    f"identifiability provenance field {field!r} differs from the fit/validation provenance."
                )


def _vector_fingerprint(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _od_rows(
    result: GravityEstimationResult,
    layout: ODParameterLayout,
    identifiability: GravityODIdentifiability | None = None,
) -> list[dict[str, object]]:
    demand = np.asarray(result.full_od_demand, dtype=np.float64)
    if demand.shape != (layout.num_od_total,) or np.any(~np.isfinite(demand)):
        raise ValueError("full_od_demand does not match the OD parameter layout.")
    free_indices = set(layout.free_od_indices)
    fixed_values = dict(zip(layout.fixed_od_indices, layout.fixed_od_values, strict=True))
    baselines = dict(zip(layout.free_od_indices, layout.free_baseline_values, strict=True))
    rows = []
    for index, key in enumerate(layout.od_keys):
        origin, destination, time_period = key
        value = float(demand[index])
        if identifiability is None:
            identifiability_values: dict[str, object] = {
                "count_information_share": "",
                "regularization_information_share": "",
                "auxiliary_information_share": "",
                "local_information_variance": "",
                "identifiability_class": "",
                "identifiability_reason": "",
            }
        elif index in free_indices:
            count_share_value = identifiability.count_information_share[index]
            regularization_share_value = identifiability.regularization_information_share[
                index
            ]
            variance_value = identifiability.local_information_variance[index]
            auxiliary_share_value = (
                None
                if identifiability.auxiliary_information_share is None
                else identifiability.auxiliary_information_share[index]
            )
            identifiability_values = {
                "count_information_share": (
                    "" if np.isnan(count_share_value) else float(count_share_value)
                ),
                "regularization_information_share": (
                    ""
                    if np.isnan(regularization_share_value)
                    else float(regularization_share_value)
                ),
                "auxiliary_information_share": (
                    ""
                    if auxiliary_share_value is None or np.isnan(auxiliary_share_value)
                    else float(auxiliary_share_value)
                ),
                "local_information_variance": (
                    "" if np.isnan(variance_value) else float(variance_value)
                ),
                "identifiability_class": str(identifiability.classification[index]),
                "identifiability_reason": str(identifiability.diagnostic_reason[index]),
            }
        else:
            identifiability_values = {
                "count_information_share": "",
                "regularization_information_share": "",
                "auxiliary_information_share": "",
                "local_information_variance": "",
                "identifiability_class": "fixed_by_policy",
                "identifiability_reason": "fixed_input_or_structural_zero_policy",
            }
        if index in free_indices:
            baseline = float(baselines[index])
            rows.append(
                {
                    "od_index": index,
                    "origin_stop_id": str(origin),
                    "destination_stop_id": str(destination),
                    "departure_time_bin": str(time_period),
                    "estimated_demand": value,
                    "cell_status": "estimated_free",
                    "inference_basis": "observed_flows_and_model_regularization",
                    "inference_score": 1.0,
                    "fixed_value": "",
                    "prior_demand": baseline,
                    "estimated_to_prior_ratio": value / baseline,
                    **identifiability_values,
                }
            )
        else:
            fixed = float(fixed_values[index])
            rows.append(
                {
                    "od_index": index,
                    "origin_stop_id": str(origin),
                    "destination_stop_id": str(destination),
                    "departure_time_bin": str(time_period),
                    "estimated_demand": value,
                    "cell_status": "fixed_zero" if fixed == 0.0 else "fixed_positive",
                    "inference_basis": "fixed_input_or_structural_zero_policy",
                    "inference_score": 0.0,
                    "fixed_value": fixed,
                    "prior_demand": "",
                    "estimated_to_prior_ratio": "",
                    **identifiability_values,
                }
            )
    return rows


def _dispersion(result: GravityEstimationResult, likelihood: str) -> float | None:
    if likelihood == "poisson":
        return None
    names = tuple(result.parameter_names)
    try:
        index = names.index("dispersion")
    except ValueError as error:
        raise ValueError("negative-binomial result has no dispersion parameter.") from error
    value = float(result.physical_parameters[index])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("negative-binomial dispersion must be finite and positive.")
    return value


def _adequacy_json(report: GravityAdequacyReport) -> dict[str, object]:
    return {
        "schema_version": report.schema_version,
        "model_fingerprint": report.model_fingerprint,
        "specification_fingerprint": report.specification_fingerprint,
        "model_specification": report.model_specification,
        "report_fingerprint": report.report_fingerprint,
        "measurements": report.measurements,
        "observed_total": report.observed_total,
        "modeled_total": report.modeled_total,
        "negative_binomial_deviance": report.negative_binomial_deviance,
        "poisson_deviance": report.poisson_deviance,
        "mae": report.mae,
        "rmse": report.rmse,
        "weighted_rmse": report.weighted_rmse,
        "threshold_counts": [list(item) for item in report.threshold_counts],
        "observed_predicted_quantiles": [list(item) for item in report.observed_predicted_quantiles],
        "grouped_summaries": [asdict(item) for item in report.grouped_summaries],
        "journey_correlations": [asdict(item) for item in report.journey_correlations],
        "findings": asdict(report.findings),
    }


def _identifiability_json(
    diagnostic: GravityODIdentifiability | None,
) -> dict[str, object]:
    """Return a compact report summary without duplicating large arrays."""
    if diagnostic is None:
        return {
            "available": False,
            "message": (
                "OD identifiability diagnostics were not available; inference_score "
                "remains only a structural fixed/free indicator."
            ),
        }
    provenance = dict(diagnostic.provenance)
    return {
        "available": True,
        "schema_version": 1,
        "artifact_identity_fingerprint": provenance.get(
            "artifact_identity_fingerprint"
        ),
        "diagnostic_artifact": provenance.get(
            "diagnostic_artifact", provenance.get("diagnostic_artifact_fingerprint")
        ),
        "od_layout_fingerprint": provenance.get("od_layout_fingerprint"),
        "model_fingerprint": provenance.get("model_fingerprint"),
        "effective_hessian_rank": diagnostic.effective_hessian_rank,
        "total_hessian_dimension": diagnostic.total_hessian_dimension,
        "discarded_eigenvalues": diagnostic.discarded_eigenvalues,
        "fixed_cells": diagnostic.fixed_cells,
        "free_cells": diagnostic.free_cells,
        "classification_counts": diagnostic.classification_counts,
        "config": asdict(diagnostic.config),
        "provenance": provenance,
        "count_share_quantiles": diagnostic._quantiles(
            diagnostic.count_information_share
        ),
        "regularization_share_quantiles": diagnostic._quantiles(
            diagnostic.regularization_information_share
        ),
        "auxiliary_share_quantiles": diagnostic._quantiles(
            diagnostic.auxiliary_information_share
        ),
    }


def _executive_messages(
    *,
    result: GravityEstimationResult,
    adequacy: GravityAdequacyReport,
    estimated_od_cells: int,
    fixed_od_cells: int,
    structural_zero_cells: int,
    identifiability: GravityODIdentifiability | None = None,
) -> tuple[str, ...]:
    messages = [
        (
            f"The fit is {result.status} with acceptance={result.acceptance}; "
            f"the optimizer used {result.iterations} iterations."
        ),
        (
            f"Observed flow totals {adequacy.observed_total:.6g}, while modeled flows total "
            f"{adequacy.modeled_total:.6g}; RMSE is {adequacy.rmse:.6g} "
            f"and MAE is {adequacy.mae:.6g}."
        ),
        (
            f"The OD table contains {estimated_od_cells} free cells estimated from the "
            f"observed flows and {fixed_od_cells} fixed cells, including "
            f"{structural_zero_cells} structural zeros."
        ),
    ]
    threshold_three = next(
        (count for threshold, count, _ in adequacy.threshold_counts if threshold == 3.0),
        None,
    )
    if threshold_three is not None:
        messages.append(
            f"{threshold_three} observations have an absolute standardized residual above 3."
        )
    messages.extend(adequacy.findings.messages)
    if identifiability is None:
        messages.append(
            "OD identifiability diagnostics were not available; inference_score "
            "remains only a structural fixed/free indicator."
        )
    else:
        counts = identifiability.classification_counts
        denominator = max(identifiability.free_cells, 1)
        labels = (
            ("count-dominated", "count_dominated"),
            ("assumption-dominated", "assumption_dominated"),
            ("auxiliary-data-dominated", "auxiliary_data_dominated"),
            ("mixed", "mixed_information"),
            ("not locally identifiable", "not_locally_identifiable"),
        )
        percentages = ", ".join(
            f"{label} {100.0 * counts.get(key, 0) / denominator:.1f}%"
            for label, key in labels
        )
        messages.append(
            f"Local OD identifiability across {identifiability.free_cells} free cells: "
            + percentages
            + "."
        )
        messages.append(
            "Identifiability shares are local linearized information diagnostics, "
            "not probabilities, posterior probabilities, or standard errors; all "
            "free OD cells remain jointly coupled through the fitted model."
        )
    messages.append(
        "These are full-data adequacy diagnostics, not an independent holdout validation."
    )
    return tuple(messages)


def _executive_markdown(
    *,
    result: GravityEstimationResult,
    adequacy: GravityAdequacyReport,
    messages: tuple[str, ...],
) -> str:
    specification = result.model_specification or {}
    likelihood = specification.get("likelihood", {})
    likelihood_family = (
        likelihood.get("family", "unknown")
        if isinstance(likelihood, Mapping)
        else "unknown"
    )
    lines = [
        "# Executive summary",
        "",
        "## Main take-home messages",
        "",
    ]
    lines.extend(f"- {message}" for message in messages)
    lines.extend(
        [
            "",
            "## Key figures",
            "",
            "| Figure | Value |",
            "|---|---:|",
            f"| Model name | `{specification.get('model_name', 'unknown')}` |",
            f"| Likelihood family | `{likelihood_family}` |",
            f"| Estimated parameters | {result.raw_parameters.size} |",
            f"| Specification fingerprint | `{result.specification_fingerprint}` |",
            f"| Model fingerprint | `{result.model_fingerprint}` |",
            f"| Fit status | `{result.status}` |",
            f"| Acceptance | `{result.acceptance}` |",
            f"| Objective | {result.objective:.9g} |",
            f"| Measurements | {adequacy.measurements} |",
            f"| Observed total | {adequacy.observed_total:.9g} |",
            f"| Modeled total | {adequacy.modeled_total:.9g} |",
            f"| RMSE | {adequacy.rmse:.9g} |",
            f"| MAE | {adequacy.mae:.9g} |",
            "",
            "The complete OD and measurement-level tables are in the accompanying CSV files.",
            "",
        ]
    )
    return "\n".join(lines)


def _detailed_markdown(
    *,
    result: GravityEstimationResult,
    adequacy: GravityAdequacyReport,
    summary: Mapping[str, object],
) -> str:
    specification = result.model_specification or {}
    likelihood = specification.get("likelihood", {})
    likelihood_family = (
        likelihood.get("family", "unknown")
        if isinstance(likelihood, Mapping)
        else "unknown"
    )
    lines = [
        "# Detailed gravity-fit report",
        "",
        "This report evaluates the fitted model on all calibration observations. It is not a holdout evaluation.",
        "",
        "## Fit and convergence",
        "",
        f"- Status: `{result.status}`; success: `{result.success}`; acceptance: `{result.acceptance}`.",
        f"- Optimizer: `{result.optimizer}`; iterations: `{result.iterations}`; evaluations: `{result.optimizer_evaluations}`.",
        f"- Objective: `{result.objective:.9g}`; scaled gradient infinity norm: `{result.scaled_gradient_inf_norm}`.",
        f"- Model name: `{specification.get('model_name', 'unknown')}`; likelihood family: `{likelihood_family}`; estimated parameters: `{result.raw_parameters.size}`.",
        f"- Specification fingerprint: `{result.specification_fingerprint}`.",
        f"- Model fingerprint: `{result.model_fingerprint}`.",
        "",
        "## Adequacy metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Measurements | {adequacy.measurements} |",
        f"| Observed total | {adequacy.observed_total:.9g} |",
        f"| Modeled total | {adequacy.modeled_total:.9g} |",
        f"| MAE | {adequacy.mae:.9g} |",
        f"| RMSE | {adequacy.rmse:.9g} |",
        f"| Weighted RMSE | {adequacy.weighted_rmse:.9g} |",
        f"| Poisson deviance | {adequacy.poisson_deviance:.9g} |",
        f"| Negative-binomial deviance | {adequacy.negative_binomial_deviance} |",
        "",
        "## Inference indicator",
        "",
        "In `full_od.csv`, `inference_score=0` identifies cells fixed by the demand input or structural-zero policy, while `inference_score=1` identifies free cells estimated using the observed flows and model regularization. This is a structural classification, not a standard error or posterior probability.",
        "",
        "## Findings",
        "",
    ]
    lines.extend(f"- {message}" for message in summary["executive_messages"])
    ident_summary = summary.get("identifiability")
    if isinstance(ident_summary, Mapping):
        lines.extend(["", "## OD identifiability", ""])
        if not ident_summary.get("available", False):
            lines.append(str(ident_summary.get("message", "")))
        else:
            lines.append(
                "The OD information shares are local linearized diagnostics. They "
                "are not probabilities, posterior probabilities, or standard errors."
            )
            lines.append("")
            lines.append(
                "Classification counts: "
                + json.dumps(ident_summary.get("classification_counts", {}), sort_keys=True)
                + "."
            )
    lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `parameters.csv`: fitted raw and physical values for every model parameter.",
            "- `full_od.csv`: every canonical origin/destination/departure-time cell.",
            "- `predicted_measurements.csv`: observed and modeled value for every measurement.",
            "- `residuals.csv`: residual, variance, standardized residual, and relative residual.",
            "- `grouped_residuals.csv`: residual diagnostics by measurement attribute.",
            "- `executive_summary.md`: short decision-oriented summary.",
            "- `report.json`: machine-readable comprehensive summary.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_gravity_detailed_report(
    *,
    result: GravityEstimationResult,
    observations: object,
    predicted_measurements: object,
    od_layout: ODParameterLayout,
    metadata: GravityValidationMetadata | None = None,
    likelihood: GravityLikelihood | str = GravityLikelihood.NEGATIVE_BINOMIAL,
    output_directory: str | Path,
    adequacy_config: GravityAdequacyConfig = GravityAdequacyConfig(),
    provenance: Mapping[str, object] | None = None,
    expected_provenance: Mapping[str, object] | None = None,
    identifiability: GravityODIdentifiability | None = None,
    require_identifiability: bool = False,
    force: bool = False,
) -> GravityDetailedReport:
    """Write CSV, JSON, Markdown, and executive reports for a gravity fit.

    The function consumes persisted fit and prediction vectors. It does not
    activate an assignment operator, rerun an optimizer, or change the
    objective function.
    """
    observed = np.asarray(observations, dtype=np.float64)
    if observed.ndim != 1:
        raise ValueError("observations must be one-dimensional.")
    if not np.all(np.isfinite(observed)):
        raise ValueError("observations must contain finite values.")
    modeled = np.asarray(predicted_measurements, dtype=np.float64)
    if modeled.ndim != 1:
        raise ValueError("predicted_measurements must be one-dimensional.")
    if not np.all(np.isfinite(modeled)):
        raise ValueError("predicted_measurements must contain finite values.")
    if observed.shape != modeled.shape:
        raise ValueError(
            "observations and predicted_measurements have different lengths: "
            f"{observed.size} != {modeled.size}."
        )
    persisted_modeled = np.asarray(result.predicted_measurements, dtype=np.float64)
    _validate_prediction_agreement(persisted_modeled, modeled)
    model_specification, specification_fingerprint = _validated_result_specification(
        result
    )
    supplied = {} if provenance is None else dict(provenance)
    for field, expected in (
        ("model_fingerprint", result.model_fingerprint),
        ("specification_fingerprint", specification_fingerprint),
        ("model_specification", model_specification),
    ):
        if field in supplied and supplied[field] != expected:
            raise ValueError(
                f"report provenance field {field!r} differs from the fitted result."
            )
    selected_metadata = metadata or GravityValidationMetadata(observed.size)
    if not isinstance(selected_metadata, GravityValidationMetadata):
        raise TypeError("metadata must be GravityValidationMetadata or None.")
    if selected_metadata.num_measurements != observed.size:
        raise ValueError(
            "metadata length does not match observations: "
            f"{selected_metadata.num_measurements} != {observed.size}."
        )
    if expected_provenance is not None:
        _validate_expected_provenance(provenance, expected_provenance)
    if require_identifiability and identifiability is None:
        raise ValueError(
            "OD identifiability diagnostics are required but were not supplied."
        )
    if identifiability is not None:
        _validate_identifiability_for_report(
            identifiability,
            result=result,
            od_layout=od_layout,
            expected_provenance=expected_provenance,
        )

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = [output / name for name in _REPORT_FILES if (output / name).exists()]
    if existing and not force:
        raise FileExistsError(
            "report output already exists; choose a new directory or use force=True: "
            + ", ".join(str(path) for path in existing)
        )

    try:
        likelihood_value = str(likelihood.value)
    except AttributeError:
        likelihood_value = str(likelihood)
    adequacy = build_gravity_adequacy_report(
        observations=observed,
        modeled=modeled,
        model_fingerprint=result.model_fingerprint,
        likelihood=likelihood_value,
        dispersion=_dispersion(result, likelihood_value),
        metadata=selected_metadata,
        config=adequacy_config,
        specification=GravityModelSpecification.from_dict(model_specification),
    )

    od_rows = _od_rows(result, od_layout, identifiability)
    measurement_rows = _metadata_rows(observed, selected_metadata)
    residual = adequacy.residual
    standardized = adequacy.standardized_nb_residual
    variance = np.maximum(modeled, np.finfo(np.float64).tiny)
    if likelihood_value == "negative_binomial":
        dispersion = _dispersion(result, likelihood_value)
        assert dispersion is not None
        variance = modeled + modeled**2 / dispersion
    for row, prediction, residual_value, standardized_value, variance_value in zip(
        measurement_rows,
        modeled,
        residual,
        standardized,
        variance,
        strict=True,
    ):
        row.update(
            {
                "modeled": float(prediction),
                "residual": float(residual_value),
                "variance": float(variance_value),
                "standardized_residual": float(standardized_value),
                "absolute_residual": float(abs(residual_value)),
                "relative_residual": (
                    ""
                    if float(row["observed"]) == 0.0
                    else float(residual_value) / float(row["observed"])
                ),
            }
        )

    prediction_fields = ["row_index", *_METADATA_FIELDS, "observed", "modeled"]
    residual_fields = [
        *prediction_fields,
        "residual",
        "variance",
        "standardized_residual",
        "absolute_residual",
        "relative_residual",
    ]
    parameter_rows = _parameter_rows(result)
    _write_csv(output / "parameters.csv", list(_PARAMETER_FIELDS), parameter_rows)
    _write_csv(output / "full_od.csv", list(_OD_FIELDS), od_rows)
    _write_csv(
        output / "predicted_measurements.csv",
        prediction_fields,
        [{field: row[field] for field in prediction_fields} for row in measurement_rows],
    )
    _write_csv(
        output / "residuals.csv",
        residual_fields,
        [{field: row[field] for field in residual_fields} for row in measurement_rows],
    )
    grouped_rows = [asdict(item) for item in adequacy.grouped_summaries]
    _write_csv(
        output / "grouped_residuals.csv",
        list(grouped_rows[0]) if grouped_rows else ["grouping", "label"],
        grouped_rows,
    )

    free_cells = len(od_layout.free_od_indices)
    fixed_cells = len(od_layout.fixed_od_indices)
    structural_zero_cells = len(od_layout.fixed_zero_indices)
    executive_messages = _executive_messages(
        result=result,
        adequacy=adequacy,
        estimated_od_cells=free_cells,
        fixed_od_cells=fixed_cells,
        structural_zero_cells=structural_zero_cells,
        identifiability=identifiability,
    )
    summary: dict[str, object] = {
        "schema_version": GRAVITY_DETAILED_REPORT_SCHEMA_VERSION,
        "status": "completed",
        "model_fingerprint": result.model_fingerprint,
        "specification_fingerprint": specification_fingerprint,
        "model_specification": model_specification,
        "fit": {
            "status": result.status,
            "success": result.success,
            "acceptance": result.acceptance,
            "message": result.message,
            "optimizer": result.optimizer,
            "iterations": result.iterations,
            "evaluations": result.optimizer_evaluations,
            "objective": result.objective,
            "scaled_gradient_inf_norm": result.scaled_gradient_inf_norm,
            "specification_fingerprint": specification_fingerprint,
        },
        "od": {
            "total_cells": od_layout.num_od_total,
            "estimated_free_cells": free_cells,
            "fixed_cells": fixed_cells,
            "structural_zero_cells": structural_zero_cells,
            "inference_indicator": {
                "definition": "0=fixed by input/structural policy; 1=estimated free cell",
                "limitation": "structural classification, not a standard error or posterior probability",
            },
        },
        "adequacy": _adequacy_json(adequacy),
        "identifiability": _identifiability_json(identifiability),
        "executive_messages": list(executive_messages),
        "provenance": {
            **supplied,
            "model_fingerprint": result.model_fingerprint,
            "specification_fingerprint": specification_fingerprint,
            "model_specification": model_specification,
        },
        "files": {name: str(output / name) for name in _REPORT_FILES},
    }
    report_hash = hashlib.sha256(
        json.dumps(summary, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    summary["report_fingerprint"] = report_hash
    _write_json(output / "report.json", summary)
    _write_text(
        output / "executive_summary.md",
        _executive_markdown(result=result, adequacy=adequacy, messages=executive_messages),
    )
    _write_text(
        output / "report.md",
        _detailed_markdown(
            result=result,
            adequacy=adequacy,
            summary={
                "executive_messages": executive_messages,
                "identifiability": _identifiability_json(identifiability),
            },
        ),
    )
    return GravityDetailedReport(
        GRAVITY_DETAILED_REPORT_SCHEMA_VERSION,
        report_hash,
        output,
        {name: output / name for name in _REPORT_FILES},
        adequacy,
        free_cells,
        fixed_cells,
        structural_zero_cells,
        executive_messages,
        identifiability,
    )


def write_persisted_gravity_detailed_report(
    *,
    fit_manifest: Mapping[str, object],
    validation_manifest: Mapping[str, object],
    observations: object,
    od_layout: ODParameterLayout,
    metadata: GravityValidationMetadata | None = None,
    likelihood: GravityLikelihood | str = GravityLikelihood.NEGATIVE_BINOMIAL,
    output_directory: str | Path,
    adequacy_config: GravityAdequacyConfig = GravityAdequacyConfig(),
    identifiability: GravityODIdentifiability | None = None,
    require_identifiability: bool = False,
    force: bool = False,
) -> GravityDetailedReport:
    """Generate a detailed report using only persisted fit/validation data.

    This entry point deliberately does not load a case context, scenario,
    routing operator, or cache.  All identity and prediction checks complete
    before the lower-level writer is allowed to create the output directory.
    """
    if not isinstance(fit_manifest, Mapping):
        raise TypeError("fit_manifest must be a mapping.")
    if not isinstance(validation_manifest, Mapping):
        raise TypeError("validation_manifest must be a mapping.")
    if not isinstance(od_layout, ODParameterLayout):
        raise TypeError("od_layout must be an ODParameterLayout.")
    _manifest_status(fit_manifest, "fit")
    _manifest_status(validation_manifest, "validation")
    raw_result = fit_manifest.get("result")
    if not isinstance(raw_result, Mapping):
        raise ValueError("fit_manifest['result'] must be a mapping.")
    result = GravityEstimationResult.from_dict(raw_result)
    _validate_restored_result(result, od_layout)
    inputs = PersistedGravityReportInputs(
        fit_manifest=fit_manifest,
        validation_manifest=validation_manifest,
        observations=np.asarray(observations),
        od_layout=od_layout,
        metadata=metadata,
        likelihood=likelihood,
    )
    persisted_provenance = validate_gravity_report_provenance(
        fit_manifest=fit_manifest,
        validation_manifest=validation_manifest,
        od_layout=od_layout,
    )
    report_provenance = {
        "source": "persisted_fit_validation",
        **persisted_provenance,
    }
    return write_gravity_detailed_report(
        result=result,
        observations=inputs.observations,
        predicted_measurements=inputs.predicted_measurements,
        od_layout=od_layout,
        metadata=inputs.metadata,
        likelihood=likelihood,
        output_directory=output_directory,
        adequacy_config=adequacy_config,
        provenance=report_provenance,
        expected_provenance=persisted_provenance,
        identifiability=identifiability,
        require_identifiability=require_identifiability,
        force=force,
    )
