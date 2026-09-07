"""Local information diagnostics for gravity-model OD cells.

The routines in this module are deliberately post-estimation diagnostics.  They
evaluate derivatives at a supplied fitted result and never alter the gravity
objective, optimizer, routing operator, or fitted parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.inference.block_coordinate._canonical import fingerprint

from .demand import generate_gravity_demand
from .estimator import GravityEstimationResult
from .objective import (
    GravityLikelihood,
    GravityObjectiveProblem,
    predict_gravity_measurements,
)


_IDENTIFIABILITY_SCHEMA_VERSION = 1
_ARRAY_NAMES = (
    "count_information_share",
    "regularization_information_share",
    "auxiliary_information_share",
    "local_information_variance",
    "classification",
    "diagnostic_reason",
)
_CANONICAL_PROVENANCE_FIELDS = (
    "artifact_identity_fingerprint",
    "assignment_fingerprint",
    "binding_fingerprint",
    "canonical_index_fingerprint",
    "compact_layout_fingerprint",
    "gravity_features_fingerprint",
    "mapping_fingerprint",
    "od_layout_fingerprint",
    "package_revision",
    "model_fingerprint",
)


def _immutable_array(value: object, *, dtype: np.dtype | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != 1:
        raise ValueError("identifiability arrays must be one-dimensional.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class GravityIdentifiabilityConfig:
    """Chunking, eigensolver, and classification settings."""

    measurement_chunk_size: int = 4096
    od_chunk_size: int = 4096
    eigenvalue_relative_tolerance: float = 1.0e-10
    eigenvalue_absolute_tolerance: float = 1.0e-12
    count_dominated_threshold: float = 0.8
    assumption_dominated_threshold: float = 0.2

    def __post_init__(self) -> None:
        for name in ("measurement_chunk_size", "od_chunk_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be a positive integer.")
            if int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
            object.__setattr__(self, name, int(value))
        for name in (
            "eigenvalue_relative_tolerance",
            "eigenvalue_absolute_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        for name in ("count_dominated_threshold", "assumption_dominated_threshold"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class GravityODIdentifiability:
    """Per-cell local information decomposition aligned to full OD order.

    ``NaN`` is the in-memory representation of JSON ``null`` for fixed cells;
    it permits the numerical arrays to remain ordinary, non-object NumPy
    arrays and therefore safe to persist in NPZ without pickle.
    """

    count_information_share: np.ndarray
    regularization_information_share: np.ndarray
    auxiliary_information_share: np.ndarray | None
    local_information_variance: np.ndarray
    classification: np.ndarray
    diagnostic_reason: np.ndarray
    effective_hessian_rank: int
    total_hessian_dimension: int
    config: GravityIdentifiabilityConfig
    provenance: Mapping[str, object]
    discarded_eigenvalues: int = 0
    fixed_cells: int = 0
    free_cells: int = 0
    clipping_required: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.config, GravityIdentifiabilityConfig):
            raise TypeError("config must be GravityIdentifiabilityConfig.")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping.")
        if self.total_hessian_dimension < 0:
            raise ValueError("total_hessian_dimension must be non-negative.")
        if not 0 <= self.effective_hessian_rank <= self.total_hessian_dimension:
            raise ValueError("effective_hessian_rank is outside the Hessian dimension.")
        size = np.asarray(self.classification).size
        classification = _immutable_array(self.classification, dtype=np.dtype("U"))
        reason = _immutable_array(self.diagnostic_reason, dtype=np.dtype("U"))
        if classification.size != size or reason.size != size:
            raise ValueError("classification and diagnostic_reason lengths must agree.")
        object.__setattr__(self, "classification", classification)
        object.__setattr__(self, "diagnostic_reason", reason)

        fixed = classification == "fixed_by_policy"
        not_identifiable = classification == "not_locally_identifiable"
        nullable = fixed | not_identifiable
        for name in (
            "count_information_share",
            "regularization_information_share",
            "local_information_variance",
        ):
            array = _immutable_array(getattr(self, name), dtype=np.dtype(np.float64))
            if array.size != size:
                raise ValueError(f"{name} must have one value per full OD cell.")
            if np.any(~np.isfinite(array[~nullable])):
                raise ValueError(f"{name} must be finite for identifiable free OD cells.")
            if np.any(~np.isnan(array[nullable])):
                raise ValueError(
                    f"{name} must be null (NaN) for fixed or non-identifiable OD cells."
                )
            object.__setattr__(self, name, array)
        if self.auxiliary_information_share is not None:
            auxiliary = _immutable_array(
                self.auxiliary_information_share, dtype=np.dtype(np.float64)
            )
            if auxiliary.size != size:
                raise ValueError(
                    "auxiliary_information_share must have one value per full OD cell."
                )
            if np.any(~np.isfinite(auxiliary[~nullable])):
                raise ValueError(
                    "auxiliary_information_share must be finite for identifiable free OD cells."
                )
            if np.any(~np.isnan(auxiliary[nullable])):
                raise ValueError(
                    "auxiliary_information_share must be null for fixed or non-identifiable OD cells."
                )
            object.__setattr__(self, "auxiliary_information_share", auxiliary)
        expected_fixed = int(np.count_nonzero(fixed))
        if self.fixed_cells not in (0, expected_fixed):
            raise ValueError("fixed_cells does not match classification.")
        expected_free = size - expected_fixed
        if self.free_cells not in (0, expected_free):
            raise ValueError("free_cells does not match classification.")
        object.__setattr__(self, "fixed_cells", expected_fixed)
        object.__setattr__(self, "free_cells", expected_free)
        object.__setattr__(self, "discarded_eigenvalues", int(self.discarded_eigenvalues))

    @property
    def num_cells(self) -> int:
        return int(self.classification.size)

    @property
    def classification_counts(self) -> dict[str, int]:
        labels, counts = np.unique(self.classification, return_counts=True)
        return {str(label): int(count) for label, count in zip(labels, counts, strict=True)}

    @staticmethod
    def _quantiles(values: np.ndarray | None) -> dict[str, float] | None:
        if values is None:
            return None
        finite = np.asarray(values)[np.isfinite(values)]
        if finite.size == 0:
            return None
        quantiles = np.quantile(finite, (0.0, 0.25, 0.5, 0.75, 1.0))
        return {
            "minimum": float(quantiles[0]),
            "q25": float(quantiles[1]),
            "median": float(quantiles[2]),
            "q75": float(quantiles[3]),
            "maximum": float(quantiles[4]),
        }

    def to_dict(self) -> dict[str, object]:
        """Return JSON metadata; numerical arrays are referenced by NPZ names."""
        provenance = dict(self.provenance)
        return {
            "schema_version": _IDENTIFIABILITY_SCHEMA_VERSION,
            "artifact_type": "gravity_od_identifiability",
            "diagnostic_artifact_fingerprint": provenance.get(
                "diagnostic_artifact_fingerprint"
            ),
            **{
                name: provenance.get(name)
                for name in _CANONICAL_PROVENANCE_FIELDS
                if name in provenance
            },
            "effective_hessian_rank": self.effective_hessian_rank,
            "total_hessian_dimension": self.total_hessian_dimension,
            "discarded_eigenvalues": self.discarded_eigenvalues,
            "fixed_cells": self.fixed_cells,
            "free_cells": self.free_cells,
            "clipping_required": self.clipping_required,
            "classification_counts": self.classification_counts,
            "config": asdict(self.config),
            "provenance": provenance,
            "arrays": {name: name for name in _ARRAY_NAMES},
            "count_share_quantiles": self._quantiles(self.count_information_share),
            "regularization_share_quantiles": self._quantiles(
                self.regularization_information_share
            ),
            "auxiliary_share_quantiles": self._quantiles(
                self.auxiliary_information_share
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        arrays: Mapping[str, object] | None = None,
    ) -> "GravityODIdentifiability":
        """Restore metadata and numerical arrays, checking all dimensions."""
        if not isinstance(payload, Mapping):
            raise TypeError("identifiability payload must be a mapping.")
        if payload.get("schema_version") != _IDENTIFIABILITY_SCHEMA_VERSION:
            raise ValueError("unsupported gravity identifiability schema version.")
        if payload.get("artifact_type") != "gravity_od_identifiability":
            raise ValueError("unexpected gravity identifiability artifact type.")
        array_map = payload.get("arrays", {})
        if not isinstance(array_map, Mapping):
            raise ValueError("identifiability arrays metadata must be a mapping.")
        supplied = {} if arrays is None else dict(arrays)

        def array(name: str) -> object:
            if name in supplied:
                return supplied[name]
            value = array_map.get(name, _MISSING)
            if value is _MISSING or isinstance(value, str):
                raise ValueError(f"identifiability array {name!r} is missing.")
            return value

        config_payload = payload.get("config")
        if not isinstance(config_payload, Mapping):
            raise ValueError("identifiability config is missing or invalid.")
        provenance = payload.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise ValueError("identifiability provenance must be a mapping.")
        for name in _CANONICAL_PROVENANCE_FIELDS:
            top_level = payload.get(name, _MISSING)
            nested = provenance.get(name, _MISSING)
            if top_level is not _MISSING and nested is not _MISSING and top_level != nested:
                raise ValueError(f"identifiability provenance mismatch for {name!r}.")
        auxiliary_value = array("auxiliary_information_share")
        if auxiliary_value is None:
            auxiliary = None
        else:
            auxiliary = np.asarray(auxiliary_value, dtype=np.float64)
        restored = cls(
            count_information_share=np.asarray(
                array("count_information_share"), dtype=np.float64
            ),
            regularization_information_share=np.asarray(
                array("regularization_information_share"), dtype=np.float64
            ),
            auxiliary_information_share=auxiliary,
            local_information_variance=np.asarray(
                array("local_information_variance"), dtype=np.float64
            ),
            classification=np.asarray(array("classification")),
            diagnostic_reason=np.asarray(array("diagnostic_reason")),
            effective_hessian_rank=int(payload.get("effective_hessian_rank", 0)),
            total_hessian_dimension=int(payload.get("total_hessian_dimension", 0)),
            config=GravityIdentifiabilityConfig(**dict(config_payload)),
            provenance=dict(provenance),
            discarded_eigenvalues=int(payload.get("discarded_eigenvalues", 0)),
            fixed_cells=int(payload.get("fixed_cells", 0)),
            free_cells=int(payload.get("free_cells", 0)),
            clipping_required=bool(payload.get("clipping_required", False)),
        )
        persisted_counts = payload.get("classification_counts")
        if persisted_counts is not None:
            if not isinstance(persisted_counts, Mapping) or {
                str(key): int(value) for key, value in persisted_counts.items()
            } != restored.classification_counts:
                raise ValueError("identifiability classification counts do not match the arrays.")
        return restored


class _Missing:
    pass


_MISSING = _Missing()


def _validated_provenance(
    problem: GravityObjectiveProblem, result: GravityEstimationResult
) -> dict[str, object]:
    operator = problem.operator
    source = getattr(operator, "operator", None)
    canonical_index = getattr(operator, "canonical_index", None)
    if canonical_index is None and source is not None:
        canonical_index = getattr(source, "canonical_index", None)
    compact = str(
        getattr(operator, "compact_layout_fingerprint", "")
        or getattr(canonical_index, "source_compact_layout_fingerprint", "")
        or problem.features.od_layout_fingerprint
    )
    features = str(problem.features.fingerprint)
    assignment = str(
        getattr(operator, "assignment_fingerprint", "")
        or getattr(canonical_index, "timetable_fingerprint", "")
    )
    mapping = str(
        getattr(operator, "mapping_fingerprint", "")
        or getattr(canonical_index, "measurement_mapping_fingerprint", "")
    )
    artifact = str(
        getattr(operator, "artifact_fingerprint", "")
        or result.direct_operator_artifact_fingerprint
        or compact
    )
    package_revision = str(
        getattr(operator, "package_version", "")
        or getattr(source, "package_version", "")
        or "unknown"
    )
    canonical_fingerprint = str(
        getattr(canonical_index, "artifact_fingerprint", "") or compact
    )
    od_layout_fingerprint = str(
        getattr(operator, "od_layout_fingerprint", "")
        or getattr(canonical_index, "source_od_layout_fingerprint", "")
        or problem.features.od_layout_fingerprint
    )
    binding_fingerprint = str(
        getattr(canonical_index, "binding_fingerprint", "") or compact
    )
    return {
        "artifact_identity_fingerprint": artifact,
        "assignment_fingerprint": assignment,
        "binding_fingerprint": binding_fingerprint,
        "canonical_index_fingerprint": canonical_fingerprint,
        "compact_layout_fingerprint": compact,
        "gravity_features_fingerprint": features,
        "mapping_fingerprint": mapping,
        "od_layout_fingerprint": od_layout_fingerprint,
        "package_revision": package_revision,
        "model_fingerprint": result.model_fingerprint,
    }


def _vector_fingerprint(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _validate_inputs(
    problem: GravityObjectiveProblem, result: GravityEstimationResult
) -> np.ndarray:
    raw = np.asarray(result.raw_parameters, dtype=np.float64)
    if raw.ndim != 1 or raw.size != problem.parameter_layout.size:
        raise ValueError(
            "fitted parameter dimension does not match the gravity parameter layout."
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError("fitted raw parameters must be finite.")
    if not isinstance(result.model_fingerprint, str) or not result.model_fingerprint:
        raise ValueError("fitted result must contain a model fingerprint.")
    if problem.features.num_cells != problem.operator.num_free_od:
        raise ValueError("gravity features and operator free-cell dimensions differ.")
    compact = getattr(problem.operator, "compact_layout_fingerprint", None)
    if compact is not None and str(compact) != problem.features.od_layout_fingerprint:
        raise ValueError("gravity features and operator compact-layout fingerprints differ.")
    if (
        result.feature_cache_fingerprint
        and result.feature_cache_fingerprint != problem.features.fingerprint
    ):
        raise ValueError("fitted result feature fingerprint does not match the problem.")
    operator_artifact = getattr(problem.operator, "artifact_fingerprint", None)
    if (
        operator_artifact
        and result.direct_operator_artifact_fingerprint
        and str(operator_artifact) != result.direct_operator_artifact_fingerprint
    ):
        raise ValueError("fitted result operator artifact fingerprint does not match the problem.")
    canonical = np.asarray(problem.features.canonical_od_index, dtype=np.int64)
    full = np.asarray(result.full_od_demand, dtype=np.float64)
    if full.ndim != 1 or not np.all(np.isfinite(full)):
        raise ValueError("fitted full OD demand must be a finite one-dimensional vector.")
    if canonical.shape != (problem.operator.num_free_od,):
        raise ValueError("canonical OD indices do not match the operator free dimension.")
    if np.any(canonical < 0) or np.any(canonical >= full.size):
        raise ValueError("canonical OD indices fall outside the fitted full OD vector.")
    if np.unique(canonical).size != canonical.size:
        raise ValueError("canonical OD indices must be unique.")
    modeled, demand = predict_gravity_measurements(raw, problem=problem)
    modeled_np = np.asarray(modeled, dtype=np.float64)
    if modeled_np.ndim != 1 or not np.all(np.isfinite(modeled_np)):
        raise ValueError("modeled measurements must be finite and one-dimensional.")
    persisted_predictions = np.asarray(result.predicted_measurements, dtype=np.float64)
    if persisted_predictions.ndim != 1 or not np.all(np.isfinite(persisted_predictions)):
        raise ValueError("fitted predicted measurements must be finite and one-dimensional.")
    persisted_demand = np.asarray(result.free_od_demand, dtype=np.float64)
    demand_np = np.asarray(demand, dtype=np.float64)
    if persisted_demand.shape != demand_np.shape or not np.all(np.isfinite(persisted_demand)):
        raise ValueError("fitted free OD demand does not match the gravity feature dimension.")
    tolerance = 5.0e-6 if problem.features.dtype == np.dtype(np.float32) else 1.0e-8
    try:
        np.testing.assert_allclose(
            modeled_np,
            persisted_predictions,
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            demand_np, persisted_demand, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            full[canonical],
            persisted_demand,
            rtol=tolerance,
            atol=tolerance,
        )
    except AssertionError as error:
        raise ValueError(
            "fitted predictions or free OD demand do not match the gravity problem."
        ) from error
    return raw


def _information_hessian(
    *,
    raw: np.ndarray,
    problem: GravityObjectiveProblem,
    config: GravityIdentifiabilityConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    parameter_count = raw.size
    raw_jax = jnp.asarray(raw)
    modeled, _ = predict_gravity_measurements(raw_jax, problem=problem)
    modeled_np = np.asarray(modeled, dtype=np.float64)
    mask = np.asarray(problem.calibration_mask, dtype=bool)
    if problem.likelihood is GravityLikelihood.POISSON:
        weights = 1.0 / np.maximum(modeled_np, float(problem.mean_floor))
    else:
        dispersion = float(problem.parameter_layout.transform(raw_jax).dispersion)
        if not np.isfinite(dispersion) or dispersion <= 0.0:
            raise ValueError("negative-binomial dispersion must be finite and positive.")
        weights = dispersion / (
            np.maximum(modeled_np, float(problem.mean_floor))
            * (dispersion + np.maximum(modeled_np, float(problem.mean_floor)))
        )

    h_counts = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    calibration_indices = np.flatnonzero(mask)
    for start in range(0, calibration_indices.size, config.measurement_chunk_size):
        indices = calibration_indices[start : start + config.measurement_chunk_size]
        index_array = jnp.asarray(indices, dtype=jnp.int32)

        def measurement_chunk(value: jax.Array) -> jax.Array:
            prediction, _ = predict_gravity_measurements(value, problem=problem)
            return prediction[index_array]

        jacobian = np.asarray(jax.jacfwd(measurement_chunk)(raw_jax), dtype=np.float64)
        if jacobian.ndim != 2 or jacobian.shape[1] != parameter_count:
            raise ValueError("measurement Jacobian chunk has an unexpected shape.")
        h_counts += jacobian.T @ (weights[indices, None] * jacobian)

    def regularization_function(value: jax.Array) -> jax.Array:
        return problem.parameter_layout.regularization(value)

    h_regularization = np.asarray(jax.hessian(regularization_function)(raw_jax), dtype=np.float64)

    h_auxiliary: np.ndarray | None = None
    channels = problem.auxiliary_observations.channels
    if channels:

        def auxiliary_negative_log_likelihood(value: jax.Array) -> jax.Array:
            demand = generate_gravity_demand(
                value,
                features=problem.features,
                parameter_layout=problem.parameter_layout,
            ).demand
            terms = tuple(
                channel.log_likelihood(
                    prediction=channel.predict(demand), raw_parameters=value
                )
                for channel in channels
            )
            return -jnp.sum(jnp.stack(terms))

        h_auxiliary = np.asarray(
            jax.hessian(auxiliary_negative_log_likelihood)(raw_jax), dtype=np.float64
        )
    return h_counts, h_regularization, h_auxiliary


def compute_gravity_od_identifiability(
    *,
    problem: GravityObjectiveProblem,
    result: GravityEstimationResult,
    config: GravityIdentifiabilityConfig = GravityIdentifiabilityConfig(),
) -> GravityODIdentifiability:
    """Compute local count/assumption information shares for every OD cell."""
    if not isinstance(problem, GravityObjectiveProblem):
        raise TypeError("problem must be GravityObjectiveProblem.")
    if not isinstance(result, GravityEstimationResult):
        raise TypeError("result must be GravityEstimationResult.")
    if not isinstance(config, GravityIdentifiabilityConfig):
        raise TypeError("config must be GravityIdentifiabilityConfig.")
    raw = _validate_inputs(problem, result)
    h_counts, h_regularization, h_auxiliary = _information_hessian(
        raw=raw, problem=problem, config=config
    )
    for name, matrix in (
        ("count information", h_counts),
        ("regularization information", h_regularization),
    ):
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} Hessian contribution is not finite.")
    if h_auxiliary is not None and not np.all(np.isfinite(h_auxiliary)):
        raise ValueError("auxiliary information Hessian contribution is not finite.")
    h_total = h_counts + h_regularization
    if h_auxiliary is not None:
        h_total = h_total + h_auxiliary
    h_counts = 0.5 * (h_counts + h_counts.T)
    h_regularization = 0.5 * (h_regularization + h_regularization.T)
    if h_auxiliary is not None:
        h_auxiliary = 0.5 * (h_auxiliary + h_auxiliary.T)
    h_total = 0.5 * (h_total + h_total.T)
    eigenvalues, eigenvectors = np.linalg.eigh(h_total)
    spectral_scale = float(np.max(np.abs(eigenvalues), initial=0.0))
    cutoff = max(
        config.eigenvalue_absolute_tolerance,
        config.eigenvalue_relative_tolerance * spectral_scale,
    )
    keep = eigenvalues > cutoff
    effective_rank = int(np.count_nonzero(keep))
    discarded = int(eigenvalues.size - effective_rank)
    if effective_rank:
        h_pseudoinverse = (
            eigenvectors[:, keep] * (1.0 / eigenvalues[keep])
        ) @ eigenvectors[:, keep].T
    else:
        h_pseudoinverse = np.zeros_like(h_total)

    full_demand = np.asarray(result.full_od_demand, dtype=np.float64)
    free_full_indices = np.asarray(problem.features.canonical_od_index, dtype=np.int64)
    fixed_mask = np.ones(full_demand.size, dtype=bool)
    fixed_mask[free_full_indices] = False
    count_share = np.full(full_demand.size, np.nan, dtype=np.float64)
    regularization_share = np.full(full_demand.size, np.nan, dtype=np.float64)
    auxiliary_share = (
        None
        if h_auxiliary is None
        else np.full(full_demand.size, np.nan, dtype=np.float64)
    )
    variance = np.full(full_demand.size, np.nan, dtype=np.float64)
    classification = np.full(full_demand.size, "fixed_by_policy", dtype="U32")
    reasons = np.full(
        full_demand.size,
        "fixed_input_or_structural_zero_policy",
        dtype="U64",
    )
    def demand_jacobian_function(value: jax.Array) -> jax.Array:
        return generate_gravity_demand(
            value,
            features=problem.features,
            parameter_layout=problem.parameter_layout,
        ).demand
    variance_cutoff = np.finfo(np.float64).eps
    clipping_required = False
    for start in range(0, free_full_indices.size, config.od_chunk_size):
        stop = start + config.od_chunk_size
        free_offsets = np.arange(start, min(stop, free_full_indices.size), dtype=np.int64)
        offset_array = jnp.asarray(free_offsets, dtype=jnp.int32)

        def demand_chunk(value: jax.Array) -> jax.Array:
            return demand_jacobian_function(value)[offset_array]

        jacobian = np.asarray(jax.jacfwd(demand_chunk)(jnp.asarray(raw)), dtype=np.float64)
        if jacobian.ndim != 2 or jacobian.shape[1] != raw.size:
            raise ValueError("OD-demand Jacobian chunk has an unexpected shape.")
        directions = jacobian @ h_pseudoinverse.T
        local_variance = np.einsum("ij,ij->i", directions, jacobian)
        for local, full_index in enumerate(free_full_indices[free_offsets]):
            value = float(local_variance[local])
            if not np.isfinite(value) or value <= variance_cutoff:
                classification[full_index] = "not_locally_identifiable"
                reasons[full_index] = "singular_or_weak_local_information"
                continue
            variance[full_index] = value
            count_info = float(directions[local] @ h_counts @ directions[local])
            regularization_info = float(
                directions[local] @ h_regularization @ directions[local]
            )
            aux_info = (
                None
                if h_auxiliary is None
                else float(directions[local] @ h_auxiliary @ directions[local])
            )
            shares = [count_info / value, regularization_info / value]
            if aux_info is not None:
                shares.append(aux_info / value)
            if any(not np.isfinite(item) for item in shares):
                classification[full_index] = "not_locally_identifiable"
                reasons[full_index] = "nonfinite_information_decomposition"
                variance[full_index] = np.nan
                continue
            raw_count, raw_regularization = shares[:2]
            raw_auxiliary = None if aux_info is None else shares[2]
            if raw_auxiliary is not None:
                auxiliary_share[full_index] = raw_auxiliary
            count_share[full_index] = raw_count
            regularization_share[full_index] = raw_regularization
            out_of_range = False
            for position, item in enumerate(shares):
                clipped = float(np.clip(item, 0.0, 1.0))
                if abs(clipped - item) <= 1.0e-6:
                    shares[position] = clipped
                else:
                    out_of_range = True
                    clipping_required = True
            count_share[full_index] = shares[0]
            regularization_share[full_index] = shares[1]
            if auxiliary_share is not None and raw_auxiliary is not None:
                auxiliary_share[full_index] = shares[2]
            if out_of_range:
                reasons[full_index] = "information_decomposition_out_of_range"
                classification[full_index] = "mixed_information"
            elif (
                auxiliary_share is not None
                and auxiliary_share[full_index] >= count_share[full_index]
                and auxiliary_share[full_index] >= regularization_share[full_index]
            ):
                classification[full_index] = "auxiliary_data_dominated"
                reasons[full_index] = "auxiliary_observed_data_information_dominates"
            elif count_share[full_index] >= config.count_dominated_threshold:
                classification[full_index] = "count_dominated"
                reasons[full_index] = "count_information_dominates_local_curvature"
            elif regularization_share[full_index] >= config.assumption_dominated_threshold:
                classification[full_index] = "assumption_dominated"
                reasons[full_index] = "regularization_or_prior_dominates_local_curvature"
            else:
                classification[full_index] = "mixed_information"
                reasons[full_index] = "counts_and_assumptions_both_contribute"

    provenance = _validated_provenance(problem, result)
    provenance["raw_parameters_fingerprint"] = _vector_fingerprint(raw)
    provenance["predicted_measurements_fingerprint"] = _vector_fingerprint(
        result.predicted_measurements
    )
    provenance["diagnostic_artifact_fingerprint"] = fingerprint(
        {
            "model_fingerprint": result.model_fingerprint,
            "hessian_rank": effective_rank,
            "hessian_dimension": raw.size,
            "config": asdict(config),
        }
    )
    return GravityODIdentifiability(
        count_information_share=count_share,
        regularization_information_share=regularization_share,
        auxiliary_information_share=auxiliary_share,
        local_information_variance=variance,
        classification=classification,
        diagnostic_reason=reasons,
        effective_hessian_rank=effective_rank,
        total_hessian_dimension=raw.size,
        config=config,
        provenance=provenance,
        discarded_eigenvalues=discarded,
        fixed_cells=int(np.count_nonzero(fixed_mask)),
        free_cells=int(free_full_indices.size),
        clipping_required=clipping_required,
    )


def _array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _persisted_array(value: np.ndarray | None) -> np.ndarray:
    """Return the on-disk representation, including an absent auxiliary array."""
    if value is None:
        return np.asarray([], dtype=np.float64)
    return np.asarray(value)


def write_gravity_od_identifiability(
    diagnostic: GravityODIdentifiability,
    output_directory: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Persist a diagnostic as JSON metadata plus a portable NPZ payload."""
    if not isinstance(diagnostic, GravityODIdentifiability):
        raise TypeError("diagnostic must be GravityODIdentifiability.")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "identifiability.json"
    arrays_path = output / "identifiability.npz"
    if not force and (metadata_path.exists() or arrays_path.exists()):
        raise FileExistsError(
            "identifiability output already exists; choose another directory or use force=True."
        )
    payload = diagnostic.to_dict()
    payload["array_digests"] = {
        name: _array_digest(_persisted_array(getattr(diagnostic, name)))
        for name in _ARRAY_NAMES
    }
    payload["arrays_file"] = arrays_path.name
    with NamedTemporaryFile(
        mode="wb", suffix=".npz", prefix=".identifiability-", dir=output, delete=False
    ) as temporary:
        np.savez_compressed(
            temporary,
            count_information_share=np.asarray(diagnostic.count_information_share),
            regularization_information_share=np.asarray(
                diagnostic.regularization_information_share
            ),
            auxiliary_information_share=_persisted_array(
                diagnostic.auxiliary_information_share
            ),
            local_information_variance=np.asarray(diagnostic.local_information_variance),
            classification=np.asarray(diagnostic.classification),
            diagnostic_reason=np.asarray(diagnostic.diagnostic_reason),
        )
        temporary_path = Path(temporary.name)
    temporary_path.replace(arrays_path)
    metadata_path_tmp = metadata_path.with_name(f".{metadata_path.name}.tmp")
    metadata_path_tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metadata_path_tmp.replace(metadata_path)
    return metadata_path


def read_gravity_od_identifiability(
    path: str | Path,
    *,
    expected_provenance: Mapping[str, object] | None = None,
) -> GravityODIdentifiability:
    """Read and verify JSON/NPZ identifiability persistence."""
    source = Path(path).expanduser().resolve()
    metadata_path = source / "identifiability.json" if source.is_dir() else source
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != _IDENTIFIABILITY_SCHEMA_VERSION:
        raise ValueError("unsupported gravity identifiability schema version.")
    if payload.get("artifact_type") != "gravity_od_identifiability":
        raise ValueError("unexpected gravity identifiability artifact type.")
    array_map = payload.get("arrays")
    if not isinstance(array_map, Mapping) or any(
        array_map.get(name) != name for name in _ARRAY_NAMES
    ):
        raise ValueError("identifiability array mapping is missing or invalid.")
    arrays_path = metadata_path.parent / str(payload.get("arrays_file", "identifiability.npz"))
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in _ARRAY_NAMES}
    if arrays["auxiliary_information_share"].size == 0:
        arrays["auxiliary_information_share"] = None
    for name in _ARRAY_NAMES:
        digests = payload.get("array_digests", {})
        if not isinstance(digests, Mapping):
            raise ValueError("identifiability array digests metadata is invalid.")
        expected_digest = digests.get(name)
        if not isinstance(expected_digest, str):
            raise ValueError(f"identifiability array digest is missing for {name!r}.")
        if expected_digest != _array_digest(_persisted_array(arrays[name])):
            raise ValueError(f"identifiability array fingerprint mismatch for {name!r}.")
    diagnostic = GravityODIdentifiability.from_dict(payload, arrays=arrays)
    for name in _CANONICAL_PROVENANCE_FIELDS:
        top_level = payload.get(name, _MISSING)
        nested = diagnostic.provenance.get(name, _MISSING)
        if top_level is not _MISSING and nested is not _MISSING and top_level != nested:
            raise ValueError(f"identifiability provenance mismatch for {name!r}.")
        if expected_provenance is not None:
            expected = expected_provenance.get(name, _MISSING)
            if expected is not _MISSING and (
                nested is _MISSING or expected != nested
            ):
                raise ValueError(f"identifiability provenance mismatch for {name!r}.")
    return diagnostic


__all__ = [
    "GravityIdentifiabilityConfig",
    "GravityODIdentifiability",
    "compute_gravity_od_identifiability",
    "read_gravity_od_identifiability",
    "write_gravity_od_identifiability",
]
