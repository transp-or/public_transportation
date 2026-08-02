"""Deterministic grouped holdout re-estimation and predictive validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, cast

import numpy as np
from scipy.special import gammaln, xlogy  # type: ignore[import-untyped]

from public_transportation.inference.block_coordinate._canonical import (
    canonical_json,
    fingerprint,
)
from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
)

from .estimator import (
    GravityEstimationResult,
    GravityEstimatorConfig,
    GravityExecutionPolicy,
    estimate_gravity_model,
)
from .objective import GravityLikelihood, GravityObjectiveProblem
from .validation import GravityValidationMetadata

GravityHoldoutUnit = Literal[
    "vehicle_journey",
    "stop_time_series",
    "line",
    "direction",
    "time_block",
    "explicit_group",
]

_STRATIFICATION_FIELDS = {
    "measurement_type",
    "line",
    "direction",
    "stop",
    "time_period",
    "origin_zone",
    "destination_zone",
}


def _immutable_bool(value: object, *, name: str, size: int) -> np.ndarray:
    array = np.array(value, dtype=bool, copy=True)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    array.setflags(write=False)
    return array


def _label_key(value: object) -> str:
    scalar = value.item() if isinstance(value, np.generic) else value
    return canonical_json({"type": type(scalar).__name__, "value": scalar})


@dataclass(frozen=True, slots=True)
class GravityHoldoutSplitConfig:
    unit: GravityHoldoutUnit
    holdout_fraction: float = 0.2
    seed: int = 0
    stratify_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.unit not in (
            "vehicle_journey",
            "stop_time_series",
            "line",
            "direction",
            "time_block",
            "explicit_group",
        ):
            raise ValueError("unsupported grouped holdout unit.")
        if not 0 < self.holdout_fraction < 1:
            raise ValueError("holdout_fraction must lie strictly between zero and one.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        if len(set(self.stratify_by)) != len(self.stratify_by):
            raise ValueError("stratify_by entries must be unique.")
        unknown = set(self.stratify_by) - _STRATIFICATION_FIELDS
        if unknown:
            raise ValueError(f"unsupported stratification fields: {sorted(unknown)}.")


@dataclass(frozen=True, slots=True)
class GravityHoldoutSplit:
    schema_version: int
    measurement_identity: str
    metadata_fingerprint: str
    config: GravityHoldoutSplitConfig
    num_measurements: int
    num_groups: int
    calibration_mask: np.ndarray
    holdout_mask: np.ndarray
    calibration_group_labels: tuple[str, ...]
    holdout_group_labels: tuple[str, ...]
    split_fingerprint: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported gravity holdout-split schema version.")
        if not self.measurement_identity or not self.metadata_fingerprint:
            raise ValueError("holdout measurement and metadata identities must be nonempty.")
        calibration = _immutable_bool(
            self.calibration_mask, name="calibration_mask", size=self.num_measurements
        )
        holdout = _immutable_bool(
            self.holdout_mask, name="holdout_mask", size=self.num_measurements
        )
        if np.any(calibration & holdout) or not np.all(calibration | holdout):
            raise ValueError("calibration and holdout masks must be complementary.")
        if not np.any(calibration) or not np.any(holdout):
            raise ValueError("both calibration and holdout sets must be nonempty.")
        object.__setattr__(self, "calibration_mask", calibration)
        object.__setattr__(self, "holdout_mask", holdout)
        if len(self.calibration_group_labels) + len(self.holdout_group_labels) != self.num_groups:
            raise ValueError("group labels do not match num_groups.")
        if set(self.calibration_group_labels) & set(self.holdout_group_labels):
            raise ValueError("a group cannot occur in both calibration and holdout sets.")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "measurement_identity": self.measurement_identity,
            "metadata_fingerprint": self.metadata_fingerprint,
            "config": asdict(self.config),
            "num_measurements": self.num_measurements,
            "num_groups": self.num_groups,
            "calibration_mask": self.calibration_mask.tolist(),
            "holdout_mask": self.holdout_mask.tolist(),
            "calibration_group_labels": list(self.calibration_group_labels),
            "holdout_group_labels": list(self.holdout_group_labels),
            "split_fingerprint": self.split_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> GravityHoldoutSplit:
        config_payload = cast(dict[str, Any], payload["config"])
        split = cls(
            int(cast(int, payload["schema_version"])),
            str(payload["measurement_identity"]),
            str(payload["metadata_fingerprint"]),
            GravityHoldoutSplitConfig(
                unit=cast(GravityHoldoutUnit, config_payload["unit"]),
                holdout_fraction=float(config_payload["holdout_fraction"]),
                seed=int(config_payload["seed"]),
                stratify_by=tuple(config_payload["stratify_by"]),
            ),
            int(cast(int, payload["num_measurements"])),
            int(cast(int, payload["num_groups"])),
            np.asarray(payload["calibration_mask"]),
            np.asarray(payload["holdout_mask"]),
            tuple(cast(list[str], payload["calibration_group_labels"])),
            tuple(cast(list[str], payload["holdout_group_labels"])),
            str(payload["split_fingerprint"]),
        )
        expected = _split_fingerprint(split)
        if split.split_fingerprint != expected:
            raise ValueError("serialized gravity holdout split fingerprint mismatch.")
        return split


@dataclass(frozen=True, slots=True)
class GravityPredictiveMetrics:
    measurements: int
    observed_total: float
    modeled_total: float
    negative_binomial_deviance: float
    poisson_deviance: float
    data_log_likelihood: float
    mae: float
    rmse: float
    weighted_rmse: float


@dataclass(frozen=True, slots=True)
class GravityHoldoutValidationReport:
    schema_version: int
    report_fingerprint: str
    split_fingerprint: str
    measurement_identity: str
    selected_specification_fingerprint: str
    fitted_model_fingerprint: str
    calibration: GravityPredictiveMetrics
    holdout: GravityPredictiveMetrics
    predicted_measurements: np.ndarray
    free_od_demand: np.ndarray
    full_od_demand: np.ndarray
    estimation_result: GravityEstimationResult

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported gravity holdout-report schema version.")
        for name in ("predicted_measurements", "free_od_demand", "full_od_demand"):
            array = np.array(getattr(self, name), copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def _metadata_fingerprint(metadata: GravityValidationMetadata) -> str:
    return fingerprint(
        {"schema_version": 1}
        | {
            name: getattr(metadata, name)
            for name in metadata.__dataclass_fields__
        }
    )


def _group_labels(
    metadata: GravityValidationMetadata,
    unit: GravityHoldoutUnit,
    explicit_group_labels: object | None,
) -> np.ndarray:
    fields = {
        "vehicle_journey": "vehicle_journey",
        "stop_time_series": "stop",
        "line": "line",
        "direction": "direction",
        "time_block": "time_period",
    }
    if unit == "explicit_group":
        if explicit_group_labels is None:
            raise ValueError("explicit_group_labels are required for explicit_group.")
        labels = np.asarray(explicit_group_labels)
        if labels.shape != (metadata.num_measurements,):
            raise ValueError("explicit_group_labels have the wrong measurement dimension.")
        return labels
    if explicit_group_labels is not None:
        raise ValueError("explicit_group_labels are valid only for explicit_group.")
    labels = getattr(metadata, fields[unit])
    if labels is None:
        raise ValueError(f"metadata.{fields[unit]} is required for holdout unit {unit}.")
    return np.asarray(labels)


def _stratum_key(
    indices: np.ndarray,
    metadata: GravityValidationMetadata,
    fields: tuple[str, ...],
) -> str:
    payload = []
    for field in fields:
        labels = getattr(metadata, field)
        if labels is None:
            raise ValueError(f"metadata.{field} is required for stratification.")
        payload.append((field, tuple(sorted({_label_key(value) for value in labels[indices]}))))
    return canonical_json(payload)


def _ordered_group_keys(keys: list[str], *, seed: int, stratum: str) -> list[str]:
    return sorted(
        keys,
        key=lambda key: fingerprint(
            {"schema_version": 1, "seed": seed, "stratum": stratum, "group": key}
        ),
    )


def _split_fingerprint(split: GravityHoldoutSplit) -> str:
    return fingerprint(
        {
            "schema_version": split.schema_version,
            "measurement_identity": split.measurement_identity,
            "metadata_fingerprint": split.metadata_fingerprint,
            "config": asdict(split.config),
            "num_measurements": split.num_measurements,
            "num_groups": split.num_groups,
            "calibration_mask": split.calibration_mask,
            "holdout_mask": split.holdout_mask,
            "calibration_group_labels": split.calibration_group_labels,
            "holdout_group_labels": split.holdout_group_labels,
        }
    )


def build_gravity_holdout_split(
    *,
    metadata: GravityValidationMetadata,
    measurement_identity: str,
    config: GravityHoldoutSplitConfig,
    explicit_group_labels: object | None = None,
) -> GravityHoldoutSplit:
    """Select complete groups deterministically, optionally within strata."""
    if not measurement_identity:
        raise ValueError("measurement_identity must be nonempty.")
    labels = _group_labels(metadata, config.unit, explicit_group_labels)
    keys = np.asarray([_label_key(value) for value in labels], dtype=object)
    unique = sorted(set(keys.tolist()))
    if len(unique) < 2:
        raise ValueError("grouped holdout requires at least two distinct groups.")
    strata: dict[str, list[str]] = {}
    for key in unique:
        indices = np.flatnonzero(keys == key)
        stratum = _stratum_key(indices, metadata, config.stratify_by)
        strata.setdefault(stratum, []).append(key)
    holdout_keys: set[str] = set()
    singleton_keys: list[str] = []
    for stratum, local_keys in sorted(strata.items()):
        ordered = _ordered_group_keys(local_keys, seed=config.seed, stratum=stratum)
        if len(ordered) == 1:
            singleton_keys.extend(ordered)
            continue
        count = int(round(config.holdout_fraction * len(ordered)))
        count = min(max(count, 1), len(ordered) - 1)
        holdout_keys.update(ordered[:count])
    if not holdout_keys:
        candidates = _ordered_group_keys(
            singleton_keys or unique, seed=config.seed, stratum="singleton-pool"
        )
        holdout_keys.add(candidates[0])
    if len(holdout_keys) == len(unique):
        holdout_keys.remove(sorted(holdout_keys)[-1])
    holdout = np.asarray([key in holdout_keys for key in keys], dtype=bool)
    calibration = ~holdout
    calibration_labels = tuple(key for key in unique if key not in holdout_keys)
    holdout_labels = tuple(key for key in unique if key in holdout_keys)
    provisional = GravityHoldoutSplit(
        1,
        measurement_identity,
        _metadata_fingerprint(metadata),
        config,
        metadata.num_measurements,
        len(unique),
        calibration,
        holdout,
        calibration_labels,
        holdout_labels,
        "pending",
    )
    return replace(provisional, split_fingerprint=_split_fingerprint(provisional))


def _nb_logpmf(y: np.ndarray, mu: np.ndarray, dispersion: float) -> np.ndarray:
    return (
        gammaln(y + dispersion)
        - gammaln(dispersion)
        - gammaln(y + 1)
        - dispersion * np.log1p(mu / dispersion)
        + xlogy(y, mu)
        - xlogy(y, dispersion + mu)
    )


def _metrics(
    observed: np.ndarray,
    modeled: np.ndarray,
    mask: np.ndarray,
    dispersion: float,
    likelihood: GravityLikelihood,
) -> GravityPredictiveMetrics:
    y = observed[mask]
    mu = modeled[mask]
    residual = y - mu
    variance = mu + mu**2 / dispersion
    saturated = np.maximum(y, np.finfo(mu.dtype).tiny)
    nb_deviance = 2 * np.sum(
        _nb_logpmf(y, saturated, dispersion) - _nb_logpmf(y, mu, dispersion)
    )
    poisson_deviance = 2 * np.sum(xlogy(y, y / mu) - (y - mu))
    if likelihood is GravityLikelihood.POISSON:
        log_likelihood = np.sum(xlogy(y, mu) - mu - gammaln(y + 1))
    else:
        log_likelihood = np.sum(_nb_logpmf(y, mu, dispersion))
    weights = 1 / variance
    return GravityPredictiveMetrics(
        int(y.size),
        float(np.sum(y)),
        float(np.sum(mu)),
        float(max(0.0, nb_deviance)),
        float(max(0.0, poisson_deviance)),
        float(log_likelihood),
        float(np.mean(np.abs(residual))),
        float(np.sqrt(np.mean(residual**2))),
        float(np.sqrt(np.sum(weights * residual**2) / np.sum(weights))),
    )


def estimate_and_validate_gravity_holdout(
    *,
    problem: GravityObjectiveProblem,
    compact_layout: CompactODAssignmentLayout,
    split: GravityHoldoutSplit,
    measurement_identity: str,
    initial_raw_parameters: object,
    estimator_config: GravityEstimatorConfig = GravityEstimatorConfig(),
    execution_policy: GravityExecutionPolicy = GravityExecutionPolicy(),
) -> GravityHoldoutValidationReport:
    """Re-estimate on calibration groups and score untouched complete groups."""
    if split.measurement_identity != measurement_identity:
        raise ValueError("holdout split and requested measurement identities differ.")
    if split.split_fingerprint != _split_fingerprint(split):
        raise ValueError("gravity holdout split fingerprint mismatch.")
    if split.num_measurements != problem.observations.size:
        raise ValueError("holdout split has the wrong measurement dimension.")
    original_mask = problem.calibration_mask
    assert original_mask is not None
    if not np.all(original_mask):
        raise ValueError("holdout re-estimation requires an initially full-data problem.")
    calibration_problem = replace(problem, calibration_mask=split.calibration_mask)
    result = estimate_gravity_model(
        problem=calibration_problem,
        compact_layout=compact_layout,
        initial_raw_parameters=initial_raw_parameters,
        config=estimator_config,
        execution=execution_policy,
    )
    observed = np.asarray(problem.observations, dtype=np.float64)
    modeled = np.asarray(result.predicted_measurements, dtype=np.float64)
    dispersion = float(result.physical_parameters[2])
    calibration = _metrics(
        observed,
        modeled,
        split.calibration_mask,
        dispersion,
        problem.likelihood,
    )
    holdout = _metrics(
        observed,
        modeled,
        split.holdout_mask,
        dispersion,
        problem.likelihood,
    )
    payload = {
        "schema_version": 1,
        "split_fingerprint": split.split_fingerprint,
        "measurement_identity": measurement_identity,
        "specification_fingerprint": problem.parameter_layout.specification.fingerprint,
        "fitted_model_fingerprint": result.model_fingerprint,
        "calibration": asdict(calibration),
        "holdout": asdict(holdout),
        "predicted_measurements": modeled,
        "free_od_demand": result.free_od_demand,
        "full_od_demand": result.full_od_demand,
    }
    return GravityHoldoutValidationReport(
        1,
        fingerprint(payload),
        split.split_fingerprint,
        measurement_identity,
        problem.parameter_layout.specification.fingerprint,
        result.model_fingerprint,
        calibration,
        holdout,
        modeled,
        result.free_od_demand,
        result.full_od_demand,
        result,
    )
