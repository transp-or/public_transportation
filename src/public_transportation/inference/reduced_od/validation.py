"""Adequacy summaries and structured predictive holdouts for reduced OD."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.special import gammaln, xlogy  # type: ignore[import-untyped]

from public_transportation.preprocessing.reduced_od.artifacts import (
    canonical_json,
    fingerprint_json,
)

from .estimator import ReducedODFitResult, estimate_minimal_gravity
from .objective import (
    MinimalGravityProblem,
    evaluate_minimal_gravity_objective,
)
from .operations import (
    GaussianRawParameterPrior,
    ReducedODFitConfig,
)
from .parameters import transform_minimal_gravity_parameters


ReducedODHoldoutUnit = Literal[
    "vehicle_journey",
    "stop_time_series",
    "line",
    "direction",
    "time_block",
    "explicit_group",
]


def _immutable_labels(value: object, *, size: int, name: str) -> np.ndarray:
    array = np.array(value, copy=True)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ReducedODMeasurementMetadata:
    number_of_measurements: int
    measurement_type: np.ndarray | None = None
    line: np.ndarray | None = None
    direction: np.ndarray | None = None
    stop: np.ndarray | None = None
    time_period: np.ndarray | None = None
    vehicle_journey: np.ndarray | None = None
    origin_zone: np.ndarray | None = None
    destination_zone: np.ndarray | None = None
    transfer_place: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.number_of_measurements <= 0:
            raise ValueError("number_of_measurements must be positive.")
        for name in (
            "measurement_type",
            "line",
            "direction",
            "stop",
            "time_period",
            "vehicle_journey",
            "origin_zone",
            "destination_zone",
            "transfer_place",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _immutable_labels(
                        value, size=self.number_of_measurements, name=name
                    ),
                )

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(
            {
                "number_of_measurements": self.number_of_measurements,
                "labels": {
                    name: None
                    if getattr(self, name) is None
                    else np.asarray(getattr(self, name)).tolist()
                    for name in (
                        "measurement_type",
                        "line",
                        "direction",
                        "stop",
                        "time_period",
                        "vehicle_journey",
                        "origin_zone",
                        "destination_zone",
                        "transfer_place",
                    )
                },
            }
        )


@dataclass(frozen=True, slots=True)
class ReducedODHoldoutConfig:
    unit: ReducedODHoldoutUnit
    fraction: float = 0.2
    seed: int = 0

    def __post_init__(self) -> None:
        if self.unit not in {
            "vehicle_journey",
            "stop_time_series",
            "line",
            "direction",
            "time_block",
            "explicit_group",
        }:
            raise ValueError("unsupported holdout unit.")
        if not 0.0 < self.fraction < 1.0:
            raise ValueError("fraction must lie strictly between zero and one.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")


@dataclass(frozen=True, slots=True)
class ReducedODHoldoutSplit:
    measurement_identity: str
    metadata_fingerprint: str
    config: ReducedODHoldoutConfig
    calibration_mask: np.ndarray
    holdout_mask: np.ndarray
    calibration_groups: tuple[str, ...]
    holdout_groups: tuple[str, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        calibration = np.array(self.calibration_mask, dtype=np.bool_, copy=True)
        holdout = np.array(self.holdout_mask, dtype=np.bool_, copy=True)
        if calibration.ndim != 1 or holdout.shape != calibration.shape:
            raise ValueError("holdout masks must be aligned vectors.")
        if np.any(calibration & holdout) or not np.all(calibration | holdout):
            raise ValueError("holdout masks must be complementary.")
        if not np.any(calibration) or not np.any(holdout):
            raise ValueError("calibration and holdout sets must be non-empty.")
        if set(self.calibration_groups) & set(self.holdout_groups):
            raise ValueError("a holdout group cannot leak into calibration.")
        calibration.setflags(write=False)
        holdout.setflags(write=False)
        object.__setattr__(self, "calibration_mask", calibration)
        object.__setattr__(self, "holdout_mask", holdout)


def _label(value: object) -> str:
    scalar = value.item() if isinstance(value, np.generic) else value
    return canonical_json({"type": type(scalar).__name__, "value": scalar})


def build_reduced_od_holdout_split(
    *,
    metadata: ReducedODMeasurementMetadata,
    measurement_identity: str,
    config: ReducedODHoldoutConfig,
    explicit_group_labels: object | None = None,
) -> ReducedODHoldoutSplit:
    """Hold out complete correlated groups with deterministic seed hashing."""
    fields = {
        "vehicle_journey": "vehicle_journey",
        "stop_time_series": "stop",
        "line": "line",
        "direction": "direction",
        "time_block": "time_period",
    }
    if not measurement_identity:
        raise ValueError("measurement_identity must be non-empty.")
    if config.unit == "explicit_group":
        if explicit_group_labels is None:
            raise ValueError("explicit_group_labels are required.")
        values = _immutable_labels(
            explicit_group_labels,
            size=metadata.number_of_measurements,
            name="explicit_group_labels",
        )
    else:
        if explicit_group_labels is not None:
            raise ValueError("explicit labels apply only to explicit_group.")
        field = fields[config.unit]
        values = getattr(metadata, field)
        if values is None:
            raise ValueError(f"metadata.{field} is required for this holdout.")
    labels = np.asarray([_label(value) for value in values], dtype=object)
    unique = sorted(set(labels.tolist()))
    if len(unique) < 2:
        raise ValueError("grouped holdout requires at least two groups.")
    ordered = sorted(
        unique,
        key=lambda group: fingerprint_json(
            {
                "seed": config.seed,
                "measurement_identity": measurement_identity,
                "group": group,
            }
        ),
    )
    count = min(max(round(config.fraction * len(ordered)), 1), len(ordered) - 1)
    held = set(ordered[:count])
    holdout = np.asarray([label in held for label in labels], dtype=np.bool_)
    calibration = ~holdout
    calibration_groups = tuple(item for item in unique if item not in held)
    holdout_groups = tuple(item for item in unique if item in held)
    provisional = {
        "measurement_identity": measurement_identity,
        "metadata_fingerprint": metadata.fingerprint,
        "unit": config.unit,
        "fraction": config.fraction,
        "seed": config.seed,
        "calibration_groups": list(calibration_groups),
        "holdout_groups": list(holdout_groups),
    }
    return ReducedODHoldoutSplit(
        measurement_identity=measurement_identity,
        metadata_fingerprint=metadata.fingerprint,
        config=config,
        calibration_mask=calibration,
        holdout_mask=holdout,
        calibration_groups=calibration_groups,
        holdout_groups=holdout_groups,
        fingerprint=fingerprint_json(provisional),
    )


@dataclass(frozen=True, slots=True)
class ReducedODPredictiveMetrics:
    measurements: int
    observed_total: float
    modeled_total: float
    poisson_deviance: float
    negative_binomial_deviance: float
    data_log_likelihood: float
    mae: float
    rmse: float
    weighted_rmse: float


def _negative_binomial_logpmf(
    observed: np.ndarray, modeled: np.ndarray, dispersion: float
) -> np.ndarray:
    return (
        gammaln(observed + dispersion)
        - gammaln(dispersion)
        - gammaln(observed + 1.0)
        - dispersion * np.log1p(modeled / dispersion)
        + xlogy(observed, modeled)
        - xlogy(observed, dispersion + modeled)
    )


def _predictive_metrics(
    observed: np.ndarray,
    modeled: np.ndarray,
    mask: np.ndarray,
    *,
    dispersion: float,
) -> ReducedODPredictiveMetrics:
    y = observed[mask]
    mu = modeled[mask]
    residual = y - mu
    deviance = 2.0 * np.sum(xlogy(y, y / mu) - (y - mu))
    poisson_log_likelihood = np.sum(xlogy(y, mu) - mu - gammaln(y + 1.0))
    if np.isinf(dispersion):
        negative_binomial_deviance = deviance
        log_likelihood = poisson_log_likelihood
        variance = mu
    else:
        saturated = np.maximum(y, np.finfo(np.float64).tiny)
        negative_binomial_deviance = 2.0 * np.sum(
            _negative_binomial_logpmf(y, saturated, dispersion)
            - _negative_binomial_logpmf(y, mu, dispersion)
        )
        log_likelihood = np.sum(_negative_binomial_logpmf(y, mu, dispersion))
        variance = mu + mu * mu / dispersion
    return ReducedODPredictiveMetrics(
        measurements=int(y.size),
        observed_total=float(np.sum(y)),
        modeled_total=float(np.sum(mu)),
        poisson_deviance=float(max(0.0, deviance)),
        negative_binomial_deviance=float(max(0.0, negative_binomial_deviance)),
        data_log_likelihood=float(log_likelihood),
        mae=float(np.mean(np.abs(residual))),
        rmse=float(np.sqrt(np.mean(residual * residual))),
        weighted_rmse=float(np.sqrt(np.mean(residual * residual / variance))),
    )


@dataclass(frozen=True, slots=True)
class ReducedODHoldoutReport:
    split_fingerprint: str
    fit: ReducedODFitResult
    calibration: ReducedODPredictiveMetrics
    holdout: ReducedODPredictiveMetrics
    predicted_measurements: np.ndarray

    def __post_init__(self) -> None:
        prediction = np.array(self.predicted_measurements, dtype=np.float64, copy=True)
        prediction.setflags(write=False)
        object.__setattr__(self, "predicted_measurements", prediction)


def validate_reduced_od_holdout(
    *,
    problem: MinimalGravityProblem,
    initial_raw_parameters: object,
    model_fingerprint: str,
    metadata: ReducedODMeasurementMetadata,
    split: ReducedODHoldoutSplit,
    fit_config: ReducedODFitConfig = ReducedODFitConfig(),
    prior: GaussianRawParameterPrior | None = None,
) -> ReducedODHoldoutReport:
    """Refit on calibration rows and score untouched grouped holdout rows."""
    if metadata.fingerprint != split.metadata_fingerprint:
        raise ValueError("holdout metadata fingerprint mismatch.")
    if split.calibration_mask.shape != problem.observations.shape:
        raise ValueError("holdout split has the wrong measurement dimension.")
    calibration_problem = replace(problem, calibration_mask=split.calibration_mask)
    fit = estimate_minimal_gravity(
        problem=calibration_problem,
        initial_raw_parameters=initial_raw_parameters,
        model_fingerprint=f"{model_fingerprint}:holdout:{split.fingerprint}",
        config=fit_config,
        prior=prior,
    )
    evaluation = evaluate_minimal_gravity_objective(fit.raw_parameters, problem=problem)
    prediction = np.asarray(evaluation.measurement_mean, dtype=np.float64)
    dispersion = reduced_od_dispersion(problem, fit.raw_parameters)
    return ReducedODHoldoutReport(
        split_fingerprint=split.fingerprint,
        fit=fit,
        calibration=_predictive_metrics(
            problem.observations,
            prediction,
            split.calibration_mask,
            dispersion=dispersion,
        ),
        holdout=_predictive_metrics(
            problem.observations,
            prediction,
            split.holdout_mask,
            dispersion=dispersion,
        ),
        predicted_measurements=prediction,
    )


def reduced_od_dispersion(
    problem: MinimalGravityProblem, raw_parameters: object
) -> float:
    """Return fitted NB dispersion, or infinity for a Poisson specification."""
    transformed = transform_minimal_gravity_parameters(
        raw_parameters, layout=problem.parameter_layout
    )
    return (
        float("inf")
        if transformed.dispersion is None
        else float(transformed.dispersion)
    )
