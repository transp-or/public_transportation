"""Detailed, provenance-aware diagnostics for observed/modelled measurements.

This module is intentionally independent of the assignment operator.  It accepts
the vectors and metadata emitted by a fitted run, and therefore can be used in a
fresh reporting process without rebuilding routing artefacts.  The analysis is
advisory: it never edits observations, parameters, masks, or model artefacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from public_transportation.version import __version__

from .metrics import poisson_deviance_residual


MEASUREMENT_RESIDUAL_ANALYSIS_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ResidualBootstrapConfig:
    """Configuration for deterministic cluster-aware uncertainty estimates."""

    replicates: int = 200
    seed: int = 0
    confidence_level: float = 0.95
    allow_observation_level: bool = False
    minimum_clusters: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.replicates, bool) or not isinstance(self.replicates, Integral) or self.replicates <= 0:
            raise ValueError("bootstrap replicates must be a positive integer.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("bootstrap seed must be an integer.")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("bootstrap confidence_level must lie strictly between zero and one.")
        if isinstance(self.minimum_clusters, bool) or not isinstance(self.minimum_clusters, Integral) or self.minimum_clusters < 2:
            raise ValueError("bootstrap minimum_clusters must be at least two.")

    @classmethod
    def from_value(cls, value: "ResidualBootstrapConfig | Mapping[str, object] | None") -> "ResidualBootstrapConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            payload = dict(value)
            if "replicates" not in payload and "n_replicates" in payload:
                payload["replicates"] = payload.pop("n_replicates")
            if "seed" not in payload and "random_seed" in payload:
                payload["seed"] = payload.pop("random_seed")
            return cls(**payload)
        raise TypeError("bootstrap_config must be ResidualBootstrapConfig, a mapping, or None.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResidualPracticalTolerances:
    """Practical thresholds used to keep candidate patterns conservative."""

    minimum_observations: int = 5
    minimum_clusters: int = 2
    minimum_absolute_bias: float = 0.1
    minimum_relative_bias: float = 0.05
    standardized_residual_threshold: float = 3.0
    isolated_residual_threshold: float = 4.0

    def __post_init__(self) -> None:
        if isinstance(self.minimum_observations, bool) or not isinstance(self.minimum_observations, Integral) or self.minimum_observations <= 0:
            raise ValueError("minimum_observations must be a positive integer.")
        if isinstance(self.minimum_clusters, bool) or not isinstance(self.minimum_clusters, Integral) or self.minimum_clusters <= 0:
            raise ValueError("minimum_clusters must be a positive integer.")
        for name in (
            "minimum_absolute_bias",
            "minimum_relative_bias",
            "standardized_residual_threshold",
            "isolated_residual_threshold",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")

    @classmethod
    def from_value(cls, value: "ResidualPracticalTolerances | Mapping[str, object] | None") -> "ResidualPracticalTolerances":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError("practical_tolerances must be ResidualPracticalTolerances, a mapping, or None.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MeasurementResidualAnalysis:
    """Result of :func:`analyze_measurement_residuals`.

    Data frames are retained for interactive use.  The manifest, candidates,
    and suggestions are JSON-compatible and can be persisted independently.
    """

    row_diagnostics: pd.DataFrame
    grouped_diagnostics: pd.DataFrame
    bootstrap_diagnostics: pd.DataFrame
    journey_diagnostics: pd.DataFrame
    candidate_patterns: tuple[dict[str, object], ...]
    suggestions: tuple[dict[str, object], ...]
    manifest: dict[str, object]
    provenance: dict[str, object]
    output_files: dict[str, Path] = field(default_factory=dict)

    @property
    def grouped_metrics(self) -> pd.DataFrame:
        """Compatibility alias used by the earlier residual-audit API."""
        return self.grouped_diagnostics

    @property
    def journey_sequence_diagnostics(self) -> pd.DataFrame:
        return self.journey_diagnostics

    @property
    def analysis_fingerprint(self) -> str:
        return str(self.manifest["analysis_fingerprint"])

    @property
    def residual_convention(self) -> str:
        return str(self.manifest["residual_convention"])

    @property
    def variance_convention(self) -> str:
        return str(self.manifest["variance_convention"])

    @property
    def likelihood_family(self) -> str:
        return str(self.manifest["likelihood_family"])


def _vector(value: object, *, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    if length is not None and array.size != length:
        raise ValueError(f"{name} has length {array.size}, expected {length}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _column(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _metadata_frame(
    observations: object,
    predictions: object,
    grouping_metadata: object | None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if isinstance(observations, pd.DataFrame):
        source = observations.copy()
        observed_name = _column(source, ("observed_value", "observed", "count", "counts", "value"))
        if observed_name is None:
            raise ValueError("observations is missing an observed-value column.")
        observed = _vector(source[observed_name], name="observations")
        if isinstance(predictions, pd.DataFrame):
            prediction_name = _column(predictions, ("predicted_value", "predicted", "modeled", "mean"))
            if prediction_name is None:
                raise ValueError("predictions is missing a modeled-value column.")
            modeled = _vector(predictions[prediction_name], name="predictions", length=observed.size)
        else:
            modeled = _vector(predictions, name="predictions", length=observed.size)
        metadata = source.drop(columns=[observed_name], errors="ignore").reset_index(drop=True)
        if isinstance(grouping_metadata, pd.DataFrame):
            extra = grouping_metadata.reset_index(drop=True)
            if len(extra) != len(metadata):
                raise ValueError("grouping_metadata must have one row per observation.")
            metadata = metadata.join(extra, how="left", rsuffix="_metadata")
        elif isinstance(grouping_metadata, Mapping):
            for key, value in grouping_metadata.items():
                metadata[str(key)] = value
        elif grouping_metadata is not None:
            raise TypeError("grouping_metadata must be a DataFrame, mapping, or None.")
        return observed, modeled, metadata.reset_index(drop=True)

    observed = _vector(observations, name="observations")
    if isinstance(predictions, pd.DataFrame):
        prediction_name = _column(predictions, ("predicted_value", "predicted", "modeled", "mean"))
        if prediction_name is None:
            raise ValueError("predictions is missing a modeled-value column.")
        modeled = _vector(predictions[prediction_name], name="predictions", length=observed.size)
    else:
        modeled = _vector(predictions, name="predictions", length=observed.size)
    if grouping_metadata is None:
        metadata = pd.DataFrame(index=np.arange(observed.size))
    elif isinstance(grouping_metadata, pd.DataFrame):
        metadata = grouping_metadata.copy()
    elif isinstance(grouping_metadata, Mapping):
        metadata = pd.DataFrame(dict(grouping_metadata))
    else:
        raise TypeError("grouping_metadata must be a DataFrame, mapping, or None.")
    if len(metadata) != observed.size:
        raise ValueError("grouping_metadata must have one row per observation.")
    return observed, modeled, metadata.reset_index(drop=True)


def _canonical_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "method_identifier": "method_id",
        "method": "method_id",
        "line_id": "line",
        "direction_id": "direction",
        "stop_id": "stop",
        "time_bin": "time_period",
        "journey": "trip_id",
        "vehicle_journey": "trip_id",
    }
    frame = metadata.copy()
    for source, target in aliases.items():
        if target not in frame.columns and source in frame.columns:
            frame[target] = frame[source]
    if "line_direction" not in frame.columns and {"line", "direction"} <= set(frame.columns):
        frame["line_direction"] = frame["line"].astype(str) + "|" + frame["direction"].astype(str)
    if "line_time_period" not in frame.columns and {"line", "time_period"} <= set(frame.columns):
        frame["line_time_period"] = frame["line"].astype(str) + "|" + frame["time_period"].astype(str)
    if "stop_measurement_type" not in frame.columns and {"stop", "measurement_type"} <= set(frame.columns):
        frame["stop_measurement_type"] = frame["stop"].astype(str) + "|" + frame["measurement_type"].astype(str)
    return frame


def _family_and_dispersion(provenance: Mapping[str, object]) -> tuple[str, float | None]:
    family = provenance.get("likelihood_family")
    likelihood = provenance.get("likelihood")
    if family is None and isinstance(likelihood, Mapping):
        family = likelihood.get("family")
    if family is None and isinstance(provenance.get("model_specification"), Mapping):
        nested = provenance["model_specification"]
        if isinstance(nested, Mapping) and isinstance(nested.get("likelihood"), Mapping):
            family = nested["likelihood"].get("family")
    normalized = "unknown" if family is None else str(family).strip().lower()
    dispersion = provenance.get("dispersion", provenance.get("negative_binomial_dispersion"))
    if dispersion is None and isinstance(likelihood, Mapping):
        dispersion = likelihood.get("dispersion")
    try:
        value = None if dispersion is None else float(dispersion)
    except (TypeError, ValueError):
        value = None
    if value is not None and (not math.isfinite(value) or value <= 0):
        raise ValueError("negative-binomial dispersion must be finite and positive.")
    return normalized, value


def _negative_binomial_deviance_residual(observed: np.ndarray, modeled: np.ndarray, dispersion: float) -> np.ndarray:
    r = float(dispersion)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        first = np.where(observed > 0, observed * np.log(observed / modeled), 0.0)
        second = (observed + r) * np.log((observed + r) / (modeled + r))
        deviance = 2.0 * (first - second)
    deviance = np.maximum(np.nan_to_num(deviance, nan=0.0, posinf=np.inf, neginf=0.0), 0.0)
    return np.sign(observed - modeled) * np.sqrt(deviance)


def _stable_frame_fingerprint(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    payload = json.dumps([str(column) for column in frame.columns], separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _metric_record(
    frame: pd.DataFrame,
    *,
    grouping: str,
    group_key: str,
    cluster_column: str | None,
) -> dict[str, object]:
    observed = frame["observed_value"].to_numpy(dtype=np.float64)
    modeled = frame["predicted_value"].to_numpy(dtype=np.float64)
    residual = frame["raw_residual"].to_numpy(dtype=np.float64)
    pearson = frame["pearson_residual"].to_numpy(dtype=np.float64)
    deviance = frame["deviance_residual"].to_numpy(dtype=np.float64)
    n = int(len(frame))
    observed_total = float(np.sum(observed))
    predicted_total = float(np.sum(modeled))
    residual_total = float(np.sum(residual))
    relative_total = None if abs(observed_total) <= 1.0e-12 else residual_total / observed_total
    clusters = None
    if cluster_column is not None and cluster_column in frame.columns:
        clusters = int(frame[cluster_column].dropna().nunique())
    finite_pearson = pearson[np.isfinite(pearson)]
    finite_deviance = deviance[np.isfinite(deviance)]
    return {
        "grouping": grouping,
        "group_key": group_key,
        "observation_count": n,
        "independent_cluster_count": clusters,
        "observed_total": observed_total,
        "modeled_total": predicted_total,
        "signed_residual_total": residual_total,
        "relative_total_bias": relative_total,
        "mean_residual": float(np.mean(residual)) if n else None,
        "mae": float(np.mean(np.abs(residual))) if n else None,
        "rmse": float(np.sqrt(np.mean(residual**2))) if n else None,
        "weighted_rmse": float(np.sqrt(np.mean(pearson[np.isfinite(pearson)] ** 2))) if finite_pearson.size else None,
        "mean_pearson_residual": float(np.mean(finite_pearson)) if finite_pearson.size else None,
        "mean_deviance_residual": float(np.mean(finite_deviance)) if finite_deviance.size else None,
        "fraction_positive_residual": float(np.mean(residual > 0.0)) if n else None,
        "fraction_negative_residual": float(np.mean(residual < 0.0)) if n else None,
        "maximum_absolute_standardized_residual": float(np.max(np.abs(finite_pearson), initial=0.0)),
        "minimum_fitted_value": float(np.min(modeled)) if n else None,
        "maximum_fitted_value": float(np.max(modeled)) if n else None,
        "zero_observation_count": int(np.count_nonzero(observed == 0.0)),
        "mean_residual_ci_lower": None,
        "mean_residual_ci_upper": None,
        "relative_total_bias_ci_lower": None,
        "relative_total_bias_ci_upper": None,
    }


def _groupings(metadata: pd.DataFrame) -> tuple[tuple[str, tuple[str, ...]], ...]:
    candidates = (
        ("overall", ()),
        ("method_id", ("method_id",)),
        ("measurement_type", ("measurement_type",)),
        ("line_direction", ("line_direction",)),
        ("line", ("line",)),
        ("direction", ("direction",)),
        ("stop", ("stop",)),
        ("time_period", ("time_period",)),
        ("line_time_period", ("line_time_period",)),
        ("stop_measurement_type", ("stop_measurement_type",)),
        ("journey_position", ("journey_position",)),
        ("origin_zone", ("origin_zone",)),
        ("destination_zone", ("destination_zone",)),
        ("trip_id", ("trip_id",)),
    )
    return tuple((name, columns) for name, columns in candidates if not columns or set(columns) <= set(metadata.columns))


def _grouped_metrics(frame: pd.DataFrame, *, cluster_column: str | None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for grouping, columns in _groupings(frame):
        if not columns:
            rows.append(_metric_record(frame, grouping=grouping, group_key="all", cluster_column=cluster_column))
            continue
        grouped = frame.groupby(list(columns), dropna=False, sort=True, observed=True)
        for values, local in grouped:
            values_tuple = values if isinstance(values, tuple) else (values,)
            labels = ["<missing>" if pd.isna(value) else str(value) for value in values_tuple]
            rows.append(_metric_record(local, grouping=grouping, group_key="|".join(labels), cluster_column=cluster_column))
    return pd.DataFrame(rows)


def _bootstrap(
    frame: pd.DataFrame,
    *,
    cluster_column: str | None,
    config: ResidualBootstrapConfig,
) -> tuple[pd.DataFrame, str, list[str]]:
    warnings: list[str] = []
    if cluster_column is None or cluster_column not in frame.columns:
        if not config.allow_observation_level:
            warnings.append(
                "No cluster identifier was available; uncertainty was not computed. "
                "Enable allow_observation_level explicitly to use row bootstrap."
            )
            return pd.DataFrame(), "unavailable", warnings
        unit_column = "__row_cluster"
        working = frame.copy()
        working[unit_column] = np.arange(len(working))
        method = "observation_bootstrap"
        warnings.append("No journey cluster was available; observation-level bootstrap is less reliable.")
    else:
        unit_column = cluster_column
        working = frame
        method = "cluster_bootstrap"
    units = working[unit_column].dropna().unique()
    if units.size < config.minimum_clusters:
        warnings.append(f"Bootstrap requires at least {config.minimum_clusters} independent clusters; found {units.size}.")
        return pd.DataFrame(), "unavailable", warnings
    rng = np.random.default_rng(config.seed)
    records: list[dict[str, object]] = []
    alpha = (1.0 - config.confidence_level) / 2.0
    for replicate in range(config.replicates):
        selected = rng.choice(units, size=units.size, replace=True)
        pieces = [working.loc[working[unit_column] == unit] for unit in selected]
        sample = pd.concat(pieces, ignore_index=True)
        for grouping, columns in _groupings(sample):
            if not columns:
                groups = [("all", sample)]
            else:
                groups = []
                for values, local in sample.groupby(list(columns), dropna=False, sort=True, observed=True):
                    values_tuple = values if isinstance(values, tuple) else (values,)
                    groups.append(("|".join("<missing>" if pd.isna(value) else str(value) for value in values_tuple), local))
            for group_key, local in groups:
                metric = _metric_record(local, grouping=grouping, group_key=group_key, cluster_column=cluster_column)
                records.append(
                    {
                        "replicate": replicate,
                        "grouping": grouping,
                        "group_key": group_key,
                        "mean_residual": metric["mean_residual"],
                        "relative_total_bias": metric["relative_total_bias"],
                    }
                )
    bootstrap = pd.DataFrame(records)
    if bootstrap.empty:
        return bootstrap, method, warnings
    summaries: list[dict[str, object]] = []
    for (grouping, group_key), local in bootstrap.groupby(["grouping", "group_key"], sort=True, observed=True):
        mean_values = pd.to_numeric(local["mean_residual"], errors="coerce").dropna().to_numpy()
        relative_values = pd.to_numeric(local["relative_total_bias"], errors="coerce").dropna().to_numpy()
        summaries.append(
            {
                "grouping": grouping,
                "group_key": group_key,
                "mean_residual_ci_lower": float(np.quantile(mean_values, alpha)) if mean_values.size else None,
                "mean_residual_ci_upper": float(np.quantile(mean_values, 1.0 - alpha)) if mean_values.size else None,
                "relative_total_bias_ci_lower": float(np.quantile(relative_values, alpha)) if relative_values.size else None,
                "relative_total_bias_ci_upper": float(np.quantile(relative_values, 1.0 - alpha)) if relative_values.size else None,
            }
        )
    return pd.DataFrame(summaries), method, warnings


def _sequence_diagnostics(frame: pd.DataFrame, cluster_column: str | None) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if cluster_column is None or cluster_column not in frame.columns:
        warnings.append("Journey-sequence diagnostics are unavailable because no trip_id/vehicle_journey column exists.")
        return pd.DataFrame(), warnings
    order_column = _column(frame, ("journey_position", "stop_sequence", "sequence", "observation_time"))
    if order_column is None:
        warnings.append("Journey-sequence diagnostics are unavailable because no within-journey ordering column exists.")
        return pd.DataFrame(), warnings
    rows: list[dict[str, object]] = []
    for journey, local in frame.groupby(cluster_column, dropna=False, sort=True, observed=True):
        ordered = local.sort_values(order_column, kind="mergesort")
        residual = ordered["raw_residual"].to_numpy(dtype=np.float64)
        pearson = ordered["pearson_residual"].to_numpy(dtype=np.float64)
        correlation = None
        if residual.size >= 3 and np.std(residual[:-1]) > 0 and np.std(residual[1:]) > 0:
            correlation = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
        signs = np.sign(residual)
        runs: list[int] = []
        for sign in signs:
            if not runs or sign != signs[sum(runs) - 1]:
                runs.append(1)
            else:
                runs[-1] += 1
        measurement = ordered.get("measurement_type", pd.Series("", index=ordered.index)).astype(str).to_numpy()
        boarding = float(np.sum(residual[measurement == "boarding"]))
        alighting = float(np.sum(residual[measurement == "alighting"]))
        rows.append(
            {
                "journey_id": "<missing>" if pd.isna(journey) else str(journey),
                "observation_count": int(residual.size),
                "adjacent_residual_correlation": correlation,
                "maximum_sign_run_length": max(runs, default=0),
                "sign_run_lengths": runs,
                "cumulative_residual": float(np.sum(residual)),
                "cumulative_boarding_alighting_imbalance": boarding - alighting,
                "first_residual": float(residual[0]) if residual.size else None,
                "last_residual": float(residual[-1]) if residual.size else None,
                "mean_pearson_residual": float(np.mean(pearson)) if pearson.size else None,
                "ordering_column": order_column,
            }
        )
    return pd.DataFrame(rows), warnings


def _pattern_and_suggestions(
    diagnostics: pd.DataFrame,
    grouped: pd.DataFrame,
    journeys: pd.DataFrame,
    *,
    tolerances: ResidualPracticalTolerances,
    bootstrap_method: str,
    residual_convention: str,
    warnings: list[str],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    candidates: list[dict[str, object]] = []
    suggestions: list[dict[str, object]] = []
    for row in grouped.to_dict(orient="records"):
        grouping = str(row["grouping"])
        if grouping == "overall":
            continue
        n = int(row["observation_count"])
        clusters = row["independent_cluster_count"]
        if n < tolerances.minimum_observations:
            continue
        if clusters is not None and clusters < tolerances.minimum_clusters:
            continue
        mean_bias = float(row["mean_residual"])
        relative = row["relative_total_bias"]
        materially_nonzero = abs(mean_bias) >= tolerances.minimum_absolute_bias or (
            relative is not None and abs(float(relative)) >= tolerances.minimum_relative_bias
        )
        if not materially_nonzero:
            continue
        lower = row.get("mean_residual_ci_lower")
        upper = row.get("mean_residual_ci_upper")
        excludes_zero = lower is not None and upper is not None and (float(lower) > 0 or float(upper) < 0)
        downgraded = bootstrap_method == "unavailable" or not excludes_zero
        if bootstrap_method != "unavailable" and not excludes_zero:
            continue
        positive_means = residual_convention == "observed_minus_modeled"
        direction = (
            "underprediction"
            if (mean_bias > 0) == positive_means
            else "overprediction"
        )
        direction_label = f"candidate_{direction}"
        if grouping == "time_period" or grouping == "line_time_period":
            label = "candidate_time_regime_pattern"
        elif grouping in {"line_direction", "line", "direction"}:
            label = "candidate_line_direction_pattern"
        elif grouping in {"stop", "stop_measurement_type"}:
            label = "candidate_stop_pattern"
        elif grouping == "trip_id":
            label = "candidate_journey_sequence_pattern"
        elif grouping == "destination_zone":
            label = "candidate_destination_pattern"
        else:
            label = "candidate_systematic_bias"
        candidate = {
            "label": label,
            "labels": [label, direction_label],
            "direction": direction,
            "grouping": grouping,
            "group_key": row["group_key"],
            "signed_bias": mean_bias,
            "relative_bias": relative,
            "confidence_interval": {"lower": lower, "upper": upper},
            "observation_count": n,
            "cluster_count": clusters,
            "uncertainty_status": "downgraded" if downgraded else "cluster_bootstrap",
            "caveat": "In-sample diagnostic pattern; not evidence of out-of-sample predictive performance.",
        }
        candidates.append(candidate)
        action = {
            "method_id": "Review the measurement method and its coverage.",
            "measurement_type": "Audit the measurement definition and units.",
            "stop": "Check stop-to-zone mapping, platform identity, coverage, and destination attractiveness.",
            "stop_measurement_type": "Check stop mapping, platform identity, and measurement coverage.",
            "line_direction": "Check route/direction coding, timetable alignment, journey assignment, and line-specific demand structure.",
            "line": "Check route coding, timetable alignment, and line-specific demand structure.",
            "direction": "Check direction coding and timetable alignment.",
            "time_period": "Check observation-to-time-bin mapping, regime boundaries, production effects, and service changes.",
            "line_time_period": "Check timetable alignment and service changes within the line/time regime.",
            "trip_id": "Check timetable timing, route assignment, stop ordering, load propagation, and boundary flows.",
            "destination_zone": "Review destination mapping, attractiveness effects, and destination regularization.",
        }.get(grouping, "Audit the affected metadata and model representation before changing the fit.")
        suggestions.append(
            {
                "category": "data_audit" if grouping in {"stop", "stop_measurement_type", "line_direction", "time_period", "line_time_period", "trip_id"} else "model_review",
                "pattern_labels": [label, direction_label],
                "severity": "high" if abs(mean_bias) >= 2 * tolerances.minimum_absolute_bias else "medium",
                "evidence_group": f"{grouping}:{row['group_key']}",
                "affected_labels": [row["group_key"]],
                "signed_bias": mean_bias,
                "relative_bias": relative,
                "confidence_interval": {"lower": lower, "upper": upper},
                "observation_count": n,
                "cluster_count": clusters,
                "suggested_action": action,
                "caveat": candidate["caveat"],
            }
        )
    finite_pearson = diagnostics["pearson_residual"].to_numpy(dtype=np.float64)
    extreme = np.flatnonzero(np.isfinite(finite_pearson) & (np.abs(finite_pearson) >= tolerances.isolated_residual_threshold))
    if extreme.size:
        suggestions.append(
            {
                "category": "data_audit",
                "severity": "medium",
                "evidence_group": "isolated_extreme_residuals",
                "affected_labels": diagnostics.iloc[extreme]["observation_id"].astype(str).tolist() if "observation_id" in diagnostics else extreme.astype(str).tolist(),
                "signed_bias": None,
                "relative_bias": None,
                "confidence_interval": None,
                "observation_count": int(extreme.size),
                "cluster_count": None,
                "suggested_action": "Audit the raw observation, timestamp, stop identifier, units, and duplicate status. Do not remove it solely because the residual is large.",
                "caveat": "An extreme in-sample residual is a review trigger, not proof that the observation is wrong.",
            }
        )
    valid_pearson = finite_pearson[np.isfinite(finite_pearson)]
    if valid_pearson.size and float(np.mean(np.abs(valid_pearson) >= tolerances.standardized_residual_threshold)) >= 0.25:
        suggestions.append(
            {
                "category": "model_review",
                "severity": "medium",
                "evidence_group": "overall:standardized_residuals",
                "affected_labels": [],
                "signed_bias": None,
                "relative_bias": None,
                "confidence_interval": None,
                "observation_count": int(len(diagnostics)),
                "cluster_count": None,
                "suggested_action": "Review variance specification, low-count behavior, exposure/offset treatment, and measurement-noise assumptions. Do not choose a negative-binomial likelihood solely from large standardized residuals; calibrate the mean first.",
                "caveat": "This is an in-sample variance diagnostic.",
            }
        )
    if not journeys.empty and journeys["cumulative_boarding_alighting_imbalance"].abs().mean() > tolerances.minimum_absolute_bias:
        candidates.append(
            {
                "label": "candidate_boarding_alighting_imbalance",
                "direction": "mixed",
                "grouping": "trip_id",
                "group_key": "journey_sequence",
                "signed_bias": float(journeys["cumulative_boarding_alighting_imbalance"].mean()),
                "relative_bias": None,
                "confidence_interval": {"lower": None, "upper": None},
                "observation_count": int(len(diagnostics)),
                "cluster_count": int(len(journeys)),
                "uncertainty_status": "diagnostic_only",
                "caveat": "Check boarding/alighting definitions, initial-onboard and terminal-outflow corrections, boundary regularization, and whether processes should be modelled separately.",
            }
        )
        suggestions.append(
            {
                "category": "data_audit",
                "severity": "high",
                "evidence_group": "journey_sequence:boarding_alighting_imbalance",
                "affected_labels": [],
                "signed_bias": float(journeys["cumulative_boarding_alighting_imbalance"].mean()),
                "relative_bias": None,
                "confidence_interval": None,
                "observation_count": int(len(diagnostics)),
                "cluster_count": int(len(journeys)),
                "suggested_action": "Check boarding and alighting measurement definitions, initial-onboard corrections, terminal-outflow corrections, boundary-flow regularization, and whether the observation processes should be modelled separately.",
                "caveat": "The balance is a diagnostic, not an automatic correction.",
            }
        )
    if "measurement_type" in grouped.columns:
        typed = grouped[grouped["grouping"] == "measurement_type"].set_index("group_key")
        if {"boarding", "alighting"} <= set(typed.index):
            boarding_bias = float(typed.loc["boarding", "mean_residual"])
            alighting_bias = float(typed.loc["alighting", "mean_residual"])
            imbalance = boarding_bias - alighting_bias
            if abs(imbalance) >= tolerances.minimum_absolute_bias:
                candidate = {
                    "label": "candidate_boarding_alighting_imbalance",
                    "labels": ["candidate_boarding_alighting_imbalance"],
                    "direction": "mixed",
                    "grouping": "measurement_type",
                    "group_key": "boarding-vs-alighting",
                    "signed_bias": imbalance,
                    "relative_bias": None,
                    "confidence_interval": None,
                    "observation_count": int(
                        typed.loc["boarding", "observation_count"]
                        + typed.loc["alighting", "observation_count"]
                    ),
                    "cluster_count": None,
                    "uncertainty_status": "diagnostic_only",
                    "caveat": "A boarding/alighting difference is a diagnostic, not an automatic correction.",
                }
                if candidate not in candidates:
                    candidates.append(candidate)
                if not any(
                    item.get("evidence_group") == "journey_sequence:boarding_alighting_imbalance"
                    for item in suggestions
                ):
                    suggestions.append(
                        {
                            "category": "data_audit",
                            "severity": "high",
                            "pattern_labels": ["candidate_boarding_alighting_imbalance"],
                            "evidence_group": "measurement_type:boarding-vs-alighting",
                            "affected_labels": ["boarding", "alighting"],
                            "signed_bias": imbalance,
                            "relative_bias": None,
                            "confidence_interval": None,
                            "observation_count": int(
                                typed.loc["boarding", "observation_count"]
                                + typed.loc["alighting", "observation_count"]
                            ),
                            "cluster_count": None,
                            "suggested_action": "Check boarding and alighting measurement definitions, initial-onboard corrections, terminal-outflow corrections, boundary-flow regularization, and whether the two observation processes should be modelled separately.",
                            "caveat": "The imbalance is an in-sample diagnostic, not evidence that either process should be deleted or forcibly equalized.",
                        }
                    )
    return tuple(candidates), tuple(suggestions)


def analyze_measurement_residuals(
    observations: object,
    predictions: object,
    *,
    variance: object | None = None,
    grouping_metadata: object | None = None,
    cluster_column: str = "trip_id",
    residual_convention: str = "observed_minus_modeled",
    bootstrap_config: ResidualBootstrapConfig | Mapping[str, object] | None = None,
    practical_tolerances: ResidualPracticalTolerances | Mapping[str, object] | None = None,
    provenance: Mapping[str, object] | None = None,
) -> MeasurementResidualAnalysis:
    """Analyze fitted measurement residuals without changing the fitted run."""
    if not isinstance(cluster_column, str) or not cluster_column.strip():
        raise ValueError("cluster_column must be a non-empty string.")
    if residual_convention not in {"observed_minus_modeled", "modeled_minus_observed"}:
        raise ValueError("residual_convention must be observed_minus_modeled or modeled_minus_observed.")
    observed, modeled, metadata = _metadata_frame(observations, predictions, grouping_metadata)
    if np.any(observed < 0.0) or np.any(modeled < 0.0):
        raise ValueError("observations and predictions must be finite non-negative values.")
    metadata = _canonical_metadata(metadata)
    if "observation_id" in metadata.columns:
        identifiers = metadata["observation_id"].to_numpy(copy=True)
        if pd.isna(identifiers).any() or pd.Series(identifiers).duplicated().any():
            raise ValueError("observation_id values must be present and unique.")
        metadata = metadata.drop(columns=["observation_id"])
    else:
        identifiers = np.arange(observed.size, dtype=np.int64)
    cluster = cluster_column if cluster_column in metadata.columns else None
    warnings: list[str] = []
    prov = {} if provenance is None else dict(provenance)
    family, dispersion = _family_and_dispersion(prov)
    if variance is None:
        variance_name = _column(metadata, ("variance", "fitted_variance"))
        if variance_name is not None:
            variance_array = _vector(metadata[variance_name], name="variance", length=observed.size)
            variance_convention = f"supplied metadata column {variance_name}"
        elif family == "negative_binomial" and dispersion is not None:
            variance_array = modeled + modeled**2 / dispersion
            variance_convention = "negative_binomial_mean_variance"
        else:
            variance_array = modeled.copy()
            variance_convention = "poisson_mean_variance_fallback"
            if family == "negative_binomial":
                warnings.append("Negative-binomial analysis had no supplied variance or dispersion; Pearson residuals use a Poisson fallback.")
    else:
        variance_array = _vector(variance, name="variance", length=observed.size)
        variance_convention = "caller_supplied_variance"
    if np.any(variance_array < 0.0):
        raise ValueError("variance must be finite and non-negative.")
    tolerances = ResidualPracticalTolerances.from_value(practical_tolerances)
    bootstrap = ResidualBootstrapConfig.from_value(bootstrap_config)
    raw = observed - modeled if residual_convention == "observed_minus_modeled" else modeled - observed
    relative = np.full(raw.shape, np.nan, dtype=np.float64)
    valid_relative = np.abs(observed) > 1.0e-12
    relative[valid_relative] = raw[valid_relative] / observed[valid_relative]
    pearson = raw / np.sqrt(np.maximum(variance_array, np.finfo(np.float64).tiny))
    if family == "negative_binomial" and dispersion is not None:
        deviance = _negative_binomial_deviance_residual(observed, modeled, dispersion)
        deviance_convention = "negative_binomial_deviance"
    else:
        deviance = poisson_deviance_residual(observed, modeled)
        deviance_convention = "poisson_deviance"
    frame = metadata.copy()
    frame.insert(0, "observation_id", identifiers)
    frame["observed_value"] = observed
    frame["predicted_value"] = modeled
    frame["variance"] = variance_array
    frame["raw_residual"] = raw
    frame["absolute_residual"] = np.abs(raw)
    frame["relative_residual"] = relative
    frame["pearson_residual"] = pearson
    frame["deviance_residual"] = deviance
    frame["positive_residual"] = raw > 0
    frame["negative_residual"] = raw < 0
    frame["zero_observation"] = observed == 0.0
    frame["support_failure"] = (observed > 0.0) & (modeled <= 1.0e-8)
    grouped = _grouped_metrics(frame, cluster_column=cluster)
    bootstrap_summary, bootstrap_method, bootstrap_warnings = _bootstrap(frame, cluster_column=cluster, config=bootstrap)
    warnings.extend(bootstrap_warnings)
    if not bootstrap_summary.empty:
        grouped = grouped.drop(
            columns=[
                "mean_residual_ci_lower",
                "mean_residual_ci_upper",
                "relative_total_bias_ci_lower",
                "relative_total_bias_ci_upper",
            ],
            errors="ignore",
        ).merge(bootstrap_summary, on=["grouping", "group_key"], how="left")
    journeys, sequence_warnings = _sequence_diagnostics(frame, cluster)
    warnings.extend(sequence_warnings)
    candidates, suggestions = _pattern_and_suggestions(
        frame,
        grouped,
        journeys,
        tolerances=tolerances,
        bootstrap_method=bootstrap_method,
        residual_convention=residual_convention,
        warnings=warnings,
    )
    if cluster is None:
        warnings.append(f"Requested cluster column {cluster_column!r} was not present; journey-level uncertainty is unavailable.")
    provenance_out: dict[str, object] = {
        "residual_convention": residual_convention,
        "variance_convention": variance_convention,
        "deviance_convention": deviance_convention,
        "likelihood_family": family,
        "model_fingerprint": prov.get("model_fingerprint", prov.get("model_identity_fingerprint")),
        "artifact_fingerprint": prov.get("artifact_fingerprint", prov.get("artifact_identity_fingerprint")),
        "package_revision": prov.get("package_revision", prov.get("repository_revision", __version__)),
        "cluster_column": cluster_column,
        "bootstrap_method": bootstrap_method,
        "bootstrap_config": bootstrap.to_dict(),
        "practical_tolerances": tolerances.to_dict(),
        "source_provenance": prov,
        "observation_fingerprint": _stable_frame_fingerprint(frame),
        "metadata_fingerprint": _stable_frame_fingerprint(metadata),
    }
    if provenance_out["model_fingerprint"] is None:
        warnings.append("Model fingerprint was not supplied; the analysis cannot identify a fitted model beyond its input vectors.")
    if provenance_out["artifact_fingerprint"] is None:
        provenance_out["artifact_fingerprint"] = hashlib.sha256(
            (str(provenance_out["observation_fingerprint"]) + str(provenance_out["metadata_fingerprint"])).encode("utf-8")
        ).hexdigest()
    fingerprint_payload = {
        "schema_version": MEASUREMENT_RESIDUAL_ANALYSIS_SCHEMA_VERSION,
        "provenance": provenance_out,
        "grouped": grouped.to_dict(orient="records"),
        "journey": journeys.to_dict(orient="records"),
        "candidates": candidates,
        "suggestions": suggestions,
    }
    analysis_fingerprint = hashlib.sha256(
        json.dumps(_json_value(fingerprint_payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    provenance_out["analysis_fingerprint"] = analysis_fingerprint
    manifest = {
        "schema_version": MEASUREMENT_RESIDUAL_ANALYSIS_SCHEMA_VERSION,
        "status": "completed",
        "scope": "in_sample",
        "analysis_fingerprint": analysis_fingerprint,
        "residual_convention": residual_convention,
        "variance_convention": variance_convention,
        "model_fingerprint": provenance_out["model_fingerprint"],
        "artifact_fingerprint": provenance_out["artifact_fingerprint"],
        "package_revision": provenance_out["package_revision"],
        "analysis_configuration": {
            "bootstrap": bootstrap.to_dict(),
            "practical_tolerances": tolerances.to_dict(),
        },
        "likelihood_family": family,
        "cluster_column": cluster_column,
        "uncertainty_method": bootstrap_method,
        "bootstrap": bootstrap.to_dict(),
        "provenance": provenance_out,
        "candidate_patterns": list(candidates),
        "suggestions": list(suggestions),
        "warnings": sorted(set(warnings)),
        "output_files": {},
        "advisory_only": True,
        "holdout_claim": False,
        "interpretation": "These are in-sample diagnostic patterns. They are not evidence of out-of-sample predictive performance.",
    }
    return MeasurementResidualAnalysis(
        row_diagnostics=frame,
        grouped_diagnostics=grouped,
        bootstrap_diagnostics=bootstrap_summary,
        journey_diagnostics=journeys,
        candidate_patterns=candidates,
        suggestions=suggestions,
        manifest=manifest,
        provenance=provenance_out,
    )


def _json_value(value: object) -> object:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [_json_value(row) for row in frame.to_dict(orient="records")]


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, na_rep="", lineterminator="\n")
    temporary.replace(path)


def _suggestions_markdown(analysis: MeasurementResidualAnalysis) -> str:
    lines = [
        "# Measurement residual suggestions",
        "",
        f"Analysis fingerprint: `{analysis.analysis_fingerprint}`.",
        f"Residual convention: `{analysis.residual_convention}`; variance convention: `{analysis.variance_convention}`.",
        "These are in-sample diagnostic patterns. They are not evidence of out-of-sample predictive performance.",
        "No observations, fitted parameters, calibration masks, or routing artifacts were modified.",
        "",
    ]
    if not analysis.suggestions:
        lines.append("No configured advisory pattern was triggered.")
    else:
        for index, suggestion in enumerate(analysis.suggestions, start=1):
            lines.extend(
                [
                    f"## {index}. {suggestion['category']} ({suggestion['severity']})",
                    "",
                    f"- Evidence: `{suggestion['evidence_group']}`.",
                    f"- Observations: {suggestion['observation_count']}; clusters: {suggestion['cluster_count']}. ",
                    f"- Suggested action: {suggestion['suggested_action']}",
                    f"- Caveat: {suggestion['caveat']}",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def write_measurement_residual_analysis(
    analysis: MeasurementResidualAnalysis,
    output_directory: str | Path,
    *,
    force: bool = False,
) -> MeasurementResidualAnalysis:
    """Persist the detailed analysis and its advisory suggestions."""
    if not isinstance(analysis, MeasurementResidualAnalysis):
        raise TypeError("analysis must be MeasurementResidualAnalysis.")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = (
        "residual_analysis.json",
        "grouped_residual_analysis.csv",
        "journey_residual_analysis.csv",
        "residual_suggestions.json",
        "residual_suggestions.md",
        "residual_analysis_manifest.json",
    )
    existing = [output / name for name in names if (output / name).exists()]
    if existing and not force:
        raise FileExistsError("measurement residual analysis output already exists: " + ", ".join(map(str, existing)))
    paths = {name: output / name for name in names}
    payload = {
        "schema_version": MEASUREMENT_RESIDUAL_ANALYSIS_SCHEMA_VERSION,
        "status": "completed",
        "scope": "in_sample",
        "analysis_fingerprint": analysis.analysis_fingerprint,
        "residual_convention": analysis.manifest.get("residual_convention"),
        "variance_convention": analysis.manifest.get("variance_convention"),
        "likelihood_family": analysis.manifest.get("likelihood_family"),
        "model_fingerprint": analysis.manifest.get("model_fingerprint"),
        "artifact_fingerprint": analysis.manifest.get("artifact_fingerprint"),
        "package_revision": analysis.manifest.get("package_revision"),
        "analysis_configuration": analysis.manifest.get("analysis_configuration", {}),
        "provenance": analysis.provenance,
        "row_diagnostics": _records(analysis.row_diagnostics),
        "grouped_diagnostics": _records(analysis.grouped_diagnostics),
        "bootstrap_diagnostics": _records(analysis.bootstrap_diagnostics),
        "journey_diagnostics": _records(analysis.journey_diagnostics),
        "candidate_patterns": list(analysis.candidate_patterns),
        "suggestions": list(analysis.suggestions),
        "warnings": analysis.manifest.get("warnings", []),
        "advisory_only": True,
    }
    _write_json(paths["residual_analysis.json"], payload)
    _write_csv(paths["grouped_residual_analysis.csv"], analysis.grouped_diagnostics)
    _write_csv(paths["journey_residual_analysis.csv"], analysis.journey_diagnostics)
    _write_json(
        paths["residual_suggestions.json"],
        {
            "schema_version": 1,
            "advisory_only": True,
            "analysis_fingerprint": analysis.analysis_fingerprint,
            "residual_convention": analysis.residual_convention,
            "variance_convention": analysis.variance_convention,
            "likelihood_family": analysis.likelihood_family,
            "provenance": analysis.provenance,
            "suggestions": list(analysis.suggestions),
        },
    )
    temporary = paths["residual_suggestions.md"].with_name(".residual_suggestions.md.tmp")
    temporary.write_text(_suggestions_markdown(analysis), encoding="utf-8")
    temporary.replace(paths["residual_suggestions.md"])
    manifest = dict(analysis.manifest)
    manifest["output_files"] = {name: str(path) for name, path in paths.items()}
    _write_json(paths["residual_analysis_manifest.json"], manifest)
    return MeasurementResidualAnalysis(
        row_diagnostics=analysis.row_diagnostics,
        grouped_diagnostics=analysis.grouped_diagnostics,
        bootstrap_diagnostics=analysis.bootstrap_diagnostics,
        journey_diagnostics=analysis.journey_diagnostics,
        candidate_patterns=analysis.candidate_patterns,
        suggestions=analysis.suggestions,
        manifest=manifest,
        provenance=analysis.provenance,
        output_files=paths,
    )


# Short aliases for callers that prefer the existing residual-audit wording.
MeasurementResidualAnalysisResult = MeasurementResidualAnalysis
BootstrapConfig = ResidualBootstrapConfig
PracticalTolerances = ResidualPracticalTolerances
