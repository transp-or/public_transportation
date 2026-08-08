"""Likelihood comparison and deterministic MAP sensitivity utilities."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from typing import Callable, Literal, Mapping

import numpy as np

from public_transportation.preprocessing.reduced_od.artifacts import canonical_json

from .estimator import ReducedODFitResult, estimate_minimal_gravity
from .objective import MinimalGravityProblem
from .operations import (
    GaussianRawParameterPrior,
    ReducedODFitConfig,
    ReducedODRawParameterBounds,
)
from .parameters import MinimalGravityParameterLayout


ComparisonProgress = Callable[[Mapping[str, object]], None]
_LIKELIHOODS = ("poisson", "negative_binomial")
_LIKELIHOOD_KEYS = frozenset(_LIKELIHOODS)


@dataclass(frozen=True, slots=True)
class ReducedODModelComparisonEntry:
    model_fingerprint: str
    artifact_fingerprint: str
    likelihood: str
    fit: ReducedODFitResult
    elapsed_seconds: float
    parameter_names: tuple[str, ...]
    initial_raw_parameters: np.ndarray
    resolved_lower_bounds: np.ndarray | None
    resolved_upper_bounds: np.ndarray | None

    def __post_init__(self) -> None:
        for field in (
            "initial_raw_parameters",
            "resolved_lower_bounds",
            "resolved_upper_bounds",
        ):
            value = getattr(self, field)
            if value is not None:
                array = np.array(value, dtype=np.float64, copy=True)
                array.setflags(write=False)
                object.__setattr__(self, field, array)


@dataclass(frozen=True, slots=True)
class ReducedODLikelihoodComparison:
    entries: tuple[ReducedODModelComparisonEntry, ...]


@dataclass(frozen=True, slots=True)
class ReducedODPriorSensitivityEntry:
    scenario: str
    prior_fingerprint: str
    fit: ReducedODFitResult
    distance_from_ml: float
    warm_start_parent: str


@dataclass(frozen=True, slots=True)
class ReducedODPriorSensitivityResult:
    ml: ReducedODFitResult
    scenarios: tuple[ReducedODPriorSensitivityEntry, ...]


def _problem_with_likelihood(
    problem: MinimalGravityProblem,
    likelihood: Literal["poisson", "negative_binomial"],
) -> MinimalGravityProblem:
    specification = replace(
        problem.parameter_layout.specification,
        likelihood=likelihood,
    )
    return replace(
        problem,
        parameter_layout=MinimalGravityParameterLayout(specification),
    )


def _validate_keys(name: str, values: Mapping[str, object]) -> None:
    keys = set(values)
    if keys != _LIKELIHOOD_KEYS:
        missing = sorted(_LIKELIHOOD_KEYS - keys)
        unexpected = sorted(keys - _LIKELIHOOD_KEYS)
        raise ValueError(
            f"{name} keys must be exactly {sorted(_LIKELIHOOD_KEYS)}; "
            f"missing={missing}, unexpected={unexpected}."
        )


def _resolve_config(
    config: ReducedODFitConfig,
    problem: MinimalGravityProblem,
) -> tuple[ReducedODFitConfig, ReducedODRawParameterBounds | None]:
    names = problem.parameter_layout.raw_parameter_names(
        problem.production_basis_labels
    )
    bounds = config.raw_parameter_bounds
    if config.named_raw_parameter_bounds is not None:
        bounds = config.named_raw_parameter_bounds.resolve(names)
    if bounds is not None and bounds.lower.shape != (problem.parameter_layout.size,):
        raise ValueError("raw parameter bounds do not match the parameter layout.")
    return (
        replace(
            config,
            raw_parameter_bounds=bounds,
            named_raw_parameter_bounds=None,
        ),
        bounds,
    )


def compare_reduced_od_likelihoods(
    *,
    problem: MinimalGravityProblem,
    initial_raw_parameters: Mapping[str, object],
    artifact_fingerprint: str,
    fit_configs: Mapping[str, ReducedODFitConfig] | None = None,
    fit_config: ReducedODFitConfig | None = None,
    priors: Mapping[str, GaussianRawParameterPrior | None] | None = None,
    progress: ComparisonProgress | None = None,
) -> ReducedODLikelihoodComparison:
    """Fit Poisson and negative binomial without rebuilding compact artifacts."""
    if not artifact_fingerprint:
        raise ValueError("artifact_fingerprint must be non-empty.")
    _validate_keys("initial_raw_parameters", initial_raw_parameters)
    if fit_configs is not None and fit_config is not None:
        raise ValueError("provide fit_configs or fit_config, not both.")
    if fit_configs is None:
        if fit_config is None:
            raise ValueError("fit_configs or fit_config is required.")
        if (
            fit_config.raw_parameter_bounds is not None
            or fit_config.named_raw_parameter_bounds is not None
        ):
            raise ValueError(
                "singular fit_config is supported only when unbounded; use "
                "likelihood-specific fit_configs for bounds."
            )
        fit_configs = {likelihood: fit_config for likelihood in _LIKELIHOODS}
    else:
        _validate_keys("fit_configs", fit_configs)
    if priors is not None:
        unexpected = sorted(set(priors) - _LIKELIHOOD_KEYS)
        if unexpected:
            raise ValueError(f"unexpected prior likelihood keys: {unexpected}.")

    prepared: list[
        tuple[
            str,
            MinimalGravityProblem,
            np.ndarray,
            ReducedODFitConfig,
            ReducedODRawParameterBounds | None,
        ]
    ] = []
    checkpoint_paths: list[str] = []
    for likelihood in _LIKELIHOODS:
        selected = _problem_with_likelihood(problem, likelihood)  # type: ignore[arg-type]
        raw = np.asarray(initial_raw_parameters[likelihood], dtype=np.float64)
        if raw.shape != (selected.parameter_layout.size,) or not np.all(
            np.isfinite(raw)
        ):
            raise ValueError(
                f"initial_raw_parameters[{likelihood!r}] must have shape "
                f"({selected.parameter_layout.size},) with finite values."
            )
        config, bounds = _resolve_config(fit_configs[likelihood], selected)
        if bounds is not None and (
            np.any(raw < bounds.lower) or np.any(raw > bounds.upper)
        ):
            raise ValueError(
                f"initial_raw_parameters[{likelihood!r}] lies outside its bounds."
            )
        prior = None if priors is None else priors.get(likelihood)
        if prior is not None and prior.mean.shape != (selected.parameter_layout.size,):
            if (
                likelihood == "poisson"
                and prior.mean.size == selected.parameter_layout.size + 1
            ):
                raise ValueError(
                    "Poisson prior must not include a dispersion coordinate."
                )
            raise ValueError(
                f"prior for {likelihood!r} does not match its parameter layout."
            )
        if config.checkpoint_path is not None:
            checkpoint_paths.append(str(config.checkpoint_path))
        prepared.append((likelihood, selected, raw, config, bounds))
    if len(checkpoint_paths) != len(set(checkpoint_paths)):
        raise ValueError("likelihood fits must use distinct checkpoint paths.")
    poisson_item, negative_binomial_item = prepared
    poisson_bounds = poisson_item[4]
    negative_binomial_bounds = negative_binomial_item[4]
    if (poisson_bounds is None) != (negative_binomial_bounds is None):
        raise ValueError("both likelihoods must either declare bounds or be unbounded.")
    if poisson_bounds is not None and negative_binomial_bounds is not None:
        poisson_names = poisson_item[1].parameter_layout.raw_parameter_names(
            poisson_item[1].production_basis_labels
        )
        negative_binomial_names = negative_binomial_item[
            1
        ].parameter_layout.raw_parameter_names(
            negative_binomial_item[1].production_basis_labels
        )
        poisson_intervals = {
            name: (poisson_bounds.lower[index], poisson_bounds.upper[index])
            for index, name in enumerate(poisson_names)
        }
        negative_binomial_intervals = {
            name: (
                negative_binomial_bounds.lower[index],
                negative_binomial_bounds.upper[index],
            )
            for index, name in enumerate(negative_binomial_names)
        }
        for name in set(poisson_names) & set(negative_binomial_names):
            if poisson_intervals[name] != negative_binomial_intervals[name]:
                raise ValueError(
                    f"shared raw parameter {name!r} must have identical bounds."
                )

    overall_started = time.perf_counter()
    if progress is not None:
        progress(
            {
                "phase": "likelihood_comparison",
                "status": "started",
                "completed_models": 0,
                "total_models": 2,
                "elapsed_seconds": 0.0,
            }
        )
    entries = []
    for index, (likelihood, selected, raw, config, bounds) in enumerate(prepared):
        model_fingerprint = hashlib.sha256(
            canonical_json(
                {
                    "artifact_fingerprint": artifact_fingerprint,
                    "likelihood": likelihood,
                    "production_mode": selected.parameter_layout.specification.production_mode,
                }
            ).encode("utf-8")
        ).hexdigest()
        started = time.perf_counter()

        def model_progress(event: Mapping[str, object]) -> None:
            if progress is not None:
                progress(
                    {
                        **event,
                        "phase": "likelihood_comparison",
                        "likelihood": likelihood,
                        "completed_models": index,
                        "total_models": 2,
                    }
                )

        try:
            fit = estimate_minimal_gravity(
                problem=selected,
                initial_raw_parameters=raw,
                model_fingerprint=model_fingerprint,
                config=config,
                prior=None if priors is None else priors.get(likelihood),
                progress=model_progress if progress is not None else None,
            )
        except Exception as error:
            if progress is not None:
                progress(
                    {
                        "phase": "likelihood_comparison",
                        "status": "failed",
                        "likelihood": likelihood,
                        "completed_models": index,
                        "total_models": 2,
                        "elapsed_seconds": time.perf_counter() - overall_started,
                        "error": str(error),
                    }
                )
            raise
        elapsed = time.perf_counter() - started
        names = selected.parameter_layout.raw_parameter_names(
            selected.production_basis_labels
        )
        entries.append(
            ReducedODModelComparisonEntry(
                model_fingerprint=model_fingerprint,
                artifact_fingerprint=artifact_fingerprint,
                likelihood=likelihood,
                fit=fit,
                elapsed_seconds=elapsed,
                parameter_names=names,
                initial_raw_parameters=raw,
                resolved_lower_bounds=None if bounds is None else bounds.lower,
                resolved_upper_bounds=None if bounds is None else bounds.upper,
            )
        )
        if progress is not None:
            total_elapsed = time.perf_counter() - overall_started
            completed = index + 1
            progress(
                {
                    "phase": "likelihood_comparison",
                    "status": "completed" if fit.status == "complete" else fit.status,
                    "likelihood": likelihood,
                    "completed_models": completed,
                    "total_models": 2,
                    "elapsed_seconds": total_elapsed,
                    "predicted_remaining_seconds": (
                        total_elapsed * (2 - completed) / completed
                    ),
                }
            )
    return ReducedODLikelihoodComparison(tuple(entries))


def run_reduced_od_prior_sensitivity(
    *,
    problem: MinimalGravityProblem,
    initial_raw_parameters: object,
    model_fingerprint: str,
    scenarios: Mapping[str, GaussianRawParameterPrior],
    fit_config: ReducedODFitConfig,
    progress: ComparisonProgress | None = None,
) -> ReducedODPriorSensitivityResult:
    """Run a small named prior grid against one already prepared problem."""
    started = time.perf_counter()
    if progress is not None:
        progress(
            {
                "phase": "prior_sensitivity",
                "status": "started",
                "completed_scenarios": 0,
                "total_scenarios": len(scenarios),
                "elapsed_seconds": 0.0,
            }
        )
    ml_config = replace(fit_config, method="ml")
    ml = estimate_minimal_gravity(
        problem=problem,
        initial_raw_parameters=initial_raw_parameters,
        model_fingerprint=model_fingerprint,
        config=ml_config,
    )
    entries = []
    map_config = replace(fit_config, method="map", resume=False)
    for index, name in enumerate(sorted(scenarios)):
        prior = scenarios[name]
        prior_fingerprint = hashlib.sha256(
            canonical_json(
                {"mean": prior.mean.tolist(), "scale": prior.scale.tolist()}
            ).encode("utf-8")
        ).hexdigest()

        def scenario_progress(event: Mapping[str, object]) -> None:
            if progress is not None:
                progress(
                    {
                        **event,
                        "phase": "prior_sensitivity",
                        "scenario": name,
                        "completed_scenarios": index,
                        "total_scenarios": len(scenarios),
                    }
                )

        try:
            fit = estimate_minimal_gravity(
                problem=problem,
                initial_raw_parameters=ml.raw_parameters,
                model_fingerprint=model_fingerprint,
                config=map_config,
                prior=prior,
                progress=scenario_progress if progress is not None else None,
            )
        except Exception as error:
            if progress is not None:
                progress(
                    {
                        "phase": "prior_sensitivity",
                        "status": "failed",
                        "scenario": name,
                        "completed_scenarios": index,
                        "total_scenarios": len(scenarios),
                        "elapsed_seconds": time.perf_counter() - started,
                        "error": str(error),
                    }
                )
            raise
        entries.append(
            ReducedODPriorSensitivityEntry(
                scenario=name,
                prior_fingerprint=prior_fingerprint,
                fit=fit,
                distance_from_ml=float(
                    np.linalg.norm(fit.raw_parameters - ml.raw_parameters)
                ),
                warm_start_parent="ml",
            )
        )
        if progress is not None:
            progress(
                {
                    "phase": "prior_sensitivity",
                    "status": "completed" if fit.status == "complete" else fit.status,
                    "scenario": name,
                    "completed_scenarios": index + 1,
                    "total_scenarios": len(scenarios),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    return ReducedODPriorSensitivityResult(ml=ml, scenarios=tuple(entries))
