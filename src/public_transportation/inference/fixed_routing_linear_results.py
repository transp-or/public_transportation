"""Versioned result contract for fixed-routing linear OD estimation."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .fixed_routing_linear_problem import FixedRoutingLinearProblem
from .fixed_routing_linear_quality import LinearEstimateQuality
from .fixed_routing_linear_trf_solver import TRFLSMRResult

Array = np.ndarray
FIXED_ROUTING_LINEAR_RESULT_SCHEMA_VERSION = 1


def _immutable(value: object) -> Array:
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class FixedRoutingLinearResult:
    """Portable, solver-independent record of one fitted configuration."""

    schema_version: int
    mode: str
    solver: str
    regularization_selection: str
    regularization_names: tuple[str, ...]
    regularization_strengths: Array
    od_layout_fingerprint: str
    assignment_fingerprint: str
    mapping_fingerprint: str
    routing_parameter: float
    free_od_indices: Array
    prior_demand: Array
    estimated_demand: Array
    solver_variable: Array
    variable_scales: Array
    lower_bounds: Array
    upper_bounds: Array
    observations: Array
    observation_weights: Array
    fixed_measurement_offset: Array
    predicted_measurements: Array
    raw_residual: Array
    weighted_residual: Array
    objective: float
    data_objective: float
    regularization_objectives: Array
    regularization_residual_norms: Array
    gradient: Array
    success: bool
    status: int
    message: str
    iterations: int
    solver_cost: float
    solver_optimality: float
    matvec_count: int
    rmatvec_count: int
    elapsed_seconds: float
    lower_active: Array
    upper_active: Array
    fixed_by_bounds: Array
    lower_multipliers: Array
    upper_multipliers: Array
    projected_gradient: Array
    projected_gradient_inf_norm: float
    feasibility_inf_norm: float
    quality_free_indices: Array
    measurement_singular_values: Array
    measurement_rank: int
    measurement_nullity: int
    measurement_rank_tolerance: float
    measurement_condition_estimate: float
    combined_rank: int
    combined_nullity: int
    effective_data_degrees_of_freedom: float
    resolution_closure_inf_norm: float
    data_mode_fractions: Array
    data_resolution_score: Array
    regularization_reliance_score: Array
    null_space_participation: Array
    classifications: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != FIXED_ROUTING_LINEAR_RESULT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported result schema version: {self.schema_version}."
            )
        if self.mode != "fixed_routing_linear":
            raise ValueError("mode must be 'fixed_routing_linear'.")
        for name in (
            "regularization_strengths",
            "free_od_indices",
            "prior_demand",
            "estimated_demand",
            "solver_variable",
            "variable_scales",
            "lower_bounds",
            "upper_bounds",
            "observations",
            "observation_weights",
            "fixed_measurement_offset",
            "predicted_measurements",
            "raw_residual",
            "weighted_residual",
            "regularization_objectives",
            "regularization_residual_norms",
            "gradient",
            "lower_active",
            "upper_active",
            "fixed_by_bounds",
            "lower_multipliers",
            "upper_multipliers",
            "projected_gradient",
            "quality_free_indices",
            "measurement_singular_values",
            "data_mode_fractions",
            "data_resolution_score",
            "regularization_reliance_score",
            "null_space_participation",
        ):
            object.__setattr__(self, name, _immutable(getattr(self, name)))
        n = self.estimated_demand.size
        m = self.observations.size
        for name in (
            "free_od_indices",
            "prior_demand",
            "solver_variable",
            "variable_scales",
            "lower_bounds",
            "upper_bounds",
            "gradient",
            "lower_active",
            "upper_active",
            "fixed_by_bounds",
            "lower_multipliers",
            "upper_multipliers",
            "projected_gradient",
            "data_resolution_score",
            "regularization_reliance_score",
            "null_space_participation",
        ):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have shape ({n},).")
        for name in (
            "observation_weights",
            "fixed_measurement_offset",
            "predicted_measurements",
            "raw_residual",
            "weighted_residual",
        ):
            if getattr(self, name).shape != (m,):
                raise ValueError(f"{name} must have shape ({m},).")
        r = len(self.regularization_names)
        if (
            self.regularization_strengths.shape != (r,)
            or self.regularization_objectives.shape != (r,)
            or self.regularization_residual_norms.shape != (r,)
        ):
            raise ValueError("regularization arrays must match regularization_names.")
        if len(self.classifications) != n:
            raise ValueError("classifications must match estimated_demand.")
        if np.unique(self.free_od_indices).size != n:
            raise ValueError("free_od_indices must be unique.")
        np.testing.assert_allclose(
            self.raw_residual, self.predicted_measurements - self.observations
        )
        np.testing.assert_allclose(
            self.weighted_residual,
            np.sqrt(self.observation_weights) * self.raw_residual,
        )
        expected = self.data_objective + float(np.sum(self.regularization_objectives))
        if not np.isclose(self.objective, expected, rtol=1e-10, atol=1e-10):
            raise ValueError(
                "objective does not equal its data and regularization parts."
            )


def build_fixed_routing_linear_result(
    problem: FixedRoutingLinearProblem,
    solver_result: TRFLSMRResult,
    quality: LinearEstimateQuality,
) -> FixedRoutingLinearResult:
    """Build the stable result record from a solved problem."""
    evaluation = solver_result.evaluation
    kkt = solver_result.kkt
    return FixedRoutingLinearResult(
        schema_version=FIXED_ROUTING_LINEAR_RESULT_SCHEMA_VERSION,
        mode="fixed_routing_linear",
        solver="trf_lsmr",
        regularization_selection=problem.regularization_selection,
        regularization_names=tuple(
            block.name for block in problem.regularization_blocks
        ),
        regularization_strengths=np.asarray(
            [block.strength for block in problem.regularization_blocks]
        ),
        od_layout_fingerprint=problem.provenance.od_layout_fingerprint,
        assignment_fingerprint=problem.provenance.assignment_fingerprint,
        mapping_fingerprint=problem.provenance.mapping_fingerprint,
        routing_parameter=problem.provenance.routing_parameter,
        free_od_indices=problem.free_od_indices,
        prior_demand=problem.prior_demand,
        estimated_demand=solver_result.demand,
        solver_variable=solver_result.solver_variable,
        variable_scales=problem.variable_scales,
        lower_bounds=problem.lower_bounds,
        upper_bounds=problem.upper_bounds,
        observations=problem.observations,
        observation_weights=problem.observation_weights,
        fixed_measurement_offset=problem.fixed_measurement_offset,
        predicted_measurements=evaluation.data_fit.prediction,
        raw_residual=evaluation.data_fit.raw_residual,
        weighted_residual=evaluation.data_fit.weighted_residual,
        objective=evaluation.objective,
        data_objective=evaluation.data_fit.objective,
        regularization_objectives=np.asarray(
            [item.objective for item in evaluation.regularization]
        ),
        regularization_residual_norms=np.asarray(
            [np.linalg.norm(item.residual) for item in evaluation.regularization]
        ),
        gradient=evaluation.gradient,
        success=solver_result.success,
        status=solver_result.status,
        message=solver_result.message,
        iterations=solver_result.iterations,
        solver_cost=solver_result.solver_cost,
        solver_optimality=solver_result.solver_optimality,
        matvec_count=solver_result.matvec_count,
        rmatvec_count=solver_result.rmatvec_count,
        elapsed_seconds=solver_result.elapsed_seconds,
        lower_active=kkt.lower_active,
        upper_active=kkt.upper_active,
        fixed_by_bounds=kkt.fixed_by_bounds,
        lower_multipliers=kkt.lower_multipliers,
        upper_multipliers=kkt.upper_multipliers,
        projected_gradient=kkt.projected_gradient,
        projected_gradient_inf_norm=kkt.projected_gradient_inf_norm,
        feasibility_inf_norm=kkt.feasibility_inf_norm,
        quality_free_indices=quality.free_indices,
        measurement_singular_values=quality.measurement_singular_values,
        measurement_rank=quality.measurement_rank,
        measurement_nullity=quality.measurement_nullity,
        measurement_rank_tolerance=quality.measurement_rank_tolerance,
        measurement_condition_estimate=quality.measurement_condition_estimate,
        combined_rank=quality.combined_rank,
        combined_nullity=quality.combined_nullity,
        effective_data_degrees_of_freedom=quality.effective_data_degrees_of_freedom,
        resolution_closure_inf_norm=quality.resolution_closure_inf_norm,
        data_mode_fractions=quality.data_mode_fractions,
        data_resolution_score=quality.data_resolution_score,
        regularization_reliance_score=quality.regularization_reliance_score,
        null_space_participation=quality.null_space_participation,
        classifications=tuple(quality.classifications),
    )


def save_fixed_routing_linear_result(
    result: FixedRoutingLinearResult, path: str | Path
) -> Path:
    """Atomically save a result as a compressed, pickle-free NPZ archive."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        field.name: np.asarray(getattr(result, field.name)) for field in fields(result)
    }
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def load_fixed_routing_linear_result(
    path: str | Path,
    *,
    expected_od_layout_fingerprint: str | None = None,
    expected_assignment_fingerprint: str | None = None,
    expected_mapping_fingerprint: str | None = None,
) -> FixedRoutingLinearResult:
    """Load and validate a result archive and optional provenance expectations."""
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {field.name for field in fields(FixedRoutingLinearResult)}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"result archive is missing fields: {sorted(missing)}.")
        values = {name: archive[name] for name in required}
    tuple_fields = {"regularization_names", "classifications"}
    scalar_fields = (
        required
        - tuple_fields
        - {name for name, value in values.items() if value.ndim > 0}
    )
    kwargs = {
        name: tuple(str(item) for item in values[name].tolist())
        if name in tuple_fields
        else values[name].item()
        if name in scalar_fields
        else values[name]
        for name in required
    }
    result = FixedRoutingLinearResult(**kwargs)
    checks = (
        ("od_layout_fingerprint", expected_od_layout_fingerprint),
        ("assignment_fingerprint", expected_assignment_fingerprint),
        ("mapping_fingerprint", expected_mapping_fingerprint),
    )
    for name, expected in checks:
        if expected is not None and getattr(result, name) != expected:
            raise ValueError(
                f"{name} mismatch: expected {expected!r}, got {getattr(result, name)!r}."
            )
    return result
