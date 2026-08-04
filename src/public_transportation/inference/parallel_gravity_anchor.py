"""Exact control-variate anchors for parallel partial gravity evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Mapping

import jax.numpy as jnp
import jax
import numpy as np

from .gravity.demand import generate_gravity_demand
from .gravity.fidelity import gravity_fidelity_problem_identity
from .gravity.objective import (
    GravityObjectiveEvaluation,
    GravityObjectiveProblem,
    _evaluation_from_mean,
    _objective_from_mean,
    gravity_value_and_gradient_adjoint,
)
from .parallel_partial_execution import FixedBudgetRoutingSelection
from .parallel_routing_executor import (
    ParallelApproximateRoutingOperator,
    PersistentParallelRoutingExecutor,
)


def _anchor_problem_identity(problem: GravityObjectiveProblem) -> str:
    digest = hashlib.sha256()
    payload = {
        "routing_problem_identity": gravity_fidelity_problem_identity(problem),
        "likelihood": problem.likelihood.value,
        "rho": float(problem.rho),
        "mean_floor": float(problem.mean_floor),
        "parameter_layout": repr(problem.parameter_layout),
    }
    digest.update(json.dumps(payload, sort_keys=True).encode())
    for value in (problem.observations, problem.calibration_mask):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ParallelGravityAnchor:
    raw_parameters: np.ndarray
    demand: np.ndarray
    routed_measurements: np.ndarray
    measurement_mean: np.ndarray
    gradient: np.ndarray
    objective: float
    problem_identity: str
    assignment_fingerprint: str
    compact_layout_fingerprint: str
    mapping_fingerprint: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "raw_parameters",
            "demand",
            "routed_measurements",
            "measurement_mean",
            "gradient",
        ):
            value = np.array(getattr(self, name), copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"anchor {name} must be finite.")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not np.isfinite(self.objective):
            raise ValueError("anchor objective must be finite.")
        if self.raw_parameters.shape != self.gradient.shape:
            raise ValueError("anchor parameter and gradient shapes must match.")
        if not all(
            (
                self.problem_identity,
                self.assignment_fingerprint,
                self.compact_layout_fingerprint,
                self.mapping_fingerprint,
            )
        ):
            raise ValueError("anchor identities must not be empty.")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "raw_parameters": self.raw_parameters.tolist(),
            "demand": self.demand.tolist(),
            "routed_measurements": self.routed_measurements.tolist(),
            "measurement_mean": self.measurement_mean.tolist(),
            "gradient": self.gradient.tolist(),
            "objective": self.objective,
            "problem_identity": self.problem_identity,
            "assignment_fingerprint": self.assignment_fingerprint,
            "compact_layout_fingerprint": self.compact_layout_fingerprint,
            "mapping_fingerprint": self.mapping_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ParallelGravityAnchor:
        if int(payload["schema_version"]) != 1:
            raise ValueError("unsupported parallel gravity anchor schema version.")
        return cls(
            raw_parameters=np.asarray(payload["raw_parameters"]),
            demand=np.asarray(payload["demand"]),
            routed_measurements=np.asarray(payload["routed_measurements"]),
            measurement_mean=np.asarray(payload["measurement_mean"]),
            gradient=np.asarray(payload["gradient"]),
            objective=float(payload["objective"]),
            problem_identity=str(payload["problem_identity"]),
            assignment_fingerprint=str(payload["assignment_fingerprint"]),
            compact_layout_fingerprint=str(payload["compact_layout_fingerprint"]),
            mapping_fingerprint=str(payload["mapping_fingerprint"]),
        )


def create_parallel_gravity_anchor(
    raw_parameters: object, *, problem: GravityObjectiveProblem
) -> ParallelGravityAnchor:
    """Evaluate and capture one exact, reusable gravity anchor."""
    raw = jnp.asarray(raw_parameters)
    def demand_function(value):
        return generate_gravity_demand(
            value, features=problem.features, parameter_layout=problem.parameter_layout
        ).demand

    demand, demand_pullback = jax.vjp(demand_function, raw)
    routed = problem.operator.jax_matvec(demand)
    offset = jnp.asarray(problem.operator.fixed_measurement_offset, dtype=demand.dtype)
    rho = jnp.asarray(problem.rho, dtype=demand.dtype)
    mean_unfloored = rho * (routed + offset)
    mean = jnp.maximum(mean_unfloored, problem.mean_floor)
    mean_gradient = jax.grad(lambda value: _objective_from_mean(value, raw, problem))(
        mean
    )
    active_mean = (mean_unfloored > problem.mean_floor).astype(mean.dtype)
    demand_cotangent = rho * problem.operator.jax_rmatvec(
        active_mean * mean_gradient
    )
    direct_gradient = jax.grad(
        lambda parameters: _objective_from_mean(mean, parameters, problem)
    )(raw)
    gradient = demand_pullback(demand_cotangent)[0] + direct_gradient
    evaluation = _evaluation_from_mean(raw, mean=mean, demand=demand, problem=problem)
    return ParallelGravityAnchor(
        raw_parameters=np.asarray(raw),
        demand=np.asarray(demand),
        routed_measurements=np.asarray(routed),
        measurement_mean=np.asarray(evaluation.measurement_mean),
        gradient=np.asarray(gradient),
        objective=float(evaluation.objective),
        problem_identity=_anchor_problem_identity(problem),
        assignment_fingerprint=problem.operator.assignment_fingerprint,
        compact_layout_fingerprint=problem.operator.compact_layout_fingerprint,
        mapping_fingerprint=problem.operator.mapping_fingerprint,
    )


def _validate_anchor(
    anchor: ParallelGravityAnchor, *, problem: GravityObjectiveProblem
) -> None:
    expected = (
        _anchor_problem_identity(problem),
        problem.operator.assignment_fingerprint,
        problem.operator.compact_layout_fingerprint,
        problem.operator.mapping_fingerprint,
    )
    observed = (
        anchor.problem_identity,
        anchor.assignment_fingerprint,
        anchor.compact_layout_fingerprint,
        anchor.mapping_fingerprint,
    )
    if observed != expected:
        raise ValueError("parallel gravity anchor identity mismatch.")


def parallel_anchored_value_and_gradient(
    raw_parameters: object,
    *,
    problem: GravityObjectiveProblem,
    executor: PersistentParallelRoutingExecutor,
    selection: FixedBudgetRoutingSelection,
    anchor: ParallelGravityAnchor,
) -> tuple[GravityObjectiveEvaluation, jnp.ndarray]:
    """Evaluate an anchored partial objective, exact at the anchor itself."""
    _validate_anchor(anchor, problem=problem)
    raw = np.asarray(raw_parameters)
    if raw.shape != anchor.raw_parameters.shape:
        raise ValueError("raw parameter shape differs from the anchor.")
    if np.array_equal(raw, anchor.raw_parameters):
        evaluation = _evaluation_from_mean(
            jnp.asarray(raw),
            mean=jnp.asarray(anchor.measurement_mean),
            demand=jnp.asarray(anchor.demand),
            problem=problem,
        )
        return evaluation, jnp.asarray(anchor.gradient)
    approximate_operator = ParallelApproximateRoutingOperator(
        problem.operator,
        executor,
        selection,
        anchor_demand=anchor.demand,
        anchor_routed_measurements=anchor.routed_measurements,
    )
    return gravity_value_and_gradient_adjoint(
        raw, problem=replace(problem, operator=approximate_operator)
    )
