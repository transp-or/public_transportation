"""Affine solver coordinates for fixed-routing linear least squares."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fixed_routing_linear_problem import FixedRoutingLinearProblem
from .fixed_routing_linear_regularization import (
    AugmentedLinearLeastSquaresSystem,
    build_augmented_linear_least_squares_system,
)
from .linear_operator import LinearOperatorProtocol

Array = np.ndarray


def _immutable_finite_vector(value: object, *, name: str, size: int) -> Array:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")
    array = np.asarray(array, dtype=np.result_type(array.dtype, np.float64))
    if array.ndim != 1 or array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class PhysicalDemandTransform:
    """Affine bijection ``x = prior_demand + scales * solver_variable``."""

    prior_demand: Array
    scales: Array
    lower_bounds: Array
    upper_bounds: Array

    def __post_init__(self) -> None:
        prior = np.asarray(self.prior_demand)
        if prior.ndim != 1:
            raise ValueError(
                f"prior_demand must be one-dimensional, got {prior.shape}."
            )
        size = prior.shape[0]
        prior = _immutable_finite_vector(prior, name="prior_demand", size=size)
        scales = _immutable_finite_vector(self.scales, name="scales", size=size)
        if np.any(scales <= 0.0):
            raise ValueError("scales must be strictly positive.")

        lower = np.asarray(self.lower_bounds)
        upper = np.asarray(self.upper_bounds)
        if lower.ndim != 1 or lower.shape != (size,):
            raise ValueError(
                f"lower_bounds must have shape ({size},), got {lower.shape}."
            )
        if upper.ndim != 1 or upper.shape != (size,):
            raise ValueError(
                f"upper_bounds must have shape ({size},), got {upper.shape}."
            )
        if lower.dtype.kind not in "iuf" or upper.dtype.kind not in "iuf":
            raise TypeError("bounds must contain real numeric values.")
        lower = np.array(lower, dtype=np.result_type(lower.dtype, np.float64), copy=True)
        upper = np.array(upper, dtype=np.result_type(upper.dtype, np.float64), copy=True)
        if np.any(np.isnan(lower)) or np.any(np.isposinf(lower)):
            raise ValueError("lower_bounds may be finite or -inf, but not NaN or +inf.")
        if np.any(np.isnan(upper)) or np.any(np.isneginf(upper)):
            raise ValueError("upper_bounds may be finite or +inf, but not NaN or -inf.")
        if np.any(lower > upper):
            raise ValueError("lower_bounds must not exceed upper_bounds.")
        if np.any(prior < lower) or np.any(prior > upper):
            raise ValueError("prior_demand must satisfy the physical demand bounds.")
        lower.setflags(write=False)
        upper.setflags(write=False)

        object.__setattr__(self, "prior_demand", prior)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)

    @classmethod
    def from_problem(
        cls, problem: FixedRoutingLinearProblem
    ) -> PhysicalDemandTransform:
        return cls(
            prior_demand=problem.prior_demand,
            scales=problem.variable_scales,
            lower_bounds=problem.lower_bounds,
            upper_bounds=problem.upper_bounds,
        )

    @property
    def size(self) -> int:
        return int(self.prior_demand.size)

    @property
    def solver_lower_bounds(self) -> Array:
        bounds = (self.lower_bounds - self.prior_demand) / self.scales
        bounds.setflags(write=False)
        return bounds

    @property
    def solver_upper_bounds(self) -> Array:
        bounds = (self.upper_bounds - self.prior_demand) / self.scales
        bounds.setflags(write=False)
        return bounds

    def demand_from_solver_variable(self, solver_variable: object) -> Array:
        value = _immutable_finite_vector(
            solver_variable, name="solver_variable", size=self.size
        )
        return self.prior_demand + self.scales * value

    def solver_variable_from_demand(self, demand: object) -> Array:
        value = _immutable_finite_vector(demand, name="demand", size=self.size)
        return (value - self.prior_demand) / self.scales

    def solver_gradient_from_physical(self, physical_gradient: object) -> Array:
        gradient = _immutable_finite_vector(
            physical_gradient, name="physical_gradient", size=self.size
        )
        return self.scales * gradient


@dataclass(frozen=True, slots=True)
class ColumnScaledLinearOperator:
    """Right-scaled operator representing ``base @ diag(scales)``."""

    base: LinearOperatorProtocol
    scales: Array

    def __post_init__(self) -> None:
        scales = _immutable_finite_vector(
            self.scales, name="scales", size=self.base.shape[1]
        )
        if np.any(scales <= 0.0):
            raise ValueError("scales must be strictly positive.")
        object.__setattr__(self, "scales", scales)

    @property
    def shape(self) -> tuple[int, int]:
        return self.base.shape

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(np.result_type(self.base.dtype, self.scales.dtype))

    def matvec(self, vector: object) -> Array:
        value = _immutable_finite_vector(
            vector, name="scaled forward vector", size=self.shape[1]
        )
        return self.base.matvec(self.scales * value)

    def rmatvec(self, vector: object) -> Array:
        return self.scales * self.base.rmatvec(vector)


@dataclass(frozen=True, slots=True)
class SolverVariableLeastSquaresSystem:
    """Augmented least-squares system expressed in solver variable ``d``."""

    operator: ColumnScaledLinearOperator
    target: Array
    lower_bounds: Array
    upper_bounds: Array
    transform: PhysicalDemandTransform
    physical_system: AugmentedLinearLeastSquaresSystem


def build_solver_variable_least_squares_system(
    problem: FixedRoutingLinearProblem,
    *,
    variable_scales: object | None = None,
) -> SolverVariableLeastSquaresSystem:
    """Transform ``A_aug x - b_aug`` using ``x = x0 + S d``."""
    transform = (
        PhysicalDemandTransform.from_problem(problem)
        if variable_scales is None
        else PhysicalDemandTransform(
            prior_demand=problem.prior_demand,
            scales=variable_scales,
            lower_bounds=problem.lower_bounds,
            upper_bounds=problem.upper_bounds,
        )
    )
    physical_system = build_augmented_linear_least_squares_system(problem)
    operator = ColumnScaledLinearOperator(
        base=physical_system.operator,
        scales=transform.scales,
    )
    target = (
        physical_system.target
        - physical_system.operator.matvec(transform.prior_demand)
    )
    target = np.array(target, copy=True)
    target.setflags(write=False)
    return SolverVariableLeastSquaresSystem(
        operator=operator,
        target=target,
        lower_bounds=transform.solver_lower_bounds,
        upper_bounds=transform.solver_upper_bounds,
        transform=transform,
        physical_system=physical_system,
    )
