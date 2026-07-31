"""Explicit regularization and augmented linear least-squares operators."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .fixed_routing_linear_objective import (
    LinearDataFitEvaluation,
    evaluate_linear_data_fit,
)
from .fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    LinearRegularizationBlock,
)
from .linear_operator import SparseLinearOperator

Array = np.ndarray


def _finite_vector(value: object, *, name: str, size: int | None = None) -> Array:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")
    array = np.asarray(array, dtype=np.result_type(array.dtype, np.float64))
    if array.ndim != 1 or (size is not None and array.shape != (size,)):
        expected = "one-dimensional" if size is None else f"shape ({size},)"
        raise ValueError(f"{name} must have {expected}, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def ridge_to_prior(
    prior_demand: object, *, strength: float, name: str = "ridge_to_prior"
) -> LinearRegularizationBlock:
    """Return ``0.5 * strength * ||x - prior||**2``."""
    prior = _finite_vector(prior_demand, name="prior_demand")
    operator = SparseLinearOperator(sparse.eye(prior.size, format="csr"))
    return LinearRegularizationBlock(name, operator, prior, strength)


def scaled_ridge_to_prior(
    prior_demand: object,
    scales: object,
    *,
    strength: float,
    name: str = "scaled_ridge_to_prior",
) -> LinearRegularizationBlock:
    """Return ``0.5 * strength * ||S^-1 (x - prior)||**2``."""
    prior = _finite_vector(prior_demand, name="prior_demand")
    scale = _finite_vector(scales, name="scales", size=prior.size)
    if np.any(scale <= 0.0):
        raise ValueError("scales must be strictly positive.")
    inverse_scale = 1.0 / scale
    operator = SparseLinearOperator(
        sparse.diags(inverse_scale, format="csr")
    )
    return LinearRegularizationBlock(
        name,
        operator,
        inverse_scale * prior,
        strength,
    )


@dataclass(frozen=True, slots=True)
class RegularizationBlockEvaluation:
    name: str
    residual: Array
    objective: float
    gradient: Array


@dataclass(frozen=True, slots=True)
class LinearLeastSquaresEvaluation:
    data_fit: LinearDataFitEvaluation
    regularization: tuple[RegularizationBlockEvaluation, ...]
    objective: float
    gradient: Array
    augmented_residual: Array


@dataclass(frozen=True, slots=True)
class AugmentedLinearLeastSquaresOperator:
    """Stack weighted measurements and regularization without assembly."""

    problem: FixedRoutingLinearProblem

    def __post_init__(self) -> None:
        _require_explicit_regularization(self.problem)

    @property
    def shape(self) -> tuple[int, int]:
        rows = self.problem.num_measurements + sum(
            block.operator.shape[0] for block in self.problem.regularization_blocks
        )
        return rows, self.problem.num_free_od

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(
            np.result_type(
                self.problem.measurement_operator.dtype,
                self.problem.observation_weights.dtype,
                *(block.operator.dtype for block in self.problem.regularization_blocks),
            )
        )

    def matvec(self, vector: object) -> Array:
        data = np.sqrt(self.problem.observation_weights) * (
            self.problem.measurement_operator.matvec(vector)
        )
        values = [data]
        for block in self.problem.regularization_blocks:
            values.append(math.sqrt(block.strength) * block.operator.matvec(vector))
        return np.concatenate(values)

    def rmatvec(self, vector: object) -> Array:
        value = _finite_vector(vector, name="augmented transpose vector")
        if value.shape != (self.shape[0],):
            raise ValueError(
                "augmented transpose vector must have shape "
                f"({self.shape[0]},), got {value.shape}."
            )
        stop = self.problem.num_measurements
        gradient = self.problem.measurement_operator.rmatvec(
            np.sqrt(self.problem.observation_weights) * value[:stop]
        )
        for block in self.problem.regularization_blocks:
            start = stop
            stop += block.operator.shape[0]
            gradient = gradient + math.sqrt(block.strength) * block.operator.rmatvec(
                value[start:stop]
            )
        return np.asarray(gradient)


@dataclass(frozen=True, slots=True)
class AugmentedLinearLeastSquaresSystem:
    """Operator and target for ``0.5 * ||A_aug x - b_aug||**2``."""

    operator: AugmentedLinearLeastSquaresOperator
    target: Array
    data_slice: slice
    regularization_slices: tuple[tuple[str, slice], ...]


def _require_explicit_regularization(problem: FixedRoutingLinearProblem) -> None:
    if problem.regularization_selection == "unspecified":
        raise ValueError(
            "regularization selection is unspecified; explicitly select 'none' "
            "or configure regularization blocks."
        )


def build_augmented_linear_least_squares_system(
    problem: FixedRoutingLinearProblem,
) -> AugmentedLinearLeastSquaresSystem:
    """Build the augmented operator and target without forming a block matrix."""
    operator = AugmentedLinearLeastSquaresOperator(problem)
    targets = [
        np.sqrt(problem.observation_weights)
        * (problem.observations - problem.fixed_measurement_offset)
    ]
    slices: list[tuple[str, slice]] = []
    stop = problem.num_measurements
    for block in problem.regularization_blocks:
        start = stop
        stop += block.operator.shape[0]
        targets.append(math.sqrt(block.strength) * block.target)
        slices.append((block.name, slice(start, stop)))
    target = np.concatenate(targets)
    target.setflags(write=False)
    return AugmentedLinearLeastSquaresSystem(
        operator=operator,
        target=target,
        data_slice=slice(0, problem.num_measurements),
        regularization_slices=tuple(slices),
    )


def evaluate_regularization_block(
    block: LinearRegularizationBlock, demand: object
) -> RegularizationBlockEvaluation:
    """Evaluate one explicitly configured regularization block."""
    unscaled = block.operator.matvec(demand) - block.target
    residual = math.sqrt(block.strength) * unscaled
    objective = float(0.5 * np.vdot(residual, residual))
    gradient = block.strength * block.operator.rmatvec(unscaled)
    residual = np.array(residual, copy=True)
    gradient = np.array(gradient, copy=True)
    residual.setflags(write=False)
    gradient.setflags(write=False)
    return RegularizationBlockEvaluation(
        name=block.name,
        residual=residual,
        objective=objective,
        gradient=gradient,
    )


def evaluate_linear_least_squares(
    problem: FixedRoutingLinearProblem, demand: object
) -> LinearLeastSquaresEvaluation:
    """Evaluate the complete explicitly selected least-squares objective."""
    _require_explicit_regularization(problem)
    data_fit = evaluate_linear_data_fit(problem, demand)
    regularization = tuple(
        evaluate_regularization_block(block, demand)
        for block in problem.regularization_blocks
    )
    objective = data_fit.objective + sum(item.objective for item in regularization)
    gradient = np.array(data_fit.gradient, copy=True)
    for item in regularization:
        gradient += item.gradient
    augmented_residual = np.concatenate(
        [data_fit.weighted_residual]
        + [item.residual for item in regularization]
    )
    gradient.setflags(write=False)
    augmented_residual.setflags(write=False)
    return LinearLeastSquaresEvaluation(
        data_fit=data_fit,
        regularization=regularization,
        objective=float(objective),
        gradient=gradient,
        augmented_residual=augmented_residual,
    )
