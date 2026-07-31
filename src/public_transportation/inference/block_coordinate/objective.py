"""Conditional objectives and projected gradients for OD blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
)
from public_transportation.inference.linear_operator import (
    DenseLinearOperator,
    SparseLinearOperator,
)

from .blocks import ODBlock
from .incremental import IncrementalLinearState
from .operator import BlockLinearOperatorProtocol
from .results import BlockObjectiveComponents

Array = np.ndarray


class UnsupportedConditionalPriorError(ValueError):
    """Raised when a prior cannot be evaluated correctly one block at a time."""


def _immutable_vector(
    value: object, *, name: str, size: int, finite: bool = True
) -> Array:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")
    array = np.array(array, dtype=np.float64, copy=True)
    if array.ndim != 1 or array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class SeparableQuadraticPrior:
    """Canonical prior ``0.5 * q*x**2 - h*x + constant``."""

    quadratic: Array
    linear: Array
    constant: float
    source_block_names: tuple[str, ...]

    def __post_init__(self) -> None:
        quadratic = _immutable_vector(
            self.quadratic, name="quadratic", size=np.asarray(self.quadratic).size
        )
        linear = _immutable_vector(
            self.linear, name="linear", size=quadratic.size
        )
        if np.any(quadratic < 0.0):
            raise ValueError("quadratic coefficients must be non-negative.")
        if not np.isfinite(self.constant) or self.constant < 0.0:
            raise ValueError("prior constant must be finite and non-negative.")
        names = tuple(str(name).strip() for name in self.source_block_names)
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("source_block_names must be nonempty and unique.")
        object.__setattr__(self, "quadratic", quadratic)
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "source_block_names", names)

    @property
    def num_variables(self) -> int:
        return self.quadratic.size

    def objective(self, flow: object) -> float:
        value = _immutable_vector(flow, name="flow", size=self.num_variables)
        return float(
            0.5 * np.dot(self.quadratic, value * value)
            - np.dot(self.linear, value)
            + self.constant
        )

    def gradient(self, flow: object) -> Array:
        value = _immutable_vector(flow, name="flow", size=self.num_variables)
        return np.asarray(self.quadratic * value - self.linear)


def _regularization_matrix(block: object, *, name: str) -> sparse.csr_array:
    if isinstance(block, SparseLinearOperator):
        return sparse.csr_array(block.matrix, copy=False)
    if isinstance(block, DenseLinearOperator):
        return sparse.csr_array(block.matrix)
    raise UnsupportedConditionalPriorError(
        f"regularization block {name!r} uses an operator whose separability "
        "cannot be proven; use ridge/scaled-ridge or a supported explicit "
        "dense/sparse separable operator"
    )


def prepare_separable_quadratic_prior(
    problem: FixedRoutingLinearProblem,
) -> SeparableQuadraticPrior:
    """Compile supported regularization blocks into a separable quadratic."""
    if problem.regularization_selection == "unspecified":
        raise ValueError(
            "regularization selection is unspecified; explicitly select 'none' "
            "or configure regularization blocks."
        )
    quadratic = np.zeros(problem.num_free_od, dtype=np.float64)
    linear = np.zeros(problem.num_free_od, dtype=np.float64)
    constant = 0.0
    names: list[str] = []
    for block in problem.regularization_blocks:
        names.append(block.name)
        if block.strength == 0.0:
            continue
        matrix = _regularization_matrix(block.operator, name=block.name)
        for row in range(matrix.shape[0]):
            start = matrix.indptr[row]
            stop = matrix.indptr[row + 1]
            columns = matrix.indices[start:stop]
            values = matrix.data[start:stop]
            nonzero = values != 0.0
            columns = columns[nonzero]
            values = values[nonzero]
            if columns.size > 1:
                raise UnsupportedConditionalPriorError(
                    f"regularization block {block.name!r} couples multiple OD "
                    f"variables in row {row}; conditional block evaluation is "
                    "not supported for this prior"
                )
            target = float(block.target[row])
            constant += 0.5 * block.strength * target * target
            if columns.size == 1:
                column = int(columns[0])
                coefficient = float(values[0])
                quadratic[column] += block.strength * coefficient * coefficient
                linear[column] += block.strength * coefficient * target
    return SeparableQuadraticPrior(quadratic, linear, constant, tuple(names))


@dataclass(frozen=True, slots=True)
class BlockObjectiveEvaluation:
    """Global objective and local first-order information at one block value."""

    components: BlockObjectiveComponents
    prediction: Array
    gradient: Array
    projected_gradient: Array
    projected_gradient_norm: float

    def __post_init__(self) -> None:
        prediction = _immutable_vector(
            self.prediction,
            name="prediction",
            size=np.asarray(self.prediction).size,
        )
        gradient = _immutable_vector(
            self.gradient, name="gradient", size=np.asarray(self.gradient).size
        )
        projected = _immutable_vector(
            self.projected_gradient,
            name="projected_gradient",
            size=gradient.size,
        )
        if not np.isfinite(self.projected_gradient_norm) or self.projected_gradient_norm < 0:
            raise ValueError("projected_gradient_norm must be finite and non-negative.")
        if not np.isclose(self.projected_gradient_norm, np.linalg.norm(projected)):
            raise ValueError("projected_gradient_norm is inconsistent with the vector.")
        object.__setattr__(self, "prediction", prediction)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "projected_gradient", projected)

    @property
    def objective(self) -> float:
        return self.components.total


def projected_gradient(
    flow: object,
    gradient: object,
    lower_bounds: object,
    upper_bounds: object,
    *,
    bound_tolerance: float = 0.0,
) -> Array:
    """Return the gradient projected onto feasible bound directions."""
    if not np.isfinite(bound_tolerance) or bound_tolerance < 0.0:
        raise ValueError("bound_tolerance must be finite and non-negative.")
    size = np.asarray(flow).size
    value = _immutable_vector(flow, name="flow", size=size)
    derivative = _immutable_vector(gradient, name="gradient", size=size)
    lower = _immutable_vector(
        lower_bounds, name="lower_bounds", size=size, finite=False
    )
    upper = _immutable_vector(
        upper_bounds, name="upper_bounds", size=size, finite=False
    )
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or np.any(lower > upper):
        raise ValueError("bounds must be ordered and may not contain NaN.")
    if np.any(value < lower - bound_tolerance) or np.any(
        value > upper + bound_tolerance
    ):
        raise ValueError("flow violates its bounds.")
    result = np.array(derivative, copy=True)
    at_lower = value <= lower + bound_tolerance
    at_upper = value >= upper - bound_tolerance
    result[at_lower & (derivative > 0.0)] = 0.0
    result[at_upper & (derivative < 0.0)] = 0.0
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ConditionalBlockObjective:
    """Exact global objective as a function of one block's local flow."""

    block: ODBlock
    block_operator: BlockLinearOperatorProtocol
    observations: Array
    observation_weights: Array
    prediction_without_block: Array
    prior_quadratic: Array
    prior_linear: Array
    prior_objective_outside_block: float
    lower_bounds: Array
    upper_bounds: Array

    def __post_init__(self) -> None:
        local = self.block.num_free_variables
        measurements = self.block_operator.num_measurements
        if self.block_operator.num_local_variables != local:
            raise ValueError("block operator width does not match the block.")
        for field_name in ("observations", "observation_weights", "prediction_without_block"):
            object.__setattr__(
                self,
                field_name,
                _immutable_vector(
                    getattr(self, field_name), name=field_name, size=measurements
                ),
            )
        for field_name in ("prior_quadratic", "prior_linear"):
            object.__setattr__(
                self,
                field_name,
                _immutable_vector(getattr(self, field_name), name=field_name, size=local),
            )
        for field_name in ("lower_bounds", "upper_bounds"):
            object.__setattr__(
                self,
                field_name,
                _immutable_vector(
                    getattr(self, field_name), name=field_name, size=local, finite=False
                ),
            )
        if np.any(self.observation_weights <= 0.0):
            raise ValueError("observation_weights must be strictly positive.")
        if not np.isfinite(self.prior_objective_outside_block):
            raise ValueError("prior_objective_outside_block must be finite.")
        if np.any(self.lower_bounds > self.upper_bounds):
            raise ValueError("lower_bounds must not exceed upper_bounds.")

    def prediction(self, local_flow: object) -> Array:
        value = _immutable_vector(
            local_flow, name="local_flow", size=self.block.num_free_variables
        )
        return self.prediction_without_block + self.block_operator.matvec(value)

    def evaluate(self, local_flow: object) -> BlockObjectiveEvaluation:
        value = _immutable_vector(
            local_flow, name="local_flow", size=self.block.num_free_variables
        )
        prediction = self.prediction_without_block + self.block_operator.matvec(value)
        residual = prediction - self.observations
        data_objective = float(0.5 * np.dot(self.observation_weights, residual * residual))
        prior_objective = float(
            self.prior_objective_outside_block
            + 0.5 * np.dot(self.prior_quadratic, value * value)
            - np.dot(self.prior_linear, value)
        )
        gradient = self.block_operator.rmatvec(self.observation_weights * residual)
        gradient = np.asarray(gradient) + self.prior_quadratic * value - self.prior_linear
        projected = projected_gradient(
            value, gradient, self.lower_bounds, self.upper_bounds
        )
        return BlockObjectiveEvaluation(
            components=BlockObjectiveComponents(data_objective, prior_objective),
            prediction=prediction,
            gradient=gradient,
            projected_gradient=projected,
            projected_gradient_norm=float(np.linalg.norm(projected)),
        )


def build_conditional_block_objective(
    problem: FixedRoutingLinearProblem,
    prior: SeparableQuadraticPrior,
    state: IncrementalLinearState,
    block: ODBlock,
    block_operator: BlockLinearOperatorProtocol,
) -> ConditionalBlockObjective:
    """Build an exact conditional objective while holding other blocks fixed."""
    if prior.num_variables != problem.num_free_od:
        raise ValueError("prior dimension does not match the problem.")
    if state.free_flow.size != problem.num_free_od:
        raise ValueError("state flow dimension does not match the problem.")
    if state.prediction.size != problem.num_measurements:
        raise ValueError("state prediction dimension does not match the problem.")
    columns = np.asarray(block.free_column_indices, dtype=np.intp)
    if columns[-1] >= problem.num_free_od:
        raise ValueError("block contains a column outside the problem.")
    current_local = state.free_flow[columns]
    prediction_without = state.prediction - block_operator.matvec(current_local)
    local_prior = float(
        0.5 * np.dot(prior.quadratic[columns], current_local * current_local)
        - np.dot(prior.linear[columns], current_local)
    )
    prior_outside = prior.objective(state.free_flow) - local_prior
    return ConditionalBlockObjective(
        block=block,
        block_operator=block_operator,
        observations=problem.observations,
        observation_weights=problem.observation_weights,
        prediction_without_block=prediction_without,
        prior_quadratic=prior.quadratic[columns],
        prior_linear=prior.linear[columns],
        prior_objective_outside_block=prior_outside,
        lower_bounds=problem.lower_bounds[columns],
        upper_bounds=problem.upper_bounds[columns],
    )
