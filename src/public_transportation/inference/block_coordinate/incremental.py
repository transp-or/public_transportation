"""Immutable algebra for incremental block-coordinate prediction updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from public_transportation.inference.linear_operator import LinearOperatorProtocol

from .blocks import ODBlock
from .operator import BlockLinearOperatorProtocol

Array = np.ndarray


def _owned_vector(
    value: object,
    *,
    name: str,
    length: int | None = None,
    require_finite: bool = True,
) -> Array:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")
    array = np.array(array, dtype=np.float64, copy=True)
    if array.ndim != 1 or (length is not None and array.shape != (length,)):
        expected = "one-dimensional" if length is None else f"shape ({length},)"
        raise ValueError(f"{name} must have {expected}, got {array.shape}.")
    if require_finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class IncrementalLinearState:
    """Current complete free flow and its cached measurement prediction."""

    free_flow: Array
    prediction: Array
    fixed_measurement_offset: Array

    def __post_init__(self) -> None:
        flow = _owned_vector(self.free_flow, name="free_flow")
        if np.any(flow < 0.0):
            raise ValueError("free_flow must be non-negative.")
        prediction = _owned_vector(self.prediction, name="prediction")
        offset = _owned_vector(
            self.fixed_measurement_offset,
            name="fixed_measurement_offset",
            length=prediction.size,
        )
        object.__setattr__(self, "free_flow", flow)
        object.__setattr__(self, "prediction", prediction)
        object.__setattr__(self, "fixed_measurement_offset", offset)


@dataclass(frozen=True, slots=True)
class BlockUpdateProposal:
    """Unapplied block change and its exact incremental prediction."""

    block_id: str
    block_fingerprint: str
    free_column_indices: tuple[int, ...]
    flow_before: Array
    flow_after: Array
    flow_delta: Array
    prediction_delta: Array
    trial_prediction: Array

    def __post_init__(self) -> None:
        columns = tuple(int(item) for item in self.free_column_indices)
        if columns != tuple(sorted(set(columns))):
            raise ValueError("free_column_indices must be unique and ascending.")
        before = _owned_vector(self.flow_before, name="flow_before", length=len(columns))
        after = _owned_vector(self.flow_after, name="flow_after", length=len(columns))
        delta = _owned_vector(self.flow_delta, name="flow_delta", length=len(columns))
        if not np.array_equal(delta, after - before):
            raise ValueError("flow_delta must equal flow_after - flow_before.")
        prediction_delta = _owned_vector(
            self.prediction_delta, name="prediction_delta"
        )
        trial_prediction = _owned_vector(
            self.trial_prediction,
            name="trial_prediction",
            length=prediction_delta.size,
        )
        object.__setattr__(self, "free_column_indices", columns)
        object.__setattr__(self, "flow_before", before)
        object.__setattr__(self, "flow_after", after)
        object.__setattr__(self, "flow_delta", delta)
        object.__setattr__(self, "prediction_delta", prediction_delta)
        object.__setattr__(self, "trial_prediction", trial_prediction)


@dataclass(frozen=True, slots=True)
class IncrementalPredictionValidation:
    """Comparison of cached and fully recomputed predictions."""

    within_tolerance: bool
    max_absolute_error: float
    relative_l2_error: float


def initialize_incremental_state(
    operator: LinearOperatorProtocol,
    free_flow: object,
    fixed_measurement_offset: object,
) -> IncrementalLinearState:
    """Build an incremental state from one complete forward product."""
    flow = _owned_vector(free_flow, name="free_flow", length=operator.shape[1])
    if np.any(flow < 0.0):
        raise ValueError("free_flow must be non-negative.")
    offset = _owned_vector(
        fixed_measurement_offset,
        name="fixed_measurement_offset",
        length=operator.shape[0],
    )
    prediction = np.asarray(operator.matvec(flow), dtype=np.float64) + offset
    return IncrementalLinearState(flow, prediction, offset)


def propose_incremental_update(
    state: IncrementalLinearState,
    block: ODBlock,
    block_operator: BlockLinearOperatorProtocol,
    trial_local_flow: object,
    *,
    lower_bounds: object | None = None,
    upper_bounds: object | None = None,
) -> BlockUpdateProposal:
    """Create a trial update without changing the source state."""
    columns = block.free_column_indices
    if block_operator.num_local_variables != len(columns):
        raise ValueError("block operator width does not match the block size.")
    if block_operator.num_measurements != state.prediction.size:
        raise ValueError("block operator height does not match the prediction size.")
    if columns and columns[-1] >= state.free_flow.size:
        raise ValueError("block contains a free-column index outside the state.")
    after = _owned_vector(trial_local_flow, name="trial_local_flow", length=len(columns))
    if np.any(after < 0.0):
        raise ValueError("trial_local_flow must be non-negative.")
    if lower_bounds is not None:
        lower = _owned_vector(
            lower_bounds,
            name="lower_bounds",
            length=len(columns),
            require_finite=False,
        )
        if np.any(np.isnan(lower)) or np.any(np.isposinf(lower)):
            raise ValueError("lower_bounds may not contain NaN or positive infinity.")
        if np.any(after < lower):
            raise ValueError("trial_local_flow violates lower_bounds.")
    if upper_bounds is not None:
        upper = _owned_vector(
            upper_bounds,
            name="upper_bounds",
            length=len(columns),
            require_finite=False,
        )
        if np.any(np.isnan(upper)) or np.any(np.isneginf(upper)):
            raise ValueError("upper_bounds may not contain NaN or negative infinity.")
        if lower_bounds is not None and np.any(lower > upper):
            raise ValueError("lower_bounds must not exceed upper_bounds.")
        if np.any(after > upper):
            raise ValueError("trial_local_flow violates upper_bounds.")
    indices = np.asarray(columns, dtype=np.intp)
    before = state.free_flow[indices]
    delta = after - before
    prediction_delta = np.asarray(block_operator.matvec(delta), dtype=np.float64)
    trial_prediction = state.prediction + prediction_delta
    return BlockUpdateProposal(
        block_id=block.block_id,
        block_fingerprint=block.fingerprint,
        free_column_indices=columns,
        flow_before=before,
        flow_after=after,
        flow_delta=delta,
        prediction_delta=prediction_delta,
        trial_prediction=trial_prediction,
    )


def apply_incremental_update(
    state: IncrementalLinearState, proposal: BlockUpdateProposal
) -> IncrementalLinearState:
    """Return a new state after verifying that a proposal is not stale."""
    if proposal.trial_prediction.shape != state.prediction.shape:
        raise ValueError("proposal prediction size does not match the state.")
    indices = np.asarray(proposal.free_column_indices, dtype=np.intp)
    if indices.size and indices[-1] >= state.free_flow.size:
        raise ValueError("proposal contains a free-column index outside the state.")
    if not np.array_equal(state.free_flow[indices], proposal.flow_before):
        raise ValueError("proposal is stale: block flow no longer matches flow_before.")
    if not np.array_equal(
        state.prediction + proposal.prediction_delta, proposal.trial_prediction
    ):
        raise ValueError("proposal trial_prediction is inconsistent with the state.")
    flow = np.array(state.free_flow, copy=True)
    flow[indices] = proposal.flow_after
    return IncrementalLinearState(
        free_flow=flow,
        prediction=proposal.trial_prediction,
        fixed_measurement_offset=state.fixed_measurement_offset,
    )


def block_data_gradient(
    block_operator: BlockLinearOperatorProtocol,
    prediction: object,
    observations: object,
    observation_weights: object,
) -> Array:
    """Return the weighted least-squares data gradient for one block."""
    count = block_operator.num_measurements
    predicted = _owned_vector(prediction, name="prediction", length=count)
    observed = _owned_vector(observations, name="observations", length=count)
    weights = _owned_vector(
        observation_weights, name="observation_weights", length=count
    )
    if np.any(weights < 0.0):
        raise ValueError("observation_weights must be non-negative.")
    return np.asarray(block_operator.rmatvec(weights * (predicted - observed)))


def validate_incremental_prediction(
    state: IncrementalLinearState,
    operator: LinearOperatorProtocol,
    *,
    absolute_tolerance: float = 1.0e-10,
    relative_tolerance: float = 1.0e-10,
) -> IncrementalPredictionValidation:
    """Detect numerical drift against a complete forward recomputation."""
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("prediction tolerances must be non-negative.")
    if operator.shape != (state.prediction.size, state.free_flow.size):
        raise ValueError("operator shape does not match the incremental state.")
    recomputed = np.asarray(operator.matvec(state.free_flow)) + state.fixed_measurement_offset
    difference = state.prediction - recomputed
    maximum = float(np.max(np.abs(difference))) if difference.size else 0.0
    denominator = max(float(np.linalg.norm(recomputed)), np.finfo(float).tiny)
    relative = float(np.linalg.norm(difference) / denominator)
    within = bool(
        np.allclose(
            state.prediction,
            recomputed,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
    )
    return IncrementalPredictionValidation(within, maximum, relative)
