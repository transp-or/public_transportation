"""Two-pass destination streaming for matrix-free measurement objectives.

The nonlinear measurement objective cannot be evaluated independently for
partial destination flows. This module first accumulates the complete
prediction and then applies its cotangent to each destination in a separate
VJP. Consequently no global OD-to-measurement Jacobian or global reverse-mode
tape is constructed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


Array = jax.Array
GroupPredictor = Callable[[Array], Array]
MeasurementObjective = Callable[[Array], Array]


@dataclass(frozen=True, slots=True)
class StreamedDestinationGroup:
    """Local free-parameter coordinates and their measurement predictor."""

    free_parameter_indices: np.ndarray
    predict_measurements: GroupPredictor
    persistent_bytes: int = 0
    label: str = ""


@dataclass(frozen=True, slots=True)
class StreamedValueAndGradient:
    """Result of the exact two-pass streamed calculation."""

    value: Array
    prediction: Array
    measurement_cotangent: Array
    gradient: Array
    num_groups: int
    maximum_group_bytes: int = 0


GroupProvider = Callable[[], Iterable[StreamedDestinationGroup]]


def _validate_group(
    *,
    group: StreamedDestinationGroup,
    group_number: int,
    parameter_size: int,
    occupied: np.ndarray | None,
    memory_ceiling_bytes: int | None,
) -> np.ndarray:
    indices = np.asarray(group.free_parameter_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(
            f"Group {group_number} free_parameter_indices must be nonempty and 1D."
        )
    if np.any(indices < 0) or np.any(indices >= parameter_size):
        raise ValueError(f"Group {group_number} contains an out-of-range index.")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"Group {group_number} contains duplicate indices.")
    if occupied is not None:
        if np.any(occupied[indices]):
            raise ValueError("Free parameter indices must be unique across groups.")
        occupied[indices] = True
    group_bytes = int(group.persistent_bytes)
    if group_bytes < 0:
        raise ValueError(f"Group {group_number} persistent_bytes must be nonnegative.")
    if memory_ceiling_bytes is not None and group_bytes > memory_ceiling_bytes:
        label = f" ({group.label})" if group.label else ""
        raise MemoryError(
            f"Group {group_number}{label} requires {group_bytes} persistent bytes, "
            f"above the {memory_ceiling_bytes}-byte ceiling."
        )
    return indices


def streamed_measurement_value_and_grad(
    *,
    parameter: Array,
    groups: Sequence[StreamedDestinationGroup],
    num_measurements: int,
    measurement_objective: MeasurementObjective,
) -> StreamedValueAndGradient:
    """Evaluate an exact measurement objective and gradient in two passes.

    Each predictor is evaluated once while accumulating the complete predicted
    measurement vector, then recomputed independently to apply the common
    measurement-space cotangent. The returned gradient contains only the
    measurement-objective contribution; parameter priors can be added
    separately.
    """
    parameter_array = jnp.asarray(parameter)
    if parameter_array.ndim != 1:
        raise ValueError("parameter must be one-dimensional.")
    if num_measurements < 0:
        raise ValueError("num_measurements must be nonnegative.")

    normalized: list[tuple[Array, GroupPredictor]] = []
    occupied = np.zeros(parameter_array.shape[0], dtype=bool)
    prediction = jnp.zeros((num_measurements,), dtype=parameter_array.dtype)
    for group_number, group in enumerate(groups):
        indices_np = _validate_group(
            group=group,
            group_number=group_number,
            parameter_size=parameter_array.shape[0],
            occupied=occupied,
            memory_ceiling_bytes=None,
        )
        indices = jnp.asarray(indices_np, dtype=jnp.int32)
        local_prediction = jnp.asarray(
            group.predict_measurements(parameter_array[indices])
        )
        if local_prediction.shape != (num_measurements,):
            raise ValueError(
                f"Group {group_number} prediction must have shape "
                f"({num_measurements},), got {local_prediction.shape}."
            )
        if local_prediction.dtype != parameter_array.dtype:
            raise ValueError(
                f"Group {group_number} prediction dtype {local_prediction.dtype} "
                f"does not match parameter dtype {parameter_array.dtype}."
            )
        prediction = prediction + local_prediction
        normalized.append((indices, group.predict_measurements))

    value, cotangent = jax.value_and_grad(measurement_objective)(prediction)
    if jnp.shape(value) != ():
        raise ValueError("measurement_objective must return a scalar.")

    gradient = jnp.zeros_like(parameter_array)
    for indices, predictor in normalized:
        _, pullback = jax.vjp(predictor, parameter_array[indices])
        (local_gradient,) = pullback(cotangent)
        gradient = gradient.at[indices].add(local_gradient)

    return StreamedValueAndGradient(
        value=value,
        prediction=prediction,
        measurement_cotangent=cotangent,
        gradient=gradient,
        num_groups=len(normalized),
        maximum_group_bytes=max(
            (int(group.persistent_bytes) for group in groups), default=0
        ),
    )


def replayable_streamed_measurement_value_and_grad(
    *,
    parameter: Array,
    group_provider: GroupProvider,
    num_measurements: int,
    measurement_objective: MeasurementObjective,
    memory_ceiling_bytes: int | None = None,
) -> StreamedValueAndGradient:
    """Stream two replayable passes while retaining no destination predictor.

    The provider must return the same group indices in the same order on every
    call. Only compact index signatures survive the first pass. Destination
    routing arrays captured by a yielded predictor can therefore be released
    before the next group is requested.
    """
    parameter_array = jnp.asarray(parameter)
    if parameter_array.ndim != 1:
        raise ValueError("parameter must be one-dimensional.")
    if num_measurements < 0:
        raise ValueError("num_measurements must be nonnegative.")
    if memory_ceiling_bytes is not None and memory_ceiling_bytes <= 0:
        raise ValueError("memory_ceiling_bytes must be positive when provided.")

    occupied = np.zeros(parameter_array.shape[0], dtype=bool)
    signatures: list[np.ndarray] = []
    prediction = jnp.zeros((num_measurements,), dtype=parameter_array.dtype)
    maximum_group_bytes = 0
    for group_number, group in enumerate(group_provider()):
        indices_np = _validate_group(
            group=group,
            group_number=group_number,
            parameter_size=parameter_array.shape[0],
            occupied=occupied,
            memory_ceiling_bytes=memory_ceiling_bytes,
        )
        local_prediction = jnp.asarray(
            group.predict_measurements(
                parameter_array[jnp.asarray(indices_np, dtype=jnp.int32)]
            )
        )
        if local_prediction.shape != (num_measurements,):
            raise ValueError(
                f"Group {group_number} prediction must have shape "
                f"({num_measurements},), got {local_prediction.shape}."
            )
        if local_prediction.dtype != parameter_array.dtype:
            raise ValueError(
                f"Group {group_number} prediction dtype {local_prediction.dtype} "
                f"does not match parameter dtype {parameter_array.dtype}."
            )
        prediction = prediction + local_prediction
        signatures.append(np.array(indices_np, dtype=np.int64, copy=True))
        maximum_group_bytes = max(maximum_group_bytes, int(group.persistent_bytes))

    value, cotangent = jax.value_and_grad(measurement_objective)(prediction)
    if jnp.shape(value) != ():
        raise ValueError("measurement_objective must return a scalar.")

    gradient = jnp.zeros_like(parameter_array)
    second_count = 0
    for group_number, group in enumerate(group_provider()):
        if group_number >= len(signatures):
            raise ValueError("Group provider yielded extra groups during replay.")
        indices_np = _validate_group(
            group=group,
            group_number=group_number,
            parameter_size=parameter_array.shape[0],
            occupied=None,
            memory_ceiling_bytes=memory_ceiling_bytes,
        )
        if not np.array_equal(indices_np, signatures[group_number]):
            raise ValueError(
                f"Group provider changed indices or ordering at group {group_number}."
            )
        indices = jnp.asarray(indices_np, dtype=jnp.int32)
        _, pullback = jax.vjp(
            group.predict_measurements, parameter_array[indices]
        )
        (local_gradient,) = pullback(cotangent)
        gradient = gradient.at[indices].add(local_gradient)
        second_count += 1
    if second_count != len(signatures):
        raise ValueError("Group provider yielded fewer groups during replay.")

    return StreamedValueAndGradient(
        value=value,
        prediction=prediction,
        measurement_cotangent=cotangent,
        gradient=gradient,
        num_groups=len(signatures),
        maximum_group_bytes=maximum_group_bytes,
    )
