"""Shared raw-to-effective parameter transformations for inference engines."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np


def smooth_bound(x: Any, bound: float) -> jnp.ndarray:
    """Map unconstrained values smoothly into ``(-bound, bound)``."""
    x_array = jnp.asarray(x)
    bound_array = jnp.asarray(bound, dtype=x_array.dtype)
    return bound_array * jnp.tanh(x_array / bound_array)


def smooth_bound_numpy(x: Any, bound: float) -> np.ndarray:
    """NumPy equivalent of :func:`smooth_bound` for result processing."""
    return float(bound) * np.tanh(np.asarray(x, dtype=float) / float(bound))


def raw_value_for_effective_center(effective_value: float, bound: float) -> float:
    """Invert :func:`smooth_bound` for a scalar prior center."""
    effective = float(effective_value)
    bound_value = float(bound)
    if not np.isfinite(bound_value) or bound_value <= 0.0:
        raise ValueError(f"bound must be positive and finite, got {bound_value!r}")
    if not np.isfinite(effective) or abs(effective) >= bound_value:
        raise ValueError(
            "effective_value must be finite and strictly inside (-bound, bound); "
            f"got effective_value={effective!r}, bound={bound_value!r}."
        )
    return float(bound_value * np.arctanh(effective / bound_value))
