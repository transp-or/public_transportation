from __future__ import annotations

from typing import Any
import jax.numpy as jnp

from .spec import AggregationSpec


def apply_mapping_spec(*, link_flow: Any, spec: AggregationSpec) -> jnp.ndarray:
    """Compute y_pred from link_flow using an AggregationSpec (JAX-safe)."""
    lf = jnp.asarray(link_flow)
    mi = jnp.asarray(spec.measurement_index, dtype=jnp.int32)
    li = jnp.asarray(spec.link_index, dtype=jnp.int32)

    contrib = lf[li]
    y = jnp.zeros((int(spec.num_measurements),), dtype=contrib.dtype)
    y = y.at[mi].add(contrib)
    return y