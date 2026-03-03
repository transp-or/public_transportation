"""
public_transportation.measurement.aggregation_numpy

NumPy helpers to aggregate assignment link flows into measurement-space using AggregationSpec.

This is the non-JAX sibling of `public_transportation.measurement.likelihood_jax`.
It is intended for post-processing and reporting (pure Python/NumPy).

Key operation
-------------
Given:
- link_flow[ℓ], ℓ=0..L-1
- spec describes K contributions (k=0..K-1):
    measurement_index[k] = m
    link_index[k] = ℓ
Compute:
- lambda_m[m] = sum_{k: measurement_index[k]=m} link_flow[ link_index[k] ]
"""

from __future__ import annotations

from typing import Any
import numpy as np

from public_transportation.measurement.mapping import AggregationSpec


def aggregate_link_flow_to_measurements(
    *,
    link_flow: np.ndarray,
    spec: AggregationSpec,
) -> np.ndarray:
    """Aggregate link flows into predicted measurements λ (before detection).

    Parameters
    ----------
    link_flow:
        Array shape (num_links,).
    spec:
        AggregationSpec with:
        - num_measurements (M)
        - measurement_index shape (K,)
        - link_index shape (K,)

    Returns
    -------
    np.ndarray
        lambda_m shape (M,)
    """
    lf = np.asarray(link_flow, dtype=float).reshape(-1)

    m = int(spec.num_measurements)
    mi = np.asarray(spec.measurement_index, dtype=np.int64).reshape(-1)
    li = np.asarray(spec.link_index, dtype=np.int64).reshape(-1)

    if mi.shape != li.shape:
        raise ValueError(f"spec.measurement_index and spec.link_index must match shape, got {mi.shape} vs {li.shape}")
    if m <= 0:
        raise ValueError(f"spec.num_measurements must be positive, got {m}")
    if mi.size == 0:
        return np.zeros((m,), dtype=float)

    if np.any(mi < 0) or np.any(mi >= m):
        bad = mi[(mi < 0) | (mi >= m)]
        raise ValueError(f"spec.measurement_index has out-of-range entries (showing up to 10): {bad[:10]!r}")

    if np.any(li < 0) or np.any(li >= lf.shape[0]):
        bad = li[(li < 0) | (li >= lf.shape[0])]
        raise ValueError(
            "spec.link_index has out-of-range entries for link_flow "
            f"(num_links={lf.shape[0]}). Bad entries (up to 10): {bad[:10]!r}"
        )

    out = np.zeros((m,), dtype=float)
    # out[mi[k]] += lf[li[k]]
    np.add.at(out, mi, lf[li])
    return out


def apply_detection_rate(*, lambda_m: np.ndarray, rho: float) -> np.ndarray:
    """Compute μ = rho * λ (rho typically in (0,1], but not enforced here)."""
    lam = np.asarray(lambda_m, dtype=float).reshape(-1)
    return float(rho) * lam


def as_1d_float(x: Any, *, name: str) -> np.ndarray:
    """Utility to coerce to 1D float array (used by reports)."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr