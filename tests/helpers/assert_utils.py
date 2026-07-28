from __future__ import annotations

from typing import Any

import numpy as np


def assert_all_finite(x: Any, *, name: str = "array") -> None:
    """
    Assert that all values in an array-like object are finite.

    Works with numpy arrays, Python sequences, and (optionally) JAX arrays.

    :param x: Array-like object.
    :param name: Name used in assertion messages.
    :raises AssertionError: If NaN or inf values are present.
    """
    arr = np.asarray(x)
    if not np.all(np.isfinite(arr)):
        bad = np.where(~np.isfinite(arr))
        raise AssertionError(f"{name} contains non-finite values at indices {bad}.")


def assert_nonnegative(x: Any, *, name: str = "array", tol: float = 0.0) -> None:
    """
    Assert that all values are >= -tol.

    :param x: Array-like object.
    :param name: Name used in assertion messages.
    :param tol: Allowed small negative tolerance.
    :raises AssertionError: If values smaller than -tol are present.
    """
    arr = np.asarray(x)
    min_val = float(arr.min()) if arr.size else 0.0
    if min_val < -tol:
        raise AssertionError(f"{name} has negative values: min={min_val} < -{tol}.")


def assert_shape(x: Any, shape: tuple[int, ...], *, name: str = "array") -> None:
    """
    Assert that an array has a specific shape.

    :param x: Array-like object.
    :param shape: Expected shape.
    :param name: Name used in assertion messages.
    :raises AssertionError: If shape does not match.
    """
    arr = np.asarray(x)
    if arr.shape != shape:
        raise AssertionError(f"{name} has shape {arr.shape}, expected {shape}.")


def assert_indices_in_range(
    idx: Any,
    *,
    lo: int,
    hi: int,
    name: str = "indices",
) -> None:
    """
    Assert that all indices satisfy lo <= idx < hi.

    :param idx: Array-like indices.
    :param lo: Inclusive lower bound.
    :param hi: Exclusive upper bound.
    :param name: Name used in assertion messages.
    :raises AssertionError: If any index is out of range.
    """
    arr = np.asarray(idx, dtype=int)
    if arr.size == 0:
        return
    bad_lo = np.where(arr < lo)[0]
    bad_hi = np.where(arr >= hi)[0]
    if bad_lo.size or bad_hi.size:
        example = None
        if bad_lo.size:
            example = int(arr[bad_lo[0]])
        elif bad_hi.size:
            example = int(arr[bad_hi[0]])
        raise AssertionError(
            f"{name} contains out-of-range values. Expected {lo} <= idx < {hi}. "
            f"Example offending value: {example}."
        )


def assert_strictly_increasing_along_links(
    node_time: Any,
    tail: Any,
    head: Any,
    *,
    name: str = "time-expanded graph",
) -> None:
    """
    Assert that time increases strictly along each directed link.

    This is a key invariant for the acyclic time-expanded network assumption.

    :param node_time: Array of node times (e.g., minutes from midnight), shape (num_nodes,).
    :param tail: Tail node indices for links, shape (num_links,).
    :param head: Head node indices for links, shape (num_links,).
    :param name: Name used in assertion messages.
    :raises AssertionError: If any link does not increase time strictly.
    """
    t = np.asarray(node_time, dtype=float)
    u = np.asarray(tail, dtype=int)
    v = np.asarray(head, dtype=int)

    if u.shape != v.shape:
        raise AssertionError(f"{name}: tail/head shape mismatch: {u.shape} vs {v.shape}")

    if u.size == 0:
        return

    dt = t[v] - t[u]
    bad = np.where(dt <= 0.0)[0]
    if bad.size:
        i = int(bad[0])
        raise AssertionError(
            f"{name}: non-increasing time on link {i}: "
            f"time[head]={t[v[i]]} - time[tail]={t[u[i]]} = {dt[i]} (must be > 0)."
        )


def assert_row_stochastic(
    probs: Any,
    *,
    axis: int = -1,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    name: str = "probs",
    mask: Any | None = None,
) -> None:
    """
    Assert that probabilities sum to 1 along an axis and are nonnegative.

    Optionally supports masking padded entries (mask=True means valid).

    :param probs: Probability array.
    :param axis: Axis over which probabilities should sum to 1.
    :param atol: Absolute tolerance.
    :param rtol: Relative tolerance.
    :param name: Name used in assertion messages.
    :param mask: Optional boolean mask with same shape as probs.
    :raises AssertionError: If constraints are violated.
    """
    p = np.asarray(probs, dtype=float)

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != p.shape:
            raise AssertionError(f"{name}: mask shape {m.shape} does not match probs shape {p.shape}.")
        p = np.where(m, p, 0.0)

    if np.any(p < -atol):
        raise AssertionError(f"{name}: contains negative probabilities (min={p.min()}).")

    s = p.sum(axis=axis)
    if not np.allclose(s, 1.0, atol=atol, rtol=rtol):
        worst = float(np.max(np.abs(s - 1.0)))
        raise AssertionError(f"{name}: rows do not sum to 1 (max deviation={worst}).")


def assert_total_flow_conservation(
    total_demand: float,
    link_flows: Any,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    name: str = "link_flows",
) -> None:
    """
    Weak conservation check for assignment outputs.

    Without opt-out, total demand must be loaded somewhere in the network.
    A simple and stable check is that total link flow is >= total demand
    (because a passenger typically traverses multiple links).

    For tiny synthetic scenarios with exactly one ride link per trip,
    you may tighten this check in specific tests.

    :param total_demand: Total OD demand injected.
    :param link_flows: Link flow array.
    :param atol: Absolute tolerance.
    :param rtol: Relative tolerance.
    :param name: Name used in assertion messages.
    """
    x = np.asarray(link_flows, dtype=float)
    assert_nonnegative(x, name=name, tol=atol)
    tot = float(x.sum())
    if tot + atol < total_demand:
        raise AssertionError(
            f"{name}: total link flow {tot} is unexpectedly smaller than total demand {total_demand}."
        )