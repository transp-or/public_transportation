"""
public_transportation.inference.results_io

IO helpers for OD+theta VI results (transport-model specific).

Design goals
------------
- Keep scripts minimal: reading/writing VI results is centralized here.
- Keep runtime dependencies light (NumPy only).
- One responsibility per function:
  - validate payload
  - load from npz
  - save to npz
  - small convenience helpers (point estimates, fingerprint check)
  - support storing/loading a fingerprint payload JSON for richer mismatch diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:  # optional dependency for richer diagnostics
    from public_transportation.assignment.id_manager import AssignmentIDManager
except Exception:  # pragma: no cover
    AssignmentIDManager = None  # type: ignore


# -----------------------------------------------------------------------------
# Data container
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ODThetaVIResults:
    """Loaded OD+theta VI results (from npz)."""

    fingerprint: str

    # Prior / baseline OD vector in assignment OD order
    f0: np.ndarray  # (num_od,)

    # Posterior samples
    theta_samples: np.ndarray  # (S,)
    f_samples: np.ndarray      # (S, num_od)

    # Convenience summaries
    theta_mean: float
    theta_sd: float
    f_mean: np.ndarray         # (num_od,)

    # VI diagnostics
    vi_losses: np.ndarray      # (num_steps,)

    # Exact JSON payload used to compute the fingerprint (optional, for diagnostics)
    fingerprint_payload_json: str | None = None

    @property
    def num_od(self) -> int:
        return int(self.f0.shape[0])

    @property
    def num_draws(self) -> int:
        return int(self.theta_samples.shape[0])

    def point_estimate(
        self,
        *,
        theta: Literal["mean", "median"] = "mean",
        f: Literal["mean", "median"] = "mean",
    ) -> tuple[float, np.ndarray]:
        """Return a point estimate (theta_hat, f_hat) from stored summaries/samples."""
        if theta == "mean":
            theta_hat = float(self.theta_mean)
        elif theta == "median":
            theta_hat = float(np.median(self.theta_samples))
        else:
            raise ValueError(f"Unknown theta estimator: {theta!r}")

        if f == "mean":
            f_hat = np.asarray(self.f_mean, dtype=float)
        elif f == "median":
            f_hat = np.asarray(np.median(self.f_samples, axis=0), dtype=float)
        else:
            raise ValueError(f"Unknown f estimator: {f!r}")

        return theta_hat, f_hat


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _require_keys(npz: Any, keys: tuple[str, ...]) -> None:
    missing = [k for k in keys if k not in npz]
    if missing:
        raise ValueError(f"Missing keys in VI results npz: {missing}")


def _as_scalar_float(x: Any, *, name: str) -> float:
    arr = np.asarray(x)
    if arr.size != 1:
        raise ValueError(f"{name} must be scalar-like, got shape {arr.shape}")
    return float(arr.reshape(()))


def _as_1d_float(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr


def _as_2d_float(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    return arr


def assert_fingerprint_matches(
    *,
    expected: str,
    got: str,
    context: str = "",
    expected_payload_json: str | None = None,
    got_payload_json: str | None = None,
) -> None:
    """Fail fast if fingerprints do not match.

    If fingerprint payload JSON strings are provided (and AssignmentIDManager is importable),
    the raised error includes a field-by-field diff of the payloads.
    """
    if str(expected) == str(got):
        return

    # Rich diagnostics when possible
    exp_payload = (expected_payload_json or "").strip() or None
    got_payload = (got_payload_json or "").strip() or None

    if AssignmentIDManager is not None and exp_payload and got_payload:
        msg = AssignmentIDManager.format_fingerprint_mismatch(
            expected_fingerprint=str(expected),
            got_fingerprint=str(got),
            expected_payload_json=str(exp_payload),
            got_payload_json=str(got_payload),
        )
        if context:
            msg = f"{msg}\n(context: {context})"
        raise ValueError(msg)

    # Fallback
    msg = "Fingerprint mismatch"
    if context:
        msg += f" ({context})"
    msg += f": expected={expected}, got={got}"
    raise ValueError(msg)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def load_od_theta_vi_results(path: str | Path) -> ODThetaVIResults:
    """Load OD+theta VI results saved via np.savez_compressed.

    Optionally loads the fingerprint_payload_json for richer diagnostics.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    with np.load(p, allow_pickle=False) as npz:
        _require_keys(
            npz,
            (
                "fingerprint",
                "f0",
                "theta_samples",
                "f_samples",
                "theta_mean",
                "theta_sd",
                "f_mean",
                "vi_losses",
            ),
        )

        fingerprint = str(npz["fingerprint"])
        if "fingerprint_payload_json" in npz:
            payload = str(npz["fingerprint_payload_json"])
            fingerprint_payload_json = payload.strip() or None
        else:
            fingerprint_payload_json = None

        f0 = _as_1d_float(npz["f0"], name="f0")
        theta_samples = _as_1d_float(npz["theta_samples"], name="theta_samples")
        f_samples = _as_2d_float(npz["f_samples"], name="f_samples")
        theta_mean = _as_scalar_float(npz["theta_mean"], name="theta_mean")
        theta_sd = _as_scalar_float(npz["theta_sd"], name="theta_sd")
        f_mean = _as_1d_float(npz["f_mean"], name="f_mean")
        vi_losses = _as_1d_float(npz["vi_losses"], name="vi_losses")

    # Consistency checks
    num_od = int(f0.shape[0])
    if f_mean.shape != (num_od,):
        raise ValueError(f"f_mean shape {f_mean.shape} must match f0 shape {(num_od,)}")

    if f_samples.shape[1] != num_od:
        raise ValueError(f"f_samples must be (S, num_od={num_od}), got {f_samples.shape}")

    if theta_samples.shape[0] != f_samples.shape[0]:
        raise ValueError(
            "theta_samples and f_samples must have same number of draws S: "
            f"{theta_samples.shape[0]} vs {f_samples.shape[0]}"
        )

    if not np.isfinite(theta_mean) or theta_mean <= 0.0:
        raise ValueError(f"theta_mean must be finite and > 0, got {theta_mean!r}")
    if not np.isfinite(theta_sd) or theta_sd < 0.0:
        raise ValueError(f"theta_sd must be finite and >= 0, got {theta_sd!r}")

    return ODThetaVIResults(
        fingerprint=fingerprint,
        f0=f0,
        theta_samples=theta_samples,
        f_samples=f_samples,
        theta_mean=theta_mean,
        theta_sd=theta_sd,
        f_mean=f_mean,
        vi_losses=vi_losses,
        fingerprint_payload_json=fingerprint_payload_json,
    )


def save_od_theta_vi_results(*, path: str | Path, results: ODThetaVIResults) -> Path:
    """Save OD+theta VI results to a compressed npz (persistence only).

    Saves the optional fingerprint_payload_json for richer diagnostics.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        p,
        fingerprint=str(results.fingerprint),
        fingerprint_payload_json=("" if results.fingerprint_payload_json is None else str(results.fingerprint_payload_json)),
        f0=np.asarray(results.f0, dtype=float),
        theta_samples=np.asarray(results.theta_samples, dtype=float),
        f_samples=np.asarray(results.f_samples, dtype=float),
        theta_mean=float(results.theta_mean),
        theta_sd=float(results.theta_sd),
        f_mean=np.asarray(results.f_mean, dtype=float),
        vi_losses=np.asarray(results.vi_losses, dtype=float),
    )
    return p