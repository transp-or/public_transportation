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
import hashlib
from pathlib import Path
from typing import Any, Literal

import numpy as np

from public_transportation.inference.runtime_profile import ODAssignmentRuntimeProfile

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

    # Reduced OD layout. ``None`` denotes a legacy all-free result file.
    free_od_indices: np.ndarray | None = None
    fixed_od_indices: np.ndarray | None = None
    fixed_od_values: np.ndarray | None = None
    od_layout_fingerprint: str | None = None
    od_layout_payload_json: str | None = None
    compact_layout_fingerprint: str | None = None
    compact_layout_payload_json: str | None = None
    runtime_profile: ODAssignmentRuntimeProfile | None = None

    @property
    def num_od(self) -> int:
        return int(self.f0.shape[0])

    @property
    def num_draws(self) -> int:
        return int(self.theta_samples.shape[0])

    @property
    def num_free_od(self) -> int:
        return self.num_od if self.free_od_indices is None else int(self.free_od_indices.size)

    @property
    def num_fixed_od(self) -> int:
        return 0 if self.fixed_od_indices is None else int(self.fixed_od_indices.size)

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


def _as_1d_int(x: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must be a 1D integer array, got shape {arr.shape}")
    return np.asarray(arr, dtype=np.int64)


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
        layout_keys = ("free_od_indices", "fixed_od_indices", "fixed_od_values")
        present_layout_keys = tuple(key for key in layout_keys if key in npz)
        if present_layout_keys and len(present_layout_keys) != len(layout_keys):
            raise ValueError(
                "VI results must contain all reduced-layout arrays or none of them; "
                f"found {present_layout_keys}."
            )
        if present_layout_keys:
            free_od_indices = _as_1d_int(npz["free_od_indices"], name="free_od_indices")
            fixed_od_indices = _as_1d_int(npz["fixed_od_indices"], name="fixed_od_indices")
            fixed_od_values = _as_1d_float(npz["fixed_od_values"], name="fixed_od_values")
        else:
            free_od_indices = fixed_od_indices = fixed_od_values = None
        od_layout_fingerprint = (
            str(npz["od_layout_fingerprint"]).strip()
            if "od_layout_fingerprint" in npz
            else None
        )
        od_layout_fingerprint = od_layout_fingerprint or None
        od_layout_payload_json = (
            str(npz["od_layout_payload_json"]).strip()
            if "od_layout_payload_json" in npz
            else None
        )
        od_layout_payload_json = od_layout_payload_json or None
        compact_layout_fingerprint = (
            str(npz["compact_layout_fingerprint"]).strip()
            if "compact_layout_fingerprint" in npz
            else None
        )
        compact_layout_fingerprint = compact_layout_fingerprint or None
        compact_layout_payload_json = (
            str(npz["compact_layout_payload_json"]).strip()
            if "compact_layout_payload_json" in npz
            else None
        )
        compact_layout_payload_json = compact_layout_payload_json or None
        runtime_keys = (
            "runtime_num_od_total",
            "runtime_num_free_od",
            "runtime_num_fixed_od",
            "runtime_num_fixed_zero_od",
            "runtime_num_fixed_positive_od",
            "runtime_assignment_active_od",
            "runtime_original_destination_groups",
            "runtime_active_destination_groups",
            "runtime_removed_destination_groups",
        )
        present_runtime_keys = tuple(key for key in runtime_keys if key in npz)
        if present_runtime_keys and len(present_runtime_keys) != len(runtime_keys):
            raise ValueError(
                "VI results must contain all runtime-profile counts or none; "
                f"found {present_runtime_keys}."
            )
        runtime_counts = (
            {key: int(np.asarray(npz[key]).reshape(())) for key in runtime_keys}
            if present_runtime_keys
            else None
        )

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

    if free_od_indices is not None:
        assert fixed_od_indices is not None and fixed_od_values is not None
        if fixed_od_indices.shape != fixed_od_values.shape:
            raise ValueError("fixed_od_indices and fixed_od_values must have equal length.")
        combined = np.concatenate((free_od_indices, fixed_od_indices))
        if np.unique(combined).size != num_od or set(combined.tolist()) != set(range(num_od)):
            raise ValueError("free and fixed OD indices must form a disjoint full partition.")
        if np.any(~np.isfinite(fixed_od_values)) or np.any(fixed_od_values < 0.0):
            raise ValueError("fixed_od_values must be finite and non-negative.")
        if fixed_od_indices.size:
            expected_fixed_samples = np.broadcast_to(
                fixed_od_values,
                (f_samples.shape[0], fixed_od_values.size),
            )
            if not np.array_equal(f_samples[:, fixed_od_indices], expected_fixed_samples):
                raise ValueError("posterior samples do not preserve declared fixed OD values exactly.")
            if not np.array_equal(f_mean[fixed_od_indices], fixed_od_values):
                raise ValueError("f_mean does not preserve declared fixed OD values exactly.")
    if od_layout_payload_json is not None:
        if od_layout_fingerprint is None:
            raise ValueError("od_layout_payload_json requires od_layout_fingerprint.")
        computed_layout_fingerprint = hashlib.sha256(
            od_layout_payload_json.encode("utf-8")
        ).hexdigest()
        if computed_layout_fingerprint != od_layout_fingerprint:
            raise ValueError(
                "OD layout fingerprint does not match its persisted canonical payload."
            )
    if compact_layout_payload_json is not None:
        if compact_layout_fingerprint is None:
            raise ValueError(
                "compact_layout_payload_json requires compact_layout_fingerprint."
            )
        computed_compact_fingerprint = hashlib.sha256(
            compact_layout_payload_json.encode("utf-8")
        ).hexdigest()
        if computed_compact_fingerprint != compact_layout_fingerprint:
            raise ValueError(
                "Compact layout fingerprint does not match its persisted canonical payload."
            )

    runtime_profile = None
    if runtime_counts is not None:
        counts = runtime_counts
        expected_free = num_od if free_od_indices is None else int(free_od_indices.size)
        expected_fixed = 0 if fixed_od_indices is None else int(fixed_od_indices.size)
        expected_zero = (
            0 if fixed_od_values is None else int(np.count_nonzero(fixed_od_values == 0.0))
        )
        expected_positive = expected_fixed - expected_zero
        expected_active = expected_free + expected_positive
        expected = {
            "runtime_num_od_total": num_od,
            "runtime_num_free_od": expected_free,
            "runtime_num_fixed_od": expected_fixed,
            "runtime_num_fixed_zero_od": expected_zero,
            "runtime_num_fixed_positive_od": expected_positive,
            "runtime_assignment_active_od": expected_active,
        }
        for key, value in expected.items():
            if counts[key] != value:
                raise ValueError(
                    f"Persisted runtime profile is inconsistent: {key}={counts[key]}, "
                    f"expected {value}."
                )
        original_groups = counts["runtime_original_destination_groups"]
        active_groups = counts["runtime_active_destination_groups"]
        removed_groups = counts["runtime_removed_destination_groups"]
        if min(original_groups, active_groups, removed_groups) < 0:
            raise ValueError("Persisted destination-group counts must be non-negative.")
        if active_groups + removed_groups != original_groups:
            raise ValueError("Persisted destination-group counts are inconsistent.")
        runtime_profile = ODAssignmentRuntimeProfile(
            num_od_total=num_od,
            num_free_od=expected_free,
            num_fixed_od=expected_fixed,
            num_fixed_zero_od=expected_zero,
            num_fixed_positive_od=expected_positive,
            assignment_active_od=expected_active,
            original_destination_groups=original_groups,
            active_destination_groups=active_groups,
            removed_destination_groups=removed_groups,
            od_layout_fingerprint=od_layout_fingerprint,
            compact_layout_fingerprint=compact_layout_fingerprint,
        )

    return ODThetaVIResults(
        fingerprint=fingerprint,
        f0=f0,
        theta_samples=theta_samples,
        f_samples=f_samples,
        theta_mean=theta_mean,
        theta_sd=theta_sd,
        f_mean=f_mean,
        vi_losses=vi_losses,
        free_od_indices=free_od_indices,
        fixed_od_indices=fixed_od_indices,
        fixed_od_values=fixed_od_values,
        od_layout_fingerprint=od_layout_fingerprint,
        od_layout_payload_json=od_layout_payload_json,
        compact_layout_fingerprint=compact_layout_fingerprint,
        compact_layout_payload_json=compact_layout_payload_json,
        runtime_profile=runtime_profile,
        fingerprint_payload_json=fingerprint_payload_json,
    )


def save_od_theta_vi_results(*, path: str | Path, results: ODThetaVIResults) -> Path:
    """Save OD+theta VI results to a compressed npz (persistence only).

    Saves the optional fingerprint_payload_json for richer diagnostics.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = dict(
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
    layout_values = (
        results.free_od_indices,
        results.fixed_od_indices,
        results.fixed_od_values,
    )
    if any(value is not None for value in layout_values):
        if not all(value is not None for value in layout_values):
            raise ValueError("All reduced-layout arrays must be supplied together.")
        payload.update(
            free_od_indices=np.asarray(results.free_od_indices, dtype=np.int64),
            fixed_od_indices=np.asarray(results.fixed_od_indices, dtype=np.int64),
            fixed_od_values=np.asarray(results.fixed_od_values, dtype=float),
        )
    if results.od_layout_fingerprint is not None:
        payload["od_layout_fingerprint"] = str(results.od_layout_fingerprint)
    if results.od_layout_payload_json is not None:
        payload["od_layout_payload_json"] = str(results.od_layout_payload_json)
    if results.compact_layout_fingerprint is not None:
        payload["compact_layout_fingerprint"] = str(results.compact_layout_fingerprint)
    if results.compact_layout_payload_json is not None:
        payload["compact_layout_payload_json"] = str(results.compact_layout_payload_json)
    if results.runtime_profile is not None:
        profile = results.runtime_profile
        if profile.od_layout_fingerprint != results.od_layout_fingerprint:
            raise ValueError("Runtime profile and result OD-layout fingerprints differ.")
        if profile.compact_layout_fingerprint != results.compact_layout_fingerprint:
            raise ValueError("Runtime profile and result compact-layout fingerprints differ.")
        payload.update(
            runtime_num_od_total=profile.num_od_total,
            runtime_num_free_od=profile.num_free_od,
            runtime_num_fixed_od=profile.num_fixed_od,
            runtime_num_fixed_zero_od=profile.num_fixed_zero_od,
            runtime_num_fixed_positive_od=profile.num_fixed_positive_od,
            runtime_assignment_active_od=profile.assignment_active_od,
            runtime_original_destination_groups=profile.original_destination_groups,
            runtime_active_destination_groups=profile.active_destination_groups,
            runtime_removed_destination_groups=profile.removed_destination_groups,
        )
    np.savez_compressed(p, **payload)
    return p
