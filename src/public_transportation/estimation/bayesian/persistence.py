from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .results import VIResult


def _to_jsonable(value: Any) -> Any:
    """Convert common Python/NumPy objects into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _save_json(data: dict[str, Any], filename: Path) -> None:
    """Save a dictionary as formatted JSON."""
    with filename.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2, sort_keys=True)


def _load_json(filename: Path) -> dict[str, Any]:
    """Load a JSON file into a Python dictionary."""
    with filename.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_vi_result(
    result: VIResult,
    output_dir: str | Path,
    *,
    save_svi_state: bool = False,
    save_params: bool = False,
) -> None:
    """
    Save a VIResult to disk.

    The saved bundle is organized as follows:

    output_dir/
        metadata.json
        arrays.npz
        losses.csv
        posterior_summary.csv
        params.npz              (optional)
        svi_state.npy           (optional)

    Notes
    -----
    - By default, `params` and `svi_state` are not saved, because they may contain
      complex, non-portable objects depending on the backend.
    - Arrays are stored in a single NPZ file for compactness and simplicity.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "guide": result.guide,
        "dim": result.dim,
        "use_base_normal_correction": result.use_base_normal_correction,
        "seed": result.seed,
        "num_steps": result.num_steps,
        "learning_rate": result.learning_rate,
        "lowrank_rank": result.lowrank_rank,
        "num_posterior_draws": result.num_posterior_draws,
        "runtime_seconds": result.runtime_seconds,
        "timestamp": result.timestamp,
    }
    _save_json(metadata, output_path / "metadata.json")

    np.savez_compressed(
        output_path / "arrays.npz",
        losses=result.losses,
        posterior_samples_theta=result.posterior_samples_theta,
        posterior_mean=result.posterior_mean,
        posterior_sd=result.posterior_sd,
        posterior_q05=result.posterior_q05,
        posterior_q50=result.posterior_q50,
        posterior_q95=result.posterior_q95,
    )

    np.savetxt(
        output_path / "losses.csv",
        result.losses,
        delimiter=",",
        header="loss",
        comments="",
    )

    posterior_summary = np.column_stack(
        [
            np.arange(result.dim),
            result.posterior_mean,
            result.posterior_sd,
            result.posterior_q05,
            result.posterior_q50,
            result.posterior_q95,
        ]
    )
    np.savetxt(
        output_path / "posterior_summary.csv",
        posterior_summary,
        delimiter=",",
        header="index,mean,sd,q05,q50,q95",
        comments="",
    )

    if save_params:
        try:
            np.savez_compressed(output_path / "params.npz", **result.params)
        except Exception as exc:
            raise ValueError(
                "Unable to save `params` with np.savez_compressed. "
                "They are likely not a plain dictionary of arrays."
            ) from exc

    if save_svi_state:
        try:
            np.save(output_path / "svi_state.npy", result.svi_state, allow_pickle=True)
        except Exception as exc:
            raise ValueError("Unable to save `svi_state`.") from exc


def load_vi_result(
    input_dir: str | Path,
    *,
    load_svi_state: bool = False,
    load_params: bool = False,
) -> VIResult:
    """
    Load a VIResult from disk.

    This reconstructs the VIResult from the files created by `save_vi_result`.

    Notes
    -----
    - If `params` or `svi_state` were not saved, they are loaded as None.
    """
    input_path = Path(input_dir)

    metadata = _load_json(input_path / "metadata.json")
    arrays = np.load(input_path / "arrays.npz")

    params = None
    if load_params and (input_path / "params.npz").exists():
        params_npz = np.load(input_path / "params.npz", allow_pickle=True)
        params = {key: params_npz[key] for key in params_npz.files}

    svi_state = None
    if load_svi_state and (input_path / "svi_state.npy").exists():
        svi_state = np.load(input_path / "svi_state.npy", allow_pickle=True).item()

    return VIResult(
        guide=metadata["guide"],
        dim=int(metadata["dim"]),
        use_base_normal_correction=bool(metadata["use_base_normal_correction"]),
        svi_state=svi_state,
        params=params,
        losses=arrays["losses"],
        posterior_samples_theta=arrays["posterior_samples_theta"],
        seed=int(metadata["seed"]),
        num_steps=int(metadata["num_steps"]),
        learning_rate=float(metadata["learning_rate"]),
        lowrank_rank=(
            None
            if metadata["lowrank_rank"] is None
            else int(metadata["lowrank_rank"])
        ),
        num_posterior_draws=int(metadata["num_posterior_draws"]),
        runtime_seconds=float(metadata["runtime_seconds"]),
        timestamp=str(metadata["timestamp"]),
        posterior_mean=arrays["posterior_mean"],
        posterior_sd=arrays["posterior_sd"],
        posterior_q05=arrays["posterior_q05"],
        posterior_q50=arrays["posterior_q50"],
        posterior_q95=arrays["posterior_q95"],
    )