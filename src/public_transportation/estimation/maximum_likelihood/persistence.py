from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..common.persistence_utils import load_json, save_json
from .results import MLResult


def save_ml_result(
    result: MLResult,
    output_dir: str | Path,
    *,
    save_scipy_result: bool = False,
) -> None:
    """Save an MLResult to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dim": result.dim,
        "objective_value": result.objective_value,
        "loglikelihood": result.loglikelihood,
        "logprior": result.logprior,
        "prior_weight": result.prior_weight,
        "gradient_norm": result.gradient_norm,
        "success": result.success,
        "message": result.message,
        "method": result.method,
        "num_iterations": result.num_iterations,
        "num_function_evaluations": result.num_function_evaluations,
        "num_gradient_evaluations": result.num_gradient_evaluations,
        "runtime_seconds": result.runtime_seconds,
        "timestamp": result.timestamp,
    }
    save_json(metadata, output_path / "metadata.json")

    np.savez_compressed(
        output_path / "arrays.npz",
        theta_hat=result.theta_hat,
        gradient=result.gradient,
        optimization_trace=result.optimization_trace,
        hessian=np.asarray([]) if result.hessian is None else result.hessian,
        covariance_matrix=(
            np.asarray([])
            if result.covariance_matrix is None
            else result.covariance_matrix
        ),
        standard_errors=(
            np.asarray([])
            if result.standard_errors is None
            else result.standard_errors
        ),
        z_values=np.asarray([]) if result.z_values is None else result.z_values,
    )

    np.savetxt(
        output_path / "theta_hat.csv",
        np.column_stack([np.arange(result.dim), result.theta_hat]),
        delimiter=",",
        header="index,estimate",
        comments="",
    )

    if result.standard_errors is not None:
        np.savetxt(
            output_path / "standard_errors.csv",
            np.column_stack([np.arange(result.dim), result.standard_errors]),
            delimiter=",",
            header="index,standard_error",
            comments="",
        )

    np.savetxt(
        output_path / "optimization_trace.csv",
        result.optimization_trace,
        delimiter=",",
        header="iteration,objective,gradient_norm",
        comments="",
    )

    if save_scipy_result:
        try:
            np.save(output_path / "scipy_result.npy", result.scipy_result, allow_pickle=True)
        except Exception as exc:
            raise ValueError("Unable to save scipy_result.") from exc


def _empty_to_none(array: np.ndarray) -> np.ndarray | None:
    """Convert empty arrays used as sentinels back to None."""
    return None if array.size == 0 else array


def load_ml_result(
    input_dir: str | Path,
    *,
    load_scipy_result: bool = False,
) -> MLResult:
    """Load an MLResult saved by save_ml_result."""
    input_path = Path(input_dir)
    metadata = load_json(input_path / "metadata.json")
    arrays = np.load(input_path / "arrays.npz")

    scipy_result: Any | None = None
    if load_scipy_result and (input_path / "scipy_result.npy").exists():
        scipy_result = np.load(input_path / "scipy_result.npy", allow_pickle=True).item()

    return MLResult(
        dim=int(metadata["dim"]),
        theta_hat=arrays["theta_hat"],
        objective_value=float(metadata["objective_value"]),
        loglikelihood=float(metadata["loglikelihood"]),
        logprior=float(metadata["logprior"]),
        prior_weight=float(metadata["prior_weight"]),
        gradient=arrays["gradient"],
        gradient_norm=float(metadata["gradient_norm"]),
        hessian=_empty_to_none(arrays["hessian"]),
        covariance_matrix=_empty_to_none(arrays["covariance_matrix"]),
        standard_errors=_empty_to_none(arrays["standard_errors"]),
        z_values=_empty_to_none(arrays["z_values"]),
        success=bool(metadata["success"]),
        message=str(metadata["message"]),
        method=str(metadata["method"]),
        num_iterations=int(metadata["num_iterations"]),
        num_function_evaluations=int(metadata["num_function_evaluations"]),
        num_gradient_evaluations=(
            None
            if metadata["num_gradient_evaluations"] is None
            else int(metadata["num_gradient_evaluations"])
        ),
        runtime_seconds=float(metadata["runtime_seconds"]),
        timestamp=str(metadata["timestamp"]),
        optimization_trace=arrays["optimization_trace"],
        scipy_result=scipy_result,
    )
