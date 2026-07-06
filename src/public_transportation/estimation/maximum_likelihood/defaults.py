from typing import Any


def recommend_ml_defaults(dim: int) -> dict[str, Any]:
    """
    Recommend conservative ML defaults.

    :param dim: Parameter dimension.
    :return: Dictionary of suggested settings for `run_ml`.
    """
    if dim <= 0:
        raise ValueError("dim must be positive.")
    if dim >= 5000:
        return {
            "method": "L-BFGS-B",
            "maxiter": 1000,
            "gtol": 1e-5,
            "compute_hessian": False,
        }
    if dim >= 1000:
        return {
            "method": "L-BFGS-B",
            "maxiter": 1000,
            "gtol": 1e-6,
            "compute_hessian": False,
        }
    return {
        "method": "BFGS",
        "maxiter": 1000,
        "gtol": 1e-6,
        "compute_hessian": True,
    }
