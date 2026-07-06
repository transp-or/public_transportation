from typing import Any


def recommend_vi_defaults(dim: int) -> dict[str, Any]:
    """
    Recommend conservative VI defaults for large-dimensional problems.

    :param dim: Parameter dimension.
    :return: Dictionary of suggested settings for `run_vi`.
    """
    if dim <= 0:
        raise ValueError("dim must be positive.")

    # Heuristics: keep it simple and stable for very large dim.
    if dim >= 5000:
        return {
            "guide": "auto_diag",
            "learning_rate": 5e-3,
            "num_steps": 10_000,
            "num_posterior_draws": 2000,
        }
    if dim >= 1000:
        return {
            "guide": "auto_lowrank",
            "lowrank_rank": 50,
            "learning_rate": 1e-2,
            "num_steps": 8_000,
            "num_posterior_draws": 2000,
        }
    return {
        "guide": "auto_lowrank",
        "lowrank_rank": 20,
        "learning_rate": 1e-2,
        "num_steps": 5_000,
        "num_posterior_draws": 2000,
    }
