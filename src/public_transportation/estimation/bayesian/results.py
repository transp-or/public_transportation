from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VIResult:
    """
    Container for variational inference results.

    :param guide: Name of the guide used.
    :param dim: Dimension of the parameter vector theta.
    :param use_base_normal_correction: Whether the base Normal(0,I) log-density was subtracted.
    :param svi_state: Final SVI state (NumPyro object).
    :param params: Learned variational parameters (PyTree).
    :param losses: ELBO losses per optimization step, shape (num_steps,).
    :param posterior_samples_theta: Samples from the variational posterior over theta,
        shape (num_draws, dim).
    """
    guide: str
    dim: int
    use_base_normal_correction: bool
    svi_state: Any
    params: Any
    losses: np.ndarray
    posterior_samples_theta: np.ndarray
    seed: int
    num_steps: int
    learning_rate: float
    lowrank_rank: int | None
    num_posterior_draws: int
    runtime_seconds: float
    timestamp: str
    posterior_mean: np.ndarray
    posterior_sd: np.ndarray
    posterior_q05: np.ndarray
    posterior_q50: np.ndarray
    posterior_q95: np.ndarray
