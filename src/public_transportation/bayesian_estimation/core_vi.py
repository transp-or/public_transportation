from datetime import datetime, timedelta
from time import perf_counter
from typing import Any, Literal

import jax
import numpy as np
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam

from .guides import make_autoguide
from .model_blackbox import LogLikFn, LogPriorFn, make_blackbox_model
from .results import VIResult


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string."""
    return str(timedelta(seconds=int(round(max(seconds, 0.0)))))

def run_vi(
    *,
    dim: int,
    data: Any,
    loglik: LogLikFn,
    logprior: LogPriorFn,
    guide: Literal["auto_diag", "auto_lowrank", "auto_mvn", "auto_normal"] = "auto_diag",
    lowrank_rank: int | None = None,
    use_base_normal_correction: bool = False,
    num_steps: int = 5_000,
    learning_rate: float = 1e-2,
    seed: int = 0,
    num_posterior_draws: int = 1_000,
    logger: Any | None = None,
    log_every: int = 100,
) -> VIResult:
    """
    Run variational inference (SVI) for a user-defined posterior target.

    :param dim: Dimension of theta (unconstrained parameter vector).
    :param data: Arbitrary user data passed to loglik(theta, data). Prefer JAX PyTrees.
    :param loglik: JAX-compatible, differentiable log-likelihood. Must return a scalar.
    :param logprior: JAX-compatible, differentiable log-prior term. Must return a scalar.
        Interpretation:
          - If `use_base_normal_correction=False` (default), best practice is:
                logprior(theta) = log p(theta) - log N(theta;0,I).
          - If `use_base_normal_correction=True`, then:
                logprior(theta) = log p(theta).
    :param guide: Autoguide choice. For thousands of parameters, prefer:
        - "auto_diag" (fast) or
        - "auto_lowrank" (captures correlations).
    :param lowrank_rank: Rank for "auto_lowrank".
    :param use_base_normal_correction: If True, subtract log N(theta;0,I) inside the model
        so `logprior` can be an absolute prior log-density.
    :param num_steps: Number of SVI optimization steps.
    :param learning_rate: Adam learning rate.
    :param seed: Random seed for initialization and posterior sampling.
    :param num_posterior_draws: Number of posterior samples to draw from the variational guide.
    :param logger: Optional logger used to report progress. It must provide a method
        `info(msg, *args)` like Python's standard `logging.Logger`. If None, no logging
        is performed.
    :param log_every: If logger is provided, emit a log message every `log_every` steps,
        and also at the last step. Must be a positive integer.
    :return: VIResult including learned parameters and posterior samples.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if num_posterior_draws <= 0:
        raise ValueError("num_posterior_draws must be positive.")
    if log_every <= 0:
        raise ValueError("log_every must be positive.")

    model = make_blackbox_model(
        dim=dim,
        loglik=loglik,
        logprior=logprior,
        data=data,
        use_base_normal_correction=use_base_normal_correction,
    )
    guide_obj = make_autoguide(model=model, guide=guide, lowrank_rank=lowrank_rank)

    optimizer = Adam(learning_rate)
    svi = SVI(model=model, guide=guide_obj, optim=optimizer, loss=Trace_ELBO())

    key = jax.random.PRNGKey(seed)

    svi_state = svi.init(key)

    start_time = perf_counter()
    last_log_time = start_time
    last_log_step = 0

    losses: list[float] = []
    for step in range(num_steps):
        svi_state, loss = svi.update(svi_state)
        loss_val = float(loss)
        losses.append(loss_val)

        if logger is not None and (step % log_every == 0 or step == num_steps - 1):
            now = perf_counter()
            completed_steps = step + 1
            elapsed_total = now - start_time
            avg_time_per_step = elapsed_total / completed_steps

            interval_steps = completed_steps - last_log_step
            interval_elapsed = now - last_log_time
            recent_time_per_step = interval_elapsed / max(interval_steps, 1)

            remaining_steps = num_steps - completed_steps
            blended_time_per_step = (
                0.3 * avg_time_per_step + 0.7 * recent_time_per_step
            )
            eta_seconds = remaining_steps * blended_time_per_step

            logger.info(
                "VI step %d/%d — ELBO loss: %.6f — elapsed: %s — ETA: %s",
                completed_steps,
                num_steps,
                loss_val,
                _format_duration(elapsed_total),
                _format_duration(eta_seconds),
            )

            last_log_time = now
            last_log_step = completed_steps

    params = svi.get_params(svi_state)

    key, subkey = jax.random.split(key)
    theta_samples = guide_obj.sample_posterior(
        subkey,
        params,
        sample_shape=(num_posterior_draws,),
    )["theta"]

    posterior_samples_theta = np.asarray(theta_samples, dtype=float)
    posterior_mean = np.mean(posterior_samples_theta, axis=0)
    posterior_sd = np.std(posterior_samples_theta, axis=0, ddof=1)
    posterior_q05 = np.quantile(posterior_samples_theta, 0.05, axis=0)
    posterior_q50 = np.quantile(posterior_samples_theta, 0.50, axis=0)
    posterior_q95 = np.quantile(posterior_samples_theta, 0.95, axis=0)
    runtime_seconds = perf_counter() - start_time
    timestamp = datetime.now().isoformat(timespec="seconds")

    return VIResult(
        guide=guide,
        dim=dim,
        use_base_normal_correction=use_base_normal_correction,
        svi_state=svi_state,
        params=params,
        losses=np.asarray(losses, dtype=float),
        posterior_samples_theta=posterior_samples_theta,
        seed=seed,
        num_steps=num_steps,
        learning_rate=learning_rate,
        lowrank_rank=lowrank_rank,
        num_posterior_draws=num_posterior_draws,
        runtime_seconds=float(runtime_seconds),
        timestamp=timestamp,
        posterior_mean=posterior_mean,
        posterior_sd=posterior_sd,
        posterior_q05=posterior_q05,
        posterior_q50=posterior_q50,
        posterior_q95=posterior_q95,
    )
