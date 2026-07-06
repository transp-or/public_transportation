# run_compare_bayesian_ml.py
"""
Estimate OD and theta with both Bayesian VI and maximum likelihood, then compare
estimates and computation times.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import jax.numpy as jnp
import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.domain import Scenario
from public_transportation.estimation.bayesian.config import VIConfig
from public_transportation.estimation.maximum_likelihood import MLConfig, run_ml
from public_transportation.inference.assignment_adapter import build_assignment_inputs
from public_transportation.inference.likelihood import (
    loglikelihood_from_link_flow,
    prepare_likelihood_inputs,
)
from public_transportation.inference.model import make_forward_inputs, forward_model
from public_transportation.inference.pipeline import (
    ODThetaEstimationRequest,
    estimate_od_theta_vi,
)
from public_transportation.inference.priors import build_f0_from_scenario_demand
from public_transportation.measurement import build_mapping_spec_strict, read_measurements_csv

DATA = Path(__file__).resolve().parent / "data"


def make_console_logger(
    name: str = "public_transportation.compare",
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def normal_logpdf(x: jnp.ndarray, loc: float | jnp.ndarray, scale: float) -> jnp.ndarray:
    x = jnp.asarray(x)
    loc = jnp.asarray(loc, dtype=x.dtype)
    scale = jnp.asarray(scale, dtype=x.dtype)
    log_two_pi = jnp.asarray(np.log(2.0 * np.pi), dtype=x.dtype)
    return -0.5 * ((x - loc) / scale) ** 2 - jnp.log(scale) - 0.5 * log_two_pi

def make_parameter_vector_from_f_and_theta(
    *,
    f: jnp.ndarray,
    theta: float | jnp.ndarray,
    f0: jnp.ndarray,
    estimate_theta: bool,
) -> jnp.ndarray:
    safe_f = jnp.maximum(jnp.asarray(f, dtype=jnp.float32), 1.0e-12)
    safe_f0 = jnp.maximum(jnp.asarray(f0, dtype=jnp.float32), 1.0e-12)
    z = jnp.log(safe_f / safe_f0)
    if not estimate_theta:
        return z
    u = jnp.log(jnp.maximum(jnp.asarray(theta, dtype=jnp.float32), 1.0e-12))
    return jnp.concatenate([z, jnp.asarray([u], dtype=jnp.float32)])

logger = make_console_logger()

# Shared hyperparameters.
baseline_theta = 10.0
sigma_z = 100.0
sigma_u = 10.0
rho = 1.0
nb_dispersion = 50.0
z_clip = 6.0
u_clip = 6.0
NUM_VI_OBJECTIVE_DRAWS = 25

# Theta handling.
# - estimate_theta=True: estimate theta in both VI and ML.
# - estimate_theta=False: estimate only OD log-deviations z and use fixed_theta.
estimate_theta = False
fixed_theta = baseline_theta

# -----------------------------------------------------------------------------
# 1) Shared data preparation
# -----------------------------------------------------------------------------
scenario = Scenario.from_folder(DATA)

rep = scenario.validate()
if rep.issues:
    print("Scenario validation issues:")
    for it in rep.issues:
        print(f"- [{it.severity.name}] {it.code}: {it.message} ({it.location})")

assignment_config = AssignmentConfig()
artifacts = prepare_assignment(scenario=scenario, config=assignment_config)
idm = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)

meas_path = DATA / "measurements_boarding_alighting.csv"
if not meas_path.exists():
    raise RuntimeError(f"Missing measurement file: {meas_path}")

table = read_measurements_csv(meas_path)
msr = build_mapping_spec_strict(
    id_manager=idm,
    table=table,
    include_link_lists_for_report=False,
)

f0 = build_f0_from_scenario_demand(
    scenario=scenario,
    id_manager=idm,
    dtype=jnp.float32,
)
f0 = jnp.asarray(f0, dtype=jnp.float32)

if estimate_theta:
    if fixed_theta is not None:
        raise ValueError("fixed_theta must be None when estimate_theta=True.")
else:
    if fixed_theta is None:
        raise ValueError("fixed_theta must be provided when estimate_theta=False.")
    fixed_theta = float(fixed_theta)
    if not np.isfinite(fixed_theta) or fixed_theta <= 0.0:
        raise ValueError(f"fixed_theta must be positive and finite, got {fixed_theta!r}.")

# -----------------------------------------------------------------------------
# 2) Bayesian VI
# -----------------------------------------------------------------------------
vi_cfg = VIConfig(
    guide="auto_lowrank",
    lowrank_rank=10,
    use_base_normal_correction=True,
    num_steps=4000,
    learning_rate=1e-2,
    seed=0,
    num_posterior_draws=1000,
    log_every=100,
)

vi_request = ODThetaEstimationRequest(
    fingerprint=str(idm.fingerprint),
    fingerprint_payload_json=idm.fingerprint_payload_json,
    f0=f0,
    y_obs=jnp.asarray(msr.y_obs, dtype=jnp.float32),
    mapping_spec=msr.spec,
    baseline_theta=float(baseline_theta),
    estimate_theta=estimate_theta,
    fixed_theta=fixed_theta,
    sigma_z=float(sigma_z),
    sigma_u=float(sigma_u),
    rho=float(rho),
    nb_dispersion=float(nb_dispersion),
    z_clip=float(z_clip),
    u_clip=float(u_clip),
    vi=vi_cfg,
    assignment_artifacts=artifacts,
    logger=logger,
)

print("Running Bayesian VI...")
t0 = perf_counter()
vi_result = estimate_od_theta_vi(vi_request)
vi_time_s = perf_counter() - t0

vi_f_mean = np.asarray(vi_result.f_mean, dtype=float)
vi_theta_mean = float(vi_result.theta_mean)
vi_total_flow_samples = vi_result.f_samples.sum(axis=1)
vi_theta_quantiles = np.quantile(
    vi_result.theta_samples,
    [0.01, 0.05, 0.50, 0.95, 0.99],
)

# -----------------------------------------------------------------------------
# 3) Maximum likelihood
# -----------------------------------------------------------------------------
assignment_inputs = build_assignment_inputs(artifacts=artifacts)
forward_inputs = make_forward_inputs(f0=f0, spec=msr.spec)
prepared_likelihood = prepare_likelihood_inputs(
    y_obs=jnp.asarray(msr.y_obs, dtype=jnp.float32),
    spec=msr.spec,
)

num_od = int(f0.shape[0])
dim = num_od + 1 if estimate_theta else num_od
mu_u = float(np.log(baseline_theta))

data: dict[str, Any] = {
    "rho": jnp.asarray(rho, dtype=jnp.float32).reshape(()),
    "r_nb": jnp.asarray(nb_dispersion, dtype=jnp.float32).reshape(()),
}


def split_theta(theta_vec: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray]:
    z = theta_vec[:num_od]
    if estimate_theta:
        u = theta_vec[num_od]
        u = jnp.clip(u, -u_clip, u_clip)
        theta = jnp.exp(u)
        return z, u, theta

    assert fixed_theta is not None
    theta = jnp.asarray(float(fixed_theta), dtype=theta_vec.dtype).reshape(())
    return z, None, theta


def loglik(theta_vec: jnp.ndarray, data_: dict[str, Any]) -> jnp.ndarray:
    z, _, theta = split_theta(theta_vec)
    z = jnp.clip(z, -z_clip, z_clip)

    out = forward_model(
        inputs=forward_inputs,
        z=z,
        theta=theta,
        rho=data_["rho"],
        assignment_inputs=assignment_inputs,
    )

    ll = loglikelihood_from_link_flow(
        link_flow=out.link_flow,
        prepared=prepared_likelihood,
        theta=theta,
        rho=data_["rho"],
        r=data_["r_nb"],
    )
    return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)


def logprior(theta_vec: jnp.ndarray) -> jnp.ndarray:
    z, u, _ = split_theta(theta_vec)
    z = jnp.clip(z, -z_clip, z_clip)
    lp = normal_logpdf(z, 0.0, sigma_z).sum()
    if estimate_theta:
        assert u is not None
        lp = lp + normal_logpdf(u, mu_u, sigma_u).sum()
    return lp


ml_cfg = MLConfig(
    method="BFGS",
    maxiter=1000,
    gtol=1.0e-6,
    prior_weight=1.0,  # 0.0 = pure ML; 1.0 = penalized ML/MAP.
    compute_hessian=True,
    log_every=10,
)

theta0 = jnp.zeros((dim,), dtype=jnp.float32)
if estimate_theta:
    theta0 = theta0.at[num_od].set(mu_u)

print("Running maximum likelihood...")
t0 = perf_counter()
ml_result = run_ml(
    dim=dim,
    data=data,
    loglik=loglik,
    logprior=logprior,
    theta0=theta0,
    config=ml_cfg,
    logger=logger,
)
ml_time_s = perf_counter() - t0

ml_z_hat = np.asarray(ml_result.theta_hat[:num_od], dtype=float)
if estimate_theta:
    ml_u_hat = float(ml_result.theta_hat[num_od])
    ml_theta_hat = float(np.exp(ml_u_hat))
else:
    assert fixed_theta is not None
    ml_u_hat = float(np.log(fixed_theta))
    ml_theta_hat = float(fixed_theta)
ml_f_hat = np.asarray(f0, dtype=float) * np.exp(ml_z_hat)

# -----------------------------------------------------------------------------
# 4) Diagnostics and comparison
# -----------------------------------------------------------------------------
# Convert the VI posterior summaries into the ML parameterization [z, u].
# This is a diagnostic point based on transformed posterior means:
#     z = log(E[f | data] / f0), u = log(E[theta | data]).
# It is not identical to E[z | data] and E[u | data].
vi_parameter_from_means = make_parameter_vector_from_f_and_theta(
    f=jnp.asarray(vi_f_mean, dtype=jnp.float32),
    theta=vi_theta_mean,
    f0=f0,
    estimate_theta=estimate_theta,
)

# Evaluate the ML/MAP objective at the ML solution and at the VI-based summary.
ml_loglik_at_ml = float(loglik(jnp.asarray(ml_result.theta_hat), data))
ml_logprior_at_ml = float(logprior(jnp.asarray(ml_result.theta_hat)))
ml_logpost_at_ml = ml_loglik_at_ml + float(ml_cfg.prior_weight) * ml_logprior_at_ml

ml_loglik_at_vi_summary = float(loglik(vi_parameter_from_means, data))
ml_logprior_at_vi_summary = float(logprior(vi_parameter_from_means))
ml_logpost_at_vi_summary = (
    ml_loglik_at_vi_summary + float(ml_cfg.prior_weight) * ml_logprior_at_vi_summary
)

# Evaluate the ML objective at a small subset of VI posterior draws. This checks
# whether the VI distribution is concentrated in a region with comparable
# objective value to the ML/MAP solution.
num_vi_draws_available = int(vi_result.theta_samples.shape[0])
num_vi_objective_draws = min(NUM_VI_OBJECTIVE_DRAWS, num_vi_draws_available)
vi_objective_draw_indices = np.linspace(
    0,
    num_vi_draws_available - 1,
    num=num_vi_objective_draws,
    dtype=int,
)

vi_sample_loglik_values: list[float] = []
vi_sample_logprior_values: list[float] = []
vi_sample_logpost_values: list[float] = []
vi_sample_theta_values: list[float] = []

for draw_index in vi_objective_draw_indices:
    vi_parameter_draw = make_parameter_vector_from_f_and_theta(
        f=jnp.asarray(vi_result.f_samples[draw_index], dtype=jnp.float32),
        theta=float(vi_result.theta_samples[draw_index]),
        f0=f0,
        estimate_theta=estimate_theta,
    )
    draw_loglik = float(loglik(vi_parameter_draw, data))
    draw_logprior = float(logprior(vi_parameter_draw))
    draw_logpost = draw_loglik + float(ml_cfg.prior_weight) * draw_logprior

    vi_sample_loglik_values.append(draw_loglik)
    vi_sample_logprior_values.append(draw_logprior)
    vi_sample_logpost_values.append(draw_logpost)
    vi_sample_theta_values.append(float(vi_result.theta_samples[draw_index]))

vi_sample_loglik_values = np.asarray(vi_sample_loglik_values, dtype=float)
vi_sample_logprior_values = np.asarray(vi_sample_logprior_values, dtype=float)
vi_sample_logpost_values = np.asarray(vi_sample_logpost_values, dtype=float)
vi_sample_theta_values = np.asarray(vi_sample_theta_values, dtype=float)
vi_sample_logpost_quantiles = np.quantile(
    vi_sample_logpost_values,
    [0.05, 0.50, 0.95],
)

# Run the same ML optimization from the VI-based summary. If this converges to a
# different solution, the objective is likely ill-conditioned or multimodal. If it
# converges to the same ML solution, the discrepancy is not an initialization artifact.
print()
print("Running maximum likelihood again, initialized at the VI-based posterior summary ...")
t0 = perf_counter()
ml_from_vi_result = run_ml(
    dim=dim,
    data=data,
    loglik=loglik,
    logprior=logprior,
    theta0=vi_parameter_from_means,
    config=ml_cfg,
    logger=logger,
)
ml_from_vi_time_s = perf_counter() - t0

ml_from_vi_z_hat = np.asarray(ml_from_vi_result.theta_hat[:num_od], dtype=float)
if estimate_theta:
    ml_from_vi_u_hat = float(ml_from_vi_result.theta_hat[num_od])
    ml_from_vi_theta_hat = float(np.exp(ml_from_vi_u_hat))
else:
    assert fixed_theta is not None
    ml_from_vi_u_hat = float(np.log(fixed_theta))
    ml_from_vi_theta_hat = float(fixed_theta)
ml_from_vi_f_hat = np.asarray(f0, dtype=float) * np.exp(ml_from_vi_z_hat)
ml_from_vi_total_flow = float(ml_from_vi_f_hat.sum())

ml_loglik_at_ml_from_vi = float(loglik(jnp.asarray(ml_from_vi_result.theta_hat), data))
ml_logprior_at_ml_from_vi = float(logprior(jnp.asarray(ml_from_vi_result.theta_hat)))
ml_logpost_at_ml_from_vi = (
    ml_loglik_at_ml_from_vi + float(ml_cfg.prior_weight) * ml_logprior_at_ml_from_vi
)

# Aggregate comparison metrics.
theta_abs_diff = abs(ml_theta_hat - vi_theta_mean)
theta_rel_diff = theta_abs_diff / abs(vi_theta_mean) if vi_theta_mean != 0 else np.nan

flow_diff = ml_f_hat - vi_f_mean
max_abs_flow_diff = float(np.max(np.abs(flow_diff)))
mean_abs_flow_diff = float(np.mean(np.abs(flow_diff)))

vi_total_flow_mean = float(vi_total_flow_samples.mean())
vi_total_flow_sd = float(vi_total_flow_samples.std())
ml_total_flow = float(ml_f_hat.sum())
total_flow_abs_diff = abs(ml_total_flow - vi_total_flow_mean)
total_flow_rel_diff = (
    total_flow_abs_diff / abs(vi_total_flow_mean)
    if vi_total_flow_mean != 0
    else np.nan
)

time_ratio = vi_time_s / ml_time_s if ml_time_s > 0 else np.nan

# Print results after all diagnostics have been computed.
print()
print("Comparison of Bayesian VI and ML")
print("--------------------------------")
print(f"estimate theta: {estimate_theta}")
if not estimate_theta:
    print(f"fixed theta: {fixed_theta:.6g}")
print(f"VI time: {vi_time_s:.3f} s")
print(f"ML time: {ml_time_s:.3f} s")
print(f"ML-from-VI time: {ml_from_vi_time_s:.3f} s")
print(f"VI/ML time ratio: {time_ratio:.3f}")
print()
print("Bayesian VI theta summary")
vi_theta_p05 = np.quantile(vi_result.theta_samples, 0.05)
vi_theta_p95 = np.quantile(vi_result.theta_samples, 0.95)
print(
    f"theta: mean={vi_result.theta_mean:.6g}, sd={vi_result.theta_sd:.6g}, "
    f"p05={vi_theta_p05:.6g}, p95={vi_theta_p95:.6g}"
)
print(
    "theta quantiles [1%, 5%, 50%, 95%, 99%]: "
    f"{np.array2string(vi_theta_quantiles, precision=6)}"
)
print()
print("Theta comparison")
print(f"theta VI mean: {vi_theta_mean:.6g}")
print(f"theta ML: {ml_theta_hat:.6g}")
print(f"theta ML from VI-based start: {ml_from_vi_theta_hat:.6g}")
print(f"theta absolute difference: {theta_abs_diff:.6g}")
print(f"theta relative difference: {theta_rel_diff:.6g}")
print()
print("Total OD flow comparison")
print(f"total OD flow VI mean: {vi_total_flow_mean:.6g}")
print(f"total OD flow VI sd: {vi_total_flow_sd:.6g}")
print(f"total OD flow ML: {ml_total_flow:.6g}")
print(f"total OD flow ML from VI-based start: {ml_from_vi_total_flow:.6g}")
print(f"total OD flow absolute difference: {total_flow_abs_diff:.6g}")
print(f"total OD flow relative difference: {total_flow_rel_diff:.6g}")
print()
print("OD vector comparison")
print(f"max |f_ML - f_VI_mean|: {max_abs_flow_diff:.6g}")
print(f"mean |f_ML - f_VI_mean|: {mean_abs_flow_diff:.6g}")
print()
print("ML diagnostics")
print(f"success: {ml_result.success}")
print(f"message: {ml_result.message}")
print(f"objective: {ml_result.objective_value:.6g}")
print(f"log-likelihood: {ml_result.loglikelihood:.6g}")
print(f"gradient norm: {ml_result.gradient_norm:.6g}")
print()
print("ML-from-VI diagnostics")
print(f"success: {ml_from_vi_result.success}")
print(f"message: {ml_from_vi_result.message}")
print(f"objective: {ml_from_vi_result.objective_value:.6g}")
print(f"log-likelihood: {ml_from_vi_result.loglikelihood:.6g}")
print(f"gradient norm: {ml_from_vi_result.gradient_norm:.6g}")
print()
print("Objective diagnostics, evaluated with the ML objective")
print(
    f"at ML solution: loglik={ml_loglik_at_ml:.6g}, "
    f"logprior={ml_logprior_at_ml:.6g}, "
    f"weighted logpost={ml_logpost_at_ml:.6g}"
)
print(
    f"at transformed VI posterior means: loglik={ml_loglik_at_vi_summary:.6g}, "
    f"logprior={ml_logprior_at_vi_summary:.6g}, "
    f"weighted logpost={ml_logpost_at_vi_summary:.6g}"
)
print(
    "at sampled VI posterior draws: "
    f"weighted logpost q05/median/q95="
    f"{np.array2string(vi_sample_logpost_quantiles, precision=6)}"
)
print(
    "best sampled VI draw under ML objective: "
    f"theta={vi_sample_theta_values[int(np.argmax(vi_sample_logpost_values))]:.6g}, "
    f"weighted logpost={float(np.max(vi_sample_logpost_values)):.6g}"
)
print(
    "at ML-from-VI solution: "
    f"loglik={ml_loglik_at_ml_from_vi:.6g}, "
    f"logprior={ml_logprior_at_ml_from_vi:.6g}, "
    f"weighted logpost={ml_logpost_at_ml_from_vi:.6g}"
)

# -----------------------------------------------------------------------------
# 5) Save combined results
# -----------------------------------------------------------------------------
out_path = DATA / "compare_vi_ml_od_theta_results.npz"
np.savez_compressed(
    out_path,
    fingerprint=str(idm.fingerprint),
    fingerprint_payload_json=idm.fingerprint_payload_json,
    f0=np.asarray(f0, dtype=float),
    estimate_theta=estimate_theta,
    fixed_theta=(np.nan if fixed_theta is None else float(fixed_theta)),
    vi_time_s=vi_time_s,
    vi_theta_samples=vi_result.theta_samples,
    vi_f_samples=vi_result.f_samples,
    vi_theta_mean=vi_theta_mean,
    vi_theta_sd=vi_result.theta_sd,
    vi_theta_quantiles=vi_theta_quantiles,
    vi_f_mean=vi_f_mean,
    vi_losses=vi_result.vi.losses,
    ml_time_s=ml_time_s,
    ml_theta_hat=ml_theta_hat,
    ml_f_hat=ml_f_hat,
    ml_z_hat=ml_z_hat,
    ml_u_hat=ml_u_hat,
    ml_parameter_hat=np.asarray(ml_result.theta_hat, dtype=float),
    ml_log_likelihood=ml_result.loglikelihood,
    ml_log_prior=ml_result.logprior,
    ml_prior_weight=ml_result.prior_weight,
    ml_objective_value=ml_result.objective_value,
    ml_gradient=np.asarray(ml_result.gradient, dtype=float),
    ml_gradient_norm=ml_result.gradient_norm,
    ml_hessian=(
        np.asarray([])
        if ml_result.hessian is None
        else np.asarray(ml_result.hessian, dtype=float)
    ),
    ml_covariance_matrix=(
        np.asarray([])
        if ml_result.covariance_matrix is None
        else np.asarray(ml_result.covariance_matrix, dtype=float)
    ),
    ml_standard_errors=(
        np.asarray([])
        if ml_result.standard_errors is None
        else np.asarray(ml_result.standard_errors, dtype=float)
    ),
    ml_optimization_trace=np.asarray(ml_result.optimization_trace, dtype=float),
    ml_success=ml_result.success,
    ml_message=ml_result.message,
    ml_method=ml_result.method,
    ml_runtime_seconds=ml_result.runtime_seconds,
    ml_from_vi_time_s=ml_from_vi_time_s,
    ml_from_vi_theta_hat=ml_from_vi_theta_hat,
    ml_from_vi_f_hat=ml_from_vi_f_hat,
    ml_from_vi_z_hat=ml_from_vi_z_hat,
    ml_from_vi_u_hat=ml_from_vi_u_hat,
    ml_from_vi_parameter_hat=np.asarray(ml_from_vi_result.theta_hat, dtype=float),
    ml_from_vi_log_likelihood=ml_from_vi_result.loglikelihood,
    ml_from_vi_log_prior=ml_from_vi_result.logprior,
    ml_from_vi_objective_value=ml_from_vi_result.objective_value,
    ml_from_vi_gradient_norm=ml_from_vi_result.gradient_norm,
    ml_from_vi_success=ml_from_vi_result.success,
    ml_from_vi_message=ml_from_vi_result.message,
    ml_loglik_at_ml=ml_loglik_at_ml,
    ml_logprior_at_ml=ml_logprior_at_ml,
    ml_logpost_at_ml=ml_logpost_at_ml,
    ml_loglik_at_vi_summary=ml_loglik_at_vi_summary,
    ml_logprior_at_vi_summary=ml_logprior_at_vi_summary,
    ml_logpost_at_vi_summary=ml_logpost_at_vi_summary,
    ml_loglik_at_ml_from_vi=ml_loglik_at_ml_from_vi,
    ml_logprior_at_ml_from_vi=ml_logprior_at_ml_from_vi,
    ml_logpost_at_ml_from_vi=ml_logpost_at_ml_from_vi,
    theta_abs_diff=theta_abs_diff,
    theta_rel_diff=theta_rel_diff,
    max_abs_flow_diff=max_abs_flow_diff,
    mean_abs_flow_diff=mean_abs_flow_diff,
    total_flow_abs_diff=total_flow_abs_diff,
    total_flow_rel_diff=total_flow_rel_diff,
    time_ratio_vi_over_ml=time_ratio,
    vi_objective_draw_indices=vi_objective_draw_indices,
    vi_sample_theta_values=vi_sample_theta_values,
    vi_sample_loglik_values=vi_sample_loglik_values,
    vi_sample_logprior_values=vi_sample_logprior_values,
    vi_sample_logpost_values=vi_sample_logpost_values,
    vi_sample_logpost_quantiles=vi_sample_logpost_quantiles,
)

print(f"Saved comparison results to: {out_path}")