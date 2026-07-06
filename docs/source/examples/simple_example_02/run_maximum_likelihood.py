"""
Example 02 — Maximum likelihood estimation: infer OD (via log-deviations z) and theta.

This script expects the same ./data files as the Bayesian example.

Workflow:
1) load Scenario + prepare assignment artifacts,
2) read measurements and build strict mapping spec,
3) build baseline OD vector f0,
4) build the OD/theta log-likelihood,
5) run maximum likelihood or penalized maximum likelihood,
6) save results (npz).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.domain import Scenario
from public_transportation.estimation.maximum_likelihood import MLConfig, run_ml
from public_transportation.inference.assignment_adapter import build_assignment_inputs
from public_transportation.inference.likelihood import (
    loglikelihood_from_link_flow,
    prepare_likelihood_inputs,
)
from public_transportation.inference.model import make_forward_inputs, forward_model
from public_transportation.inference.priors import build_f0_from_scenario_demand
from public_transportation.measurement import build_mapping_spec_strict, read_measurements_csv

DATA = Path(__file__).resolve().parent / "data"


def make_console_logger(
    name: str = "public_transportation.ml",
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
    """Normal log-density, implemented directly to avoid a NumPyro dependency."""
    x = jnp.asarray(x)
    loc = jnp.asarray(loc, dtype=x.dtype)
    scale = jnp.asarray(scale, dtype=x.dtype)
    log_two_pi = jnp.asarray(np.log(2.0 * np.pi), dtype=x.dtype)
    return -0.5 * ((x - loc) / scale) ** 2 - jnp.log(scale) - 0.5 * log_two_pi


logger = make_console_logger()


# -----------------------------------------------------------------------------
# 1) Load scenario + prepare assignment artifacts
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


# -----------------------------------------------------------------------------
# 2) Measurements -> strict mapping spec
# -----------------------------------------------------------------------------
meas_path = DATA / "measurements_boarding_alighting.csv"
if not meas_path.exists():
    raise RuntimeError(
        f"Missing measurement file: {meas_path}. "
        "Expected 'measurements_boarding_alighting.csv' next to the scenario data."
    )

table = read_measurements_csv(meas_path)
msr = build_mapping_spec_strict(
    id_manager=idm,
    table=table,
    include_link_lists_for_report=False,
)


# -----------------------------------------------------------------------------
# 3) Baseline OD vector f0
# -----------------------------------------------------------------------------
f0 = build_f0_from_scenario_demand(
    scenario=scenario,
    id_manager=idm,
    dtype=jnp.float32,
)
f0 = jnp.asarray(f0, dtype=jnp.float32)


# -----------------------------------------------------------------------------
# 4) Build the OD/theta log-likelihood
# -----------------------------------------------------------------------------
baseline_theta = 10.0
sigma_z = 100.0
sigma_u = 10.0
rho = 1.0
nb_dispersion = 50.0
z_clip = 6.0
u_clip = 6.0

if baseline_theta <= 0.0:
    raise ValueError("baseline_theta must be strictly positive.")
if sigma_z <= 0.0:
    raise ValueError("sigma_z must be strictly positive.")
if sigma_u <= 0.0:
    raise ValueError("sigma_u must be strictly positive.")

assignment_inputs = build_assignment_inputs(artifacts=artifacts)
forward_inputs = make_forward_inputs(f0=f0, spec=msr.spec)
prepared_likelihood = prepare_likelihood_inputs(
    y_obs=jnp.asarray(msr.y_obs, dtype=jnp.float32),
    spec=msr.spec,
)

num_od = int(f0.shape[0])
dim = num_od + 1
mu_u = float(np.log(baseline_theta))

data: dict[str, Any] = {
    "rho": jnp.asarray(rho, dtype=jnp.float32).reshape(()),
    "r_nb": jnp.asarray(nb_dispersion, dtype=jnp.float32).reshape(()),
}


def split_theta(theta_vec: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Split optimization vector into OD log-deviations z and log-theta u."""
    z = theta_vec[:num_od]
    u = theta_vec[num_od]
    return z, u


def loglik(theta_vec: jnp.ndarray, data_: dict[str, Any]) -> jnp.ndarray:
    """Negative-binomial measurement log-likelihood for OD/theta parameters."""
    z, u = split_theta(theta_vec)
    z = jnp.clip(z, -z_clip, z_clip)
    u = jnp.clip(u, -u_clip, u_clip)

    theta = jnp.exp(u)
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
    """Same Gaussian prior specification as the Bayesian example."""
    z, u = split_theta(theta_vec)
    z = jnp.clip(z, -z_clip, z_clip)
    u = jnp.clip(u, -u_clip, u_clip)

    return normal_logpdf(z, 0.0, sigma_z).sum() + normal_logpdf(u, mu_u, sigma_u).sum()


# -----------------------------------------------------------------------------
# 5) Run maximum likelihood estimation
# -----------------------------------------------------------------------------
ml_cfg = MLConfig(
    method="BFGS",
    maxiter=1000,
    gtol=1.0e-6,
    # 0.0 gives pure ML. 1.0 gives penalized ML/MAP using the same prior as VI.
    prior_weight=0.0,
    compute_hessian=True,
    log_every=10,
)

theta0 = jnp.zeros((dim,), dtype=jnp.float32)
theta0 = theta0.at[num_od].set(mu_u)

print("Running maximum likelihood estimation for OD (z) + theta ...")
ml_result = run_ml(
    dim=dim,
    data=data,
    loglik=loglik,
    logprior=logprior,
    theta0=theta0,
    config=ml_cfg,
    logger=logger,
)

z_hat = np.asarray(ml_result.theta_hat[:num_od], dtype=float)
u_hat = float(ml_result.theta_hat[num_od])
theta_hat = float(np.exp(u_hat))
f_hat = np.asarray(f0, dtype=float) * np.exp(z_hat)

print()
print("Maximum likelihood summary")
print(f"success: {ml_result.success}")
print(f"message: {ml_result.message}")
print(f"objective: {ml_result.objective_value:.6g}")
print(f"log-likelihood: {ml_result.loglikelihood:.6g}")
print(f"log-prior: {ml_result.logprior:.6g}")
print(f"prior weight: {ml_result.prior_weight:.6g}")
print(f"theta_hat: {theta_hat:.6g}")
print(f"Total OD flow: {f_hat.sum():.6g}")
print(f"gradient norm: {ml_result.gradient_norm:.6g}")


# -----------------------------------------------------------------------------
# 6) Save results
# -----------------------------------------------------------------------------
out_path = DATA / "ml_od_theta_results.npz"
np.savez_compressed(
    out_path,
    fingerprint=str(idm.fingerprint),
    fingerprint_payload_json=idm.fingerprint_payload_json,
    f0=np.asarray(f0, dtype=float),
    theta_hat=theta_hat,
    f_hat=f_hat,
    z_hat=z_hat,
    u_hat=u_hat,
    parameter_hat=np.asarray(ml_result.theta_hat, dtype=float),
    log_likelihood=ml_result.loglikelihood,
    log_prior=ml_result.logprior,
    prior_weight=ml_result.prior_weight,
    objective_value=ml_result.objective_value,
    gradient=np.asarray(ml_result.gradient, dtype=float),
    gradient_norm=ml_result.gradient_norm,
    hessian=(
        np.asarray([])
        if ml_result.hessian is None
        else np.asarray(ml_result.hessian, dtype=float)
    ),
    covariance_matrix=(
        np.asarray([])
        if ml_result.covariance_matrix is None
        else np.asarray(ml_result.covariance_matrix, dtype=float)
    ),
    standard_errors=(
        np.asarray([])
        if ml_result.standard_errors is None
        else np.asarray(ml_result.standard_errors, dtype=float)
    ),
    optimization_trace=np.asarray(ml_result.optimization_trace, dtype=float),
    success=ml_result.success,
    message=ml_result.message,
    method=ml_result.method,
    runtime_seconds=ml_result.runtime_seconds,
)

print(f"Saved ML results to: {out_path}")