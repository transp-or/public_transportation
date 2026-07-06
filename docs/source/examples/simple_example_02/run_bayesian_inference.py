"""
Example 02 — Bayesian estimation only (VI): infer OD (via log-deviations z) and theta.

This script expects the following files next to it in ./data:
- metadata.json
- stops.csv
- lines.csv
- trips.csv
- stop_times.csv
- time_bins.csv
- demand.csv
- measurements_boarding_alighting.csv

Workflow (minimal):
1) load Scenario + prepare assignment artifacts,
2) read measurements and build strict mapping spec,
3) build baseline OD vector f0 (from scenario demand, aligned to assignment OD indexing),
4) run VI inference (OD + theta),
5) save results (npz).
"""

from __future__ import annotations

import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.domain import Scenario
from public_transportation.estimation.bayesian.config import VIConfig
from public_transportation.inference.pipeline import ODThetaEstimationRequest, estimate_od_theta_vi
from public_transportation.inference.priors import build_f0_from_scenario_demand
from public_transportation.measurement import build_mapping_spec_strict, read_measurements_csv

DATA = Path(__file__).resolve().parent / "data"


def make_console_logger(name: str = "public_transportation.vi", level: int = logging.INFO) -> logging.Logger:
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

config = AssignmentConfig()
artifacts = prepare_assignment(scenario=scenario, config=config)

# Build ID manager (assignment indexing)
idm = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)

# -----------------------------------------------------------------------------
# 2) Measurements -> strict mapping spec (JAX-safe aggregation spec)
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
# 3) Baseline OD vector f0 (aligned to assignment OD indexing)
# -----------------------------------------------------------------------------
f0 = build_f0_from_scenario_demand(scenario=scenario, id_manager=idm, dtype=jnp.float32)

# -----------------------------------------------------------------------------
# 4) Run VI inference (OD + theta)
# -----------------------------------------------------------------------------

# baseline theta used to center the prior (pick something reasonable)
# If your artifacts/config exposes a default theta elsewhere, plug it here.
baseline_theta = 10.0  # minutes (adjust if you have a better baseline)

vi_cfg = VIConfig(
    guide="auto_diag",
    lowrank_rank=None,
    use_base_normal_correction=True,
    num_steps=4000,
    learning_rate=1e-2,
    seed=0,
    num_posterior_draws=1000,
    log_every=100,
)

request = ODThetaEstimationRequest(
    fingerprint=str(idm.fingerprint),
    fingerprint_payload_json=idm.fingerprint_payload_json,
    f0=jnp.asarray(f0),
    y_obs=jnp.asarray(msr.y_obs, dtype=jnp.float32),
    mapping_spec=msr.spec,
    baseline_theta=float(baseline_theta),
    sigma_z=100.0,
    sigma_u=10,
    rho=1.0,
    nb_dispersion=50.0,
    z_clip=6.0,
    u_clip=6.0,
    vi=vi_cfg,
    assignment_artifacts=artifacts,
    logger=logger,
)

print("Running VI (NumPyro) for OD (z) + theta ...")
result = estimate_od_theta_vi(request)

print()
print("Posterior summary (VI samples)")
print(
    f"theta: mean={result.theta_mean:.6g}, sd={result.theta_sd:.6g}, "
    f"p05={np.quantile(result.theta_samples, 0.05):.6g}, "
    f"p95={np.quantile(result.theta_samples, 0.95):.6g}"
)
print(f"Total OD flow: mean={result.f_samples.sum(axis=1).mean():.6g}, sd={result.f_samples.sum(axis=1).std():.6g}")

# -----------------------------------------------------------------------------
# 5) Save results
# -----------------------------------------------------------------------------
out_path = DATA / "vi_od_theta_results.npz"
np.savez_compressed(
    out_path,
    fingerprint=result.fingerprint,
    fingerprint_payload_json=(
        "" if result.fingerprint_payload_json is None else result.fingerprint_payload_json
    ),
    f0=result.f0,
    theta_samples=result.theta_samples,
    f_samples=result.f_samples,
    theta_mean=result.theta_mean,
    theta_sd=result.theta_sd,
    f_mean=result.f_mean,
    vi_losses=result.vi.losses,
)
print(f"Saved VI results to: {out_path}")