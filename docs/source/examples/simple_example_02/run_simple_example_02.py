"""Example 02: two OD pairs, two partially-overlapping lines, time-varying demand.

This script expects the following files next to it in ./data:
- metadata.json
- stops.csv
- lines.csv
- trips.csv
- stop_times.csv
- time_bins.csv
- demand.csv

Workflow:
1) load the Scenario from the data folder,
2) run the assignment,
3) generate one HTML report per OD-group (so costs/flows match exactly what the model uses),
4) print a compact summary of totals by link type,
5) run Bayesian estimation (VI) for OD + theta using observed measurements.

Constraints:
- No new helper functions are defined in this script.
- Use only the domain/assignment API provided by the package.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import _assign_core, assign, prepare_assignment
from public_transportation.assignment.costs import link_costs
from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.bayesian_estimation import (
    build_vi_report_data,
    compute_all_diagnostics,
    generate_vi_report_html,
    generate_vi_report_plots,
    run_vi,
    save_vi_result,
)
from public_transportation.domain import Scenario
from public_transportation.measurement import (
    apply_mapping_spec,
    build_mapping_spec_strict,
    build_measurement_vectors,
    read_measurements_csv,
    write_mapping_report_html,
)
from public_transportation.measurement.likelihood_jax import measurement_loglik_from_link_flow
from public_transportation.viz.time_expanded_report import write_time_expanded_report_from_assignment

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", True)

DATA = Path(__file__).resolve().parent / "data"

import logging

def make_console_logger(name: str = "public_transportation.vi", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if you rerun in an interactive session
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger

logger = make_console_logger()

# -----------------------------------------------------------------------------
# 1) Load scenario from data
# -----------------------------------------------------------------------------
scenario = Scenario.from_folder(DATA)

# Optional: validate and print any issues
rep = scenario.validate()
if rep.issues:
    print("Scenario validation issues:")
    for it in rep.issues:
        print(f"- [{it.severity.name}] {it.code}: {it.message} ({it.location})")

# -----------------------------------------------------------------------------
# 2) Assign demand (baseline run)
# -----------------------------------------------------------------------------
if scenario.demand is None or not scenario.demand.records:
    raise RuntimeError("No demand records found.")

od_values = jnp.asarray([float(r.flow) for r in scenario.demand.records], dtype=jnp.float32)
config = AssignmentConfig()

artifacts = prepare_assignment(scenario=scenario, config=config)
res = assign(
    od_values=od_values,
    artifacts=artifacts,
    theta=None,
    # No per-group flows here: grouping is an internal feature of the assignment module.
    return_group_link_flows=False,
)

print("Example 02 — overlapping lines, time-varying demand")
print(f"OD records: {len(scenario.demand.records)}")
print(f"Total demand: {float(np.asarray(od_values).sum()):.6g}")
print(f"Theta used: {res.theta:.6g}")

# -----------------------------------------------------------------------------
# 3) Generate reporting (single report; no OD grouping logic in this script)
# -----------------------------------------------------------------------------
_ = res.link_flow
_ = res.link_cost

report_dir = DATA
report_dir.mkdir(parents=True, exist_ok=True)

report_path = report_dir / "time_expanded_report.html"

write_time_expanded_report_from_assignment(
    scenario=scenario,
    assignment=res,
    config=config,
    output_path=report_path,
    title="Example 02 — time-expanded graph report",
    svg_scale_x=1.8,
    svg_scale_y=2.4,
)

print(f"Report written to: {report_path}")

# -----------------------------------------------------------------------------
# 4) Load observed measurements and build aligned vectors (y_obs, y_pred)
# -----------------------------------------------------------------------------
meas_path = report_dir / "measurements_boarding_alighting.csv"
if not meas_path.exists():
    raise RuntimeError(
        f"Missing measurement file: {meas_path}. "
        "Expected 'measurements_boarding_alighting.csv' next to the scenario data."
    )

# Read as MeasurementTable (validated)
measurements_table = read_measurements_csv(meas_path)

# Build the assignment ID manager (explicit)
idm = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)

# Structural mapping for inference (JAX-safe spec)
msr = build_mapping_spec_strict(
    id_manager=idm,
    table=measurements_table,
    include_link_lists_for_report=True,  # set False if you don’t need link lists
)

y_obs = msr.y_obs
spec = msr.spec

# Predicted vector at baseline assignment (for quick sanity check)
y_pred = apply_mapping_spec(link_flow=res.link_flow, spec=spec)

y_obs_np = np.asarray(y_obs, dtype=float)
y_pred_np = np.asarray(y_pred, dtype=float)

if y_obs_np.shape != y_pred_np.shape:
    raise RuntimeError(f"Shape mismatch: y_obs {y_obs_np.shape} vs y_pred {y_pred_np.shape}")

print()
print("Measurement mapping summary")
print(f"Measurements: {int(y_obs_np.shape[0])}")
print(f"y_obs sum: {float(y_obs_np.sum()):.6g}")
print(f"y_pred sum: {float(y_pred_np.sum()):.6g}")

# Optional: write a mapping report for validation
mapping_report_path = report_dir / "measurement_mapping_report.html"

mvr = build_measurement_vectors(
    assignment=res,
    id_manager=idm,
    table=measurements_table,
    include_link_lists_for_report=True,
    enrich_predictions=True,  # fills predicted_value in MappingInfo entries
)

write_mapping_report_html(
    info=mvr.info,
    id_manager=idm,
    assignment_link_flow=np.asarray(res.link_flow, dtype=float),
    output_path=mapping_report_path,
)

print(f"Mapping report written to: {mapping_report_path}")

# -----------------------------------------------------------------------------
# 5) Bayesian estimation (VI): infer OD (via log-deviations z) and theta
# -----------------------------------------------------------------------------
# Baseline OD vector f0 is the scenario demand itself (strict alignment = scenario order here)
# We keep the inference parameterization:
#   z in R^(num_od), f = f0 * exp(z)  (positivity guaranteed)
#   u in R, theta = exp(u)            (positivity guaranteed)
num_od = int(od_values.shape[0])
f0 = jnp.asarray(od_values, dtype=jnp.float32)
parameter_names = [
    f"log-deviation {r.origin_stop_id}->{r.dest_stop_id} / {r.time_bin_id}"
    for r in scenario.demand.records
] + ["log(theta)"]

# Precompute base link costs once (inference will reuse it)
base_link_cost = jnp.asarray(
    link_costs(graph=artifacts.graph, cost_parts=artifacts.cost_parts, config=artifacts.config),
    dtype=jnp.float32,
)

# Extract ODGroups arrays once (avoid passing Python objects into JAX tracing)
odg = artifacts.od_groups
group_dest_node = jnp.asarray(odg.group_dest_node)
group_link_mask = jnp.asarray(odg.group_link_mask)
od_origin_node = jnp.asarray(odg.od_origin_node)
group_od_index_padded = jnp.asarray(odg.group_od_index_padded)
group_od_mask = jnp.asarray(odg.group_od_mask)

# Likelihood inputs
y_obs = jnp.asarray(y_obs_np, dtype=jnp.float32)
spec_num_measurements = int(spec.num_measurements)
SPEC_NUM_MEASUREMENTS = spec_num_measurements  # Python int constant for JAX jit
spec_measurement_index = jnp.asarray(spec.measurement_index, dtype=jnp.int32)
spec_link_index = jnp.asarray(spec.link_index, dtype=jnp.int32)

# Fixed measurement parameters (keep simple for now)
rho = jnp.asarray(1.0, dtype=jnp.float32)   # detection rate (1.0 => flows are already "counts")
r_nb = jnp.asarray(50.0, dtype=jnp.float32) # NB dispersion (larger => closer to Poisson)

# VI parameter vector: [z(0:num_od), u_theta]
dim = num_od + 1

# Priors (in the unconstrained space):
#  z_i ~ Normal(0, sigma_z)
#  u   ~ Normal(mu_u, sigma_u)   where theta = exp(u)
sigma_z = 1.0
mu_u = float(np.log(float(res.theta)))  # center prior at baseline theta used by assignment
sigma_u = 0.7

sqrt_2pi = jnp.asarray(np.sqrt(2.0 * np.pi), dtype=jnp.float32)

def logprior(theta_vec: jnp.ndarray) -> jnp.ndarray:
    z = theta_vec[:num_od]
    u = theta_vec[num_od]

    # Normal logpdf: -0.5*((x-m)/s)^2 - log(s*sqrt(2pi))
    lp_z = (-0.5 * jnp.sum((z / sigma_z) ** 2)) - z.shape[0] * jnp.log(sigma_z * sqrt_2pi)
    lp_u = (-0.5 * ((u - mu_u) / sigma_u) ** 2) - jnp.log(sigma_u * sqrt_2pi)
    return (lp_z + lp_u).reshape(())

def loglik(theta_vec: jnp.ndarray, data: dict) -> jnp.ndarray:
    # NOTE: NumPyro initializes `theta` by sampling from N(0, I).
    # With exp() reparameterizations, rare but not-too-rare initial draws can
    # produce extreme values (very large theta or OD multipliers), which can
    # make the assignment/likelihood numerically unstable (NaNs) under JIT.
    # We therefore clamp the unconstrained parameters to a safe range.
    z = jnp.clip(theta_vec[:num_od], -4.0, 4.0)
    u = jnp.clip(theta_vec[num_od], -3.0, 3.0)
    theta_pos = jnp.exp(u).astype(jnp.float32)

    # f = f0 * exp(z)  (positivity guaranteed)
    f = data["f0"] * jnp.exp(z)

    # Assignment core -> link_flow (JAX-safe)
    link_flow, _ = _assign_core(
        graph=data["graph"],
        od_values=f,
        base_link_cost=data["base_link_cost"],
        theta=theta_pos,
        group_dest_node=data["group_dest_node"],
        group_link_mask=data["group_link_mask"],
        od_origin_node=data["od_origin_node"],
        group_od_index_padded=data["group_od_index_padded"],
        group_od_mask=data["group_od_mask"],
        return_group_link_flows=False,
    )

    ll = measurement_loglik_from_link_flow(
        link_flow=link_flow,
        y_obs=data["y_obs"],
        theta=theta_pos,
        rho=data["rho"],
        r=data["r_nb"],
        spec_num_measurements=SPEC_NUM_MEASUREMENTS,
        spec_measurement_index=data["spec_measurement_index"],
        spec_link_index=data["spec_link_index"],
    ).reshape(())

    # If ll becomes non-finite (rare with clamping), penalize softly.
    return jnp.where(jnp.isfinite(ll), ll, jnp.asarray(-1.0e30, dtype=jnp.float32))

data = {
    "graph": artifacts.graph,
    "base_link_cost": base_link_cost,
    "group_dest_node": group_dest_node,
    "group_link_mask": group_link_mask,
    "od_origin_node": od_origin_node,
    "group_od_index_padded": group_od_index_padded,
    "group_od_mask": group_od_mask,
    "f0": f0,
    "y_obs": y_obs,
    "rho": rho,
    "r_nb": r_nb,
    "spec_measurement_index": spec_measurement_index,
    "spec_link_index": spec_link_index,
}

print()

print("Running VI (NumPyro) for OD (z) + theta ...")
print(f"Baseline assignment link_flow: min={float(np.asarray(res.link_flow).min()):.6g}, max={float(np.asarray(res.link_flow).max()):.6g}")

vi = run_vi(
    dim=dim,
    data=data,
    loglik=loglik,
    logprior=logprior,
    guide="auto_diag",
    use_base_normal_correction=True,  # logprior is absolute in theta_vec space
    num_steps=300,
    learning_rate=1e-2,
    seed=0,
    num_posterior_draws=1000,
    logger=logger,
)

# -----------------------------------------------------------------------------
# 6) Save VI results and generate report
# -----------------------------------------------------------------------------
run_dir = DATA / datetime.now().strftime("vi_run_%Y%m%d_%H%M%S")
run_dir.mkdir(parents=True, exist_ok=True)

# Save VI result (core object)
save_vi_result(vi, run_dir / "vi")


# Generate report diagnostics and plots
diagnostics = compute_all_diagnostics(vi, parameter_names=parameter_names)
figures_dir = run_dir / "figures"
figure_files = generate_vi_report_plots(
    vi,
    figures_dir,
    diagnostics=diagnostics,
    parameter_names=parameter_names,
)

report_data = build_vi_report_data(
    vi,
    diagnostics,
    figure_files=figure_files,
)

report_path = run_dir / "vi_report.html"
generate_vi_report_html(report_data, report_path)

print(f"VI report written to: {report_path}")

print(f"VI report figures written to: {figures_dir}")
print(f"Generated figure files: {figure_files}")

samples = vi.posterior_samples_theta  # shape (S, dim)
z_s = samples[:, :num_od]
u_s = samples[:, num_od]
theta_s = np.exp(u_s)

f_s = np.asarray(f0)[None, :] * np.exp(z_s)

# (optional but useful) save OD posterior samples
np.save(run_dir / "theta_samples.npy", samples)
np.save(run_dir / "od_samples.npy", f_s)

print()
print("Posterior summary (VI samples)")
print(f"theta: mean={theta_s.mean():.6g}, sd={theta_s.std():.6g}, "
      f"p05={np.quantile(theta_s,0.05):.6g}, p95={np.quantile(theta_s,0.95):.6g}")
print(f"Total OD flow: mean={f_s.sum(axis=1).mean():.6g}, sd={f_s.sum(axis=1).std():.6g}")

# Optional: write posterior means back to console in scenario order
f_mean = f_s.mean(axis=0)
np.save(run_dir / "od_mean.npy", f_mean)
print()
print("OD posterior mean (scenario order):")
for k, r in enumerate(scenario.demand.records):
    print(
        f"- {r.origin_stop_id}->{r.dest_stop_id} (tb={getattr(r,'time_bin_id',getattr(r,'time_bin_index','?'))}): "
        f"{f_mean[k]:.6g}"
    )