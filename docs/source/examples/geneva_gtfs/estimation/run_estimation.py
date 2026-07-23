"""Estimate the Geneva OD matrix by ML, MAP, or Bayesian variational inference."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.domain import Scenario, read_fixed_demand_csv
from public_transportation.estimation.bayesian.config import VIConfig
from public_transportation.estimation.maximum_likelihood import MLConfig, run_ml
from public_transportation.inference.maximum_likelihood_pipeline import build_od_theta_ml_problem
from public_transportation.inference.od_parameter_layout import build_od_parameter_layout
from public_transportation.inference.pipeline import ODThetaEstimationRequest, estimate_od_theta_vi
from public_transportation.inference.priors import build_f0_from_scenario_demand
from public_transportation.measurement import build_mapping_spec_strict, read_measurements_csv


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PREPROCESSING = ROOT / "pre_processing" / "results"
RESULTS = Path(__file__).resolve().parent / "results"
NETWORK_FILES = ("metadata.json", "stops.csv", "lines.csv", "trips.csv", "stop_times.csv", "time_bins.csv")


def _scenario_folder() -> tempfile.TemporaryDirectory[str]:
    temporary = tempfile.TemporaryDirectory()
    target = Path(temporary.name)
    for name in NETWORK_FILES:
        shutil.copy2(DATA / name, target / name)
    shutil.copy2(PREPROCESSING / "demand.csv", target / "demand.csv")
    return temporary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("ml", "map", "vi"), default="ml")
    parser.add_argument("--maxiter", type=int, default=200, help="ML/MAP optimizer iterations")
    parser.add_argument("--vi-steps", type=int, default=1500)
    parser.add_argument("--posterior-draws", type=int, default=500)
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    with _scenario_folder() as folder:
        scenario = Scenario.from_folder(folder, strict=True)
        fixed = read_fixed_demand_csv(DATA / "fixed_demand.csv", scenario=scenario)
        layout = build_od_parameter_layout(scenario=scenario, fixed_demand=fixed)
        artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
        id_manager = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)
        measurements = read_measurements_csv(PREPROCESSING / "measurements_boarding_alighting.csv")
        mapped = build_mapping_spec_strict(
            id_manager=id_manager,
            table=measurements,
            include_link_lists_for_report=False,
        )
        f0 = jnp.asarray(
            build_f0_from_scenario_demand(scenario=scenario, id_manager=id_manager),
            dtype=jnp.float32,
        )
        request = ODThetaEstimationRequest(
            fingerprint=str(id_manager.fingerprint),
            fingerprint_payload_json=id_manager.fingerprint_payload_json,
            f0=f0,
            y_obs=jnp.asarray(mapped.y_obs, dtype=jnp.float32),
            mapping_spec=mapped.spec,
            baseline_theta=5.0,
            od_layout=layout,
            estimate_theta=False,
            fixed_theta=5.0,
            sigma_z=1.0,
            sigma_u=0.7,
            rho=1.0,
            nb_dispersion=100.0,
            z_clip=6.0,
            u_clip=6.0,
            vi=VIConfig(
                guide="auto_lowrank",
                lowrank_rank=10,
                use_base_normal_correction=True,
                num_steps=args.vi_steps,
                learning_rate=1.0e-2,
                seed=0,
                num_posterior_draws=args.posterior_draws,
                log_every=100,
            ),
            assignment_artifacts=artifacts,
        )

        if args.method == "vi":
            result = estimate_od_theta_vi(request)
            f_hat = np.asarray(result.f_mean, dtype=float)
            theta_hat = float(result.theta_mean)
            success = True
            objective = float(np.asarray(result.vi.losses)[-1])
            runtime = float(result.vi.runtime_seconds)
            parameter = np.asarray([])
            runtime_profile = result.runtime_profile
            method_arrays = {
                "f_samples": np.asarray(result.f_samples, dtype=float),
                "f_q05": np.quantile(np.asarray(result.f_samples, dtype=float), 0.05, axis=0),
                "f_q50": np.quantile(np.asarray(result.f_samples, dtype=float), 0.50, axis=0),
                "f_q95": np.quantile(np.asarray(result.f_samples, dtype=float), 0.95, axis=0),
                "theta_samples": np.asarray(result.theta_samples, dtype=float),
                "losses": np.asarray(result.vi.losses, dtype=float),
            }
        else:
            problem = build_od_theta_ml_problem(request)
            result = run_ml(
                dim=problem.dim,
                data=problem.data,
                loglik=problem.loglik,
                logprior=problem.logprior,
                theta0=problem.theta0,
                config=MLConfig(
                    method="L-BFGS-B",
                    maxiter=args.maxiter,
                    gtol=1.0e-5,
                    prior_weight=(0.0 if args.method == "ml" else 1.0),
                    compute_hessian=False,
                    log_every=10,
                ),
            )
            f_hat, theta_hat = problem.decode(result.theta_hat)
            success = bool(result.success)
            objective = float(result.objective_value)
            runtime = float(result.runtime_seconds)
            parameter = np.asarray(result.theta_hat, dtype=float)
            runtime_profile = problem.runtime_profile
            optimizer_message = str(result.message)
            gradient_norm = float(result.gradient_norm)
            method_arrays = {}

        output = RESULTS / f"{args.method}_results.npz"
        np.savez_compressed(
            output,
            method=args.method,
            fingerprint=str(id_manager.fingerprint),
            f0=np.asarray(f0, dtype=float),
            f_hat=np.asarray(f_hat, dtype=float),
            theta_hat=theta_hat,
            parameter_hat=parameter,
            success=success,
            objective=objective,
            runtime_seconds=runtime,
            free_od_indices=np.asarray(layout.free_od_indices, dtype=np.int64),
            fixed_od_indices=np.asarray(layout.fixed_od_indices, dtype=np.int64),
            fixed_od_values=np.asarray(layout.fixed_od_values, dtype=float),
            **method_arrays,
        )
        summary = {
            "method": args.method,
            "success": success,
            "objective": objective,
            "runtime_seconds": runtime,
            "theta_hat": theta_hat,
            "total_estimated_demand": float(np.asarray(f_hat).sum()),
            **(
                {}
                if args.method == "vi"
                else {
                    "optimizer_message": optimizer_message,
                    "gradient_norm": gradient_norm,
                }
            ),
            **runtime_profile.as_dict(),
        }
        (RESULTS / f"{args.method}_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
