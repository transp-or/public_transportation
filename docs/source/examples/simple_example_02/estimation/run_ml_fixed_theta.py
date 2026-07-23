"""
Maximum-likelihood estimation with theta fixed to 1.0.

Reads ../data and ../pre_processing/results. Writes results/ml_fixed_theta_results.npz.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp
import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.domain import Scenario, read_fixed_demand_csv
from public_transportation.estimation.maximum_likelihood import MLConfig, run_ml
from public_transportation.inference.maximum_likelihood_pipeline import build_od_theta_ml_problem
from public_transportation.inference.od_parameter_layout import build_od_parameter_layout
from public_transportation.inference.pipeline import (
    ODThetaEstimationRequest,
)
from public_transportation.inference.priors import build_f0_from_scenario_demand
from public_transportation.measurement import build_mapping_spec_strict, read_measurements_csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PREPROCESSING_RESULTS = ROOT / "pre_processing" / "results"
RESULTS = Path(__file__).resolve().parent / "results"

NETWORK_FILES = [
    "metadata.json",
    "stops.csv",
    "lines.csv",
    "trips.csv",
    "stop_times.csv",
    "time_bins.csv",
]

baseline_theta = 1.0
sigma_z = 100.0
sigma_u = 10.0
rho = 1.0
nb_dispersion = 50.0
z_clip = 6.0
u_clip = 6.0


def make_console_logger(
    name: str = "public_transportation.estimation",
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



def prepare_scenario_folder() -> tempfile.TemporaryDirectory[str]:
    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)

    for name in NETWORK_FILES:
        src = DATA / name
        if not src.exists():
            raise FileNotFoundError(f"Missing required file: {src}")
        shutil.copy2(src, tmp_path / name)

    demand_path = PREPROCESSING_RESULTS / "demand.csv"
    if not demand_path.exists():
        raise FileNotFoundError(
            f"Missing generated demand file: {demand_path}. "
            "Run pre_processing/run_preprocessing.py first."
        )
    shutil.copy2(demand_path, tmp_path / "demand.csv")

    return tmp


def prepare_shared_inputs():
    measurements_path = PREPROCESSING_RESULTS / "measurements_boarding_alighting.csv"
    if not measurements_path.exists():
        raise FileNotFoundError(
            f"Missing generated measurement file: {measurements_path}. "
            "Run pre_processing/run_preprocessing.py first."
        )

    scenario = Scenario.from_folder(Path(SCENARIO_DIR))
    fixed_demand = read_fixed_demand_csv(DATA / "fixed_demand.csv", scenario=scenario)
    od_layout = build_od_parameter_layout(scenario=scenario, fixed_demand=fixed_demand)
    rep = scenario.validate()
    if rep.issues:
        print("Scenario validation issues:")
        for issue in rep.issues:
            print(f"- [{issue.severity.name}] {issue.code}: {issue.message} ({issue.location})")

    assignment_config = AssignmentConfig()
    artifacts = prepare_assignment(scenario=scenario, config=assignment_config)
    idm = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)

    table = read_measurements_csv(measurements_path)
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
    return scenario, artifacts, idm, msr, f0, od_layout




def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    logger = make_console_logger("public_transportation.ml")
    estimate_theta = False
    fixed_theta = baseline_theta

    with prepare_scenario_folder() as scenario_dir:
        global SCENARIO_DIR
        SCENARIO_DIR = scenario_dir
        _, artifacts, idm, msr, f0, od_layout = prepare_shared_inputs()

        ml_request = ODThetaEstimationRequest(
            fingerprint=str(idm.fingerprint),
            fingerprint_payload_json=idm.fingerprint_payload_json,
            f0=f0,
            y_obs=jnp.asarray(msr.y_obs, dtype=jnp.float32),
            mapping_spec=msr.spec,
            baseline_theta=float(baseline_theta),
            od_layout=od_layout,
            estimate_theta=estimate_theta,
            fixed_theta=fixed_theta,
            sigma_z=float(sigma_z),
            sigma_u=float(sigma_u),
            rho=float(rho),
            nb_dispersion=float(nb_dispersion),
            z_clip=float(z_clip),
            u_clip=float(u_clip),
            assignment_artifacts=artifacts,
        )
        ml_problem = build_od_theta_ml_problem(ml_request)

        ml_cfg = MLConfig(
            method="BFGS",
            maxiter=1000,
            gtol=1.0e-6,
            prior_weight=0.0,
            compute_hessian=True,
            log_every=10,
        )

        print("Running maximum likelihood...")
        print(f"estimate theta: {estimate_theta}")
        if fixed_theta is not None:
            print(f"fixed theta: {fixed_theta:.6g}")

        t0 = perf_counter()
        ml_result = run_ml(
            dim=ml_problem.dim,
            data=ml_problem.data,
            loglik=ml_problem.loglik,
            logprior=ml_problem.logprior,
            theta0=ml_problem.theta0,
            config=ml_cfg,
            logger=logger,
        )
        ml_time_s = perf_counter() - t0

        f_hat, theta_hat = ml_problem.decode(ml_result.theta_hat)
        z_hat = np.asarray(ml_result.theta_hat[: ml_problem.num_free_od], dtype=float)
        u_hat = float(np.log(theta_hat))

        out_path = RESULTS / "ml_fixed_theta_results.npz"
        np.savez_compressed(
            out_path,
            fingerprint=str(idm.fingerprint),
            fingerprint_payload_json=idm.fingerprint_payload_json,
            f0=np.asarray(f0, dtype=float),
            num_od_total=od_layout.num_od_total,
            num_free_od=od_layout.num_free,
            num_fixed_od=od_layout.num_fixed,
            free_od_indices=np.asarray(od_layout.free_od_indices, dtype=np.int64),
            fixed_od_indices=np.asarray(od_layout.fixed_od_indices, dtype=np.int64),
            fixed_od_values=np.asarray(od_layout.fixed_od_values, dtype=float),
            od_layout_fingerprint=od_layout.fingerprint,
            od_layout_payload_json=od_layout.fingerprint_payload_json,
            compact_layout_fingerprint=ml_problem.compact_layout_fingerprint,
            compact_layout_payload_json=ml_problem.compact_layout_payload_json,
            **{
                f"runtime_{key}": value
                for key, value in ml_problem.runtime_profile.as_dict().items()
                if not key.endswith("fingerprint")
            },
            estimate_theta=estimate_theta,
            fixed_theta=(np.nan if fixed_theta is None else float(fixed_theta)),
            ml_time_s=ml_time_s,
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
            hessian=(np.asarray([]) if ml_result.hessian is None else np.asarray(ml_result.hessian, dtype=float)),
            covariance_matrix=(np.asarray([]) if ml_result.covariance_matrix is None else np.asarray(ml_result.covariance_matrix, dtype=float)),
            standard_errors=(np.asarray([]) if ml_result.standard_errors is None else np.asarray(ml_result.standard_errors, dtype=float)),
            optimization_trace=np.asarray(ml_result.optimization_trace, dtype=float),
            success=ml_result.success,
            message=ml_result.message,
            method=ml_result.method,
            runtime_seconds=ml_result.runtime_seconds,
        )

        print()
        print("Maximum likelihood summary")
        for line in ml_problem.runtime_profile.format_lines():
            print(line)
        print(f"runtime: {ml_time_s:.3f} s")
        print(f"success: {ml_result.success}")
        print(f"message: {ml_result.message}")
        print(f"theta_hat: {theta_hat:.6g}")
        print(f"total OD flow: {float(f_hat.sum()):.6g}")
        print(f"gradient norm: {ml_result.gradient_norm:.6g}")
        print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
