"""
Run both Bayesian VI and maximum-likelihood estimation for simple_example_01.

Reads:
    ../data/
    ../pre_processing/results/demand.csv
    ../pre_processing/results/measurements_boarding_alighting.csv

Writes:
    results/compare_vi_ml_od_theta_results.npz
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
from public_transportation.estimation.bayesian.config import VIConfig
from public_transportation.estimation.maximum_likelihood import MLConfig, run_ml
from public_transportation.inference.maximum_likelihood_pipeline import build_od_theta_ml_problem
from public_transportation.inference.od_parameter_layout import build_od_parameter_layout
from public_transportation.inference.pipeline import (
    ODThetaEstimationRequest,
    estimate_od_theta_vi,
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

baseline_theta = 5.0
sigma_z = 100.0
sigma_u = 10.0
rho = 1.0
nb_dispersion = 50.0
z_clip = 6.0
u_clip = 6.0

estimate_theta = False
fixed_theta = baseline_theta


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


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    logger = make_console_logger()

    measurements_path = PREPROCESSING_RESULTS / "measurements_boarding_alighting.csv"
    if not measurements_path.exists():
        raise FileNotFoundError(
            f"Missing generated measurement file: {measurements_path}. "
            "Run pre_processing/run_preprocessing.py first."
        )

    if estimate_theta:
        if fixed_theta is not None:
            raise ValueError("fixed_theta must be None when estimate_theta=True.")
    else:
        if fixed_theta is None:
            raise ValueError("fixed_theta must be provided when estimate_theta=False.")
        if not np.isfinite(fixed_theta) or fixed_theta <= 0.0:
            raise ValueError(f"fixed_theta must be positive and finite, got {fixed_theta!r}.")

    with prepare_scenario_folder() as scenario_dir:
        scenario = Scenario.from_folder(Path(scenario_dir))
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
            od_layout=od_layout,
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

        ml_problem = build_od_theta_ml_problem(vi_request)

        ml_cfg = MLConfig(
            method="BFGS",
            maxiter=1000,
            gtol=1.0e-6,
            prior_weight=0.0,
            compute_hessian=True,
            log_every=10,
        )

        print("Running maximum likelihood...")
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

        ml_f_hat, ml_theta_hat = ml_problem.decode(ml_result.theta_hat)
        ml_z_hat = np.asarray(ml_result.theta_hat[: ml_problem.num_free_od], dtype=float)
        ml_u_hat = float(np.log(ml_theta_hat))

        vi_total_flow_mean = float(vi_total_flow_samples.mean())
        vi_total_flow_sd = float(vi_total_flow_samples.std())
        ml_total_flow = float(ml_f_hat.sum())

        theta_abs_diff = abs(ml_theta_hat - vi_theta_mean)
        theta_rel_diff = theta_abs_diff / abs(vi_theta_mean) if vi_theta_mean != 0 else np.nan

        flow_diff = ml_f_hat - vi_f_mean
        max_abs_flow_diff = float(np.max(np.abs(flow_diff)))
        mean_abs_flow_diff = float(np.mean(np.abs(flow_diff)))

        total_flow_abs_diff = abs(ml_total_flow - vi_total_flow_mean)
        total_flow_rel_diff = (
            total_flow_abs_diff / abs(vi_total_flow_mean)
            if vi_total_flow_mean != 0.0
            else np.nan
        )

        time_ratio = vi_time_s / ml_time_s if ml_time_s > 0 else np.nan

        print()
        print("Comparison of Bayesian VI and ML")
        print("--------------------------------")
        for line in vi_result.runtime_profile.format_lines():
            print(line)
        print(f"estimate theta: {estimate_theta}")
        if not estimate_theta:
            print(f"fixed theta: {fixed_theta:.6g}")
        print(f"VI time: {vi_time_s:.3f} s")
        print(f"ML time: {ml_time_s:.3f} s")
        print(f"VI/ML time ratio: {time_ratio:.3f}")
        print()
        print(f"theta VI mean: {vi_theta_mean:.6g}")
        print(f"theta ML: {ml_theta_hat:.6g}")
        print(f"theta absolute difference: {theta_abs_diff:.6g}")
        print(f"theta relative difference: {theta_rel_diff:.6g}")
        print()
        print(f"total OD flow VI mean: {vi_total_flow_mean:.6g}")
        print(f"total OD flow VI sd: {vi_total_flow_sd:.6g}")
        print(f"total OD flow ML: {ml_total_flow:.6g}")
        print(f"total OD flow absolute difference: {total_flow_abs_diff:.6g}")
        print(f"total OD flow relative difference: {total_flow_rel_diff:.6g}")
        print()
        print(f"max |f_ML - f_VI_mean|: {max_abs_flow_diff:.6g}")
        print(f"mean |f_ML - f_VI_mean|: {mean_abs_flow_diff:.6g}")
        print()
        print("ML diagnostics")
        print(f"success: {ml_result.success}")
        print(f"message: {ml_result.message}")
        print(f"objective: {ml_result.objective_value:.6g}")
        print(f"log-likelihood: {ml_result.loglikelihood:.6g}")
        print(f"gradient norm: {ml_result.gradient_norm:.6g}")

        out_path = RESULTS / "compare_vi_ml_od_theta_results.npz"
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
            compact_layout_fingerprint=vi_result.compact_layout_fingerprint,
            compact_layout_payload_json=vi_result.compact_layout_payload_json,
            **{
                f"runtime_{key}": value
                for key, value in vi_result.runtime_profile.as_dict().items()
                if not key.endswith("fingerprint")
            },
            estimate_theta=estimate_theta,
            fixed_theta=(np.nan if fixed_theta is None else float(fixed_theta)),
            vi_time_s=vi_time_s,
            vi_theta_samples=vi_result.theta_samples,
            vi_f_samples=vi_result.f_samples,
            vi_theta_mean=vi_theta_mean,
            vi_theta_sd=vi_result.theta_sd,
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
            theta_abs_diff=theta_abs_diff,
            theta_rel_diff=theta_rel_diff,
            max_abs_flow_diff=max_abs_flow_diff,
            mean_abs_flow_diff=mean_abs_flow_diff,
            total_flow_abs_diff=total_flow_abs_diff,
            total_flow_rel_diff=total_flow_rel_diff,
            time_ratio_vi_over_ml=time_ratio,
        )

        print()
        print(f"Saved comparison results to: {out_path}")
        print()
        print("Next step:")
        print("  cd ../post_processing")
        print("  python run_postprocessing.py")


if __name__ == "__main__":
    main()
