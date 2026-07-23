"""
Bayesian VI estimation with theta fixed to 1.0.

Reads ../data and ../pre_processing/results. Writes results/vi_fixed_theta_results.npz.
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
from public_transportation.inference.od_parameter_layout import build_od_parameter_layout
from public_transportation.estimation.bayesian.config import VIConfig
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



def run_bayesian(*, idm, msr, f0, od_layout, artifacts, estimate_theta: bool, fixed_theta: float | None, logger):
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

    t0 = perf_counter()
    vi_result = estimate_od_theta_vi(vi_request)
    vi_time_s = perf_counter() - t0
    return vi_result, vi_time_s


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    logger = make_console_logger("public_transportation.bayesian")
    estimate_theta = False
    fixed_theta = baseline_theta

    with prepare_scenario_folder() as scenario_dir:
        global SCENARIO_DIR
        SCENARIO_DIR = scenario_dir
        _, artifacts, idm, msr, f0, od_layout = prepare_shared_inputs()

        print("Running Bayesian VI...")
        print(f"estimate theta: {estimate_theta}")
        if fixed_theta is not None:
            print(f"fixed theta: {fixed_theta:.6g}")

        vi_result, vi_time_s = run_bayesian(
            idm=idm,
            msr=msr,
            f0=f0,
            od_layout=od_layout,
            artifacts=artifacts,
            estimate_theta=estimate_theta,
            fixed_theta=fixed_theta,
            logger=logger,
        )

        out_path = RESULTS / "vi_fixed_theta_results.npz"
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
            theta_samples=vi_result.theta_samples,
            f_samples=vi_result.f_samples,
            theta_mean=vi_result.theta_mean,
            theta_sd=vi_result.theta_sd,
            f_mean=vi_result.f_mean,
            vi_losses=vi_result.vi.losses,
        )

        print()
        print("Bayesian VI summary")
        for line in vi_result.runtime_profile.format_lines():
            print(line)
        print(f"runtime: {vi_time_s:.3f} s")
        print(f"theta mean: {float(vi_result.theta_mean):.6g}")
        print(f"total OD flow mean: {float(vi_result.f_samples.sum(axis=1).mean()):.6g}")
        print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
