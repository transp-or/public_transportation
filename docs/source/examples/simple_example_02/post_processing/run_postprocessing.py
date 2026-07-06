"""
Post-process estimation results for simple_example_02.

Reads:
    ../data/
    ../pre_processing/results/demand.csv
    ../pre_processing/results/measurements_boarding_alighting.csv
    ../estimation/results/compare_vi_ml_od_theta_results.npz

Writes:
    results/bayesian/
    results/ml/
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import assign, prepare_assignment
from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.domain import Scenario
from public_transportation.inference.fingerprint_debug import (
    assert_results_compatible_with_id_manager,
)
from public_transportation.inference.results_io import load_od_theta_vi_results
from public_transportation.measurement import (
    build_mapping_spec_strict,
    read_measurements_csv,
)
from public_transportation.viz.inference_comparison_report import (
    compute_od_and_flow_comparison,
    write_od_theta_comparison_report_html,
)
from public_transportation.viz.time_expanded_report import (
    write_time_expanded_report_from_assignment,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PREPROCESSING_RESULTS = ROOT / "pre_processing" / "results"
ESTIMATION_RESULTS = ROOT / "estimation" / "results"
RESULTS = Path(__file__).resolve().parent / "results"

NETWORK_FILES = [
    "metadata.json",
    "stops.csv",
    "lines.csv",
    "trips.csv",
    "stop_times.csv",
    "time_bins.csv",
]


def prepare_scenario_folder() -> tempfile.TemporaryDirectory[str]:
    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)

    for filename in NETWORK_FILES:
        src = DATA / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing required data file: {src}")
        shutil.copy2(src, tmp_path / filename)

    demand_path = PREPROCESSING_RESULTS / "demand.csv"
    if not demand_path.exists():
        raise FileNotFoundError(
            f"Missing generated demand file: {demand_path}. "
            "Run pre_processing/run_preprocessing.py first."
        )
    shutil.copy2(demand_path, tmp_path / "demand.csv")

    return tmp


def theta_mode_from_samples(theta_samples: np.ndarray, *, bins: int = 60) -> float:
    x = np.asarray(theta_samples, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("theta_samples contains no finite values.")
    if x.size == 1:
        return float(x[0])

    hist, edges = np.histogram(x, bins=int(bins))
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


def load_point_estimates_from_results(
    *,
    results_path: Path,
    method: str,
    theta_choice: str,
    f_choice: str,
) -> tuple[np.ndarray, float, np.ndarray, str | None, str | None]:
    with np.load(results_path, allow_pickle=False) as result:
        estimate_theta = (
            bool(np.asarray(result["estimate_theta"]).reshape(-1)[0])
            if "estimate_theta" in result.files
            else None
        )
        fixed_theta = (
            float(np.asarray(result["fixed_theta"]).reshape(-1)[0])
            if "fixed_theta" in result.files
            else None
        )
        if fixed_theta is not None and np.isnan(fixed_theta):
            fixed_theta = None

        f0 = np.asarray(result["f0"], dtype=float).reshape(-1)

        if method == "bayesian":
            if "vi_f_mean" in result.files:
                f_mean_key = "vi_f_mean"
                f_samples_key = "vi_f_samples"
                theta_mean_key = "vi_theta_mean"
                theta_samples_key = "vi_theta_samples"
            else:
                f_mean_key = "f_mean"
                f_samples_key = "f_samples"
                theta_mean_key = "theta_mean"
                theta_samples_key = "theta_samples"

            if f_choice == "mean":
                f_hat = np.asarray(result[f_mean_key], dtype=float).reshape(-1)
            else:
                f_samples = np.asarray(result[f_samples_key], dtype=float)
                f_hat = np.asarray(np.median(f_samples, axis=0), dtype=float).reshape(-1)

            if estimate_theta is False and fixed_theta is not None:
                theta_hat = float(fixed_theta)
            elif theta_choice == "mean":
                theta_hat = float(result[theta_mean_key])
            elif theta_choice == "median":
                theta_hat = float(np.median(np.asarray(result[theta_samples_key], dtype=float)))
            else:
                theta_hat = theta_mode_from_samples(np.asarray(result[theta_samples_key], dtype=float))

        elif method == "ml":
            f_hat = np.asarray(result["ml_f_hat" if "ml_f_hat" in result.files else "f_hat"], dtype=float).reshape(-1)

            if estimate_theta is False and fixed_theta is not None:
                theta_hat = float(fixed_theta)
            else:
                theta_key = "ml_theta_hat" if "ml_theta_hat" in result.files else "theta_hat"
                theta_hat = float(result[theta_key])

        else:
            raise ValueError(f"Unknown method: {method!r}")

    return f_hat, theta_hat, f0, str(estimate_theta), None if fixed_theta is None else str(fixed_theta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=str,
        default=str(ESTIMATION_RESULTS / "compare_vi_ml_od_theta_results.npz"),
        help="Path to estimation result .npz.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="both",
        choices=["bayesian", "ml", "both"],
        help=(
            "Which result to postprocess when the .npz contains both. "
            "Use 'both' to generate outputs for Bayesian and ML results."
        ),
    )
    parser.add_argument(
        "--theta",
        type=str,
        default="mean",
        choices=["mean", "median", "mode"],
        help="Theta point estimate for Bayesian results. Ignored if theta was fixed or method is ML.",
    )
    parser.add_argument(
        "--f",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="OD point estimate for Bayesian results. Ignored for ML.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=1.0,
        help="Detection rate rho used for predicted measurements.",
    )
    parser.add_argument("--svg-scale-x", type=float, default=1.8)
    parser.add_argument("--svg-scale-y", type=float, default=2.4)
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(
            f"Missing estimation result file: {results_path}. "
            "Run estimation/run_both.py first."
        )

    measurements_path = PREPROCESSING_RESULTS / "measurements_boarding_alighting.csv"
    if not measurements_path.exists():
        raise FileNotFoundError(
            f"Missing measurement file: {measurements_path}. "
            "Run pre_processing/run_preprocessing.py first."
        )

    with prepare_scenario_folder() as scenario_dir:
        scenario = Scenario.from_folder(Path(scenario_dir))

        rep = scenario.validate()
        if rep.issues:
            print("Scenario validation issues:")
            for issue in rep.issues:
                print(
                    f"- [{issue.severity.name}] {issue.code}: "
                    f"{issue.message} ({issue.location})"
                )

        config = AssignmentConfig()
        artifacts = prepare_assignment(scenario=scenario, config=config)
        idm = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)

        table = read_measurements_csv(measurements_path)
        msr = build_mapping_spec_strict(
            id_manager=idm,
            table=table,
            include_link_lists_for_report=False,
        )
        y_obs = np.asarray(msr.y_obs, dtype=float).reshape(-1)
        mapping_spec = msr.spec

        methods_to_process = ["bayesian", "ml"] if args.method == "both" else [args.method]

        for method in methods_to_process:
            out_dir = RESULTS / method
            out_dir.mkdir(parents=True, exist_ok=True)

            f_hat, theta_hat, f0, estimate_theta, fixed_theta = load_point_estimates_from_results(
                results_path=results_path,
                method=method,
                theta_choice=args.theta,
                f_choice=args.f,
            )

            if not np.isfinite(theta_hat) or theta_hat <= 0.0:
                raise RuntimeError(f"Invalid theta_hat={theta_hat!r} for method {method!r}")

            if f_hat.shape != f0.shape:
                raise RuntimeError(
                    f"OD length mismatch for method {method!r}: f_hat {f_hat.shape} vs f0 {f0.shape}"
                )

            assignment_est = assign(
                od_values=f_hat,
                artifacts=artifacts,
                theta=theta_hat,
                return_group_link_flows=False,
            )

            assignment_prior = assign(
                od_values=f0,
                artifacts=artifacts,
                theta=theta_hat,
                return_group_link_flows=False,
            )

            te_prior_path = out_dir / "time_expanded_report_prior.html"
            write_time_expanded_report_from_assignment(
                scenario=scenario,
                assignment=assignment_prior,
                config=config,
                output_path=te_prior_path,
                title=f"Time-expanded graph report — prior OD (theta={theta_hat:.6g})",
                svg_scale_x=float(args.svg_scale_x),
                svg_scale_y=float(args.svg_scale_y),
            )
            print(f"Wrote: {te_prior_path}")

            te_post_path = out_dir / "time_expanded_report_estimated.html"
            write_time_expanded_report_from_assignment(
                scenario=scenario,
                assignment=assignment_est,
                config=config,
                output_path=te_post_path,
                title=f"Time-expanded graph report — estimated OD ({method}, theta={theta_hat:.6g})",
                svg_scale_x=float(args.svg_scale_x),
                svg_scale_y=float(args.svg_scale_y),
            )
            print(f"Wrote: {te_post_path}")

            bundle = compute_od_and_flow_comparison(
                scenario=scenario,
                assignment_artifacts=artifacts,
                id_manager=idm,
                mapping_spec=mapping_spec,
                y_obs=y_obs,
                fingerprint_expected=str(idm.fingerprint),
                fingerprint_results=str(idm.fingerprint),
                theta_hat=float(theta_hat),
                f0=f0,
                f_hat=f_hat,
                rho=float(args.rho),
            )

            cmp_path = out_dir / "inference_comparison_report.html"
            write_od_theta_comparison_report_html(
                bundle=bundle,
                output_path=cmp_path,
                title=f"Inference postprocessing — {method}",
                extra_links={
                    "Time-expanded report (prior OD)": te_prior_path.name,
                    "Time-expanded report (estimated OD)": te_post_path.name,
                },
            )
            print(f"Wrote: {cmp_path}")

            np.savez_compressed(
                out_dir / "postprocessed_assignment.npz",
                method=method,
                estimate_theta=estimate_theta,
                fixed_theta=("" if fixed_theta is None else fixed_theta),
                theta_hat=float(theta_hat),
                f0=f0,
                f_hat=f_hat,
                link_flow_est=np.asarray(assignment_est.link_flow, dtype=float),
                link_flow_prior=np.asarray(assignment_prior.link_flow, dtype=float),
            )

    print()
    print("Post-processing complete.")
    print(f"Outputs written to: {RESULTS}")


if __name__ == "__main__":
    main()