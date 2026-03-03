"""
Example 02 — Postprocessing VI results: assign estimated OD with estimated theta,
and generate a comparison report.

Inputs (expected in ./data):
- metadata.json
- stops.csv
- lines.csv
- trips.csv
- stop_times.csv
- time_bins.csv
- demand.csv
- measurements_boarding_alighting.csv
- vi_od_theta_results.npz

Outputs (written to ./data by default):
- time_expanded_report_estimated.html
- inference_comparison_report.html
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np

from public_transportation.domain import Scenario
from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment, assign
from public_transportation.assignment.id_manager import AssignmentIDManager

from public_transportation.measurement import read_measurements_csv, build_mapping_spec_strict


from public_transportation.inference.results_io import load_od_theta_vi_results  # your chosen API name
# If your file uses load_od_theta_vi_results_npz instead, update this import accordingly.
from public_transportation.inference.fingerprint_debug import assert_results_compatible_with_id_manager

from public_transportation.viz.time_expanded_report import write_time_expanded_report_from_assignment

from public_transportation.viz.inference_comparison_report import (
    compute_od_and_flow_comparison,
    write_od_theta_comparison_report_html,
)


DATA = Path(__file__).resolve().parent / "data"


def _theta_mode_from_samples(theta_samples: np.ndarray, *, bins: int = 60) -> float:
    """Simple histogram-based mode (no external deps)."""
    x = np.asarray(theta_samples, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("theta_samples contains no finite values.")
    if x.size == 1:
        return float(x[0])

    hist, edges = np.histogram(x, bins=int(bins))
    k = int(np.argmax(hist))
    # bin center
    return float(0.5 * (edges[k] + edges[k + 1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=str,
        default=str(DATA / "vi_od_theta_results.npz"),
        help="Path to VI results .npz (must match current scenario/graph fingerprint)",
    )
    parser.add_argument(
        "--theta",
        type=str,
        required=True,
        choices=["mean", "median", "mode"],
        help="Theta point estimate to use for assignment (no default).",
    )
    parser.add_argument(
        "--f",
        type=str,
        default="mean",
        choices=["mean", "median"],
        help="OD point estimate to use (default: mean).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DATA),
        help="Output directory for HTML reports.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=1.0,
        help="Detection rate rho used for predicted measurements (mu = rho * lambda).",
    )
    parser.add_argument(
        "--svg-scale-x",
        type=float,
        default=1.8,
        help="SVG horizontal scale for the time-expanded report.",
    )
    parser.add_argument(
        "--svg-scale-y",
        type=float,
        default=2.4,
        help="SVG vertical scale for the time-expanded report.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1) Load scenario + prepare assignment artifacts (same as inference)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 2) Measurements -> mapping spec (same “truth” as inference)
    # -------------------------------------------------------------------------
    meas_path = DATA / "measurements_boarding_alighting.csv"
    if not meas_path.exists():
        raise RuntimeError(f"Missing measurement file: {meas_path}")

    table = read_measurements_csv(meas_path)
    msr = build_mapping_spec_strict(
        id_manager=idm,
        table=table,
        include_link_lists_for_report=False,
    )
    y_obs = np.asarray(msr.y_obs, dtype=float).reshape(-1)
    mapping_spec = msr.spec

    # -------------------------------------------------------------------------
    # 3) Load VI results
    # -------------------------------------------------------------------------
    res = load_od_theta_vi_results(args.results)

    # Fail fast if someone mixes scenario/results (rich diagnostics)
    # This uses fingerprint payload diffs when available.
    assert_results_compatible_with_id_manager(
        results=res,
        id_manager=idm,
        context=(
            "run_postprocessing: results file does not match the current scenario/graph indexing. "
            "Common causes: changed GTFS inputs, re-generated time-expanded graph with different settings, "
            "or using results from a different data folder."
        ),
    )

    # -------------------------------------------------------------------------
    # 4) Choose point estimates (theta: required choice; f: mean/median)
    # -------------------------------------------------------------------------
    if args.theta == "mean":
        theta_hat = float(res.theta_mean)
    elif args.theta == "median":
        theta_hat = float(np.median(np.asarray(res.theta_samples, dtype=float)))
    else:
        theta_hat = _theta_mode_from_samples(np.asarray(res.theta_samples, dtype=float))

    if not np.isfinite(theta_hat) or theta_hat <= 0.0:
        raise RuntimeError(f"Invalid theta_hat={theta_hat!r}")

    if args.f == "mean":
        f_hat = np.asarray(res.f_mean, dtype=float).reshape(-1)
    else:
        f_hat = np.asarray(np.median(np.asarray(res.f_samples, dtype=float), axis=0), dtype=float).reshape(-1)

    # Prior OD for comparison (baseline used in inference)
    f0 = np.asarray(res.f0, dtype=float).reshape(-1)

    if f_hat.shape != f0.shape:
        raise RuntimeError(f"OD length mismatch: f_hat {f_hat.shape} vs f0 {f0.shape}")

    # -------------------------------------------------------------------------
    # 5) Assign estimated OD with estimated theta
    # -------------------------------------------------------------------------
    assignment_est = assign(
        od_values=f_hat,
        artifacts=artifacts,
        theta=theta_hat,
        return_group_link_flows=False,
    )
    link_flow_est = np.asarray(assignment_est.link_flow, dtype=float).reshape(-1)

    # -------------------------------------------------------------------------
    # 6) Write reports
    # -------------------------------------------------------------------------
    # (a) time-expanded reports (prior vs estimated) under the SAME theta_hat
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
        title=f"Time-expanded graph report — estimated OD (theta={theta_hat:.6g})",
        svg_scale_x=float(args.svg_scale_x),
        svg_scale_y=float(args.svg_scale_y),
    )
    print(f"Wrote: {te_post_path}")

    # (b) comparison report (OD + link flows + measurements)
    bundle = compute_od_and_flow_comparison(
        scenario=scenario,
        assignment_artifacts=artifacts,
        id_manager=idm,
        mapping_spec=mapping_spec,
        y_obs=y_obs,
        fingerprint_expected=str(idm.fingerprint),
        fingerprint_results=str(res.fingerprint),
        theta_hat=float(theta_hat),
        f0=f0,
        f_hat=f_hat,
        rho=float(args.rho),
    )

    cmp_path = out_dir / "inference_comparison_report.html"
    write_od_theta_comparison_report_html(
        bundle=bundle,
        output_path=cmp_path,
        title="Inference postprocessing — OD and flow comparison",
        extra_links={
            "Time-expanded report (prior OD)": te_prior_path.name,
            "Time-expanded report (estimated OD)": te_post_path.name,
        },
    )
    print(f"Wrote: {cmp_path}")


if __name__ == "__main__":
    main()