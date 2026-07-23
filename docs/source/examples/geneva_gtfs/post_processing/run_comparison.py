"""Assign an estimated OD matrix and compare its link flows with the truth."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import assign, prepare_assignment
from public_transportation.domain import Scenario


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PREPROCESSING = ROOT / "pre_processing" / "results"
ESTIMATION = ROOT / "estimation" / "results"
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
    args = parser.parse_args()
    estimate_path = ESTIMATION / f"{args.method}_results.npz"
    if not estimate_path.exists():
        raise FileNotFoundError(f"Run estimation/run_estimation.py --method {args.method} first")

    estimated = np.load(estimate_path)
    truth = np.load(PREPROCESSING / "true_assignment.npz")
    f_hat = np.asarray(estimated["f_hat"], dtype=float)
    f_true = np.asarray(truth["od_values"], dtype=float)
    true_link_flow = np.asarray(truth["link_flow"], dtype=float)

    with _scenario_folder() as folder:
        scenario = Scenario.from_folder(folder, strict=True)
        artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
        assigned = assign(
            od_values=jnp.asarray(f_hat, dtype=jnp.float32),
            artifacts=artifacts,
            theta=float(estimated["theta_hat"]),
        )
        estimated_link_flow = np.asarray(assigned.link_flow, dtype=float)

    od_error = f_hat - f_true
    link_error = estimated_link_flow - true_link_flow
    active = np.flatnonzero((f_true > 0.0) | (f_hat > 0.0))
    summary = {
        "method": args.method,
        "num_compared_od_cells": int(f_true.size),
        "num_active_od_cells": int(active.size),
        "true_total_demand": float(f_true.sum()),
        "estimated_total_demand": float(f_hat.sum()),
        "od_mae_active": float(np.mean(np.abs(od_error[active]))),
        "od_rmse_active": float(np.sqrt(np.mean(np.square(od_error[active])))),
        "link_flow_mae": float(np.mean(np.abs(link_error))),
        "link_flow_rmse": float(np.sqrt(np.mean(np.square(link_error)))),
        "link_flow_max_abs_error": float(np.max(np.abs(link_error))),
        "link_flow_correlation": float(np.corrcoef(true_link_flow, estimated_link_flow)[0, 1]),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.method}_comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        RESULTS / f"{args.method}_estimated_assignment.npz",
        f_hat=f_hat,
        estimated_link_flow=estimated_link_flow,
        true_link_flow=true_link_flow,
    )

    keys = [
        (record.origin_stop_id, record.dest_stop_id, record.time_bin_id)
        for record in scenario.demand.records
    ]
    with (RESULTS / f"{args.method}_od_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("origin_stop_id", "dest_stop_id", "time_bin_id", "true_flow", "estimated_flow", "error"))
        for index in active:
            writer.writerow((*keys[index], f_true[index], f_hat[index], od_error[index]))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
