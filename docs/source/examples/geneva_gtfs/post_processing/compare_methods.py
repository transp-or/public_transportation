"""Create a common ML/MAP/VI benchmark report for the Geneva example."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ESTIMATION = ROOT / "estimation" / "results"
PREPROCESSING = ROOT / "pre_processing" / "results"
RESULTS = Path(__file__).resolve().parent / "results"
METHODS = ("ml", "map", "vi")


def main() -> None:
    truth = np.load(PREPROCESSING / "true_assignment.npz")
    f_true = np.asarray(truth["od_values"], dtype=float)
    active = f_true > 0.0
    rows: list[dict[str, object]] = []

    for method in METHODS:
        estimate = np.load(ESTIMATION / f"{method}_results.npz")
        estimation_summary = json.loads(
            (ESTIMATION / f"{method}_summary.json").read_text(encoding="utf-8")
        )
        comparison = json.loads(
            (RESULTS / f"{method}_comparison.json").read_text(encoding="utf-8")
        )
        row: dict[str, object] = {
            "method": method.upper(),
            "runtime_seconds": float(estimation_summary["runtime_seconds"]),
            "success": bool(estimation_summary["success"]),
            "termination": estimation_summary.get("optimizer_message", "completed VI schedule"),
            "od_mae_active": float(comparison["od_mae_active"]),
            "od_rmse_active": float(comparison["od_rmse_active"]),
            "link_flow_mae": float(comparison["link_flow_mae"]),
            "link_flow_rmse": float(comparison["link_flow_rmse"]),
            "link_flow_correlation": float(comparison["link_flow_correlation"]),
            "estimated_total_demand": float(comparison["estimated_total_demand"]),
            "coverage_90_active": "",
            "mean_interval_width_90_active": "",
        }
        if method == "vi":
            lower = np.asarray(estimate["f_q05"], dtype=float)
            upper = np.asarray(estimate["f_q95"], dtype=float)
            covered = (lower <= f_true) & (f_true <= upper)
            row["coverage_90_active"] = float(np.mean(covered[active]))
            row["mean_interval_width_90_active"] = float(np.mean((upper - lower)[active]))
            losses = np.asarray(estimate["losses"], dtype=float)
            row["vi_initial_loss"] = float(losses[0])
            row["vi_final_loss"] = float(losses[-1])
        rows.append(row)

    fastest = min(float(row["runtime_seconds"]) for row in rows)
    for row in rows:
        row["runtime_relative_to_fastest"] = float(row["runtime_seconds"]) / fastest

    payload = {
        "experiment": {
            "fixed_theta": 5.0,
            "num_free_od": 96,
            "num_frozen_zero_od": 15032,
            "num_measurements": 8967,
            "true_total_demand": float(f_true.sum()),
            "vi_interval": "equal-tail empirical 5%-95% interval from 500 variational draws",
        },
        "methods": rows,
    }
    (RESULTS / "method_comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
