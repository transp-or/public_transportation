"""Create a common ML/MAP/VI benchmark report for the Geneva example."""

from __future__ import annotations

import csv
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

    fields = (
        "method",
        "runtime_seconds",
        "runtime_relative_to_fastest",
        "success",
        "termination",
        "od_mae_active",
        "od_rmse_active",
        "link_flow_mae",
        "link_flow_rmse",
        "link_flow_correlation",
        "estimated_total_demand",
        "coverage_90_active",
        "mean_interval_width_90_active",
    )
    with (RESULTS / "method_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

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

    lines = [
        "# Geneva estimation-method comparison",
        "",
        "All methods use the same timetable, observations, fixed theta, OD support, and frozen-cell layout.",
        "",
        "| Method | Runtime (s) | Relative runtime | OD RMSE | Link RMSE | Link correlation | 90% coverage | Mean 90% width |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in rows:
        coverage = row["coverage_90_active"]
        width = row["mean_interval_width_90_active"]
        lines.append(
            f"| {row['method']} | {float(row['runtime_seconds']):.1f} | "
            f"{float(row['runtime_relative_to_fastest']):.1f}x | "
            f"{float(row['od_rmse_active']):.3f} | {float(row['link_flow_rmse']):.3f} | "
            f"{float(row['link_flow_correlation']):.6f} | "
            f"{('—' if coverage == '' else f'{float(coverage):.1%}')} | "
            f"{('—' if width == '' else f'{float(width):.3f}')} |"
        )
    lines.extend(
        (
            "",
            "ML reached its configured iteration cap; MAP satisfied the optimizer termination criterion; VI completed its fixed 1,000-step schedule.",
            "The prior is intentionally inaccurate, so this benchmark is a stress test of regularization rather than a favorable MAP/VI setup.",
            "",
        )
    )
    (RESULTS / "method_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
