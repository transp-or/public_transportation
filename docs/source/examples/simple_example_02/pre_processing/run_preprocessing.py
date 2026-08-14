"""
Pre-processing for simple_example_02.

This script has two modes.

1. Investigation mode

   Used to inspect route-choice sensitivity to theta before generating synthetic
   estimation data. It assigns the true demand for several theta values and
   writes one time-expanded report per theta value.

   Interpretation of theta in the assignment model:
       small theta -> strong sensitivity to cost -> concentrated on low-cost paths;
       large theta -> weak sensitivity to cost -> more dispersed route choice.

   Example:
       python run_preprocessing.py --mode investigate
       python run_preprocessing.py --mode investigate --theta-grid 0.25 0.5 0.75 1 1.5 2 3

2. Data-generation mode

   Used to generate the synthetic estimation files with a selected theta value.

   Example:
       python run_preprocessing.py --mode generate --theta 1.0

Inputs:
    ../data/metadata.json
    ../data/stops.csv
    ../data/lines.csv
    ../data/trips.csv
    ../data/stop_times.csv
    ../data/time_bins.csv
    ../data/true_demand.csv
    ../data/prior_demand.csv

Investigation outputs:
    results/theta_investigation/theta_sensitivity_summary.csv
        Includes aggregate flow, link-use, route-choice dispersion diagnostics,
        and a generalized-cost ratio relative to the lowest-theta assignment.
    results/theta_investigation/time_expanded_report_true_theta_*.html
    results/theta_investigation/assignment_theta_*.npz

Data-generation outputs:
    results/demand.csv
    results/measurements_boarding_alighting.csv
    results/time_expanded_report_true.html
    results/synthetic_measurement_link_mapping.json
    results/true_assignment.npz
    results/preprocessing_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from public_transportation.assignment.assign import assign, prepare_assignment
from public_transportation.assignment.config import AssignmentConfig
from public_transportation.assignment.graph_sentinels import (
    LINK_TYPE_ACCESS,
    LINK_TYPE_EGRESS,
)
from public_transportation.domain import TimeOfDay
from public_transportation.domain.scenario import Scenario
from public_transportation.measurement.io import write_measurements_csv
from public_transportation.measurement.schema import (
    MeasurementRecord,
    MeasurementTable,
    MeasurementType,
)
from public_transportation.viz.time_expanded_report import (
    write_time_expanded_report_from_assignment,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = Path(__file__).resolve().parent / "results"

NETWORK_FILES = [
    "metadata.json",
    "stops.csv",
    "lines.csv",
    "trips.csv",
    "stop_times.csv",
    "time_bins.csv",
]

DEFAULT_THETA_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
MEASUREMENT_FLOW_TOL = 1.0e-4


def _copy_required_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required input file: {src}")
    shutil.copy2(src, dst)


def _prepare_scenario_folder(*, demand_file: Path) -> tempfile.TemporaryDirectory[str]:
    """Create a temporary Scenario folder with the selected demand file as demand.csv."""
    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)

    for filename in NETWORK_FILES:
        _copy_required_file(DATA / filename, tmp_path / filename)

    _copy_required_file(demand_file, tmp_path / "demand.csv")
    return tmp


def _hms_from_seconds(seconds: int) -> str:
    seconds = max(int(seconds), 0)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _load_true_demand_scenario() -> tuple[Scenario, jnp.ndarray, tempfile.TemporaryDirectory[str]]:
    """Load the scenario with true_demand.csv as demand.csv.

    The returned temporary directory must remain alive while the scenario is used.
    """
    true_demand_path = DATA / "true_demand.csv"
    if not true_demand_path.exists():
        raise FileNotFoundError(f"Missing true demand file: {true_demand_path}")

    tmp = _prepare_scenario_folder(demand_file=true_demand_path)
    scenario = Scenario.from_folder(Path(tmp.name))

    report = scenario.validate()
    if report.issues:
        print("Scenario validation issues:")
        for issue in report.issues:
            print(
                f"- [{issue.severity.name}] {issue.code}: "
                f"{issue.message} ({issue.location})"
            )

    if scenario.demand is None or not scenario.demand.records:
        raise RuntimeError("No true demand records found.")

    od_values = jnp.asarray(
        [float(record.flow) for record in scenario.demand.records],
        dtype=jnp.float32,
    )
    return scenario, od_values, tmp



def _assign_true_demand(
    *,
    scenario: Scenario,
    od_values: jnp.ndarray,
    theta: float,
) -> tuple[AssignmentConfig, object, object]:
    config = AssignmentConfig()
    artifacts = prepare_assignment(scenario=scenario, config=config)
    assignment = assign(
        od_values=od_values,
        artifacts=artifacts,
        theta=float(theta),
        return_group_link_flows=False,
    )
    return config, artifacts, assignment


def _link_flow_diagnostics(
    *,
    link_flow: np.ndarray,
    link_cost: np.ndarray,
    total_od_flow: float,
) -> dict[str, float | int]:
    """Compute aggregate route-choice dispersion diagnostics from link flows.

    These are not path-level probabilities, but they are useful indicators for
    choosing theta during investigation mode.

    Interpretation:
        - average_links_per_passenger increases when passengers use longer or
          more circuitous alternatives.
        - effective_positive_links is an entropy-based measure of how many links
          carry meaningful flow.
        - top_5_link_flow_share and top_10_link_flow_share measure concentration
          of flow on the busiest links.
        - average_generalized_cost_per_passenger is the flow-weighted average
          generalized cost per passenger. In the investigation summary, it is
          compared with the value obtained for the smallest theta in the grid,
          used as a proxy for the deterministic low-cost assignment.
    """
    x = np.asarray(link_flow, dtype=float).reshape(-1)
    c = np.asarray(link_cost, dtype=float).reshape(-1)
    if c.shape != x.shape:
        raise ValueError(f"link_cost shape {c.shape} does not match link_flow shape {x.shape}.")
    positive = x[x > MEASUREMENT_FLOW_TOL]

    total_link_flow = float(x.sum())
    if total_od_flow <= 0.0:
        average_links_per_passenger = np.nan
    else:
        average_links_per_passenger = total_link_flow / float(total_od_flow)

    if total_od_flow <= 0.0:
        average_generalized_cost_per_passenger = np.nan
    else:
        average_generalized_cost_per_passenger = float(np.sum(x * c) / float(total_od_flow))

    if positive.size == 0 or positive.sum() <= 0.0:
        return {
            "total_link_flow": total_link_flow,
            "average_links_per_passenger": float(average_links_per_passenger),
            "average_generalized_cost_per_passenger": float(average_generalized_cost_per_passenger),
            "max_link_flow": 0.0,
            "mean_positive_link_flow": 0.0,
            "num_positive_links": 0,
            "effective_positive_links": 0.0,
            "top_5_link_flow_share": 0.0,
            "top_10_link_flow_share": 0.0,
        }

    shares = positive / positive.sum()
    entropy = -float(np.sum(shares * np.log(shares)))
    effective_positive_links = float(np.exp(entropy))

    descending = np.sort(positive)[::-1]
    top_5_share = float(descending[:5].sum() / positive.sum())
    top_10_share = float(descending[:10].sum() / positive.sum())

    return {
        "total_link_flow": total_link_flow,
        "average_links_per_passenger": float(average_links_per_passenger),
        "average_generalized_cost_per_passenger": float(average_generalized_cost_per_passenger),
        "max_link_flow": float(positive.max()),
        "mean_positive_link_flow": float(positive.mean()),
        "num_positive_links": int(positive.size),
        "effective_positive_links": effective_positive_links,
        "top_5_link_flow_share": top_5_share,
        "top_10_link_flow_share": top_10_share,
    }


def _write_theta_investigation(
    *,
    theta_grid: list[float],
    svg_scale_x: float,
    svg_scale_y: float,
) -> None:
    scenario, od_values, tmp = _load_true_demand_scenario()
    with tmp:
        output_dir = RESULTS / "theta_investigation"
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = output_dir / "theta_sensitivity_summary.csv"
        rows: list[dict[str, float | int | str]] = []
        lowest_theta_average_cost: float | None = None

        print("Theta investigation mode")
        print(f"OD records: {len(scenario.demand.records) if scenario.demand else 0}")
        print(f"Total true demand: {float(np.asarray(od_values).sum()):.6g}")
        print()
        print("Theta interpretation:")
        print("  small theta -> strong sensitivity to cost -> concentrated on low-cost paths")
        print("  large theta -> weak sensitivity to cost -> more dispersed route choice")
        print()
        print("Diagnostics interpretation:")
        print("  average_links_per_passenger should not grow excessively with theta")
        print("  effective_positive_links indicates how diffuse the assignment is")
        print("  top_5_link_flow_share and top_10_link_flow_share indicate concentration")
        print("  cost_ratio_to_lowest_theta compares average generalized cost with the lowest-theta assignment")
        print()

        for theta in theta_grid:
            if theta <= 0.0 or not np.isfinite(theta):
                raise ValueError(f"Theta values must be positive and finite, got {theta!r}")

            config, _, assignment = _assign_true_demand(
                scenario=scenario,
                od_values=od_values,
                theta=float(theta),
            )
            link_flow = np.asarray(assignment.link_flow, dtype=float).reshape(-1)
            total_od_flow = float(np.asarray(od_values).sum())
            diagnostics = _link_flow_diagnostics(
                link_flow=link_flow,
                link_cost=np.asarray(assignment.link_cost, dtype=float).reshape(-1),
                total_od_flow=total_od_flow,
            )
            average_cost = float(diagnostics["average_generalized_cost_per_passenger"])
            if lowest_theta_average_cost is None:
                lowest_theta_average_cost = average_cost
            cost_ratio_to_lowest_theta = (
                np.nan
                if lowest_theta_average_cost <= 0.0
                else average_cost / lowest_theta_average_cost
            )

            report_path = output_dir / f"time_expanded_report_true_theta_{theta:g}.html"
            write_time_expanded_report_from_assignment(
                scenario=scenario,
                assignment=assignment,
                config=config,
                output_path=report_path,
                title=f"Example 02 — true-demand assignment, theta={theta:g}",
                svg_scale_x=float(svg_scale_x),
                svg_scale_y=float(svg_scale_y),
            )

            npz_path = output_dir / f"assignment_theta_{theta:g}.npz"
            np.savez_compressed(
                npz_path,
                theta=float(theta),
                od_values=np.asarray(od_values, dtype=float),
                link_flow=link_flow,
                link_cost=np.asarray(assignment.link_cost, dtype=float),
            )

            row = {
                "theta": float(theta),
                "total_od_flow": total_od_flow,
                "total_link_flow": diagnostics["total_link_flow"],
                "average_links_per_passenger": diagnostics["average_links_per_passenger"],
                "average_generalized_cost_per_passenger": diagnostics[
                    "average_generalized_cost_per_passenger"
                ],
                "cost_ratio_to_lowest_theta": float(cost_ratio_to_lowest_theta),
                "max_link_flow": diagnostics["max_link_flow"],
                "mean_positive_link_flow": diagnostics["mean_positive_link_flow"],
                "num_positive_links": diagnostics["num_positive_links"],
                "effective_positive_links": diagnostics["effective_positive_links"],
                "top_5_link_flow_share": diagnostics["top_5_link_flow_share"],
                "top_10_link_flow_share": diagnostics["top_10_link_flow_share"],
                "report": report_path.name,
                "assignment_npz": npz_path.name,
            }
            rows.append(row)

            print(f"theta={theta:g}")
            print(f"  total link flow:             {row['total_link_flow']:.6g}")
            print(f"  average links per passenger: {row['average_links_per_passenger']:.6g}")
            print(f"  average generalized cost:    {row['average_generalized_cost_per_passenger']:.6g}")
            print(f"  cost ratio to lowest theta:  {row['cost_ratio_to_lowest_theta']:.6g}")
            print(f"  max link flow:               {row['max_link_flow']:.6g}")
            print(f"  positive links:              {row['num_positive_links']}")
            print(f"  effective positive links:    {row['effective_positive_links']:.6g}")
            print(f"  top 5 link-flow share:       {row['top_5_link_flow_share']:.6g}")
            print(f"  top 10 link-flow share:      {row['top_10_link_flow_share']:.6g}")
            print(f"  report:                      {report_path}")
            print()

        with summary_path.open("w", newline="", encoding="utf-8") as file:
            fieldnames = [
                "theta",
                "total_od_flow",
                "total_link_flow",
                "average_links_per_passenger",
                "average_generalized_cost_per_passenger",
                "cost_ratio_to_lowest_theta",
                "max_link_flow",
                "mean_positive_link_flow",
                "num_positive_links",
                "effective_positive_links",
                "top_5_link_flow_share",
                "top_10_link_flow_share",
                "report",
                "assignment_npz",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print("Theta investigation complete.")
        print(f"Summary written to: {summary_path}")
        print()
        print("Inspect the summary CSV and HTML reports before selecting theta.")
        print("A useful theta should avoid both extremes: nearly deterministic assignment and overly diffuse assignment.")
        print("Prefer values where average_links_per_passenger, effective_positive_links, and cost_ratio_to_lowest_theta are moderate.")
        print("Then run, for example:")
        print("  python run_preprocessing.py --mode generate --theta 1.0")


def _build_measurements_and_mapping(
    *,
    scenario: Scenario,
    artifacts: object,
    assignment: object,
) -> tuple[MeasurementTable, Path]:
    graph = artifacts.graph
    link_flow = np.asarray(assignment.link_flow)
    link_type = np.asarray(graph.link_type)
    tail = np.asarray(graph.tail)
    head = np.asarray(graph.head)
    node_time_s = np.asarray(graph.node_time_s)
    node_stop_index = np.asarray(graph.node_stop_index)
    node_trip_index = np.asarray(graph.node_trip_index)

    stop_ids = list(graph.node_stop_id)
    trip_ids = list(graph.trip_id)

    def stop_id_from_node(node: int) -> str:
        stop_index = int(node_stop_index[node])
        return str(stop_ids[stop_index])

    def trip_id_from_node(node: int) -> str | None:
        trip_index = int(node_trip_index[node])
        if trip_index < 0:
            return None
        return str(trip_ids[trip_index])

    def node_hms(node: int) -> str:
        return _hms_from_seconds(int(node_time_s[node]))

    agg: dict[tuple[str, str, str | None, str], float] = {}
    contrib: dict[tuple[str, str, str | None, str], list[dict[str, float | int]]] = {}

    access_link_ids = np.where(link_type == LINK_TYPE_ACCESS)[0]
    for link_id in access_link_ids:
        value = float(link_flow[link_id])
        if abs(value) <= MEASUREMENT_FLOW_TOL:
            continue

        node = int(head[link_id])
        key = (
            "boarding",
            stop_id_from_node(node),
            trip_id_from_node(node),
            node_hms(node),
        )
        agg[key] = agg.get(key, 0.0) + value
        contrib.setdefault(key, []).append({"link_id": int(link_id), "value": value})

    egress_link_ids = np.where(link_type == LINK_TYPE_EGRESS)[0]
    for link_id in egress_link_ids:
        value = float(link_flow[link_id])
        if abs(value) <= MEASUREMENT_FLOW_TOL:
            continue

        node = int(tail[link_id])
        key = (
            "alighting",
            stop_id_from_node(node),
            trip_id_from_node(node),
            node_hms(node),
        )
        agg[key] = agg.get(key, 0.0) + value
        contrib.setdefault(key, []).append({"link_id": int(link_id), "value": value})

    records: list[MeasurementRecord] = []

    for (measurement_type, stop_id, trip_id, time_hms), value in sorted(agg.items()):
        if trip_id is None:
            continue

        hh, mm, ss = (int(part) for part in time_hms.split(":"))
        time = TimeOfDay(seconds_from_midnight=hh * 3600 + mm * 60 + ss)

        records.append(
            MeasurementRecord(
                method_id="synthetic_from_true_assignment",
                measurement_type=MeasurementType(measurement_type),
                stop_id=stop_id,
                time=time,
                value=float(value),
                trip_id=trip_id,
                line_id=None,
            )
        )

    mapping_path = RESULTS / "synthetic_measurement_link_mapping.json"
    mapping_records = []
    for (measurement_type, stop_id, trip_id, time_hms), value in sorted(agg.items()):
        if trip_id is None:
            continue
        parts = sorted(
            contrib.get((measurement_type, stop_id, trip_id, time_hms), []),
            key=lambda item: int(item["link_id"]),
        )
        links = []
        for part in parts:
            link_id = int(part["link_id"])
            tail_node = int(tail[link_id])
            head_node = int(head[link_id])
            links.append(
                {
                    "link_id": link_id,
                    "contribution": float(part["value"]),
                    "link_type": int(link_type[link_id]),
                    "tail_node": tail_node,
                    "tail_stop": stop_id_from_node(tail_node),
                    "tail_time": node_hms(tail_node),
                    "head_node": head_node,
                    "head_stop": stop_id_from_node(head_node),
                    "head_time": node_hms(head_node),
                }
            )
        mapping_records.append(
            {
                "measurement_type": measurement_type,
                "stop_id": stop_id,
                "trip_id": trip_id,
                "time": time_hms,
                "value": float(value),
                "contributing_links": links,
            }
        )
    mapping_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "description": (
                    "Synthetic boarding/alighting measurements and their "
                    "contributing assigned links."
                ),
                "assignment_report": "time_expanded_report_true.html",
                "measurements": mapping_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return MeasurementTable.from_records(records), mapping_path


def _generate_data(
    *,
    theta: float,
    svg_scale_x: float,
    svg_scale_y: float,
) -> None:
    if theta <= 0.0 or not np.isfinite(theta):
        raise ValueError(f"theta must be positive and finite, got {theta!r}")

    RESULTS.mkdir(parents=True, exist_ok=True)

    prior_demand_path = DATA / "prior_demand.csv"
    if not prior_demand_path.exists():
        raise FileNotFoundError(f"Missing prior demand file: {prior_demand_path}")

    generated_demand_path = RESULTS / "demand.csv"
    shutil.copy2(prior_demand_path, generated_demand_path)

    scenario, od_values, tmp = _load_true_demand_scenario()
    with tmp:
        config = AssignmentConfig()
        artifacts = prepare_assignment(scenario=scenario, config=config)

        assignment = assign(
            od_values=od_values,
            artifacts=artifacts,
            theta=float(theta),
            return_group_link_flows=False,
        )

        print("Data-generation mode")
        print(f"OD records: {len(scenario.demand.records) if scenario.demand else 0}")
        print(f"Total true demand: {float(np.asarray(od_values).sum()):.6g}")
        print(f"Theta used for synthetic assignment: {assignment.theta:.6g}")
        print("Theta interpretation:")
        print("  small theta -> strong sensitivity to cost -> concentrated on low-cost paths")
        print("  large theta -> weak sensitivity to cost -> more dispersed route choice")

        report_path = RESULTS / "time_expanded_report_true.html"
        write_time_expanded_report_from_assignment(
            scenario=scenario,
            assignment=assignment,
            config=config,
            output_path=report_path,
            title=f"Example 02 — true-demand time-expanded graph report, theta={theta:g}",
            svg_scale_x=float(svg_scale_x),
            svg_scale_y=float(svg_scale_y),
        )

        measurement_table, mapping_path = _build_measurements_and_mapping(
            scenario=scenario,
            artifacts=artifacts,
            assignment=assignment,
        )

        measurements_path = RESULTS / "measurements_boarding_alighting.csv"
        write_measurements_csv(measurement_table, measurements_path)

        true_assignment_path = RESULTS / "true_assignment.npz"
        np.savez_compressed(
            true_assignment_path,
            theta=float(assignment.theta),
            od_values=np.asarray(od_values, dtype=float),
            link_flow=np.asarray(assignment.link_flow, dtype=float),
            link_cost=np.asarray(assignment.link_cost, dtype=float),
        )

        summary_path = RESULTS / "preprocessing_summary.json"
        summary = {
            "mode": "generate",
            "source_data_dir": str(DATA),
            "results_dir": str(RESULTS),
            "true_demand_file": str(DATA / "true_demand.csv"),
            "prior_demand_file": str(prior_demand_path),
            "generated_demand_file": str(generated_demand_path),
            "measurements_file": str(measurements_path),
            "time_expanded_report": str(report_path),
            "measurement_mapping_json": str(mapping_path),
            "true_assignment_file": str(true_assignment_path),
            "num_od_records": len(scenario.demand.records) if scenario.demand else 0,
            "total_true_demand": float(np.asarray(od_values).sum()),
            "theta": float(assignment.theta),
            "num_measurements": len(measurement_table.records),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("Pre-processing complete.")
    print()
    print("Generated files:")
    print(f"  {generated_demand_path}")
    print(f"  {measurements_path}")
    print(f"  {report_path}")
    print(f"  {mapping_path}")
    print(f"  {true_assignment_path}")
    print(f"  {summary_path}")
    print()
    print("Next step:")
    print("  cd ../estimation")
    print("  python run_both.py")
    print()
    print("The estimation scripts should read:")
    print(f"  stable scenario files from: {DATA}")
    print(f"  generated estimation inputs from: {RESULTS}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="investigate",
        choices=["investigate", "generate"],
        help="Use 'investigate' to compare theta values, or 'generate' to create estimation data.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=None,
        help="Theta used in generate mode. Required when --mode generate.",
    )
    parser.add_argument(
        "--theta-grid",
        type=float,
        nargs="+",
        default=DEFAULT_THETA_GRID,
        help="Theta values inspected in investigation mode.",
    )
    parser.add_argument("--svg-scale-x", type=float, default=1.8)
    parser.add_argument("--svg-scale-y", type=float, default=2.4)
    args = parser.parse_args()

    if args.mode == "investigate":
        _write_theta_investigation(
            theta_grid=[float(theta) for theta in args.theta_grid],
            svg_scale_x=float(args.svg_scale_x),
            svg_scale_y=float(args.svg_scale_y),
        )
        return

    if args.theta is None:
        raise ValueError("--theta is required when --mode generate.")

    _generate_data(
        theta=float(args.theta),
        svg_scale_x=float(args.svg_scale_x),
        svg_scale_y=float(args.svg_scale_y),
    )


if __name__ == "__main__":
    main()
