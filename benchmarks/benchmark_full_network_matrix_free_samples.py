"""Run isolated fixed-routing benchmarks across representative TPG groups.

Each sample runs in a fresh subprocess so compiler state from one OD shape
cannot accumulate and invalidate the configured memory ceiling. This benchmark
is expensive and opt-in.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
SINGLE = ROOT / "benchmarks/benchmark_full_network_matrix_free_group.py"
DEFAULT_GROUPS = (1667, 273, 595, 140, 105, 130)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", type=int, default=DEFAULT_GROUPS)
    parser.add_argument("--memory-ceiling-gib", type=float, default=24.0)
    parser.add_argument("--warm-evaluations", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sample-directory",
        type=Path,
        default=ROOT / "benchmarks/full_network_matrix_free_samples",
    )
    args = parser.parse_args()
    if len(set(args.groups)) != len(args.groups):
        parser.error("groups must not contain duplicates")

    args.sample_directory.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("MPLCONFIGDIR", "/tmp/public-transportation-mpl")
    environment.setdefault("XDG_CACHE_HOME", "/tmp/public-transportation-xdg")
    environment.setdefault(
        "UV_CACHE_DIR", "/tmp/public-transportation-uv-cache"
    )
    reports = []
    started = perf_counter()
    for ordinal, group_index in enumerate(args.groups, start=1):
        destination = args.sample_directory / f"group_{group_index}.json"
        command = [
            sys.executable,
            str(SINGLE),
            "--group-index",
            str(group_index),
            "--memory-ceiling-gib",
            str(args.memory_ceiling_gib),
            "--warm-evaluations",
            str(args.warm_evaluations),
            "--output",
            str(destination),
        ]
        print(
            f"[{ordinal}/{len(args.groups)}] benchmarking destination group "
            f"{group_index} in an isolated process",
            flush=True,
        )
        child_started = perf_counter()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=None,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Destination group {group_index} failed with exit code "
                f"{completed.returncode}."
            )
        report = json.loads(destination.read_text(encoding="utf-8"))
        report["orchestrator_wall_seconds"] = perf_counter() - child_started
        reports.append(report)
        print(
            f"  {report['sample']['free_cells']} free cells; warm value-gradient "
            f"{report['kernels']['value_and_gradient']['warm']['median_seconds']:.3f} s; "
            f"peak {report['memory']['peak_rss_bytes'] / 1024**3:.2f} GiB",
            flush=True,
        )

    warm = sorted(
        item["kernels"]["value_and_gradient"]["warm"]["median_seconds"]
        for item in reports
    )
    middle = len(warm) // 2
    median_warm = (
        warm[middle]
        if len(warm) % 2
        else 0.5 * (warm[middle - 1] + warm[middle])
    )
    forward = sorted(
        item["kernels"]["flow_loading"]["warm"]["median_seconds"]
        for item in reports
    )
    median_forward = (
        forward[middle]
        if len(forward) % 2
        else 0.5 * (forward[middle - 1] + forward[middle])
    )
    routing = sorted(
        item["preparation_seconds"]["fixed_routing_preparation"]
        for item in reports
    )
    median_routing = (
        routing[middle]
        if len(routing) % 2
        else 0.5 * (routing[middle - 1] + routing[middle])
    )
    combined = {
        "schema_version": 1,
        "mode": "isolated_representative_destination_samples",
        "groups": list(args.groups),
        "samples": reports,
        "summary": {
            "num_samples": len(reports),
            "median_sample_value_gradient_seconds": median_warm,
            "median_sample_forward_seconds": median_forward,
            "median_sample_routing_preparation_seconds": median_routing,
            "sequential_full_network_value_gradient_seconds": median_warm * 1898,
            "sequential_full_network_value_gradient_hours": median_warm
            * 1898
            / 3600.0,
            "sequential_two_pass_streamed_seconds": (median_forward + median_warm)
            * 1898,
            "sequential_two_pass_streamed_hours": (median_forward + median_warm)
            * 1898
            / 3600.0,
            "maximum_peak_rss_bytes": max(
                item["memory"]["peak_rss_bytes"] for item in reports
            ),
            "total_orchestrator_seconds": perf_counter() - started,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(combined["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
