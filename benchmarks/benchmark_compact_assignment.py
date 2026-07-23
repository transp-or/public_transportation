"""Benchmark full versus compact assignment for frozen-zero OD demand.

Run from the repository root, for example::

    .venv/bin/python benchmarks/benchmark_compact_assignment.py --example simple_example_02

Timings are deliberately kept outside pytest because they are machine-dependent.
The script always checks numerical equivalence before reporting performance.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from statistics import median

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.domain import FixedODDemand, FixedODRecord, Scenario
from public_transportation.inference.assignment_adapter import (
    assign_link_flow,
    build_assignment_inputs,
)
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.od_parameter_layout import build_od_parameter_layout


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = ROOT / "docs/source/examples"
SCENARIO_FILES = (
    "metadata.json",
    "stops.csv",
    "lines.csv",
    "trips.csv",
    "stop_times.csv",
    "time_bins.csv",
)


def _load_example(name: str, destination: Path) -> Scenario:
    source = EXAMPLES_ROOT / name
    for filename in SCENARIO_FILES:
        shutil.copy2(source / "data" / filename, destination / filename)
    shutil.copy2(source / "pre_processing/results/demand.csv", destination / "demand.csv")
    return Scenario.from_folder(destination)


def _fixed_zero_records(scenario: Scenario, indices: tuple[int, ...]) -> FixedODDemand:
    records = tuple(scenario.demand.records)
    return FixedODDemand(
        records=tuple(
            FixedODRecord(
                records[index].origin_stop_id,
                records[index].dest_stop_id,
                records[index].time_bin_id,
                0.0,
            )
            for index in indices
        )
    )


def _distributed_indices(num_od: int, fraction: float) -> tuple[int, ...]:
    count = min(num_od - 1, int(round(num_od * fraction)))
    if count <= 0:
        return ()
    return tuple(sorted(np.linspace(0, num_od - 1, count, dtype=int).tolist()))


def _destination_indices(scenario: Scenario, fraction: float) -> tuple[int, ...]:
    records = tuple(scenario.demand.records)
    target = min(len(records) - 1, int(round(len(records) * fraction)))
    by_destination: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_destination.setdefault(str(record.dest_stop_id), []).append(index)
    selected: list[int] = []
    for destination in sorted(by_destination, key=lambda key: (-len(by_destination[key]), key)):
        candidate = by_destination[destination]
        if len(selected) + len(candidate) >= len(records):
            continue
        selected.extend(candidate)
        if len(selected) >= target:
            break
    return tuple(sorted(selected))


def _timed_call(function, argument) -> tuple[float, np.ndarray]:
    started = time.perf_counter()
    value = function(argument)
    value.block_until_ready()
    return time.perf_counter() - started, np.asarray(value)


def _median_warm_time(function, argument, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        elapsed, _ = _timed_call(function, argument)
        timings.append(elapsed)
    return median(timings)


def _measure_path(*, function, gradient, z, repeats: int) -> dict[str, float | np.ndarray]:
    compile_forward_s, forward_value = _timed_call(function, z)
    warm_forward_s = _median_warm_time(function, z, repeats)
    compile_gradient_s, gradient_value = _timed_call(gradient, z)
    warm_gradient_s = _median_warm_time(gradient, z, repeats)
    return {
        "compile_forward_s": compile_forward_s,
        "warm_forward_s": warm_forward_s,
        "compile_gradient_s": compile_gradient_s,
        "warm_gradient_s": warm_gradient_s,
        "flow": forward_value,
        "gradient": gradient_value,
    }


def _benchmark_case(*, scenario, artifacts, frozen_indices, label: str, repeats: int) -> dict:
    parameter_layout = build_od_parameter_layout(
        scenario=scenario,
        fixed_demand=_fixed_zero_records(scenario, frozen_indices),
    )
    compact_layout = build_compact_od_assignment_layout(parameter_layout=parameter_layout)
    full_inputs = build_assignment_inputs(artifacts=artifacts)
    compact_inputs = build_assignment_inputs(
        artifacts=artifacts,
        compact_layout=compact_layout,
    )
    z = jnp.linspace(-0.2, 0.3, parameter_layout.num_free, dtype=jnp.float32)
    theta = jnp.asarray(5.0, dtype=jnp.float32)

    def full_forward(z_free):
        return assign_link_flow(
            inputs=full_inputs,
            f=parameter_layout.reconstruct_jax(z_free),
            theta=theta,
        )

    def compact_forward(z_free):
        return assign_link_flow(
            inputs=compact_inputs,
            f=compact_layout.assemble_compact_jax(z_free),
            theta=theta,
        )

    def objective(forward, z_free):
        flow = forward(z_free)
        return jnp.square(flow).sum()

    full_jit = jax.jit(full_forward)
    full_gradient = jax.jit(jax.grad(lambda value: objective(full_forward, value)))
    jax.clear_caches()
    full = _measure_path(
        function=full_jit,
        gradient=full_gradient,
        z=z,
        repeats=repeats,
    )

    compact_jit = jax.jit(compact_forward)
    compact_gradient = jax.jit(jax.grad(lambda value: objective(compact_forward, value)))
    jax.clear_caches()
    compact = _measure_path(
        function=compact_jit,
        gradient=compact_gradient,
        z=z,
        repeats=repeats,
    )

    if not np.allclose(full["flow"], compact["flow"], rtol=2e-6, atol=2e-6):
        raise RuntimeError(f"{label}: compact and full link flows differ.")
    if not np.allclose(full["gradient"], compact["gradient"], rtol=3e-5, atol=3e-5):
        raise RuntimeError(f"{label}: compact and full gradients differ.")

    original_groups = int(artifacts.od_groups.group_dest_node.shape[0])
    active_groups = int(compact_inputs.group_dest_node.shape[0])
    result = {
        "case": label,
        "num_od_total": parameter_layout.num_od_total,
        "num_frozen_zero": parameter_layout.num_fixed_zero,
        "num_assignment_active": compact_layout.num_active,
        "original_destination_groups": original_groups,
        "active_destination_groups": active_groups,
    }
    for metric in (
        "compile_forward_s",
        "warm_forward_s",
        "compile_gradient_s",
        "warm_gradient_s",
    ):
        full_value = float(full[metric])
        compact_value = float(compact[metric])
        result[f"full_{metric}"] = full_value
        result[f"compact_{metric}"] = compact_value
        result[f"{metric}_speedup"] = full_value / compact_value
    return result


def run_benchmark(*, example: str, repeats: int) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"pt-{example}-") as temporary:
        scenario = _load_example(example, Path(temporary))
        artifacts = prepare_assignment(scenario=scenario, config=AssignmentConfig())
        num_od = len(scenario.demand.records)
        cases = (
            ("all_free", ()),
            ("distributed_90pct_zero", _distributed_indices(num_od, 0.9)),
            ("destination_concentrated_zero", _destination_indices(scenario, 0.6)),
        )
        results = [
            _benchmark_case(
                scenario=scenario,
                artifacts=artifacts,
                frozen_indices=indices,
                label=label,
                repeats=repeats,
            )
            for label, indices in cases
        ]
    return {"example": example, "repeats": repeats, "cases": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example",
        choices=("simple_example_01", "simple_example_02", "all"),
        default="all",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    examples = (
        ("simple_example_01", "simple_example_02")
        if args.example == "all"
        else (args.example,)
    )
    payload = {
        "jax_backend": jax.default_backend(),
        "results": [run_benchmark(example=name, repeats=args.repeats) for name in examples],
    }
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
