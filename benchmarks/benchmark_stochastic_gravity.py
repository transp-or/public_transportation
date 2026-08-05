"""Synthetic benchmark for the streaming persisted-shard gravity backend."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import numpy as np

from benchmarks.benchmark_sharded_gravity_operator import _inputs, _layout, _problem
from public_transportation.inference.sharded_fixed_routing import (
    FixedRoutingPreparationConfig,
    prepare_fixed_routing_sharded,
)
from public_transportation.inference.sharded_matrix_free_operator import (
    ShardedMatrixFreeFixedRoutingMeasurementOperator,
)
from public_transportation.inference.stochastic_gravity import (
    StochasticGravityConfig,
    stochastic_gravity_value_and_gradient,
)
from public_transportation.measurement.mapping import AggregationSpec


def run_benchmark(
    *,
    destination_groups: int = 32,
    od_cells: int = 256,
    measurements: int = 128,
    efforts: tuple[float, ...] = (10.0, 25.0, 50.0, 100.0),
) -> dict[str, object]:
    inputs = _inputs(nodes=128, degree=2, groups=destination_groups, od_cells=od_cells)
    spec = AggregationSpec(
        num_measurements=measurements,
        measurement_index=np.arange(measurements, dtype=np.int32),
        link_index=np.arange(measurements, dtype=np.int32) % inputs.graph.num_links,
    )
    with tempfile.TemporaryDirectory(prefix="stochastic-gravity-") as temporary:
        root = Path(temporary)
        prepared = prepare_fixed_routing_sharded(
            inputs=inputs,
            theta=1.0,
            config=FixedRoutingPreparationConfig(
                maximum_groups_per_shard=2,
                cache_directory=root / "routing",
                checkpoint_directory=root / "checkpoints",
                resident_shard_limit=1,
            ),
        )
        operator = ShardedMatrixFreeFixedRoutingMeasurementOperator(
            inputs=inputs,
            routing=prepared.routing,
            spec=spec,
            compact_layout=_layout(od_cells),
            resident_shard_limit=1,
            operator_shards_per_batch=1,
        )
        problem, raw = _problem(operator)
        warm_effort = min(item for item in efforts if item < 100.0)
        sampled_warmup = stochastic_gravity_value_and_gradient(
            raw,
            problem=problem,
            config=StochasticGravityConfig(
                effort_percent=warm_effort, concurrency=1
            ),
        )
        measured = []
        # Exercise sampled efforts first so their peak-RSS observation is not
        # contaminated by a preceding exact evaluation on platforms that expose
        # only process high-water RSS.
        for effort in (item for item in efforts if item != 100.0):
            result = stochastic_gravity_value_and_gradient(
                raw,
                problem=problem,
                config=StochasticGravityConfig(effort_percent=effort, concurrency=1),
            )
            measured.append((effort, result))
        exact_warmup = stochastic_gravity_value_and_gradient(
            raw,
            problem=problem,
            config=StochasticGravityConfig(effort_percent=100.0),
        )
        exact = stochastic_gravity_value_and_gradient(
            raw,
            problem=problem,
            config=StochasticGravityConfig(effort_percent=100.0),
        )
        if 100.0 in efforts:
            measured.append((100.0, exact))
        assert exact.gradient is not None and exact.predicted_measurements is not None
        rows = []
        for effort, result in measured:
            assert result.gradient is not None and result.predicted_measurements is not None
            rows.append(
                {
                    "effort_percent": effort,
                    "realized_effort_percent": result.selection.realized_effort_percent,
                    "selected_shards": len(result.selection.selected_shard_ids),
                    "wall_time_seconds": result.wall_time_seconds,
                    "forward_seconds": result.forward_seconds,
                    "reverse_seconds": result.reverse_seconds,
                    "peak_rss_bytes": result.peak_rss_bytes,
                    "compiled_forward_shapes": len(operator._partial_compiled_forward),
                    "compiled_reverse_shapes": len(operator._partial_compiled_reverse),
                    "measurement_relative_error": float(
                        np.linalg.norm(
                            result.predicted_measurements - exact.predicted_measurements
                        )
                        / max(np.linalg.norm(exact.predicted_measurements), 1.0e-12)
                    ),
                    "gradient_relative_error": float(
                        np.linalg.norm(result.gradient - exact.gradient)
                        / max(np.linalg.norm(exact.gradient), 1.0e-12)
                    ),
                    "quality": None if result.quality is None else asdict(result.quality),
                    "exact": result.exact,
                }
            )
        return {
            "configuration": {
                "destination_groups": destination_groups,
                "persisted_shards": prepared.routing.num_shards,
                "od_cells": od_cells,
                "measurements": measurements,
                "concurrency": 1,
                "prepared_batch_retention": False,
                "sampled_warmup_effort_percent": warm_effort,
                "sampled_warmup_seconds": sampled_warmup.wall_time_seconds,
                "exact_warmup_seconds": exact_warmup.wall_time_seconds,
            },
            "results": rows,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_benchmark()
    text = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(text)
    else:
        arguments.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
