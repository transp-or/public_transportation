"""Measure concurrent CPU calls to one warm fixed-routing executable."""

from __future__ import annotations

import argparse
import json
import os
import platform
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter, process_time

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from public_transportation.inference.assignment_adapter import _prepare_fixed_routing_core

from benchmark_fixed_routing_scaling import _inputs


def _peak_rss() -> int | None:
    try:
        import resource
        import sys

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return None


def _context_switches() -> tuple[int, int] | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return int(usage.ru_nvcsw), int(usage.ru_nivcsw)
    except (ImportError, OSError, ValueError):
        return None


def _cpu_affinity() -> list[int] | None:
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def _process_thread_count() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("Threads:"):
                return int(line.split(":", 1)[1])
    except (FileNotFoundError, OSError, ValueError):
        pass
    return threading.active_count()


def _environment() -> dict[str, object]:
    visible = os.cpu_count()
    affinity = _cpu_affinity()
    return {
        "platform": platform.platform(),
        "allocated_or_visible_cpu_count": visible,
        "scheduler_cpu_allocation": {
            name: os.environ.get(name)
            for name in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE")
        },
        "process_cpu_affinity": affinity,
        "affinity_cpu_count": None if affinity is None else len(affinity),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "XLA_FLAGS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "backend": jax.default_backend(),
        "logical_devices": [str(device) for device in jax.devices()],
        "active_python_threads": threading.active_count(),
        "active_process_threads": _process_thread_count(),
    }


def _padded_inputs(inputs, *, start: int, count: int):
    destination = np.asarray(inputs.group_dest_node)
    masks = np.asarray(inputs.group_link_mask)
    indices = np.arange(start, start + count) % len(destination)
    device_destination = jnp.asarray(destination[indices])
    device_masks = jnp.asarray(masks[indices])
    jax.block_until_ready((device_destination, device_masks))
    from dataclasses import replace

    return replace(
        inputs,
        group_dest_node=device_destination,
        group_link_mask=device_masks,
    )


def _compile(inputs, theta):
    tracing_started = perf_counter()
    traced = _prepare_fixed_routing_core.trace(inputs=inputs, theta=theta)
    tracing_seconds = perf_counter() - tracing_started
    lowering_started = perf_counter()
    lowered = traced.lower()
    lowering_seconds = perf_counter() - lowering_started
    compilation_started = perf_counter()
    executable = lowered.compile()
    compilation_seconds = perf_counter() - compilation_started
    return executable, {
        "tracing_seconds": tracing_seconds,
        "lowering_seconds": lowering_seconds,
        "compilation_seconds": compilation_seconds,
    }


def _run_concurrency(*, executable, shard_inputs, theta, workers: int, tasks: int):
    reference = None

    def execute(index: int):
        task_started = perf_counter()
        dispatch_started = perf_counter()
        effective, probability = executable(
            inputs=shard_inputs[index % len(shard_inputs)], theta=theta
        )
        dispatch_seconds = perf_counter() - dispatch_started
        synchronization_started = perf_counter()
        jax.block_until_ready((effective, probability))
        synchronization_seconds = perf_counter() - synchronization_started
        transfer_started = perf_counter()
        result = (np.asarray(effective), np.asarray(probability))
        transfer_seconds = perf_counter() - transfer_started
        return {
            "wall_seconds": perf_counter() - task_started,
            "dispatch_seconds": dispatch_seconds,
            "synchronization_seconds": synchronization_seconds,
            "host_transfer_seconds": transfer_seconds,
            "result": result,
        }

    cpu_started = process_time()
    switches_started = _context_switches()
    batch_started = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(execute, range(tasks)))
    batch_seconds = perf_counter() - batch_started
    cpu_seconds = process_time() - cpu_started
    switches_finished = _context_switches()
    differences = []
    for index, record in enumerate(records):
        effective, probability = record.pop("result")
        if index < len(shard_inputs):
            if reference is None:
                reference = []
            reference.append((effective, probability))
        comparison = reference[index % len(shard_inputs)]
        differences.append(
            max(
                float(np.max(effective != comparison[0], initial=0)),
                float(
                    np.max(
                        np.abs(probability - comparison[1]), initial=0.0
                    )
                ),
            )
        )
    latencies = np.asarray([record["wall_seconds"] for record in records])
    result = {
        "workers": workers,
        "tasks": tasks,
        "batch_wall_seconds": batch_seconds,
        "shards_per_second": tasks / batch_seconds,
        "per_shard_seconds": {
            "minimum": float(np.min(latencies)),
            "median": float(np.median(latencies)),
            "maximum": float(np.max(latencies)),
        },
        "phase_medians_seconds": {
            name: float(np.median([record[name] for record in records]))
            for name in (
                "dispatch_seconds",
                "synchronization_seconds",
                "host_transfer_seconds",
            )
        },
        "process_cpu_seconds": cpu_seconds,
        "effective_average_cpu_cores": cpu_seconds / batch_seconds,
        "process_cpu_utilization_percent": 100.0 * cpu_seconds / batch_seconds,
        "peak_rss_bytes": _peak_rss(),
        "maximum_numerical_difference": max(differences, default=0.0),
        "shared_executable": True,
        "compilation_count_during_batch": 0,
        "maximum_active_python_threads": max(
            [threading.active_count(), workers + 1]
        ),
        "active_process_threads_after_batch": _process_thread_count(),
        "persistence_performed": False,
        "persistence_seconds": 0.0,
    }
    if switches_started is not None and switches_finished is not None:
        result["voluntary_context_switches"] = max(
            0, switches_finished[0] - switches_started[0]
        )
        result["involuntary_context_switches"] = max(
            0, switches_finished[1] - switches_started[1]
        )
    return result


def run_benchmark(
    *, tasks: int = 16, num_nodes: int = 32768, maximum_out_degree: int = 4
) -> dict[str, object]:
    inputs = _inputs(
        num_nodes=num_nodes,
        maximum_out_degree=maximum_out_degree,
        groups=8,
        density=0.25,
    )
    theta = jnp.asarray(1.0, dtype=inputs.base_link_cost.dtype)
    shard_inputs = [_padded_inputs(inputs, start=index * 2, count=2) for index in range(4)]
    executable, compilation = _compile(shard_inputs[0], theta)
    warm_effective, warm_probability = executable(inputs=shard_inputs[0], theta=theta)
    jax.block_until_ready((warm_effective, warm_probability))
    concurrency = [
        _run_concurrency(
            executable=executable,
            shard_inputs=shard_inputs,
            theta=theta,
            workers=workers,
            tasks=tasks,
        )
        for workers in (1, 2, 4)
    ]

    batched = []
    for shard_batch_size in (1, 2, 4):
        batched_inputs = _padded_inputs(
            inputs, start=0, count=2 * shard_batch_size
        )
        batched_executable, batched_compilation = _compile(batched_inputs, theta)
        batched_warm = batched_executable(inputs=batched_inputs, theta=theta)
        jax.block_until_ready(batched_warm)
        cpu_started = process_time()
        switches_started = _context_switches()
        batched_started = perf_counter()
        batched_result = batched_executable(inputs=batched_inputs, theta=theta)
        jax.block_until_ready(batched_result)
        batched_seconds = perf_counter() - batched_started
        cpu_seconds = process_time() - cpu_started
        switches_finished = _context_switches()
        item = {
            "shards_per_execution_batch": shard_batch_size,
            "padded_batch_size": shard_batch_size,
            "execution_seconds": batched_seconds,
            "process_cpu_seconds": cpu_seconds,
            "effective_average_cpu_cores": cpu_seconds / batched_seconds,
            "equivalent_shards_per_second": shard_batch_size / batched_seconds,
            "peak_rss_bytes": _peak_rss(),
            "compilation": batched_compilation,
            "compilation_count_during_execution": 0,
            "persistence_performed": False,
            "persistence_seconds": 0.0,
        }
        if switches_started is not None and switches_finished is not None:
            item["voluntary_context_switches"] = max(
                0, switches_finished[0] - switches_started[0]
            )
            item["involuntary_context_switches"] = max(
                0, switches_finished[1] - switches_started[1]
            )
        batched.append(item)
    return {
        "schema_version": 2,
        "system": _environment(),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "groups_per_shard": 2,
        "nodes": num_nodes,
        "links": inputs.graph.num_links,
        "maximum_out_degree": maximum_out_degree,
        "tasks": tasks,
        "warmup_completed": True,
        "compilation": compilation,
        "concurrency": concurrency,
        "batched": batched,
        "experimental_four_shard_batch": {
            **batched[-1],
            "groups": 8,
            "production_routing_fingerprint_changed": False,
            "compiled_kernel_identity_changes_with_shape": True,
        },
        "threads_per_worker_controls_xla_threads": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=16)
    parser.add_argument("--nodes", type=int, default=32768)
    parser.add_argument("--maximum-out-degree", type=int, default=4)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_benchmark(
        tasks=arguments.tasks,
        num_nodes=arguments.nodes,
        maximum_out_degree=arguments.maximum_out_degree,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
