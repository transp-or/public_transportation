"""Explicit CPU multi-device experiment for independent routing shards.

Run in a fresh process whose ``XLA_FLAGS`` sets
``--xla_force_host_platform_device_count=N``.  The benchmark uses ``pmap``;
merely exposing devices is deliberately not treated as an experiment.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from time import perf_counter, process_time

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.inference.assignment_adapter import (
    _prepare_fixed_routing_core,
)

from benchmark_fixed_routing_scaling import _inputs


def _run(*, devices: int, num_nodes: int) -> dict[str, object]:
    available = jax.devices("cpu")
    if len(available) < devices:
        raise RuntimeError(
            f"requested {devices} logical CPU devices but found {len(available)}; "
            "set --xla_force_host_platform_device_count before importing JAX."
        )
    inputs = _inputs(
        num_nodes=num_nodes,
        maximum_out_degree=4,
        groups=2 * devices,
        density=0.25,
    )
    theta = jnp.asarray(1.0, dtype=inputs.base_link_cost.dtype)
    destinations = np.asarray(inputs.group_dest_node).reshape(devices, 2)
    masks = np.asarray(inputs.group_link_mask).reshape(
        devices, 2, inputs.graph.num_links
    )

    def per_device(destination, mask):
        local = replace(
            inputs,
            group_dest_node=destination,
            group_link_mask=mask,
        )
        return _prepare_fixed_routing_core(inputs=local, theta=theta)

    executable = jax.pmap(per_device, devices=available[:devices])
    warm = executable(jnp.asarray(destinations), jnp.asarray(masks))
    jax.block_until_ready(warm)
    cpu_started = process_time()
    started = perf_counter()
    effective, probability = executable(
        jnp.asarray(destinations), jnp.asarray(masks)
    )
    jax.block_until_ready((effective, probability))
    wall = perf_counter() - started
    cpu = process_time() - cpu_started

    single_inputs = replace(
        inputs,
        group_dest_node=jnp.asarray(destinations.reshape(-1)),
        group_link_mask=jnp.asarray(
            masks.reshape(-1, inputs.graph.num_links)
        ),
    )
    reference = _prepare_fixed_routing_core(
        inputs=single_inputs, theta=theta
    )
    jax.block_until_ready(reference)
    effective_reference, probability_reference = map(np.asarray, reference)
    effective_host = np.asarray(effective).reshape(effective_reference.shape)
    probability_host = np.asarray(probability).reshape(
        probability_reference.shape
    )
    placed_devices = sorted(
        str(shard.device) for shard in probability.addressable_shards
    )
    return {
        "schema_version": 1,
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "backend": jax.default_backend(),
        "logical_devices": [str(device) for device in available],
        "requested_devices": devices,
        "placed_devices": placed_devices,
        "explicit_pmap": True,
        "wall_seconds": wall,
        "process_cpu_seconds": cpu,
        "effective_average_cpu_cores": cpu / wall,
        "shards_per_second": devices / wall,
        "maximum_mask_difference": int(
            np.max(effective_host != effective_reference, initial=0)
        ),
        "maximum_probability_difference": float(
            np.max(
                np.abs(probability_host - probability_reference), initial=0.0
            )
        ),
        "nodes": num_nodes,
        "links": inputs.graph.num_links,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--nodes", type=int, default=32768)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = _run(devices=args.devices, num_nodes=args.nodes)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
