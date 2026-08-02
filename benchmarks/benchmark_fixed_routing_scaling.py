"""Controlled scaling benchmark for complete-link fixed-routing preparation."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.assignment.jax_graph_types import JaxGraph
from public_transportation.inference.assignment_adapter import (
    AssignmentInputs,
    _prepare_fixed_routing_core,
)
from public_transportation.inference.sharded_fixed_routing import (
    FixedRoutingPreparationConfig,
    load_fixed_routing_shard,
    prepare_fixed_routing_sharded,
)


def _dag(*, num_nodes: int, maximum_out_degree: int) -> JaxGraph:
    edges = [
        (tail, tail + step)
        for tail in range(num_nodes)
        for step in range(1, maximum_out_degree + 1)
        if tail + step < num_nodes
    ]
    tail = np.asarray([item[0] for item in edges], dtype=np.int32)
    head = np.asarray([item[1] for item in edges], dtype=np.int32)
    num_links = len(edges)
    out_links = np.zeros((num_nodes, maximum_out_degree), dtype=np.int32)
    out_mask = np.zeros_like(out_links, dtype=bool)
    cursor = np.zeros(num_nodes + 1, dtype=np.int32)
    for node in range(num_nodes):
        indices = np.flatnonzero(tail == node)
        out_links[node, : len(indices)] = indices
        out_mask[node, : len(indices)] = True
        cursor[node + 1] = cursor[node] + len(indices)
    zeros_i = jnp.zeros(num_nodes, dtype=jnp.int32)
    return JaxGraph(
        num_nodes=num_nodes,
        num_links=num_links,
        tail=jnp.asarray(tail),
        head=jnp.asarray(head),
        topo_order=jnp.arange(num_nodes, dtype=jnp.int32),
        topo_order_rev=jnp.arange(num_nodes - 1, -1, -1, dtype=jnp.int32),
        node_time=jnp.arange(num_nodes, dtype=jnp.float32),
        node_stop_index=zeros_i,
        node_time_s=zeros_i,
        node_kind=zeros_i,
        node_trip_index=jnp.full(num_nodes, -1, dtype=jnp.int32),
        out_start=jnp.asarray(cursor),
        out_links_csr=jnp.arange(num_links, dtype=jnp.int32),
        out_links=jnp.asarray(out_links),
        out_mask=jnp.asarray(out_mask),
        link_type=jnp.zeros(num_links, dtype=jnp.int32),
        travel_time=jnp.ones(num_links, dtype=jnp.float32),
        capacity=jnp.full(num_links, jnp.inf, dtype=jnp.float32),
        link_trip_index=jnp.full(num_links, -1, dtype=jnp.int32),
        node_time_bin_index=jnp.full(num_nodes, -1, dtype=jnp.int32),
    )


def _inputs(
    *, num_nodes: int, maximum_out_degree: int, groups: int, density: float
) -> AssignmentInputs:
    graph = _dag(num_nodes=num_nodes, maximum_out_degree=maximum_out_degree)
    rng = np.random.default_rng(19332)
    mask = rng.random((groups, graph.num_links)) < density
    return AssignmentInputs(
        graph=graph,
        base_link_cost=jnp.ones(graph.num_links, dtype=jnp.float32),
        group_dest_node=jnp.full(groups, num_nodes - 1, dtype=jnp.int32),
        group_link_mask=jnp.asarray(mask),
        od_origin_node=jnp.zeros(groups, dtype=jnp.int32),
        group_od_index_padded=jnp.arange(groups, dtype=jnp.int32)[:, None],
        group_od_mask=jnp.ones((groups, 1), dtype=bool),
    )


def _case(
    *, num_nodes: int, maximum_out_degree: int, groups: int, density: float
) -> dict[str, object]:
    inputs = _inputs(
        num_nodes=num_nodes,
        maximum_out_degree=maximum_out_degree,
        groups=groups,
        density=density,
    )
    theta = jnp.asarray(1.0, dtype=jnp.float32)
    executable = _prepare_fixed_routing_core.trace(
        inputs=inputs, theta=theta
    ).lower().compile()
    first = executable(inputs=inputs, theta=theta)
    jax.block_until_ready(first)
    samples = []
    for _ in range(3):
        started = perf_counter()
        result = executable(inputs=inputs, theta=theta)
        jax.block_until_ready(result)
        samples.append(perf_counter() - started)
    return {
        "nodes": num_nodes,
        "links": inputs.graph.num_links,
        "groups": groups,
        "enabled_density": density,
        "maximum_out_degree": maximum_out_degree,
        "median_warm_seconds": float(np.median(samples)),
        "groups_links": groups * inputs.graph.num_links,
        "groups_enabled_links": int(
            groups * inputs.graph.num_links * density
        ),
    }


def run() -> dict[str, object]:
    cases = []
    for nodes in (256, 1024, 4096):
        for groups in (1, 4):
            for density in (0.1, 0.5, 1.0):
                cases.append(
                    _case(
                        num_nodes=nodes,
                        maximum_out_degree=2,
                        groups=groups,
                        density=density,
                    )
                )
    for degree in (1, 4, 8):
        cases.append(
            _case(
                num_nodes=1024,
                maximum_out_degree=degree,
                groups=4,
                density=0.25,
            )
        )
    parallel_inputs = _inputs(
        num_nodes=8192,
        maximum_out_degree=4,
        groups=16,
        density=0.25,
    )
    parallel_cases = []
    reference = None
    with tempfile.TemporaryDirectory(prefix="routing-scaling-") as temporary:
        root = Path(temporary)
        for workers in (1, 2, 4):
            started = perf_counter()
            prepared = prepare_fixed_routing_sharded(
                inputs=parallel_inputs,
                theta=1.0,
                config=FixedRoutingPreparationConfig(
                    maximum_groups_per_shard=2,
                    construction_workers=workers,
                    cache_directory=root / f"cache-{workers}",
                    checkpoint_directory=root / f"checkpoint-{workers}",
                ),
            )
            elapsed = perf_counter() - started
            probability = np.concatenate(
                [
                    load_fixed_routing_shard(
                        routing=prepared.routing, descriptor=descriptor
                    ).group_link_probability
                    for descriptor in prepared.plan.descriptors
                ]
            )
            if reference is None:
                reference = probability
                difference = 0.0
                serial_seconds = elapsed
            else:
                difference = float(np.max(np.abs(probability - reference)))
            parallel_cases.append(
                {
                    "workers": workers,
                    "elapsed_seconds": elapsed,
                    "speedup": serial_seconds / elapsed,
                    "parallel_efficiency": serial_seconds / elapsed / workers,
                    "compilation_count": prepared.compilation_count,
                    "peak_rss_bytes": prepared.peak_rss_bytes,
                    "maximum_probability_difference": difference,
                    "shards": prepared.routing.num_shards,
                    "groups_per_shard": prepared.plan.groups_per_full_shard,
                }
            )
    return {
        "schema_version": 1,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "cases": cases,
        "kernel_complexity": "O(groups * (nodes * max_out_degree + links))",
        "full_graph_traversal_confirmed_by_code": True,
        "parallel_cases": parallel_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run()
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
