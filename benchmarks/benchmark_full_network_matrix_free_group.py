"""Bounded matrix-free fixed-routing benchmark for one full-network group."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
TPG = ROOT.parent / "public_transport_TPG"
DEFAULT_SCENARIO = TPG / "processed_data_for_models/full_network"


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _storage_bytes(tree) -> int:
    import jax
    import numpy as np

    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            total += int(np.prod(leaf.shape, dtype=np.int64)) * np.dtype(
                leaf.dtype
            ).itemsize
    return total


def _compile(kernel, *arguments):
    import jax

    started = perf_counter()
    traced = kernel.trace(*arguments)
    tracing = perf_counter() - started
    started = perf_counter()
    lowered = traced.lower()
    lowering = perf_counter() - started
    started = perf_counter()
    compiled = lowered.compile()
    compilation = perf_counter() - started
    started = perf_counter()
    first = compiled(*arguments)
    jax.block_until_ready(first)
    first_execution = perf_counter() - started
    return compiled, {
        "tracing_seconds": tracing,
        "lowering_seconds": lowering,
        "compilation_seconds": compilation,
        "first_execution_seconds": first_execution,
        # Rendering compiler IR can itself allocate gigabytes for this graph.
        "lowered_text_bytes": None,
    }


def _warm(compiled, arguments, evaluations):
    import jax
    import numpy as np

    samples = []
    for _ in range(evaluations):
        started = perf_counter()
        result = compiled(*arguments)
        jax.block_until_ready(result)
        samples.append(perf_counter() - started)
    return {
        "samples_seconds": samples,
        "median_seconds": float(np.median(samples)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-folder", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--group-index", type=int, default=140)
    parser.add_argument("--theta", type=float, default=5.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--nb-dispersion", type=float, default=10.0)
    parser.add_argument("--z-bound", type=float, default=6.0)
    parser.add_argument("--sigma-z", type=float, default=5.0)
    parser.add_argument("--memory-ceiling-gib", type=float, default=12.0)
    parser.add_argument("--warm-evaluations", type=int, default=5)
    parser.add_argument(
        "--loader-mode",
        choices=("ordinary", "rematerialized", "custom_adjoint"),
        default="ordinary",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.memory_ceiling_gib <= 0 or args.warm_evaluations <= 0:
        parser.error("memory ceiling and warm evaluations must be positive")
    if args.theta <= 0 or args.nb_dispersion <= 0 or args.sigma_z <= 0:
        parser.error("theta, NB dispersion and sigma_z must be positive")

    process_started = perf_counter()
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    import jax
    import jax.numpy as jnp
    import jaxlib
    import numpy as np
    from public_transportation.assignment import AssignmentConfig
    from public_transportation.assignment.assign import prepare_assignment
    from public_transportation.assignment.id_manager import AssignmentIDManager
    from public_transportation.domain import Scenario, read_fixed_demand_csv
    from public_transportation.estimation.common.model_blackbox import _as_scalar
    from public_transportation.inference.assignment_adapter import (
        assign_link_flow_fixed_routing,
        assign_link_flow_fixed_routing_custom_adjoint,
        prepare_fixed_routing,
    )
    from public_transportation.inference.fixed_routing_group import (
        build_single_free_group_assignment,
    )
    from public_transportation.inference.od_parameter_layout import (
        build_od_parameter_layout,
    )
    from public_transportation.inference.parameterization import smooth_bound
    from public_transportation.measurement import (
        build_mapping_spec_strict,
        read_measurements_csv,
    )
    from public_transportation.measurement.likelihood_jax import (
        negbinom_loglikelihood,
        predict_measurements_from_link_flow,
    )

    stages: dict[str, float] = {}
    scenario_folder = args.scenario_folder.resolve()
    started = perf_counter()
    scenario = Scenario.from_folder(scenario_folder, strict=True)
    stages["scenario_loading_and_validation"] = perf_counter() - started
    started = perf_counter()
    fixed = read_fixed_demand_csv(
        scenario_folder / "fixed_demand.csv", scenario=scenario
    )
    layout = build_od_parameter_layout(scenario=scenario, fixed_demand=fixed)
    stages["od_layout"] = perf_counter() - started
    started = perf_counter()
    artifacts = prepare_assignment(
        scenario=scenario, config=AssignmentConfig(), cache_policy="off"
    )
    stages["assignment_preparation"] = perf_counter() - started
    started = perf_counter()
    selected = build_single_free_group_assignment(
        artifacts=artifacts, layout=layout, group_index=args.group_index
    )
    jax.block_until_ready(selected.inputs)
    stages["single_group_extraction"] = perf_counter() - started

    ceiling = int(args.memory_ceiling_gib * 1024**3)
    routing_projection = int(selected.inputs.graph.num_links) * 5
    projected_before_routing = _rss_bytes() + routing_projection
    if projected_before_routing > ceiling:
        raise MemoryError(
            "Single-group routing preflight rejected: current peak RSS plus one "
            f"boolean mask and float32 probability vector is {projected_before_routing} "
            f"bytes, above the {ceiling}-byte ceiling."
        )

    started = perf_counter()
    routing = prepare_fixed_routing(inputs=selected.inputs, theta=args.theta)
    jax.block_until_ready(routing)
    stages["fixed_routing_preparation"] = perf_counter() - started

    started = perf_counter()
    id_manager = AssignmentIDManager.build(
        scenario=scenario, graph=artifacts.graph
    )
    stages["id_manager"] = perf_counter() - started
    started = perf_counter()
    measurement_table = read_measurements_csv(
        scenario_folder / "measurements_boarding_alighting.csv"
    )
    stages["measurement_csv"] = perf_counter() - started
    started = perf_counter()
    mapping = build_mapping_spec_strict(
        id_manager=id_manager,
        table=measurement_table,
        include_link_lists_for_report=False,
    )
    stages["strict_measurement_mapping"] = perf_counter() - started

    spec_measurement = jnp.asarray(mapping.spec.measurement_index, dtype=jnp.int32)
    spec_link = jnp.asarray(mapping.spec.link_index, dtype=jnp.int32)
    y_obs = jnp.asarray(mapping.y_obs, dtype=jnp.float32)
    demand0 = selected.baseline_demand
    rho = jnp.asarray(args.rho, dtype=jnp.float32)
    r_nb = jnp.asarray(args.nb_dispersion, dtype=jnp.float32)
    z_bound = jnp.asarray(args.z_bound, dtype=jnp.float32)
    sigma_z = jnp.asarray(args.sigma_z, dtype=jnp.float32)

    def ordinary_load(demand):
        return assign_link_flow_fixed_routing(
            inputs=selected.inputs, routing=routing, f=demand
        )

    def custom_adjoint_load(demand):
        return assign_link_flow_fixed_routing_custom_adjoint(
            inputs=selected.inputs, routing=routing, f=demand
        )

    if args.loader_mode == "ordinary":
        load = ordinary_load
    elif args.loader_mode == "rematerialized":
        load = jax.checkpoint(ordinary_load)
    else:
        load = custom_adjoint_load

    def aggregate(link_flow):
        return predict_measurements_from_link_flow(
            link_flow,
            spec_num_measurements=mapping.spec.num_measurements,
            spec_measurement_index=spec_measurement,
            spec_link_index=spec_link,
        )

    def likelihood(prediction):
        mu = jnp.maximum(rho * prediction, jnp.asarray(1e-9, prediction.dtype))
        return negbinom_loglikelihood(y_obs=y_obs, mu=mu, r=r_nb)

    normalizer = jnp.log(sigma_z) + 0.5 * jnp.log(
        jnp.asarray(2.0 * np.pi, dtype=jnp.float32)
    )

    def objective(parameter):
        bounded = smooth_bound(parameter, z_bound)
        prediction = aggregate(load(demand0 * jnp.exp(bounded)))
        prior = jnp.sum(-0.5 * jnp.square(parameter / sigma_z) - normalizer)
        return -(_as_scalar(likelihood(prediction)) + prior)

    memory_snapshots = {"before_kernel_compilation": _rss_bytes()}
    parameter0 = jnp.zeros((demand0.shape[0],), dtype=jnp.float32)
    print("Compiling bounded value-and-gradient kernel...", flush=True)
    compiled_value_gradient, value_gradient_compile = _compile(
        jax.jit(jax.value_and_grad(objective)), parameter0
    )
    value_gradient_warm = _warm(
        compiled_value_gradient, (parameter0,), args.warm_evaluations
    )
    objective_value, gradient = compiled_value_gradient(parameter0)
    jax.block_until_ready((objective_value, gradient))
    memory_snapshots["after_value_and_gradient"] = _rss_bytes()

    print("Profiling separated forward phases...", flush=True)
    compiled_loading, loading_compile = _compile(jax.jit(load), demand0)
    loading_warm = _warm(compiled_loading, (demand0,), args.warm_evaluations)
    link_flow = compiled_loading(demand0)
    jax.block_until_ready(link_flow)
    compiled_aggregation, aggregation_compile = _compile(
        jax.jit(aggregate), link_flow
    )
    aggregation_warm = _warm(
        compiled_aggregation, (link_flow,), args.warm_evaluations
    )
    prediction = compiled_aggregation(link_flow)
    jax.block_until_ready(prediction)
    compiled_likelihood, likelihood_compile = _compile(
        jax.jit(likelihood), prediction
    )
    likelihood_warm = _warm(
        compiled_likelihood, (prediction,), args.warm_evaluations
    )
    memory_snapshots["after_separated_forward"] = _rss_bytes()
    print("Profiling separate value-only and gradient-only kernels...", flush=True)
    compiled_value, value_compile = _compile(jax.jit(objective), parameter0)
    value_warm = _warm(compiled_value, (parameter0,), args.warm_evaluations)
    compiled_gradient, gradient_compile = _compile(
        jax.jit(jax.grad(objective)), parameter0
    )
    gradient_warm = _warm(
        compiled_gradient, (parameter0,), args.warm_evaluations
    )
    memory_snapshots["after_all_diagnostic_kernels"] = _rss_bytes()

    enabled_links = int(np.count_nonzero(np.asarray(routing.effective_group_link_mask)))
    mapped_links = np.asarray(mapping.spec.link_index, dtype=np.int64)
    enabled = np.asarray(routing.effective_group_link_mask[0], dtype=bool)
    local_measurements = np.unique(
        np.asarray(mapping.spec.measurement_index)[enabled[mapped_links]]
    )
    report = {
        "schema_version": 1,
        "mode": "bounded_single_destination_matrix_free_reference",
        "loader_mode": args.loader_mode,
        "safety": {
            "selected_groups": 1,
            "global_fixed_routing_constructed": False,
            "measurement_operator_constructed": False,
            "jacobian_constructed": False,
            "map_optimization_started": False,
            "memory_ceiling_bytes": ceiling,
            "routing_preflight_bytes": routing_projection,
            "observed_memory_ceiling_exceeded": _rss_bytes() > ceiling,
        },
        "platform": {
            "system": platform.platform(),
            "architecture": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
        },
        "sample": {
            "original_group_index": args.group_index,
            "free_cells": int(demand0.shape[0]),
            "destination_node": int(np.asarray(selected.inputs.group_dest_node[0])),
            "enabled_links": enabled_links,
            "local_measurements": int(local_measurements.size),
            "total_measurements": int(mapping.spec.num_measurements),
        },
        "dimensions": {
            "nodes": int(selected.inputs.graph.num_nodes),
            "links": int(selected.inputs.graph.num_links),
            "full_groups": int(artifacts.od_groups.group_dest_node.shape[0]),
            "candidate_od": layout.num_od_total,
            "free_od": layout.num_free,
        },
        "preparation_seconds": stages,
        "persistent_array_bytes": {
            "assignment_artifacts": int(artifacts.cache_metrics.logical_bytes),
            "single_group_inputs": _storage_bytes(selected.inputs),
            "single_group_routing": _storage_bytes(routing),
            "measurement_mapping": int(
                np.asarray(mapping.spec.measurement_index).nbytes
                + np.asarray(mapping.spec.link_index).nbytes
                + np.asarray(mapping.y_obs).nbytes
            ),
        },
        "kernels": {
            "flow_loading": {"compile": loading_compile, "warm": loading_warm},
            "measurement_aggregation": {
                "compile": aggregation_compile,
                "warm": aggregation_warm,
            },
            "likelihood": {"compile": likelihood_compile, "warm": likelihood_warm},
            "value_only": {"compile": value_compile, "warm": value_warm},
            "gradient_only": {
                "compile": gradient_compile,
                "warm": gradient_warm,
            },
            "value_and_gradient": {
                "compile": value_gradient_compile,
                "warm": value_gradient_warm,
            },
        },
        "numerical": {
            "objective_at_baseline": float(np.asarray(objective_value)),
            "gradient_norm": float(np.linalg.norm(np.asarray(gradient))),
            "prediction_nonzero": int(np.count_nonzero(np.asarray(prediction))),
            "prediction_sum": float(np.asarray(prediction).sum()),
        },
        "memory": {
            "peak_rss_bytes": _rss_bytes(),
            "peak_rss_snapshots": memory_snapshots,
            "device_memory_stats": jax.devices()[0].memory_stats(),
        },
        "total_process_seconds": perf_counter() - process_started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
