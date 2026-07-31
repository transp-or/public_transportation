"""Bounded local pilot for sparse and matrix-free fixed-routing estimation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.domain import Scenario, read_fixed_demand_csv
from public_transportation.inference.assignment_adapter import (
    build_assignment_inputs,
    prepare_fixed_routing,
)
from public_transportation.inference.compact_od_assignment_layout import (
    build_compact_od_assignment_layout,
)
from public_transportation.inference.fixed_routing_linear_problem import (
    FixedRoutingLinearProblem,
    FixedRoutingLinearProvenance,
)
from public_transportation.inference.fixed_routing_linear_backend import (
    SparseOperatorSelectionConfig,
    prepare_fixed_routing_linear_measurement_backend,
)
from public_transportation.inference.fixed_routing_sharded_builder import (
    ShardedConstructionConfig,
    load_complete_sharded_fixed_routing_cache,
    plan_sharded_fixed_routing_operator,
    prepare_sharded_fixed_routing_measurement_operator,
)
from public_transportation.inference.fixed_routing_origin_support import (
    OriginSupportConfig,
    analyze_fixed_routing_origin_support,
)
from public_transportation.inference.sharded_sparse_operator import (
    ShardedSparseLinearOperator,
)
from public_transportation.inference.fixed_routing_linear_regularization import (
    scaled_ridge_to_prior,
)
from public_transportation.inference.fixed_routing_linear_scalable_quality import (
    ScalableQualityConfig,
    analyze_linear_estimate_quality_scalable,
)
from public_transportation.inference.fixed_routing_linear_trf_solver import (
    TRFLSMRConfig,
    solve_trf_lsmr,
)
from public_transportation.inference.fixed_routing_measurement_operator import (
    assignment_inputs_fingerprint,
    measurement_mapping_fingerprint,
)
from public_transportation.inference.od_parameter_layout import (
    build_od_parameter_layout,
)
from public_transportation.measurement import (
    build_mapping_spec_strict,
    read_measurements_csv,
)

ROOT = Path(__file__).resolve().parents[1]
GENEVA = ROOT / "docs/source/examples/geneva_gtfs"
NETWORK_FILES = (
    "metadata.json",
    "stops.csv",
    "lines.csv",
    "trips.csv",
    "stop_times.csv",
    "time_bins.csv",
)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _scenario_copy(source: Path, demand_path: Path) -> tempfile.TemporaryDirectory[str]:
    temporary = tempfile.TemporaryDirectory(prefix="pt-linear-pilot-")
    destination = Path(temporary.name)
    for name in NETWORK_FILES:
        shutil.copy2(source / name, destination / name)
    shutil.copy2(demand_path, destination / "demand.csv")
    return temporary


def run_pilot(args: argparse.Namespace) -> dict[str, object]:
    total_start = perf_counter()
    phases: dict[str, float] = {}
    temporary = _scenario_copy(args.scenario_folder, args.demand)
    try:
        started = perf_counter()
        scenario = Scenario.from_folder(Path(temporary.name), strict=True)
        phases["scenario_loading_seconds"] = perf_counter() - started

        started = perf_counter()
        fixed = read_fixed_demand_csv(args.fixed_demand, scenario=scenario)
        layout = build_od_parameter_layout(scenario=scenario, fixed_demand=fixed)
        compact = build_compact_od_assignment_layout(parameter_layout=layout)
        phases["od_layout_seconds"] = perf_counter() - started

        started = perf_counter()
        artifacts = prepare_assignment(
            scenario=scenario,
            config=AssignmentConfig(),
            cache_policy="off",
        )
        inputs = build_assignment_inputs(artifacts=artifacts, compact_layout=compact)
        phases["assignment_preparation_seconds"] = perf_counter() - started

        started = perf_counter()
        id_manager = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)
        mapping = build_mapping_spec_strict(
            id_manager=id_manager,
            table=read_measurements_csv(args.measurements),
            include_link_lists_for_report=False,
        )
        phases["measurement_mapping_seconds"] = perf_counter() - started

        dimensions = {
            "nodes": int(artifacts.graph.num_nodes),
            "links": int(artifacts.graph.num_links),
            "destination_groups": int(inputs.group_dest_node.shape[0]),
            "active_od": compact.num_active,
            "free_od": compact.num_free,
            "fixed_positive_od": len(compact.fixed_compact_indices),
            "measurements": mapping.spec.num_measurements,
            "logical_operator_entries": compact.num_free
            * mapping.spec.num_measurements,
        }
        if args.preflight_only and args.operator_mode != "sharded":
            return {
                "schema_version": 1,
                "mode": "preflight",
                "dimensions": dimensions,
                "phases": phases,
                "peak_rss_bytes": _peak_rss_bytes(),
                "total_seconds": perf_counter() - total_start,
            }

        if args.operator_mode == "sharded":
            sharded_config = ShardedConstructionConfig(
                od_chunk_size=args.operator_chunk_size,
                measurement_block_size=args.measurement_block_size,
                worker_memory_budget_bytes=(
                    args.memory_budget_bytes
                    if args.memory_budget_bytes is not None
                    else 512 * 1024 * 1024
                ),
                zero_tolerance=args.zero_tolerance,
                compressed_shards=args.compressed_shards,
                origin_support_chunk_size=args.origin_support_chunk_size,
                support_edge_block_size=args.support_edge_block_size,
                target_nonzeros_per_storage_shard=args.target_nonzeros_per_storage_shard,
                maximum_nonzeros_per_storage_shard=args.maximum_nonzeros_per_storage_shard,
                maximum_patterns_per_storage_shard=args.maximum_patterns_per_storage_shard,
                maximum_storage_shards=args.maximum_storage_shards,
                manifest_checkpoint_shards=args.manifest_checkpoint_shards,
            )
            started = perf_counter()
            inputs_fingerprint = assignment_inputs_fingerprint(inputs)
            phases["assignment_input_fingerprint_seconds"] = perf_counter() - started
            built = None
            if not args.preflight_only:
                started = perf_counter()
                built = load_complete_sharded_fixed_routing_cache(
                    directory=args.operator_cache,
                    inputs=inputs,
                    spec=mapping.spec,
                    compact_layout=compact,
                    assignment_fingerprint=inputs_fingerprint,
                    od_layout_fingerprint=layout.fingerprint,
                    theta=args.theta,
                    config=sharded_config,
                    assignment_inputs_fingerprint_value=inputs_fingerprint,
                )
                phases["complete_cache_probe_seconds"] = perf_counter() - started
            started = perf_counter()
            routing = (
                None
                if built is not None and not args.analyze_origin_support
                else prepare_fixed_routing(inputs=inputs, theta=args.theta)
            )
            phases["fixed_routing_preparation_seconds"] = perf_counter() - started
            origin_support_metrics = None
            if args.analyze_origin_support:
                assert routing is not None
                analyzed = analyze_fixed_routing_origin_support(
                    inputs=inputs,
                    routing=routing,
                    spec=mapping.spec,
                    compact_layout=compact,
                    config=OriginSupportConfig(
                        origin_chunk_size=args.origin_support_chunk_size,
                        worker_memory_budget_bytes=(
                            args.memory_budget_bytes
                            if args.memory_budget_bytes is not None
                            else 512 * 1024 * 1024
                        ),
                        materialize=False,
                    ),
                )
                origin_support_metrics = asdict(analyzed.metrics)
                phases["origin_support_analysis_seconds"] = (
                    analyzed.metrics.support_discovery_seconds
                )
            if args.preflight_only:
                assert routing is not None
                plan, _ = plan_sharded_fixed_routing_operator(
                    inputs=inputs,
                    routing=routing,
                    spec=mapping.spec,
                    compact_layout=compact,
                    config=sharded_config,
                )
                plan_payload = asdict(plan)
                plan_payload.pop("expected_shards")
                plan_payload.pop("storage_shards")
                return {
                    "schema_version": 2,
                    "mode": "sharded_preflight",
                    "dimensions": dimensions,
                    "sharded_plan": plan_payload,
                    "origin_support_metrics": origin_support_metrics,
                    "phases": phases,
                    "peak_rss_bytes": _peak_rss_bytes(),
                    "total_seconds": perf_counter() - total_start,
                }
            if built is None:
                assert routing is not None
                built = prepare_sharded_fixed_routing_measurement_operator(
                    directory=args.operator_cache,
                    inputs=inputs,
                    routing=routing,
                    spec=mapping.spec,
                    compact_layout=compact,
                    assignment_fingerprint=inputs_fingerprint,
                    od_layout_fingerprint=layout.fingerprint,
                    config=sharded_config,
                )
            operator = ShardedSparseLinearOperator(
                args.operator_cache,
                max_cached_shards=args.max_cached_shards,
                memory_budget_bytes=args.memory_budget_bytes,
            )
            fixed_offset = operator.fixed_measurement_offset
            operator_selection = {
                "requested_mode": "sharded",
                "selected_mode": "sharded",
                "reason": "explicit bounded sharded benchmark",
            }
            operator_metrics = {
                "cache_complete": built.manifest.complete,
                "reused_shards": built.reused_shards,
                "rebuilt_shards": built.rebuilt_shards,
                "rejected_shards": built.rejected_shards,
                "aggregate_nonzeros": built.manifest.aggregate_nonzeros,
                "num_shards": built.plan.num_shards,
                "candidate_entries": built.plan.candidate_entries,
                "support_discovery_seconds": built.support_discovery_seconds,
                "lowering_seconds": built.lowering_seconds,
                "compilation_seconds": built.compilation_seconds,
                "dispatch_seconds": built.dispatch_seconds,
                "synchronization_seconds": built.synchronization_seconds,
                "transfer_seconds": built.transfer_seconds,
                "zero_filtering_seconds": built.zero_filtering_seconds,
                "shard_persistence_seconds": built.shard_persistence_seconds,
                "manifest_seconds": built.manifest_seconds,
                "manifest_write_count": built.manifest_write_count,
                "cumulative_manifest_bytes": built.cumulative_manifest_bytes,
                "recovery_scan_seconds": built.recovery_scan_seconds,
                "finalization_seconds": built.finalization_seconds,
                "total_preparation_seconds": built.total_seconds,
                "construction_batches": built.construction_batches,
                "dispatch_count": built.dispatch_count,
                "synchronization_count": built.synchronization_count,
                "support_edge_blocks": built.support_edge_blocks,
                "origins_per_dispatch": built.origins_per_dispatch,
                "supported_edges_per_dispatch": built.supported_edges_per_dispatch,
                "output_values_per_dispatch": built.output_values_per_dispatch,
                "dispatch_time_quantiles": built.dispatch_time_quantiles,
                "synchronization_time_quantiles": built.synchronization_time_quantiles,
                "group_timing_seconds": built.group_timing_seconds,
                "padded_buffer_allocations": built.padded_buffer_allocations,
                "routing_array_dispatch_uses": built.routing_array_dispatch_uses,
                "origin_support_metrics": origin_support_metrics,
            }
            phases["operator_total_preparation_seconds"] = built.total_seconds
        else:
            backend = prepare_fixed_routing_linear_measurement_backend(
            inputs=inputs,
            spec=mapping.spec,
            compact_layout=compact,
            theta=args.theta,
            routing_factory=lambda: prepare_fixed_routing(
                inputs=inputs, theta=args.theta
            ),
            assignment_fingerprint=assignment_inputs_fingerprint(inputs),
            od_layout_fingerprint=layout.fingerprint,
            cache_directory=args.operator_cache,
            config=SparseOperatorSelectionConfig(
                mode=args.operator_mode,
                memory_budget_bytes=args.memory_budget_bytes,
                estimated_density=args.estimated_density,
                expected_matvec_calls=(
                    args.product_repeats + args.solver_iterations * args.lsmr_iterations
                ),
                expected_rmatvec_calls=(
                    args.product_repeats + args.solver_iterations * args.lsmr_iterations
                ),
                estimated_construction_seconds=args.estimated_construction_seconds,
                matrix_free_product_seconds=args.matrix_free_product_seconds,
                sparse_product_seconds=args.sparse_product_seconds,
                zero_tolerance=args.zero_tolerance,
                chunk_size=args.operator_chunk_size,
            ),
            )
            phases.update(
                {
                    "fixed_routing_preparation_seconds": backend.metrics.fixed_routing_preparation_seconds,
                    "operator_cache_lookup_seconds": backend.metrics.cache_lookup_seconds,
                    "operator_cache_load_seconds": backend.metrics.cache_load_seconds,
                    "operator_construction_seconds": backend.metrics.operator_construction_seconds,
                    "operator_cache_persistence_seconds": backend.metrics.cache_persistence_seconds,
                    "operator_cpu_sparse_conversion_seconds": backend.metrics.device_transfer_and_sparse_conversion_seconds,
                    "operator_total_preparation_seconds": backend.metrics.total_preparation_seconds,
                }
            )
            operator = backend.operator
            fixed_offset = backend.fixed_measurement_offset
            operator_selection = asdict(backend.selection)
            operator_metrics = asdict(backend.metrics)
        prior = np.asarray(layout.free_baseline_values, dtype=float)
        scales = np.maximum(prior, 1.0)
        problem = FixedRoutingLinearProblem(
            measurement_operator=operator,
            fixed_measurement_offset=fixed_offset,
            observations=np.asarray(mapping.y_obs),
            observation_weights=np.ones(mapping.spec.num_measurements),
            prior_demand=prior,
            lower_bounds=np.zeros(layout.num_free),
            upper_bounds=np.full(layout.num_free, np.inf),
            provenance=FixedRoutingLinearProvenance(
                od_layout_fingerprint=layout.fingerprint,
                assignment_fingerprint=assignment_inputs_fingerprint(inputs),
                mapping_fingerprint=measurement_mapping_fingerprint(mapping.spec),
                routing_parameter=args.theta,
            ),
            regularization_selection="configured",
            regularization_blocks=(
                scaled_ridge_to_prior(
                    prior,
                    scales,
                    strength=args.regularization_strength,
                ),
            ),
            variable_scales=scales,
            free_od_indices=np.asarray(layout.free_od_indices),
        )

        forward_samples = []
        transpose_samples = []
        predictions = []
        cotangent = np.ones(problem.num_measurements)
        for _ in range(args.product_repeats):
            started = perf_counter()
            prediction = problem.measurement_operator.matvec(prior)
            forward_samples.append(perf_counter() - started)
            predictions.append(prediction)
            started = perf_counter()
            problem.measurement_operator.rmatvec(cotangent)
            transpose_samples.append(perf_counter() - started)
        reproducibility_error = max(
            float(np.max(np.abs(value - predictions[0]), initial=0.0))
            for value in predictions[1:]
        ) if len(predictions) > 1 else 0.0

        solver = None
        quality = None
        if args.solver_iterations > 0:
            solved = solve_trf_lsmr(
                problem,
                config=TRFLSMRConfig(
                    tolerance=args.solver_tolerance,
                    lsmr_tolerance=args.solver_tolerance,
                    max_iterations=args.solver_iterations,
                    lsmr_max_iterations=args.lsmr_iterations,
                    diagonal_preconditioner=args.diagonal_preconditioner,
                    success_policy=args.success_policy,
                    kkt_tolerance=args.kkt_tolerance,
                ),
            )
            solver = {
                "success": solved.success,
                "status": solved.status,
                "message": solved.message,
                "stopping_condition": solved.stopping_condition,
                "iterations": solved.iterations,
                "elapsed_seconds": solved.elapsed_seconds,
                "objective": solved.evaluation.objective,
                "data_objective": solved.evaluation.data_fit.objective,
                "projected_gradient_inf_norm": solved.kkt.projected_gradient_inf_norm,
                "feasibility_inf_norm": solved.kkt.feasibility_inf_norm,
                "matvec_count": solved.matvec_count,
                "rmatvec_count": solved.rmatvec_count,
                "preconditioner_seconds": solved.preconditioner_seconds,
                "preparation_matvec_count": solved.preparation_matvec_count,
                "final_matvec_count": solved.final_matvec_count,
                "final_rmatvec_count": solved.final_rmatvec_count,
            }
            if args.quality_samples > 0:
                diagnosed = analyze_linear_estimate_quality_scalable(
                    problem,
                    solved.demand,
                    config=ScalableQualityConfig(
                        smallest_singular_values=args.smallest_singular_values,
                        spectral_max_iterations=args.spectral_iterations,
                        resolution_samples=args.quality_samples,
                        linear_solve_max_iterations=args.quality_linear_iterations,
                        random_seed=args.random_seed,
                    ),
                )
                quality = {
                    "spectral_converged": diagnosed.spectral_converged,
                    "spectral_message": diagnosed.spectral_message,
                    "largest_singular_value_estimate": diagnosed.largest_singular_value_estimate,
                    "smallest_singular_value_estimates": diagnosed.smallest_singular_value_estimates.tolist(),
                    "estimated_rank_upper_bound": diagnosed.estimated_rank_upper_bound,
                    "estimated_nullity_lower_bound": diagnosed.estimated_nullity_lower_bound,
                    "condition_estimate": diagnosed.condition_estimate,
                    "resolution_samples": diagnosed.resolution_samples,
                    "resolution_converged_samples": diagnosed.resolution_converged_samples,
                    "effective_data_degrees_of_freedom_estimate": diagnosed.effective_data_degrees_of_freedom_estimate,
                    "effective_data_degrees_of_freedom_standard_error": diagnosed.effective_data_degrees_of_freedom_standard_error,
                }

        if isinstance(operator, ShardedSparseLinearOperator):
            operator_metrics.update(
                {
                    "loading_seconds": operator.loading_seconds,
                    "loading_policy": operator.loading_policy,
                    "file_open_count": operator.file_open_count,
                    "bytes_read": operator.bytes_read,
                    "shard_load_count": operator.shard_load_count,
                    "shard_cache_hit_count": operator.shard_cache_hit_count,
                    "shard_eviction_count": operator.shard_eviction_count,
                    "sparse_matrix_calls": operator.sparse_matrix_calls,
                    "uses_merged_operator": operator.uses_merged_operator,
                    "merge_seconds": operator.merge_seconds,
                    "merged_csr_seconds": operator.merged_csr_seconds,
                    "merged_transpose_seconds": operator.merged_transpose_seconds,
                    "merged_storage_bytes": operator.merged_storage_bytes,
                }
            )
        return {
            "schema_version": 1,
            "mode": "fixed_routing_linear_pilot",
            "platform": {
                "system": platform.platform(),
                "jax_backend": jax.default_backend(),
                "jax_device": str(jax.devices()[0]),
            },
            "configuration": {
                "theta": args.theta,
                "regularization_strength": args.regularization_strength,
                "product_repeats": args.product_repeats,
                "solver_iterations": args.solver_iterations,
                "lsmr_iterations": args.lsmr_iterations,
                "quality_samples": args.quality_samples,
                "random_seed": args.random_seed,
                "operator_mode": args.operator_mode,
                "memory_budget_bytes": args.memory_budget_bytes,
                "estimated_density": args.estimated_density,
                "zero_tolerance": args.zero_tolerance,
                "diagonal_preconditioner": args.diagonal_preconditioner,
                "success_policy": args.success_policy,
                "kkt_tolerance": args.kkt_tolerance,
                "estimated_construction_seconds": args.estimated_construction_seconds,
                "matrix_free_product_seconds": args.matrix_free_product_seconds,
                "sparse_product_seconds": args.sparse_product_seconds,
            },
            "operator_selection": operator_selection,
            "operator_metrics": operator_metrics,
            "dimensions": dimensions,
            "phases": phases,
            "products": {
                "forward_seconds": forward_samples,
                "transpose_seconds": transpose_samples,
                "forward_median_seconds": float(np.median(forward_samples)),
                "transpose_median_seconds": float(np.median(transpose_samples)),
                "reproducibility_max_abs_difference": reproducibility_error,
            },
            "solver": solver,
            "quality": quality,
            "peak_rss_bytes": _peak_rss_bytes(),
            "total_seconds": perf_counter() - total_start,
        }
    finally:
        temporary.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-folder", type=Path, default=GENEVA / "data")
    parser.add_argument(
        "--demand",
        type=Path,
        default=GENEVA / "pre_processing/results/demand.csv",
    )
    parser.add_argument(
        "--fixed-demand", type=Path, default=GENEVA / "data/fixed_demand.csv"
    )
    parser.add_argument(
        "--measurements",
        type=Path,
        default=GENEVA / "pre_processing/results/measurements_boarding_alighting.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/fixed_routing_linear_pilot_geneva.json",
    )
    parser.add_argument("--theta", type=float, default=5.0)
    parser.add_argument("--regularization-strength", type=float, default=1.0)
    parser.add_argument("--product-repeats", type=int, default=2)
    parser.add_argument("--solver-iterations", type=int, default=2)
    parser.add_argument("--lsmr-iterations", type=int, default=10)
    parser.add_argument("--solver-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--quality-samples", type=int, default=2)
    parser.add_argument("--smallest-singular-values", type=int, default=1)
    parser.add_argument("--spectral-iterations", type=int, default=10)
    parser.add_argument("--quality-linear-iterations", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=1729)
    parser.add_argument(
        "--operator-mode",
        choices=("matrix_free", "sparse", "auto", "sharded"),
        default="auto",
    )
    parser.add_argument(
        "--operator-cache", type=Path, default=ROOT / ".cache/fixed-routing-operators"
    )
    parser.add_argument("--memory-budget-bytes", type=int)
    parser.add_argument("--estimated-density", type=float, default=0.1)
    parser.add_argument("--zero-tolerance", type=float, default=0.0)
    parser.add_argument("--operator-chunk-size", type=int, default=128)
    parser.add_argument("--measurement-block-size", type=int, default=512)
    parser.add_argument("--max-cached-shards", type=int)
    parser.add_argument("--compressed-shards", action="store_true")
    parser.add_argument("--analyze-origin-support", action="store_true")
    parser.add_argument("--origin-support-chunk-size", type=int, default=64)
    parser.add_argument("--support-edge-block-size", type=int, default=2048)
    parser.add_argument("--target-nonzeros-per-storage-shard", type=int, default=2048)
    parser.add_argument("--maximum-nonzeros-per-storage-shard", type=int, default=8192)
    parser.add_argument("--maximum-patterns-per-storage-shard", type=int, default=256)
    parser.add_argument("--maximum-storage-shards", type=int, default=256)
    parser.add_argument("--manifest-checkpoint-shards", type=int, default=16)
    parser.add_argument("--estimated-construction-seconds", type=float)
    parser.add_argument("--matrix-free-product-seconds", type=float)
    parser.add_argument("--sparse-product-seconds", type=float, default=0.0)
    parser.add_argument("--diagonal-preconditioner", action="store_true")
    parser.add_argument(
        "--success-policy", choices=("scipy", "kkt", "both"), default="scipy"
    )
    parser.add_argument("--kkt-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if (
        args.theta <= 0.0
        or args.regularization_strength < 0.0
        or args.product_repeats <= 0
        or args.solver_iterations < 0
        or args.lsmr_iterations <= 0
        or args.solver_tolerance <= 0.0
        or args.quality_samples < 0
        or args.smallest_singular_values <= 0
        or args.spectral_iterations <= 0
        or args.quality_linear_iterations <= 0
        or args.memory_budget_bytes is not None and args.memory_budget_bytes <= 0
        or not 0.0 <= args.estimated_density <= 1.0
        or args.zero_tolerance < 0.0
        or args.operator_chunk_size <= 0
        or args.measurement_block_size <= 0
        or args.max_cached_shards is not None and args.max_cached_shards <= 0
        or args.origin_support_chunk_size <= 0
        or args.support_edge_block_size <= 0
        or args.target_nonzeros_per_storage_shard <= 0
        or args.maximum_nonzeros_per_storage_shard
        < args.target_nonzeros_per_storage_shard
        or args.maximum_patterns_per_storage_shard <= 0
        or args.maximum_storage_shards <= 0
        or args.manifest_checkpoint_shards <= 0
        or args.estimated_construction_seconds is not None
        and args.estimated_construction_seconds < 0.0
        or args.matrix_free_product_seconds is not None
        and args.matrix_free_product_seconds < 0.0
        or args.sparse_product_seconds < 0.0
        or args.kkt_tolerance <= 0.0
    ):
        parser.error("pilot counts and tolerances are outside their valid ranges")
    return args


def main() -> None:
    args = parse_args()
    report = run_pilot(args)
    _atomic_json(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
