"""Run bounded fixed-routing support preflight without global support planning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.assignment import AssignmentConfig
from public_transportation.assignment.assign import prepare_assignment
from public_transportation.assignment.id_manager import AssignmentIDManager
from public_transportation.domain import Scenario, read_fixed_demand_csv
from public_transportation.inference.assignment_adapter import (
    build_assignment_inputs,
    prepare_fixed_routing,
)
from public_transportation.inference.block_coordinate._canonical import (
    canonical_json,
    fingerprint,
)
from public_transportation.inference.block_coordinate.config import BlockSizingConfig
from public_transportation.inference.block_coordinate.blocks import ODBlock
from public_transportation.inference.block_coordinate.fixed_routing_selected_block_builder import (
    FixedRoutingSelectedBlockBuilder,
    SelectedBlockBuilderConfig,
    SelectedBlockBuilderProvenance,
    SelectedBlockDiagnosticStop,
)
from public_transportation.inference.block_coordinate.partition import (
    ODBlockPartition,
    partition_assignment_od_blocks,
)
from public_transportation.inference.block_coordinate.selected_blocks import (
    select_representative_block_ids,
)
from public_transportation.inference.block_coordinate.support_preflight import (
    SupportPreflightBudget,
    SupportPreflightConfig,
    SupportPreflightFingerprints,
    SupportPreflightMode,
    authorize_block_coordinate_pilot,
    run_support_preflight,
)
from public_transportation.inference.compact_od_assignment_layout import (
    CompactODAssignmentLayout,
    build_compact_od_assignment_layout,
)
from public_transportation.inference.fixed_routing_measurement_operator import (
    assignment_inputs_fingerprint,
    measurement_mapping_fingerprint,
)
from public_transportation.inference.fixed_routing_origin_support import (
    OriginSupportConfig,
    analyze_fixed_routing_origin_support,
)
from public_transportation.inference.od_parameter_layout import (
    build_od_parameter_layout,
)
from public_transportation.measurement import (
    build_mapping_spec_strict,
    read_measurements_csv,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs/source/examples/simple_example_02"
NETWORK_FILES = (
    "metadata.json",
    "stops.csv",
    "lines.csv",
    "trips.csv",
    "stop_times.csv",
    "time_bins.csv",
)


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_payload(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(payload) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _scenario_copy(source: Path, demand: Path) -> tempfile.TemporaryDirectory[str]:
    temporary = tempfile.TemporaryDirectory(prefix="support-preflight-")
    destination = Path(temporary.name)
    for name in NETWORK_FILES:
        shutil.copy2(source / name, destination / name)
    shutil.copy2(demand, destination / "demand.csv")
    return temporary


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.stop_after is not None and not args.benchmark_od_batching:
        raise ValueError("--stop-after requires --benchmark-od-batching.")
    if args.stop_after is not None and args.phase_progress_file is None:
        raise ValueError("--stop-after requires --phase-progress-file.")
    total_started = perf_counter()
    phases: dict[str, float] = {}
    temporary = _scenario_copy(args.scenario_folder, args.demand)
    try:
        started = perf_counter()
        scenario = Scenario.from_folder(temporary.name, strict=True)
        phases["scenario_seconds"] = perf_counter() - started
        started = perf_counter()
        fixed = read_fixed_demand_csv(args.fixed_demand, scenario=scenario)
        layout = build_od_parameter_layout(scenario=scenario, fixed_demand=fixed)
        compact = build_compact_od_assignment_layout(parameter_layout=layout)
        phases["layout_seconds"] = perf_counter() - started
        started = perf_counter()
        artifacts = prepare_assignment(
            scenario=scenario, config=AssignmentConfig(), cache_policy="off"
        )
        inputs = build_assignment_inputs(artifacts=artifacts, compact_layout=compact)
        phases["assignment_seconds"] = perf_counter() - started
        started = perf_counter()
        manager = AssignmentIDManager.build(scenario=scenario, graph=artifacts.graph)
        mapping = build_mapping_spec_strict(
            id_manager=manager,
            table=read_measurements_csv(args.measurements),
            include_link_lists_for_report=False,
        )
        phases["measurement_mapping_seconds"] = perf_counter() - started
        started = perf_counter()
        partition = partition_assignment_od_blocks(
            inputs=inputs,
            parameter_layout=layout,
            compact_layout=compact,
            sizing=BlockSizingConfig(
                mode="explicit",
                maximum_free_variables_per_block=args.maximum_variables_per_block,
            ),
        )
        phases["partition_seconds"] = perf_counter() - started
        dimensions = {
            "nodes": int(artifacts.graph.num_nodes),
            "links": int(artifacts.graph.num_links),
            "destination_groups": int(inputs.group_dest_node.shape[0]),
            "candidate_od": layout.num_od_total,
            "active_od": compact.num_active,
            "free_od": compact.num_free,
            "frozen_zero_od": len(layout.fixed_zero_indices),
            "frozen_positive_od": len(layout.fixed_positive_indices),
            "measurements": mapping.spec.num_measurements,
            "blocks": partition.num_blocks,
        }
        budget = SupportPreflightBudget(
            maximum_elapsed_seconds=args.maximum_elapsed_seconds,
            maximum_process_rss_bytes=args.maximum_rss_bytes,
            maximum_temporary_bytes=args.maximum_temporary_bytes,
            maximum_retained_support_bytes=args.maximum_retained_bytes,
            maximum_support_rows_per_block=args.maximum_support_rows_per_block,
            maximum_nonzeros_per_block=args.maximum_nonzeros_per_block,
            maximum_block_operator_bytes=args.maximum_block_operator_bytes,
        )
        mode = SupportPreflightMode(args.mode.replace("-", "_"))
        config = SupportPreflightConfig(
            mode=mode,
            destination_group_ids=(
                None
                if args.destination_group is None
                else tuple(args.destination_group)
            ),
            sample_count=args.sample_count,
            sampling_seed=args.sampling_seed,
            origin_chunk_size=args.origin_chunk_size,
            checkpoint_directory=args.checkpoint_directory,
            checkpoint_interval_groups=args.checkpoint_interval_groups,
            checkpoint_interval_seconds=args.checkpoint_interval_seconds,
            progress_interval_groups=args.progress_interval_groups,
            construction_workers=args.construction_workers,
            threads_per_worker=args.threads_per_worker,
            persist_selected_block_support=args.persist_selected_block_support,
            authorize_exact_materialized_plan=args.authorize_exact_materialized_plan,
            budget=budget,
        )
        fingerprints = SupportPreflightFingerprints(
            scenario=fingerprint(
                {
                    name: _file_fingerprint(args.scenario_folder / name)
                    for name in NETWORK_FILES
                }
            ),
            assignment_inputs=assignment_inputs_fingerprint(inputs),
            od_layout=layout.fingerprint,
            fixed_demand=_file_fingerprint(args.fixed_demand),
            measurement_mapping=measurement_mapping_fingerprint(mapping.spec),
            routing=fingerprint({"theta": args.theta}),
            partition=partition.fingerprint,
        )
        base = {
            "schema_version": 1,
            "command": "bounded_support_preflight",
            "dimensions": dimensions,
            "phases": phases,
            "configuration": asdict(config),
            "fingerprints": asdict(fingerprints),
        }
        if mode is SupportPreflightMode.EXACT_MATERIALIZED_PLAN:
            logical_bytes = compact.num_free * mapping.spec.num_measurements
            if logical_bytes > budget.maximum_retained_support_bytes:
                raise RuntimeError(
                    "exact materialized support rejected before routing: logical support exceeds retained-state budget"
                )
            started = perf_counter()
            routing = prepare_fixed_routing(inputs=inputs, theta=args.theta)
            phases["routing_seconds"] = perf_counter() - started
            materialized = analyze_fixed_routing_origin_support(
                inputs=inputs,
                routing=routing,
                spec=mapping.spec,
                compact_layout=compact,
                config=OriginSupportConfig(
                    origin_chunk_size=args.origin_chunk_size,
                    worker_memory_budget_bytes=budget.maximum_temporary_bytes,
                    max_materialized_entries=max(1, logical_bytes),
                    materialize=True,
                ),
            )
            phases["support_seconds"] = materialized.metrics.support_discovery_seconds
            report = dict(base)
            report.update(
                {
                    "complete": True,
                    "full_network_coverage": True,
                    "materialized_support_metrics": asdict(materialized.metrics),
                    "pilot_authorization": {
                        "accepted": False,
                        "reasons": [
                            "materialized support is a compatibility benchmark, not bounded pilot evidence"
                        ],
                    },
                    "phases": phases,
                    "total_seconds": perf_counter() - total_started,
                }
            )
            _atomic_payload(args.output, report)
            return report

        def progress(event) -> None:
            payload = dict(base)
            payload["progress"] = asdict(event)
            payload["complete"] = False
            _atomic_payload(args.output, payload)
            print(
                f"support invocation={event.invocation_count} "
                f"{event.processed_groups}/{event.total_groups} groups; "
                f"free={event.processed_free_columns}; rss={event.current_rss_bytes / 1024**3:.2f} GiB; "
                f"retained={event.retained_state_bytes / 1024**2:.2f} MiB; elapsed={event.elapsed_seconds:.1f}s",
                f"invocation_elapsed={event.current_invocation_elapsed_seconds:.1f}s",
                flush=True,
            )

        result = run_support_preflight(
            inputs=inputs,
            theta=args.theta,
            spec=mapping.spec,
            compact_layout=compact,
            partition=partition,
            fingerprints=fingerprints,
            config=config,
            resume=args.resume,
            progress_callback=progress,
        )
        phases["support_seconds"] = result.elapsed_seconds
        selected_block_results: list[dict[str, object]] = []
        od_batch_benchmark: list[dict[str, object]] = []
        if (
            args.construct_representative_blocks
            or args.persist_selected_block_support
            or args.benchmark_od_batching
        ):
            selected_ids = select_representative_block_ids(
                result, explicit_block_ids=args.selected_block
            )
            blocks_by_id = {block.block_id: block for block in partition.blocks}
            builder_config = SelectedBlockBuilderConfig(
                cache_directory=args.block_cache_directory,
                support_directory=args.selected_support_directory,
                od_chunk_size=args.od_chunk_size,
                od_batch_size=args.od_batch_size,
                measurement_chunk_size=args.measurement_chunk_size,
                mapped_edge_chunk_size=args.mapped_edge_chunk_size,
                maximum_variables=args.maximum_variables_per_block,
                maximum_support_rows=args.maximum_support_rows_per_block,
                maximum_nonzeros=args.maximum_nonzeros_per_block,
                maximum_temporary_bytes=args.maximum_temporary_bytes,
                maximum_retained_block_bytes=args.maximum_block_operator_bytes,
                per_worker_memory_ceiling_bytes=args.maximum_temporary_bytes,
                storage_dtype=args.storage_dtype,
            )
            provenance = SelectedBlockBuilderProvenance(
                fingerprints=fingerprints,
                semantic_preflight_fingerprint=config.semantics_fingerprint,
                theta=args.theta,
            )

            def construction_progress(event) -> None:
                print(
                    f"block={event.block_id} od_chunks={event.completed_od_chunks}/"
                    f"{event.total_od_chunks} measurement_chunks="
                    f"{event.completed_measurement_chunks} candidates="
                    f"{event.candidate_entries}",
                    flush=True,
                )

            builder = FixedRoutingSelectedBlockBuilder(
                inputs=inputs,
                spec=mapping.spec,
                compact_layout=compact,
                partition=partition,
                provenance=provenance,
                config=builder_config,
                progress=construction_progress,
            )
            selected_support_started = perf_counter()
            selected_support_artifacts = {
                block_id: builder.prepare_support(blocks_by_id[block_id])
                for block_id in selected_ids
            }
            phases["selected_support_persistence_seconds"] = (
                perf_counter() - selected_support_started
            )
            if args.construct_representative_blocks:
                for block_id in selected_ids:
                    block = blocks_by_id[block_id]
                    cold = builder.build_result(block)
                    local = np.linspace(0.25, 1.25, block.num_free_variables)
                    cotangent = np.linspace(-0.5, 0.5, mapping.spec.num_measurements)
                    started = perf_counter()
                    forward = cold.operator.matvec(local)
                    forward_seconds = perf_counter() - started
                    started = perf_counter()
                    transpose = cold.operator.rmatvec(cotangent)
                    transpose_seconds = perf_counter() - started
                    left = float(np.dot(forward, cotangent))
                    right = float(np.dot(local, transpose))
                    builder.release_all()
                    warm_builder = FixedRoutingSelectedBlockBuilder(
                        inputs=inputs,
                        spec=mapping.spec,
                        compact_layout=compact,
                        partition=partition,
                        provenance=provenance,
                        config=builder_config,
                    )
                    warm = warm_builder.build_result(block)
                    selected_block_results.append(
                        {
                            "block_id": block_id,
                            "support_artifact": {
                                "path": str(selected_support_artifacts[block_id].path),
                                "fingerprint": selected_support_artifacts[
                                    block_id
                                ].fingerprint,
                                "disk_bytes": selected_support_artifacts[
                                    block_id
                                ].disk_bytes,
                            },
                            "cold": {
                                "cache_hit": cold.cache_hit,
                                "support_artifact_load_seconds": cold.support_artifact_load_seconds,
                                "support_discovery_seconds": cold.support_discovery_seconds,
                                "routing_preparation_seconds": cold.routing_preparation_seconds,
                                "numerical_construction_seconds": cold.numerical_construction_seconds,
                                "sparse_assembly_seconds": cold.sparse_assembly_seconds,
                                "persistence_seconds": cold.persistence_seconds,
                                "exact_nonzeros": cold.exact_nonzeros,
                                "support_rows": cold.support_rows,
                                "disk_bytes": cold.disk_bytes,
                                "resident_bytes": cold.resident_bytes,
                                "peak_temporary_bytes": cold.peak_temporary_bytes,
                                "construction_count": cold.construction_count,
                                "reuse_count": cold.reuse_count,
                                "estimate_observed_memory_ratio": cold.estimate_observed_memory_ratio,
                            },
                            "resource_estimate": asdict(cold.estimate),
                            "diagnostics": asdict(cold.diagnostics),
                            "forward_seconds": forward_seconds,
                            "transpose_seconds": transpose_seconds,
                            "adjoint_absolute_error": abs(left - right),
                            "zero_outside_support": bool(
                                np.all(
                                    np.delete(
                                        forward,
                                        np.asarray(
                                            cold.support_artifact.support_rows,
                                            dtype=np.int64,
                                        ),
                                    )
                                    == 0.0
                                )
                            ),
                            "warm_cache_hit": warm.cache_hit,
                            "warm_cache_load_seconds": warm.cache_load_seconds,
                            "warm_identical": bool(
                                np.array_equal(
                                    cast(Any, warm.operator.compact_matrix).toarray(),
                                    cast(Any, cold.operator.compact_matrix).toarray(),
                                )
                            ),
                        }
                    )
                    warm_builder.release_all()
            if args.benchmark_od_batching:
                if args.synthetic_od_columns <= 0:
                    raise ValueError("synthetic_od_columns must be positive.")
                source_group = max(
                    set(
                        group
                        for block in partition.blocks
                        for group in block.destination_group_indices
                    ),
                    key=lambda group: sum(
                        block.num_free_variables
                        for block in partition.blocks
                        if block.destination_group_indices == (group,)
                    ),
                )
                source_active = np.asarray(
                    [
                        active
                        for block in partition.blocks
                        if block.destination_group_indices == (source_group,)
                        for active in block.active_od_indices
                    ],
                    dtype=np.int64,
                )
                synthetic_columns = args.synthetic_od_columns
                repeated_active = np.resize(source_active, synthetic_columns)
                synthetic_origins = np.asarray(inputs.od_origin_node)[repeated_active]
                synthetic_inputs = replace(
                    inputs,
                    od_origin_node=jnp.asarray(
                        synthetic_origins, dtype=inputs.od_origin_node.dtype
                    ),
                )
                coordinates = tuple(range(synthetic_columns))
                synthetic_compact = CompactODAssignmentLayout(
                    num_od_total=synthetic_columns,
                    active_full_indices=coordinates,
                    removed_zero_full_indices=(),
                    full_to_compact=coordinates,
                    free_full_indices=coordinates,
                    free_compact_indices=coordinates,
                    free_baseline_values=tuple(1.0 for _ in coordinates),
                    fixed_compact_indices=(),
                    fixed_compact_values=(),
                )
                benchmark_block = ODBlock(
                    block_id="synthetic-batching-block",
                    free_column_indices=coordinates,
                    active_od_indices=coordinates,
                    destination_group_indices=(source_group,),
                    time_bin_ids=("synthetic",),
                )
                synthetic_partition = ODBlockPartition(
                    blocks=(benchmark_block,),
                    num_free_variables=synthetic_columns,
                )
                synthetic_fingerprints = replace(
                    fingerprints,
                    assignment_inputs=assignment_inputs_fingerprint(synthetic_inputs),
                    od_layout=synthetic_compact.fingerprint,
                    fixed_demand=fingerprint({"synthetic_fixed_demand": "none"}),
                    partition=synthetic_partition.fingerprint,
                )
                synthetic_provenance = SelectedBlockBuilderProvenance(
                    fingerprints=synthetic_fingerprints,
                    semantic_preflight_fingerprint=config.semantics_fingerprint,
                    theta=args.theta,
                )
                reference_matrix = None
                batch_sizes = (
                    (args.od_batch_size,)
                    if args.stop_after is not None
                    else (1, 2, 4, 8, None)
                )
                for batch_size in batch_sizes:
                    label = "auto" if batch_size is None else str(batch_size)
                    batch_config = replace(
                        builder_config,
                        cache_directory=(
                            args.block_cache_directory / f"od-batch-{label}"
                        ),
                        od_batch_size=batch_size,
                        maximum_variables=max(
                            builder_config.maximum_variables, synthetic_columns
                        ),
                        support_directory=(
                            args.selected_support_directory / "synthetic-od-batching"
                            if args.selected_support_directory is not None
                            else args.block_cache_directory
                            / "synthetic-od-batching-support"
                        ),
                        maximum_retained_blocks=0,
                    )
                    batch_builder = FixedRoutingSelectedBlockBuilder(
                        inputs=synthetic_inputs,
                        spec=mapping.spec,
                        compact_layout=synthetic_compact,
                        partition=synthetic_partition,
                        provenance=synthetic_provenance,
                        config=batch_config,
                        progress_file=args.phase_progress_file,
                        durable_progress=args.durable_progress,
                        diagnostic_stop_after=args.stop_after,
                    )
                    try:
                        cold = batch_builder.build_result(benchmark_block)
                    except SelectedBlockDiagnosticStop as stopped:
                        od_batch_benchmark.append(
                            {
                                "label": label,
                                "block_id": benchmark_block.block_id,
                                "block_variables": benchmark_block.num_free_variables,
                                "diagnostic_stop": asdict(stopped.event),
                                "phase_progress_file": str(args.phase_progress_file),
                                "numerical_cache_published": bool(
                                    tuple(
                                        batch_config.cache_directory.glob("block-*.npz")
                                    )
                                ),
                            }
                        )
                        break
                    matrix = cast(Any, cold.operator.compact_matrix).toarray()
                    if reference_matrix is None:
                        reference_matrix = matrix
                    maximum_difference = float(
                        np.max(np.abs(matrix - reference_matrix), initial=0.0)
                    )
                    warm = FixedRoutingSelectedBlockBuilder(
                        inputs=synthetic_inputs,
                        spec=mapping.spec,
                        compact_layout=synthetic_compact,
                        partition=synthetic_partition,
                        provenance=synthetic_provenance,
                        config=batch_config,
                    ).build_result(benchmark_block)
                    od_batch_benchmark.append(
                        {
                            "label": label,
                            "block_id": benchmark_block.block_id,
                            "block_variables": benchmark_block.num_free_variables,
                            "cold_numerical_seconds": cold.numerical_construction_seconds,
                            "cold_total_seconds": (
                                cold.support_artifact_load_seconds
                                + cold.support_discovery_seconds
                                + cold.routing_preparation_seconds
                                + cold.numerical_construction_seconds
                                + cold.sparse_assembly_seconds
                                + cold.persistence_seconds
                            ),
                            "warm_load_seconds": warm.cache_load_seconds,
                            "warm_cache_hit": warm.cache_hit,
                            "maximum_reference_difference": maximum_difference,
                            "resource_estimate": asdict(cold.estimate),
                            "diagnostics": asdict(cold.diagnostics),
                        }
                    )
        report = dict(base)
        report.update(
            {
                "complete": result.complete,
                "full_network_coverage": result.full_network_coverage,
                "preflight": asdict(result),
                "pilot_authorization": asdict(authorize_block_coordinate_pilot(result)),
                "selected_blocks": selected_block_results,
                "od_batch_benchmark": od_batch_benchmark,
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "jax": jax.__version__,
                    "jax_backend": jax.default_backend(),
                    "devices": [str(value) for value in jax.devices()],
                },
                "phases": phases,
                "total_seconds": perf_counter() - total_started,
            }
        )
        _atomic_payload(args.output, report)
        return report
    finally:
        temporary.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "structural",
            "sampled-exact-support",
            "streaming-exact-support",
            "exact-materialized-plan",
        ),
        default="sampled-exact-support",
    )
    parser.add_argument("--scenario-folder", type=Path, default=EXAMPLE / "data")
    parser.add_argument(
        "--demand", type=Path, default=EXAMPLE / "pre_processing/results/demand.csv"
    )
    parser.add_argument(
        "--fixed-demand", type=Path, default=EXAMPLE / "data/fixed_demand.csv"
    )
    parser.add_argument(
        "--measurements",
        type=Path,
        default=EXAMPLE / "pre_processing/results/measurements_boarding_alighting.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/support_preflight_simple_example_02.json",
    )
    parser.add_argument(
        "--checkpoint-directory", type=Path, default=ROOT / ".cache/support-preflight"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--construct-representative-blocks", action="store_true")
    parser.add_argument("--benchmark-od-batching", action="store_true")
    parser.add_argument("--synthetic-od-columns", type=int, default=64)
    parser.add_argument("--selected-block", action="append", default=[])
    parser.add_argument(
        "--block-cache-directory", type=Path, default=ROOT / ".cache/selected-blocks"
    )
    parser.add_argument("--selected-support-directory", type=Path)
    parser.add_argument("--persist-selected-block-support", action="store_true")
    parser.add_argument(
        "--storage-dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--measurement-chunk-size", type=int, default=512)
    parser.add_argument("--od-chunk-size", type=int, default=32)
    parser.add_argument(
        "--od-batch-size",
        type=lambda value: None if value == "auto" else int(value),
        default=None,
    )
    parser.add_argument("--mapped-edge-chunk-size", type=int, default=2048)
    parser.add_argument("--phase-progress-file", type=Path)
    parser.add_argument("--durable-progress", action="store_true")
    parser.add_argument(
        "--stop-after",
        choices=("tracing", "lowering", "compilation", "execution"),
        help="diagnostic probe boundary; requires --benchmark-od-batching",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate and report only; this command never estimates",
    )
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--destination-group", type=int, action="append")
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--origin-chunk-size", type=int, default=32)
    parser.add_argument("--maximum-variables-per-block", type=int, default=512)
    parser.add_argument("--maximum-elapsed-seconds", type=float, default=3600.0)
    parser.add_argument("--maximum-rss-bytes", type=int, default=16 * 1024**3)
    parser.add_argument("--maximum-temporary-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--maximum-retained-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--maximum-support-rows-per-block", type=int, default=100_000)
    parser.add_argument("--maximum-nonzeros-per-block", type=int, default=10_000_000)
    parser.add_argument(
        "--maximum-block-operator-bytes", type=int, default=512 * 1024**2
    )
    parser.add_argument("--checkpoint-interval-groups", type=int, default=1)
    parser.add_argument("--checkpoint-interval-seconds", type=float, default=60.0)
    parser.add_argument("--progress-interval-groups", type=int, default=1)
    parser.add_argument("--construction-workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--authorize-exact-materialized-plan", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
