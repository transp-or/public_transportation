"""Current-version direct-scheduled case-study driver.

The command intentionally keeps each stage independent and persists only
identity-bearing summaries. Run it from the case-study root.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from public_transportation.inference.gravity.estimator import (
    GravityEstimatorConfig,
    GravityExecutionPolicy,
    estimate_gravity_model,
)
from public_transportation.inference.gravity.operations import GravityJSONLProgressSink
from public_transportation.inference.gravity.preflight import (
    GravityPreflightPhase,
    run_gravity_preflight,
)
from public_transportation.preprocessing import run_structural_zero_preprocessing

from adapter import (
    CaseContext,
    activate,
    evaluate_once,
    gravity_problem,
    initial_raw_parameters,
    load_context,
    write_json,
)


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "value") and type(value).__module__ == "enum":
        return value.value
    return value


def _paths(context: CaseContext, stage: str) -> tuple[Path, Path]:
    manifests = context.settings.results / "manifests"
    logs = context.settings.results / "logs"
    manifests.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return manifests / f"{stage}.json", logs / f"{stage}.jsonl"


def _base(context: CaseContext, stage: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "completed",
        "stage": stage,
        "package_revision": context.settings.package_revision,
        "python": sys.version,
        "platform": platform.platform(),
        "assignment_fingerprint": context.id_manager.fingerprint,
        "od_layout_fingerprint": context.parameter_layout.fingerprint,
        "compact_layout_fingerprint": context.compact_layout.fingerprint,
        "canonical_index_fingerprint": context.canonical_index.artifact_fingerprint,
        "binding_fingerprint": context.canonical_index.binding_fingerprint,
        "mapping_fingerprint": context.mapping.info.fingerprint,
        "artifact_identity_fingerprint": context.identity.fingerprint,
    }


def _activation_summary(activated: object) -> dict[str, object]:
    decision = getattr(activated, "decision")
    construction = getattr(activated, "construction")
    termination = getattr(activated, "termination")
    source = None if construction is None else getattr(construction, "source", None)
    return {
        "decision": _jsonable(decision),
        "termination": _jsonable(termination),
        "construction": None if source is None else {
            "total_seconds": getattr(source, "total_seconds", None),
            "construction_seconds": getattr(source, "construction_seconds", None),
            "artifact_directory": getattr(source, "artifact_directory", None),
            "checkpoint_directory": getattr(source, "checkpoint_directory", None),
            "reused_shards": getattr(source, "reused_shards", None),
            "rebuilt_shards": getattr(source, "rebuilt_shards", None),
            "worker_failures": getattr(source, "worker_failures", None),
        },
    }


def check(root: Path) -> None:
    context = load_context(root)
    path, log_path = _paths(context, "check")
    payload = _base(context, "check") | {
        "scenario": str(context.settings.scenario),
        "measurements": str(context.settings.measurements),
        "fixed_demand": str(context.settings.fixed_demand),
        "num_physical_od_cells": context.parameter_layout.num_od_total,
        "num_free_od_cells": context.compact_layout.num_free,
        "num_fixed_positive_cells": context.compact_layout.num_fixed_positive,
        "num_removed_zero_cells": context.compact_layout.num_removed_zero,
        "num_measurements": context.mapping.spec.num_measurements,
    }
    write_json(path, payload)
    sink = GravityJSONLProgressSink(log_path, durable=True, context={"stage": "check"})
    sink({"phase": "input_audit", "status": "completed", "current_unit": "canonical_mapping"})
    print(json.dumps(_jsonable(payload), sort_keys=True))


def structural_zeros(root: Path) -> None:
    context = load_context(root)
    path, log_path = _paths(context, "structural-zeros")
    sink = GravityJSONLProgressSink(log_path, durable=True, context={"stage": "structural-zeros"})
    config_path = context.settings.root / "config/structural_zeros.toml"

    def progress(event: object) -> None:
        # StructuralZeroProgress exposes ETA as properties; materialize them
        # into the durable event rather than relying on a human presentation.
        if hasattr(event, "phase") and hasattr(event, "completed"):
            completed = int(getattr(event, "completed"))
            total = int(getattr(event, "total"))
            sink({
                "phase": str(getattr(event, "phase")),
                "status": "completed" if total and completed == total else "running",
                "completed_units": completed,
                "total_units": total,
                "elapsed_seconds": float(getattr(event, "elapsed_seconds")),
                "throughput_units_per_second": getattr(event, "throughput_units_per_second"),
                "estimated_remaining_seconds": getattr(event, "estimated_remaining_seconds"),
                "eta_confidence": str(getattr(event, "eta_confidence")),
                "current_unit": getattr(event, "message", None),
            })
        else:
            sink(event)

    result = run_structural_zero_preprocessing(config_path, progress=progress)
    payload = _base(context, "structural-zeros") | {
        "scenario_fingerprint": result.scenario_fingerprint,
        "configuration_fingerprint": result.config.fingerprint,
        "topology_fingerprint": result.topology.fingerprint,
        "num_cells": result.analysis.num_cells,
        "num_structural_zero": result.analysis.num_structural_zero,
        "fixed_demand_output": str(result.outputs.fixed_demand),
        "audit_output": str(result.outputs.audit),
        "summary_output": str(result.outputs.summary),
    }
    write_json(path, payload)
    print(json.dumps(_jsonable(payload), sort_keys=True))


def prepare(root: Path) -> None:
    context = load_context(root)
    path, log_path = _paths(context, "prepare")
    sink = GravityJSONLProgressSink(log_path, durable=True, context={"stage": "prepare"})
    activated = activate(context, progress=sink)
    payload = _base(context, "prepare") | {
        **_activation_summary(activated),
        "artifact_root": str(context.settings.results / "artifacts"),
        "checkpoint_root": str(context.settings.results / "checkpoints"),
    }
    if activated.operator is None:
        payload["status"] = "deadline_stopped"
    else:
        payload["operator_representation"] = activated.operator.representation
        payload["operator_fingerprint"] = context.identity.fingerprint
    write_json(path, payload)
    print(json.dumps(_jsonable(payload), sort_keys=True))
    if activated.operator is None:
        raise SystemExit("preparation stopped at a resumable deadline; rerun prepare")


def _operator_and_problem(root: Path, stage: str) -> tuple[CaseContext, object, object, object, Path]:
    context = load_context(root)
    _, log_path = _paths(context, stage)
    sink = GravityJSONLProgressSink(log_path, durable=True, context={"stage": stage})
    activated = activate(context, progress=sink)
    if activated.operator is None:
        raise RuntimeError("no complete direct-scheduled artifact is available; run prepare")
    problem, parameters = gravity_problem(context, activated.operator)
    return context, activated.operator, problem, parameters, log_path


def preflight(root: Path) -> None:
    context, operator, problem, parameters, _ = _operator_and_problem(root, "preflight")
    path, _ = _paths(context, "preflight")
    result = run_gravity_preflight(
        problem=problem,
        raw_parameters=initial_raw_parameters(parameters, context.settings.model),
        stop_after=GravityPreflightPhase.RECOMMENDATION,
    )
    payload = _base(context, "preflight") | {"result": _jsonable(result)}
    write_json(path, payload)
    print(json.dumps(_jsonable(payload), sort_keys=True))


def benchmark(root: Path) -> None:
    context, operator, problem, parameters, _ = _operator_and_problem(root, "benchmark")
    path, _ = _paths(context, "benchmark")
    started = __import__("time").perf_counter()
    evaluation = evaluate_once(problem, initial_raw_parameters(parameters, context.settings.model))
    evaluation["elapsed_seconds"] = __import__("time").perf_counter() - started
    payload = _base(context, "benchmark") | {"result": evaluation}
    write_json(path, payload)
    print(json.dumps(_jsonable(payload), sort_keys=True))


def fit(root: Path, resume: bool = False) -> None:
    context, operator, problem, parameters, log_path = _operator_and_problem(root, "fit")
    path, _ = _paths(context, "fit")
    model = context.settings.model
    checkpoint = context.settings.results / "checkpoints/gravity.json"
    wall = float(model.get("wall_time_seconds", 0.0))
    execution = GravityExecutionPolicy(
        gradient_strategy=str(model.get("gradient_strategy", "auto")),
        wall_time_seconds=None if wall == 0 else wall,
        checkpoint_path=checkpoint,
        progress_interval=int(model.get("progress_interval", 1)),
        jax_compilation_cache_directory=context.settings.results / "jax-cache",
    )
    config = GravityEstimatorConfig(
        maximum_iterations=int(model.get("maximum_iterations", 100)),
        gradient_tolerance=float(model.get("gradient_tolerance", 1.0e-6)),
        objective_tolerance=float(model.get("objective_tolerance", 1.0e-9)),
    )
    sink = GravityJSONLProgressSink(log_path, durable=True, context={"stage": "fit"})
    fit_started = perf_counter()

    def fit_progress(event: object) -> None:
        iteration = int(getattr(event, "iteration"))
        elapsed = float(getattr(event, "elapsed_seconds"))
        total = config.maximum_iterations
        rate = iteration / elapsed if iteration and elapsed > 0 else None
        remaining = None if rate is None else max(0.0, (total - iteration) / rate)
        sink({
            "phase": "gravity_fit",
            "status": "completed" if iteration >= total else "running",
            "completed_units": iteration,
            "total_units": total,
            "elapsed_seconds": elapsed,
            "throughput_units_per_second": rate,
            "estimated_remaining_seconds": remaining,
            "eta_confidence": "low" if iteration < 3 else "medium",
            "current_unit": f"iteration-{iteration}",
            "iteration": iteration,
            "objective": float(getattr(event, "objective")),
            "gradient_inf_norm": float(getattr(event, "gradient_inf_norm")),
            "checkpoint_written": bool(getattr(event, "checkpoint_written")),
            "wall_clock_seconds": perf_counter() - fit_started,
        })

    result = estimate_gravity_model(
        problem=problem,
        compact_layout=context.compact_layout,
        initial_raw_parameters=initial_raw_parameters(parameters, context.settings.model),
        config=config,
        execution=execution,
        resume=resume,
        progress=fit_progress,
    )
    payload = _base(context, "fit") | {"result": _jsonable(result)}
    write_json(path, payload)
    write_json(context.settings.results / "fits/result.json", result)
    print(json.dumps(_jsonable(payload), sort_keys=True))
    if result.status not in {"converged", "iteration_limit", "stopped_by_time_budget"}:
        raise SystemExit(f"gravity fit failed: {result.status}")


def validate(root: Path) -> None:
    context, operator, problem, parameters, _ = _operator_and_problem(root, "validate")
    fit_path, _ = _paths(context, "fit")
    if not fit_path.is_file():
        raise RuntimeError("fit manifest is missing; run fit before validate")
    fit_payload = json.loads(fit_path.read_text(encoding="utf-8"))
    raw = np.asarray(fit_payload["result"]["raw_parameters"], dtype=np.float64)
    evaluation = evaluate_once(problem, raw)
    path, _ = _paths(context, "validate")
    fit_result = fit_payload["result"]
    accepted = fit_result["status"] == "converged" and bool(fit_result["success"])
    payload = _base(context, "validate") | {
        "fit_status": fit_result["status"],
        "objective": evaluation["objective"],
        "gradient_inf_norm": evaluation["gradient_inf_norm"],
        "predicted_measurements": evaluation["measurement_mean"],
        "acceptance": "accepted" if accepted else "diagnostic_only",
    }
    write_json(path, payload)
    print(json.dumps(_jsonable(payload), sort_keys=True))


STAGES = {
    "check": check,
    "structural-zeros": structural_zeros,
    "prepare": prepare,
    "preflight": preflight,
    "benchmark": benchmark,
    "fit": fit,
    "validate": validate,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=tuple(STAGES))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--resume", action="store_true",
        help="resume the identity-matching gravity checkpoint (fit stage only)",
    )
    args = parser.parse_args()
    if args.resume and args.stage != "fit":
        parser.error("--resume is valid only for the fit stage")
    if args.stage == "fit":
        fit(args.root.resolve(), resume=args.resume)
    else:
        STAGES[args.stage](args.root.resolve())


if __name__ == "__main__":
    main()
