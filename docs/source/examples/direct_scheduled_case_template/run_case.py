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
from threading import Event, Thread
from time import perf_counter
from typing import Mapping

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
from public_transportation.preprocessing import (
    ODTimeExpansionInterrupted,
    run_structural_zero_preprocessing,
)
from public_transportation.inference.measurement_support_preflight import (
    UnsupportedPositiveBoardingError,
)

from adapter import (
    CaseContext,
    CaseSettings,
    activate,
    bootstrap_prior_demand,
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


class _StageProgress:
    """Durable stage progress with heartbeats for uninstrumented work."""

    def __init__(self, settings: CaseSettings, stage: str) -> None:
        self.settings = settings
        self.stage = stage
        self.manifest_path = settings.results / "manifests" / f"{stage}.json"
        self.log_path = settings.results / "logs" / f"{stage}.jsonl"
        self.sink = GravityJSONLProgressSink(
            self.log_path,
            durable=True,
            context={"stage": stage},
        )
        self._started = perf_counter()
        self._phase = "initialization"
        self._current_unit = "load_context"
        self._stop = Event()
        self._heartbeat_thread: Thread | None = None
        self._finished = False

    @property
    def elapsed_seconds(self) -> float:
        return perf_counter() - self._started

    def _event(self, event: Mapping[str, object]) -> None:
        payload = dict(event)
        payload.setdefault("schema_version", 1)
        payload.setdefault("stage", self.stage)
        payload.setdefault("elapsed_seconds", self.elapsed_seconds)
        self.sink(payload)

    def start(self) -> None:
        """Write the first durable event before context loading begins."""
        self._event(
            {
                "phase": "initialization",
                "status": "started",
                "elapsed_seconds": 0.0,
                "current_unit": "load_context",
                "estimated_remaining_seconds": None,
                "eta_confidence": "unavailable",
            }
        )
        self._heartbeat_thread = Thread(
            target=self._heartbeat_loop,
            name=f"{self.stage}-progress-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        interval = float(self.settings.progress_interval_seconds)
        while not self._stop.wait(interval):
            self._event(
                {
                    "phase": self._phase,
                    "status": "running",
                    "current_unit": self._current_unit,
                    "estimated_remaining_seconds": None,
                    "eta_confidence": "unavailable",
                    "heartbeat": True,
                }
            )

    def __call__(self, event: object) -> None:
        """Forward adapter/package events and track the active phase."""
        if isinstance(event, Mapping):
            payload = dict(event)
        else:
            payload = {
                "phase": self._phase,
                "status": "running",
                "current_unit": self._current_unit,
                "source_event_type": type(event).__name__,
            }
        phase = payload.get("phase")
        current_unit = payload.get("current_unit")
        if isinstance(phase, str):
            self._phase = phase
        if isinstance(current_unit, str) and current_unit:
            self._current_unit = current_unit
        payload.setdefault("phase", self._phase)
        payload.setdefault("status", "running")
        payload.setdefault("current_unit", self._current_unit)
        self._event(payload)

    def phase_started(self, phase: str, current_unit: str) -> None:
        self._phase = phase
        self._current_unit = current_unit
        self(
            {
                "phase": phase,
                "status": "running",
                "current_unit": current_unit,
                "estimated_remaining_seconds": None,
                "eta_confidence": "unavailable",
            }
        )

    def phase_completed(self, phase: str, current_unit: str) -> None:
        self._phase = phase
        self._current_unit = current_unit
        self(
            {
                "phase": phase,
                "status": "completed",
                "current_unit": current_unit,
            }
        )

    def finish(
        self,
        status: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        self._stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=max(1.0, self.settings.progress_interval_seconds * 2.0))
        payload: dict[str, object] = {
            "phase": "stage_completion",
            "status": status,
            "current_unit": self.stage,
        }
        if error is not None:
            payload.update(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            report_path = getattr(error, "report_path", None)
            if report_path is not None:
                payload["positive_boarding_support_report_path"] = str(report_path)
        self._event(payload)


def _start_stage(root: Path, stage: str) -> tuple[CaseSettings, _StageProgress]:
    """Load only settings, then open progress before context construction."""
    settings = CaseSettings.load(root)
    settings.results.joinpath("manifests").mkdir(parents=True, exist_ok=True)
    settings.results.joinpath("logs").mkdir(parents=True, exist_ok=True)
    progress = _StageProgress(settings, stage)
    progress.start()
    return settings, progress


def _failure_manifest(
    settings: CaseSettings,
    progress: _StageProgress,
    error: BaseException,
    *,
    context: CaseContext | None = None,
    status: str = "failed",
) -> dict[str, object]:
    existing: dict[str, object] = {}
    if progress.manifest_path.is_file():
        try:
            candidate = json.loads(progress.manifest_path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                existing = candidate
        except (OSError, json.JSONDecodeError):
            existing = {}
    if existing:
        payload = existing
        payload["status"] = status
    elif context is None:
        payload: dict[str, object] = {
            "schema_version": 1,
            "status": status,
            "stage": progress.stage,
            "package_revision": settings.package_revision,
        }
    else:
        payload = _base(context, progress.stage)
        payload["status"] = status
    payload.update(
        {
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )
    report_path = getattr(error, "report_path", None)
    if report_path is not None:
        payload["positive_boarding_support_report_path"] = str(report_path)
    write_json(progress.manifest_path, payload)
    return payload


def _is_deadline_stop(error: BaseException) -> bool:
    return isinstance(error, TimeoutError) or (
        isinstance(error, SystemExit) and "stopped" in str(error).lower()
    )


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
        # These identify the exact fixed-demand file used to build the
        # context, rather than merely the configured fallback path.
        "fixed_demand": str(context.fixed_demand_path),
        "fixed_demand_source": context.fixed_demand_source,
        "fixed_demand_sha256": context.fixed_demand_sha256,
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
    settings, progress = _start_stage(root, "check")
    context: CaseContext | None = None
    try:
        context = load_context(root, settings=settings, progress=progress)
        support_audit_path = settings.results / "audit/feasibility_support.json"
        if not support_audit_path.is_file():
            raise ValueError(
                "check requires the feasibility support audit generated by feature construction: "
                f"{support_audit_path}"
            )
        support_audit = json.loads(support_audit_path.read_text(encoding="utf-8"))
        if not isinstance(support_audit, dict):
            raise ValueError(f"feasibility support audit is not a JSON object: {support_audit_path}")
        if support_audit.get("status") != "completed":
            raise ValueError(
                "feasibility support audit is not completed: "
                f"{support_audit_path}"
            )
        if any(
            int(support_audit.get(field, 0)) != 0
            for field in (
                "unsupported_free_cells",
                "cells_present_only_in_bootstrap_support",
                "cells_present_only_in_feature_construction_support",
            )
        ):
            raise ValueError(
                "feasibility support audit reports a support-set mismatch: "
                f"{support_audit_path}"
            )
        payload = _base(context, "check") | {
            "scenario": str(context.settings.scenario),
            "measurements": str(context.settings.measurements),
            "num_physical_od_cells": context.parameter_layout.num_od_total,
            "num_free_od_cells": context.compact_layout.num_free,
            "num_fixed_positive_cells": context.compact_layout.num_fixed_positive,
            "num_removed_zero_cells": context.compact_layout.num_removed_zero,
            "num_measurements": context.mapping.spec.num_measurements,
            "feasibility_support_audit": str(support_audit_path),
            "feasibility_support_contract_fingerprint": support_audit["contract"]["fingerprint"],
        }
        write_json(progress.manifest_path, payload)
        progress({"phase": "input_audit", "status": "completed", "current_unit": "canonical_mapping"})
        progress.finish("completed")
        print(json.dumps(_jsonable(payload), sort_keys=True))
    except BaseException as error:
        status = "deadline_stopped" if _is_deadline_stop(error) else (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        _failure_manifest(settings, progress, error, context=context, status=status)
        progress.finish(status, error=error)
        raise


def bootstrap_prior(root: Path, resume: bool = False) -> None:
    settings, progress = _start_stage(root, "bootstrap-prior")

    def normalized_progress(event: Mapping[str, object]) -> None:
        normalized = dict(event)
        normalized.setdefault(
            "phase",
            "expand_od_time" if "completed_chunks" in normalized else "materialize_prior",
        )
        if "completed_cells" in normalized:
            normalized.setdefault("completed_units", normalized["completed_cells"])
        if "total_cells" in normalized:
            normalized.setdefault("total_units", normalized["total_cells"])
        if "eta_seconds" in normalized:
            normalized.setdefault(
                "estimated_remaining_seconds", normalized["eta_seconds"]
            )
        normalized.setdefault(
            "current_unit", normalized.get("next_chunk", normalized.get("chunk"))
        )
        progress(normalized)

    try:
        progress({
            "phase": "bootstrap_prior",
            "status": "running",
            "current_unit": "candidate_od_time_expansion",
            "resume": resume,
        })
        audit = bootstrap_prior_demand(
            root,
            resume=resume,
            settings=settings,
            progress=normalized_progress,
        )
        payload = {
            "schema_version": 1,
            "status": "completed",
            "stage": "bootstrap-prior",
            "package_revision": settings.package_revision,
            "checkpoint_directory": audit["checkpoint_directory"],
            "prior_generation": audit,
        }
        write_json(progress.manifest_path, payload)
        progress({
            "phase": "bootstrap_prior",
            "status": "completed",
            "completed_units": 1,
            "total_units": 1,
            "current_unit": "prior_demand.csv",
            "output_file": audit["output_file"],
        })
        progress.finish("completed")
        print(json.dumps(_jsonable(payload), sort_keys=True))
    except BaseException as error:
        status = "deadline_stopped" if _is_deadline_stop(error) else (
            "interrupted" if isinstance(error, (KeyboardInterrupt, ODTimeExpansionInterrupted)) else "failed"
        )
        payload = _failure_manifest(settings, progress, error, status=status)
        if isinstance(error, ODTimeExpansionInterrupted):
            payload["checkpoint_directory"] = str(error.checkpoint_directory)
            write_json(progress.manifest_path, payload)
        progress.finish(status, error=error)
        if isinstance(error, ODTimeExpansionInterrupted):
            raise SystemExit(error.exit_code) from error
        raise


def structural_zeros(root: Path) -> None:
    settings, progress = _start_stage(root, "structural-zeros")
    context: CaseContext | None = None
    try:
        config_path = settings.root / "config/structural_zeros.toml"
        progress.phase_started("structural_zero_preprocessing", "configuration_and_topology")

        def structural_progress(event: object) -> None:
            if hasattr(event, "phase") and hasattr(event, "completed"):
                completed = int(getattr(event, "completed"))
                total = int(getattr(event, "total"))
                progress({
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
                progress(event)

        result = run_structural_zero_preprocessing(config_path, progress=structural_progress)
        progress.phase_completed("structural_zero_preprocessing", "structural_zero_outputs")
        progress.phase_started("context_initialization", "load_context")
        context = load_context(root, settings=settings, progress=progress)
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
        write_json(progress.manifest_path, payload)
        progress.finish("completed")
        print(json.dumps(_jsonable(payload), sort_keys=True))
    except BaseException as error:
        status = "deadline_stopped" if _is_deadline_stop(error) else (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        _failure_manifest(settings, progress, error, context=context, status=status)
        progress.finish(status, error=error)
        raise


def prepare(root: Path) -> None:
    settings, progress = _start_stage(root, "prepare")
    context: CaseContext | None = None
    try:
        context = load_context(root, settings=settings, progress=progress)
        progress.phase_started("operator_activation", "direct_scheduled_operator")
        activated = activate(context, progress=progress)
        progress.phase_completed("operator_activation", "direct_scheduled_operator")
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
        write_json(progress.manifest_path, payload)
        progress.finish(str(payload["status"]))
        print(json.dumps(_jsonable(payload), sort_keys=True))
        if activated.operator is None:
            raise SystemExit("preparation stopped at a resumable deadline; rerun prepare")
    except BaseException as error:
        status = "deadline_stopped" if _is_deadline_stop(error) else (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        _failure_manifest(settings, progress, error, context=context, status=status)
        progress.finish(status, error=error)
        raise


def _operator_and_problem(
    root: Path,
    stage: str,
    settings: CaseSettings,
    progress: _StageProgress,
) -> tuple[CaseContext, object, object, object]:
    context = load_context(root, settings=settings, progress=progress)
    progress.phase_started("operator_activation", "direct_scheduled_operator")
    activated = activate(context, progress=progress)
    if activated.operator is None:
        raise RuntimeError("no complete direct-scheduled artifact is available; run prepare")
    progress.phase_completed("operator_activation", "direct_scheduled_operator")
    problem, parameters = gravity_problem(context, activated.operator)
    return context, activated.operator, problem, parameters


def preflight(root: Path) -> None:
    settings, progress = _start_stage(root, "preflight")
    context: CaseContext | None = None
    try:
        context, operator, problem, parameters = _operator_and_problem(
            root, "preflight", settings, progress
        )
        progress.phase_started("preflight_evaluation", "recommendation")
        result = run_gravity_preflight(
            problem=problem,
            raw_parameters=initial_raw_parameters(parameters, context.settings.model),
            stop_after=GravityPreflightPhase.RECOMMENDATION,
        )
        progress.phase_completed("preflight_evaluation", "recommendation")
        payload = _base(context, "preflight") | {"result": _jsonable(result)}
        write_json(progress.manifest_path, payload)
        progress.finish("completed")
        print(json.dumps(_jsonable(payload), sort_keys=True))
    except BaseException as error:
        status = "deadline_stopped" if _is_deadline_stop(error) else (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        _failure_manifest(settings, progress, error, context=context, status=status)
        progress.finish(status, error=error)
        raise


def benchmark(root: Path) -> None:
    settings, progress = _start_stage(root, "benchmark")
    context: CaseContext | None = None
    try:
        context, operator, problem, parameters = _operator_and_problem(
            root, "benchmark", settings, progress
        )
        progress.phase_started("benchmark_evaluation", "objective_and_gradient")
        started = perf_counter()
        evaluation = evaluate_once(problem, initial_raw_parameters(parameters, context.settings.model))
        evaluation["elapsed_seconds"] = perf_counter() - started
        progress.phase_completed("benchmark_evaluation", "objective_and_gradient")
        payload = _base(context, "benchmark") | {"result": evaluation}
        write_json(progress.manifest_path, payload)
        progress.finish("completed")
        print(json.dumps(_jsonable(payload), sort_keys=True))
    except BaseException as error:
        status = "deadline_stopped" if _is_deadline_stop(error) else (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        _failure_manifest(settings, progress, error, context=context, status=status)
        progress.finish(status, error=error)
        raise


def fit(root: Path, resume: bool = False) -> None:
    settings, progress = _start_stage(root, "fit")
    context: CaseContext | None = None
    try:
        context, operator, problem, parameters = _operator_and_problem(
            root, "fit", settings, progress
        )
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
        fit_started = perf_counter()

        def fit_progress(event: object) -> None:
            iteration = int(getattr(event, "iteration"))
            elapsed = float(getattr(event, "elapsed_seconds"))
            total = config.maximum_iterations
            rate = iteration / elapsed if iteration and elapsed > 0 else None
            remaining = None if rate is None else max(0.0, (total - iteration) / rate)
            progress({
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

        progress.phase_started("gravity_fit", "initialization")
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
        write_json(progress.manifest_path, payload)
        write_json(context.settings.results / "fits/result.json", result)
        stage_status = "completed" if result.status in {
            "converged", "iteration_limit", "stopped_by_time_budget"
        } else "failed"
        progress.finish(stage_status)
        print(json.dumps(_jsonable(payload), sort_keys=True))
        if stage_status == "failed":
            raise SystemExit(f"gravity fit failed: {result.status}")
    except BaseException as error:
        status = "deadline_stopped" if _is_deadline_stop(error) else (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        _failure_manifest(settings, progress, error, context=context, status=status)
        progress.finish(status, error=error)
        raise


def validate(root: Path) -> None:
    settings, progress = _start_stage(root, "validate")
    context: CaseContext | None = None
    try:
        context, operator, problem, parameters = _operator_and_problem(
            root, "validate", settings, progress
        )
        progress.phase_started("fit_result_loading", "fit_manifest")
        fit_path = context.settings.results / "manifests/fit.json"
        if not fit_path.is_file():
            raise RuntimeError("fit manifest is missing; run fit before validate")
        fit_payload = json.loads(fit_path.read_text(encoding="utf-8"))
        raw = np.asarray(fit_payload["result"]["raw_parameters"], dtype=np.float64)
        progress.phase_completed("fit_result_loading", "fit_manifest")
        progress.phase_started("validation_evaluation", "objective_and_predictions")
        evaluation = evaluate_once(problem, raw)
        progress.phase_completed("validation_evaluation", "objective_and_predictions")
        fit_result = fit_payload["result"]
        accepted = fit_result["status"] == "converged" and bool(fit_result["success"])
        payload = _base(context, "validate") | {
            "fit_status": fit_result["status"],
            "objective": evaluation["objective"],
            "gradient_inf_norm": evaluation["gradient_inf_norm"],
            "predicted_measurements": evaluation["measurement_mean"],
            "acceptance": "accepted" if accepted else "diagnostic_only",
        }
        write_json(progress.manifest_path, payload)
        progress.finish("completed")
        print(json.dumps(_jsonable(payload), sort_keys=True))
    except BaseException as error:
        status = "deadline_stopped" if _is_deadline_stop(error) else (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        _failure_manifest(settings, progress, error, context=context, status=status)
        progress.finish(status, error=error)
        raise


STAGES = {
    "bootstrap-prior": bootstrap_prior,
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
        help="resume the identity-matching bootstrap or gravity checkpoint",
    )
    args = parser.parse_args()
    if args.resume and args.stage not in {"bootstrap-prior", "fit"}:
        parser.error("--resume is valid only for bootstrap-prior and fit stages")
    try:
        if args.stage == "fit":
            fit(args.root.resolve(), resume=args.resume)
        elif args.stage == "bootstrap-prior":
            bootstrap_prior(args.root.resolve(), resume=args.resume)
        else:
            STAGES[args.stage](args.root.resolve())
    except UnsupportedPositiveBoardingError as error:
        # This is an expected, actionable case-data/support failure.  The
        # stage has already persisted its manifest and JSONL failure event;
        # avoid burying the report in a generic Python traceback.
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
