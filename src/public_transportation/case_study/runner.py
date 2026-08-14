"""Independent, configuration-driven stages for a generic case study."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from importlib.metadata import distribution
from pathlib import Path
from typing import Any, Sequence

import jax
import numpy as np

from public_transportation.inference.reduced_od import (
    GaussianRawParameterPrior,
    MinimalGravitySpecification,
    ReducedODFitConfig,
    benchmark_minimal_gravity_objective,
    build_minimal_gravity_problem,
    default_minimal_gravity_raw_parameters,
    estimate_minimal_gravity,
    generate_minimal_gravity_demand,
    load_reduced_od_artifacts,
    preflight_reduced_od_j0,
)
from public_transportation.inference.reduced_od.contracts import (
    JourneyODTimeKey,
    ReducedODProblemContract,
)
from public_transportation.inference.reduced_od.reconstruction import reconstruct_full_od
from public_transportation.preprocessing.materialize_time_bins import materialize_time_bins
from public_transportation.inference.reduced_od import prepare_reduced_od_artifacts
from public_transportation.preprocessing.structural_zeros import run_structural_zero_preprocessing
from public_transportation.preprocessing.time_discretization import (
    TimeDiscretizationConfig,
    recommend_time_discretization,
)

from .adapter import GenericCaseAdapter, GenericCaseHook
from .config import CaseStudyConfig, load_case_study_config


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


class GenericCaseRunner:
    """Run exactly one named case stage and persist only under results/."""

    def __init__(
        self,
        config: CaseStudyConfig,
        *,
        json_progress: bool = False,
        custom_hook: GenericCaseHook | None = None,
    ):
        self.config = config
        self.adapter = GenericCaseAdapter(config, custom_hook=custom_hook)
        self.json_progress = json_progress
        self.results = config.paths.results_directory
        self.results.mkdir(parents=True, exist_ok=True)

    def emit(self, event: dict[str, object]) -> None:
        event = {
            "case": self.config.case_name,
            "configuration_fingerprint": self.config.fingerprint,
            **event,
        }
        if self.json_progress:
            print(json.dumps(event, sort_keys=True, default=str), flush=True)

    def package_info(self) -> dict[str, object]:
        installed = distribution("public_transportation")
        package_path = Path(__import__("public_transportation").__file__).resolve()
        lockfile = self.config.source_file.parent.parent / "uv.lock"
        lock_text = lockfile.read_text(encoding="utf-8") if lockfile.is_file() else ""
        package_block = ""
        marker = 'name = "public-transportation"'
        if marker in lock_text:
            package_block = lock_text[lock_text.index(marker) : lock_text.index(marker) + 4096]
        direct_url_text = installed.read_text("direct_url.json") or ""
        try:
            direct_url = json.loads(direct_url_text) if direct_url_text else {}
        except json.JSONDecodeError:
            direct_url = {}
        direct_url_info = direct_url.get("dir_info", {}) if isinstance(direct_url, dict) else {}
        editable = (
            'source = { editable = "." }' in package_block
            or bool(isinstance(direct_url_info, dict) and direct_url_info.get("editable"))
        )
        revision_match = re.search(r'\brev = "([0-9a-fA-F]+)"', package_block)
        lock_source = (
            "editable" if editable else
            ("git" if "source = { git =" in package_block else
             ("registry" if "source = { registry =" in package_block else "unknown"))
        )
        return {
            "distribution_version": installed.version,
            "package_file": str(package_path),
            "python_version": platform.python_version(),
            "jax_version": getattr(jax, "__version__", "unknown"),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "jax_x64": bool(jax.config.x64_enabled),
            "uv_lock": str(lockfile) if lockfile.is_file() else None,
            "uv_lock_sha256": hashlib.sha256(lock_text.encode("utf-8")).hexdigest() if lock_text else None,
            "editable_local_package_detected": editable,
            "locked_source_type": lock_source,
            "locked_revision": revision_match.group(1) if revision_match else None,
            "installed_direct_url": direct_url.get("url") if isinstance(direct_url, dict) else None,
            "command": list(sys.argv),
            "working_directory": str(Path.cwd()),
            "package_source_contract": (
                "editable-local-lock-entry" if editable else
                ("immutable-lock-entry" if package_block else "lockfile-not-found")
            ),
        }

    def _stage_path(self, category: str, name: str) -> Path:
        path = (self.results / category / name).resolve()
        if self.results.resolve() not in path.parents:
            raise ValueError(f"stage output escaped results directory: {path}")
        return path

    def check(self) -> dict[str, object]:
        self.emit({"stage": "check", "status": "started"})
        audit = self.adapter.audit().to_dict()
        package = self.package_info()
        manifest = {
            "case": self.config.case_name,
            "configuration_fingerprint": self.config.fingerprint,
            "configuration_files": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.config.package_config_paths
                if path.is_file()
            },
            "input_checksums": dict(audit["source_checksums"]),
            "package": package,
        }
        payload = {"package": package, "audit": audit, "manifest": manifest}
        output = _write_json(self._stage_path("audit", "input_audit.json"), payload)
        _write_json(self._stage_path("audit", "run_manifest.json"), manifest)
        self.emit({"stage": "check", "status": "completed", "output": str(output)})
        return payload

    def time_discretization(self) -> dict[str, object]:
        self.adapter.audit()
        settings = self.config.time_discretization
        report = recommend_time_discretization(
            self.adapter.data.measurements.records,
            TimeDiscretizationConfig(
                base_resolution_minutes=settings.base_resolution_minutes,
                min_bin_minutes=settings.min_bin_minutes,
                max_bin_minutes=settings.max_bin_minutes,
                max_bins=settings.max_bins,
                num_od_pairs=settings.num_od_pairs,
                max_od_cells=settings.max_od_cells,
                horizon_start_s=settings.horizon_start_s,
                horizon_end_s=settings.horizon_end_s,
            ),
        )
        output = _write_json(self._stage_path("audit", "time_discretization_recommendation.json"), report)
        self.emit({"stage": "time-discretization", "status": "completed", "output": str(output)})
        return report

    def materialize_bins(
        self,
        *,
        candidate: str | None,
        reviewer: str | None,
        overwrite: bool = False,
    ) -> dict[str, object]:
        if not candidate or not reviewer:
            raise ValueError("materialize-bins requires --candidate and --reviewer approval.")
        recommendation = self._stage_path("audit", "time_discretization_recommendation.json")
        if not recommendation.is_file():
            raise FileNotFoundError("run time-discretization before materialize-bins.")
        output = self._stage_path("generated_inputs", "time_bins.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        bins = materialize_time_bins(
            recommendation,
            output,
            candidate_name=candidate,
            overwrite=overwrite,
        )
        manifest = {
            "candidate": candidate,
            "reviewer": reviewer,
            "recommendation": str(recommendation),
            "time_bins": bins,
            "configuration_fingerprint": self.config.fingerprint,
        }
        _write_json(self._stage_path("generated_inputs", "time_bins_manifest.json"), manifest)
        self.emit({"stage": "materialize-bins", "status": "completed", "output": str(output), "reviewer": reviewer})
        return manifest

    def structural_zeros(self) -> dict[str, object]:
        configured_output = self.config.structural_zero_config.output.folder.resolve()
        if self.results.resolve() not in configured_output.parents:
            raise ValueError(
                "structural-zero output is outside the configured results directory: "
                f"{configured_output}"
            )
        result = run_structural_zero_preprocessing(self.config.structural_zero_config_file)
        output_folder = result.outputs.folder.resolve()
        if self.results.resolve() not in output_folder.parents:
            raise ValueError(
                "structural-zero output escaped the configured results directory: "
                f"{output_folder}"
            )
        payload = {
            "num_cells": result.analysis.num_cells,
            "num_structural_zero": result.analysis.num_structural_zero,
            "num_retained": result.analysis.num_retained,
            "num_fixed_merged": result.reconciliation.num_merged,
            "num_free_after_merge": result.analysis.num_cells - result.reconciliation.num_merged,
            "primary_reason_counts": result.analysis.reason_counts,
            "output_folder": str(output_folder),
            "configuration_fingerprint": self.config.fingerprint,
        }
        output = _write_json(self._stage_path("audit", "structural_zero_summary.json"), payload)
        self.emit({"stage": "structural-zeros", "status": "completed", "output": str(output)})
        return payload

    def prepare(self) -> dict[str, object]:
        self.adapter.audit()
        self.emit({"stage": "prepare", "status": "started"})
        prepared = prepare_reduced_od_artifacts(
            scenario=self.adapter.data.scenario,
            measurements=self.adapter.data.measurements,
            configuration=self.config.reduced_od_config,
            inputs=self.adapter.preparation_inputs(),
            output_directory=self._stage_path("artifacts", "reduced_od"),
            cache_policy="reuse_or_build",
            progress=self.emit,
        )
        payload = {
            "directory": str(prepared.directory),
            "fingerprints": dict(prepared.fingerprints),
            "dimensions": prepared.dimensions,
            "phase_diagnostics": list(prepared.phase_diagnostics),
            "configuration_fingerprint": self.config.fingerprint,
        }
        output = _write_json(self._stage_path("artifacts", "prepare_summary.json"), payload)
        self.emit({"stage": "prepare", "status": "completed", "output": str(output)})
        return payload

    def _artifacts_and_problem(self, likelihood: str | None = None):
        artifacts = load_reduced_od_artifacts(
            configuration=self.config.reduced_od_config,
            artifact_directory=self._stage_path("artifacts", "reduced_od"),
        )
        selected_likelihood = likelihood or str(self.config.model["likelihood"])
        production_mode = str(self.config.model["production_mode"])
        basis = None
        labels = None
        columns = 0
        if production_mode == "estimated_basis":
            columns = 1
            basis = np.ones((artifacts.features.number_of_origin_time_groups, columns), dtype=np.float64)
            labels = ("global_log_scale",)
        specification = MinimalGravitySpecification(
            likelihood=selected_likelihood,
            production_mode=production_mode,
            production_basis_columns=columns,
        )
        built = build_minimal_gravity_problem(
            artifacts=artifacts,
            specification=specification,
            production_basis=basis,
            production_basis_labels=labels,
        )
        return artifacts, built

    def preflight(self) -> dict[str, object]:
        self.adapter.audit()
        artifacts, built = self._artifacts_and_problem()
        payload = preflight_reduced_od_j0(
            configuration=self.config.reduced_od_config,
            artifact_directory=self._stage_path("artifacts", "reduced_od"),
            specification=built.problem.parameter_layout.specification,
            production_basis=built.problem.production_basis,
        )
        payload["configuration_fingerprint"] = self.config.fingerprint
        payload["package"] = self.package_info()
        output = _write_json(self._stage_path("preflight", "preflight.json"), payload)
        self.emit({"stage": "preflight", "status": "completed", "output": str(output), "artifact_count": len(artifacts.fingerprints)})
        return payload

    def benchmark(self) -> dict[str, object]:
        _, built = self._artifacts_and_problem()
        initial = default_minimal_gravity_raw_parameters(built.problem.parameter_layout)
        timing = benchmark_minimal_gravity_objective(problem=built.problem, raw_parameters=initial)
        payload = {"benchmark": timing.to_dict(), "model_fingerprint": built.model_fingerprint, "configuration_fingerprint": self.config.fingerprint}
        output = _write_json(self._stage_path("audit", "warm_benchmark.json"), payload)
        self.emit({"stage": "benchmark", "status": "completed", "output": str(output)})
        return payload

    def fit(self, *, method: str, likelihood: str) -> dict[str, object]:
        if method not in {"ml", "map"}:
            raise ValueError("fit --method must be ml or map.")
        _, built = self._artifacts_and_problem(likelihood)
        initial = default_minimal_gravity_raw_parameters(built.problem.parameter_layout)
        fit_config = ReducedODFitConfig(method=method, maximum_iterations=int(self.config.model.get("maximum_iterations", 25)), gradient_tolerance=float(self.config.model.get("gradient_tolerance", 1.0e-5)), function_tolerance=float(self.config.model.get("function_tolerance", 1.0e-8)))
        prior = None
        if method == "map":
            prior = GaussianRawParameterPrior(np.zeros(initial.size), np.full(initial.size, float(self.config.model.get("prior_scale", 2.0))))
        result = estimate_minimal_gravity(problem=built.problem, initial_raw_parameters=initial, model_fingerprint=built.model_fingerprint, config=fit_config, prior=prior, progress=self.emit)
        payload = {
            "method": method,
            "likelihood": likelihood,
            "status": result.status,
            "success": result.success,
            "message": result.message,
            "objective": result.objective,
            "raw_parameters": result.raw_parameters.tolist(),
            "model_fingerprint": built.model_fingerprint,
            "configuration_fingerprint": self.config.fingerprint,
        }
        output = _write_json(self._stage_path("fits", f"{method}_{likelihood}.json"), payload)
        self.emit({"stage": "fit", "status": "completed", "output": str(output)})
        return payload

    def reconstruct(self, *, fit_path: str | Path) -> dict[str, object]:
        fit_file = Path(fit_path).resolve()
        if not fit_file.is_file():
            raise FileNotFoundError(f"fit result does not exist: {fit_file}")
        fit = json.loads(fit_file.read_text(encoding="utf-8"))
        artifacts, built = self._artifacts_and_problem(str(fit["likelihood"]))
        raw = np.asarray(fit["raw_parameters"], dtype=np.float64)
        generated = generate_minimal_gravity_demand(raw, problem=built.problem)
        free_keys = tuple(artifacts.measurement_response.free_cell_keys)
        free_journey_keys = tuple(JourneyODTimeKey(*key.tuple) for key in free_keys)
        fixed_items = tuple(sorted(artifacts.fixed_demand.items()))
        all_keys = tuple(sorted(free_journey_keys + tuple(JourneyODTimeKey(*key.tuple) for key, _ in fixed_items)))
        free_set = set(free_journey_keys)
        free_indices = np.asarray([index for index, key in enumerate(all_keys) if key in free_set], dtype=np.int64)
        fixed_indices = np.asarray([index for index, key in enumerate(all_keys) if key not in free_set], dtype=np.int64)
        fixed_by_key = {
            JourneyODTimeKey(*key.tuple): float(value)
            for key, value in fixed_items
        }
        fixed_values = np.asarray(
            [fixed_by_key[all_keys[index]] for index in fixed_indices],
            dtype=np.float64,
        )
        contract = ReducedODProblemContract(self.config.reduced_od_config.fingerprint, artifacts.fingerprints["timetable_index"], artifacts.fingerprints["measurement_response"], all_keys, free_indices, fixed_indices, fixed_values)
        reconstructed = reconstruct_full_od(contract=contract, free_cell_keys=free_keys, free_demand=np.asarray(generated.demand))
        payload = {"keys": [list(key.tuple) for key in reconstructed.keys], "demand": reconstructed.demand.tolist(), "estimated": reconstructed.estimated.tolist(), "configuration_fingerprint": self.config.fingerprint}
        output = _write_json(self._stage_path("validation", "reconstructed_od.json"), payload)
        self.emit({"stage": "reconstruct", "status": "completed", "output": str(output)})
        return payload

    def diagnose(self, *, fit_path: str | Path) -> dict[str, object]:
        fit_file = Path(fit_path).resolve()
        if not fit_file.is_file():
            raise FileNotFoundError(f"fit result does not exist: {fit_file}")
        fit = json.loads(fit_file.read_text(encoding="utf-8"))
        payload = {"fit": str(fit_file), "method": fit.get("method"), "likelihood": fit.get("likelihood"), "status": fit.get("status"), "scientific_diagnostics": "Run grouped validation and detailed assignment explicitly for the accepted fit.", "configuration_fingerprint": self.config.fingerprint}
        output = _write_json(self._stage_path("diagnostics", "fit_diagnostics.json"), payload)
        self.emit({"stage": "diagnose", "status": "completed", "output": str(output)})
        return payload

    def validate_detailed(self, *, od_path: str | Path) -> dict[str, object]:
        path = Path(od_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"OD result does not exist: {path}")
        payload = {"od_result": str(path), "status": "not_run", "message": "Detailed assignment is an explicit case hook; no generic fallback is permitted.", "configuration_fingerprint": self.config.fingerprint}
        output = _write_json(self._stage_path("validation", "detailed_validation.json"), payload)
        self.emit({"stage": "validate-detailed", "status": "completed", "output": str(output)})
        return payload


def run_case_stage(
    config_path: str | Path,
    stage: str,
    *,
    case_root: str | Path | None = None,
    candidate: str | None = None,
    reviewer: str | None = None,
    overwrite: bool = False,
    method: str = "ml",
    likelihood: str = "poisson",
    fit_path: str | Path | None = None,
    od_path: str | Path | None = None,
    json_progress: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    config = load_case_study_config(config_path, case_root=case_root)
    if dry_run:
        output_categories = {
            "check": "audit",
            "time-discretization": "audit",
            "materialize-bins": "generated_inputs",
            "structural-zeros": "audit",
            "prepare": "artifacts",
            "preflight": "preflight",
            "benchmark": "audit",
            "fit": "fits",
            "diagnose": "diagnostics",
            "reconstruct": "validation",
            "validate-detailed": "validation",
        }
        if stage not in output_categories:
            raise ValueError(f"unknown case-study stage: {stage}")
        return {
            "status": "dry_run",
            "case": config.case_name,
            "stage": stage,
            "configuration_fingerprint": config.fingerprint,
            "results_directory": str(config.paths.results_directory),
            "output_category": output_categories[stage],
        }
    runner = GenericCaseRunner(config, json_progress=json_progress)
    actions = {
        "check": runner.check,
        "time-discretization": runner.time_discretization,
        "structural-zeros": runner.structural_zeros,
        "prepare": runner.prepare,
        "preflight": runner.preflight,
        "benchmark": runner.benchmark,
    }
    if stage in actions:
        return actions[stage]()
    if stage == "materialize-bins":
        return runner.materialize_bins(candidate=candidate, reviewer=reviewer, overwrite=overwrite)
    if stage == "fit":
        return runner.fit(method=method, likelihood=likelihood)
    if stage == "diagnose":
        if fit_path is None:
            raise ValueError("diagnose requires --fit.")
        return runner.diagnose(fit_path=fit_path)
    if stage == "reconstruct":
        if fit_path is None:
            raise ValueError("reconstruct requires --fit.")
        return runner.reconstruct(fit_path=fit_path)
    if stage == "validate-detailed":
        if od_path is None:
            raise ValueError("validate-detailed requires --od.")
        return runner.validate_detailed(od_path=od_path)
    raise ValueError(f"unknown case-study stage: {stage}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "time-discretization", "materialize-bins", "structural-zeros", "prepare", "preflight", "benchmark", "fit", "diagnose", "reconstruct", "validate-detailed"))
    parser.add_argument("--config", type=Path, default=Path("config/case.toml"))
    parser.add_argument("--case-root", type=Path, default=None)
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--reviewer", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--method", choices=("ml", "map"), default="ml")
    parser.add_argument("--likelihood", choices=("poisson", "negative_binomial"), default="poisson")
    parser.add_argument("--fit", type=Path, default=None)
    parser.add_argument("--od", type=Path, default=None)
    parser.add_argument("--json-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = run_case_stage(
            args.config,
            args.stage,
            case_root=args.case_root,
            candidate=args.candidate,
            reviewer=args.reviewer,
            overwrite=args.overwrite,
            method=args.method,
            likelihood=args.likelihood,
            fit_path=args.fit,
            od_path=args.od,
            json_progress=args.json_progress,
            dry_run=args.dry_run,
        )
    except Exception as error:  # CLI boundary: preserve a non-zero status.
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
