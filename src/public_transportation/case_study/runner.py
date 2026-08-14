"""Independent, configuration-driven stages for a generic case study."""

from __future__ import annotations

import argparse
import csv
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


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
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

    def _source_checksums(self) -> dict[str, str]:
        """Return checksums for all configured source and contract files."""
        paths = self.config.paths
        candidates = [
            ("measurements", paths.measurements),
            ("candidate_demand", paths.candidate_demand),
            ("od_pairs", paths.od_pairs),
            ("prior_demand", paths.prior_demand),
            ("fixed_demand", paths.fixed_demand),
            ("production_inputs", paths.production_inputs),
            ("destination_attractiveness", paths.destination_attractiveness),
            ("od_universe_pair_file", self.config.od_universe.pair_file),
            ("prior_input_file", self.config.prior_demand.input_file),
        ]
        checksums = {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in candidates
            if path is not None and path.is_file()
        }
        if paths.scenario_directory.is_dir():
            for path in sorted(item for item in paths.scenario_directory.rglob("*") if item.is_file()):
                checksums[f"scenario/{path.relative_to(paths.scenario_directory)}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        for path in self.config.package_config_paths:
            if path.is_file():
                checksums[f"configuration/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return checksums

    def _stage_provenance(self) -> dict[str, object]:
        return {
            "source_checksums": self._source_checksums(),
            "package": self.package_info(),
            "configuration_fingerprint": self.config.fingerprint,
            "od_universe_policy": {
                "source": self.config.od_universe.source,
                "level": self.config.od_universe.level,
                "include_same_stop": self.config.od_universe.include_same_stop,
                "active_service_only": self.config.od_universe.active_service_only,
                "connectivity_policy": self.config.od_universe.connectivity_policy,
            },
            "service_day": self.config.service_day,
            "timezone": self.config.timezone,
        }

    def _component_provenance(self, data: Any) -> dict[str, object]:
        """Serialize explicit production/attractiveness semantics and dimensions."""
        expansion = data.od_time_expansion
        origin_groups = (
            set()
            if expansion is None
            else {(cell.origin_stop_id, cell.time_bin_id) for cell in expansion.cells}
        )
        destination_groups = (
            set()
            if expansion is None
            else {(cell.destination_stop_id, cell.time_bin_id) for cell in expansion.cells}
        )
        result: dict[str, object] = {}
        for name, groups in (
            ("production", origin_groups),
            ("destination_attractiveness", destination_groups),
        ):
            spec = self.config.model.get(name)
            if not isinstance(spec, dict):
                result[name] = {"mode": "legacy_compatibility"}
                continue
            mode = str(spec["mode"])
            scope = str(spec["correction_scope"])
            estimated_dimension = 0
            if mode == "estimated":
                estimated_dimension = 1 if scope == "global" else max(len(groups) - 1, 0)
            result[name] = {
                "mode": mode,
                "baseline": spec.get("baseline"),
                "correction_scope": scope,
                "transformation": spec.get("transformation"),
                "constraint": spec.get("constraint"),
                "regularization": spec.get("regularization"),
                "prior_scale": spec.get("prior_scale"),
                "group_count": len(groups),
                "estimated_dimension": estimated_dimension,
                "active_parameters": estimated_dimension,
                "inactive_parameters": 0,
                "derived_from_prior": False,
                "derived_from_legacy_demand": False,
            }
        return result

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
            "input_semantics": audit.get("input_semantics", "legacy_time_dependent_demand"),
            "od_universe_policy": {
                "source": self.config.od_universe.source,
                "level": self.config.od_universe.level,
                "include_same_stop": self.config.od_universe.include_same_stop,
                "active_service_only": self.config.od_universe.active_service_only,
                "connectivity_policy": self.config.od_universe.connectivity_policy,
            },
            "prior_demand": {
                "source": self.config.prior_demand.source,
                "semantics": self.config.prior_demand.semantics,
                "expansion": self.config.prior_demand.expansion,
            },
            "production_specification": self.config.model.get("production"),
            "destination_attractiveness_specification": self.config.model.get("destination_attractiveness"),
        }
        payload = {"package": package, "audit": audit, "manifest": manifest}
        output = _write_json(self._stage_path("audit", "input_audit.json"), payload)
        _write_json(self._stage_path("audit", "run_manifest.json"), manifest)
        self.emit({"stage": "check", "status": "completed", "output": str(output)})
        return payload

    def od_universe(self) -> dict[str, object]:
        """Validate/write the independent candidate pair universe."""
        data = self.adapter.data
        if data.candidate_od_universe is None:
            payload = {
                "status": "legacy_compatibility",
                "input_semantics": "legacy_time_dependent_demand",
                "message": "OD universe is derived from legacy demand.csv; no pair-only input was supplied.",
            }
            _write_json(self._stage_path("audit", "od_universe.json"), payload)
            return payload
        universe = data.candidate_od_universe
        _write_csv(
            self._stage_path("audit", "od_pairs.csv"),
            ("origin_stop_id", "destination_stop_id"),
            [pair.tuple for pair in universe.pairs],
        )
        _write_csv(
            self._stage_path("audit", "od_universe_exclusions.csv"),
            ("origin_stop_id", "destination_stop_id", "reason", "detail"),
            [item.tuple for item in universe.exclusions],
        )
        payload = {
            **universe.audit,
            **self._stage_provenance(),
        }
        _write_json(self._stage_path("audit", "od_universe.json"), payload)
        self.emit({"stage": "od-universe", "status": "completed", "output": str(self._stage_path("audit", "od_universe.json"))})
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

    def expand_od(self) -> dict[str, object]:
        """Expand the approved time bins across the independent OD universe."""
        data = self.adapter.data
        if data.od_time_expansion is None:
            payload = {
                "status": "legacy_compatibility",
                "input_semantics": "legacy_time_dependent_demand",
                "message": "OD--time cells are read from legacy demand.csv.",
            }
            _write_json(self._stage_path("audit", "od_time_expansion.json"), payload)
            return payload
        expansion = data.od_time_expansion
        _write_csv(
            self._stage_path("audit", "od_time_exclusions.csv"),
            ("origin_stop_id", "destination_stop_id", "time_bin_id", "reason", "detail"),
            [
                (item.origin_stop_id, item.destination_stop_id, item.time_bin_id, item.reason, item.detail)
                for item in expansion.exclusions
            ],
        )
        _write_csv(
            self._stage_path("generated_inputs", "candidate_od_time.csv"),
            ("origin_stop_id", "destination_stop_id", "time_bin_id"),
            [cell.tuple for cell in expansion.cells],
        )
        approved_bins_fingerprint = hashlib.sha256(
            json.dumps([list(item) for item in expansion.time_bins], separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        payload = {
            **expansion.audit,
            "approved_time_bins_fingerprint": approved_bins_fingerprint,
            **self._stage_provenance(),
        }
        _write_json(self._stage_path("audit", "od_time_expansion.json"), payload)
        if data.prior_demand is not None:
            _write_csv(
                self._stage_path("generated_inputs", "prior_demand.csv"),
                ("origin_stop_id", "destination_stop_id", "time_bin_id", "prior_value"),
                [(*cell.tuple, value) for cell, value in sorted(data.prior_demand.items())],
            )
            _write_json(
                self._stage_path("audit", "prior_generation.json"),
                {
                    **self._stage_provenance(),
                    **(
                        {}
                        if data.prior_generation is None
                        else data.prior_generation.audit
                    ),
                    "source": self.config.prior_demand.source,
                    "semantics": self.config.prior_demand.semantics,
                    "expansion": self.config.prior_demand.expansion,
                    "cell_count": len(data.prior_demand),
                    "universe_fingerprint": data.candidate_od_universe.fingerprint,
                    "od_time_expansion_fingerprint": expansion.fingerprint,
                    "approved_time_bins_fingerprint": approved_bins_fingerprint,
                },
            )
        _write_json(
            self._stage_path("audit", "production_attractiveness_provenance.json"),
            {
                **self._stage_provenance(),
                "components": self._component_provenance(data),
                "input_files": {
                    "production": None if self.config.paths.production_inputs is None else str(self.config.paths.production_inputs),
                    "destination_attractiveness": None if self.config.paths.destination_attractiveness is None else str(self.config.paths.destination_attractiveness),
                },
                "derived_from_prior": False,
                "derived_from_legacy_demand": False,
                "approved_time_bins_fingerprint": approved_bins_fingerprint,
            },
        )
        self.emit({"stage": "expand-od", "status": "completed", "output": str(self._stage_path("audit", "od_time_expansion.json"))})
        return payload

    def structural_zeros(self) -> dict[str, object]:
        data = self.adapter.data
        if data.od_time_expansion is not None:
            fixed = {
                (item.origin_stop_id, item.destination_stop_id, item.time_bin_id): 0.0
                for item in data.od_time_expansion.exclusions
            }
            _write_csv(
                self._stage_path("generated_inputs", "fixed_demand.csv"),
                ("origin_stop_id", "dest_stop_id", "time_bin_id", "flow"),
                [(*key, value) for key, value in sorted(fixed.items())],
            )
            _write_csv(
                self._stage_path("audit", "structural_zero_audit.csv"),
                ("origin_stop_id", "destination_stop_id", "time_bin_id", "reason", "detail"),
                [
                    (
                        item.origin_stop_id,
                        item.destination_stop_id,
                        item.time_bin_id,
                        item.reason,
                        item.detail,
                    )
                    for item in data.od_time_expansion.exclusions
                ],
            )
            payload = {
                "num_cells": data.od_time_expansion.audit["expanded_od_time_count"],
                "num_structural_zero": len(fixed),
                "num_retained": data.od_time_expansion.cell_count,
                "num_fixed_merged": len(fixed),
                "num_free_after_merge": data.od_time_expansion.cell_count,
                "primary_reason_counts": data.od_time_expansion.audit["exclusion_counts"],
                "input_semantics": data.input_semantics,
                "output_folder": str(self._stage_path("generated_inputs", "")),
                "configuration_fingerprint": self.config.fingerprint,
            }
            output = _write_json(self._stage_path("audit", "structural_zero_summary.json"), payload)
            self.emit({"stage": "structural-zeros", "status": "completed", "output": str(output)})
            return payload
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

        def component_mode(name: str, legacy: str | None) -> str:
            spec = self.config.model.get(name)
            if isinstance(spec, dict):
                mode = str(spec["mode"])
                return "estimated_basis" if mode == "estimated" else "provided"
            if name == "production" and legacy is not None:
                return legacy
            return "provided"

        def centered_basis(
            count: int, *, component: str, scope: str, label: str
        ) -> tuple[np.ndarray, tuple[str, ...]]:
            if scope == "global":
                return np.ones((count, 1), dtype=np.float64), (f"{label}.global",)
            if scope not in {"origin_time", "destination", "destination_time"}:
                raise ValueError(
                    f"{component} estimated correction scope {scope!r} is not supported by the reduced-OD runner."
                )
            if count < 2:
                raise ValueError(
                    f"{component} estimated correction requires at least two active groups."
                )
            # The final group is the reference implied by a sum-to-zero basis:
            # each column is e_i - e_last, so the represented effects sum to zero.
            basis = np.zeros((count, count - 1), dtype=np.float64)
            basis[:-1, :] = np.eye(count - 1)
            basis[-1, :] = -1.0
            labels = tuple(f"{label}[{index}]" for index in range(count - 1))
            return basis, labels

        legacy_production_mode = self.config.model.get("production_mode")
        production_mode = component_mode(
            "production",
            None if legacy_production_mode is None else str(legacy_production_mode),
        )
        production_basis = None
        production_labels = None
        production_columns = 0
        production_spec = self.config.model.get("production")
        if production_mode == "estimated_basis":
            scope = "global" if not isinstance(production_spec, dict) else str(production_spec["correction_scope"])
            production_basis, production_labels = centered_basis(
                artifacts.features.number_of_origin_time_groups,
                component="production",
                scope=scope,
                label="production.deviation",
            )
            production_columns = production_basis.shape[1]

        attraction_mode = component_mode("destination_attractiveness", None)
        attraction_basis = None
        attraction_labels = None
        attraction_columns = 0
        attraction_spec = self.config.model.get("destination_attractiveness")
        if attraction_mode == "estimated_basis":
            scope = "destination" if not isinstance(attraction_spec, dict) else str(attraction_spec["correction_scope"])
            attraction_basis, attraction_labels = centered_basis(
                len(artifacts.features.destination_ids),
                component="destination_attractiveness",
                scope=scope,
                label="destination_attractiveness.deviation",
            )
            attraction_columns = attraction_basis.shape[1]
        specification = MinimalGravitySpecification(
            likelihood=selected_likelihood,
            production_mode=production_mode,
            production_basis_columns=production_columns,
            destination_attractiveness_mode=attraction_mode,
            destination_attractiveness_basis_columns=attraction_columns,
        )
        built = build_minimal_gravity_problem(
            artifacts=artifacts,
            specification=specification,
            production_basis=production_basis,
            production_basis_labels=production_labels,
            destination_attractiveness_basis=attraction_basis,
            destination_attractiveness_basis_labels=attraction_labels,
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
            destination_attractiveness_basis=built.problem.destination_attractiveness_basis,
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
        fitted_demand = generate_minimal_gravity_demand(
            result.raw_parameters, problem=built.problem
        )
        payload = {
            "method": method,
            "likelihood": likelihood,
            "status": result.status,
            "success": result.success,
            "message": result.message,
            "objective": result.objective,
            "raw_parameters": result.raw_parameters.tolist(),
            "fitted_productions": [
                [list(key), float(value)]
                for key, value in zip(
                    built.problem.features.origin_time_group_keys,
                    np.asarray(fitted_demand.productions),
                    strict=True,
                )
            ],
            "fitted_destination_attractiveness": [
                [list(key.tuple), float(value)]
                for key, value in zip(
                    built.problem.features.cell_keys,
                    np.asarray(fitted_demand.destination_attractiveness),
                    strict=True,
                )
            ],
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
            "od-universe": "audit",
            "time-discretization": "audit",
            "materialize-bins": "generated_inputs",
            "expand-od": "audit",
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
        "od-universe": runner.od_universe,
        "time-discretization": runner.time_discretization,
        "expand-od": runner.expand_od,
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
    parser.add_argument("stage", choices=("check", "od-universe", "time-discretization", "materialize-bins", "expand-od", "structural-zeros", "prepare", "preflight", "benchmark", "fit", "diagnose", "reconstruct", "validate-detailed"))
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
