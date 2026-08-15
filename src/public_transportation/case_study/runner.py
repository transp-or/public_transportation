"""Independent, configuration-driven stages for a generic case study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
import time
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
from public_transportation.preprocessing.od_universe import (
    ODTimeExpansionInterrupted,
    TimetableFeasibilityIndex,
    _read_pair_priors,
    expansion_contract_fingerprint,
    run_candidate_od_time_expansion,
)
from public_transportation.inference.reduced_od import prepare_reduced_od_artifacts
from public_transportation.preprocessing.structural_zeros import run_structural_zero_preprocessing
from public_transportation.preprocessing.time_discretization import (
    TimeDiscretizationConfig,
    recommend_time_discretization,
)
from public_transportation.preprocessing.reduced_od import JourneyTimePeriod

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


def _write_csv_atomic(path: Path, header: Sequence[str], rows: Any) -> Path:
    """Write a potentially large row iterator without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def _write_json_atomic(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
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
        return self.adapter._current_source_checksums()

    def _stage_provenance(self) -> dict[str, object]:
        return {
            "source_checksums": self._source_checksums(),
            "scenario_checksums": self._source_checksums(),
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

    def _package_revision(self) -> str:
        """Stable package identity used by freshness manifests."""
        return self.adapter._current_package_revision()

    def _component_provenance(self, data: Any | None = None, *, expansion: Any | None = None) -> dict[str, object]:
        """Serialize explicit production/attractiveness semantics and dimensions."""
        if expansion is None:
            if data is None:
                raise ValueError("component provenance requires case data or an expansion")
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
        universe = self.adapter.build_od_universe()
        if universe is None:
            payload = {
                "status": "legacy_compatibility",
                "input_semantics": "legacy_time_dependent_demand",
                "message": "OD universe is derived from legacy demand.csv; no pair-only input was supplied.",
            }
            _write_json(self._stage_path("audit", "od_universe.json"), payload)
            return payload
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
            "complexity_preflight": self.adapter.complexity_preflight(
                pair_count=universe.pair_count,
                pair_level_exclusions=len(universe.exclusions),
                time_bin_count=0,
            ),
            "configured_max_bins": self.config.time_discretization.max_bins,
            "worst_case_od_time_cells": universe.pair_count * self.config.time_discretization.max_bins,
            "maximum_configured_od_cells": self.config.time_discretization.max_od_cells,
            "worst_case_within_budget": (
                self.config.time_discretization.max_od_cells is None
                or universe.pair_count * self.config.time_discretization.max_bins
                <= self.config.time_discretization.max_od_cells
            ),
            **self._stage_provenance(),
        }
        _write_json(self._stage_path("audit", "od_universe.json"), payload)
        self.emit({"stage": "od-universe", "status": "completed", "output": str(self._stage_path("audit", "od_universe.json"))})
        return payload

    def time_discretization(self) -> dict[str, object]:
        settings = self.config.time_discretization
        base = self.adapter.load_base_data()
        od_audit = self.adapter.load_current_od_universe_audit(required=False)
        if base.input_semantics == "independent_od_universe":
            if od_audit is None:
                if settings.num_od_pairs is None or settings.max_od_cells is None:
                    raise FileNotFoundError(
                        "run od-universe before time-discretization for an independent OD case"
                    )
                pair_count = settings.num_od_pairs
                od_fingerprint = f"manual-approved:{pair_count}"
            else:
                pair_count = int(od_audit["retained_pair_count"])
                od_fingerprint = str(od_audit["fingerprint"])
        else:
            pair_count = self.adapter._persisted_pair_count()
            if pair_count is None:
                raise ValueError("legacy time-discretization requires a candidate demand pair count")
            od_fingerprint, _, _ = self.adapter._od_universe_identity()
        complexity = self.adapter.complexity_preflight(
            pair_count=pair_count,
            time_bin_count=0,
            raise_on_exceed=False,
        )
        report = recommend_time_discretization(
            base.measurements.records,
            TimeDiscretizationConfig(
                base_resolution_minutes=settings.base_resolution_minutes,
                min_bin_minutes=settings.min_bin_minutes,
                max_bin_minutes=settings.max_bin_minutes,
                max_bins=settings.max_bins,
                num_od_pairs=pair_count,
                max_od_cells=settings.max_od_cells,
                horizon_start_s=settings.horizon_start_s,
                horizon_end_s=settings.horizon_end_s,
                allow_infeasible_budget=True,
            ),
        )
        report["complexity_preflight"] = complexity
        report["configuration_fingerprint"] = self.config.fingerprint
        report["package_revision"] = self._package_revision()
        report["od_universe_fingerprint"] = od_fingerprint
        report["retained_pair_count"] = pair_count
        report["time_discretization_fingerprint"] = self.config.time_discretization_fingerprint
        report["source_checksums"] = self.adapter._current_source_checksums()
        output = _write_json(self._stage_path("audit", "time_discretization_recommendation.json"), report)
        if report.get("status") == "blocked":
            raise ValueError(
                "no candidate time discretization satisfies max_od_cells; "
                f"recommendation written to {output}. Review the budget and rerun."
            )
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
        try:
            recommendation_payload = json.loads(recommendation.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("time-discretization recommendation is not valid JSON") from error
        if not isinstance(recommendation_payload, dict):
            raise ValueError("time-discretization recommendation must contain an object")
        od_fingerprint, pair_count, _ = self.adapter._od_universe_identity()
        expected = {
            "configuration_fingerprint": self.config.fingerprint,
            "package_revision": self._package_revision(),
            "od_universe_fingerprint": od_fingerprint,
            "retained_pair_count": pair_count,
            "time_discretization_fingerprint": self.config.time_discretization_fingerprint,
            "source_checksums": self.adapter._current_source_checksums(),
        }
        stale_reasons = [
            key
            for key, value in expected.items()
            if recommendation_payload.get(key) != value
        ]
        if stale_reasons:
            raise ValueError(
                "STALE ARTIFACT: time_discretization_recommendation.json\n"
                "Reason: " + ", ".join(stale_reasons) +
                "\nAction: rerun time-discretization."
            )
        if recommendation_payload.get("status") == "blocked" or not isinstance(recommendation_payload.get("recommendation"), dict):
            raise ValueError("time-discretization recommendation is blocked; review its budget before materialize-bins")
        candidates = recommendation_payload.get("candidates", [])
        selected = recommendation_payload.get("recommendation")
        named = candidate == "recommendation" or (
            isinstance(selected, dict) and selected.get("name") == candidate
        )
        if candidate != "recommendation" and not any(
            isinstance(item, dict) and item.get("name") == candidate and item.get("valid") is True
            for item in candidates if isinstance(candidates, list)
        ):
            named = False
        if not named:
            raise ValueError(f"candidate {candidate!r} is not a valid candidate in the current recommendation")
        output = self._stage_path("generated_inputs", "time_bins.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        _, manifest_path = self.adapter._time_bin_artifact_paths()
        if output.is_file() and not overwrite:
            stale = self.adapter.stale_artifacts()
            if stale:
                reasons = ", ".join(item["reason"] for item in stale)
                raise ValueError(
                    f"STALE ARTIFACT: {output}\nReason: {reasons}\n"
                    "Action: review the new recommendation and pass --overwrite explicitly."
                )
        bins = materialize_time_bins(
            recommendation,
            output,
            candidate_name=candidate,
            overwrite=overwrite,
        )
        periods = tuple(JourneyTimePeriod(item["bin_id"], int(item["start_s"]), int(item["end_s"])) for item in bins)
        manifest = {
            "candidate": candidate,
            "reviewer": reviewer,
            "recommendation": str(recommendation),
            "recommendation_fingerprint": hashlib.sha256(recommendation.read_bytes()).hexdigest(),
            "time_bins": bins,
            "time_bins_fingerprint": self.adapter._time_bins_fingerprint(periods),
            "configuration_fingerprint": self.config.fingerprint,
            "package_revision": self._package_revision(),
            "od_universe_fingerprint": od_fingerprint,
            "retained_pair_count": pair_count,
            "time_discretization_fingerprint": self.config.time_discretization_fingerprint,
            "source_checksums": self.adapter._current_source_checksums(),
        }
        _write_json(manifest_path, manifest)
        self.emit({"stage": "materialize-bins", "status": "completed", "output": str(output), "reviewer": reviewer})
        return manifest

    def _iter_checkpoint_rows(self, checkpoint: Path):
        manifest_path = checkpoint / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError("OD-time expansion checkpoint is incomplete; resume or complete expand-od first")
        for index in sorted(int(item) for item in manifest.get("completed_chunks", [])):
            chunk = checkpoint / f"chunk-{index:06d}.jsonl"
            if not chunk.is_file():
                raise ValueError(f"OD-time expansion checkpoint chunk is missing: {chunk}")
            with chunk.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        row = json.loads(line)
                        if row.get("status") not in {"retained", "excluded"}:
                            raise ValueError(f"invalid expansion row status in {chunk}")
                        yield row

    def _expansion_components_from_groups(
        self,
        origin_groups: set[tuple[str, str]],
        destination_groups: set[tuple[str, str]],
    ) -> dict[str, object]:
        """Component provenance from streaming group sets (without retaining cells)."""
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
            estimated_dimension = 0 if mode != "estimated" else (1 if scope == "global" else max(len(groups) - 1, 0))
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

    def expand_od(self, *, resume: bool = False, fresh: bool = False) -> dict[str, object]:
        """Expand approved bins with durable, deterministic checkpoints."""
        if resume and fresh:
            raise ValueError("expand-od accepts either --resume or --fresh, not both")
        if self.config.od_universe.source == "legacy_time_dependent_demand":
            payload = {
                "status": "legacy_compatibility",
                "input_semantics": "legacy_time_dependent_demand",
                "message": "OD--time cells are read from legacy demand.csv.",
            }
            _write_json_atomic(self._stage_path("audit", "od_time_expansion.json"), payload)
            return payload
        base = self.adapter.load_base_data()
        universe = self.adapter.load_persisted_universe()
        periods = self.adapter._approved_time_periods(require_materialized=True)
        bins = tuple(JourneyTimePeriod(item.period_id, item.start_seconds, item.end_seconds) for item in periods)
        approved_bins_fingerprint = self.adapter._time_bins_fingerprint(bins)
        reduced = self.config.reduced_od_config
        feasibility = {
            "maximum_transfers": reduced.journeys.maximum_transfers,
            "maximum_initial_wait_seconds": reduced.journeys.maximum_waiting_seconds,
            "maximum_journey_seconds": reduced.journeys.maximum_journey_seconds,
            "maximum_waiting_seconds": reduced.journeys.maximum_waiting_seconds,
            "timetable_policy": "required",
        }
        expansion_config: dict[str, object] = {
            **feasibility,
            "chunk_size_pairs": self.config.expansion.chunk_size_pairs,
            "progress_interval_seconds": self.config.expansion.progress_interval_seconds,
            "checkpoint_enabled": self.config.expansion.checkpoint_enabled,
            "resume_requires_explicit_flag": self.config.expansion.resume_requires_explicit_flag,
            "maximum_temporary_bytes": self.config.expansion.maximum_temporary_bytes,
            "configuration_fingerprint": self.config.fingerprint,
            "package_revision": self._package_revision(),
            "source_checksums": self._source_checksums(),
            "scenario_checksums": self._source_checksums(),
            "od_universe_fingerprint": universe.fingerprint,
            "approved_time_bins_fingerprint": approved_bins_fingerprint,
            "reduced_od_fingerprint": self.config.reduced_od_config.fingerprint,
            "physical_stop_mapping": sorted(universe.physical_stop_mapping.items()),
        }
        expansion_fingerprint = expansion_contract_fingerprint(universe, bins, expansion_config)
        checkpoint_root = self.results / "checkpoints" / "od_time_expansion"
        checkpoint = (checkpoint_root / expansion_fingerprint).resolve()
        if self.results.resolve() not in checkpoint.parents:
            raise ValueError("expansion checkpoint escaped the configured results directory")
        existing_checkpoints: list[Path] = []
        if checkpoint_root.is_dir():
            for candidate in sorted(checkpoint_root.iterdir()):
                if not candidate.is_dir() or ".archive-" in candidate.name:
                    continue
                manifest_path = candidate / "manifest.json"
                if manifest_path.is_file():
                    existing_checkpoints.append(candidate)
                    try:
                        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as error:
                        raise ValueError(f"invalid expansion checkpoint manifest: {manifest_path}") from error
                    if (
                        not fresh
                        and not resume
                        and existing_manifest.get("od_universe_fingerprint") == universe.fingerprint
                        and candidate != checkpoint
                    ):
                        raise ValueError(
                            "an expansion checkpoint exists for this OD universe but its contract fingerprint differs; "
                            "pass --fresh explicitly after reviewing it"
                        )
        if fresh:
            for candidate in existing_checkpoints:
                if (candidate / ".lock").exists():
                    raise RuntimeError(f"cannot archive active expansion checkpoint {candidate}")
                archive = candidate.with_name(f"{candidate.name}.archive-{time.time_ns()}")
                os.replace(candidate, archive)
        elif checkpoint.exists() and not resume and any(checkpoint.iterdir()):
            raise FileExistsError(
                f"checkpoint already exists at {checkpoint}; pass --resume or --fresh explicitly"
            )
        marker = self._stage_path("audit", "od_time_expansion.json")
        incomplete_payload = {
            "status": "running",
            "input_semantics": "independent_od_universe",
            "universe_fingerprint": universe.fingerprint,
            "expansion_fingerprint": expansion_fingerprint,
            "approved_time_bins_fingerprint": approved_bins_fingerprint,
            "checkpoint_directory": str(checkpoint),
            "configuration_fingerprint": self.config.fingerprint,
            "source_checksums": self._source_checksums(),
        }
        _write_json_atomic(marker, incomplete_payload)
        index = TimetableFeasibilityIndex.from_scenario(
            base.scenario,
            physical_stop_mapping=universe.physical_stop_mapping,
        )
        try:
            result = run_candidate_od_time_expansion(
                universe,
                bins,
                scenario=base.scenario,
                feasibility_index=index,
                configuration=expansion_config,
                checkpoint_directory=checkpoint,
                resume=resume,
                progress=lambda event: self.emit({"stage": "expand-od", **dict(event)}),
            )
        except ODTimeExpansionInterrupted as error:
            _write_json_atomic(
                marker,
                {
                    **incomplete_payload,
                    "status": "interrupted",
                    "checkpoint_directory": str(error.checkpoint_directory),
                },
            )
            raise

        def rows():
            return self._iter_checkpoint_rows(result.checkpoint_directory)
        origin_groups: set[tuple[str, str]] = set()
        destination_groups: set[tuple[str, str]] = set()
        exclusion_counts: dict[str, int] = {}

        def retained_rows():
            for row in rows():
                if row["status"] == "retained":
                    origin_groups.add((str(row["origin_stop_id"]), str(row["time_bin_id"])))
                    destination_groups.add((str(row["destination_stop_id"]), str(row["time_bin_id"])))
                    yield (row["origin_stop_id"], row["destination_stop_id"], row["time_bin_id"])

        def excluded_rows():
            for row in rows():
                if row["status"] == "excluded":
                    reason = str(row.get("reason", ""))
                    exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
                    yield (
                        row["origin_stop_id"],
                        row["destination_stop_id"],
                        row["time_bin_id"],
                        reason,
                        row.get("detail", ""),
                    )

        _write_csv_atomic(
            self._stage_path("generated_inputs", "candidate_od_time.csv"),
            ("origin_stop_id", "destination_stop_id", "time_bin_id"),
            retained_rows(),
        )
        _write_csv_atomic(
            self._stage_path("audit", "od_time_exclusions.csv"),
            ("origin_stop_id", "destination_stop_id", "time_bin_id", "reason", "detail"),
            excluded_rows(),
        )
        source = self.config.prior_demand.source
        prior_value = float(self.config.prior_demand.value or 1.0)
        pair_values: dict[tuple[str, str], float] = {}
        if source == "external_file":
            if self.config.prior_demand.input_file is None:
                raise ValueError("external_file prior requires prior_demand input_file")
            pair_values = _read_pair_priors(self.config.prior_demand.input_file)
        elif source != "all_ones":
            raise NotImplementedError(f"checkpointed expansion does not support prior source {source!r}")
        if source == "all_ones" and (not np.isfinite(prior_value) or prior_value <= 0.0):
            raise ValueError("all_ones prior value must be finite and positive")

        def prior_rows():
            seen_pairs: set[tuple[str, str]] = set()
            for row in retained_rows():
                key = (str(row[0]), str(row[1]))
                seen_pairs.add(key)
                value = prior_value if source == "all_ones" else pair_values.get(key)
                if value is None:
                    raise ValueError(f"external prior is missing retained pair {key!r}")
                yield (*row, float(value))
            if source == "external_file":
                extra = sorted(set(pair_values) - seen_pairs)
                if extra:
                    raise ValueError(f"external prior contains pairs not retained by expansion: {extra}")

        _write_csv_atomic(
            self._stage_path("generated_inputs", "prior_demand.csv"),
            ("origin_stop_id", "destination_stop_id", "time_bin_id", "prior_value"),
            prior_rows(),
        )
        complexity = self.adapter.complexity_preflight(
            pair_count=universe.pair_count,
            pair_level_exclusions=len(universe.exclusions),
            time_bin_count=len(periods),
        )
        payload = {
            "status": "completed",
            "input_semantics": "independent_od_universe",
            "input_pair_count": universe.pair_count,
            "time_bin_count": len(bins),
            "expanded_od_time_count": result.total_cells,
            "retained_cell_count": result.retained_cells,
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
            "retained_cells": result.retained_cells,
            "excluded_cells": result.excluded_cells,
            "universe_fingerprint": universe.fingerprint,
            "pair_universe_fingerprint": universe.fingerprint,
            "expansion_fingerprint": result.expansion_fingerprint,
            "semantic_checksum": result.semantic_checksum,
            "feasibility_settings": feasibility,
            "complexity_preflight": complexity,
            "approved_time_bins_fingerprint": approved_bins_fingerprint,
            "checkpoint_directory": str(result.checkpoint_directory),
            "checkpoint_reused": result.checkpoint_reused,
            "completed_chunks": result.completed_chunks,
            "total_chunks": result.total_chunks,
            **self._stage_provenance(),
        }
        _write_json_atomic(marker, payload)
        prior_parameters: dict[str, object]
        if source == "all_ones":
            prior_parameters = {"value": prior_value, "expansion": "one_per_retained_od_time_cell"}
        else:
            prior_parameters = {
                "prior_file": str(self.config.prior_demand.input_file),
                "expansion": "pair_value_repeated_over_retained_bins",
            }
        prior_payload = {
            **self._stage_provenance(),
            "source": source,
            "semantics": self.config.prior_demand.semantics,
            "parameters": prior_parameters,
            "cell_count": result.retained_cells,
            "universe_fingerprint": universe.fingerprint,
            "od_time_expansion_fingerprint": result.expansion_fingerprint,
            "approved_time_bins_fingerprint": approved_bins_fingerprint,
            "generator_fingerprint": hashlib.sha256(
                json.dumps(
                    {"source": source, "semantics": self.config.prior_demand.semantics, "parameters": prior_parameters},
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
        }
        prior_payload["fingerprint"] = hashlib.sha256(
            json.dumps(
                {"generator_fingerprint": prior_payload["generator_fingerprint"], "expansion_fingerprint": result.expansion_fingerprint},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        _write_json_atomic(self._stage_path("audit", "prior_generation.json"), prior_payload)
        _write_json_atomic(
            self._stage_path("audit", "production_attractiveness_provenance.json"),
            {
                **self._stage_provenance(),
                "components": self._expansion_components_from_groups(origin_groups, destination_groups),
                "input_files": {
                    "production": None if self.config.paths.production_inputs is None else str(self.config.paths.production_inputs),
                    "destination_attractiveness": None if self.config.paths.destination_attractiveness is None else str(self.config.paths.destination_attractiveness),
                },
                "derived_from_prior": False,
                "derived_from_legacy_demand": False,
                "approved_time_bins_fingerprint": approved_bins_fingerprint,
            },
        )
        self.emit({"stage": "expand-od", "status": "completed", "output": str(marker), "checkpoint_reused": result.checkpoint_reused})
        return payload

    def structural_zeros(self) -> dict[str, object]:
        if self.config.od_universe.source != "legacy_time_dependent_demand":
            expansion_audit_path = self._stage_path("audit", "od_time_expansion.json")
            if not expansion_audit_path.is_file():
                raise FileNotFoundError("run expand-od before structural-zeros")
            audit_payload = json.loads(expansion_audit_path.read_text(encoding="utf-8"))
            if audit_payload.get("status") != "completed":
                raise ValueError("OD-time expansion is incomplete; complete expand-od before structural-zeros")
            reason_counts: dict[str, int] = {}

            def fixed_rows():
                for item in self.adapter.iter_persisted_expansion_records():
                    if item["status"] == "excluded":
                        reason = str(item.get("reason", ""))
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                        yield (item["origin_stop_id"], item["destination_stop_id"], item["time_bin_id"], 0.0)

            _write_csv_atomic(
                self._stage_path("generated_inputs", "fixed_demand.csv"),
                ("origin_stop_id", "dest_stop_id", "time_bin_id", "fixed_flow"),
                fixed_rows(),
            )

            def audit_rows():
                for item in self.adapter.iter_persisted_expansion_records():
                    if item["status"] == "excluded":
                        yield (
                            item["origin_stop_id"],
                            item["destination_stop_id"],
                            item["time_bin_id"],
                            item.get("reason", ""),
                            item.get("detail", ""),
                        )

            _write_csv_atomic(
                self._stage_path("audit", "structural_zero_audit.csv"),
                ("origin_stop_id", "destination_stop_id", "time_bin_id", "reason", "detail"),
                audit_rows(),
            )
            payload = {
                "num_cells": audit_payload.get("expanded_od_time_count"),
                "num_structural_zero": audit_payload.get("excluded_cells", 0),
                "num_retained": audit_payload.get("retained_cell_count", 0),
                "num_fixed_merged": audit_payload.get("excluded_cells", 0),
                "num_free_after_merge": audit_payload.get("retained_cell_count", 0),
                "primary_reason_counts": dict(sorted(reason_counts.items())),
                "input_semantics": "independent_od_universe",
                "expansion_fingerprint": audit_payload.get("expansion_fingerprint"),
                "expansion_status": audit_payload.get("status"),
                "checkpoint_reused": audit_payload.get("checkpoint_reused", False),
                "completed_chunks": audit_payload.get("completed_chunks"),
                "checkpoint_directory": audit_payload.get("checkpoint_directory"),
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
        self.emit({"stage": "prepare", "status": "started"})
        data = self.adapter.load_persisted_data()
        prepared = prepare_reduced_od_artifacts(
            scenario=data.scenario,
            measurements=data.measurements,
            configuration=self.config.reduced_od_config,
            inputs=self.adapter.build_preparation_inputs(),
            output_directory=self._stage_path("artifacts", "reduced_od"),
            cache_policy="reuse_or_build",
            progress=self.emit,
        )
        expansion_report: dict[str, object] = {"status": "legacy_compatibility"}
        expansion_audit_path = self._stage_path("audit", "od_time_expansion.json")
        if expansion_audit_path.is_file():
            audit_payload = json.loads(expansion_audit_path.read_text(encoding="utf-8"))
            expansion_report = {
                "status": audit_payload.get("status"),
                "checkpoint_reused": audit_payload.get("checkpoint_reused", False),
                "checkpoint_directory": audit_payload.get("checkpoint_directory"),
                "completed_chunks": audit_payload.get("completed_chunks"),
                "total_chunks": audit_payload.get("total_chunks"),
                "retained_cells": audit_payload.get("retained_cell_count"),
                "structural_zero_cells": audit_payload.get("excluded_cells"),
                "expansion_fingerprint": audit_payload.get("expansion_fingerprint"),
            }
        payload = {
            "directory": str(prepared.directory),
            "fingerprints": dict(prepared.fingerprints),
            "dimensions": prepared.dimensions,
            "phase_diagnostics": list(prepared.phase_diagnostics),
            "expansion": expansion_report,
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
    resume: bool = False,
    fresh: bool = False,
) -> dict[str, object]:
    config = load_case_study_config(config_path, case_root=case_root)
    if (resume or fresh) and stage != "expand-od":
        raise ValueError("--resume/--fresh are only valid with the expand-od stage")
    if dry_run:
        if resume or fresh:
            raise ValueError("--resume/--fresh are only valid for a real expand-od run")
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
        "expand-od": lambda: runner.expand_od(resume=resume, fresh=fresh),
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
    parser.add_argument("--resume", action="store_true", help="resume the matching expand-od checkpoint")
    parser.add_argument("--fresh", action="store_true", help="archive a matching checkpoint and start expand-od afresh")
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
            resume=args.resume,
            fresh=args.fresh,
        )
    except ODTimeExpansionInterrupted as error:
        print(json.dumps({"status": "interrupted", "error": str(error), "checkpoint_directory": str(error.checkpoint_directory)}, sort_keys=True), file=sys.stderr)
        return error.exit_code
    except Exception as error:  # CLI boundary: preserve a non-zero status.
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
