"""End-to-end TOML-driven structural-zero preprocessing service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

from public_transportation.domain.scenario import Scenario

from .classification import analyze_structural_zeros
from ..canonical_timetable import build_canonical_timetable_index
from .config import StructuralZeroConfig, load_structural_zero_config
from .persistence import StructuralZeroOutputPaths, write_structural_zero_outputs
from .progress import StructuralZeroProgress, emit_phase
from .reconciliation import (
    FixedDemandReconciliationResult,
    load_and_reconcile_fixed_demand,
)
from .scenario_fingerprint import fingerprint_scenario
from .topology import StructuralZeroTopology, build_structural_zero_topology
from .types import ODTimeKey, StructuralZeroAnalysisResult


@dataclass(frozen=True, slots=True)
class StructuralZeroExecutionResult:
    """Complete in-memory and persisted result of one preprocessing run."""

    config: StructuralZeroConfig
    scenario_fingerprint: str
    topology: StructuralZeroTopology
    analysis: StructuralZeroAnalysisResult
    reconciliation: FixedDemandReconciliationResult
    outputs: StructuralZeroOutputPaths


def run_structural_zero_preprocessing(
    config_file: str | Path,
    *,
    progress: Callable[[StructuralZeroProgress], None] | None = None,
) -> StructuralZeroExecutionResult:
    """Execute the complete workflow using only a TOML configuration path."""
    started = perf_counter()
    emit_phase(progress, "load_scenario", completed=0)
    config = load_structural_zero_config(config_file)
    scenario = Scenario.from_folder(
        config.scenario.folder,
        strict=True,
        demand_file=config.scenario.demand_file,
    )
    scenario_fingerprint = fingerprint_scenario(scenario)
    candidate_keys = _scenario_demand_keys(scenario)
    emit_phase(progress, "load_scenario", completed=1, started=started)
    phase_started = perf_counter()
    emit_phase(progress, "build_topology", completed=0)
    timetable_index = build_canonical_timetable_index(scenario)
    topology = build_structural_zero_topology(
        scenario,
        config.assignment,
        timetable_index=timetable_index,
    )
    emit_phase(progress, "build_topology", completed=1, started=phase_started)
    analysis = analyze_structural_zeros(
        topology,
        config,
        scenario_fingerprint=scenario_fingerprint,
        keys=candidate_keys,
        progress=progress,
    )
    phase_started = perf_counter()
    emit_phase(progress, "reconcile_fixed_demand", completed=0)
    reconciliation = load_and_reconcile_fixed_demand(
        analysis,
        config,
        scenario=scenario,
    )
    emit_phase(
        progress, "reconcile_fixed_demand", completed=1, started=phase_started
    )
    outputs = write_structural_zero_outputs(
        analysis, reconciliation, config, progress=progress
    )
    result = StructuralZeroExecutionResult(
        config=config,
        scenario_fingerprint=scenario_fingerprint,
        topology=topology,
        analysis=analysis,
        reconciliation=reconciliation,
        outputs=outputs,
    )
    emit_phase(progress, "complete", completed=1, started=started)
    return result


def _scenario_demand_keys(scenario: Scenario) -> tuple[ODTimeKey, ...]:
    keys = tuple(
        ODTimeKey(
            str(record.origin_stop_id),
            str(record.dest_stop_id),
            str(record.time_bin_id),
        )
        for record in scenario.demand.records
    )
    if not keys:
        raise ValueError("Scenario demand must contain at least one OD/time cell.")
    if len(keys) != len(set(keys)):
        raise ValueError("Scenario demand contains duplicate OD/time keys.")
    return tuple(sorted(keys))
