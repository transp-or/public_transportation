"""End-to-end TOML-driven structural-zero preprocessing service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from public_transportation.domain.scenario import Scenario

from .classification import analyze_structural_zeros
from .config import StructuralZeroConfig, load_structural_zero_config
from .persistence import StructuralZeroOutputPaths, write_structural_zero_outputs
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
) -> StructuralZeroExecutionResult:
    """Execute the complete workflow using only a TOML configuration path."""
    config = load_structural_zero_config(config_file)
    scenario = Scenario.from_folder(
        config.scenario.folder,
        strict=True,
        demand_file=config.scenario.demand_file,
    )
    scenario_fingerprint = fingerprint_scenario(scenario)
    candidate_keys = _scenario_demand_keys(scenario)
    topology = build_structural_zero_topology(scenario, config.assignment)
    analysis = analyze_structural_zeros(
        topology,
        config,
        scenario_fingerprint=scenario_fingerprint,
        keys=candidate_keys,
    )
    reconciliation = load_and_reconcile_fixed_demand(
        analysis,
        config,
        scenario=scenario,
    )
    outputs = write_structural_zero_outputs(analysis, reconciliation, config)
    return StructuralZeroExecutionResult(
        config=config,
        scenario_fingerprint=scenario_fingerprint,
        topology=topology,
        analysis=analysis,
        reconciliation=reconciliation,
        outputs=outputs,
    )


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
