"""Atomic, versioned checkpoint journal for block-coordinate estimation."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ._canonical import canonical_json
from .checkpoint import (
    BLOCK_COORDINATE_CHECKPOINT_SCHEMA_VERSION,
    BlockCoordinateFingerprints,
)
from .incremental import BlockUpdateProposal
from .progress import DiagnosticValue
from .results import (
    BlockConvergenceDiagnostics,
    BlockCoordinateState,
    BlockObjectiveComponents,
)

COMPACT_CHECKPOINT_NAME = "compact-checkpoint.json"


def _diagnostic(payload: dict) -> DiagnosticValue:
    return DiagnosticValue(
        value=payload["value"],
        kind=payload["kind"],
        computed_at_sweep=payload["computed_at_sweep"],
    )


def _state_payload(state: BlockCoordinateState) -> dict:
    return {
        "current_free_flow": state.current_free_flow.tolist(),
        "best_free_flow": state.best_free_flow.tolist(),
        "current_prediction": state.current_prediction.tolist(),
        "fixed_measurement_offset": state.fixed_measurement_offset.tolist(),
        "current_objective": state.current_objective,
        "best_objective": state.best_objective,
        "current_components": asdict(state.current_components),
        "best_components": asdict(state.best_components),
        "sweep": state.sweep,
        "schedule_position": state.schedule_position,
        "accepted_updates": state.accepted_updates,
        "rejected_updates": state.rejected_updates,
        "elapsed_seconds": state.elapsed_seconds,
        "block_schedule": list(state.block_schedule),
        "random_state_json": state.random_state_json,
        "diagnostics": asdict(state.diagnostics),
    }


def _state_from_payload(
    payload: dict, fingerprints: BlockCoordinateFingerprints
) -> BlockCoordinateState:
    diagnostics = payload["diagnostics"]
    return BlockCoordinateState(
        current_free_flow=payload["current_free_flow"],
        best_free_flow=payload["best_free_flow"],
        current_prediction=payload["current_prediction"],
        fixed_measurement_offset=payload["fixed_measurement_offset"],
        current_objective=payload["current_objective"],
        best_objective=payload["best_objective"],
        current_components=BlockObjectiveComponents(**payload["current_components"]),
        best_components=BlockObjectiveComponents(**payload["best_components"]),
        sweep=payload["sweep"],
        schedule_position=payload["schedule_position"],
        accepted_updates=payload["accepted_updates"],
        rejected_updates=payload["rejected_updates"],
        elapsed_seconds=payload["elapsed_seconds"],
        block_schedule=tuple(payload["block_schedule"]),
        random_state_json=payload["random_state_json"],
        diagnostics=BlockConvergenceDiagnostics(
            latest_block_projected_gradient=_diagnostic(
                diagnostics["latest_block_projected_gradient"]
            ),
            estimated_global_projected_gradient=_diagnostic(
                diagnostics["estimated_global_projected_gradient"]
            ),
            exact_global_projected_gradient=_diagnostic(
                diagnostics["exact_global_projected_gradient"]
            ),
            maximum_block_flow_change=diagnostics["maximum_block_flow_change"],
            initialization_objective_improvement=diagnostics[
                "initialization_objective_improvement"
            ],
            current_sweep_objective_improvement=diagnostics[
                "current_sweep_objective_improvement"
            ],
            previous_sweep_objective_improvement=diagnostics[
                "previous_sweep_objective_improvement"
            ],
        ),
        fingerprints=fingerprints,
    )


def _atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid checkpoint JSON from {path}.") from error
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint payload in {path} must be a JSON object.")
    return value


class BlockCheckpointStore:
    """Durable compact checkpoint plus committed incremental journal."""

    def __init__(
        self,
        directory: Path | str,
        fingerprints: BlockCoordinateFingerprints,
    ) -> None:
        self.directory = Path(directory).expanduser()
        self.fingerprints = fingerprints
        self.journal_sequence = 0

    @property
    def compact_path(self) -> Path:
        return self.directory / COMPACT_CHECKPOINT_NAME

    def initialize(self, state: BlockCoordinateState) -> None:
        if self.compact_path.exists():
            raise FileExistsError(
                f"checkpoint already exists at {self.compact_path}; resume it "
                "or choose a new checkpoint directory"
            )
        self.journal_sequence = 0
        self._write_compact(state)

    def _write_compact(self, state: BlockCoordinateState) -> None:
        payload = {
            "schema_version": BLOCK_COORDINATE_CHECKPOINT_SCHEMA_VERSION,
            "kind": "compact",
            "fingerprints": asdict(self.fingerprints),
            "fingerprint": self.fingerprints.fingerprint,
            "journal_sequence": self.journal_sequence,
            "state": _state_payload(state),
        }
        _atomic_write(self.compact_path, payload)

    def append_accepted_update(
        self,
        *,
        proposal: BlockUpdateProposal,
        objective_before: float,
        state_after: BlockCoordinateState,
        best_solution_updated: bool,
    ) -> int:
        sequence = self.journal_sequence + 1
        record = {
            "schema_version": BLOCK_COORDINATE_CHECKPOINT_SCHEMA_VERSION,
            "kind": "accepted_block_update",
            "sequence": sequence,
            "fingerprint": self.fingerprints.fingerprint,
            "block_id": proposal.block_id,
            "block_fingerprint": proposal.block_fingerprint,
            "free_column_indices": list(proposal.free_column_indices),
            "flow_before": proposal.flow_before.tolist(),
            "flow_after": proposal.flow_after.tolist(),
            "prediction_delta": proposal.prediction_delta.tolist(),
            "objective_before": objective_before,
            "objective_after": state_after.current_objective,
            "best_solution_updated": best_solution_updated,
            "state_metadata": {
                key: value
                for key, value in _state_payload(state_after).items()
                if key
                not in {
                    "current_free_flow",
                    "current_prediction",
                    "fixed_measurement_offset",
                    "best_free_flow",
                }
            },
        }
        journal = self.directory / f"journal-{sequence:012d}.json"
        commit = self.directory / f"journal-{sequence:012d}.commit"
        _atomic_write(journal, record)
        digest = hashlib.sha256(journal.read_bytes()).hexdigest()
        _atomic_write(
            commit,
            {
                "schema_version": BLOCK_COORDINATE_CHECKPOINT_SCHEMA_VERSION,
                "sequence": sequence,
                "journal_sha256": digest,
            },
        )
        self.journal_sequence = sequence
        return sequence

    def compact(self, state: BlockCoordinateState) -> None:
        self._write_compact(state)
        for sequence in range(1, self.journal_sequence + 1):
            for suffix in ("json", "commit"):
                path = self.directory / f"journal-{sequence:012d}.{suffix}"
                if path.exists():
                    path.unlink()

    def load(self) -> BlockCoordinateState:
        compact = _read_json(self.compact_path)
        if compact.get("schema_version") != BLOCK_COORDINATE_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported block-coordinate checkpoint schema version.")
        if compact.get("fingerprint") != self.fingerprints.fingerprint:
            raise ValueError("checkpoint fingerprints do not match the requested run.")
        stored_fingerprints = BlockCoordinateFingerprints(**compact["fingerprints"])
        if stored_fingerprints != self.fingerprints:
            raise ValueError("checkpoint authoritative fingerprints do not match.")
        state = _state_from_payload(compact["state"], self.fingerprints)
        self.journal_sequence = int(compact["journal_sequence"])
        sequence = self.journal_sequence + 1
        while True:
            journal = self.directory / f"journal-{sequence:012d}.json"
            commit = self.directory / f"journal-{sequence:012d}.commit"
            if not journal.exists() or not commit.exists():
                break
            marker = _read_json(commit)
            digest = hashlib.sha256(journal.read_bytes()).hexdigest()
            if marker.get("sequence") != sequence or marker.get("journal_sha256") != digest:
                raise ValueError(f"journal commit marker {sequence} is invalid.")
            record = _read_json(journal)
            if record.get("fingerprint") != self.fingerprints.fingerprint:
                raise ValueError(f"journal {sequence} fingerprint does not match.")
            if record.get("sequence") != sequence:
                raise ValueError(f"journal sequence {sequence} is inconsistent.")
            columns = np.asarray(record["free_column_indices"], dtype=np.intp)
            flow = np.array(state.current_free_flow, copy=True)
            before = np.asarray(record["flow_before"], dtype=float)
            if not np.array_equal(flow[columns], before):
                raise ValueError(f"journal {sequence} cannot be replayed from this state.")
            flow[columns] = np.asarray(record["flow_after"], dtype=float)
            prediction = state.current_prediction + np.asarray(
                record["prediction_delta"], dtype=float
            )
            metadata = record["state_metadata"]
            payload = {
                **metadata,
                "current_free_flow": flow.tolist(),
                "current_prediction": prediction.tolist(),
                "fixed_measurement_offset": state.fixed_measurement_offset.tolist(),
                "best_free_flow": (
                    flow.tolist()
                    if record["best_solution_updated"]
                    else state.best_free_flow.tolist()
                ),
            }
            state = _state_from_payload(payload, self.fingerprints)
            self.journal_sequence = sequence
            sequence += 1
        return state
