"""Versioned metadata contracts for durable block-coordinate checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

from ._canonical import fingerprint

BLOCK_COORDINATE_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BlockCoordinateFingerprints:
    scenario: str
    assignment_inputs: str
    od_layout: str
    fixed_demand: str
    measurements: str
    prior: str
    routing: str
    partition: str
    solver_semantics: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} fingerprint must be nonempty.")

    @property
    def fingerprint(self) -> str:
        return fingerprint(self)


@dataclass(frozen=True, slots=True)
class BlockCheckpointMetadata:
    fingerprints: BlockCoordinateFingerprints
    checkpoint_sequence: int
    journal_sequence: int
    sweep: int
    schedule_position: int
    committed: bool
    schema_version: int = BLOCK_COORDINATE_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BLOCK_COORDINATE_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported block-coordinate checkpoint schema version.")
        for name in (
            "checkpoint_sequence",
            "journal_sequence",
            "sweep",
            "schedule_position",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
