"""Sparse, deterministic input contract for frozen OD demand cells."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .scenario import Scenario


FixedODKey = tuple[str, str, str]

REQUIRED_COLUMNS = ("origin_stop_id", "dest_stop_id", "time_bin_id")
OPTIONAL_COLUMNS = ("fixed_flow",)
ALLOWED_COLUMNS = frozenset((*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS))


@dataclass(frozen=True, slots=True, order=True)
class FixedODRecord:
    """One OD/time-bin cell whose demand is fixed rather than estimated."""

    origin_stop_id: str
    dest_stop_id: str
    time_bin_id: str
    fixed_flow: float = 0.0

    @property
    def key(self) -> FixedODKey:
        """Return the complete logical key used for strict alignment."""
        return (self.origin_stop_id, self.dest_stop_id, self.time_bin_id)


@dataclass(frozen=True, slots=True)
class FixedODDemand:
    """Canonical, key-sorted collection of sparse fixed-demand records."""

    records: tuple[FixedODRecord, ...]

    def __len__(self) -> int:
        return len(self.records)

    def as_dict(self) -> dict[FixedODKey, float]:
        """Return fixed values keyed by ``(origin, destination, time_bin)``."""
        return {record.key: record.fixed_flow for record in self.records}


def _parse_identifier(*, value: str | None, column: str, row_number: int) -> str:
    parsed = "" if value is None else str(value).strip()
    if not parsed:
        raise ValueError(
            f"fixed demand row {row_number}: {column} must be a non-empty identifier."
        )
    return parsed


def _parse_fixed_flow(*, value: str | None, row_number: int) -> float:
    text = "" if value is None else str(value).strip()
    if text == "":
        return 0.0
    try:
        flow = float(text)
    except ValueError as error:
        raise ValueError(
            f"fixed demand row {row_number}: fixed_flow must be numeric, got {text!r}."
        ) from error
    if not math.isfinite(flow):
        raise ValueError(
            f"fixed demand row {row_number}: fixed_flow must be finite, got {text!r}."
        )
    if flow < 0.0:
        raise ValueError(
            f"fixed demand row {row_number}: fixed_flow must be non-negative, got {flow}."
        )
    return flow


def _validate_key_against_scenario(
    *,
    key: FixedODKey,
    stop_ids: set[str],
    time_bin_ids: set[str],
    demand_keys: set[FixedODKey],
    row_number: int,
) -> None:
    origin, destination, time_bin_id = key
    if origin not in stop_ids:
        raise ValueError(
            f"fixed demand row {row_number}: unknown origin_stop_id {origin!r}."
        )
    if destination not in stop_ids:
        raise ValueError(
            f"fixed demand row {row_number}: unknown dest_stop_id {destination!r}."
        )
    if time_bin_id not in time_bin_ids:
        raise ValueError(
            f"fixed demand row {row_number}: unknown time_bin_id {time_bin_id!r}."
        )

    if key not in demand_keys:
        raise ValueError(
            f"fixed demand row {row_number}: OD/time-bin key {key!r} is not present "
            "in scenario.demand.records."
        )


def read_fixed_demand_csv(
    path: str | Path,
    *,
    scenario: Scenario,
) -> FixedODDemand:
    """Read and strictly validate a sparse fixed-demand CSV file.

    Every data row freezes one complete ``(origin, destination, time-bin)`` key.
    Missing or blank ``fixed_flow`` values default to zero. Records are returned
    in canonical key order, independently of the order of rows in the file.
    """
    file_path = Path(path)
    stop_ids = {str(stop.stop_id) for stop in scenario.stops}
    time_bin_ids = {str(time_bin.bin_id) for time_bin in scenario.time_bins}
    demand_keys: set[FixedODKey] = {
        (str(record.origin_stop_id), str(record.dest_stop_id), str(record.time_bin_id))
        for record in scenario.demand.records
    }
    with file_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("fixed demand CSV must contain a header row.")

        fieldnames = [str(name).strip() for name in reader.fieldnames]
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("fixed demand CSV contains duplicate column names.")
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(f"fixed demand CSV is missing required columns: {missing}.")
        unexpected = sorted(set(fieldnames) - ALLOWED_COLUMNS)
        if unexpected:
            raise ValueError(f"fixed demand CSV contains unexpected columns: {unexpected}.")

        # DictReader retains the original header strings as keys. Normalize them
        # once so headers with harmless surrounding whitespace remain usable.
        header_map = dict(zip(reader.fieldnames, fieldnames, strict=True))
        seen: dict[FixedODKey, int] = {}
        records: list[FixedODRecord] = []

        for row_number, raw_row in enumerate(reader, start=2):
            extra_values = raw_row.get(None)
            if extra_values and any(str(value).strip() for value in extra_values):
                raise ValueError(
                    f"fixed demand row {row_number}: row contains more values than the header."
                )
            row = {header_map[key]: value for key, value in raw_row.items() if key is not None}
            origin = _parse_identifier(
                value=row.get("origin_stop_id"),
                column="origin_stop_id",
                row_number=row_number,
            )
            destination = _parse_identifier(
                value=row.get("dest_stop_id"),
                column="dest_stop_id",
                row_number=row_number,
            )
            time_bin_id = _parse_identifier(
                value=row.get("time_bin_id"),
                column="time_bin_id",
                row_number=row_number,
            )
            fixed_flow = _parse_fixed_flow(
                value=row.get("fixed_flow"),
                row_number=row_number,
            )
            key = (origin, destination, time_bin_id)
            if key in seen:
                raise ValueError(
                    f"fixed demand row {row_number}: duplicate OD/time-bin key {key!r}; "
                    f"first defined on row {seen[key]}."
                )
            _validate_key_against_scenario(
                key=key,
                stop_ids=stop_ids,
                time_bin_ids=time_bin_ids,
                demand_keys=demand_keys,
                row_number=row_number,
            )
            seen[key] = row_number
            records.append(
                FixedODRecord(
                    origin_stop_id=origin,
                    dest_stop_id=destination,
                    time_bin_id=time_bin_id,
                    fixed_flow=fixed_flow,
                )
            )

    records.sort(key=lambda record: record.key)
    return FixedODDemand(records=tuple(records))
