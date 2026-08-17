"""Independent candidate OD universes, OD--time expansion, and priors.

This module deliberately keeps three concepts separate:

* an ordered origin/destination pair universe;
* the approved time-bin expansion of that universe; and
* numerical prior values used by a statistical model.

None of the pair-level fingerprints include ``Scenario.time_bins``.  This is
important when a case owner changes the temporal resolution after reviewing
the count timestamps: the OD universe remains the same and only the later
OD--time expansion is invalidated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import signal
import tempfile
import time
from bisect import bisect_left
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Mapping, Sequence

from public_transportation.domain import Scenario

if TYPE_CHECKING:  # pragma: no cover - import only for static analysis
    from .canonical_timetable import CanonicalTimetableIndex



def canonical_json(value: object) -> str:
    """Serialize a fingerprint payload without a retired backend dependency."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


ODUniverseSource = Literal["file", "network_ordered_pairs"]
ODUniverseLevel = Literal["stop", "physical_stop"]
ConnectivityPolicy = Literal["none", "directed_reachable"]
TimetablePolicy = Literal["required", "defer", "none"]
EXPANSION_ALGORITHM_VERSION = 1


def _text(value: Any, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} must be a non-empty identifier.")
    parsed = str(value).strip()
    if not parsed:
        raise ValueError(f"{name} must be a non-empty identifier.")
    return parsed


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class CandidateODPair:
    """One ordered candidate origin/destination pair."""

    origin_stop_id: str
    destination_stop_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "origin_stop_id", _text(self.origin_stop_id, "origin_stop_id")
        )
        object.__setattr__(
            self,
            "destination_stop_id",
            _text(self.destination_stop_id, "destination_stop_id"),
        )

    @property
    def tuple(self) -> tuple[str, str]:
        return (self.origin_stop_id, self.destination_stop_id)


@dataclass(frozen=True, slots=True)
class ODUniverseExclusion:
    """Audit record for one pair removed by an explicit rule."""

    origin_stop_id: str
    destination_stop_id: str
    reason: str
    detail: str = ""

    @property
    def tuple(self) -> tuple[str, str, str, str]:
        return (
            self.origin_stop_id,
            self.destination_stop_id,
            self.reason,
            self.detail,
        )


@dataclass(frozen=True, slots=True)
class CandidateODUniverse:
    """Validated immutable pair universe and its exclusion audit."""

    pairs: tuple[CandidateODPair, ...]
    exclusions: tuple[ODUniverseExclusion, ...]
    source: ODUniverseSource
    level: ODUniverseLevel
    include_same_stop: bool
    active_service_only: bool
    connectivity_policy: ConnectivityPolicy
    physical_stop_mapping: Mapping[str, str]
    generator_fingerprint: str
    directed_reachability: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.pairs != tuple(sorted(self.pairs)):
            raise ValueError("candidate OD pairs must use canonical sorted order.")
        if len(set(self.pairs)) != len(self.pairs):
            raise ValueError("candidate OD pairs must be unique.")
        if self.source not in {"file", "network_ordered_pairs"}:
            raise ValueError("unsupported OD-universe source.")
        if self.level not in {"stop", "physical_stop"}:
            raise ValueError("unsupported OD-universe level.")
        if self.connectivity_policy not in {"none", "directed_reachable"}:
            raise ValueError("unsupported connectivity policy.")

    @property
    def pair_count(self) -> int:
        return len(self.pairs)

    @property
    def fingerprint(self) -> str:
        return _sha256_payload(
            {
                "generator_fingerprint": self.generator_fingerprint,
                "level": self.level,
                "pairs": [list(pair.tuple) for pair in self.pairs],
            }
        )

    @property
    def audit(self) -> dict[str, object]:
        counts: dict[str, int] = defaultdict(int)
        for exclusion in self.exclusions:
            counts[exclusion.reason] += 1
        return {
            "source": self.source,
            "level": self.level,
            "include_same_stop": self.include_same_stop,
            "active_service_only": self.active_service_only,
            "connectivity_policy": self.connectivity_policy,
            "input_pair_count": self.pair_count + len(self.exclusions),
            "retained_pair_count": self.pair_count,
            "exclusion_counts": dict(sorted(counts.items())),
            "fingerprint": self.fingerprint,
            "generator_fingerprint": self.generator_fingerprint,
        }


@dataclass(frozen=True, slots=True, order=True)
class CandidateODTimeCell:
    """One candidate OD pair assigned to one approved time interval."""

    origin_stop_id: str
    destination_stop_id: str
    time_bin_id: str

    @property
    def tuple(self) -> tuple[str, str, str]:
        return (self.origin_stop_id, self.destination_stop_id, self.time_bin_id)


@dataclass(frozen=True, slots=True)
class ODTimeExclusion:
    """Audit record for one pair/time-bin cell removed by a rule."""

    origin_stop_id: str
    destination_stop_id: str
    time_bin_id: str
    reason: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ODTimeExpansion:
    """Immutable expanded candidate cells and complete exclusion audit."""

    universe_fingerprint: str
    cells: tuple[CandidateODTimeCell, ...]
    exclusions: tuple[ODTimeExclusion, ...]
    time_bins: tuple[tuple[str, int, int], ...]
    policies: Mapping[str, object]
    fingerprint: str

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    @property
    def audit(self) -> dict[str, object]:
        counts: dict[str, int] = defaultdict(int)
        for exclusion in self.exclusions:
            counts[exclusion.reason] += 1
        expanded_count = len(self.cells) + len(self.exclusions)
        pair_count = (
            expanded_count // len(self.time_bins)
            if self.time_bins
            else 0
        )
        return {
            "input_pair_count": pair_count,
            "time_bin_count": len(self.time_bins),
            "expanded_od_time_count": expanded_count,
            "retained_cell_count": len(self.cells),
            "exclusion_counts": dict(sorted(counts.items())),
            "universe_fingerprint": self.universe_fingerprint,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ODTimeExpansionRunResult:
    """Summary of a checkpointed OD--time expansion.

    Rows are deliberately not retained here.  The durable JSONL chunks are
    the source of truth and can be streamed by downstream stages.
    """

    checkpoint_directory: Path
    expansion_fingerprint: str
    semantic_checksum: str
    status: str
    total_chunks: int
    completed_chunks: int
    total_cells: int
    retained_cells: int
    excluded_cells: int
    next_chunk: int
    checkpoint_reused: bool = False


class ODTimeExpansionInterrupted(RuntimeError):
    """Raised when checkpointed expansion is interrupted safely."""

    exit_code = 130

    def __init__(self, checkpoint_directory: Path, message: str = "OD-time expansion interrupted") -> None:
        super().__init__(message)
        self.checkpoint_directory = checkpoint_directory


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _atomic_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    maximum_bytes: int,
) -> str:
    """Atomically write one bounded chunk without a second serialized copy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    written = 0
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            for row in rows:
                line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                encoded = line.encode("utf-8")
                written += len(encoded)
                if written > maximum_bytes:
                    raise MemoryError("one expansion chunk exceeds expansion.maximum_temporary_bytes")
                stream.write(line)
                digest.update(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return digest.hexdigest()


def _memory_bytes() -> int | None:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if platform.system() == "Darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return None


def _semantic_checksum(rows: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            canonical_json(
                [
                    str(row["origin_stop_id"]),
                    str(row["destination_stop_id"]),
                    str(row["time_bin_id"]),
                    str(row["status"]),
                    str(row.get("reason", "")),
                ]
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expansion_contract_fingerprint(
    universe: CandidateODUniverse,
    time_periods: Sequence[object],
    configuration: Mapping[str, object] | None = None,
) -> str:
    """Return the deterministic contract fingerprint for checkpoint reuse."""
    bins = tuple(_period_tuple(period) for period in time_periods)
    payload = {
        "algorithm_version": EXPANSION_ALGORITHM_VERSION,
        "universe_fingerprint": universe.fingerprint,
        "time_bins": [list(item) for item in bins],
        "configuration": dict(configuration or {}),
    }
    return _sha256_payload(payload)


def _expansion_rows_for_pair(
    pair: CandidateODPair,
    bins: tuple[tuple[str, int, int], ...],
    universe: CandidateODUniverse,
    *,
    scenario: Scenario | None,
    departures: set[str],
    arrivals: set[str],
    reachable: Mapping[str, set[str]],
    maximum_transfers: int,
    maximum_initial_wait_seconds: int,
    maximum_journey_seconds: int,
    maximum_waiting_seconds: int,
    timetable_policy: TimetablePolicy,
    timetable_feasibility: Callable[[CandidateODPair, tuple[str, int, int]], bool] | None,
    feasibility_index: TimetableFeasibilityIndex | None,
    feasibility_contract: ScheduledFeasibilityContract | None,
    timetable_index: "CanonicalTimetableIndex | None",
    feasible_destinations_by_origin_period: Mapping[
        tuple[str, str], frozenset[str]
    ] | None = None,
) -> list[dict[str, object]]:
    pair_reason: str | None = None
    if pair.origin_stop_id == pair.destination_stop_id and not universe.include_same_stop:
        pair_reason = "same_node"
    elif universe.active_service_only and scenario is not None and pair.origin_stop_id not in departures:
        pair_reason = "inactive_origin"
    elif universe.active_service_only and scenario is not None and pair.destination_stop_id not in arrivals:
        pair_reason = "inactive_destination"
    elif (
        universe.connectivity_policy == "directed_reachable"
        and scenario is not None
        and pair.destination_stop_id not in reachable.get(pair.origin_stop_id, set())
    ):
        pair_reason = "static_unreachable"
    rows: list[dict[str, object]] = []
    for period in bins:
        row: dict[str, object] = {
            "origin_stop_id": pair.origin_stop_id,
            "destination_stop_id": pair.destination_stop_id,
            "time_bin_id": period[0],
            "status": "excluded",
            "reason": pair_reason or "",
            "detail": "",
        }
        if pair_reason is None:
            if timetable_feasibility is not None:
                feasible: bool | None = bool(timetable_feasibility(pair, period))
            elif timetable_policy == "none":
                feasible = True
            elif timetable_policy == "defer":
                feasible = None
            elif scenario is None or scenario.timetable is None:
                feasible = False
            elif feasible_destinations_by_origin_period is not None:
                origin = universe.physical_stop_mapping.get(
                    pair.origin_stop_id, pair.origin_stop_id
                )
                destination = universe.physical_stop_mapping.get(
                    pair.destination_stop_id, pair.destination_stop_id
                )
                feasible = destination in feasible_destinations_by_origin_period[
                    (origin, period[0])
                ]
            elif feasibility_index is not None:
                if feasibility_contract is None:
                    feasible = feasibility_index.is_feasible(
                        pair,
                        period,
                        maximum_transfers=maximum_transfers,
                        maximum_initial_wait_seconds=maximum_initial_wait_seconds,
                        maximum_waiting_seconds=maximum_waiting_seconds,
                        maximum_journey_seconds=maximum_journey_seconds,
                    )
                else:
                    feasible = feasibility_contract.is_feasible(
                        feasibility_index, pair, period
                    )
            else:
                feasible = _timetable_feasible(
                    scenario,
                    pair,
                    period,
                    mapping=universe.physical_stop_mapping,
                    maximum_transfers=maximum_transfers,
                    maximum_initial_wait_seconds=maximum_initial_wait_seconds,
                    maximum_journey_seconds=maximum_journey_seconds,
                    maximum_waiting_seconds=maximum_waiting_seconds,
                    timetable_index=timetable_index,
                )
            if feasible is True or feasible is None:
                row["status"] = "retained"
                row["reason"] = ""
            else:
                row["reason"] = "timetable_infeasible"
        rows.append(row)
    return rows


def run_candidate_od_time_expansion(
    universe: CandidateODUniverse,
    time_periods: Sequence[object],
    scenario: Scenario | None = None,
    feasibility_index: TimetableFeasibilityIndex | None = None,
    configuration: Mapping[str, object] | None = None,
    checkpoint_directory: str | Path | None = None,
    resume: bool = False,
    progress: Callable[[Mapping[str, object]], None] | None = None,
    timetable_feasibility: Callable[[CandidateODPair, tuple[str, int, int]], bool] | None = None,
    timetable_index: "CanonicalTimetableIndex | None" = None,
) -> ODTimeExpansionRunResult:
    """Expand candidate cells in deterministic, resumable pair chunks.

    Only one chunk is materialized in memory.  Completed chunks are immutable
    JSONL files, each accompanied by a checksum in the atomic manifest.
    """
    if configuration is None:
        config: dict[str, object] = {}
    elif isinstance(configuration, Mapping):
        config = dict(configuration)
    elif is_dataclass(configuration):
        config = dict(asdict(configuration))
    else:
        config = dict(vars(configuration))
    bins = tuple(_period_tuple(period) for period in time_periods)
    if not bins:
        raise ValueError("at least one approved time period is required")
    if bins != tuple(sorted(bins, key=lambda item: (item[1], item[2], item[0]))):
        raise ValueError("approved time periods must be sorted by start/end/id")
    if any(left[2] > right[1] for left, right in zip(bins, bins[1:], strict=False)):
        raise ValueError("approved time periods must not overlap")
    chunk_size = int(config.get("chunk_size_pairs", 512))
    interval = float(config.get("progress_interval_seconds", 5.0))
    maximum_temporary_bytes = int(config.get("maximum_temporary_bytes", 8 * 1024**3))
    if chunk_size <= 0 or interval <= 0 or maximum_temporary_bytes <= 0:
        raise ValueError("expansion chunk and resource settings must be positive")
    if checkpoint_directory is None:
        raise ValueError("checkpoint_directory is required")
    checkpoint = Path(checkpoint_directory).expanduser().resolve()
    checkpoint.mkdir(parents=True, exist_ok=True)
    feasibility_contract = ScheduledFeasibilityContract.from_mapping(config)
    config.setdefault("feasibility_contract_version", feasibility_contract.version)
    config.setdefault(
        "feasibility_contract_fingerprint", feasibility_contract.fingerprint
    )
    expansion_fingerprint = expansion_contract_fingerprint(universe, bins, config)
    manifest_path = checkpoint / "manifest.json"
    progress_path = checkpoint / "progress.json"
    if manifest_path.exists() and not resume:
        raise FileExistsError(
            f"checkpoint already exists at {checkpoint}; pass --resume or explicitly archive it before a fresh run"
        )
    if not manifest_path.exists() and not resume and any(checkpoint.iterdir()):
        raise FileExistsError(
            f"checkpoint directory contains existing files at {checkpoint}; pass --resume only for a valid manifest"
        )
    if resume and not manifest_path.exists():
        raise FileNotFoundError(f"cannot resume without checkpoint manifest: {manifest_path}")
    total_chunks = (len(universe.pairs) + chunk_size - 1) // chunk_size
    total_rows = len(universe.pairs) * len(bins)
    departures, arrivals = (
        (set(), set())
        if scenario is None
        else _service_activity(
            scenario,
            universe.physical_stop_mapping,
            timetable_index=timetable_index,
        )
    )
    if universe.directed_reachability:
        reachable = {
            origin: set(destinations)
            for origin, destinations in universe.directed_reachability.items()
        }
    else:
        reachable = (
            {}
            if scenario is None
            else _directed_reachability(
                scenario,
                universe.physical_stop_mapping,
                timetable_index=timetable_index,
            )
        )
    if (
        timetable_feasibility is None
        and feasibility_index is None
        and config.get("timetable_policy", "required") == "required"
        and scenario is not None
        and scenario.timetable is not None
    ):
        feasibility_index = TimetableFeasibilityIndex.from_scenario(
            scenario,
            physical_stop_mapping=universe.physical_stop_mapping,
            timetable_index=timetable_index,
        )
    policy = str(config.get("timetable_policy", "required"))
    if policy not in {"required", "defer", "none"}:
        raise ValueError("unsupported timetable_policy")
    limits = {
        "maximum_transfers": int(config.get("maximum_transfers", 2)),
        "maximum_initial_wait_seconds": int(config.get("maximum_initial_wait_seconds", 3600)),
        "maximum_journey_seconds": int(config.get("maximum_journey_seconds", 7200)),
        "maximum_waiting_seconds": int(config.get("maximum_waiting_seconds", 3600)),
    }
    if any(value < 0 for value in limits.values()) or limits["maximum_journey_seconds"] <= 0:
        raise ValueError("feasibility limits must be non-negative (journey time positive)")
    feasible_destinations_by_origin_period = None
    if (
        timetable_feasibility is None
        and feasibility_index is not None
        and policy == "required"
        and scenario is not None
        and scenario.timetable is not None
    ):
        feasible_destinations_by_origin_period = _precompute_temporal_feasibility(
            universe=universe,
            bins=bins,
            feasibility_index=feasibility_index,
            feasibility_contract=feasibility_contract,
            **limits,
        )
    contract = {
        "status": "running",
        "algorithm_version": EXPANSION_ALGORITHM_VERSION,
        "expansion_fingerprint": expansion_fingerprint,
        "configuration": config,
        "configuration_fingerprint": config.get("configuration_fingerprint"),
        "package_revision": config.get("package_revision"),
        "scenario_checksums": config.get("scenario_checksums", config.get("source_checksums", {})),
        "od_universe_fingerprint": universe.fingerprint,
        "approved_time_bins_fingerprint": config.get("approved_time_bins_fingerprint"),
        "approved_time_bins": [list(item) for item in bins],
        "chunk_size_pairs": chunk_size,
        "total_pairs": len(universe.pairs),
        "total_time_bins": len(bins),
        "total_rows": total_rows,
        "total_chunks": total_chunks,
        "completed_chunks": [],
        "completed_cells": 0,
        "retained_cells": 0,
        "excluded_cells": 0,
        "next_chunk": 0,
        "semantic_checksum": None,
    }
    reused = bool(resume)
    completed: set[int] = set()
    chunk_checksums: dict[str, str] = {}
    if resume:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("expansion_fingerprint") != expansion_fingerprint:
            raise ValueError("checkpoint fingerprint does not match the current expansion contract")
        if existing.get("status") == "completed":
            # A completed checkpoint is reusable, but still verify every chunk.
            pass
        for item in existing.get("completed_chunks", []):
            index = int(item)
            chunk_path = checkpoint / f"chunk-{index:06d}.jsonl"
            if not chunk_path.is_file():
                raise ValueError(f"checkpoint chunk is missing: {chunk_path}")
            expected_checksum = str(existing.get("chunk_checksums", {}).get(str(index), ""))
            if not expected_checksum or _file_sha256(chunk_path) != expected_checksum:
                raise ValueError(f"checkpoint chunk checksum mismatch: {chunk_path}")
            completed.add(index)
            chunk_checksums[str(index)] = expected_checksum
        contract.update(
            {
                "completed_chunks": sorted(completed),
                "completed_cells": int(existing.get("completed_cells", 0)),
                "retained_cells": int(existing.get("retained_cells", 0)),
                "excluded_cells": int(existing.get("excluded_cells", 0)),
                "next_chunk": int(existing.get("next_chunk", max(completed, default=-1) + 1)),
                "chunk_checksums": chunk_checksums,
            }
        )
    else:
        contract["chunk_checksums"] = {}
        _atomic_json(manifest_path, contract)

    start = time.monotonic()
    rates: list[float] = []
    lock_path = checkpoint / ".lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(lock_fd)
    except FileExistsError as error:
        raise RuntimeError(f"another expand-od process is using checkpoint {checkpoint}") from error

    def report(status: str, *, chunk: int | None = None, final: bool = False) -> None:
        completed_cells = int(contract["completed_cells"])
        elapsed = max(time.monotonic() - start, 1e-9)
        rate = completed_cells / elapsed if completed_cells else 0.0
        if rate > 0.0:
            rates.append(rate)
        eta = None
        if len(rates) >= 3 and rates[-3:]:
            eta = max(total_rows - completed_cells, 0) / (sum(rates[-3:]) / len(rates[-3:]))
        event: dict[str, object] = {
            "status": status,
            "chunk": chunk,
            "completed_chunks": len(completed),
            "total_chunks": total_chunks,
            "completed_cells": completed_cells,
            "total_cells": total_rows,
            "retained_cells": int(contract["retained_cells"]),
            "excluded_cells": int(contract["excluded_cells"]),
            "elapsed_seconds": elapsed,
            "rolling_rate_cells_per_second": rates[-3:] if rates else [],
            "eta_seconds": eta,
            "memory_bytes": _memory_bytes(),
            "next_chunk": int(contract["next_chunk"]),
            "checkpoint_directory": str(checkpoint),
            "expansion_fingerprint": expansion_fingerprint,
        }
        if final:
            event["checkpoint_reusable"] = False
        _atomic_json(progress_path, event)
        if progress is not None:
            progress(event)

    old_sigterm = None
    active_chunk_path: Path | None = None
    active_chunk_index: int | None = None

    def stop_handler(signum: int, _frame: object) -> None:
        raise ODTimeExpansionInterrupted(checkpoint, f"OD-time expansion interrupted by signal {signum}")

    try:
        try:
            old_sigterm = signal.signal(signal.SIGTERM, stop_handler)
        except ValueError:
            old_sigterm = None
        if not resume:
            report("started")
        else:
            report("resuming")
        for chunk_index in range(total_chunks):
            if chunk_index in completed:
                continue
            pair_slice = universe.pairs[chunk_index * chunk_size : (chunk_index + 1) * chunk_size]
            rows: list[dict[str, object]] = []
            active_chunk_path = checkpoint / f"chunk-{chunk_index:06d}.jsonl"
            active_chunk_index = chunk_index
            last_progress = time.monotonic()
            for pair in pair_slice:
                rows.extend(
                    _expansion_rows_for_pair(
                        pair,
                        bins,
                        universe,
                        scenario=scenario,
                        departures=departures,
                        arrivals=arrivals,
                        reachable=reachable,
                        timetable_feasibility=timetable_feasibility,
                        feasibility_index=feasibility_index,
                        feasibility_contract=feasibility_contract,
                        timetable_policy=policy,  # type: ignore[arg-type]
                        timetable_index=timetable_index,
                        feasible_destinations_by_origin_period=(
                            feasible_destinations_by_origin_period
                        ),
                        **limits,
                    )
                )
                if time.monotonic() - last_progress >= interval:
                    report("running", chunk=chunk_index)
                    last_progress = time.monotonic()
            checksum = _atomic_jsonl(
                active_chunk_path, rows, maximum_bytes=maximum_temporary_bytes
            )
            chunk_checksums[str(chunk_index)] = checksum
            completed.add(chunk_index)
            contract["completed_chunks"] = sorted(completed)
            contract["chunk_checksums"] = dict(chunk_checksums)
            contract["completed_cells"] = int(contract["completed_cells"]) + len(rows)
            contract["retained_cells"] = int(contract["retained_cells"]) + sum(row["status"] == "retained" for row in rows)
            contract["excluded_cells"] = int(contract["excluded_cells"]) + sum(row["status"] == "excluded" for row in rows)
            contract["next_chunk"] = chunk_index + 1
            _atomic_json(manifest_path, contract)
            report("running", chunk=chunk_index)
            active_chunk_path = None
            active_chunk_index = None
        def persisted_rows():
            for chunk_index in sorted(completed):
                chunk_path = checkpoint / f"chunk-{chunk_index:06d}.jsonl"
                with chunk_path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        if line.strip():
                            yield json.loads(line)

        semantic = _semantic_checksum(persisted_rows())
        contract.update({"status": "completed", "semantic_checksum": semantic, "checkpoint_reusable": False})
        _atomic_json(manifest_path, contract)
        report("completed", final=True)
        return ODTimeExpansionRunResult(
            checkpoint,
            expansion_fingerprint,
            semantic,
            "completed",
            total_chunks,
            len(completed),
            int(contract["completed_cells"]),
            int(contract["retained_cells"]),
            int(contract["excluded_cells"]),
            int(contract["next_chunk"]),
            reused,
        )
    except (KeyboardInterrupt, ODTimeExpansionInterrupted) as error:
        if active_chunk_path is not None and active_chunk_index not in completed:
            try:
                active_chunk_path.unlink()
            except FileNotFoundError:
                pass
        contract["status"] = "interrupted"
        contract["next_chunk"] = min((index for index in range(total_chunks) if index not in completed), default=total_chunks)
        _atomic_json(manifest_path, contract)
        try:
            report("interrupted", final=True)
        except BaseException:
            # A progress sink must not prevent the typed interruption from
            # reaching the CLI after durable state has been written.
            pass
        if isinstance(error, ODTimeExpansionInterrupted):
            raise
        raise ODTimeExpansionInterrupted(checkpoint) from error
    finally:
        if old_sigterm is not None:
            signal.signal(signal.SIGTERM, old_sigterm)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True, slots=True)
class PriorGenerationResult:
    """Generated prior values and semantic provenance."""

    values: Mapping[CandidateODTimeCell, float]
    source: str
    semantics: str
    parameters: Mapping[str, object]
    generator_fingerprint: str
    fingerprint: str

    @property
    def audit(self) -> dict[str, object]:
        return {
            "source": self.source,
            "semantics": self.semantics,
            "parameters": dict(self.parameters),
            "cell_count": len(self.values),
            "generator_fingerprint": self.generator_fingerprint,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class PriorCheckpointMaterializationResult:
    """Audit information for a prior streamed from a completed expansion."""

    output_path: Path
    source: str
    semantics: str
    value: float | None
    cell_count: int
    expansion_fingerprint: str
    generator_fingerprint: str
    fingerprint: str
    output_sha256: str

    @property
    def audit(self) -> dict[str, object]:
        return {
            "source": self.source,
            "semantics": self.semantics,
            "value": self.value,
            "cell_count": self.cell_count,
            "expansion_fingerprint": self.expansion_fingerprint,
            "generator_fingerprint": self.generator_fingerprint,
            "fingerprint": self.fingerprint,
            "output_path": str(self.output_path),
            "output_sha256": self.output_sha256,
        }


@dataclass(frozen=True, slots=True)
class ScheduledFeasibilityContract:
    """Single feasibility policy shared by OD expansion and feature support.

    The contract intentionally contains only timetable-feasibility semantics;
    candidate-universe rules (for example directed reachability) remain in the
    surrounding OD-universe contract.  Its fingerprint is persisted with
    support audits so a prior cannot be reused after a semantic change.
    """

    maximum_transfers: int = 2
    maximum_initial_wait_seconds: int = 3600
    maximum_journey_seconds: int = 7200
    maximum_waiting_seconds: int = 3600
    version: int = 1

    def __post_init__(self) -> None:
        if self.maximum_transfers < 0 or self.maximum_initial_wait_seconds < 0:
            raise ValueError("transfer and initial-wait limits cannot be negative")
        if self.maximum_journey_seconds <= 0 or self.maximum_waiting_seconds < 0:
            raise ValueError("journey time must be positive and waiting time non-negative")

    @property
    def fingerprint(self) -> str:
        return _sha256_payload(
            {
                "version": self.version,
                "maximum_transfers": self.maximum_transfers,
                "maximum_initial_wait_seconds": self.maximum_initial_wait_seconds,
                "maximum_journey_seconds": self.maximum_journey_seconds,
                "maximum_waiting_seconds": self.maximum_waiting_seconds,
            }
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ScheduledFeasibilityContract":
        return cls(
            maximum_transfers=int(values.get("maximum_transfers", 2)),
            maximum_initial_wait_seconds=int(
                values.get("maximum_initial_wait_seconds", 3600)
            ),
            maximum_journey_seconds=int(values.get("maximum_journey_seconds", 7200)),
            maximum_waiting_seconds=int(values.get("maximum_waiting_seconds", 3600)),
            version=int(values.get("feasibility_contract_version", 1)),
        )

    def path_metrics(
        self,
        index: "TimetableFeasibilityIndex",
        *,
        origin: str,
        period: tuple[str, int, int],
    ) -> Mapping[str, "TimetablePathMetrics"]:
        return index.path_metrics(
            origin,
            period,
            maximum_transfers=self.maximum_transfers,
            maximum_initial_wait_seconds=self.maximum_initial_wait_seconds,
            maximum_waiting_seconds=self.maximum_waiting_seconds,
            maximum_journey_seconds=self.maximum_journey_seconds,
        )

    def is_feasible(
        self,
        index: "TimetableFeasibilityIndex",
        pair: CandidateODPair,
        period: tuple[str, int, int],
    ) -> bool:
        origin = index.physical_stop_mapping.get(pair.origin_stop_id, pair.origin_stop_id)
        destination = index.physical_stop_mapping.get(
            pair.destination_stop_id, pair.destination_stop_id
        )
        return destination in self.path_metrics(index, origin=origin, period=period)


@dataclass(frozen=True, slots=True)
class TimetablePathMetrics:
    """Minimum path metrics produced by :class:`ScheduledFeasibilityContract`."""

    minimum_transfers: int
    minimum_initial_wait_seconds: int
    minimum_journey_seconds: int
    feasible_departure_count: int
    earliest_arrival_seconds: int


@dataclass(frozen=True, slots=True)
class TimetableFeasibilityIndex:
    """Reusable schedule indexes for OD--time feasibility checks.

    The old evaluator rebuilt trip sequences and scanned every stop-time record
    for every candidate cell.  This index performs that work once per
    expansion and keeps sorted departure/arrival lists for logarithmic lookup.
    """

    sequences: Mapping[str, tuple[object, ...]]
    departures_by_stop: Mapping[str, tuple[tuple[int, str, int], ...]]
    departure_seconds_by_stop: Mapping[str, tuple[int, ...]]
    arrivals_by_stop: Mapping[str, tuple[tuple[int, str, int], ...]]
    physical_stop_mapping: Mapping[str, str]

    @classmethod
    def from_scenario(
        cls,
        scenario: Scenario,
        *,
        physical_stop_mapping: Mapping[str, str] | None = None,
        timetable_index: "CanonicalTimetableIndex | None" = None,
    ) -> "TimetableFeasibilityIndex":
        mapping = dict(physical_stop_mapping or {str(stop.stop_id): str(stop.stop_id) for stop in scenario.stops})
        sequences = _trip_sequences(
            scenario, mapping, timetable_index=timetable_index
        )
        departures: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
        arrivals: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
        for trip_id, sequence in sequences.items():
            for index, stop_time in enumerate(sequence):
                stop = mapping[str(stop_time.stop_id)]
                departure = _departure_seconds(stop_time)
                arrival = _arrival_seconds(stop_time)
                departures[stop].append((departure, trip_id, index))
                arrivals[stop].append((arrival, trip_id, index))
        ordered_departures = {
            stop: tuple(sorted(items, key=lambda item: (item[0], item[1], item[2])))
            for stop, items in departures.items()
        }
        ordered_arrivals = {
            stop: tuple(sorted(items, key=lambda item: (item[0], item[1], item[2])))
            for stop, items in arrivals.items()
        }
        return cls(
            sequences=MappingProxyType(dict(sequences)),
            departures_by_stop=MappingProxyType(ordered_departures),
            departure_seconds_by_stop=MappingProxyType(
                {stop: tuple(item[0] for item in items) for stop, items in ordered_departures.items()}
            ),
            arrivals_by_stop=MappingProxyType(ordered_arrivals),
            physical_stop_mapping=MappingProxyType(mapping),
        )

    def is_feasible(
        self,
        pair: CandidateODPair,
        period: tuple[str, int, int],
        *,
        maximum_transfers: int,
        maximum_initial_wait_seconds: int,
        maximum_waiting_seconds: int,
        maximum_journey_seconds: int,
    ) -> bool:
        origin = self.physical_stop_mapping.get(pair.origin_stop_id, pair.origin_stop_id)
        destination = self.physical_stop_mapping.get(pair.destination_stop_id, pair.destination_stop_id)
        return destination in self.path_metrics(
            origin,
            period,
            maximum_transfers=maximum_transfers,
            maximum_initial_wait_seconds=maximum_initial_wait_seconds,
            maximum_waiting_seconds=maximum_waiting_seconds,
            maximum_journey_seconds=maximum_journey_seconds,
        )

    def feasible_destinations(
        self,
        origin: str,
        period: tuple[str, int, int],
        *,
        maximum_transfers: int,
        maximum_initial_wait_seconds: int,
        maximum_waiting_seconds: int,
        maximum_journey_seconds: int,
    ) -> frozenset[str]:
        """Return every destination feasible from one origin/time period.

        The search state is identical to the historical per-destination
        evaluator, but it is performed once per origin and period.  Expansion
        can therefore classify all destinations in that origin/time slice
        without repeating the same timetable traversal.
        """
        return frozenset(
            self.path_metrics(
                origin,
                period,
                maximum_transfers=maximum_transfers,
                maximum_initial_wait_seconds=maximum_initial_wait_seconds,
                maximum_waiting_seconds=maximum_waiting_seconds,
                maximum_journey_seconds=maximum_journey_seconds,
            )
        )

    def path_metrics(
        self,
        origin: str,
        period: tuple[str, int, int],
        *,
        maximum_transfers: int,
        maximum_initial_wait_seconds: int,
        maximum_waiting_seconds: int,
        maximum_journey_seconds: int,
    ) -> Mapping[str, TimetablePathMetrics]:
        """Return feasible destinations and metrics under one exact contract."""
        origin = self.physical_stop_mapping.get(origin, origin)
        start, end = period[1], period[2]
        origin_departures = self.departures_by_stop.get(origin, ())
        departure_seconds = self.departure_seconds_by_stop.get(origin, ())
        lower = bisect_left(departure_seconds, start)
        upper_bound = min(end, start + maximum_initial_wait_seconds + 1)
        upper = bisect_left(departure_seconds, upper_bound)
        first_candidates = [
            (trip_id, index, departure)
            for departure, trip_id, index in origin_departures[lower:upper]
        ]
        queue: deque[tuple[str, int, int, int]] = deque(
            (trip_id, index, departure, 0)
            for trip_id, index, departure in first_candidates
        )
        visited: set[tuple[str, int, int]] = set()
        destination_metrics: dict[str, dict[str, Any]] = {}
        while queue:
            trip_id, board_index, first_departure, transfers = queue.popleft()
            state_key = (trip_id, board_index, transfers)
            if state_key in visited:
                continue
            visited.add(state_key)
            sequence = self.sequences[trip_id]
            for alight_index in range(board_index + 1, len(sequence)):
                alight = sequence[alight_index]
                arrival = _arrival_seconds(alight)
                if arrival - first_departure > maximum_journey_seconds:
                    break
                stop = self.physical_stop_mapping.get(str(alight.stop_id), str(alight.stop_id))
                candidate = destination_metrics.setdefault(
                    stop,
                    {
                        "transfers": [],
                        "initial_wait": [],
                        "journey": [],
                        "departures": set(),
                        "arrivals": [],
                    },
                )
                candidate["transfers"].append(transfers)
                candidate["initial_wait"].append(first_departure - start)
                candidate["journey"].append(arrival - first_departure)
                candidate["departures"].add((trip_id, board_index, first_departure))
                candidate["arrivals"].append(arrival)
                if transfers >= maximum_transfers:
                    continue
                departures = self.departures_by_stop.get(stop, ())
                seconds = self.departure_seconds_by_stop.get(stop, ())
                first = bisect_left(seconds, arrival)
                for departure, next_trip_id, next_index in departures[first:]:
                    if departure - arrival > maximum_waiting_seconds:
                        break
                    if next_trip_id != trip_id:
                        queue.append((next_trip_id, next_index, first_departure, transfers + 1))
        return {
            destination: TimetablePathMetrics(
                minimum_transfers=min(values["transfers"]),
                minimum_initial_wait_seconds=min(values["initial_wait"]),
                minimum_journey_seconds=min(values["journey"]),
                feasible_departure_count=len(values["departures"]),
                earliest_arrival_seconds=min(values["arrivals"]),
            )
            for destination, values in destination_metrics.items()
        }


def _precompute_temporal_feasibility(
    *,
    universe: CandidateODUniverse,
    bins: Sequence[tuple[str, int, int]],
    feasibility_index: TimetableFeasibilityIndex,
    maximum_transfers: int,
    maximum_initial_wait_seconds: int,
    maximum_waiting_seconds: int,
    maximum_journey_seconds: int,
    feasibility_contract: ScheduledFeasibilityContract | None = None,
) -> dict[tuple[str, str], frozenset[str]]:
    """Evaluate each distinct origin/time slice once.

    The historical expansion searched the timetable independently for every
    destination.  The feasibility index can instead return all destinations
    reached by the same search state, so this cache preserves the exact
    per-pair result while eliminating repeated traversal work.
    """
    contract = feasibility_contract or ScheduledFeasibilityContract(
        maximum_transfers=maximum_transfers,
        maximum_initial_wait_seconds=maximum_initial_wait_seconds,
        maximum_journey_seconds=maximum_journey_seconds,
        maximum_waiting_seconds=maximum_waiting_seconds,
    )
    origins = sorted(
        {
            universe.physical_stop_mapping.get(
                pair.origin_stop_id, pair.origin_stop_id
            )
            for pair in universe.pairs
        }
    )
    return {
        (origin, period[0]): frozenset(
            contract.path_metrics(
                feasibility_index, origin=origin, period=period
            )
        )
        for origin in origins
        for period in bins
    }


def _mapping_for_level(
    scenario: Scenario,
    *,
    level: ODUniverseLevel,
    physical_stop_mapping: Mapping[str, str] | None,
) -> dict[str, str]:
    stop_ids = {str(stop.stop_id) for stop in scenario.stops}
    if level == "stop":
        return {stop_id: stop_id for stop_id in sorted(stop_ids)}
    if physical_stop_mapping is None:
        return {stop_id: stop_id for stop_id in sorted(stop_ids)}
    normalized = {
        _text(stop_id, "physical-stop mapping stop_id"): _text(
            physical_id, "physical-stop mapping physical_stop_id"
        )
        for stop_id, physical_id in physical_stop_mapping.items()
    }
    missing = sorted(stop_ids - set(normalized))
    unknown = sorted(set(normalized) - stop_ids)
    if missing or unknown:
        raise ValueError(
            "physical-stop mapping must cover exactly the scenario stops; "
            f"missing={missing}, unknown={unknown}."
        )
    # Pair identifiers at ``physical_stop`` level are physical IDs, whereas
    # timetable records still use platform/stop IDs.  Keep both namespaces in
    # the index so the same mapping can resolve either kind of identifier.
    return {
        **normalized,
        **{physical_id: physical_id for physical_id in set(normalized.values())},
    }


def _trip_sequences(
    scenario: Scenario,
    mapping: Mapping[str, str],
    *,
    timetable_index: "CanonicalTimetableIndex | None" = None,
) -> dict[str, tuple[object, ...]]:
    if scenario.timetable is None:
        return {}
    if timetable_index is not None:
        return {
            str(trip_id): tuple(sequence)
            for trip_id, sequence in timetable_index.trip_sequences.items()
        }
    by_trip: dict[str, list[object]] = defaultdict(list)
    for stop_time in scenario.timetable.stop_times:
        by_trip[str(stop_time.trip_id)].append(stop_time)
    return {
        trip_id: tuple(
            sorted(
                values,
                key=lambda item: int(getattr(item, "sequence")),
            )
        )
        for trip_id, values in by_trip.items()
    }


def _service_activity(
    scenario: Scenario,
    mapping: Mapping[str, str],
    *,
    timetable_index: "CanonicalTimetableIndex | None" = None,
) -> tuple[set[str], set[str]]:
    departures: set[str] = set()
    arrivals: set[str] = set()
    if scenario.timetable is None:
        return departures, arrivals
    stop_times = (
        scenario.timetable.stop_times
        if timetable_index is None
        else timetable_index.stop_times
    )
    for stop_time in stop_times:
        physical = mapping[str(stop_time.stop_id)]
        departures.add(physical)
        arrivals.add(physical)
    return departures, arrivals


def _directed_reachability(
    scenario: Scenario,
    mapping: Mapping[str, str],
    *,
    timetable_index: "CanonicalTimetableIndex | None" = None,
) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for sequence in _trip_sequences(
        scenario, mapping, timetable_index=timetable_index
    ).values():
        for left, right in zip(sequence, sequence[1:], strict=False):
            adjacency[mapping[str(left.stop_id)]].add(mapping[str(right.stop_id)])
    nodes = set(mapping.values())
    reachable: dict[str, set[str]] = {}
    for origin in sorted(nodes):
        seen = {origin}
        queue: deque[str] = deque([origin])
        while queue:
            current = queue.popleft()
            for destination in sorted(adjacency.get(current, ())):
                if destination not in seen:
                    seen.add(destination)
                    queue.append(destination)
        reachable[origin] = seen
    return reachable


def _read_pair_file(
    path: str | Path,
    *,
    allowed_ids: set[str],
    identifier_label: str,
) -> tuple[CandidateODPair, ...]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"candidate OD-pair file does not exist: {source}")
    pairs: list[CandidateODPair] = []
    with source.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("candidate OD-pair file has no header.")
        required = {"origin_stop_id", "destination_stop_id"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(
                "candidate OD-pair file is missing required columns: "
                f"{missing}. Expected origin_stop_id,destination_stop_id."
            )
        extra = sorted(set(reader.fieldnames) - required)
        if extra:
            raise ValueError(
                "candidate OD-pair file must be pair-only; unexpected columns: "
                f"{extra}. Remove time-bin and flow columns."
            )
        for row_number, row in enumerate(reader, start=2):
            origin = _text(row.get("origin_stop_id"), f"OD pair row {row_number} origin_stop_id")
            destination = _text(
                row.get("destination_stop_id"),
                f"OD pair row {row_number} destination_stop_id",
            )
            if origin not in allowed_ids or destination not in allowed_ids:
                raise ValueError(
                    f"OD pair row {row_number} references an unknown {identifier_label}: "
                    f"{origin!r}, {destination!r}."
                )
            pairs.append(CandidateODPair(origin, destination))
    if len(set(pairs)) != len(pairs):
        raise ValueError("candidate OD-pair file contains duplicate ordered pairs.")
    return tuple(sorted(pairs))


def generate_candidate_od_pairs(
    scenario: Scenario,
    *,
    source: ODUniverseSource = "network_ordered_pairs",
    level: ODUniverseLevel = "stop",
    include_same_stop: bool = False,
    active_service_only: bool = True,
    connectivity_policy: ConnectivityPolicy = "directed_reachable",
    od_pairs_path: str | Path | None = None,
    physical_stop_mapping: Mapping[str, str] | None = None,
    timetable_index: "CanonicalTimetableIndex | None" = None,
) -> CandidateODUniverse:
    """Generate and validate an immutable ordered candidate OD universe.

    The function never reads ``scenario.demand`` or ``scenario.time_bins``.
    ``source='file'`` requires a pair-only CSV with no time-bin membership.
    """
    if source not in {"file", "network_ordered_pairs"}:
        raise ValueError("source must be 'file' or 'network_ordered_pairs'.")
    if connectivity_policy not in {"none", "directed_reachable"}:
        raise ValueError("connectivity_policy must be 'none' or 'directed_reachable'.")
    mapping = _mapping_for_level(
        scenario,
        level=level,
        physical_stop_mapping=physical_stop_mapping,
    )
    if source == "file":
        if od_pairs_path is None:
            raise ValueError("source='file' requires od_pairs_path.")
        raw_pairs = _read_pair_file(
            od_pairs_path,
            allowed_ids=set(mapping.values()),
            identifier_label=("physical stop" if level == "physical_stop" else "network stop"),
        )
    else:
        nodes = sorted(set(mapping.values()))
        raw_pairs = (
            CandidateODPair(origin, destination)
            for origin in nodes
            for destination in nodes
        )
    departures, arrivals = _service_activity(
        scenario, mapping, timetable_index=timetable_index
    )
    reachable = _directed_reachability(
        scenario, mapping, timetable_index=timetable_index
    )
    retained: list[CandidateODPair] = []
    exclusions: list[ODUniverseExclusion] = []
    for pair in raw_pairs:
        if not include_same_stop and pair.origin_stop_id == pair.destination_stop_id:
            exclusions.append(ODUniverseExclusion(*pair.tuple, "same_node"))
            continue
        if active_service_only and pair.origin_stop_id not in departures:
            exclusions.append(ODUniverseExclusion(*pair.tuple, "inactive_origin"))
            continue
        if active_service_only and pair.destination_stop_id not in arrivals:
            exclusions.append(ODUniverseExclusion(*pair.tuple, "inactive_destination"))
            continue
        if (
            connectivity_policy == "directed_reachable"
            and pair.destination_stop_id not in reachable.get(pair.origin_stop_id, set())
        ):
            exclusions.append(ODUniverseExclusion(*pair.tuple, "static_unreachable"))
            continue
        retained.append(pair)
    generator_fingerprint = _sha256_payload(
        {
            "source": source,
            "level": level,
            "include_same_stop": include_same_stop,
            "active_service_only": active_service_only,
            "connectivity_policy": connectivity_policy,
            "od_pairs_path": None if od_pairs_path is None else str(Path(od_pairs_path).resolve()),
            "mapping": sorted(mapping.items()),
            "network_nodes": sorted(set(mapping.values())),
            "directed_edges": sorted(
                (origin, destination)
                for origin, destinations in reachable.items()
                for destination in destinations
            ),
        }
    )
    return CandidateODUniverse(
        pairs=tuple(sorted(retained)),
        exclusions=tuple(sorted(exclusions, key=lambda item: item.tuple)),
        source=source,
        level=level,
        include_same_stop=include_same_stop,
        active_service_only=active_service_only,
        connectivity_policy=connectivity_policy,
        physical_stop_mapping=MappingProxyType(dict(mapping)),
        generator_fingerprint=generator_fingerprint,
        directed_reachability=MappingProxyType(
            {
                origin: tuple(sorted(destinations))
                for origin, destinations in sorted(reachable.items())
            }
        ),
    )


def _period_tuple(period: object) -> tuple[str, int, int]:
    if isinstance(period, (tuple, list)) and len(period) == 3:
        period_id, start, end = period
        start_i, end_i = int(start), int(end)
        if end_i <= start_i:
            raise ValueError(f"time period {period_id!r} must have end > start.")
        return _text(period_id, "time period id"), start_i, end_i
    period_id = getattr(period, "period_id", getattr(period, "bin_id", None))
    start = getattr(period, "start_seconds", None)
    end = getattr(period, "end_seconds", None)
    if start is None:
        start = getattr(getattr(period, "start", None), "seconds_from_midnight", None)
    if end is None:
        end = getattr(getattr(period, "end", None), "seconds_from_midnight", None)
    if period_id is None or start is None or end is None:
        raise ValueError("time periods must expose id/bin_id and start/end seconds.")
    start_i, end_i = int(start), int(end)
    if end_i <= start_i:
        raise ValueError(f"time period {period_id!r} must have end > start.")
    return _text(period_id, "time period id"), start_i, end_i


def _arrival_seconds(stop_time: object) -> int:
    value = getattr(stop_time, "arrival_s", None)
    if value is not None:
        return int(value)
    arrival = getattr(stop_time, "arrival", None)
    if hasattr(arrival, "seconds_from_midnight"):
        return int(arrival.seconds_from_midnight)
    return int(arrival)


def _departure_seconds(stop_time: object) -> int:
    value = getattr(stop_time, "departure_s", None)
    if value is not None:
        return int(value)
    departure = getattr(stop_time, "departure", None)
    if hasattr(departure, "seconds_from_midnight"):
        return int(departure.seconds_from_midnight)
    return int(departure)


def _timetable_feasible(
    scenario: Scenario,
    pair: CandidateODPair,
    period: tuple[str, int, int],
    *,
    mapping: Mapping[str, str],
    maximum_transfers: int,
    maximum_initial_wait_seconds: int,
    maximum_journey_seconds: int,
    maximum_waiting_seconds: int,
    timetable_index: "CanonicalTimetableIndex | None" = None,
) -> bool:
    """Small schedule search used by expansion for an explicit feasibility rule."""
    sequences = _trip_sequences(
        scenario, mapping, timetable_index=timetable_index
    )
    if not sequences:
        return False
    origin = mapping[pair.origin_stop_id]
    destination = mapping[pair.destination_stop_id]
    start, end = period[1], period[2]
    boardings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for trip_id, sequence in sequences.items():
        for index, stop_time in enumerate(sequence):
            if mapping[str(stop_time.stop_id)] == origin:
                departure = _departure_seconds(stop_time)
                if start <= departure < end and departure - start <= maximum_initial_wait_seconds:
                    boardings[trip_id].append((index, departure))
    queue: deque[tuple[str, int, int, int]] = deque(
        (trip_id, index, departure, 0)
        for trip_id, entries in boardings.items()
        for index, departure in entries
    )
    visited: set[tuple[str, int, int]] = set()
    while queue:
        trip_id, board_index, first_departure, transfers = queue.popleft()
        state_key = (trip_id, board_index, transfers)
        if state_key in visited:
            continue
        visited.add(state_key)
        sequence = sequences[trip_id]
        for alight_index in range(board_index + 1, len(sequence)):
            alight = sequence[alight_index]
            arrival = _arrival_seconds(alight)
            if arrival - first_departure > maximum_journey_seconds:
                break
            stop = mapping[str(alight.stop_id)]
            if stop == destination:
                return True
            if transfers >= maximum_transfers:
                continue
            for next_trip_id, next_sequence in sequences.items():
                for next_index, next_stop_time in enumerate(next_sequence):
                    if mapping[str(next_stop_time.stop_id)] != stop:
                        continue
                    departure = _departure_seconds(next_stop_time)
                    if departure < arrival:
                        continue
                    if departure - arrival > maximum_waiting_seconds:
                        break
                    if next_trip_id == trip_id:
                        continue
                    queue.append((next_trip_id, next_index, first_departure, transfers + 1))
                    break
    return False


def expand_candidate_od_time_cells(
    universe: CandidateODUniverse,
    time_periods: Sequence[object],
    *,
    scenario: Scenario | None = None,
    maximum_transfers: int = 2,
    maximum_initial_wait_seconds: int = 3600,
    maximum_journey_seconds: int = 7200,
    maximum_waiting_seconds: int = 3600,
    timetable_policy: TimetablePolicy = "required",
    timetable_feasibility: Callable[[CandidateODPair, tuple[str, int, int]], bool] | None = None,
    feasibility_index: TimetableFeasibilityIndex | None = None,
    timetable_index: "CanonicalTimetableIndex | None" = None,
) -> ODTimeExpansion:
    """Expand a pair universe across approved bins and audit each exclusion."""
    if maximum_transfers < 0 or maximum_initial_wait_seconds < 0 or maximum_journey_seconds <= 0 or maximum_waiting_seconds < 0:
        raise ValueError("feasibility limits must be non-negative (journey time positive).")
    if timetable_policy not in {"required", "defer", "none"}:
        raise ValueError("unsupported timetable_policy.")
    bins = tuple(_period_tuple(period) for period in time_periods)
    if not bins:
        raise ValueError("at least one approved time period is required.")
    if len({item[0] for item in bins}) != len(bins):
        raise ValueError("approved time period identifiers must be unique.")
    if bins != tuple(sorted(bins, key=lambda item: (item[1], item[2], item[0]))):
        raise ValueError("approved time periods must be sorted by start/end/id.")
    if any(left[2] > right[1] for left, right in zip(bins, bins[1:], strict=False)):
        raise ValueError("approved time periods must not overlap.")
    mapping = universe.physical_stop_mapping
    departures, arrivals = (
        (set(), set())
        if scenario is None
        else _service_activity(
            scenario, mapping, timetable_index=timetable_index
        )
    )
    if universe.directed_reachability:
        reachable = {
            origin: set(destinations)
            for origin, destinations in universe.directed_reachability.items()
        }
    else:
        reachable = (
            {}
            if scenario is None
            else _directed_reachability(
                scenario, mapping, timetable_index=timetable_index
            )
        )
    if (
        timetable_feasibility is None
        and feasibility_index is None
        and timetable_policy == "required"
        and scenario is not None
        and scenario.timetable is not None
    ):
        feasibility_index = TimetableFeasibilityIndex.from_scenario(
            scenario,
            physical_stop_mapping=mapping,
            timetable_index=timetable_index,
        )
    feasible_destinations_by_origin_period = None
    if (
        timetable_feasibility is None
        and feasibility_index is not None
        and timetable_policy == "required"
        and scenario is not None
        and scenario.timetable is not None
    ):
        feasibility_contract = ScheduledFeasibilityContract(
            maximum_transfers=maximum_transfers,
            maximum_initial_wait_seconds=maximum_initial_wait_seconds,
            maximum_journey_seconds=maximum_journey_seconds,
            maximum_waiting_seconds=maximum_waiting_seconds,
        )
        feasible_destinations_by_origin_period = _precompute_temporal_feasibility(
            universe=universe,
            bins=bins,
            feasibility_index=feasibility_index,
            feasibility_contract=feasibility_contract,
            maximum_transfers=maximum_transfers,
            maximum_initial_wait_seconds=maximum_initial_wait_seconds,
            maximum_waiting_seconds=maximum_waiting_seconds,
            maximum_journey_seconds=maximum_journey_seconds,
        )
    cells: list[CandidateODTimeCell] = []
    exclusions: list[ODTimeExclusion] = []
    for pair in universe.pairs:
        pair_reason: str | None = None
        if pair.origin_stop_id == pair.destination_stop_id and not universe.include_same_stop:
            pair_reason = "same_node"
        elif universe.active_service_only and scenario is not None and pair.origin_stop_id not in departures:
            pair_reason = "inactive_origin"
        elif universe.active_service_only and scenario is not None and pair.destination_stop_id not in arrivals:
            pair_reason = "inactive_destination"
        elif universe.connectivity_policy == "directed_reachable" and scenario is not None and pair.destination_stop_id not in reachable.get(pair.origin_stop_id, set()):
            pair_reason = "static_unreachable"
        for period in bins:
            if pair_reason is not None:
                exclusions.append(ODTimeExclusion(*pair.tuple, period[0], pair_reason))
                continue
            feasible: bool | None
            if timetable_feasibility is not None:
                feasible = bool(timetable_feasibility(pair, period))
            elif timetable_policy == "none":
                feasible = True
            elif timetable_policy == "defer":
                feasible = None
            elif scenario is None or scenario.timetable is None:
                feasible = False
            elif feasible_destinations_by_origin_period is not None:
                origin = mapping.get(pair.origin_stop_id, pair.origin_stop_id)
                destination = mapping.get(
                    pair.destination_stop_id, pair.destination_stop_id
                )
                feasible = destination in feasible_destinations_by_origin_period[
                    (origin, period[0])
                ]
            elif feasibility_index is not None:
                feasible = feasibility_index.is_feasible(
                    pair,
                    period,
                    maximum_transfers=maximum_transfers,
                    maximum_initial_wait_seconds=maximum_initial_wait_seconds,
                    maximum_waiting_seconds=maximum_waiting_seconds,
                    maximum_journey_seconds=maximum_journey_seconds,
                )
            else:
                feasible = _timetable_feasible(
                    scenario,
                    pair,
                    period,
                    mapping=mapping,
                    maximum_transfers=maximum_transfers,
                    maximum_initial_wait_seconds=maximum_initial_wait_seconds,
                    maximum_journey_seconds=maximum_journey_seconds,
                    maximum_waiting_seconds=maximum_waiting_seconds,
                    timetable_index=timetable_index,
                )
            if feasible is False:
                exclusions.append(
                    ODTimeExclusion(*pair.tuple, period[0], "timetable_infeasible")
                )
            else:
                cells.append(CandidateODTimeCell(*pair.tuple, period[0]))
    feasibility_contract = ScheduledFeasibilityContract(
        maximum_transfers=maximum_transfers,
        maximum_initial_wait_seconds=maximum_initial_wait_seconds,
        maximum_journey_seconds=maximum_journey_seconds,
        maximum_waiting_seconds=maximum_waiting_seconds,
    )
    policies = MappingProxyType(
        {
            "maximum_transfers": maximum_transfers,
            "maximum_initial_wait_seconds": maximum_initial_wait_seconds,
            "maximum_journey_seconds": maximum_journey_seconds,
            "maximum_waiting_seconds": maximum_waiting_seconds,
            "timetable_policy": timetable_policy,
            "feasibility_contract_version": feasibility_contract.version,
            "feasibility_contract_fingerprint": feasibility_contract.fingerprint,
        }
    )
    fingerprint = _sha256_payload(
        {
            "universe_fingerprint": universe.fingerprint,
            "time_bins": [list(item) for item in bins],
            "cells": [list(item.tuple) for item in sorted(cells)],
            "exclusions": [
                [item.origin_stop_id, item.destination_stop_id, item.time_bin_id, item.reason]
                for item in sorted(exclusions, key=lambda value: (value.origin_stop_id, value.destination_stop_id, value.time_bin_id, value.reason))
            ],
            "policies": dict(policies),
        }
    )
    return ODTimeExpansion(
        universe_fingerprint=universe.fingerprint,
        cells=tuple(sorted(cells)),
        exclusions=tuple(
            sorted(
                exclusions,
                key=lambda item: (
                    item.origin_stop_id,
                    item.destination_stop_id,
                    item.time_bin_id,
                    item.reason,
                ),
            )
        ),
        time_bins=bins,
        policies=policies,
        fingerprint=fingerprint,
    )


def _read_pair_priors(path: str | Path) -> dict[tuple[str, str], float]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"prior demand file does not exist: {source}")
    result: dict[tuple[str, str], float] = {}
    with source.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("prior demand file has no header.")
        required = {"origin_stop_id", "destination_stop_id", "prior_value"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"prior demand file is missing required columns: {missing}")
        extra = sorted(set(reader.fieldnames) - required)
        if extra:
            raise ValueError(
                "prior demand file must be pair-level and independent of time bins; "
                f"unexpected columns: {extra}."
            )
        for row_number, row in enumerate(reader, start=2):
            key = (
                _text(row.get("origin_stop_id"), f"prior row {row_number} origin_stop_id"),
                _text(row.get("destination_stop_id"), f"prior row {row_number} destination_stop_id"),
            )
            if key in result:
                raise ValueError(f"prior demand file contains duplicate pair {key!r}.")
            try:
                value = float(row.get("prior_value", ""))
            except (TypeError, ValueError) as error:
                raise ValueError(f"prior row {row_number} prior_value must be numeric.") from error
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"prior row {row_number} prior_value must be finite and non-negative.")
            result[key] = value
    return result


def generate_prior_demand(
    expansion: ODTimeExpansion,
    *,
    source: str = "all_ones",
    value: float = 1.0,
    semantics: str = "neutral_seed",
    prior_file: str | Path | None = None,
) -> PriorGenerationResult:
    """Generate prior values only after OD--time expansion.

    The default ``all_ones`` prior is a neutral numerical seed, never an
    observation or a production/attractiveness estimate.
    """
    allowed_sources = {
        "all_ones",
        "external_file",
        "distance_decay",
        "travel_time_decay",
        "gravity_seed",
        "destination_attractiveness_seed",
    }
    if source not in allowed_sources:
        raise ValueError(f"unsupported prior source {source!r}.")
    if not isinstance(semantics, str) or not semantics.strip():
        raise ValueError("prior semantics must be a non-empty string.")
    if source == "all_ones":
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("all_ones value must be finite and positive.")
        pair_values: dict[tuple[str, str], float] = {}
        values = {cell: float(value) for cell in expansion.cells}
        parameters: dict[str, object] = {"value": float(value), "expansion": "one_per_retained_od_time_cell"}
    elif source == "external_file":
        if prior_file is None:
            raise ValueError("source='external_file' requires prior_file.")
        pair_values = _read_pair_priors(prior_file)
        required_pairs = {(cell.origin_stop_id, cell.destination_stop_id) for cell in expansion.cells}
        missing = sorted(required_pairs - set(pair_values))
        extra = sorted(set(pair_values) - required_pairs)
        if missing or extra:
            raise ValueError(f"external prior pairs do not match retained cells; missing={missing}, extra={extra}.")
        values = {cell: pair_values[(cell.origin_stop_id, cell.destination_stop_id)] for cell in expansion.cells}
        parameters = {"prior_file": str(Path(prior_file).expanduser().resolve()), "expansion": "pair_value_repeated_over_retained_bins"}
    else:
        raise NotImplementedError(
            f"prior generator {source!r} is reserved for a future explicit implementation."
        )
    generator_fingerprint = _sha256_payload({"source": source, "semantics": semantics, "parameters": parameters})
    fingerprint = _sha256_payload({"generator_fingerprint": generator_fingerprint, "expansion_fingerprint": expansion.fingerprint, "values": [[*cell.tuple, values[cell]] for cell in sorted(values)]})
    return PriorGenerationResult(
        values=MappingProxyType(dict(sorted(values.items()))),
        source=source,
        semantics=semantics,
        parameters=MappingProxyType(parameters),
        generator_fingerprint=generator_fingerprint,
        fingerprint=fingerprint,
    )


def materialize_prior_demand_from_checkpoint(
    checkpoint_directory: str | Path,
    output_path: str | Path,
    *,
    source: str = "all_ones",
    value: float = 1.0,
    semantics: str = "neutral_seed",
    prior_file: str | Path | None = None,
    scenario: Scenario | None = None,
    package_revision: str | None = None,
    expansion_fingerprint: str | None = None,
    configuration_fingerprint: str | None = None,
    approved_time_bins: Sequence[object] | None = None,
    approved_time_bins_fingerprint: str | None = None,
    scenario_fingerprint: str | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> PriorCheckpointMaterializationResult:
    """Stream a prior from a validated, completed OD-time checkpoint.

    The final output is written through a temporary file and is replaced only
    after every immutable chunk, checksum, row count, and semantic checksum has
    been validated.  Interrupted or corrupt checkpoints therefore cannot leave
    a new partial scenario demand file behind.
    """
    checkpoint = Path(checkpoint_directory).expanduser().resolve()
    manifest_path = checkpoint / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"checkpoint manifest is not valid JSON: {manifest_path}") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("checkpoint manifest must contain a JSON object")
    if manifest.get("status") != "completed":
        raise ValueError(
            "cannot materialize a prior from an incomplete checkpoint; "
            f"status={manifest.get('status')!r}"
        )
    checkpoint_expansion = str(manifest.get("expansion_fingerprint", ""))
    if not checkpoint_expansion:
        raise ValueError("checkpoint manifest has no expansion fingerprint")
    if expansion_fingerprint is not None and expansion_fingerprint != checkpoint_expansion:
        raise ValueError("checkpoint expansion fingerprint does not match the requested fingerprint")
    checkpoint_revision = str(manifest.get("package_revision", ""))
    if not checkpoint_revision:
        raise ValueError("checkpoint manifest has no package identity")
    if package_revision is not None and package_revision != checkpoint_revision:
        raise ValueError("checkpoint package revision does not match the requested revision")
    recorded_configuration_fingerprint = manifest.get("configuration_fingerprint")
    if configuration_fingerprint is not None and recorded_configuration_fingerprint != configuration_fingerprint:
        raise ValueError("checkpoint configuration fingerprint does not match the requested configuration")
    recorded_time_bins_fingerprint = manifest.get("approved_time_bins_fingerprint")
    if approved_time_bins_fingerprint is not None and recorded_time_bins_fingerprint != approved_time_bins_fingerprint:
        raise ValueError("checkpoint approved time-bin fingerprint does not match the requested bins")
    if approved_time_bins is not None:
        expected_bins = tuple(_period_tuple(item) for item in approved_time_bins)
        recorded_bins = tuple(
            _period_tuple(item) for item in manifest.get("approved_time_bins", ())
        )
        if expected_bins != recorded_bins:
            raise ValueError("checkpoint approved time bins do not match the scenario")
    recorded_scenario_fingerprint = manifest.get("scenario_checksums", {})
    if scenario_fingerprint is not None:
        if not isinstance(recorded_scenario_fingerprint, Mapping):
            raise ValueError("checkpoint has no scenario checksum payload")
        if recorded_scenario_fingerprint.get("scenario_fingerprint") != scenario_fingerprint:
            raise ValueError("checkpoint scenario fingerprint does not match the requested scenario")

    try:
        total_chunks = int(manifest["total_chunks"])
        total_rows = int(manifest["total_rows"])
        expected_completed_cells = int(manifest["completed_cells"])
        expected_retained_cells = int(manifest["retained_cells"])
        expected_excluded_cells = int(manifest["excluded_cells"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint manifest is missing numeric completion fields") from error
    if (
        total_chunks < 0
        or total_rows < 0
        or expected_completed_cells < 0
        or expected_retained_cells < 0
        or expected_excluded_cells < 0
    ):
        raise ValueError("checkpoint manifest contains negative completion fields")
    completed_chunks_raw = manifest.get("completed_chunks")
    if not isinstance(completed_chunks_raw, list):
        raise ValueError("checkpoint manifest has no completed chunk list")
    completed_chunks = {int(item) for item in completed_chunks_raw}
    if len(completed_chunks_raw) != len(completed_chunks):
        raise ValueError("checkpoint manifest contains duplicate completed chunks")
    if completed_chunks != set(range(total_chunks)):
        raise ValueError("completed checkpoint does not contain every expected chunk")
    checksums = manifest.get("chunk_checksums")
    if not isinstance(checksums, Mapping):
        raise ValueError("checkpoint manifest has no chunk checksums")
    for chunk_index in range(total_chunks):
        chunk_path = checkpoint / f"chunk-{chunk_index:06d}.jsonl"
        expected_checksum = str(checksums.get(str(chunk_index), ""))
        if not chunk_path.is_file() or not expected_checksum:
            raise ValueError(f"checkpoint chunk is missing or unsigned: {chunk_path}")
        if _file_sha256(chunk_path) != expected_checksum:
            raise ValueError(f"checkpoint chunk checksum mismatch: {chunk_path}")

    if not isinstance(semantics, str) or not semantics.strip():
        raise ValueError("prior semantics must be a non-empty string")
    if source == "all_ones":
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("all_ones value must be finite and positive")
        pair_values: dict[tuple[str, str], float] | None = None
        parameters: dict[str, object] = {
            "value": float(value),
            "expansion": "one_per_retained_od_time_cell",
        }
    elif source == "external_file":
        if prior_file is None:
            raise ValueError("source='external_file' requires prior_file")
        pair_values = _read_pair_priors(prior_file)
        parameters = {
            "prior_file": str(Path(prior_file).expanduser().resolve()),
            "expansion": "pair_value_repeated_over_retained_bins",
        }
    elif source in {
        "distance_decay",
        "travel_time_decay",
        "gravity_seed",
        "destination_attractiveness_seed",
    }:
        raise NotImplementedError(
            f"prior generator {source!r} is reserved for a future explicit implementation."
        )
    else:
        raise ValueError(f"unsupported prior source {source!r}")

    generator_fingerprint = _sha256_payload(
        {"source": source, "semantics": semantics, "parameters": parameters}
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    prior_digest = hashlib.sha256()
    prior_digest.update(
        canonical_json(
            {
                "expansion_fingerprint": checkpoint_expansion,
                "generator_fingerprint": generator_fingerprint,
            }
        ).encode("utf-8")
    )
    prior_digest.update(b"\n")
    row_count = 0
    retained_count = 0
    excluded_count = 0
    retained_pairs: set[tuple[str, str]] = set()
    semantic_digest = hashlib.sha256()
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("origin_stop_id", "dest_stop_id", "time_bin_id", "flow"),
            )
            writer.writeheader()
            for chunk_index in range(total_chunks):
                chunk_path = checkpoint / f"chunk-{chunk_index:06d}.jsonl"
                with chunk_path.open("r", encoding="utf-8") as chunk_stream:
                    for line_number, line in enumerate(chunk_stream, start=1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise ValueError(
                                f"checkpoint chunk contains invalid JSON: {chunk_path}:{line_number}"
                            ) from error
                        if not isinstance(row, Mapping):
                            raise ValueError(f"checkpoint row is not an object: {chunk_path}:{line_number}")
                        for field in ("origin_stop_id", "destination_stop_id", "time_bin_id", "status"):
                            if field not in row:
                                raise ValueError(f"checkpoint row is missing {field!r}: {chunk_path}:{line_number}")
                        status = str(row["status"])
                        if status not in {"retained", "excluded"}:
                            raise ValueError(f"checkpoint row has invalid status {status!r}")
                        semantic_digest.update(
                            canonical_json(
                                [
                                    str(row["origin_stop_id"]),
                                    str(row["destination_stop_id"]),
                                    str(row["time_bin_id"]),
                                    status,
                                    str(row.get("reason", "")),
                                ]
                            ).encode("utf-8")
                        )
                        semantic_digest.update(b"\n")
                        row_count += 1
                        if status == "excluded":
                            excluded_count += 1
                            continue
                        retained_count += 1
                        key = (str(row["origin_stop_id"]), str(row["destination_stop_id"]))
                        retained_pairs.add(key)
                        row_value = float(value) if pair_values is None else pair_values.get(key)
                        if row_value is None:
                            raise ValueError(f"external prior is missing retained pair {key!r}")
                        if not math.isfinite(row_value) or row_value < 0.0:
                            raise ValueError(f"prior value for pair {key!r} is invalid")
                        cell_payload = [str(row["origin_stop_id"]), str(row["destination_stop_id"]), str(row["time_bin_id"]), row_value]
                        prior_digest.update(canonical_json(cell_payload).encode("utf-8"))
                        prior_digest.update(b"\n")
                        output_row = {
                            "origin_stop_id": str(row["origin_stop_id"]),
                            "dest_stop_id": str(row["destination_stop_id"]),
                            "time_bin_id": str(row["time_bin_id"]),
                            "flow": row_value,
                        }
                        writer.writerow(output_row)
                if progress is not None:
                    progress(
                        {
                            "phase": "materialize_prior",
                            "status": "running",
                            "completed_chunks": chunk_index + 1,
                            "total_chunks": total_chunks,
                            "completed_cells": row_count,
                            "total_cells": total_rows,
                            "retained_cells": retained_count,
                            "excluded_cells": excluded_count,
                            "checkpoint_directory": str(checkpoint),
                            "expansion_fingerprint": checkpoint_expansion,
                            "current_unit": f"chunk-{chunk_index:06d}",
                        }
                    )
            stream.flush()
            os.fsync(stream.fileno())
        if row_count != total_rows or row_count != expected_completed_cells:
            raise ValueError("checkpoint row count does not match the manifest")
        if retained_count != expected_retained_cells or excluded_count != expected_excluded_cells:
            raise ValueError("checkpoint retained/excluded counts do not match the manifest")
        semantic = semantic_digest.hexdigest()
        if semantic != str(manifest.get("semantic_checksum", "")):
            raise ValueError("checkpoint semantic checksum mismatch")
        prior_fingerprint = prior_digest.hexdigest()
        if pair_values is not None:
            extra_pairs = sorted(set(pair_values) - retained_pairs)
            if extra_pairs:
                raise ValueError(f"external prior contains pairs absent from the checkpoint: {extra_pairs}")
        if progress is not None:
            progress(
                {
                    "phase": "materialize_prior",
                    "status": "completed",
                    "completed_chunks": total_chunks,
                    "total_chunks": total_chunks,
                    "completed_cells": row_count,
                    "total_cells": total_rows,
                    "retained_cells": retained_count,
                    "excluded_cells": excluded_count,
                    "checkpoint_directory": str(checkpoint),
                    "expansion_fingerprint": checkpoint_expansion,
                    "current_unit": output.name,
                }
            )
        os.replace(temporary_path, output)
        temporary_path = None
        result = PriorCheckpointMaterializationResult(
            output_path=output,
            source=source,
            semantics=semantics,
            value=float(value) if source == "all_ones" else None,
            cell_count=retained_count,
            expansion_fingerprint=checkpoint_expansion,
            generator_fingerprint=generator_fingerprint,
            fingerprint=prior_fingerprint,
            output_sha256=_file_sha256(output),
        )
        return result
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
