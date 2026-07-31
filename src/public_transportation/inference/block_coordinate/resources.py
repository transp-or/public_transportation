"""Bounded resource preflight and deterministic execution recommendations."""

from __future__ import annotations

import math
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Callable, Literal, Sequence

import numpy as np

from ._canonical import fingerprint
from .blocks import ODBlock
from .config import BlockCoordinateMAPConfig, BlockSizingConfig
from .operator import BlockLinearOperatorProtocol

ResourceProfile = Literal["auto", "laptop", "workstation", "server"]

_PROFILE_MEMORY_FRACTIONS: dict[str, float] = {
    "laptop": 0.55,
    "workstation": 0.68,
    "server": 0.78,
}


@dataclass(frozen=True, slots=True)
class MachineResourceSnapshot:
    """Resources visible to the current process, or an injected test snapshot."""

    available_memory_bytes: int
    logical_cpu_count: int
    physical_cpu_count: int
    available_cache_bytes: int
    coordinator_rss_bytes: int = 0
    assignment_rss_bytes: int = 0

    def __post_init__(self) -> None:
        for name in (
            "available_memory_bytes",
            "logical_cpu_count",
            "physical_cpu_count",
            "available_cache_bytes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in ("coordinator_rss_bytes", "assignment_rss_bytes"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.physical_cpu_count > self.logical_cpu_count:
            raise ValueError("physical_cpu_count cannot exceed logical_cpu_count.")


@dataclass(frozen=True, slots=True)
class BlockPreflightSample:
    """Measured cost of constructing and exercising one representative block."""

    block_id: str
    variables: int
    nonzeros: int
    operator_memory_bytes: int
    local_solver_memory_bytes: int
    construction_seconds: float
    matvec_seconds: float
    rmatvec_seconds: float
    checkpoint_bytes: int
    cache_bytes: int

    def __post_init__(self) -> None:
        if not self.block_id.strip():
            raise ValueError("block_id must be nonempty.")
        for name in (
            "variables",
            "operator_memory_bytes",
            "local_solver_memory_bytes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in ("nonzeros", "checkpoint_bytes", "cache_bytes"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        for name in ("construction_seconds", "matvec_seconds", "rmatvec_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")

    @property
    def worker_peak_bytes(self) -> int:
        return self.operator_memory_bytes + self.local_solver_memory_bytes


@dataclass(frozen=True, slots=True)
class ResourcePreflightConfig:
    resource_profile: ResourceProfile = "auto"
    maximum_sampled_blocks: int = 3
    memory_safety_factor: float = 1.35
    cache_safety_factor: float = 1.20
    requested_workers: int | None = None
    requested_threads_per_worker: int | None = None

    def __post_init__(self) -> None:
        if self.resource_profile not in {"auto", "laptop", "workstation", "server"}:
            raise ValueError("invalid resource profile.")
        if self.maximum_sampled_blocks <= 0:
            raise ValueError("maximum_sampled_blocks must be positive.")
        for name in ("memory_safety_factor", "cache_safety_factor"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 1.0:
                raise ValueError(f"{name} must be finite and at least one.")
        for name in ("requested_workers", "requested_threads_per_worker"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided.")


@dataclass(frozen=True, slots=True)
class BlockResourceRecommendation:
    resource_profile: ResourceProfile
    maximum_variables_per_block: int
    maximum_nonzeros_per_block: int
    block_count: int
    worker_count: int
    threads_per_worker: int
    expected_peak_memory_bytes: int
    expected_cache_bytes: int
    estimated_first_sweep_seconds: float
    estimated_cache_hit_sweep_seconds: float
    uncertainty_fraction: float
    reason: str

    def __post_init__(self) -> None:
        if self.resource_profile not in {"auto", "laptop", "workstation", "server"}:
            raise ValueError("invalid resource profile.")
        for name in (
            "maximum_variables_per_block",
            "maximum_nonzeros_per_block",
            "block_count",
            "worker_count",
            "threads_per_worker",
            "expected_peak_memory_bytes",
            "expected_cache_bytes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in (
            "estimated_first_sweep_seconds",
            "estimated_cache_hit_sweep_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not math.isfinite(self.uncertainty_fraction) or not 0.0 <= self.uncertainty_fraction <= 1.0:
            raise ValueError("uncertainty_fraction must be in [0, 1].")
        if not self.reason.strip():
            raise ValueError("recommendation reason must be nonempty.")

    @property
    def fingerprint(self) -> str:
        return fingerprint(self)


@dataclass(frozen=True, slots=True)
class AcceptedBlockResourceProposal:
    recommendation_fingerprint: str
    accepted: bool

    def __post_init__(self) -> None:
        if not self.recommendation_fingerprint.strip():
            raise ValueError("recommendation_fingerprint must be nonempty.")
        if not self.accepted:
            raise ValueError("a resource proposal must be explicitly accepted.")


def detect_machine_resources(*, cache_directory: Path) -> MachineResourceSnapshot:
    """Measure portable process-visible resources without allocating large arrays."""
    logical = os.cpu_count() or 1
    physical = logical
    available_memory = 0
    if platform.system() == "Darwin":
        try:
            output = subprocess.run(
                ["/usr/bin/vm_stat"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            page_match = re.search(r"page size of (\d+) bytes", output)
            page_size = int(page_match.group(1)) if page_match else 4096
            available_pages = 0
            for label in ("Pages free", "Pages inactive", "Pages speculative"):
                match = re.search(rf"^{label}:\s+(\d+)\.", output, re.MULTILINE)
                if match:
                    available_pages += int(match.group(1))
            available_memory = page_size * available_pages
            physical = int(
                subprocess.run(
                    ["/usr/sbin/sysctl", "-n", "hw.physicalcpu"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    else:
        try:
            available_memory = int(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
            )
        except (OSError, ValueError):
            pass
    if available_memory <= 0:
        # Conservative fallback used only when the platform exposes no memory API.
        available_memory = 1024**3
    physical = min(logical, max(1, physical))
    cache_path = Path(cache_directory).expanduser()
    while not cache_path.exists() and cache_path != cache_path.parent:
        cache_path = cache_path.parent
    return MachineResourceSnapshot(
        available_memory_bytes=available_memory,
        logical_cpu_count=logical,
        physical_cpu_count=physical,
        available_cache_bytes=shutil.disk_usage(cache_path).free,
    )


def select_representative_blocks(
    blocks: Sequence[ODBlock], *, maximum_samples: int = 3
) -> tuple[ODBlock, ...]:
    """Choose deterministic small, median, and large blocks within a hard bound."""
    if maximum_samples <= 0:
        raise ValueError("maximum_samples must be positive.")
    ordered = sorted(
        blocks,
        key=lambda block: (
            block.estimated_nonzeros or 0,
            block.num_free_variables,
            block.block_id,
        ),
    )
    if len(ordered) <= maximum_samples:
        return tuple(ordered)
    positions = np.linspace(0, len(ordered) - 1, maximum_samples, dtype=int)
    return tuple(ordered[int(position)] for position in positions)


def measure_representative_blocks(
    blocks: Sequence[ODBlock],
    *,
    operator_factory: Callable[[ODBlock], BlockLinearOperatorProtocol],
    maximum_samples: int = 3,
    local_solver_memory_estimator: Callable[[ODBlock], int] | None = None,
    checkpoint_bytes_per_variable: int = 32,
) -> tuple[BlockPreflightSample, ...]:
    """Bound preflight work by constructing and timing only selected blocks."""
    selected = select_representative_blocks(blocks, maximum_samples=maximum_samples)
    samples: list[BlockPreflightSample] = []
    for block in selected:
        started = perf_counter()
        operator = operator_factory(block)
        construction_seconds = perf_counter() - started
        vector = np.ones(operator.num_local_variables, dtype=operator.dtype)
        started = perf_counter()
        prediction = operator.matvec(vector)
        matvec_seconds = perf_counter() - started
        started = perf_counter()
        operator.rmatvec(prediction)
        rmatvec_seconds = perf_counter() - started
        preparation = getattr(operator, "preparation_metrics", None)
        operator_bytes = int(
            getattr(operator, "retained_bytes", max(1, block.num_free_variables * 8))
        )
        disk_bytes = int(getattr(preparation, "disk_bytes", operator_bytes))
        nonzeros = int(
            getattr(preparation, "nonzero_entries", block.estimated_nonzeros or 0)
        )
        solver_bytes = (
            int(local_solver_memory_estimator(block))
            if local_solver_memory_estimator is not None
            else max(1, 8 * (3 * block.num_free_variables + operator.num_measurements))
        )
        samples.append(
            BlockPreflightSample(
                block_id=block.block_id,
                variables=block.num_free_variables,
                nonzeros=nonzeros,
                operator_memory_bytes=max(1, operator_bytes),
                local_solver_memory_bytes=max(1, solver_bytes),
                construction_seconds=construction_seconds,
                matvec_seconds=matvec_seconds,
                rmatvec_seconds=rmatvec_seconds,
                checkpoint_bytes=checkpoint_bytes_per_variable * block.num_free_variables,
                cache_bytes=max(0, disk_bytes),
            )
        )
        release = getattr(operator, "release", None)
        if callable(release):
            release()
    return tuple(samples)


def _resolved_profile(profile: ResourceProfile, machine: MachineResourceSnapshot) -> str:
    if profile != "auto":
        return profile
    gib = machine.available_memory_bytes / (1024**3)
    if gib < 24:
        return "laptop"
    if gib < 96:
        return "workstation"
    return "server"


def recommend_block_resources(
    *,
    samples: Sequence[BlockPreflightSample],
    machine: MachineResourceSnapshot,
    total_variables: int,
    total_nonzeros: int,
    config: ResourcePreflightConfig = ResourcePreflightConfig(),
) -> BlockResourceRecommendation:
    """Scale bounded measurements into a conservative, reproducible proposal."""
    if not samples:
        raise ValueError("at least one representative block sample is required.")
    if total_variables <= 0 or total_nonzeros <= 0:
        raise ValueError("total_variables and total_nonzeros must be positive.")
    profile = _resolved_profile(config.resource_profile, machine)
    memory_budget = int(
        machine.available_memory_bytes * _PROFILE_MEMORY_FRACTIONS[profile]
    )
    fixed_memory = machine.coordinator_rss_bytes + machine.assignment_rss_bytes
    usable_memory = memory_budget - fixed_memory
    if usable_memory <= 0:
        raise MemoryError("resource preflight: fixed process memory exhausts the job budget.")

    bytes_per_variable = max(
        sample.worker_peak_bytes / sample.variables for sample in samples
    ) * config.memory_safety_factor
    bytes_per_nonzero = max(
        sample.operator_memory_bytes / max(1, sample.nonzeros) for sample in samples
    ) * config.memory_safety_factor
    threads = config.requested_threads_per_worker or max(
        1, machine.logical_cpu_count // machine.physical_cpu_count
    )
    cpu_workers = machine.logical_cpu_count // threads
    requested_workers = config.requested_workers or cpu_workers
    measured_worker_peak = math.ceil(
        max(sample.worker_peak_bytes for sample in samples)
        * config.memory_safety_factor
    )
    memory_workers = usable_memory // measured_worker_peak
    candidate_workers = min(requested_workers, cpu_workers, memory_workers)
    if candidate_workers <= 0:
        raise MemoryError("resource preflight: no measured worker fits safely.")

    # Iterate because fewer memory-safe workers allow larger blocks and vice versa.
    worker_count = candidate_workers
    while worker_count > 0:
        per_worker_budget = usable_memory // worker_count
        maximum_variables = min(total_variables, int(per_worker_budget / bytes_per_variable))
        maximum_nonzeros = min(total_nonzeros, int(per_worker_budget / bytes_per_nonzero))
        if maximum_variables >= 1 and maximum_nonzeros >= 1:
            break
        worker_count -= 1
    if worker_count == 0:
        raise MemoryError("resource preflight: no worker fits the measured block safely.")

    block_count = max(
        math.ceil(total_variables / maximum_variables),
        math.ceil(total_nonzeros / maximum_nonzeros),
    )
    maximum_variables = max(1, math.ceil(total_variables / block_count))
    maximum_nonzeros = max(1, math.ceil(total_nonzeros / block_count))
    scale = total_variables / sum(sample.variables for sample in samples)
    construction = sum(sample.construction_seconds for sample in samples) * scale
    products = sum(
        sample.matvec_seconds + sample.rmatvec_seconds for sample in samples
    ) * scale
    expected_cache = int(
        max(1, sum(sample.cache_bytes for sample in samples) * scale * config.cache_safety_factor)
    )
    if expected_cache > machine.available_cache_bytes:
        raise OSError("resource preflight: estimated operator cache exceeds available storage.")
    expected_peak = fixed_memory + int(
        worker_count * maximum_variables * bytes_per_variable
    )
    ratios = [sample.worker_peak_bytes / sample.variables for sample in samples]
    spread = (max(ratios) - min(ratios)) / max(ratios) if len(ratios) > 1 else 0.25
    uncertainty = min(1.0, max(0.10, spread))
    return BlockResourceRecommendation(
        resource_profile=config.resource_profile,
        maximum_variables_per_block=maximum_variables,
        maximum_nonzeros_per_block=maximum_nonzeros,
        block_count=block_count,
        worker_count=worker_count,
        threads_per_worker=threads,
        expected_peak_memory_bytes=max(1, expected_peak),
        expected_cache_bytes=expected_cache,
        estimated_first_sweep_seconds=(construction + products) / worker_count,
        estimated_cache_hit_sweep_seconds=products / worker_count,
        uncertainty_fraction=uncertainty,
        reason=(
            f"{profile} policy uses {_PROFILE_MEMORY_FRACTIONS[profile]:.0%} of "
            "available memory; worker count is bounded by measured peak memory and CPUs."
        ),
    )


def validate_resource_acceptance(
    recommendation: BlockResourceRecommendation,
    acceptance: AcceptedBlockResourceProposal,
) -> None:
    """Reject stale or unaccepted automatic proposals before partition/execution."""
    if acceptance.recommendation_fingerprint != recommendation.fingerprint:
        raise ValueError("accepted resource proposal does not match the recommendation.")


def apply_accepted_resource_recommendation(
    recommendation: BlockResourceRecommendation,
    acceptance: AcceptedBlockResourceProposal,
    *,
    map_config: BlockCoordinateMAPConfig,
) -> tuple[BlockSizingConfig, BlockCoordinateMAPConfig]:
    """Convert an explicitly accepted proposal into partition and runtime configs."""
    validate_resource_acceptance(recommendation, acceptance)
    sizing = BlockSizingConfig(
        mode="auto",
        maximum_free_variables_per_block=recommendation.maximum_variables_per_block,
        maximum_operator_nonzeros_per_block=recommendation.maximum_nonzeros_per_block,
    )
    execution = replace(
        map_config,
        construction_workers=recommendation.worker_count,
        solver_workers=recommendation.worker_count,
        threads_per_worker=recommendation.threads_per_worker,
    )
    return sizing, execution
