"""Explicit, restart-safe post-fit route-flow export contracts.

The gravity fit does not reconstruct passenger paths.  This module keeps the
expensive route export separate: callers provide rows from the saved fit and
the case's validated routing implementation, while this layer records a cost
plan and writes a normalized sparse CSV with observable checkpoints.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .viewer_bundle import _ROUTE_FLOW_COLUMNS, _route_flow_row


@dataclass(frozen=True, slots=True)
class GravityRouteFlowExportPlan:
    """Cost and provenance summary shown before route computation starts."""

    available: bool
    source: str
    expected_rows: int | None
    estimated_bytes: int | None
    checkpoint_every_rows: int
    resource_requirements: str
    reason: str
    assignment_fingerprint: str | None = None
    cache_identity: str | None = None
    contains_od_path_contributions: bool | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def plan_gravity_route_flow_export(
    *,
    routing_cache: str | Path | None,
    expected_assignment_fingerprint: str | None = None,
    cache_metadata: Mapping[str, object] | None = None,
    expected_rows: int | None = None,
    estimated_bytes: int | None = None,
    checkpoint_every_rows: int = 10_000,
) -> GravityRouteFlowExportPlan:
    """Inspect persisted routing metadata without preparing or running routing."""
    if checkpoint_every_rows <= 0:
        raise ValueError("checkpoint_every_rows must be positive.")
    path = None if routing_cache is None else Path(routing_cache).expanduser()
    available = path is not None and path.exists()
    reason = "routing cache is available for an explicit post-fit export"
    if not available:
        reason = "no persisted routing cache was found; route computation is required"
    metadata = {} if cache_metadata is None else dict(cache_metadata)
    cache_fingerprint = metadata.get("assignment_fingerprint")
    if expected_assignment_fingerprint is not None and cache_fingerprint != expected_assignment_fingerprint:
        available = False
        reason = (
            "routing cache assignment fingerprint is missing or differs from the fitted result"
        )
    if available and metadata.get("contains_od_path_contributions") is False:
        available = False
        reason = "routing cache lacks OD-to-path contributions; a post-fit route computation is required"
    return GravityRouteFlowExportPlan(
        available=available,
        source=str(path) if path is not None else "",
        expected_rows=expected_rows,
        estimated_bytes=estimated_bytes,
        checkpoint_every_rows=checkpoint_every_rows,
        resource_requirements=(
            "stream one departure-time partition at a time; retain the saved fit "
            "and routing cache, with temporary output plus checkpoint metadata"
        ),
        reason=reason,
        assignment_fingerprint=(None if cache_fingerprint is None else str(cache_fingerprint)),
        cache_identity=(None if metadata.get("cache_identity") is None else str(metadata["cache_identity"])),
        contains_od_path_contributions=(
            None if metadata.get("contains_od_path_contributions") is None
            else bool(metadata["contains_od_path_contributions"])
        ),
    )


def export_gravity_route_flows(
    *,
    route_flow_rows: Iterable[Mapping[str, object]],
    output_path: str | Path,
    checkpoint_path: str | Path | None = None,
    checkpoint_every_rows: int = 10_000,
    progress: Callable[[int], Any] | None = None,
) -> dict[str, object]:
    """Stream normalized route-flow rows and checkpoint progress.

    The supplied iterable must be produced by the saved-fit routing
    implementation. No estimation, assignment preparation, or numerical
    reconstruction happens here.
    """
    if checkpoint_every_rows <= 0:
        raise ValueError("checkpoint_every_rows must be positive.")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path).expanduser().resolve()
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    count = 0

    def save_checkpoint(status: str) -> None:
        if checkpoint is None:
            return
        payload = {"status": status, "rows_written": count, "output_path": str(destination)}
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    try:
        import csv

        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(_ROUTE_FLOW_COLUMNS), extrasaction="raise")
            writer.writeheader()
            for raw in route_flow_rows:
                writer.writerow(_route_flow_row(raw))
                count += 1
                if count % checkpoint_every_rows == 0:
                    save_checkpoint("running")
                    if progress is not None:
                        progress(count)
        os.replace(temporary, destination)
        save_checkpoint("completed")
        if progress is not None:
            progress(count)
    except Exception:
        temporary.unlink(missing_ok=True)
        save_checkpoint("failed")
        raise
    return {"status": "completed", "rows_written": count, "output_path": str(destination)}


__all__ = [
    "GravityRouteFlowExportPlan",
    "export_gravity_route_flows",
    "plan_gravity_route_flow_export",
]
