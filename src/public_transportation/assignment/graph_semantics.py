"""Canonical identity helpers for assignment graph semantics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from .config import AssignmentConfig

ASSIGNMENT_GRAPH_SEMANTICS_VERSION = 3


def transfer_geometry_identity_payload(scenario: Any) -> dict[str, Any]:
    """Return explicit transfer inputs, including the legacy identity fallback."""
    geometry = getattr(scenario, "transfer_geometry", None)
    stops = getattr(scenario, "stops", ())
    stop_values = stops.values() if isinstance(stops, dict) else stops
    stop_ids = sorted(str(getattr(stop, "stop_id")) for stop in stop_values)
    explicit_mapping = {} if geometry is None else dict(geometry.platform_to_station)
    platform_to_station = [
        [platform, explicit_mapping.get(platform, platform)] for platform in stop_ids
    ]
    distance_records = (
        []
        if geometry is None
        else [
            [station, platform, distance]
            for (
                station,
                platform,
            ), distance in geometry.hub_platform_distances_m.items()
        ]
    )
    return {
        "platform_to_station": platform_to_station,
        "hub_platform_distances_m": distance_records,
    }


def assignment_semantics_fingerprint(*, scenario: Any, config: AssignmentConfig) -> str:
    """Fingerprint all graph/cost settings and effective transfer inputs."""
    payload = {
        "graph_semantics_version": ASSIGNMENT_GRAPH_SEMANTICS_VERSION,
        "transfer_geometry": transfer_geometry_identity_payload(scenario),
        "assignment_config": asdict(config),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
