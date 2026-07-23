"""Canonical semantic fingerprint for a loaded public-transport scenario."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from public_transportation.domain.scenario import Scenario


def scenario_fingerprint_payload_json(scenario: Scenario) -> str:
    """Return canonical JSON covering every domain input used by preprocessing."""
    timetable = scenario.timetable
    metadata = _dataclass_payload(scenario.metadata)
    # This field has a clock-based default when omitted from metadata.json. It
    # is provenance, not a model input, so excluding it keeps reloads stable.
    metadata.pop("created_at", None)
    payload = {
        "metadata": metadata,
        "stops": sorted(
            (_dataclass_payload(stop) for stop in scenario.stops),
            key=lambda item: str(item["stop_id"]),
        ),
        "lines": sorted(
            (_dataclass_payload(line) for line in scenario.lines),
            key=lambda item: str(item["line_id"]),
        ),
        "time_bins": sorted(
            (_dataclass_payload(time_bin) for time_bin in scenario.time_bins),
            key=lambda item: str(item["bin_id"]),
        ),
        "demand": sorted(
            (_dataclass_payload(record) for record in scenario.demand.records),
            key=lambda item: (
                str(item["origin_stop_id"]),
                str(item["dest_stop_id"]),
                str(item["time_bin_id"]),
            ),
        ),
        "timetable": None,
    }
    if timetable is not None:
        payload["timetable"] = {
            "trips": sorted(
                (_dataclass_payload(trip) for trip in timetable.trips),
                key=lambda item: str(item["trip_id"]),
            ),
            "stop_times": sorted(
                (_dataclass_payload(stop_time) for stop_time in timetable.stop_times),
                key=lambda item: (
                    str(item["trip_id"]),
                    int(item["sequence"]),
                    str(item["stop_id"]),
                ),
            ),
        }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def fingerprint_scenario(scenario: Scenario) -> str:
    """Return the SHA-256 fingerprint of the canonical scenario payload."""
    payload = scenario_fingerprint_payload_json(scenario)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dataclass_payload(value: Any) -> dict[str, Any]:
    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError(f"Expected a dataclass instance, got {type(value).__name__}.")
    return {
        field.name: _jsonable(getattr(value, field.name))
        for field in fields(value)
        if field.init
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _dataclass_payload(value)
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported scenario fingerprint value: {type(value).__name__}.")
