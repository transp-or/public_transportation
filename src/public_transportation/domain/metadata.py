from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class Metadata:
    """
    Scenario metadata (provenance and units).

    :param title: Human-friendly title.
    :param description: Optional description.
    :param timezone: IANA timezone name.
    :param cost_unit: Description of cost unit (e.g., "minutes", "generalized_minutes", "CHF").
    :param created_at: ISO timestamp string.
    :param sources: Optional list of data sources.
    :param extra: Arbitrary extra metadata (GUI-friendly).
    """
    title: str
    description: str | None = None
    timezone: str = "Europe/Zurich"
    cost_unit: str = "minutes"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    sources: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)