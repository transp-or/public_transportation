from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Convert common Python/NumPy/JAX-ish objects into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def save_json(data: dict[str, Any], filename: str | Path) -> None:
    """Save a dictionary as formatted JSON."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, sort_keys=True)


def load_json(filename: str | Path) -> dict[str, Any]:
    """Load a JSON file into a Python dictionary."""
    with Path(filename).open("r", encoding="utf-8") as f:
        return json.load(f)
