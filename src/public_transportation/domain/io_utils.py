from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    """
    Read a tabular file based on extension.

    Supported:
    - .csv
    - .parquet
    - .json (records-oriented)

    :param path: File path.
    :return: DataFrame.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".json":
        return pd.read_json(p, orient="records")
    raise ValueError(f"Unsupported table format: {p.suffix}. Use csv/parquet/json.")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """
    Write a DataFrame based on extension.

    :param df: DataFrame to write.
    :param path: Output file path.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        df.to_csv(p, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(p, index=False)
        return
    if suffix == ".json":
        df.to_json(p, orient="records", indent=2)
        return
    raise ValueError(f"Unsupported table format: {p.suffix}. Use csv/parquet/json.")


def write_dataclass_json(obj: Any, path: str | Path) -> None:
    """
    Write a dataclass (or dict-like) to JSON.

    :param obj: Dataclass instance or dict.
    :param path: Output file.
    """
    import json

    p = Path(path)
    if is_dataclass(obj):
        payload = asdict(obj)
    elif isinstance(obj, dict):
        payload = obj
    else:
        raise TypeError("obj must be a dataclass instance or dict.")
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json_dict(path: str | Path) -> dict[str, Any]:
    """
    Read JSON file as dict.

    :param path: Input file.
    :return: Parsed dict.
    """
    import json

    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))