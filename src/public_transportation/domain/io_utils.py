from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .issues import Issue, Severity, ValidationReport


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a tabular file based on extension.

    Supported:
    - .csv
    - .parquet
    - .json (records-oriented)

    This helper does not perform schema validation, but it can be used in
    conjunction with model-level validation (see :func:`validate_time_columns`).

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


# ============================================================================
# Time parsing helpers (for user-friendly IO)
# ============================================================================


def _parse_time_to_seconds(value: Any) -> int | None:
    """Parse a time value into seconds-from-midnight.

    Accepted inputs:
    - int/float (interpreted as seconds)
    - "HH:MM" or "HH:MM:SS" strings

    Returns None for missing values (NaN/None/empty string).

    :param value: Value to parse.
    :return: Seconds-from-midnight, or None if missing.
    :raises ValueError: If the value cannot be parsed or is out of range.
    """
    if value is None:
        return None

    # pandas may pass NaN as float
    try:
        if pd.isna(value):
            return None
    except Exception:
        # pd.isna may fail for some custom objects
        pass

    if isinstance(value, (int,)):
        sec = int(value)
        if sec < 0:
            raise ValueError(f"Time seconds must be >= 0, got {sec}.")
        return sec

    if isinstance(value, (float,)):
        if not float(value).is_integer():
            raise ValueError(f"Time seconds must be an integer value, got {value}.")
        sec = int(value)
        if sec < 0:
            raise ValueError(f"Time seconds must be >= 0, got {sec}.")
        return sec

    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return None
        parts = s.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid time format '{value}'. Expected HH:MM or HH:MM:SS.")
        try:
            hh = int(parts[0])
            mm = int(parts[1])
            ss = int(parts[2]) if len(parts) == 3 else 0
        except ValueError as e:
            raise ValueError(f"Invalid time format '{value}'. Expected integers in HH:MM(:SS).") from e

        if hh < 0 or mm < 0 or ss < 0:
            raise ValueError(f"Invalid time '{value}': negative component.")
        if mm >= 60 or ss >= 60:
            raise ValueError(f"Invalid time '{value}': minutes/seconds out of range.")
        sec = hh * 3600 + mm * 60 + ss
        return sec

    raise ValueError(f"Unsupported time value type: {type(value).__name__}.")


def coerce_time_columns_to_seconds(
    df: pd.DataFrame,
    columns: list[str],
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Coerce selected columns to seconds-from-midnight.

    Columns are converted using :func:`_parse_time_to_seconds`. The resulting
    dtype is pandas nullable integer (Int64), allowing missing values.

    :param df: Input DataFrame.
    :param columns: Column names to coerce.
    :param inplace: If True, modify df in place.
    :return: DataFrame with coerced columns.
    :raises KeyError: If any requested column is missing.
    :raises ValueError: If any value cannot be parsed.
    """
    out = df if inplace else df.copy()

    missing = [c for c in columns if c not in out.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    for c in columns:
        parsed = out[c].map(_parse_time_to_seconds)
        # Ensure integer semantics with NA support
        out[c] = parsed.astype("Int64")

    return out


def validate_time_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    allow_none: bool = True,
    max_seconds: int = 48 * 3600,
) -> ValidationReport:
    """Validate that time columns are parseable and within range.

    This produces a :class:`~public_transportation.issues.ValidationReport`
    rather than raising, so it can be used as part of Scenario/Timetable
    validation.

    :param df: DataFrame containing the columns.
    :param columns: Columns to validate.
    :param allow_none: If True, missing values are allowed.
    :param max_seconds: Upper bound for allowed times (default 48h).
    :return: ValidationReport.
    """
    report = ValidationReport()

    for col in columns:
        if col not in df.columns:
            report.add(
                Issue(
                    code="TIME_COL_MISSING",
                    message=f"Missing required time column '{col}'.",
                    severity=Severity.ERROR,
                    location=f"table[{col}]",
                )
            )
            continue

        for idx, val in df[col].items():
            try:
                sec = _parse_time_to_seconds(val)
            except ValueError as e:
                report.add(
                    Issue(
                        code="TIME_VALUE_INVALID",
                        message=str(e),
                        severity=Severity.ERROR,
                        location=f"table[{col}][row={idx}]",
                    )
                )
                continue

            if sec is None:
                if not allow_none:
                    report.add(
                        Issue(
                            code="TIME_VALUE_MISSING",
                            message=f"Missing time value in column '{col}'.",
                            severity=Severity.ERROR,
                            location=f"table[{col}][row={idx}]",
                        )
                    )
                continue

            if sec < 0 or sec > max_seconds:
                report.add(
                    Issue(
                        code="TIME_VALUE_RANGE",
                        message=f"Time value {sec} out of range [0, {max_seconds}] in column '{col}'.",
                        severity=Severity.ERROR,
                        location=f"table[{col}][row={idx}]",
                    )
                )

    return report