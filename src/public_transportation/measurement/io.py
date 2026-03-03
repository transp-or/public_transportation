from __future__ import annotations

from pathlib import Path

from public_transportation.domain.time_of_day import TimeOfDay

from .schema import MeasurementRecord, MeasurementTable, MeasurementType


def _parse_required_str(row: dict, col: str) -> str:
    v = row.get(col, None)
    if v is None:
        raise ValueError(f"Missing required column {col!r}.")
    s = str(v).strip()
    if not s:
        raise ValueError(f"Column {col!r} must be non-empty.")
    return s


def _parse_optional_str(row: dict, col: str) -> str | None:
    v = row.get(col, None)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _parse_time_hms(row: dict, col: str = "time") -> TimeOfDay:
    t = _parse_required_str(row, col)
    # Time must be HH:MM:SS (as requested)
    parts = t.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid time {t!r}: expected HH:MM:SS.")
    return TimeOfDay.parse(t)


def _parse_measurement_type(row: dict, col: str = "measurement_type") -> MeasurementType:
    s = _parse_required_str(row, col)
    try:
        return MeasurementType(s)
    except ValueError as e:
        raise ValueError(
            f"Invalid measurement_type {s!r}. Expected one of "
            f"{[m.value for m in MeasurementType]}."
        ) from e


def _parse_float(row: dict, col: str = "value") -> float:
    v = row.get(col, None)
    if v is None:
        raise ValueError(f"Missing required column {col!r}.")
    try:
        return float(v)
    except Exception as e:
        raise ValueError(f"Invalid float in column {col!r}: {v!r}.") from e


def record_from_row(row: dict) -> MeasurementRecord:
    """Convert one dict-like row into a MeasurementRecord."""
    method_id = _parse_required_str(row, "method_id")
    mtype = _parse_measurement_type(row, "measurement_type")
    stop_id = _parse_required_str(row, "stop_id")
    t = _parse_time_hms(row, "time")
    value = _parse_float(row, "value")
    trip_id = _parse_optional_str(row, "trip_id")
    line_id = _parse_optional_str(row, "line_id")
    return MeasurementRecord(
        method_id=method_id,
        measurement_type=mtype,
        stop_id=stop_id,
        time=t,
        value=value,
        trip_id=trip_id,
        line_id=line_id,
    )


def read_measurements_csv(path: str | Path) -> MeasurementTable:
    """Read measurements from CSV.

    Expected columns:
      - method_id (str, required)
      - measurement_type (load|boarding|alighting, required)
      - stop_id (str, required)
      - time (HH:MM:SS, required)
      - value (float, required)
      - trip_id (str, optional)
      - line_id (str, optional)

    Duplicates are NOT allowed.
    """
    import csv

    p = Path(path)
    with p.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header.")
        rows = list(reader)

    records = [record_from_row(r) for r in rows]
    return MeasurementTable.from_records(records)


def write_measurements_csv(table: MeasurementTable, path: str | Path) -> None:
    """Write measurements to CSV using the canonical schema."""
    import csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method_id", "measurement_type", "stop_id", "time", "value", "trip_id", "line_id"]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in table.records:
            w.writerow(
                {
                    "method_id": r.method_id,
                    "measurement_type": r.measurement_type.value,
                    "stop_id": r.stop_id,
                    "time": r.time.to_string(include_seconds=True),
                    "value": r.value,
                    "trip_id": r.trip_id or "",
                    "line_id": r.line_id or "",
                }
            )


def read_measurements_parquet(path: str | Path) -> MeasurementTable:
    """Read measurements from Parquet.

    Requires pandas + pyarrow (or fastparquet). Duplicates are NOT allowed.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise ImportError("Reading Parquet requires pandas.") from e

    p = Path(path)
    df = pd.read_parquet(p)
    rows = df.to_dict(orient="records")
    records = [record_from_row(r) for r in rows]
    return MeasurementTable.from_records(records)


def write_measurements_parquet(table: MeasurementTable, path: str | Path) -> None:
    """Write measurements to Parquet (pandas backend)."""
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise ImportError("Writing Parquet requires pandas.") from e

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in table.records:
        rows.append(
            {
                "method_id": r.method_id,
                "measurement_type": r.measurement_type.value,
                "stop_id": r.stop_id,
                "time": r.time.to_string(include_seconds=True),
                "value": float(r.value),
                "trip_id": r.trip_id,
                "line_id": r.line_id,
            }
        )

    df = pd.DataFrame(rows)
    df.to_parquet(p, index=False)