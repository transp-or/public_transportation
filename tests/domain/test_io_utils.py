from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from public_transportation.domain.io_utils import (
    read_json_dict,
    read_table,
    write_dataclass_json,
    write_table,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _df_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stop_id": ["A", "B", "C"],
            "x": [1, 2, 3],
            "y": [1.5, 2.5, 3.5],
        }
    )


def _assert_frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> None:
    # index is not written, but we keep it robust anyway
    pd.testing.assert_frame_equal(
        a.reset_index(drop=True),
        b.reset_index(drop=True),
        check_dtype=False,  # allow minor dtype changes (e.g., int -> int64)
    )


def _parquet_available() -> bool:
    df = pd.DataFrame({"x": [1]})
    try:
        # if pyarrow/fastparquet missing, this raises ImportError/ValueError
        tmp = df.copy()
        # Use a BytesIO roundtrip to avoid filesystem assumptions (still needs engine).
        import io

        buf = io.BytesIO()
        tmp.to_parquet(buf, index=False)
        buf.seek(0)
        _ = pd.read_parquet(buf)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------
# read_table
# ---------------------------------------------------------------------


def test_read_table_csv(tmp_path: Path) -> None:
    df = _df_sample()
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)

    got = read_table(p)

    _assert_frames_equal(got, df)


@pytest.mark.skipif(not _parquet_available(), reason="Parquet engine (pyarrow/fastparquet) not available")
def test_read_table_parquet(tmp_path: Path) -> None:
    df = _df_sample()
    p = tmp_path / "data.parquet"
    df.to_parquet(p, index=False)

    got = read_table(p)

    _assert_frames_equal(got, df)


def test_read_table_json_records(tmp_path: Path) -> None:
    df = _df_sample()
    p = tmp_path / "data.json"
    df.to_json(p, orient="records", indent=2)

    got = read_table(p)

    _assert_frames_equal(got, df)


def test_read_table_unsupported_extension_raises(tmp_path: Path) -> None:
    p = tmp_path / "data.xlsx"
    p.write_text("dummy", encoding="utf-8")

    with pytest.raises(ValueError) as e:
        _ = read_table(p)

    msg = str(e.value)
    assert "Unsupported table format" in msg
    assert "csv/parquet/json" in msg


# ---------------------------------------------------------------------
# write_table
# ---------------------------------------------------------------------


def test_write_table_csv_and_roundtrip(tmp_path: Path) -> None:
    df = _df_sample()
    p = tmp_path / "out.csv"

    write_table(df, p)
    got = pd.read_csv(p)

    _assert_frames_equal(got, df)


@pytest.mark.skipif(not _parquet_available(), reason="Parquet engine (pyarrow/fastparquet) not available")
def test_write_table_parquet_and_roundtrip(tmp_path: Path) -> None:
    df = _df_sample()
    p = tmp_path / "out.parquet"

    write_table(df, p)
    got = pd.read_parquet(p)

    _assert_frames_equal(got, df)


def test_write_table_json_and_roundtrip(tmp_path: Path) -> None:
    df = _df_sample()
    p = tmp_path / "out.json"

    write_table(df, p)
    got = pd.read_json(p, orient="records")

    _assert_frames_equal(got, df)

    # Also sanity-check that file looks like JSON array / records
    text = p.read_text(encoding="utf-8").strip()
    assert text.startswith("[")
    assert text.endswith("]")


def test_write_table_unsupported_extension_raises(tmp_path: Path) -> None:
    df = _df_sample()
    p = tmp_path / "out.xlsx"

    with pytest.raises(ValueError) as e:
        write_table(df, p)

    msg = str(e.value)
    assert "Unsupported table format" in msg
    assert "csv/parquet/json" in msg


# ---------------------------------------------------------------------
# write_dataclass_json / read_json_dict
# ---------------------------------------------------------------------


@dataclass(slots=True)
class _DC:
    a: int
    b: str


def test_write_dataclass_json_writes_dict_payload(tmp_path: Path) -> None:
    obj = _DC(a=12, b="hello")
    p = tmp_path / "obj.json"

    write_dataclass_json(obj, p)
    got = read_json_dict(p)

    assert got == {"a": 12, "b": "hello"}


def test_write_dataclass_json_accepts_dict(tmp_path: Path) -> None:
    payload = {"x": 1, "y": {"z": 2}}
    p = tmp_path / "payload.json"

    write_dataclass_json(payload, p)
    got = read_json_dict(p)

    assert got == payload


def test_write_dataclass_json_rejects_other_types(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"

    with pytest.raises(TypeError) as e:
        write_dataclass_json(["not", "a", "dict"], p)

    assert "dataclass instance or dict" in str(e.value)


def test_read_json_dict_reads_dict(tmp_path: Path) -> None:
    p = tmp_path / "in.json"
    p.write_text('{"k": 1, "v": "x"}', encoding="utf-8")

    got = read_json_dict(p)

    assert got == {"k": 1, "v": "x"}