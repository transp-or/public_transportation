"""Materialize a reviewed time-discretization recommendation as ``time_bins.csv``.

The profiling command intentionally writes JSON only.  This separate command is
the explicit, reviewable handoff into a case-study scenario: it validates a
candidate from that JSON report and writes the canonical three-column
``time_bins.csv`` consumed by :class:`public_transportation.domain.Scenario`.
Existing files are never overwritten unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from public_transportation.preprocessing.time_discretization import SCHEMA_VERSION


def _read_report(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read recommendation report {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Recommendation report is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Recommendation report must contain a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported recommendation schema version: "
            f"{payload.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    return payload


def _select_candidate(report: Mapping[str, Any], candidate_name: str) -> Mapping[str, Any]:
    if candidate_name == "recommendation":
        candidate = report.get("recommendation")
        if not isinstance(candidate, Mapping):
            raise ValueError("Recommendation report has no valid 'recommendation' object")
        return candidate

    candidates = report.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("Recommendation report has no candidate list")
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping) and candidate.get("name") == candidate_name
    ]
    if not matches:
        raise ValueError(f"No candidate named {candidate_name!r} was found in the report")
    if len(matches) > 1:
        raise ValueError(f"Candidate name {candidate_name!r} is not unique")
    return matches[0]


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer number of seconds")
    return int(value)


def _validated_time_bins(candidate: Mapping[str, Any]) -> list[dict[str, int | str]]:
    if candidate.get("valid") is not True:
        reasons = candidate.get("invalid_reasons", [])
        raise ValueError(f"Selected candidate is invalid: {reasons!r}")
    raw_bins = candidate.get("time_bins")
    if not isinstance(raw_bins, Sequence) or isinstance(raw_bins, (str, bytes)) or not raw_bins:
        raise ValueError("Selected candidate contains no time_bins")

    result: list[dict[str, int | str]] = []
    previous_end: int | None = None
    identifiers: set[str] = set()
    for index, raw_bin in enumerate(raw_bins):
        if not isinstance(raw_bin, Mapping):
            raise ValueError(f"time_bins[{index}] must be a JSON object")
        bin_id = raw_bin.get("bin_id")
        if not isinstance(bin_id, str) or not bin_id.strip():
            raise ValueError(f"time_bins[{index}].bin_id must be a non-empty string")
        if bin_id in identifiers:
            raise ValueError(f"Duplicate time-bin identifier: {bin_id!r}")
        identifiers.add(bin_id)
        start_s = _integer(raw_bin.get("start_s"), field=f"time_bins[{index}].start_s")
        end_s = _integer(raw_bin.get("end_s"), field=f"time_bins[{index}].end_s")
        if start_s < 0 or end_s <= start_s:
            raise ValueError(f"time_bins[{index}] must satisfy 0 <= start_s < end_s")
        if previous_end is not None and start_s != previous_end:
            raise ValueError(
                "Selected time bins must form contiguous half-open intervals; "
                f"bin {index} starts at {start_s}, previous bin ends at {previous_end}"
            )
        result.append({"bin_id": bin_id, "start_s": start_s, "end_s": end_s})
        previous_end = end_s
    return result


def materialize_time_bins(
    recommendation_path: str | Path,
    output_path: str | Path,
    *,
    candidate_name: str = "recommendation",
    overwrite: bool = False,
) -> list[dict[str, int | str]]:
    """Validate and atomically write one candidate as canonical ``time_bins.csv``."""

    recommendation = Path(recommendation_path)
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing time-bin file {output}; use --overwrite explicitly"
        )
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output.parent}")

    candidate = _select_candidate(_read_report(recommendation), candidate_name)
    time_bins = _validated_time_bins(candidate)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=("bin_id", "start_s", "end_s"))
            writer.writeheader()
            writer.writerows(time_bins)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return time_bins


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommendation-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        default="recommendation",
        help="Candidate name from the report, or 'recommendation' (default)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing time_bins.csv",
    )
    args = parser.parse_args(argv)
    bins = materialize_time_bins(
        args.recommendation_json,
        args.output,
        candidate_name=args.candidate,
        overwrite=args.overwrite,
    )
    print(f"Wrote {len(bins)} time bins to {args.output}")


if __name__ == "__main__":
    main()
