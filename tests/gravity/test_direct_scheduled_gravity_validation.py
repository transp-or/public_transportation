"""Fresh-process acceptance test for the public temporal gravity example."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs/source/examples/direct_scheduled_gravity_validation.py"


def _run(cache_directory: Path, *, require_reuse: bool) -> dict[str, object]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--cache-directory",
        str(cache_directory),
    ]
    if require_reuse:
        command.append("--require-cache-reuse")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    summary_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("SUMMARY ")
    )
    return json.loads(summary_line.removeprefix("SUMMARY "))


def test_public_gravity_objective_and_gradient_reuse_in_fresh_process(tmp_path):
    first = _run(tmp_path, require_reuse=False)
    second = _run(tmp_path, require_reuse=True)

    assert not first["cache_reused"]
    assert first["progress_event_count"] > 0
    assert first["construction_seconds"] >= 0.0
    assert first["objective_absolute_difference"] <= 5.0e-3
    assert first["gradient_maximum_absolute_difference"] <= 5.0e-3
    assert second["cache_reused"]
    assert second["progress_event_count"] >= 1
    assert second["construction_progress_event_count"] == 0
    assert second["construction_seconds"] is None
    assert second["objective_absolute_difference"] <= 5.0e-3
    assert second["gradient_maximum_absolute_difference"] <= 5.0e-3
