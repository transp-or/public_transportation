from __future__ import annotations

import io

import pytest

from public_transportation.preprocessing import (
    StructuralZeroProgress,
    structural_zero_tqdm_progress,
)
from public_transportation.preprocessing.structural_zeros.progress import (
    ProgressEmitter,
)


def test_progress_event_validates_counts() -> None:
    event = StructuralZeroProgress("phase", 1, 2, 0.5, "working")
    assert event.throughput_units_per_second == pytest.approx(2.0)
    assert event.estimated_remaining_seconds == pytest.approx(0.5)
    assert event.eta_confidence == "low"
    assert event.completed == 1
    with pytest.raises(ValueError, match="0 <= completed <= total"):
        StructuralZeroProgress("phase", 3, 2, 0.0)


def test_progress_eta_is_unavailable_until_work_completes() -> None:
    event = StructuralZeroProgress("phase", 0, 100, 2.0)
    assert event.throughput_units_per_second is None
    assert event.estimated_remaining_seconds is None
    assert event.eta_confidence == "unavailable"


def test_tqdm_adapter_changes_phase_and_closes() -> None:
    stream = io.StringIO()
    adapter = structural_zero_tqdm_progress(file=stream)
    with adapter as progress:
        progress(StructuralZeroProgress("first", 0, 2, 0.0))
        progress(StructuralZeroProgress("first", 2, 2, 0.1))
        progress(StructuralZeroProgress("second", 0, 1, 0.0))
        progress(StructuralZeroProgress("second", 1, 1, 0.1))
    rendered = stream.getvalue()
    assert "first" in rendered
    assert "second" in rendered
    assert adapter._bar is None


def test_disabled_tqdm_adapter_emits_nothing() -> None:
    stream = io.StringIO()
    with structural_zero_tqdm_progress(enabled=False, file=stream) as progress:
        progress(StructuralZeroProgress("phase", 0, 1, 0.0))
        progress(StructuralZeroProgress("phase", 1, 1, 0.1))
    assert stream.getvalue() == ""


def test_large_loop_count_throttle_emits_bounded_updates() -> None:
    events = []
    emitter = ProgressEmitter(events.append, phase="large", total=250)
    emitter.start()
    for completed in range(1, 251):
        emitter.update(completed)
    assert [event.completed for event in events] == list(range(0, 251, 25))


def test_default_tqdm_stream_is_stderr(capsys) -> None:
    with structural_zero_tqdm_progress() as progress:
        progress(StructuralZeroProgress("phase", 0, 1, 0.0))
        progress(StructuralZeroProgress("phase", 1, 1, 0.1))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "phase" in captured.err
