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


def test_structural_zero_progress_exposes_common_hierarchical_payload() -> None:
    event = StructuralZeroProgress("path_metrics", 2, 5, 4.0, "working")

    assert event.schema_version == 1
    assert event.status == "running"
    assert event.completed_units == 2
    assert event.total_units == 5
    assert event.current_unit == "path_metrics-000002"
    assert event.predicted_remaining_seconds == pytest.approx(6.0)

    payload = event.as_dict()
    assert payload["phase"] == "path_metrics"
    assert payload["completed_units"] == 2
    assert payload["total_units"] == 5
    assert payload["predicted_remaining_seconds"] == pytest.approx(6.0)
    assert payload["work_stack"][0]["name"] == "path_metrics"
    assert payload["work_stack"][0]["status"] == "running"
    assert '"schema_version":1' in event.to_json_line()


def test_structural_zero_progress_terminal_payload_has_zero_remaining() -> None:
    event = StructuralZeroProgress("complete", 3, 3, 1.0)
    payload = event.as_dict()
    assert event.status == "completed"
    assert event.current_unit is None
    assert payload["predicted_remaining_seconds"] == pytest.approx(0.0)
    assert payload["status"] == "completed"


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


def test_progress_emitter_uses_shared_eta_and_suppresses_sink_failures() -> None:
    events: list[StructuralZeroProgress] = []
    emitter = ProgressEmitter(events.append, phase="large", total=4)
    emitter.start()
    for completed in range(1, 5):
        emitter.emit(completed)

    assert events[0].predicted_remaining_seconds is None
    assert events[1].predicted_remaining_seconds is not None
    assert events[1].eta_confidence == "low"
    assert events[-1].completed_units == events[-1].total_units == 4
    assert events[-1].predicted_remaining_seconds == pytest.approx(0.0)
    assert events[-1].eta_confidence == "high"

    def broken_sink(_event: StructuralZeroProgress) -> None:
        raise OSError("progress destination unavailable")

    failing = ProgressEmitter(broken_sink, phase="large", total=1)
    failing.start()
    failing.emit(1)
    assert failing.reporting_failures == 2  # initial and terminal events
    assert "progress destination unavailable" in (failing.last_reporting_error or "")


def test_default_tqdm_stream_is_stderr(capsys) -> None:
    with structural_zero_tqdm_progress() as progress:
        progress(StructuralZeroProgress("phase", 0, 1, 0.0))
        progress(StructuralZeroProgress("phase", 1, 1, 0.1))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "phase" in captured.err
