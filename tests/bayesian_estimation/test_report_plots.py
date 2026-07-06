from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

from public_transportation.estimation.bayesian.report_plots import (
    generate_vi_report_plots,
    _plot_loss_curve,
    _plot_loss_curve_recent,
    _plot_posterior_sd_rank,
    _plot_posterior_intervals_top,
)


matplotlib.use("Agg")


def _dummy_result(
    *,
    losses=None,
    posterior_sd=None,
):
    return SimpleNamespace(
        losses=np.asarray(losses if losses is not None else [10.0, 8.0, 6.5, 6.0]),
        posterior_sd=np.asarray(
            posterior_sd if posterior_sd is not None else [0.2, 0.8, 0.1, 0.5]
        ),
    )


def _dummy_diagnostics(*, recent_window_size=2, top_uncertain_parameters=None):
    if top_uncertain_parameters is None:
        top_uncertain_parameters = [
            {"name": "beta_time", "median": -1.0, "q05": -1.5, "q95": -0.5},
            {"name": "beta_cost", "median": -0.4, "q05": -0.8, "q95": -0.1},
            {"name": "asc_bus", "median": 0.2, "q05": -0.2, "q95": 0.6},
        ]

    return {
        "optimization": {
            "recent_window_size": recent_window_size,
        },
        "posterior": {
            "top_uncertain_parameters": top_uncertain_parameters,
        },
    }


def _assert_png_created(path: Path):
    assert path.exists()
    assert path.is_file()
    assert path.stat().st_size > 0
    with path.open("rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


def test_generate_vi_report_plots_creates_output_directory(tmp_path):
    output_dir = tmp_path / "plots"
    result = _dummy_result()
    diagnostics = _dummy_diagnostics()

    assert not output_dir.exists()

    produced = generate_vi_report_plots(
        result=result,
        output_dir=output_dir,
        diagnostics=diagnostics,
    )

    assert output_dir.exists()
    assert output_dir.is_dir()
    assert set(produced) == {
        "loss_curve",
        "loss_curve_recent",
        "posterior_sd_rank",
        "posterior_intervals_top",
    }


def test_generate_vi_report_plots_returns_relative_paths(tmp_path):
    output_dir = tmp_path / "vi_plots"
    result = _dummy_result()
    diagnostics = _dummy_diagnostics()

    produced = generate_vi_report_plots(
        result=result,
        output_dir=output_dir,
        diagnostics=diagnostics,
    )

    assert produced == {
        "loss_curve": "vi_plots/loss_curve.png",
        "loss_curve_recent": "vi_plots/loss_curve_recent.png",
        "posterior_sd_rank": "vi_plots/posterior_sd_rank.png",
        "posterior_intervals_top": "vi_plots/posterior_intervals_top.png",
    }


def test_generate_vi_report_plots_writes_all_png_files(tmp_path):
    output_dir = tmp_path / "plots"
    result = _dummy_result()
    diagnostics = _dummy_diagnostics()

    produced = generate_vi_report_plots(
        result=result,
        output_dir=output_dir,
        diagnostics=diagnostics,
    )

    for relative_name in produced.values():
        path = tmp_path / relative_name
        _assert_png_created(path)


def test_generate_vi_report_plots_accepts_string_output_dir(tmp_path):
    output_dir = tmp_path / "plots"
    result = _dummy_result()
    diagnostics = _dummy_diagnostics()

    produced = generate_vi_report_plots(
        result=result,
        output_dir=str(output_dir),
        diagnostics=diagnostics,
    )

    assert produced["loss_curve"] == "plots/loss_curve.png"
    _assert_png_created(output_dir / "loss_curve.png")


def test_generate_vi_report_plots_computes_diagnostics_when_omitted(monkeypatch, tmp_path):
    output_dir = tmp_path / "plots"
    result = _dummy_result()
    diagnostics = _dummy_diagnostics(recent_window_size=3)
    seen = {}

    def fake_compute_all_diagnostics(result_arg, *, parameter_names, top_k):
        seen["result"] = result_arg
        seen["parameter_names"] = parameter_names
        seen["top_k"] = top_k
        return diagnostics

    monkeypatch.setattr(
        "public_transportation.estimation.bayesian.report_plots.compute_all_diagnostics",
        fake_compute_all_diagnostics,
    )

    produced = generate_vi_report_plots(
        result=result,
        output_dir=output_dir,
        diagnostics=None,
        parameter_names=["a", "b", "c", "d"],
        top_k=3,
    )

    assert seen == {
        "result": result,
        "parameter_names": ["a", "b", "c", "d"],
        "top_k": 3,
    }
    assert set(produced) == {
        "loss_curve",
        "loss_curve_recent",
        "posterior_sd_rank",
        "posterior_intervals_top",
    }


def test_generate_vi_report_plots_does_not_compute_diagnostics_when_provided(monkeypatch, tmp_path):
    output_dir = tmp_path / "plots"
    result = _dummy_result()
    diagnostics = _dummy_diagnostics()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("compute_all_diagnostics should not have been called")

    monkeypatch.setattr(
        "public_transportation.estimation.bayesian.report_plots.compute_all_diagnostics",
        fail_if_called,
    )

    generate_vi_report_plots(
        result=result,
        output_dir=output_dir,
        diagnostics=diagnostics,
    )


def test_generate_vi_report_plots_overwrites_existing_png_files(tmp_path):
    output_dir = tmp_path / "plots"
    output_dir.mkdir()
    existing = output_dir / "loss_curve.png"
    existing.write_text("not a png")

    result = _dummy_result()
    diagnostics = _dummy_diagnostics()

    generate_vi_report_plots(
        result=result,
        output_dir=output_dir,
        diagnostics=diagnostics,
    )

    _assert_png_created(existing)


def test_plot_loss_curve_creates_png(tmp_path):
    result = _dummy_result(losses=[5.0, 4.0, 3.0])
    output_file = tmp_path / "loss.png"

    _plot_loss_curve(result=result, output_file=output_file)

    _assert_png_created(output_file)


def test_plot_loss_curve_accepts_single_loss(tmp_path):
    result = _dummy_result(losses=[5.0])
    output_file = tmp_path / "loss_single.png"

    _plot_loss_curve(result=result, output_file=output_file)

    _assert_png_created(output_file)


def test_plot_loss_curve_handles_numpy_loss_array(tmp_path):
    result = _dummy_result(losses=np.array([3.0, 2.5, 2.0]))
    output_file = tmp_path / "loss_np.png"

    _plot_loss_curve(result=result, output_file=output_file)

    _assert_png_created(output_file)


@pytest.mark.parametrize(
    "recent_window_size, expected_success",
    [
        (1, True),
        (2, True),
        (10, True),
        (0, True),
        (-5, True),
    ],
)
def test_plot_loss_curve_recent_clamps_window_size(
    tmp_path,
    recent_window_size,
    expected_success,
):
    result = _dummy_result(losses=[10.0, 9.0, 8.0, 7.0])
    diagnostics = _dummy_diagnostics(recent_window_size=recent_window_size)
    output_file = tmp_path / f"recent_{recent_window_size}.png"

    _plot_loss_curve_recent(
        result=result,
        diagnostics=diagnostics,
        output_file=output_file,
    )

    _assert_png_created(output_file)


def test_plot_loss_curve_recent_raises_for_missing_optimization_key(tmp_path):
    result = _dummy_result()
    diagnostics = {"posterior": {}}
    output_file = tmp_path / "recent.png"

    with pytest.raises(KeyError):
        _plot_loss_curve_recent(
            result=result,
            diagnostics=diagnostics,
            output_file=output_file,
        )


def test_plot_loss_curve_recent_raises_for_missing_recent_window_size(tmp_path):
    result = _dummy_result()
    diagnostics = {"optimization": {}}
    output_file = tmp_path / "recent.png"

    with pytest.raises(KeyError):
        _plot_loss_curve_recent(
            result=result,
            diagnostics=diagnostics,
            output_file=output_file,
        )


def test_plot_posterior_sd_rank_creates_png(tmp_path):
    result = _dummy_result(posterior_sd=[0.3, 0.1, 0.8])
    output_file = tmp_path / "sd_rank.png"

    _plot_posterior_sd_rank(result=result, output_file=output_file)

    _assert_png_created(output_file)


def test_plot_posterior_sd_rank_accepts_single_parameter(tmp_path):
    result = _dummy_result(posterior_sd=[0.3])
    output_file = tmp_path / "sd_rank_single.png"

    _plot_posterior_sd_rank(result=result, output_file=output_file)

    _assert_png_created(output_file)


def test_plot_posterior_sd_rank_handles_unsorted_input(tmp_path):
    result = _dummy_result(posterior_sd=[0.1, 0.9, 0.3, 0.7])
    output_file = tmp_path / "sd_rank_unsorted.png"

    _plot_posterior_sd_rank(result=result, output_file=output_file)

    _assert_png_created(output_file)


def test_plot_posterior_intervals_top_creates_png(tmp_path):
    diagnostics = _dummy_diagnostics(
        top_uncertain_parameters=[
            {"name": "a", "median": 0.0, "q05": -1.0, "q95": 1.0},
            {"name": "b", "median": 2.0, "q05": 1.5, "q95": 3.0},
        ]
    )
    output_file = tmp_path / "intervals.png"

    _plot_posterior_intervals_top(
        diagnostics=diagnostics,
        output_file=output_file,
        top_k=10,
    )

    _assert_png_created(output_file)


def test_plot_posterior_intervals_top_respects_top_k(monkeypatch, tmp_path):
    diagnostics = _dummy_diagnostics(
        top_uncertain_parameters=[
            {"name": "a", "median": 0.0, "q05": -1.0, "q95": 1.0},
            {"name": "b", "median": 2.0, "q05": 1.5, "q95": 3.0},
            {"name": "c", "median": 4.0, "q05": 3.5, "q95": 5.0},
        ]
    )
    output_file = tmp_path / "intervals_top1.png"

    _plot_posterior_intervals_top(
        diagnostics=diagnostics,
        output_file=output_file,
        top_k=1,
    )

    _assert_png_created(output_file)


def test_plot_posterior_intervals_top_handles_empty_list(tmp_path):
    diagnostics = _dummy_diagnostics(top_uncertain_parameters=[])
    output_file = tmp_path / "intervals_empty.png"

    _plot_posterior_intervals_top(
        diagnostics=diagnostics,
        output_file=output_file,
        top_k=10,
    )

    _assert_png_created(output_file)


def test_plot_posterior_intervals_top_handles_top_k_zero(tmp_path):
    diagnostics = _dummy_diagnostics()
    output_file = tmp_path / "intervals_top0.png"

    _plot_posterior_intervals_top(
        diagnostics=diagnostics,
        output_file=output_file,
        top_k=0,
    )

    _assert_png_created(output_file)


def test_plot_posterior_intervals_top_converts_names_to_strings(tmp_path):
    diagnostics = _dummy_diagnostics(
        top_uncertain_parameters=[
            {"name": 123, "median": 0.0, "q05": -1.0, "q95": 1.0},
        ]
    )
    output_file = tmp_path / "intervals_numeric_name.png"

    _plot_posterior_intervals_top(
        diagnostics=diagnostics,
        output_file=output_file,
        top_k=10,
    )

    _assert_png_created(output_file)


@pytest.mark.parametrize(
    "missing_key",
    ["name", "median", "q05", "q95"],
)
def test_plot_posterior_intervals_top_raises_for_missing_item_key(
    tmp_path,
    missing_key,
):
    item = {"name": "a", "median": 0.0, "q05": -1.0, "q95": 1.0}
    del item[missing_key]
    diagnostics = _dummy_diagnostics(top_uncertain_parameters=[item])
    output_file = tmp_path / "intervals_missing_key.png"

    with pytest.raises(KeyError):
        _plot_posterior_intervals_top(
            diagnostics=diagnostics,
            output_file=output_file,
            top_k=10,
        )


def test_generate_vi_report_plots_handles_empty_uncertain_parameter_list(tmp_path):
    output_dir = tmp_path / "plots"
    result = _dummy_result()
    diagnostics = _dummy_diagnostics(top_uncertain_parameters=[])

    produced = generate_vi_report_plots(
        result=result,
        output_dir=output_dir,
        diagnostics=diagnostics,
    )

    assert set(produced) == {
        "loss_curve",
        "loss_curve_recent",
        "posterior_sd_rank",
        "posterior_intervals_top",
    }
    _assert_png_created(output_dir / "posterior_intervals_top.png")


def test_generate_vi_report_plots_uses_top_k_for_interval_plot(tmp_path):
    output_dir = tmp_path / "plots"
    result = _dummy_result()
    diagnostics = _dummy_diagnostics(
        top_uncertain_parameters=[
            {"name": "a", "median": 0.0, "q05": -1.0, "q95": 1.0},
            {"name": "b", "median": 2.0, "q05": 1.5, "q95": 3.0},
            {"name": "c", "median": 4.0, "q05": 3.5, "q95": 5.0},
        ]
    )

    produced = generate_vi_report_plots(
        result=result,
        output_dir=output_dir,
        diagnostics=diagnostics,
        top_k=1,
    )

    assert "posterior_intervals_top" in produced
    _assert_png_created(output_dir / "posterior_intervals_top.png")


def test_generate_vi_report_plots_raises_if_loss_data_missing(tmp_path):
    output_dir = tmp_path / "plots"
    result = SimpleNamespace(posterior_sd=np.asarray([0.1, 0.2]))
    diagnostics = _dummy_diagnostics()

    with pytest.raises(AttributeError):
        generate_vi_report_plots(
            result=result,
            output_dir=output_dir,
            diagnostics=diagnostics,
        )


def test_generate_vi_report_plots_raises_if_posterior_sd_missing(tmp_path):
    output_dir = tmp_path / "plots"
    result = SimpleNamespace(losses=np.asarray([3.0, 2.0, 1.0]))
    diagnostics = _dummy_diagnostics()

    with pytest.raises(AttributeError):
        generate_vi_report_plots(
            result=result,
            output_dir=output_dir,
            diagnostics=diagnostics,
        )