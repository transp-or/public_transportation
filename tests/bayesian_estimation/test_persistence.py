# tests/variational/test_persistence.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from public_transportation.estimation.bayesian.persistence import (
    _load_json,
    _save_json,
    _to_jsonable,
    load_vi_result,
    save_vi_result,
)
from public_transportation.estimation.bayesian.results import VIResult


def _make_vi_result(
    *,
    params=None,
    svi_state=None,
    dim: int = 3,
    losses: np.ndarray | None = None,
) -> VIResult:
    if losses is None:
        losses = np.asarray([10.0, 5.0, 2.5], dtype=float)

    posterior_samples_theta = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [1.5, 2.5, 3.5],
            [2.0, 3.0, 4.0],
        ],
        dtype=float,
    )

    return VIResult(
        guide="diagonal",
        dim=dim,
        use_base_normal_correction=True,
        svi_state=svi_state,
        params=params,
        losses=losses,
        posterior_samples_theta=posterior_samples_theta,
        seed=123,
        num_steps=1000,
        learning_rate=0.01,
        lowrank_rank=None,
        num_posterior_draws=posterior_samples_theta.shape[0],
        runtime_seconds=12.5,
        timestamp="2026-07-02T10:00:00",
        posterior_mean=np.asarray([1.5, 2.5, 3.5], dtype=float),
        posterior_sd=np.asarray([0.5, 0.5, 0.5], dtype=float),
        posterior_q05=np.asarray([1.05, 2.05, 3.05], dtype=float),
        posterior_q50=np.asarray([1.5, 2.5, 3.5], dtype=float),
        posterior_q95=np.asarray([1.95, 2.95, 3.95], dtype=float),
    )


def test_to_jsonable_converts_common_objects():
    value = {
        Path("abc"): {
            "path": Path("some/file.txt"),
            "array": np.asarray([1, 2, 3], dtype=np.int64),
            "scalar": np.float64(1.25),
            "nested": (np.int64(4), Path("x")),
        }
    }

    converted = _to_jsonable(value)

    assert converted == {
        "abc": {
            "path": "some/file.txt",
            "array": [1, 2, 3],
            "scalar": 1.25,
            "nested": [4, "x"],
        }
    }


def test_save_and_load_json_round_trip(tmp_path: Path):
    filename = tmp_path / "data.json"
    data = {
        "b": np.asarray([1, 2]),
        "a": Path("folder/file.txt"),
        "c": {"x": np.int64(7)},
    }

    _save_json(data, filename)
    loaded = _load_json(filename)

    assert loaded == {
        "a": "folder/file.txt",
        "b": [1, 2],
        "c": {"x": 7},
    }

    raw = filename.read_text(encoding="utf-8")
    assert raw.startswith("{\n")
    assert '"a"' in raw


def test_save_vi_result_creates_expected_files_by_default(tmp_path: Path):
    result = _make_vi_result()

    save_vi_result(result, tmp_path)

    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "arrays.npz").exists()
    assert (tmp_path / "losses.csv").exists()
    assert (tmp_path / "posterior_summary.csv").exists()
    assert not (tmp_path / "params.npz").exists()
    assert not (tmp_path / "svi_state.npy").exists()


def test_save_vi_result_metadata_content(tmp_path: Path):
    result = _make_vi_result()

    save_vi_result(result, tmp_path)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert metadata == {
        "guide": "diagonal",
        "dim": 3,
        "use_base_normal_correction": True,
        "seed": 123,
        "num_steps": 1000,
        "learning_rate": 0.01,
        "lowrank_rank": None,
        "num_posterior_draws": 3,
        "runtime_seconds": 12.5,
        "timestamp": "2026-07-02T10:00:00",
    }


def test_save_vi_result_arrays_content(tmp_path: Path):
    result = _make_vi_result()

    save_vi_result(result, tmp_path)

    arrays = np.load(tmp_path / "arrays.npz")

    assert np.allclose(arrays["losses"], result.losses)
    assert np.allclose(arrays["posterior_samples_theta"], result.posterior_samples_theta)
    assert np.allclose(arrays["posterior_mean"], result.posterior_mean)
    assert np.allclose(arrays["posterior_sd"], result.posterior_sd)
    assert np.allclose(arrays["posterior_q05"], result.posterior_q05)
    assert np.allclose(arrays["posterior_q50"], result.posterior_q50)
    assert np.allclose(arrays["posterior_q95"], result.posterior_q95)


def test_save_vi_result_writes_losses_csv(tmp_path: Path):
    result = _make_vi_result()

    save_vi_result(result, tmp_path)

    content = (tmp_path / "losses.csv").read_text(encoding="utf-8")

    assert content.splitlines()[0] == "loss"
    loaded = np.loadtxt(tmp_path / "losses.csv", delimiter=",", skiprows=1)
    assert np.allclose(loaded, result.losses)


def test_save_vi_result_writes_posterior_summary_csv(tmp_path: Path):
    result = _make_vi_result()

    save_vi_result(result, tmp_path)

    content = (tmp_path / "posterior_summary.csv").read_text(encoding="utf-8")
    assert content.splitlines()[0] == "index,mean,sd,q05,q50,q95"

    loaded = np.loadtxt(tmp_path / "posterior_summary.csv", delimiter=",", skiprows=1)
    expected = np.column_stack(
        [
            np.arange(result.dim),
            result.posterior_mean,
            result.posterior_sd,
            result.posterior_q05,
            result.posterior_q50,
            result.posterior_q95,
        ]
    )
    assert np.allclose(loaded, expected)


def test_load_vi_result_round_trip_without_optional_state(tmp_path: Path):
    result = _make_vi_result()

    save_vi_result(result, tmp_path)
    loaded = load_vi_result(tmp_path)

    assert loaded.guide == result.guide
    assert loaded.dim == result.dim
    assert loaded.use_base_normal_correction == result.use_base_normal_correction
    assert loaded.svi_state is None
    assert loaded.params is None
    assert np.allclose(loaded.losses, result.losses)
    assert np.allclose(loaded.posterior_samples_theta, result.posterior_samples_theta)
    assert loaded.seed == result.seed
    assert loaded.num_steps == result.num_steps
    assert loaded.learning_rate == result.learning_rate
    assert loaded.lowrank_rank == result.lowrank_rank
    assert loaded.num_posterior_draws == result.num_posterior_draws
    assert loaded.runtime_seconds == result.runtime_seconds
    assert loaded.timestamp == result.timestamp
    assert np.allclose(loaded.posterior_mean, result.posterior_mean)
    assert np.allclose(loaded.posterior_sd, result.posterior_sd)
    assert np.allclose(loaded.posterior_q05, result.posterior_q05)
    assert np.allclose(loaded.posterior_q50, result.posterior_q50)
    assert np.allclose(loaded.posterior_q95, result.posterior_q95)


def test_save_and_load_params_when_requested(tmp_path: Path):
    params = {
        "loc": np.asarray([1.0, 2.0, 3.0]),
        "scale": np.asarray([0.1, 0.2, 0.3]),
    }
    result = _make_vi_result(params=params)

    save_vi_result(result, tmp_path, save_params=True)
    loaded = load_vi_result(tmp_path, load_params=True)

    assert (tmp_path / "params.npz").exists()
    assert loaded.params is not None
    assert set(loaded.params) == {"loc", "scale"}
    assert np.allclose(loaded.params["loc"], params["loc"])
    assert np.allclose(loaded.params["scale"], params["scale"])


def test_params_are_not_loaded_unless_requested(tmp_path: Path):
    result = _make_vi_result(params={"loc": np.asarray([1.0])})

    save_vi_result(result, tmp_path, save_params=True)
    loaded = load_vi_result(tmp_path, load_params=False)

    assert (tmp_path / "params.npz").exists()
    assert loaded.params is None


def test_load_params_missing_file_returns_none(tmp_path: Path):
    result = _make_vi_result(params=None)

    save_vi_result(result, tmp_path, save_params=False)
    loaded = load_vi_result(tmp_path, load_params=True)

    assert loaded.params is None


def test_save_params_accepts_array_like_values(tmp_path: Path):
    result = _make_vi_result(
        params={
            "scalar": 1.25,
            "vector": [1.0, 2.0, 3.0],
        }
    )

    save_vi_result(result, tmp_path, save_params=True)
    loaded = load_vi_result(tmp_path, load_params=True)

    assert (tmp_path / "params.npz").exists()
    assert loaded.params is not None
    assert set(loaded.params) == {"scalar", "vector"}
    assert np.allclose(loaded.params["scalar"], np.asarray(1.25))
    assert np.allclose(loaded.params["vector"], np.asarray([1.0, 2.0, 3.0]))


def test_save_and_load_svi_state_when_requested(tmp_path: Path):
    svi_state = {"iteration": 12, "step_size": 0.01}
    result = _make_vi_result(svi_state=svi_state)

    save_vi_result(result, tmp_path, save_svi_state=True)
    loaded = load_vi_result(tmp_path, load_svi_state=True)

    assert (tmp_path / "svi_state.npy").exists()
    assert loaded.svi_state == svi_state


def test_svi_state_is_not_loaded_unless_requested(tmp_path: Path):
    result = _make_vi_result(svi_state={"iteration": 12})

    save_vi_result(result, tmp_path, save_svi_state=True)
    loaded = load_vi_result(tmp_path, load_svi_state=False)

    assert (tmp_path / "svi_state.npy").exists()
    assert loaded.svi_state is None


def test_load_svi_state_missing_file_returns_none(tmp_path: Path):
    result = _make_vi_result(svi_state=None)

    save_vi_result(result, tmp_path, save_svi_state=False)
    loaded = load_vi_result(tmp_path, load_svi_state=True)

    assert loaded.svi_state is None


def test_save_vi_result_creates_nested_output_directory(tmp_path: Path):
    output_dir = tmp_path / "nested" / "vi" / "result"
    result = _make_vi_result()

    save_vi_result(result, output_dir)

    assert output_dir.exists()
    assert (output_dir / "metadata.json").exists()


def test_load_vi_result_preserves_lowrank_rank_when_present(tmp_path: Path):
    result = _make_vi_result()
    result = VIResult(
        **{
            **result.__dict__,
            "guide": "lowrank",
            "lowrank_rank": 2,
        }
    )

    save_vi_result(result, tmp_path)
    loaded = load_vi_result(tmp_path)

    assert loaded.guide == "lowrank"
    assert loaded.lowrank_rank == 2


def test_load_vi_result_raises_if_metadata_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_vi_result(tmp_path)


def test_load_vi_result_raises_if_arrays_missing(tmp_path: Path):
    result = _make_vi_result()

    save_vi_result(result, tmp_path)
    (tmp_path / "arrays.npz").unlink()

    with pytest.raises(FileNotFoundError):
        load_vi_result(tmp_path)


def test_load_vi_result_raises_if_required_array_missing(tmp_path: Path):
    result = _make_vi_result()

    save_vi_result(result, tmp_path)

    np.savez_compressed(
        tmp_path / "arrays.npz",
        losses=result.losses,
        # posterior_samples_theta intentionally omitted
        posterior_mean=result.posterior_mean,
        posterior_sd=result.posterior_sd,
        posterior_q05=result.posterior_q05,
        posterior_q50=result.posterior_q50,
        posterior_q95=result.posterior_q95,
    )

    with pytest.raises(KeyError):
        load_vi_result(tmp_path)


def test_round_trip_accepts_string_paths(tmp_path: Path):
    result = _make_vi_result()
    output_dir = tmp_path / "string-path"

    save_vi_result(result, str(output_dir))
    loaded = load_vi_result(str(output_dir))

    assert np.allclose(loaded.posterior_mean, result.posterior_mean)


def test_loaded_arrays_are_numpy_arrays(tmp_path: Path):
    result = _make_vi_result()

    save_vi_result(result, tmp_path)
    loaded = load_vi_result(tmp_path)

    assert isinstance(loaded.losses, np.ndarray)
    assert isinstance(loaded.posterior_samples_theta, np.ndarray)
    assert isinstance(loaded.posterior_mean, np.ndarray)
    assert isinstance(loaded.posterior_sd, np.ndarray)
    assert isinstance(loaded.posterior_q05, np.ndarray)
    assert isinstance(loaded.posterior_q50, np.ndarray)
    assert isinstance(loaded.posterior_q95, np.ndarray)