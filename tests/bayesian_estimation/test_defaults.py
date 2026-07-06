# tests/estimation/test_defaults.py
from __future__ import annotations

import pytest

from public_transportation.estimation.bayesian.defaults import recommend_vi_defaults


@pytest.mark.parametrize("dim", [0, -1, -10])
def test_recommend_vi_defaults_rejects_nonpositive_dimension(dim):
    with pytest.raises(ValueError, match="dim must be positive"):
        recommend_vi_defaults(dim)


@pytest.mark.parametrize("dim", [1, 10, 999])
def test_recommend_vi_defaults_for_small_dimensions(dim):
    defaults = recommend_vi_defaults(dim)

    assert defaults == {
        "guide": "auto_lowrank",
        "lowrank_rank": 20,
        "learning_rate": 1e-2,
        "num_steps": 5_000,
        "num_posterior_draws": 2000,
    }


@pytest.mark.parametrize("dim", [1000, 1001, 4999])
def test_recommend_vi_defaults_for_medium_dimensions(dim):
    defaults = recommend_vi_defaults(dim)

    assert defaults == {
        "guide": "auto_lowrank",
        "lowrank_rank": 50,
        "learning_rate": 1e-2,
        "num_steps": 8_000,
        "num_posterior_draws": 2000,
    }


@pytest.mark.parametrize("dim", [5000, 5001, 100_000])
def test_recommend_vi_defaults_for_large_dimensions(dim):
    defaults = recommend_vi_defaults(dim)

    assert defaults == {
        "guide": "auto_diag",
        "learning_rate": 5e-3,
        "num_steps": 10_000,
        "num_posterior_draws": 2000,
    }


@pytest.mark.parametrize(
    ("dim", "expected_guide", "expected_rank"),
    [
        (999, "auto_lowrank", 20),
        (1000, "auto_lowrank", 50),
        (4999, "auto_lowrank", 50),
        (5000, "auto_diag", None),
    ],
)
def test_recommend_vi_defaults_thresholds(dim, expected_guide, expected_rank):
    defaults = recommend_vi_defaults(dim)

    assert defaults["guide"] == expected_guide
    if expected_rank is None:
        assert "lowrank_rank" not in defaults
    else:
        assert defaults["lowrank_rank"] == expected_rank


def test_recommend_vi_defaults_returns_independent_dictionaries():
    first = recommend_vi_defaults(100)
    second = recommend_vi_defaults(100)

    assert first == second
    assert first is not second

    first["learning_rate"] = 123.0
    assert second["learning_rate"] == 1e-2


@pytest.mark.parametrize("dim", [1, 1000, 5000])
def test_recommend_vi_defaults_common_required_keys(dim):
    defaults = recommend_vi_defaults(dim)

    assert "guide" in defaults
    assert "learning_rate" in defaults
    assert "num_steps" in defaults
    assert "num_posterior_draws" in defaults

    assert defaults["learning_rate"] > 0
    assert defaults["num_steps"] > 0
    assert defaults["num_posterior_draws"] > 0