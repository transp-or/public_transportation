# tests/estimation/test_guides.py
from __future__ import annotations

import pytest
import numpyro
import numpyro.distributions as dist
from numpyro.infer.autoguide import (
    AutoDiagonalNormal,
    AutoLowRankMultivariateNormal,
    AutoMultivariateNormal,
    AutoNormal,
)

from public_transportation.estimation.bayesian.guides import (
    make_autoguide,
    resolve_lowrank_rank,
)


def _simple_model() -> None:
    numpyro.sample("x", dist.Normal(0.0, 1.0))


@pytest.mark.parametrize(
    ("guide_name", "expected_type"),
    [
        ("auto_diag", AutoDiagonalNormal),
        ("auto_lowrank", AutoLowRankMultivariateNormal),
        ("auto_mvn", AutoMultivariateNormal),
        ("auto_normal", AutoNormal),
    ],
)
def test_make_autoguide_returns_expected_type(guide_name, expected_type):
    guide = make_autoguide(model=_simple_model, guide=guide_name)
    assert isinstance(guide, expected_type)


def test_make_autoguide_default_is_auto_diagonal_normal():
    guide = make_autoguide(model=_simple_model)
    assert isinstance(guide, AutoDiagonalNormal)


def test_auto_lowrank_uses_default_rank_when_none():
    guide = make_autoguide(
        model=_simple_model,
        guide="auto_lowrank",
        lowrank_rank=None,
    )

    assert isinstance(guide, AutoLowRankMultivariateNormal)
    assert guide.rank == 20


def test_auto_lowrank_uses_custom_rank():
    guide = make_autoguide(
        model=_simple_model,
        guide="auto_lowrank",
        lowrank_rank=7,
    )

    assert isinstance(guide, AutoLowRankMultivariateNormal)
    assert guide.rank == 7


def test_auto_lowrank_converts_rank_to_int():
    guide = make_autoguide(
        model=_simple_model,
        guide="auto_lowrank",
        lowrank_rank=3.0,
    )

    assert isinstance(guide, AutoLowRankMultivariateNormal)
    assert guide.rank == 3


@pytest.mark.parametrize("invalid_rank", [0, -1, -10])
def test_auto_lowrank_rejects_nonpositive_rank(invalid_rank):
    with pytest.raises(ValueError, match="lowrank_rank must be a positive integer"):
        make_autoguide(
            model=_simple_model,
            guide="auto_lowrank",
            lowrank_rank=invalid_rank,
        )


def test_auto_lowrank_rejects_non_integer_convertible_rank():
    with pytest.raises((TypeError, ValueError)):
        make_autoguide(
            model=_simple_model,
            guide="auto_lowrank",
            lowrank_rank="not-an-int",
        )


@pytest.mark.parametrize(
    "guide_name",
    ["unknown", "diag", "auto_full", "", None],
)
def test_make_autoguide_rejects_unknown_guide(guide_name):
    with pytest.raises(ValueError, match="Unknown guide"):
        make_autoguide(model=_simple_model, guide=guide_name)


def test_guide_keeps_reference_to_model():
    guide = make_autoguide(model=_simple_model, guide="auto_diag")
    assert guide.model is _simple_model


@pytest.mark.parametrize(
    ("dim", "requested", "expected"),
    [(1000, 20, 20), (10, 20, 10), (1, None, 1), (0, 50, None)],
)
def test_resolve_lowrank_rank_caps_rank_at_reduced_dimension(dim, requested, expected):
    assert resolve_lowrank_rank(dim=dim, lowrank_rank=requested) == expected


@pytest.mark.parametrize("rank", [0, -1])
def test_resolve_lowrank_rank_rejects_invalid_rank_even_for_zero_dimension(rank):
    with pytest.raises(ValueError, match="positive integer"):
        resolve_lowrank_rank(dim=0, lowrank_rank=rank)
