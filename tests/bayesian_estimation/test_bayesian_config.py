# tests/bayesian_estimation/test_config.py
from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass

import pytest

from public_transportation.estimation.bayesian.config import VIConfig


def test_vi_config_is_frozen_slotted_dataclass():
    cfg = VIConfig()

    assert is_dataclass(cfg)
    assert not hasattr(cfg, "__dict__")

    with pytest.raises(FrozenInstanceError):
        cfg.num_steps = 10


def test_vi_config_default_values():
    cfg = VIConfig()

    assert cfg.guide == "auto_diag"
    assert cfg.lowrank_rank is None
    assert cfg.use_base_normal_correction is False
    assert cfg.num_steps == 5_000
    assert cfg.learning_rate == 1e-2
    assert cfg.seed == 0
    assert cfg.num_posterior_draws == 1_000
    assert cfg.log_every == 100


def test_vi_config_field_order_is_stable():
    assert [field.name for field in fields(VIConfig)] == [
        "guide",
        "lowrank_rank",
        "use_base_normal_correction",
        "num_steps",
        "learning_rate",
        "seed",
        "num_posterior_draws",
        "log_every",
    ]


@pytest.mark.parametrize(
    "guide",
    ["auto_diag", "auto_lowrank", "auto_mvn", "auto_normal"],
)
def test_vi_config_accepts_documented_guides(guide: str):
    cfg = VIConfig(guide=guide)

    assert cfg.guide == guide


def test_vi_config_accepts_custom_values():
    cfg = VIConfig(
        guide="auto_lowrank",
        lowrank_rank=5,
        use_base_normal_correction=True,
        num_steps=123,
        learning_rate=0.005,
        seed=42,
        num_posterior_draws=321,
        log_every=17,
    )

    assert cfg.guide == "auto_lowrank"
    assert cfg.lowrank_rank == 5
    assert cfg.use_base_normal_correction is True
    assert cfg.num_steps == 123
    assert cfg.learning_rate == pytest.approx(0.005)
    assert cfg.seed == 42
    assert cfg.num_posterior_draws == 321
    assert cfg.log_every == 17


def test_vi_config_supports_asdict_serialization():
    cfg = VIConfig(
        guide="auto_mvn",
        lowrank_rank=None,
        use_base_normal_correction=True,
        num_steps=10,
        learning_rate=0.1,
        seed=3,
        num_posterior_draws=20,
        log_every=2,
    )

    assert asdict(cfg) == {
        "guide": "auto_mvn",
        "lowrank_rank": None,
        "use_base_normal_correction": True,
        "num_steps": 10,
        "learning_rate": 0.1,
        "seed": 3,
        "num_posterior_draws": 20,
        "log_every": 2,
    }


def test_vi_config_is_hashable():
    cfg1 = VIConfig()
    cfg2 = VIConfig()
    cfg3 = VIConfig(seed=1)

    assert hash(cfg1) == hash(cfg2)
    assert cfg1 == cfg2
    assert cfg1 != cfg3


def test_vi_config_rejects_unknown_field():
    with pytest.raises(TypeError):
        VIConfig(unknown_parameter=1)


def test_vi_config_currently_does_not_validate_runtime_values():
    # VIConfig is intentionally only a frozen container. Runtime validation,
    # if any, belongs in the VI engine or pipeline layer.
    cfg = VIConfig(
        guide="not_a_documented_guide",  # type checkers reject this; runtime dataclasses do not.
        lowrank_rank=-1,
        num_steps=-100,
        learning_rate=-0.5,
        seed=-3,
        num_posterior_draws=-10,
        log_every=0,
    )

    assert cfg.guide == "not_a_documented_guide"
    assert cfg.lowrank_rank == -1
    assert cfg.num_steps == -100
    assert cfg.learning_rate == -0.5
    assert cfg.seed == -3
    assert cfg.num_posterior_draws == -10
    assert cfg.log_every == 0