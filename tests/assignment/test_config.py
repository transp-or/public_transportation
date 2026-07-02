from __future__ import annotations

from dataclasses import fields

import pytest

from public_transportation.assignment.config import AssignmentConfig


def _field_names() -> set[str]:
    return {field.name for field in fields(AssignmentConfig)}


def test_default_config_is_valid():
    cfg = AssignmentConfig()
    cfg.validate()


def test_config_is_slots_dataclass():
    cfg = AssignmentConfig()

    assert hasattr(cfg, "__slots__")
    assert not hasattr(cfg, "__dict__")


def test_expected_core_fields_exist():
    names = _field_names()

    expected = {
        "beta_in_vehicle",
        "beta_transfer",
        "beta_wait",
        "max_access_deviation_min",
        "max_transfer_wait_min",
        "min_dwell_s",
        "egress_cost",
        "use_global_egress_links",
        "mask_strategy",
        "beta_early",
        "beta_late",
        "theta_default",
        "theta_min",
        "use_capacity_penalty",
        "capacity_penalty_alpha",
        "capacity_penalty_kappa",
    }

    assert expected.issubset(names)


def test_no_obsolete_beta_access_field():
    assert "beta_access" not in _field_names()


def test_custom_valid_config_is_valid():
    cfg = AssignmentConfig(
        beta_in_vehicle=1.2,
        beta_transfer=2.0,
        beta_wait=1.1,
        max_access_deviation_min=20.0,
        max_transfer_wait_min=45.0,
        min_dwell_s=2,
        egress_cost=0.5,
        use_global_egress_links=True,
        mask_strategy="link_mask",
        beta_early=3.0,
        beta_late=4.0,
        theta_default=6.0,
        theta_min=0.2,
        use_capacity_penalty=True,
        capacity_penalty_alpha=0.7,
        capacity_penalty_kappa=1.5,
    )

    cfg.validate()


@pytest.mark.parametrize(
    ("field_name", "bad_value", "expected_message"),
    [
        ("beta_in_vehicle", -1.0, "beta_in_vehicle must be non-negative"),
        ("beta_transfer", -1.0, "beta_transfer must be non-negative"),
        ("beta_wait", -1.0, "beta_wait must be non-negative"),
        ("egress_cost", -1.0, "egress_cost must be non-negative"),
        ("beta_early", -1.0, "beta_early must be non-negative"),
        ("beta_late", -1.0, "beta_late must be non-negative"),
        ("max_access_deviation_min", -1.0, "max_access_deviation_min must be non-negative"),
        ("max_transfer_wait_min", -1.0, "max_transfer_wait_min must be non-negative"),
    ],
)
def test_non_negative_fields_reject_negative_values(
    field_name: str,
    bad_value: float,
    expected_message: str,
):
    cfg = AssignmentConfig(**{field_name: bad_value})

    with pytest.raises(ValueError, match=expected_message):
        cfg.validate()


@pytest.mark.parametrize("value", [0, -1])
def test_min_dwell_s_must_be_strictly_positive(value: int):
    cfg = AssignmentConfig(min_dwell_s=value)

    with pytest.raises(ValueError, match="min_dwell_s must be strictly positive"):
        cfg.validate()


@pytest.mark.parametrize("value", [1, 2, 30])
def test_min_dwell_s_accepts_positive_values(value: int):
    cfg = AssignmentConfig(min_dwell_s=value)

    cfg.validate()


@pytest.mark.parametrize("mask_strategy", ["link_mask", "egress_by_head"])
def test_mask_strategy_accepts_supported_values(mask_strategy: str):
    cfg = AssignmentConfig(mask_strategy=mask_strategy)

    cfg.validate()


@pytest.mark.parametrize("mask_strategy", ["", "node_mask", "invalid", "LINK_MASK"])
def test_mask_strategy_rejects_unsupported_values(mask_strategy: str):
    cfg = AssignmentConfig(mask_strategy=mask_strategy)

    with pytest.raises(ValueError, match="mask_strategy must be one of"):
        cfg.validate()


@pytest.mark.parametrize("theta_default", [0.0, -1.0])
def test_theta_default_must_be_strictly_positive(theta_default: float):
    cfg = AssignmentConfig(theta_default=theta_default)

    with pytest.raises(ValueError, match="theta_default must be strictly positive"):
        cfg.validate()


@pytest.mark.parametrize("theta_default", [0.1, 1.0, 10.0])
def test_theta_default_accepts_positive_values(theta_default: float):
    cfg = AssignmentConfig(theta_default=theta_default)

    cfg.validate()


def test_theta_min_is_present_but_not_validated_by_current_implementation():
    cfg = AssignmentConfig(theta_min=-1.0)

    # This documents the current implementation. If theta_min validation is
    # added later, this test should be updated accordingly.
    cfg.validate()


def test_capacity_penalty_parameters_are_not_restricted_when_penalty_disabled():
    cfg = AssignmentConfig(
        use_capacity_penalty=False,
        capacity_penalty_alpha=-10.0,
        capacity_penalty_kappa=-1.0,
    )

    # The current implementation validates capacity penalty parameters only
    # when use_capacity_penalty is True.
    cfg.validate()


def test_capacity_penalty_alpha_must_be_non_negative_when_enabled():
    cfg = AssignmentConfig(
        use_capacity_penalty=True,
        capacity_penalty_alpha=-0.1,
        capacity_penalty_kappa=1.0,
    )

    with pytest.raises(ValueError, match="capacity_penalty_alpha must be non-negative"):
        cfg.validate()


@pytest.mark.parametrize("kappa", [0.0, -1.0])
def test_capacity_penalty_kappa_must_be_positive_when_enabled(kappa: float):
    cfg = AssignmentConfig(
        use_capacity_penalty=True,
        capacity_penalty_alpha=0.0,
        capacity_penalty_kappa=kappa,
    )

    with pytest.raises(ValueError, match="capacity_penalty_kappa must be > 0"):
        cfg.validate()


@pytest.mark.parametrize("kappa", [0.1, 1.0, 10.0])
def test_capacity_penalty_kappa_accepts_positive_values_when_enabled(kappa: float):
    cfg = AssignmentConfig(
        use_capacity_penalty=True,
        capacity_penalty_alpha=0.0,
        capacity_penalty_kappa=kappa,
    )

    cfg.validate()


def test_use_global_egress_links_boolean_values_are_accepted():
    AssignmentConfig(use_global_egress_links=True).validate()
    AssignmentConfig(use_global_egress_links=False).validate()


def test_boolean_flags_are_not_retyped_by_validate():
    cfg = AssignmentConfig(
        use_global_egress_links=False,
        use_capacity_penalty=False,
    )

    cfg.validate()

    assert cfg.use_global_egress_links is False
    assert cfg.use_capacity_penalty is False


def test_validate_does_not_change_valid_values():
    cfg = AssignmentConfig(
        beta_in_vehicle=1.25,
        beta_transfer=2.5,
        beta_wait=0.75,
        max_access_deviation_min=12.0,
        max_transfer_wait_min=25.0,
        min_dwell_s=3,
        egress_cost=1.0,
        use_global_egress_links=False,
        mask_strategy="egress_by_head",
        beta_early=1.5,
        beta_late=2.5,
        theta_default=4.0,
        theta_min=0.25,
        use_capacity_penalty=True,
        capacity_penalty_alpha=0.4,
        capacity_penalty_kappa=2.0,
    )

    before = {field.name: getattr(cfg, field.name) for field in fields(cfg)}
    cfg.validate()
    after = {field.name: getattr(cfg, field.name) for field in fields(cfg)}

    assert after == before


def test_adding_future_fields_does_not_break_core_field_test():
    # This test intentionally checks only that the current core fields exist.
    # It does not require the field set to be exactly equal to a fixed list,
    # so adding future fields, for instance a Dial algorithm selector or
    # JAX-graph execution mode, will not break the test suite.
    assert "theta_default" in _field_names()
    assert "mask_strategy" in _field_names()