from __future__ import annotations

import pytest

from public_transportation.assignment.config import AssignmentConfig


# ---------------------------------------------------------
# Happy path
# ---------------------------------------------------------


def test_default_config_is_valid():
    cfg = AssignmentConfig()
    cfg.validate()  # should not raise


def test_custom_valid_config_is_valid():
    cfg = AssignmentConfig(
        beta_in_vehicle=1.0,
        beta_transfer=2.0,
        beta_wait=1.2,
        beta_access=0.8,
        beta_early=3.0,
        beta_late=4.0,
        theta_default=6.0,
        use_capacity_penalty=False,
        capacity_penalty_alpha=0.0,
        capacity_penalty_kappa=1.0,
    )
    cfg.validate()  # should not raise


# ---------------------------------------------------------
# Non-negativity checks
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "beta_in_vehicle",
        "beta_transfer",
        "beta_wait",
        "beta_access",
        "beta_early",
        "beta_late",
    ],
)
def test_time_weights_must_be_non_negative(field: str):
    kwargs = {field: -0.01}
    cfg = AssignmentConfig(**kwargs)
    with pytest.raises(ValueError) as excinfo:
        cfg.validate()
    msg = str(excinfo.value)
    assert field in msg
    assert "non-negative" in msg


# ---------------------------------------------------------
# theta_default
# ---------------------------------------------------------


@pytest.mark.parametrize("theta", [0.0, -1.0, -1e-9])
def test_theta_default_must_be_strictly_positive(theta: float):
    cfg = AssignmentConfig(theta_default=theta)
    with pytest.raises(ValueError) as excinfo:
        cfg.validate()
    assert "theta_default" in str(excinfo.value)
    assert "strictly positive" in str(excinfo.value)


def test_theta_default_positive_is_ok():
    cfg = AssignmentConfig(theta_default=1e-12)
    cfg.validate()  # should not raise


# ---------------------------------------------------------
# Capacity penalty validation
# ---------------------------------------------------------


def test_capacity_penalty_checks_are_skipped_when_disabled():
    # Even if alpha is negative, we do not validate it unless use_capacity_penalty is True
    cfg = AssignmentConfig(use_capacity_penalty=False, capacity_penalty_alpha=-1.0, capacity_penalty_kappa=-2.0)
    cfg.validate()  # should not raise


def test_capacity_penalty_alpha_must_be_non_negative_when_enabled():
    cfg = AssignmentConfig(use_capacity_penalty=True, capacity_penalty_alpha=-0.1)
    with pytest.raises(ValueError) as excinfo:
        cfg.validate()
    assert "capacity_penalty_alpha" in str(excinfo.value)
    assert "non-negative" in str(excinfo.value)


@pytest.mark.parametrize("kappa", [0.0, -1.0, -1e-9])
def test_capacity_penalty_kappa_must_be_positive_when_enabled(kappa: float):
    cfg = AssignmentConfig(use_capacity_penalty=True, capacity_penalty_alpha=0.0, capacity_penalty_kappa=kappa)
    with pytest.raises(ValueError) as excinfo:
        cfg.validate()
    assert "capacity_penalty_kappa" in str(excinfo.value)
    assert "> 0" in str(excinfo.value)