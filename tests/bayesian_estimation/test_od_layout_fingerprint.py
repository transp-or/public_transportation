from __future__ import annotations

from dataclasses import replace

import pytest

from public_transportation.inference.od_parameter_layout import (
    ODParameterLayout,
    assert_od_layout_fingerprint_matches,
)


def _layout() -> ODParameterLayout:
    return ODParameterLayout(
        num_od_total=3,
        od_keys=(("a", "b", "t"), ("b", "c", "t"), ("c", "d", "t")),
        free_od_indices=(1,),
        fixed_od_indices=(0, 2),
        fixed_od_values=(0.0, 7.5),
        free_baseline_values=(20.0,),
        fixed_zero_indices=(0,),
        fixed_positive_indices=(2,),
    )


def test_layout_fingerprint_is_stable_and_sha256_sized():
    first = _layout().fingerprint
    second = _layout().fingerprint
    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


def test_layout_fingerprint_changes_with_fixed_value_or_free_baseline():
    layout = _layout()
    changed_fixed = replace(
        layout,
        fixed_od_values=(0.0, 8.0),
    )
    changed_baseline = replace(layout, free_baseline_values=(21.0,))
    assert changed_fixed.fingerprint != layout.fingerprint
    assert changed_baseline.fingerprint != layout.fingerprint


def test_layout_fingerprint_changes_with_partition():
    layout = _layout()
    repartitioned = ODParameterLayout(
        num_od_total=3,
        od_keys=layout.od_keys,
        free_od_indices=(0, 1),
        fixed_od_indices=(2,),
        fixed_od_values=(7.5,),
        free_baseline_values=(10.0, 20.0),
        fixed_zero_indices=(),
        fixed_positive_indices=(2,),
    )
    assert repartitioned.fingerprint != layout.fingerprint


def test_layout_fingerprint_mismatch_has_actionable_diagnostic():
    with pytest.raises(ValueError, match="fixed-demand file"):
        assert_od_layout_fingerprint_matches(
            expected="expected",
            got="actual",
            context="result.npz",
        )
