from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "docs/source/examples"

PUBLIC_EXAMPLE_APPLICABILITY = {
    "bayesian_estimation": "generic_dense_gaussian_map",
    "network_model": "loading_partitioning_and_missing_measurement_rejection",
    "simple_example_01": "complete_fixed_routing_block_map",
    "simple_example_02": "complete_regularized_fixed_routing_block_map",
    "geneva_gtfs": "structural_preflight_bounded_anytime_and_slow_sweep",
}


def test_every_public_example_has_explicit_block_coordinate_applicability():
    discovered = {
        path.name
        for path in EXAMPLES.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    assert discovered == set(PUBLIC_EXAMPLE_APPLICABILITY)
    assert all(PUBLIC_EXAMPLE_APPLICABILITY.values())
