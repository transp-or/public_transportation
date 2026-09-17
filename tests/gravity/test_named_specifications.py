from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from public_transportation.inference.gravity import (
    GravityFeatures,
    GravityModelSpecification,
    GravityParameterLayout,
    generate_gravity_demand,
    gravity_demand_numpy_reference,
    gravity_model_specification_from_mapping,
    gravity_model_specification_from_preset,
)


def _features() -> GravityFeatures:
    origin = np.repeat(np.arange(2), 4)
    destination = np.tile(np.arange(2), 4)
    time = np.tile(np.repeat(np.arange(2), 2), 2)
    return GravityFeatures(
        origin_index=origin,
        canonical_od_index=np.arange(8),
        destination_index=destination,
        departure_time_index=time,
        origin_time_group_index=np.repeat(np.arange(4), 2),
        journey_time=np.asarray((4, 8, 5, 7, 6, 9, 3, 10), dtype=np.float64),
        transfer_count=np.zeros(8, dtype=np.int64),
        structural_feasible=np.ones(8, dtype=bool),
        origin_time_totals=np.asarray((10, 20, 30, 40), dtype=np.float64),
        destination_attractiveness=np.ones(8, dtype=np.float64),
        num_origins=2,
        num_destinations=2,
        num_departure_times=2,
        od_layout_fingerprint="named-specification",
        origin_zone_index=origin,
        destination_zone_index=destination,
        time_period_index=time,
    )


def test_zone_time_preset_is_explicit_and_neutral_exposure_is_conserving():
    features = _features()
    specification = gravity_model_specification_from_preset(
        "zone_time", features=features
    )
    assert specification.production_source == "unit_exposure"
    assert specification.uses_external_od_matrix is False
    assert features.origin_zone_time_index is not None
    assert features.destination_zone_time_index is not None
    layout = GravityParameterLayout(specification)
    raw = np.linspace(-0.2, 0.3, layout.size)
    result = generate_gravity_demand(raw, features=features, parameter_layout=layout)
    reference = gravity_demand_numpy_reference(
        raw, features=features, parameter_layout=layout
    )
    np.testing.assert_allclose(result.demand, reference, rtol=3e-6, atol=3e-6)
    assert np.all(np.asarray(result.origin_time_sums) > 0)
    restored = GravityModelSpecification.from_dict(specification.to_dict())
    assert restored == specification
    assert GravityParameterLayout.from_dict(layout.to_dict()).names == layout.names


def test_unit_exposure_accepts_absent_observed_totals():
    features = replace(_features(), origin_time_totals=None)
    # The constructor restores a neutral one-per-origin/time placeholder.
    assert np.all(features.origin_time_totals == 1)
    specification = gravity_model_specification_from_preset("zone", features=features)
    layout = GravityParameterLayout(specification)
    result = generate_gravity_demand(
        np.zeros(layout.size), features=features, parameter_layout=layout
    )
    np.testing.assert_allclose(
        np.asarray(result.origin_time_sums),
        np.ones(features.num_origin_time_groups),
    )


def test_destination_time_only_and_unrestricted_redundancy_are_rejected():
    from public_transportation.inference.gravity import (
        GravityConstraint,
        GravityEffectScope,
        GravityTermSpecification,
    )

    with pytest.raises(ValueError, match="not valid|cancel"):
        GravityModelSpecification(
            terms=(
                GravityTermSpecification(
                    "destination.time",
                    "destination_attractiveness",
                    GravityEffectScope.TIME_PERIOD,
                    grouping="time_period_index",
                    group_count=2,
                ),
            )
        )

    with pytest.raises(ValueError, match="constraint|unrestricted"):
        GravityModelSpecification(
            terms=(
                GravityTermSpecification(
                    "p.zone",
                    "production",
                    GravityEffectScope.ORIGIN_ZONE,
                    grouping="origin_zone_index",
                    group_count=2,
                    constraint=GravityConstraint.NONE,
                ),
                GravityTermSpecification(
                    "p.time",
                    "production",
                    GravityEffectScope.TIME_PERIOD,
                    grouping="time_period_index",
                    group_count=2,
                ),
            )
        )


def test_mapping_parser_accepts_unit_exposure_and_terms():
    specification = gravity_model_specification_from_mapping(
        {
            "schema_version": 1,
            "production": {
                "baseline": "unit_exposure",
                "terms": [
                    {
                        "name": "production.origin_zone",
                        "scope": "origin_zone",
                        "regularization": {"type": "ridge", "strength": 1.0},
                    }
                ],
            },
            "destination_attractiveness": {
                "terms": [
                    {
                        "name": "destination.zone",
                        "scope": "destination_zone",
                        "regularization": {"type": "ridge", "strength": 1.0},
                    }
                ]
            },
        },
        features=_features(),
    )
    assert specification.production_source == "unit_exposure"
    assert len(specification.terms) == 2


def test_mapping_parser_expands_named_preset_with_neutral_default():
    specification = gravity_model_specification_from_mapping(
        {"schema_version": 1, "preset": "zone_time"}, features=_features()
    )
    assert specification.preset == "zone_time"
    assert specification.production_source == "unit_exposure"
    assert specification.schema_version == 4


def test_mapping_parser_infers_two_way_centered_dimensions():
    specification = gravity_model_specification_from_mapping(
        {
            "schema_version": 1,
            "production": {
                "baseline": "unit_exposure",
                "terms": [
                    {
                        "name": "production.zone_time",
                        "scope": "origin_zone_time",
                        "constraint": "two_way_centered",
                        "regularization": {"type": "ridge", "strength": 1.0},
                    }
                ],
            },
        },
        features=_features(),
    )
    term = specification.production_terms[0]
    assert term.row_group_count == 2
    assert term.column_group_count == 2
    assert term.parameter_count == 1


def test_a_priori_od_matrix_is_rejected_at_feature_boundary():
    with pytest.raises(ValueError, match="a priori OD matrix"):
        replace(_features(), origin_time_totals=np.ones((2, 2), dtype=np.float64))


@pytest.mark.parametrize(
    "source",
    ("od_matrix", "a_priori_od_matrix", "prior_demand", "baseline-od-matrix"),
)
def test_obsolete_od_matrix_production_sources_are_rejected(source):
    with pytest.raises(ValueError, match="a priori OD matrix"):
        GravityModelSpecification(production_source=source)


def test_obsolete_od_matrix_component_source_is_rejected():
    from public_transportation.inference.gravity import (
        GravityComponentSpecification,
        GravityEffectScope,
        GravityParameterization,
    )

    with pytest.raises(ValueError, match="a priori OD matrix"):
        GravityModelSpecification(
            components=(
                GravityComponentSpecification(
                    "production",
                    GravityEffectScope.GLOBAL,
                    GravityParameterization.LOG_MULTIPLIER,
                    source="a_priori_od_matrix",
                ),
            )
        )


@pytest.mark.parametrize("field", ("od_matrix", "a_priori_od_matrix", "prior_demand"))
def test_obsolete_od_matrix_mapping_fields_are_rejected(field):
    with pytest.raises(ValueError, match="a priori OD matrix"):
        gravity_model_specification_from_mapping({field: [[1.0]]})

    with pytest.raises(ValueError, match="a priori OD matrix"):
        gravity_model_specification_from_mapping({"production": {field: [[1.0]]}})


def test_obsolete_od_matrix_fields_are_rejected_in_serialized_contracts():
    specification = GravityModelSpecification().to_dict()
    specification["od_matrix"] = [[1.0]]
    with pytest.raises(ValueError, match="a priori OD matrix"):
        GravityModelSpecification.from_dict(specification)

    feature_payload = _features().to_dict()
    feature_payload["a_priori_od_matrix"] = [[1.0]]
    with pytest.raises(ValueError, match="a priori OD matrix"):
        GravityFeatures.from_dict(feature_payload)
