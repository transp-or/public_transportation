from __future__ import annotations

import numpy as np
import pytest

from public_transportation.inference.gravity import (
    GravityAggregateBin,
    GravityAggregateHistogram,
    GravityAggregateObservation,
    GravityAggregateStratum,
    GravityAggregateUncertainty,
    GravityAttributeResponseOperator,
    GravityAttributeResponseProvenance,
    GravityAttributeSupportError,
    GravityRouteShare,
    validate_aggregate_support,
)


def provenance() -> GravityAttributeResponseProvenance:
    return GravityAttributeResponseProvenance(
        od_layout_fingerprint="od-layout",
        assignment_fingerprint="assignment",
        graph_fingerprint="graph",
        timetable_fingerprint="timetable",
        feasibility_fingerprint="feasibility",
    )


def operator(*, representation="dense") -> GravityAttributeResponseOperator:
    categories = (
        ("morning", "short"),
        ("morning", "long"),
        ("evening", "short"),
        ("evening", "long"),
    )
    # Three free cells.  The first cell has two routes, split 70/30 between
    # short and long, which must not be replaced by a mean journey-time value.
    route_shares = (
        (
            GravityRouteShare("morning", "short", 0.7),
            GravityRouteShare("morning", "long", 0.3),
        ),
        (
            GravityRouteShare("morning", "short", 0.1),
            GravityRouteShare("morning", "long", 0.9),
        ),
        (
            GravityRouteShare("evening", "short", 0.4),
            GravityRouteShare("evening", "long", 0.6),
        ),
    )
    return GravityAttributeResponseOperator.from_route_shares(
        attribute="travel_time",
        unit="seconds",
        category_labels=categories,
        route_shares=route_shares,
        fixed_attribute_offset=np.asarray((1.0, 2.0, 0.0, 0.0)),
        provenance=provenance(),
        representation=representation,
        shard_size=2,
    )


def histogram(*, unsupported=False) -> GravityAggregateHistogram:
    bins = (
        GravityAggregateBin("short", 0.0, 600.0),
        GravityAggregateBin("long", 600.0, 7200.0),
    )
    if unsupported:
        bins = bins + (GravityAggregateBin("unserved", 7200.0, 7201.0),)
    return GravityAggregateHistogram(
        attribute="travel_time",
        unit="seconds",
        support=(0.0, 7201.0) if unsupported else (0.0, 7200.0),
        bins=bins,
        strata=(
            GravityAggregateStratum(
                "morning",
                (7, 3, 1) if unsupported else (7, 3),
                11 if unsupported else 10,
            ),
            GravityAggregateStratum(
                "evening", (4, 6, 0) if unsupported else (4, 6), 10
            ),
        ),
    )


def test_route_shares_preserve_multi_route_distribution_and_fixed_offset():
    item = operator()
    demand = np.asarray((10.0, 20.0, 30.0))

    expected = np.asarray((10.0, 23.0, 12.0, 18.0))
    np.testing.assert_allclose(item.jax_matvec(demand), expected - (1.0, 2.0, 0.0, 0.0))
    np.testing.assert_allclose(
        np.asarray(item.jax_matvec(demand)) + item.fixed_attribute_offset,
        expected,
    )
    np.testing.assert_allclose(item.route_share_mass, np.ones(3))
    assert item.supported_category_labels == item.category_labels


def test_forward_and_adjoint_satisfy_dot_product_identity_for_dense_and_shards():
    vector = np.asarray((0.2, 1.5, 3.0))
    residual = np.asarray((2.0, -1.0, 0.5, 4.0))
    for item in (operator(), operator(representation="sharded")):
        np.testing.assert_allclose(
            np.vdot(np.asarray(item.jax_matvec(vector)), residual),
            np.vdot(vector, np.asarray(item.jax_rmatvec(residual))),
        )


def test_sharded_artifact_round_trip_keeps_identity_and_products(tmp_path):
    item = operator(representation="sharded")
    path = item.save(tmp_path / "attribute-response.npz")
    loaded = GravityAttributeResponseOperator.load(path)

    assert loaded.fingerprint == item.fingerprint
    assert loaded.to_dict() == item.to_dict()
    np.testing.assert_allclose(
        loaded.jax_matvec(np.ones(3)), item.jax_matvec(np.ones(3))
    )


def test_provenance_and_response_identity_are_separate_from_count_operator():
    item = operator()
    altered = GravityAttributeResponseOperator.from_route_shares(
        attribute="travel_time",
        unit="seconds",
        category_labels=item.category_labels,
        route_shares=(
            (
                GravityRouteShare("morning", "short", 0.5),
                GravityRouteShare("morning", "long", 0.5),
            ),
            (
                GravityRouteShare("morning", "short", 0.1),
                GravityRouteShare("morning", "long", 0.9),
            ),
            (
                GravityRouteShare("evening", "short", 0.4),
                GravityRouteShare("evening", "long", 0.6),
            ),
        ),
        fixed_attribute_offset=item.fixed_attribute_offset,
        provenance=item.provenance,
    )
    assert altered.fingerprint != item.fingerprint
    assert item.provenance.od_layout_fingerprint == "od-layout"


def test_invalid_route_shares_and_mass_are_rejected():
    with pytest.raises(ValueError, match="exceed unit mass"):
        GravityAttributeResponseOperator.from_route_shares(
            category_labels=(("all", "short"),),
            route_shares=((GravityRouteShare("all", "short", 1.1),),),
            fixed_attribute_offset=(0.0,),
            provenance=provenance(),
        )
    with pytest.raises(ValueError, match="undeclared category"):
        GravityAttributeResponseOperator.from_route_shares(
            category_labels=(("all", "short"),),
            route_shares=((GravityRouteShare("all", "long", 1.0),),),
            fixed_attribute_offset=(0.0,),
            provenance=provenance(),
        )


def test_positive_observation_in_unsupported_bin_fails_with_audit():
    item = operator()
    observed = histogram(unsupported=True)

    with pytest.raises(GravityAttributeSupportError) as error:
        validate_aggregate_support(item, observed)
    report = error.value.report
    assert report["status"] == "rejected"
    assert report["unsupported_positive_mass"] == 1
    assert report["unsupported_positive_bins"] == [
        {
            "stratum": "morning",
            "bin_label": "unserved",
            "observed_mass": 1,
            "cause": "attribute bin absent from route response",
        }
    ]

    # A zero-count undeclared bin is not a positive unsupported observation.
    zero_observation = histogram(unsupported=True)
    zero_observation = GravityAggregateHistogram(
        attribute=zero_observation.attribute,
        unit=zero_observation.unit,
        support=zero_observation.support,
        bins=zero_observation.bins,
        strata=(
            GravityAggregateStratum("morning", (7, 3, 0), 10),
            GravityAggregateStratum("evening", (4, 6, 0), 10),
        ),
    )
    supported = validate_aggregate_support(item, zero_observation)
    assert supported["status"] == "supported"


def test_fixed_positive_attribute_offset_counts_as_model_support():
    item = GravityAttributeResponseOperator(
        attribute="travel_time",
        unit="seconds",
        category_labels=(("all", "short"),),
        num_free_od=1,
        matrix=np.zeros((1, 1)),
        fixed_attribute_offset=np.asarray((4.0,)),
        provenance=provenance(),
    )
    observed = GravityAggregateHistogram(
        attribute="travel_time",
        unit="seconds",
        support=(0.0, 600.0),
        bins=(GravityAggregateBin("short", 0.0, 600.0),),
        strata=(GravityAggregateStratum("all", (2,), 2),),
    )
    report = validate_aggregate_support(item, observed)
    assert report["status"] == "supported"


def test_positive_observation_in_unsupported_stratum_fails_before_estimation():
    item = operator()
    observed = GravityAggregateHistogram(
        attribute="travel_time",
        unit="seconds",
        support=(0.0, 7200.0),
        bins=(
            GravityAggregateBin("short", 0.0, 600.0),
            GravityAggregateBin("long", 600.0, 7200.0),
        ),
        strata=(GravityAggregateStratum("night", (3, 2), 5),),
    )
    with pytest.raises(GravityAttributeSupportError) as error:
        validate_aggregate_support(item, observed)
    report = error.value.report
    assert report["unsupported_positive_mass"] == 5
    assert report["unsupported_positive_bins"] == [
        {
            "stratum": "night",
            "bin_label": "short",
            "observed_mass": 3,
            "cause": "stratum absent from route response",
        },
        {
            "stratum": "night",
            "bin_label": "long",
            "observed_mass": 2,
            "cause": "stratum absent from route response",
        },
    ]


def test_support_audit_records_aggregate_identity():
    item = operator()
    aggregate = GravityAggregateObservation(
        schema_version=1,
        channel_name="gps_trip_attributes",
        kind="trip_attribute_distribution",
        histograms=(histogram(),),
        metadata={
            "collection_period": "fixture",
            "valid_journeys": 20,
            "excluded_journeys": 0,
            "cleaning_reasons": {},
            "apc_overlap_policy": "recorded",
        },
        uncertainty=GravityAggregateUncertainty("multinomial"),
        source_path="fixture.json",
        file_sha256="file",
        content_sha256="content",
    )
    report = validate_aggregate_support(
        item, aggregate.histograms[0], aggregate=aggregate
    )
    assert report["aggregate_fingerprint"] == aggregate.fingerprint
    assert report["unsupported_positive_mass"] == 0
