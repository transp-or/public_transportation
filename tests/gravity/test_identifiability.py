from __future__ import annotations

from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from public_transportation.inference.gravity import (
    GravityIdentifiabilityConfig,
    GravityLikelihood,
    GravityODIdentifiability,
    GravityObservationBundle,
    compute_gravity_od_identifiability,
    read_gravity_od_identifiability,
    write_gravity_detailed_report,
    write_gravity_od_identifiability,
)
from public_transportation.inference.gravity.identifiability import (
    _information_hessian,
    _jacobian,
)
from public_transportation.inference.gravity.objective import (
    predict_gravity_measurements,
)
from public_transportation.inference.od_parameter_layout import ODParameterLayout

from tests.gravity.test_phase4_validation import validation_case


class _AuxiliaryChannel:
    name = "synthetic_auxiliary"
    kind = "synthetic"
    fingerprint = "synthetic-auxiliary-fingerprint"
    observed = np.asarray((100.0,))

    def validate(self, *, num_free_od: int, dtype: np.dtype) -> None:
        del num_free_od, dtype

    def predict(self, demand):
        import jax.numpy as jnp

        return jnp.sum(demand)

    def log_likelihood(self, *, prediction, raw_parameters):
        del raw_parameters
        return -((prediction - 100.0) ** 2)

    def report(self):
        return {"name": self.name, "kind": self.kind}


class _CustomVJPMeasurementOperator:
    """Small production-like operator with a VJP but no JVP registration."""

    def __init__(self, base):
        matrix = jnp.asarray(base.matrix)
        self._matrix = matrix

        @jax.custom_vjp
        def matvec(value):
            return matrix @ value

        def matvec_forward(value):
            return matrix @ value, None

        def matvec_reverse(_, cotangent):
            return (matrix.T @ cotangent,)

        matvec.defvjp(matvec_forward, matvec_reverse)
        self._matvec = matvec
        self.fixed_measurement_offset = base.fixed_measurement_offset
        self.num_free_od = base.num_free_od
        self.num_measurements = base.num_measurements
        self.compact_layout_fingerprint = base.compact_layout_fingerprint
        self.assignment_fingerprint = base.assignment_fingerprint
        self.graph_fingerprint = base.graph_fingerprint
        self.mapping_fingerprint = base.mapping_fingerprint
        self.dtype = base.dtype

    def jax_matvec(self, value):
        return self._matvec(value)

    def jax_rmatvec(self, value):
        return self._matrix.T @ value


@pytest.fixture(scope="module")
def fitted_case():
    with jax.enable_x64():
        return validation_case()


def _full_layout(
    num_cells: int = 6, *, fixed: tuple[int, ...] = ()
) -> ODParameterLayout:
    fixed_set = set(fixed)
    free = tuple(index for index in range(num_cells) if index not in fixed_set)
    return ODParameterLayout(
        num_od_total=num_cells,
        od_keys=tuple(
            (f"o{index // 3}", f"d{index % 3}", "am") for index in range(num_cells)
        ),
        free_od_indices=free,
        fixed_od_indices=tuple(fixed),
        fixed_od_values=tuple(0.0 for _ in fixed),
        free_baseline_values=tuple(1.0 for _ in free),
        fixed_zero_indices=tuple(fixed),
        fixed_positive_indices=(),
    )


def test_poisson_count_information_is_analytic_and_finite(fitted_case):
    problem, _, result = fitted_case
    with jax.enable_x64():
        diagnostic = compute_gravity_od_identifiability(
            problem=replace(problem, likelihood=GravityLikelihood.POISSON),
            result=result,
            config=GravityIdentifiabilityConfig(
                measurement_chunk_size=1,
                od_chunk_size=1,
            ),
        )
    np.testing.assert_allclose(diagnostic.count_information_share, 1.0)
    np.testing.assert_allclose(diagnostic.regularization_information_share, 0.0)
    assert diagnostic.effective_hessian_rank > 0
    assert np.all(np.isfinite(diagnostic.local_information_variance))


def test_negative_binomial_weights_and_chunking_are_stable(fitted_case):
    problem, _, result = fitted_case
    with jax.enable_x64():
        one = compute_gravity_od_identifiability(
            problem=problem,
            result=result,
            config=GravityIdentifiabilityConfig(
                measurement_chunk_size=1,
                od_chunk_size=1,
            ),
        )
        many = compute_gravity_od_identifiability(
            problem=problem,
            result=result,
            config=GravityIdentifiabilityConfig(),
        )
    np.testing.assert_allclose(
        one.count_information_share, many.count_information_share
    )
    np.testing.assert_allclose(
        one.local_information_variance, many.local_information_variance
    )
    assert np.all(
        (one.count_information_share >= 0) & (one.count_information_share <= 1)
    )


def test_negative_binomial_information_uses_expected_weights(fitted_case):
    problem, _, result = fitted_case
    with jax.enable_x64():
        h_counts, h_regularization, _ = _information_hessian(
            raw=np.asarray(result.raw_parameters, dtype=np.float64),
            problem=problem,
            config=GravityIdentifiabilityConfig(measurement_chunk_size=64),
        )
        means, _ = predict_gravity_measurements(result.raw_parameters, problem=problem)
        means = np.asarray(means, dtype=np.float64)
        dispersion = float(
            problem.parameter_layout.transform(result.raw_parameters).dispersion
        )
        jacobian = np.asarray(
            jax.jacfwd(
                lambda value: predict_gravity_measurements(value, problem=problem)[0]
            )(result.raw_parameters),
            dtype=np.float64,
        )
    weights = dispersion / (means * (dispersion + means))
    expected = jacobian.T @ (weights[:, None] * jacobian)
    np.testing.assert_allclose(h_counts, expected)
    assert h_regularization.shape == expected.shape


def test_fixed_cells_are_null_and_policy_classified(fitted_case):
    problem, _, result = fitted_case
    expanded = replace(result, full_od_demand=np.r_[result.full_od_demand, 0.0, 0.0])
    with jax.enable_x64():
        diagnostic = compute_gravity_od_identifiability(
            problem=problem, result=expanded
        )
    assert diagnostic.fixed_cells == 2
    assert diagnostic.classification[-1] == "fixed_by_policy"
    assert np.isnan(diagnostic.count_information_share[-1])
    assert np.isnan(diagnostic.local_information_variance[-1])


def test_auxiliary_information_is_kept_separate(fitted_case):
    problem, _, result = fitted_case
    problem = replace(
        problem,
        auxiliary_observations=GravityObservationBundle((_AuxiliaryChannel(),)),
    )
    with jax.enable_x64():
        diagnostic = compute_gravity_od_identifiability(problem=problem, result=result)
    assert diagnostic.auxiliary_information_share is not None
    assert np.all(np.isfinite(diagnostic.auxiliary_information_share))
    assert diagnostic.to_dict()["auxiliary_share_quantiles"] is not None


def test_singular_threshold_marks_cells_not_locally_identifiable(fitted_case):
    problem, _, result = fitted_case
    with jax.enable_x64():
        diagnostic = compute_gravity_od_identifiability(
            problem=problem,
            result=result,
            config=GravityIdentifiabilityConfig(eigenvalue_absolute_tolerance=1.0e12),
        )
    assert diagnostic.effective_hessian_rank == 0
    assert set(diagnostic.classification) == {"not_locally_identifiable"}
    assert np.all(np.isnan(diagnostic.local_information_variance))


@pytest.mark.parametrize(
    "field,value",
    [
        ("measurement_chunk_size", 0),
        ("od_chunk_size", 0),
        ("eigenvalue_relative_tolerance", -1.0),
        ("eigenvalue_absolute_tolerance", float("nan")),
        ("count_dominated_threshold", 1.1),
    ],
)
def test_identifiability_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        GravityIdentifiabilityConfig(**{field: value})


def test_identifiability_defaults_to_reverse_jacobians():
    assert GravityIdentifiabilityConfig().jacobian_mode == "reverse"


def test_identifiability_rejects_unknown_jacobian_mode():
    with pytest.raises(ValueError, match="jacobian_mode"):
        GravityIdentifiabilityConfig(jacobian_mode="diagonal")


def test_reverse_and_forward_jacobians_agree_for_plain_function():
    value = jnp.asarray((0.2, -0.4, 0.7))

    def function(item):
        return jnp.asarray((item[0] ** 2 + item[1], item[1] * item[2]))

    reverse = np.asarray(_jacobian(function, value, mode="reverse"))
    forward = np.asarray(_jacobian(function, value, mode="forward"))
    np.testing.assert_allclose(reverse, forward)


def test_reverse_and_forward_information_hessian_agree(fitted_case):
    problem, _, result = fitted_case
    with jax.enable_x64():
        reverse = _information_hessian(
            raw=np.asarray(result.raw_parameters, dtype=np.float64),
            problem=problem,
            config=GravityIdentifiabilityConfig(
                measurement_chunk_size=2,
                jacobian_mode="reverse",
            ),
        )
        forward = _information_hessian(
            raw=np.asarray(result.raw_parameters, dtype=np.float64),
            problem=problem,
            config=GravityIdentifiabilityConfig(
                measurement_chunk_size=2,
                jacobian_mode="forward",
            ),
        )
    for reverse_matrix, forward_matrix in zip(reverse, forward, strict=True):
        if reverse_matrix is None:
            assert forward_matrix is None
        else:
            np.testing.assert_allclose(reverse_matrix, forward_matrix)


def test_custom_vjp_operator_uses_reverse_mode_by_default(fitted_case):
    problem, _, result = fitted_case
    custom_problem = replace(
        problem, operator=_CustomVJPMeasurementOperator(problem.operator)
    )
    with jax.enable_x64():
        diagnostic = compute_gravity_od_identifiability(
            problem=custom_problem,
            result=result,
            config=GravityIdentifiabilityConfig(
                measurement_chunk_size=2,
                od_chunk_size=2,
            ),
        )
    assert diagnostic.config.jacobian_mode == "reverse"
    assert np.all(np.isfinite(diagnostic.local_information_variance))
    with jax.enable_x64(), pytest.raises(ValueError, match="jacobian_mode='reverse'"):
        compute_gravity_od_identifiability(
            problem=custom_problem,
            result=result,
            config=GravityIdentifiabilityConfig(jacobian_mode="forward"),
        )


def test_persistence_roundtrip_and_digests(fitted_case, tmp_path):
    problem, _, result = fitted_case
    with jax.enable_x64():
        diagnostic = compute_gravity_od_identifiability(problem=problem, result=result)
    path = write_gravity_od_identifiability(diagnostic, tmp_path / "identifiability")
    restored = read_gravity_od_identifiability(path)
    assert isinstance(restored, GravityODIdentifiability)
    np.testing.assert_array_equal(restored.classification, diagnostic.classification)
    np.testing.assert_allclose(
        restored.local_information_variance, diagnostic.local_information_variance
    )
    metadata = json.loads(path.read_text())
    assert metadata["schema_version"] == 1
    assert metadata["artifact_type"] == "gravity_od_identifiability"


def test_persistence_retains_jacobian_mode_and_reads_old_metadata(
    fitted_case, tmp_path
):
    problem, _, result = fitted_case
    with jax.enable_x64():
        diagnostic = compute_gravity_od_identifiability(
            problem=problem,
            result=result,
            config=GravityIdentifiabilityConfig(jacobian_mode="forward"),
        )
    path = write_gravity_od_identifiability(diagnostic, tmp_path / "selected-mode")
    metadata_path = path
    metadata = json.loads(metadata_path.read_text())
    assert metadata["config"]["jacobian_mode"] == "forward"
    restored = read_gravity_od_identifiability(metadata_path)
    assert restored.config.jacobian_mode == "forward"

    metadata["config"].pop("jacobian_mode")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    restored_old = read_gravity_od_identifiability(metadata_path)
    assert restored_old.config.jacobian_mode == "reverse"


def test_report_includes_identifiability_columns_and_summary(fitted_case, tmp_path):
    problem, _, result = fitted_case
    od_layout = _full_layout()
    with jax.enable_x64():
        diagnostic = compute_gravity_od_identifiability(problem=problem, result=result)
    report = write_gravity_detailed_report(
        result=result,
        observations=problem.observations,
        predicted_measurements=result.predicted_measurements,
        od_layout=od_layout,
        likelihood=problem.likelihood,
        identifiability=diagnostic,
        output_directory=tmp_path / "report",
    )
    header = (report.files["full_od.csv"]).read_text().splitlines()[0]
    assert "count_information_share" in header
    summary = json.loads(report.files["report.json"].read_text())
    assert summary["identifiability"]["available"] is True


def test_report_requires_diagnostic_and_rejects_provenance_before_output(
    fitted_case, tmp_path
):
    problem, _, result = fitted_case
    od_layout = _full_layout()
    with pytest.raises(ValueError, match="required"):
        write_gravity_detailed_report(
            result=result,
            observations=problem.observations,
            predicted_measurements=result.predicted_measurements,
            od_layout=od_layout,
            likelihood=problem.likelihood,
            require_identifiability=True,
            output_directory=tmp_path / "required",
        )
    with jax.enable_x64():
        diagnostic = compute_gravity_od_identifiability(problem=problem, result=result)
    bad = replace(
        diagnostic,
        provenance={**diagnostic.provenance, "model_fingerprint": "wrong"},
    )
    destination = tmp_path / "mismatch"
    with pytest.raises(ValueError, match="model_fingerprint"):
        write_gravity_detailed_report(
            result=result,
            observations=problem.observations,
            predicted_measurements=result.predicted_measurements,
            od_layout=od_layout,
            likelihood=problem.likelihood,
            identifiability=bad,
            output_directory=destination,
        )
    assert not destination.exists()


def test_diagnostic_does_not_modify_fitted_result(fitted_case):
    problem, _, result = fitted_case
    before = {
        "objective": result.objective,
        "gradient": np.array(result.gradient, copy=True),
        "raw": np.array(result.raw_parameters, copy=True),
        "predicted": np.array(result.predicted_measurements, copy=True),
    }
    with jax.enable_x64():
        compute_gravity_od_identifiability(problem=problem, result=result)
    assert result.objective == before["objective"]
    np.testing.assert_array_equal(result.gradient, before["gradient"])
    np.testing.assert_array_equal(result.raw_parameters, before["raw"])
    np.testing.assert_array_equal(result.predicted_measurements, before["predicted"])
