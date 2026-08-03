"""Genuinely matrix-free fixed-routing measurement products."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np

from public_transportation.measurement.likelihood_jax import (
    predict_measurements_from_link_flow,
)
from public_transportation.measurement.mapping import AggregationSpec

from .assignment_adapter import (
    AssignmentInputs,
    FixedRoutingInputs,
    assign_link_flow_fixed_routing,
    assign_link_flow_fixed_routing_custom_adjoint,
    validate_fixed_routing_compatibility,
)
from .compact_od_assignment_layout import CompactODAssignmentLayout
from .fixed_routing_measurement_operator import (
    assignment_inputs_fingerprint,
    measurement_mapping_fingerprint,
)
from .measurement_operator_protocol import GravityOperatorCapabilities

Array = np.ndarray
JaxProduct = Callable[[jnp.ndarray], jnp.ndarray]
Clock = Callable[[], float]


def _finite_vector(value: object, *, name: str, size: int) -> Array:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values.")
    array = np.asarray(array, dtype=np.result_type(array.dtype, np.float64))
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


@dataclass(frozen=True, slots=True)
class MatrixFreePreparationDiagnostics:
    zero_offset_fast_path: bool
    fixed_positive_cells: int
    validation_seconds: float
    routing_preparation_seconds: float
    offset_compilation_seconds: float
    offset_execution_seconds: float
    total_preparation_seconds: float
    forward_compilation_count: int
    forward_compilation_seconds: float
    transpose_compilation_count: int
    transpose_compilation_seconds: float
    forward_execution_count: int
    forward_execution_seconds: float
    transpose_execution_count: int
    transpose_execution_seconds: float
    deadline_exceeded: bool
    indivisible_operation_overshoot: bool
    forward_tracing_seconds: float
    forward_lowering_seconds: float
    forward_first_execution_seconds: float
    forward_warm_execution_seconds: float
    transpose_tracing_seconds: float
    transpose_lowering_seconds: float
    transpose_first_execution_seconds: float
    transpose_warm_execution_seconds: float
    deadline_phase: str | None
    captured_constant_bytes: int | None
    forward_input_shape: tuple[int, ...]
    transpose_input_shape: tuple[int, ...]
    input_dtype: str
    backend: str
    devices: tuple[str, ...]


class MatrixFreePreparationDeadlineError(RuntimeError):
    """Raised between indivisible phases after a preparation deadline expires."""

    def __init__(
        self, message: str, *, diagnostics: MatrixFreePreparationDiagnostics
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True, slots=True)
class MatrixFreeStorageMetrics:
    """Memory fields consumed by gravity lineage without implying a matrix."""

    stored_bytes: int = 0
    peak_construction_bytes: int = 0


@dataclass(slots=True)
class MatrixFreeFixedRoutingMeasurementOperator:
    """Apply fixed-routing measurement and transpose products without storing A.

    Contract validation is eager. Complete forward and transpose products are
    compiled independently on first use. The fixed offset is exactly zero and
    requires no JAX work when there are no positive fixed-demand cells.
    """

    inputs: AssignmentInputs
    routing: FixedRoutingInputs
    spec: AggregationSpec
    compact_layout: CompactODAssignmentLayout
    preparation_deadline: float | None = None
    clock: Clock = field(default=perf_counter, repr=False, compare=False)
    fixed_measurement_offset: Array = field(init=False, repr=False)
    _forward_function: JaxProduct = field(init=False, repr=False, compare=False)
    _transpose_function: JaxProduct = field(init=False, repr=False, compare=False)
    _forward_product: JaxProduct | None = field(
        init=False, default=None, repr=False, compare=False
    )
    _transpose_product: JaxProduct | None = field(
        init=False, default=None, repr=False, compare=False
    )
    _validation_seconds: float = field(init=False, default=0.0, repr=False)
    _routing_preparation_seconds: float = field(init=False, default=0.0, repr=False)
    _offset_compilation_seconds: float = field(init=False, default=0.0, repr=False)
    _offset_execution_seconds: float = field(init=False, default=0.0, repr=False)
    _forward_compilation_count: int = field(init=False, default=0, repr=False)
    _forward_compilation_seconds: float = field(init=False, default=0.0, repr=False)
    _transpose_compilation_count: int = field(init=False, default=0, repr=False)
    _transpose_compilation_seconds: float = field(init=False, default=0.0, repr=False)
    _forward_execution_count: int = field(init=False, default=0, repr=False)
    _forward_execution_seconds: float = field(init=False, default=0.0, repr=False)
    _transpose_execution_count: int = field(init=False, default=0, repr=False)
    _transpose_execution_seconds: float = field(init=False, default=0.0, repr=False)
    _deadline_exceeded: bool = field(init=False, default=False, repr=False)
    _indivisible_overshoot: bool = field(init=False, default=False, repr=False)
    _zero_offset_fast_path: bool = field(init=False, default=False, repr=False)
    _forward_tracing_seconds: float = field(init=False, default=0.0, repr=False)
    _forward_lowering_seconds: float = field(init=False, default=0.0, repr=False)
    _forward_first_execution_seconds: float = field(
        init=False, default=0.0, repr=False
    )
    _forward_warm_execution_seconds: float = field(
        init=False, default=0.0, repr=False
    )
    _transpose_tracing_seconds: float = field(init=False, default=0.0, repr=False)
    _transpose_lowering_seconds: float = field(init=False, default=0.0, repr=False)
    _transpose_first_execution_seconds: float = field(
        init=False, default=0.0, repr=False
    )
    _transpose_warm_execution_seconds: float = field(
        init=False, default=0.0, repr=False
    )
    _deadline_phase: str | None = field(init=False, default=None, repr=False)

    @property
    def product_capabilities(self) -> GravityOperatorCapabilities:
        return GravityOperatorCapabilities()

    def __post_init__(self) -> None:
        if self.preparation_deadline is not None and not math.isfinite(
            self.preparation_deadline
        ):
            raise ValueError("preparation_deadline must be finite when provided.")
        self._check_deadline("contract validation")
        validation_started = self.clock()
        validate_fixed_routing_compatibility(inputs=self.inputs, routing=self.routing)
        if self.compact_layout.num_active != self.inputs.od_origin_node.shape[0]:
            raise ValueError(
                "compact layout active dimension does not match assignment inputs."
            )
        measurement_indices = np.asarray(self.spec.measurement_index)
        link_indices = np.asarray(self.spec.link_index)
        if measurement_indices.shape != link_indices.shape:
            raise ValueError(
                "measurement and link mapping indices must have equal shapes."
            )
        if measurement_indices.size and (
            np.any(measurement_indices < 0)
            or np.any(measurement_indices >= self.spec.num_measurements)
            or np.any(link_indices < 0)
            or np.any(link_indices >= self.inputs.graph.num_links)
        ):
            raise ValueError("measurement mapping indices are out of bounds.")
        self._validation_seconds = max(0.0, self.clock() - validation_started)
        self._check_deadline("numerical function preparation")

        free_indices = jnp.asarray(
            self.compact_layout.free_compact_indices, dtype=jnp.int32
        )
        measurement_index = jnp.asarray(measurement_indices, dtype=jnp.int32)
        link_index = jnp.asarray(link_indices, dtype=jnp.int32)
        num_active = self.compact_layout.num_active
        num_measurements = self.spec.num_measurements

        def aggregate(active_demand: jnp.ndarray) -> jnp.ndarray:
            link_flow = assign_link_flow_fixed_routing_custom_adjoint(
                inputs=self.inputs,
                routing=self.routing,
                f=active_demand,
            )
            return predict_measurements_from_link_flow(
                link_flow,
                spec_num_measurements=num_measurements,
                spec_measurement_index=measurement_index,
                spec_link_index=link_index,
            )

        def aggregate_forward_mode(active_demand: jnp.ndarray) -> jnp.ndarray:
            link_flow = assign_link_flow_fixed_routing(
                inputs=self.inputs,
                routing=self.routing,
                f=active_demand,
            )
            return predict_measurements_from_link_flow(
                link_flow,
                spec_num_measurements=num_measurements,
                spec_measurement_index=measurement_index,
                spec_link_index=link_index,
            )

        def forward(free_demand: jnp.ndarray) -> jnp.ndarray:
            active = jnp.zeros((num_active,), dtype=self.inputs.base_link_cost.dtype)
            active = active.at[free_indices].set(
                free_demand, indices_are_sorted=True, unique_indices=True
            )
            return aggregate(active)

        def forward_mode(free_demand: jnp.ndarray) -> jnp.ndarray:
            active = jnp.zeros((num_active,), dtype=self.inputs.base_link_cost.dtype)
            active = active.at[free_indices].set(
                free_demand, indices_are_sorted=True, unique_indices=True
            )
            return aggregate_forward_mode(active)

        @jax.custom_jvp
        def transformable_forward(free_demand: jnp.ndarray) -> jnp.ndarray:
            return forward(free_demand)

        @transformable_forward.defjvp
        def transformable_forward_jvp(primals, tangents):
            (free_demand,) = primals
            (demand_tangent,) = tangents
            return forward(free_demand), forward_mode(demand_tangent)

        def transpose(measurement_cotangent: jnp.ndarray) -> jnp.ndarray:
            zero_free = jnp.zeros(
                (self.compact_layout.num_free,),
                dtype=self.inputs.base_link_cost.dtype,
            )
            _, pullback = jax.vjp(forward, zero_free)
            return pullback(measurement_cotangent)[0]

        self._forward_function = transformable_forward
        self._transpose_function = transpose

        fixed_count = len(self.compact_layout.fixed_compact_indices)
        if fixed_count == 0:
            offset = np.zeros((num_measurements,), dtype=self.dtype)
            offset.setflags(write=False)
            self.fixed_measurement_offset = offset
            self._zero_offset_fast_path = True
            return

        self._check_deadline("fixed-offset routing preparation")
        preparation_started = self.clock()
        fixed_indices = jnp.asarray(
            self.compact_layout.fixed_compact_indices, dtype=jnp.int32
        )
        fixed_values = jnp.asarray(
            self.compact_layout.fixed_compact_values,
            dtype=self.inputs.base_link_cost.dtype,
        )
        fixed_active = jnp.zeros((num_active,), dtype=self.inputs.base_link_cost.dtype)
        fixed_active = fixed_active.at[fixed_indices].set(fixed_values)
        self._routing_preparation_seconds = max(
            0.0, self.clock() - preparation_started
        )
        self._check_deadline("fixed-offset compilation")
        compilation_started = self.clock()
        compiled_offset = jax.jit(aggregate).lower(fixed_active).compile()
        self._offset_compilation_seconds = max(
            0.0, self.clock() - compilation_started
        )
        self._check_after_indivisible("fixed-offset compilation")
        self._check_deadline("fixed-offset execution")
        execution_started = self.clock()
        offset_device = compiled_offset(fixed_active)
        offset_device.block_until_ready()
        offset = np.asarray(offset_device)
        self._offset_execution_seconds = max(0.0, self.clock() - execution_started)
        self._check_after_indivisible("fixed-offset execution")
        if not np.all(np.isfinite(offset)) or np.any(offset < 0.0):
            raise ValueError("matrix-free fixed measurement offset is invalid.")
        offset = np.array(offset, copy=True)
        offset.setflags(write=False)
        self.fixed_measurement_offset = offset

    @property
    def shape(self) -> tuple[int, int]:
        return self.spec.num_measurements, self.compact_layout.num_free

    @property
    def num_free_od(self) -> int:
        return self.compact_layout.num_free

    @property
    def num_measurements(self) -> int:
        return self.spec.num_measurements

    @property
    def compact_layout_fingerprint(self) -> str:
        return self.compact_layout.fingerprint

    @property
    def is_matrix_free(self) -> bool:
        return True

    @property
    def assignment_fingerprint(self) -> str:
        return assignment_inputs_fingerprint(self.inputs)

    @property
    def graph_fingerprint(self) -> str:
        return assignment_inputs_fingerprint(self.inputs)

    @property
    def mapping_fingerprint(self) -> str:
        return measurement_mapping_fingerprint(self.spec)

    @property
    def theta(self) -> float:
        return float(np.asarray(self.routing.theta))

    @property
    def representation(self) -> str:
        return "matrix_free"

    @property
    def metrics(self) -> MatrixFreeStorageMetrics:
        return MatrixFreeStorageMetrics()

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.inputs.base_link_cost.dtype)

    @property
    def diagnostics(self) -> MatrixFreePreparationDiagnostics:
        return MatrixFreePreparationDiagnostics(
            zero_offset_fast_path=self._zero_offset_fast_path,
            fixed_positive_cells=len(self.compact_layout.fixed_compact_indices),
            validation_seconds=self._validation_seconds,
            routing_preparation_seconds=self._routing_preparation_seconds,
            offset_compilation_seconds=self._offset_compilation_seconds,
            offset_execution_seconds=self._offset_execution_seconds,
            total_preparation_seconds=(
                self._validation_seconds
                + self._routing_preparation_seconds
                + self._offset_compilation_seconds
                + self._offset_execution_seconds
            ),
            forward_compilation_count=self._forward_compilation_count,
            forward_compilation_seconds=self._forward_compilation_seconds,
            transpose_compilation_count=self._transpose_compilation_count,
            transpose_compilation_seconds=self._transpose_compilation_seconds,
            forward_execution_count=self._forward_execution_count,
            forward_execution_seconds=self._forward_execution_seconds,
            transpose_execution_count=self._transpose_execution_count,
            transpose_execution_seconds=self._transpose_execution_seconds,
            deadline_exceeded=self._deadline_exceeded,
            indivisible_operation_overshoot=self._indivisible_overshoot,
            forward_tracing_seconds=self._forward_tracing_seconds,
            forward_lowering_seconds=self._forward_lowering_seconds,
            forward_first_execution_seconds=self._forward_first_execution_seconds,
            forward_warm_execution_seconds=self._forward_warm_execution_seconds,
            transpose_tracing_seconds=self._transpose_tracing_seconds,
            transpose_lowering_seconds=self._transpose_lowering_seconds,
            transpose_first_execution_seconds=self._transpose_first_execution_seconds,
            transpose_warm_execution_seconds=self._transpose_warm_execution_seconds,
            deadline_phase=self._deadline_phase,
            captured_constant_bytes=None,
            forward_input_shape=(self.num_free_od,),
            transpose_input_shape=(self.num_measurements,),
            input_dtype=str(self.dtype),
            backend=jax.default_backend(),
            devices=tuple(str(device) for device in jax.devices()),
        )

    def _check_deadline(self, phase: str) -> None:
        if self.preparation_deadline is None or self.clock() < self.preparation_deadline:
            return
        self._deadline_exceeded = True
        self._deadline_phase = phase
        raise MatrixFreePreparationDeadlineError(
            f"matrix-free preparation deadline reached before {phase}.",
            diagnostics=self.diagnostics,
        )

    def _check_after_indivisible(self, phase: str) -> None:
        if self.preparation_deadline is None or self.clock() < self.preparation_deadline:
            return
        self._deadline_exceeded = True
        self._indivisible_overshoot = True
        self._deadline_phase = phase
        raise MatrixFreePreparationDeadlineError(
            f"matrix-free preparation deadline exceeded during indivisible {phase}.",
            diagnostics=self.diagnostics,
        )

    def _compile_forward(self, value: jnp.ndarray) -> None:
        self._check_deadline("forward compilation")
        started = self.clock()
        traced = jax.jit(self._forward_function).trace(value)
        self._forward_tracing_seconds += max(0.0, self.clock() - started)
        self._check_after_indivisible("forward tracing")
        started = self.clock()
        lowered = traced.lower()
        self._forward_lowering_seconds += max(0.0, self.clock() - started)
        self._check_after_indivisible("forward lowering")
        started = self.clock()
        self._forward_product = lowered.compile()
        self._forward_compilation_seconds += max(0.0, self.clock() - started)
        self._forward_compilation_count += 1
        self._check_after_indivisible("forward compilation")

    def _compile_transpose(self, value: jnp.ndarray) -> None:
        self._check_deadline("transpose compilation")
        started = self.clock()
        traced = jax.jit(self._transpose_function).trace(value)
        self._transpose_tracing_seconds += max(0.0, self.clock() - started)
        self._check_after_indivisible("transpose tracing")
        started = self.clock()
        lowered = traced.lower()
        self._transpose_lowering_seconds += max(0.0, self.clock() - started)
        self._check_after_indivisible("transpose lowering")
        started = self.clock()
        self._transpose_product = lowered.compile()
        self._transpose_compilation_seconds += max(0.0, self.clock() - started)
        self._transpose_compilation_count += 1
        self._check_after_indivisible("transpose compilation")

    def prepare_device_products(
        self,
        *,
        products: Literal["forward", "forward_and_transpose"] = (
            "forward_and_transpose"
        ),
    ) -> MatrixFreePreparationDiagnostics:
        """Compile and time reusable products under the absolute deadline."""
        if products not in ("forward", "forward_and_transpose"):
            raise ValueError("unsupported matrix-free preparation product set.")
        forward_value = jnp.zeros((self.num_free_od,), dtype=self.dtype)
        if self._forward_product is None:
            self._compile_forward(forward_value)
        assert self._forward_product is not None
        for warm in (False, True):
            phase = "forward warm execution" if warm else "forward first execution"
            self._check_deadline(phase)
            started = self.clock()
            result = self._forward_product(forward_value)
            result.block_until_ready()
            elapsed = max(0.0, self.clock() - started)
            if warm:
                self._forward_warm_execution_seconds = elapsed
            else:
                self._forward_first_execution_seconds = elapsed
            self._forward_execution_seconds += elapsed
            self._forward_execution_count += 1
            self._check_after_indivisible(phase)
        if products == "forward_and_transpose":
            transpose_value = jnp.zeros((self.num_measurements,), dtype=self.dtype)
            if self._transpose_product is None:
                self._compile_transpose(transpose_value)
            assert self._transpose_product is not None
            for warm in (False, True):
                phase = (
                    "transpose warm execution"
                    if warm
                    else "transpose first execution"
                )
                self._check_deadline(phase)
                started = self.clock()
                result = self._transpose_product(transpose_value)
                result.block_until_ready()
                elapsed = max(0.0, self.clock() - started)
                if warm:
                    self._transpose_warm_execution_seconds = elapsed
                else:
                    self._transpose_first_execution_seconds = elapsed
                self._transpose_execution_seconds += elapsed
                self._transpose_execution_count += 1
                self._check_after_indivisible(phase)
        return self.diagnostics

    def matvec(self, vector: object) -> Array:
        value = _finite_vector(vector, name="forward vector", size=self.shape[1])
        device_value = jnp.asarray(value, dtype=self.inputs.base_link_cost.dtype)
        if self._forward_product is None:
            self._compile_forward(device_value)
        self._check_deadline("forward execution")
        started = self.clock()
        assert self._forward_product is not None
        result = self._forward_product(device_value)
        result.block_until_ready()
        self._forward_execution_seconds += max(0.0, self.clock() - started)
        self._forward_execution_count += 1
        self._check_after_indivisible("forward execution")
        return np.asarray(result)

    def jax_matvec(self, vector: jax.Array) -> jax.Array:
        """Traceable device-native forward product with no synchronization."""
        value = jnp.asarray(vector, dtype=self.inputs.base_link_cost.dtype)
        if value.shape != (self.num_free_od,):
            raise ValueError(
                f"forward vector must have shape ({self.num_free_od},), "
                f"got {value.shape}."
            )
        return self._forward_function(value)

    def rmatvec(self, vector: object) -> Array:
        value = _finite_vector(vector, name="transpose vector", size=self.shape[0])
        device_value = jnp.asarray(value, dtype=self.inputs.base_link_cost.dtype)
        if self._transpose_product is None:
            self._compile_transpose(device_value)
        self._check_deadline("transpose execution")
        started = self.clock()
        assert self._transpose_product is not None
        result = self._transpose_product(device_value)
        result.block_until_ready()
        self._transpose_execution_seconds += max(0.0, self.clock() - started)
        self._transpose_execution_count += 1
        self._check_after_indivisible("transpose execution")
        return np.asarray(result)

    def jax_rmatvec(self, vector: jax.Array) -> jax.Array:
        """Traceable device-native transpose product with no synchronization."""
        value = jnp.asarray(vector, dtype=self.inputs.base_link_cost.dtype)
        if value.shape != (self.num_measurements,):
            raise ValueError(
                f"transpose vector must have shape ({self.num_measurements},), "
                f"got {value.shape}."
            )
        return self._transpose_function(value)

    def jax_matmat(self, matrix: jax.Array) -> jax.Array:
        """Apply the forward product to a small parameter-direction batch."""
        value = jnp.asarray(matrix, dtype=self.inputs.base_link_cost.dtype)
        if value.ndim != 2 or value.shape[0] != self.num_free_od:
            raise ValueError(
                "forward matrix must have shape "
                f"({self.num_free_od}, k), got {value.shape}."
            )
        return jax.vmap(self._forward_function, in_axes=1, out_axes=1)(value)
