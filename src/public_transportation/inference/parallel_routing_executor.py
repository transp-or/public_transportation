"""Persistent dynamically scheduled executor for fixed-shape routing batches."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import os
from threading import Lock, get_ident
from time import perf_counter
from typing import Callable, Literal, Sequence

import numpy as np
import jax

from .parallel_partial_execution import (
    FixedBudgetRoutingSelection,
    RoutingMicroshardPlan,
    RoutingWorkUnit,
)


ParallelOperation = Literal["matvec", "rmatvec"]


class ParallelRoutingExecutionInterrupted(RuntimeError):
    """Raised at a batch boundary when cancellation or a deadline prevents work."""


@dataclass(frozen=True, slots=True)
class ParallelRoutingExecutorConfig:
    worker_count: int = 8
    threads_per_worker: int = 1
    supported_group_batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16)
    maximum_retained_batch_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.worker_count <= 0 or self.threads_per_worker <= 0:
            raise ValueError("worker_count and threads_per_worker must be positive.")
        sizes = self.supported_group_batch_sizes
        if not sizes or any(value <= 0 for value in sizes):
            raise ValueError("supported group batch sizes must be positive.")
        if tuple(sorted(set(sizes))) != sizes:
            raise ValueError(
                "supported group batch sizes must be unique and increasing."
            )
        if self.maximum_retained_batch_bytes < 0:
            raise ValueError("maximum_retained_batch_bytes must be nonnegative.")


@dataclass(frozen=True, slots=True)
class RoutingExecutionBatch:
    batch_id: str
    work_ids: tuple[str, ...]
    destination_group_indices: tuple[int, ...]
    predicted_cost: float
    padded_groups: int


@dataclass(frozen=True, slots=True)
class RoutingBatchExecutionObservation:
    batch_id: str
    operation: ParallelOperation
    worker_thread_id: int
    queue_wait_seconds: float
    execution_seconds: float
    predicted_cost: float
    destination_groups: int
    padded_groups: int
    prepared_cache_hit: bool = False


@dataclass(frozen=True, slots=True)
class ParallelRoutingExecutionResult:
    value: np.ndarray
    operation: ParallelOperation
    observations: tuple[RoutingBatchExecutionObservation, ...]
    wall_seconds: float
    worker_count: int
    threads_per_worker: int
    selected_work_ids: tuple[str, ...]
    dispatch_order: tuple[str, ...]
    retained_batch_count: int = 0

    @property
    def worker_thread_ids(self) -> tuple[int, ...]:
        return tuple(sorted({item.worker_thread_id for item in self.observations}))

    @property
    def total_queue_wait_seconds(self) -> float:
        return sum(item.queue_wait_seconds for item in self.observations)

    @property
    def total_worker_execution_seconds(self) -> float:
        return sum(item.execution_seconds for item in self.observations)


@dataclass(slots=True)
class _RetainedEvaluation:
    selected_work_ids: tuple[str, ...]
    expansion_weights: dict[str, float]
    prepared_by_batch: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]


def plan_fixed_shape_routing_batches(
    microshard_plan: RoutingMicroshardPlan,
    *,
    selected_work_ids: Sequence[str] | None = None,
    supported_group_batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16),
    minimum_batches: int = 1,
) -> tuple[RoutingExecutionBatch, ...]:
    """Pack selected microshards deterministically into supported group shapes."""
    if (
        not supported_group_batch_sizes
        or tuple(sorted(set(supported_group_batch_sizes)))
        != supported_group_batch_sizes
    ):
        raise ValueError(
            "supported_group_batch_sizes must be unique, positive, and increasing."
        )
    if any(value <= 0 for value in supported_group_batch_sizes):
        raise ValueError(
            "supported_group_batch_sizes must be unique, positive, and increasing."
        )
    if minimum_batches <= 0:
        raise ValueError("minimum_batches must be positive.")
    by_id = {item.work_id: item for item in microshard_plan.microshards}
    selected = (
        tuple(by_id)
        if selected_work_ids is None
        else tuple(str(item) for item in selected_work_ids)
    )
    if not selected:
        raise ValueError("selected_work_ids must not be empty.")
    if len(selected) != len(set(selected)):
        raise ValueError("selected_work_ids must be unique.")
    unknown = sorted(set(selected) - set(by_id))
    if unknown:
        raise ValueError(f"unknown selected routing work: {unknown}.")
    ordered = [by_id[item] for item in sorted(selected)]
    total_groups = sum(len(item.destination_group_indices) for item in ordered)
    desired_capacity = max(1, int(np.ceil(total_groups / minimum_batches)))
    maximum = next(
        (
            size
            for size in supported_group_batch_sizes
            if size >= desired_capacity
        ),
        supported_group_batch_sizes[-1],
    )
    largest_unit = max(len(item.destination_group_indices) for item in ordered)
    if largest_unit > supported_group_batch_sizes[-1]:
        raise ValueError(
            "one microshard exceeds the largest supported destination-group batch."
        )
    minimum_unit_capacity = next(
        size for size in supported_group_batch_sizes if size >= largest_unit
    )
    maximum = max(maximum, minimum_unit_capacity)

    packed: list[list[RoutingWorkUnit]] = []
    current: list[RoutingWorkUnit] = []
    current_groups = 0
    for item in ordered:
        groups = len(item.destination_group_indices)
        if current and current_groups + groups > maximum:
            packed.append(current)
            current = []
            current_groups = 0
        current.append(item)
        current_groups += groups
    if current:
        packed.append(current)

    result = []
    for index, members in enumerate(packed):
        groups = tuple(
            group for item in members for group in item.destination_group_indices
        )
        if len(groups) != len(set(groups)):
            raise ValueError("selected microshards overlap destination groups.")
        padded = next(size for size in supported_group_batch_sizes if size >= len(groups))
        result.append(
            RoutingExecutionBatch(
                batch_id=f"routing-batch-{index:05d}",
                work_ids=tuple(item.work_id for item in members),
                destination_group_indices=groups,
                predicted_cost=sum(item.predicted_cost for item in members),
                padded_groups=padded,
            )
        )
    return tuple(result)


class PersistentParallelRoutingExecutor:
    """Reuse worker threads and dispatch expensive batches first dynamically."""

    def __init__(
        self,
        *,
        operator,
        microshard_plan: RoutingMicroshardPlan,
        config: ParallelRoutingExecutorConfig = ParallelRoutingExecutorConfig(),
    ) -> None:
        if microshard_plan.problem_fingerprint != operator.assignment_fingerprint:
            raise ValueError("microshard plan and routing operator identities differ.")
        self.operator = operator
        self.microshard_plan = microshard_plan
        self.config = config
        self._pool = ThreadPoolExecutor(
            max_workers=min(config.worker_count, os.cpu_count() or 1),
            thread_name_prefix="partial-routing",
        )
        self._closed = False
        self._execution_lock = Lock()
        self._retained_evaluations: dict[str, _RetainedEvaluation] = {}

    @property
    def effective_worker_count(self) -> int:
        return min(self.config.worker_count, os.cpu_count() or 1)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if not self._closed:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._retained_evaluations.clear()
            self._closed = True

    def __enter__(self) -> PersistentParallelRoutingExecutor:
        if self._closed:
            raise RuntimeError("a closed routing executor cannot be re-entered.")
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def _run_batch(
        self,
        operation: ParallelOperation,
        vector: object,
        batch: RoutingExecutionBatch,
        submitted: float,
        expansion_weights: dict[str, float],
        prepared_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        | None,
        retain_prepared: bool,
    ) -> tuple[
        np.ndarray,
        RoutingBatchExecutionObservation,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    ]:
        started = perf_counter()
        function = (
            self.operator.partial_matvec
            if operation == "matvec"
            else self.operator.partial_rmatvec
        )
        by_id = {item.work_id: item for item in self.microshard_plan.microshards}
        group_weights = tuple(
            expansion_weights.get(work_id, 1.0)
            for work_id in batch.work_ids
            for _ in by_id[work_id].destination_group_indices
        )
        cache_hit = prepared_arrays is not None
        if prepared_arrays is None and retain_prepared:
            prepared_arrays = self.operator.prepare_partial_batch(
                destination_group_indices=batch.destination_group_indices,
                padded_groups=batch.padded_groups,
                group_weights=group_weights,
            )
        value = function(
            vector,
            destination_group_indices=batch.destination_group_indices,
            padded_groups=batch.padded_groups,
            group_weights=group_weights,
            prepared_arrays=prepared_arrays,
        )
        finished = perf_counter()
        return (
            value,
            RoutingBatchExecutionObservation(
                batch_id=batch.batch_id,
                operation=operation,
                worker_thread_id=get_ident(),
                queue_wait_seconds=max(0.0, started - submitted),
                execution_seconds=max(0.0, finished - started),
                predicted_cost=batch.predicted_cost,
                destination_groups=len(batch.destination_group_indices),
                padded_groups=batch.padded_groups,
                prepared_cache_hit=cache_hit,
            ),
            prepared_arrays if retain_prepared else None,
        )

    def _execute_internal(
        self,
        operation: ParallelOperation,
        vector: object,
        *,
        selected_work_ids: Sequence[str] | None = None,
        expansion_weights: dict[str, float] | None = None,
        prepared_by_batch: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ]
        | None = None,
        retain_prepared: bool = False,
        cancellation_requested: Callable[[], bool] | None = None,
        absolute_deadline: float | None = None,
    ) -> tuple[
        ParallelRoutingExecutionResult,
        dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ]:
        if self._closed:
            raise RuntimeError("routing executor is closed.")
        if operation not in {"matvec", "rmatvec"}:
            raise ValueError(f"unsupported routing operation: {operation!r}.")
        if not self._execution_lock.acquire(blocking=False):
            raise RuntimeError("one routing executor cannot run concurrent products.")
        try:
            batches = plan_fixed_shape_routing_batches(
                self.microshard_plan,
                selected_work_ids=selected_work_ids,
                supported_group_batch_sizes=self.config.supported_group_batch_sizes,
                minimum_batches=self.effective_worker_count,
            )
            dispatch = tuple(
                sorted(batches, key=lambda item: (-item.predicted_cost, item.batch_id))
            )
            selected = tuple(
                work_id for batch in batches for work_id in batch.work_ids
            )
            weights = {} if expansion_weights is None else dict(expansion_weights)
            if set(weights) - set(selected):
                raise ValueError("expansion weights contain unselected work identities.")
            if any(not np.isfinite(value) or value <= 0.0 for value in weights.values()):
                raise ValueError("expansion weights must be positive and finite.")
            prepared_by_batch = {} if prepared_by_batch is None else prepared_by_batch
            retain_ids: set[str] = set()
            retained_bytes = 0
            if retain_prepared:
                links = int(self.operator.routing.num_links)
                width = int(self.operator.inputs.group_od_index_padded.shape[1])
                bytes_per_group = links * (
                    self.operator.dtype.itemsize + np.dtype(bool).itemsize
                ) + width * (np.dtype(np.int32).itemsize + np.dtype(bool).itemsize)
                for batch in batches:
                    predicted = batch.padded_groups * bytes_per_group
                    if (
                        retained_bytes + predicted
                        <= self.config.maximum_retained_batch_bytes
                    ):
                        retain_ids.add(batch.batch_id)
                        retained_bytes += predicted
            started = perf_counter()
            pending: dict[
                Future[
                    tuple[
                        np.ndarray,
                        RoutingBatchExecutionObservation,
                        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
                    ]
                ],
                RoutingExecutionBatch,
            ] = {}
            cursor = 0
            completed: dict[
                str,
                tuple[
                    np.ndarray,
                    RoutingBatchExecutionObservation,
                    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
                ],
            ] = {}

            def dispatch_next() -> bool:
                nonlocal cursor
                if cursor >= len(dispatch):
                    return False
                if cancellation_requested is not None and cancellation_requested():
                    raise ParallelRoutingExecutionInterrupted(
                        "partial routing cancelled before a batch boundary."
                    )
                if absolute_deadline is not None and perf_counter() >= absolute_deadline:
                    raise ParallelRoutingExecutionInterrupted(
                        "partial routing deadline reached before a batch boundary."
                    )
                batch = dispatch[cursor]
                cursor += 1
                submitted = perf_counter()
                future = self._pool.submit(
                    self._run_batch,
                    operation,
                    vector,
                    batch,
                    submitted,
                    weights,
                    prepared_by_batch.get(batch.batch_id),
                    batch.batch_id in retain_ids,
                )
                pending[future] = batch
                return True

            for _ in range(min(self.effective_worker_count, len(dispatch))):
                dispatch_next()
            while pending:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    batch = pending.pop(future)
                    completed[batch.batch_id] = future.result()
                    if cursor < len(dispatch):
                        dispatch_next()

            ordered_results = [completed[item.batch_id] for item in batches]
            value = np.zeros_like(ordered_results[0][0])
            retained = {}
            for batch, (contribution, _, prepared) in zip(
                batches, ordered_results, strict=True
            ):
                value += contribution
                if prepared is not None:
                    retained[batch.batch_id] = prepared
            return (
                ParallelRoutingExecutionResult(
                    value=value,
                    operation=operation,
                    observations=tuple(item[1] for item in ordered_results),
                    wall_seconds=perf_counter() - started,
                    worker_count=self.effective_worker_count,
                    threads_per_worker=self.config.threads_per_worker,
                    selected_work_ids=selected,
                    dispatch_order=tuple(item.batch_id for item in dispatch),
                    retained_batch_count=len(retained),
                ),
                retained,
            )
        except BaseException:
            for future in locals().get("pending", {}):
                future.cancel()
            raise
        finally:
            self._execution_lock.release()

    def execute(
        self,
        operation: ParallelOperation,
        vector: object,
        *,
        selected_work_ids: Sequence[str] | None = None,
        expansion_weights: dict[str, float] | None = None,
        cancellation_requested: Callable[[], bool] | None = None,
        absolute_deadline: float | None = None,
    ) -> ParallelRoutingExecutionResult:
        return self._execute_internal(
            operation,
            vector,
            selected_work_ids=selected_work_ids,
            expansion_weights=expansion_weights,
            cancellation_requested=cancellation_requested,
            absolute_deadline=absolute_deadline,
        )[0]

    def forward_evaluation(
        self,
        evaluation_id: str,
        vector: object,
        *,
        selected_work_ids: Sequence[str],
        expansion_weights: dict[str, float],
    ) -> ParallelRoutingExecutionResult:
        if not evaluation_id or evaluation_id in self._retained_evaluations:
            raise ValueError("evaluation_id must be nonempty and new.")
        result, retained = self._execute_internal(
            "matvec",
            vector,
            selected_work_ids=selected_work_ids,
            expansion_weights=expansion_weights,
            retain_prepared=True,
        )
        self._retained_evaluations[evaluation_id] = _RetainedEvaluation(
            selected_work_ids=tuple(selected_work_ids),
            expansion_weights=dict(expansion_weights),
            prepared_by_batch=retained,
        )
        return result

    def reverse_evaluation(
        self, evaluation_id: str, vector: object, *, release: bool = True
    ) -> ParallelRoutingExecutionResult:
        retained = self._retained_evaluations.get(evaluation_id)
        if retained is None:
            raise ValueError("unknown or released evaluation_id.")
        try:
            result, _ = self._execute_internal(
                "rmatvec",
                vector,
                selected_work_ids=retained.selected_work_ids,
                expansion_weights=retained.expansion_weights,
                prepared_by_batch=retained.prepared_by_batch,
            )
            return result
        finally:
            if release:
                self._retained_evaluations.pop(evaluation_id, None)

    def release_evaluation(self, evaluation_id: str) -> bool:
        return self._retained_evaluations.pop(evaluation_id, None) is not None


class ParallelExactRoutingOperator:
    """Gravity-adjoint adapter executing 100% work through a persistent executor."""

    def __init__(self, base_operator, executor: PersistentParallelRoutingExecutor):
        if executor.operator is not base_operator:
            raise ValueError("parallel executor and base operator must be identical.")
        self.base_operator = base_operator
        self.executor = executor

    def __getattr__(self, name):
        return getattr(self.base_operator, name)

    @property
    def representation(self) -> str:
        return "parallel_exact_sharded_matrix_free"

    def jax_matvec(self, vector: jax.Array) -> jax.Array:
        output = jax.ShapeDtypeStruct(
            (self.num_measurements,), self.base_operator.inputs.base_link_cost.dtype
        )
        return jax.pure_callback(
            lambda value: self.executor.execute("matvec", value).value,
            output,
            vector,
            vmap_method="sequential",
        )

    def jax_rmatvec(self, vector: jax.Array) -> jax.Array:
        output = jax.ShapeDtypeStruct(
            (self.num_free_od,), self.base_operator.inputs.base_link_cost.dtype
        )
        return jax.pure_callback(
            lambda value: self.executor.execute("rmatvec", value).value,
            output,
            vector,
            vmap_method="sequential",
        )

    def jax_matmat(self, matrix: jax.Array) -> jax.Array:
        return jax.vmap(self.jax_matvec, in_axes=1, out_axes=1)(matrix)


class ParallelApproximateRoutingOperator:
    """Matched weighted forward/reverse adapter for one sub-100% selection."""

    def __init__(
        self,
        base_operator,
        executor: PersistentParallelRoutingExecutor,
        selection: FixedBudgetRoutingSelection,
        anchor_demand: object | None = None,
        anchor_routed_measurements: object | None = None,
    ) -> None:
        if executor.operator is not base_operator:
            raise ValueError("parallel executor and base operator must be identical.")
        if selection.microshard_plan_fingerprint != executor.microshard_plan.plan_fingerprint:
            raise ValueError("routing selection and microshard plan identities differ.")
        if selection.exact:
            raise ValueError("effort 100 must use the established exact backend.")
        self.base_operator = base_operator
        self.executor = executor
        self.selection = selection
        if (anchor_demand is None) != (anchor_routed_measurements is None):
            raise ValueError("anchor demand and routed measurements must be provided together.")
        self.anchor_demand = (
            None if anchor_demand is None else np.asarray(anchor_demand)
        )
        self.anchor_routed_measurements = (
            None
            if anchor_routed_measurements is None
            else np.asarray(anchor_routed_measurements)
        )
        if self.anchor_demand is not None and self.anchor_demand.shape != (
            self.num_free_od,
        ):
            raise ValueError("anchor demand has the wrong shape.")
        if (
            self.anchor_routed_measurements is not None
            and self.anchor_routed_measurements.shape != (self.num_measurements,)
        ):
            raise ValueError("anchor routed measurements have the wrong shape.")
        self._counter = 0
        self._pending_evaluation_id: str | None = None

    def __getattr__(self, name):
        return getattr(self.base_operator, name)

    @property
    def representation(self) -> str:
        return "parallel_approximate_sharded_matrix_free"

    def _forward_callback(self, value):
        if self._pending_evaluation_id is not None:
            self.executor.release_evaluation(self._pending_evaluation_id)
        evaluation_id = (
            f"partial-{self.selection.selection_fingerprint}-{self._counter}"
        )
        self._counter += 1
        routed_value = (
            value if self.anchor_demand is None else np.asarray(value) - self.anchor_demand
        )
        result = self.executor.forward_evaluation(
            evaluation_id,
            routed_value,
            selected_work_ids=self.selection.selected_work_ids,
            expansion_weights=self.selection.weight_by_work_id,
        )
        self._pending_evaluation_id = evaluation_id
        return (
            result.value
            if self.anchor_routed_measurements is None
            else self.anchor_routed_measurements + result.value
        )

    def _reverse_callback(self, value):
        if self._pending_evaluation_id is None:
            raise RuntimeError("partial reverse requires its matching forward evaluation.")
        evaluation_id = self._pending_evaluation_id
        self._pending_evaluation_id = None
        return self.executor.reverse_evaluation(evaluation_id, value).value

    def jax_matvec(self, vector: jax.Array) -> jax.Array:
        output = jax.ShapeDtypeStruct(
            (self.num_measurements,), self.base_operator.inputs.base_link_cost.dtype
        )
        return jax.pure_callback(
            self._forward_callback, output, vector, vmap_method="sequential"
        )

    def jax_rmatvec(self, vector: jax.Array) -> jax.Array:
        output = jax.ShapeDtypeStruct(
            (self.num_free_od,), self.base_operator.inputs.base_link_cost.dtype
        )
        return jax.pure_callback(
            self._reverse_callback, output, vector, vmap_method="sequential"
        )

    def jax_matmat(self, matrix: jax.Array) -> jax.Array:
        raise NotImplementedError(
            "partial routing supports the matched adjoint strategy, not matmat."
        )

    def release_pending(self) -> bool:
        if self._pending_evaluation_id is None:
            return False
        evaluation_id = self._pending_evaluation_id
        self._pending_evaluation_id = None
        return self.executor.release_evaluation(evaluation_id)
