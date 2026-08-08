"""Scaling benchmark for Phase-7 compressed products and ``H=B Phi``."""

from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np
from scipy import sparse

from public_transportation.inference.reduced_od import (
    build_basis_response,
    build_reduced_response_operator_from_coo,
)


def _atoms(
    measurements: int, cells: int, classes: int, entries_per_class: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for cell in range(cells):
        response_class = cell % classes
        for offset in range(entries_per_class):
            rows.append((response_class * 37 + offset * 101) % measurements)
            columns.append(cell)
            values.append(1.0 + (response_class + offset) % 7 / 10.0)
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(columns, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
    )


def _basis(cells: int, parameters: int) -> sparse.csr_matrix:
    rows = np.repeat(np.arange(cells, dtype=np.int64), 2)
    columns = np.empty(2 * cells, dtype=np.int64)
    columns[0::2] = np.arange(cells) % parameters
    columns[1::2] = (np.arange(cells) * 7 + 3) % parameters
    values = np.empty(2 * cells, dtype=np.float64)
    values[0::2] = 1.0
    values[1::2] = 0.25
    return sparse.coo_matrix(
        (values, (rows, columns)), shape=(cells, parameters)
    ).tocsr()


def _run(cells: int, measurements: int, parameters: int, repeats: int) -> dict[str, int | float | str]:
    classes = max(1, cells // 4)
    rows, columns, values = _atoms(measurements, cells, classes, 6)
    build_started = time.perf_counter()
    operator = build_reduced_response_operator_from_coo(
        number_of_measurements=measurements,
        number_of_free_cells=cells,
        measurement_index=rows,
        free_cell_index=columns,
        response_values=values,
    )
    operator_build_seconds = time.perf_counter() - build_started
    demand = np.linspace(0.5, 1.5, cells)
    measurement_weights = np.linspace(-1.0, 1.0, measurements)
    operator.matvec(demand)
    operator.rmatvec(measurement_weights)
    product_started = time.perf_counter()
    for _ in range(repeats):
        operator.matvec(demand)
        operator.rmatvec(measurement_weights)
    product_seconds = (time.perf_counter() - product_started) / (2 * repeats)
    phi = _basis(cells, parameters)
    basis_started = time.perf_counter()
    basis_response = build_basis_response(operator, phi, storage="auto")
    basis_build_seconds = time.perf_counter() - basis_started
    diagnostics = operator.diagnostics
    return {
        "measurements": measurements,
        "free_cells": cells,
        "response_classes": diagnostics.number_of_response_classes,
        "original_nnz": diagnostics.original_nnz,
        "compressed_nnz": diagnostics.compressed_nnz,
        "compression_ratio": diagnostics.compression_ratio,
        "operator_build_seconds": operator_build_seconds,
        "mean_product_seconds": product_seconds,
        "operator_retained_bytes": diagnostics.retained_bytes,
        "basis_parameters": parameters,
        "basis_nnz": basis_response.diagnostics.basis_nnz,
        "h_nnz": basis_response.diagnostics.result_nnz,
        "h_storage": basis_response.diagnostics.storage,
        "h_build_seconds": basis_build_seconds,
        "h_retained_bytes": basis_response.diagnostics.retained_bytes,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, nargs="+", default=[1000, 5000, 20000])
    parser.add_argument("--measurements", type=int, default=2000)
    parser.add_argument("--parameters", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=20)
    arguments = parser.parse_args()
    if any(value < 1 for value in arguments.cells):
        parser.error("all --cells values must be positive")
    if min(arguments.measurements, arguments.parameters, arguments.repeats) < 1:
        parser.error("measurements, parameters, and repeats must be positive")
    print(
        json.dumps(
            [
                _run(
                    cells,
                    arguments.measurements,
                    arguments.parameters,
                    arguments.repeats,
                )
                for cells in arguments.cells
            ],
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
