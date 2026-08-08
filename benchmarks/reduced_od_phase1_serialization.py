"""Microbenchmark Phase-1 reduced-OD metadata serialization and hashing."""

from __future__ import annotations

import json
from statistics import median
from time import perf_counter

import numpy as np

from public_transportation.inference.reduced_od import (
    JourneyODTimeKey,
    ReducedODProblemContract,
)


def main() -> None:
    num_od = 100_000
    keys = tuple(
        JourneyODTimeKey(f"O{i // 1000:04d}", f"D{i % 1000:04d}", "T1")
        for i in range(num_od)
    )
    free = np.arange(num_od, dtype=np.int64)
    contract = ReducedODProblemContract(
        configuration_fingerprint="configuration",
        timetable_artifact_fingerprint="timetable",
        response_artifact_fingerprint="response",
        od_keys=keys,
        free_od_indices=free,
        fixed_od_indices=np.empty(0, dtype=np.int64),
        fixed_od_values=np.empty(0, dtype=np.float64),
    )

    timings: list[float] = []
    payload = ""
    for _ in range(7):
        started = perf_counter()
        payload = contract.fingerprint_payload_json
        _ = contract.fingerprint
        timings.append(perf_counter() - started)

    print(
        json.dumps(
            {
                "num_od": num_od,
                "payload_bytes": len(payload.encode("utf-8")),
                "median_serialize_and_hash_seconds": median(timings),
                "retained_array_bytes": (
                    contract.free_od_indices.nbytes
                    + contract.fixed_od_indices.nbytes
                    + contract.fixed_od_values.nbytes
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
