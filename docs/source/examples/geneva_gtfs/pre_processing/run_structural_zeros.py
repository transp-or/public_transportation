"""Run TOML-driven structural-zero preprocessing for the Geneva snapshot."""

from __future__ import annotations

import json
from pathlib import Path

from public_transportation.preprocessing import run_structural_zero_preprocessing


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result = run_structural_zero_preprocessing(ROOT / "structural_zeros.toml")
    print(
        json.dumps(
            {
                "num_cells": result.analysis.num_cells,
                "num_structural_zero": result.analysis.num_structural_zero,
                "num_retained": result.analysis.num_retained,
                "num_fixed_merged": result.reconciliation.num_merged,
                "num_free_after_merge": (
                    result.analysis.num_cells - result.reconciliation.num_merged
                ),
                "primary_reason_counts": result.analysis.reason_counts,
                "output_folder": str(result.outputs.folder),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
