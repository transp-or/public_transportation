"""Create a deliberate metadata mismatch for cache-rejection benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    with np.load(args.path, allow_pickle=False) as archive:
        content = {
            name: np.array(archive[name], copy=True) for name in archive.files
        }
    metadata = json.loads(str(content["metadata"].item()))
    provenance = json.loads(metadata["provenance_payload_json"])
    provenance["assignment_config"]["max_transfer_wait_min"] = -999.0
    metadata["provenance_payload_json"] = json.dumps(
        provenance, sort_keys=True, separators=(",", ":")
    )
    content["metadata"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    with args.path.open("wb") as stream:
        np.savez_compressed(stream, **content)


if __name__ == "__main__":
    main()
