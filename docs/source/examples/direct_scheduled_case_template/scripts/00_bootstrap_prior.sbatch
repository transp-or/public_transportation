#!/bin/bash
#SBATCH --job-name=od-bootstrap-prior
# Bootstrap may scan a large OD--time universe. These are deliberately
# conservative defaults; the case owner may increase them after a pilot.
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=results/logs/bootstrap-prior-%j.out
#SBATCH --error=results/logs/bootstrap-prior-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
uv run --frozen python run_case.py bootstrap-prior
