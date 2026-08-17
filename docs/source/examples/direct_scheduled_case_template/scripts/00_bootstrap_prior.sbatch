#!/bin/bash
#SBATCH --job-name=od-bootstrap-prior
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=results/logs/bootstrap-prior-%j.out
#SBATCH --error=results/logs/bootstrap-prior-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
uv run --frozen python run_case.py bootstrap-prior
