#!/bin/bash
#SBATCH --job-name=od-preflight
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=results/logs/preflight-%j.out
#SBATCH --error=results/logs/preflight-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
uv run --frozen python run_case.py preflight
