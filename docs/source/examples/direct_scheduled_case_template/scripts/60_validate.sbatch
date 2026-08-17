#!/bin/bash
#SBATCH --job-name=od-validate
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results/logs/validate-%j.out
#SBATCH --error=results/logs/validate-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
uv run --frozen python run_case.py validate
