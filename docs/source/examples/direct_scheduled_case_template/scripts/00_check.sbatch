#!/bin/bash
#SBATCH --job-name=od-check
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=results/logs/check-%j.out
#SBATCH --error=results/logs/check-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
uv run --frozen python run_case.py check
