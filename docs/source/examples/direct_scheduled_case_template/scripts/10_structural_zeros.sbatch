#!/bin/bash
#SBATCH --job-name=od-structural-zeros
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=results/logs/structural-zeros-%j.out
#SBATCH --error=results/logs/structural-zeros-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
uv run --frozen python run_case.py structural-zeros
