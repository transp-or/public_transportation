#!/bin/bash
#SBATCH --job-name=od-fit
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/logs/fit-%j.out
#SBATCH --error=results/logs/fit-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
uv run --frozen python run_case.py fit
