#!/bin/bash
#SBATCH --job-name=od-prepare
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=results/logs/prepare-%j.out
#SBATCH --error=results/logs/prepare-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
uv run --frozen python run_case.py prepare
