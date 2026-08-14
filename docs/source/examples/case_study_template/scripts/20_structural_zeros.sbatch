#!/usr/bin/env bash
#SBATCH --job-name=case-structural-zeroes
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
JAX_ENABLE_X64=true uv run --frozen python run.py structural-zeros --json-progress
