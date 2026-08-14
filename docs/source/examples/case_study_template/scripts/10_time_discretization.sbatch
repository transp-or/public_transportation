#!/usr/bin/env bash
#SBATCH --job-name=case-time-bins
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
JAX_ENABLE_X64=true uv run --frozen python run.py time-discretization --json-progress
