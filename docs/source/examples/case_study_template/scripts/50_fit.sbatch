#!/usr/bin/env bash
#SBATCH --job-name=case-fit
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
JAX_ENABLE_X64=true uv run --frozen python run.py fit --method map --likelihood poisson --json-progress
