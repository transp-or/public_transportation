#!/usr/bin/env bash
#SBATCH --job-name=case-probe
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
JAX_ENABLE_X64=true uv run --frozen python run.py check --json-progress
