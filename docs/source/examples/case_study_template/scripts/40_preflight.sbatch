#!/usr/bin/env bash
#SBATCH --job-name=case-preflight
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
JAX_ENABLE_X64=true uv run --frozen python run.py preflight --json-progress
