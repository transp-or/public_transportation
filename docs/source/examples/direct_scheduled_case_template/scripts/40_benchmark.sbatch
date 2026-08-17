#!/bin/bash
#SBATCH --job-name=od-benchmark
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=results/logs/benchmark-%j.out
#SBATCH --error=results/logs/benchmark-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
uv run --frozen python run_case.py benchmark
