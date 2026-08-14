#!/usr/bin/env bash
#SBATCH --job-name=case-fit
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/.." && pwd)"
JAX_ENABLE_X64=true uv run --frozen python run.py fit --method map --likelihood poisson --json-progress
