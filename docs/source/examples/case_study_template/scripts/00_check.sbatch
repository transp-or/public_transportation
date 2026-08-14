#!/usr/bin/env bash
#SBATCH --job-name=case-check
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/.." && pwd)"
JAX_ENABLE_X64=true uv run --frozen python run.py check --json-progress
