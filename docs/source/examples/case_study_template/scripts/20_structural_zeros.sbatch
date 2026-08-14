#!/usr/bin/env bash
#SBATCH --job-name=case-structural-zeroes
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/.." && pwd)"
JAX_ENABLE_X64=true uv run --frozen python run.py structural-zeros --json-progress
