#!/usr/bin/env bash
#SBATCH --job-name=case-prepare
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/.." && pwd)"
JAX_ENABLE_X64=true uv run --frozen python run.py prepare --json-progress
