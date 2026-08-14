#!/usr/bin/env bash
#SBATCH --job-name=case-preflight
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/.." && pwd)"
JAX_ENABLE_X64=true uv run --frozen python run.py preflight --json-progress
