#!/usr/bin/env bash
#SBATCH --job-name=case-materialize-bins
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/.." && pwd)"
CANDIDATE="${CASE_BIN_CANDIDATE:-recommendation}"
REVIEWER="${CASE_BIN_REVIEWER:-${USER:-case-owner}}"
JAX_ENABLE_X64=true uv run --frozen python run.py materialize-bins \
  --candidate "$CANDIDATE" --reviewer "$REVIEWER" --json-progress
