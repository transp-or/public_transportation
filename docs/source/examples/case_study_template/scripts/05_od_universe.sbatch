#!/usr/bin/env bash
#SBATCH --job-name=case-od-universe
#SBATCH --output=results/logs/%x-%j.out
set -euo pipefail
cd "$(dirname "$0")/.."
JAX_ENABLE_X64=true uv run --frozen python run.py od-universe --json-progress
