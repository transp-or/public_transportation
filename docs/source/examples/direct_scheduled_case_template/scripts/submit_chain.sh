#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs results/manifests results/checkpoints results/artifacts

check=$(sbatch --parsable scripts/00_check.sbatch)
zeros=$(sbatch --parsable --dependency="afterok:${check}" scripts/10_structural_zeros.sbatch)
prepare=$(sbatch --parsable --dependency="afterok:${zeros}" scripts/20_prepare.sbatch)
preflight=$(sbatch --parsable --dependency="afterok:${prepare}" scripts/30_preflight.sbatch)
benchmark=$(sbatch --parsable --dependency="afterok:${preflight}" scripts/40_benchmark.sbatch)
fit=$(sbatch --parsable --dependency="afterok:${benchmark}" scripts/50_fit.sbatch)
validate=$(sbatch --parsable --dependency="afterok:${fit}" scripts/60_validate.sbatch)
printf '%s\n' \
  "check=${check}" "structural-zeros=${zeros}" "prepare=${prepare}" \
  "preflight=${preflight}" "benchmark=${benchmark}" "fit=${fit}" \
  "validate=${validate}"
