#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
probe=$(sbatch --parsable scripts/probe.sbatch)
check=$(sbatch --parsable --dependency="afterok:${probe}" scripts/00_check.sbatch)
bins=$(sbatch --parsable --dependency="afterok:${check}" scripts/10_time_discretization.sbatch)
zeros=$(sbatch --parsable --dependency="afterok:${bins}" scripts/20_structural_zeros.sbatch)
prepare=$(sbatch --parsable --dependency="afterok:${zeros}" scripts/30_prepare.sbatch)
preflight=$(sbatch --parsable --dependency="afterok:${prepare}" scripts/40_preflight.sbatch)
fit=$(sbatch --parsable --dependency="afterok:${preflight}" scripts/50_fit.sbatch)
printf '%s\n' \
  "probe=${probe}" "check=${check}" "bins=${bins}" "zeros=${zeros}" \
  "prepare=${prepare}" "preflight=${preflight}" "fit=${fit}"
