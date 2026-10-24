#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
run_root="${repo_root}/.local_data/sam/dp4_randomsegments_lr1"
status="${run_root}/queue.status"
verify_log="${run_root}/cache_build/verify.log"
grep -q '^verified complete: 223670 frames, 2308032 actors, 0 failed masks;' "${verify_log}"
[[ ! -e "${run_root}/smoke2/training.status" ]]
[[ ! -e "${run_root}/full/training.status" ]]

phase=smoke
on_exit() {
  code=$?
  if (( code != 0 )); then
    printf 'pipeline_failed phase=%s code=%s %s\n' \
      "${phase}" "${code}" "$(date -Is)" > "${status}"
  fi
}
trap on_exit EXIT

printf 'smoke_retry_started %s\n' "$(date -Is)" > "${status}"
SMOKE_STEPS=2 MIN_GPU_FREE_MIB=23000 SAM_TRAIN_SAMPLING=randomsegments \
  bash scripts/run_sam_dp4_full.sh
grep -q '^training_complete ' "${run_root}/smoke2/training.status"
printf 'smoke_complete %s\n' "$(date -Is)" > "${status}"

phase=full_training
printf 'full_training_started %s\n' "$(date -Is)" > "${status}"
MIN_GPU_FREE_MIB=23000 SAM_TRAIN_SAMPLING=randomsegments \
  bash scripts/run_sam_dp4_full.sh
grep -q '^training_complete ' "${run_root}/full/training.status"
printf 'full_training_complete %s\n' "$(date -Is)" > "${status}"
