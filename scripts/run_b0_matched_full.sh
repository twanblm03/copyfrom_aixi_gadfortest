#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

run_root="${repo_root}/.local_data/b0_matched"
mkdir -p "${run_root}"
status="${run_root}/full_training.status"
log="${run_root}/full_training.log"

# Match the completed SAM run: fixed TRAIN center frames, seed 1, four GPUs,
# batch 16, 30 epochs, and clipping from epoch 8 after SAM's epoch-7 resume.
export MODE=disabled
export SEED=1
export EPOCHS=30
export TEST_FREQ=1
export BATCH=16
export TEST_BATCH=4
export GRADIENT_CLIPPING=1
export GRADIENT_CLIPPING_START_EPOCH=8
export TAG=b0_fixedcenter_matched_sam_seed1
export MASTER_PORT=29619
export CHECKPOINT_ROOT="${run_root}/full_checkpoints"

mapfile -t gpu_memory < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [[ "${#gpu_memory[@]}" -ne 4 ]]; then
  echo "Expected four visible GPUs; found ${#gpu_memory[@]}" >&2
  exit 2
fi
for used in "${gpu_memory[@]}"; do
  if (( used >= 2048 )); then
    echo "Expected free GPUs; memory used MiB: ${gpu_memory[*]}" >&2
    exit 2
  fi
done

printf 'training_started %s\n' "$(date -Is)" > "${status}"
trap 'printf "training_interrupted %s\n" "$(date -Is)" > "${status}"; exit 130' INT TERM

set +e
bash scripts/run_sam_person_full.sh > "${log}" 2>&1
code=$?
set -e
trap - INT TERM
if (( code == 0 )); then
  printf 'training_complete %s\n' "$(date -Is)" > "${status}"
else
  printf 'training_failed code=%s %s\n' "${code}" "$(date -Is)" > "${status}"
fi
exit "${code}"
