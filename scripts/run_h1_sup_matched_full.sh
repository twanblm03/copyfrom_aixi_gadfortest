#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

run_root="${repo_root}/.local_data/h1_sup_matched"
mkdir -p "${run_root}"
status="${run_root}/full_training.status"
log="${run_root}/full_training.log"

# Match the current B0/SAM training protocol while enabling only the H1-Sup
# auxiliary group heatmap target used by the historical H1-Sup experiment.
export MODE=disabled
export HEATMAP_VARIANT=h1_sup
export HEATMAP_SIGMA_RATIO=0.15
export SEED=1
export EPOCHS=30
export TEST_FREQ=1
export BATCH=16
export TEST_BATCH=4
export GRADIENT_CLIPPING=1
export GRADIENT_CLIPPING_START_EPOCH=8
export TAG=h1_sup_fixedcenter_matched_seed1
export MASTER_PORT=29620
export CHECKPOINT_ROOT="${run_root}/full_checkpoints"

mapfile -t gpu_free < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if [[ "${#gpu_free[@]}" -ne 4 ]]; then
  echo "Expected four visible GPUs; found ${#gpu_free[@]}" >&2
  exit 2
fi
for free in "${gpu_free[@]}"; do
  if (( free < 20000 )); then
    echo "Need at least 20 GiB free per GPU for shared H1-Sup training; free MiB: ${gpu_free[*]}" >&2
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
