#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
variant="${1:-}"
case "${variant}" in
  b0)
    mode=disabled
    heatmap_variant=off
    heatmap_sigma_ratio=0.25
    tag=b0_center_lr1_noclip_seed1
    master_port=29621
    ;;
  h1_sup)
    mode=disabled
    heatmap_variant=h1_sup
    heatmap_sigma_ratio=0.15
    tag=h1_sup_center_lr1_noclip_seed1
    master_port=29622
    ;;
  sam)
    mode=sam_roi_blend
    heatmap_variant=off
    heatmap_sigma_ratio=0.25
    tag=sam_center_lr1_noclip_seed1
    master_port=29623
    ;;
  *)
    echo "Usage: bash scripts/run_lr1_center_full.sh {b0|h1_sup|sam}" >&2
    exit 2
    ;;
esac

run_root="${repo_root}/.local_data/historical_lr_center/${variant}"
mkdir -p "${run_root}"
status="${run_root}/full_training.status"
log="${run_root}/full_training.log"
if [[ -e "${status}" ]]; then
  echo "Existing run status; refusing to overwrite ${status}" >&2
  exit 2
fi

# Restore the historical backbone LR and no-clipping settings while keeping
# the fixed center frames and four-GPU DDP protocol required by the SAM cache.
export MODE="${mode}"
export HEATMAP_VARIANT="${heatmap_variant}"
export HEATMAP_SIGMA_RATIO="${heatmap_sigma_ratio}"
export BACKBONE_LR_SCALE=1
export GRADIENT_CLIPPING=0
export SEED=1
export EPOCHS=30
export TEST_FREQ=1
export BATCH=16
export TEST_BATCH=4
export TAG="${tag}"
export MASTER_PORT="${master_port}"
export CHECKPOINT_ROOT="${run_root}/full_checkpoints"

mapfile -t gpu_free < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if [[ "${#gpu_free[@]}" -ne 4 ]]; then
  echo "Expected four visible GPUs; found ${#gpu_free[@]}" >&2
  exit 2
fi
for free in "${gpu_free[@]}"; do
  if (( free < 20000 )); then
    echo "Need at least 20 GiB free per GPU; free MiB: ${gpu_free[*]}" >&2
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
