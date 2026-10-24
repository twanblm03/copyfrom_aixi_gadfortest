#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
variant="${1:-}"
case "${variant}" in
  b0)
    heatmap_variant=off
    heatmap_sigma_ratio=0.25
    tag=b0_randomseg_dp4_lr1_noclip_seed1
    ;;
  h1_sup)
    heatmap_variant=h1_sup
    heatmap_sigma_ratio=0.15
    tag=h1_sup_randomseg_dp4_lr1_noclip_seed1
    ;;
  *)
    echo "Usage: bash scripts/run_lr1_randomsegments_dp4_full.sh {b0|h1_sup}" >&2
    exit 2
    ;;
esac

smoke_steps="${SMOKE_STEPS:-0}"
if ! [[ "${smoke_steps}" =~ ^[0-9]+$ ]]; then
  echo "SMOKE_STEPS must be a non-negative integer" >&2
  exit 2
fi
if (( smoke_steps > 0 )); then
  tag="${tag}_smoke${smoke_steps}"
  run_root="${repo_root}/.local_data/historical_lr_randomsegments_dp4/smoke_${variant}_${smoke_steps}"
  epochs=1
  warmup_epochs=1
  test_freq=31
else
  run_root="${repo_root}/.local_data/historical_lr_randomsegments_dp4/${variant}"
  epochs=30
  warmup_epochs=5
  test_freq=1
fi
mkdir -p "${run_root}"
status="${run_root}/full_training.status"
log="${run_root}/full_training.log"
if [[ -e "${status}" ]]; then
  echo "Existing run status; refusing to overwrite ${status}" >&2
  exit 2
fi

# DataParallel gathers all 16 model outputs before the criterion; H1-Sup's
# unchanged torch.stack(...).mean() is exactly the historical full-batch mean.
export MODE=disabled
export DISTRIBUTED=0
export PARALLEL_TRANSPORT=host
export FIXED_CENTER_TRAIN=0
export HEATMAP_VARIANT="${heatmap_variant}"
export HEATMAP_SIGMA_RATIO="${heatmap_sigma_ratio}"
export BACKBONE_LR_SCALE=1
export GRADIENT_CLIPPING=0
export SEED=1
export EPOCHS="${epochs}"
export WARMUP_EPOCHS="${warmup_epochs}"
export TEST_FREQ="${test_freq}"
export MAX_TRAIN_STEPS="${smoke_steps}"
export BATCH=16
export TEST_BATCH=4
export TAG="${tag}"
export CHECKPOINT_ROOT="${run_root}/full_checkpoints"

mapfile -t gpu_free < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if [[ "${#gpu_free[@]}" -ne 4 ]]; then
  echo "Expected four visible GPUs; found ${#gpu_free[@]}" >&2
  exit 2
fi
for free in "${gpu_free[@]}"; do
  if (( free < 25000 )); then
    echo "Need at least 25,000 MiB free per GPU; free MiB: ${gpu_free[*]}" >&2
    exit 2
  fi
done

printf '%s\n' "$$" > "${run_root}/full_training.pid"
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
