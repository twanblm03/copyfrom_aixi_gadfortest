#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
smoke_steps="${SMOKE_STEPS:-0}"
if ! [[ "${smoke_steps}" =~ ^[0-9]+$ ]]; then
  echo "SMOKE_STEPS must be a non-negative integer" >&2
  exit 2
fi
min_gpu_free_mib="${MIN_GPU_FREE_MIB:-30000}"
if ! [[ "${min_gpu_free_mib}" =~ ^[0-9]+$ ]]; then
  echo "MIN_GPU_FREE_MIB must be a non-negative integer" >&2
  exit 2
fi

sampling="${SAM_TRAIN_SAMPLING:-randomsegments}"
case "${sampling}" in
  center)
    fixed_center_train=1
    manifest="${repo_root}/.local_data/sam/full_manifest.jsonl"
    ;;
  randomsegments)
    fixed_center_train=0
    manifest="${repo_root}/.local_data/sam/randomsegments_manifest.jsonl"
    ;;
  *)
    echo "SAM_TRAIN_SAMPLING must be center or randomsegments" >&2
    exit 2
    ;;
esac
run_base="${repo_root}/.local_data/sam/dp4_${sampling}_lr1"
tag="sam_${sampling}_dp4_lr1_noclip_seed1"
if (( smoke_steps > 0 )); then
  run_root="${run_base}/smoke${smoke_steps}"
  tag="${tag}_smoke${smoke_steps}"
  epochs=1
  warmup_epochs=1
  test_freq=31
else
  run_root="${run_base}/full"
  epochs=30
  warmup_epochs=5
  test_freq=1
fi
mkdir -p "${run_root}"
status="${run_root}/training.status"
log="${run_root}/training.log"
if [[ -e "${status}" ]]; then
  echo "Existing run status; refusing to overwrite ${status}" >&2
  exit 2
fi

while true; do
  mapfile -t gpu_free < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  enough=1
  if [[ "${#gpu_free[@]}" -ne 4 ]]; then
    enough=0
  else
    for free in "${gpu_free[@]}"; do
      if (( free < min_gpu_free_mib )); then
        enough=0
      fi
    done
  fi
  if (( enough )); then
    break
  fi
  printf 'Waiting for at least %s MiB free on each GPU; free MiB: %s (%s)\n' \
    "${min_gpu_free_mib}" "${gpu_free[*]}" "$(date -Is)"
  sleep 60
done

export MODE=sam_roi_blend
export DISTRIBUTED=0
export PARALLEL_TRANSPORT=host
export FIXED_CENTER_TRAIN="${fixed_center_train}"
export SAM_MANIFEST="${manifest}"
export PROJECTED_CACHE=1
export CONTEXT_FRACTION=0.5
export HEATMAP_VARIANT=off
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
export CHECKPOINT_ROOT="${run_root}/checkpoints"

printf '%s\n' "$$" > "${run_root}/training.pid"
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
