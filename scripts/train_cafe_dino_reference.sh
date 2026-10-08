#!/usr/bin/env bash
# DINOv2 reference configuration recovered from cherixi/CAFE_codebase experiment commits.
# Usage: PREFLIGHT_STEPS=2 bash scripts/train_cafe_dino_reference.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_path="${DATA_PATH:-/share/share/aixi/Cafe_Dataset/Cafe_Dataset/Cafe_Dataset/Dataset}"
groundtruth="${GROUNDTRUTH:-${repo_root}/evaluation/gt_tracks.txt}"
tracks_path="${TRACKS_PATH:-${data_path}/cafe/gt_tracks.pkl}"
videomae_feats_path="${VIDEOMAE_FEATS_PATH:-/share/share/aixi/videomae_features_giant}"
# Validated for num_frame=8 on this machine; GPU 5 lacks enough free headroom.
device="${DEVICE:-0,1,2,4}"
IFS="," read -r -a device_list <<< "${device}"
world_size="${NPROC_PER_NODE:-${#device_list[@]}}"
master_port="${MASTER_PORT:-29512}"
python_bin="${PYTHON_BIN:-python}"
checkpoint_root="${CHECKPOINT_ROOT:-/tmp/cafe_checkpoints}"

# These are the senior's DINOv2-B/14 partial-finetuning values.
batch="${BATCH:-16}"
num_frame="${NUM_FRAME:-8}"
# The original DataParallel command uses 52 globally; DDP test_batch is per rank.
test_batch="${TEST_BATCH:-$((52 / world_size))}"
epochs="${EPOCHS:-30}"
test_freq="${TEST_FREQ:-1}"
log_interval="${LOG_INTERVAL:-10}"
use_mae="${USE_MAE:-1}"
preflight_steps="${PREFLIGHT_STEPS:-0}"
resume_args=()
mae_args=()
preflight_args=()

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python executable not found: ${python_bin}. Set PYTHON_BIN or activate gad." >&2
  exit 1
fi
if [[ "${world_size}" -ne "${#device_list[@]}" ]]; then
  echo "NPROC_PER_NODE (${world_size}) must match DEVICE (${device})." >&2
  exit 1
fi
if [[ ! -d "${data_path}/cafe" || ! -f "${tracks_path}" || ! -f "${groundtruth}" ]]; then
  echo "Dataset, tracks, or evaluation ground truth is unavailable." >&2
  exit 1
fi
mkdir -p "${checkpoint_root}"
if [[ ! -w "${checkpoint_root}" ]]; then
  echo "Checkpoint root is not writable: ${checkpoint_root}" >&2
  exit 1
fi

python_bin_path="$(command -v "${python_bin}")"
python_root="$(cd "$(dirname "${python_bin_path}")/.." && pwd)"
if [[ -f "${python_root}/lib/libstdc++.so.6" ]]; then
  export LD_LIBRARY_PATH="${python_root}/lib:${LD_LIBRARY_PATH:-}"
fi
if [[ -d "${repo_root}/.local_deps/xformers_016" ]]; then
  export PYTHONPATH="${repo_root}/.local_deps/xformers_016${PYTHONPATH:+:${PYTHONPATH}}"
fi

if [[ "${use_mae}" == "1" ]]; then
  if [[ ! -f "${videomae_feats_path}/1_0.npy" ]]; then
    echo "VideoMAE-v2 feature cache is missing: ${videomae_feats_path}" >&2
    exit 1
  fi
  mae_args=(--mae_version v2 --videomae_feats_path "${videomae_feats_path}")
elif [[ "${use_mae}" == "0" ]]; then
  mae_args=(--no_mae)
else
  echo "USE_MAE must be 0 or 1, got: ${use_mae}" >&2
  exit 1
fi

if [[ -n "${RESUME_PATH:-}" ]]; then
  if [[ ! -f "${RESUME_PATH}" ]]; then
    echo "Resume checkpoint not found: ${RESUME_PATH}" >&2
    exit 1
  fi
  resume_args=(--load_model --model_path "${RESUME_PATH}")
fi

if ! [[ "${preflight_steps}" =~ ^[0-9]+$ ]]; then
  echo "PREFLIGHT_STEPS must be a non-negative integer." >&2
  exit 1
fi
if [[ "${preflight_steps}" -gt 0 ]]; then
  if [[ -n "${RESUME_PATH:-}" ]]; then
    echo "PREFLIGHT_STEPS cannot be combined with RESUME_PATH." >&2
    exit 1
  fi
  epochs=1
  test_freq=2  # skip validation; this checks forward/backward/optimizer memory only
  preflight_args=(--max_train_steps "${preflight_steps}" --report_peak_memory)
  echo "DINO reference memory preflight: ${preflight_steps} optimizer step(s)."
fi

cd "${repo_root}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
CUDA_VISIBLE_DEVICES="${device}" "${python_bin}" -m torch.distributed.run \
  --nproc_per_node="${world_size}" \
  --master_port "${master_port}" \
  train.py \
  --data_path "${data_path}" \
  --tracks_path "${tracks_path}" \
  --groundtruth "${groundtruth}" \
  --checkpoint_root "${checkpoint_root}" \
  --split "${SPLIT:-place}" \
  --backbone dinov2_vitb14 \
  --unfreeze_blocks 2 \
  --frozen_batch_norm \
  --backbone_lr_scale 0.1 \
  --batch "${batch}" \
  --test_batch "${test_batch}" \
  --num_frame "${num_frame}" \
  --optimizer adamw \
  --weight_decay 1e-4 \
  --lr_schedule cyclic \
  --lr_step 4 \
  --lr_step_down 25 \
  "${mae_args[@]}" \
  --epochs "${epochs}" \
  --test_freq "${test_freq}" \
  --log_interval "${log_interval}" \
  --device "${device}" \
  --distributed \
  "${preflight_args[@]}" \
  "${resume_args[@]}"
