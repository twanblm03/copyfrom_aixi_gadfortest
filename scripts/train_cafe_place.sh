#!/usr/bin/env bash
set -euo pipefail

# Café-paper training configuration, adapted to DINOv2: fine-tune only the
# last two blocks at 0.1x head LR, on four GPUs with global batch size 16.
# Override DATA_PATH, GROUNDTRUTH, DEVICE, or PYTHON_BIN as needed.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_path="${DATA_PATH:-/share/share/aixi/Cafe_Dataset/Cafe_Dataset/Cafe_Dataset/Dataset}"
groundtruth="${GROUNDTRUTH:-${repo_root}/evaluation/gt_tracks.txt}"
local_tracks="${repo_root}/.local_data/cafe/gt_tracks.pkl"
tracks_path="${TRACKS_PATH:-${local_tracks}}"
device="${DEVICE:-0,2,3,4}"
IFS=',' read -r -a device_list <<< "${device}"
world_size="${NPROC_PER_NODE:-${#device_list[@]}}"
master_port="${MASTER_PORT:-29500}"
python_bin="${PYTHON_BIN:-python}"
# This machine's default NCCL P2P path stalls.  Use shared-memory transport;
use_mae="${USE_MAE:-0}"
videomae_feats_path="${VIDEOMAE_FEATS_PATH:-${repo_root}/.local_data/videomae_v2_giant}"
# these remain overridable for machines with a healthy P2P fabric.
nccl_p2p_disable="${NCCL_P2P_DISABLE:-1}"
nccl_ib_disable="${NCCL_IB_DISABLE:-1}"
resume_args=()

mae_args=(--no_mae)
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python executable not found: ${python_bin}. Set PYTHON_BIN or activate gad." >&2
  exit 1
fi


# Prefer the interpreter environment's C++ runtime over the host's older one.
python_bin_path="$(command -v "${python_bin}")"
python_root="$(cd "$(dirname "${python_bin_path}")/.." && pwd)"
if [[ -f "${python_root}/lib/libstdc++.so.6" ]]; then
  export LD_LIBRARY_PATH="${python_root}/lib:${LD_LIBRARY_PATH:-}"
fi
if [[ -d "${repo_root}/.local_deps/xformers_016" ]]; then
  export PYTHONPATH="${repo_root}/.local_deps/xformers_016${PYTHONPATH:+:${PYTHONPATH}}"
fi
if [[ "${world_size}" -ne "${#device_list[@]}" ]]; then
  echo "NPROC_PER_NODE (${world_size}) must match the number of GPUs in DEVICE (${device})." >&2
  exit 1
fi

# Fall back to the NAS copy until the optional local cache is present.
if [[ ! -f "${tracks_path}" ]]; then
  tracks_path="${data_path}/cafe/gt_tracks.pkl"
fi

if [[ ! -d "${data_path}/cafe" || ! -f "${tracks_path}" ]]; then
  echo "Café data or gt_tracks.pkl not found under: ${data_path}" >&2
  exit 1
fi

if [[ ! -f "${groundtruth}" ]]; then
  echo "Evaluation ground truth not found: ${groundtruth}" >&2
  echo "Set GROUNDTRUTH to the official evaluation/gt_tracks.txt before training." >&2
  exit 1
fi

if [[ -n "${RESUME_PATH:-}" ]]; then
  if [[ ! -f "${RESUME_PATH}" ]]; then
    echo "Resume checkpoint not found: ${RESUME_PATH}" >&2
    exit 1
  fi
  resume_args=(--load_model --model_path "${RESUME_PATH}")
fi

if [[ "${use_mae}" == "1" ]]; then
  if [[ ! -f "${videomae_feats_path}/1_0.npy" ]]; then
    echo "VideoMAE-v2 feature cache is missing or incomplete: ${videomae_feats_path}" >&2
    exit 1
  fi
  mae_args=(--mae_version v2 --videomae_feats_path "${videomae_feats_path}")
elif [[ "${use_mae}" != "0" ]]; then
  echo "USE_MAE must be 0 or 1, got: ${use_mae}" >&2
  exit 1
fi

cd "${repo_root}"
NCCL_P2P_DISABLE="${nccl_p2p_disable}" NCCL_IB_DISABLE="${nccl_ib_disable}" \
CUDA_VISIBLE_DEVICES="${device}" "${python_bin}" -m torch.distributed.run \
  --nproc_per_node="${world_size}" \
  --master_port "${master_port}" \
  train.py \
  --data_path "${data_path}" \
  --tracks_path "${tracks_path}" \
  --groundtruth "${groundtruth}" \
  --split place \
  --backbone dinov2_vitb14 \
  --unfreeze_blocks 2 \
  --frozen_batch_norm \
  --batch 16 \
  --backbone_lr_scale 0.1 \
  --test_batch 4 \
  --num_frame 5 \
  "${mae_args[@]}" \
  --device "${device}" \
  --distributed \
  "${resume_args[@]}"
