#!/usr/bin/env bash
set -euo pipefail

# Paper configuration: fine-tune only the last two DINOv2 blocks with a global
# batch of 16.  Set USE_MAE=1 to train with verified VideoMAE-v2 Giant features.
# Override DATA_PATH, GROUNDTRUTH, DEVICE, and VIDEOMAE_FEATS_PATH as needed.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_path="${DATA_PATH:-/nas/mzy/dataset/Cafe_Dataset/Cafe_Dataset/Dataset}"
groundtruth="${GROUNDTRUTH:-/nas/mzy/dataset/Cafe_Dataset/Cafe_Dataset/evaluation/gt_tracks.txt}"
local_tracks="${repo_root}/.local_data/cafe/gt_tracks.pkl"
tracks_path="${TRACKS_PATH:-${local_tracks}}"
device="${DEVICE:-0,1,2,3}"
IFS=',' read -r -a device_list <<< "${device}"
world_size="${NPROC_PER_NODE:-${#device_list[@]}}"
master_port="${MASTER_PORT:-29500}"
python_bin="${PYTHON_BIN:-/data2/mzy/GAD/.miniconda3/envs/gad/bin/python}"
use_mae="${USE_MAE:-0}"
videomae_feats_path="${VIDEOMAE_FEATS_PATH:-${repo_root}/.local_data/videomae_v2_giant}"
# This machine's default NCCL P2P path stalls.  Use shared-memory transport;
# these remain overridable for machines with a healthy P2P fabric.
nccl_p2p_disable="${NCCL_P2P_DISABLE:-1}"
nccl_ib_disable="${NCCL_IB_DISABLE:-1}"
resume_args=()
mae_args=(--no_mae)

if [[ ! -x "${python_bin}" ]]; then
  echo "Python environment not found: ${python_bin}. Set PYTHON_BIN or activate gad." >&2
  exit 1
fi

if [[ "${world_size}" -ne "${#device_list[@]}" ]]; then
  echo "NPROC_PER_NODE (${world_size}) must match the number of GPUs in DEVICE (${device})." >&2
  exit 1
fi

# Fall back to the NAS copy until the optional local cache is present.
if [[ ! -f "${tracks_path}" ]]; then
  tracks_path="${data_path}/cafe/gt_tracks.pkl"
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
  --split view \
  --backbone dinov2_vitb14 \
  --unfreeze_blocks 2 \
  --frozen_batch_norm \
  --batch 16 \
  --backbone_lr_scale 0.1 \
  --test_batch 4 \
  --num_frame 8 \
  "${mae_args[@]}" \
  --device "${device}" \
  --distributed \
  "${resume_args[@]}"
