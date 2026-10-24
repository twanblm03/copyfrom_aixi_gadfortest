#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/data2/mzy/GAD/.miniconda3/envs/gad/bin/python}"
data_path="${DATA_PATH:-/nas/mzy/dataset/Cafe_Dataset/Cafe_Dataset/Dataset}"
tracks_path="${TRACKS_PATH:-/data2/mzy/GAD/CAFE_codebase/.local_data/cafe/gt_tracks.pkl}"
groundtruth="${GROUNDTRUTH:-/nas/mzy/dataset/Cafe_Dataset/Cafe_Dataset/evaluation/gt_tracks.txt}"
manifest="${SAM_MANIFEST:-${repo_root}/.local_data/sam/full_manifest.jsonl}"
cache_root="${SAM_CACHE_ROOT:-${repo_root}/.local_data/sam/cache}"
cache_version="${SAM_CACHE_VERSION:-raw_bc294f454bfdfad21798}"
checkpoint_root="${CHECKPOINT_ROOT:-${repo_root}/.local_data/sam/full_checkpoints}"
gpus="${GPUS:-0,1,2,3}"
split="${SPLIT:-place}"
mode="${MODE:-sam_roi_blend}"
heatmap_variant="${HEATMAP_VARIANT:-off}"
heatmap_sigma_ratio="${HEATMAP_SIGMA_RATIO:-0.25}"
projected_cache="${PROJECTED_CACHE:-1}"
context_fraction="${CONTEXT_FRACTION:-0.5}"
seed="${SEED:-1}"
backbone_lr_scale="${BACKBONE_LR_SCALE:-0.1}"
tag="${TAG:-sam_person_blend50_full_seed1}"
master_port="${MASTER_PORT:-29618}"
distributed="${DISTRIBUTED:-1}"
parallel_transport="${PARALLEL_TRANSPORT:-native}"
epochs="${EPOCHS:-30}"
warmup_epochs="${WARMUP_EPOCHS:-5}"
test_freq="${TEST_FREQ:-1}"
batch="${BATCH:-16}"
test_batch="${TEST_BATCH:-4}"
max_train_steps="${MAX_TRAIN_STEPS:-0}"
clip_manifest="${CLIP_MANIFEST:-}"
resume_checkpoint="${RESUME_CHECKPOINT:-}"
gradient_clipping="${GRADIENT_CLIPPING:-0}"
gradient_clipping_start_epoch="${GRADIENT_CLIPPING_START_EPOCH:-1}"
fixed_center_train="${FIXED_CENTER_TRAIN:-1}"

if [[ "${split}" != "place" && "${split}" != "view" ]]; then
  echo "SPLIT must be place or view" >&2
  exit 2
fi
if [[ "${mode}" != "disabled" && "${mode}" != "sam_roi" && "${mode}" != "sam_roi_blend" ]]; then
  echo "MODE must be disabled, sam_roi, or sam_roi_blend" >&2
  exit 2
fi
if [[ "${heatmap_variant}" != "off" && "${heatmap_variant}" != "h1_sup" ]]; then
  echo "HEATMAP_VARIANT must be off or h1_sup for this full-run launcher" >&2
  exit 2
fi
if [[ "${mode}" != "disabled" && "${heatmap_variant}" != "off" ]]; then
  echo "SAM and heatmap modes cannot be combined in this launcher" >&2
  exit 2
fi
if [[ "${distributed}" != "0" && "${distributed}" != "1" ]]; then
  echo "DISTRIBUTED must be 0 or 1" >&2
  exit 2
fi
if [[ "${parallel_transport}" != "native" && "${parallel_transport}" != "host" ]]; then
  echo "PARALLEL_TRANSPORT must be native or host" >&2
  exit 2
fi
if [[ "${distributed}" == "1" && "${parallel_transport}" != "native" ]]; then
  echo "Host transport requires DataParallel (DISTRIBUTED=0)" >&2
  exit 2
fi
if [[ "${fixed_center_train}" != "0" && "${fixed_center_train}" != "1" ]]; then
  echo "FIXED_CENTER_TRAIN must be 0 or 1" >&2
  exit 2
fi
IFS=',' read -r -a gpu_list <<< "${gpus}"
if [[ "${#gpu_list[@]}" -ne 4 ]]; then
  echo "This launcher requires exactly four GPU IDs" >&2
  exit 2
fi
for required in "${python_bin}" "${data_path}/cafe" "${tracks_path}" "${groundtruth}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required input: ${required}" >&2
    exit 2
  fi
done

mask_args=()
clip_args=()
resume_args=()
gradient_args=()
frame_args=()
if [[ "${fixed_center_train}" == "1" ]]; then
  frame_args=(--fixed_center_train)
fi
if [[ -n "${clip_manifest}" ]]; then
  [[ -f "${clip_manifest}" ]] || { echo "Missing clip manifest: ${clip_manifest}" >&2; exit 2; }
  clip_args=(--clip_manifest "${clip_manifest}")
fi
if [[ -n "${resume_checkpoint}" ]]; then
  [[ -f "${resume_checkpoint}" ]] || { echo "Missing resume checkpoint: ${resume_checkpoint}" >&2; exit 2; }
  resume_args=(--load_model --model_path "${resume_checkpoint}")
fi
if [[ "${gradient_clipping}" == "1" ]]; then
  gradient_args=(--gradient_clipping --max_norm 1.0
                 --gradient_clipping_start_epoch "${gradient_clipping_start_epoch}")
elif [[ "${gradient_clipping}" != "0" ]]; then
  echo "GRADIENT_CLIPPING must be 0 or 1" >&2
  exit 2
fi
if [[ "${mode}" != "disabled" ]]; then
  [[ -f "${manifest}" ]] || { echo "Missing SAM manifest: ${manifest}" >&2; exit 2; }
  [[ -d "${cache_root}/${cache_version}" ]] || { echo "Missing SAM cache: ${cache_root}/${cache_version}" >&2; exit 2; }
  mask_args=(--sam_mask_manifest "${manifest}"
             --sam_cache_root "${cache_root}"
             --sam_cache_version "${cache_version}"
             --sam_feature_height 23 --sam_feature_width 40)
  if [[ "${projected_cache}" == "1" ]]; then
    mask_args+=(--sam_projected_cache)
  fi
fi

cd "${repo_root}"
export TORCH_HOME="${TORCH_HOME:-${repo_root}/.local_data/torch}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export CUDA_VISIBLE_DEVICES="${gpus}"

train_args=(train.py \
  --data_path "${data_path}" --tracks_path "${tracks_path}" \
  --groundtruth "${groundtruth}" --split "${split}" --val_mode \
  --research_b0_repro --heatmap_variant "${heatmap_variant}" \
  --heatmap_sigma_ratio "${heatmap_sigma_ratio}" --no_mae \
  --backbone resnet18 --image_width 1280 --image_height 720 \
  --num_frame 5 --batch "${batch}" --test_batch "${test_batch}" \
  --epochs "${epochs}" --test_freq "${test_freq}" \
  --max_train_steps "${max_train_steps}" \
  --optimizer adam --lr 0.00001 --max_lr 0.0001 --warmup_epochs "${warmup_epochs}" \
  --lr_schedule linear --weight_decay 0 --backbone_lr_scale "${backbone_lr_scale}" \
  --selection_metric group_mAP_1.0 --random_seed "${seed}" \
  "${frame_args[@]}" --train_mask_mode "${mode}" \
  --annotation_workers 16 \
  --mask_context_fraction "${context_fraction}" \
  --checkpoint_root "${checkpoint_root}" --experiment_tag "${tag}" \
  --device "${gpus}" --parallel_transport "${parallel_transport}" \
  --log_interval 10 --report_peak_memory \
  "${mask_args[@]}" "${clip_args[@]}" "${resume_args[@]}" "${gradient_args[@]}")
if [[ "${distributed}" == "1" ]]; then
  exec "${python_bin}" -m torch.distributed.run \
    --nproc_per_node=4 --master_port "${master_port}" "${train_args[@]}" --distributed
else
  exec "${python_bin}" "${train_args[@]}"
fi
