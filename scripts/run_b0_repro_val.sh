#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
split="${SPLIT:-place}"
seed_spec="${SEEDS:-1,2,3}"
device="${DEVICE:-0,1,2,3}"
python_bin="${PYTHON_BIN:-python}"
data_path="${DATA_PATH:-/share/share/aixi/Cafe_Dataset/Cafe_Dataset/Cafe_Dataset/Dataset}"
tracks_path="${TRACKS_PATH:-${data_path}/cafe/gt_tracks.pkl}"
groundtruth="${GROUNDTRUTH:-/share/share/aixi/Cafe_Dataset/Cafe_Dataset/Cafe_Dataset/evaluation/gt_tracks.txt}"
run_root="${RUN_ROOT:-/mzy/cafe_runs/b0-repro}"
result_path="${RESULT_PATH:-${run_root}/results}"
checkpoint_root="${CHECKPOINT_ROOT:-${run_root}/checkpoints/${split}}"
dry_run=0

usage() {
  printf '%s\n' \
    'Usage: bash scripts/run_b0_repro_val.sh [options]' \
    '  --split {place|view}  Dataset split (default: place)' \
    '  --seeds LIST          Comma-separated seeds (default: 1,2,3)' \
    '  --gpus LIST           CUDA device IDs (default: 0,1,2,3)' \
    '  --dry-run             Print commands without running training'
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split)
      [[ $# -ge 2 ]] || die '--split requires a value'
      split="$2"
      shift 2
      ;;
    --seeds|--seed)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      seed_spec="$2"
      shift 2
      ;;
    --gpus|--device)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      device="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ "${split}" == 'place' || "${split}" == 'view' ]] || die 'split must be place or view'
device="${device//[[:space:]]/}"
[[ "${device}" =~ ^[0-9]+(,[0-9]+)*$ ]] || die 'gpus must be comma-separated integers'
read -r -a seeds <<< "${seed_spec//,/ }"
[[ ${#seeds[@]} -gt 0 ]] || die 'at least one seed is required'

declare -A seen=()
for seed in "${seeds[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || die "invalid seed: ${seed}"
  [[ -z "${seen[${seed}]:-}" ]] || die "duplicate seed: ${seed}"
  seen["${seed}"]=1
done

if [[ ${dry_run} -eq 0 ]]; then
  command -v "${python_bin}" >/dev/null 2>&1 || die "Python not found: ${python_bin}"
  [[ -d "${data_path}/cafe" ]] || die "dataset not found: ${data_path}/cafe"
  [[ -f "${tracks_path}" ]] || die "tracks not found: ${tracks_path}"
  [[ -f "${groundtruth}" ]] || die "ground truth not found: ${groundtruth}"
fi

printf 'Café B0-repro: TRAIN -> VAL\n'
printf '  split=%s seeds=%s GPUs=%s\n' "${split}" "${seeds[*]}" "${device}"
printf '  model=ResNet-18, frames=5, original GADTR\n'
printf '  optimizer=Adam, lr=1e-5->1e-4, warmup=5, linear decay, wd=0\n'

cd "${repo_root}"
for seed in "${seeds[@]}"; do
  tag="b0_repro_val_${split}_seed${seed}"
  command_args=(
    "${python_bin}" train.py
    --dataset cafe
    --val_mode
    --split "${split}"
    --data_path "${data_path}"
    --tracks_path "${tracks_path}"
    --groundtruth "${groundtruth}"
    --result_path "${result_path}"
    --checkpoint_root "${checkpoint_root}"
    --experiment_tag "${tag}"
    --random_seed "${seed}"
    --device "${device}"
    --image_width 1280
    --image_height 720
    --num_frame 5
    --num_class 6
    --num_boxes 14
    --backbone resnet18
    --hidden_dim 256
    --crop_size 5
    --gar_nheads 4
    --gar_enc_layers 6
    --gar_ffn_dim 512
    --position_embedding sine
    --num_group_tokens 12
    --group_threshold 0.5
    --distance_threshold 0.2
    --temperature 0.2
    --ce_loss_coef 1
    --eos_coef 1
    --group_eos_coef 1
    --group_ce_loss_coef 1
    --group_code_loss_coef 5
    --consistency_loss_coef 2
    --set_cost_group_class 1
    --set_cost_membership 1
    --batch 16
    --test_batch 16
    --epochs 30
    --test_freq 1
    --lr 1e-5
    --max_lr 1e-4
    --warmup_epochs 5
    --lr_schedule linear
    --weight_decay 0
    --drop_rate 0.1
    --selection_metric group_mAP_1.0
    --eval_type gt_base
    --eval_image_width 1280
    --eval_image_height 720
  )

  printf '\nseed=%s tag=%s\n' "${seed}" "${tag}"
  if [[ ${dry_run} -eq 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${device}"
    printf '%q ' "${command_args[@]}"
    printf '\n'
  else
    CUDA_VISIBLE_DEVICES="${device}" "${command_args[@]}"
  fi
done
