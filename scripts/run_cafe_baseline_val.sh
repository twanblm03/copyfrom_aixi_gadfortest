#!/usr/bin/env bash
# Run the mask-free Cafe baseline on TRAIN -> VAL for one or more seeds.
#
# Experiment contract:
#   * --val_mode is always enabled (TRAIN is optimized; VAL is evaluated).
#   * Seeds run sequentially with identical arguments on the same GPU set.
#   * The split, seed, data paths, and GPU mapping are locked by this wrapper.
#   * Each seed gets a stable tag: baseline_val_<split>_seed<seed>.
#   * No mask, SAM, heatmap, or other proposed incremental module is enabled.
#   * train.py records the full resolved configuration in each run's args.json.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

split="${SPLIT:-place}"
seed_spec="${SEEDS:-${SEED:-1,2,3}}"
device="${DEVICE:-0,1,2,3}"
data_path="${DATA_PATH:-/share/share/aixi/Cafe_Dataset/Cafe_Dataset/Cafe_Dataset/Dataset}"
tracks_path="${TRACKS_PATH:-}"
groundtruth="${GROUNDTRUTH:-${repo_root}/evaluation/gt_tracks.txt}"
videomae_feats_path="${VIDEOMAE_FEATS_PATH:-/share/share/aixi/videomae_features_giant}"
checkpoint_root="${CHECKPOINT_ROOT:-}"
python_bin="${PYTHON_BIN:-python}"
resume_path="${RESUME_PATH:-}"
world_size="${NPROC_PER_NODE:-}"
master_port="${MASTER_PORT:-29520}"
use_mae="${USE_MAE:-1}"
dry_run=0
extra_train_args=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_cafe_baseline_val.sh [options] [-- TRAIN.PY_ARGS...]

Required experiment behavior:
  Runs the current mask-free Cafe model on TRAIN -> VAL.  --val_mode is
  always passed explicitly; this launcher never trains on VAL or evaluates
  TEST.  Multiple seeds are run sequentially and fail fast.

Options:
  --split {place|view}    Dataset split protocol (default: place)
  --seeds LIST            Comma- or space-separated seeds (default: 1,2,3)
  --seed SEED             Alias for a single seed
  --gpus LIST             CUDA device IDs, e.g. 0,1,2,3
  --data-path PATH        Dataset root containing cafe/
  --tracks-path PATH      gt_tracks.pkl (default: DATA_PATH/cafe/gt_tracks.pkl)
  --groundtruth PATH      Evaluation gt_tracks.txt
  --checkpoint-root PATH  Checkpoint root (default: .local_data/checkpoints/baseline_val/SPLIT)
  --python-bin PATH       Python interpreter (default: python)
  --resume-path PATH      Resume one seed from a checkpoint; restores history/selection
  --master-port PORT      First torchrun port; increments per seed (default: 29520)
  --nproc-per-node N      Must equal the number of IDs in --gpus
  --dry-run               Validate arguments and print commands without running
  -h, --help              Show this help

The following environment variables provide the same core overrides:
  SPLIT, SEEDS (or SEED), DEVICE, DATA_PATH, TRACKS_PATH, GROUNDTRUTH,
  CHECKPOINT_ROOT, PYTHON_BIN, RESUME_PATH, MASTER_PORT, and NPROC_PER_NODE.

Baseline hyperparameters can be overridden through environment variables:
  BATCH=16 TEST_BATCH=4 NUM_FRAME=8 EPOCHS=30 TEST_FREQ=1 LOG_INTERVAL=10
  USE_MAE=1 VIDEOMAE_FEATS_PATH=/path/to/cache

Arguments after -- are forwarded to train.py.  Protocol-defining arguments
(tag, split, seed, paths, devices, --val_mode/--final_test, and resume options)
are rejected there so all seeds remain paired and independent.

Examples:
  bash scripts/run_cafe_baseline_val.sh \
    --split place --seeds 1,2,3 --gpus 0,1,2,3 \
    --data-path /datasets/Cafe/Dataset

  USE_MAE=0 bash scripts/run_cafe_baseline_val.sh \
    --split view --seed 1 --gpus 0 -- --micro_batch 4

  bash scripts/run_cafe_baseline_val.sh --split place --seed 1 \
    --resume-path /path/to/epoch10.pth --gpus 0,1,2,3
EOF
}

die() {
  echo "Error: $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split)
      [[ $# -ge 2 ]] || die "--split requires a value"
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
    --data-path)
      [[ $# -ge 2 ]] || die "--data-path requires a value"
      data_path="$2"
      shift 2
      ;;
    --tracks-path)
      [[ $# -ge 2 ]] || die "--tracks-path requires a value"
      tracks_path="$2"
      shift 2
      ;;
    --groundtruth)
      [[ $# -ge 2 ]] || die "--groundtruth requires a value"
      groundtruth="$2"
      shift 2
      ;;
    --checkpoint-root)
      [[ $# -ge 2 ]] || die "--checkpoint-root requires a value"
      checkpoint_root="$2"
      shift 2
      ;;
    --python-bin)
      [[ $# -ge 2 ]] || die "--python-bin requires a value"
      python_bin="$2"
      shift 2
      ;;
    --resume-path)
      [[ $# -ge 2 ]] || die "--resume-path requires a value"
      resume_path="$2"
      shift 2
      ;;
    --master-port)
      [[ $# -ge 2 ]] || die "--master-port requires a value"
      master_port="$2"
      shift 2
      ;;
    --nproc-per-node)
      [[ $# -ge 2 ]] || die "--nproc-per-node requires a value"
      world_size="$2"
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
    --)
      shift
      extra_train_args=("$@")
      break
      ;;
    *)
      die "unknown launcher option '$1' (put train.py arguments after --)"
      ;;
  esac
done

[[ "${split}" == "place" || "${split}" == "view" ]] || \
  die "--split must be 'place' or 'view', got '${split}'"

# Normalize optional whitespace around the comma-separated CUDA IDs.
device="${device//[[:space:]]/}"
[[ "${device}" =~ ^[0-9]+(,[0-9]+)*$ ]] || \
  die "--gpus must be a comma-separated list of integer CUDA IDs, got '${device}'"
IFS=',' read -r -a device_list <<< "${device}"
if [[ -z "${world_size}" ]]; then
  world_size="${#device_list[@]}"
fi
[[ "${world_size}" =~ ^[1-9][0-9]*$ ]] || \
  die "--nproc-per-node must be a positive integer, got '${world_size}'"
[[ "${world_size}" -eq "${#device_list[@]}" ]] || \
  die "NPROC_PER_NODE (${world_size}) must match the number of GPUs (${#device_list[@]})"

[[ "${master_port}" =~ ^[0-9]+$ ]] || die "--master-port must be an integer"
[[ "${master_port}" -ge 1024 && "${master_port}" -le 65535 ]] || \
  die "--master-port must be between 1024 and 65535"

# Accept either "1,2,3" or "1 2 3", then reject empty/duplicate seeds.
read -r -a seed_list <<< "${seed_spec//,/ }"
[[ "${#seed_list[@]}" -gt 0 ]] || die "at least one seed is required"
declare -A seen_seeds=()
for seed in "${seed_list[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || die "seeds must be non-negative integers, got '${seed}'"
  [[ -z "${seen_seeds[${seed}]:-}" ]] || die "duplicate seed '${seed}'"
  seen_seeds["${seed}"]=1
done
if [[ -n "${resume_path}" ]]; then
  [[ "${#seed_list[@]}" -eq 1 ]] || \
    die "--resume-path is only valid with exactly one seed"
  [[ -f "${resume_path}" ]] || die "resume checkpoint not found: ${resume_path}"
  resume_dir="$(cd "$(dirname "${resume_path}")" && pwd)"
  resume_path="${resume_dir}/$(basename "${resume_path}")"
fi
last_port=$((master_port + ${#seed_list[@]} - 1))
[[ "${last_port}" -le 65535 ]] || die "per-seed master port would exceed 65535"

[[ "${use_mae}" == "0" || "${use_mae}" == "1" ]] || \
  die "USE_MAE must be 0 or 1, got '${use_mae}'"

if [[ -z "${tracks_path}" ]]; then
  tracks_path="${data_path}/cafe/gt_tracks.pkl"
fi
if [[ -z "${checkpoint_root}" ]]; then
  checkpoint_root="${repo_root}/.local_data/checkpoints/baseline_val/${split}"
fi

# These values define data access, model selection, or run identity.  Keeping
# them out of passthrough args prevents a duplicate argparse option from
# silently changing only part of a paired multi-seed experiment.
for arg in "${extra_train_args[@]}"; do
  option_name="${arg%%=*}"
  case "${option_name}" in
    --dataset|--val_mode|--final_test|--split|--random_seed|--device|--distributed|--local_rank|\
    --data_path|--tracks_path|--groundtruth|--checkpoint_root|--experiment_tag|\
    --resume-path|--resume_path|--load_model|--model_path)
      die "'${option_name}' is controlled by this launcher, not by arguments after --"
      ;;
  esac
done

if [[ "${dry_run}" -eq 0 ]]; then
  command -v "${python_bin}" >/dev/null 2>&1 || \
    die "Python executable not found: ${python_bin}"
  [[ -d "${data_path}/cafe" ]] || die "Cafe dataset directory not found: ${data_path}/cafe"
  [[ -f "${tracks_path}" ]] || die "track file not found: ${tracks_path}"
  [[ -f "${groundtruth}" ]] || die "evaluation ground truth not found: ${groundtruth}"
  if [[ "${use_mae}" == "1" ]]; then
    [[ -f "${videomae_feats_path}/1_0.npy" ]] || \
      die "VideoMAE-v2 cache is missing or incomplete: ${videomae_feats_path}"
  fi
  mkdir -p "${checkpoint_root}"
  [[ -w "${checkpoint_root}" ]] || die "checkpoint root is not writable: ${checkpoint_root}"
fi

# Prefer the active interpreter's C++ runtime, matching the existing launchers.
if [[ "${dry_run}" -eq 0 ]]; then
  python_bin_path="$(command -v "${python_bin}")"
  python_root="$(cd "$(dirname "${python_bin_path}")/.." && pwd)"
  if [[ -f "${python_root}/lib/libstdc++.so.6" ]]; then
    export LD_LIBRARY_PATH="${python_root}/lib:${LD_LIBRARY_PATH:-}"
  fi
fi
if [[ -d "${repo_root}/.local_deps/xformers_016" ]]; then
  export PYTHONPATH="${repo_root}/.local_deps/xformers_016${PYTHONPATH:+:${PYTHONPATH}}"
fi

mae_args=(--no_mae)
if [[ "${use_mae}" == "1" ]]; then
  mae_args=(--mae_version v2 --videomae_feats_path "${videomae_feats_path}")
fi

resume_args=()
if [[ -n "${resume_path}" ]]; then
  resume_args=(--load_model --model_path "${resume_path}")
fi

baseline_args=(
  --backbone dinov2_vitb14
  --unfreeze_blocks 2
  --frozen_batch_norm
  --backbone_lr_scale 0.1
  --batch "${BATCH:-16}"
  --test_batch "${TEST_BATCH:-4}"
  --num_frame "${NUM_FRAME:-8}"
  --optimizer adamw
  --weight_decay 1e-4
  --lr_schedule cyclic
  --lr_step 4
  --lr_step_down 25
  --epochs "${EPOCHS:-30}"
  --test_freq "${TEST_FREQ:-1}"
  --log_interval "${LOG_INTERVAL:-10}"
  "${mae_args[@]}"
)

echo "Cafe mask-free validation baseline"
echo "  protocol: TRAIN -> VAL (--val_mode); TEST is untouched"
echo "  split: ${split}"
echo "  seeds: ${seed_list[*]}"
echo "  GPUs: ${device} (${world_size} process(es))"
echo "  data: ${data_path}"
echo "  tracks: ${tracks_path}"
echo "  checkpoints: ${checkpoint_root}"
echo "  VideoMAE-v2: $([[ "${use_mae}" == "1" ]] && echo enabled || echo disabled)"
if [[ -n "${resume_path}" ]]; then
  echo "  resume: ${resume_path} (history and selection state are restored)"
fi

cd "${repo_root}"
for index in "${!seed_list[@]}"; do
  seed="${seed_list[${index}]}"
  seed_port=$((master_port + index))
  experiment_tag="baseline_val_${split}_seed${seed}"
  command_args=(
    "${python_bin}" -m torch.distributed.run
    --nproc_per_node="${world_size}"
    --master_port "${seed_port}"
    train.py
    "${baseline_args[@]}"
    "${resume_args[@]}"
    "${extra_train_args[@]}"
    --dataset cafe
    --val_mode
    --split "${split}"
    --random_seed "${seed}"
    --experiment_tag "${experiment_tag}"
    --data_path "${data_path}"
    --tracks_path "${tracks_path}"
    --groundtruth "${groundtruth}"
    --checkpoint_root "${checkpoint_root}"
    --device "${device}"
    --distributed
  )

  echo
  echo "[$((index + 1))/${#seed_list[@]}] seed=${seed}, tag=${experiment_tag}, master_port=${seed_port}"
  if [[ "${dry_run}" -eq 1 ]]; then
    printf 'NCCL_P2P_DISABLE=%q NCCL_IB_DISABLE=%q CUDA_VISIBLE_DEVICES=%q ' \
      "${NCCL_P2P_DISABLE:-1}" "${NCCL_IB_DISABLE:-1}" "${device}"
    printf '%q ' "${command_args[@]}"
    printf '\n'
    continue
  fi

  NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
  NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
  CUDA_VISIBLE_DEVICES="${device}" \
    "${command_args[@]}"
done

echo
if [[ "${dry_run}" -eq 1 ]]; then
  echo "Dry run complete; no training was started."
else
  echo "All validation-baseline seeds completed."
fi
