#!/usr/bin/env bash
set -euo pipefail

# A plan is printed by default. Only --run may start train.py/torchrun.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/data2/mzy/GAD/.miniconda3/envs/gad/bin/python}"
data_path="${DATA_PATH:-/nas/mzy/dataset/Cafe_Dataset/Cafe_Dataset/Dataset}"
tracks_path="${TRACKS_PATH:-/data2/mzy/GAD/CAFE_codebase/.local_data/cafe/gt_tracks.pkl}"
groundtruth="${GROUNDTRUTH:-/nas/mzy/dataset/Cafe_Dataset/Cafe_Dataset/evaluation/gt_tracks.txt}"
sam_repo="${SAM_REPO:-${repo_root}/.local_deps/segment-anything}"
sam_checkpoint="${SAM_CHECKPOINT:-${repo_root}/.local_data/sam/checkpoints/sam_vit_b_01ec64.pth}"
sam_checkpoint_sha256="${SAM_CHECKPOINT_SHA256:-}"
sam_encoder_chunk_size="${SAM_ENCODER_CHUNK_SIZE:-1}"
sam_output_height="${SAM_OUTPUT_HEIGHT:-23}"
sam_output_width="${SAM_OUTPUT_WIDTH:-40}"
device="${DEVICE:-${GPUS:-0,1,2,3}}"
split="${SPLIT:-place}"
distributed="${DISTRIBUTED:-0}"
parallel_transport="${PARALLEL_TRANSPORT:-host}"
master_port="${MASTER_PORT:-29623}"
epochs="${EPOCHS:-30}"
warmup_epochs="${WARMUP_EPOCHS:-5}"
test_freq="${TEST_FREQ:-1}"
batch="${BATCH:-16}"
test_batch="${TEST_BATCH:-4}"
max_train_steps="${MAX_TRAIN_STEPS:-0}"
clip_manifest="${CLIP_MANIFEST:-}"
resume_checkpoint="${RESUME_CHECKPOINT:-}"
fixed_center_train="${FIXED_CENTER_TRAIN:-0}"
seed="${SEED:-1}"
lr="${LR:-0.00001}"
max_lr="${MAX_LR:-0.0001}"
backbone_lr_scale="${BACKBONE_LR_SCALE:-1.0}"
annotation_workers="${ANNOTATION_WORKERS:-16}"
checkpoint_root="${CHECKPOINT_ROOT:-${repo_root}/.local_data/sam_encoder/checkpoints}"
launch_root="${LAUNCH_ROOT:-${repo_root}/.local_data/sam_encoder/launches}"
tag="${TAG:-}"
action="print"
final_test=0
best_on_test=0
keep_all_checkpoints=0

usage() {
  cat <<'EOF'
Usage: scripts/train_cafe_sam_encoder.sh [--dry-run | --run] [options]

Without --run this only prints paths, environment and the exact command.
Frozen SAM ViT-B replaces ResNet for BOTH actor and scene features.

  --device IDS                 Comma-separated physical GPU IDs (DEVICE)
  --split place|view           Dataset split (SPLIT)
  --final-test                 Train TRAIN+VAL; evaluate TEST at final epoch
  --best-on-test               With --final-test, evaluate/select on TEST each epoch
  --keep-all-checkpoints      Retain every epoch checkpoint (train flag: --keep_all_checkpoints)
  --save-every-epoch          Alias of --keep-all-checkpoints
  --distributed 0|1            DataParallel/single GPU or torchrun (default 0)
  --parallel-transport host|native
                               host requires --distributed 0 (default host)
  --epochs N                  Epoch limit (EPOCHS, default 30)
  --warmup-epochs N           Must be in [1, epochs] (default 5)
  --max-train-steps N          Optimizer steps PER epoch; 0=full (default 0)
  --batch N                   Global batch, including all DP GPUs (default 16)
  --test-batch N              Evaluation batch (default 4)
  --clip-manifest PATH         Optional bounded TRAIN/VAL clip selection
  --resume PATH                Resume only the same SAM architecture
  --tag NAME                   New experiment label; never defaults to B0
  --checkpoint-root PATH      Root for new experiment checkpoint directories
  --python PATH                GAD Python interpreter (PYTHON_BIN)
  --sam-repo PATH              Local official segment-anything source
  --sam-checkpoint PATH        Official SAM ViT-B weights
  --sam-checkpoint-sha256 HEX  Optional expected hash, verified before launch
  --sam-encoder-chunk-size N  Frozen encoder frames per forward (default 1)
  --sam-output-height N       GAD feature grid height (default 23)
  --sam-output-width N        GAD feature grid width (default 40)
  -h, --help                   Show launcher help

Paths can also be set with DATA_PATH, TRACKS_PATH, GROUNDTRUTH, SAM_REPO,
SAM_CHECKPOINT, CHECKPOINT_ROOT, LAUNCH_ROOT. Additional settings: SEED,
WARMUP_EPOCHS, LR, MAX_LR, TEST_FREQ, ANNOTATION_WORKERS, MASTER_PORT,
BACKBONE_LR_SCALE (default 1.0), FIXED_CENTER_TRAIN=0|1, OMP_NUM_THREADS,
MKL_NUM_THREADS, TORCH_HOME. TEST requires explicit --final-test.
Every run already retains its latest/final epoch checkpoint. No mask cache,
heatmap branch, VideoMAE or fusion is enabled.
EOF
}

need_value() {
  [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
}
while (($#)); do
  case "$1" in
    --run|--dry-run)
      next_action="print"; [[ "$1" == --run ]] && next_action="run"
      [[ "${action}" == print || "${action}" == "${next_action}" ]] || {
        echo '--run and --dry-run cannot be combined' >&2; exit 2;
      }
      if [[ "$1" == --dry-run && "${action}" == run ]]; then
        echo '--run and --dry-run cannot be combined' >&2; exit 2
      fi
      # Track explicit dry-run separately so order cannot enable training.
      if [[ "$1" == --run && "${explicit_dry_run:-0}" == 1 ]]; then
        echo '--run and --dry-run cannot be combined' >&2; exit 2
      fi
      [[ "$1" != --dry-run ]] || explicit_dry_run=1
      action="${next_action}"; shift ;;
    -h|--help) usage; exit 0 ;;
    --final-test) final_test=1; shift ;;
    --best-on-test) best_on_test=1; shift ;;
    --keep-all-checkpoints|--save-every-epoch) keep_all_checkpoints=1; shift ;;
    --device|--split|--distributed|--parallel-transport|--epochs|--warmup-epochs|--max-train-steps|--batch|--test-batch|--clip-manifest|--resume|--tag|--checkpoint-root|--python|--sam-repo|--sam-checkpoint|--sam-checkpoint-sha256|--sam-encoder-chunk-size|--sam-output-height|--sam-output-width)
      need_value "$@"
      option="$1"; value="$2"; shift 2
      case "${option}" in
        --device) device="${value}" ;; --split) split="${value}" ;;
        --distributed) distributed="${value}" ;; --parallel-transport) parallel_transport="${value}" ;;
        --epochs) epochs="${value}" ;; --max-train-steps) max_train_steps="${value}" ;;
        --warmup-epochs) warmup_epochs="${value}" ;;
        --batch) batch="${value}" ;; --test-batch) test_batch="${value}" ;;
        --clip-manifest) clip_manifest="${value}" ;; --resume) resume_checkpoint="${value}" ;;
        --tag) tag="${value}" ;; --checkpoint-root) checkpoint_root="${value}" ;;
        --python) python_bin="${value}" ;; --sam-repo) sam_repo="${value}" ;;
        --sam-checkpoint) sam_checkpoint="${value}" ;;
        --sam-checkpoint-sha256) sam_checkpoint_sha256="${value}" ;;
        --sam-encoder-chunk-size) sam_encoder_chunk_size="${value}" ;;
        --sam-output-height) sam_output_height="${value}" ;; --sam-output-width) sam_output_width="${value}" ;;
      esac ;;
    *) echo "Unknown launcher argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "${split}" == place || "${split}" == view ]] || { echo 'SPLIT must be place or view' >&2; exit 2; }
[[ "${distributed}" == 0 || "${distributed}" == 1 ]] || { echo 'DISTRIBUTED must be 0 or 1' >&2; exit 2; }
[[ "${parallel_transport}" == host || "${parallel_transport}" == native ]] || {
  echo 'PARALLEL_TRANSPORT must be host or native' >&2; exit 2;
}
[[ "${distributed}" == 0 || "${parallel_transport}" == native ]] || {
  echo 'DDP requires PARALLEL_TRANSPORT=native; host is DataParallel only' >&2; exit 2;
}
[[ "${fixed_center_train}" == 0 || "${fixed_center_train}" == 1 ]] || {
  echo 'FIXED_CENTER_TRAIN must be 0 or 1' >&2; exit 2;
}
device="${device//[[:space:]]/}"
[[ "${device}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'DEVICE must be comma-separated GPU IDs' >&2; exit 2; }
IFS=',' read -r -a device_list <<< "${device}"
for numeric_name in epochs warmup_epochs test_freq batch test_batch sam_encoder_chunk_size sam_output_height sam_output_width; do
  numeric_value="${!numeric_name}"
  [[ "${numeric_value}" =~ ^[1-9][0-9]*$ ]] || { echo "${numeric_name} must be a positive integer" >&2; exit 2; }
done
for numeric_name in max_train_steps seed annotation_workers; do
  [[ "${!numeric_name}" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "${numeric_name} must be a non-negative integer" >&2; exit 2; }
done
((warmup_epochs <= epochs)) || { echo 'WARMUP_EPOCHS must be in [1, EPOCHS]; set --warmup-epochs for a bounded pilot' >&2; exit 2; }
if [[ "${best_on_test}" == 1 && ( "${final_test}" != 1 || "${test_freq}" != 1 ) ]]; then
  echo '--best-on-test requires --final-test and TEST_FREQ=1' >&2; exit 2
fi
if [[ "${final_test}" == 1 && -n "${clip_manifest}" ]]; then
  echo '--clip-manifest is supported only for TRAIN->VAL diagnostics' >&2; exit 2
fi
if [[ -n "${sam_checkpoint_sha256}" && ! "${sam_checkpoint_sha256}" =~ ^[0-9A-Fa-f]{64}$ ]]; then
  echo 'SAM_CHECKPOINT_SHA256 must contain 64 hexadecimal characters' >&2; exit 2
fi
protocol="development"
protocol_args=(--val_mode)
if [[ "${final_test}" == 1 ]]; then
  protocol="final_test"
  protocol_args=(--final_test)
fi
if [[ "${best_on_test}" == 1 ]]; then
  protocol="best_on_test"
  protocol_args+=(--best_on_test)
fi
if [[ -z "${tag}" ]]; then
  tag="sam_encoder_all_${split}_seed${seed}"
  [[ "${protocol}" == development ]] || tag="sam_encoder_all_${split}_${protocol}_seed${seed}"
elif [[ "${protocol}" != development && "${tag}" != *"${protocol}"* ]]; then
  tag="${tag}_${protocol}"
fi
[[ "${tag}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$ ]] || { echo 'Invalid experiment tag' >&2; exit 2; }

cd "${repo_root}"
export TORCH_HOME="${TORCH_HOME:-${repo_root}/.local_data/torch}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export CUDA_VISIBLE_DEVICES="${device}"

optional_args=()
[[ -z "${clip_manifest}" ]] || optional_args+=(--clip_manifest "${clip_manifest}")
[[ -z "${resume_checkpoint}" ]] || optional_args+=(--load_model --model_path "${resume_checkpoint}")
[[ "${fixed_center_train}" == 0 ]] || optional_args+=(--fixed_center_train)
[[ "${keep_all_checkpoints}" == 0 ]] || optional_args+=(--keep_all_checkpoints)
[[ -z "${sam_checkpoint_sha256}" ]] || optional_args+=(--sam_checkpoint_sha256 "${sam_checkpoint_sha256}")
train_args=(train.py
  --data_path "${data_path}" --tracks_path "${tracks_path}" --groundtruth "${groundtruth}"
  --split "${split}" "${protocol_args[@]}" --research_b0_repro --no_mae
  --backbone sam_vit_b --freeze_backbone --heatmap_variant off --train_mask_mode disabled
  --sam_repo "${sam_repo}" --sam_checkpoint "${sam_checkpoint}"
  --sam_encoder_chunk_size "${sam_encoder_chunk_size}"
  --sam_output_height "${sam_output_height}" --sam_output_width "${sam_output_width}"
  --image_width 1280 --image_height 720 --num_frame 5 --num_boxes 14 --crop_size 5
  --hidden_dim 256 --num_group_tokens 12 --batch "${batch}" --test_batch "${test_batch}"
  --epochs "${epochs}" --test_freq "${test_freq}" --max_train_steps "${max_train_steps}"
  --optimizer adam --lr "${lr}" --max_lr "${max_lr}" --warmup_epochs "${warmup_epochs}"
  --lr_schedule linear --weight_decay 0 --backbone_lr_scale "${backbone_lr_scale}"
  --selection_metric group_mAP_1.0 --random_seed "${seed}"
  --annotation_workers "${annotation_workers}" --checkpoint_root "${checkpoint_root}"
  --experiment_tag "${tag}" --device "${device}" --parallel_transport "${parallel_transport}"
  --log_interval 10 --report_peak_memory "${optional_args[@]}")
command=("${python_bin}")
if [[ "${distributed}" == 1 ]]; then
  command+=(-m torch.distributed.run "--nproc_per_node=${#device_list[@]}" --master_port "${master_port}")
  train_args+=(--distributed)
fi
command+=("${train_args[@]}")

print_plan() {
  printf 'Frozen SAM-only backbone; actor + scene use %sx%s SAM features.\n' "${sam_output_height}" "${sam_output_width}"
  printf 'Protocol: %s\nCheckpoint retention: %s (latest/final epoch always retained)\n' \
    "${protocol}" "$([[ "${keep_all_checkpoints}" == 1 ]] && printf 'all epochs' || printf 'latest + selected best')"
  printf 'Repository: %s\nData: %s\nTracks: %s\nEvaluation GT: %s\n' "${repo_root}" "${data_path}" "${tracks_path}" "${groundtruth}"
  printf 'SAM source: %s\nSAM weights: %s\nExpected SHA256: %s\n' "${sam_repo}" "${sam_checkpoint}" "${sam_checkpoint_sha256:-computed on --run}"
  printf 'Checkpoints: %s\nLaunch records: %s\nTag: %s\n' "${checkpoint_root}" "${launch_root}" "${tag}"
  printf 'Clip manifest: %s\nResume: %s\n' "${clip_manifest:-full protocol splits}" "${resume_checkpoint:-new run}"
  printf 'Command:\n  cd %q\n  ' "${repo_root}"
  printf '%q ' env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" "TORCH_HOME=${TORCH_HOME}" \
    "OMP_NUM_THREADS=${OMP_NUM_THREADS}" "MKL_NUM_THREADS=${MKL_NUM_THREADS}" \
    "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}" "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}" "${command[@]}"
  printf '\n'
}

if [[ "${action}" != run ]]; then
  print_plan
  printf 'Dry run only: no files were written and no training process was started. Use --run to execute.\n'
  exit 0
fi

command -v "${python_bin}" >/dev/null 2>&1 || { echo "Python executable not found: ${python_bin}" >&2; exit 2; }
for required_dir in "${data_path}/cafe" "${sam_repo}/segment_anything"; do
  [[ -d "${required_dir}" ]] || { echo "Missing required directory: ${required_dir}" >&2; exit 2; }
done
for required_file in "${tracks_path}" "${groundtruth}" "${sam_checkpoint}"; do
  [[ -f "${required_file}" ]] || { echo "Missing required file: ${required_file}" >&2; exit 2; }
done
for optional_file in "${clip_manifest}" "${resume_checkpoint}"; do
  [[ -z "${optional_file}" || -f "${optional_file}" ]] || { echo "Missing requested file: ${optional_file}" >&2; exit 2; }
done

# CPU-only provenance. No model import, CUDA query, training or queue operation.
actual_sam_sha="$("${python_bin}" - "${sam_checkpoint}" "${sam_checkpoint_sha256}" <<'PY'
import hashlib
import sys
digest = hashlib.sha256()
with open(sys.argv[1], 'rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
actual = digest.hexdigest()
if sys.argv[2] and actual != sys.argv[2].lower():
    raise SystemExit('SAM checkpoint SHA256 mismatch: ' + actual)
print(actual)
PY
)"
if [[ -z "${sam_checkpoint_sha256}" ]]; then
  command+=(--sam_checkpoint_sha256 "${actual_sam_sha}")
fi
sam_checkpoint_sha256="${actual_sam_sha}"
launch_dir="$("${python_bin}" - "${repo_root}" "${launch_root}" "${tag}" "${data_path}" \
  "${tracks_path}" "${groundtruth}" "${sam_repo}" "${sam_checkpoint}" "${actual_sam_sha}" \
  "${clip_manifest}" "${resume_checkpoint}" "${command[@]}" <<'PY'
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile

(repo, launch_root, tag, data_path, tracks, groundtruth, sam_repo,
 sam_checkpoint, sam_sha, clip_manifest, resume) = sys.argv[1:12]
command = sys.argv[12:]
stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
root = Path(repo)

def file_sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

input_files = {'tracks': tracks, 'groundtruth': groundtruth}
if clip_manifest:
    input_files['clip_manifest'] = clip_manifest
if resume:
    input_files['resume_checkpoint'] = resume
sources = sorted(root.glob('*.py')) + [root / 'scripts/train_cafe_sam_encoder.sh']
for folder in ('models', 'dataloader', 'evaluation', 'util'):
    sources.extend(sorted((root / folder).rglob('*.py')))
sources.extend(sorted((Path(sam_repo) / 'segment_anything').rglob('*.py')))
environment = {name: os.environ.get(name, '') for name in (
    'CUDA_VISIBLE_DEVICES', 'TORCH_HOME', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'NCCL_P2P_DISABLE', 'NCCL_IB_DISABLE')}
def source_key(path):
    # Path.is_relative_to is unavailable in the project's Python 3.8.
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)

record = {
    'created_at_asia_shanghai': stamp, 'repo_root': repo, 'experiment_tag': tag,
    'protocol': ('best_on_test' if '--best_on_test' in command else
                 'final_test' if '--final_test' in command else 'development'),
    'checkpoint_policy': {'keep_all_checkpoints': '--keep_all_checkpoints' in command,
                          'save_last_epoch': True},
    'backbone': 'sam_vit_b', 'sam_frozen': True,
    'actor_and_scene_from_sam': True, 'data_path': str(Path(data_path).resolve()),
    'sam_repo': str(Path(sam_repo).resolve()),
    'sam_checkpoint': str(Path(sam_checkpoint).resolve()), 'sam_checkpoint_sha256': sam_sha,
    'command': command, 'environment': environment,
    'inputs': {key: {'path': str(Path(value).resolve()), 'sha256': file_sha(value)}
               for key, value in input_files.items()},
    'source_sha256': {source_key(path): file_sha(path) for path in sources if path.is_file()},
    'full_parser_args': 'Saved by train.py in result/<experiment>/args.json after initialization',
}
Path(launch_root).mkdir(parents=True, exist_ok=True)
folder = Path(tempfile.mkdtemp(prefix=tag + '_', dir=launch_root))
(folder / 'launch.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
run_text = ('#!/usr/bin/env bash\nset -euo pipefail\ncd ' + shlex.quote(repo) + '\n' +
            'exec env ' + ' '.join(shlex.quote(key + '=' + value) for key, value in environment.items()) +
            ' ' + ' '.join(shlex.quote(arg) for arg in command) + '\n')
(folder / 'command.sh').write_text(run_text)
print(folder)
PY
)"
print_plan
printf 'Launch provenance saved to %s/launch.json\n' "${launch_dir}"
exec "${command[@]}"
