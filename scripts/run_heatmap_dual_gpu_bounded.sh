#!/usr/bin/env bash
# Fixed, bounded recovery workflow for this server. Never uses physical GPU0.
set -euo pipefail

workflow_root=''
workflow_python=.miniconda3/envs/gad/bin/python
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) workflow_root="$2"; shift 2 ;;
    --python) workflow_python="$2"; shift 2 ;;
    --help)
      printf 'Usage: bash scripts/run_heatmap_dual_gpu_bounded.sh --root NEW_OUTPUT_DIRECTORY [--python PYTHON]\n'
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ -z "$workflow_root" || -e "$workflow_root" ]]; then
  printf 'A new, non-existing --root is required; existing artifacts are never overwritten.\n' >&2
  exit 2
fi
if [[ ! -x "$workflow_python" || ! -f configs/heatmap/b0.json ]]; then
  printf 'Run from the project root with an existing Python environment.\n' >&2
  exit 2
fi
mkdir -p results/heatmap
exec 9>results/heatmap/dual_gpu_23.lock
if ! flock -n 9; then
  printf 'Another dual-GPU workflow holds this project lock.\n' >&2
  exit 2
fi
mkdir -p "$(dirname "$workflow_root")"
mkdir "$workflow_root"
exec > >(tee -a "$workflow_root/workflow.log") 2>&1
workflow_stage=preflight
workflow_finish() {
  workflow_rc=$?
  printf 'stage=%s\nexit_code=%s\nfinished_at=%s\n' \
    "$workflow_stage" "$workflow_rc" "$(date --iso-8601=seconds)" > "$workflow_root/workflow_status.txt"
  printf 'WORKFLOW_EXIT stage=%s exit_code=%s\n' "$workflow_stage" "$workflow_rc"
}
trap workflow_finish EXIT
run_recorded() {
  printf '$ '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-f13b1d06-5ee1-64b8-bc92-4e5c28d72e7c,GPU-bc83c5ed-4690-7c23-37a5-8f78022962fa
printf 'STARTED_AT=%s\nWORKFLOW_PID=%s\nCUDA_DEVICE_ORDER=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
  "$(date --iso-8601=seconds)" "$$" "$CUDA_DEVICE_ORDER" "$CUDA_VISIBLE_DEVICES"
cp -- scripts/run_heatmap_dual_gpu_bounded.sh "$workflow_root/workflow_script.sh"
cp -- scripts/run_heatmap_paired_pilot.sh "$workflow_root/paired_pilot_script.sh"
run_recorded nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
  --query-gpu=index,uuid,temperature.gpu,memory.used,utilization.gpu --format=csv
run_recorded timeout --kill-after=5s 30s "$workflow_python" -c \
  "import torch; torch.set_num_threads(2); torch.cuda.init(); assert torch.cuda.device_count()==2; m=torch.nn.DataParallel(torch.nn.Linear(4,4).cuda(),device_ids=[0,1]); x=torch.randn(16,4,device='cuda:0',requires_grad=True); m(x).square().mean().backward(); [torch.cuda.synchronize(i) for i in range(2)]; assert torch.isfinite(x.grad).all(); print('TWO_GPU_FORWARD_BACKWARD_OK',flush=True)"
run_recorded timeout --kill-after=10s 3m "$workflow_python" -m unittest discover \
  -s tests/heatmap -p test_contract.py -v 2>&1 | tee "$workflow_root/contract_tests.log"

workflow_stage=b0_overfit500
# Fresh common initialization, not a weights-only pseudo-resume of step200.
run_recorded timeout --signal=TERM --kill-after=30s 30m "$workflow_python" tools/run_heatmap_research.py \
  --stage overfit --config configs/heatmap/b0.json --steps 500 \
  --train-clips 16 --eval-clips 32 --workers 2 --cache-fixed-overfit --train-eval-every 100 \
  --device cuda:0 --parallel-gpus 0,1 --output "$workflow_root/b0_overfit500"

workflow_stage=paired_pilot500
# Same GPUs, worker policy, effective batch, seed and steps for all four variants.
# Stop on any failure. Total pilot wall bound is 120 minutes, not full training.
run_recorded timeout --signal=TERM --kill-after=30s 120m bash "$workflow_root/paired_pilot_script.sh" \
  --steps 500 --seed 1 --root "$workflow_root/paired_pilot500" \
  --gpus 0,1 --python "$workflow_python" --workers 4 --persistent-workers

workflow_stage=paired_validation
run_recorded "$workflow_python" tools/report_heatmap_research.py \
  --root "$workflow_root" --paired-root "$workflow_root/paired_pilot500" \
  --destination "$workflow_root/summary" --visualize
workflow_stage=completed
