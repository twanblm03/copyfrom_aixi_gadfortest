#!/usr/bin/env bash
# B only: contracts, 3 optimizer updates, then 100 fixed-TRAIN updates. NO full stage.
set -euo pipefail
crowd_repo=''
crowd_root=''
crowd_predecessor=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) crowd_repo="$2"; shift 2 ;;
    --output-root) crowd_root="$2"; shift 2 ;;
    --after-official-view) crowd_predecessor="$2"; shift 2 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ "$crowd_repo" != /* || "$crowd_root" != /* || "$crowd_predecessor" != /* ]]; then
  printf 'Explicit absolute repository, preparation root and predecessor paths required.\n' >&2; exit 2
fi
crowd_python="$crowd_repo/.miniconda3/envs/gad/bin/python"
crowd_code="$crowd_root/audit/source_snapshot"
if [[ ! -x "$crowd_python" || ! -f "$crowd_root/geometry_train64/geometry_summary.json" || -e "$crowd_root/smoke3_b" || -e "$crowd_root/overfit100_b" ]]; then
  printf 'Prepared frozen source/geometry and unused bounded output paths required.\n' >&2; exit 2
fi
exec > >(tee -a "$crowd_root/bounded_workflow.log") 2>&1
export TZ=Asia/Shanghai PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-f13b1d06-5ee1-64b8-bc92-4e5c28d72e7c,GPU-bc83c5ed-4690-7c23-37a5-8f78022962fa
crowd_stage=waiting_for_predecessor
crowd_finish() {
  crowd_rc=$?
  printf 'stage=%s\nexit_code=%s\nfinished_at=%s\n' "$crowd_stage" "$crowd_rc" "$(date --iso-8601=seconds)" > "$crowd_root/bounded_workflow_status.txt"
  printf 'BOUNDED_EXIT stage=%s exit_code=%s; NO_FULL_TRAINING\n' "$crowd_stage" "$crowd_rc"
}
trap crowd_finish EXIT
crowd_run() { printf '$ '; printf '%q ' "$@"; printf '\n'; "$@"; }
printf 'BOUNDED_ONLY_START=%s\nWORKFLOW_PID=%s\nWAIT_FOR=%s\nNO_A_NO_FULL_NO_SEED2_3\n' "$(date --iso-8601=seconds)" "$$" "$crowd_predecessor"
exec 9>"$crowd_repo/results/heatmap/dual_gpu_23.lock"
# Wait in the persistent foreground session; never contend with ongoing TEST.
if ! flock -w 7200 9; then printf 'Predecessor wait exceeded 2 hours; no CUDA work launched.\n' >&2; exit 2; fi
crowd_stage=preflight
cd "$crowd_code"
crowd_run "$crowd_python" - "$crowd_root" "$crowd_predecessor" <<'PY'
import hashlib,json,math,pathlib,sys
root,prior=map(pathlib.Path,sys.argv[1:])
read=lambda p:json.loads(p.read_text())
status=dict(line.split('=',1) for line in (prior/'workflow_status.txt').read_text().splitlines())
assert status['stage']=='completed' and status['exit_code']=='0','predecessor failed/incomplete; no launch'
run=prior/'h1_sup_full';record=read(run/'run_record.json');progress=read(run/'progress.json')
assert record['status']=='completed' and record['stage']=='full' and record['test_evaluations']==1
assert record['protocol']=='view_trainval_test_official_reference_aligned'
assert progress['epoch']==30 and progress['step']==13680
manifest=read(run/'test_manifest.json');metrics=read(run/'metrics_test.json')
assert manifest['checkpoint_epoch']==30 and manifest['optimizer_steps']==13680
assert (run/'prediction_export_test.txt').is_file()
for key in ('group_mAP_1.0','group_mAP_0.5','outlier_mIoU'):assert math.isfinite(metrics[key])
assert hashlib.sha256(pathlib.Path(record['checkpoint_path']).read_bytes()).hexdigest()==manifest['checkpoint_sha256']
geometry=read(root/'geometry_train64/geometry_summary.json')
assert geometry['status']=='completed' and geometry['geometry_gate_pass']
audit=read(root/'audit/run_record.json');assert audit['status']=='completed' and audit['variant']=='h1_sup_crowding'
for path,expected in read(root/'audit/source_manifest.json').items():
    assert hashlib.sha256((root/'audit/source_snapshot'/path).read_bytes()).hexdigest()==expected,path
print('PREDECESSOR_TRAIN_AND_TEST_COMPLETE_FROZEN_B_SOURCE_OK',flush=True)
PY
crowd_run "$crowd_python" -c 'from tools.check_heatmap_gpu_allocation import require_idle_allocation; require_idle_allocation(); print("GPU2_3_IDLE_OK",flush=True)'
crowd_stage=contract_tests
crowd_run timeout --kill-after=10s 3m "$crowd_python" -m unittest discover -s tests/heatmap -v 2>&1 | tee "$crowd_root/contract_host.log"
crowd_stage=smoke3_b
crowd_run timeout --signal=TERM --kill-after=30s 15m "$crowd_python" tools/run_heatmap_research.py \
  --stage smoke --config configs/heatmap/h1_sup_crowding.json --workers 4 \
  --steps 3 --train-clips 32 --eval-clips 3 --device cuda:0 --parallel-gpus 0,1 --output "$crowd_root/smoke3_b"
crowd_stage=overfit100_b
crowd_run timeout --signal=TERM --kill-after=30s 30m "$crowd_python" tools/run_heatmap_research.py \
  --stage overfit --config configs/heatmap/h1_sup_crowding.json --workers 0 \
  --steps 100 --train-clips 16 --eval-clips 3 --cache-fixed-overfit --train-eval-every 50 \
  --device cuda:0 --parallel-gpus 0,1 --output "$crowd_root/overfit100_b"
crowd_stage=verify_bounded
crowd_run "$crowd_python" - "$crowd_root" <<'PY'
import json,math,pathlib,sys
root=pathlib.Path(sys.argv[1]);records=[]
for name,expected in (('smoke3_b',3),('overfit100_b',100)):
    p=root/name;r=json.loads((p/'run_record.json').read_text());records.append(r)
    assert r['status']=='completed' and r['variant']=='h1_sup_crowding'
    summary=json.loads((p/'training_summary.json').read_text());assert summary['steps']==expected
    steps=[json.loads(line) for line in (p/'steps.jsonl').read_text().splitlines()]
    assert len(steps)==expected
    assert all(math.isfinite(s['gradient_norm']) and s['mask_grad_norm']>0 for s in steps)
    assert all(math.isfinite(v) for s in steps for v in s['losses'].values())
assert records[0]['initial_full_state_hash']==records[1]['initial_full_state_hash']
print('B_ONLY_BOUNDED_CHECKS_PASS; NO_FULL_TRAINING',flush=True)
PY
crowd_stage=completed
