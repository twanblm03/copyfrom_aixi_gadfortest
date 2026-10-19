#!/usr/bin/env bash
# User-authorized B seed1 only. Place TRAIN->VAL; no A/view/H2/seed2/3 queue.
set -euo pipefail
crowd_repo=''
crowd_root=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) crowd_repo="$2"; shift 2 ;;
    --output-root) crowd_root="$2"; shift 2 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ "$crowd_repo" != /* || "$crowd_root" != /* || ! -f "$crowd_root/audit/run_record.json" ]]; then
  printf 'Absolute repository/root and prepared CPU audit required.\n' >&2; exit 2
fi
crowd_python="$crowd_repo/.miniconda3/envs/gad/bin/python"
crowd_code="$crowd_root/audit/source_snapshot"
if [[ ! -x "$crowd_python" || -e "$crowd_root/smoke3" || -e "$crowd_root/overfit100" || -e "$crowd_root/h1_sup_crowding_full" ]]; then
  printf 'Existing gad environment and unused smoke/overfit/full paths required.\n' >&2; exit 2
fi
exec 9>"$crowd_repo/results/heatmap/dual_gpu_23.lock"
if ! flock -n 9; then printf 'Another project job holds GPU2/3 lock.\n' >&2; exit 2; fi
exec > >(tee -a "$crowd_root/workflow.log") 2>&1
export TZ=Asia/Shanghai PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-f13b1d06-5ee1-64b8-bc92-4e5c28d72e7c,GPU-bc83c5ed-4690-7c23-37a5-8f78022962fa
crowd_stage=source_verification
crowd_finish() {
  crowd_rc=$?
  printf 'stage=%s\nexit_code=%s\nfinished_at=%s\n' "$crowd_stage" "$crowd_rc" "$(date --iso-8601=seconds)" > "$crowd_root/workflow_status.txt"
  printf 'WORKFLOW_EXIT stage=%s exit_code=%s\n' "$crowd_stage" "$crowd_rc"
}
trap crowd_finish EXIT
crowd_run() { printf '$ '; printf '%q ' "$@"; printf '\n'; "$@"; }
printf 'STARTED_AT=%s\nWORKFLOW_PID=%s\nFROZEN_SOURCE=%s\nFULL_OUTPUT=%s\nB_ONLY_UNPAIRED_EXPLORATORY_PLACE_TRAIN_VAL\n' "$(date --iso-8601=seconds)" "$$" "$crowd_code" "$crowd_root/h1_sup_crowding_full"
cd "$crowd_code"
crowd_run "$crowd_python" - "$crowd_root" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);read=lambda p:json.loads(p.read_text())
audit=root/'audit';r=read(audit/'run_record.json')
assert r['status']=='completed' and r['stage']=='audit' and r['variant']=='h1_sup_crowding'
assert r['protocol']=='place_train_val_b0_repro'
data=read(audit/'dataset_audit.json');assert data['train_clips']==6444 and data['val_clips']==1386
for path,expected in read(audit/'source_manifest.json').items():
    assert hashlib.sha256((audit/'source_snapshot'/path).read_bytes()).hexdigest()==expected,path
assert (audit/'source_snapshot/tools/official_cafe_reference/crop_and_resize.cpp').is_file()
base=read(pathlib.Path('configs/heatmap/h1_sup.json'));candidate=read(pathlib.Path('configs/heatmap/h1_sup_crowding.json'))
assert candidate.pop('heatmap_target_mode')=='nearest_center' and base==candidate
geometry=read(root/'geometry_train64/geometry_summary.json')
assert geometry['status']=='completed' and geometry['geometry_gate_pass']
assert geometry['checked_clips']==64 and not geometry['val_loaded'] and not geometry['test_loaded']
gsrc=root/'geometry_train64'
for path,expected in read(gsrc/'source_manifest.json').items():
    assert hashlib.sha256((gsrc/'source_snapshot'/path).read_bytes()).hexdigest()==expected
    assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==expected,path
print('FROZEN_B_SOURCE_CONFIG_AND_TRAIN_GEOMETRY_OK',r['source_manifest_hash'],flush=True)
PY
crowd_run "$crowd_python" -c 'from tools.check_heatmap_gpu_allocation import require_idle_allocation; require_idle_allocation(); print("GPU2_3_IDLE_OK",flush=True)'
crowd_run nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,name,temperature.gpu,memory.used,utilization.gpu --format=csv
crowd_stage=contract_tests
crowd_run timeout --kill-after=10s 4m "$crowd_python" -m unittest discover -s tests/heatmap -v 2>&1 | tee "$crowd_root/contracts_host.log"
crowd_stage=smoke3
crowd_run timeout --signal=TERM --kill-after=30s 15m "$crowd_python" tools/run_heatmap_research.py \
  --stage smoke --config configs/heatmap/h1_sup_crowding.json --workers 4 --steps 3 \
  --train-clips 32 --eval-clips 3 --device cuda:0 --parallel-gpus 0,1 --output "$crowd_root/smoke3"
crowd_stage=overfit100
crowd_run timeout --signal=TERM --kill-after=30s 30m "$crowd_python" tools/run_heatmap_research.py \
  --stage overfit --config configs/heatmap/h1_sup_crowding.json --workers 0 --steps 100 \
  --train-clips 16 --eval-clips 3 --cache-fixed-overfit --train-eval-every 50 \
  --device cuda:0 --parallel-gpus 0,1 --output "$crowd_root/overfit100"
crowd_stage=bounded_validation
crowd_run "$crowd_python" - "$crowd_root" <<'PY'
import json,math,pathlib,sys
from tools.run_heatmap_research import json_write
root=pathlib.Path(sys.argv[1]);read=lambda p:json.loads(p.read_text());records=[]
cfg=read(pathlib.Path('configs/heatmap/h1_sup_crowding.json'))
weights=dict(loss_ce=cfg['ce_loss_coef'],loss_group_ce=cfg['group_ce_loss_coef'],
    loss_group_code=cfg['group_code_loss_coef'],loss_consistency=cfg['consistency_loss_coef'],
    loss_mask_focal=cfg['heatmap_focal_coef'],loss_mask_dice=cfg['heatmap_dice_coef'],loss_mask_empty=cfg['heatmap_empty_coef'])
for name,expected in (('smoke3',3),('overfit100',100)):
    p=root/name;r=read(p/'run_record.json');records.append(r)
    assert r['status']=='completed' and r['variant']=='h1_sup_crowding'
    assert r['heatmap_target_mode']=='nearest_center' and r['parallel_gpus']==[0,1]
    assert read(p/'training_summary.json')['steps']==expected
    trace=[json.loads(line) for line in (p/'steps.jsonl').read_text().splitlines()]
    assert len(trace)==expected
    assert all(math.isfinite(s['gradient_norm']) and s['mask_grad_norm']>0 for s in trace)
    assert all(math.isfinite(v) for s in trace for v in s['losses'].values())
    if name=='overfit100':
        mean=lambda rows,key:sum(s['losses'][key] for s in rows)/len(rows)
        first=sum(weights[k]*mean(trace[:10],k) for k in weights)
        last=sum(weights[k]*mean(trace[-10:],k) for k in weights)
        dice_first=mean(trace[:10],'loss_mask_dice');dice_last=mean(trace[-10:],'loss_mask_dice')
        assert last<first and dice_last<dice_first,'short-overfit loss did not improve; no full launch'
        json_write(root/'bounded_checks.json',dict(all_finite=True,mask_gradient_nonzero=True,
            first10_weighted_loss=first,last10_weighted_loss=last,first10_mask_dice=dice_first,last10_mask_dice=dice_last,
            engineering_only_no_ap_parameter_selection=True,full_training_authorized=True,strict_A_B_pair=False))
for key in ('init_hash','initial_full_state_hash','initial_rng_hash','data_manifest_hash','split_hash','source_manifest_hash','evaluator_hash'):
    assert records[0][key]==records[1][key],key
assert records[0]['init_hash']=='2a5629056bbe7a665dddbfe242cba6852483c241d59d0c9c46eabf2ad9ef479b'
assert records[0]['initial_full_state_hash']=='16cdabdf5ab3bf64008b78f0c54e813d3d6b65e1daad23a31613bdc5c111380b'
print('B_SMOKE_OVERFIT_GATES_PASS_FRESH_FULL_NEXT',flush=True)
PY
crowd_stage=full_training
crowd_run "$crowd_python" tools/run_heatmap_research.py \
  --stage full --config configs/heatmap/h1_sup_crowding.json --workers 4 --eval-clips 0 \
  --device cuda:0 --parallel-gpus 0,1 --output "$crowd_root/h1_sup_crowding_full"
crowd_stage=final_verification
crowd_run "$crowd_python" - "$crowd_root" <<'PY'
import json,math,pathlib,sys,torch
root=pathlib.Path(sys.argv[1]);p=root/'h1_sup_crowding_full';read=lambda f:json.loads(f.read_text())
r=read(p/'run_record.json');ref=read(root/'smoke3/run_record.json')
assert r['status']=='completed' and r['variant']=='h1_sup_crowding' and r['stage']=='full'
assert read(p/'training_summary.json')['steps']==12060
for key in ('init_hash','initial_full_state_hash','initial_rng_hash','data_manifest_hash','split_hash','source_manifest_hash','evaluator_hash'):
    assert r[key]==ref[key],key
epochs=[read(p/('metrics_epoch%d.json'%e)) for e in range(1,31)]
assert all(m['clip_count']==1386 for m in epochs)
best=max(range(30),key=lambda i:epochs[i]['group_mAP_1.0'])+1
saved=torch.load(p/'checkpoints/best_val.pth',map_location='cpu');assert saved['epoch']==best
assert read(p/'metrics_val.json')['prediction_hash']==epochs[best-1]['prediction_hash']
assert not list(p.glob('metrics_test*'))
print('B_FULL_30_EPOCHS_BEST_VAL_VERIFIED',best,flush=True)
PY
crowd_stage=completed
