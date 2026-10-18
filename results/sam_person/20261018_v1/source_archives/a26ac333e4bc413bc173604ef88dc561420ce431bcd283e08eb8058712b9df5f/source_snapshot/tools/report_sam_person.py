"""Collect existing bounded SAM artifacts; never invent missing metrics."""
import csv
import json
from pathlib import Path
import sys
import statistics

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from util.sam_person import atomic_json,file_hash


def main():
    out=ROOT/'outputs/sam_person/20261018_v1'
    qc=json.loads((out/'qc_metrics.json').read_text())
    generation=json.loads((out/'generation_budget.json').read_text())
    records=[]
    for variant in ('B0','S2F'):
        path=out/'paired'/variant/'results.json'
        if path.exists():records.extend(json.loads(path.read_text()))
    checks=[]
    for stage in ('reference','smoke','overfit','pilot'):
        pair=[next((r for r in records if r['variant']==v and r['stage']==stage),None) for v in ('B0','S2F')]
        keys=['base_checkpoint_or_init_hash','config_hash','evaluator_hash','input_manifest_hash','split_hash','seed','effective_batch','steps']
        if stage!='reference':keys.append('sample_order_hash')
        checks.append(dict(stage=stage,present=all(pair),checks={k:bool(all(pair) and pair[0].get(k)==pair[1].get(k)) for k in keys}))
    costs=dict(sam_gpu_hours_including_failed_attempts=generation['seconds']/3600,
        sam_attempted_prompts=generation['prompts'],
        generation_sessions=[dict(status=s['status'],failure_reason=s.get('failure_reason'),
            load_seconds=s['load_seconds'],frames=len(s['frames']),peak_memory_bytes=s.get('peak_memory_bytes')) for s in generation['sessions']],
        cache_bytes=sum(p.stat().st_size for p in (out/'cache').rglob('*') if p.is_file()),
        train_gpu_hours_including_evaluation=0.,training_ledgers={})
    for variant in ('B0','S2F'):
        path=out/'training_ledgers'/(variant+'.json')
        if path.exists():
            ledger=json.loads(path.read_text());costs['training_ledgers'][variant]=ledger
            costs['train_gpu_hours_including_evaluation']+=ledger['seconds']/3600
    frames=[f for s in generation['sessions'] for f in s['frames']]
    for key in ('image_encoder_seconds','decoder_seconds','read_seconds','write_seconds'):
        values=[f[key] for f in frames]
        costs[key]=dict(total=sum(values),mean=statistics.mean(values) if values else None,
            median=statistics.median(values) if values else None,first=values[0] if values else None)
    costs['decoder_seconds_per_prompt']=sum(f['decoder_seconds'] for f in frames)/max(sum(f['actors'] for f in frames),1)
    atomic_json(out/'cost_summary.json',costs)
    atomic_json(out/'paired_checks.json',checks)
    table=[]
    for r in records:
        metrics=r.get('metrics') or {}
        row={k:v for k,v in r.items() if not isinstance(v,(dict,list))}
        row.update(group_map_1_0=metrics.get('group_mAP_1.0'),group_map_0_5=metrics.get('group_mAP_0.5'),
            outlier_miou=metrics.get('outlier_mIoU'),membership_f1=metrics.get('membership_f1'),
            raw_success_rate=qc['all']['raw_success_rate'],usable_mask_coverage=qc['all']['usable_coverage'],
            qc_fallback_rate=qc['all']['fallback_rate'],sam_gpu_hours=costs['sam_gpu_hours_including_failed_attempts'],
            cache_bytes=costs['cache_bytes'])
        table.append(row)
    # Failed generation attempts stay visible beside successful runs.
    for i,s in enumerate(generation['sessions']):
        if s['status']=='failed':table.append(dict(run_id='generation_session_%d'%i,variant='SAM-QC',
            status='failed',failure_reason=s.get('failure_reason'),group_map_1_0=None,group_map_0_5=None,outlier_miou=None))
    registry=ROOT/'results/sam_person/run_registry.csv';registry.parent.mkdir(parents=True,exist_ok=True)
    fields=sorted({k for r in table for k in r})
    with open(registry,'w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:('null' if v is None else v) for k,v in row.items()} for row in table)
    atomic_json(ROOT/'results/sam_person/run_registry.json',table)
    print(json.dumps(dict(paired_checks=checks,costs=costs),indent=2))


if __name__=='__main__':main()
