"""Read-only delivery checks plus a verification record; no model imports."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path('/mzy/copyfrom_aixi_gadfortest')
OUT=ROOT/'outputs/sam_person/20261018_v1'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(2**20),b''):
            h.update(b)
    return h.hexdigest()


def main():
    frozen=load(OUT/'source_manifest_delivery.json')
    changed=[name for name,digest in frozen.items() if sha(ROOT/name)!=digest]
    assert not changed,changed
    diff=subprocess.check_output(['git','-C',str(ROOT),'diff','--binary','--','tools/run_heatmap_research.py'])
    assert diff==(OUT/'user_initial.diff').read_bytes()
    whitespace=subprocess.run(['git','-C',str(ROOT),'-c','core.whitespace=cr-at-eol','diff','--check','--','dataloader/cafe.py','models/b0_repro.py'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    assert whitespace.returncode==0,whitespace.stdout
    report=ROOT/'docs/sam_person/report.md'
    missing=[]
    targets=re.findall(r'\[[^\]]+\]\(([^)]+)\)',report.read_text())
    for target in targets:
        if not target.startswith(('https://','http://')) and not (report.parent/target).exists():
            missing.append(target)
    assert not missing,missing
    records=load(ROOT/'results/sam_person/run_registry.json')
    accepted=[r for r in records if r.get('included_in_final_pair')]
    assert len(accepted)==10
    for r in accepted:
        assert Path(r['metrics_artifact']).exists(),r['metrics_artifact']
        if r['stage']!='overfit':
            raw=load(Path(r['metrics_artifact']))
            suffix='reference' if r['stage']=='reference' else 'val'
            prediction=Path(r['run_id'])/('prediction_'+suffix+'.txt')
            assert sha(prediction)==raw['metrics']['prediction_sha256']
    actors=[json.loads(line) for line in (OUT/'cache_actor_records.jsonl').open()]
    assert len(actors)==2054
    arrays={r['storage_array_path'] for r in actors}
    metadata={r['storage_metadata_path'] for r in actors}
    assert len(arrays)==len(metadata)==200
    assert all(Path(p).is_file() for p in arrays|metadata)
    assert all(r['raw_mask_valid'] and len(r['prompt_hash'])==64 and len(r['mask_checksum'])==64 for r in actors)
    summary=load(OUT/'final_summary.json')
    assert summary['full_training_run'] is False
    assert summary['costs']['within_budget']
    assert summary['all_optimizer_updates_including_superseded']=={'B0':420,'S2F':423}
    checks=load(OUT/'paired_checks_final.json')
    assert all(all(r['checks'].values()) for r in checks)
    assert 'Ran 6 tests' in (OUT/'cpu_acceptance_stable.log').read_text()
    data_hashes=load(OUT/'input_asset_hashes.json')
    # Large immutable assets were hashed by the immediately preceding audit.
    assert data_hashes['qc_manifest']['sha256']==sha(OUT/'qc_manifest.jsonl')
    assert data_hashes['unchanged_evaluator']['sha256']==sha(ROOT/'evaluation/cafe_eval.py')
    previous=load(OUT/'delivery_verification.json')
    previous.update(source_manifest_delivery_matches=True,user_initial_heatmap_diff_preserved=True,
        report_missing_local_links=missing,report_local_links_checked=sum(not t.startswith(('https://','http://')) for t in targets),
        accepted_stage_records=len(accepted),registry_records=len(records),actor_records=len(actors),
        raw_cache_frames=len(arrays),accepted_prediction_checksums_match=True,
        all_fairness_checks_passed=True,within_all_gpu_budgets=True,all_optimizer_updates_including_superseded=summary['all_optimizer_updates_including_superseded'])
    (OUT/'delivery_verification.json').write_text(json.dumps(previous,indent=2)+'\n')
    print(json.dumps(previous,indent=2))


if __name__=='__main__':
    main()
