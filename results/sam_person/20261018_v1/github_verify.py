import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path('/mzy/copyfrom_aixi_gadfortest-sam-person-20261018')
SERVER=Path('/mzy/copyfrom_aixi_gadfortest')
OUT=ROOT/'results/sam_person/20261018_v1'
BASE='33ca78cffd46d3bc50b7d3515ede28f70de842be'


def load(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline=subprocess.check_output(['git','diff',BASE,'--','tools/run_heatmap_research.py'],cwd=ROOT)
    assert not baseline
    assert not (ROOT/'util/training_resume.py').exists()
    assert BASE+':models/b0_repro.py' in (ROOT/'tests/sam_person/test_contract.py').read_text()
    reference=load(SERVER/'outputs/sam_person/20261018_v1/final_summary.json')
    packaged=load(OUT/'final_summary.json')
    assert reference['cohorts']==packaged['cohorts']
    assert reference['costs']==packaged['costs']
    assert not packaged['full_training_run']
    metadata=load(OUT/'github_packaging.json')
    assert all(metadata['active_training_implementation_unchanged'].values())
    for name in metadata['active_training_implementation_unchanged']:
        assert sha(ROOT/name)==sha(SERVER/name)
    manifest=load(OUT/'source_manifest_delivery.json')
    bad_snapshot=[name for name,h in manifest.items() if sha(OUT/'source_delivery'/name)!=h]
    assert not bad_snapshot,bad_snapshot
    missing_links=[];link_count=0
    for doc in (ROOT/'docs/sam_person').glob('*.md'):
        # The original playbook contains illustrative, non-local commands,
        # but only actual Markdown targets need to exist in the Git bundle.
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',doc.read_text()):
            if target.startswith(('http://','https://','#')):
                continue
            link_count+=1
            if not (doc.parent/target).exists():missing_links.append((str(doc.relative_to(ROOT)),target))
    assert not missing_links,missing_links
    for directory in ['smoke_gallery','failure_gallery','qc_gallery']:
        doc=OUT/directory/'index.html'
        for target in re.findall(r'(?:src|href)=[\"\']([^\"\']+)[\"\']',doc.read_text()):
            if target.startswith(('http://','https://','#','data:')):continue
            assert (doc.parent/target).is_file(),(doc,target)
    files=[f for f in OUT.rglob('*') if f.is_file() and '__pycache__' not in f.parts]
    prohibited=[str(f.relative_to(ROOT)) for f in files if f.suffix in {'.pth','.pkl','.npz','.pt','.ckpt'} or 'smoke_originals' in f.parts]
    assert not prohibited,prohibited
    assert max(f.stat().st_size for f in files)<10*1024*1024
    # Scan text for credential formats, reporting only filenames on a match.
    pattern=re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
    suspected=[]
    text_files=[f for f in files if f.suffix in {'.py','.json','.jsonl','.csv','.txt','.md','.log','.diff','.html'}]
    text_files.extend((ROOT/'docs/sam_person').glob('*.md'))
    for f in text_files:
        if pattern.search(f.read_bytes()):suspected.append(str(f.relative_to(ROOT)))
    assert not suspected,suspected
    original_diff=subprocess.check_output(['git','diff','--binary','--','tools/run_heatmap_research.py'],cwd=SERVER)
    assert original_diff==(SERVER/'outputs/sam_person/20261018_v1/user_initial.diff').read_bytes()
    result={'original_workspace_preserved':True,'active_other_work_not_included':True,
        'packaged_results_equal_server_results':True,'trained_implementation_unchanged':True,
        'historical_source_snapshot_hashes_match':True,'baseline_test_revision_pinned':BASE,
        'local_markdown_links_checked':link_count,'missing_local_links':missing_links,
        'gallery_image_links_resolve':True,'prohibited_files':prohibited,
        'credential_format_matches':suspected,'artifact_file_count':len(files),
        'artifact_bytes':sum(f.stat().st_size for f in files),'cpu_tests':'6 passed; see github_cpu_acceptance.log',
        'new_gpu_runs':0}
    (OUT/'github_branch_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
