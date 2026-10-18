import json,hashlib,subprocess,shutil
from pathlib import Path
root=Path('/mzy/copyfrom_aixi_gadfortest');out=root/'outputs/sam_person/20261018_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
archive=out/'source_archives'
existing=out/'source_manifest.json'
if existing.exists():
 old=sha(existing);target=archive/old
 if not target.exists():
  target.mkdir(parents=True);shutil.copyfile(existing,target/'source_manifest.json')
  shutil.copytree(out/'source_snapshot',target/'source_snapshot')
  if (out/'implementation.diff').exists():shutil.copyfile(out/'implementation.diff',target/'implementation.diff')
manifest={}
for folder in ('models','dataloader','evaluation','util','tools','tests/sam_person'):
 for p in sorted((root/folder).rglob('*.py')):
  rel=str(p.relative_to(root));manifest[rel]=sha(p)
  dst=out/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
p=out/'source_manifest.json';p.write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
diff=subprocess.check_output(['git','diff','--binary'],cwd=root);(out/'implementation.diff').write_bytes(diff)
cpath=root/'configs/sam_person/pilot_resolved.yaml';c=json.loads(cpath.read_text())
c['code_manifest']=str(p);c['code_hash']=sha(p);c['dirty_diff_hash']=hashlib.sha256(diff).hexdigest()
cpath.write_text(json.dumps(c,indent=2)+'\n')
rows=[json.loads(l) for l in (out/'qc_manifest.jsonl').read_text().splitlines()]
classes=set();individuals=multi=0
for v,clip in sorted({(r['video_id'],r['clip_id']) for r in rows}):
 path=Path(next(r['source_image_path'] for r in rows if r['video_id']==v and r['clip_id']==clip)).parent.parent/'ann.json'
 ann=json.loads(path.read_text());groups=set()
 for actor in ann['figures']:
  if actor['label']=='individual':individuals+=1
  else:groups.add(actor['label'])
  for attribute in actor.get('attributes',[]):
   val=attribute.get('value')
   if isinstance(val,dict) and val.get('key'):classes.add(val['key'])
 multi+=int(len(groups)>1)
summary=dict(classes=sorted(classes),individual_annotation_rows=individuals,multi_group_clips=multi,unique_source_keys=len({r['source_image_key'] for r in rows}),unique_image_content_hashes=len({r['source_image_hash'] for r in rows}),provenance='annotation labels used only for coverage audit; not provider or feature selection')
(out/'diagnostic_coverage.json').write_text(json.dumps(summary,indent=2)+'\n')
print('source hash',c['code_hash']);print(summary)
