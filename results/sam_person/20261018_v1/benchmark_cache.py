import json,sys,time,statistics
from pathlib import Path
root=Path('/mzy/copyfrom_aixi_gadfortest');sys.path.insert(0,str(root))
from util.sam_person import read_frame,feature_masks,atomic_json
c=json.loads((root/'configs/sam_person/qc_resolved.yaml').read_text())
row=json.loads((root/'outputs/sam_person/20261018_v1/qc_manifest.jsonl').read_text().splitlines()[0])
import torch
torch.set_num_threads(4)
reads=[];projections=[]
for i in range(20):
 start=time.monotonic();masks,meta=read_frame(c['cache_root'],c['cache_version'],row);reads.append(time.monotonic()-start)
 start=time.monotonic();feature_masks(masks,c['image_hw'],c['feature_hw']);projections.append(time.monotonic()-start)
result=dict(scope='one fixed raw frame repeated 20 times, warm cache after first; checksum+decode and mask resize+area projection; no image source rehash',actors=len(row['actors']),source=row['source_image_key'],read_seconds=reads,projection_seconds=projections,read_median_seconds=statistics.median(reads[1:]),projection_median_seconds=statistics.median(projections[1:]))
atomic_json(root/'outputs/sam_person/20261018_v1/cache_read_benchmark.json',result)
print(json.dumps(result))
