import sys,json,time
from pathlib import Path
root=Path('/mzy/copyfrom_aixi_gadfortest');sys.path.insert(0,str(root))
import torch
from tools.sam_person_masks import load_config,datasets
from tools.run_heatmap_research import seed_all,collate,get_batch
from util.sam_person import checkpoint_state,atomic_json
from models import build_model
import models.sam_roi as roi_module
out=root/'outputs/sam_person/20261018_v1'
c=load_config(root/'configs/sam_person/qc_resolved.yaml');args,train,val=datasets(c)
args.train_mask_mode='disabled';seed_all(1)
model,_=build_model(args)
p=json.loads((root/'configs/sam_person/pilot_resolved.yaml').read_text())
checkpoint=torch.load(p['checkpoint'],map_location='cpu');state,_=checkpoint_state(checkpoint);model.load_state_dict(state,strict=True)
clips=json.loads((out/'paired/B0/input_manifest.json').read_text())['val'][:4]
samples=[val.load_samples([(tuple(r['clip']),f) for f in r['frames']]) for r in clips]
batch=collate(samples);device=torch.device('cuda:0');started=time.time()
model.to(device).eval();x,targets,infos,boxes,dummy,valid=get_batch(batch,device)
ones=torch.ones(4,5,14,23,40,device=device)
captured={}
def hook(module,inputs,output):
 if inputs[0].shape[1]==512:
  captured['boxes']=torch.cat(inputs[1]);captured['roi']=output
handle=model.roi_align.register_forward_hook(hook)
def run(mode):
 model.train_mask_mode=mode
 with torch.no_grad():return model(x,boxes,dummy,actor_frame_valid=valid,actor_masks=ones,raw_mask_valid=valid)
def diff(a,b):return {k:float((a[k]-b[k]).abs().max()) for k in a}
base=run('disabled');repeat=run('disabled')
direct=run('sam_roi')
weight=model.roi_align(ones.reshape(-1,1,23,40),[b[None] for b in captured['boxes']])
interior=weight[valid.reshape(-1)]
result=dict(tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,disabled_repeat=diff(base,repeat),direct_ones=diff(base,direct),sampled_ones_min=float(interior.min()),sampled_ones_max=float(interior.max()),sampled_ones_max_deviation=float((interior-1).abs().max()))
original_filter=roi_module.filter_actor_rois
def complement(rois,boxes,masks,actor_valid,raw_valid,align,eps=1e-6):
 count=rois.shape[0];h,w=masks.shape[-2:]
 weights=(1-align((1-masks.detach()).reshape(count,1,h,w).to(rois),[b[None] for b in boxes])).clamp(0,1)
 use=actor_valid.reshape(-1)&raw_valid.reshape(-1)&(weights.mean((1,2,3))>eps)
 weights=torch.where(use[:,None,None,None],weights,torch.ones_like(weights))
 return rois*weights,dict(valid=int(use.sum()))
roi_module.filter_actor_rois=complement
stable=run('sam_roi');result['complement_ones']=diff(base,stable)
roi_module.filter_actor_rois=original_filter
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
strict=run('disabled');strict_ones=run('sam_roi');result['tf32_disabled_direct_ones']=diff(strict,strict_ones)
torch.cuda.synchronize();result['gpu_seconds']=time.time()-started
atomic_json(out/'gpu_ones_diagnosis.json',result);print(json.dumps(result,indent=2))
atomic_json(out/'training_ledgers/diagnostic.json',dict(seconds=result['gpu_seconds'],status='complete',scope='GPU ones diagnosis, physical GPU4'))
