"""Read-only, fixed-TRAIN mask-head gradient diagnosis; no optimizer or VAL tuning."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
import torch
import torch.nn.functional as F

from run_heatmap_research import (read_dataset,build_model,seed_all,collate,get_batch,
    record_context,update_registry,json_write,json_default,file_hash,dataset_manifest,tensor_hash)
from models.heatmap import prepare_targets,heatmap_losses


def gradient_vector(loss,parameters):
    gradients=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True) if loss.requires_grad else [None]*len(parameters)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(parameters,gradients)])


def diagnose(model,criterion,batch,args,device):
    core=model.module if isinstance(model,torch.nn.DataParallel) else model
    if core.mask_head is None:
        raise ValueError('a heatmap variant is required for mask-head diagnostics')
    # Common features remain numerically identical in eval mode. Only derivatives
    # w.r.t. mask-head parameters are needed, so do not retain backbone activations.
    for parameter in core.parameters(): parameter.requires_grad_(False)
    for parameter in core.mask_head.parameters(): parameter.requires_grad_(True)
    model.eval()
    x,targets,infos,boxes,dummy,valid=get_batch(batch,device)
    outputs=model(x,boxes,dummy,actor_frame_valid=valid)
    prepared=prepare_targets(outputs,targets,args)
    for target in prepared: target['dummy_idx']=(target['actions']!=args.num_class+1).long()
    indices=criterion.group_matcher(outputs,prepared)
    original_costs=copy.deepcopy(criterion.group_matcher.last_cost_scales)
    losses=criterion(outputs,targets,log=False)
    diagnostic_args=copy.copy(args)
    diagnostic_args.heatmap_focal_gamma=0
    diagnostic_args.heatmap_focal_alpha=-1
    # Counterfactual gradient only: exact SAME assignment and predictions.
    bce=heatmap_losses(outputs,prepared,indices,diagnostic_args)['loss_mask_focal']
    named=list(core.mask_head.named_parameters())
    parameters=[parameter for _,parameter in named]
    vectors={name:gradient_vector(losses[name]*weight,parameters) for name,weight in criterion.weight_dict.items()}
    vectors['diagnostic_soft_bce']=gradient_vector(bce*args.heatmap_focal_coef,parameters)
    scales={}
    for name,vector in vectors.items():
        if not torch.isfinite(vector).all(): raise FloatingPointError('nonfinite diagnostic gradient '+name)
        offset=0; by_parameter={}
        for key,parameter in named:
            by_parameter[key]=float(vector[offset:offset+parameter.numel()].norm())
            offset+=parameter.numel()
        scales[name]=dict(weighted_gradient_norm=float(vector.norm()),by_parameter=by_parameter)
    cosines={}
    for first in ('loss_group_code','loss_mask_focal','loss_mask_dice','diagnostic_soft_bce'):
        for second in ('loss_mask_focal','loss_mask_dice','loss_mask_empty','loss_group_code'):
            if first==second: continue
            a,b=vectors[first],vectors[second]
            denominator=a.norm()*b.norm()
            cosines[first+' vs '+second]=float(torch.dot(a,b)/denominator) if denominator>0 else None
    total=sum(vector for name,vector in vectors.items() if name!='diagnostic_soft_bce')
    response=outputs['spatial_response'].sum(2)/valid.sum(1)[:,None].clamp(min=1)
    member,nonmember,matched_mask,unmatched_mask=[],[],[],[]
    assignments=[]
    for b,(slots,groups) in enumerate(indices):
        actor_valid=valid[b].any(0)
        assignments.append(dict(clip=[infos[b]['vid'],infos[b]['sid']],slots=slots.tolist(),groups=groups.tolist()))
        used=set(slots.tolist())
        for slot,group in zip(slots,groups):
            truth=targets[b]['members'][0,group].bool()
            member.extend(response[b,slot,truth & actor_valid].detach().cpu().tolist())
            nonmember.extend(response[b,slot,~truth & actor_valid].detach().cpu().tolist())
            matched_mask.append(float(outputs['group_masks'][b,slot].mean()))
        unmatched_mask.extend(float(outputs['group_masks'][b,slot].mean()) for slot in range(args.num_group_tokens) if slot not in used)
    return dict(mode='eval; common params frozen only for derivative memory; no optimization',
        derivative_scope='mask_head only, each original loss weighted as configured',
        loss_values={key:float(value.detach()) for key,value in losses.items()},
        diagnostic_soft_bce_value=float(bce.detach()),gradient_scales=scales,gradient_cosines=cosines,
        total_weighted_mask_gradient_norm=float(total.norm()),matching_cost_scales=original_costs,
        alpha=float(core.mask_head.alpha) if core.mask_head.raw_alpha is not None else None,
        mask_bias=float(core.mask_head.bias),mask_min=float(outputs['group_masks'].min()),mask_max=float(outputs['group_masks'].max()),
        membership_max_mean=float(outputs['membership'].max(1).values.mean()),
        membership_over_threshold=int((outputs['membership'].max(1).values>args.group_threshold).sum()),
        no_group_slots=int((outputs['pred_activities'].argmax(-1)==args.num_class).sum()),
        total_slots=len(infos)*args.num_group_tokens,member_responses=member,nonmember_responses=nonmember,
        matched_mask_means=matched_mask,unmatched_mask_means=unmatched_mask,assignments=assignments,
        caveat='Head-gradient geometry at fixed checkpoint/assignment is not an efficacy or intervention experiment; BCE counterfactual was not trained')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,help='source overfit run with resolved config and selected clips')
    parser.add_argument('--checkpoint',help='default: source run checkpoints/last.pth')
    parser.add_argument('--initial',action='store_true',help='diagnose paired initial state rather than checkpoint')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--parallel-gpus',default='0,1')
    cli=parser.parse_args()
    import os
    os.chdir(str(ROOT))
    run=Path(cli.run)
    config=json.loads((run/'resolved_config.json').read_text()); config.pop('runtime',None)
    args=argparse.Namespace(**config)
    args.attention_probe=False
    cli.stage='diagnose'; cli.steps=0
    cli.checkpoint=None if cli.initial else cli.checkpoint or str(run/'checkpoints/last.pth')
    out=Path(cli.output); out.mkdir(parents=True,exist_ok=False)
    log=open(out/'train.log','w',buffering=1)
    class Tee:
        def __init__(self,stream): self.stream=stream
        def write(self,value): self.stream.write(value); log.write(value)
        def flush(self): self.stream.flush(); log.flush()
    sys.stdout=Tee(sys.stdout); sys.stderr=Tee(sys.stderr)
    started=time.monotonic(); record=record_context(out,args,cli)
    record['checkpoint_selection_rule']='source fixed step or original initial state; no optimization'
    record['source_run']=str(run); update_registry(record)
    try:
        seed_all(args.random_seed); torch.set_num_threads(4)
        train_set,val_set=read_dataset(args); train_set.is_training=False
        selected=json.loads((run/'selected_clips.json').read_text())['train']
        lookup={tuple(clip):i for i,clip in enumerate(train_set.frames)}
        batch=collate([train_set[lookup[tuple(clip)]] for clip in selected])
        record['data_manifest_hash'],record['split_hash']=dataset_manifest(train_set,val_set,out)
        json_write(out/'selected_clips.json',dict(train=selected,val=[]))
        model,criterion=build_model(args)
        if cli.checkpoint:
            state=torch.load(cli.checkpoint,map_location='cpu')
            model.load_state_dict(state['state_dict'],strict=True)
            record['checkpoint_source_hash']=file_hash(cli.checkpoint)
            record['source_step']=state.get('step')
        record['init_hash']=tensor_hash(model)
        device=torch.device(cli.device); model.to(device); criterion.to(device)
        devices=[int(i) for i in cli.parallel_gpus.split(',')] if cli.parallel_gpus else []
        if devices:
            if device.type!='cuda' or devices[0]!=(device.index or 0): raise ValueError('invalid device mapping')
            model=torch.nn.DataParallel(model,device_ids=devices)
        result=diagnose(model,criterion,batch,args,device)
        json_write(out/'gradient_diagnostics.json',result)
        print(json.dumps(result,default=json_default,allow_nan=False),flush=True)
        record.update(status='completed',diagnostic_path=str(out/'gradient_diagnostics.json'))
        if device.type=='cuda': record['peak_memory']=torch.cuda.max_memory_allocated(device)/1024**2
    except BaseException as error:
        record.update(status='failed',failure_reason=type(error).__name__+': '+str(error))
        (out/'failure.txt').write_text(traceback.format_exc()); traceback.print_exc(); raise
    finally:
        record['wall_time']=time.monotonic()-started
        json_write(out/'run_record.json',record); update_registry(record)
        print('RUN_RECORD',json.dumps(record,default=json_default),flush=True)


if __name__=='__main__': main()
