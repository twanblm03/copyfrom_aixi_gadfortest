"""Small paired B0/S2F experiment; no full-training or TEST entrypoint."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import build_model
from tools.sam_person_masks import load_config,read_manifest,datasets
from tools.run_heatmap_research import seed_all,collate,get_batch,tensor_hash,b0_learning_rate,evaluate
from util.sam_person import MaskReader,atomic_json,file_hash,canonical,digest,checkpoint_state
from util.group_export import make_txt
from evaluation.cafe_eval import GAD_Evaluation,make_clip_key,make_image_key


def selected_samples(ds, clips, reader, c):
    samples=[];rows=[]
    for clip in clips:
        key=tuple(clip['clip'])
        image,target,info=ds.load_samples([(key,f) for f in clip['frames']])
        masks,raw=reader.clip(info,target['boxes'],target['actor_frame_valid'],c['image_hw'],c['feature_hw'])
        target['sam_masks'],target['raw_mask_valid']=masks,raw
        samples.append((image,target,info))
        h=__import__('hashlib').sha256(image.numpy().tobytes())
        h.update(target['boxes'].numpy().tobytes())
        rows.append(dict(clip=list(key),frames=info['fid'],image_boxes_hash=h.hexdigest()))
        print('cached pilot input',key,flush=True)
    return samples,rows


def forward(model,batch,device,mode):
    x,targets,infos,boxes,dummy,valid=get_batch(batch,device)
    kwargs=dict(actor_frame_valid=valid)
    if mode in ('sam_roi', 'sam_roi_blend'):
        kwargs.update(actor_masks=torch.stack([t['sam_masks'] for t in targets]),
                      raw_mask_valid=torch.stack([t['raw_mask_valid'] for t in targets]))
    return model(x,boxes,dummy,**kwargs),targets,infos,boxes


def evaluate_masks(model,loader,args,out,device,mode,suffix):
    model.eval()
    prediction=out/('prediction_'+suffix+'.txt');prediction.write_text('')
    expected=set();stats=[];tp=fp=fn=0;activity_wrong=0;sets=[];forward_times=[]
    started=time.monotonic()
    for batch in loader:
        torch.cuda.synchronize();forward_start=time.monotonic()
        with torch.no_grad():
            outputs,targets,infos,boxes=forward(model,batch,device,mode)
        torch.cuda.synchronize();forward_times.append(time.monotonic()-forward_start)
        make_txt(boxes,infos,outputs,{i:i for i in range(1,25)},str(prediction),args)
        expected.update(make_clip_key(make_image_key(info['vid'],info['sid'],info['fid'][0])) for info in infos)
        if model.last_mask_stats:stats.append(dict(model.last_mask_stats))
        for target in targets:target['dummy_idx']=(target['actions']!=7).long()
        from models.group_matcher import build_group_matcher
        matched=build_group_matcher(args)(outputs,targets)
        for b,(slots,groups) in enumerate(matched):
            valid=targets[b]['dummy_idx'][0].bool()
            scores,chosen=outputs['membership'][b].max(0)
            for slot,group in zip(slots,groups):
                truth=targets[b]['members'][0,group].bool() & valid
                pred=(chosen==slot.to(chosen.device)) & (scores>args.group_threshold) & valid
                tp+=int((truth&pred).sum());fp+=int((~truth&pred).sum());fn+=int((truth&~pred).sum())
                activity_wrong+=int(outputs['pred_activities'][b,slot].argmax()!=targets[b]['activities'][0,group])
                sets.append(dict(clip=[infos[b]['vid'],infos[b]['sid']],slot=int(slot),group=int(group),
                    missing=int((truth&~pred).sum()),extra=int((~truth&pred).sum()),
                    member_set_iou=float((truth&pred).sum())/max(1,int((truth|pred).sum()))))
    observed={make_clip_key(make_image_key(*line.split()[:3])) for line in prediction.read_text().splitlines()}
    if observed!=expected:raise ValueError('export clip coverage mismatch')
    gt=out/('groundtruth_'+suffix+'.txt')
    with open(args.groundtruth) as stream:
        gt.write_text(''.join(line for line in stream if make_clip_key(make_image_key(*line.split()[:3])) in expected))
    with open(gt) as truth,open(args.labelmap) as labels,open(prediction) as predictions:
        metrics=GAD_Evaluation(argparse.Namespace(eval_type=args.eval_type,groundtruth=truth,labelmap=labels)).evaluate(predictions)
    metrics={k:(None if np.isscalar(v) and not np.isfinite(v) else v) for k,v in metrics.items()}
    metrics.update(membership_precision=tp/max(tp+fp,1),membership_recall=tp/max(tp+fn,1),
        membership_f1=2*tp/max(2*tp+fp+fn,1),membership_scope='Hungarian matched GT groups, decoded argmax+threshold; supplementary proxy',
        matched_group_activity_errors=activity_wrong,matched_group_details=sets,
        mask_counts={k:sum(s[k] for s in stats) for k in stats[0]} if stats else None,
        clip_count=len(expected),prediction_sha256=file_hash(prediction),eval_seconds=time.monotonic()-started)
    metrics.update(cached_input_h2d_forward_seconds=forward_times,
        cached_input_h2d_forward_clips_per_second=len(expected)/sum(forward_times))
    atomic_json(out/('metrics_'+suffix+'.json'),metrics)
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--variant',choices=['B0','S2F'],required=True)
    p.add_argument('--stages',nargs='+',choices=['reference','smoke','overfit','pilot'],default=['reference','smoke','overfit','pilot'])
    cli=p.parse_args();c=json.loads(Path(cli.config).read_text())
    if not Path(c['gate_file']).is_file() or not json.loads(Path(c['gate_file']).read_text())['passed']:
        raise ValueError('P1/P2 engineering gate has not passed')
    if file_hash(c['code_manifest'])!=c['code_hash']:
        raise ValueError('frozen code manifest changed')
    for relative,expected in json.loads(Path(c['code_manifest']).read_text()).items():
        if file_hash(ROOT/relative)!=expected:raise ValueError('source changed after experiment freeze: '+relative)
    qc=load_config(c['qc_config']);rows=read_manifest(c['qc_manifest'])
    if c['overfit_steps']>200 or c['pilot_steps']>500 or c['train_clips']>64 or c['val_clips']>32:
        raise ValueError('bounded plan exceeds manual')
    if c['train_gpu_hours']>2 or not c['allocation_confirmed']:
        raise ValueError('missing or excessive GPU allocation')
    output=Path(c['output_root'])/cli.variant
    output.mkdir(parents=True,exist_ok=True)
    with open(ROOT/'docs/sam_person/commands_actual.md','a') as stream:
        env=' '.join(k+'='+shlex.quote(os.environ[k]) for k in ('CUDA_DEVICE_ORDER','CUDA_VISIBLE_DEVICES',
            'OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONPATH','TORCH_HOME') if k in os.environ)
        stream.write('\n```bash\n'+env+' '+shlex.join([sys.executable]+sys.argv)+'\n```\n')
    reader=MaskReader(qc,rows);reader.preflight()
    args,train,val=datasets(qc)
    args.batch=c['batch'];args.test_batch=c['batch'];args.random_seed=c['seed']
    args.train_mask_mode='disabled' if cli.variant=='B0' else 'sam_roi'
    args.mask_empty_eps=qc['empty_eps']
    args.data_path=str(Path(qc['data_root']).parent)
    clips=json.loads(Path(c['clip_manifest']).read_text())
    train_clips=[clips['train'][i] for i in np.linspace(0,len(clips['train'])-1,c['train_clips'],dtype=int)]
    val_clips=[clips['val'][i] for i in np.linspace(0,len(clips['val'])-1,c['val_clips'],dtype=int)]
    train_samples,train_rows=selected_samples(train,train_clips,reader,qc)
    val_samples,val_rows=selected_samples(val,val_clips,reader,qc)
    atomic_json(output/'input_manifest.json',dict(train=train_rows,val=val_rows))
    atomic_json(output/'config_actual.json',dict(config=c,args=vars(args),cli=vars(cli),
        torch=torch.__version__,torchvision=__import__('torchvision').__version__,python=sys.version,
        numpy=np.__version__,scipy=__import__('scipy').__version__,pillow=__import__('PIL').__version__,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32))
    device=torch.device('cuda:0')
    seed_all(c['seed']);model,criterion=build_model(args)
    checkpoint=torch.load(c['checkpoint'],map_location='cpu')
    state,key_mapping=checkpoint_state(checkpoint)
    model.load_state_dict(state,strict=True)
    init_hash=tensor_hash(model)
    model.to(device);criterion.to(device)
    baseline_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
    registry=[]
    # Each process has a conservative half-share. Both shares include model
    # setup, failed attempts, train evaluation and checkpoint export.
    ledger_path=Path(c['training_ledger_root'])/(cli.variant+'.json')
    ledger_path.parent.mkdir(parents=True,exist_ok=True)
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(seconds=0.,stages=[])
    if ledger.get('active_since'):ledger['seconds']+=time.time()-ledger.pop('active_since')
    session_start=time.time();ledger['active_since']=session_start;atomic_json(ledger_path,ledger)
    def elapsed():return ledger['seconds']+time.time()-session_start
    try:
        for stage in cli.stages:
            out=output/stage
            if out.exists():raise FileExistsError('retain existing run; select a new output_root')
            out.mkdir();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
            model.load_state_dict(baseline_state,strict=True);seed_all(c['seed'])
            record=dict(run_id=str(out),variant=cli.variant,stage=stage,seed=c['seed'],steps=0,
                base_checkpoint_or_init_hash=init_hash,base_checkpoint_sha256=file_hash(c['checkpoint']),
                checkpoint_key_mapping=key_mapping,
                code_revision=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
                config_hash=file_hash(cli.config),evaluator_hash=file_hash(ROOT/'evaluation/cafe_eval.py'),
                code_hash=c['code_hash'],dirty_diff_hash=c['dirty_diff_hash'],
                protocol=qc['protocol'],input_manifest_hash=file_hash(output/'input_manifest.json'),
                train_clip_manifest_hash=digest(canonical(train_rows)),val_clip_manifest_hash=digest(canonical(val_rows)),
                sampled_frame_hash=digest(canonical(dict(train=train_rows,val=val_rows))),split_hash=c['split_hash'],
                checkpoint_selection_rule='fixed last bounded step',effective_batch=c['batch'],
                lr_schedule='B0 full TRAIN epoch clock; paired same Adam, no microbatches',
                sam_version=qc['sam_family'],checkpoint_hash=qc['checkpoint_sha256'],cache_version=qc['cache_version'],
                prompt_policy='box_only',status='running',failure_reason=None,metrics=None)
            order=[];losses=[];mask_counts=[]
            val_loader=DataLoader(val_samples,batch_size=c['batch'],collate_fn=collate,shuffle=False,num_workers=0)
            if stage=='reference':
                record['metrics']=evaluate_masks(model,val_loader,args,out,device,args.train_mask_mode,'reference')
                if cli.variant=='B0':
                    batch=next(iter(val_loader))
                    model.eval()
                    with torch.no_grad():
                        before,_,_,_=forward(model,batch,device,'disabled')
                        ones_batch=(batch[0],[dict(t,sam_masks=torch.ones_like(t['sam_masks']),
                            raw_mask_valid=torch.ones_like(t['raw_mask_valid'])) for t in batch[1]],batch[2])
                        model.train_mask_mode='sam_roi'
                        after,_,_,_=forward(model,ones_batch,device,'sam_roi')
                        model.train_mask_mode='disabled'
                        model.last_mask_stats=None
                    record['real_gpu_ones_max_error']={k:float((before[k]-after[k]).abs().max()) for k in before}
                    for k in before:torch.testing.assert_allclose(before[k],after[k],rtol=1e-5,atol=1e-6)
                    record['real_gpu_ones_tolerance']=dict(rtol=1e-5,atol=1e-6)
            else:
                samples=train_samples[:c['overfit_clips']] if stage in ('smoke','overfit') else train_samples
                generator=torch.Generator().manual_seed(c['seed'])
                loader=DataLoader(samples,batch_size=c['batch'],collate_fn=collate,shuffle=True,
                    generator=generator,num_workers=0,drop_last=True)
                if not len(loader):raise ValueError('no full batch in pilot')
                limit={'smoke':3,'overfit':c['overfit_steps'],'pilot':c['pilot_steps']}[stage]
                optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(.9,.999),eps=1e-8,weight_decay=0)
                train_eval_loader=DataLoader(samples,batch_size=c['batch'],collate_fn=collate,shuffle=False)
                if stage=='overfit':
                    record['train_before']=evaluate_masks(model,train_eval_loader,args,out,device,args.train_mask_mode,'train_before')
                while record['steps']<limit and elapsed()<c['train_gpu_hours']*3600/2:
                    for batch in loader:
                        if elapsed()>=c['train_gpu_hours']*3600/2:break
                        model.train();optimizer.zero_grad();torch.cuda.synchronize();step_start=time.monotonic()
                        outputs,targets,infos,boxes=forward(model,batch,device,args.train_mask_mode)
                        terms=criterion(outputs,targets,log=False)
                        loss=sum(terms[k]*v for k,v in criterion.weight_dict.items())
                        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                        loss.backward()
                        grads=[p.grad for p in model.parameters() if p.grad is not None]
                        grad_norm=sum(float(g.detach().square().sum()) for g in grads)**.5
                        if not math.isfinite(grad_norm):raise FloatingPointError('nonfinite gradients')
                        optimizer.step();record['steps']+=1;torch.cuda.synchronize()
                        if model.last_mask_stats:mask_counts.append(dict(model.last_mask_stats))
                        order.append([[info['vid'],info['sid'],info['fid']] for info in infos])
                        item=dict(step=record['steps'],loss=float(loss),terms={k:float(v) for k,v in terms.items()},
                            grad_norm=grad_norm,seconds=time.monotonic()-step_start,lr=args.lr)
                        losses.append(item)
                        if stage=='overfit' and record['steps']==3:
                            atomic_json(out/'technical_smoke.json',dict(variant=cli.variant,steps=3,
                                status='passed',finite_loss=True,finite_gradients=True,
                                init_hash=init_hash,code_hash=c['code_hash'],
                                sample_order_hash=digest(canonical(order)),integrated_in_overfit=True))
                        with open(out/'steps.jsonl','a') as stream:stream.write(json.dumps(item)+'\n')
                        atomic_json(ledger_path,dict(ledger,active_since=session_start,latest_elapsed=elapsed()))
                        if record['steps']%10==0 or record['steps']==1:print(cli.variant,stage,item,flush=True)
                        if record['steps']>=limit:break
                record['sample_order_hash']=digest(canonical(order));atomic_json(out/'sample_order.json',order)
                record['first_loss']=losses[0]['loss'] if losses else None
                record['last_loss']=losses[-1]['loss'] if losses else None
                if stage=='overfit':record['train_after']=evaluate_masks(model,train_eval_loader,args,out,device,args.train_mask_mode,'train_after')
                if stage=='pilot':record['metrics']=evaluate_masks(model,val_loader,args,out,device,args.train_mask_mode,'val')
                torch.save(dict(state_dict=model.state_dict(),args=vars(args),step=record['steps']),out/'last.pth')
                reloaded=torch.load(out/'last.pth',map_location='cpu');model.load_state_dict(reloaded['state_dict'],strict=True)
                record['mask_counts']={k:sum(s[k] for s in mask_counts) for k in mask_counts[0]} if mask_counts else None
                record['status']='complete' if record['steps']==limit else 'partial'
            if record['status']=='running':record['status']='complete'
            record['seconds']=time.monotonic()-start
            record['peak_memory_bytes']=torch.cuda.max_memory_allocated()
            record['train_gpu_hours_including_evaluation']=record['seconds']/3600
            atomic_json(out/'result.json',record);registry.append(record);atomic_json(output/'results.json',registry)
            print('STAGE_RESULT',cli.variant,stage,record['status'],record['steps'],flush=True)
    except BaseException as error:
        atomic_json(output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        ledger['seconds']=elapsed();ledger.pop('active_since',None)
        ledger['status']='finished';atomic_json(ledger_path,ledger)


if __name__=='__main__':main()
