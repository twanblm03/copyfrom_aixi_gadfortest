"""Bounded, recorded heatmap research on frozen ResNet B0 (TRAIN -> VAL)."""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataloader.dataloader import read_dataset, TRAIN_CAFE_P, VAL_CAFE_P, TEST_CAFE_P
from evaluation.cafe_eval import GAD_Evaluation, make_clip_key, make_image_key
from models import build_model
from models.heatmap import gaussian_targets, roi_responses, prepare_targets
from models.heatmap_criterion import HeatmapMatcher
from util.group_export import make_txt


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def json_default(item):
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(item, np.generic):
        return item.item()
    if isinstance(item, Path):
        return str(item)
    raise TypeError(type(item).__name__)


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + '\n')


def git(*args):
    return subprocess.check_output(['git'] + list(args), cwd=str(ROOT)).decode()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collate(batch):
    images, targets, infos = zip(*batch)
    return torch.stack(images), list(targets), list(infos)


def subset_indices(dataset,count):
    """Fixed coverage across the canonical manifest, not a single video's prefix."""
    count=min(count or len(dataset),len(dataset))
    return np.linspace(0,len(dataset)-1,count,dtype=int).tolist()


def b0_learning_rate(args, epoch):
    """The original B0 epoch schedule, not a subset-pass schedule."""
    if epoch < args.warmup_epochs:
        if args.warmup_epochs == 1:
            return args.max_lr
        return args.lr + (args.max_lr-args.lr)*epoch/(args.warmup_epochs-1)
    decay_epochs = args.epochs-args.warmup_epochs
    if not decay_epochs:
        return args.max_lr
    return args.max_lr*max(0.,(args.epochs-(epoch+1))/decay_epochs)


def cache_fixed_samples(dataset, indices):
    """CPU-only optimization for deterministic overfit inputs, never augmentation."""
    if dataset.is_training:
        raise ValueError('only deterministic overfit samples can be cached')
    samples, rows = [], []
    for index in indices:
        image, target, info = dataset[index]
        hasher = hashlib.sha256(image.contiguous().numpy().tobytes())
        for key,value in sorted(target.items()):
            hasher.update(key.encode())
            hasher.update(value.contiguous().numpy().tobytes())
        rows.append(dict(clip=[info['vid'],info['sid']],frames=info['fid'],content_hash=hasher.hexdigest()))
        samples.append((image,target,info))
    return samples, dict(samples=rows, image_bytes=sum(image.numel()*image.element_size() for image,_,_ in samples),
                         preprocessing='unchanged B0, deterministic frames; CPU cache only')


def get_batch(batch, device):
    images, targets, infos = batch
    targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
    boxes = torch.stack([t['boxes'] for t in targets])
    dummy = torch.stack([t['actions'][0] == 7 for t in targets])
    valid = torch.stack([t['actor_frame_valid'] for t in targets])
    return images.to(device), targets, infos, boxes, dummy, valid


def tensor_hash(model, common=True):
    hasher = hashlib.sha256()
    for key, value in model.state_dict().items():
        if common and key.startswith('mask_head.'):
            continue
        hasher.update(key.encode())
        hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def record_context(out, args, cli):
    revision = git('rev-parse', 'HEAD').strip()
    diff = git('diff', '--binary')
    (out/'dirty.diff').write_text(diff)
    sources = {}
    snapshot=out/'source_snapshot'
    for folder in ('models', 'dataloader', 'evaluation', 'util', 'tools', 'configs/heatmap', 'tests/heatmap'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.json'):
                sources[str(path.relative_to(ROOT))] = file_hash(path)
                destination=snapshot/path.relative_to(ROOT)
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,destination)
    json_write(out/'source_manifest.json', sources)
    config = vars(args).copy()
    config['runtime']=vars(cli)
    json_write(out/'resolved_config.json', config)
    (out/'command.txt').write_text(shlex.join([sys.executable] + sys.argv) + '\n')
    (out/'code_revision.txt').write_text(revision+'\n')
    environment = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
        torchvision=__import__('torchvision').__version__, cuda=torch.version.cuda,
        scipy=__import__('scipy').__version__, pillow=__import__('PIL').__version__,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpus=[dict(index=i, name=torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())])
    json_write(out/'environment.json', environment)
    json_write(out/'existing_results.json', sorted(str(p.relative_to(ROOT)) for p in (ROOT/'outputs').rglob('args.json')
                                                   if 'heatmap' not in str(p)))
    return dict(run_id=out.as_posix(), variant=args.heatmap_variant, protocol='place_train_val_b0_repro',
        code_revision=revision, dirty_diff_hash=digest(diff.encode()),
        baseline_revision=git('rev-parse','exp/b0-repro').strip(),
        source_manifest_hash=file_hash(out/'source_manifest.json'), config_hash=file_hash(out/'resolved_config.json'),
        evaluator_hash=file_hash(ROOT/'evaluation/cafe_eval.py'), seed=args.random_seed,
        init_source=cli.checkpoint or 'ImageNet ResNet18 + paired seeded B0 heads', init_hash=None,
        membership_normalizer='softmax_K', target_type='Gaussian member max' if args.heatmap_variant!='off' else None,
        mask_resolution=[23,40] if args.heatmap_variant!='off' else None, K=args.num_group_tokens, T=args.num_frame,
        threshold_protocol='B0 argmax K then max>0.5; no-group order unchanged',
        epochs_or_steps=cli.steps if cli.stage!='full' else args.epochs, effective_batch=args.batch,
        precision='fp32', checkpoint_path=None, checkpoint_selection_rule='last bounded step' if cli.stage!='full' else args.selection_metric,
        group_map_1_0=None, group_map_0_5=None, outlier_miou=None, peak_memory=None, status='running', failure_reason=None,
        stage=cli.stage, wall_time=None, split_hash=None, data_manifest_hash=None)


def dataset_manifest(train_set, val_set, out):
    manifest = {}
    for name, dataset in (('train', train_set), ('val', val_set)):
        rows = []
        for clip in sorted(dataset.frames):
            ann = Path(dataset.image_path)/str(clip[0])/str(clip[1])/'ann.json'
            rows.append(dict(clip=list(clip), annotation_hash=file_hash(ann),
                             frames=dataset.anns[clip]['num_frames'], groups=len(dataset.anns[clip]['activities']),
                             actors=len(dataset.anns[clip]['actions'])))
        manifest[name] = rows
    json_write(out/'data_manifest.json', manifest)
    split = dict(train=TRAIN_CAFE_P, val=VAL_CAFE_P, test=TEST_CAFE_P)
    json_write(out/'split.json', split)
    return file_hash(out/'data_manifest.json'), file_hash(out/'split.json')


def evaluate(model, loader, args, out, device, suffix='val', probe=False):
    model.eval()
    prediction_path = out/('prediction_export_'+suffix+'.txt')
    prediction_path.write_text('')
    expected, probe_rows, diagnostic_rows = set(), [], []
    for batch_index, batch in enumerate(loader):
        with torch.no_grad():
            images, targets, infos, boxes, dummy, valid = get_batch(batch, device)
            outputs = model(images, boxes, dummy, actor_frame_valid=valid)
            make_txt(boxes, infos, outputs, {i:i for i in range(1,25)}, str(prediction_path), args)
            for info in infos:
                expected.add(make_clip_key(make_image_key(info['vid'], info['sid'], info['fid'][0])))
            if probe:
                for target in targets:
                    target['dummy_idx'] = (target['actions'] != args.num_class+1).long()
                assignment = __import__('models.group_matcher', fromlist=['build_group_matcher']).build_group_matcher(args)(outputs, targets)
                _, responses = roi_responses(outputs['group_spatial_attention'], boxes, valid,
                                             args.heatmap_roi_size, args.heatmap_roi_sampling)
                for b, (slots, groups) in enumerate(assignment):
                    for slot, group in zip(slots, groups):
                        members = targets[b]['members'][0,group].bool()
                        actor_valid = valid[b].any(0)
                        probe_rows.append(dict(clip=[infos[b]['vid'],infos[b]['sid']], slot=int(slot), group=int(group),
                            member_response=responses[b,slot,members & actor_valid].cpu().tolist(),
                            nonmember_response=responses[b,slot,~members & actor_valid].cpu().tolist()))
                if batch_index < 4:
                    np.savez_compressed(out/('attention_'+str(batch_index)+'.npz'),
                        attention=outputs['group_spatial_attention'].cpu().numpy(),boxes=boxes.cpu().numpy())
            if 'group_masks' in outputs:
                for target in targets:
                    target['dummy_idx']=(target['actions']!=args.num_class+1).long()
                prepared=prepare_targets(outputs,targets,args)
                matched=HeatmapMatcher(args)(outputs,prepared)
                member_scores,nonmember_scores,ious=[],[],[]
                tp=fp=fn=0
                mean_response=outputs['spatial_response'].sum(2)/valid.sum(1)[:,None].clamp(min=1)
                for b,(slots,groups) in enumerate(matched):
                    for slot,group in zip(slots,groups):
                        actor_valid=valid[b].any(0)
                        truth=targets[b]['members'][0,group].bool() & actor_valid
                        prediction=(outputs['membership'][b,slot]>args.group_threshold) & actor_valid
                        tp+=int((prediction & truth).sum()); fp+=int((prediction & ~truth).sum()); fn+=int((~prediction & truth).sum())
                        ious.append(float((prediction & truth).sum())/max(int((prediction | truth).sum()),1))
                        member_scores.extend(mean_response[b,slot,truth].cpu().tolist())
                        nonmember_scores.extend(mean_response[b,slot,~truth & actor_valid].cpu().tolist())
                diagnostic_rows.append(dict(mask_min=float(outputs['group_masks'].min()), mask_max=float(outputs['group_masks'].max()),
                    mask_mean=float(outputs['group_masks'].mean()), valid_frames=int(valid.sum()),
                    membership_saturation=float((outputs['membership'].max(1).values>.99).float().mean()),
                    member_tp=tp,member_fp=fp,member_fn=fn,matched_set_ious=ious,
                    member_responses=member_scores,nonmember_responses=nonmember_scores,
                    no_group_slots=int((outputs['pred_activities'].argmax(-1)==args.num_class).sum())))
                if batch_index < 4:
                    np.savez_compressed(out/('heatmaps_'+suffix+'_'+str(batch_index)+'.npz'),
                        masks=outputs['group_masks'].cpu().numpy(),response=outputs['spatial_response'].cpu().numpy(),
                        boxes=boxes.cpu().numpy(),membership=outputs['membership'].cpu().numpy(),
                        activities=outputs['pred_activities'].cpu().numpy(),
                        gt_members=targets[0]['members'].cpu().numpy(),gt_activities=targets[0]['activities'].cpu().numpy(),
                        gt_heatmaps=prepared[0]['group_heatmaps'].cpu().numpy(),
                        matched_slots=matched[0][0].cpu().numpy(),matched_groups=matched[0][1].cpu().numpy(),
                        clip=np.array([infos[0]['vid'],infos[0]['sid']]),frame_ids=np.array(infos[0]['fid']))
        if batch_index % 100 == 0:
            print('evaluation batch',batch_index, 'clips',len(expected),flush=True)
    observed = {make_clip_key(make_image_key(*line.split()[:3])) for line in prediction_path.read_text().splitlines()}
    if observed != expected:
        raise RuntimeError('evaluator export coverage mismatch: '+str(expected-observed))
    # Official evaluator counts only predicted clip keys: explicitly restrict
    # GT and verify full selected-clip coverage instead of silently dropping GT.
    gt_path = out/('groundtruth_'+suffix+'.txt')
    with open(args.groundtruth) as source:
        gt_path.write_text(''.join(line for line in source if make_clip_key(make_image_key(*line.split()[:3])) in expected))
    with open(gt_path) as gt, open(args.labelmap) as labels, open(prediction_path) as predictions:
        evaluator_args = argparse.Namespace(eval_type=args.eval_type,groundtruth=gt,labelmap=labels)
        result = GAD_Evaluation(evaluator_args).evaluate(predictions)
    # Undefined outlier IoU (no outliers in either set) remains null, not zero.
    result = {key: (None if np.isscalar(value) and not np.isfinite(value) else value) for key,value in result.items()}
    result.update(clip_count=len(expected),prediction_hash=file_hash(prediction_path))
    json_write(out/('metrics_'+suffix+'.json'),result)
    json_write(out/('heatmap_diagnostics_'+suffix+'.json'),diagnostic_rows)
    if probe:
        json_write(out/'attention_probe.json',dict(aggregation='last layer, native head average, no occupancy sigmoid',samples=probe_rows))
    return result


def oracle(dataset, indices, args, ratio):
    rows = []
    for idx in indices:
        clip = dataset.frames[idx]
        selected = dataset.select_frames(clip)
        # Use actual transformed targets. Image loading also verifies files.
        _, target, _ = dataset.load_samples(selected)
        boxes, valid, members = target['boxes'], target['actor_frame_valid'], target['members'][0]
        maps, group_valid = gaussian_targets(boxes,members,valid,23,40,ratio,args.heatmap_sigma_min,args.heatmap_sigma_max)
        _, response = roi_responses(maps[None],boxes[None],valid[None],args.heatmap_roi_size,args.heatmap_roi_sampling)
        for g in range(len(members)):
            actor_valid = valid.any(0)
            if not group_valid[g].any():
                continue
            active = boxes[...,2]*boxes[...,3]
            for actor in range(len(actor_valid)):
                if actor_valid[actor]:
                    other = boxes[0,:, :2] - boxes[0,actor,:2]
                    distance = other.norm(dim=-1)
                    distance[actor] = float('inf')
                    rows.append(dict(clip=list(clip),group=g,actor=actor,member=bool(members[g,actor]),
                        response=float(response[0,g,actor]),response_vector=response[0,:,actor].tolist(),
                        area=float(active[:,actor][valid[:,actor]].mean()),
                        nearest_center=float(distance[valid[0]].min()) if valid[0].sum()>1 else None))
    return rows


def oracle_metrics(rows, threshold):
    tp=fp=fn=0
    sets = {}
    for row in rows:
        predicted = row['response'] > threshold
        tp += int(predicted and row['member']); fp += int(predicted and not row['member']); fn += int(not predicted and row['member'])
        key = tuple(row['clip'])+(row['group'],)
        truth,prediction = sets.setdefault(key,(set(),set()))
        if row['member']: truth.add(row['actor'])
        if predicted: prediction.add(row['actor'])
    ious = [len(gt & pred)/len(gt | pred) for gt,pred in sets.values() if gt | pred]
    return dict(threshold=float(threshold),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1),
        set_group_iou=float(np.mean(ious)) if ious else None, exact_group_fraction=float(np.mean(np.array(ious)==1)) if ious else None,
        groups=len(sets), actors=len(rows))


def run_oracle(train_set,val_set,args,out,cli):
    train_set.is_training=False
    report = {}
    for ratio in (.15,.25):
        train = oracle(train_set,subset_indices(train_set,cli.train_clips),args,ratio)
        val = oracle(val_set,subset_indices(val_set,cli.eval_clips),args,ratio)
        thresholds = np.linspace(0,1,201)
        selected = max((oracle_metrics(train,float(t)) for t in thresholds),key=lambda m:m['f1'])
        result = dict(train=selected,diagnostic_val=oracle_metrics(val,selected['threshold']),
            calibration='train-only global raw ROI threshold; NOT a model mAP',
            train_rows=train,val_rows=val)
        def softmax_rows(rows,alpha):
            converted=[]
            for row in rows:
                response=torch.zeros(args.num_group_tokens)
                response[:len(row['response_vector'])]=torch.tensor(row['response_vector'])
                probability=(response*alpha).softmax(0)
                copy=dict(row)
                copy['response']=float(probability.max()) if int(probability.argmax())==row['group'] else 0.
                converted.append(copy)
            return converted
        candidates={alpha:oracle_metrics(softmax_rows(train,alpha),args.group_threshold) for alpha in (5.,10.,20.,40.)}
        chosen=max(candidates,key=lambda alpha:(candidates[alpha]['f1'],-alpha))
        result['h2_s_oracle']=dict(alpha_candidates_train=candidates,selected_alpha=chosen,
            diagnostic_val=oracle_metrics(softmax_rows(val,chosen),args.group_threshold),
            protocol='GT target slots + zero occupancy unmatched slots; softmax over all 12, B0 threshold 0.5')
        members = [r['response'] for r in train if r['member']]
        nonmembers = [r['response'] for r in train if not r['member']]
        result['alpha_train_suggestion'] = float(math.log(11)/max(np.median(members)-np.median(nonmembers),.01))
        result['val_buckets'] = {name:oracle_metrics([r for r in val if condition(r)],selected['threshold'])
            for name,condition in dict(near=lambda r:r['nearest_center'] is not None and r['nearest_center']<.1,
                far=lambda r:r['nearest_center'] is None or r['nearest_center']>=.1,
                small=lambda r:r['area']<.01,large=lambda r:r['area']>=.01).items()}
        result['bucket_definition']='actor-restricted universe; both-empty group sets omitted from IoU average'
        report[str(ratio)] = result
    json_write(out/'oracle.json',report)
    return report


def train_bounded(model, criterion, dataset, val_loader, args, cli, out, device):
    core=model.module if isinstance(model,torch.nn.DataParallel) else model
    fixed = cli.stage in ('smoke','overfit','pilot')
    dataset.is_training = cli.stage != 'overfit'
    indices = subset_indices(dataset,cli.train_clips) if fixed else list(range(len(dataset)))
    if len(indices) < args.batch:
        raise ValueError('train subset must contain at least effective_batch clips')
    generator = torch.Generator().manual_seed(args.random_seed)
    training_data = Subset(dataset,indices)
    if cli.cache_fixed_overfit:
        training_data,cache_info=cache_fixed_samples(dataset,indices)
        json_write(out/'fixed_input_cache.json',cache_info)
        print('Cached fixed overfit inputs:',len(training_data),'clips,',cache_info['image_bytes'],'image bytes',flush=True)
    loader = DataLoader(training_data,batch_size=args.batch,shuffle=True,drop_last=True,
                        num_workers=0 if cli.cache_fixed_overfit else cli.workers,collate_fn=collate,generator=generator,
                        persistent_workers=cli.persistent_workers)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,betas=(.9,.999),eps=1e-8,weight_decay=args.weight_decay)
    protocol_steps_per_epoch = len(dataset)//args.batch
    records, step, best = [],0,-float('inf')
    limit = cli.steps
    if cli.stage=='smoke': limit=min(limit,3)
    if cli.stage=='overfit' and limit>500: raise ValueError('overfit bound is 500')
    if cli.stage=='pilot' and limit>1000: raise ValueError('pilot bound is 1000')
    if cli.stage=='full': limit=args.epochs*len(loader)
    started = time.monotonic()
    epoch = 0
    (out/'checkpoints').mkdir()
    sample_order = []
    train_eval_loader = DataLoader(training_data,batch_size=args.test_batch,shuffle=False,
                                  num_workers=0,collate_fn=collate) if cli.train_eval_every else None
    def train_diagnostic(current_step):
        # Evaluation must not perturb the training RNG stream.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if device.type=='cuda' else []):
            suffix='train_step'+str(current_step)
            metrics=evaluate(model,train_eval_loader,args,out,device,suffix=suffix)
        if core.mask_head:
            rows=json.loads((out/('heatmap_diagnostics_'+suffix+'.json')).read_text())
            members=[value for row in rows for value in row['member_responses']]
            nonmembers=[value for row in rows for value in row['nonmember_responses']]
            metrics.update(member_response_mean=float(np.mean(members)) if members else None,
                nonmember_response_mean=float(np.mean(nonmembers)) if nonmembers else None,
                no_group_slots=sum(row['no_group_slots'] for row in rows),
                member_tp=sum(row['member_tp'] for row in rows),member_fp=sum(row['member_fp'] for row in rows),
                member_fn=sum(row['member_fn'] for row in rows))
        with open(out/'train_evaluation_curve.jsonl','a') as stream:
            stream.write(json.dumps(dict(step=current_step,metrics=metrics),default=json_default,allow_nan=False)+'\n')
        print('TRAIN_DIAGNOSTIC',current_step,json.dumps(metrics,default=json_default,allow_nan=False),flush=True)
    if train_eval_loader is not None:
        train_diagnostic(0)
    while step < limit:
        for batch in loader:
            # Small subset passes are not full B0 epochs. Preserve the original
            # 402-step epoch schedule for bounded runs too (same first 402 steps).
            protocol_epoch = step//protocol_steps_per_epoch
            for group in optimizer.param_groups:
                group['lr'] = b0_learning_rate(args,protocol_epoch)
            model.train()
            optimizer.zero_grad()
            losses, grad_norm = {},0.
            images,targets,infos = batch
            sample_order.append([[info['vid'],info['sid'],info['fid']] for info in infos])
            micro = args.micro_batch or args.batch
            for start in range(0,len(images),micro):
                end=min(start+micro,len(images))
                x,tgt,info,boxes,dummy,valid = get_batch((images[start:end],targets[start:end],infos[start:end]),device)
                outputs = model(x,boxes,dummy,actor_frame_valid=valid)
                terms = criterion(outputs,tgt,log=False)
                loss=sum(terms[key]*weight for key,weight in criterion.weight_dict.items())
                if not torch.isfinite(loss): raise FloatingPointError('non-finite training loss: '+str(terms))
                (loss*(end-start)/len(images)).backward()
                for key,value in terms.items(): losses[key]=losses.get(key,0)+float(value.detach())*(end-start)/len(images)
            grads = [p.grad.detach().norm().square() for p in model.parameters() if p.grad is not None]
            grad_norm=float(torch.stack(grads).sum().sqrt())
            if not math.isfinite(grad_norm): raise FloatingPointError('non-finite gradient')
            optimizer.step()
            step+=1
            if device.type=='cuda': torch.cuda.synchronize(device)
            record=dict(step=step,epoch=epoch,protocol_epoch=protocol_epoch,lr=optimizer.param_groups[0]['lr'],
                losses=losses,gradient_norm=grad_norm,seconds=time.monotonic()-started,
                mask_range=[float(outputs['group_masks'].min()),float(outputs['group_masks'].max())] if 'group_masks' in outputs else None,
                mask_grad_norm=sum(float(p.grad.norm()) for p in core.mask_head.parameters() if p.grad is not None) if core.mask_head else None,
                cost_scales=criterion.group_matcher.last_cost_scales,
                peak_memory_mb=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None)
            records.append(record)
            with open(out/'steps.jsonl','a') as stream: stream.write(json.dumps(record)+'\n')
            if step==1 or step%10==0 or step==limit:
                print(json.dumps(record),flush=True)
            if train_eval_loader is not None and step%cli.train_eval_every==0:
                torch.save(dict(state_dict=core.state_dict(),optimizer=optimizer.state_dict(),step=step,args=vars(args)),
                           out/'checkpoints'/('step'+str(step)+'.pth'))
                train_diagnostic(step)
            if step>=limit: break
        epoch+=1
        if cli.stage=='full':
            metrics=evaluate(model,val_loader,args,out,device,'epoch'+str(epoch))
            if metrics['group_mAP_1.0']>best:
                best=metrics['group_mAP_1.0']
                torch.save(dict(state_dict=core.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                                metrics=metrics,args=vars(args)),out/'checkpoints'/'best_val.pth')
    checkpoint=out/'checkpoints'/'last.pth'
    torch.save(dict(state_dict=core.state_dict(),optimizer=optimizer.state_dict(),step=step,args=vars(args)),checkpoint)
    # Verify actual serialized state reload before evaluation export.
    restored=torch.load(checkpoint,map_location=device)
    core.load_state_dict(restored['state_dict'],strict=True)
    json_write(out/'sample_order.json',sample_order)
    json_write(out/'training_summary.json',dict(steps=step,seconds=time.monotonic()-started,
        seconds_per_step=(time.monotonic()-started)/step,first_loss=records[0]['losses'],last_loss=records[-1]['losses'],
        protocol_steps_per_epoch=protocol_steps_per_epoch,
        learning_rate_protocol='original B0 epochs counted by full TRAIN optimizer steps, not subset passes',
        cached_fixed_inputs=cli.cache_fixed_overfit, train_eval_every=cli.train_eval_every,
        persistent_workers=cli.persistent_workers,
        micro_batch=args.micro_batch, microbatch_protocol_warning='BatchNorm/legacy consistency reduction differ from full B0 batch' if args.micro_batch else None))
    if cli.stage=='full':
        checkpoint=out/'checkpoints'/'best_val.pth'
        core.load_state_dict(torch.load(checkpoint,map_location=device)['state_dict'],strict=True)
    return str(checkpoint)


def update_registry(record):
    path=ROOT/'results/heatmap/run_registry.csv'
    path.parent.mkdir(parents=True,exist_ok=True)
    # Concurrent jobs use disjoint GPUs, but share the registry. Lock and
    # atomically replace it so neither job drops historical/current rows.
    with open(path.with_suffix('.lock'),'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=[]
        if path.exists():
            with open(path) as stream: rows=list(csv.DictReader(stream))
        rows=[row for row in rows if row['run_id']!=record['run_id']]
        fields=list(record)
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
        rows.append(record)
        temporary=path.with_suffix('.'+str(os.getpid())+'.tmp')
        with open(temporary,'w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/heatmap/b0.json')
    parser.add_argument('--stage',choices=['audit','oracle','evaluate','smoke','overfit','pilot','full'],required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--parallel-gpus',default='',help='logical GPU indices within inherited allocation, e.g. 0,1,2')
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--train-clips',type=int,default=64)
    parser.add_argument('--eval-clips',type=int,default=16,help='0 means all VAL, never TEST')
    parser.add_argument('--batch',type=int)
    parser.add_argument('--micro-batch',type=int)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--checkpoint')
    parser.add_argument('--allow-unused-bbox-checkpoint',action='store_true',
                        help='allow only obsolete bbox_emb keys, absent from frozen B0 forward; recorded explicitly')
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--cache-fixed-overfit',action='store_true',help='cache deterministic fixed TRAIN inputs on CPU, overfit only')
    parser.add_argument('--train-eval-every',type=int,default=0,help='train-only overfit diagnostics/checkpoints every N steps, including step0')
    parser.add_argument('--persistent-workers',action='store_true',help='keep training input workers alive across subset passes; pair all configurations consistently')
    cli=parser.parse_args()
    os.chdir(str(ROOT))
    args=argparse.Namespace(**json.loads(Path(cli.config).read_text()))
    for attr,value in (('batch',cli.batch),('micro_batch',cli.micro_batch),('random_seed',cli.seed)):
        if value is not None: setattr(args,attr,value)
    if not args.val_mode or args.split!='place' or args.use_mae or not args.no_mae or not args.research_b0_repro:
        raise ValueError('first-round runner is frozen to place TRAIN->VAL, no VideoMAE')
    frozen = dict(backbone='resnet18',num_frame=5,num_group_tokens=12,hidden_dim=256,
        image_width=1280,image_height=720,optimizer='adam',lr_schedule='linear',
        lr=1e-5,max_lr=1e-4,warmup_epochs=5,epochs=30,weight_decay=0,
        backbone_lr_scale=1,precision='fp32',gradient_clipping=False)
    changed = {key:(value,getattr(args,key,None)) for key,value in frozen.items() if getattr(args,key,None)!=value}
    if changed:
        raise ValueError('frozen first-round B0 parameters differ (expected, actual): '+str(changed))
    if cli.steps<1 or args.batch<1 or cli.train_clips<1 or cli.eval_clips<0:
        raise ValueError('invalid bounded budget')
    if cli.train_eval_every<0 or ((cli.cache_fixed_overfit or cli.train_eval_every) and cli.stage!='overfit'):
        raise ValueError('fixed-input cache and train evaluation curve are overfit-only')
    if cli.persistent_workers and (cli.workers<1 or cli.cache_fixed_overfit):
        raise ValueError('persistent workers require positive workers and no fixed CPU cache')
    if (cli.stage=='overfit' and cli.steps>500) or (cli.stage=='pilot' and cli.steps>1000):
        raise ValueError('exceeds bounded overfit/pilot step budget')
    if cli.stage=='full' and args.micro_batch:
        raise ValueError('Full B0 protocol requires unsplit effective batch; microbatch runs are diagnostics')
    if cli.stage=='full' and (cli.eval_clips!=0 or args.batch!=16 or args.selection_metric!='group_mAP_1.0'):
        raise ValueError('Full protocol requires --eval-clips 0, batch16, selection_metric group_mAP_1.0')
    args.attention_probe=cli.probe
    out=Path(cli.output)
    out.mkdir(parents=True,exist_ok=False)
    log=open(out/'train.log','w',buffering=1)
    class Tee:
        def __init__(self,console): self.console=console
        def write(self,value): self.console.write(value); log.write(value)
        def flush(self): self.console.flush(); log.flush()
    sys.stdout=Tee(sys.stdout); sys.stderr=Tee(sys.stderr)
    started=time.monotonic()
    record=record_context(out,args,cli)
    update_registry(record)
    try:
        seed_all(args.random_seed)
        torch.set_num_threads(4)
        train_set,val_set=read_dataset(args)
        # Canonical subset identities, independent of os.listdir order.
        train_set.frames.sort(); val_set.frames.sort()
        record['data_manifest_hash'],record['split_hash']=dataset_manifest(train_set,val_set,out)
        json_write(out/'selected_clips.json',dict(train=[train_set.frames[i] for i in subset_indices(train_set,0 if cli.stage=='full' else cli.train_clips)],
            val=[val_set.frames[i] for i in subset_indices(val_set,cli.eval_clips)]))
        if cli.stage=='oracle':
            run_oracle(train_set,val_set,args,out,cli)
        elif cli.stage=='audit':
            json_write(out/'dataset_audit.json',dict(train_clips=len(train_set),val_clips=len(val_set),
                max_actors=max(len(a['actions']) for a in train_set.anns.values()),
                tracks_hash=file_hash(Path(args.data_path)/'cafe/gt_tracks.pkl'),gt_hash=file_hash(args.groundtruth),
                note='all-individual clips excluded by cafe_read_annotations; TEST not loaded'))
        else:
            device=torch.device(cli.device)
            if device.type=='cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable in this execution environment')
            model,criterion=build_model(args)
            if cli.checkpoint:
                loaded=torch.load(cli.checkpoint,map_location='cpu')
                state={key.removeprefix('module.') if hasattr(key,'removeprefix') else key[7:] if key.startswith('module.') else key:value
                       for key,value in loaded.get('state_dict',loaded).items()}
                if cli.allow_unused_bbox_checkpoint:
                    removed=[key for key in state if key.startswith('bbox_emb.')]
                    for key in removed: del state[key]
                    json_write(out/'checkpoint_compatibility.json',dict(ignored_keys=removed,
                        reason='bbox_emb is absent from exp/b0-repro model and forward; released reference only'))
                incompatible=model.load_state_dict(state,strict=False)
                if incompatible.unexpected_keys or any(not key.startswith('mask_head.') for key in incompatible.missing_keys):
                    raise RuntimeError(str(incompatible))
                record['checkpoint_source_hash']=file_hash(cli.checkpoint)
            record['init_hash']=tensor_hash(model)
            rng=torch.get_rng_state().numpy().tobytes()
            if torch.cuda.is_available():
                rng+=b''.join(state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all())
            record['initial_rng_hash']=digest(rng)
            record['mask_head_parameters']=sum(p.numel() for p in model.mask_head.parameters()) if model.mask_head else 0
            record['total_parameters']=sum(p.numel() for p in model.parameters())
            model.to(device); criterion.to(device)
            if cli.parallel_gpus:
                devices=[int(value) for value in cli.parallel_gpus.split(',')]
                if device.type!='cuda' or devices[0]!=(device.index or 0) or any(i>=torch.cuda.device_count() for i in devices):
                    raise ValueError('parallel GPUs must exist within inherited allocation and begin at --device')
                model=torch.nn.DataParallel(model,device_ids=devices)
                record['parallel_gpus']=devices
            count=cli.eval_clips or len(val_set)
            val_loader=DataLoader(Subset(val_set,subset_indices(val_set,count)),batch_size=args.test_batch,
                                  shuffle=False,num_workers=cli.workers,collate_fn=collate)
            if cli.stage!='evaluate':
                record['checkpoint_path']=train_bounded(model,criterion,train_set,val_loader,args,cli,out,device)
            if cli.stage=='overfit':
                train_set.is_training=False
                overfit_loader=DataLoader(Subset(train_set,subset_indices(train_set,cli.train_clips)),
                    batch_size=args.test_batch,shuffle=False,num_workers=cli.workers,collate_fn=collate)
                evaluate(model,overfit_loader,args,out,device,suffix='overfit_train')
            metrics=evaluate(model,val_loader,args,out,device,probe=cli.probe)
            if cli.stage in ('smoke','evaluate'):
                repeat=evaluate(model,val_loader,args,out,device,suffix='repeat')
                if metrics['prediction_hash']!=repeat['prediction_hash']:
                    raise AssertionError('Repeated evaluation prediction content changed')
            record.update(group_map_1_0=metrics['group_mAP_1.0'],group_map_0_5=metrics['group_mAP_0.5'],
                          outlier_miou=metrics['outlier_mIoU'])
            if device.type=='cuda': record['peak_memory']=torch.cuda.max_memory_allocated(device)/1024**2
        record['status']='completed'
    except BaseException as error:
        record.update(status='failed',failure_reason=type(error).__name__+': '+str(error))
        (out/'failure.txt').write_text(traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        record['wall_time']=time.monotonic()-started
        json_write(out/'run_record.json',record)
        update_registry(record)
        print('RUN_RECORD',json.dumps(record),flush=True)


if __name__=='__main__':
    main()
