"""Audit, freeze, validate, generate and inspect bounded CAFE person masks."""
import argparse
import collections
import fcntl
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
from util.sam_person import (SCHEMA, atomic_json, canonical, digest, file_hash,
    frame_fingerprint, cache_paths, read_frame, write_frame, feature_masks)


def read_manifest(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def version_payload(c):
    return {k: c[k] for k in ('sam_family', 'sam_commit', 'sam_source_hash',
        'checkpoint_sha256', 'sam_config_hash', 'prompt_policy', 'multimask_output',
        'precision', 'mask_threshold', 'postprocess_options', 'schema_version')}


def version(c):
    return 'raw_' + digest(canonical(version_payload(c)))[:20]


def load_config(path):
    # JSON is valid YAML, allowing the original environments to run without
    # installing PyYAML or upgrading any dependency.
    c = json.loads(Path(path).read_text())
    for key in ('project_root', 'data_root', 'tracks_path', 'sam_python', 'sam_repo', 'checkpoint'):
        if not c.get(key) or not Path(c[key]).exists():
            raise ValueError('unresolved required path: ' + key)
    if c['schema_version'] != SCHEMA or c['protocol'] != 'place_train_val_b0_repro':
        raise ValueError('unsupported schema/protocol')
    if c['box_coordinate_convention'] != 'normalized_cxcywh_to_original_pixel_xyxy':
        raise ValueError('unknown box coordinates')
    if c['prompt_policy'] != 'box_only' or c['multimask_output'] or c['precision'] != 'fp32':
        raise ValueError('first run is FP32 single box-only candidate')
    if c['sam_family'] != 'sam_v1_vit_b' or Path(c['checkpoint']).name != 'sam_vit_b_01ec64.pth':
        raise ValueError('this offline adapter supports verified SAM v1 ViT-B only')
    if c['max_frames'] > 200 or c['max_actor_prompts'] > 3000 or c['max_gpu_minutes'] > 60:
        raise ValueError('generation plan exceeds first-round limits')
    if any(not np.isfinite(c[k]) or c[k]<=0 for k in ('max_frames','max_actor_prompts','max_gpu_minutes')):
        raise ValueError('budget limits must be finite and positive')
    if sum(c['per_split_limits'].values())>c['max_frames'] or not 0<c['empty_eps']<1:
        raise ValueError('split plan/empty epsilon invalid')
    if c['cache_version'] != version(c):
        raise ValueError('model/config/prompt cache version mismatch')
    return c


def datasets(c):
    import pickle
    from dataloader.dataloader import TRAIN_CAFE_P, VAL_CAFE_P
    from dataloader.cafe import CafeDataset, cafe_read_annotations
    args = argparse.Namespace(**json.loads((ROOT/'configs/heatmap/b0.json').read_text()))
    args.data_path = str(Path(c['data_root']).parent)
    args.tracks_path = c['tracks_path']
    # Reuse the independently frozen full TRAIN/VAL manifest as the sampling
    # frame, then call the actual annotation parser and dataset. This avoids
    # thousands of unrelated NFS reads in every bounded pilot process.
    source=json.loads(Path(c['sampling_split_manifest']).read_text())
    if file_hash(c['sampling_split_manifest'])!=c['sampling_split_manifest_sha256']:
        raise ValueError('sampling split manifest changed')
    rng=random.Random(c['sample_seed'])
    selected={}
    for split in ('train','val'):
        proxy=argparse.Namespace(frames=[tuple(r['clip']) for r in source[split]])
        selected[split]=choose_clips(proxy,c['per_split_limits'][split]//args.num_frame,rng)
    with open(c['tracks_path'],'rb') as stream:
        tracks=pickle.load(stream)
    result=[]
    for split,videos in (('train',TRAIN_CAFE_P),('val',VAL_CAFE_P)):
        anns=cafe_read_annotations(c['data_root'],videos,args.num_class,clip_ids=selected[split])
        if set(anns)!=set(selected[split]):
            raise ValueError('sampling manifest/actual retained clip mismatch')
        expected={tuple(r['clip']):r['annotation_hash'] for r in source[split]}
        for key in selected[split]:
            path=Path(c['data_root'])/str(key[0])/str(key[1])/'ann.json'
            if file_hash(path)!=expected[key]:raise ValueError('annotation differs from audited split manifest')
        result.append(CafeDataset(sorted(anns),anns,tracks,c['data_root'],args,is_training=False))
    train,val=result
    # Fixed segment centers are a declared bounded pilot input protocol for
    # BOTH variants, using the real B0 validation sampling method.
    train.is_training = False
    train.frames = sorted(train.frames)
    val.frames = sorted(val.frames)
    return args, train, val


def choose_clips(ds, count, rng):
    buckets = collections.defaultdict(list)
    for clip in sorted(ds.frames):
        buckets[clip[0]].append(clip)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    chosen = []
    while len(chosen) < count:
        progressed = False
        for vid in sorted(buckets):
            if buckets[vid] and len(chosen) < count:
                chosen.append(buckets[vid].pop())
                progressed = True
        if not progressed:
            break
    return chosen


def manifest(c, out):
    import torch
    args, train, val = datasets(c)
    all_rows, clips = [], {}
    split_records = json.loads(Path(c['sampling_split_manifest']).read_text())
    for split, ds in (('train', train), ('val', val)):
        selected = sorted(ds.frames)
        clips[split] = []
        for clip in selected:
            sampled = ds.select_frames(clip)
            _, target, info = ds.load_samples(sampled)
            clips[split].append(dict(clip=list(clip), frames=info['fid']))
            for i, fid in enumerate(info['fid']):
                path = Path(ds.image_path)/str(clip[0])/str(clip[1])/'images'/('frames_%d.jpg' % fid)
                with Image.open(path) as image:
                    if image.mode != 'RGB':
                        raise ValueError('B0 source must be RGB, got ' + image.mode)
                    w, h = image.size
                actors = []
                for actor_row in range(args.num_boxes):
                    if not target['actor_frame_valid'][i, actor_row]:
                        continue
                    cx, cy, bw, bh = target['boxes'][i, actor_row].tolist()
                    box = [float(v) for v in np.asarray([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2])*[w,h,w,h]]
                    if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                        raise ValueError('invalid real loader box')
                    actors.append(dict(actor_row=actor_row,
                        actor_uid='cafe/%d/%d/row_%d' % (clip[0], clip[1], actor_row),
                        actor_id_scope='video/clip/zero-based annotation row; track ID=row+1',
                        source_box_xyxy=box, box_hash=digest(canonical(box)),
                        box_source=c['actor_box_source']))
                all_rows.append(dict(schema_version=SCHEMA, dataset='CAFE', protocol=c['protocol'],
                    split=split, video_id=clip[0], clip_id=clip[1], source_frame_index=fid,
                    source_image_key='%d/%d/images/frames_%d.jpg' % (clip[0],clip[1],fid),
                    source_image_path=str(path), source_image_hash=file_hash(path), original_hw=[h,w],
                    actors=actors, invalid_actor_rows=[j for j in range(args.num_boxes)
                        if not target['actor_frame_valid'][i,j]],
                    sampling='real CafeDataset._sample_segment_centers * framesEach'))
            print('manifest', split, list(clip), flush=True)
    if len(all_rows) > c['max_frames'] or sum(len(r['actors']) for r in all_rows)>c['max_actor_prompts']:
        raise ValueError('manifest exceeds frame/prompt budget')
    hashes = {split:{r['source_image_hash'] for r in all_rows if r['split']==split} for split in ('train','val')}
    overlap = hashes['train'] & hashes['val']
    if overlap:
        raise ValueError('train/val duplicate image content in selected manifest')
    path = Path(out)
    if path.exists():
        raise FileExistsError('manifest is frozen; choose a new output')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b''.join(canonical(row)+b'\n' for row in all_rows))
    atomic_json(path.with_suffix('.splits.json'), split_records)
    atomic_json(path.with_suffix('.clips.json'), clips)
    atomic_json(path.with_suffix('.summary.json'), dict(frames=len(all_rows),
        actor_prompts=sum(len(r['actors']) for r in all_rows), selected_duplicate_hashes=len(overlap),
        train_val_clip_overlap=len({tuple(r['clip']) for r in split_records['train']} & {tuple(r['clip']) for r in split_records['val']}),
        full_split_image_overlap='not exhaustively hashed; selected raw frames checked',
        split_hash=digest(canonical(split_records)), manifest_sha256=file_hash(path),
        original_video_absolute_frame_mapping='unavailable; identity is clip-relative image key',
        torch=torch.__version__))


def validate(c, rows, assets=True):
    if assets:
        if file_hash(c['checkpoint']) != c['checkpoint_sha256']:
            raise ValueError('checkpoint hash mismatch')
        source = {str(p.relative_to(Path(c['sam_repo']))):file_hash(p)
                  for p in sorted(Path(c['sam_repo']).rglob('*.py'))}
        if digest(canonical(source)) != c['sam_source_hash']:
            raise ValueError('SAM source changed')
    keys = set()
    for row in rows:
        key = row['source_image_key']
        if key in keys or row['split'] not in ('train','val'):
            raise ValueError('duplicate source or forbidden split')
        keys.add(key)
        if file_hash(row['source_image_path']) != row['source_image_hash']:
            raise ValueError('source image fingerprint changed')
        with Image.open(row['source_image_path']) as image:
            if list(image.size[::-1]) != row['original_hw'] or image.mode != 'RGB':
                raise ValueError('source dimensions/color changed')
        uids = set()
        for actor in row['actors']:
            box = actor['source_box_xyxy']
            if actor['actor_uid'] in uids or digest(canonical(box)) != actor['box_hash']:
                raise ValueError('actor UID or box fingerprint mismatch')
            uids.add(actor['actor_uid'])
            if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError('degenerate box')
    print(json.dumps(dict(status='validated', frames=len(rows), actors=sum(len(r['actors']) for r in rows))))


class Provider:
    def __init__(self, c):
        import torch
        sys.path.insert(0, c['sam_repo'])
        from segment_anything import SamPredictor, sam_model_registry
        self.torch, self.c = torch, c
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = sam_model_registry['vit_b'](checkpoint=c['checkpoint']).eval().to(c['device'])
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if float(self.model.mask_threshold) != c['mask_threshold']:
            raise ValueError('SAM binary threshold mismatch')
        self.predictor = SamPredictor(self.model)

    def sync(self):
        if self.c['device'].startswith('cuda'):
            self.torch.cuda.synchronize()

    def encode(self, image):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('provider needs raw RGB uint8[H,W,3]')
        self.sync()
        start = time.monotonic()
        with self.torch.no_grad():
            self.predictor.set_image(image, image_format='RGB')
        self.sync()
        return time.monotonic()-start

    def predict(self, box, hw):
        self.sync()
        start = time.monotonic()
        with self.torch.no_grad():
            masks, scores, _ = self.predictor.predict(box=np.asarray(box, np.float32), multimask_output=False)
        self.sync()
        if masks.shape != (1,) + tuple(hw) or scores.shape != (1,):
            raise ValueError('SAM final mask candidate/HW axes unexpected')
        if not np.isfinite(scores).all():
            raise ValueError('nonfinite quality')
        return masks[0].astype(bool), float(scores[0]), time.monotonic()-start


def generate(c, rows, cli):
    import torch
    limit_frames = min(cli.max_frames or c['max_frames'], c['max_frames'])
    limit_prompts = min(cli.max_actor_prompts or c['max_actor_prompts'], c['max_actor_prompts'])
    limit_seconds = min(cli.max_gpu_minutes or c['max_gpu_minutes'], c['max_gpu_minutes'])*60
    ledger_path = Path(c['generation_ledger'])
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(ledger_path)+'.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(seconds=0., prompts=0, sessions=[])
        if ledger.get('active_since'):
            ledger['seconds'] += time.time()-ledger.pop('active_since')
            if ledger['sessions'] and ledger['sessions'][-1]['status']=='running':
                ledger['sessions'][-1].update(status='interrupted',failure_reason='process ended without finalization; elapsed gap and issued prompts conservatively charged')
        if ledger['seconds'] >= limit_seconds:
            atomic_json(ledger_path, ledger)
            print('partial: cumulative time budget exhausted')
            return
        existing, prompts = [], 0
        for row in rows:
            _, marker = cache_paths(c['cache_root'], c['cache_version'], row)
            if marker.exists():
                if not cli.resume:
                    raise FileExistsError('completed cache exists; use --resume')
                read_frame(c['cache_root'], c['cache_version'], row, verify_source=True)
                existing.append(row['source_image_key'])
                prompts += len(row['actors'])
        session = dict(start=time.time(), frames=[], load_seconds=None, status='running')
        ledger['sessions'].append(session)
        ledger['active_since'] = time.time()
        atomic_json(ledger_path, ledger)
        def charge(persist=True):
            now = time.time()
            ledger['seconds'] += now-ledger['active_since']
            ledger['active_since'] = now
            if persist:atomic_json(ledger_path, ledger)
        try:
            start = time.monotonic()
            provider = Provider(c)
            provider.sync()
            session['load_seconds'] = time.monotonic()-start
            session['environment']=dict(python=sys.version,torch=torch.__version__,
                torchvision=__import__('torchvision').__version__,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                device=c['device'],precision='fp32',
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32)
            charge()
            torch.cuda.reset_peak_memory_stats()
            for row in rows:
                if row['source_image_key'] in existing:
                    continue
                if (len(existing)+len(session['frames']) >= limit_frames or
                    prompts+len(row['actors']) > limit_prompts or
                    ledger['prompts']+len(row['actors']) > c['max_actor_prompts'] or
                    ledger['seconds'] >= limit_seconds):
                    break
                read_start = time.monotonic()
                image = np.asarray(Image.open(row['source_image_path']).convert('RGB'))
                read_seconds = time.monotonic()-read_start
                encode_seconds = provider.encode(image)
                charge()
                masks, records = [], []
                # Reserve the entire frame before issuing prompts. This is a
                # durable upper bound if interrupted mid-frame. Time recovery
                # uses the persisted active_since; no per-actor NFS renames are
                # necessary, and uncompleted reservations are never refunded.
                ledger['prompts'] += len(row['actors'])
                atomic_json(ledger_path, ledger)
                for actor in row['actors']:
                    if ledger['seconds'] >= limit_seconds:
                        # Leave this frame incomplete; attempted prompts remain
                        # charged in the persistent ledger, never silently reset.
                        session['status'] = 'partial'
                        return
                    try:
                        mask, score, seconds = provider.predict(actor['source_box_xyxy'], row['original_hw'])
                        ok = bool(mask.any())
                        status, failure = ('success', None) if ok else ('empty', 'raw mask empty')
                    except (ValueError, RuntimeError) as error:
                        mask = np.zeros(row['original_hw'], bool)
                        score, seconds, ok = None, None, False
                        status, failure = 'failed', repr(error)
                    charge(persist=False)
                    masks.append(mask)
                    records.append(dict(actor_uid=actor['actor_uid'], generation_status=status,
                        raw_mask_valid=ok, predicted_quality=score, selected_candidate=0,
                        failure_reason=failure, quality_flags=[], decoder_seconds=seconds,
                        mask_checksum=digest(mask.tobytes()), model=version_payload(c)))
                write_start = time.monotonic()
                write_frame(c['cache_root'], c['cache_version'], row,
                    np.stack(masks) if masks else np.zeros((0,)+tuple(row['original_hw']),bool), records)
                session['frames'].append(dict(key=row['source_image_key'], actors=len(masks),
                    image_encoder_seconds=encode_seconds, read_seconds=read_seconds,
                    write_seconds=time.monotonic()-write_start,
                    decoder_seconds=sum(r['decoder_seconds'] or 0 for r in records)))
                prompts += len(masks)
                charge()
                print('generated', row['source_image_key'], len(masks), 'cumulative seconds', round(ledger['seconds'],2), flush=True)
            session['status'] = 'complete' if len(existing)+len(session['frames'])==len(rows) else 'partial'
        except BaseException as error:
            session['status'], session['failure_reason'] = 'failed', repr(error)
            raise
        finally:
            charge()
            ledger.pop('active_since', None)
            session['peak_memory_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None
            atomic_json(ledger_path, ledger)


def qc(c, rows, cli):
    import torch
    from torchvision.ops import RoIAlign
    from models.sam_roi import filter_actor_rois
    from scipy.ndimage import label
    records, missing = [], []
    for row in rows:
        _, marker = cache_paths(c['cache_root'], c['cache_version'], row)
        if not marker.exists():
            missing.append(dict(source=row['source_image_key'], actors=len(row['actors'])))
            for actor in row['actors']:
                records.append(dict(split=row['split'], source=row['source_image_key'],
                    actor_uid=actor['actor_uid'], generation_status='unprocessed', raw_mask_valid=None))
            continue
        masks, meta = read_frame(c['cache_root'], c['cache_version'], row, verify_source=True)
        if not len(masks):
            continue
        h,w = row['original_hw']
        grid = feature_masks(masks, c['image_hw'], c['feature_hw'])
        boxes = torch.tensor([a['source_box_xyxy'] for a in row['actors']])
        boxes_feature = boxes/torch.tensor([w,h,w,h])*torch.tensor([c['feature_hw'][1],c['feature_hw'][0]]*2)
        align = RoIAlign((5,5),1.,1,True)
        roi_weights = align(grid[:,None],[b[None] for b in boxes_feature])[:,0]
        for i,(actor,mask,record) in enumerate(zip(row['actors'],masks,meta['records'])):
            x1,y1,x2,y2 = actor['source_box_xyxy']
            ix1,iy1,ix2,iy2 = max(0,int(np.floor(x1))),max(0,int(np.floor(y1))),min(w,int(np.ceil(x2))),min(h,int(np.ceil(y2)))
            area = int(mask.sum())
            overlap = [float((mask&m).sum()/max(1,(mask|m).sum())) for j,m in enumerate(masks) if j!=i]
            flags=[]
            if overlap and max(overlap)>.9: flags.append('near_duplicate_neighbor_proxy')
            if area and float(mask[iy1:iy2,ix1:ix2].sum())/area < .7: flags.append('outside_box_proxy')
            empty=bool(roi_weights[i].mean() <= c['empty_eps'])
            records.append(dict(split=row['split'],source=row['source_image_key'],actor_uid=actor['actor_uid'],
                generation_status=record['generation_status'],raw_mask_valid=record['raw_mask_valid'],
                failure_reason=record['failure_reason'],predicted_quality=record['predicted_quality'],
                pixel_iou=None,dice=None,mask_pixels=area,
                mask_box_area_ratio=area/max((x2-x1)*(y2-y1),1e-10),
                inside_box_fraction=float(mask[iy1:iy2,ix1:ix2].sum())/area if area else None,
                connected_components=int(label(mask)[1]),max_neighbor_mask_iou=max(overlap) if overlap else None,
                feature_roi_mean=float(roi_weights[i].mean()),feature_grid_empty=empty,
                usable=bool(record['raw_mask_valid'] and not empty),quality_flags=flags,
                actor_scale_area=(x2-x1)*(y2-y1)/(w*h),view=(row['video_id']-1)%4+1,
                occlusion='unannotated',review_status='unreviewed'))
    def summary(selected):
        attempted=[r for r in selected if r['generation_status']!='unprocessed']
        success=sum(bool(r['raw_mask_valid']) for r in attempted)
        return dict(total=len(selected),attempted=len(attempted),unprocessed=len(selected)-len(attempted),
            raw_success=success,failed=len(attempted)-success,
            processing_coverage=len(attempted)/len(selected) if selected else None,
            raw_success_rate=success/len(attempted) if attempted else None,
            usable_coverage=sum(r.get('usable',False) for r in attempted)/len(selected) if selected else None,
            fallback_rate=sum(not r.get('usable',False) for r in attempted)/len(attempted) if attempted else None,
            feature_grid_empty=sum(r.get('feature_grid_empty',False) for r in attempted),
            pixel_iou=None,dice=None)
    result=dict(status='partial' if missing else 'complete',evidence='proxies only; no independent person pixel GT',
        scope_requested=cli.scope,completed_scope=summary([r for r in records if r['generation_status']!='unprocessed']),
        denominator_note='all/records retain full manifest; raw success and fallback use attempted denominator; usable uses full manifest',
        manifest_frames=len(rows),completed_frames=len(rows)-len(missing),all=summary(records),
        splits={s:summary([r for r in records if r['split']==s]) for s in ('train','val')},
        frames={s:summary([r for r in records if r['source']==s]) for s in sorted({r['source'] for r in records})},
        scale_buckets={s:summary([r for r in records if r.get('actor_scale_area',-1)>=lo and r.get('actor_scale_area',-1)<hi])
            for s,lo,hi in [('small',0,.01),('medium',.01,.05),('large',.05,2)]},
        missing=missing,records=records)
    atomic_json(cli.out,result)
    print(json.dumps({k:result[k] for k in ('status','manifest_frames','completed_frames','all','splits')}),flush=True)
    if cli.require_complete and missing:
        raise RuntimeError('incomplete manifest: see explicit missing table')


def visualize(c, rows, cli):
    out=Path(cli.out)
    out.mkdir(parents=True,exist_ok=True)
    index=[]
    # Predeclared first 8 frames per split, plus every structural failure;
    # each actor is shown separately so neighboring identities are inspectable.
    counts=collections.Counter()
    for row in rows:
        _,marker=cache_paths(c['cache_root'],c['cache_version'],row)
        if not marker.exists(): continue
        masks,meta=read_frame(c['cache_root'],c['cache_version'],row)
        fail=any(not r['raw_mask_valid'] for r in meta['records'])
        if counts[row['split']]>=8 and not fail: continue
        counts[row['split']]+=1
        original=np.asarray(Image.open(row['source_image_path']).convert('RGB'))
        thumbs=[]
        for actor,mask,record in zip(row['actors'],masks,meta['records']):
            base=Image.fromarray(original.copy()); draw=ImageDraw.Draw(base)
            for neighbor in row['actors']:
                draw.rectangle(neighbor['source_box_xyxy'],outline=(160,160,160),width=2)
            draw.rectangle(actor['source_box_xyxy'],outline=(255,40,30),width=4)
            overlay=original.copy().astype(np.float32)
            overlay[mask]=overlay[mask]*.55+np.array([10,220,80])*.45
            overlay=Image.fromarray(overlay.astype(np.uint8))
            ImageDraw.Draw(overlay).rectangle(actor['source_box_xyxy'],outline=(255,40,30),width=4)
            width=480; height=round(width*original.shape[0]/original.shape[1])
            panel=Image.new('RGB',(width*2,height+24),'white')
            panel.paste(base.resize((width,height)),(0,24));panel.paste(overlay.resize((width,height)),(width,24))
            ImageDraw.Draw(panel).text((4,4),actor['actor_uid']+' '+record['generation_status'],fill='black')
            name='%s_%d_%d_%d_row%d.jpg'%(row['split'],row['video_id'],row['clip_id'],row['source_frame_index'],actor['actor_row'])
            panel.save(out/name)
            index.append('<figure><img loading="lazy" width="960" src="%s"><figcaption>%s</figcaption></figure>'%(name,actor['actor_uid']))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>SAM fixed actor gallery</title>'+''.join(index))
    for split in ('train','val'):
        subset=[r for r in rows if r['split']==split]
        if not subset:continue
        clip=(subset[0]['video_id'],subset[0]['clip_id'])
        track_rows=sorted([r for r in subset if (r['video_id'],r['clip_id'])==clip],key=lambda r:r['source_frame_index'])
        if len(track_rows)<2:continue
        common=set(a['actor_uid'] for a in track_rows[0]['actors'])
        for row in track_rows:common &= {a['actor_uid'] for a in row['actors']}
        for uid in sorted(common)[::max(1,len(common)-1)][:2]:
            timeline=Image.new('RGB',(480*len(track_rows),294),'white')
            complete=True
            for column,row in enumerate(track_rows):
                try:masks,meta=read_frame(c['cache_root'],c['cache_version'],row)
                except FileNotFoundError:complete=False;break
                i=next(i for i,a in enumerate(row['actors']) if a['actor_uid']==uid)
                rgb=np.asarray(Image.open(row['source_image_path']).convert('RGB'))
                paint=rgb.astype(float).copy();paint[masks[i]]=paint[masks[i]]*.55+np.array([10,220,80])*.45
                frame=Image.fromarray(paint.astype(np.uint8))
                ImageDraw.Draw(frame).rectangle(row['actors'][i]['source_box_xyxy'],outline='red',width=4)
                frame.thumbnail((480,270));timeline.paste(frame,(column*480,24))
                ImageDraw.Draw(timeline).text((column*480+4,4),uid+' fid='+str(row['source_frame_index']),fill='black')
            if complete:
                name='timeline_'+split+'_'+uid.replace('/','_')+'.jpg';timeline.save(out/name)
                with open(out/'index.html','a') as stream:stream.write('<p><a href="'+name+'">'+name+'</a></p>')
    print('gallery actor panels',len(index),flush=True)


def audit(c, out):
    result=dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        user_initial_diff_sha256=file_hash(ROOT/'outputs/sam_person/20261018_v1/user_initial.diff'),
        python=sys.version,executable=sys.executable,config=c,
        evaluator_hash=file_hash(ROOT/'evaluation/cafe_eval.py'),
        data_code_hashes={f:file_hash(ROOT/f) for f in ('dataloader/cafe.py','dataloader/dataloader.py','models/b0_repro.py')})
    atomic_json(out,result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['audit','manifest','validate','generate','qc','visualize'])
    p.add_argument('--config',required=True);p.add_argument('--manifest');p.add_argument('--out')
    p.add_argument('--max-frames',type=int);p.add_argument('--max-actor-prompts',type=int);p.add_argument('--max-gpu-minutes',type=float)
    p.add_argument('--resume',action='store_true');p.add_argument('--require-complete',action='store_true')
    p.add_argument('--scope',choices=['completed','all'],default='all')
    cli=p.parse_args()
    c=load_config(cli.config)
    log=ROOT/'docs/sam_person/commands_actual.md'
    with open(log,'a') as stream:
        env=' '.join(k+'='+shlex.quote(os.environ[k]) for k in ('CUDA_DEVICE_ORDER','CUDA_VISIBLE_DEVICES',
            'OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONPATH','TORCH_HOME') if k in os.environ)
        stream.write('\n```bash\n'+env+' '+shlex.join([sys.executable]+sys.argv)+'\n```\n')
    if cli.command=='audit': audit(c,cli.out);return
    if cli.command=='manifest': manifest(c,cli.out);return
    rows=read_manifest(cli.manifest)
    if cli.command=='validate': validate(c,rows);return
    if cli.command=='generate': validate(c,rows);generate(c,rows,cli);return
    if cli.command=='qc': qc(c,rows,cli);return
    visualize(c,rows,cli)


if __name__=='__main__':
    main()
