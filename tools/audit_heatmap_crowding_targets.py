"""Fixed TRAIN-only target geometry check; no model, optimizer, VAL or TEST.

The engineering gate is frozen before execution: >=1% of valid member-frames
must tighten, and >=2 IoU-crowded clips must change. A failed gate blocks full
training, not an invitation to select a new radius from validation results.
"""
import argparse
import csv
import json
from pathlib import Path
import pickle
import shlex
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from dataloader.cafe import CafeDataset, cafe_read_annotations, cafe_all_frames
from dataloader.dataloader import TRAIN_CAFE_P
from models.heatmap import gaussian_targets, frame_validity, xyxy_grid
from tools.run_heatmap_research import file_hash, json_write, subset_indices


def center_responses(maps, boxes):
    groups, frames, height, width = maps.shape
    grid = (2*boxes[...,:2]-1)[:,:,None,:].repeat(groups,1,1,1)
    return F.grid_sample(maps.reshape(groups*frames,1,height,width), grid,
                         mode='bilinear', padding_mode='zeros', align_corners=False).reshape(groups,frames,-1)


def analyze(boxes, members, valid, args):
    maps = [gaussian_targets(boxes,members,valid,23,40,args.heatmap_sigma_ratio,
                             args.heatmap_sigma_min,args.heatmap_sigma_max,mode)[0]
            for mode in ('box','nearest_center')]
    centers = boxes[...,:2]*boxes.new_tensor([40,23])
    distances = (centers[:,:,None]-centers[:,None,:]).square().sum(-1).sqrt()
    pairs = valid[:,:,None] & valid[:,None,:] & ~torch.eye(boxes.shape[1],dtype=torch.bool)[None]
    nearest = distances.masked_fill(~pairs,float('inf')).min(-1).values
    sigma_a = (boxes[...,2:]*boxes.new_tensor([40,23])*args.heatmap_sigma_ratio).clamp(args.heatmap_sigma_min,args.heatmap_sigma_max)
    sigma_b = torch.minimum(sigma_a,.35*nearest[...,None]).clamp(min=args.heatmap_sigma_min)
    grouped = members.bool().any(0)
    member_frames = valid & grouped[None]
    tightened = (sigma_b < sigma_a).any(-1) & member_frames
    labels = members.argmax(0); labels[~grouped] = -1
    relevant = pairs & (labels[:,None] != labels[None,:])[None] & (grouped[:,None] | grouped[None,:])[None]
    corners = xyxy_grid(boxes,1,1)
    intersection = (torch.minimum(corners[:,:,None,2:],corners[:,None,:,2:]) -
                    torch.maximum(corners[:,:,None,:2],corners[:,None,:,:2])).clamp(min=0).prod(-1)
    area = boxes[...,2:].prod(-1)
    iou = intersection/(area[:,:,None]+area[:,None,:]-intersection).clamp(min=1e-12)
    min_distance = float(distances.masked_fill(~relevant,float('inf')).min())
    bucket = '<1' if min_distance < 1 else '1-2' if min_distance < 2 else '>=2' if np.isfinite(min_distance) else 'no_relevant_pair'
    row = dict(valid_member_frames=int(member_frames.sum()), tightened_member_frames=int(tightened.sum()),
               crowded_iou_0_3=bool(((iou>=.3)&relevant).any()), distance_bucket=bucket,
               min_cross_group_or_individual_grid_distance=min_distance if np.isfinite(min_distance) else None,
               coincident_valid_pair_frames=int(((distances<=1e-6)&pairs).sum())//2,
               floor_both_axes_member_frames=int(((sigma_a==args.heatmap_sigma_min).all(-1)&member_frames).sum()),
               changed=not torch.equal(maps[0],maps[1]))
    masks = dict(member=members.bool()[:,None,:]&valid[None],
                 cross_group=(~members.bool())[:,None,:]&grouped[None,None,:]&valid[None],
                 individual=(~grouped)[None,None,:].expand(members.shape[0],boxes.shape[0],-1)&valid[None])
    for suffix, target in zip(('a','b'),maps):
        row['mass_'+suffix] = float(target.sum())
        responses = center_responses(target,boxes)
        for name, mask in masks.items():
            row[name+'_count'] = int(mask.sum())
            row[name+'_response_sum_'+suffix] = float(responses[mask].sum())
    assert torch.isfinite(maps[1]).all() and bool((maps[1]<=maps[0]).all())
    return row, maps, sigma_a, sigma_b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/heatmap/h1_sup_crowding.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--train-clips',type=int,default=64)
    cli = parser.parse_args()
    args = argparse.Namespace(**json.loads(Path(cli.config).read_text()))
    if args.split!='place' or not args.val_mode or args.heatmap_variant!='h1_sup' or args.heatmap_target_mode!='nearest_center':
        raise ValueError('only the prespecified place TRAIN H1-Sup crowding control is supported')
    if cli.train_clips<2: raise ValueError('at least two TRAIN clips are required')
    out = Path(cli.output); out.mkdir(parents=True,exist_ok=False)
    (out/'command.txt').write_text(shlex.join([sys.executable]+sys.argv)+'\n')
    json_write(out/'resolved_config.json',vars(args))
    sources = {}
    for source in ('models/heatmap.py','dataloader/cafe.py','dataloader/dataloader.py',
                   'tools/audit_heatmap_crowding_targets.py',cli.config):
        path = ROOT/source; sources[source] = file_hash(path)
        destination = out/'source_snapshot'/source; destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
    json_write(out/'source_manifest.json',sources)
    json_write(out/'gate_definition.json',dict(min_tightened_fraction=.01,min_changed_crowded_clips=2,
        fixed_distance_ratio=.35,selection='uniformly spaced sorted TRAIN clips; fixed temporal segment centers'))
    torch.set_num_threads(2)
    try:
        data = Path(args.data_path)/'cafe'
        anns = cafe_read_annotations(str(data),TRAIN_CAFE_P,args.num_class)
        track_path = Path(args.tracks_path) if args.tracks_path else data/'gt_tracks.pkl'
        with track_path.open('rb') as stream: tracks = pickle.load(stream)
        dataset = CafeDataset(sorted(cafe_all_frames(anns)),anns,tracks,str(data),args,is_training=False)
        rows = []; (out/'targets').mkdir()
        for index in subset_indices(dataset,cli.train_clips):
            clip = dataset.frames[index]
            # Use the real loader, rather than a second track/validity implementation.
            images, target, info = dataset[index]; del images
            boxes, members = target['boxes'], target['members'][0]
            valid = frame_validity(boxes[None],target['actions']==args.num_class+1,target['actor_frame_valid'][None])[0]
            row, maps, sigma_a, sigma_b = analyze(boxes,members,valid,args)
            ann_path = data/str(clip[0])/str(clip[1])/'ann.json'
            actor_ids = [figure['id'] for figure in json.loads(ann_path.read_text())['figures']]
            row.update(vid=clip[0],cid=clip[1],frame_ids=info['fid'],actor_ids=actor_ids,annotation_hash=file_hash(ann_path))
            rows.append(row)
            np.savez_compressed(str(out/'targets'/('%s_%s.npz'%clip)),boxes=boxes.numpy(),members=members.numpy(),
                valid=valid.numpy(),target_a=maps[0].numpy(),target_b=maps[1].numpy(),
                sigma_a=sigma_a.numpy(),sigma_b=sigma_b.numpy(),actor_ids=np.array(actor_ids),frame_ids=np.array(info['fid']))
            print(json.dumps(row),flush=True)
        json_write(out/'geometry_rows.json',rows)
        with (out/'geometry_rows.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        total = lambda key: sum(row[key] for row in rows)
        fraction = total('tightened_member_frames')/max(total('valid_member_frames'),1)
        changed_crowded = sum(row['changed'] and row['crowded_iou_0_3'] for row in rows)
        report = dict(status='completed',train_dataset_clips=len(dataset),checked_clips=len(rows),
            val_loaded=False,test_loaded=False,optimizer_updates=0,prediction_metrics=False,
            sampling='fixed TRAIN coverage; deterministic segment-center frames, not training RNG',
            tightened_member_frames=total('tightened_member_frames'),valid_member_frames=total('valid_member_frames'),
            tightened_member_fraction=fraction,changed_clips=total('changed'),
            crowded_clips=total('crowded_iou_0_3'),changed_crowded_clips=changed_crowded,
            coincident_valid_pair_frames=total('coincident_valid_pair_frames'),
            floor_both_axes_member_frames=total('floor_both_axes_member_frames'),
            mass_a=total('mass_a'),mass_b=total('mass_b'),mass_ratio=total('mass_b')/max(total('mass_a'),1e-12),
            distance_bucket_clip_counts={key:sum(r['distance_bucket']==key for r in rows) for key in ('<1','1-2','>=2','no_relevant_pair')},
            geometry_gate_pass=fraction>=.01 and changed_crowded>=2,
            gate_note='geometry only, not evidence of better grouping; no radius/threshold search',
            tracks_hash=file_hash(track_path))
        for name in ('member','cross_group','individual'):
            report[name+'_count']=total(name+'_count')
            for suffix in ('a','b'):
                report[name+'_center_response_'+suffix]=total(name+'_response_sum_'+suffix)/max(total(name+'_count'),1)
        json_write(out/'geometry_summary.json',report)
        print('GEOMETRY_SUMMARY',json.dumps(report),flush=True)
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise


if __name__=='__main__': main()
