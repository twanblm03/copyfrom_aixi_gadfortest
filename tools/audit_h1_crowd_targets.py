"""Audit crowd Gaussian geometry on fixed TRAIN clips without JPEG/GPU loading."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from dataloader.cafe import CafeDataset, cafe_read_annotations
from dataloader.dataloader import TRAIN_CAFE_P, TRAIN_CAFE_V
from models.heatmap import gaussian_targets
from tools.sam_person_full import actor_boxes_for_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-args', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--clips', type=int, default=128)
    parser.add_argument('--cap-ratio', type=float, default=.35)
    args = parser.parse_args()
    if args.clips < 1:
        raise ValueError('clips must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.reference_args.read_text())
    if not config['val_mode'] or config['sam_mask_manifest']:
        raise ValueError('expected mask-free TRAIN->VAL reference')
    torch.set_num_threads(2)
    np.random.seed(104729)
    videos = TRAIN_CAFE_P if config['split'] == 'place' else TRAIN_CAFE_V
    data_root = Path(config['data_path']) / 'cafe'
    available = sorted((int(v), int(c.name)) for v in videos
                       for c in (data_root / v).iterdir() if c.is_dir() and c.name.isdigit())
    selected = [available[i] for i in np.linspace(0, len(available)-1,
                                                min(args.clips, len(available)), dtype=int)]
    annotations = cafe_read_annotations(str(data_root), videos, config['num_class'],
                                        clip_ids=set(selected))
    with open(config['tracks_path'], 'rb') as stream:
        tracks = pickle.load(stream)
    dataset = CafeDataset(sorted(annotations), annotations, tracks, str(data_root),
                          argparse.Namespace(**config), is_training=True)
    rows, baseline_mass, crowd_mass, leakage_before, leakage_after = [], [], [], [], []
    counts = dict(valid_actor_frames=0, shrunk_actor_frames=0, nearest_cap_below_floor=0,
                  group_frames=0, overlapping_cross_group_pairs=0)
    for clip in sorted(annotations):
        frames = dataset.select_frames(clip)
        geometry = [actor_boxes_for_frame(dataset, clip, fid) for _, fid in frames]
        boxes = torch.tensor(np.stack([item[0] for item in geometry]))
        valid = torch.tensor(np.stack([item[1] for item in geometry]))
        ann = annotations[clip]
        members = F.pad(ann['members'], (0, config['num_boxes']-ann['members'].shape[1]))
        old, group_valid = gaussian_targets(boxes, members, valid, 23, 40,
            config['heatmap_sigma_ratio'], config['heatmap_sigma_min'], config['heatmap_sigma_max'])
        new, new_valid = gaussian_targets(boxes, members, valid, 23, 40,
            config['heatmap_sigma_ratio'], config['heatmap_sigma_min'], config['heatmap_sigma_max'],
            target_mode='crowd', crowd_cap_ratio=args.cap_ratio)
        if not (torch.isfinite(new).all() and torch.equal(group_valid, new_valid)
                and (new <= old + 1e-7).all()):
            raise AssertionError('invalid target change for %s' % (clip,))
        centers = boxes[..., :2] * torch.tensor([40., 23.])
        distances = (centers[:, :, None]-centers[:, None, :]).square().sum(-1).sqrt()
        pair_valid = valid[:, :, None] & valid[:, None, :] & ~torch.eye(
            boxes.shape[1], dtype=torch.bool)[None]
        nearest = distances.masked_fill(~pair_valid, float('inf')).min(-1).values
        sigma = (boxes[..., 2:]*torch.tensor([40., 23.])*config['heatmap_sigma_ratio']).clamp(
            config['heatmap_sigma_min'], config['heatmap_sigma_max'])
        cap = (nearest*args.cap_ratio).clamp(min=config['heatmap_sigma_min'])
        shrunk = (sigma > cap[..., None]).any(-1) & valid
        counts['valid_actor_frames'] += int(valid.sum())
        counts['shrunk_actor_frames'] += int(shrunk.sum())
        counts['nearest_cap_below_floor'] += int(((nearest*args.cap_ratio <
                                                 config['heatmap_sigma_min']) & valid).sum())
        counts['group_frames'] += int(group_valid.sum())
        baseline_mass.extend(old.sum((-1,-2))[group_valid].tolist())
        crowd_mass.extend(new.sum((-1,-2))[group_valid].tolist())
        # [G,T,H,W] -> one map per group/frame; sample every actor center.
        g,t,h,w = old.shape
        grid = (2*boxes[..., :2]-1)[None].expand(g,-1,-1,-1).reshape(g*t,1,-1,2)
        old_response = F.grid_sample(old.reshape(g*t,1,h,w), grid,
                                    align_corners=False).reshape(g,t,-1)
        new_response = F.grid_sample(new.reshape(g*t,1,h,w), grid,
                                    align_corners=False).reshape(g,t,-1)
        nonmember = ~members.bool()[:,None,:] & valid[None] & group_valid[...,None]
        leakage_before.extend(old_response[nonmember].tolist())
        leakage_after.extend(new_response[nonmember].tolist())
        corners = torch.cat([boxes[...,:2]-boxes[...,2:]/2,
                             boxes[...,:2]+boxes[...,2:]/2], -1)
        lo = torch.maximum(corners[:,:,None,:2], corners[:,None,:,:2])
        hi = torch.minimum(corners[:,:,None,2:], corners[:,None,:,2:])
        inter = (hi-lo).clamp(min=0).prod(-1)
        area = boxes[...,2:].prod(-1)
        iou = inter/(area[:,:,None]+area[:,None,:]-inter).clamp(min=1e-12)
        group_id = F.pad(ann['membership'], (0, config['num_boxes']-len(ann['membership'])), value=-1)
        different = group_id[:,None] != group_id[None,:]
        cross_pairs = (pair_valid & different[None] & (iou >= .3)).sum()//2
        counts['overlapping_cross_group_pairs'] += int(cross_pairs)
        rows.append(dict(clip=list(clip), frames=[fid for _,fid in frames],
                         valid=int(valid.sum()), shrunk=int(shrunk.sum()),
                         cross_group_overlap_pairs=int(cross_pairs)))
    if not rows or not counts['shrunk_actor_frames']:
        raise RuntimeError('no effective crowd target change; inspect before full training')
    report = dict(status='passed', split=config['split'], partition='TRAIN only',
        sampling='evenly spread candidate clips, seeded random temporal segments; no JPEG',
        sampling_seed=104729, clips=len(rows), candidate_clips_requested=args.clips,
        reference_args=str(args.reference_args.resolve()),
        reference_args_sha256=hashlib.sha256(args.reference_args.read_bytes()).hexdigest(),
        cap_ratio=args.cap_ratio, sigma_min=config['heatmap_sigma_min'], grid=[23,40],
        counts=counts, changed_fraction=counts['shrunk_actor_frames']/counts['valid_actor_frames'],
        mean_target_mass_before=float(np.mean(baseline_mass)),
        mean_target_mass_after=float(np.mean(crowd_mass)),
        mean_nonmember_center_response_before=float(np.mean(leakage_before)),
        mean_nonmember_center_response_after=float(np.mean(leakage_after)),
        caveat='Target geometry only; lower mass changes auxiliary gradient scale. Not grouping/model evidence.',
        selected_clips=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='selected_clips'}, indent=2))


if __name__ == '__main__':
    main()
