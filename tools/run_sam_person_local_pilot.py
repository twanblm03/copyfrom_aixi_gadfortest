"""Paired local B0, original SAM, and context-preserving SAM pilot."""
import argparse
import json
import math
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from dataloader.cafe import CafeDataset, cafe_read_annotations
from dataloader.dataloader import TRAIN_CAFE_P, VAL_CAFE_P
from models import build_model
from tools.run_heatmap_research import collate, seed_all
from tools.run_sam_person_pilot import evaluate_masks, forward, selected_samples
from tools.sam_person_full import iter_manifest, source_config
from util.sam_person import MaskReader, atomic_json, checkpoint_state, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--checkpoint', required=True, help='shared B0 warm start')
    p.add_argument('--output', required=True)
    p.add_argument('--cache-root', default=str(ROOT / '.local_data/sam/cache'))
    p.add_argument('--sam-repo', default=str(ROOT / '.local_deps/segment-anything'))
    p.add_argument('--sam-checkpoint', default=str(ROOT / '.local_data/sam/checkpoints/sam_vit_b_01ec64.pth'))
    p.add_argument('--data-root', default='/nas/mzy/dataset/Cafe_Dataset/Cafe_Dataset/Dataset')
    p.add_argument('--tracks-path', default=str(ROOT.parent.parent / '.local_data/cafe/gt_tracks.pkl'))
    p.add_argument('--groundtruth', default='/nas/mzy/dataset/Cafe_Dataset/Cafe_Dataset/evaluation/gt_tracks.txt')
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--context-fraction', type=float, default=0.5)
    cli = p.parse_args()
    if cli.steps < 1 or cli.batch < 1:
        p.error('steps and batch must be positive')
    out = Path(cli.output).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    c = source_config(argparse.Namespace(sam_repo=cli.sam_repo,
        checkpoint=cli.sam_checkpoint, cache_root=cli.cache_root))
    reader = MaskReader(c, list(iter_manifest(cli.manifest)))
    reader.preflight()
    clips = json.loads(Path(cli.manifest + '.clips.json').read_text())
    with open(cli.tracks_path, 'rb') as stream:
        tracks = pickle.load(stream)
    base = json.loads((ROOT / 'configs/heatmap/b0.json').read_text())
    base.update(data_path=cli.data_root, tracks_path=cli.tracks_path,
                groundtruth=cli.groundtruth,
                labelmap=str(ROOT / 'label_map/group_action_list.pbtxt'),
                train_mask_mode='disabled', mask_empty_eps=1e-6,
                mask_context_fraction=0.0, random_seed=cli.seed,
                batch=cli.batch, test_batch=cli.batch)
    args = argparse.Namespace(**base)
    selected = []
    for split, videos in [('train', TRAIN_CAFE_P), ('val', VAL_CAFE_P)]:
        ids = {tuple(item['clip']) for item in clips[split]}
        annotations = cafe_read_annotations(
            str(Path(cli.data_root) / 'cafe'), videos, args.num_class,
            clip_ids=ids)
        dataset = CafeDataset(sorted(annotations), annotations, tracks,
            str(Path(cli.data_root) / 'cafe'), args, is_training=False)
        samples, rows = selected_samples(dataset, clips[split], reader,
            dict(image_hw=[720,1280], feature_hw=[23,40]))
        selected.append(samples)
        atomic_json(out / (split + '_inputs.json'), rows)
    train_samples, val_samples = selected
    val_loader = DataLoader(val_samples, batch_size=cli.batch,
        shuffle=False, collate_fn=collate, num_workers=0)
    checkpoint = torch.load(cli.checkpoint, map_location='cpu', weights_only=False)
    state, mapping = checkpoint_state(checkpoint)
    atomic_json(out / 'protocol.json', dict(
        base_checkpoint=cli.checkpoint, base_checkpoint_sha256=file_hash(cli.checkpoint),
        checkpoint_mapping=mapping, source_manifest_sha256=file_hash(cli.manifest),
        sam_cache_version=c['cache_version'], evaluator_sha256=file_hash(ROOT/'evaluation/cafe_eval.py'),
        seed=cli.seed, batch=cli.batch, steps=cli.steps,
        variants=['B0','S2F','BLEND'], context_fraction=cli.context_fraction,
        optimizer='Adam lr=1e-5, weight_decay=0; independent reset per variant',
        selection='fixed final step; full 8-clip VAL evaluated only before and after'))
    device = torch.device('cuda:0')
    for name, mode, fraction in [('B0','disabled',0.0),
                                 ('S2F','sam_roi',0.0),
                                 ('BLEND','sam_roi_blend',cli.context_fraction)]:
        variant = out / name
        variant.mkdir()
        seed_all(cli.seed)
        args.train_mask_mode = mode
        args.mask_context_fraction = fraction
        model, criterion = build_model(args)
        model.load_state_dict(state, strict=True)
        model.to(device)
        criterion.to(device)
        metrics_before = evaluate_masks(model, val_loader, args, variant, device, mode, 'before')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)
        generator = torch.Generator().manual_seed(cli.seed)
        loader = DataLoader(train_samples, batch_size=cli.batch, shuffle=True,
            generator=generator, drop_last=True, collate_fn=collate, num_workers=0)
        steps = 0
        started = time.monotonic()
        order = []
        while steps < cli.steps:
            for batch in loader:
                model.train()
                optimizer.zero_grad()
                outputs, targets, infos, _ = forward(model, batch, device, mode)
                terms = criterion(outputs, targets, log=False)
                loss = sum(terms[key] * weight for key, weight in criterion.weight_dict.items())
                if not torch.isfinite(loss):
                    raise FloatingPointError('%s nonfinite loss at step %d' % (name, steps + 1))
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError('%s nonfinite gradient at step %d' % (name, steps + 1))
                optimizer.step()
                steps += 1
                order.append([[info['vid'], info['sid'], info['fid']] for info in infos])
                with open(variant / 'steps.jsonl', 'a') as stream:
                    stream.write(json.dumps(dict(step=steps, loss=float(loss.detach()))) + '\n')
                if steps % 30 == 0:
                    print(name, 'step', steps, 'loss', float(loss.detach()), flush=True)
                if steps >= cli.steps:
                    break
        metrics_after = evaluate_masks(model, val_loader, args, variant, device, mode, 'after')
        torch.save(dict(state_dict=model.cpu().state_dict(), args=vars(args), steps=steps), variant/'last.pth')
        atomic_json(variant/'result.json', dict(name=name, mode=mode,
            context_fraction=fraction, steps=steps, seconds=time.monotonic()-started,
            sample_order=order, metrics_before=metrics_before, metrics_after=metrics_after))
        print('PILOT_RESULT', name, metrics_after['group_mAP_1.0'],
              metrics_after['group_mAP_0.5'], metrics_after['outlier_mIoU'], flush=True)


if __name__ == '__main__':
    main()
