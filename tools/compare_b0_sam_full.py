#!/usr/bin/env python3
"""Re-evaluate the selected B0 and SAM checkpoints on the same VAL clips."""

import json
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.cafe_eval import GAD_Evaluation


SAM_TAG = 'sam_person_blend50_full_resume7_clip1'
B0_TAG = 'b0_fixedcenter_matched_sam_seed1'
COMMON_FIELDS = (
    'dataset', 'split', 'val_mode', 'data_path', 'tracks_path',
    'groundtruth', 'labelmap', 'research_b0_repro', 'heatmap_variant',
    'no_mae', 'backbone', 'image_width', 'image_height', 'num_frame',
    'num_class', 'num_boxes', 'fixed_center_train', 'random_sampling',
    'random_seed', 'epochs', 'batch', 'test_batch', 'micro_batch',
    'optimizer', 'lr', 'max_lr', 'warmup_epochs', 'lr_schedule',
    'weight_decay', 'backbone_lr_scale', 'selection_metric', 'eval_split',
)


def load_run(tag):
    matches = sorted(path for path in (ROOT / 'result').glob('*/summary.json')
                     if f'_[{tag}]_' in path.parent.name)
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one {tag} summary, got {matches}')
    summary_path = matches[0]
    summary = json.loads(summary_path.read_text())
    history = summary['history']
    if [row['epoch'] for row in history['train']] != list(range(1, 31)):
        raise RuntimeError(f'{tag} does not have complete TRAIN history')
    if [row['epoch'] for row in history['val']] != list(range(1, 31)):
        raise RuntimeError(f'{tag} does not have complete VAL history')
    if history['test']:
        raise RuntimeError(f'{tag} unexpectedly contains TEST evaluation')
    best_epoch = summary['best']['group_mAP_1.0']['epoch']
    if best_epoch != max(history['val'], key=lambda row: row['group_mAP_1.0'])['epoch']:
        raise RuntimeError(f'{tag} best epoch does not match VAL history')
    exp_name = summary_path.parent.name
    prediction_path = ROOT / 'outputs' / exp_name / f'pred_group_epoch_{best_epoch}.txt'
    if not prediction_path.is_file():
        raise RuntimeError(f'Missing selected prediction: {prediction_path}')
    return {
        'tag': tag,
        'args': summary['args'],
        'history': history,
        'best_epoch': best_epoch,
        'prediction_path': prediction_path,
        'summary_path': summary_path,
    }


def clip_ids(path):
    with path.open() as stream:
        return {(int(parts[0]), int(parts[1]))
                for line in stream if (parts := line.split(' ', 3))}


def metrics_for(evaluator, run):
    with run['prediction_path'].open() as stream:
        metrics = evaluator.evaluate(stream)
    row = run['history']['val'][run['best_epoch'] - 1]
    for key in ('group_mAP_1.0', 'group_mAP_0.5', 'outlier_mIoU'):
        if not math.isclose(metrics[key], row[key], rel_tol=0, abs_tol=1e-6):
            raise RuntimeError(f"{run['tag']} {key} disagrees with training summary")
    return metrics


def main():
    sam = load_run(SAM_TAG)
    b0 = load_run(B0_TAG)
    mismatches = {field: [sam['args'].get(field), b0['args'].get(field)]
                  for field in COMMON_FIELDS
                  if sam['args'].get(field) != b0['args'].get(field)}
    if mismatches:
        raise RuntimeError(f'Core protocols differ: {mismatches}')
    if sam['args']['train_mask_mode'] != 'sam_roi_blend' or b0['args']['train_mask_mode'] != 'disabled':
        raise RuntimeError('Mask mode mismatch')
    if not (b0['args']['gradient_clipping']
            and b0['args']['gradient_clipping_start_epoch'] == 8):
        raise RuntimeError('B0 clipping schedule mismatch')

    sam_clips = clip_ids(sam['prediction_path'])
    b0_clips = clip_ids(b0['prediction_path'])
    if sam_clips != b0_clips or len(sam_clips) != 1386:
        raise RuntimeError(f'VAL clip mismatch: SAM {len(sam_clips)}, B0 {len(b0_clips)}')

    args = SimpleNamespace(**sam['args'])
    with open(args.labelmap) as labelmap, open(args.groundtruth) as groundtruth:
        args.labelmap = labelmap
        args.groundtruth = groundtruth
        evaluator = GAD_Evaluation(args, clip_ids=sam_clips)
        sam_metrics = metrics_for(evaluator, sam)
        b0_metrics = metrics_for(evaluator, b0)
        support = Counter()
        for clip in evaluator.gt_groups_ids:
            for group_id, members in evaluator.gt_groups_ids[clip][0].items():
                if len(members) >= 2:
                    category = next(iter(evaluator.gt_groups_activity[clip][0][group_id]))
                    support[category] += 1

    categories = evaluator.categories
    report = {
        'protocol': 'fixed_center_place_train_val_seed1_30epoch_four_gpu',
        'val_clips': len(sam_clips),
        'caveat': 'SAM restarted after epoch 7; B0 ran continuously. Both clip from epoch 8.',
        'runs': {},
        'sam_minus_b0': {},
        'per_class': [],
    }
    for label, run, metrics in (('sam', sam, sam_metrics), ('b0', b0, b0_metrics)):
        report['runs'][label] = {
            'best_epoch': run['best_epoch'],
            'group_mAP_1.0': metrics['group_mAP_1.0'],
            'group_mAP_0.5': metrics['group_mAP_0.5'],
            'outlier_mIoU': metrics['outlier_mIoU'],
            'summary_path': str(run['summary_path'].relative_to(ROOT)),
            'prediction_path': str(run['prediction_path'].relative_to(ROOT)),
        }
    for key in ('group_mAP_1.0', 'group_mAP_0.5', 'outlier_mIoU'):
        report['sam_minus_b0'][key] = sam_metrics[key] - b0_metrics[key]
    for index, category in enumerate(categories):
        report['per_class'].append({
            'name': category['name'],
            'id': category['id'],
            'gt_groups': support[category['id']],
            'sam_AP_1.0': float(sam_metrics['group_APs_1.0'][index]),
            'b0_AP_1.0': float(b0_metrics['group_APs_1.0'][index]),
            'sam_AP_0.5': float(sam_metrics['group_APs_0.5'][index]),
            'b0_AP_0.5': float(b0_metrics['group_APs_0.5'][index]),
        })

    output = ROOT / 'results/sam_person/b0_matched_full_20261019.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
