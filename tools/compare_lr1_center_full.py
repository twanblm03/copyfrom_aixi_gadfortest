#!/usr/bin/env python3
"""Re-evaluate three completed center-frame, historical-LR VAL runs."""

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.cafe_eval import GAD_Evaluation, make_groups, read_text_file
from compare_b0_sam_full import COMMON_FIELDS, clip_ids, load_run, metrics_for


TAGS = {
    'b0': 'b0_center_lr1_noclip_seed1',
    'h1_sup': 'h1_sup_center_lr1_noclip_seed1',
    'sam': 'sam_center_lr1_noclip_seed1',
}


def grouping_coverage(evaluator, prediction_path):
    """GT group recall ignoring confidence, with and without activity labels."""
    with prediction_path.open() as stream:
        boxes, group_labels, activities, _, scores = read_text_file(
            stream, evaluator.eval_type, mode='pred')
    pred_groups, pred_activities, _ = make_groups(boxes, group_labels, activities, scores)
    valid_classes = {category['id'] for category in evaluator.categories}
    counts = Counter()
    total_iou, total_same_iou = 0.0, 0.0
    for clip, groups in evaluator.gt_groups_ids.items():
        candidates = []
        if clip in pred_groups:
            for group_id, members in pred_groups[clip][0].items():
                if group_id >= 0 and len(members) >= 2:
                    category = next(iter(pred_activities[clip][0][group_id]))
                    candidates.append((set(members), category))
        for group_id, members in groups[0].items():
            if len(members) < 2:
                continue
            category = next(iter(evaluator.gt_groups_activity[clip][0][group_id]))
            if category not in valid_classes:
                continue
            truth = set(members)
            overlaps = [(len(truth & pred) / len(truth | pred), pred_category)
                        for pred, pred_category in candidates]
            best = max((iou for iou, _ in overlaps), default=0.0)
            same = max((iou for iou, pred_category in overlaps if pred_category == category),
                       default=0.0)
            counts['gt_groups'] += 1
            counts['structural_at_1.0'] += best >= 1.0
            counts['structural_at_0.5'] += best >= .5
            counts['class_correct_at_1.0'] += same >= 1.0
            counts['class_correct_at_0.5'] += same >= .5
            total_iou += best
            total_same_iou += same
    total = counts['gt_groups']
    return {
        'gt_groups': total,
        'mean_best_group_iou': total_iou / total,
        'mean_best_same_class_iou': total_same_iou / total,
        **{key + '_recall': counts[key] / total
           for key in ('structural_at_1.0', 'structural_at_0.5',
                       'class_correct_at_1.0', 'class_correct_at_0.5')},
    }


def main():
    runs = {name: load_run(tag) for name, tag in TAGS.items()}
    baseline = runs['b0']['args']
    common = tuple(field for field in COMMON_FIELDS if field != 'heatmap_variant')
    for name, run in runs.items():
        diffs = {field: [baseline.get(field), run['args'].get(field)]
                 for field in common if baseline.get(field) != run['args'].get(field)}
        if diffs:
            raise RuntimeError(f'{name} core protocol mismatch: {diffs}')
        args = run['args']
        if args['gradient_clipping'] or args['backbone_lr_scale'] != 1:
            raise RuntimeError(f'{name} is not historical-LR/no-clipping training')
        checkpoint = ROOT / '.local_data/historical_lr_center' / name / 'full_checkpoints'
        dirs = list(checkpoint.iterdir())
        if len(dirs) != 1 or not all((dirs[0] / filename).is_file()
                                      for filename in ('best_val.pth', 'epoch30.pth')):
            raise RuntimeError(f'{name} selected/final checkpoint missing')

    expected = {'b0': ('off', 'disabled'), 'h1_sup': ('h1_sup', 'disabled'),
                'sam': ('off', 'sam_roi_blend')}
    for name, (heatmap, mask) in expected.items():
        args = runs[name]['args']
        if (args['heatmap_variant'], args['train_mask_mode']) != (heatmap, mask):
            raise RuntimeError(f'{name} variant mismatch')
    if runs['h1_sup']['args']['heatmap_sigma_ratio'] != .15:
        raise RuntimeError('H1-Sup sigma ratio mismatch')
    if not runs['sam']['args']['sam_projected_cache'] or runs['sam']['args']['mask_context_fraction'] != .5:
        raise RuntimeError('SAM cache or context setting mismatch')

    clips = {name: clip_ids(run['prediction_path']) for name, run in runs.items()}
    if len(clips['b0']) != 1386 or any(value != clips['b0'] for value in clips.values()):
        raise RuntimeError('Selected prediction VAL clip coverage differs')

    args = SimpleNamespace(**baseline)
    with open(args.labelmap) as labelmap, open(args.groundtruth) as groundtruth:
        args.labelmap, args.groundtruth = labelmap, groundtruth
        evaluator = GAD_Evaluation(args, clip_ids=clips['b0'])
        metrics = {name: metrics_for(evaluator, run) for name, run in runs.items()}
        coverage = {name: grouping_coverage(evaluator, run['prediction_path'])
                    for name, run in runs.items()}
        support = Counter()
        for clip in evaluator.gt_groups_ids:
            for group_id, members in evaluator.gt_groups_ids[clip][0].items():
                if len(members) >= 2:
                    category = next(iter(evaluator.gt_groups_activity[clip][0][group_id]))
                    support[category] += 1

    keys = ('group_mAP_1.0', 'group_mAP_0.5', 'outlier_mIoU')
    report = {
        'protocol': 'fixed_center_place_train_val_seed1_30epoch_four_gpu_lrscale1_no_clip',
        'val_clips': len(clips['b0']),
        'selection_metric': 'group_mAP_1.0',
        'runs': {
            name: {
                'best_epoch': run['best_epoch'],
                **{key: float(metrics[name][key]) for key in keys},
                'summary_path': str(run['summary_path'].relative_to(ROOT)),
                'prediction_path': str(run['prediction_path'].relative_to(ROOT)),
            }
            for name, run in runs.items()
        },
        'delta_vs_b0': {
            name: {key: float(metrics[name][key] - metrics['b0'][key]) for key in keys}
            for name in ('h1_sup', 'sam')
        },
        'grouping_coverage': coverage,
        'per_class': [
            {
                'id': category['id'],
                'name': category['name'],
                'gt_groups': support[category['id']],
                **{f'{name}_AP_{threshold}': float(metrics[name][f'group_APs_{threshold}'][index])
                   for name in runs for threshold in ('1.0', '0.5')},
            }
            for index, category in enumerate(evaluator.categories)
        ],
    }
    output = ROOT / 'results/sam_person/lr1_center_full_20261020.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
