#!/usr/bin/env python3
"""Compare the selected B0, SAM, and H1-Sup weights on one VAL protocol."""

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.cafe_eval import GAD_Evaluation
from compare_b0_sam_full import COMMON_FIELDS, clip_ids, load_run, metrics_for


def main():
    runs = {
        'b0': load_run('b0_fixedcenter_matched_sam_seed1'),
        'sam': load_run('sam_person_blend50_full_resume7_clip1'),
        'h1_sup': load_run('h1_sup_fixedcenter_matched_seed1'),
    }
    reference = runs['b0']['args']
    for name, run in runs.items():
        differences = {field: [reference.get(field), run['args'].get(field)]
                       for field in COMMON_FIELDS
                       if reference.get(field) != run['args'].get(field)}
        if differences:
            raise RuntimeError(f'{name} core protocol mismatch: {differences}')
    if not (runs['b0']['args']['heatmap_variant'] == 'off'
            and runs['sam']['args']['heatmap_variant'] == 'off'
            and runs['h1_sup']['args']['heatmap_variant'] == 'h1_sup'):
        raise RuntimeError('Unexpected heatmap variant')
    if not (runs['b0']['args']['train_mask_mode'] == 'disabled'
            and runs['h1_sup']['args']['train_mask_mode'] == 'disabled'
            and runs['sam']['args']['train_mask_mode'] == 'sam_roi_blend'):
        raise RuntimeError('Unexpected SAM mask mode')
    if runs['h1_sup']['args']['heatmap_sigma_ratio'] != 0.15:
        raise RuntimeError('H1-Sup sigma ratio differs from historical H1-Sup')
    for name in ('b0', 'h1_sup'):
        a = runs[name]['args']
        if not (a['gradient_clipping'] and a['gradient_clipping_start_epoch'] == 8):
            raise RuntimeError(f'{name} gradient clipping schedule mismatch')

    clips = {name: clip_ids(run['prediction_path']) for name, run in runs.items()}
    if len(clips['b0']) != 1386 or any(value != clips['b0'] for value in clips.values()):
        raise RuntimeError('Selected predictions have different VAL clip coverage')

    args = SimpleNamespace(**runs['b0']['args'])
    with open(args.labelmap) as labelmap, open(args.groundtruth) as groundtruth:
        args.labelmap = labelmap
        args.groundtruth = groundtruth
        evaluator = GAD_Evaluation(args, clip_ids=clips['b0'])
        metrics = {name: metrics_for(evaluator, run) for name, run in runs.items()}
        support = Counter()
        for clip in evaluator.gt_groups_ids:
            for group_id, members in evaluator.gt_groups_ids[clip][0].items():
                if len(members) >= 2:
                    category = next(iter(evaluator.gt_groups_activity[clip][0][group_id]))
                    support[category] += 1

    keys = ('group_mAP_1.0', 'group_mAP_0.5', 'outlier_mIoU')
    report = {
        'protocol': 'fixed_center_place_train_val_seed1_30epoch_four_gpu',
        'val_clips': len(clips['b0']),
        'caveat': 'SAM restarted after epoch 7; B0 and H1-Sup ran continuously with clipping from epoch 8.',
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
            for name in ('sam', 'h1_sup')
        },
        'per_class': [
            {
                'id': category['id'],
                'name': category['name'],
                'gt_groups': support[category['id']],
                **{
                    f'{name}_AP_{threshold}': float(metrics[name][f'group_APs_{threshold}'][index])
                    for name in runs for threshold in ('1.0', '0.5')
                },
            }
            for index, category in enumerate(evaluator.categories)
        ],
    }
    output = ROOT / 'results/sam_person/b0_sam_h1_matched_full_20261019.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
