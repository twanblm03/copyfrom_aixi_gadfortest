#!/usr/bin/env python3
"""Compare paired B0/H1-Sup runs with center and random segment TRAIN frames."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.cafe_eval import GAD_Evaluation
from compare_b0_sam_full import clip_ids, load_run, metrics_for


TAGS = {
    'center_b0': 'b0_center_lr1_noclip_seed1',
    'center_h1_sup': 'h1_sup_center_lr1_noclip_seed1',
    'random_b0': 'b0_randomseg_lr1_noclip_seed1',
    'random_h1_sup': 'h1_sup_randomseg_lr1_noclip_seed1',
}
ALLOWED_ARG_DIFFERENCES = {'checkpoint_root', 'experiment_tag', 'fixed_center_train',
                           'heatmap_variant', 'heatmap_sigma_ratio',
                           'peak_cuda_memory_mb'}


def main():
    runs = {name: load_run(tag) for name, tag in TAGS.items()}
    baseline = runs['center_b0']['args']
    for name, run in runs.items():
        args = run['args']
        differences = {key: [baseline.get(key), args.get(key)]
                       for key in set(baseline) | set(args)
                       if key not in ALLOWED_ARG_DIFFERENCES and baseline.get(key) != args.get(key)}
        if differences:
            raise RuntimeError(f'{name} unexpected protocol differences: {differences}')
        expected_center = name.startswith('center_')
        expected_variant = 'h1_sup' if name.endswith('h1_sup') else 'off'
        if args['fixed_center_train'] != expected_center or args['heatmap_variant'] != expected_variant:
            raise RuntimeError(f'{name} sampling or variant mismatch')
        if args['train_mask_mode'] != 'disabled' or args['gradient_clipping'] or args['backbone_lr_scale'] != 1:
            raise RuntimeError(f'{name} optimization protocol mismatch')
        if expected_variant == 'h1_sup' and args['heatmap_sigma_ratio'] != .15:
            raise RuntimeError(f'{name} heatmap sigma ratio mismatch')

    clips = {name: clip_ids(run['prediction_path']) for name, run in runs.items()}
    if len(clips['center_b0']) != 1386 or any(value != clips['center_b0'] for value in clips.values()):
        raise RuntimeError('VAL clip coverage differs')

    args = SimpleNamespace(**baseline)
    with open(args.labelmap) as labelmap, open(args.groundtruth) as groundtruth:
        args.labelmap, args.groundtruth = labelmap, groundtruth
        evaluator = GAD_Evaluation(args, clip_ids=clips['center_b0'])
        metrics = {name: metrics_for(evaluator, run) for name, run in runs.items()}

    keys = ('group_mAP_1.0', 'group_mAP_0.5', 'outlier_mIoU')
    advantage = {
        sampling: {
            key: float(metrics[f'{sampling}_h1_sup'][key] - metrics[f'{sampling}_b0'][key])
            for key in keys
        }
        for sampling in ('center', 'random')
    }
    report = {
        'question': 'Does random segment TRAIN frame sampling recover H1-Sup advantage over B0 under four-GPU DDP?',
        'val_clips': len(clips['center_b0']),
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
        'h1_minus_b0': advantage,
        'sampling_effect_on_h1_advantage': {
            key: float(advantage['random'][key] - advantage['center'][key]) for key in keys
        },
        'per_class': [
            {
                'id': category['id'],
                'name': category['name'],
                **{f'{name}_AP_{threshold}': float(metrics[name][f'group_APs_{threshold}'][index])
                   for name in runs for threshold in ('1.0', '0.5')},
            }
            for index, category in enumerate(evaluator.categories)
        ],
    }
    output = ROOT / 'results/sam_person/h1_sampling_4run.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    center_gap = advantage['center']['group_mAP_1.0']
    random_gap = advantage['random']['group_mAP_1.0']
    if random_gap > 0:
        conclusion = ('随机分段采帧组中 H1-Sup 高于 B0，历史的相对优势在当前四卡协议下'
                      '重新出现；采帧方式是导致上轮结果变化的候选因素。')
    else:
        conclusion = ('随机分段采帧组中 H1-Sup 仍未高于 B0；单独改变采帧方式不足以'
                      '恢复历史优势，需要继续检查双卡 DataParallel 与四卡 DDP 等差异。')
    rows = [
        '# H1-Sup 随机分段采帧复现结果',
        '',
        '四项训练均为 30 epoch、同一 1,386 条 VAL clip，并按修正后的 group mAP@1.0 选模。',
        '',
        '| TRAIN 采帧 | B0 mAP@1.0 | H1-Sup mAP@1.0 | H1-Sup − B0 |',
        '| --- | ---: | ---: | ---: |',
    ]
    for sampling, label in (('center', '固定中心'), ('random', '随机分段')):
        rows.append(f'| {label} | {metrics[f"{sampling}_b0"]["group_mAP_1.0"]:.4f} '
                    f'| {metrics[f"{sampling}_h1_sup"]["group_mAP_1.0"]:.4f} '
                    f'| {advantage[sampling]["group_mAP_1.0"]:+.4f} |')
    rows += [
        '',
        f'相对优势变化：{random_gap - center_gap:+.4f} mAP@1.0 点。',
        '',
        conclusion,
        '',
        '此对照仅改变当前四卡协议的 TRAIN 采帧。历史双卡 DataParallel 还存在'
        '每卡 BatchNorm 批量、H1-Sup mask loss 的跨 rank 归一化等差异，'
        '不能仅凭本次结果断定历史差异的唯一原因。',
        '',
        f'逐类 AP、mAP@0.5、outlier mIoU 和运行路径见 `{output.relative_to(ROOT)}`。',
    ]
    (output.with_suffix('.md')).write_text('\n'.join(rows) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
