#!/usr/bin/env python3
"""Compare matched four-GPU DataParallel B0/H1-Sup full runs."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.cafe_eval import GAD_Evaluation
from compare_b0_sam_full import clip_ids, load_run, metrics_for


TAGS = {
    'b0': 'b0_randomseg_dp4_lr1_noclip_seed1',
    'h1_sup': 'h1_sup_randomseg_dp4_lr1_noclip_seed1',
}
ALLOWED_DIFFERENCES = {'checkpoint_root', 'experiment_tag',
                       'heatmap_variant', 'heatmap_sigma_ratio',
                       'peak_cuda_memory_mb'}


def main():
    runs = {name: load_run(tag) for name, tag in TAGS.items()}
    baseline = runs['b0']['args']
    for name, run in runs.items():
        args = run['args']
        unexpected = {key: [baseline.get(key), args.get(key)]
                      for key in set(baseline) | set(args)
                      if key not in ALLOWED_DIFFERENCES and baseline.get(key) != args.get(key)}
        if unexpected:
            raise RuntimeError(f'{name} protocol differs: {unexpected}')
        expected_variant = 'h1_sup' if name == 'h1_sup' else 'off'
        if args['heatmap_variant'] != expected_variant or args['fixed_center_train']:
            raise RuntimeError(f'{name} variant or sampling mismatch')
        if args['distributed'] or args.get('parallel_transport') != 'host':
            raise RuntimeError(f'{name} is not host-staged DataParallel')
        if args['train_mask_mode'] != 'disabled' or args['gradient_clipping'] or args['backbone_lr_scale'] != 1:
            raise RuntimeError(f'{name} optimization protocol mismatch')
        if name == 'h1_sup' and args['heatmap_sigma_ratio'] != .15:
            raise RuntimeError('H1-Sup heatmap sigma mismatch')
    clips = {name: clip_ids(run['prediction_path']) for name, run in runs.items()}
    if len(clips['b0']) != 1386 or clips['b0'] != clips['h1_sup']:
        raise RuntimeError('VAL clip coverage differs')

    args = SimpleNamespace(**baseline)
    with open(args.labelmap) as labelmap, open(args.groundtruth) as groundtruth:
        args.labelmap, args.groundtruth = labelmap, groundtruth
        evaluator = GAD_Evaluation(args, clip_ids=clips['b0'])
        metrics = {name: metrics_for(evaluator, run) for name, run in runs.items()}

    keys = ('group_mAP_1.0', 'group_mAP_0.5', 'outlier_mIoU')
    difference = {key: float(metrics['h1_sup'][key] - metrics['b0'][key]) for key in keys}
    report = {
        'protocol': 'random_segment_train_center_val_4gpu_host_staged_DataParallel_batch16_seed1_30epoch',
        'val_clips': len(clips['b0']),
        'selection_metric': 'group_mAP_1.0',
        'runs': {name: {
            'best_epoch': run['best_epoch'],
            **{key: float(metrics[name][key]) for key in keys},
            'summary_path': str(run['summary_path'].relative_to(ROOT)),
            'prediction_path': str(run['prediction_path'].relative_to(ROOT)),
        } for name, run in runs.items()},
        'h1_minus_b0': difference,
        'per_class': [{
            'id': category['id'], 'name': category['name'],
            **{f'{name}_AP_{threshold}': float(metrics[name][f'group_APs_{threshold}'][index])
               for name in runs for threshold in ('1.0', '0.5')},
        } for index, category in enumerate(evaluator.categories)],
    }
    output = ROOT / 'results/sam_person/h1_dp4_randomseg_full.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = [
        '# 四卡 DataParallel：随机分段采帧的 B0 / H1-Sup', '',
        '同一 batch16、30 epoch、1,386 条固定中心帧 VAL clips；热图损失在 H1-Sup 的完整 batch16 上求平均。', '',
        '| 模型 | 最佳 epoch | group mAP@1.0 | group mAP@0.5 | outlier mIoU |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for name in ('b0', 'h1_sup'):
        row = report['runs'][name]
        lines.append(f'| {name} | {row["best_epoch"]} | {row["group_mAP_1.0"]:.4f} '
                     f'| {row["group_mAP_0.5"]:.4f} | {row["outlier_mIoU"]:.4f} |')
    lines += ['', f'H1-Sup − B0：{difference["group_mAP_1.0"]:+.4f} mAP@1.0 点。', '',
              '与历史双卡 DataParallel 相比，本轮每卡 batch 从 8 变为 4，BatchNorm 本地统计可能不同。',
              f'逐类 AP 和运行路径见 `{output.relative_to(ROOT)}`。']
    output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
