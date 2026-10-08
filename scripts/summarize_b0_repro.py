#!/usr/bin/env python3
"""Report B0-repro metrics at each seed's primary-selected VAL epoch."""

import argparse
import json
import math
import statistics
from pathlib import Path


METRICS = ('group_mAP_1.0', 'group_mAP_0.5', 'outlier_mIoU')
EXPECTED_ARGS = {
    'dataset': 'cafe',
    'val_mode': True,
    'image_width': 1280,
    'image_height': 720,
    'random_sampling': False,
    'num_frame': 5,
    'num_class': 6,
    'backbone': 'resnet18',
    'dilation': False,
    'frozen_batch_norm': False,
    'hidden_dim': 256,
    'num_boxes': 14,
    'crop_size': 5,
    'gar_nheads': 4,
    'gar_enc_layers': 6,
    'gar_ffn_dim': 512,
    'position_embedding': 'sine',
    'num_group_tokens': 12,
    'aux_loss': False,
    'group_threshold': 0.5,
    'distance_threshold': 0.2,
    'temperature': 0.2,
    'ce_loss_coef': 1.0,
    'eos_coef': 1.0,
    'group_eos_coef': 1.0,
    'group_ce_loss_coef': 1.0,
    'group_code_loss_coef': 5.0,
    'consistency_loss_coef': 2.0,
    'set_cost_group_class': 1.0,
    'set_cost_membership': 1.0,
    'epochs': 30,
    'test_freq': 1,
    'batch': 16,
    'test_batch': 16,
    'lr': 1e-5,
    'max_lr': 1e-4,
    'warmup_epochs': 5,
    'lr_schedule': 'linear',
    'weight_decay': 0.0,
    'drop_rate': 0.1,
    'gradient_clipping': False,
    'max_train_steps': 0,
    'selection_metric': 'group_mAP_1.0',
    'eval_type': 'gt_base',
    'eval_image_width': 1280,
    'eval_image_height': 720,
}


def parse_seeds(value):
    seeds = [int(field) for field in value.replace(',', ' ').split()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError('seeds must be a non-empty unique list')
    return seeds


def load_run(summary_path, split, seed):
    with summary_path.open(encoding='utf-8') as stream:
        summary = json.load(stream)
    run_args = summary.get('args', {})
    if summary.get('completed') is not True:
        return None
    if run_args.get('protocol') != 'development':
        return None
    if run_args.get('split') != split or run_args.get('random_seed') != seed:
        return None
    if run_args.get('experiment_tag') != f'b0_repro_val_{split}_seed{seed}':
        return None
    if any(run_args.get(key) != value for key, value in EXPECTED_ARGS.items()):
        return None

    history = summary.get('history', {})
    train_history = history.get('train', [])
    val_history = history.get('val', [])
    if [entry.get('epoch') for entry in train_history] != list(range(1, 31)):
        return None
    if [entry.get('epoch') for entry in val_history] != list(range(1, 31)):
        return None

    selection = summary.get('selection_state', {})
    epoch = selection.get('epoch')
    if selection.get('metric') != 'group_mAP_1.0' or not isinstance(epoch, int):
        return None
    selected = next(
        (entry for entry in val_history if entry.get('epoch') == epoch),
        None,
    )
    if selected is None or any(metric not in selected for metric in METRICS):
        return None
    metrics = {metric: float(selected[metric]) for metric in METRICS}
    if any(not math.isfinite(value) for value in metrics.values()):
        return None
    primary_values = [float(entry['group_mAP_1.0']) for entry in val_history]
    selected_value = selection.get('value')
    if (not all(math.isfinite(value) for value in primary_values)
            or not isinstance(selected_value, (int, float))
            or epoch != primary_values.index(max(primary_values)) + 1
            or not math.isclose(selected_value, metrics['group_mAP_1.0'])):
        return None
    return epoch, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_root', type=Path)
    parser.add_argument('--split', choices=['place', 'view'], default='place')
    parser.add_argument('--seeds', type=parse_seeds, default=parse_seeds('1,2,3'))
    args = parser.parse_args()

    records = []
    for seed in args.seeds:
        candidates = []
        for summary_path in args.result_root.rglob('summary.json'):
            try:
                run = load_run(summary_path, args.split, seed)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if run is not None:
                candidates.append((summary_path.stat().st_mtime_ns, summary_path, run))
        if not candidates:
            raise SystemExit(f'no complete B0-repro run found for seed {seed}')
        _, summary_path, (epoch, metrics) = max(candidates)
        records.append((seed, epoch, metrics, summary_path.parent))

    print('seed  epoch  mAP@1.0  mAP@0.5  outlier_mIoU')
    for seed, epoch, metrics, _ in records:
        print(f'{seed:4d}  {epoch:5d}  {metrics["group_mAP_1.0"]:8.2f}  '
              f'{metrics["group_mAP_0.5"]:8.2f}  {metrics["outlier_mIoU"]:12.2f}')
    print('mean         ' + '  '.join(
        f'{statistics.fmean(record[2][metric] for record in records):8.2f}' for metric in METRICS))
    print('std          ' + '  '.join(
        f'{statistics.pstdev(record[2][metric] for record in records):8.2f}' for metric in METRICS))


if __name__ == '__main__':
    main()
