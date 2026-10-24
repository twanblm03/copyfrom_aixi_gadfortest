#!/usr/bin/env python3
"""Reject view experiments that drift from the matched place DP4 protocol.

Plan paths are relative to --root. A plan contains ``runs`` keyed by b0,
h1_sup, and sam, each with reference_args and resolved_args JSON paths. A
list of entries with a variant field is also accepted. Queue metadata belongs
in the plan, never inside a resolved trainer args dictionary.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('b0', 'h1_sup', 'sam')
EVALUATOR_SHA256 = 'f1cd5f05b63ec769c149ad1b8926922eb03bb70ab3665fb7e6bf75cec28b40a0'
OUTPUT_FIELDS = {'experiment_tag', 'checkpoint_root', 'result_path'}
SAM_PATH_FIELDS = {'sam_mask_manifest', 'sam_cache_root'}
METHOD_FIELDS = {
    'train_mask_mode', 'heatmap_variant', 'heatmap_sigma_ratio',
    'sam_mask_manifest', 'sam_cache_root', 'sam_cache_version',
    'sam_projected_cache',
}
FIXED_ARGS = {
    'dataset': 'cafe', 'research_b0_repro': True, 'val_mode': True,
    'backbone': 'resnet18', 'no_mae': True, 'use_mae': False,
    'image_width': 1280, 'image_height': 720, 'num_frame': 5,
    'num_group_tokens': 12, 'fixed_center_train': False,
    'random_sampling': False, 'frozen_batch_norm': False,
    'freeze_backbone': False, 'backbone_lr_scale': 1,
    'random_seed': 1, 'epochs': 30, 'test_freq': 1,
    'selection_metric': 'group_mAP_1.0', 'batch': 16,
    'micro_batch': 0, 'test_batch': 4, 'optimizer': 'adam',
    'lr': 1e-5, 'max_lr': 1e-4, 'warmup_epochs': 5,
    'weight_decay': 0, 'lr_schedule': 'linear', 'max_train_steps': 0,
    'gradient_clipping': False, 'distributed': False,
    'parallel_transport': 'host', 'device': '0,1,2,3',
    'mask_context_fraction': 0.5, 'load_model': False, 'model_path': '',
    'protocol': 'development', 'eval_split': 'val', 'eval_type': 'gt_base',
}


class ContractError(ValueError):
    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details


def _equal(left, right):
    # A JSON boolean must not silently match numeric 0/1 in a protocol check.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def _require(condition, message, **details):
    if not condition:
        raise ContractError(message, **details)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def _path(root, value):
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _check_args(args, variant, split):
    _require(isinstance(args, dict), 'trainer args must be a JSON object',
             variant=variant, split=split)
    for key, expected in dict(FIXED_ARGS, split=split).items():
        _require(key in args and _equal(args[key], expected),
                 'fixed training protocol mismatch', variant=variant,
                 split=split, field=key, expected=expected, actual=args.get(key))
    method = {
        'train_mask_mode': 'sam_roi_blend' if variant == 'sam' else 'disabled',
        'heatmap_variant': 'h1_sup' if variant == 'h1_sup' else 'off',
        'heatmap_sigma_ratio': 0.15 if variant == 'h1_sup' else 0.25,
        'sam_projected_cache': variant == 'sam',
    }
    for key, expected in method.items():
        _require(key in args and _equal(args[key], expected),
                 'method configuration mismatch', variant=variant,
                 split=split, field=key, expected=expected, actual=args.get(key))
    for key in ('sam_mask_manifest', 'sam_cache_root', 'sam_cache_version'):
        expected_nonempty = variant == 'sam'
        _require(isinstance(args.get(key), str) and
                 bool(args[key]) == expected_nonempty,
                 'SAM cache configuration mismatch', variant=variant,
                 split=split, field=key, actual=args.get(key))
    _require(isinstance(args.get('experiment_tag'), str) and
             re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}', args['experiment_tag']),
             'invalid experiment tag', variant=variant, split=split)
    _require(isinstance(args.get('checkpoint_root'), str) and args['checkpoint_root'],
             'checkpoint root is required', variant=variant, split=split)


def validate_contract(place_args, view_args, evaluator_sha256, root=ROOT):
    """Return audit details or raise ContractError before launching any job."""
    root = Path(root).resolve()
    _require(evaluator_sha256 == EVALUATOR_SHA256,
             'corrected evaluator SHA256 mismatch', expected=EVALUATOR_SHA256,
             actual=evaluator_sha256)
    for label, runs in (('place', place_args), ('view', view_args)):
        _require(set(runs) == set(VARIANTS), 'exactly three methods are required',
                 split=label, expected=list(VARIANTS), actual=sorted(runs))

    pairwise = {}
    for variant in VARIANTS:
        previous, planned = place_args[variant], view_args[variant]
        _check_args(previous, variant, 'place')
        _check_args(planned, variant, 'view')
        _require(set(previous) == set(planned), 'trainer args fields differ',
                 variant=variant, missing=sorted(set(previous) - set(planned)),
                 extra=sorted(set(planned) - set(previous)))
        allowed = {'split'} | OUTPUT_FIELDS
        if variant == 'sam':
            allowed |= SAM_PATH_FIELDS
        changes = {key: {'place': previous[key], 'view': planned[key]}
                   for key in sorted(previous)
                   if not _equal(previous[key], planned[key])}
        unexpected = {key: change for key, change in changes.items()
                      if key not in allowed}
        _require(not unexpected, 'place-to-view configuration drift',
                 variant=variant, unexpected_differences=unexpected)
        _require(previous['experiment_tag'] != planned['experiment_tag'],
                 'view experiment tag would reuse place output', variant=variant)
        _require(_path(root, previous['checkpoint_root']) !=
                 _path(root, planned['checkpoint_root']),
                 'view checkpoint root would reuse place output', variant=variant)
        if variant == 'sam':
            for key in SAM_PATH_FIELDS:
                _require(_path(root, previous[key]) != _path(root, planned[key]),
                         'view SAM paths must be isolated from place', field=key)
        pairwise[variant] = {
            'status': 'passed', 'args_fields_checked': len(previous),
            'unchanged_fields': len(previous) - len(changes),
            'allowed_changed_fields': sorted(allowed), 'changes': changes,
            'place_args_sha256': hashlib.sha256(_canonical(previous)).hexdigest(),
            'view_args_sha256': hashlib.sha256(_canonical(planned)).hexdigest(),
        }

    # Ignore only the method switches and output destinations. Everything else
    # must be shared, including dormant heatmap settings and SAM geometry.
    ignored = METHOD_FIELDS | OUTPUT_FIELDS
    common = {key: value for key, value in view_args['b0'].items()
              if key not in ignored}
    for variant in ('h1_sup', 'sam'):
        candidate = {key: value for key, value in view_args[variant].items()
                     if key not in ignored}
        _require(set(common) == set(candidate), 'common args fields differ',
                 variant=variant)
        differences = {key: {'b0': common[key], variant: candidate[key]}
                       for key in sorted(common)
                       if not _equal(common[key], candidate[key])}
        _require(not differences, 'common training configuration differs',
                 variant=variant, unexpected_differences=differences)

    tags = [view_args[name]['experiment_tag'] for name in VARIANTS]
    roots = [_path(root, view_args[name]['checkpoint_root']) for name in VARIANTS]
    _require(len(set(tags)) == 3, 'view methods must have distinct experiment tags')
    _require(len(set(roots)) == 3, 'view methods must have distinct checkpoint roots')
    place_tags = {args['experiment_tag'] for args in place_args.values()}
    place_roots = {_path(root, args['checkpoint_root']) for args in place_args.values()}
    _require(not (set(tags) & place_tags), 'view tag collides with a place method')
    _require(not (set(roots) & place_roots), 'view checkpoint root collides with place')
    return {
        'schema_version': 1, 'status': 'passed',
        'protocol': 'new_dp4_corrected_evaluator', 'split': 'view',
        'evaluator': {'path': 'evaluation/cafe_eval.py',
                      'sha256': evaluator_sha256, 'status': 'passed'},
        'pairwise_place_to_view': pairwise,
        'shared_view_args': common, 'shared_fields_checked': len(common),
        'method_specific_fields': sorted(METHOD_FIELDS),
        'fixed_training_args': FIXED_ARGS,
    }


def audit_plan(root, plan_path):
    root = Path(root).resolve()
    plan_path = _path(root, plan_path)
    plan = json.loads(plan_path.read_text())
    _require(isinstance(plan, dict), 'plan must be a JSON object')
    runs = plan.get('runs')
    if isinstance(runs, list):
        _require(all(isinstance(entry, dict) and
                     isinstance(entry.get('variant'), str) for entry in runs),
                 'list plan entries require a variant string')
        names = [entry.get('variant') for entry in runs]
        _require(len(names) == len(set(names)), 'duplicate method in plan')
        runs = {entry['variant']: entry for entry in runs}
    _require(isinstance(runs, dict) and set(runs) == set(VARIANTS),
             'plan runs must contain b0, h1_sup, and sam')
    place_args, view_args, sources = {}, {}, {}
    for variant in VARIANTS:
        entry = runs[variant]
        _require(isinstance(entry, dict), 'invalid method plan', variant=variant)
        sources[variant] = {}
        for field, target in (('reference_args', place_args), ('resolved_args', view_args)):
            _require(isinstance(entry.get(field), str) and entry[field],
                     'args JSON path is required in plan', variant=variant, field=field)
            path = _path(root, entry[field])
            target[variant] = json.loads(path.read_text())
            sources[variant][field] = str(path)
    evaluator = root / 'evaluation/cafe_eval.py'
    evaluator_sha256 = hashlib.sha256(evaluator.read_bytes()).hexdigest()
    report = validate_contract(place_args, view_args, evaluator_sha256, root)
    report.update(plan_path=str(plan_path), sources=sources)
    return report


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.%d.tmp' % os.getpid())
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    os.replace(str(temporary), str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit_plan(args.root, args.plan)
        code = 0
    except (ContractError, OSError, ValueError, TypeError, KeyError) as error:
        report = {'schema_version': 1, 'status': 'failed', 'error': str(error),
                  'details': getattr(error, 'details', {})}
        code = 2
    if args.output:
        _write_report(_path(args.root.resolve(), args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == '__main__':
    sys.exit(main())
