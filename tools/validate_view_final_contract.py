#!/usr/bin/env python3
"""Validate frozen TRAIN+VAL -> TEST view runs before launching any training.

Reference args are the reviewed development view DP4 configs. Only the final
split policy and independent output/SAM cache destinations may differ. Trainer
args retain the original 115 fields and add only the two explicit TEST
selection/checkpoint retention controls. Queue policy belongs in the plan.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('b0', 'h1_sup', 'sam')
EVALUATOR_SHA256 = 'f1cd5f05b63ec769c149ad1b8926922eb03bb70ab3665fb7e6bf75cec28b40a0'
CACHE_VERSION = 'raw_bc294f454bfdfad21798'
DERIVED_FIELDS = {'protocol', 'eval_split', 'use_mae', 'mae_dim'}
OUTPUT_FIELDS = {'experiment_tag', 'checkpoint_root', 'result_path'}
METHOD_FIELDS = {'train_mask_mode', 'heatmap_variant', 'heatmap_sigma_ratio',
                 'sam_mask_manifest', 'sam_cache_root', 'sam_cache_version',
                 'sam_projected_cache'}
FINAL_CHANGED_FIELDS = {'val_mode', 'protocol', 'eval_split',
                        'experiment_tag', 'checkpoint_root'}
CONTROL_FIELDS = {'best_on_test', 'keep_all_checkpoints'}
SAM_PATH_FIELDS = {'sam_mask_manifest', 'sam_cache_root'}
EVALUATION_POLICY = {
    'split': 'test', 'epochs': list(range(1, 31)),
    'selection_metric': 'group_mAP_1.0', 'selection_label': 'best-on-TEST',
    'checkpoint': 'best_test.pth', 'keep_epoch30': True, 'keep_all_epochs': True,
}
FIXED_ARGS = {
    'dataset': 'cafe', 'research_b0_repro': True, 'split': 'view',
    'val_mode': False, 'protocol': 'best_on_test', 'eval_split': 'test',
    'best_on_test': True, 'keep_all_checkpoints': True,
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
    'eval_type': 'gt_base', 'clip_manifest': '',
}


class ContractError(ValueError):
    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details


def _require(condition, message, **details):
    if not condition:
        raise ContractError(message, **details)


def _equal(left, right):
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    return left == right


def _path(root, value):
    path = Path(value)
    return (path if path.is_absolute() else Path(root) / path).resolve()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def _independent(left, right):
    return left != right and left not in right.parents and right not in left.parents


def _check_args(args, variant, final):
    label = 'final' if final else 'reference'
    expected_fields = 117 if final else 115
    _require(isinstance(args, dict) and len(args) == expected_fields,
             'trainer args must contain 115 original fields plus only 2 explicit final controls',
             variant=variant, stage=label,
             actual=len(args) if isinstance(args, dict) else None)
    expected = dict(FIXED_ARGS)
    if not final:
        for field in CONTROL_FIELDS:
            expected.pop(field)
        expected.update(val_mode=True, protocol='development', eval_split='val')
    expected.update(
        train_mask_mode='sam_roi_blend' if variant == 'sam' else 'disabled',
        heatmap_variant='h1_sup' if variant == 'h1_sup' else 'off',
        heatmap_sigma_ratio=0.15 if variant == 'h1_sup' else 0.25,
        sam_projected_cache=variant == 'sam',
        sam_cache_version=CACHE_VERSION if variant == 'sam' else '')
    for key, value in expected.items():
        _require(key in args and _equal(args[key], value),
                 'fixed training protocol mismatch', stage=label,
                 variant=variant, field=key, expected=value, actual=args.get(key))
    for key in SAM_PATH_FIELDS:
        _require(isinstance(args.get(key), str) and bool(args[key]) == (variant == 'sam'),
                 'SAM cache path mismatch', stage=label, variant=variant, field=key)
    _require(isinstance(args.get('experiment_tag'), str) and
             re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}', args['experiment_tag']),
             'invalid experiment tag', variant=variant, stage=label)
    _require(isinstance(args.get('checkpoint_root'), str) and args['checkpoint_root'],
             'checkpoint root is required', variant=variant, stage=label)


def validate_contract(references, planned, evaluator_sha256, evaluation_policy, root=ROOT):
    root = Path(root).resolve()
    _require(evaluator_sha256 == EVALUATOR_SHA256, 'corrected evaluator SHA256 mismatch',
             expected=EVALUATOR_SHA256, actual=evaluator_sha256)
    _require(_canonical(evaluation_policy) == _canonical(EVALUATION_POLICY),
             'final evaluation policy must explicitly use best-on-TEST and retain every epoch',
             expected=EVALUATION_POLICY, actual=evaluation_policy)
    for label, collection in (('reference', references), ('final', planned)):
        _require(isinstance(collection, dict) and set(collection) == set(VARIANTS),
                 'exactly b0, h1_sup and sam are required', stage=label)
    pairwise = {}
    for variant in VARIANTS:
        previous, future = references[variant], planned[variant]
        _check_args(previous, variant, False)
        _check_args(future, variant, True)
        _require(set(previous) | CONTROL_FIELDS == set(future) and not (set(previous) & CONTROL_FIELDS),
                 'trainer args fields differ beyond the 2 explicit final controls', variant=variant)
        changes = {key: {'development': previous.get(key, '<absent>'), 'final': future[key]}
                   for key in sorted(future)
                   if key not in previous or not _equal(previous[key], future[key])}
        allowed = FINAL_CHANGED_FIELDS | CONTROL_FIELDS | (SAM_PATH_FIELDS if variant == 'sam' else set())
        unexpected = {key: value for key, value in changes.items() if key not in allowed}
        _require(not unexpected, 'development-to-final configuration drift',
                 variant=variant, unexpected_differences=unexpected)
        _require(FINAL_CHANGED_FIELDS <= set(changes),
                 'final split and independent outputs must all change', variant=variant)
        if variant == 'sam':
            for key in SAM_PATH_FIELDS:
                _require(_independent(_path(root, previous[key]), _path(root, future[key])),
                         'final SAM paths must be independent from development paths', field=key)
        pairwise[variant] = {
            'status': 'passed', 'reference_fields_checked': 115, 'args_fields_checked': 117,
            'changes': changes, 'unchanged_original_fields': 115 - len(set(changes) - CONTROL_FIELDS),
            'reference_args_sha256': hashlib.sha256(_canonical(previous)).hexdigest(),
            'final_args_sha256': hashlib.sha256(_canonical(future)).hexdigest(),
        }
    ignored = METHOD_FIELDS | OUTPUT_FIELDS
    common = {key: value for key, value in planned['b0'].items() if key not in ignored}
    _require(len(common) == 107, '105 original shared fields plus 2 common controls are required', actual=len(common))
    for variant in ('h1_sup', 'sam'):
        candidate = {key: value for key, value in planned[variant].items() if key not in ignored}
        _require(_canonical(candidate) == _canonical(common),
                 'common training configuration differs', variant=variant)
    tags = [planned[name]['experiment_tag'] for name in VARIANTS]
    roots = [_path(root, planned[name]['checkpoint_root']) for name in VARIANTS]
    _require(len(set(tags)) == 3 and len(set(roots)) == 3,
             'final methods must have distinct tags and checkpoint roots')
    _require(not set(tags) & {args['experiment_tag'] for args in references.values()},
             'final tag collides with development')
    for final_root in roots:
        for previous in references.values():
            _require(_independent(final_root, _path(root, previous['checkpoint_root'])),
                     'final checkpoint root overlaps development outputs')
    return {
        'schema_version': 1, 'status': 'passed', 'split': 'view',
        'protocol': 'new_dp4_corrected_evaluator', 'training_protocol': 'best_on_test',
        'evaluation_policy': EVALUATION_POLICY, 'shared_fields_checked': len(common),
        'shared_original_fields_checked': 105, 'new_control_fields': sorted(CONTROL_FIELDS),
        'shared_final_args': common, 'pairwise_development_to_final': pairwise,
        'evaluator': {'sha256': evaluator_sha256, 'status': 'passed'},
    }


def validate_runtime_policy(train_path):
    """Check the actual evaluation/selection guards without running training."""
    source = Path(train_path).read_text()
    tree = ast.parse(source)
    main = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'main'), None)
    _require(main is not None, 'trainer main function missing')
    parents = {child: node for node in ast.walk(main) for child in ast.iter_child_nodes(node)}

    def guarded(node, expression):
        child = node
        while child in parents:
            parent = parents[child]
            if isinstance(parent, ast.If) and child in parent.body and ast.dump(parent.test) == expression:
                return True
            child = parent
        return False

    val_test = ast.dump(ast.parse('args.val_mode or args.best_on_test', mode='eval').body)
    evaluate_test = ast.dump(ast.parse('should_evaluate', mode='eval').body)
    guard_expected = ast.dump(ast.parse(
        '(epoch % args.test_freq == 0) if (args.val_mode or args.best_on_test) else (epoch == args.epochs)',
        mode='eval').body)
    assignments = [node for node in ast.walk(main) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'should_evaluate'
                           for target in node.targets)]
    _require(len(assignments) == 1 and ast.dump(assignments[0].value) == guard_expected,
             'trainer does not evaluate best-on-TEST at every configured epoch')
    validates, selections, changes = [], [], []
    for node in ast.walk(main):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'validate':
            validates.append(node.lineno)
            _require(guarded(node, evaluate_test), 'evaluation bypasses evaluation policy guard', line=node.lineno)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'update_best':
            selections.append(node.lineno)
            _require(guarded(node, val_test), 'TEST selection is not explicitly gated by best_on_test', line=node.lineno)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'is_new_best'
                                               for target in node.targets):
            if isinstance(node.value, ast.Constant) and node.value.value is False:
                continue
            changes.append(node.lineno)
            _require(guarded(node, val_test), 'best checkpoint selection bypasses explicit policy', line=node.lineno)
    _require(len(validates) == 1 and len(selections) == 1 and len(changes) == 1,
             'unexpected trainer evaluation or selection implementation')
    has_test_checkpoint = any(isinstance(node, ast.Constant) and node.value == 'best_test.pth'
                              for node in ast.walk(tree))
    has_retention_control = any(isinstance(node, ast.Attribute) and node.attr == 'keep_all_checkpoints'
                                and isinstance(node.value, ast.Name) and node.value.id == 'args'
                                for node in ast.walk(tree))
    _require(has_test_checkpoint and has_retention_control,
             'trainer lacks explicit best_test checkpoint or all-epoch retention control')
    return {'status': 'passed', 'train_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'evaluation_policy_guard_line': assignments[0].lineno,
            'evaluation_lines': validates, 'explicit_selection_policy_lines': selections}


def load_trainer_parser(root, args):
    """Load the real parser; run_name prevents main() and any CUDA training."""
    root = Path(root).resolve()
    old_argv, old_path, old_cwd = sys.argv[:], sys.path[:], Path.cwd()
    old_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
    old_bytecode = sys.dont_write_bytecode
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root))
        os.chdir(str(root))
        sys.argv = [str(root / 'train.py'), '--groundtruth', str(_path(root, args['groundtruth'])),
                    '--labelmap', str(_path(root, args['labelmap']))]
        namespace = runpy.run_path(str(root / 'train.py'), run_name='final_contract_parser')
        for value in vars(namespace['args']).values():
            if hasattr(value, 'close'):
                value.close()
        return namespace['parser']
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
        os.chdir(str(old_cwd))
        sys.dont_write_bytecode = old_bytecode
        if old_cuda is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda


def build_argv(parser, args):
    argv = []
    for action in parser._actions:
        if action.dest in ('help', argparse.SUPPRESS) or action.dest not in args:
            continue
        option = next((item for item in action.option_strings if item.startswith('--')), None)
        _require(option is not None, 'trainer action lacks a long option', field=action.dest)
        value = args[action.dest]
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(option)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                argv.append(option)
        else:
            argv.extend([option, str(value)])
    return argv


def verify_argv(parser, argv, expected, root):
    old_cwd = Path.cwd()
    try:
        os.chdir(str(root))
        try:
            parsed = parser.parse_args(argv)
        except SystemExit as error:
            raise ContractError('real trainer parser rejected argv', exit_code=error.code)
    finally:
        os.chdir(str(old_cwd))
    actual = {}
    for key, value in vars(parsed).items():
        actual[key] = value.name if hasattr(value, 'name') and hasattr(value, 'close') else value
        if hasattr(value, 'close'):
            value.close()
    actual.update(protocol=('development' if actual['val_mode'] else
                            'best_on_test' if actual.get('best_on_test') else 'final_test'),
                  eval_split='val' if actual['val_mode'] else 'test',
                  use_mae=not actual['no_mae'],
                  mae_dim=(1408 if actual['mae_version'] == 'v2' else 768) if not actual['no_mae'] else 0)
    differences = {key: {'expected': expected.get(key), 'actual': actual.get(key)}
                   for key in set(expected) | set(actual)
                   if key not in expected or key not in actual or not _equal(expected[key], actual[key])}
    _require(not differences, 'real trainer parser roundtrip mismatch', differences=differences)
    return {'status': 'passed', 'args_fields_checked': len(actual)}


def audit_plan(root, plan_path):
    root = Path(root).resolve()
    plan_path = _path(root, plan_path)
    plan = json.loads(plan_path.read_text())
    runs = plan.get('runs')
    _require(isinstance(runs, dict) and set(runs) == set(VARIANTS),
             'plan requires exactly b0, h1_sup and sam')
    references, planned, sources = {}, {}, {}
    for variant in VARIANTS:
        sources[variant] = {}
        for field, collection in (('reference_args', references), ('resolved_args', planned)):
            path = _path(root, runs[variant][field])
            collection[variant] = json.loads(path.read_text())
            sources[variant][field] = str(path)
    evaluator = root / 'evaluation/cafe_eval.py'
    report = validate_contract(references, planned, hashlib.sha256(evaluator.read_bytes()).hexdigest(),
                               plan.get('evaluation_policy'), root)
    report['runtime_policy'] = validate_runtime_policy(root / 'train.py')
    parser = load_trainer_parser(root, planned['b0'])
    report['argv_roundtrips'] = {}
    for variant in VARIANTS:
        argv_path = _path(root, runs[variant]['argv'])
        sources[variant]['argv'] = str(argv_path)
        report['argv_roundtrips'][variant] = verify_argv(
            parser, json.loads(argv_path.read_text()), planned[variant], root)
    report.update(plan_path=str(plan_path), sources=sources)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit_plan(args.root, args.plan)
        code = 0
    except (ContractError, OSError, ValueError, TypeError, KeyError, StopIteration) as error:
        report = {'schema_version': 1, 'status': 'failed', 'error': str(error),
                  'details': getattr(error, 'details', {})}
        code = 2
    if args.output:
        output = _path(args.root, args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + '.%d.tmp' % os.getpid())
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        os.replace(str(temporary), str(output))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == '__main__':
    sys.exit(main())
