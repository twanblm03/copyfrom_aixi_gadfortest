#!/usr/bin/env python3
"""Prepare view argv and resolved args from the real matched place runs.

This imports train.py with runpy under a non-main name to inspect its parser.
It never calls training, reads datasets, loads a checkpoint, or uses a GPU.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.validate_view_training_contract import EVALUATOR_SHA256, audit_plan


REFERENCES = {
    'b0': 'result/[cafe]_GAD_[b0_randomseg_dp4_lr1_noclip_seed1]_[2026-10-20_21-51-49]/args.json',
    'h1_sup': 'result/[cafe]_GAD_[h1_sup_randomseg_dp4_lr1_noclip_seed1]_[2026-10-20_21-52-07]/args.json',
    'sam': 'result/[cafe]_GAD_[sam_randomsegments_dp4_lr1_noclip_seed1]_[2026-10-21_19-57-40]/args.json',
}
DERIVED = ('protocol', 'eval_split', 'use_mae', 'mae_dim')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n')


def close_files(namespace):
    for value in vars(namespace).values():
        if isinstance(value, io.IOBase):
            value.close()


def trainer_parser(groundtruth):
    original = sys.argv
    try:
        # The historical default groundtruth is intentionally absent locally.
        # Supplying the real path lets the parser open its FileType safely.
        sys.argv = ['train.py', '--groundtruth', groundtruth]
        module = runpy.run_path(str(ROOT / 'train.py'),
                               run_name='__view_protocol_config_parser__')
        close_files(module['args'])
        return module['parser']
    finally:
        sys.argv = original


def argv_for(parser, wanted):
    argv, visited = [], set()
    for action in parser._actions:
        if not action.option_strings or action.dest in visited:
            continue
        visited.add(action.dest)
        value = wanted[action.dest]
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            options = [item for item in parser._actions
                       if item.dest == action.dest and
                       isinstance(item, (argparse._StoreTrueAction, argparse._StoreFalseAction))]
            matching = [item for item in options if item.const is value]
            if matching:
                argv.append(matching[0].option_strings[0])
            elif value is not action.default:
                raise ValueError('No flag represents %s=%r' % (action.dest, value))
        else:
            argv.extend((action.option_strings[0], str(value)))
    return argv


def roundtrip(parser, argv, wanted):
    parsed = parser.parse_args(argv)
    try:
        resolved = {key: value.name if isinstance(value, io.IOBase) else value
                    for key, value in vars(parsed).items()}
    finally:
        close_files(parsed)
    resolved.update(
        protocol='development' if resolved['val_mode'] else 'final_test',
        eval_split='val' if resolved['val_mode'] else 'test',
        use_mae=not resolved['no_mae'],
        mae_dim=(0 if resolved['no_mae'] else
                 1408 if resolved['mae_version'] == 'v2' else 768))
    if resolved != wanted:
        differences = {key: {'wanted': wanted.get(key), 'parsed': resolved.get(key)}
                       for key in sorted(set(wanted) | set(resolved))
                       if wanted.get(key) != resolved.get(key) or
                       (key in wanted) != (key in resolved)}
        raise ValueError('Trainer argv roundtrip differs: %r' % differences)
    return resolved


def main():
    output = ROOT / 'configs/view_dp4'
    run_root = ROOT / '.local_data/view_dp4_20261021'
    references = {variant: json.loads((ROOT / path).read_text())
                  for variant, path in REFERENCES.items()}
    parser = trainer_parser(references['b0']['groundtruth'])
    plan = {
        'schema_version': 1, 'protocol': 'new_dp4_corrected_evaluator',
        'split': 'view', 'method_order': ['b0', 'h1_sup', 'sam'],
        'evaluator_sha256': EVALUATOR_SHA256,
        'derived_args': list(DERIVED), 'runs': {},
    }
    for variant, previous in references.items():
        wanted = dict(previous, split='view',
                      experiment_tag='view_' + previous['experiment_tag'],
                      checkpoint_root=str(run_root / variant / 'checkpoints'))
        if variant == 'sam':
            wanted.update(
                sam_mask_manifest=str(ROOT / '.local_data/sam/view_randomsegments_manifest.jsonl'),
                sam_cache_root=str(ROOT / '.local_data/sam/view_cache'))
        parser_keys = {action.dest for action in parser._actions if action.option_strings}
        if parser_keys != set(wanted) - set(DERIVED):
            raise ValueError('Real trainer parser fields no longer match the place args')
        argv = argv_for(parser, wanted)
        resolved = roundtrip(parser, argv, wanted)
        args_path, argv_path = output / (variant + '.args.json'), output / (variant + '.argv.json')
        write_json(args_path, resolved)
        write_json(argv_path, argv)
        plan['runs'][variant] = {
            'reference_args': REFERENCES[variant],
            'resolved_args': str(args_path.relative_to(ROOT)),
            'argv': str(argv_path.relative_to(ROOT)),
            'args_fields': len(resolved),
            'roundtrip': 'passed',
            'reference_args_sha256': hashlib.sha256((ROOT / REFERENCES[variant]).read_bytes()).hexdigest(),
        }
    plan_path = output / 'contract.plan.json'
    write_json(plan_path, plan)
    audit = audit_plan(ROOT, plan_path)
    audit['trainer_argv_roundtrip'] = {
        'status': 'passed', 'derived_fields': list(DERIVED),
        'fields_per_method': {name: row['args_fields'] for name, row in plan['runs'].items()},
        'loader': 'runpy.run_path(train.py, run_name=__view_protocol_config_parser__)',
    }
    write_json(run_root / 'config_audit.json', audit)
    print(json.dumps({
        'status': audit['status'], 'plan': str(plan_path),
        'audit': str(run_root / 'config_audit.json'),
        'fields_per_method': {name: row['args_fields'] for name, row in plan['runs'].items()},
        'shared_fields_checked': audit['shared_fields_checked'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
