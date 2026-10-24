#!/usr/bin/env python3
"""Resume interrupted view B0 after place SAM, using the original frozen trainer."""
import argparse
import copy
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil

RESUME_FIELDS = {'load_model', 'model_path', 'experiment_tag', 'checkpoint_root'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def validate_checkpoint(checkpoint, planned, steps_per_epoch):
    epoch = checkpoint['epoch']
    if not isinstance(epoch, int) or not 0 < epoch < planned['epochs']:
        raise ValueError('checkpoint epoch cannot be continued')
    if checkpoint.get('protocol') != 'development' or checkpoint.get('eval_split') != 'val':
        raise ValueError('checkpoint protocol differs from TRAIN->VAL')
    if not checkpoint.get('state_dict') or not checkpoint.get('optimizer', {}).get('state'):
        raise ValueError('checkpoint lacks model or Adam state')
    for split in ('train', 'val'):
        rows = checkpoint['history'][split]
        if [row['epoch'] for row in rows] != list(range(1, epoch + 1)):
            raise ValueError('checkpoint history is incomplete: ' + split)
        for row in rows:
            if not math.isfinite(row['loss']):
                raise ValueError('checkpoint history has nonfinite loss')
            if split == 'train' and row.get('skipped_nonfinite_steps') != 0:
                raise ValueError('checkpoint contains skipped optimizer steps')
    if checkpoint['history'].get('test'):
        raise ValueError('checkpoint contains TEST results')
    optimizer, scheduler = checkpoint['optimizer'], checkpoint['scheduler']
    if scheduler['last_epoch'] != epoch:
        raise ValueError('scheduler does not match checkpoint epoch')
    groups = optimizer['param_groups']
    if len(groups) != 2 or [group['lr'] for group in groups] != scheduler['_last_lr']:
        raise ValueError('Adam and scheduler learning rates differ')
    expected_steps = epoch * steps_per_epoch
    if any(float(state['step']) != expected_steps for state in optimizer['state'].values()):
        raise ValueError('Adam update count differs from completed epochs')
    selection = checkpoint['selection_state']
    if selection['metric'] != planned['selection_metric'] or selection['mode'] != 'max':
        raise ValueError('checkpoint selection metric changed')
    if not 1 <= selection['epoch'] <= epoch or not math.isfinite(selection['value']):
        raise ValueError('checkpoint best-selection state is invalid')
    return dict(saved_epoch=epoch, next_epoch=epoch + 1, target_epochs=planned['epochs'],
                completed_optimizer_steps=expected_steps,
                next_learning_rates=[group['lr'] for group in groups],
                best_epoch=selection['epoch'], best_value=selection['value'],
                rng_state_saved=any('rng' in key.lower() for key in checkpoint))


def validate_args(original, resumed):
    if set(original) != set(resumed):
        raise ValueError('resumed args fields differ')
    changes = {key: dict(original=original[key], resumed=resumed[key])
               for key in original if original[key] != resumed[key]}
    if set(changes) != RESUME_FIELDS or resumed['load_model'] is not True:
        raise ValueError('continuation changed training configuration: ' + str(changes))
    return changes


def frozen_module(plan):
    path = Path(plan['snapshot_root']) / 'tools/queue_view_dp4.py'
    spec = importlib.util.spec_from_file_location('frozen_view_queue', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(source_path):
    import torch
    source_path = Path(source_path).resolve()
    source = read(source_path)
    parent = Path(source['queue_root'])
    original = read(source['runs']['b0']['resolved_args'])
    checkpoints = list(Path(original['checkpoint_root']).glob('*/epoch*.pth'))
    if not checkpoints:
        raise ValueError('no B0 checkpoint is available')
    checkpoint_path = max(checkpoints, key=lambda path: int(path.stem[5:]))
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    audit = validate_checkpoint(checkpoint, original, source['expected_inventory']['steps_per_epoch'])
    epoch = audit['saved_epoch']
    base = parent / ('continuation_e%d_20261022' % epoch)
    if base.exists():
        raise ValueError('continuation already prepared: ' + str(base))
    previous_result = Path(source['workspace']) / 'result' / checkpoint_path.parent.name
    if read(previous_result / 'args.json') != original:
        raise ValueError('interrupted run args do not match the frozen B0 configuration')
    source_best = Path(checkpoint['selection_state']['checkpoint_path'])
    best = torch.load(str(source_best), map_location='cpu', weights_only=False)
    if best['epoch'] != audit['best_epoch']:
        raise ValueError('historical best checkpoint does not match selection history')
    base.mkdir(parents=True)
    inputs = base / 'inputs'
    inputs.mkdir()
    resume_checkpoint = inputs / checkpoint_path.name
    historical_best = inputs / 'historical_best_val.pth'
    shutil.copy2(str(checkpoint_path), str(resume_checkpoint))
    shutil.copy2(str(source_best), str(historical_best))
    resumed = dict(original, load_model=True, model_path=str(resume_checkpoint),
                   experiment_tag=original['experiment_tag'] + '_resume_e%d' % epoch,
                   checkpoint_root=str(base / 'b0/checkpoints'))
    changes = validate_args(original, resumed)
    argv = read(source['runs']['b0']['argv'])
    for key in ('model_path', 'experiment_tag', 'checkpoint_root'):
        argv[argv.index('--' + key) + 1] = resumed[key]
    argv.append('--load_model')
    args_path, argv_path = base / 'b0.args.json', base / 'b0.argv.json'
    write(args_path, resumed)
    write(argv_path, argv)
    runner = base / 'resume_view_queue.py'
    shutil.copy2(str(Path(__file__).resolve()), str(runner))
    plan = copy.deepcopy(source)
    plan['queue_root'] = str(base)
    plan['runs']['b0']['resolved_args'] = str(args_path)
    plan['runs']['b0']['argv'] = str(argv_path)
    plan['continuation'] = dict(parent_queue_root=str(parent), source_plan=str(source_path),
                                previous_result=str(previous_result),
                                source_checkpoint=str(checkpoint_path),
                                historical_best=str(historical_best),
                                original_args=source['runs']['b0']['resolved_args'],
                                checkpoint_audit=audit, args_changes=changes,
                                user_request='view B0中断了，我们把它排在现在SAM的后面，接着训练。')
    for path in (source_path, resume_checkpoint, historical_best, args_path, argv_path, runner):
        plan['frozen_files'].append(dict(path=str(path), sha256=sha(path)))
    write(base / 'continuation_audit.json', dict(status='passed', **plan['continuation']))
    write(base / 'queue.plan.json', plan)
    print(json.dumps(dict(plan=str(base / 'queue.plan.json'), **audit), indent=2))


def run_or_check(plan_path, check):
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    module = frozen_module(plan)
    details = plan['continuation']
    parent = Path(details['parent_queue_root'])

    class ContinuationQueue(module.Queue):
        def wait_dependency(self):
            if details.get('scheduling') == 'start_when_memory_available':
                self.status('place_dependency_released_by_user',
                            message='Resume view B0 as soon as four-card memory checks pass',
                            user_request=details['latest_user_request'])
                return
            super().wait_dependency()
            if details.get('sam_recovery_plan'):
                upstream = read(details['sam_recovery_plan'])
                public = Path(upstream['sam_recovery']['public_run_root'])
                (public.parent / 'queue.status').write_text('pipeline_complete ' + module.now() + '\n')

        def status(self, stage, **kwargs):
            super().status(stage, **kwargs)
            state = read(self.base / 'queue_state.json')
            state.update(continuation_root=str(self.base),
                         saved_epoch=details['checkpoint_audit']['saved_epoch'],
                         next_epoch=details['checkpoint_audit']['next_epoch'])
            module.atomic_json(parent / 'queue_state.json', state)
            (parent / 'queue.status').write_text(stage + ' ' + state['updated_at'] + '\n')

        def verify_training(self, variant, planned):
            if variant == 'b0':
                outputs = [path for path in (self.root / 'result').iterdir()
                           if ('_[' + planned['experiment_tag'] + ']_') in path.name]
                if len(outputs) != 1:
                    raise RuntimeError('expected one resumed B0 result directory')
                result = outputs[0]
                target = Path(planned['checkpoint_root']) / result.name / 'best_val.pth'
                summary = read(result / 'summary.json')
                if not target.exists():
                    # The trainer preserves earlier selection state. If no later
                    # epoch improves it, retain the reviewed historical best.
                    selected = summary['best'][planned['selection_metric']]
                    previous = details['checkpoint_audit']
                    if selected != dict(epoch=previous['best_epoch'], value=previous['best_value']):
                        raise RuntimeError('resumed best checkpoint is missing')
                    shutil.copy2(details['historical_best'], str(target))
                    module.atomic_json(self.base / 'b0/historical_best_retained.json',
                                       dict(source=details['historical_best'], destination=str(target),
                                            selected=selected))
            super().verify_training(variant, planned)

    queue = ContinuationQueue(plan_path)
    queue.verify_frozen()
    changes = validate_args(read(details['original_args']), read(plan['runs']['b0']['resolved_args']))
    if changes != details['args_changes']:
        raise ValueError('continuation args audit changed')
    # This checks every original method against its matching place protocol.
    # The four explicit continuation fields above are the only exception.
    queue.audit_contract()
    if check:
        print('Frozen training protocol and continuation inputs: passed', flush=True)
        return
    parent_lock = (parent / 'queue.lock').open('a')
    fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    queue.run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'check', 'run'])
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.plan)
    else:
        run_or_check(args.plan, args.command == 'check')


if __name__ == '__main__':
    main()
