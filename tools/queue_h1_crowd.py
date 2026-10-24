#!/usr/bin/env python3
"""Run one matched place crowd H1 only after the existing queues complete.

``check`` performs CPU checks only. ``run`` starts a waiting successor and
launches the frozen trainer after both dependencies and the preflight pass.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import fcntl


HELPER = Path(__file__).with_name('queue_view_dp4.py')
SPEC = importlib.util.spec_from_file_location('crowd_view_queue_helpers', str(HELPER))
view_queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(view_queue)

EVALUATOR_SHA256 = 'f1cd5f05b63ec769c149ad1b8926922eb03bb70ab3665fb7e6bf75cec28b40a0'
NEW_ARGS = {'heatmap_target_mode': 'crowd', 'heatmap_crowd_cap_ratio': .35}
OUTPUT_ARGS = {'experiment_tag', 'checkpoint_root'}
VARIANT = 'h1_crowd'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def validate_args(anchor, planned):
    """Only two output destinations and the two target fields may differ."""
    require(not (set(NEW_ARGS) & set(anchor)), 'anchor unexpectedly contains target-mode fields')
    require(set(planned) == set(anchor) | set(NEW_ARGS), 'crowd trainer args schema differs')
    require(all(same(planned[key], value) for key, value in NEW_ARGS.items()),
            'crowd target definition differs from the registered target')
    changed = {key for key in anchor if not same(anchor[key], planned[key])}
    require(changed == OUTPUT_ARGS, 'anchor argument drift: ' + repr(sorted(changed)))
    fixed = dict(split='place', random_seed=1, heatmap_variant='h1_sup',
                 train_mask_mode='disabled', epochs=30, test_freq=1, batch=16,
                 test_batch=4, fixed_center_train=False, selection_metric='group_mAP_1.0',
                 protocol='development', eval_split='val', device='0,1,2,3',
                 distributed=False, parallel_transport='host', gradient_clipping=False,
                 heatmap_sigma_ratio=.15, heatmap_sigma_min=.5, heatmap_sigma_max=4.)
    require(all(key in anchor and same(anchor[key], value) for key, value in fixed.items()),
            'anchor is not the matched place DP4 H1 protocol')
    return dict(status='passed', anchor_fields=len(anchor), crowd_fields=len(planned),
                changed_anchor_fields=sorted(changed), added_target_fields=NEW_ARGS)


def validate_baseline(args, summary, checkpoint_dir):
    summary_args = dict(summary.get('args', {}))
    # Trainer appends this observed memory statistic after saving args.json.
    # It is not a parser argument or an optimization setting.
    summary_args.pop('peak_cuda_memory_mb', None)
    require(same(summary_args, args), 'baseline summary trainer args differ from args.json')
    history = summary['history']
    for split in ('train', 'val'):
        rows = history.get(split, [])
        require([item['epoch'] for item in rows] == list(range(1, 31)),
                'baseline does not have 30 complete ' + split + ' epochs')
        keys = ('loss',) if split == 'train' else (
            'loss', 'group_mAP_1.0', 'group_mAP_0.5', 'outlier_mIoU')
        require(all(all(math.isfinite(row[key]) for key in keys) for row in rows),
                'baseline contains nonfinite metrics')
        if split == 'train':
            require(all(row.get('skipped_nonfinite_steps') == 0 for row in rows),
                    'baseline skipped nonfinite optimizer steps')
    require(not history.get('test'), 'baseline unexpectedly used TEST')
    best = summary['best']['group_mAP_1.0']
    best_row = max(history['val'], key=lambda item: item['group_mAP_1.0'])
    require(best['epoch'] == best_row['epoch'] and best['value'] == best_row['group_mAP_1.0'],
            'baseline best VAL selection disagrees with complete history')
    for name in ('best_val.pth', 'epoch30.pth'):
        require((Path(checkpoint_dir) / name).is_file(), 'baseline checkpoint missing: ' + name)
    return dict(status='passed', train_epochs=30, val_epochs=30, test_epochs=0, best=best)


def validate_geometry(report, reference_sha256):
    fixed = dict(status='passed', split='place', partition='TRAIN only',
                 cap_ratio=.35, sigma_min=.5, grid=[23, 40],
                 reference_args_sha256=reference_sha256)
    require(all(key in report and same(report[key], value) for key, value in fixed.items()),
            'geometry preflight does not match the place crowd target/anchor')
    # This also rejects NaN/Inf hidden in per-clip rows or unrecognized fields.
    try:
        json.dumps(report, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise RuntimeError('geometry preflight contains nonfinite/unserializable data') from error
    counts = report['counts']
    valid, changed = counts['valid_actor_frames'], counts['shrunk_actor_frames']
    require(report['clips'] > 0 and counts['group_frames'] > 0 and 0 < changed <= valid,
            'geometry preflight found no effective crowd target change')
    require(math.isclose(report['changed_fraction'], changed / valid, rel_tol=1e-12),
            'geometry preflight changed fraction disagrees with counts')
    before, after = report['mean_target_mass_before'], report['mean_target_mass_after']
    require(math.isfinite(before) and math.isfinite(after) and 0 <= after < before,
            'geometry preflight did not change rasterized target mass')
    for key in ('mean_nonmember_center_response_before', 'mean_nonmember_center_response_after'):
        require(math.isfinite(report[key]) and report[key] >= 0,
                'geometry preflight contains invalid center response')
    return dict(status='passed', clips=report['clips'], changed_fraction=report['changed_fraction'])


PARSER_SCRIPT = r'''
import argparse, io, json, sys
from pathlib import Path
payload = json.load(sys.stdin)
root = Path(payload['root'])
sys.path.insert(0, str(root))
source = (root / 'train.py').read_text()
prefix, marker, _ = source.partition('\nargs = parser.parse_args()')
if not marker:
    raise RuntimeError('trainer parser boundary changed')
namespace = {'__name__': 'crowd_parser_contract', '__file__': str(root / 'train.py')}
exec(compile(prefix, str(root / 'train.py'), 'exec'), namespace)
parser = namespace['parser']
expected = payload['expected']
derived = {'protocol', 'eval_split', 'use_mae', 'mae_dim'}
known = {action.dest for action in parser._actions}
if set(expected) - known != derived:
    raise RuntimeError('trainer argument schema changed')
argv = []
for action in parser._actions:
    if action.dest not in expected or not action.option_strings:
        continue
    value = expected[action.dest]
    option = next((s for s in action.option_strings if s.startswith('--')), action.option_strings[0])
    if isinstance(action, argparse._StoreTrueAction):
        if value: argv.append(option)
    elif isinstance(action, argparse._StoreFalseAction):
        if not value: argv.append(option)
    elif isinstance(action, argparse._StoreAction):
        argv.extend([option, str(value)])
    else:
        raise RuntimeError('unsupported parser action: ' + action.dest)
parsed = vars(parser.parse_args(argv))
for key, value in list(parsed.items()):
    if isinstance(value, io.IOBase):
        parsed[key] = value.name
        value.close()
parsed.update(protocol='development' if parsed['val_mode'] else 'final_test',
              eval_split='val' if parsed['val_mode'] else 'test',
              use_mae=not parsed['no_mae'],
              mae_dim=0 if parsed['no_mae'] else (1408 if parsed['mae_version'] == 'v2' else 768))
differences = {k: [expected.get(k), parsed.get(k)] for k in set(expected) | set(parsed)
               if k not in expected or k not in parsed or expected[k] != parsed[k]
               or (isinstance(expected[k], bool) != isinstance(parsed[k], bool))}
if differences:
    raise RuntimeError('real trainer parser roundtrip mismatch: ' + repr(differences))
print('CROWD_PARSER_RESULT=' + json.dumps(dict(status='passed', fields=len(parsed), argv=argv)))
'''


def parser_roundtrip(root, python, expected, cwd):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([python, '-c', PARSER_SCRIPT],
                            input=json.dumps(dict(root=str(root), expected=expected)),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            cwd=str(cwd), env=environment)
    require(result.returncode == 0, 'CPU trainer parser check failed: ' + result.stderr[-6000:])
    lines = [line for line in result.stdout.splitlines() if line.startswith('CROWD_PARSER_RESULT=')]
    require(len(lines) == 1, 'CPU parser did not return exactly one result')
    return json.loads(lines[0].split('=', 1)[1])


def dependency_stage(view_state, view_statuses, view_audits, place_status, place_alive,
                     view_locked, view_alive=False):
    """Pure state gate, independently testable without processes or GPUs."""
    stage = view_state.get('stage', '')
    if stage in ('failed', 'interrupted'):
        raise RuntimeError('view queue did not complete normally: ' + stage)
    place = place_status.split()[0] if place_status.strip() else ''
    if place in ('training_failed', 'training_interrupted', 'pipeline_failed'):
        raise RuntimeError('place SAM did not complete normally: ' + place)
    if place != 'training_complete':
        require(place_alive, 'place SAM exited without training_complete')
    for name, status in view_statuses.items():
        if status.split()[:1] in (['training_failed'], ['training_interrupted']):
            raise RuntimeError('view training dependency failed: ' + name)
    if stage != 'complete':
        require(view_locked or view_alive, 'view queue exited without complete')
        return 'waiting_existing_view_queue'
    require(view_state.get('completed_runs') == ['b0', 'h1_sup', 'sam'],
            'view queue completion inventory differs')
    for name in ('b0', 'h1_sup', 'sam'):
        require(view_statuses.get(name, '').split()[:1] == ['training_complete'],
                'view dependency missing training_complete: ' + name)
        audit = view_audits.get(name, {})
        require(audit.get('status') == 'passed' and audit.get('train_epochs') == 30
                and audit.get('val_epochs') == 30 and audit.get('test_epochs') == 0,
                'view dependency completion audit not passed: ' + name)
    if view_locked or view_alive:
        return 'waiting_view_queue_exit'
    if place != 'training_complete':
        return 'waiting_place_sam'
    if place_alive:
        return 'waiting_place_sam_exit'
    return 'ready'


def lock_is_held(path):
    # A lock file is not created or rewritten by this read-only dependency check.
    if not Path(path).is_file():
        return False
    with Path(path).open('r') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


class CrowdQueue(view_queue.Queue):
    def verify_frozen(self):
        manifest = self.plan['source_manifest']
        require(view_queue.digest(manifest['path']) == manifest['sha256'],
                'crowd source manifest changed')
        require(same(view_queue.read_json(manifest['path'])['files'], self.plan['frozen_files']),
                'crowd plan and frozen source manifest differ')
        super().verify_frozen()

    def audit_contract(self):
        require(self.plan.get('schema_version') == 1 and
                self.plan.get('protocol') == 'new_dp4_corrected_evaluator' and
                self.plan.get('method_order') == [VARIANT] and
                set(self.plan['runs']) == {VARIANT}, 'unexpected successor plan schema')
        self.verify_frozen()
        upstream = view_queue.read_json(self.plan['upstream_plan'])
        for key in ('workspace', 'python', 'gpu_uuids', 'minimum_free_mib', 'poll_seconds', 'dependency'):
            require(same(self.plan[key], upstream[key]), 'upstream queue protocol drift: ' + key)
        require(same(self.plan['environment'], dict(upstream['environment'], PYTHONDONTWRITEBYTECODE='1')),
                'upstream queue environment drift')
        baseline = self.plan['baseline']
        anchor = view_queue.read_json(baseline['args'])
        geometry = validate_geometry(view_queue.read_json(self.plan['preflight']['geometry_report']),
                                     view_queue.digest(baseline['args']))
        planned = view_queue.read_json(self.plan['runs'][VARIANT]['resolved_args'])
        argument_audit = validate_args(anchor, planned)
        require(Path(planned['checkpoint_root']).resolve() ==
                (self.base / VARIANT / 'checkpoints').resolve(), 'checkpoint isolation differs')
        require(planned['experiment_tag'] == 'place_h1_crowd_randomseg_dp4_lr1_noclip_seed1',
                'unexpected crowd experiment tag')
        require(Path(baseline['status']).read_text().split()[:1] == ['training_complete'],
                'baseline training not complete')
        baseline_audit = validate_baseline(anchor, view_queue.read_json(baseline['summary']),
                                           baseline['checkpoint_dir'])
        require(view_queue.digest(self.snapshot / 'evaluation/cafe_eval.py') == EVALUATOR_SHA256,
                'corrected evaluator SHA256 mismatch')
        paired_box = dict(anchor, heatmap_target_mode='box', heatmap_crowd_cap_ratio=.35)
        parser_roundtrip(self.snapshot, self.python, paired_box, self.root)
        parsed = parser_roundtrip(self.snapshot, self.python, planned, self.root)
        require(same(parsed['argv'], view_queue.read_json(self.plan['runs'][VARIANT]['argv'])),
                'frozen training argv differs from real parser roundtrip')
        report = dict(status='passed', protocol=self.plan['protocol'], args=argument_audit,
                      baseline=baseline_audit, geometry=geometry, parser_roundtrip='passed',
                      evaluator_sha256=EVALUATOR_SHA256, checked_at=view_queue.now())
        view_queue.atomic_json(self.base / 'launch_config_audit.json', report)
        return report

    def dependency_state(self):
        dependency = self.plan['dependency']
        view = self.plan['view_dependency']
        state = view_queue.read_json(view['state'])
        statuses, audits = {}, {}
        for name in ('b0', 'h1_sup', 'sam'):
            run = Path(view['queue_root']) / name
            statuses[name] = (run / 'training.status').read_text() if (run / 'training.status').exists() else ''
            if (run / 'completion_audit.json').exists():
                audits[name] = view_queue.read_json(run / 'completion_audit.json')
        place_status = Path(dependency['status']).read_text()
        place_alive = any(view_queue.process_matches(p['pid'], p['start_ticks'])
                          for p in dependency['processes'])
        process = view.get('process')
        view_alive = bool(process and view_queue.process_matches(process['pid'], process['start_ticks']))
        stage = dependency_stage(state, statuses, audits, place_status, place_alive,
                                 lock_is_held(view['lock']), view_alive)
        return stage, state

    def wait_dependency(self):
        while True:
            stage, state = self.dependency_state()
            if stage == 'ready':
                return
            self.status(stage, existing_view_stage=state.get('stage'))
            time.sleep(self.interval)

    def preflight(self):
        self.verify_frozen()
        hook = self.plan['preflight']
        require(hook.get('cpu_only') is True and hook.get('argv'), 'CPU preflight hook missing')
        self.status('cpu_target_preflight')
        self.command(hook['argv'], self.base / 'target_preflight.log',
                     env=dict(self.env, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'),
                     cwd=self.snapshot)
        view_queue.atomic_json(self.base / 'target_preflight.json',
                               dict(status='passed', argv=hook['argv'], completed_at=view_queue.now()))

    def run(self):
        with (self.base / 'queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(not (self.base / 'queue.pid').exists(), 'successor queue was already launched')
            (self.base / 'queue.pid').write_text(str(os.getpid()) + '\n')
            signal.signal(signal.SIGTERM, self.interrupt)
            signal.signal(signal.SIGINT, self.interrupt)
            try:
                self.audit_contract()
                status = self.base / VARIANT / 'training.status'
                require(not status.exists(), 'crowd training already has a status')
                planned = view_queue.read_json(self.plan['runs'][VARIANT]['resolved_args'])
                require(not any(('_[' + planned['experiment_tag'] + ']_') in p.name
                                for p in (self.root / 'result').iterdir()),
                        'crowd experiment already has output')
                self.wait_dependency()
                self.preflight()
                self.run_training(VARIANT)
                self.status('complete', completed_runs=[VARIANT])
            except Exception as error:
                self.status('failed', error=repr(error))
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'check'))
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(argv)
    queue = CrowdQueue(args.plan)
    if args.command == 'check':
        report = queue.audit_contract()
        print(json.dumps(report, indent=2))
    else:
        queue.run()


if __name__ == '__main__':
    main()
