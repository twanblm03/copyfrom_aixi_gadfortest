#!/usr/bin/env python3
"""Insert frozen SAM encoder training after the live, audited final-view B0.

Only the predecessor controller is paused; its training child keeps running.
The predecessor is retired after its child exits and its complete run is audited.
The original H1 and mask-SAM contracts remain in their original snapshot.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.%d.tmp' % os.getpid())
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def process_info(identity):
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != identity['boot_id']:
        return None
    try:
        fields = Path('/proc/%s/stat' % identity['pid']).read_text().rsplit(')', 1)[1].split()
        if int(fields[19]) != identity['start_ticks'] or fields[0] == 'Z':
            return None
        return dict(state=fields[0], parent=int(fields[1]))
    except (OSError, ValueError, IndexError):
        return None


def completed_child_exit(identity):
    """Read the unreaped child's wait status while its parent is still paused."""
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != identity['boot_id']:
        raise RuntimeError('boot changed during B0 training')
    fields = Path('/proc/%s/stat' % identity['pid']).read_text().rsplit(')', 1)[1].split()
    if int(fields[19]) != identity['start_ticks'] or fields[0] != 'Z':
        raise RuntimeError('B0 exited without an identifiable child exit status')
    wait_status = int(fields[49])
    if wait_status != 0:
        raise RuntimeError('B0 failed: raw child wait status %d' % wait_status)
    return dict(raw_wait_status=wait_status, exit_code=0)


METHOD_FIELDS = {'backbone', 'freeze_backbone', 'experiment_tag', 'checkpoint_root'}


def validate_contract(reference, planned):
    missing = set(reference) - set(planned)
    changes = {key for key in reference if reference[key] != planned.get(key)}
    if missing or changes != METHOD_FIELDS:
        raise RuntimeError('unexpected B0/SAM common-argument drift: ' + str((missing, changes)))
    extras = set(planned) - set(reference)
    expected_extras = {'sam_repo', 'sam_checkpoint', 'sam_checkpoint_sha256',
                       'sam_encoder_chunk_size', 'sam_output_height', 'sam_output_width',
                       'heatmap_target_mode', 'heatmap_crowd_cap_ratio'}
    if extras != expected_extras:
        raise RuntimeError('unexpected additional SAM arguments: ' + str(extras))
    expected = dict(backbone='sam_vit_b', freeze_backbone=True, train_mask_mode='disabled',
                    heatmap_variant='off', split='view', val_mode=False, best_on_test=True,
                    keep_all_checkpoints=True, protocol='best_on_test', eval_split='test',
                    epochs=30, batch=16, test_batch=4, num_frame=5, load_model=False,
                    model_path='', device='0,1,2,3', parallel_transport='host',
                    sam_encoder_chunk_size=1, sam_output_height=23, sam_output_width=40)
    expected.update(heatmap_target_mode='box', heatmap_crowd_cap_ratio=.35)
    for key, value in expected.items():
        if planned.get(key) != value:
            raise RuntimeError('SAM protocol mismatch: ' + key)
    return dict(status='passed', common_fields_checked=len(reference),
                method_changes=sorted(changes), sam_fields=sorted(extras))


def verify_summary(summary):
    history = summary['history']
    if history.get('val'):
        raise RuntimeError('unexpected VAL evaluation')
    for split in ('train', 'test'):
        if [row['epoch'] for row in history[split]] != list(range(1, 31)):
            raise RuntimeError('incomplete 30-epoch ' + split + ' history')
        for row in history[split]:
            required = ('loss',) if split == 'train' else (
                'group_mAP_1.0', 'group_mAP_0.5', 'outlier_mIoU', 'loss')
            if any(key not in row or not isinstance(row[key], (int, float))
                   or not math.isfinite(row[key]) for key in required):
                raise RuntimeError('missing/nonfinite required ' + split + ' metric')
            if any(not math.isfinite(value) for value in row.values() if isinstance(value, (int, float))):
                raise RuntimeError('nonfinite training/evaluation metric')
            if split == 'train' and row.get('skipped_nonfinite_steps') != 0:
                raise RuntimeError('skipped or unreported optimizer steps')
    selected = max(history['test'], key=lambda row: row['group_mAP_1.0'])
    for metric in ('group_mAP_1.0', 'group_mAP_0.5', 'outlier_mIoU', 'loss'):
        if summary['best'][metric] != dict(epoch=selected['epoch'], value=selected[metric]):
            raise RuntimeError('reported metrics do not come from one selected TEST epoch')
    return selected


def load_predecessor(plan):
    path = Path(plan['predecessor_tool'])
    spec = importlib.util.spec_from_file_location('sam_queue_predecessor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Queue(plan['predecessor_plan'])


def predecessor_action(plan, action):
    queue = load_predecessor(plan)
    if action == 'audit-b0':
        queue.verify_frozen()
        queue.verify_training('b0', read_json(queue.plan['runs']['b0']['resolved_args']))
        (queue.base / 'b0/training.status').write_text('training_complete ' + now() + '\n')
        atomic_json(queue.base / 'successor_handoff.json', dict(
            stage='b0_complete_handed_off', successor_plan=plan['plan_path'], checked_at=now()))
        return
    lock = (queue.base / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, queue.interrupt)
    signal.signal(signal.SIGINT, queue.interrupt)
    try:
        queue.verify_frozen()
        queue.audit_contract()
        queue.run_training('h1_sup')  # Wait for and audit the already running H1.
        queue.prepare_cache()
        queue.run_training('sam')
        queue.status('complete', completed_runs=['b0', 'h1_sup', 'sam'],
                     sam_encoder_successor=plan['plan_path'])
    except Exception as error:
        queue.status('failed', error=repr(error))
        raise


class Queue:
    def __init__(self, plan_path):
        self.path = Path(plan_path).resolve()
        self.plan = read_json(self.path)
        self.base = Path(self.plan['queue_root'])
        self.root = Path(self.plan['workspace'])
        self.snapshot = Path(self.plan['snapshot_root'])
        self.python = self.plan['python']
        self.interval = self.plan.get('poll_seconds', 30)
        self.env = dict(os.environ, **self.plan['environment'], PYTHONPATH=str(self.snapshot))
        self.child = None
        self.predecessor_paused = False
        self.retired = False
        self.stage = None

    def status(self, stage, **details):
        state = dict(stage=stage, updated_at=now(), pid=os.getpid(), **details)
        atomic_json(self.base / 'queue_state.json', state)
        if stage != self.stage:
            print(json.dumps(state, ensure_ascii=False), flush=True)
            self.stage = stage

    def check(self):
        for entry in self.plan['frozen_files']:
            if digest(entry['path']) != entry['sha256']:
                raise RuntimeError('frozen source/asset changed: ' + entry['path'])
        args = read_json(self.plan['run']['resolved_args'])
        report = validate_contract(read_json(self.plan['reference_args']), args)
        atomic_json(self.base / 'config_audit.json', dict(report, checked_at=now()))
        return report

    def command(self, argv, log, env=None):
        path = self.base / log
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('ab', buffering=0) as stream:
            stream.write(('QUEUE ' + now() + ' ' + json.dumps(argv) + '\n').encode())
            self.child = subprocess.Popen(argv, cwd=self.root, env=env or self.env,
                stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True)
            atomic_json(path.with_suffix('.process.json'), dict(pid=self.child.pid,
                command=argv, started_at=now()))
            code = self.child.wait()
            self.child = None
        if code:
            raise RuntimeError('exit %d; inspect %s' % (code, path))

    def attach(self):
        manager, child = self.plan['predecessor_manager'], self.plan['predecessor_child']
        state, training = process_info(manager), process_info(child)
        old_state = read_json(self.plan['predecessor_state'])
        if (not state or not training or training['parent'] != manager['pid']
                or old_state['stage'] != 'training_b0' or old_state['pid'] != manager['pid']):
            raise RuntimeError('predecessor changed before handoff; refusing to signal')
        self.predecessor_paused = True
        os.kill(manager['pid'], signal.SIGSTOP)
        deadline = time.monotonic() + 10
        while True:
            state = process_info(manager)
            if state and state['state'] in ('T', 't'):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('predecessor did not pause')
            time.sleep(.1)
        training = process_info(child)
        old_state = read_json(self.plan['predecessor_state'])
        if (not training or training['parent'] != manager['pid']
                or old_state['stage'] != 'training_b0' or old_state['pid'] != manager['pid']):
            raise RuntimeError('B0 advanced during handoff; resume the original controller')
        atomic_json(self.base / 'handoff.json', dict(
            manager=manager, training_child=child, child_left_running=True, attached_at=now()))

    def wait_b0(self):
        while process_info(self.plan['predecessor_child']):
            state = process_info(self.plan['predecessor_manager'])
            if not state or state['state'] not in ('T', 't'):
                raise RuntimeError('paused predecessor controller changed')
            self.status('waiting_b0_completion', training_pid=self.plan['predecessor_child']['pid'])
            time.sleep(self.interval)
        state = process_info(self.plan['predecessor_manager'])
        if not state or state['state'] not in ('T', 't'):
            raise RuntimeError('predecessor was not paused at B0 exit')
        exit_result = completed_child_exit(self.plan['predecessor_child'])
        atomic_json(self.base / 'b0_exit.json', dict(exit_result, checked_at=now()))
        # Retire only after its exact training child has exited. SIGTERM of the
        # controller cannot then terminate the running B0 or the independent H1.
        manager = self.plan['predecessor_manager']
        if process_info(manager):
            os.kill(manager['pid'], signal.SIGTERM)
            os.kill(manager['pid'], signal.SIGCONT)
            deadline = time.monotonic() + 60
            while process_info(manager):
                if time.monotonic() >= deadline:
                    raise RuntimeError('predecessor failed to retire')
                time.sleep(1)
        self.retired = True
        self.predecessor_paused = False
        self.status('auditing_b0_completion')
        self.command([self.python, '-u', __file__, 'audit-b0', '--plan', str(self.path)],
                     'b0_completion.log', env=dict(self.env, CUDA_VISIBLE_DEVICES=''))

    def wait_memory(self):
        consecutive = 0
        while consecutive < 2:
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                '--format=csv,noheader,nounits'], text=True)
            rows = [line.split(',') for line in output.strip().splitlines()]
            if len(rows) != 4 or any(int(row[0]) != i or row[1].strip() != self.plan['gpu_uuids'][i]
                                     for i, row in enumerate(rows)):
                raise RuntimeError('physical GPU index/UUID mapping changed')
            free = [int(row[2]) for row in rows]
            consecutive = consecutive + 1 if min(free) >= self.plan['minimum_free_mib'] else 0
            self.status('waiting_gpu_memory', free_mib=free, consecutive_checks=consecutive)
            if consecutive < 2:
                time.sleep(self.interval)

    def verify_training(self):
        args = read_json(self.plan['run']['resolved_args'])
        results = [p for p in (self.root / 'result').iterdir()
                   if '_[' + args['experiment_tag'] + ']_' in p.name]
        if len(results) != 1:
            raise RuntimeError('expected exactly one SAM result')
        result = results[0]
        actual = read_json(result / 'args.json')
        if (set(actual) != set(args) | {'sam_encoder_metadata'}
                or {key: actual[key] for key in args} != args):
            raise RuntimeError('actual trainer arguments differ from frozen SAM contract')
        metadata = actual['sam_encoder_metadata']
        if (metadata.get('checkpoint_sha256') != args['sam_checkpoint_sha256']
                or metadata.get('family') != 'sam_v1_vit_b'
                or metadata.get('frozen') is not True):
            raise RuntimeError('actual SAM encoder metadata differs from the contract')
        summary = read_json(result / 'summary.json')
        if ({key: summary['args'].get(key) for key in actual} != actual
                or set(summary['args']) - set(actual) - {'peak_cuda_memory_mb'}):
            raise RuntimeError('summary arguments changed')
        selected = verify_summary(summary)
        checkpoint = Path(args['checkpoint_root']) / result.name
        for name in ['epoch%d.pth' % n for n in range(1, 31)] + ['best_test.pth']:
            if not (checkpoint / name).is_file():
                raise RuntimeError('missing checkpoint: ' + name)
        if (checkpoint / 'best_val.pth').exists():
            raise RuntimeError('unexpected VAL checkpoint')
        import torch
        best = torch.load(checkpoint / 'best_test.pth', map_location='cpu', weights_only=False)
        selection = best['selection_state']
        if (best['epoch'] != selected['epoch'] or best['eval_split'] != 'test'
                or selection['epoch'] != selected['epoch']
                or selection['value'] != selected['group_mAP_1.0']
                or selection.get('selection_label') != 'best-on-TEST'
                or best.get('sam_encoder_metadata', {}).get('checkpoint_sha256') != args['sam_checkpoint_sha256']):
            raise RuntimeError('selected checkpoint provenance mismatch')
        del best
        expected = None
        for epoch in range(1, 31):
            clips = set()
            with (self.root / args['result_path'] / result.name / ('pred_group_epoch_%d.txt' % epoch)).open() as stream:
                for line in stream:
                    if line.strip():
                        fields = line.split()
                        clips.add((int(fields[0]), int(fields[1])))
            if (len(clips) != 2414 or {v for v, _ in clips} != {4, 8, 12, 16, 20, 24}
                    or (expected is not None and clips != expected)):
                raise RuntimeError('incomplete or changed TEST clip inventory')
            expected = clips
        atomic_json(self.base / 'sam_encoder/completion_audit.json', dict(status='passed',
            result_dir=str(result), checkpoint_dir=str(checkpoint), selected_epoch=selected['epoch'],
            selected_metrics=selected, epoch30_metrics=summary['history']['test'][-1],
            train_epochs=30, test_epochs=30, test_clips=2414, completed_at=now()))

    def restore_predecessor(self):
        if self.predecessor_paused and not self.retired and process_info(self.plan['predecessor_manager']):
            os.kill(self.plan['predecessor_manager']['pid'], signal.SIGCONT)
            self.predecessor_paused = False
            atomic_json(self.base / 'handoff_rollback.json', dict(
                action='resumed_original_controller', checked_at=now()))

    def interrupt(self, signum, _frame):
        if self.child and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGTERM)
        self.restore_predecessor()
        self.status('interrupted', signal=signum)
        raise SystemExit(128 + signum)

    def run(self):
        lock = (self.base / 'queue.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (self.base / 'queue.process.json').exists():
            raise RuntimeError('queue already launched; explicit recovery required')
        stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
        atomic_json(self.base / 'queue.process.json', dict(pid=os.getpid(), start_ticks=int(stat[19]),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), plan=str(self.path), started_at=now()))
        signal.signal(signal.SIGTERM, self.interrupt)
        signal.signal(signal.SIGINT, self.interrupt)
        try:
            self.check()
            self.attach()
            self.wait_b0()
            self.check()
            self.wait_memory()
            self.status('sam_encoder_dp4_smoke')
            self.command([self.python, '-u', str(self.snapshot / 'tools/smoke_sam_encoder_dp4.py'),
                '--args-json', self.plan['run']['resolved_args'], '--output', str(self.base / 'dp4_smoke.json')],
                'dp4_smoke.log')
            if not read_json(self.base / 'dp4_smoke.json').get('passed'):
                raise RuntimeError('SAM DP4 smoke did not pass')
            self.check()
            self.wait_memory()
            status = self.base / 'sam_encoder/training.status'
            status.parent.mkdir(parents=True, exist_ok=True)
            args = read_json(self.plan['run']['resolved_args'])
            if any('_[' + args['experiment_tag'] + ']_' in p.name for p in (self.root / 'result').iterdir()):
                raise RuntimeError('SAM experiment tag already exists')
            self.status('training_sam_encoder')
            status.write_text('training_started ' + now() + '\n')
            try:
                self.command([self.python, '-u', str(self.snapshot / 'train.py')] + read_json(self.plan['run']['argv']),
                             'sam_encoder/training.log')
                self.verify_training()
            except BaseException:
                status.write_text('training_failed ' + now() + '\n')
                raise
            status.write_text('training_complete ' + now() + '\n')
            self.status('continuing_original_mask_sam_queue')
            self.command([self.python, '-u', __file__, 'continue-mask', '--plan', str(self.path)], 'mask_queue.log')
            self.status('complete', completed_runs=['sam_encoder', 'sam_mask_roi_blend'])
        except Exception as error:
            self.restore_predecessor()
            self.status('failed', error=repr(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'run', 'audit-b0', 'continue-mask'])
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    queue = Queue(args.plan)
    if args.command == 'check':
        print(json.dumps(queue.check(), indent=2))
    elif args.command == 'run':
        queue.run()
    else:
        predecessor_action(queue.plan, args.command)


if __name__ == '__main__':
    main()
