#!/usr/bin/env python3
"""Frozen view TRAIN+VAL -> TEST queue with explicitly selected best-on-TEST."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.queue_view_dp4 import Queue as FrozenQueue, atomic_json, now, process_matches, read_json

METRICS = ('group_mAP_1.0', 'group_mAP_0.5', 'outlier_mIoU', 'loss')
TEST_VIDEOS = {4, 8, 12, 16, 20, 24}


def process_state(pid, ticks):
    try:
        fields = Path('/proc/%s/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return fields[0] if int(fields[19]) == int(ticks) and fields[0] != 'Z' else None
    except (OSError, ValueError, IndexError):
        return None


def active_preparation(manifest, command):
    """Only the exact final-cache CPU command is a preparation dependency."""
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            tokens = path.read_bytes().split(b'\0')
        except OSError:
            continue
        if (command.encode() in tokens and str(manifest).encode() in tokens
                and any(token.endswith(b'/prepare_view_final_sam_cache.py') for token in tokens)):
            return True
    return False


def selected_test_row(summary):
    history = summary['history']
    if history.get('val'):
        raise RuntimeError('final protocol unexpectedly evaluated VAL')
    for split in ('train', 'test'):
        if [row['epoch'] for row in history[split]] != list(range(1, 31)):
            raise RuntimeError('final protocol requires all 30 TRAIN/TEST epochs')
        for row in history[split]:
            required = ('loss',) if split == 'train' else METRICS
            if (any(not math.isfinite(row[key]) for key in required)
                    or any(not math.isfinite(value) for value in row.values()
                           if isinstance(value, (int, float)))):
                raise RuntimeError('nonfinite final training/TEST metrics')
            if split == 'train' and row.get('skipped_nonfinite_steps') != 0:
                raise RuntimeError('final training skipped a nonfinite optimizer step')
    # max is stable: the first epoch wins exact ties, matching strict > selection.
    selected = max(history['test'], key=lambda row: row['group_mAP_1.0'])
    for metric in METRICS:
        best = summary['best'][metric]
        if best['epoch'] != selected['epoch'] or best['value'] != selected[metric]:
            raise RuntimeError('best-on-TEST metrics must all come from the selected epoch')
    return selected, history['test'][-1]


class Queue(FrozenQueue):
    def audit_contract(self):
        self.command([self.python, str(self.snapshot / 'tools/validate_view_final_contract.py'),
                      '--root', str(self.snapshot), '--plan', self.plan['contract_plan'],
                      '--output', str(self.base / 'launch_config_audit.json')],
                     self.base / 'contract.log', env=dict(self.env, CUDA_VISIBLE_DEVICES=''))

    def wait_dependency(self):
        handoff = read_json(self.plan['handoff_file'])
        child = handoff['current_training']['pid']
        child_ticks = handoff['current_training_start_ticks']
        while process_matches(child, child_ticks):
            self.status('waiting_development_b0_exit', training_pid=child)
            time.sleep(self.interval)
        pid, ticks = handoff['queue_pid'], handoff['queue_start_ticks']
        if pid == os.getpid() or handoff['current_variant'] != 'b0':
            raise RuntimeError('invalid predecessor queue handoff')
        state = process_state(pid, ticks)
        if state is not None:
            if state not in ('T', 't'):
                raise RuntimeError('predecessor manager is no longer paused; inspect before handoff')
            self.status('retiring_paused_development_queue', predecessor_pid=pid)
            if process_matches(pid, ticks):
                os.kill(pid, signal.SIGTERM)
            if process_matches(pid, ticks):
                os.kill(pid, signal.SIGCONT)
            deadline = time.monotonic() + 60
            while process_matches(pid, ticks):
                if time.monotonic() >= deadline:
                    raise RuntimeError('predecessor manager did not exit after TERM/CONT')
                time.sleep(min(self.interval, 2))
        # Its TERM handler may mark the just-completed B0 failed. Re-audit after
        # the manager has exited, then restore complete only with full evidence.
        predecessor = FrozenQueue(self.plan['predecessor_plan'])
        report = dict(predecessor_pid=pid, training_pid=child, checked_at=now())
        try:
            args = read_json(predecessor.plan['runs']['b0']['resolved_args'])
            predecessor.verify_training('b0', args)
            (predecessor.base / 'b0/training.status').write_text('training_complete ' + now() + '\n')
            report.update(status='complete', development_b0_audit=str(predecessor.base / 'b0/completion_audit.json'))
        except Exception as error:
            report.update(status='development_b0_incomplete', error=repr(error),
                          action='preserved artifacts; final B0 starts independently from scratch')
        atomic_json(self.base / 'predecessor_completion_audit.json', report)

    def verify_training(self, variant, planned):
        outputs = [path for path in (self.root / 'result').iterdir()
                   if ('_[' + planned['experiment_tag'] + ']_') in path.name]
        if len(outputs) != 1:
            raise RuntimeError('expected one final result directory: ' + variant)
        result = outputs[0]
        actual = read_json(result / 'args.json')
        if actual != planned or len(actual) != 117:
            raise RuntimeError('actual final trainer arguments differ from all 117 frozen fields')
        if (actual['val_mode'] or not actual['best_on_test'] or actual['eval_split'] != 'test'
                or actual['load_model'] or actual['model_path']):
            raise RuntimeError('final run must start from scratch and select best-on-TEST')
        summary = read_json(result / 'summary.json')
        summary_args = summary['args']
        if (not set(planned).issubset(summary_args)
                or {key: summary_args[key] for key in planned} != planned
                or set(summary_args) - set(planned) - {'peak_cuda_memory_mb'}):
            raise RuntimeError('final summary arguments differ from launch contract')
        selected, final = selected_test_row(summary)
        checkpoint = Path(planned['checkpoint_root']) / result.name
        for name in ['epoch%d.pth' % epoch for epoch in range(1, 31)] + ['best_test.pth']:
            if not (checkpoint / name).is_file():
                raise RuntimeError('missing final checkpoint: ' + str(checkpoint / name))
        if (checkpoint / 'best_val.pth').exists():
            raise RuntimeError('final run unexpectedly saved a VAL checkpoint')
        import torch
        state = torch.load(checkpoint / 'best_test.pth', map_location='cpu', weights_only=False)
        if (state['epoch'] != selected['epoch'] or state['eval_split'] != 'test'
                or state['selection_state']['epoch'] != selected['epoch']
                or state['selection_state']['value'] != selected['group_mAP_1.0']
                or state['selection_state'].get('split') != 'test'
                or state['selection_state'].get('selection_label') != 'best-on-TEST'):
            raise RuntimeError('best_test checkpoint differs from first argmax TEST epoch')
        del state
        prediction_clips = None
        expected_count = self.plan.get('expected_inventory', {}).get('test_clips', 2414)
        prediction_root = self.root / planned['result_path'] / result.name
        for epoch in range(1, 31):
            clips = set()
            with (prediction_root / ('pred_group_epoch_%d.txt' % epoch)).open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    tokens = line.split()
                    video, clip = int(tokens[0]), int(tokens[1])
                    if video not in TEST_VIDEOS:
                        raise RuntimeError('prediction contains a non-TEST camera view')
                    clips.add((video, clip))
            if ({video for video, _ in clips} != TEST_VIDEOS
                    or (expected_count is not None and len(clips) != expected_count)
                    or (prediction_clips is not None and clips != prediction_clips)):
                raise RuntimeError('TEST clip inventory changed or is incomplete')
            prediction_clips = clips
        atomic_json(self.base / variant / 'completion_audit.json', dict(status='passed',
            result_dir=str(result), checkpoint_dir=str(checkpoint), args_fields_checked=117,
            train_epochs=30, val_epochs=0, test_epochs=30, test_clips=len(prediction_clips),
            selection_label='best-on-TEST', selection_metric='group_mAP_1.0',
            selected_epoch=selected['epoch'], selected_metrics=selected, epoch30_metrics=final,
            best_checkpoint=str(checkpoint / 'best_test.pth'), completed_at=now()))

    def cache_args(self):
        cache = self.plan['cache']
        args = ['--manifest', cache['manifest'], '--cache-root', cache['cache_root'],
                '--cache-version', cache['cache_version'], '--sam-repo', cache['sam_repo'],
                '--checkpoint', cache['checkpoint']]
        sources = cache.get('sources') or [dict(manifest=manifest, cache_root=root)
            for manifest, root in zip(cache['source_manifests'], cache['source_cache_roots'])]
        if len(sources) != 2:
            raise RuntimeError('final cache requires both approved place and view sources')
        for source in sources:
            args += ['--source-manifest', source['manifest'], '--source-cache-root', source['cache_root']]
        return args

    def ensure_manifest(self):
        from tools.prepare_view_final_sam_cache import inspect_manifest
        cache = self.plan['cache']
        manifest = Path(cache['manifest'])
        while active_preparation(manifest, 'manifest'):
            self.status('waiting_cpu_final_manifest')
            time.sleep(self.interval)
        if not all(Path(str(manifest) + suffix).is_file() for suffix in ('', '.summary.json', '.clips.json')):
            self.status('building_final_manifest')
            self.command([self.python, '-u', str(self.snapshot / 'tools/prepare_view_final_sam_cache.py'),
                          'manifest'] + self.cache_args() + ['--data-root', cache['data_root'],
                          '--tracks-path', cache['tracks_path'], '--report',
                          str(self.base / 'sam_cache/manifest.json')], self.base / 'sam_cache/manifest.log',
                         env=dict(self.env, CUDA_VISIBLE_DEVICES=''))
        inventory = inspect_manifest(manifest)
        expected = self.plan.get('expected_inventory', {})
        if inventory['clips']['train'] != 7298 or (
                expected.get('test_clips') is not None and inventory['clips']['test'] != expected['test_clips']):
            raise RuntimeError('final SAM manifest clip inventory differs from approved dataset')
        return inventory

    def prepare_cache(self):
        self.verify_frozen()
        self.ensure_manifest()
        cache = self.plan['cache']
        while active_preparation(cache['manifest'], 'reuse'):
            self.status('waiting_cpu_final_cache_reuse')
            time.sleep(self.interval)
        tool = str(self.snapshot / 'tools/prepare_view_final_sam_cache.py')
        cpu = dict(self.env, CUDA_VISIBLE_DEVICES='')
        self.status('reusing_final_sam_cache')
        report = self.base / 'sam_cache/reuse.json'
        self.command([self.python, '-u', tool, 'reuse'] + self.cache_args() +
                     ['--report', str(report)], self.base / 'sam_cache/reuse.log', env=cpu)
        missing = read_json(report)['coverage'].get('missing', 0)
        if missing:
            self.wait_memory('final_sam_cache_generation')
            self.status('generating_final_sam_cache', missing_frames=missing)
            env = dict(self.env, PYTHON_BIN=self.python, SAM_MANIFEST=cache['manifest'],
                SAM_CACHE_ROOT=cache['cache_root'], SAM_REPO=cache['sam_repo'], SAM_CHECKPOINT=cache['checkpoint'],
                SAM_LOG_DIR=str(self.base / 'sam_cache/generation'), GPUS='0,1,2,3',
                VERIFY='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            self.command(['bash', str(self.snapshot / 'scripts/generate_sam_person_full.sh')],
                         self.base / 'sam_cache/generation.log', env=env)
        self.status('verifying_complete_final_sam_cache')
        for command in ('audit', 'verify'):
            report = self.base / ('sam_cache/%s.json' % command)
            self.command([self.python, '-u', tool, command] + self.cache_args() +
                         ['--verify-workers', '4', '--report', str(report)],
                         self.base / ('sam_cache/%s.log' % command), env=cpu)
            verified = read_json(report)
            if verified['status'] != 'complete' or (command == 'verify' and verified.get('full_verification') != 'passed'):
                raise RuntimeError('complete final cache did not pass ' + command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'check'])
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    queue = Queue(args.plan)
    if args.command == 'check':
        queue.verify_frozen()
        queue.audit_contract()
        print('Frozen final files and best-on-TEST contract: passed', flush=True)
    else:
        queue.run()


if __name__ == '__main__':
    main()
