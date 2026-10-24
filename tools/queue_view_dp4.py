#!/usr/bin/env python3
"""Durable, fail-closed queue for the three matched view DP4 experiments."""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

VARIANTS = ('b0', 'h1_sup', 'sam')


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.%d.tmp' % os.getpid())
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(str(temporary), str(path))


def process_matches(pid, start_ticks):
    try:
        # comm may contain spaces/parentheses; fields after its last ')' start at 3.
        fields = Path('/proc/%s/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and int(fields[19]) == int(start_ticks)
    except (OSError, ValueError, IndexError):
        return False


def dependency_state(status_text, alive):
    status = status_text.split()[0] if status_text.strip() else ''
    if status in ('training_failed', 'training_interrupted'):
        raise RuntimeError('place SAM did not finish normally: ' + status_text.strip())
    if status == 'training_complete':
        return 'waiting_dependency_exit' if alive else 'ready'
    if not alive:
        raise RuntimeError('place SAM exited without training_complete; inspect its log')
    return 'waiting_place_sam'


def parse_gpus(output, expected_uuids, minimum):
    rows = [line.split(',') for line in output.strip().splitlines() if line.strip()]
    if len(rows) != len(expected_uuids):
        raise RuntimeError('expected exactly four physical GPUs')
    values = []
    for index, (row, expected) in enumerate(zip(rows, expected_uuids)):
        if len(row) != 3 or int(row[0].strip()) != index or row[1].strip() != expected:
            raise RuntimeError('GPU index/UUID changed; refusing to change DP4 device mapping')
        values.append(int(row[2].strip()))
    return all(value >= minimum for value in values), values


class Queue:
    def __init__(self, plan_path):
        self.plan_path = Path(plan_path).resolve()
        self.plan = read_json(self.plan_path)
        self.root = Path(self.plan['workspace'])
        self.base = Path(self.plan['queue_root'])
        self.snapshot = Path(self.plan['snapshot_root'])
        self.python = self.plan['python']
        self.interval = self.plan.get('poll_seconds', 30)
        self.env = os.environ.copy()
        self.env.update(self.plan['environment'])
        self.env['PYTHONPATH'] = str(self.snapshot)
        self.child = None
        self.current = None
        self.base.mkdir(parents=True, exist_ok=True)

    def status(self, stage, **details):
        state = dict(stage=stage, updated_at=now(), pid=os.getpid(), **details)
        atomic_json(self.base / 'queue_state.json', state)
        (self.base / 'queue.status').write_text(stage + ' ' + state['updated_at'] + '\n')
        # The JSON heartbeat carries changing memory figures without flooding logs.
        if stage != self.current:
            print(json.dumps(state, ensure_ascii=False), flush=True)
            self.current = stage

    def verify_frozen(self):
        for entry in self.plan['frozen_files']:
            if digest(entry['path']) != entry['sha256']:
                raise RuntimeError('frozen input changed: ' + entry['path'])

    def command(self, argv, log, env=None, cwd=None):
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('ab', buffering=0) as stream:
            stream.write(('\nQUEUE command at ' + now() + ': ' + json.dumps(argv) + '\n').encode())
            self.child = subprocess.Popen(argv, cwd=str(cwd or self.root),
                                          env=env or self.env, stdout=stream,
                                          stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL, start_new_session=True)
            atomic_json(log.with_suffix('.process.json'),
                        dict(pid=self.child.pid, command=argv, started_at=now()))
            code = self.child.wait()
            self.child = None
        if code:
            raise RuntimeError('command failed (exit %d); inspect %s' % (code, log))

    def interrupt(self, signum, _frame):
        if self.child is not None and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGTERM)
        self.status('interrupted', signal=signum)
        raise SystemExit(128 + signum)

    def wait_dependency(self):
        dependency = self.plan['dependency']
        while True:
            status_path = Path(dependency['status'])
            text = status_path.read_text() if status_path.exists() else ''
            alive = any(process_matches(p['pid'], p['start_ticks']) for p in dependency['processes'])
            stage = dependency_state(text, alive)
            if stage == 'ready':
                return
            self.status(stage, dependency_status=text.strip())
            time.sleep(self.interval)

    def wait_memory(self, next_stage):
        consecutive = 0
        while consecutive < 2:
            try:
                output = subprocess.check_output([
                    'nvidia-smi', '--query-gpu=index,uuid,memory.free',
                    '--format=csv,noheader,nounits'], text=True)
            except subprocess.CalledProcessError:
                self.status('waiting_gpu_query', next_stage=next_stage)
                time.sleep(self.interval)
                continue
            enough, values = parse_gpus(output, self.plan['gpu_uuids'],
                                       self.plan['minimum_free_mib'])
            consecutive = consecutive + 1 if enough else 0
            self.status('waiting_gpu_memory', next_stage=next_stage,
                        free_mib=values, minimum_free_mib=self.plan['minimum_free_mib'],
                        consecutive_checks=consecutive)
            if consecutive < 2:
                time.sleep(self.interval)

    def audit_contract(self):
        self.command([self.python, str(self.snapshot / 'tools/validate_view_training_contract.py'),
                      '--root', str(self.snapshot), '--plan', self.plan['contract_plan'],
                      '--output', str(self.base / 'launch_config_audit.json')],
                     self.base / 'contract.log')

    def run_training(self, variant):
        self.verify_frozen()
        entry = self.plan['runs'][variant]
        run_root = self.base / variant
        run_root.mkdir(parents=True, exist_ok=True)
        status = run_root / 'training.status'
        if status.exists():
            raise RuntimeError('refusing to overwrite existing run: ' + str(status))
        planned = read_json(entry['resolved_args'])
        existing = [p for p in (self.root / 'result').iterdir()
                    if ('_[' + planned['experiment_tag'] + ']_') in p.name]
        if existing:
            raise RuntimeError('experiment tag already has output: ' + planned['experiment_tag'])
        self.wait_memory(variant)
        self.status('training_' + variant)
        status.write_text('training_started ' + now() + '\n')
        try:
            self.command([self.python, '-u', str(self.snapshot / 'train.py')] + read_json(entry['argv']),
                         run_root / 'training.log')
            self.verify_training(variant, planned)
        except BaseException:
            status.write_text('training_failed ' + now() + '\n')
            raise
        status.write_text('training_complete ' + now() + '\n')

    def verify_training(self, variant, planned):
        outputs = [p for p in (self.root / 'result').iterdir()
                   if ('_[' + planned['experiment_tag'] + ']_') in p.name]
        if len(outputs) != 1:
            raise RuntimeError('expected exactly one result directory for ' + variant)
        result = outputs[0]
        actual = read_json(result / 'args.json')
        if actual != planned:
            differences = {key: [planned.get(key), actual.get(key)] for key in set(planned) | set(actual)
                           if planned.get(key) != actual.get(key)}
            raise RuntimeError('actual trainer arguments differ: ' + json.dumps(differences))
        summary = read_json(result / 'summary.json')
        for split in ('train', 'val'):
            if [item['epoch'] for item in summary['history'][split]] != list(range(1, 31)):
                raise RuntimeError('incomplete 30-epoch TRAIN/VAL history: ' + variant)
            for item in summary['history'][split]:
                keys = ('loss',) if split == 'train' else ('loss', 'group_mAP_1.0', 'group_mAP_0.5', 'outlier_mIoU')
                if any(not math.isfinite(item[key]) for key in keys):
                    raise RuntimeError('nonfinite training/validation metrics: ' + variant)
                if split == 'train' and item.get('skipped_nonfinite_steps') != 0:
                    raise RuntimeError('training skipped a nonfinite optimizer step: ' + variant)
        if summary['history'].get('test'):
            raise RuntimeError('unexpected TEST evaluation')
        checkpoint = Path(planned['checkpoint_root']) / result.name
        for name in ('best_val.pth', 'epoch30.pth'):
            if not (checkpoint / name).is_file():
                raise RuntimeError('missing checkpoint: ' + str(checkpoint / name))
        atomic_json(self.base / variant / 'completion_audit.json',
                    dict(status='passed', result_dir=str(result), checkpoint_dir=str(checkpoint),
                         args_fields_checked=len(actual), train_epochs=30, val_epochs=30,
                         test_epochs=0, best=summary['best'], completed_at=now()))

    def cache_args(self):
        c = self.plan['cache']
        return ['--manifest', c['manifest'], '--source-manifest', c['source_manifest'],
                '--source-cache-root', c['source_cache_root'], '--cache-root', c['cache_root'],
                '--cache-version', c['cache_version'], '--sam-repo', c['sam_repo'],
                '--checkpoint', c['checkpoint']]

    def ensure_manifest(self):
        c = self.plan['cache']
        manifest = Path(c['manifest'])
        while not all(Path(str(manifest) + suffix).is_file()
                      for suffix in ('', '.summary.json', '.clips.json')):
            # A CPU builder was already authorized/started during queue setup.
            active = False
            for path in Path('/proc').glob('[0-9]*/cmdline'):
                try:
                    command = path.read_bytes().split(b'\0')
                    active = (b'manifest' in command and b'--output' in command and
                              any(b'sam_person_full.py' in token for token in command) and
                              any(b'view_randomsegments_manifest.jsonl' in token for token in command))
                except OSError:
                    continue
                if active:
                    break
            if active:
                self.status('waiting_cpu_view_manifest')
                time.sleep(self.interval)
                continue
            if manifest.exists():
                raise RuntimeError('manifest companions missing; inspect manifest log')
            self.status('building_view_manifest')
            env = dict(self.env, CUDA_VISIBLE_DEVICES='')
            self.command([self.python, '-u', str(self.snapshot / 'tools/sam_person_full.py'),
                          'manifest', '--split', 'view', '--train-all-frames',
                          '--data-root', c['data_root'], '--tracks-path', c['tracks_path'],
                          '--output', str(manifest)], self.base / 'sam_cache/manifest.log', env=env)
        summary = read_json(str(manifest) + '.summary.json')
        if (summary['split'] != 'view' or summary['clips'] != {'train': 4857, 'val': 2441}
                or summary['frames'] != 174532 or summary['manifest_sha256'] != digest(manifest)):
            raise RuntimeError('unexpected view manifest inventory/hash')
        return summary

    def prepare_cache(self):
        self.verify_frozen()
        self.ensure_manifest()
        # CPU reuse is also allowed to finish before the queued GPU stages.
        # Queue setup records its report; otherwise this command is idempotent.
        report = Path(self.plan['cache']['reuse_report'])
        while True:
            alive = False
            for path in Path('/proc').glob('[0-9]*/cmdline'):
                try:
                    tokens = path.read_bytes().split(b'\0')
                    alive = (b'reuse' in tokens and any(b'prepare_view_sam_cache.py' in t for t in tokens)
                             and (str(report).encode() in tokens or
                                  self.plan['cache']['manifest'].encode() in tokens or
                                  b'.local_data/sam/view_cache_reuse.json' in tokens))
                except OSError:
                    continue
                if alive:
                    break
            if not alive:
                break
            self.status('waiting_cpu_cache_reuse')
            time.sleep(self.interval)
        if not report.exists() or read_json(report).get('status') not in ('complete', 'reuse_complete_missing_generation'):
            self.status('reusing_view_cache')
            self.command([self.python, '-u', str(self.snapshot / 'tools/prepare_view_sam_cache.py'),
                          'reuse'] + self.cache_args() + ['--report', str(report)],
                         self.base / 'sam_cache/reuse.log', env=dict(self.env, CUDA_VISIBLE_DEVICES=''))
        coverage = read_json(report)['coverage']
        if coverage.get('missing', 0):
            self.wait_memory('sam_cache_generation')
            self.status('generating_view_sam_cache', missing_frames=coverage['missing'])
            c = self.plan['cache']
            env = dict(self.env, PYTHON_BIN=self.python, SAM_MANIFEST=c['manifest'],
                       SAM_CACHE_ROOT=c['cache_root'], SAM_REPO=c['sam_repo'], SAM_CHECKPOINT=c['checkpoint'],
                       SAM_LOG_DIR=str(self.base / 'sam_cache/generation'), GPUS='0,1,2,3',
                       VERIFY='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            self.command(['bash', str(self.snapshot / 'scripts/generate_sam_person_full.sh')],
                         self.base / 'sam_cache/generation.log', env=env)
        self.status('verifying_complete_view_cache')
        self.command([self.python, '-u', str(self.snapshot / 'tools/prepare_view_sam_cache.py'),
                      'verify'] + self.cache_args() + ['--verify-workers', '4', '--report',
                                                       str(self.base / 'sam_cache/verification.json')],
                     self.base / 'sam_cache/verification.log', env=dict(self.env, CUDA_VISIBLE_DEVICES=''))
        verified = read_json(self.base / 'sam_cache/verification.json')
        if verified['status'] != 'complete' or verified.get('full_verification') != 'passed':
            raise RuntimeError('full SAM cache verification did not pass')

    def run(self):
        lock = (self.base / 'queue.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (self.base / 'queue.pid').exists():
            raise RuntimeError('queue was already launched; refusing duplicate training')
        (self.base / 'queue.pid').write_text(str(os.getpid()) + '\n')
        signal.signal(signal.SIGTERM, self.interrupt)
        signal.signal(signal.SIGINT, self.interrupt)
        try:
            self.verify_frozen()
            self.audit_contract()
            self.wait_dependency()
            self.run_training('b0')
            self.run_training('h1_sup')
            self.prepare_cache()
            self.run_training('sam')
            self.status('complete', completed_runs=list(VARIANTS))
        except Exception as error:
            self.status('failed', error=repr(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'check'])
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    queue = Queue(args.plan)
    if args.command == 'check':
        queue.verify_frozen()
        queue.audit_contract()
        print('Frozen files and configuration contract: passed', flush=True)
    else:
        queue.run()


if __name__ == '__main__':
    main()
