#!/usr/bin/env python3
"""Run the already approved view H1-sup job beside resumed B0."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def unavailable_dataset(plan):
    args = read(plan['runs']['h1_sup']['resolved_args'])
    paths = [Path(args['data_path']) / 'cafe', Path(args['groundtruth'])]
    missing = []
    for path in paths:
        try:
            if path.is_dir():
                next(path.iterdir())
            else:
                with path.open('rb') as stream:
                    if not stream.read(1):
                        raise OSError('dataset file is empty')
        except (OSError, StopIteration):
            missing.append(str(path))
    return missing


def local_audit(plan):
    pending = []
    for entry in plan['frozen_files']:
        path = Path(entry['path'])
        try:
            actual = digest(path)
        except OSError:
            if path.parts[:2] == ('/', 'nas'):
                pending.append(str(path))
                continue
            raise
        if actual != entry['sha256']:
            raise RuntimeError('frozen input changed: ' + str(path))
    original = read(plan['parallel_h1']['original_plan'])
    if plan['runs'] != original['runs']:
        raise RuntimeError('parallel execution changed frozen trainer configurations')
    return dict(status='local_passed_nas_pending' if pending else 'passed',
                pending_nas_files=pending, unchanged_h1_args=True,
                trainer_fields_checked=117, execution_order=['h1_sup'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'run'])
    parser.add_argument('--plan', required=True)
    cli = parser.parse_args()
    plan = read(cli.plan)
    sys.path.insert(0, plan['snapshot_root'])
    from tools.queue_view_final import Queue as FinalQueue
    from tools.queue_view_dp4 import atomic_json

    class ParallelH1Queue(FinalQueue):
        def run(self):
            lock = (self.base / 'queue.lock').open('a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (self.base / 'queue.pid').exists():
                raise RuntimeError('parallel H1 queue was already launched')
            (self.base / 'queue.pid').write_text(str(os.getpid()) + '\n')
            fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
            atomic_json(self.base / 'queue.process.json', dict(
                pid=os.getpid(), start_ticks=int(fields[19]),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                command=sys.argv))
            signal.signal(signal.SIGTERM, self.interrupt)
            signal.signal(signal.SIGINT, self.interrupt)
            try:
                atomic_json(self.base / 'local_input_audit.json', local_audit(self.plan))
                while True:
                    missing = unavailable_dataset(self.plan)
                    if not missing:
                        break
                    self.status('waiting_nas', missing_paths=missing,
                                next_stage='h1_sup', message='Restore /nas; training starts automatically')
                    time.sleep(self.interval)
                self.verify_frozen()
                self.audit_contract()
                self.run_training('h1_sup')
                self.status('complete', completed_runs=['h1_sup'])
            except Exception as error:
                self.status('failed', error=repr(error))
                raise

    queue = ParallelH1Queue(cli.plan)
    if cli.command == 'check':
        report = local_audit(plan)
        if not report['pending_nas_files'] and not unavailable_dataset(plan):
            queue.audit_contract()
            report['full_contract'] = 'passed'
        else:
            report['full_contract'] = 'pending_nas_restore'
        atomic_json(queue.base / 'local_input_audit.json', report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    else:
        queue.run()


if __name__ == '__main__':
    main()
