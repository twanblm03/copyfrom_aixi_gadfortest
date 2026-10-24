#!/usr/bin/env python3
"""Restore place SAM to epoch 30 before releasing the view continuation."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'run'])
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    plan = read(args.plan)
    snapshot = Path(plan['snapshot_root'])
    spec = importlib.util.spec_from_file_location('frozen_queue', str(snapshot / 'tools/queue_view_dp4.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    details = plan['sam_recovery']

    class SamRecovery(module.Queue):
        def status(self, stage, **kwargs):
            super().status(stage, **kwargs)
            full = Path(details['public_run_root'])
            if stage == 'complete': token = 'training_complete'
            elif stage == 'failed': token = 'training_failed'
            elif stage == 'interrupted': token = 'training_interrupted'
            elif stage == 'training_place_sam': token = 'training_started'
            else: token = 'training_waiting'
            text = token + ' ' + module.now() + ' recovery=' + str(self.base) + '\n'
            (full / 'training.status').write_text(text)
            (self.base / 'dependency.status').write_text(text)
            module.atomic_json(full / 'recovery_state.json', read(self.base / 'queue_state.json'))
            pipeline = full.parent / 'queue.status'
            pipeline.write_text(('pipeline_complete' if stage == 'complete' else 'recovery_' + stage)
                                + ' ' + module.now() + '\n')

        def wait_nas(self):
            code = 'from PIL import Image\nimport sys\nfor p in sys.argv[1:]:\n with Image.open(p) as im: im.load()\n'
            while True:
                self.status('checking_nas')
                try:
                    result = subprocess.run([self.python, '-c', code] + details['nas_probe_images'],
                                            env=dict(self.env, CUDA_VISIBLE_DEVICES=''),
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            timeout=15, text=True)
                    if result.returncode == 0:
                        return
                    error = result.stderr[-250:]
                except subprocess.TimeoutExpired:
                    error = 'NAS decode probe timed out'
                self.status('waiting_nas', error=error)
                time.sleep(self.interval)

        def verify_training(self, variant, planned):
            outputs = [p for p in (self.root / 'result').iterdir()
                       if ('_[' + planned['experiment_tag'] + ']_') in p.name]
            if len(outputs) != 1:
                raise RuntimeError('expected one SAM recovery output')
            result = outputs[0]
            target = Path(planned['checkpoint_root']) / result.name / 'best_val.pth'
            if not target.exists():
                selected = read(result / 'summary.json')['best'][planned['selection_metric']]
                previous = details['checkpoint_audit']
                if selected != dict(epoch=previous['best_epoch'], value=previous['best_value']):
                    raise RuntimeError('best SAM checkpoint is missing')
                shutil.copy2(details['historical_best'], str(target))
                module.atomic_json(self.base / 'place_sam/historical_best_retained.json',
                                   dict(source=details['historical_best'], destination=str(target), selected=selected))
            super().verify_training(variant, planned)

        def run(self):
            lock = (self.base / 'queue.lock').open('a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (self.base / 'queue.pid').exists():
                raise RuntimeError('SAM recovery was already launched')
            (self.base / 'queue.pid').write_text(str(os.getpid()) + '\n')
            (Path(details['public_run_root']) / 'training.pid').write_text(str(os.getpid()) + '\n')
            signal.signal(signal.SIGTERM, self.interrupt)
            signal.signal(signal.SIGINT, self.interrupt)
            try:
                self.verify_frozen()
                self.audit_contract()
                self.wait_nas()
                self.run_training('place_sam')
                self.status('complete', resumed_epoch=27, completed_epoch=30)
            except Exception as error:
                self.status('failed', error=repr(error))
                raise

    queue = SamRecovery(args.plan)
    queue.verify_frozen()
    original, resumed = read(details['original_args']), read(plan['runs']['place_sam']['resolved_args'])
    changed = {key for key in set(original) | set(resumed) if original.get(key) != resumed.get(key)}
    if changed != {'load_model', 'model_path', 'experiment_tag', 'checkpoint_root'} or resumed['load_model'] is not True:
        raise RuntimeError('SAM recovery changed training hyperparameters')
    queue.audit_contract()
    if args.command == 'check':
        print('SAM recovery inputs, frozen source, evaluator and training configuration: passed', flush=True)
    else:
        queue.run()


if __name__ == '__main__':
    main()
