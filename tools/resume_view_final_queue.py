#!/usr/bin/env python3
"""Recover the interrupted final-view B0 at epoch 4 using frozen training code."""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / '.local_data/view_final_20261023/source_snapshot'
sys.path.insert(0, str(SNAPSHOT))
spec = importlib.util.spec_from_file_location('recovery_frozen_view_final', SNAPSHOT / 'tools/queue_view_final.py')
FROZEN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FROZEN)
from tools.queue_view_dp4 import atomic_json, digest, now, process_matches, read_json

RESUME_FIELDS = {'experiment_tag', 'checkpoint_root', 'load_model', 'model_path'}
METRICS = FROZEN.METRICS


def validate_args(original, recovered, checkpoint_path):
    if len(original) != 117 or set(original) != set(recovered):
        raise RuntimeError('resume must retain all 117 original trainer fields')
    changed = {key for key in original if original[key] != recovered[key]}
    if changed != RESUME_FIELDS:
        raise RuntimeError('only the four explicit B0 recovery fields may change: ' + str(changed))
    if (original['load_model'] or original['model_path'] or not recovered['load_model']
            or recovered['model_path'] != str(checkpoint_path)
            or recovered['val_mode'] or not recovered['best_on_test']
            or recovered['epochs'] != 30 or recovered['eval_split'] != 'test'):
        raise RuntimeError('invalid final-view continuation protocol')
    for key in ('checkpoint_root', 'experiment_tag'):
        if not recovered[key] or recovered[key] == original[key]:
            raise RuntimeError('recovery requires independent output identity')
    return dict(status='passed', fields_checked=117, allowed_changes=sorted(changed))


def freeze_copy(source, destination, expected=None):
    source, destination = Path(source), Path(destination)
    checksum = digest(source)
    if expected is not None and checksum != expected:
        raise RuntimeError('inherited source hash changed: ' + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest(destination) != checksum:
            raise RuntimeError('refusing to overwrite different recovery artifact: ' + str(destination))
    else:
        temporary = destination.with_name(destination.name + '.%d.tmp' % os.getpid())
        shutil.copyfile(str(source), str(temporary))
        if digest(temporary) != checksum:
            raise RuntimeError('recovery artifact copy checksum mismatch')
        os.replace(str(temporary), str(destination))
    return dict(path=str(destination.resolve()), sha256=checksum, source=str(source.resolve()))


def resume_argv(original, args):
    argv = list(original)
    for key in ('experiment_tag', 'checkpoint_root', 'model_path'):
        option = '--' + key
        if argv.count(option) != 1:
            raise RuntimeError('ambiguous frozen argv option: ' + option)
        argv[argv.index(option) + 1] = str(args[key])
    if '--load_model' in argv:
        raise RuntimeError('original final argv is already a resume')
    argv.append('--load_model')
    return argv


def prepare(original_plan, recovery_root, parallel_h1_plan=None):
    original_plan, recovery_root = Path(original_plan).resolve(), Path(recovery_root).resolve()
    plan = read_json(original_plan)
    if recovery_root.exists() and any(path.name != 'parallel_h1' for path in recovery_root.iterdir()):
        raise FileExistsError('recovery directory already contains main recovery artifacts: ' + str(recovery_root))
    args = read_json(plan['runs']['b0']['resolved_args'])
    results = [path for path in (Path(plan['workspace']) / 'result').iterdir()
               if ('_[' + args['experiment_tag'] + ']_') in path.name]
    if len(results) != 1:
        raise RuntimeError('expected one interrupted final B0 result')
    result = results[0]
    checkpoints = Path(args['checkpoint_root']) / result.name
    import torch
    state = torch.load(checkpoints / 'epoch4.pth', map_location='cpu', weights_only=False)
    history = copy.deepcopy(state['history'])
    if (state['epoch'] != 4 or state['eval_split'] != 'test' or history.get('val')
            or any([row['epoch'] for row in history[split]] != [1, 2, 3, 4]
                   for split in ('train', 'test'))):
        raise RuntimeError('source checkpoint is not a complete epoch-4 final run')
    steps = {float(value['step']) for value in state['optimizer']['state'].values() if 'step' in value}
    if steps != {1824.} or any(abs(group['lr'] - 1e-4) > 1e-14 for group in state['optimizer']['param_groups']):
        raise RuntimeError('source Adam progress/next LR differs from epoch-4 final protocol')
    prefix_selected = max(history['test'], key=lambda row: row['group_mAP_1.0'])
    if state['selection_state']['epoch'] != prefix_selected['epoch']:
        raise RuntimeError('source checkpoint selection state differs from retained TEST history')
    del state
    recovery_root.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for name in ['epoch%d.pth' % epoch for epoch in range(1, 5)] + ['best_test.pth']:
        inputs[name] = freeze_copy(checkpoints / name, recovery_root / 'inputs' / name)
    predictions = Path(plan['workspace']) / args['result_path'] / result.name
    for epoch in range(1, 5):
        name = 'pred_group_epoch_%d.txt' % epoch
        inputs[name] = freeze_copy(predictions / name, recovery_root / 'inputs' / name)
    recovered = dict(args, experiment_tag=args['experiment_tag'] + '_resume_e4_power',
                     checkpoint_root=str(recovery_root / 'b0/checkpoints'),
                     load_model=True, model_path=inputs['epoch4.pth']['path'])
    validate_args(args, recovered, inputs['epoch4.pth']['path'])
    args_path, argv_path = recovery_root / 'b0.args.json', recovery_root / 'b0.argv.json'
    atomic_json(args_path, recovered)
    atomic_json(argv_path, resume_argv(read_json(plan['runs']['b0']['argv']), recovered))
    plan['queue_root'] = str(recovery_root)
    plan['runs']['b0'] = dict(plan['runs']['b0'], resolved_args=str(args_path), argv=str(argv_path))
    plan['recovery'] = dict(original_plan=str(original_plan), original_result=str(result),
        resume_epoch=4, next_epoch=5, inputs=inputs, history_prefix=history,
        original_b0_args=args, selection_label='best-on-TEST', rng_state_restored=False,
        note='model/Adam/scheduler/history/best resume; original checkpoint has no RNG state')
    if parallel_h1_plan:
        plan['parallel_h1_plan'] = str(Path(parallel_h1_plan).resolve())
    pinned = [original_plan, args_path, argv_path, Path(__file__).resolve()] + [Path(item['path']) for item in inputs.values()]
    if parallel_h1_plan:
        pinned.append(Path(parallel_h1_plan).resolve())
    for path in pinned:
        plan['frozen_files'].append(dict(path=str(path), sha256=digest(path)))
    output = recovery_root / 'queue.plan.json'
    atomic_json(output, plan)
    print(str(output), flush=True)
    return output


class Queue(FROZEN.Queue):
    def validate_recovery(self):
        recovery = self.plan['recovery']
        original = read_json(recovery['original_plan'])
        actual = read_json(self.plan['runs']['b0']['resolved_args'])
        result = validate_args(read_json(original['runs']['b0']['resolved_args']), actual,
                               recovery['inputs']['epoch4.pth']['path'])
        for variant in ('h1_sup', 'sam'):
            if (read_json(self.plan['runs'][variant]['resolved_args']) != read_json(original['runs'][variant]['resolved_args'])
                    or read_json(self.plan['runs'][variant]['argv']) != read_json(original['runs'][variant]['argv'])):
                raise RuntimeError('recovery must preserve original final ' + variant + ' configuration')
        if read_json(self.plan['runs']['b0']['argv']) != resume_argv(read_json(original['runs']['b0']['argv']), actual):
            raise RuntimeError('recovered B0 argv differs beyond four resume fields')
        return result

    def wait_nas(self):
        args = read_json(self.plan['runs']['b0']['resolved_args'])
        while True:
            try:
                with open(args['groundtruth'], 'rb') as stream:
                    if not stream.read(64):
                        raise OSError('groundtruth file is empty')
                for video in range(1, 25):
                    folder = Path(args['data_path']) / 'cafe' / str(video)
                    clips = sorted(path for path in folder.iterdir() if path.name.isdigit())
                    if not clips:
                        raise OSError('no clips visible in ' + str(folder))
                    for probe in (clips[0] / 'ann.json', clips[0] / 'images/frames_0.jpg'):
                        with probe.open('rb') as stream:
                            if not stream.read(64):
                                raise OSError('empty dataset probe: ' + str(probe))
                return
            except OSError as error:
                self.status('waiting_nas', error=str(error), next_stage='resume_view_b0_epoch5')
                time.sleep(self.interval)

    def verify_frozen(self):
        self.wait_nas()
        super().verify_frozen()

    def local_check(self):
        pending = []
        for entry in self.plan['frozen_files']:
            if str(entry['path']).startswith('/nas/'):
                pending.append(entry['path'])
            elif digest(entry['path']) != entry['sha256']:
                raise RuntimeError('local frozen input changed: ' + entry['path'])
        report = dict(status='local_checks_passed', recovery=self.validate_recovery(), nas_pending=pending,
                      full_contract='will run after NAS gate', checked_at=now())
        atomic_json(self.base / 'local_check.json', report)
        return report

    def audit_contract(self):
        # contract_plan still names the three original from-scratch final runs.
        super().audit_contract()
        report = self.validate_recovery()
        from tools.validate_view_final_contract import load_trainer_parser, verify_argv
        planned = read_json(self.plan['runs']['b0']['resolved_args'])
        parser = load_trainer_parser(self.snapshot, planned)
        report['argv'] = verify_argv(parser, read_json(self.plan['runs']['b0']['argv']), planned, self.snapshot)
        atomic_json(self.base / 'recovery_config_audit.json', report)

    def wait_dependency(self):
        # Old PIDs/ticks are from an earlier boot. Never signal that handoff.
        self.status('power_recovery_ready', next_epoch=5)

    def run_training(self, variant):
        if variant != 'h1_sup' or not self.plan.get('parallel_h1_plan'):
            return super().run_training(variant)
        parallel = FROZEN.Queue(self.plan['parallel_h1_plan'])
        planned = read_json(self.plan['runs']['h1_sup']['resolved_args'])
        if read_json(parallel.plan['runs']['h1_sup']['resolved_args']) != planned:
            raise RuntimeError('parallel H1 configuration differs from approved final run')
        process_file = Path(self.plan.get('parallel_h1_process_file', parallel.base / 'queue.process.json'))
        status_file = parallel.base / 'h1_sup/training.status'
        while True:
            tokens = status_file.read_text().split() if status_file.exists() else []
            status = tokens[0] if tokens else ''
            if status == 'training_complete':
                parallel.verify_training('h1_sup', planned)
                self.status('parallel_h1_complete', completion_audit=str(parallel.base / 'h1_sup/completion_audit.json'))
                return
            if status in ('training_failed', 'training_interrupted'):
                raise RuntimeError('parallel H1 stopped: ' + status)
            if process_file.exists():
                process = read_json(process_file)
                boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                if process['boot_id'] != boot or not process_matches(process['pid'], process['start_ticks']):
                    raise RuntimeError('parallel H1 controller exited without complete status')
            self.status('waiting_parallel_h1', training_status=status or 'awaiting_launch')
            time.sleep(self.interval)

    def inherit_artifacts(self, result, checkpoint, selected_epoch):
        inputs = self.plan['recovery']['inputs']
        predictions = self.root / read_json(self.plan['runs']['b0']['resolved_args'])['result_path'] / result.name
        inherited = {}
        for epoch in range(1, 5):
            for name, destination in [('epoch%d.pth' % epoch, checkpoint),
                                      ('pred_group_epoch_%d.txt' % epoch, predictions)]:
                item = inputs[name]
                inherited[name] = freeze_copy(item['path'], destination / name, item['sha256'])
        if selected_epoch <= 4:
            item = inputs['best_test.pth']
            inherited['best_test.pth'] = freeze_copy(item['path'], checkpoint / 'best_test.pth', item['sha256'])
        return inherited

    def verify_training(self, variant, planned):
        if variant != 'b0':
            return super().verify_training(variant, planned)
        self.validate_recovery()
        results = [path for path in (self.root / 'result').iterdir()
                   if ('_[' + planned['experiment_tag'] + ']_') in path.name]
        if len(results) != 1:
            raise RuntimeError('expected one recovered final B0 result')
        result = results[0]
        if read_json(result / 'args.json') != planned or len(planned) != 117:
            raise RuntimeError('recovered trainer differs from 117 frozen fields')
        summary = read_json(result / 'summary.json')
        args = summary['args']
        if (not set(planned).issubset(args) or {key: args[key] for key in planned} != planned
                or set(args) - set(planned) - {'peak_cuda_memory_mb'}):
            raise RuntimeError('recovered summary arguments differ')
        for split in ('train', 'test'):
            if summary['history'][split][:4] != self.plan['recovery']['history_prefix'][split]:
                raise RuntimeError('recovered history changed the original first four epochs')
        selected, final = FROZEN.selected_test_row(summary)
        checkpoint = Path(planned['checkpoint_root']) / result.name
        inherited = self.inherit_artifacts(result, checkpoint, selected['epoch'])
        for name in ['epoch%d.pth' % epoch for epoch in range(1, 31)] + ['best_test.pth']:
            if not (checkpoint / name).is_file():
                raise RuntimeError('missing recovered checkpoint: ' + name)
        if (checkpoint / 'best_val.pth').exists():
            raise RuntimeError('unexpected VAL checkpoint')
        import torch
        best = torch.load(checkpoint / 'best_test.pth', map_location='cpu', weights_only=False)
        selection = best['selection_state']
        if (best['epoch'] != selected['epoch'] or best['eval_split'] != 'test'
                or selection['epoch'] != selected['epoch'] or selection['value'] != selected['group_mAP_1.0']
                or selection.get('split') != 'test' or selection.get('selection_label') != 'best-on-TEST'):
            raise RuntimeError('recovered best_test differs from global first argmax TEST epoch')
        selected_path = selection.get('checkpoint_path')
        del best
        expected = None
        predictions = self.root / planned['result_path'] / result.name
        for epoch in range(1, 31):
            clips = set()
            with (predictions / ('pred_group_epoch_%d.txt' % epoch)).open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = line.split()
                    video, clip = int(row[0]), int(row[1])
                    if video not in FROZEN.TEST_VIDEOS:
                        raise RuntimeError('recovery prediction contains non-TEST video')
                    clips.add((video, clip))
            if len(clips) != 2414 or {video for video, _ in clips} != FROZEN.TEST_VIDEOS or (expected is not None and clips != expected):
                raise RuntimeError('recovery TEST clip inventory differs')
            expected = clips
        atomic_json(self.base / 'b0/completion_audit.json', dict(status='passed',
            result_dir=str(result), checkpoint_dir=str(checkpoint), args_fields_checked=117,
            resume_epoch=4, train_epochs=30, test_epochs=30, val_epochs=0, test_clips=2414,
            selection_label='best-on-TEST', selected_epoch=selected['epoch'], selected_metrics=selected,
            epoch30_metrics=final, inherited_artifacts=inherited,
            best_checkpoint=str(checkpoint / 'best_test.pth'), stored_selection_path=selected_path,
            selection_path_note='copied early best retains its original provenance path', completed_at=now()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'check', 'run'])
    parser.add_argument('--original-plan')
    parser.add_argument('--recovery-root')
    parser.add_argument('--parallel-h1-plan')
    parser.add_argument('--plan')
    args = parser.parse_args()
    if args.command == 'prepare':
        if not args.original_plan or not args.recovery_root:
            parser.error('prepare requires --original-plan and --recovery-root')
        prepare(args.original_plan, args.recovery_root, args.parallel_h1_plan)
    else:
        if not args.plan:
            parser.error('check/run require --plan')
        queue = Queue(args.plan)
        if queue.snapshot != SNAPSHOT:
            raise RuntimeError('recovery must use the original final source snapshot')
        if args.command == 'check':
            print(json.dumps(queue.local_check(), indent=2), flush=True)
        else:
            queue.run()


if __name__ == '__main__':
    main()
