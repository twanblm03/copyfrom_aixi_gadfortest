#!/usr/bin/env python3
"""Freeze the reviewed SAM replacement and the exact live B0 handoff identity."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.queue_sam_encoder_after_b0 import atomic_json, digest, now, read_json, validate_contract


def identity(pid):
    fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
    if fields[0] in ('Z', 'T', 't'):
        raise RuntimeError('expected an active predecessor process')
    return dict(pid=pid, start_ticks=int(fields[19]),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def prepare(args):
    previous_path = Path(args.predecessor_plan).resolve()
    previous = read_json(previous_path)
    base = Path(args.queue_root).resolve()
    if base.exists():
        raise FileExistsError('use an unused queue directory: ' + str(base))
    manager, child = identity(args.manager_pid), identity(args.child_pid)
    state = read_json(Path(previous['queue_root']) / 'queue_state.json')
    fields = Path('/proc/%d/stat' % child['pid']).read_text().rsplit(')', 1)[1].split()
    if state['stage'] != 'training_b0' or state['pid'] != manager['pid'] or int(fields[1]) != manager['pid']:
        raise RuntimeError('B0 process/controller identity differs from the plan')
    reference_path = Path(previous['snapshot_root']) / 'configs/view_final_dp4/b0.resolved_args.json'
    reference = read_json(reference_path)
    old_argv = read_json(Path(previous['snapshot_root']) / 'configs/view_final_dp4/b0.argv.json')
    snapshot = base / 'source_snapshot'
    snapshot.mkdir(parents=True)
    # Snapshot source only: no data, weights, output directories or mutable logs.
    sources = list(ROOT.glob('*.py'))
    for folder in ('models', 'dataloader', 'evaluation', 'util', 'tools', 'tests', 'configs', 'label_map'):
        sources.extend(p for p in (ROOT / folder).rglob('*')
                       if p.is_file() and '__pycache__' not in p.parts
                       and p.suffix in ('.py', '.json', '.pbtxt', '.txt'))
    sources += [ROOT / 'scripts/train_cafe_sam_encoder.sh', ROOT / 'docs/sam_encoder/README.md']
    for source in sources:
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    sam_source = ROOT / '.local_deps/segment-anything/segment_anything'
    sam_repo = snapshot / '.local_deps/segment-anything'
    shutil.copytree(sam_source, sam_repo / 'segment_anything',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    checkpoint = ROOT / '.local_data/sam/checkpoints/sam_vit_b_01ec64.pth'
    planned = dict(reference, backbone='sam_vit_b', freeze_backbone=True,
        experiment_tag='view_final_sam_encoder_full_trainval_dp4_seed1',
        checkpoint_root=str(base / 'sam_encoder/checkpoints'),
        sam_repo=str(sam_repo), sam_checkpoint=str(checkpoint),
        sam_checkpoint_sha256=digest(checkpoint), sam_encoder_chunk_size=1,
        sam_output_height=23, sam_output_width=40,
        heatmap_target_mode='box', heatmap_crowd_cap_ratio=.35)
    report = validate_contract(reference, planned)
    argv = list(old_argv)
    for key in ('backbone', 'experiment_tag', 'checkpoint_root'):
        option = '--' + key
        if argv.count(option) != 1:
            raise RuntimeError('ambiguous original argv: ' + option)
        argv[argv.index(option) + 1] = str(planned[key])
    argv.append('--freeze_backbone')
    for key in ('sam_repo', 'sam_checkpoint', 'sam_checkpoint_sha256', 'sam_encoder_chunk_size',
                'sam_output_height', 'sam_output_width', 'heatmap_target_mode', 'heatmap_crowd_cap_ratio'):
        argv.extend(['--' + key, str(planned[key])])
    args_path, argv_path = base / 'sam_encoder.args.json', base / 'sam_encoder.argv.json'
    atomic_json(args_path, planned)
    atomic_json(argv_path, argv)
    # Load the new frozen real parser, without entering main or initializing CUDA.
    spec = importlib.util.spec_from_file_location('sam_queue_contract_validator',
        snapshot / 'tools/validate_view_final_contract.py')
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    parser = validator.load_trainer_parser(snapshot, planned)
    report['argv_roundtrip'] = validator.verify_argv(parser, argv, planned, snapshot)
    report['selection_code'] = validator.validate_runtime_policy(snapshot / 'train.py')
    paths = list(p for p in snapshot.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.extend([args_path, argv_path, reference_path, previous_path, checkpoint,
                  Path(reference['tracks_path']), Path(reference['groundtruth']),
                  ROOT / 'tools/resume_view_final_queue.py'])
    plan_path = base / 'queue.plan.json'
    plan = dict(plan_path=str(plan_path), workspace=str(ROOT), queue_root=str(base),
        snapshot_root=str(snapshot), python=previous['python'], poll_seconds=30,
        predecessor_plan=str(previous_path), predecessor_tool=str(ROOT / 'tools/resume_view_final_queue.py'),
        predecessor_state=str(Path(previous['queue_root']) / 'queue_state.json'),
        predecessor_manager=manager, predecessor_child=child,
        reference_args=str(reference_path), run=dict(resolved_args=str(args_path), argv=str(argv_path)),
        gpu_uuids=previous['gpu_uuids'], minimum_free_mib=20000,
        environment=previous['environment'],
        sequence=['wait_and_audit_b0', 'sam_encoder_dp4_smoke', 'sam_encoder_full_30_epochs',
                  'wait_and_audit_parallel_h1', 'original_sam_mask_roi_blend'],
        frozen_files=[dict(path=str(path), sha256=digest(path)) for path in sorted(set(paths))],
        created_at=now())
    atomic_json(plan_path, plan)
    atomic_json(base / 'preparation_audit.json', dict(report, checked_at=now()))
    print(json.dumps(dict(status='prepared', plan=str(plan_path),
                         source_files=len(sources), frozen_files=len(plan['frozen_files']),
                         args_fields=len(planned)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predecessor-plan', required=True)
    parser.add_argument('--queue-root', required=True)
    parser.add_argument('--manager-pid', type=int, required=True)
    parser.add_argument('--child-pid', type=int, required=True)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
