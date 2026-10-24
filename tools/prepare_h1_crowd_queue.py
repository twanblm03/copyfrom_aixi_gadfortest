#!/usr/bin/env python3
"""Freeze a matched place crowd H1 successor; preparation never starts jobs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from queue_h1_crowd import (CrowdQueue, EVALUATOR_SHA256, NEW_ARGS, VARIANT,
                            parser_roundtrip, require, validate_args,
                            validate_baseline, validate_geometry, view_queue)


ROOT = Path(__file__).resolve().parents[1]
TAG = 'place_h1_crowd_randomseg_dp4_lr1_noclip_seed1'
ANCHOR_NAME = '[cafe]_GAD_[h1_sup_randomseg_dp4_lr1_noclip_seed1]_[2026-10-20_21-52-07]'


def write(path, value):
    view_queue.atomic_json(path, value)


def process_identity(pid):
    try:
        fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start_ticks=int(fields[19]))
    except (OSError, IndexError, ValueError):
        return None


def prepare(root, base, reference_args, view_plan, preflight_report):
    root, base = Path(root).resolve(), Path(base).resolve()
    reference_args = Path(reference_args).resolve()
    view_plan = Path(view_plan).resolve()
    preflight_report = Path(preflight_report).resolve()
    snapshot = base / 'source_snapshot'
    require(not snapshot.exists() and not (base / 'queue.pid').exists()
            and not (base / 'queue.plan.json').exists(), 'crowd snapshot/queue already exists')
    require(not (base / VARIANT).exists(), 'crowd run directory already exists')
    existing = root / 'result'
    require(not any(('_[' + TAG + ']_') in p.name for p in existing.iterdir()),
            'crowd experiment tag already has output')
    require(preflight_report.is_file(), 'TRAIN geometry preflight report is required')
    geometry = view_queue.read_json(preflight_report)
    validate_geometry(geometry, view_queue.digest(reference_args))

    old_plan = view_queue.read_json(view_plan)
    require(old_plan.get('protocol') == 'new_dp4_corrected_evaluator'
            and old_plan.get('method_order') == ['b0', 'h1_sup', 'sam'],
            'existing view queue protocol differs')
    for item in old_plan['frozen_files']:
        require(view_queue.digest(item['path']) == item['sha256'],
                'existing view frozen input changed: ' + item['path'])
    require(view_queue.digest(root / 'evaluation/cafe_eval.py') == EVALUATOR_SHA256,
            'corrected evaluator SHA256 mismatch')

    anchor = view_queue.read_json(reference_args)
    baseline_summary = reference_args.with_name('summary.json')
    baseline_status = root / '.local_data/historical_lr_randomsegments_dp4/h1_sup/full_training.status'
    checkpoint_dir = Path(anchor['checkpoint_root']) / reference_args.parent.name
    require(baseline_status.read_text().split()[:1] == ['training_complete'],
            'matched place baseline has not completed')
    validate_baseline(anchor, view_queue.read_json(baseline_summary), checkpoint_dir)
    planned = dict(anchor, **NEW_ARGS)
    planned.update(experiment_tag=TAG, checkpoint_root=str(base / VARIANT / 'checkpoints'))
    validate_args(anchor, planned)

    python = old_plan['python']
    # Verify the actual new parser before writing a frozen experiment.
    parser_roundtrip(root, python, dict(anchor, heatmap_target_mode='box',
                                       heatmap_crowd_cap_ratio=.35), root)
    parsed = parser_roundtrip(root, python, planned, root)
    argv = parsed['argv']
    base.mkdir(parents=True, exist_ok=True)
    staging = base / ('source_snapshot.preparing.%d' % os.getpid())
    require(not staging.exists(), 'snapshot staging directory already exists')
    staging.mkdir()
    try:
        files = [root / 'train.py', root / 'test.py']
        for folder in ('models', 'util', 'dataloader', 'evaluation', 'tools', 'tests'):
            files.extend(path for path in (root / folder).rglob('*.py')
                         if '__pycache__' not in path.parts)
        files.extend((root / 'configs/heatmap').glob('*.json'))
        files.extend((root / 'label_map').glob('*.pbtxt'))
        require((root / 'tests/heatmap/test_crowd_targets.py') in files,
                'CPU crowd target contract tests are missing')
        for source in sorted(set(files)):
            target = staging / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(target))
        config = staging / 'configs/h1_crowd'
        config.mkdir(parents=True)
        for source, name in ((reference_args, 'anchor.args.json'),
                             (baseline_summary, 'anchor.summary.json'),
                             (baseline_status, 'anchor.status'),
                             (preflight_report, 'target_geometry_audit.json')):
            shutil.copy2(str(source), str(config / name))
        write(config / 'crowd.args.json', planned)
        write(config / 'crowd.argv.json', argv)
        parser_roundtrip(staging, python, planned, root)
        # Publish only a complete source tree; never touch the current queue.
        staging.rename(snapshot)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    view_base = Path(old_plan['queue_root'])
    view_state = view_queue.read_json(view_base / 'queue_state.json')
    process = process_identity(int(view_state['pid']))
    plan = dict(schema_version=1, protocol='new_dp4_corrected_evaluator', split='place',
                workspace=str(root), queue_root=str(base), snapshot_root=str(snapshot),
                python=python, method_order=[VARIANT],
                environment=dict(old_plan['environment'], PYTHONDONTWRITEBYTECODE='1'),
                gpu_uuids=old_plan['gpu_uuids'], minimum_free_mib=old_plan['minimum_free_mib'],
                poll_seconds=old_plan.get('poll_seconds', 30),
                upstream_plan=str(view_plan),
                dependency=old_plan['dependency'],
                view_dependency=dict(queue_root=str(view_base), state=str(view_base / 'queue_state.json'),
                                     lock=str(view_base / 'queue.lock'), process=process),
                baseline=dict(args=str(snapshot / 'configs/h1_crowd/anchor.args.json'),
                              summary=str(snapshot / 'configs/h1_crowd/anchor.summary.json'),
                              status=str(snapshot / 'configs/h1_crowd/anchor.status'),
                              checkpoint_dir=str(checkpoint_dir), reference_args=str(reference_args)),
                runs={VARIANT: dict(reference_args=str(snapshot / 'configs/h1_crowd/anchor.args.json'),
                                    resolved_args=str(snapshot / 'configs/h1_crowd/crowd.args.json'),
                                    argv=str(snapshot / 'configs/h1_crowd/crowd.argv.json'))},
                preflight=dict(cpu_only=True,
                               geometry_report=str(snapshot / 'configs/h1_crowd/target_geometry_audit.json'),
                               argv=[python, '-m', 'unittest', 'discover', '-s', 'tests/heatmap',
                                     '-p', 'test_crowd_targets.py', '-v']),
                expected_inventory=dict(train_clips=6444, val_clips=1386,
                                        steps_per_epoch=402, total_optimizer_steps=12060),
                prepared_at=view_queue.now())
    pinned = [path for path in snapshot.rglob('*') if path.is_file()]
    pinned.extend([Path(anchor['tracks_path']), Path(anchor['groundtruth']),
                   root / anchor['labelmap'], root / '.local_data/torch/hub/checkpoints/resnet18-f37072fd.pth',
                   view_plan])
    plan['frozen_files'] = [dict(path=str(path), sha256=view_queue.digest(path))
                            for path in sorted(set(pinned))]
    manifest = base / 'source_manifest.json'
    write(manifest, dict(files=plan['frozen_files']))
    plan['source_manifest'] = dict(path=str(manifest), sha256=view_queue.digest(manifest))
    plan_path = base / 'queue.plan.json'
    write(plan_path, plan)
    # Treat the snapshot as immutable, with hashes checked again at every launch.
    for path in snapshot.rglob('*'):
        path.chmod(0o555 if path.is_dir() else 0o444)
    snapshot.chmod(0o555)
    report = CrowdQueue(plan_path).audit_contract()
    return dict(plan=str(plan_path), snapshot_root=str(snapshot),
                frozen_files=len(plan['frozen_files']), status=report['status'],
                auto_training_runs=[TAG], launched=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--reference-args', type=Path)
    parser.add_argument('--view-plan', type=Path)
    parser.add_argument('--preflight-report', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    base = (args.base or root / '.local_data/h1_crowd_20261022').resolve()
    report = prepare(root, base, args.reference_args or root / 'result' / ANCHOR_NAME / 'args.json',
                     args.view_plan or root / '.local_data/view_dp4_20261021/queue.plan.json',
                     args.preflight_report or base / 'target_geometry_audit.json')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
