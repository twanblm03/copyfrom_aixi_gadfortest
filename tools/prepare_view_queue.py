#!/usr/bin/env python3
"""Freeze the reviewed source/configuration for the authorized view queue."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    base = ROOT / '.local_data/view_dp4_20261021'
    snapshot = base / 'source_snapshot'
    if snapshot.exists() or (base / 'queue.pid').exists():
        raise RuntimeError('snapshot/queue already exists; refusing to alter frozen experiment')
    contract = json.loads((ROOT / 'configs/view_dp4/contract.plan.json').read_text())
    # Pin reference args inside the snapshot so future report edits cannot drift.
    for name, entry in contract['runs'].items():
        for key in ('reference_args', 'resolved_args', 'argv'):
            source = ROOT / entry[key]
            relative = Path('configs/view_dp4') / (name + '.reference_args.json') if key == 'reference_args' else Path(entry[key])
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(target))
            entry[key] = str(target)
    source_files = [ROOT / 'train.py', ROOT / 'test.py']
    for folder in ('models', 'util', 'dataloader', 'evaluation', 'tools'):
        source_files.extend(path for path in (ROOT / folder).rglob('*.py') if '__pycache__' not in path.parts)
    source_files.extend((ROOT / 'configs/heatmap').glob('*.json'))
    source_files.extend((ROOT / 'label_map').glob('*.pbtxt'))
    source_files.append(ROOT / 'scripts/generate_sam_person_full.sh')
    for source in sorted(set(source_files)):
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(target))
    frozen_contract = snapshot / 'configs/view_dp4/contract.plan.json'
    write(frozen_contract, contract)
    runs = contract['runs']
    b0 = json.loads(Path(runs['b0']['resolved_args']).read_text())
    sam = json.loads(Path(runs['sam']['resolved_args']).read_text())
    plan = dict(
        schema_version=1, protocol='new_dp4_corrected_evaluator', split='view',
        workspace=str(ROOT), queue_root=str(base), snapshot_root=str(snapshot),
        python='/data2/mzy/GAD/.miniconda3/envs/gad/bin/python',
        method_order=['b0', 'h1_sup', 'sam'], contract_plan=str(frozen_contract), runs=runs,
        dependency=dict(status=str(ROOT / '.local_data/sam/dp4_randomsegments_lr1/full/training.status'),
                        processes=[dict(pid=3581790, start_ticks=594624224),
                                   dict(pid=3581798, start_ticks=594624230)]),
        gpu_uuids=['GPU-60b1d519-b004-babb-f4f2-1bb5f7c74c03',
                   'GPU-0995f70e-550d-c945-c43f-fbcfcb7285fa',
                   'GPU-3c30f17e-a770-2830-d95a-e960f8df1be7',
                   'GPU-b3966074-67ea-97fd-40e1-dffed27984dd'],
        minimum_free_mib=25000, poll_seconds=30,
        environment=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='0,1,2,3',
                         TORCH_HOME=str(ROOT / '.local_data/torch'), OMP_NUM_THREADS='4',
                         MKL_NUM_THREADS='4', NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1',
                         PYTHONUNBUFFERED='1'),
        cache=dict(manifest=sam['sam_mask_manifest'], cache_root=sam['sam_cache_root'],
                   cache_version=sam['sam_cache_version'],
                   source_manifest=str(ROOT / '.local_data/sam/randomsegments_manifest.jsonl'),
                   source_cache_root=str(ROOT / '.local_data/sam/cache'),
                   sam_repo=str(ROOT / '.local_deps/segment-anything'),
                   checkpoint=str(ROOT / '.local_data/sam/checkpoints/sam_vit_b_01ec64.pth'),
                   reuse_report=str(ROOT / '.local_data/sam/view_cache_reuse.json'),
                   data_root=b0['data_path'], tracks_path=b0['tracks_path']),
        expected_inventory=dict(train_clips=4857, val_clips=2441, manifest_frames=174532,
                                steps_per_epoch=303, total_optimizer_steps=9090))
    pinned = list(path for path in snapshot.rglob('*') if path.is_file())
    pinned.extend([Path(b0['tracks_path']), Path(b0['groundtruth']), ROOT / b0['labelmap'],
                   ROOT / '.local_data/torch/hub/checkpoints/resnet18-f37072fd.pth',
                   Path(plan['cache']['checkpoint'])])
    plan['frozen_files'] = [dict(path=str(path), sha256=sha(path)) for path in sorted(set(pinned))]
    write(base / 'queue.plan.json', plan)
    write(base / 'source_manifest.json', dict(files=plan['frozen_files']))
    print(json.dumps(dict(plan=str(base / 'queue.plan.json'), frozen_files=len(plan['frozen_files']),
                          snapshot_root=str(snapshot)), indent=2))


if __name__ == '__main__':
    main()
