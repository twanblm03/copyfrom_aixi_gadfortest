#!/usr/bin/env python3
"""Freeze the user-approved three-view, best-on-TEST experiment queue."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def main():
    base = ROOT / '.local_data/view_final_20261023'
    snapshot = base / 'source_snapshot'
    if (base / 'queue.plan.json').exists() or (base / 'queue.pid').exists():
        raise RuntimeError('refusing to replace an existing final queue')
    for filename in ('queue_view_final.py', 'prepare_view_final_sam_cache.py', 'validate_view_final_contract.py'):
        source = ROOT / 'tools' / filename
        if not source.is_file():
            raise FileNotFoundError(source)
    if shutil.disk_usage(str(ROOT)).free < 40 * 1024 ** 3:
        raise RuntimeError('need 40 GiB free for all three sets of epoch checkpoints and cache preparation')
    contract = json.loads((ROOT / 'configs/view_final_dp4/contract.plan.json').read_text())
    for variant, run in contract['runs'].items():
        for key in ('reference_args', 'resolved_args', 'argv'):
            source = Path(run[key])
            target = snapshot / 'configs/view_final_dp4' / (variant + '.' + key + '.json')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(target))
            run[key] = str(target)
    for filename in ('queue_view_final.py', 'prepare_view_final_sam_cache.py', 'validate_view_final_contract.py'):
        shutil.copy2(str(ROOT / 'tools' / filename), str(snapshot / 'tools' / filename))
    contract_path = snapshot / 'configs/view_final_dp4/contract.plan.json'
    write(contract_path, contract)
    previous_path = ROOT / '.local_data/view_dp4_20261021/continuation_e4_nas_20261022/queue.plan.json'
    previous = json.loads(previous_path.read_text())
    handoff = ROOT / '.local_data/view_dp4_20261021/final_protocol_handoff_20261023.json'
    handoff_copy = base / 'predecessor_handoff.json'
    shutil.copy2(str(handoff), str(handoff_copy))
    b0 = json.loads(Path(contract['runs']['b0']['resolved_args']).read_text())
    sam = json.loads(Path(contract['runs']['sam']['resolved_args']).read_text())
    cache = dict(manifest=sam['sam_mask_manifest'], cache_root=sam['sam_cache_root'],
                 cache_version=sam['sam_cache_version'],
                 sam_repo=str(ROOT / '.local_deps/segment-anything'),
                 checkpoint=str(ROOT / '.local_data/sam/checkpoints/sam_vit_b_01ec64.pth'),
                 data_root=b0['data_path'], tracks_path=b0['tracks_path'],
                 manifest_report=str(ROOT / '.local_data/sam/view_final_prepare_20261023/manifest.report.json'),
                 sources=[dict(manifest=str(ROOT / '.local_data/sam/randomsegments_manifest.jsonl'),
                               cache_root=str(ROOT / '.local_data/sam/cache')),
                          dict(manifest=str(ROOT / '.local_data/sam/view_randomsegments_manifest.jsonl'),
                               cache_root=str(ROOT / '.local_data/sam/view_cache'))])
    plan = dict(schema_version=1, workspace=str(ROOT), queue_root=str(base), snapshot_root=str(snapshot),
                python=previous['python'], environment=copy.deepcopy(previous['environment']),
                gpu_uuids=previous['gpu_uuids'], minimum_free_mib=20000, poll_seconds=10,
                runs=contract['runs'], contract_plan=str(contract_path),
                protocol='best_on_test', evaluation_policy=contract['evaluation_policy'],
                method_order=['b0', 'h1_sup', 'sam'],
                handoff_file=str(handoff_copy), predecessor_plan=str(previous_path),
                predecessor_public_root=str(ROOT / '.local_data/view_dp4_20261021'),
                cache=cache, expected_inventory=dict(train_clips=7298, test_clips=2414, steps_per_epoch=456),
                initializer='ImageNet ResNet18; load_model=false for all three methods',
                source_changes=['Explicit best-on-TEST mode and all-epoch checkpoint retention',
                                'Report secondary best metrics at the mAP@1.0-selected epoch',
                                'Plot TEST history with explicit TEST curve labels',
                                'SAM guard permits view TRAIN+VAL -> TEST',
                                'Existing bounded same-image NAS I/O retry retained'])
    pinned = [path for path in snapshot.rglob('*') if path.is_file() and '__pycache__' not in path.parts]
    pinned.extend([handoff_copy, previous_path, Path(b0['tracks_path']), Path(b0['groundtruth']),
                   ROOT / b0['labelmap'], ROOT / '.local_data/torch/hub/checkpoints/resnet18-f37072fd.pth',
                   Path(cache['checkpoint'])])
    plan['frozen_files'] = [dict(path=str(path), sha256=sha(path)) for path in sorted(set(pinned))]
    write(base / 'queue.plan.json', plan)
    write(base / 'source_manifest.json', dict(files=plan['frozen_files'], source_changes=plan['source_changes']))
    print(json.dumps(dict(plan=str(base / 'queue.plan.json'), frozen_files=len(plan['frozen_files']),
                         protocol=plan['protocol'], evaluation_policy=plan['evaluation_policy']), indent=2))


if __name__ == '__main__':
    main()
