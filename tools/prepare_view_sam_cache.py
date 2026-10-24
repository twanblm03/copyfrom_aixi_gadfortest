"""Reuse reviewed place SAM arrays in an isolated, complete view cache.

Only protocol/split metadata changes during reuse. Arrays are immutable hard
links where possible, and all source cache files remain untouched. This tool
does not import SAM or use CUDA. Missing frames are left for the existing full
cache generator; audit/verify fail until every manifest frame is available.
"""
import argparse
from collections import Counter
import copy
import errno
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.sam_person_full import iter_manifest, source_config, verify
from tools.sam_person_masks import version_payload
from util.sam_person import (SCHEMA, atomic_json, cache_paths, canonical,
                             file_hash, frame_fingerprint, projected_paths,
                             read_projected_frame)

IMAGE_HW = (720, 1280)
FEATURE_HW = (23, 40)


def frame_key(row):
    return (row['video_id'], row['clip_id'], row['source_frame_index'])


def semantic_identity(row):
    """All source metadata except the partition labels must remain identical."""
    return {key: value for key, value in row.items()
            if key not in ('protocol', 'split', 'sampling')}


def partition_videos():
    # Keep audit independent of model/torch/CUDA initialization.
    return {'train': {1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22},
            'val': {3, 7, 11, 15, 19, 23}}


def inspect_manifest(path):
    """Check partition, uniqueness and frozen companion inventory."""
    counts = Counter()
    frames = {'train': {}, 'val': {}}
    for row in iter_manifest(path):
        split = row['split']
        if (row['schema_version'] != SCHEMA or row['dataset'] != 'CAFE'
                or row['protocol'] != 'view_train_val_b0_repro'
                or split not in frames
                or row['video_id'] not in partition_videos()[split]):
            raise ValueError('invalid view manifest partition: %s' % (frame_key(row),))
        clip = (row['video_id'], row['clip_id'])
        ids = frames[split].setdefault(clip, set())
        if row['source_frame_index'] in ids:
            raise ValueError('duplicate manifest frame: %s' % (frame_key(row),))
        ids.add(row['source_frame_index'])
        counts[split] += 1
    inventory = json.loads(Path(str(path) + '.clips.json').read_text())
    for split in ('train', 'val'):
        expected = {tuple(item['clip']): set(item['frames']) for item in inventory[split]}
        if len(expected) != len(inventory[split]) or frames[split] != expected:
            raise ValueError('manifest differs from frozen clip/frame inventory: ' + split)
    summary = json.loads(Path(str(path) + '.summary.json').read_text())
    clip_counts = {split: len(frames[split]) for split in frames}
    if (summary['clips'] != clip_counts or summary['frames'] != sum(counts.values())
            or summary['manifest_sha256'] != file_hash(path)
            or summary['sampling'] != 'all TRAIN frame candidates; fixed VAL segment centers'):
        raise ValueError('view manifest summary/hash/sampling mismatch')
    return dict(frames=sum(counts.values()), frames_by_split=dict(counts), clips=clip_counts,
                manifest_sha256=summary['manifest_sha256'])


def validate_bundle(root, cache_version, row, expected_model, verify_source=False):
    """Verify identities and both array checksums before any reuse."""
    raw_array, raw_marker = cache_paths(root, cache_version, row)
    meta = json.loads(raw_marker.read_text())
    if (meta['schema_version'] != SCHEMA or meta['cache_version'] != cache_version
            or meta['frame_fingerprint'] != frame_fingerprint(row)
            or canonical(meta['source']) != canonical(row)
            or meta['mask_shape'] != [len(row['actors'])] + row['original_hw']
            or len(meta['records']) != len(row['actors'])):
        raise ValueError('raw cache identity mismatch: ' + str(raw_marker))
    for actor, record in zip(row['actors'], meta['records']):
        if (record['actor_uid'] != actor['actor_uid']
                or record.get('model') != expected_model):
            raise ValueError('SAM model/prompt/actor configuration mismatch: ' + str(raw_marker))
    if file_hash(raw_array) != meta['array_sha256']:
        raise ValueError('raw cache checksum mismatch: ' + str(raw_array))
    _, projected = read_projected_frame(root, cache_version, row, IMAGE_HW, FEATURE_HW)
    if (projected['raw_array_sha256'] != meta['array_sha256']
            or projected['raw_valid'] != [bool(r['raw_mask_valid']) for r in meta['records']]):
        raise ValueError('projected/raw identity mismatch: ' + str(raw_marker))
    if verify_source and file_hash(row['source_image_path']) != row['source_image_hash']:
        raise ValueError('source image changed: ' + row['source_image_key'])
    return meta, projected


def install_array(source, destination):
    """Publish a whole array without replacing a prior completed result."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_hash(source) != file_hash(destination):
            raise ValueError('existing destination array differs: ' + str(destination))
        return 'existing_array'
    try:
        os.link(str(source), str(destination))
        return 'hardlinked_arrays'
    except OSError as error:
        if error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP):
            raise
    temporary = destination.with_name(destination.name + '.%d.copy.tmp' % os.getpid())
    try:
        shutil.copyfile(str(source), str(temporary))
        if file_hash(temporary) != file_hash(source):
            raise ValueError('copy checksum mismatch: ' + str(source))
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()
    return 'copied_arrays'


def source_index(path):
    # Store compact fingerprints rather than the >1 GB full source rows.
    result = {}
    for row in iter_manifest(path):
        key = frame_key(row)
        if key in result:
            raise ValueError('duplicate source manifest identity: %s' % (key,))
        result[key] = frame_fingerprint(row)
    return result


def reuse_row(row, source_root, target_root, cache_version, expected_model,
              approved_source_fingerprint, source_manifest_sha256):
    """Return reused/existing/missing; reject any source mutation or corruption."""
    target_raw, target_marker = cache_paths(target_root, cache_version, row)
    if target_marker.exists():
        validate_bundle(target_root, cache_version, row, expected_model)
        return Counter(existing=1)
    if approved_source_fingerprint is None:
        return Counter(missing=1)
    source_raw, source_marker = cache_paths(source_root, cache_version, row)
    source_meta = json.loads(source_marker.read_text())
    source_row = source_meta['source']
    if (frame_fingerprint(source_row) != approved_source_fingerprint
            or semantic_identity(source_row) != semantic_identity(row)):
        raise ValueError('source/new manifest data differ: %s' % (frame_key(row),))
    source_meta, projected_meta = validate_bundle(
        source_root, cache_version, source_row, expected_model)
    source_projected, source_projected_marker = projected_paths(
        source_root, cache_version, row, IMAGE_HW, FEATURE_HW)
    target_projected, target_projected_marker = projected_paths(
        target_root, cache_version, row, IMAGE_HW, FEATURE_HW)
    counters = Counter(reused=1)
    counters[install_array(source_raw, target_raw)] += 1
    counters[install_array(source_projected, target_projected)] += 1
    provenance = dict(operation='reuse unchanged reviewed arrays; partition metadata only',
                      source_manifest_sha256=source_manifest_sha256,
                      source_cache_root=str(Path(source_root).resolve()),
                      source_frame_fingerprint=approved_source_fingerprint,
                      source_raw_marker_sha256=file_hash(source_marker),
                      source_projected_marker_sha256=file_hash(source_projected_marker))
    new_raw = copy.deepcopy(source_meta)
    new_raw.update(source=row, frame_fingerprint=frame_fingerprint(row), reuse_provenance=provenance)
    new_projected = copy.deepcopy(projected_meta)
    new_projected.update(frame_fingerprint=frame_fingerprint(row), reuse_provenance=provenance)
    # Raw completion marker is published last; interrupted reuse is resumable.
    atomic_json(target_projected_marker, new_projected)
    atomic_json(target_marker, new_raw)
    return counters


def audit_cache(manifest, root, cache_version):
    counters = Counter()
    examples = []
    for row in iter_manifest(manifest):
        paths = cache_paths(root, cache_version, row) + projected_paths(
            root, cache_version, row, IMAGE_HW, FEATURE_HW)
        complete = all(path.is_file() for path in paths)
        counters['present' if complete else 'missing'] += 1
        counters[row['split'] + ('_present' if complete else '_missing')] += 1
        if not complete and len(examples) < 20:
            examples.append(row['source_image_key'])
    return dict(counters, missing_examples=examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['reuse', 'audit', 'verify'])
    parser.add_argument('--manifest', default=str(ROOT / '.local_data/sam/view_randomsegments_manifest.jsonl'))
    parser.add_argument('--source-manifest', default=str(ROOT / '.local_data/sam/randomsegments_manifest.jsonl'))
    parser.add_argument('--source-cache-root', default=str(ROOT / '.local_data/sam/cache'))
    parser.add_argument('--cache-root', default=str(ROOT / '.local_data/sam/view_cache'))
    parser.add_argument('--cache-version', default='raw_bc294f454bfdfad21798')
    parser.add_argument('--sam-repo', default=str(ROOT / '.local_deps/segment-anything'))
    parser.add_argument('--checkpoint', default=str(ROOT / '.local_data/sam/checkpoints/sam_vit_b_01ec64.pth'))
    parser.add_argument('--report', required=True)
    parser.add_argument('--verify-workers', type=int, default=4)
    cli = parser.parse_args()
    source_root, target_root = Path(cli.source_cache_root).resolve(), Path(cli.cache_root).resolve()
    if source_root == target_root or source_root in target_root.parents or target_root in source_root.parents:
        parser.error('view cache root must be isolated from the source place cache root')
    report = dict(command=cli.command, status='running', manifest=str(Path(cli.manifest).resolve()),
                  cache_root=str(target_root), cache_version=cli.cache_version)
    try:
        report['manifest_inventory'] = inspect_manifest(cli.manifest)
        config = source_config(cli)
        if config['cache_version'] != cli.cache_version:
            raise ValueError('requested cache version differs from reviewed SAM configuration')
        if cli.command == 'reuse':
            index = source_index(cli.source_manifest)
            source_hash = file_hash(cli.source_manifest)
            report['source_manifest_sha256'] = source_hash
            counters = Counter()
            for position, row in enumerate(iter_manifest(cli.manifest), 1):
                counters.update(reuse_row(row, source_root, target_root, cli.cache_version,
                                         version_payload(config), index.get(frame_key(row)), source_hash))
                if position % 1000 == 0:
                    report['reuse'] = dict(counters)
                    atomic_json(cli.report, report)
                    print('reuse', position, dict(counters), flush=True)
            report['reuse'] = dict(counters)
        report['coverage'] = audit_cache(cli.manifest, target_root, cli.cache_version)
        if cli.command != 'reuse' and report['coverage'].get('missing', 0):
            raise ValueError('view cache incomplete; every random TRAIN candidate is required')
        if cli.command == 'verify':
            # Existing exhaustive verifier checks JPEGs, packed masks, per-actor
            # checksums, projected arrays, and all raw/projected identities.
            cli.shard, cli.shards, cli.projected = 0, 1, True
            verify(cli)
            report['full_verification'] = 'passed'
        report['status'] = 'complete' if not report['coverage'].get('missing', 0) else 'reuse_complete_missing_generation'
    except Exception as error:
        report.update(status='failed', error=repr(error))
        atomic_json(cli.report, report)
        print(json.dumps(report, sort_keys=True), flush=True)
        raise
    atomic_json(cli.report, report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
