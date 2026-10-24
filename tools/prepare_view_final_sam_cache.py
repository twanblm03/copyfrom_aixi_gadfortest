"""CPU preparation of isolated SAM masks for view TRAIN+VAL -> TEST.

TRAIN means all retained clips from the first three camera views. Every frame
reachable by random segment sampling is required. TEST contains only fixed
segment centers from the fourth view. No model training, predictions, metrics,
checkpoint selection, or CUDA initialization happens here.
"""
import argparse
from collections import Counter
import errno
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from tools.prepare_view_sam_cache import (
    audit_cache, frame_key, reuse_row, source_index, validate_bundle)
from tools.sam_person_full import actor_boxes_for_frame, iter_manifest, source_config, verify
from tools.sam_person_masks import version_payload
from util.sam_person import (SCHEMA, atomic_json, cache_paths, canonical, digest,
                             file_hash, frame_fingerprint, projected_paths)

PROTOCOL = 'view_trainval_test_b0_repro'
CACHE_VERSION = 'raw_bc294f454bfdfad21798'
SAMPLING = 'all TRAIN+VAL frame candidates; fixed TEST segment centers'
EXPECTED_TRAIN_CLIPS = 7298
EXPECTED_TEST_CLIPS = 2414
PARTITIONS = {
    'train': {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23},
    'test': {4, 8, 12, 16, 20, 24},
}
TRANSIENT_IO = {errno.EIO, errno.ESTALE, errno.ETIMEDOUT, errno.EHOSTDOWN,
                errno.EHOSTUNREACH, errno.ENETDOWN, errno.ENETUNREACH}


def retry_io(operation, attempts=6):
    for attempt in range(attempts):
        try:
            return operation()
        except OSError as error:
            if error.errno not in TRANSIENT_IO or attempt + 1 == attempts:
                raise
            delay = min(2 ** (attempt + 1), 30)
            print('transient NAS error; retry in %ss: %s' % (delay, error), flush=True)
            time.sleep(delay)


def candidate_frames(num_frames, frames_each, split, num_sampled_frames=5):
    if num_frames < 1 or frames_each < 1 or num_sampled_frames != 5:
        raise ValueError('invalid final manifest frame sampling metadata')
    if split == 'train':
        indices = range(num_frames)
    elif split == 'test':
        edges = np.linspace(0, num_frames, num_sampled_frames + 1)
        indices = np.minimum(np.floor((edges[:-1] + edges[1:]) / 2).astype(np.int64),
                             num_frames - 1).tolist()
    else:
        raise ValueError('final manifest split must be train/test')
    return [int(index * frames_each) for index in indices]


def inspect_manifest(path, expected_train_clips=EXPECTED_TRAIN_CLIPS,
                     expected_test_clips=EXPECTED_TEST_CLIPS):
    """Fail on partial TRAIN candidates, wrong TEST centers, or mixed protocols."""
    path = Path(path)
    inventory_path = Path(str(path) + '.clips.json')
    inventory = json.loads(inventory_path.read_text())
    if set(inventory) != {'train', 'test'}:
        raise ValueError('final inventory must contain only train/test')
    expected = {}
    for split in ('train', 'test'):
        expected[split] = {}
        for item in inventory[split]:
            clip = tuple(item['clip'])
            frames = candidate_frames(item['num_frames'], item['frames_each'], split)
            if (len(clip) != 2 or clip[0] not in PARTITIONS[split]
                    or clip in expected[split] or item['frames'] != frames):
                raise ValueError('inventory omits candidates or has wrong partition/TEST centers')
            expected[split][clip] = set(frames)
    if expected_train_clips is not None and len(expected['train']) != expected_train_clips:
        raise ValueError('final TRAIN+VAL clip count differs from complete retained dataset')
    if expected_test_clips is not None and len(expected['test']) != expected_test_clips:
        raise ValueError('final TEST clip count differs from complete retained dataset')
    observed = {'train': {}, 'test': {}}
    counts = Counter()
    for row in iter_manifest(path):
        split = row['split']
        if (row['schema_version'] != SCHEMA or row['dataset'] != 'CAFE'
                or row['protocol'] != PROTOCOL or split not in observed
                or row['video_id'] not in PARTITIONS[split]):
            raise ValueError('invalid final manifest partition/protocol: %s' % (frame_key(row),))
        clip = (row['video_id'], row['clip_id'])
        ids = observed[split].setdefault(clip, set())
        if row['source_frame_index'] in ids:
            raise ValueError('duplicate final manifest frame: %s' % (frame_key(row),))
        ids.add(row['source_frame_index'])
        counts[split] += 1
    if observed != expected:
        raise ValueError('manifest differs from complete TRAIN candidates/fixed TEST inventory')
    summary = json.loads(Path(str(path) + '.summary.json').read_text())
    clip_counts = {split: len(expected[split]) for split in expected}
    if (summary.get('protocol') != PROTOCOL or summary.get('val_mode') is not False
            or summary['clips'] != clip_counts or summary['frames'] != sum(counts.values())
            or summary['manifest_sha256'] != file_hash(path)
            or summary['inventory_sha256'] != file_hash(inventory_path)
            or summary['sampling'] != SAMPLING or summary.get('num_sampled_frames') != 5):
        raise ValueError('final manifest summary/hash/sampling mismatch')
    return dict(protocol=PROTOCOL, frames=sum(counts.values()), frames_by_split=dict(counts),
                clips=clip_counts, manifest_sha256=summary['manifest_sha256'],
                inventory_sha256=summary['inventory_sha256'])


def make_row(dataset, clip, fid, split):
    path = Path(dataset.image_path) / str(clip[0]) / str(clip[1]) / 'images' / ('frames_%d.jpg' % fid)

    def inspect_image():
        with Image.open(path) as image:
            if image.mode != 'RGB':
                raise ValueError('expected RGB image: %s' % path)
            width, height = image.size
        return width, height, file_hash(path)

    width, height, image_hash = retry_io(inspect_image)
    boxes, frame_valid = actor_boxes_for_frame(dataset, clip, fid)
    actors = []
    for actor_row in range(dataset.num_boxes):
        if not frame_valid[actor_row]:
            continue
        cx, cy, bw, bh = boxes[actor_row].tolist()
        box = [float(value) for value in np.asarray(
            [cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2]) * [width, height, width, height]]
        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('invalid actor box: %s' % path)
        actors.append(dict(actor_row=actor_row,
            actor_uid='cafe/%d/%d/row_%d' % (clip[0], clip[1], actor_row),
            actor_id_scope='video/clip/zero-based annotation row; track ID=row+1',
            source_box_xyxy=box, box_hash=digest(canonical(box)),
            box_source='given GT actor tracklets gt_tracks.pkl, matching B0; visible vs amodal semantics not independently documented'))
    return dict(schema_version=SCHEMA, dataset='CAFE', protocol=PROTOCOL, split=split,
        video_id=clip[0], clip_id=clip[1], source_frame_index=fid,
        source_image_key='%d/%d/images/frames_%d.jpg' % (clip[0], clip[1], fid),
        source_image_path=str(path), source_image_hash=image_hash,
        original_hw=[height, width], actors=actors,
        invalid_actor_rows=[index for index in range(dataset.num_boxes) if not frame_valid[index]],
        sampling=('all TRAIN+VAL candidates for CafeDataset._sample_random_segment_frames * framesEach'
                  if split == 'train' else 'real CafeDataset._sample_segment_centers * framesEach'))


def build_manifest(cli, report):
    from dataloader.dataloader import read_dataset
    output = Path(cli.manifest).resolve()
    if output.exists():
        report['manifest_inventory'] = inspect_manifest(output)
        report['status'] = 'complete'
        return
    config = json.loads((ROOT / 'configs/heatmap/b0.json').read_text())
    config.update(split='view', val_mode=False, data_path=str(Path(cli.data_root).resolve()),
                  tracks_path=str(Path(cli.tracks_path).resolve()), sam_mask_manifest='',
                  clip_manifest='', fixed_center_train=False, use_mae=False,
                  annotation_workers=cli.annotation_workers)
    datasets = retry_io(lambda: read_dataset(argparse.Namespace(**config)))
    output.parent.mkdir(parents=True, exist_ok=True)
    inventory = {'train': [], 'test': []}
    frames = actors = 0
    temporary = output.with_name(output.name + '.tmp')
    if temporary.exists():
        raise FileExistsError('unfinished manifest exists; preserve for inspection: ' + str(temporary))
    try:
        with open(temporary, 'wb') as stream:
            for split, dataset in zip(('train', 'test'), datasets):
                dataset.is_training = False
                if split == 'train' and len(dataset.frames) != EXPECTED_TRAIN_CLIPS:
                    raise ValueError('final TRAIN+VAL must retain exactly 7298 clips')
                if split == 'test' and len(dataset.frames) != EXPECTED_TEST_CLIPS:
                    raise ValueError('final TEST must retain exactly 2414 clips')
                for clip in sorted(dataset.frames):
                    if clip[0] not in PARTITIONS[split]:
                        raise ValueError('dataloader returned unexpected final view partition')
                    annotation = dataset.anns[clip]
                    num_frames, interval = int(annotation['num_frames']), int(annotation['interval'])
                    frame_ids = candidate_frames(num_frames, interval, split)
                    if split == 'test' and [fid for _, fid in dataset.select_frames(clip)] != frame_ids:
                        raise ValueError('current dataloader TEST centers differ from frozen manifest algorithm')
                    inventory[split].append(dict(clip=list(clip), frames=frame_ids,
                                                  num_frames=num_frames, frames_each=interval))
                    for fid in dict.fromkeys(frame_ids):
                        row = make_row(dataset, clip, fid, split)
                        stream.write(canonical(row) + b'\n')
                        frames += 1
                        actors += len(row['actors'])
                    if len(inventory[split]) % 100 == 0:
                        stream.flush()
                        report.update(stage='manifest_' + split, frames=frames,
                                      clips={key: len(value) for key, value in inventory.items()})
                        atomic_json(cli.report, report)
                        print('manifest', split, len(inventory[split]), 'clips', frames, 'frames', flush=True)
            stream.flush()
            os.fsync(stream.fileno())
        inventory_path = str(output) + '.clips.json'
        atomic_json(inventory_path, inventory)
        summary = dict(protocol=PROTOCOL, split='view', val_mode=False, frames=frames,
            actor_prompts=actors, clips={key: len(value) for key, value in inventory.items()},
            manifest_sha256=file_hash(temporary), inventory_sha256=file_hash(inventory_path),
            sampling=SAMPLING, num_sampled_frames=5,
            data_root=str(Path(cli.data_root).resolve()), tracks_sha256=file_hash(cli.tracks_path),
            purpose='SAM image/GT tracklet cache only; no TEST predictions, evaluation, or model selection')
        atomic_json(str(output) + '.summary.json', summary)
        os.replace(str(temporary), str(output))
    except Exception:
        # Preserve the incomplete file and progress report for NAS diagnostics.
        raise
    report['manifest_inventory'] = inspect_manifest(output)
    report['status'] = 'complete'


def isolated_sources(source_roots, target_root):
    target = Path(target_root).resolve()
    for root in source_roots:
        source = Path(root).resolve()
        if source == target or source in target.parents or target in source.parents:
            raise ValueError('final SAM cache must be isolated from every source cache')
    return target


def reuse_sources(cli, report, model):
    sources = []
    for manifest, root in zip(cli.source_manifest, cli.source_cache_root):
        print('index source manifest', manifest, flush=True)
        sources.append((Path(root).resolve(), source_index(manifest), file_hash(manifest)))
    counters = Counter()
    for position, row in enumerate(iter_manifest(cli.manifest), 1):
        _, target_marker = cache_paths(cli.cache_root, cli.cache_version, row)
        if target_marker.exists():
            validate_bundle(cli.cache_root, cli.cache_version, row, model)
            counters['existing'] += 1
        else:
            for source_root, index, source_hash in sources:
                fingerprint = index.get(frame_key(row))
                if fingerprint is None:
                    continue
                paths = cache_paths(source_root, cli.cache_version, row) + projected_paths(
                    source_root, cli.cache_version, row, (720, 1280), (23, 40))
                if not all(path.is_file() for path in paths):
                    continue
                counters.update(reuse_row(row, source_root, cli.cache_root,
                    cli.cache_version, model, fingerprint, source_hash))
                counters['source_' + str(source_root)] += 1
                break
            else:
                counters['missing'] += 1
        if position % 1000 == 0:
            report['reuse'] = dict(counters)
            atomic_json(cli.report, report)
            print('reuse', position, dict(counters), flush=True)
    report['reuse'] = dict(counters)


def verify_model_records(manifest, cache_root, cache_version, model):
    """Bind every generated/reused frame to the reviewed SAM model and prompt."""
    checked = 0
    for row in iter_manifest(manifest):
        _, marker = cache_paths(cache_root, cache_version, row)
        meta = json.loads(marker.read_text())
        if (meta['schema_version'] != SCHEMA or meta['cache_version'] != cache_version
                or meta['frame_fingerprint'] != frame_fingerprint(row)
                or canonical(meta['source']) != canonical(row)
                or len(meta['records']) != len(row['actors'])):
            raise ValueError('final raw marker identity mismatch: ' + str(marker))
        for actor, record in zip(row['actors'], meta['records']):
            if record['actor_uid'] != actor['actor_uid'] or record.get('model') != model:
                raise ValueError('final SAM model/prompt/actor identity mismatch: ' + str(marker))
        checked += 1
        if checked % 1000 == 0:
            print('verified model identities', checked, 'frames', flush=True)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['manifest', 'reuse', 'audit', 'verify'])
    parser.add_argument('--manifest', default=str(ROOT / '.local_data/sam/view_final_trainval_test_manifest.jsonl'))
    parser.add_argument('--cache-root', default=str(ROOT / '.local_data/sam/view_final_cache'))
    parser.add_argument('--source-manifest', action='append')
    parser.add_argument('--source-cache-root', action='append')
    parser.add_argument('--cache-version', default=CACHE_VERSION)
    parser.add_argument('--sam-repo', default=str(ROOT / '.local_deps/segment-anything'))
    parser.add_argument('--checkpoint', default=str(ROOT / '.local_data/sam/checkpoints/sam_vit_b_01ec64.pth'))
    parser.add_argument('--data-root', default='/nas/mzy/dataset/Cafe_Dataset/Cafe_Dataset/Dataset')
    parser.add_argument('--tracks-path', default=str(ROOT.parent.parent / '.local_data/cafe/gt_tracks.pkl'))
    parser.add_argument('--annotation-workers', type=int, default=4)
    parser.add_argument('--report', required=True)
    parser.add_argument('--verify-workers', type=int, default=4)
    cli = parser.parse_args()
    if cli.source_manifest is None and cli.source_cache_root is None:
        cli.source_manifest = [str(ROOT / '.local_data/sam/randomsegments_manifest.jsonl'),
                               str(ROOT / '.local_data/sam/view_randomsegments_manifest.jsonl')]
        cli.source_cache_root = [str(ROOT / '.local_data/sam/cache'), str(ROOT / '.local_data/sam/view_cache')]
    if (not cli.source_manifest or not cli.source_cache_root
            or len(cli.source_manifest) != len(cli.source_cache_root)):
        parser.error('provide a source-cache-root for every source-manifest')
    isolated_sources(cli.source_cache_root, cli.cache_root)
    Path(cli.report).parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(cli.manifest) + '.prepare.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(command=cli.command, status='running', protocol=PROTOCOL,
                  manifest=str(Path(cli.manifest).resolve()),
                  cache_root=str(Path(cli.cache_root).resolve()), cache_version=cli.cache_version)
    with open(lock_path, 'a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(cli.report, report)
        try:
            if cli.cache_version != CACHE_VERSION:
                raise ValueError('final protocol must preserve the reviewed SAM cache version')
            if cli.command == 'manifest':
                build_manifest(cli, report)
            else:
                report['manifest_inventory'] = inspect_manifest(cli.manifest)
                config = source_config(cli)
                if config['cache_version'] != cli.cache_version:
                    raise ValueError('requested SAM version differs from reviewed source/checkpoint/config')
                if cli.command == 'reuse':
                    reuse_sources(cli, report, version_payload(config))
                report['coverage'] = audit_cache(cli.manifest, cli.cache_root, cli.cache_version)
                if cli.command != 'reuse' and report['coverage'].get('missing', 0):
                    raise ValueError('final SAM cache incomplete; every TRAIN candidate and TEST center is required')
                if cli.command == 'verify':
                    report['model_identity_frames'] = verify_model_records(
                        cli.manifest, cli.cache_root, cli.cache_version, version_payload(config))
                    cli.shard, cli.shards, cli.projected = 0, 1, True
                    verify(cli)
                    report['full_verification'] = 'passed'
                report['status'] = ('reuse_complete_missing_generation'
                                    if report['coverage'].get('missing', 0) else 'complete')
        except Exception as error:
            report.update(status='failed', error=repr(error))
            atomic_json(cli.report, report)
            raise
        atomic_json(cli.report, report)
        print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
