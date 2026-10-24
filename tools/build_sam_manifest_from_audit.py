"""Build a fixed-center SAM manifest from an audited split and local tracks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import pickle
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from tools.sam_person_full import actor_boxes_for_frame
from util.sam_person import SCHEMA, atomic_json, canonical, digest, file_hash


def clip_rows(item, data_root, tracks, num_boxes):
    split, record = item
    clip = tuple(record['clip'])
    frame_count = int(record['frames'])
    keys = sorted(tracks[clip])
    if len(keys) != frame_count or len(keys) < 2 or keys[0] != 0:
        raise ValueError('audited frame count and track indices disagree: %s' % (clip,))
    interval = keys[1] - keys[0]
    if interval < 1 or keys != [i * interval for i in range(frame_count)]:
        raise ValueError('track frame interval is not uniform: %s' % (clip,))
    edges = np.linspace(0, frame_count, 6)
    centers = np.floor((edges[:-1] + edges[1:]) / 2).astype(np.int64)
    frame_ids = [int(i * interval) for i in centers]
    actor_count = int(record['actors'])
    if actor_count > num_boxes:
        raise ValueError('actor count exceeds B0 padding: %s' % (clip,))
    proxy = SimpleNamespace(anns={clip: {'boxes': np.empty((actor_count, 4))}},
                            tracks=tracks, num_boxes=num_boxes)
    rows = []
    for fid in dict.fromkeys(frame_ids):
        path = Path(data_root) / 'cafe' / str(clip[0]) / str(clip[1]) / 'images' / ('frames_%d.jpg' % fid)
        with Image.open(path) as image:
            if image.mode != 'RGB':
                raise ValueError('expected RGB JPEG: %s' % path)
            w, h = image.size
        boxes, frame_valid = actor_boxes_for_frame(proxy, clip, fid)
        actors = []
        for actor_row in range(num_boxes):
            if not frame_valid[actor_row]:
                continue
            cx, cy, bw, bh = boxes[actor_row].tolist()
            box = [float(v) for v in np.asarray(
                [cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2]) * [w, h, w, h]]
            if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError('invalid actor box: %s frame %d' % (clip, fid))
            actors.append(dict(actor_row=actor_row,
                actor_uid='cafe/%d/%d/row_%d' % (clip[0], clip[1], actor_row),
                actor_id_scope='video/clip/zero-based annotation row; track ID=row+1',
                source_box_xyxy=box, box_hash=digest(canonical(box)),
                box_source='given GT actor tracklets gt_tracks.pkl, matching B0; visible vs amodal semantics not independently documented'))
        rows.append(dict(schema_version=SCHEMA, dataset='CAFE',
            protocol='place_train_val_b0_repro', split=split,
            video_id=clip[0], clip_id=clip[1], source_frame_index=fid,
            source_image_key='%d/%d/images/frames_%d.jpg' % (clip[0], clip[1], fid),
            source_image_path=str(path), source_image_hash=file_hash(path),
            original_hw=[h, w], actors=actors,
            invalid_actor_rows=[i for i in range(num_boxes) if not frame_valid[i]],
            sampling='real CafeDataset._sample_segment_centers * framesEach'))
    return frame_ids, rows, interval


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='audited TRAIN/VAL data_manifest.json')
    p.add_argument('--output', required=True)
    p.add_argument('--data-root', default='/nas/mzy/dataset/Cafe_Dataset/Cafe_Dataset/Dataset')
    p.add_argument('--tracks-path', default=str(ROOT.parent.parent / '.local_data/cafe/gt_tracks.pkl'))
    p.add_argument('--clip-manifest', default='', help='optional bounded clip list for exact comparison')
    p.add_argument('--workers', type=int, default=16)
    cli = p.parse_args()
    if cli.workers < 1:
        p.error('--workers must be positive')
    source = json.loads(Path(cli.source).read_text())
    if set(source) != {'train', 'val'}:
        raise ValueError('audited source must contain TRAIN and VAL only')
    selected = json.loads(Path(cli.clip_manifest).read_text()) if cli.clip_manifest else None
    if selected is not None:
        selected = {split:{tuple(row['clip']) for row in selected[split]}
                    for split in ('train','val')}
    entries = []
    for split in ('train', 'val'):
        entries.extend((split, record) for record in source[split]
                       if selected is None or tuple(record['clip']) in selected[split])
    if selected is not None:
        for split in ('train','val'):
            found={tuple(record['clip']) for row_split,record in entries if row_split==split}
            if found!=selected[split]:
                raise ValueError('bounded clips differ from audited split: '+split)
    with open(cli.tracks_path, 'rb') as stream:
        tracks = pickle.load(stream)
    output = Path(cli.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    clips = {'train': [], 'val': []}
    frame_hashes = {'train': set(), 'val': set()}
    intervals = {}
    frame_count = actor_count = 0
    try:
        with ThreadPoolExecutor(max_workers=cli.workers) as pool, open(temporary, 'wb') as stream:
            work = pool.map(lambda item: clip_rows(item, cli.data_root, tracks, 14), entries)
            for index, ((split, record), (frame_ids, rows, interval)) in enumerate(zip(entries, work), 1):
                clips[split].append(dict(clip=record['clip'], frames=frame_ids))
                intervals[interval] = intervals.get(interval, 0) + 1
                for row in rows:
                    stream.write(canonical(row) + b'\n')
                    frame_count += 1
                    actor_count += len(row['actors'])
                    frame_hashes[split].add(row['source_image_hash'])
                if index % 100 == 0:
                    print('manifest', index, '/', len(entries), 'clips,',
                          frame_count, 'unique frames', flush=True)
        os.replace(str(temporary), str(output))
    finally:
        if temporary.exists():
            temporary.unlink()
    overlap = frame_hashes['train'] & frame_hashes['val']
    if overlap:
        raise ValueError('TRAIN/VAL sampled image content overlap')
    atomic_json(str(output)+'.clips.json', clips)
    atomic_json(str(output)+'.summary.json', dict(
        frames=frame_count, actor_prompts=actor_count,
        clips={split:len(clips[split]) for split in clips},
        interval_counts=intervals, source_split_sha256=file_hash(cli.source),
        tracks_sha256=file_hash(cli.tracks_path), manifest_sha256=file_hash(output),
        selected_duplicate_hashes=len(overlap),
        annotation_hashes='in audited source; individual ann.json not rehashed here',
        box_contract='exactly compared to real CafeDataset on bounded 120-frame pilot'))
    print('manifest complete:', frame_count, 'frames,', actor_count, 'prompts', flush=True)


if __name__ == '__main__':
    main()
