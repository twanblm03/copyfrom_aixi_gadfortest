"""Build and fill a resumable SAM cache for full Café TRAIN/VAL."""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from util.sam_person import (SCHEMA, atomic_json, cache_paths, canonical, digest,
                             feature_masks, file_hash, projected_paths,
                             read_frame, read_projected_frame, write_frame,
                             write_projected_frame)
from tools.sam_person_masks import Provider, version, version_payload


SAM_COMMIT = 'dca509fe793f601edb92606367a655c15ac00fdf'
SAM_SOURCE_HASH = '79c08dc4649b51b257382e0fb7647fe9ab99b046bc84623ffd445ff383598a68'
SAM_CHECKPOINT_HASH = 'ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912'
SAM_CONFIG_HASH = '77bc4faf728118625ddd184dc4e6375bd1b5c636fc66fb429f9fc5276f331cc5'


def source_config(cli):
    sam_repo = Path(cli.sam_repo).resolve()
    checkpoint = Path(cli.checkpoint).resolve()
    source = {str(p.relative_to(sam_repo)): file_hash(p)
              for p in sorted((sam_repo / 'segment_anything').rglob('*.py'))}
    if digest(canonical(source)) != SAM_SOURCE_HASH:
        raise ValueError('SAM source differs from reviewed revision')
    if file_hash(checkpoint) != SAM_CHECKPOINT_HASH:
        raise ValueError('SAM checkpoint differs from reviewed ViT-B weight')
    c = dict(schema_version=SCHEMA, sam_family='sam_v1_vit_b', sam_commit=SAM_COMMIT,
             sam_source_hash=SAM_SOURCE_HASH, checkpoint_sha256=SAM_CHECKPOINT_HASH,
             sam_config_hash=SAM_CONFIG_HASH, prompt_policy='box_only',
             multimask_output=False, precision='fp32', mask_threshold=0.0,
             postprocess_options=dict(raw_binary=True, clip_to_box=False,
                                      largest_component=False, nms=False,
                                      custom_cuda_extension='not required by original SAM'),
             sam_repo=str(sam_repo), checkpoint=str(checkpoint),
             cache_root=str(Path(cli.cache_root).resolve()), device='cuda:0')
    c['cache_version'] = version(c)
    return c


def iter_manifest(path):
    with open(path) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def actor_boxes_for_frame(dataset, clip, fid):
    """Reproduce CafeDataset.load_samples track validity without decoding JPEGs."""
    n = dataset.anns[clip]['boxes'].shape[0]
    boxes = np.ones((n, 4), dtype=np.float64)
    valid = np.zeros(dataset.num_boxes, dtype=bool)
    for track in dataset.tracks[clip][fid]:
        actor_id, x1, y1, x2, y2 = track
        if x1 < 0.0 and y2 < 0.0:
            x1, y1, x2, y2 = 0.0, 0.0, 1e-8, 1e-8
        cx, cy = (x1+x2)/2, (y1+y2)/2
        bw, bh = x2-x1, y2-y1
        if actor_id <= n:
            boxes[int(actor_id-1)] = [cx, cy, bw, bh]
            if actor_id >= 1:
                valid[int(actor_id-1)] = bool(np.isfinite([x1,y1,x2,y2]).all()
                    and bw > 1e-7 and bh > 1e-7 and x2 > 0 and y2 > 0
                    and x1 < 1 and y1 < 1)
    if n < dataset.num_boxes:
        boxes = np.vstack([boxes, np.zeros((dataset.num_boxes-n, 4))])
    return boxes.astype(np.float32), valid


def build_manifest(cli):
    import pickle
    from dataloader.dataloader import (read_dataset, TRAIN_CAFE_P, VAL_CAFE_P,
                                       TRAIN_CAFE_V, VAL_CAFE_V)
    from dataloader.cafe import CafeDataset, cafe_read_annotations
    config = json.loads((ROOT / 'configs/heatmap/b0.json').read_text())
    split_name = getattr(cli, 'split', 'place')
    train_videos, val_videos = ((TRAIN_CAFE_P, VAL_CAFE_P) if split_name == 'place'
                                else (TRAIN_CAFE_V, VAL_CAFE_V))
    config.update(split=split_name, data_path=str(Path(cli.data_root).resolve()),
                  tracks_path=str(Path(cli.tracks_path).resolve()),
                  sam_mask_manifest='', use_mae=False,
                  annotation_workers=16)
    args = argparse.Namespace(**config)
    if cli.limit_train or cli.limit_val:
        with open(cli.tracks_path, 'rb') as stream:
            tracks = pickle.load(stream)
        datasets = []
        for videos, limit in [(train_videos, cli.limit_train),
                              (val_videos, cli.limit_val)]:
            available = sorted((int(video), int(clip)) for video in videos
                               for clip in os.listdir(Path(args.data_path) / 'cafe' / video)
                               if clip.isdigit())
            if limit:
                available = [available[i] for i in np.linspace(
                    0, len(available) - 1, min(limit, len(available)), dtype=int)]
            annotations = cafe_read_annotations(
                str(Path(args.data_path) / 'cafe'), videos, args.num_class,
                clip_ids=set(available))
            datasets.append(CafeDataset(sorted(annotations), annotations, tracks,
                str(Path(args.data_path) / 'cafe'), args, is_training=False))
        train, val = datasets
    else:
        train, val = read_dataset(args)
    output = Path(cli.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    clips = {'train': [], 'val': []}
    seen = set()
    frames = actors_total = 0
    try:
        with open(temporary, 'wb') as stream:
            for split, dataset, limit in [('train', train, cli.limit_train),
                                          ('val', val, cli.limit_val)]:
                dataset.is_training = False
                selected = sorted(dataset.frames)
                for clip in selected:
                    if split == 'train' and cli.train_all_frames:
                        annotation = dataset.anns[clip]
                        sampled = [(clip, int(index * annotation['interval']))
                                   for index in range(int(annotation['num_frames']))]
                    else:
                        sampled = dataset.select_frames(clip)
                    frame_ids = [fid for _, fid in sampled]
                    clips[split].append(dict(clip=list(clip), frames=frame_ids))
                    for fid in frame_ids:
                        path = Path(dataset.image_path) / str(clip[0]) / str(clip[1]) / 'images' / ('frames_%d.jpg' % fid)
                        key = '%d/%d/images/frames_%d.jpg' % (clip[0], clip[1], fid)
                        if key in seen:
                            continue
                        seen.add(key)
                        with Image.open(path) as image:
                            if image.mode != 'RGB':
                                raise ValueError('expected RGB image: %s' % path)
                            w, h = image.size
                        boxes, frame_valid = actor_boxes_for_frame(dataset, clip, fid)
                        selected_actors = []
                        for actor_row in range(args.num_boxes):
                            if not frame_valid[actor_row]:
                                continue
                            cx, cy, bw, bh = boxes[actor_row].tolist()
                            box = [float(v) for v in np.asarray(
                                [cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2]) * [w, h, w, h]]
                            if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                                raise ValueError('invalid actor box: %s' % key)
                            selected_actors.append(dict(actor_row=actor_row,
                                actor_uid='cafe/%d/%d/row_%d' % (clip[0], clip[1], actor_row),
                                actor_id_scope='video/clip/zero-based annotation row; track ID=row+1',
                                source_box_xyxy=box, box_hash=digest(canonical(box)),
                                box_source='given GT actor tracklets gt_tracks.pkl, matching B0; visible vs amodal semantics not independently documented'))
                        row = dict(schema_version=SCHEMA, dataset='CAFE',
                            protocol=split_name + '_train_val_b0_repro', split=split,
                            video_id=clip[0], clip_id=clip[1], source_frame_index=fid,
                            source_image_key=key, source_image_path=str(path),
                            source_image_hash=file_hash(path), original_hw=[h, w],
                            actors=selected_actors,
                            invalid_actor_rows=[i for i in range(args.num_boxes)
                                                if not frame_valid[i]],
                            sampling=('all TRAIN candidates for CafeDataset._sample_random_segment_frames * framesEach'
                                      if split == 'train' and cli.train_all_frames else
                                      'real CafeDataset._sample_segment_centers * framesEach'))
                        stream.write(canonical(row) + b'\n')
                        frames += 1
                        actors_total += len(selected_actors)
                    if len(clips[split]) % 100 == 0:
                        print('manifest', split, len(clips[split]), 'clips', frames, 'unique frames', flush=True)
        os.replace(str(temporary), str(output))
    finally:
        if temporary.exists():
            temporary.unlink()
    atomic_json(str(output) + '.clips.json', clips)
    atomic_json(str(output) + '.summary.json', dict(
        split=split_name,
        frames=frames, actor_prompts=actors_total, clips={s:len(v) for s,v in clips.items()},
        manifest_sha256=file_hash(output),
        sampling=('all TRAIN frame candidates; fixed VAL segment centers'
                  if cli.train_all_frames else 'fixed segment centers'),
        data_root=str(Path(cli.data_root).resolve()), tracks_sha256=file_hash(cli.tracks_path)))
    print('manifest complete:', frames, 'frames,', actors_total, 'prompts', flush=True)


def generate(cli):
    c = source_config(cli)
    print('SAM cache version', c['cache_version'], 'shard', cli.shard, '/', cli.shards, flush=True)
    provider = Provider(c)
    done = failed = skipped = 0
    for index, row in enumerate(iter_manifest(cli.manifest)):
        if index % cli.shards != cli.shard:
            continue
        _, marker = cache_paths(c['cache_root'], c['cache_version'], row)
        if marker.exists():
            read_frame(c['cache_root'], c['cache_version'], row, verify_source=False)
            skipped += 1
            continue
        with Image.open(row['source_image_path']) as image:
            rgb = np.asarray(image.convert('RGB'))
        if file_hash(row['source_image_path']) != row['source_image_hash']:
            raise ValueError('source JPEG changed: ' + row['source_image_key'])
        provider.encode(rgb)
        masks, records = [], []
        for actor in row['actors']:
            try:
                mask, score, seconds = provider.predict(actor['source_box_xyxy'], row['original_hw'])
                valid = bool(mask.any())
                status = 'success' if valid else 'empty'
                failure = None if valid else 'raw mask empty'
            except (RuntimeError, ValueError) as error:
                mask = np.zeros(row['original_hw'], bool)
                score, seconds, valid = None, None, False
                status, failure = 'failed', repr(error)
                failed += 1
            masks.append(mask)
            records.append(dict(actor_uid=actor['actor_uid'], generation_status=status,
                raw_mask_valid=valid, predicted_quality=score, selected_candidate=0,
                failure_reason=failure, quality_flags=[], decoder_seconds=seconds,
                mask_checksum=digest(mask.tobytes()), model=version_payload(c)))
        packed = np.stack(masks) if masks else np.zeros((0,) + tuple(row['original_hw']), bool)
        write_frame(c['cache_root'], c['cache_version'], row, packed, records)
        done += 1
        if done % 100 == 0:
            print('generated shard', cli.shard, done, 'frames; failures', failed,
                  'skipped', skipped, flush=True)
    print('generation complete: shard', cli.shard, 'done', done,
          'skipped', skipped, 'failures', failed, flush=True)


def project(cli):
    c = source_config(cli)
    image_hw = (720, 1280)
    feature_hw = (23, 40)
    done = skipped = 0
    for index, row in enumerate(iter_manifest(cli.manifest)):
        if index % cli.shards != cli.shard:
            continue
        _, marker = projected_paths(c['cache_root'], c['cache_version'], row,
                                    image_hw, feature_hw)
        if marker.exists():
            read_projected_frame(c['cache_root'], c['cache_version'], row,
                                 image_hw, feature_hw)
            skipped += 1
            continue
        masks, meta = read_frame(c['cache_root'], c['cache_version'], row)
        weights = feature_masks(masks, image_hw, feature_hw).numpy()
        write_projected_frame(c['cache_root'], c['cache_version'], row,
                              image_hw, feature_hw, weights,
                              [r['raw_mask_valid'] for r in meta['records']],
                              meta['array_sha256'])
        done += 1
        if done % 500 == 0:
            print('projected shard', cli.shard, done, 'frames', flush=True)
    print('projection complete: shard', cli.shard, 'done', done,
          'skipped', skipped, flush=True)


def verify_shard(task):
    c, manifest, shard, shards, projected = task
    frames = actors = failures = 0
    for index, row in enumerate(iter_manifest(manifest)):
        if index % shards != shard:
            continue
        masks, meta = read_frame(c['cache_root'], c['cache_version'], row, verify_source=True)
        if projected:
            _, derived = read_projected_frame(c['cache_root'], c['cache_version'],
                                               row, (720, 1280), (23, 40))
            if (derived['raw_array_sha256'] != meta['array_sha256'] or
                    derived['raw_valid'] != [bool(r['raw_mask_valid']) for r in meta['records']]):
                raise ValueError('projected/raw cache mismatch: '+row['source_image_key'])
        frames += 1
        actors += len(masks)
        failures += sum(not item['raw_mask_valid'] for item in meta['records'])
        if frames % 1000 == 0:
            print('verified shard', shard, frames, 'frames', flush=True)
    return frames, actors, failures


def verify(cli):
    c = source_config(cli)
    if cli.shards != 1:
        jobs = [(c, cli.manifest, cli.shard, cli.shards, cli.projected)]
    else:
        jobs = [(c, cli.manifest, shard, cli.verify_workers, cli.projected)
                for shard in range(cli.verify_workers)]
    if len(jobs) == 1:
        results = [verify_shard(jobs[0])]
    else:
        with multiprocessing.get_context('fork').Pool(len(jobs)) as pool:
            results = pool.map(verify_shard, jobs)
    frames, actors, failures = map(sum, zip(*results))
    print('verified complete:', frames, 'frames,', actors, 'actors,', failures,
          'failed masks; workers', len(jobs), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['manifest', 'generate', 'project', 'verify'])
    p.add_argument('--manifest', default='')
    p.add_argument('--output', default='')
    p.add_argument('--split', choices=['place', 'view'], default='place',
                   help='official TRAIN/VAL partition used by manifest generation')
    p.add_argument('--data-root', default='/nas/mzy/dataset/Cafe_Dataset/Cafe_Dataset/Dataset')
    p.add_argument('--tracks-path', default=str(ROOT.parent.parent / '.local_data/cafe/gt_tracks.pkl'))
    p.add_argument('--cache-root', default=str(ROOT / '.local_data/sam/cache'))
    p.add_argument('--sam-repo', default=str(ROOT / '.local_deps/segment-anything'))
    p.add_argument('--checkpoint', default=str(ROOT / '.local_data/sam/checkpoints/sam_vit_b_01ec64.pth'))
    p.add_argument('--limit-train', type=int, default=0)
    p.add_argument('--limit-val', type=int, default=0)
    p.add_argument('--train-all-frames', action='store_true',
                   help='cache every TRAIN frame reachable by random segment sampling; keep VAL centers')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--projected', action='store_true', help='also verify feature-grid cache')
    p.add_argument('--verify-workers', type=int, default=4)
    cli = p.parse_args()
    if cli.command == 'manifest':
        if not cli.output:
            p.error('--output is required for manifest')
        build_manifest(cli)
    else:
        if not cli.manifest:
            p.error('--manifest is required')
        if not 0 <= cli.shard < cli.shards:
            p.error('invalid shard index')
        if cli.verify_workers < 1:
            p.error('--verify-workers must be positive')
        {'generate': generate, 'project': project, 'verify': verify}[cli.command](cli)


if __name__ == '__main__':
    main()
