"""Versioned raw person masks. No SAM imports or label-dependent selection."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


SCHEMA = 'sam-person-raw-v1'


def checkpoint_state(checkpoint):
    state=checkpoint.get('state_dict',checkpoint.get('model',checkpoint))
    prefixed=[key.startswith('module.') for key in state]
    if any(prefixed) and not all(prefixed):
        raise ValueError('mixed checkpoint DataParallel prefixes')
    if prefixed and all(prefixed):
        return {key[7:]:value for key,value in state.items()},'strip uniform DataParallel module. prefix; tensor values unchanged'
    return state,'identity'


def canonical(value):
    def encode(item):
        if isinstance(item,np.ndarray):return item.tolist()
        if isinstance(item,np.generic):return item.item()
        if isinstance(item,Path):return str(item)
        raise TypeError(type(item).__name__)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False,default=encode).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.%d.tmp' % os.getpid())
    temporary.write_bytes(canonical(value) + b'\n')
    os.replace(str(temporary), str(path))


def frame_fingerprint(row):
    return digest(canonical({k: row[k] for k in (
        'schema_version', 'dataset', 'protocol', 'split', 'video_id', 'clip_id',
        'source_frame_index', 'source_image_key', 'source_image_hash', 'original_hw', 'actors')}))


def cache_paths(root, version, row):
    # IDs are scoped to clip rows, never raw annotation ID or file order.
    base = Path(root) / version / ('video_%02d' % row['video_id']) / (
        'clip_%04d_frame_%06d' % (row['clip_id'], row['source_frame_index']))
    return base.with_suffix('.npz'), base.with_suffix('.json')


def write_frame(root, version, row, masks, records):
    masks = np.asarray(masks, dtype=bool)
    shape = (len(row['actors']),) + tuple(row['original_hw'])
    if masks.shape != shape or len(records) != shape[0]:
        raise ValueError('raw mask shape/actor count mismatch')
    array_path, done_path = cache_paths(root, version, row)
    array_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = array_path.with_name(array_path.name + '.%d.tmp' % os.getpid())
    # C-order flatten, MSB-first np.packbits; exact shape in completion record.
    with open(temporary, 'wb') as stream:
        np.savez_compressed(stream, packed=np.packbits(masks.reshape(-1)))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(array_path))
    atomic_json(done_path, dict(schema_version=SCHEMA, cache_version=version,
        frame_fingerprint=frame_fingerprint(row), source=row,
        mask_encoding='numpy.packbits C-order uint8 MSB-first; truncate to product(mask_shape)',
        mask_shape=list(shape), array_sha256=file_hash(array_path), records=records))


def read_frame(root, version, row, verify_source=False):
    array_path, done_path = cache_paths(root, version, row)
    if not done_path.is_file() or not array_path.is_file():
        raise FileNotFoundError('incomplete/missing raw mask cache: %s' % done_path)
    meta = json.loads(done_path.read_text())
    if (meta['schema_version'] != SCHEMA or meta['cache_version'] != version or
            meta['frame_fingerprint'] != frame_fingerprint(row)):
        raise ValueError('stale cache model/source/box identity: %s' % done_path)
    if verify_source and file_hash(row['source_image_path']) != row['source_image_hash']:
        raise ValueError('source image changed')
    if file_hash(array_path) != meta['array_sha256']:
        raise ValueError('corrupt mask array')
    shape = (len(row['actors']),) + tuple(row['original_hw'])
    if list(shape) != meta['mask_shape'] or len(meta['records']) != shape[0]:
        raise ValueError('cache dimensions/identities changed')
    with np.load(array_path, allow_pickle=False) as data:
        unpacked = np.unpackbits(data['packed'])
    if len(unpacked) < int(np.prod(shape)):
        raise ValueError('truncated mask bits')
    masks = unpacked[:int(np.prod(shape))].reshape(shape).astype(bool)
    for actor, mask, record in zip(row['actors'], masks, meta['records']):
        if record['actor_uid'] != actor['actor_uid'] or record['mask_checksum'] != digest(mask.tobytes()):
            raise ValueError('actor identity/mask checksum mismatch')
    return masks, meta


def transform_geometry(image, boxes, masks, resize_hw, crop_xyxy=None, flip=False):
    """One shared resize -> optional crop -> flip. Masks remain discrete here.

    Boxes are absolute continuous XYXY with half-open image boundaries.
    The actual B0 adapter uses resize only; crop/flip are tested for reuse.
    """
    h, w = image.shape[:2]
    rh, rw = resize_hw
    boxes = np.asarray(boxes, dtype=np.float32).copy() * np.array([rw/w, rh/h, rw/w, rh/h])
    image = np.asarray(Image.fromarray(image).resize((rw, rh), Image.BILINEAR))
    masks = np.stack([np.asarray(Image.fromarray(m.astype(np.uint8)).resize((rw, rh), Image.NEAREST))
                      for m in masks]) if len(masks) else np.zeros((0, rh, rw), np.uint8)
    if crop_xyxy is not None:
        x1, y1, x2, y2 = crop_xyxy
        image, masks = image[y1:y2, x1:x2], masks[:, y1:y2, x1:x2]
        boxes -= np.array([x1, y1, x1, y1])
    if flip:
        image, masks = image[:, ::-1].copy(), masks[:, :, ::-1].copy()
        left = boxes[:, 0].copy()
        boxes[:, 0], boxes[:, 2] = image.shape[1] - boxes[:, 2], image.shape[1] - left
    return image, boxes, masks.astype(bool)


def feature_masks(masks, image_hw, feature_hw):
    import torch
    import torch.nn.functional as F
    if not len(masks):
        return torch.zeros((0,) + tuple(feature_hw))
    # Match the B0 image resize; then project continuous area coverage. Do not
    # re-threshold after the feature projection (small people retain weight).
    resized = np.stack([np.asarray(Image.fromarray(m.astype(np.uint8)).resize(
        (image_hw[1], image_hw[0]), Image.NEAREST)).copy() for m in masks])
    return F.interpolate(torch.from_numpy(resized).float()[:, None], size=feature_hw,
                         mode='area')[:, 0]


class MaskReader:
    """Strict manifest lookup; source fingerprint verified at preflight."""
    def __init__(self, config, manifest):
        self.config = config
        self.rows = {}
        for row in manifest:
            key = (row['video_id'], row['clip_id'], row['source_frame_index'])
            if key in self.rows:
                raise ValueError('duplicate manifest identity')
            self.rows[key] = row

    def preflight(self):
        for row in self.rows.values():
            read_frame(self.config['cache_root'], self.config['cache_version'], row, verify_source=True)

    def clip(self, info, boxes, valid, image_hw, feature_hw):
        import torch
        t, n = boxes.shape[:2]
        grid = torch.ones((t, n) + tuple(feature_hw))
        raw_valid = torch.zeros(t, n, dtype=torch.bool)
        for index, fid in enumerate(info['fid']):
            row = self.rows[(info['vid'], info['sid'], fid)]
            masks, meta = read_frame(self.config['cache_root'], self.config['cache_version'], row)
            expected = {i for i in range(n) if bool(valid[index, i])}
            if expected != {a['actor_row'] for a in row['actors']}:
                raise ValueError('dataloader/manifest actor validity mismatch')
            projected = feature_masks(masks, image_hw, feature_hw)
            for actor, mask, record in zip(row['actors'], projected, meta['records']):
                i = actor['actor_row']
                cx, cy, bw, bh = boxes[index, i].tolist()
                hw = row['original_hw']
                xyxy = np.array([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2]) * [hw[1], hw[0], hw[1], hw[0]]
                if not np.allclose(xyxy, actor['source_box_xyxy'], atol=1e-3, rtol=1e-5):
                    raise ValueError('loader/source box mismatch')
                grid[index, i] = mask
                raw_valid[index, i] = record['raw_mask_valid']
        return grid, raw_valid
