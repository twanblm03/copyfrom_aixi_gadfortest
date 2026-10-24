"""Random TRAIN frame sampling remains within usable SAM cache frames."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import torch

from dataloader.cafe import CafeDataset
from util.sam_person import MaskReader, cache_paths, projected_paths


ROOT = Path(__file__).resolve().parents[2]


class RandomFrameCache(unittest.TestCase):
    def test_random_segments_skip_missing_cache_files(self):
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(**json.loads((ROOT / 'configs/heatmap/b0.json').read_text()))
            args.sam_cache_root = temp
            args.sam_cache_version = 'v1'
            args.sam_projected_cache = True
            args.sam_feature_height = 23
            args.sam_feature_width = 40
            args.fixed_center_train = False
            clip = (1, 2)
            rows = []
            for index in range(10):
                row = dict(video_id=clip[0], clip_id=clip[1],
                           source_frame_index=index * 6, actors=[])
                rows.append(row)
                if index not in (0, 1, 2, 4, 6, 8):
                    continue
                for pair in (cache_paths(temp, 'v1', row),
                             projected_paths(temp, 'v1', row, (720, 1280), (23, 40))):
                    for path in pair:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.touch()
            manifest = Path(temp) / 'manifest.jsonl'
            manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            args.sam_mask_manifest = str(manifest)
            dataset = CafeDataset([clip], {clip: dict(num_frames=10, interval=6)},
                                  {}, temp, args, is_training=True)
            selected = {dataset.select_frames(clip)[0][1] for _ in range(40)}
            self.assertEqual(selected, {0, 6})
            for _ in range(10):
                self.assertTrue(all(fid // 6 in dataset.sam_available_indices[clip]
                                    for _, fid in dataset.select_frames(clip)))

            # A whole temporal segment without any cache frame is rejected
            # before training, rather than producing a later worker crash.
            projected_paths(temp, 'v1', rows[4], (720, 1280), (23, 40))[1].unlink()
            with self.assertRaisesRegex(ValueError, 'no usable TRAIN frame'):
                CafeDataset([clip], {clip: dict(num_frames=10, interval=6)},
                            {}, temp, args, is_training=True)

    def test_missing_train_cache_falls_back_to_box_roi(self):
        row = dict(video_id=1, clip_id=2, source_frame_index=0,
                   actors=[dict(actor_row=0, source_box_xyxy=[0, 0, 1, 1])])
        with tempfile.TemporaryDirectory() as temp:
            reader = MaskReader(dict(cache_root=temp, cache_version='v1',
                                     projected=True, allow_cache_fallback=True), [row])
            with warnings.catch_warnings(record=True) as seen:
                masks, valid = reader.clip(dict(vid=1, sid=2, fid=[0]),
                    torch.zeros(1, 1, 4), torch.ones(1, 1, dtype=torch.bool),
                    (720, 1280), (23, 40))
            self.assertTrue(torch.all(masks == 1))
            self.assertFalse(valid.any())
            self.assertEqual(len(seen), 1)

            strict_reader = MaskReader(dict(cache_root=temp, cache_version='v1',
                                            projected=True), [row])
            with self.assertRaises(FileNotFoundError):
                strict_reader.clip(dict(vid=1, sid=2, fid=[0]),
                    torch.zeros(1, 1, 4), torch.ones(1, 1, dtype=torch.bool),
                    (720, 1280), (23, 40))


if __name__ == '__main__':
    unittest.main()
