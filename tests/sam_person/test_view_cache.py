"""CPU contracts for isolated view cache reuse and complete candidate coverage."""
import argparse
import copy
import errno
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from tools.prepare_view_sam_cache import (audit_cache, inspect_manifest,
    install_array, reuse_row)
from tools.sam_person_full import build_manifest
from util.sam_person import (SCHEMA, atomic_json, cache_paths, canonical,
    digest, file_hash, frame_fingerprint, projected_paths, read_frame,
    read_projected_frame, write_frame, write_projected_frame)


class ViewCache(unittest.TestCase):
    def fixture(self, folder):
        folder = Path(folder)
        image = folder / 'image'
        image.write_bytes(b'unchanged source image')
        row = dict(schema_version=SCHEMA, dataset='CAFE',
                   protocol='place_train_val_b0_repro', split='train',
                   video_id=3, clip_id=2, source_frame_index=0,
                   source_image_key='3/2/images/frames_0.jpg',
                   source_image_path=str(image), source_image_hash=file_hash(image),
                   original_hw=[7, 13], actors=[dict(actor_uid='3/2/row0',
                   actor_row=0, source_box_xyxy=[1, 1, 7, 6])],
                   invalid_actor_rows=[], sampling='all TRAIN candidates')
        mask = np.zeros((1, 7, 13), bool)
        mask[0, 1:6, 1:7] = True
        model = dict(family='reviewed SAM', prompt='box', threshold=0.0)
        records = [dict(actor_uid=row['actors'][0]['actor_uid'],
                        mask_checksum=digest(mask[0].tobytes()),
                        raw_mask_valid=True, model=model)]
        source, target = folder / 'place_cache', folder / 'view_cache'
        write_frame(source, 'v1', row, mask, records)
        raw_path, _ = cache_paths(source, 'v1', row)
        write_projected_frame(source, 'v1', row, (720, 1280), (23, 40),
                              np.ones((1, 23, 40), np.float32), [True], file_hash(raw_path))
        view = copy.deepcopy(row)
        view.update(protocol='view_train_val_b0_repro', split='val', sampling='VAL centers')
        return source, target, row, view, model, mask

    def test_reuse_relabels_markers_preserves_arrays_and_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target, row, view, model, mask = self.fixture(folder)
            before = {str(p.relative_to(source)): file_hash(p) for p in source.rglob('*') if p.is_file()}
            result = reuse_row(view, source, target, 'v1', model,
                               frame_fingerprint(row), 'approved-source-manifest')
            self.assertEqual(result['reused'], 1)
            self.assertEqual(result['hardlinked_arrays'], 2)
            restored, meta = read_frame(target, 'v1', view, verify_source=True)
            np.testing.assert_array_equal(restored, mask)
            projected, _ = read_projected_frame(target, 'v1', view, (720, 1280), (23, 40))
            self.assertEqual(projected.shape, (1, 23, 40))
            self.assertEqual(meta['reuse_provenance']['source_frame_fingerprint'], frame_fingerprint(row))
            self.assertEqual(meta['source'], view)
            after = {str(p.relative_to(source)): file_hash(p) for p in source.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(reuse_row(view, source, target, 'v1', model,
                                      frame_fingerprint(row), 'approved-source-manifest')['existing'], 1)

    def test_reuse_rejects_changed_boxes_source_manifest_and_model(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target, row, view, model, _ = self.fixture(folder)
            changed = copy.deepcopy(view)
            changed['actors'][0]['source_box_xyxy'][0] = 2
            with self.assertRaisesRegex(ValueError, 'manifest data differ'):
                reuse_row(changed, source, target, 'v1', model, frame_fingerprint(row), 'hash')
            with self.assertRaisesRegex(ValueError, 'manifest data differ'):
                reuse_row(view, source, target, 'v1', model, 'wrong approved fingerprint', 'hash')
            with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
                reuse_row(view, source, target, 'v1', dict(model, threshold=0.5),
                          frame_fingerprint(row), 'hash')
            self.assertFalse(target.exists())

    def test_all_candidate_files_required_and_partial_reuse_resumable(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target, row, view, model, _ = self.fixture(folder)
            manifest = Path(folder) / 'view.jsonl'
            missing = copy.deepcopy(view)
            missing['source_frame_index'] = 6
            manifest.write_bytes(canonical(view) + b'\n' + canonical(missing) + b'\n')
            self.assertEqual(reuse_row(missing, source, target, 'v1', model, None, 'hash')['missing'], 1)
            reuse_row(view, source, target, 'v1', model, frame_fingerprint(row), 'hash')
            report = audit_cache(manifest, target, 'v1')
            self.assertEqual((report['present'], report['missing']), (1, 1))
            # Completion raw marker missing after an interrupted reuse: existing
            # arrays/projected marker can be safely reused and finalized again.
            cache_paths(target, 'v1', view)[1].unlink()
            result = reuse_row(view, source, target, 'v1', model, frame_fingerprint(row), 'hash')
            self.assertEqual(result['existing_array'], 2)
            projected_paths(target, 'v1', view, (720, 1280), (23, 40))[1].unlink()
            self.assertEqual(audit_cache(manifest, target, 'v1')['missing'], 2)

    def test_copy_fallback_and_existing_array_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            source, destination = Path(folder) / 'a', Path(folder) / 'b'
            source.write_bytes(b'exact bytes')
            with mock.patch('tools.prepare_view_sam_cache.os.link', side_effect=OSError(errno.EXDEV, 'cross-device')):
                self.assertEqual(install_array(source, destination), 'copied_arrays')
            self.assertEqual(source.read_bytes(), destination.read_bytes())
            destination.write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'destination array differs'):
                install_array(source, destination)

    def test_manifest_inventory_rejects_missing_frame_and_wrong_partition(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, _, view, _, _ = self.fixture(folder)
            path = Path(folder) / 'view.jsonl'
            path.write_bytes(canonical(view) + b'\n')
            atomic_json(str(path) + '.clips.json', {'train': [], 'val': [{'clip': [3, 2], 'frames': [0]}]})
            atomic_json(str(path) + '.summary.json', dict(clips={'train': 0, 'val': 1}, frames=1,
                        manifest_sha256=file_hash(path), sampling='all TRAIN frame candidates; fixed VAL segment centers'))
            self.assertEqual(inspect_manifest(path)['frames'], 1)
            atomic_json(str(path) + '.clips.json', {'train': [], 'val': [{'clip': [3, 2], 'frames': [0, 6]}]})
            with self.assertRaisesRegex(ValueError, 'frozen clip/frame inventory'):
                inspect_manifest(path)
            view['video_id'] = 5  # Official view TRAIN video, invalid as VAL.
            path.write_bytes(canonical(view) + b'\n')
            with self.assertRaisesRegex(ValueError, 'partition'):
                inspect_manifest(path)

    def test_manifest_default_and_view_select_real_partition(self):
        # Exercise configuration dispatch without reading the full dataset.
        class EmptyDataset:
            frames = []
        for split in ('place', 'view'):
            with tempfile.TemporaryDirectory() as folder:
                tracks = Path(folder) / 'tracks'
                tracks.write_bytes(b'tracks fingerprint')
                cli = argparse.Namespace(split=split, data_root=folder, tracks_path=str(tracks),
                    limit_train=0, limit_val=0, output=str(Path(folder) / 'manifest'), train_all_frames=True)
                if split == 'place':
                    del cli.split  # Existing callers retain the place default.
                with mock.patch('dataloader.dataloader.read_dataset', return_value=(EmptyDataset(), EmptyDataset())) as read:
                    build_manifest(cli)
                self.assertEqual(read.call_args[0][0].split, split)
                summary = json.loads(Path(cli.output + '.summary.json').read_text())
                self.assertEqual(summary['split'], split)
                self.assertEqual(summary['sampling'], 'all TRAIN frame candidates; fixed VAL segment centers')

    def test_view_manifest_records_all_train_candidates_and_val_centers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            datasets = []
            for video in (5, 7):
                clip = (video, 2)
                images = root / 'cafe' / str(video) / '2' / 'images'
                images.mkdir(parents=True)
                for index in range(6):
                    Image.new('RGB', (13, 7), color=(index, video, 0)).save(images / ('frames_%d.jpg' % (index * 6)))
                datasets.append(SimpleNamespace(frames=[clip], num_boxes=14,
                    anns={clip: dict(num_frames=6, interval=6, boxes=np.empty((1, 4)))},
                    tracks={clip: {index * 6: [] for index in range(6)}},
                    image_path=str(root / 'cafe'),
                    select_frames=lambda clip: [(clip, fid) for fid in (0, 6, 18, 24, 30)]))
            tracks = root / 'tracks'
            tracks.write_bytes(b'tracks')
            cli = argparse.Namespace(split='view', data_root=folder, tracks_path=str(tracks),
                limit_train=0, limit_val=0, output=str(root / 'manifest'), train_all_frames=True)
            with mock.patch('dataloader.dataloader.read_dataset', return_value=datasets):
                build_manifest(cli)
            report = inspect_manifest(cli.output)
            self.assertEqual(report['frames_by_split'], {'train': 6, 'val': 5})
            rows = [json.loads(line) for line in Path(cli.output).read_text().splitlines()]
            self.assertTrue(all(row['protocol'] == 'view_train_val_b0_repro' for row in rows))
            self.assertEqual({row['source_frame_index'] for row in rows if row['split'] == 'train'},
                             {0, 6, 12, 18, 24, 30})


if __name__ == '__main__':
    unittest.main()
