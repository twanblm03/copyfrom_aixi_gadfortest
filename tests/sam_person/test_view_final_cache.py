"""CPU checks for complete final view sampling and isolated multi-source reuse."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from tools import prepare_view_final_sam_cache as final
from util.sam_person import (SCHEMA, atomic_json, cache_paths, canonical, digest,
    file_hash, frame_fingerprint, read_frame, write_frame, write_projected_frame)


class FinalViewCache(unittest.TestCase):
    def row(self, root, video, split='train', frame=0):
        image = Path(root) / ('image_%d_%d.jpg' % (video, frame))
        image.write_bytes(b'unchanged image bytes')
        return dict(schema_version=SCHEMA, dataset='CAFE', protocol=final.PROTOCOL,
            split=split, video_id=video, clip_id=2, source_frame_index=frame,
            source_image_key='%d/2/images/frames_%d.jpg' % (video, frame),
            source_image_path=str(image), source_image_hash=file_hash(image),
            original_hw=[7, 13], actors=[dict(actor_uid='%d/2/row0' % video,
            actor_row=0, source_box_xyxy=[1, 1, 7, 6])], invalid_actor_rows=[],
            sampling='partition sampling')

    def manifest(self, root, rows, inventory):
        path = Path(root) / 'final.jsonl'
        path.write_bytes(b''.join(canonical(row) + b'\n' for row in rows))
        atomic_json(str(path) + '.clips.json', inventory)
        atomic_json(str(path) + '.summary.json', dict(protocol=final.PROTOCOL,
            val_mode=False, clips={key: len(value) for key, value in inventory.items()},
            frames=len(rows), manifest_sha256=file_hash(path),
            inventory_sha256=file_hash(str(path) + '.clips.json'), sampling=final.SAMPLING,
            num_sampled_frames=5))
        return path

    def test_final_sampling_includes_old_val_candidates_and_test_centers(self):
        self.assertEqual(final.candidate_frames(6, 6, 'train'), [0, 6, 12, 18, 24, 30])
        self.assertEqual(final.candidate_frames(6, 6, 'test'), [0, 6, 18, 24, 30])
        self.assertEqual(final.candidate_frames(2, 6, 'test'), [0, 0, 6, 6, 6])
        self.assertIn(7, final.PARTITIONS['train'])
        self.assertNotIn(7, final.PARTITIONS['test'])
        with self.assertRaises(ValueError):
            final.candidate_frames(6, 6, 'val')

    def test_inspect_rejects_missing_random_candidate_even_if_both_files_omit_it(self):
        with tempfile.TemporaryDirectory() as root:
            rows = [self.row(root, 7, frame=frame) for frame in (0, 6, 12, 18, 24, 30)]
            inventory = dict(train=[dict(clip=[7, 2], num_frames=6, frames_each=6,
                                         frames=[0, 6, 12, 18, 24, 30])], test=[])
            path = self.manifest(root, rows, inventory)
            self.assertEqual(final.inspect_manifest(path, expected_train_clips=1, expected_test_clips=0)['frames'], 6)
            rows = [row for row in rows if row['source_frame_index'] != 12]
            inventory['train'][0]['frames'].remove(12)
            self.manifest(root, rows, inventory)
            with self.assertRaisesRegex(ValueError, 'omits candidates'):
                final.inspect_manifest(path, expected_train_clips=1, expected_test_clips=0)

    def test_inspect_rejects_wrong_test_centers_protocol_and_count(self):
        with tempfile.TemporaryDirectory() as root:
            rows = [self.row(root, 8, split='test', frame=frame) for frame in (0, 6, 18, 24, 30)]
            inventory = dict(train=[], test=[dict(clip=[8, 2], num_frames=6,
                frames_each=6, frames=[0, 6, 18, 24, 30])])
            path = self.manifest(root, rows, inventory)
            self.assertEqual(final.inspect_manifest(path, expected_train_clips=0, expected_test_clips=1)['frames'], 5)
            with self.assertRaisesRegex(ValueError, 'clip count'):
                final.inspect_manifest(path)
            rows[0]['protocol'] = 'view_train_val_b0_repro'
            self.manifest(root, rows, inventory)
            with self.assertRaisesRegex(ValueError, 'partition/protocol'):
                final.inspect_manifest(path, expected_train_clips=0, expected_test_clips=1)
            rows[0]['protocol'] = final.PROTOCOL
            inventory['test'][0]['frames'] = [0, 6, 12, 24, 30]
            self.manifest(root, rows, inventory)
            with self.assertRaisesRegex(ValueError, 'TEST centers'):
                final.inspect_manifest(path, expected_train_clips=0, expected_test_clips=1)

    def cache_fixture(self, root, name, row, model):
        cache = Path(root) / name
        raw = np.zeros((1, 7, 13), bool)
        raw[0, 1:6, 1:7] = True
        records = [dict(actor_uid=row['actors'][0]['actor_uid'], raw_mask_valid=True,
                        mask_checksum=digest(raw[0].tobytes()), model=model)]
        write_frame(cache, 'v1', row, raw, records)
        write_projected_frame(cache, 'v1', row, (720, 1280), (23, 40),
            np.ones((1, 23, 40), np.float32), [True], file_hash(cache_paths(cache, 'v1', row)[0]))
        manifest = Path(root) / (name + '.jsonl')
        manifest.write_bytes(canonical(row) + b'\n')
        return cache, manifest, raw

    def test_reuse_uses_both_sources_preserves_bytes_and_relabels_markers(self):
        with tempfile.TemporaryDirectory() as root:
            targets = [self.row(root, video) for video in (3, 7)]
            old_rows = copy.deepcopy(targets)
            old_rows[0].update(protocol='place_train_val_b0_repro', sampling='all TRAIN')
            old_rows[1].update(protocol='view_train_val_b0_repro', split='val', sampling='VAL centers')
            model = dict(reviewed='SAM v1', prompt='box')
            fixtures = [self.cache_fixture(root, name, row, model)
                        for name, row in zip(('place', 'view'), old_rows)]
            before = [{str(path.relative_to(cache)): file_hash(path)
                       for path in cache.rglob('*') if path.is_file()}
                      for cache, _, _ in fixtures]
            path = Path(root) / 'target.jsonl'
            path.write_bytes(b''.join(canonical(row) + b'\n' for row in targets))
            target = Path(root) / 'final_cache'
            cli = SimpleNamespace(source_manifest=[str(item[1]) for item in fixtures],
                source_cache_root=[str(item[0]) for item in fixtures], manifest=str(path),
                cache_root=str(target), cache_version='v1', report=str(Path(root) / 'report'))
            report = {}
            final.reuse_sources(cli, report, model)
            self.assertEqual(report['reuse']['reused'], 2)
            for index, (row, fixture) in enumerate(zip(targets, fixtures)):
                raw, marker = read_frame(target, 'v1', row, verify_source=True)
                np.testing.assert_array_equal(raw, fixture[2])
                self.assertEqual(marker['source']['protocol'], final.PROTOCOL)
                self.assertEqual(marker['source']['split'], 'train')
                self.assertEqual(marker['reuse_provenance']['source_frame_fingerprint'],
                                 frame_fingerprint(old_rows[index]))
                after = {str(path.relative_to(fixture[0])): file_hash(path)
                         for path in fixture[0].rglob('*') if path.is_file()}
                self.assertEqual(before[index], after)

    def test_reuse_rejects_corrupt_identity_and_uses_second_source_when_first_missing(self):
        with tempfile.TemporaryDirectory() as root:
            target_row = self.row(root, 7)
            old = copy.deepcopy(target_row)
            old.update(protocol='view_train_val_b0_repro', split='val')
            model = dict(reviewed='SAM v1')
            cache, manifest, _ = self.cache_fixture(root, 'view', old, model)
            absent = Path(root) / 'absent_cache'
            target_manifest = Path(root) / 'target.jsonl'
            target_manifest.write_bytes(canonical(target_row) + b'\n')
            cli = SimpleNamespace(source_manifest=[str(manifest), str(manifest)],
                source_cache_root=[str(absent), str(cache)], manifest=str(target_manifest),
                cache_root=str(Path(root) / 'target_cache'), cache_version='v1',
                report=str(Path(root) / 'report'))
            report = {}
            final.reuse_sources(cli, report, model)
            self.assertEqual(report['reuse']['reused'], 1)
            cli.cache_root = str(Path(root) / 'different_target')
            target_row['source_image_hash'] = 'changed'
            target_manifest.write_bytes(canonical(target_row) + b'\n')
            with self.assertRaisesRegex(ValueError, 'manifest data differ'):
                final.reuse_sources(cli, {}, model)

    def test_cache_roots_cannot_share_or_nest(self):
        for source, target in [('/tmp/cache', '/tmp/cache'), ('/tmp/cache', '/tmp/cache/final'),
                               ('/tmp/cache/place', '/tmp/cache')]:
            with self.assertRaisesRegex(ValueError, 'isolated'):
                final.isolated_sources([source], target)

    def test_verify_rejects_unreviewed_model_even_with_unchanged_array(self):
        with tempfile.TemporaryDirectory() as root:
            row = self.row(root, 7)
            model = dict(reviewed='SAM v1', prompt='box')
            cache, manifest, _ = self.cache_fixture(root, 'target', row, model)
            self.assertEqual(final.verify_model_records(manifest, cache, 'v1', model), 1)
            _, marker = cache_paths(cache, 'v1', row)
            metadata = json.loads(marker.read_text())
            metadata['records'][0]['model']['prompt'] = 'different prompt'
            atomic_json(marker, metadata)
            with self.assertRaisesRegex(ValueError, 'model/prompt/actor'):
                final.verify_model_records(manifest, cache, 'v1', model)

    def test_manifest_uses_trainval_test_loader_and_preserves_short_test_centers(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            track_file = root / 'tracks.pkl'
            track_file.write_bytes(b'identity of provided tracks')
            datasets = []
            for video, num_frames in ((7, 6), (8, 2)):
                clip = (video, 2)
                datasets.append(SimpleNamespace(frames=[clip], num_boxes=14,
                    anns={clip: dict(num_frames=num_frames, interval=6)},
                    select_frames=lambda clip: [(clip, fid) for fid in [0, 0, 6, 6, 6]]))
            cli = SimpleNamespace(manifest=str(root / 'manifest.jsonl'),
                data_root=str(root), tracks_path=str(track_file), annotation_workers=2,
                report=str(root / 'report.json'))
            report = {}
            inspect = final.inspect_manifest
            with mock.patch('dataloader.dataloader.read_dataset', return_value=datasets) as read:
                with mock.patch.object(final, 'EXPECTED_TRAIN_CLIPS', 1), \
                     mock.patch.object(final, 'EXPECTED_TEST_CLIPS', 1), \
                     mock.patch.object(final, 'make_row', side_effect=lambda dataset, clip, fid, split:
                                       self.row(root, clip[0], split, fid)), \
                     mock.patch.object(final, 'inspect_manifest', side_effect=lambda path:
                                       inspect(path, expected_train_clips=1, expected_test_clips=1)):
                    final.build_manifest(cli, report)
            args = read.call_args[0][0]
            self.assertEqual(args.split, 'view')
            self.assertFalse(args.val_mode)
            self.assertFalse(args.fixed_center_train)
            self.assertEqual(args.sam_mask_manifest, '')
            self.assertEqual(report['manifest_inventory']['frames_by_split'], {'train': 6, 'test': 2})
            self.assertEqual(report['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
