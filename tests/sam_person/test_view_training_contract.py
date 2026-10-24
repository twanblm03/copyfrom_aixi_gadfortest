import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.validate_view_training_contract import (
    ContractError, EVALUATOR_SHA256, FIXED_ARGS, VARIANTS,
    main, validate_contract,
)


def matched_args():
    place, view = {}, {}
    for variant in VARIANTS:
        previous = dict(FIXED_ARGS, split='place', custom_model_dimension=256,
                        experiment_tag='place_' + variant,
                        checkpoint_root='place/' + variant, result_path='./outputs/',
                        heatmap_variant='h1_sup' if variant == 'h1_sup' else 'off',
                        heatmap_sigma_ratio=0.15 if variant == 'h1_sup' else 0.25,
                        train_mask_mode='sam_roi_blend' if variant == 'sam' else 'disabled',
                        sam_projected_cache=variant == 'sam',
                        sam_mask_manifest='place/manifest.jsonl' if variant == 'sam' else '',
                        sam_cache_root='place/cache' if variant == 'sam' else '',
                        sam_cache_version='raw_reviewed' if variant == 'sam' else '')
        planned = dict(previous, split='view', experiment_tag='view_' + variant,
                       checkpoint_root='view/' + variant)
        if variant == 'sam':
            planned.update(sam_mask_manifest='view/manifest.jsonl', sam_cache_root='view/cache')
        place[variant], view[variant] = previous, planned
    return place, view


class ViewTrainingContract(unittest.TestCase):
    def test_matched_split_change_passes_with_explicit_audit(self):
        place, view = matched_args()
        report = validate_contract(place, view, EVALUATOR_SHA256)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['shared_view_args']['batch'], 16)
        self.assertEqual(set(report['pairwise_place_to_view']['b0']['changes']),
                         {'split', 'experiment_tag', 'checkpoint_root'})

    def test_unknown_training_drift_or_args_metadata_is_rejected(self):
        for field, value in (('custom_model_dimension', 512), ('queue_phase', 'waiting')):
            with self.subTest(field=field):
                place, view = matched_args()
                view['b0'][field] = value
                with self.assertRaises(ContractError):
                    validate_contract(place, view, EVALUATOR_SHA256)

    def test_all_methods_cannot_silently_change_frozen_hyperparameters(self):
        for field, value in (('epochs', 10), ('test_batch', 1), ('parallel_transport', 'native'),
                             ('distributed', True), ('gradient_clipping', True),
                             ('backbone_lr_scale', 0.1), ('batch', True)):
            with self.subTest(field=field):
                place, view = matched_args()
                for collection in (place, view):
                    for args in collection.values():
                        args[field] = value
                with self.assertRaisesRegex(ContractError, 'fixed training protocol'):
                    validate_contract(place, view, EVALUATOR_SHA256)

    def test_shared_field_difference_cannot_hide_in_matching_place_reference(self):
        place, view = matched_args()
        place['sam']['custom_model_dimension'] = 128
        view['sam']['custom_model_dimension'] = 128
        with self.assertRaisesRegex(ContractError, 'common training configuration'):
            validate_contract(place, view, EVALUATOR_SHA256)

    def test_method_switches_and_sam_model_version_are_checked(self):
        for variant, field, value in (
                ('h1_sup', 'heatmap_sigma_ratio', 0.25),
                ('b0', 'heatmap_variant', 'h1_sup'),
                ('sam', 'train_mask_mode', 'sam_roi'),
                ('sam', 'sam_projected_cache', False),
                ('sam', 'sam_cache_version', 'different_sam_model')):
            with self.subTest(variant=variant, field=field):
                place, view = matched_args()
                view[variant][field] = value
                with self.assertRaises(ContractError):
                    validate_contract(place, view, EVALUATOR_SHA256)

    def test_sam_cache_isolation_uses_resolved_paths(self):
        place, view = matched_args()
        view['sam']['sam_cache_root'] = 'view/../place/cache'
        with self.assertRaisesRegex(ContractError, 'SAM paths must be isolated'):
            validate_contract(place, view, EVALUATOR_SHA256)

    def test_stale_evaluator_and_output_collisions_are_rejected(self):
        place, view = matched_args()
        with self.assertRaisesRegex(ContractError, 'evaluator SHA256 mismatch'):
            validate_contract(place, view, 'legacy_evaluator')
        for field in ('experiment_tag', 'checkpoint_root'):
            with self.subTest(field=field):
                candidate = copy.deepcopy(view)
                candidate['sam'][field] = candidate['b0'][field]
                with self.assertRaises(ContractError):
                    validate_contract(place, candidate, EVALUATOR_SHA256)

    def test_cli_writes_failure_audit_and_exits_before_any_training(self):
        place, view = matched_args()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'evaluation').mkdir()
            (root / 'evaluation/cafe_eval.py').write_text('legacy evaluator')
            runs = {}
            for variant in VARIANTS:
                previous, planned = variant + '.place.json', variant + '.view.json'
                (root / previous).write_text(json.dumps(place[variant]))
                (root / planned).write_text(json.dumps(view[variant]))
                runs[variant] = dict(reference_args=previous, resolved_args=planned)
            (root / 'plan.json').write_text(json.dumps(dict(runs=runs, queue_phase='waiting')))
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(['--root', str(root), '--plan', 'plan.json', '--output', 'audit.json'])
            self.assertEqual(code, 2)
            report = json.loads((root / 'audit.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertIn('evaluator SHA256 mismatch', report['error'])


if __name__ == '__main__':
    unittest.main()
