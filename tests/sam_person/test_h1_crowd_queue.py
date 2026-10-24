"""CPU contracts for the successor's causal comparison and dependency gates."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('test_crowd_queue', str(ROOT / 'tools/queue_h1_crowd.py'))
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


def arguments():
    return dict(split='place', random_seed=1, heatmap_variant='h1_sup',
                train_mask_mode='disabled', epochs=30, test_freq=1, batch=16,
                test_batch=4, fixed_center_train=False, selection_metric='group_mAP_1.0',
                protocol='development', eval_split='val', device='0,1,2,3',
                distributed=False, parallel_transport='host', gradient_clipping=False,
                heatmap_sigma_ratio=.15, heatmap_sigma_min=.5, heatmap_sigma_max=4.,
                experiment_tag='old', checkpoint_root='/old', result_path='./outputs/', lr=1e-5)


def complete():
    return (dict(stage='complete', completed_runs=['b0', 'h1_sup', 'sam']),
            dict.fromkeys(('b0', 'h1_sup', 'sam'), 'training_complete date'),
            {name: dict(status='passed', train_epochs=30, val_epochs=30, test_epochs=0)
             for name in ('b0', 'h1_sup', 'sam')})


class CrowdArguments(unittest.TestCase):
    def test_geometry_must_match_train_target_and_anchor(self):
        report = dict(status='passed', split='place', partition='TRAIN only', cap_ratio=.35,
                      sigma_min=.5, grid=[23, 40], reference_args_sha256='anchor', clips=2,
                      counts=dict(valid_actor_frames=10, shrunk_actor_frames=5, group_frames=4),
                      changed_fraction=.5, mean_target_mass_before=10., mean_target_mass_after=8.,
                      mean_nonmember_center_response_before=.1,
                      mean_nonmember_center_response_after=.05)
        self.assertEqual(queue.validate_geometry(report, 'anchor')['status'], 'passed')
        for field, value in [('split', 'view'), ('partition', 'VAL'), ('cap_ratio', .2),
                             ('grid', [46, 80]), ('reference_args_sha256', 'wrong'),
                             ('changed_fraction', 0.), ('mean_target_mass_after', float('inf'))]:
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                queue.validate_geometry(dict(report, **{field: value}), 'anchor')
        no_change = copy.deepcopy(report)
        no_change['counts']['shrunk_actor_frames'] = 0
        with self.assertRaises(RuntimeError):
            queue.validate_geometry(no_change, 'anchor')

    def test_only_target_definition_and_two_outputs_differ(self):
        anchor = arguments()
        planned = dict(anchor, **queue.NEW_ARGS)
        planned.update(experiment_tag='crowd', checkpoint_root='/crowd')
        self.assertEqual(queue.validate_args(anchor, planned)['status'], 'passed')
        for field, value in [('lr', 1e-4), ('result_path', '/new'), ('random_seed', 2),
                             ('heatmap_crowd_cap_ratio', .2), ('unknown', True),
                             ('heatmap_target_mode', 'box')]:
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                queue.validate_args(anchor, dict(planned, **{field: value}))

    def test_schema_and_json_boolean_types_are_preserved(self):
        anchor = arguments()
        planned = dict(anchor, **queue.NEW_ARGS)
        planned.update(experiment_tag='crowd', checkpoint_root='/crowd')
        del planned['lr']
        with self.assertRaises(RuntimeError):
            queue.validate_args(anchor, planned)
        planned = dict(anchor, **queue.NEW_ARGS)
        planned.update(experiment_tag='crowd', checkpoint_root='/crowd', distributed=0)
        with self.assertRaises(RuntimeError):
            queue.validate_args(anchor, planned)

    def test_full_baseline_and_best_selection_required(self):
        args = arguments()
        history = dict(train=[dict(epoch=n, loss=1., skipped_nonfinite_steps=0)
                              for n in range(1, 31)],
                       val=[dict(epoch=n, loss=1., group_mAP_1_0=0.,
                                 **{'group_mAP_1.0': float(n), 'group_mAP_0.5': 40.,
                                    'outlier_mIoU': 70.}) for n in range(1, 31)])
        summary = dict(args=args, history=history,
                       best={'group_mAP_1.0': dict(epoch=30, value=30.)})
        with tempfile.TemporaryDirectory() as folder:
            for name in ('best_val.pth', 'epoch30.pth'):
                (Path(folder) / name).touch()
            self.assertEqual(queue.validate_baseline(args, summary, folder)['status'], 'passed')
            observed = copy.deepcopy(summary)
            observed['args']['peak_cuda_memory_mb'] = [[9603., 15578.]]
            self.assertEqual(queue.validate_baseline(args, observed, folder)['status'], 'passed')
            for mutate in ('best', 'partial', 'skip', 'test', 'nonfinite'):
                changed = copy.deepcopy(summary)
                if mutate == 'best': changed['best']['group_mAP_1.0']['epoch'] = 29
                if mutate == 'partial': changed['history']['val'].pop()
                if mutate == 'skip': changed['history']['train'][0]['skipped_nonfinite_steps'] = 1
                if mutate == 'test': changed['history']['test'] = [dict(epoch=30)]
                if mutate == 'nonfinite': changed['history']['val'][0]['loss'] = float('nan')
                with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                    queue.validate_baseline(args, changed, folder)


class CrowdDependencyGates(unittest.TestCase):
    def test_ready_requires_both_dependencies_and_process_exit(self):
        state, statuses, audits = complete()
        for place_alive, view_locked, view_alive, expected in (
                (False, False, False, 'ready'),
                (True, False, False, 'waiting_place_sam_exit'),
                (False, True, False, 'waiting_view_queue_exit'),
                (False, False, True, 'waiting_view_queue_exit')):
            with self.subTest(expected=expected):
                self.assertEqual(queue.dependency_stage(state, statuses, audits,
                    'training_complete date', place_alive, view_locked, view_alive), expected)

    def test_running_view_and_place_are_waited_for(self):
        state, statuses, audits = complete()
        self.assertEqual(queue.dependency_stage(dict(stage='training_h1_sup'), {}, {},
            'training_started date', True, True), 'waiting_existing_view_queue')
        self.assertEqual(queue.dependency_stage(state, statuses, audits,
            'training_started date', True, False), 'waiting_place_sam')

    def test_dependency_failure_or_disappearance_is_terminal(self):
        state, statuses, audits = complete()
        cases = [
            (dict(stage='failed'), statuses, audits, 'training_complete date', False, False),
            (dict(stage='interrupted'), statuses, audits, 'training_complete date', False, False),
            (state, statuses, audits, 'training_failed code=1 date', False, False),
            (state, statuses, audits, 'training_interrupted date', True, False),
            (state, statuses, audits, 'training_started date', False, False),
            (dict(stage='training_b0'), {}, {}, 'training_complete date', False, False),
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(RuntimeError):
                queue.dependency_stage(*case)

    def test_complete_marker_does_not_replace_completion_audits(self):
        state, statuses, audits = complete()
        for mutate in ('missing_status', 'failed_run', 'missing_audit', 'failed_audit',
                       'short_history', 'test_used', 'wrong_inventory'):
            s, st, a = copy.deepcopy((state, statuses, audits))
            if mutate == 'missing_status': st.pop('sam')
            if mutate == 'failed_run': st['b0'] = 'training_failed date'
            if mutate == 'missing_audit': a.pop('h1_sup')
            if mutate == 'failed_audit': a['sam']['status'] = 'failed'
            if mutate == 'short_history': a['sam']['train_epochs'] = 29
            if mutate == 'test_used': a['sam']['test_epochs'] = 1
            if mutate == 'wrong_inventory': s['completed_runs'] = ['b0', 'h1_sup']
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                queue.dependency_stage(s, st, a, 'training_complete date', False, False)

    def test_existing_lock_is_read_only_and_not_created(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'queue.lock'
            self.assertFalse(queue.lock_is_held(path))
            self.assertFalse(path.exists())
            path.write_text('preserve')
            with path.open('r') as stream:
                queue.fcntl.flock(stream, queue.fcntl.LOCK_EX | queue.fcntl.LOCK_NB)
                self.assertTrue(queue.lock_is_held(path))
            self.assertFalse(queue.lock_is_held(path))
            self.assertEqual(path.read_text(), 'preserve')

    def test_exact_process_identity_rejects_other_start_ticks(self):
        import os
        pid = os.getpid()
        fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
        ticks = int(fields[19])
        self.assertTrue(queue.view_queue.process_matches(pid, ticks))
        self.assertFalse(queue.view_queue.process_matches(pid, ticks + 1))


if __name__ == '__main__':
    unittest.main()
