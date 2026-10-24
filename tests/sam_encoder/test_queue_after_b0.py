"""CPU-only audit tests; process signals and queue actions are mocked."""
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'queue_sam_encoder_review', ROOT / 'tools/queue_sam_encoder_after_b0.py')
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


def contracts():
    planned = dict(
        backbone='sam_vit_b', freeze_backbone=True, experiment_tag='sam_encoder',
        checkpoint_root='/tmp/sam_encoder', train_mask_mode='disabled',
        heatmap_variant='off', split='view', val_mode=False, best_on_test=True,
        keep_all_checkpoints=True, protocol='best_on_test', eval_split='test',
        epochs=30, batch=16, test_batch=4, num_frame=5, load_model=False,
        model_path='', device='0,1,2,3', parallel_transport='host', lr=1e-5,
        sam_repo='/tmp/sam_source', sam_checkpoint='/tmp/sam.pth',
        sam_checkpoint_sha256='a' * 64, sam_encoder_chunk_size=1,
        sam_output_height=23, sam_output_width=40,
        heatmap_target_mode='box', heatmap_crowd_cap_ratio=.35)
    reference = {key: value for key, value in planned.items()
                 if not key.startswith('sam_') and key not in (
                     'heatmap_target_mode', 'heatmap_crowd_cap_ratio')}
    reference.update(backbone='resnet18', freeze_backbone=False,
                     experiment_tag='b0', checkpoint_root='/tmp/b0')
    return reference, planned


def summary_fixture():
    train = [dict(epoch=i, loss=1. / i, skipped_nonfinite_steps=0)
             for i in range(1, 31)]
    test = [dict(epoch=i, loss=1. / i, outlier_mIoU=30. + i,
                 **{'group_mAP_1.0': 20. if i in (2, 3) else 1., 'group_mAP_0.5': float(i)})
            for i in range(1, 31)]
    selected = test[1]
    return dict(history=dict(train=train, test=test, val=[]), best={
        metric: dict(epoch=2, value=selected[metric])
        for metric in ('group_mAP_1.0', 'group_mAP_0.5', 'outlier_mIoU', 'loss')})


class ContractTests(unittest.TestCase):
    def test_only_encoder_and_output_identity_change(self):
        report = QUEUE.validate_contract(*contracts())
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(set(report['method_changes']), QUEUE.METHOD_FIELDS)

    def test_shared_protocol_drift_is_rejected(self):
        for field, value in [('lr', 1e-4), ('num_frame', 3), ('load_model', True),
                             ('batch', 8), ('parallel_transport', 'native')]:
            reference, planned = contracts()
            planned[field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                QUEUE.validate_contract(reference, planned)

    def test_missing_or_extra_field_is_rejected(self):
        for mode in ('missing', 'extra'):
            reference, planned = contracts()
            if mode == 'missing':
                del planned['lr']
            else:
                planned['unapproved_flag'] = True
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                QUEUE.validate_contract(reference, planned)

    def test_sam_geometry_or_heatmap_branch_cannot_change(self):
        for field, value in [('sam_output_height', 36), ('sam_encoder_chunk_size', 2),
                             ('heatmap_target_mode', 'crowd')]:
            reference, planned = contracts()
            planned[field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                QUEUE.validate_contract(reference, planned)


class SummaryTests(unittest.TestCase):
    def test_first_tied_primary_epoch_supplies_all_metrics(self):
        selected = QUEUE.verify_summary(summary_fixture())
        self.assertEqual(selected['epoch'], 2)
        self.assertEqual(selected['group_mAP_0.5'], 2.)

    def test_independent_secondary_peak_is_rejected(self):
        summary = summary_fixture()
        summary['best']['group_mAP_0.5'] = dict(epoch=30, value=30.)
        with self.assertRaises(RuntimeError):
            QUEUE.verify_summary(summary)

    def test_incomplete_or_val_history_is_rejected(self):
        for split in ('train', 'test', 'val'):
            summary = summary_fixture()
            if split == 'val':
                summary['history']['val'] = [dict(epoch=1)]
            else:
                summary['history'][split].pop()
            with self.subTest(split=split), self.assertRaises(RuntimeError):
                QUEUE.verify_summary(summary)

    def test_nonfinite_and_skipped_training_are_rejected(self):
        for mode in ('train_nan', 'test_inf', 'skipped', 'missing_steps'):
            summary = summary_fixture()
            if mode == 'train_nan':
                summary['history']['train'][0]['loss'] = float('nan')
            elif mode == 'test_inf':
                summary['history']['test'][0]['loss'] = float('inf')
            elif mode == 'skipped':
                summary['history']['train'][0]['skipped_nonfinite_steps'] = 1
            else:
                del summary['history']['train'][0]['skipped_nonfinite_steps']
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                QUEUE.verify_summary(summary)

    def test_missing_train_loss_cannot_be_certified_complete(self):
        summary = summary_fixture()
        del summary['history']['train'][0]['loss']
        with self.assertRaises((RuntimeError, KeyError)):
            QUEUE.verify_summary(summary)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='sam_queue_review_')
        self.addCleanup(self.temporary.cleanup)
        self.queue = QUEUE.Queue.__new__(QUEUE.Queue)
        self.queue.base = Path(self.temporary.name)
        self.manager = dict(pid=77701, start_ticks=1, boot_id='test-boot')
        self.child = dict(pid=77702, start_ticks=2, boot_id='test-boot')
        self.queue.plan = dict(predecessor_manager=self.manager,
                               predecessor_child=self.child,
                               predecessor_state='mock-state.json')
        self.queue.predecessor_paused = False
        self.queue.retired = False

    def test_attach_stops_controller_pid_only(self):
        state = dict(stage='training_b0', pid=self.manager['pid'])
        with mock.patch.object(QUEUE, 'process_info', side_effect=[
                dict(state='S'), dict(parent=self.manager['pid']),
                dict(state='T'), dict(parent=self.manager['pid'])]), \
                mock.patch.object(QUEUE, 'read_json', return_value=state), \
                mock.patch.object(QUEUE.os, 'kill') as kill, \
                mock.patch.object(QUEUE.os, 'killpg') as killpg:
            self.queue.attach()
        kill.assert_called_once_with(self.manager['pid'], signal.SIGSTOP)
        killpg.assert_not_called()
        report = json.loads((self.queue.base / 'handoff.json').read_text())
        self.assertTrue(report['child_left_running'])

    def test_child_exit_during_attach_is_rejected_and_can_rollback(self):
        state = dict(stage='training_b0', pid=self.manager['pid'])
        with mock.patch.object(QUEUE, 'process_info', side_effect=[
                dict(state='S'), dict(parent=self.manager['pid']),
                dict(state='T'), None]), \
                mock.patch.object(QUEUE, 'read_json', return_value=state), \
                mock.patch.object(QUEUE.os, 'kill') as kill:
            with self.assertRaises(RuntimeError):
                self.queue.attach()
            self.assertTrue(self.queue.predecessor_paused)
            kill.assert_called_once_with(self.manager['pid'], signal.SIGSTOP)
        with mock.patch.object(QUEUE, 'process_info', return_value=dict(state='T')), \
                mock.patch.object(QUEUE.os, 'kill') as kill:
            self.queue.restore_predecessor()
            kill.assert_called_once_with(self.manager['pid'], signal.SIGCONT)
        self.assertFalse(self.queue.predecessor_paused)

    def test_retired_controller_is_never_resumed(self):
        self.queue.predecessor_paused = True
        self.queue.retired = True
        with mock.patch.object(QUEUE.os, 'kill') as kill:
            self.queue.restore_predecessor()
        kill.assert_not_called()

    def test_nonzero_child_exit_rejects_before_retirement(self):
        self.queue.status = mock.Mock()
        with mock.patch.object(QUEUE, 'process_info', side_effect=[None, dict(state='T')]), \
                mock.patch.object(QUEUE, 'completed_child_exit', side_effect=RuntimeError('exit=1')), \
                mock.patch.object(QUEUE.os, 'kill') as kill:
            with self.assertRaises(RuntimeError):
                self.queue.wait_b0()
        kill.assert_not_called()
        self.assertFalse(self.queue.retired)


if __name__ == '__main__':
    unittest.main()
