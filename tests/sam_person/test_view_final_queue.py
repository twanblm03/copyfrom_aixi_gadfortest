"""CPU checks for final queue handoff and same-epoch TEST selection."""
import copy
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tools.queue_view_final import Queue, selected_test_row


def summary_fixture():
    train = [dict(epoch=epoch, loss=1., skipped_nonfinite_steps=0.) for epoch in range(1, 31)]
    test = [dict(epoch=epoch, loss=1. / epoch, **{'group_mAP_1.0': 20. if epoch in (3, 20) else 10.,
            'group_mAP_0.5': 30. + epoch, 'outlier_mIoU': 50. + epoch}) for epoch in range(1, 31)]
    return dict(history=dict(train=train, val=[], test=test), best={key: dict(epoch=3, value=test[2][key])
                for key in ('group_mAP_1.0', 'group_mAP_0.5', 'outlier_mIoU', 'loss')})


class FinalQueue(unittest.TestCase):
    def test_best_on_test_ties_use_first_epoch_and_metrics_share_that_epoch(self):
        summary = summary_fixture()
        selected, last = selected_test_row(summary)
        self.assertEqual((selected['epoch'], last['epoch']), (3, 30))
        summary['best']['group_mAP_0.5'] = dict(epoch=30, value=60.)
        with self.assertRaisesRegex(RuntimeError, 'all come from'):
            selected_test_row(summary)

    def test_rejects_val_nonfinite_and_missing_epoch(self):
        summary = summary_fixture()
        for mutate in (lambda value: value['history']['val'].append({'epoch': 1}),
                       lambda value: value['history']['train'][0].update(skipped_nonfinite_steps=1),
                       lambda value: value['history']['test'][0].update(loss=float('nan')),
                       lambda value: value['history']['test'].pop()):
            invalid = copy.deepcopy(summary)
            mutate(invalid)
            with self.assertRaises(RuntimeError):
                selected_test_row(invalid)

    def queue_fixture(self, root):
        root = Path(root)
        queue = Queue.__new__(Queue)
        queue.root, queue.base = root, root / 'queue'
        result = root / 'result/[cafe]_GAD_[test_final]_time'
        result.mkdir(parents=True)
        checkpoint_root = root / 'checkpoints'
        checkpoint = checkpoint_root / result.name
        checkpoint.mkdir(parents=True)
        args = dict(experiment_tag='test_final', checkpoint_root=str(checkpoint_root),
            result_path='./outputs/', val_mode=False, best_on_test=True, eval_split='test',
            load_model=False, model_path='')
        args.update({'unused_%d' % index: index for index in range(117 - len(args))})
        (result / 'args.json').write_text(json.dumps(args))
        summary = summary_fixture()
        summary['args'] = dict(args, peak_cuda_memory_mb={'0': 15000., '1': 14000.})
        (result / 'summary.json').write_text(json.dumps(summary))
        prediction_root = root / args['result_path'] / result.name
        prediction_root.mkdir(parents=True)
        for epoch in range(1, 31):
            (checkpoint / ('epoch%d.pth' % epoch)).write_bytes(b'checkpoint')
            (prediction_root / ('pred_group_epoch_%d.txt' % epoch)).write_text(''.join(
                '%d 2 0 1 1 7 6 1 1 .9\n' % video for video in (4, 8, 12, 16, 20, 24)))
        (checkpoint / 'best_test.pth').write_bytes(b'checkpoint')
        queue.plan = dict(expected_inventory={'test_clips': 6})
        state = dict(epoch=3, eval_split='test', selection_state={'epoch': 3, 'value': 20.,
                     'split': 'test', 'selection_label': 'best-on-TEST'})
        return queue, result, checkpoint, args, state

    def test_final_acceptance_writes_selected_and_epoch30_independently(self):
        with tempfile.TemporaryDirectory() as root:
            queue, result, checkpoint, args, state = self.queue_fixture(root)
            with mock.patch('torch.load', return_value=state) as load:
                queue.verify_training('b0', args)
            self.assertEqual(load.call_args.kwargs['map_location'], 'cpu')
            audit = json.loads((queue.base / 'b0/completion_audit.json').read_text())
            self.assertEqual(audit['selection_label'], 'best-on-TEST')
            self.assertEqual(audit['selected_metrics']['epoch'], 3)
            self.assertEqual(audit['epoch30_metrics']['epoch'], 30)
            (checkpoint / 'epoch12.pth').unlink()
            with self.assertRaisesRegex(RuntimeError, 'missing final checkpoint'):
                queue.verify_training('b0', args)

    def test_final_acceptance_rejects_wrong_checkpoint_and_non_test_predictions(self):
        with tempfile.TemporaryDirectory() as root:
            queue, result, checkpoint, args, state = self.queue_fixture(root)
            with mock.patch('torch.load', return_value=dict(state, epoch=20)):
                with self.assertRaisesRegex(RuntimeError, 'first argmax'):
                    queue.verify_training('b0', args)
            prediction_root = queue.root / args['result_path'] / result.name
            (prediction_root / 'pred_group_epoch_30.txt').write_text('7 2 0 1 1 7 6 1 1 .9\n')
            with mock.patch('torch.load', return_value=state):
                with self.assertRaisesRegex(RuntimeError, 'non-TEST'):
                    queue.verify_training('b0', args)

    def handoff_queue(self, root):
        root = Path(root)
        handoff = root / 'handoff.json'
        handoff.write_text(json.dumps(dict(current_training={'pid': 10},
            current_training_start_ticks=101, queue_pid=20, queue_start_ticks=202,
            current_variant='b0')))
        queue = Queue.__new__(Queue)
        queue.base, queue.interval = root / 'final', 0
        queue.status = mock.Mock()
        queue.plan = dict(handoff_file=str(handoff), predecessor_plan='old-plan')
        old = SimpleNamespace(base=root / 'old', plan={'runs': {'b0': {'resolved_args': str(root / 'old.args')}}},
                              verify_training=mock.Mock())
        (old.base / 'b0').mkdir(parents=True)
        (root / 'old.args').write_text('{}')
        return queue, old

    def test_handoff_waits_for_child_then_only_signals_exact_paused_manager(self):
        with tempfile.TemporaryDirectory() as root:
            queue, old = self.handoff_queue(root)
            calls = []
            # child alive once, exits; manager matches for TERM, CONT, then exits.
            matches = [True, False, True, True, False]
            with mock.patch('tools.queue_view_final.FrozenQueue', return_value=old), \
                 mock.patch('tools.queue_view_final.process_matches', side_effect=matches), \
                 mock.patch('tools.queue_view_final.process_state', return_value='T'), \
                 mock.patch('tools.queue_view_final.time.sleep'), \
                 mock.patch('tools.queue_view_final.os.kill', side_effect=lambda pid, sig: calls.append((pid, sig))):
                queue.wait_dependency()
            self.assertEqual(calls, [(20, signal.SIGTERM), (20, signal.SIGCONT)])
            old.verify_training.assert_called_once_with('b0', {})
            self.assertTrue((old.base / 'b0/training.status').read_text().startswith('training_complete '))

    def test_incomplete_old_b0_does_not_block_new_from_scratch_queue(self):
        with tempfile.TemporaryDirectory() as root:
            queue, old = self.handoff_queue(root)
            old.verify_training.side_effect = RuntimeError('epoch30 missing')
            with mock.patch('tools.queue_view_final.FrozenQueue', return_value=old), \
                 mock.patch('tools.queue_view_final.process_matches', return_value=False), \
                 mock.patch('tools.queue_view_final.process_state', return_value=None), \
                 mock.patch('tools.queue_view_final.os.kill') as kill:
                queue.wait_dependency()
            kill.assert_not_called()
            audit = json.loads((queue.base / 'predecessor_completion_audit.json').read_text())
            self.assertEqual(audit['status'], 'development_b0_incomplete')


if __name__ == '__main__':
    unittest.main()
