"""CPU evidence for an epoch-4 continuation with preserved TEST selection."""
import copy
import json
from pathlib import Path
import tempfile
from unittest import mock
import unittest

from tools import resume_view_final_queue as resume


class ResumeFinal(unittest.TestCase):
    def arguments(self, root):
        args = dict(experiment_tag='original', checkpoint_root=str(Path(root) / 'original_ckpt'),
            load_model=False, model_path='', val_mode=False, best_on_test=True,
            eval_split='test', epochs=30, result_path='./outputs/')
        args.update({'unused_%d' % index: index for index in range(117 - len(args))})
        recovered = dict(args, experiment_tag='resumed', checkpoint_root=str(Path(root) / 'recovered_ckpt'),
                         load_model=True, model_path=str(Path(root) / 'epoch4.pth'))
        return args, recovered

    def test_only_explicit_four_resume_fields_can_change(self):
        with tempfile.TemporaryDirectory() as root:
            original, args = self.arguments(root)
            self.assertEqual(resume.validate_args(original, args, args['model_path'])['fields_checked'], 117)
            for key, value in [('epochs', 26), ('unused_0', 100), ('best_on_test', False)]:
                changed = dict(args, **{key: value})
                with self.assertRaises(RuntimeError):
                    resume.validate_args(original, changed, args['model_path'])

    def test_resume_argv_keeps_original_settings_and_rejects_duplicate_options(self):
        _, args = self.arguments('/tmp/fixture')
        original = ['--experiment_tag', 'old', '--checkpoint_root', 'old', '--model_path', '', '--epochs', '30']
        argv = resume.resume_argv(original, args)
        self.assertEqual(argv[-1], '--load_model')
        self.assertEqual(argv[argv.index('--epochs') + 1], '30')
        with self.assertRaises(RuntimeError):
            resume.resume_argv(original + ['--load_model'], args)

    def test_frozen_copy_checks_source_and_never_overwrites_different_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, destination = root / 'original', root / 'copy'
            source.write_bytes(b'original epoch4')
            item = resume.freeze_copy(source, destination)
            resume.freeze_copy(source, destination, item['sha256'])
            source.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                resume.freeze_copy(source, destination, item['sha256'])
            self.assertEqual(destination.read_bytes(), b'original epoch4')

    def resumed_fixture(self, root):
        root = Path(root)
        _, args = self.arguments(root)
        queue = resume.Queue.__new__(resume.Queue)
        queue.root, queue.base = root, root / 'queue'
        queue.validate_recovery = mock.Mock(return_value={'status': 'passed'})
        result = root / 'result/[cafe]_GAD_[resumed]_time'
        result.mkdir(parents=True)
        (result / 'args.json').write_text(json.dumps(args))
        train = [dict(epoch=epoch, loss=1., skipped_nonfinite_steps=0.) for epoch in range(1, 31)]
        test = [dict(epoch=epoch, loss=1. / epoch, **{'group_mAP_1.0': 20. if epoch == 4 else 10.,
                'group_mAP_0.5': 30. + epoch, 'outlier_mIoU': 50. + epoch}) for epoch in range(1, 31)]
        history = dict(train=train, val=[], test=test)
        summary = dict(args=dict(args, peak_cuda_memory_mb=[]), history=history,
                       best={key: dict(epoch=4, value=test[3][key]) for key in resume.METRICS})
        (result / 'summary.json').write_text(json.dumps(summary))
        checkpoint = Path(args['checkpoint_root']) / result.name
        checkpoint.mkdir(parents=True)
        predictions = root / args['result_path'] / result.name
        predictions.mkdir(parents=True)
        text = ''.join('%d %d 0 1 1 7 6 1 1 .9\n' % ((4, 8, 12, 16, 20, 24)[i % 6], i)
                       for i in range(2414))
        inputs = {}
        for epoch in range(1, 31):
            ckpt_name, pred_name = 'epoch%d.pth' % epoch, 'pred_group_epoch_%d.txt' % epoch
            if epoch <= 4:
                for name, data in ((ckpt_name, ('ckpt%d' % epoch).encode()), (pred_name, text.encode())):
                    source = root / 'inputs' / name
                    source.parent.mkdir(exist_ok=True)
                    source.write_bytes(data)
                    inputs[name] = dict(path=str(source), sha256=resume.digest(source))
            else:
                (checkpoint / ckpt_name).write_bytes(b'new checkpoint')
                (predictions / pred_name).write_text(text)
        best_source = root / 'inputs/best_test.pth'
        best_source.write_bytes(b'original best epoch4')
        inputs['best_test.pth'] = dict(path=str(best_source), sha256=resume.digest(best_source))
        args_file = root / 'args.json'
        args_file.write_text(json.dumps(args))
        queue.plan = dict(runs={'b0': {'resolved_args': str(args_file)}},
            recovery=dict(inputs=inputs, history_prefix={key: value[:4] for key, value in history.items()}))
        best = dict(epoch=4, eval_split='test', selection_state=dict(epoch=4, value=20.,
                    split='test', selection_label='best-on-TEST', checkpoint_path='/old/best_test.pth'))
        return queue, args, result, checkpoint, best

    def test_complete_resume_inherits_early_checkpoints_predictions_and_best(self):
        with tempfile.TemporaryDirectory() as root:
            queue, args, result, checkpoint, best = self.resumed_fixture(root)
            with mock.patch('torch.load', return_value=best):
                queue.verify_training('b0', args)
            self.assertEqual((checkpoint / 'best_test.pth').read_bytes(), b'original best epoch4')
            self.assertTrue((checkpoint / 'epoch1.pth').is_file())
            audit = json.loads((queue.base / 'b0/completion_audit.json').read_text())
            self.assertEqual(audit['selected_epoch'], 4)
            self.assertEqual(audit['epoch30_metrics']['epoch'], 30)
            self.assertEqual(audit['stored_selection_path'], '/old/best_test.pth')

    def test_changed_prefix_and_corrupt_early_artifact_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            queue, args, result, checkpoint, best = self.resumed_fixture(root)
            summary_file = result / 'summary.json'
            summary = json.loads(summary_file.read_text())
            summary['history']['train'][0]['loss'] = 99.
            summary_file.write_text(json.dumps(summary))
            with self.assertRaisesRegex(RuntimeError, 'original first four'):
                queue.verify_training('b0', args)
            summary['history']['train'][0]['loss'] = 1.
            summary_file.write_text(json.dumps(summary))
            (checkpoint / 'epoch1.pth').write_bytes(b'unexpected data')
            with self.assertRaisesRegex(RuntimeError, 'overwrite'):
                queue.verify_training('b0', args)

    def test_nas_failure_waits_instead_of_starting_or_exiting(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'args.json'
            path.write_text(json.dumps({'groundtruth': '/nas/unmounted/gt.txt'}))
            queue = resume.Queue.__new__(resume.Queue)
            queue.plan = {'runs': {'b0': {'resolved_args': str(path)}}}
            queue.status, queue.interval = mock.Mock(), 0
            with mock.patch('builtins.open', side_effect=FileNotFoundError('NAS not mounted')), \
                 mock.patch('tools.resume_view_final_queue.time.sleep', side_effect=StopIteration):
                with self.assertRaises(StopIteration):
                    queue.wait_nas()
            self.assertEqual(queue.status.call_args.args[0], 'waiting_nas')

    def test_local_check_never_reads_nas(self):
        with tempfile.TemporaryDirectory() as root:
            queue = resume.Queue.__new__(resume.Queue)
            queue.base = Path(root)
            queue.plan = dict(frozen_files=[{'path': '/nas/absent/gt.txt', 'sha256': 'pending'}])
            queue.validate_recovery = mock.Mock(return_value={'status': 'passed'})
            with mock.patch('tools.resume_view_final_queue.digest', side_effect=AssertionError('should not hash NAS')):
                report = queue.local_check()
            self.assertEqual(report['status'], 'local_checks_passed')
            self.assertEqual(report['nas_pending'], ['/nas/absent/gt.txt'])


if __name__ == '__main__':
    unittest.main()
