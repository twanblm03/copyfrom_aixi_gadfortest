"""Resume checks reject inconsistent epoch/optimizer state or protocol drift."""
import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[2] / 'tools/resume_view_queue.py'
SPEC = importlib.util.spec_from_file_location('view_resume', PATH)
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


class ViewResume(unittest.TestCase):
    def setUp(self):
        self.planned = dict(epochs=30, selection_metric='group_mAP_1.0')
        self.checkpoint = dict(
            epoch=3, protocol='development', eval_split='val', state_dict={'weight': 1},
            optimizer=dict(state={0: dict(step=909)},
                           param_groups=[dict(lr=7.75e-5), dict(lr=7.75e-5)]),
            scheduler=dict(last_epoch=3, _last_lr=[7.75e-5, 7.75e-5]),
            history=dict(train=[dict(epoch=i, loss=1., skipped_nonfinite_steps=0) for i in (1, 2, 3)],
                         val=[dict(epoch=i, loss=1.) for i in (1, 2, 3)], test=[]),
            selection_state=dict(metric='group_mAP_1.0', mode='max', epoch=3, value=2.))

    def test_complete_checkpoint_continues_at_four(self):
        result = resume.validate_checkpoint(self.checkpoint, self.planned, 303)
        self.assertEqual((result['next_epoch'], result['target_epochs']), (4, 30))
        self.assertEqual(result['completed_optimizer_steps'], 909)
        self.assertFalse(result['rng_state_saved'])

    def test_misaligned_optimizer_and_scheduler_are_rejected(self):
        for field in ('step', 'epoch', 'lr'):
            c = copy.deepcopy(self.checkpoint)
            if field == 'step': c['optimizer']['state'][0]['step'] = 908
            elif field == 'epoch': c['scheduler']['last_epoch'] = 2
            else: c['optimizer']['param_groups'][0]['lr'] = 1e-5
            with self.subTest(field=field), self.assertRaises(ValueError):
                resume.validate_checkpoint(c, self.planned, 303)

    def test_missing_history_or_test_use_are_rejected(self):
        for field in ('history', 'test', 'skipped'):
            c = copy.deepcopy(self.checkpoint)
            if field == 'history': c['history']['val'].pop()
            elif field == 'test': c['history']['test'] = [dict(epoch=1)]
            else: c['history']['train'][0]['skipped_nonfinite_steps'] = 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                resume.validate_checkpoint(c, self.planned, 303)

    def test_resume_flags_do_not_allow_batch_or_epoch_drift(self):
        original = dict(batch=16, epochs=30, load_model=False, model_path='',
                        experiment_tag='b0', checkpoint_root='/old')
        resumed = dict(original, load_model=True, model_path='/epoch3.pth',
                       experiment_tag='b0_resume', checkpoint_root='/resume')
        self.assertEqual(set(resume.validate_args(original, resumed)), resume.RESUME_FIELDS)
        for key, value in [('batch', 8), ('epochs', 33)]:
            changed = dict(resumed, **{key: value})
            with self.subTest(field=key), self.assertRaises(ValueError):
                resume.validate_args(original, changed)


if __name__ == '__main__':
    unittest.main()
