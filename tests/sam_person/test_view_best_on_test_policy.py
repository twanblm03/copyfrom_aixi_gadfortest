"""Exercise the staged view policy without loading models or using GPUs."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / '.local_data/view_final_20261023/source_snapshot'
SPEC = importlib.util.spec_from_file_location('selection_policy', str(SNAPSHOT / 'util/checkpoint_selection.py'))
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)
TREE = ast.parse((SNAPSHOT / 'train.py').read_text())
MAIN = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == 'main')


def assigned_expression(name):
    found = [node.value for node in ast.walk(MAIN)
             if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                                                     for target in node.targets)]
    if len(found) != 1:
        raise AssertionError('expected one assignment: ' + name)
    return compile(ast.Expression(found[0]), '<staged policy>', 'eval')


class BestOnTestPolicy(unittest.TestCase):
    def test_test_runs_for_all_thirty_epochs(self):
        expression = assigned_expression('should_evaluate')
        args = SimpleNamespace(val_mode=False, best_on_test=True, test_freq=1, epochs=30)
        epochs = [epoch for epoch in range(1, 31) if eval(expression, {'args': args, 'epoch': epoch})]
        self.assertEqual(epochs, list(range(1, 31)))

    def test_existing_final_once_and_validation_modes_still_work(self):
        expression = assigned_expression('should_evaluate')
        for val, best_test, frequency, expected in [
            (False, False, 1, [30]), (True, False, 2, list(range(2, 31, 2)))]:
            args = SimpleNamespace(val_mode=val, best_on_test=best_test, test_freq=frequency, epochs=30)
            epochs = [epoch for epoch in range(1, 31) if eval(expression, {'args': args, 'epoch': epoch})]
            self.assertEqual(epochs, expected)

    def test_secondary_metric_peak_cannot_change_selected_epoch(self):
        first = dict(zip(('group_mAP_1.0', 'group_mAP_0.5', 'outlier_mIoU', 'loss'), (20, 35, 60, 1.2)))
        second = dict(zip(first, (19, 50, 75, .7)))
        best = selection.selected_epoch_metrics({}, 6, first, True)
        after = selection.selected_epoch_metrics(best, 7, second, False)
        self.assertEqual(after, best)
        self.assertTrue(all(value['epoch'] == 6 for value in after.values()))
        self.assertEqual(after['group_mAP_0.5']['value'], 35.)

    def test_primary_improvement_moves_all_metrics_to_one_checkpoint(self):
        metrics = dict(zip(('group_mAP_1.0', 'group_mAP_0.5', 'outlier_mIoU', 'loss'), (21, 32, 58, 1.4)))
        best = selection.selected_epoch_metrics({'old': {}}, 9, metrics, True)
        self.assertEqual(set(best), set(metrics))
        self.assertTrue(all(value['epoch'] == 9 for value in best.values()))

    def test_retention_flag_bypasses_old_checkpoint_deletion(self):
        expression = assigned_expression('removed_paths')
        removal = Mock(return_value=['old.pth'])
        values = dict(args=SimpleNamespace(keep_all_checkpoints=True),
                      remove_older_checkpoints=removal, checkpoint_dir='/tmp/run', checkpoint_path='/tmp/run/epoch30.pth')
        self.assertEqual(eval(expression, values), [])
        removal.assert_not_called()


if __name__ == '__main__':
    unittest.main()
